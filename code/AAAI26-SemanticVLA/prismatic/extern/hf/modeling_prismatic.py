"""
modeling_prismatic.py

HuggingFace 风格的核心模型定义：PrismaticPreTrainedModel 和 PrismaticForConditionalGeneration。
继承自 `transformers.PretrainedModel`，独立自包含，复现 `prismatic.models.vlms.prismatic.py` 的逻辑。
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)
from prismatic.models.router import MoEAggregator

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# 日志记录器
logger = logging.getLogger(__name__)

from prismatic.models.modeling_llama import replace_llama_spda_forward
replace_llama_spda_forward()

# from prismatic.models.modeling_llama_fastv import replace_llama_fastv_forward
# replace_llama_fastv_forward()


# === Monkey-Patch 工具函数 ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """将返回元组的函数包装为只返回第一个元素。"""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers 会覆盖名称包含 `gamma` 的参数，因此需要修补 VisionBackbone.LayerScale。
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    """LayerScale 修补后的前向传播：将 `gamma` 重命名为 `scale_factor` 并使用 mul。"""
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    """对单个 LayerScale 模块应用修补：克隆 gamma → scale_factor，替换 forward，删除 gamma。"""
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic 视觉 Backbone (nn.Module) 定义（支持双 Backbone 融合） ===
class PrismaticVisionBackbone(nn.Module):
    """
    Prismatic 模型的视觉 backbone，负责图像特征提取。

    支持单 backbone（如 SigLIP）和双 backbone 融合（如 SigLIP + DINOv2）两种配置。
    双 backbone 模式下，两个模型的特征沿特征维度拼接。
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        初始化视觉 backbone。

        参数:
            use_fused_vision_backbone: 是否使用双 backbone 并融合特征
            image_sizes: 每个 backbone 的输入图像尺寸列表
            timm_model_ids: 每个 backbone 的 TIMM 模型 ID 列表
            timm_override_act_layers: 每个 backbone 的激活层覆盖列表
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # 默认值，后续可覆盖

        # 校验双 backbone 数量
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic 模型最多支持 2 个（融合）视觉 backbone！")

        # 创建主特征提取器
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # 如果使用双 backbone 融合，创建第二个特征提取器
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # 修补 LayerScale 模块以兼容 HF 参数命名
        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        创建基于 TIMM 的特征提取器模型。

        参数:
            model_id: 要加载的 TIMM 模型 ID
            img_size: 模型的输入图像尺寸
            act_layer: 激活层类型覆盖（可选）

        返回:
            配置好的特征提取器模型
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch forward 函数，提取倒数第二层的特征
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        """
        修补所有 LayerScale 模块以兼容 HF 参数命名。

        HF Transformers 会覆盖名称包含 'gamma' 的参数，因此需要重命名并修改 forward 方法。
        """
        # 修补主特征提取器
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # 如果存在，修补第二个特征提取器
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """返回视觉 backbone 输出的 patch 数量。"""
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        """返回视觉 backbone 期望的输入图像数量。"""
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        设置视觉 backbone 的输入图像数量。

        参数:
            num_images_in_input: 期望的输入图像数量
        """
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        视觉 backbone 的前向传播。

        如果 `use_fused_vision_backbone == True`，同时使用 SigLIP 和 DINOv2 提取视觉特征
        （否则仅使用 SigLIP）。支持多图输入（但仅限双 backbone 融合模式）。

        参数:
            pixel_values (torch.Tensor): 输入图像的像素值, (B, C, H, W)
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # 拆分 pixel_values :: [bsz, 2*3, resolution, resolution] → 分别提取特征
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return (patches, patches_fused)  # Tuple[Tensor]

        else:
            assert self.use_fused_vision_backbone, "多图输入必须使用双 backbone 融合模式！"

            # 将 pixel_values 拆分为单张图像（每张 6 通道：3 给 SigLIP + 3 给 DINOv2）
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # 逐图处理并收集 patch
            all_patches = []
            for img in images:
                # 再将每张图拆分为两组各 3 通道
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # 分别通过 SigLIP 和 DINOv2 提取特征
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                all_patches.append((patches, patches_fused))

            return all_patches  # List[Tuple[Tensor]]


# === Prismatic 投影器 (nn.Module) 定义 ===
class PrismaticProjector(nn.Module):
    """将视觉 patch 特征从 vision_dim 投影到 LLM 的 embedding 空间 (llm_dim=4096)。

    单 backbone 模式：2 层 MLP（Linear → GELU → Linear）
    双 backbone 融合模式：3 层 MLP，第一层从 vision_dim 放大到 4*vision_dim 的中间维度。
    """

    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # 根据是否双 backbone 选择不同的 MLP 结构和投影因子
        if not self.use_fused_vision_backbone:
            # 单 backbone：简洁的 2 层 MLP
            #   vision_dim → llm_dim → llm_dim
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            # 双 backbone 融合：3 层 MLP，先放大再压缩
            #   vision_dim → 4*vision_dim → llm_dim → llm_dim
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """
        将视觉 patch 投影到 LLM embedding 空间。

        参数:
            img_patches: [B, num_patches, vision_dim]
        返回:
            [B, num_patches, llm_dim]
        """
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === 主要 HF 模型类定义 ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Prismatic 视觉条件语言模型的输出基类，同时暴露视觉特征。"""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # VLM 额外输出
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    """Prismatic 预训练模型基类，定义通用的 HF 属性和权重初始化。"""
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # 重要：此 HF 移植版不适用于从头训练，仅用于推理和微调！
        #   因此此 init_weights 代码并不完整；如需从头训练 VLM，请使用主代码库：
        #   https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """检查 LLM 是否支持 SDPA 注意力。"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    """Prismatic 条件生成模型：视觉 backbone + 投影器 + LLM backbone。

    支持多种视觉聚合策略（concat / MoE）和可选的 MM Sampler 用于 token 压缩。
    """

    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)
        self.config = config

        # [校验] 对 config 字段和依赖版本进行基本校验
        if config.use_fused_vision_backbone is None:
            raise ValueError("缺少 config 字段 `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM 版本必须 >= 0.9.10 且 < 1.0.0（不兼容大版本更新）；"
                "如需紧急支持最新 TIMM 版本，请提交 GitHub Issue。"
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"期望 `transformers==4.40.1` 和 `tokenizers==0.19.1`，但当前为 "
                f"`transformers=={transformers.__version__}` 和 `tokenizers=={tokenizers.__version__}`；"
                f"依赖版本变化可能导致推理时的性能回退。如有疑问，请使用上述指定版本。"
            )

        # 实例化 PrismaticVisionBackbone（可支持双 backbone 融合）
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # 根据视觉聚合策略创建投影层
        if config.vision_aggregate_type == 'moe':
            # MoE（混合专家）聚合：先分别投影，再由 Router 动态融合
            assert config.use_fused_vision_backbone, \
                '使用 MoE 聚合时必须设置 `use_fused_vision_backbone=True`'
            self.moe_router = MoEAggregator(
                num_experts=2,
                seq_dim=config.text_config.hidden_size,
                router_method='mlp'
            )
            self.featurizer_proj = PrismaticProjector(
                False,
                vision_dim=self.vision_backbone.featurizer.embed_dim,
                llm_dim=config.text_config.hidden_size,
            )
            self.fused_featurizer_proj = PrismaticProjector(
                False,
                vision_dim=self.vision_backbone.fused_featurizer.embed_dim,
                llm_dim=config.text_config.hidden_size,
            )
        elif config.vision_aggregate_type == 'concat':
            # Concat 聚合：沿 hidden dim 拼接后统一投影
            self.projector = PrismaticProjector(
                config.use_fused_vision_backbone,
                vision_dim=self.vision_backbone.embed_dim,
                llm_dim=config.text_config.hidden_size,
            )
        else:
            NotImplementedError

        # 可选的 MM Sampler：对 ViT patch 进行压缩以减少 token 数量
        if config.featurizer_cfg.get("use_mm_sampler", False):
            self.featurizer_sampler_proj = PrismaticProjector(
                False,
                vision_dim=self.vision_backbone.featurizer.embed_dim,
                llm_dim=config.text_config.hidden_size,
            )
        if config.fused_featurizer_cfg.get("use_mm_sampler", False):
            self.fused_featurizer_sampler_proj = PrismaticProjector(
                False,
                vision_dim=self.vision_backbone.fused_featurizer.embed_dim,
                llm_dim=config.text_config.hidden_size,
            )

        # 实例化 LLM Backbone（Llama-2 7B）
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        for layer in self.language_model.model.layers:
            layer.get_num_patches = self.get_num_patches
            layer.get_num_images_in_input = self.get_num_images_in_input
            layer.self_attn.num_action_tokens = self.config.num_action_tokens

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size

        # HF 样板代码：通过 `_init_weights()` 初始化权重并设置梯度检查点
        self.post_init()

    def get_num_patches(self) -> int:
        """返回视觉 backbone 输出的每张图 patch 数量。"""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """返回视觉 backbone 期望的输入图像数量。"""
        return self.vision_backbone.get_num_images_in_input()

    # === `PreTrainedModel` 样板方法 ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        # 注意：Llama-2 和 Mistral 不绑定权重（无操作）
        self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # 同步更新 config/实例变量
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        用噪声动作特征替换 input_embeddings 中动作 token 位置的 embedding。

        [扩散] 在前向传播中，将 placeholder（全零）的动作 token embedding
        替换为从噪声动作投影得到的 embedding。

        参数:
            input_embeddings: 形状 (B, S, D) 的张量
            all_actions_mask: 形状 (B, S) 的布尔张量，标记动作 token 位置
            noisy_action_features: 形状 (B, K, D) 的张量，K 为每个样本中动作 token 的数量

        返回:
            修改后的 input_embeddings 张量
        """
        # 克隆输入以避免修改原始张量
        new_input_embeddings = input_embeddings.clone()

        # 创建与 input_embeddings 同形状的占位张量
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # 为拼接创建 batch 索引
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # 获取每个样本中 mask 为 True 的位置索引
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # 将噪声动作特征放置到正确位置
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # 利用 mask 合并原始 embedding 和噪声动作 embedding
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        """从 labels 中提取动作掩码：当前动作 + 未来动作 token 的位置。"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _aggregate_patch_features(self, patch_features, language_embeddings=None):
        """
        聚合来自双 backbone 的 patch 特征，支持三种策略：
        1. concat：沿 hidden dim 拼接 → 通过 projector 投影
        2. MoE：分别投影 → MoE Router 融合
        3. 可选 MM Sampler：将部分 patch 作为文本引导 token 单独处理
        """
        patches, patches_fused = patch_features

        # 处理 SigLIP 侧的 MM Sampler（文本引导 token 压缩）
        text_agg_embed = None
        if self.config.featurizer_cfg["use_mm_sampler"] and self.config.featurizer_cfg["text_topk"] > 0:
            nt = self.config.featurizer_cfg["text_topk"]
            text_agg_embed, patches = patches[:, :nt, :], patches[:, nt:, :]

        # 处理 DINOv2 侧的 MM Sampler
        text_agg_embed_fused = None
        if self.config.fused_featurizer_cfg["use_mm_sampler"] and self.config.fused_featurizer_cfg["text_topk"] > 0:
            nt = self.config.fused_featurizer_cfg["text_topk"]
            text_agg_embed_fused, patches_fused = patches_fused[:, :nt, :], patches_fused[:, nt:, :]

        patch_features = (patches, patches_fused)

        if self.config.vision_aggregate_type == 'moe':
            # MoE 聚合：先分别投影到 LLM 空间，再由 Router 根据语言特征动态融合
            assert patch_features[0].shape[1] == patch_features[1].shape[1], \
                f"两个 backbone 的 token 数量必须相同，当前为 {[x.shape for x in patch_features]}"
            average_language_embedding = language_embeddings.mean(dim=1)
            patches, patches_fused = patch_features
            patches_llm_proj = self.featurizer_proj(patches)             # SigLIP → [B, L, 4096]
            patches_fused_llm_proj = self.fused_featurizer_proj(patches_fused)  # DINOv2 → [B, L, 4096]
            image_embeds = self.moe_router(
                [patches_llm_proj, patches_fused_llm_proj], average_language_embedding
            )
        elif self.config.vision_aggregate_type == 'concat':
            # Concat 聚合：沿 hidden dim 拼接再投影
            assert patch_features[0].shape[1] == patch_features[1].shape[1], \
                f"两个 backbone 的 token 数量必须相同，当前为 {[x.shape for x in patch_features]}"
            image_embeds = torch.cat(patch_features, dim=2)  # 沿 hidden dim 拼接 SigLIP 和 DINOv2
            image_embeds = self.projector(image_embeds)       # 投影到 LLM embedding 空间
        else:
            raise NotImplementedError

        # 如果使用了 MM Sampler，将被剥离的文本引导 token 投影后拼回序列前部
        if text_agg_embed is not None:
            image_embeds = torch.cat([self.featurizer_sampler_proj(text_agg_embed), image_embeds], dim=1)
        if text_agg_embed_fused is not None:
            image_embeds = torch.cat([self.fused_featurizer_sampler_proj(text_agg_embed_fused), image_embeds], dim=1)

        return image_embeds  # (B, L, D)

    def _process_vision_features(self, pixel_values, language_embeddings=None, instructions=None):
        """
        处理视觉特征：通过视觉 backbone 提取 patch 特征，然后聚合。

        支持原始 PrismaticVisionBackbone 类以及包装类（如 FiLM 包装器）。
        多图输入时，逐图处理后在 patch 维度拼接。
        """
        if isinstance(self.vision_backbone, PrismaticVisionBackbone):  # 原始类
            patch_features = self.vision_backbone(pixel_values)
        else:  # 包装类，额外接收 language_embeddings
            patch_features = self.vision_backbone(pixel_values, language_embeddings, instructions)

        if self.vision_backbone.get_num_images_in_input() == 1:
            image_embeds = self._aggregate_patch_features(patch_features, language_embeddings)  # (B, 256, D)
        else:
            all_image_embeds = []
            for img_patch_features in patch_features:
                all_image_embeds.append(self._aggregate_patch_features(img_patch_features, language_embeddings))

            # 沿 patch 维度拼接所有图像， (B, 256 * num_images, D)
            image_embeds = torch.cat(all_image_embeds, dim=1)

        return image_embeds

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """
        处理本体感知特征并附加到视觉特征末尾。

        将 proprio 向量通过 ProprioProjector 投影为单个 token，
        拼接到 vision patch token 序列的末尾。
        """
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (B, num_patches * num_images, llm_dim)
            # proprio: (B, proprio_dim) 或 (proprio_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (B, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (B, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (B, 1, llm_dim)
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """
        构建多模态 embedding 和 attention mask。

        将 vision patch embedding 插入到 text embedding 的 <BOS> token 之后：
        [<BOS>, patches..., text...]
        """
        # 为 vision patch 创建 attention mask（全部为 True）
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # 构建多模态 embedding：在 <BOS> (位置 1:) 之后插入 vision patch
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """
        构建多模态 labels：vision patch 位置的 label 设为 IGNORE_INDEX 以忽略 loss。
        """
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Prismatic VLM 核心 `forward()` 逻辑 ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        instructions: Optional[List[str]|torch.LongTensor] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """
        VLM 核心前向传播，返回 PrismaticCausalLMOutputWithPast 实例。

        支持三种模式：
        1. 缓存生成模式（input_ids.shape[1] == 1）：跳过视觉编码，仅 LLM forward
        2. 纯文本模式（pixel_values is None）：仅 LLM forward
        3. 多模态模式：视觉编码 → 投影 → 组装多模态序列 → LLM forward
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 仅在非训练时使用缓存（即使梯度检查点关闭）
        use_cache = use_cache and not self.training

        # 投影特征的占位符
        projected_patch_embeddings = None

        # === 模式 1：带缓存的生成（input_ids 仅包含单个 token） ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "当前仅支持 batch size = 1 的生成！"
            assert past_key_values is not None, "缓存生成时必须提供 `past_key_values`！"
            assert labels is None, "缓存生成时不应出现 `labels`！"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === 模式 2：纯文本前向（无图像） ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "纯文本 forward 缺少 `input_ids`！"
            assert past_key_values is None, "纯文本 forward 不应出现 `past_key_values`！"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === 模式 3：多模态前向（图像 + 文本） ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "多模态 forward 不应出现 `past_key_values`！"

            # 获取输入 embedding（来自 LLM embedding 层）
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            # 提取动作掩码（标记哪些位置是动作 token）
            all_actions_mask = self._process_action_masks(labels)

            # 提取语言部分的输入 embedding（去除动作 token 部分）
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)

            # 获取视觉特征
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, instructions)

            # 如果提供了本体感知状态，拼接到 vision patch 末尾
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

            # [扩散] 如果提供了扩散时间步 embedding，拼接到 patch 末尾
            if diffusion_timestep_embeddings is not None:
                projected_patch_embeddings = torch.cat(
                    (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
                )

            # 处理动作 embedding
            if noisy_actions is not None:
                # 重新获取动作掩码
                all_actions_mask = self._process_action_masks(labels)

                # 将噪声动作变形为单个动作 token
                # noisy_actions: (B, chunk_len, action_dim) → (B, chunk_len * action_dim, 1)
                B = noisy_actions.shape[0]
                noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

                # 将噪声动作 token 投影到 LLM embedding 空间
                noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

                # 替换动作 token 的 embedding 为噪声动作 embedding
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, noisy_action_features
                )
            else:
                # 将动作 token 的 embedding 置零（后续会加上位置编码）
                all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
                input_embeddings = input_embeddings * ~all_actions_mask

            # 构建多模态 embedding 和 attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # 构建多模态 labels
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

            # 送入 LLM
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === 非法输入 ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("非齐次（文本, 图像）输入批次 -- forward() 不支持混合批次！")

        else:
            raise ValueError(
                "PrismaticForConditionalGeneration `forward()` 调用参数无效：\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # 解包 `language_model_output` 并返回
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin 方法 ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """
        生成前的输入准备，借鉴 `LlamaForCausalLM` 并简化为 batch_size=1。

        处理缓存：有 `past_key_values` 时仅取最后一个 token；
        `inputs_embeds` 仅在第一步生成时使用。
        """
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("当前不支持 batch size > 1 的生成！")

        # 使用缓存：仅保留最后未处理的 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # inputs_embeds 仅在第一步使用
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 确保 pixel_values 被保留
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # 委托给 Language Model（不同模型处理方式不同）
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    """
    OpenVLA 动作预测模型：在 PrismaticForConditionalGeneration 基础上
    增加动作 token 化/去 token 化、连续动作头和扩散推理等功能。
    """

    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # 计算动作离散化的 bin
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 计算去 token 化所用的词表大小（减去 pad_to_multiple_of 的填充）
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

        # 动作 token 总数：-1 表示使用默认的 ACTION_DIM * NUM_ACTIONS_CHUNK
        self.TOTAL_ACTION_TOKENS = -1
        if self.config.num_action_tokens == -1:
            self.TOTAL_ACTION_TOKENS = ACTION_DIM * NUM_ACTIONS_CHUNK
        else:
            self.TOTAL_ACTION_TOKENS = self.config.num_action_tokens * NUM_ACTIONS_CHUNK

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """
        为动作预测准备输入：在序列末尾追加动作 placeholder token 和 STOP token。

        输入:  [BOS, patches..., text...]
        输出: [BOS, patches..., text..., action_placeholder(56), STOP]
        """
        # 追加动作 placeholder token（值为 1）
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], self.TOTAL_ACTION_TOKENS)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 追加 STOP token（训练时出现在非因果双向自注意力中）
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # 扩展 attention mask 以匹配新序列长度
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """
        为动作预测创建 labels 张量：扩展 labels 序列以覆盖新增的 action token 和 STOP token。
        """
        # 用任意动作 token 索引填充扩展部分
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # 最后一个 label token 设为 STOP
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """
        使用数据集统计量将归一化动作反归一化。

        支持两种归一化方式：
        - BOUNDS：使用 min/max 反归一化到原始范围
        - BOUNDS_Q99：使用 1%/99% 分位数反归一化
        """
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("不支持的动作/本体感知归一化类型！")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        """
        运行扩散模型的动作预测（反向扩散过程）。

        x_T（纯噪声）→ 迭代去噪 → x_0（干净动作），
        每一步以 VLM 的视觉-语言隐变量、当前噪声动作和时间步 embedding 为条件。
        """
        # 设置扩散时间步
        action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)
        # 克隆 vision embedding 供每个时间步复用
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        # 反向扩散：迭代去噪生成动作预测
        for t in action_head.noise_scheduler.timesteps:
            # 扩散时间步的 sinusoid 编码
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # 将扩散时间步 embedding 拼接到 vision patch 末尾
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # 变形并投影噪声动作到 LLM embedding 空间
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # 替换动作 token embedding 为噪声动作 embedding
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # 构建多模态 embedding 和 attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # LLM 前向（需要 hidden_states）
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # 提取动作部分的 hidden states
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + self.TOTAL_ACTION_TOKENS,
                :,
            ]  # (B, total_action_tokens, D)

            # 预测噪声并更新：x_t → x_{t-1}
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        """
        L1 回归连续动作预测 或 离散 token 动作预测。

        - 有 action_head：通过 MLP 直接从 hidden states 回归连续动作值
        - 无 action_head：通过 logits argmax 解码为离散 bin，再映射为连续值
        """
        # 将动作 token embedding 置零（仅保留位置编码信息）
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # 构建多模态 embedding
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # LLM 前向
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # 提取动作 token 位置的 hidden states
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + self.TOTAL_ACTION_TOKENS,
            :,
        ]  # (B, total_action_tokens, D)

        # 根据不同预测方式生成动作
        if action_head is not None:
            # L1 回归：直接从 hidden states 回归连续动作
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # 离散 token 预测：argmax → bin center 映射
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + self.TOTAL_ACTION_TOKENS,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        instructions=None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        从输入序列预测动作，支持多种预测方式。

        流程：
        1. 构建完整的推理序列（文本 + action placeholder + STOP）
        2. 视觉编码 + 投影 → LLM 前向
        3. 提取动作 hidden states → 回归/扩散/离散解码
        4. 反归一化 → 连续动作向量

        参数:
            input_ids: 输入 token id 序列
            unnorm_key: 反归一化统计量的键名
            proprio: 本体感知特征（可选）
            proprio_projector: 本体感知投影器（可选）
            action_head: L1 回归或扩散动作头（可选）
            noisy_action_projector: 扩散模式下噪声动作的投影器（可选）
            instructions: 指令文本（可选，用于 FiLM/Vision-Language 条件注入）
            **kwargs: 额外参数，包括 pixel_values 和 attention_mask

        返回:
            (未归一化动作向量, 动作 hidden states) 的元组
        """
        # 如果 prompt 中冒号后没有空 token ('')（在 "OUT:" 或 "ASSISTANT:" 之后），
        # 插入它以匹配训练时的输入格式
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # 创建假的 labels 张量（动作掩码所需）
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # 获取 prompt 的 token 数（减去动作 token 和 STOP token）
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        # 追加动作 placeholder token + STOP token
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # 更新 labels 以供后续动作掩码计算
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # 获取输入 embedding 和动作掩码
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # 提取语言 embedding（去除动作部分）
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # 视觉特征提取 + 聚合
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, instructions)

        # 如果提供了本体感知特征，拼接到 vision patch 末尾
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # 判断是否使用扩散（有 noisy_action_projector 且 action_head 有 noise_scheduler）
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # 计算 patch 数量（包含可能的 proprio token 和/或扩散时间步 embedding）
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            # 从标准高斯分布采样初始噪声，作为反向扩散的起点
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # 运行扩散预测
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # 运行回归或离散 token 预测
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
            )

        # 反归一化动作
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """
        校验并解析反归一化统计量的键名。

        如果未指定 unnorm_key 且模型仅在单个数据集上训练，自动使用该数据集的统计量；
        如果在多个数据集上训练，必须显式指定。
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"模型在多个数据集上训练，请从以下选项中指定 `unnorm_key` "
                f"以选择用于动作反归一化的统计量：{norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"指定的 `unnorm_key` 不在可用数据集统计量集合中，"
            f"请从以下选项中选择：{norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """获取策略动作空间的维度。"""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """获取指定数据集的所有统计量。"""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
