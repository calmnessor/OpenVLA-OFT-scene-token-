"""
modeling_prismatic.py

HuggingFace 风格的 PrismaticPreTrainedModel 和 PrismaticForConditionalGeneration 类定义。
继承自 transformers.PretrainedModel。独立且自包含，但逻辑完全复刻 prismatic.models.vlms.prismatic.py。

这是整个 VLA 模型的核心文件，包含：
  - PrismaticVisionBackbone: 视觉编码器（SigLIP / DINOv2 融合）
  - PrismaticProjector: 视觉特征 → LLM 空间的投影层
  - PrismaticForConditionalGeneration.forward(): 多模态前向传播
  - OpenVLAForActionPrediction.predict_action(): 动作预测的总调度入口
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

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Set up logger
logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数：Monkey-Patch 与 HF 兼容性修复
# ============================================================================

def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """
    TIMM 的 get_intermediate_layers 返回 tuple，但 HF 期望返回 Tensor。
    这个包装器自动解包元组的第一个元素。
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper


# HF Transformers 会覆盖名字包含 'gamma' 的参数，TIMM 的 LayerScale 正好用了 gamma。
# 所以需要把 gamma 重命名为 scale_factor，并替换 forward 方法。
#   TIMM: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   Transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960

def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    """替代 TIMM LayerScale 的 forward，使用 scale_factor 而非 gamma。"""
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    """对单个 LayerScale 模块应用 monkey-patch：重命名参数 + 替换 forward。"""
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# ============================================================================
# PrismaticVisionBackbone: 视觉编码器
# ============================================================================

class PrismaticVisionBackbone(nn.Module):
    """
    Prismatic 视觉骨干网络，处理图像特征提取。

    支持两种模式：
      单骨干（如仅 SigLIP）：一张图 → 一组建模特征
      融合骨干（SigLIP + DINOv2）：一张图 → 两组特征拼接，语义更丰富

    多图输入时，每张图都会被分别编码，然后沿 patch 维度拼接。
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Args:
            use_fused_vision_backbone: 是否使用双骨干融合（SigLIP + DINOv2）
            image_sizes: 每个骨干的输入图像尺寸
            timm_model_ids: 每个骨干的 TIMM 模型 ID
            timm_override_act_layers: 激活层覆盖（可选）
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # 默认输入图像数，后续可覆盖

        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic 模型最多支持 2 个（融合）视觉骨干！")

        # 创建主 featurizer（通常是 SigLIP）
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # 如果使用融合骨干，创建第二个 featurizer（通常是 DINOv2）
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim  # 总嵌入维度 = 两者之和

        # 对所有 LayerScale 模块打补丁
        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        创建 TIMM 视觉模型作为特征提取器。

        关键技巧：用 get_intermediate_layers 取倒数第二层的输出，
        而不是最后一层（CLS token 层），以保留更多空间信息。
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch：用倒数第二层的 patch features 替代默认 forward
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        """对主骨干和融合骨干的所有 LayerScale 模块打补丁。"""
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """返回每张图像输出的 patch 数量（不含 CLS token）。"""
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        """返回模型期望的输入图像数量。"""
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """设置输入图像数量。"""
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        视觉骨干的前向传播。

        单骨干、单图:
          pixel_values (B, 3, H, W) → featurizer → (B, num_patches, embed_dim)

        融合骨干、单图:
          pixel_values (B, 6, H, W) → 拆分为两通道组 → 分别编码 → 沿嵌入维拼接
          → (B, num_patches, embed_siglip + embed_dinov2)

        融合骨干、多图:
          pixel_values (B, 6*N, H, W) → 拆分为 N 张图 → 每张编码后沿 patch 维拼接
          → (B, num_patches * N, embed_total)
        """
        # === 单图模式 ===
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # 融合骨干：拆分为 SigLIP(3通道) 和 DINOv2(3通道)，分别编码后拼接
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
            return torch.cat([patches, patches_fused], dim=2)  # 沿嵌入维拼接

        # === 多图模式 ===
        else:
            assert self.use_fused_vision_backbone, "多图输入需要使用融合骨干！"

            # 拆分多张图（每张 6 通道：SigLIP 3ch + DINOv2 3ch）
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            all_patches = []
            for img in images:
                # 每张图再拆分为 SigLIP 和 DINOv2 通道
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)
                combined_patches = torch.cat([patches, patches_fused], dim=2)  # 沿嵌入维
                all_patches.append(combined_patches)

            # 所有图的 patch 沿序列维拼接
            return torch.cat(all_patches, dim=1)


# ============================================================================
# PrismaticProjector: 视觉→语言投影层
# ============================================================================

class PrismaticProjector(nn.Module):
    """
    将视觉特征从视觉编码器的维度投影到 LLM 的嵌入维度。

    单骨干模式: 2 层 MLP (vision_dim → llm_dim → llm_dim)
    融合骨干模式: 3 层 MLP (vision_dim → 4*vision_dim → llm_dim → llm_dim)
    """

    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        if not self.use_fused_vision_backbone:
            # 单骨干: 两层投影
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            # 融合骨干: 三层投影（因为融合后的视觉维度更大，需要更深的投影）
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# HuggingFace 模型基类
# ============================================================================

@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Prismatic VLM 的输出类：除了标准 LLM 输出外，额外包含投影后的视觉特征。"""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # VLM 特有：投影后的视觉 patch 特征
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    """
    Prismatic 预训练模型基类。

    注意：这个 HF 移植版本不是为从头训练设计的，只用于推理和微调！
    从头训练 VLM 请用原始代码库: https://github.com/TRI-ML/prismatic-vlms
    """
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        """权重初始化（仅用于微调场景，非从头训练的精确初始化）。"""
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
        """检查 LLM 是否支持 SDPA（PyTorch 2.0 的 scaled dot-product attention）。"""
        return self.language_model._supports_sdpa


# ============================================================================
# PrismaticForConditionalGeneration: 核心 VLM 模型
# ============================================================================

class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    """
    Prismatic VLM 主类，组装 Vision Backbone + Projector + LLM Backbone。

    提供:
      - forward(): 训练/推理的多模态前向传播
      - prepare_inputs_for_generation(): KV-cache 生成支持
    """

    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # 验证依赖版本
        if config.use_fused_vision_backbone is None:
            raise ValueError("缺少 config 字段 `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM 版本需 >= 0.9.10 且 < 1.0.0；如需支持最新 TIMM 版本请提交 GitHub Issue。"
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"期望 transformers==4.40.1 和 tokenizers==0.19.1，"
                f"但当前 transformers=={transformers.__version__} tokenizers=={tokenizers.__version__}；"
                f"依赖版本差异可能导致推理问题。"
            )

        # 1. 视觉骨干
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # 2. 视觉→语言投影层
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # 3. LLM 骨干（如 Llama-2）
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size  # 通常是 4096

        self.post_init()

    # === HuggingFace 标准接口 ===

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
        self.language_model.tie_weights()  # Llama-2 和 Mistral 不绑定权重（no-op）

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings
        return updated_embeddings

    # ========================================================================
    # 内部辅助方法：构建多模态输入
    # ========================================================================

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        用噪声动作嵌入替换 input_embeddings 中动作 token 位置的原始嵌入。

        用于 Diffusion 模式：将纯噪声（或当前去噪状态）的嵌入填入序列的动作占位符位置。

        Args:
            input_embeddings:       (B, seq_len, D) — 原始文本嵌入
            all_actions_mask:       (B, seq_len)   — 布尔掩码，True=动作位置
            noisy_action_features:  (B, K, D)      — 噪声动作的嵌入向量

        Returns:
            替换后的 input_embeddings，动作位置被噪声嵌入替代
        """
        new_input_embeddings = input_embeddings.clone()

        # 创建与 input_embeddings 同形状的零张量，作为接收容器
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # 批量索引：为每个样本构造 (batch_idx, seq_idx) 坐标对
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # 获取每个样本中动作 token 位置的索引
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # 将噪声嵌入放到正确的位置
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # 用掩码合并：动作位置用噪声嵌入，其余保留原文嵌入
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )
        return new_input_embeddings

    def _process_action_masks(self, labels):
        """
        从 labels 中提取动作掩码。

        动作在序列中占两个部分：
          - current_action_mask: 当前动作 token 位置
          - next_actions_mask:   下一帧动作 token 位置（序列预测目标）

        all_actions_mask = current | next，覆盖所有动作 token。
        """
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """
        处理视觉特征。

        FiLM 模式: 将语言嵌入注入视觉编码过程（条件化视觉特征）。
        非 FiLM:   标准视觉编码。
        """
        if use_film:
            patch_features = self.vision_backbone(pixel_values, language_embeddings)
        else:
            patch_features = self.vision_backbone(pixel_values)

        # 投影到 LLM 嵌入空间
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """
        将本体感知（proprioception）状态投影并拼接到视觉特征末尾。

        proprio: 机械臂的关节角度、夹爪状态等 → 投影到 llm_dim → 作为额外 token 拼入序列。
        """
        if proprio_projector is not None and proprio is not None:
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (B, proprio_dim)
            proprio_features = proprio_projector(proprio)                       # (B, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)                # (B, 1, llm_dim)

            # 拼接到视觉 patch 序列末尾（作为第 N+1 个"伪视觉 token"）
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _process_scene_tokens(self, projected_patch_embeddings, scene_tokens, lang_feat=None):
        """
        将 VGGT-Omega 场景 token（全局 3D 几何先验）拼接到视觉特征末尾。

        VGGT-Omega 的 register token 编码了多视图的全局 3D 几何信息，
        作为 VLA 策略的空间先验。

        Q-Former 模式 (scene_projector.use_qformer=True):
          N*16 VGGT registers → Q-Former 压缩 → K 个 task-conditioned tokens
          → FiLM 调制 → Modality Embedding → 拼入序列

        简单模式 (scene_projector.use_qformer=False):
          N*16 registers → Linear + LayerNorm 投影 → 拼入序列
        """
        if scene_tokens is not None:
            scene_embeddings = self.scene_projector(scene_tokens, lang_feat)
            return torch.cat([projected_patch_embeddings, scene_embeddings], dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """
        构建多模态嵌入和多模态注意力掩码。

        在 <BOS> token 之后插入视觉 patch 嵌入：
          [BOS] [patch_1 ... patch_N] [text_tokens ...]

        视觉 patch 的 attention_mask 全为 True（都参与注意力计算）。
        """
        # 为视觉 patch 创建注意力掩码（全 1）
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # 拼接: [BOS(1)] + [视觉 patches] + [文本(从第2个token开始)]
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
        构建多模态 labels。

        视觉 patch 位置的标签设为 IGNORE_INDEX（-100），
        因为视觉 token 不需要预测（只预测文本/动作 token）。
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

    # ========================================================================
    # forward(): 核心多模态前向传播
    # ========================================================================

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
        proprio=None,                       # 本体感知状态
        proprio_projector=None,             # 本体感知投影器
        noisy_actions=None,                 # 噪声动作（Diffusion 模式用）
        noisy_action_projector=None,        # 噪声动作投影器
        diffusion_timestep_embeddings=None, # 扩散时间步嵌入
        use_film: bool = False,             # 是否使用 FiLM 条件化
        scene_tokens: Optional[torch.Tensor] = None,  # VGGT-Omega 场景 token
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """
        多模态前向传播，处理三种模式：
          1. 缓存生成模式 (input_ids.shape[1] == 1): 单 token 自回归解码
          2. 纯文本模式 (pixel_values is None): 标准 LLM 前向
          3. 多模态模式: 图像 + 文本 → LLM

        多模态模式是核心，流程如下：
          文本输入 → token 嵌入 → 分离动作位置掩码
          图像输入 → 视觉编码 → 投影 → 拼接 proprio/scene/diffusion
          动作位置处理（零向量 or 噪声嵌入）
          拼接嵌入序列 → LLM 前向 → 输出
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 训练时不使用 KV cache（即使传入 use_cache=True）
        use_cache = use_cache and not self.training

        projected_patch_embeddings = None

        # ================================================================
        # 模式1: 缓存生成（自回归解码，每次只处理一个新 token）
        # ================================================================
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "缓存生成仅支持 batch_size=1！"
            assert past_key_values is not None, "缓存生成必须提供 past_key_values！"
            assert labels is None, "缓存生成不应传入 labels！"

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

        # ================================================================
        # 模式2: 纯文本（无图像输入）
        # ================================================================
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "纯文本模式需要 input_ids！"
            assert past_key_values is None, "纯文本模式不应有 past_key_values！"

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

        # ================================================================
        # 模式3: 多模态（图像 + 文本）— 训练和批量推理的核心路径
        # ================================================================
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "多模态前向不应有 past_key_values！"

            # --- Step 1: 文本 → token 嵌入 ---
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            # --- Step 2: 提取动作掩码，分离纯语言部分 ---
            all_actions_mask = self._process_action_masks(labels)

            # 纯语言嵌入（不含动作 token），用于 FiLM 条件化
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )

            # --- Step 3: 视觉编码 → 投影 ---
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

            # --- Step 4: 拼接额外 token（proprio / scene / diffusion timestep）---
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

            # 从语言嵌入中提取平均特征用于 Q-Former / FiLM 条件化
            lang_feat = language_embeddings.mean(dim=1) if language_embeddings.shape[1] > 0 else None

            projected_patch_embeddings = self._process_scene_tokens(
                projected_patch_embeddings, scene_tokens, lang_feat
            )

            if diffusion_timestep_embeddings is not None:
                projected_patch_embeddings = torch.cat(
                    (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
                )

            # --- Step 5: 处理动作 token 嵌入 ---
            if noisy_actions is not None:
                # Diffusion 模式: 用噪声动作嵌入替换动作位置的嵌入
                B = noisy_actions.shape[0]
                noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)
                noisy_action_features = noisy_action_projector(noisy_actions)  # 投影到 llm_dim
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, noisy_action_features
                )
            else:
                # L1 回归 / 离散模式: 动作位置填零（仅靠 positional embedding 区分）
                all_actions_mask = all_actions_mask.unsqueeze(-1)
                input_embeddings = input_embeddings * ~all_actions_mask

            # --- Step 6: 拼接多模态嵌入序列 ---
            # 最终序列: [BOS] [vision patches (+ proprio + scene + diff)] [text (+ zero/noise actions)]
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # --- Step 7: 构建 labels（视觉 patch 位置标记为忽略）---
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

            # --- Step 8: LLM 前向 ---
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,   # 直接传入嵌入，不用 token ID
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # 返回输出
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

    # ========================================================================
    # prepare_inputs_for_generation: KV-cache 自回归生成支持
    # ========================================================================

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
        为自回归生成准备输入（借鉴 LlamaForCausalLM，简化为 batch_size=1）。

        有缓存时只取最后一个 token；首次生成时如果传了 inputs_embeds 则优先使用。
        """
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # 有缓存：只取最后一个未处理的 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 如果传了 inputs_embeds 且是首次生成（无缓存），用嵌入；否则用 token ID
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 保留 pixel_values 和缓存信息
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


# ============================================================================
# OpenVLAForActionPrediction: VLA 动作预测模型
# ============================================================================

class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    """
    在 PrismaticForConditionalGeneration 基础上扩展动作预测能力。

    核心方法 predict_action() 是动作预测的总调度入口，统一了三种模式:
      1. 离散 token 预测 (action_head is None, noisy_action_projector is None)
      2. L1 回归预测     (action_head is not None, noisy_action_projector is None)
      3. 扩散去噪预测   (action_head is not None, noisy_action_projector is not None)
    """
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats  # 各数据集的归一化统计量

        # 离散预测模式下的动作分箱（与 action_tokenizer.py 一致）
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 恢复真实的词表大小（减去 pad_to_multiple_of 的填充）
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

        # 场景投影器：将 VGGT-Omega 的 3D 几何 token 投影到 LLM 空间
        # Q-Former 模式: N*16 VGGT registers → K 个 task-conditioned queries
        # 简单模式 (基线): N*16 registers → Linear + LayerNorm
        from prismatic.models.scene_projector import SceneProjector
        self.scene_projector = SceneProjector(
            scene_dim=2048,            # VGGT-Omega register token 维度
            llm_dim=self.llm_dim,     # LLM 隐藏维度（如 4096）
            num_queries=8,            # Q-Former 压缩目标：8 个 task-conditioned tokens
            num_layers=2,             # Q-Former 层数
            num_heads=8,              # 注意力头数
            qformer_dropout=0.1,      # Q-Former dropout
            film_bottleneck=256,      # FiLM 瓶颈维度
            use_qformer=True,         # True: 完整管线; False: 简单投影 (基线)
        )

    # ========================================================================
    # 辅助方法：准备动作预测的输入
    # ========================================================================

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """
        在输入序列末尾添加动作预测所需的占位 token 和停止 token。

        添加内容:
          1. ACTION_DIM * NUM_ACTIONS_CHUNK 个占位 token（动作位置）
          2. 1 个 STOP token（</s>），标记序列结束

        例如 LIBERO: 添加 7*8=56 个动作占位 + 1 个 </s> = 57 个 token
        """
        # 添加动作占位 token
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 添加停止 token（训练时非因果双向自注意力需要它）
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # 扩展注意力掩码
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """
        为动作预测创建伪 labels（用于计算动作掩码，非真正的训练 labels）。

        动作位置用任意非 IGNORE_INDEX 的 token 填充（让动作掩码机制能识别出来），
        最后一个位置设为 STOP token。
        """
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1]))
            .to(labels.device)
            .to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # 最后一个 token 是 STOP token
        labels[:, -1] = STOP_INDEX

        return labels

    # ========================================================================
    # 动作逆归一化
    # ========================================================================

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """
        将 [-1, 1] 空间的归一化动作还原到物理空间。

        根据训练时使用的归一化类型选择还原方式:
          BOUNDS:     action = 0.5 * (normalized + 1) * (max - min) + min
          BOUNDS_Q99: action = 0.5 * (normalized + 1) * (q99 - q01) + q01
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

        # 反归一化: [-1, 1] → [low, high]
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions

    # ========================================================================
    # 扩散预测子流程：50 步 DDIM 去噪
    # ========================================================================

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
        扩散去噪循环：从纯噪声开始，逐步去噪 50 步，得到干净动作。

        每步流程:
          1. 计算当前时间步嵌入
          2. 将当前噪声动作嵌入 LLM 空间
          3. 替换序列中的动作占位符
          4. LLM 前向 → 取动作位置的 hidden states
          5. MLP 预测噪声 → DDIM 去噪一步
        """
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise  # 起始: 纯高斯噪声

        for t in action_head.noise_scheduler.timesteps:
            # Step 1: 时间步编码
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # Step 2: 拼接时间步嵌入到视觉 token 后
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # Step 3: 将噪声动作投影到 LLM 嵌入空间
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # Step 4: 替换动作 token 位置的嵌入
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # Step 5: 构建多模态嵌入 → LLM 前向
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,      # 需要 hidden states
                return_dict=True,
            )

            # Step 6: 取动作位置的 hidden states → 预测噪声
            last_hidden_states = language_model_output.hidden_states[-1]
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, chunk_len * action_dim, D)

            # Step 7: MLP 预测噪声 → DDIM scheduler 去噪一步
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(
                noise_pred, t, curr_noisy_actions
            ).prev_sample

        # 最终 reshape 为 (NUM_ACTIONS_CHUNK, ACTION_DIM)
        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    # ========================================================================
    # L1 回归 / 离散预测子流程
    # ========================================================================

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
        L1 回归或离散 token 预测（非 Diffusion 模式）。

        流程:
          1. 动作位置填零 → 构建多模态嵌入
          2. LLM 前向 → 取动作位置的 hidden states
          3. 如果有 action_head → L1 回归（MLP 直接输出连续值）
             如果没有 → 离散 token 预测（argmax + bin_centers 查表）
        """
        # Step 1: 动作位置填零
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Step 2: 构建多模态嵌入 → LLM 前向
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

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

        # Step 3: 取动作位置的 hidden states
        last_hidden_states = language_model_output.hidden_states[-1]
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, chunk_len * action_dim, D)

        # Step 4: 根据是否有 action_head 选择预测方式
        if action_head is not None:
            # ====== OpenVLA-OFT: L1 回归 ======
            normalized_actions = action_head.predict_action(actions_hidden_states)  # MLP 直接输出
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # ====== 原始 OpenVLA: 离散 token 预测 ======
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)              # 取概率最大的 token
                .cpu()
                .numpy()
            )
            # 解码: token ID → bin index → bin center → 连续值
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    # ========================================================================
    # predict_action(): 动作预测的总调度入口 ★★★
    # ========================================================================

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        scene_tokens=None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        从输入序列预测动作，统一调度三种预测模式。

        三种模式的分发逻辑:
          use_diffusion = (noisy_action_projector is not None) and hasattr(action_head, "noise_scheduler")

          1. 离散 token 预测: action_head is None
          2. L1 回归:         action_head is not None, use_diffusion is False
          3. 扩散去噪:       use_diffusion is True

        完整流程:
          1. 确保 prompt 末尾有特殊空 token（''）
          2. 添加动作占位 token + STOP token
          3. 构建伪 labels（用于动作掩码）
          4. 文本嵌入 → 视觉编码 → 拼接 proprio + scene
          5. 根据模式调用不同的预测子流程
          6. 逆归一化: [-1, 1] → 物理尺度

        Args:
            input_ids: 输入 token IDs
            unnorm_key: 逆归一化用的数据集 key（如 "libero_spatial"）
            proprio: 本体感知状态
            proprio_projector: 本体感知投影器
            action_head: L1 回归头或 Diffusion 头
            noisy_action_projector: 噪声动作投影器（仅 Diffusion 模式）
            use_film: 是否使用 FiLM
            scene_tokens: VGGT-Omega 场景 token
            **kwargs: pixel_values, attention_mask

        Returns:
            (未归一化动作, 动作 hidden states)
        """
        # === 0. 确保 prompt 末尾有特殊空 token (token ID 29871) ===
        # 训练时冒号 (':') 后面会跟一个空 token，推理时也要匹配这个格式
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # === 1. 创建伪 labels（仅用于动作掩码计算，非训练用） ===
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # 计算 prompt token 数量（不含动作和 STOP token）
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        # === 2. 添加动作占位 token 和 STOP token ===
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # === 3. 文本 → 嵌入，获取动作掩码 ===
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # 纯语言嵌入（FiLM 用）
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # === 4. 视觉编码 → 投影 ===
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # === 5. 拼接 proprio ===
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # === 6. 拼接 VGGT-Omega 场景 token ===
        if scene_tokens is not None:
            lang_feat = language_embeddings.mean(dim=1) if language_embeddings.shape[1] > 0 else None
            projected_patch_embeddings = self._process_scene_tokens(
                projected_patch_embeddings, scene_tokens, lang_feat
            )

        # === 7. 判断预测模式 ===
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # === 8. 计算总 patch 数量（视觉 + proprio + scene + diffusion timestep） ===
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if scene_tokens is not None:
            # Q-Former 模式: K 个压缩 token；简单模式: N*16 个 token
            if self.scene_projector.use_qformer:
                NUM_PATCHES += self.scene_projector.num_queries  # K compressed tokens
            else:
                NUM_PATCHES += scene_tokens.shape[1]  # N * 16 raw tokens
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1  # 扩散时间步嵌入

        # === 9. 执行预测 ===
        if use_diffusion:
            # ----- 模式3: 扩散去噪 -----
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
                device=input_embeddings.device,
                dtype=input_embeddings.dtype,
            )  # 从纯高斯噪声开始
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings, all_actions_mask, noise, action_head,
                projected_patch_embeddings, labels, attention_mask,
                NUM_PATCHES, NUM_PROMPT_TOKENS, noisy_action_projector,
            )
        else:
            # ----- 模式1&2: 离散 token 或 L1 回归 -----
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings, all_actions_mask, projected_patch_embeddings,
                attention_mask, labels, NUM_PATCHES, NUM_PROMPT_TOKENS, action_head,
            )

        # === 10. 逆归一化: [-1, 1] → 物理空间 ===
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    # ========================================================================
    # 归一化统计量查询
    # ========================================================================

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """
        验证并解析逆归一化 key。

        如果只训练了一个数据集，可以省略 unnorm_key 自动推断；
        多数据集则必须显式指定。
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"模型在多个数据集上训练，请从以下选项中选择 unnorm_key: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"unnorm_key '{unnorm_key}' 不在可用统计量中，请从以下选择: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """获取策略动作空间的维度。"""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """获取指定数据集的动作统计量（用于逆归一化）。"""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
