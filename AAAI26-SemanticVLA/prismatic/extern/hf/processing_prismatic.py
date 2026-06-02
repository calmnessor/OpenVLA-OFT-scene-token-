"""
processing_prismatic.py

Prismatic VLM 的 HuggingFace 风格预处理器定义，继承自 `ProcessorMixin`。
默认配置为 `siglip-224px+7b`。
"""

from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === 图像处理 ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """将 PIL.Image 补零填充为正方形，在高度/宽度两侧对称添加边框。"""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    """Prismatic 图像处理器：封装 torchvision 变换，支持单/双视觉 backbone 和多图输入。"""
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        初始化 PrismaticImageProcessor，封装 torchvision 图像变换。
        变换由 TIMM 创建，并根据自定义 `image_resize_strategy` 进行调整。

        参数:
            use_fused_vision_backbone: 是否使用双视觉 backbone 融合（SigLIP + DINOv2）
            image_resize_strategy: 图像缩放策略，可选 < resize-naive | resize-crop | letterbox >
            input_sizes: [TIMM :: `data_cfg`] 输入图像尺寸，格式 (channels, width, height)
            interpolations: [TIMM :: `data_cfg`] 插值方法，默认 "bicubic"
            means: [TIMM :: `data_cfg`] 归一化均值，双 backbone 时传入两个元组
            stds: [TIMM :: `data_cfg`] 归一化标准差，双 backbone 时传入两个元组
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # 处理 None 默认值
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        # TIMM `data_cfg` 参数
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # 通过 TIMM 获取 torchvision 变换，解析为可序列化的函数式参数
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,           # 设为 1.0 跳过裁剪（初始 Resize 已设置 `input_size`）
                crop_mode="center",     # 默认裁剪模式 -- 当 `crop_pct == 1.0` 时不生效
                is_training=False,      # 加载变换时不使用数据增强
            )

            # [校验] 确保变换结构符合预期（Resize → CenterCrop → ToTensor → Normalize）
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"TIMM 图像变换结构/尺寸不符合预期: `{transform}`")

            # HF ImageProcessor 必须可 JSON 序列化，因此不能直接存储 torchvision 对象。
            # 改为解析变换参数，后续通过 `torchvision.transforms.functional` (tvf) 调用。
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # 根据 image_resize_strategy 调整缩放策略
            if self.image_resize_strategy == "resize-naive":
                # 简单缩放：强制正方形尺寸
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                # 信箱填充：先补零为正方形，再缩放，保留原始宽高比
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                # 缩放裁剪：先缩放后中心裁剪（TIMM 默认行为）
                pass
            else:
                raise ValueError(f"不支持的图像缩放策略 `{self.image_resize_strategy}`！")

        # 将 **kwargs 传递给父类
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """
        对单张图像应用函数式变换，等价于 TIMM 的 Compose([Resize → CenterCrop → ToTensor → Normalize])。

        双 backbone 模式下，两个 backbone 的图像 Tensor 沿 channel 维度堆叠，
        例如 SigLIP + DINOv2 → [6, 224, 224]，在模型侧再拆分。
        """
        # 如果启用了 letterbox 策略，先补零填充为正方形
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # [约定] 融合 Backbone 期望 "channel-stacked" 输入，在模型侧拆分处理
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)                              # → [C, H, W], 值域 [0, 1]
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])  # → 标准化
            imgs_t.append(img_idx_t)

        # [约定] imgs_t 是 Tensor 列表，每个形状 [3, input_size, input_size]，沿 dim=0 堆叠
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        预处理单张或批量图像。

        注意：与 `transformers::BaseImageProcessor` 不同，这里仅处理 PIL.Image.Image 类型以保持简洁。

        参数:
            images: 单张或批量的 PIL.Image.Image 实例
            return_tensors: 返回的 BatchFeature 张量格式（如 "pt" 表示 torch）；为 None 时返回 np.ndarray

        返回:
            `transformers::BatchFeature` 实例，包含单个键 "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # 对每张图调用 apply_transform（返回 torch.Tensor 列表），然后堆叠为 batch 维度
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # 返回 BatchFeature；为兼容性，构造函数期望 Dict[str, np.ndarray]，因此做转换
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        """便捷调用入口，等同于 self.preprocess(images, **kwargs)。"""
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor：同时封装 ImageProcessor 和 Tokenizer ===
# 参考: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    """Prismatic 多模态处理器，组合图像预处理和文本分词两个子处理器。"""
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        预处理批量的文本和图像，用于 Prismatic VLM 前向传播。

        文本通过底层 LLM 的 tokenizer 编码，图像通过 PrismaticImageProcessor 处理。

        参数:
            text: 待编码的文本（或文本列表）
            images: 待预处理的 PIL.Image.Image（或图像列表）
            padding: 序列填充策略，True="longest" | "max_length" | False
            truncation: 输出序列的截断策略，需指定 `max_length`
            max_length: 截断的最大 token 数
            return_tensors: 返回的张量类型（通常为 "pt" 或 TensorType.PYTORCH）

        返回:
            BatchFeature，包含 `input_ids`、`attention_mask` 和 `pixel_values`
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [校验] 图像和文本输入数量必须匹配
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("批次格式错误：图像数量与文本输入数量必须一致！")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer 分发工具方法，详见 `PreTrainedTokenizerBase` 文档 ===

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        """批量解码 token ID 序列为文本字符串。"""
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        """解码单个 token ID 序列为文本字符串。"""
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        """返回模型期望的所有输入名称（tokenizer + image_processor 的输入名并集）。"""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
