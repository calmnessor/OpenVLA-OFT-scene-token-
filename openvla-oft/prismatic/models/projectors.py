"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class GeoRegisterPredictor(nn.Module):
    """
    Predicts lightweight geometry register tokens from the VLA's projected visual tokens.

    The module uses a small set of learnable queries that cross-attend to the visual
    tokens, mirroring the role of VGGT-Omega scene/register tokens while keeping
    inference cheap.
    """

    def __init__(
        self,
        llm_dim: int,
        num_registers: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.num_registers = num_registers

        self.register_queries = nn.Parameter(torch.empty(1, num_registers, llm_dim))
        nn.init.trunc_normal_(self.register_queries, std=0.02)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(llm_dim),
                        "visual_norm": nn.LayerNorm(llm_dim),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=llm_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(llm_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(llm_dim, mlp_ratio * llm_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_ratio * llm_dim, llm_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = visual_tokens.shape[0]
        geo_tokens = self.register_queries.expand(batch_size, -1, -1).to(dtype=visual_tokens.dtype)

        for layer in self.layers:
            query = layer["query_norm"](geo_tokens)
            key_value = layer["visual_norm"](visual_tokens)
            attn_out, _ = layer["cross_attn"](query=query, key=key_value, value=key_value, need_weights=False)
            geo_tokens = geo_tokens + attn_out
            geo_tokens = geo_tokens + layer["ffn"](layer["ffn_norm"](geo_tokens))

        return self.out_norm(geo_tokens)


class TeacherRegisterProjector(nn.Module):
    """Projects cached VGGT-Omega scene registers into the VLA hidden dimension."""

    def __init__(self, teacher_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(teacher_dim),
            nn.Linear(teacher_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, teacher_registers: torch.Tensor) -> torch.Tensor:
        return self.net(teacher_registers)


class DepthAuxHead(nn.Module):
    """
    Lightweight patch-level depth classifier for training-only auxiliary supervision.

    It consumes projected visual tokens and predicts quantized depth bins for each
    visual patch token. The head is discarded at inference.
    """

    def __init__(self, llm_dim: int, num_depth_bins: int = 32) -> None:
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.net = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, num_depth_bins),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        param_dtype = self.net[0].weight.dtype
        return self.net(visual_tokens.to(dtype=param_dtype))


def geometry_register_loss(
    student_registers: torch.Tensor,
    teacher_registers: torch.Tensor,
    relation_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Cosine feature matching plus relation matching between student and teacher registers.

    Relation matching compares Gram matrices, making the loss less brittle when
    individual register slots are not perfectly semantically aligned.
    """

    if teacher_registers.dim() == 4:
        teacher_registers = teacher_registers.mean(dim=1)
    if teacher_registers.shape[1] == student_registers.shape[1] + 1:
        # VGGT-Omega exposes [camera_token, scene_register_0, ...].
        # Distillation targets only the scene registers.
        teacher_registers = teacher_registers[:, 1:]
    if student_registers.shape[1] != teacher_registers.shape[1]:
        raise ValueError(
            f"student/teacher register counts differ: {student_registers.shape[1]} vs {teacher_registers.shape[1]}"
        )

    student = F.normalize(student_registers.float(), dim=-1)
    teacher = F.normalize(teacher_registers.float(), dim=-1)
    feature_loss = 1.0 - (student * teacher).sum(dim=-1).mean()

    student_rel = torch.bmm(student, student.transpose(1, 2))
    teacher_rel = torch.bmm(teacher, teacher.transpose(1, 2))
    relation_loss = F.mse_loss(student_rel, teacher_rel)

    loss = feature_loss + relation_weight * relation_loss
    metrics = {
        "geo_register_feature_loss": feature_loss.detach().item(),
        "geo_register_relation_loss": relation_loss.detach().item(),
    }
    return loss, metrics


def depth_aux_loss(
    depth_logits: torch.Tensor,
    teacher_depth: torch.Tensor,
    num_depth_bins: int = 32,
    min_depth: float = 1e-3,
    max_depth: float = 100.0,
    num_images: int = 1,
) -> torch.Tensor | None:
    """
    Converts cached teacher depth maps to log-depth bins and supervises visual tokens.

    Depth is aligned per input image. If the visual token count cannot be
    factored into equal square grids per image, the loss returns None so callers
    can skip unsafe supervision instead of silently misaligning labels.
    """

    if teacher_depth.dim() == 5 and teacher_depth.shape[-1] == 1:
        teacher_depth = teacher_depth.squeeze(-1)
    if teacher_depth.dim() == 3:
        teacher_depth = teacher_depth.unsqueeze(1)
    if teacher_depth.dim() != 4:
        raise ValueError(f"teacher_depth must have shape [B,H,W] or [B,V,H,W], got {tuple(teacher_depth.shape)}")

    batch_size, num_tokens, _ = depth_logits.shape
    if num_images <= 0 or num_tokens % num_images != 0:
        return None

    patches_per_image = num_tokens // num_images
    grid_size = int(patches_per_image**0.5)
    if grid_size * grid_size != patches_per_image:
        return None

    if teacher_depth.shape[1] < num_images:
        if teacher_depth.shape[1] == 1:
            teacher_depth = teacher_depth.expand(-1, num_images, -1, -1)
        else:
            return None
    teacher_depth = teacher_depth[:, :num_images]

    depth = teacher_depth.reshape(batch_size * num_images, 1, *teacher_depth.shape[-2:])
    depth = depth.float().clamp(min=min_depth, max=max_depth)
    depth = F.interpolate(depth, size=(grid_size, grid_size), mode="area")
    depth = depth.reshape(batch_size, num_images * patches_per_image)

    log_depth = torch.log(depth)
    log_min = torch.log(torch.tensor(min_depth, device=depth.device, dtype=depth.dtype))
    log_max = torch.log(torch.tensor(max_depth, device=depth.device, dtype=depth.dtype))
    labels = ((log_depth - log_min) / (log_max - log_min + 1e-8) * (num_depth_bins - 1)).long()
    labels = labels.clamp(0, num_depth_bins - 1)

    return F.cross_entropy(depth_logits.float().reshape(-1, num_depth_bins), labels.reshape(-1))
