"""
scene_projector.py

Projection layer for VGGT-Omega scene tokens (registers) into the LLM embedding space.
Scene tokens encode global 3D geometry of the multi-view input and serve as a
spatial prior for the VLA policy.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class SceneProjector(nn.Module):
    """Project VGGT-Omega scene tokens (per-frame registers) into LLM embedding space.

    VGGT-Omega outputs 16 register tokens per input frame (embed_dim=1024).
    These are flattened into [B, N*16, 1024] and projected to [B, N*16, llm_dim].
    """

    def __init__(self, scene_dim: int = 2048, llm_dim: int = 4096):
        super().__init__()
        self.projector = nn.Sequential(OrderedDict([
            ("scene_linear", nn.Linear(scene_dim, llm_dim, bias=True)),
            ("scene_norm", nn.LayerNorm(llm_dim)),
        ]))

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, N*16, 1024] — flattened registers from VGGT-Omega
        Returns:
            [B, N*16, llm_dim]
        """
        return self.projector(scene_tokens)
