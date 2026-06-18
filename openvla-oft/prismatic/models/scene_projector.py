"""
scene_projector.py

Projection layer for VGGT-Omega scene tokens (registers) into the LLM embedding space.
Uses a single Linear layer — VGGT tokens are already well-normalized,
and LLM internal RMSNorm handles any remaining distribution mismatch.
"""

import torch
import torch.nn as nn


class SceneProjector(nn.Module):
    """Project VGGT-Omega scene tokens into LLM embedding space via Linear layer.

    VGGT-Omega outputs 16 register tokens per input frame.
    These are concatenated along the feature dimension and projected to [B, N*16, llm_dim].
    
    No LayerNorm needed: VGGT tokens are already normalized, and the llama
    backbone applies RMSNorm after concatenation.
    """

    def __init__(self, scene_dim: int = 2048, llm_dim: int = 4096):
        super().__init__()
        self.projector = nn.Linear(scene_dim, llm_dim, bias=True)

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, N*16, scene_dim] — flattened registers from VGGT-Omega
        Returns:
            [B, N*16, llm_dim]
        """
        return self.projector(scene_tokens)
