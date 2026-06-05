"""
scene_projector.py

Task-conditioned scene token pipeline: Q-Former compression → FiLM modulation → Modality Embedding.

Architecture:
  VGGT scene tokens [B, N*16, 2048] + lang_feat [B, llm_dim]
    → SceneQFormer: compress N*16 → K query tokens via Self-Attn + Cross-Attn
    → FiLM: language-conditioned per-channel modulation (γ, β from lang_feat)
    → Modality Embedding: additive type marker "this is VGGT geometry-semantic token"
    → Output: [B, K, llm_dim]

Design decisions:
  - FiLM uses bottleneck (llm_dim → 256 → llm_dim) for γ,β to prevent overfitting
  - FiLM is zero-initialized (identity at training start) to preserve Q-Former output
  - Modality Embedding uses zero-initialized learned vector for same reason
  - Two modes: Q-Former (compression) and Simple (baseline, no compression)
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from .scene_qformer import SceneQFormer


class SceneTokenFiLM(nn.Module):
    """
    Language-conditioned Feature-wise Linear Modulation for scene tokens.

    Applies per-channel scaling and shifting conditioned on language:
        x = (1 + γ) ⊙ x + β

    γ and β are learned projections from the average language embedding, passed
    through a bottleneck to prevent overfitting. Both are zero-initialized so
    the modulation starts as an identity transformation.

    Unlike the ViT FiLM which modulates intermediate block features, this FiLM
    operates on the final compressed Q-Former output tokens.
    """

    def __init__(self, llm_dim: int = 4096, bottleneck: int = 256):
        super().__init__()
        self.llm_dim = llm_dim

        # Bottleneck: llm_dim → bottleneck → llm_dim for both γ and β
        self.scale_fc1 = nn.Linear(llm_dim, bottleneck)
        self.scale_fc2 = nn.Linear(bottleneck, llm_dim)
        self.shift_fc1 = nn.Linear(llm_dim, bottleneck)
        self.shift_fc2 = nn.Linear(bottleneck, llm_dim)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        """Zero-initialize final projection layers for identity at training start."""
        nn.init.zeros_(self.scale_fc2.weight)
        nn.init.zeros_(self.scale_fc2.bias)
        nn.init.zeros_(self.shift_fc2.weight)
        nn.init.zeros_(self.shift_fc2.bias)

    def forward(self, scene_tokens: torch.Tensor, lang_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, K, llm_dim] — compressed Q-Former output tokens
            lang_feat:    [B, llm_dim]   — averaged language embedding
        Returns:
            [B, K, llm_dim] — language-modulated scene tokens
        """
        gamma = self.scale_fc2(self.act(self.scale_fc1(lang_feat)))  # [B, llm_dim]
        beta = self.shift_fc2(self.act(self.shift_fc1(lang_feat)))   # [B, llm_dim]

        # Apply FiLM: x = (1 + γ) * x + β  (per-channel, applied batch-wise)
        gamma = gamma.unsqueeze(1)  # [B, 1, llm_dim]
        beta = beta.unsqueeze(1)    # [B, 1, llm_dim]

        return scene_tokens * (1 + gamma) + beta


class SceneProjector(nn.Module):
    """
    Project and modulate VGGT-Omega scene tokens into LLM embedding space.

    Two modes:
      use_qformer=True:  Full pipeline with Q-Former compression, FiLM modulation,
                         and modality embedding. Compresses N*16 → K tokens.
      use_qformer=False: Simple Linear + LayerNorm projection (baseline).
                         N*16 → N*16 tokens.

    The pipeline in Q-Former mode:
      1. SceneQFormer: N*16 VGGT registers → K learnable queries with language bias
      2. SceneTokenFiLM: language-conditioned channel modulation
      3. Modality Embedding: additive type marker to distinguish geometry tokens
    """

    def __init__(
        self,
        scene_dim: int = 2048,
        llm_dim: int = 4096,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        qformer_dropout: float = 0.1,
        film_bottleneck: int = 256,
        use_qformer: bool = True,
    ):
        super().__init__()
        self.scene_dim = scene_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        self.use_qformer = use_qformer

        if use_qformer:
            # Q-Former: compress N*16 → K task-conditioned tokens
            self.qformer = SceneQFormer(
                scene_dim=scene_dim,
                llm_dim=llm_dim,
                num_queries=num_queries,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=qformer_dropout,
            )
            # FiLM: language-conditioned channel modulation
            self.film = SceneTokenFiLM(llm_dim=llm_dim, bottleneck=film_bottleneck)
            # Modality Embedding: marks tokens as "VGGT geometry-semantic"
            self.modality_embedding = nn.Parameter(torch.zeros(1, 1, llm_dim))
        else:
            # Simple baseline: Linear + LayerNorm
            self.projector = nn.Sequential(OrderedDict([
                ("scene_linear", nn.Linear(scene_dim, llm_dim, bias=True)),
                ("scene_norm", nn.LayerNorm(llm_dim)),
            ]))

    @property
    def output_token_count(self) -> int:
        """Returns the number of output tokens from this projector."""
        return self.num_queries if self.use_qformer else None  # None means no compression

    def forward(self, scene_tokens: torch.Tensor, lang_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, N*16, 2048] — raw VGGT-Omega register tokens
            lang_feat:    [B, llm_dim]    — averaged language embedding (required for Q-Former mode)
        Returns:
            If use_qformer: [B, K, llm_dim] — compressed, task-conditioned, modality-marked tokens
            Else:           [B, N*16, llm_dim] — projected tokens (no compression)
        """
        if self.use_qformer:
            assert lang_feat is not None, "lang_feat required for Q-Former mode"

            # Step 1: Q-Former compression with language-conditioned query bias
            compressed = self.qformer(scene_tokens, lang_feat)  # [B, K, llm_dim]

            # Step 2: FiLM channel modulation
            compressed = self.film(compressed, lang_feat)  # [B, K, llm_dim]

            # Step 3: Add modality embedding (marks tokens as geometry-semantic type)
            compressed = compressed + self.modality_embedding  # [B, K, llm_dim]

            return compressed
        else:
            return self.projector(scene_tokens)
