"""
scene_qformer.py

Q-Former module for task-conditioned VGGT-Omega scene token selection and compression.

Core insight: VGGT-Omega register tokens encode 3D geometry in a distributed manner —
no single register carries complete geometric semantics. Q-Former's Self-Attention across
learnable queries enables collaborative synthesis of these distributed 3D encodings,
while Cross-Attention extracts only task-relevant geometric features conditioned on
language embeddings.

Architecture:
  VGGT scene tokens [B, N*16, 2048]
    → SceneProj (2048 → llm_dim)
    → Language-conditioned Query Bias
    → Q-Former Layers (Self-Attn → Cross-Attn → FFN) × num_layers
    → FiLM channel modulation
    → Modality Embedding (additive type marker)
    → Compressed tokens [B, K, llm_dim]
"""

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """
    Single Q-Former block: Self-Attention → Cross-Attention → FFN.

    Self-Attention operates on learnable query tokens (mutual refinement),
    Cross-Attention attends queries to VGGT scene tokens (information extraction),
    FFN transforms per-query features.

    Uses Pre-LN (LayerNorm before each sublayer) for training stability.
    """

    def __init__(self, hidden_dim: int = 4096, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Self-Attention (query tokens attend to each other)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.sa_norm = nn.LayerNorm(hidden_dim)
        self.sa_dropout = nn.Dropout(dropout)

        # Cross-Attention (queries attend to VGGT scene tokens)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ca_norm = nn.LayerNorm(hidden_dim)
        self.ca_dropout = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, scene_features: torch.Tensor,
                sa_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            queries:        [B, K, hidden_dim] — learnable query tokens
            scene_features: [B, N*16, hidden_dim] — projected VGGT scene tokens
            sa_mask:        [K, K] — optional causal/self-attention mask
        Returns:
            queries: [B, K, hidden_dim] — updated query tokens
        """
        # Self-Attention: queries refine each other
        residual = queries
        queries = self.sa_norm(queries)
        queries, _ = self.self_attn(queries, queries, queries, attn_mask=sa_mask)
        queries = residual + self.sa_dropout(queries)

        # Cross-Attention: queries extract info from scene tokens
        residual = queries
        queries = self.ca_norm(queries)
        queries, _ = self.cross_attn(queries, scene_features, scene_features)
        queries = residual + self.ca_dropout(queries)

        # Feed-Forward
        residual = queries
        queries = self.ffn_norm(queries)
        queries = self.ffn(queries)
        queries = residual + self.ffn_dropout(queries)

        return queries


class SceneQFormer(nn.Module):
    """
    Q-Former for task-conditioned VGGT scene token compression.

    Takes N*16 VGGT register tokens and compresses them to K learnable query tokens
    through language-conditioned query bias and multi-layer Q-Former blocks.

    Key features:
    - Language-conditioned query bias: language embedding → bias added to learnable queries,
      steering each query toward task-relevant geometric concepts
    - Self-Attention + Cross-Attention: queries collaboratively synthesize distributed 3D
      encodings before extracting task-relevant features from scene tokens
    - Single-stage end-to-end training: action L1 loss directly supervises geometry selection
    """

    def __init__(
        self,
        scene_dim: int = 2048,
        llm_dim: int = 4096,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scene_dim = scene_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        self.num_layers = num_layers

        # Scene token projection: VGGT register dim → LLM embedding dim
        self.scene_proj = nn.Linear(scene_dim, llm_dim, bias=True)
        self.scene_proj_norm = nn.LayerNorm(llm_dim)

        # Learnable query tokens (initialized with small values for stable training)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)

        # Language-conditioned query bias: steer queries toward task-relevant concepts
        self.lang_to_bias = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim * num_queries),
        )

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim=llm_dim, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.output_norm = nn.LayerNorm(llm_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize language-to-bias projection with near-zero weights for stable training start."""
        # Zero-init the last layer of lang_to_bias so queries start unbiased
        nn.init.zeros_(self.lang_to_bias[-1].weight)
        nn.init.zeros_(self.lang_to_bias[-1].bias)
        # Zero-init output_proj last layer for residual-like start
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, scene_tokens: torch.Tensor, lang_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, N*16, 2048] — raw VGGT-Omega register tokens
            lang_feat:    [B, llm_dim]    — averaged language embedding for task conditioning
        Returns:
            [B, num_queries, llm_dim] — compressed, task-conditioned geometry tokens
        """
        B = scene_tokens.shape[0]

        # Project scene tokens to LLM space
        scene_features = self.scene_proj(scene_tokens)
        scene_features = self.scene_proj_norm(scene_features)

        # Expand learnable queries to batch
        queries = self.query_tokens.expand(B, -1, -1)

        # Language-conditioned query bias: lang_feat → per-query additive bias
        bias = self.lang_to_bias(lang_feat).view(B, self.num_queries, self.llm_dim)
        queries = queries + bias

        # Pass through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, scene_features)

        # Residual output projection
        queries = queries + self.output_proj(queries)
        queries = self.output_norm(queries)

        return queries
