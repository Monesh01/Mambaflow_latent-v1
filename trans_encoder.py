import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from x_transformers import Encoder


class XTEncoder(nn.Module):
    """
    6-Stack Transformer Text Encoder with Rotary Position Embeddings (RoPE), RMSNorm, and Flash Attention.
    Pure multi-head self-attention architecture for high-fidelity linguistic representation (dim=192, depth=6, heads=6, ff_mult=3).
    """
    def __init__(
        self,
        dim=192,
        depth=6,
        heads=6,
        ff_mult=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        layer_dropout=0.0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads if heads is not None else max(1, dim // 32),
            ff_swish=True,
            ff_mult=ff_mult,
            attn_flash=True,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            ff_no_bias=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            layer_dropout=layer_dropout,
        )

    def _encoder_step(self, x, mask=None):
        return self.encoder(x, mask=mask)

    def forward(self, x, mask=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint.checkpoint(self._encoder_step, x, mask, use_reentrant=False)
        return self.encoder(x, mask=mask)


# Export alias
TextEncoder = XTEncoder