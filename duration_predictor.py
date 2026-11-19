import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    3-Layer ResNet-Conv1D Per-Phoneme Duration Predictor with Speaker Conditioning:
    - Input: Concatenation of 192D text representation and 192D projected speaker embedding (384D total)
    - Speaker Projection: Linear(d_spk -> dims) + RMSNorm on normalized average speaker embedding
    - Hidden Dimension: 256D
    - Layer 1: Conv1D (384 -> 256, k=3, padding=1) -> LayerNorm(256) -> SiLU -> Dropout(0.1)
    - Layer 2: Conv1D (256 -> 256, k=3, padding=1) -> LayerNorm(256) -> SiLU -> Dropout(0.1) + Residual
    - Layer 3: Conv1D (256 -> 256, k=3, padding=1) -> LayerNorm(256) -> SiLU -> Dropout(0.1) + Residual
    - Output Head: Linear(256, 1) predicting per-phoneme continuous log1p(duration) in frames.
    - Inference: Clamped expm1 integer durations per phoneme, summed for utterance-level length.
    """
    def __init__(
        self,
        dims=192,
        d_spk=192,
        hidden_dim=256,
        kernel_size=3,
        dropout_p=0.1,
        **kwargs,
    ):
        super().__init__()
        self.dims = dims
        self.d_spk = kwargs.get("d_spk", d_spk)
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        # Speaker projection: normalized speaker embedding -> dims
        self.spk_proj = nn.Linear(self.d_spk, dims, bias=True)

        # RMSNorm on text and speaker representations to maintain stable dynamic range
        self.norm_text = nn.RMSNorm(dims)
        self.norm_spk = nn.RMSNorm(dims)

        self.in_dims = dims + dims  # 192D text + 192D spk = 384D

        # Layer 1: in_dims -> hidden_dim Conv -> LayerNorm -> SiLU -> Dropout
        self.conv1 = nn.Conv1d(
            self.in_dims,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, bias=False)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_p)

        # Layer 2: hidden_dim -> hidden_dim Conv -> LayerNorm -> SiLU -> Dropout + Residual
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, bias=False)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_p)

        # Layer 3: hidden_dim -> hidden_dim Conv -> LayerNorm -> SiLU -> Dropout + Residual
        self.conv3 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.norm3 = nn.LayerNorm(hidden_dim, bias=False)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(dropout_p)

        # Output linear head predicting per-phoneme continuous log1p(duration)
        self.linear = nn.Linear(hidden_dim, 1, bias=True)

    def forward(
        self,
        mu_text,
        spk_emb=None,
        mask=None,
        length_scale=1.0,
        **kwargs,
    ):
        """
        Args:
            mu_text:      [B, T_text, 192] - Text representation (detached enc_out)
            spk_emb:      [B, 192]         - Normalized speaker embedding
            mask:         [B, T_text]      - Text token boolean mask (True = valid token)
            length_scale: float            - Duration scaling factor during inference
        Returns:
            dur_int:       [B, T_text]     - Integer frame counts per phoneme (clamped >= 1 for valid tokens)
            dur_pred:      [B, T_text]     - Continuous log1p(duration) predictions per phoneme
            pred_frames:   [B]             - Total predicted utterance frames (sum of dur_int)
        """
        if spk_emb is None and "spk" in kwargs:
            spk_emb = kwargs["spk"]
        if spk_emb is None and "speaker_emb" in kwargs:
            spk_emb = kwargs["speaker_emb"]

        x_det = mu_text.detach()
        B, T_text, _ = x_det.shape

        # Apply RMSNorm to text representation
        x_norm = self.norm_text(x_det)

        # Compute normalized speaker summary
        if spk_emb is not None:
            if spk_emb.ndim == 1:
                spk_emb = spk_emb.unsqueeze(0)
            spk_norm = F.normalize(spk_emb.float(), p=2, dim=-1).to(dtype=x_det.dtype)
            spk_summary = self.norm_spk(self.spk_proj(spk_norm))  # [B, 192]
        else:
            spk_summary = torch.zeros(B, self.dims, device=x_det.device, dtype=x_det.dtype)

        # Broadcast normalized speaker summary across text tokens (192D + 192D = 384D)
        spk_exp = spk_summary.unsqueeze(1).expand(-1, T_text, -1)  # [B, T_text, 192]
        h = torch.cat([x_norm, spk_exp], dim=-1)                   # [B, T_text, 384]

        m_c = mask.unsqueeze(-1).to(dtype=h.dtype) if (mask is not None and mask.dim() == 2) else (mask.to(dtype=h.dtype) if mask is not None else None)

        # Layer 1
        if m_c is not None:
            h = h * m_c
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.drop1(h)
        if m_c is not None:
            h = h * m_c

        # Layer 2
        res2 = h
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.drop2(h)
        h = h + res2
        if m_c is not None:
            h = h * m_c

        # Layer 3
        res3 = h
        h = self.conv3(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm3(h)
        h = self.act3(h)
        h = self.drop3(h)
        h = h + res3
        if m_c is not None:
            h = h * m_c

        # Output projection per phoneme -> [B, T_text]
        dur_pred = self.linear(h).squeeze(-1)  # [B, T_text]

        if mask is not None:
            m_b = mask.bool() if mask.dim() == 2 else mask.squeeze(-1).bool()
            dur_pred = dur_pred.masked_fill(~m_b, 0.0)
        else:
            m_b = torch.ones(B, T_text, device=x_det.device, dtype=torch.bool)

        # Convert continuous log1p to integer frame counts: >= 1 for valid tokens
        dur_int = torch.clamp(
            torch.round(torch.expm1(dur_pred.clamp(min=0.0)) * float(length_scale)),
            min=1.0,
        ).long()
        dur_int = dur_int.masked_fill(~m_b, 0)

        # Sum of phoneme durations gives total utterance-level length
        pred_frames = dur_int.sum(dim=-1).clamp(min=1)  # [B]

        return dur_int, dur_pred, pred_frames


# Backward-compatibility alias
duration = DurationPredictor
DurationPredictor3Layer = DurationPredictor
