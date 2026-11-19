import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Continuous Sinusoidal Positional Embedding for arbitrary sequence lengths.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim).float() / half_dim)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, length, device=None, dtype=None):
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        sin_in = positions[:, None] * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(sin_in), torch.cos(sin_in)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        if dtype is not None:
            emb = emb.to(dtype=dtype)
        return emb.unsqueeze(0)  # [1, T, dim]


class TimestepEmbedding(nn.Module):
    """
    Continuous Sinusoidal Timestep Embedding followed by a 2-layer MLP.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.half_dim = d_model // 2
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, self.half_dim).float() / self.half_dim)
        self.register_buffer("inv_freq", inv_freq)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=True),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model, bias=True),
        )

    def forward(self, t):
        # t: [B] or [B, 1]
        t = t.view(-1).float()
        sin_in = t[:, None] * self.inv_freq[None, :] * 1000.0
        emb = torch.cat([torch.sin(sin_in), torch.cos(sin_in)], dim=-1)
        if emb.size(-1) < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.size(-1)))
        return self.mlp(emb)


class PointwiseDenseMLPBlock(nn.Module):
    """
    Pointwise Dense MLP Block (bias=False) with SiLU activation.
    """
    def __init__(self, d_model, mlp_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class BiMambaDecoderBlock(nn.Module):
    """
    BiMamba-2 Decoder Block without Cross-Attention:
    - Pure Sequence Mixing via Bidirectional Mamba-2 (Forward + Backward SSD SSM)
    - AdaLN Modulation via Timestep and Speaker Conditioning (ada_g alone)
    - Pointwise Dense MLP (2x hidden expansion by default)
    - Optional Gated Text Injection into RMS-normalized hidden states (320D per-channel
      negative-initialized sigmoid vector gates, enabled on later layers e.g. 3, 4, 5):
      h_norm1 = norm1(x) + sigmoid(gate1) * text_proj
      h_norm2 = norm2(x) + sigmoid(gate2) * text_proj
      (Residual highway x remains untouched.)
    """
    def __init__(
        self,
        d_model=320,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        mlp_mult=2,
        use_text_gate=True,
        gate_init=-2.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_text_gate = use_text_gate

        # AdaLN: projection strictly for global conditioning g (timestep + speaker embedding)
        self.ada_g = nn.Linear(d_model, 6 * d_model, bias=True)
        nn.init.zeros_(self.ada_g.weight)
        nn.init.zeros_(self.ada_g.bias)

        # Independent per-channel (d_model-dim) learnable sigmoid-gated vectors,
        # initialized negative so sigmoid(gate) starts small (< 0.5) on later layers
        if use_text_gate:
            self.gate1 = nn.Parameter(torch.full((d_model,), float(gate_init)))
            self.gate2 = nn.Parameter(torch.full((d_model,), float(gate_init)))
        else:
            self.gate1 = None
            self.gate2 = None

        # BiMamba-2
        self.norm1 = nn.RMSNorm(d_model)
        self.mamba_fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            bias=False,
            conv_bias=False,
        )
        self.mamba_bwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            bias=False,
            conv_bias=False,
        )
        self.bimamba_proj = nn.Sequential(
            nn.RMSNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model, bias=False),
        )

        # Dense MLP
        self.norm2 = nn.RMSNorm(d_model)
        self.dense_mlp = PointwiseDenseMLPBlock(
            d_model=d_model,
            mlp_hidden_dim=int(mlp_mult * d_model),
        )

    def forward(self, x, g, text_proj=None, audio_mask=None, **kwargs):
        """
        Args:
            x:          [B, T, d_model] - Audio sequence hidden state
            g:          [B, d_model]    - Combined timestep and speaker conditioning vector
            text_proj:  [B, T, d_model] - Globally precomputed duration-aligned text projection
            audio_mask: [B, T]          - Valid audio frames boolean mask
        Returns:
            x:          [B, T, d_model]
        """
        B, T, D = x.shape
        m_audio = audio_mask.unsqueeze(-1).to(dtype=x.dtype) if audio_mask is not None else None

        # Modulation strictly from timestep and speaker embedding (g)
        mod_g = self.ada_g(g).unsqueeze(1)  # [B, 1, 6 * d_model]
        scale1, shift1, alpha1, scale2, shift2, alpha2 = mod_g.chunk(6, dim=-1)

        # 1. BiMamba-2 Sub-block (per-channel sigmoid-gated RMS-normed text added to norm1(x))
        h_norm1 = self.norm1(x)
        if text_proj is not None and self.gate1 is not None:
            h_norm1 = h_norm1 + torch.sigmoid(self.gate1).view(1, 1, -1) * text_proj
        h_mod1 = h_norm1 * (1.0 + scale1) + shift1
        if m_audio is not None:
            h_mod1 = h_mod1 * m_audio

        fwd = self.mamba_fwd(h_mod1)

        # Reverse input for backward Mamba-2 pass
        rev_in = torch.flip(h_mod1, dims=[1])
        bwd_rev = self.mamba_bwd(rev_in)
        bwd = torch.flip(bwd_rev, dims=[1])

        cat_mamba = torch.cat([fwd, bwd], dim=-1)   # [B, T, 2 * d_model]
        h_mamba = self.bimamba_proj(cat_mamba)       # [B, T, d_model]
        if m_audio is not None:
            h_mamba = h_mamba * m_audio

        x = x + alpha1 * h_mamba

        # 2. Dense MLP Sub-block (per-channel sigmoid-gated RMS-normed text added to norm2(x))
        h_norm2 = self.norm2(x)
        if text_proj is not None and self.gate2 is not None:
            h_norm2 = h_norm2 + torch.sigmoid(self.gate2).view(1, 1, -1) * text_proj
        h_mod2 = h_norm2 * (1.0 + scale2) + shift2
        if m_audio is not None:
            h_mod2 = h_mod2 * m_audio

        h_mlp = self.dense_mlp(h_mod2)
        if m_audio is not None:
            h_mlp = h_mlp * m_audio

        x = x + alpha2 * h_mlp
        return x


class MambaDecoder(nn.Module):
    """
    MambaFlow-TTS Continuous Flow Matching Decoder:
    - Input: Concatenation of noisy continuous latent (64D) and duration-aligned text representation (192D) = 256D
    - Initial projection: Linear(256 -> d_model)
    - Stack of N BiMambaDecoderBlock layers (expand=2, mlp_mult=2)
    - Later layers (default: 3, 4, 5 for 6 blocks) additionally receive per-channel gated text_proj at norm1/norm2
    - Output projection: Linear(d_model -> 64) initialized to zero
    - Completely eliminates cross-attention layers in favor of direct sequence mixing
    """
    def __init__(
        self,
        d_in=256,
        d_cond=192,
        d_out=64,
        d_model=320,
        d_enc=192,
        d_spk=192,
        num_blocks=6,
        expand=2,
        d_state=64,
        d_conv=4,
        headdim=64,
        mlp_mult=2,
        text_inject_layers=None,
        gate_init=-2.0,
        sigma_min=1e-4,
        logit_normal_mean=0.0,
        logit_normal_std=1.0,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model
        self.d_enc = d_enc
        self.d_spk = d_spk
        self.num_blocks = num_blocks
        if text_inject_layers is None:
            self.text_inject_layers = set(range(num_blocks // 2, num_blocks))
        else:
            self.text_inject_layers = set(text_inject_layers)
        self.sigma_min = sigma_min
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.gradient_checkpointing = gradient_checkpointing

        # Positional Embedding for decoder sequence inputs
        self.pos_embed = SinusoidalPositionalEmbedding(d_model)

        # Learnable null text embedding for CFG unconditional conditioning
        self.null_text_emb = nn.Parameter(torch.zeros(1, 1, d_enc))

        # Conditioning Projections: Timestep + Speaker
        self.t_embedder = TimestepEmbedding(d_model)
        self.spk_proj = nn.Linear(d_spk, d_model, bias=False)
        self.cond_norm = nn.RMSNorm(d_model)

        # Globally precomputed projection for aligned text (shared across layers):
        # text_proj = RMSNorm(Linear(192 -> d))(aligned_text)
        self.text_proj = nn.Sequential(
            nn.Linear(d_enc, d_model, bias=True),
            nn.RMSNorm(d_model),
        )

        # Input & Output Projections
        self.in_proj = nn.Linear(d_in, d_model, bias=True)
        self.final_norm = nn.RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Decoder Blocks (expand=2, mlp_mult=2; gated text_proj injection on later layers e.g. 3, 4, 5)
        self.blocks = nn.ModuleList([
            BiMambaDecoderBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                mlp_mult=mlp_mult,
                use_text_gate=(idx in self.text_inject_layers),
                gate_init=gate_init,
                **kwargs,
            )
            for idx in range(num_blocks)
        ])

    def _sample_t(self, batch_size, device):
        normal_noise = torch.randn(batch_size, device=device)
        sample = normal_noise * self.logit_normal_std + self.logit_normal_mean
        t = torch.sigmoid(sample)
        return torch.clamp(t, min=1e-5, max=1.0 - 1e-5)

    def _forward_blocks(self, x, g, text_proj, audio_mask):
        for idx, block in enumerate(self.blocks):
            block_text = text_proj if idx in self.text_inject_layers else None
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, g, block_text, audio_mask, use_reentrant=False)
            else:
                x = block(x, g, block_text, audio_mask)
        return x

    def forward_velocity(self, x_t, aligned_text, t, spk_emb=None, audio_mask=None):
        """
        Core velocity prediction forward pass:
        Args:
            x_t:          [B, T, 64]   - Continuous latent state at timestep t
            aligned_text: [B, T, 192]  - Duration-aligned text representation
            t:            [B]          - Continuous timestep in [0, 1]
            spk_emb:      [B, 192]     - Normalized speaker embedding
            audio_mask:   [B, T]       - Valid audio frame boolean mask
        Returns:
            v_pred:       [B, T, 64]   - Predicted velocity field
        """
        B, T, _ = x_t.shape

        # In_proj: support 64D or 256D
        if self.in_proj.in_features == x_t.size(-1):
            h = self.in_proj(x_t)
        else:
            cat_in = torch.cat([x_t, aligned_text], dim=-1)
            h = self.in_proj(cat_in)

        # Decoder sequence positional embedding
        h = h + self.pos_embed(T, device=h.device, dtype=h.dtype)
        if audio_mask is not None:
            h = h * audio_mask.unsqueeze(-1).to(dtype=h.dtype)

        # Precompute Projections Globally:
        # 1. Project aligned text once before decoder blocks: text_proj = RMSNorm(Linear(192 -> d))(aligned_text)
        text_proj = self.text_proj(aligned_text)

        # 2. Combine global conditioning once: g = t_emb + spk_emb
        t_emb = self.t_embedder(t)
        if spk_emb is not None:
            spk_norm = F.normalize(spk_emb.float(), p=2, dim=-1).to(dtype=h.dtype)
            s_emb = self.spk_proj(spk_norm)
            g = self.cond_norm(t_emb + s_emb)
        else:
            g = self.cond_norm(t_emb)

        # Process through BiMamba-2 layers with AdaLN and norm-level gated text injection
        h = self._forward_blocks(h, g, text_proj, audio_mask)

        # Final projection to 64-dim velocity
        h = self.final_norm(h)
        v_pred = self.out_proj(h)

        if audio_mask is not None:
            m_c = audio_mask.unsqueeze(-1).to(dtype=v_pred.dtype)
            v_pred = v_pred * m_c

        return v_pred

    def forward(
        self,
        aligned_text=None,
        spk_emb=None,
        audio_mask=None,
        target=None,
        n_timesteps=25,
        temperature=1.0,
        solver="euler",
        t_end=1.0,
        cfg_scale=1.0,
        cond_drop_mask=None,
        **kwargs,
    ):
        """
        Unified Forward method for Training (CFM Velocity Matching) and Inference (ODE integration).
        """
        # Backward-compatibility fallback for text_repr
        if aligned_text is None and "text_repr" in kwargs:
            aligned_text = kwargs["text_repr"]

        if target is not None:
            # ======================== TRAINING MODE ========================
            B, T_targ, _ = target.shape
            device = target.device

            if audio_mask is None:
                full_audio_mask = torch.ones(B, T_targ, device=device, dtype=torch.bool)
            else:
                full_audio_mask = (audio_mask if audio_mask.dim() == 2 else audio_mask.squeeze(-1)).bool()
                if full_audio_mask.size(1) > T_targ:
                    full_audio_mask = full_audio_mask[:, :T_targ]
                elif full_audio_mask.size(1) < T_targ:
                    pad_len = T_targ - full_audio_mask.size(1)
                    full_audio_mask = F.pad(full_audio_mask, (0, pad_len), value=False)

            if aligned_text.size(1) != T_targ:
                min_T = min(aligned_text.size(1), T_targ)
                aligned_text = aligned_text[:, :min_T, :]
                target = target[:, :min_T, :]
                full_audio_mask = full_audio_mask[:, :min_T]
                T_targ = min_T

            # Sample Gaussian prior x0
            x_0 = torch.randn_like(target)

            # Sample continuous timesteps
            t = self._sample_t(B, device)

            # Optimal transport interpolation: x_t = (1 - (1 - sigma_min)t) * x_0 + t * target
            t_expand = t.view(B, 1, 1)
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * target

            # Target velocity: u_t = target - (1 - sigma_min) * x_0
            u_target = target - (1.0 - self.sigma_min) * x_0

            # Apply CFG conditioning dropout if specified using learnable null_text_emb
            spk_use = spk_emb
            text_use = aligned_text
            if cond_drop_mask is not None:
                m_drop = cond_drop_mask.view(-1, 1, 1)
                text_use = torch.where(m_drop, self.null_text_emb.to(dtype=text_use.dtype), text_use)
                if spk_use is not None:
                    spk_use = spk_use.masked_fill(cond_drop_mask.view(-1, 1), 0.0)

            v_pred = self.forward_velocity(
                x_t=x_t,
                aligned_text=text_use,
                t=t,
                spk_emb=spk_use,
                audio_mask=full_audio_mask,
            )

            return v_pred, u_target, t, full_audio_mask

        else:
            # ======================== INFERENCE MODE ========================
            # Integrate Continuous Flow Matching ODE from t=0 to t=t_end
            B, T_audio, _ = aligned_text.shape
            device = aligned_text.device

            if audio_mask is None:
                audio_mask = torch.ones(B, T_audio, device=device, dtype=torch.bool)

            # Sample Gaussian prior
            x = torch.randn(B, T_audio, self.d_out, device=device, dtype=aligned_text.dtype) * float(temperature)

            time_schedule = kwargs.get("time_schedule", kwargs.get("schedule", "uniform"))
            sway_coef = kwargs.get("sway_coef", -1.0)
            u = torch.linspace(0.0, float(t_end), n_timesteps + 1, device=device)
            u_norm = u / float(t_end)

            if time_schedule == "sway":
                t_norm = u_norm + float(sway_coef) * (torch.cos(torch.pi / 2.0 * u_norm) - 1.0 + u_norm)
                timesteps = torch.clamp(t_norm * float(t_end), min=0.0, max=float(t_end))
            elif time_schedule == "sine":
                t_norm = torch.sin(torch.pi / 2.0 * u_norm)
                timesteps = torch.clamp(t_norm * float(t_end), min=0.0, max=float(t_end))
            elif time_schedule == "sway_sine":
                t_sway = u_norm + float(sway_coef) * (torch.cos(torch.pi / 2.0 * u_norm) - 1.0 + u_norm)
                t_norm = torch.sin(torch.pi / 2.0 * t_sway)
                timesteps = torch.clamp(t_norm * float(t_end), min=0.0, max=float(t_end))
            else:
                timesteps = u

            for i in range(n_timesteps):
                t_val = timesteps[i]
                dt = (timesteps[i + 1] - timesteps[i]).item()
                t_vec = torch.full((B,), t_val.item(), device=device, dtype=torch.float32)

                if cfg_scale > 1.0 and spk_emb is not None:
                    # Conditional velocity
                    v_cond = self.forward_velocity(x, aligned_text, t_vec, spk_emb=spk_emb, audio_mask=audio_mask)
                    # Unconditional velocity with learnable null_text_emb (in-distribution prior)
                    uncond_text = self.null_text_emb.to(dtype=aligned_text.dtype).expand(B, T_audio, -1)
                    uncond_spk = torch.zeros_like(spk_emb)
                    v_uncond = self.forward_velocity(x, uncond_text, t_vec, spk_emb=uncond_spk, audio_mask=audio_mask)
                    v = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
                else:
                    v = self.forward_velocity(x, aligned_text, t_vec, spk_emb=spk_emb, audio_mask=audio_mask)

                if solver == "midpoint" and i < n_timesteps:
                    # Midpoint step
                    x_mid = x + 0.5 * dt * v
                    t_mid = t_vec + 0.5 * dt
                    if cfg_scale > 1.0 and spk_emb is not None:
                        v_cond_mid = self.forward_velocity(x_mid, aligned_text, t_mid, spk_emb=spk_emb, audio_mask=audio_mask)
                        v_uncond_mid = self.forward_velocity(x_mid, uncond_text, t_mid, spk_emb=uncond_spk, audio_mask=audio_mask)
                        v_mid = v_uncond_mid + float(cfg_scale) * (v_cond_mid - v_uncond_mid)
                    else:
                        v_mid = self.forward_velocity(x_mid, aligned_text, t_mid, spk_emb=spk_emb, audio_mask=audio_mask)
                    x = x + dt * v_mid
                else:
                    # Euler step
                    x = x + dt * v

            if audio_mask is not None:
                m_c = audio_mask.unsqueeze(-1).to(dtype=x.dtype)
                x = x * m_c

            return x
