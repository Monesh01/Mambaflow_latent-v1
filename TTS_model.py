import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
try:
    from .trans_encoder import XTEncoder
    from .duration_predictor import duration
    from .mamba_decoder import MambaDecoder, SinusoidalPositionalEmbedding
except (ImportError, ValueError):
    from trans_encoder import XTEncoder
    from duration_predictor import duration
    from mamba_decoder import MambaDecoder, SinusoidalPositionalEmbedding


class SoftBoundaryResidualConv1d(nn.Module):
    """
    Residual 1D Convolution applied right after duration-based repeat_interleave
    to smooth hard discrete phoneme boundaries into soft continuous transitions.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: [B, T, dim]
        res = self.conv(x.transpose(1, 2)).transpose(1, 2)
        res = self.act(res)
        res = self.norm(res)
        return x + res


class MambaFlowTTSModel(nn.Module):
    """
    MambaFlow-TTS Non-Autoregressive Continuous Flow Matching Architecture:
    - 6-Stack Transformer Text Encoder (XTEncoder: 192-dim, 6 layers, 6 heads, ff_mult=3, RoPE, RMSNorm)
    - 3-Layer Speaker-Conditioned Per-Phoneme Duration Predictor (192D text + 192D spk -> 256D hidden -> per-phoneme continuous log1p)
    - Length Regulation: repeat_interleave expands T_text, 192D -> T_audio, 192D (strict monotonic time progression, no cross-attention collapse)
    - 6-Layer BiMamba-2 Decoder (d_dec=320 or 384, expand=2, mlp_mult=2):
        * Channel concatenation: [x_t (64D) ; H_aligned (192D)] -> 256D in_proj
        * Bidirectional Mamba-2 Sequence Mixing (O(T) linear time/space complexity)
        * Decoupled Dual Modulation on Timestep and Speaker Embedding
        * Pointwise Dense MLP (2x hidden dim, bias=False)
        * Later-layer (3, 4, 5) norm-level 320D vector-gated text_proj injection (negative gate init)
        * Zero-initialized velocity projection head out_proj (Linear(d_model -> 64))
    """
    def __init__(
        self,
        encoder=None,
        duration_module=None,
        decoder_module=None,
        n_vocab=179,
        d_text=None,
        d_enc=192,
        d_codec=64,
        d_dec=320,
        expand=2,
        num_dec_blocks=6,
        emb_dropout=0.0,
        cfg_dropout=0.15,
        gradient_checkpointing=False,
        logit_normal_mean=0.0,
        logit_normal_std=1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_codec = d_codec
        self.d_dec = d_dec
        self.expand = expand
        self.num_dec_blocks = num_dec_blocks
        self.d_text = d_enc if d_text is None else d_text
        self.cfg_dropout = cfg_dropout
        self.n_vocab = n_vocab

        self.txt_emb = nn.Embedding(n_vocab, self.d_text)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 6-Stack Transformer Text Encoder with Pure Attention (192-dim, 6 layers, 6 heads, ff_mult=3)
        self.encoder = encoder if encoder is not None else XTEncoder(
            dim=d_enc,
            depth=6,
            heads=6 if d_enc == 192 else max(1, d_enc // 32),
            ff_mult=3,
            gradient_checkpointing=gradient_checkpointing,
            attn_dropout=0.0,
            ff_dropout=0.0,
            layer_dropout=0.0,
        )

        # 3-Layer Speaker-Conditioned Per-Phoneme Duration Predictor
        dur_hidden = kwargs.get("duration_hidden_dim", 256)
        d_spk = kwargs.get("d_spk", 192)
        self.d_spk = d_spk
        self.duration = duration_module if duration_module is not None else duration(
            dims=d_enc,
            d_spk=d_spk,
            hidden_dim=dur_hidden,
            kernel_size=3,
            dropout_p=0.1,
        )

        # Residual 1D Convolution right after repeat_interleave to provide soft phoneme boundaries
        self.post_interleave_conv = SoftBoundaryResidualConv1d(self.d_enc, kernel_size=3)

        # Positional embedding for aligned text sequence
        self.pos_embed = SinusoidalPositionalEmbedding(self.d_enc)

        decoder_kwargs = kwargs.copy()
        for k in ["duration_hidden_dim", "d_spk", "d_enc", "d_codec", "d_dec", "gate_init", "text_inject_layers", "num_dec_blocks", "gradient_checkpointing", "logit_normal_mean", "logit_normal_std", "ca_layer_indices", "use_gaussian_mask", "mlp_mult", "d_state", "expand", "aux_layer_idx", "use_aux_head", "aux_ce_weight"]:
            decoder_kwargs.pop(k, None)

        mlp_mult = kwargs.get("mlp_mult", 2)
        d_state = kwargs.get("d_state", 64)
        gate_init = kwargs.get("gate_init", -2.0)
        text_inject_layers = kwargs.get("text_inject_layers", None)

        # BiMamba-2 Decoder (d_in = 64D latent + 192D aligned text = 256D, expand=2, mlp_mult=2)
        self.decoder = decoder_module if decoder_module is not None else MambaDecoder(
            d_in=d_codec + d_enc,
            d_cond=d_enc,
            d_out=d_codec,
            d_model=d_dec,
            expand=expand,
            d_enc=d_enc,
            d_spk=d_spk,
            num_blocks=num_dec_blocks,
            gradient_checkpointing=gradient_checkpointing,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            mlp_mult=mlp_mult,
            text_inject_layers=text_inject_layers,
            gate_init=gate_init,
            d_state=d_state,
            **decoder_kwargs,
        )

        self.gradient_checkpointing = gradient_checkpointing

    @property
    def null_text_emb(self):
        return getattr(self.decoder, "null_text_emb", None)

    @property
    def gradient_checkpointing(self):
        return getattr(self, "_gradient_checkpointing", False)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = bool(value)
        if hasattr(self.encoder, "gradient_checkpointing"):
            self.encoder.gradient_checkpointing = bool(value)
        if hasattr(self.decoder, "gradient_checkpointing"):
            self.decoder.gradient_checkpointing = bool(value)

    def forward(
        self,
        x,
        target_latent=None,
        durations=None,
        spk_emb=None,
        mask=None,
        audio_mask=None,
        n_timesteps=25,
        temperature=1.0,
        length_scale=1.0,
        solver="euler",
        time_schedule="uniform",
        sway_coef=-1.0,
        t_end=1.0,
        skip_decoder=False,
        cfg_scale=1.0,
        **kwargs,
    ):
        if "schedule" in kwargs and (time_schedule == "uniform" or time_schedule is None):
            time_schedule = kwargs.pop("schedule")
        if spk_emb is None and "spk" in kwargs:
            spk_emb = kwargs["spk"]
        if spk_emb is None and "speaker_emb" in kwargs:
            spk_emb = kwargs["speaker_emb"]

        # 1. Text embedding with dropout
        x_in = self.emb_dropout(self.txt_emb(x))

        # 2. 6-Stack Pure Transformer Text Encoder (XTEncoder, 192-dim, RoPE, RMSNorm)
        enc_out = self.encoder(x_in, mask=mask)
        if mask is not None:
            m_t = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            enc_out = enc_out * m_t

        # 3. Per-Phoneme Duration Predictor
        dur_int, dur_pred, pred_frames = self.duration(
            enc_out.detach(),
            spk_emb=spk_emb,
            mask=mask,
            length_scale=length_scale,
        )

        if skip_decoder:
            return (None, None, None, None), enc_out, dur_pred

        # 4. Length Regulation via repeat_interleave
        # Use ground-truth durations during training/eval with target, otherwise use predicted durations
        dur_use = durations if (durations is not None and (self.training or target_latent is not None)) else dur_int
        dur_use = torch.clamp(dur_use, min=0)

        # Expand text representation T_text -> T_audio
        aligned_list = []
        B = enc_out.size(0)
        for b in range(B):
            if mask is not None:
                valid_len = int(mask[b].sum().item())
            else:
                valid_len = enc_out.size(1)
            b_enc = enc_out[b, :valid_len]
            b_dur = dur_use[b, :valid_len]
            if b_dur.sum() == 0:
                b_dur = b_dur.clone()
                b_dur[0] = 1
            aligned_b = torch.repeat_interleave(b_enc, b_dur, dim=0)
            aligned_list.append(aligned_b)

        aligned_text = pad_sequence(aligned_list, batch_first=True)  # [B, T_audio, 192]

        # Residual Conv1d right after repeat_interleave to provide soft phoneme boundaries
        aligned_text = self.post_interleave_conv(aligned_text)

        # Positional embedding so the text representation entering decoder layers knows frame positions
        aligned_text = aligned_text + self.pos_embed(aligned_text.size(1), device=aligned_text.device, dtype=aligned_text.dtype)

        # 5. Decoder Pass
        if target_latent is not None:
            # Training Mode: CFM Velocity Matching
            T_target = target_latent.size(1)
            min_T = min(aligned_text.size(1), T_target)
            target = target_latent[:, :min_T, :]
            aligned_text = aligned_text[:, :min_T, :]

            if audio_mask is not None:
                step_audio_mask = audio_mask[:, :min_T]
            else:
                step_audio_mask = torch.ones(B, min_T, device=x.device, dtype=torch.bool)

            drop_mask = None
            drop_spk = spk_emb
            if self.training and self.cfg_dropout > 0.0:
                drop_mask = torch.rand(B, device=x.device) < self.cfg_dropout
                if drop_spk is not None:
                    drop_spk = drop_spk.masked_fill(drop_mask.view(-1, 1), 0.0)

            dec_res = self.decoder(
                aligned_text=aligned_text,
                spk_emb=drop_spk,
                audio_mask=step_audio_mask,
                target=target,
                cond_drop_mask=drop_mask,
                **decoder_kwargs if 'decoder_kwargs' in locals() else {},
            )
            v_pred, u_target, t_step, dec_audio_mask = dec_res[:4]

            return (v_pred, u_target, t_step, dec_audio_mask, drop_mask), enc_out, dur_pred

        else:
            # Inference Mode: Continuous ODE integration
            T_audio = aligned_text.size(1)
            step_audio_mask = torch.ones(B, T_audio, device=x.device, dtype=torch.bool)

            dec_out = self.decoder(
                aligned_text=aligned_text,
                spk_emb=spk_emb,
                audio_mask=step_audio_mask,
                target=None,
                n_timesteps=n_timesteps,
                temperature=temperature,
                solver=solver,
                time_schedule=time_schedule,
                sway_coef=sway_coef,
                t_end=t_end,
                cfg_scale=cfg_scale,
                **kwargs,
            )
            return dec_out, enc_out, pred_frames

    @torch.no_grad()
    def sample(
        self,
        x,
        text_lengths=None,
        spk_emb=None,
        n_timesteps=25,
        temperature=1.0,
        cfg_scale=2.0,
        solver="euler",
        time_schedule="uniform",
        sway_coef=-1.0,
        length_scale=1.0,
        **kwargs,
    ):
        if "schedule" in kwargs and (time_schedule == "uniform" or time_schedule is None):
            time_schedule = kwargs.pop("schedule")
        device = x.device
        if text_lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=device)[None, :] < text_lengths[:, None]
        else:
            mask = None

        dec_out, enc_out, pred_frames = self.forward(
            x=x,
            target_latent=None,
            spk_emb=spk_emb,
            mask=mask,
            n_timesteps=n_timesteps,
            temperature=temperature,
            cfg_scale=cfg_scale,
            solver=solver,
            time_schedule=time_schedule,
            sway_coef=sway_coef,
            length_scale=length_scale,
            **kwargs,
        )
        return dec_out


# Backward compatibility alias
Model = MambaFlowTTSModel
