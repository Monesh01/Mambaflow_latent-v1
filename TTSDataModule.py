import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import DataLoader
try:
    from TTSDatasetModule import MyDataset, collate_fn, masked_huber_loss, masked_mse_loss
    from TTS_model import MambaFlowTTSModel
except (ImportError, ValueError):
    from TTSModel.libri.TTSDatasetModule import MyDataset, collate_fn, masked_huber_loss, masked_mse_loss
    from TTSModel.libri.TTS_model import MambaFlowTTSModel


class TTSDataModule(pl.LightningDataModule):
    """
    LightningDataModule managing data loading for LibriTTS-R Multi-Speaker TTS.
    """
    def __init__(
        self,
        train_file="/home/monesh/TTSModel/libri/train.csv",
        val_file="/home/monesh/TTSModel/libri/val.csv",
        batch_size=16,
        num_workers=4,
        target_dim=64,
        mount_prefix="/home/monesh/TTSModel/libri",
        prefetch_data=False,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_dim = target_dim
        self.mount_prefix = mount_prefix
        self.prefetch_data = prefetch_data
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.train_dataset is None and self.train_file and os.path.exists(self.train_file):
                print(f"[+] Setting up Train Dataset from: {self.train_file}")
                self.train_dataset = MyDataset(
                    csv_path=self.train_file,
                    mount_prefix=self.mount_prefix,
                    target_dim=self.target_dim,
                    prefetch_data=self.prefetch_data,
                    is_train=True,
                )
            if self.val_dataset is None and self.val_file and os.path.exists(self.val_file):
                print(f"[+] Setting up Val Dataset from: {self.val_file}")
                self.val_dataset = MyDataset(
                    csv_path=self.val_file,
                    mount_prefix=self.mount_prefix,
                    target_dim=self.target_dim,
                    prefetch_data=self.prefetch_data,
                    is_train=False,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )


class MambaFlowTTS(pl.LightningModule):
    """
    LightningModule for MambaFlow-TTS:
    - Text Encoder (XTEncoder, 192D)
    - Speaker-conditioned per-phoneme duration predictor (DurationPredictor)
    - Length regulation via repeat_interleave
    - BiMamba-2 Decoder (expand=2, mlp_mult=2, no cross-attention)
    - CFM Velocity matching loss + Per-phoneme Duration Huber loss
    """
    def __init__(
        self,
        model=None,
        lr=2e-4,
        lr_decoder=None,
        d_codec=64,
        patience=50,
        factor=0.5,
        max_epochs=100,
        scheduler_type="cosine",
        cosine_t_max=None,
        **kwargs,
    ):
        super().__init__()
        self.d_codec = d_codec
        d_enc = kwargs.get("d_enc", 192)
        d_dec = kwargs.get("d_dec", 320)
        expand = kwargs.get("expand", 2)
        mlp_mult = kwargs.get("mlp_mult", 2)
        gate_init = kwargs.get("gate_init", -2.0)
        text_inject_layers = kwargs.get("text_inject_layers", None)
        num_dec_blocks = kwargs.get("num_dec_blocks", 6)
        dur_hidden = kwargs.get("duration_hidden_dim", 256)
        d_spk = kwargs.get("d_spk", 192)

        if model is None:
            self.model = MambaFlowTTSModel(
                n_vocab=179,
                d_enc=d_enc,
                d_codec=d_codec,
                d_dec=d_dec,
                expand=expand,
                mlp_mult=mlp_mult,
                gate_init=gate_init,
                text_inject_layers=text_inject_layers,
                d_spk=d_spk,
                duration_hidden_dim=dur_hidden,
                num_dec_blocks=num_dec_blocks,
                cfg_dropout=kwargs.get("cfg_dropout", 0.15),
                gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
                logit_normal_mean=kwargs.get("logit_normal_mean", 0.0),
                logit_normal_std=kwargs.get("logit_normal_std", 1.0),
            )
        else:
            self.model = model

        self.train_decoder_and_duration = kwargs.get("train_decoder_and_duration", False)
        self.train_decoder_only = kwargs.get("train_decoder_only", False)
        self.train_duration_only = kwargs.get("train_duration_only", False)
        self.duration_weight = kwargs.get("duration_weight", 0.1)
        self.duration_beta = kwargs.get("duration_beta", 0.1)
        self.cfm_loss_type = kwargs.get("cfm_loss_type", "mse")
        self.lr = lr
        self.lr_decoder = lr_decoder
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.total_steps = kwargs.get("total_steps", 15000)
        self.min_lr = kwargs.get("min_lr", 5e-6)
        self.patience = patience
        self.factor = factor
        self.max_epochs = max_epochs
        self.scheduler_type = scheduler_type
        self.cosine_t_max = cosine_t_max
        self.override_lr = kwargs.get("override_lr", None)
        self.resume_step = kwargs.get("resume_step", None)
        self.save_hyperparameters(ignore=["model"])
        self.strict_loading = False
        self.monitor = kwargs.get("monitor", "val_loss")

        if self.train_decoder_and_duration:
            self.freeze_encoder_modules()
        elif self.train_decoder_only:
            self.freeze_non_decoder_modules()
        elif self.train_duration_only:
            self.freeze_non_duration_modules()

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 1. Remap legacy ada_lin to ada_g if needed
        keys_to_remap = {}
        for k in list(state_dict.keys()):
            if "ada_lin" in k:
                keys_to_remap[k] = k.replace("ada_lin", "ada_g")
        for old_k, new_k in keys_to_remap.items():
            print(f"[+] on_load_checkpoint: Remapping legacy {old_k} -> {new_k}")
            state_dict[new_k] = state_dict.pop(old_k)

        # 2. Drop obsolete (ada_x, cross_attn, larope, etc.) or shape-mismatched keys
        model_state = self.state_dict()
        keys_to_remove = []
        for k, v in list(state_dict.items()):
            if "ada_x" in k or "cross_attn" in k or "larope" in k or "gaussian_lambda" in k:
                keys_to_remove.append(k)
            elif k not in model_state:
                keys_to_remove.append(k)
            elif v.shape != model_state[k].shape:
                print(f"[-] on_load_checkpoint: Shape mismatch for {k}: ckpt {v.shape} vs model {model_state[k].shape}")
                keys_to_remove.append(k)

        for k in keys_to_remove:
            print(f"[-] on_load_checkpoint: Dropping legacy/mismatched key: {k}")
            del state_dict[k]

        # 3. If architecture changed (keys dropped or new layers added), clear optimizer states to avoid shape crash
        missing_keys = [k for k in model_state.keys() if k not in state_dict]
        if keys_to_remove or missing_keys:
            if "optimizer_states" in checkpoint:
                print(f"[+] on_load_checkpoint: Architecture modified ({len(keys_to_remove)} dropped, {len(missing_keys)} new). Cleared obsolete optimizer states.")
                checkpoint.pop("optimizer_states", None)
            if "lr_schedulers" in checkpoint:
                checkpoint.pop("lr_schedulers", None)

        # 4. If override_lr or cosine_t_max is explicitly set, reconfigure CosineAnnealingLR cleanly for remaining epochs
        if (self.override_lr is not None or self.cosine_t_max is not None) and self.scheduler_type == "cosine":
            ckpt_epoch = int(checkpoint.get("epoch", 0))
            loops_ep = checkpoint.get("loops", {}).get("fit_loop", {}).get("epoch_progress", {}).get("current", {})
            started_ep = int(loops_ep.get("started", ckpt_epoch + 1))
            rem_epochs = max(1, int(self.max_epochs) - started_ep)
            t_max = int(self.cosine_t_max) if self.cosine_t_max is not None else rem_epochs
            new_lr = float(self.override_lr) if self.override_lr is not None else float(self.lr)

            n_groups = 3
            if "optimizer_states" in checkpoint and len(checkpoint["optimizer_states"]) > 0:
                opt_state = checkpoint["optimizer_states"][0]
                param_groups = opt_state.get("param_groups", [])
                if param_groups:
                    n_groups = len(param_groups)
                    for pg in param_groups:
                        pg["lr"] = new_lr
                        pg["initial_lr"] = new_lr

            if "lr_schedulers" in checkpoint and len(checkpoint["lr_schedulers"]) > 0:
                sched_state = checkpoint["lr_schedulers"][0]
                n_sched = len(sched_state.get("base_lrs", [])) or n_groups
                sched_state["T_max"] = t_max
                sched_state["eta_min"] = float(self.min_lr)
                sched_state["base_lrs"] = [new_lr] * n_sched
                sched_state["last_epoch"] = 0
                sched_state["_step_count"] = 1
                sched_state["_last_lr"] = [new_lr] * n_sched

            if "callbacks" in checkpoint:
                for cb_key, cb_state in checkpoint["callbacks"].items():
                    if "EarlyStopping" in str(cb_key) and isinstance(cb_state, dict):
                        cb_state["wait_count"] = 0
                        cb_state["patience"] = int(self.patience)

            print(
                f"[+] on_load_checkpoint: Reset CosineAnnealingLR from Epoch {started_ep} -> {self.max_epochs} "
                f"(T_max={t_max}, start_lr={new_lr:.2e}, min_lr={self.min_lr:.2e})"
            )

    def freeze_encoder_modules(self):
        if hasattr(self.model, "txt_emb") and self.model.txt_emb is not None:
            for p in self.model.txt_emb.parameters():
                p.requires_grad = False
        if hasattr(self.model, "encoder") and self.model.encoder is not None:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        if hasattr(self.model, "post_interleave_conv") and self.model.post_interleave_conv is not None:
            for p in self.model.post_interleave_conv.parameters():
                p.requires_grad = False
        if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
            for p in self.model.pos_embed.parameters():
                p.requires_grad = False

    def freeze_non_decoder_modules(self):
        self.freeze_encoder_modules()
        if hasattr(self.model, "duration") and self.model.duration is not None:
            for p in self.model.duration.parameters():
                p.requires_grad = False

    def freeze_non_duration_modules(self):
        self.freeze_encoder_modules()
        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            for p in self.model.decoder.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        lr_base = self.lr if self.override_lr is None else self.override_lr
        lr_dec = (self.lr_decoder if self.lr_decoder is not None else lr_base) if self.override_lr is None else self.override_lr

        if self.train_duration_only:
            params = [p for p in self.model.duration.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr_base, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2)
        elif self.train_decoder_only:
            params = [p for p in self.model.decoder.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr_dec, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2)
        else:
            enc_params = [p for p in list(self.model.txt_emb.parameters()) + list(self.model.encoder.parameters()) if p.requires_grad]
            if hasattr(self.model, "post_interleave_conv"):
                enc_params += [p for p in self.model.post_interleave_conv.parameters() if p.requires_grad]
            if hasattr(self.model, "pos_embed"):
                enc_params += [p for p in self.model.pos_embed.parameters() if p.requires_grad]
            dur_params = [p for p in self.model.duration.parameters() if p.requires_grad]
            dec_params = [p for p in self.model.decoder.parameters() if p.requires_grad]

            # Ensure all remaining trainable model parameters are included in optimizer
            assigned = set(enc_params + dur_params + dec_params)
            for p in self.model.parameters():
                if p.requires_grad and p not in assigned:
                    dec_params.append(p)

            optimizer = torch.optim.AdamW([
                {"params": enc_params, "lr": lr_base},
                {"params": dur_params, "lr": lr_base},
                {"params": dec_params, "lr": lr_dec},
            ], betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2)

        if self.scheduler_type == "cosine":
            t_max = self.cosine_t_max if self.cosine_t_max is not None else self.max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.min_lr)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.factor, patience=self.patience, min_lr=self.min_lr,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": self.monitor}}

    def _shared_step(self, batch):
        text = batch['ipa']
        text_lengths = batch['ipa_lengths']
        audio_lengths = batch['lengths']
        target_mel = batch['mel']
        durations = batch.get('durations', None)
        spk_emb = batch.get('spk_emb', None)
        device = text.device

        max_text_len = text.size(1)
        max_audio_len = target_mel.size(1)

        text_mask = torch.arange(max_text_len, device=device)[None, :] < text_lengths[:, None]
        audio_mask = torch.arange(max_audio_len, device=device)[None, :] < audio_lengths[:, None]

        model_out, enc_out, dur_pred = self.model(
            x=text,
            target_latent=target_mel,
            durations=durations,
            spk_emb=spk_emb,
            mask=text_mask,
            audio_mask=audio_mask,
            skip_decoder=self.train_duration_only,
        )

        v_pred, u_target, t_step, step_audio_mask = None, None, None, audio_mask
        if model_out is not None:
            if len(model_out) >= 5:
                v_pred, u_target, t_step, step_audio_mask = model_out[:4]
            elif len(model_out) == 4:
                v_pred, u_target, t_step, step_audio_mask = model_out

        if v_pred is not None:
            min_T = min(v_pred.size(1), u_target.size(1))
            v_pred = v_pred[:, :min_T, :]
            u_target = u_target[:, :min_T, :]
            if step_audio_mask is not None:
                step_audio_mask = step_audio_mask[:, :min_T]

        # 1. CFM Velocity Loss (MSE vs Huber)
        if v_pred is not None:
            if self.cfm_loss_type == "mse":
                cfm_loss = masked_mse_loss(v_pred, u_target, step_audio_mask)
            else:
                cfm_loss = masked_huber_loss(v_pred, u_target, step_audio_mask, delta=1.0)
        else:
            cfm_loss = torch.tensor(0.0, device=device)

        # 2. Per-Phoneme Duration Huber Loss
        if durations is not None:
            target_dur_log = torch.log1p(durations.float())
            dur_loss = masked_huber_loss(dur_pred, target_dur_log, text_mask, delta=self.duration_beta)
        else:
            dur_loss = torch.tensor(0.0, device=device)

        if self.train_duration_only:
            total_loss = dur_loss
        elif self.train_decoder_only:
            total_loss = cfm_loss
        else:
            total_loss = cfm_loss + self.duration_weight * dur_loss

        losses = {
            "cfm_loss": cfm_loss,
            "dur_loss": dur_loss,
        }
        return total_loss, losses

    def training_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_cfm", losses["cfm_loss"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_dur", losses["dur_loss"], prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        dev = batch["mel"].device
        devices = [dev] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(42 + batch_idx)
            loss, losses = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_loss_cfm", losses["cfm_loss"], prog_bar=True, on_epoch=True)
        self.log("val_loss_dur", losses["dur_loss"], prog_bar=True, on_epoch=True)
        return loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


# Backward compatibility alias
TTSMODEL = MambaFlowTTS
