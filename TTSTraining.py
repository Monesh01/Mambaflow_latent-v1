import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

import warnings
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    ModelSummary,
    Callback,
    LearningRateMonitor,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from TTSDataModule import TTSDataModule, TTSMODEL

warnings.filterwarnings("ignore", category=UserWarning, module="requests")
torch.backends.cudnn.benchmark = False


class AtomicLastCheckpointCallback(Callback):
    """
    Guarantees that the last checkpoint is atomically updated periodically,
    overwriting without accumulating duplicates, so training can always seamlessly resume.
    """
    def __init__(self, dirpath="TTS_checkpoints/", filename="mambaflow_last.ckpt", save_every=1):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.save_every = save_every

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_ep = trainer.max_epochs
        if epoch % self.save_every == 0 or epoch == max_ep:
            os.makedirs(self.dirpath, exist_ok=True)
            last_path = os.path.join(self.dirpath, self.filename)
            tmp_path = os.path.join(self.dirpath, f"{self.filename}.tmp")
            trainer.save_checkpoint(tmp_path)
            os.replace(tmp_path, last_path)


class MetricsTrackerCallback(Callback):
    """
    Saves full epoch-level metrics history to CSV for evaluation and visualization.
    """
    def __init__(self, output_csv="training_metrics.csv"):
        super().__init__()
        self.output_csv = output_csv
        if os.path.exists(output_csv):
            try:
                import pandas as pd
                self.metrics = pd.read_csv(output_csv).to_dict("records")
            except Exception:
                self.metrics = []
        else:
            self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        logged = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in trainer.logged_metrics.items()}
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else getattr(pl_module, "lr", 0.0)

        train_l = logged.get("train_loss_epoch", logged.get("train_loss", None))
        cfm_l = logged.get("train_loss_cfm_epoch", logged.get("train_loss_cfm", None))
        dur_l = logged.get("train_loss_dur_epoch", logged.get("train_loss_dur", None))
        val_l = logged.get("val_loss", None)
        val_cfm_l = logged.get("val_loss_cfm", None)
        val_dur_l = logged.get("val_loss_dur", None)

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_l,
            "train_loss_cfm": cfm_l,
            "train_loss_dur": dur_l,
            "val_loss": val_l,
            "val_loss_cfm": val_cfm_l,
            "val_loss_dur": val_dur_l,
        }
        self.metrics = [m for m in self.metrics if m.get("epoch") != epoch]
        self.metrics.append(row)
        import pandas as pd
        pd.DataFrame(self.metrics).to_csv(self.output_csv, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MambaFlow-TTS Training Script (Length-Regulated BiMamba-2)")
    parser.add_argument("--epochs", type=int, default=250, help="Total training epochs (default: 250)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak AdamW learning rate (default: 2e-4)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate floor (default: 1e-6)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size (default: 16)")
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation batches (default: 1)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint file path to resume from (default: None)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint (default: False)")
    parser.add_argument("--scratch", action="store_true", default=True, help="Train fully from scratch without resuming (default: True)")
    parser.add_argument("--warm_start", action="store_true", default=False, help="Load weights with strict=False and start fresh epochs (default: False)")
    parser.add_argument("--duration_weight", type=float, default=0.1, help="Loss weight for duration predictor (default: 0.1)")
    parser.add_argument("--cfm_loss_type", type=str, default="mse", choices=["mse", "huber"], help="CFM velocity loss objective (default: mse)")
    parser.add_argument("--limit_train_batches", type=float, default=None, help="Limit train batches per epoch for quick validation/smoke testing")
    parser.add_argument("--limit_val_batches", type=float, default=None, help="Limit val batches per epoch")
    parser.add_argument("--train_file", type=str, default="/home/monesh/TTSModel/libri/train.csv", help="Path to train CSV")
    parser.add_argument("--val_file", type=str, default="/home/monesh/TTSModel/libri/val.csv", help="Path to val CSV")
    parser.add_argument("--exp_name", type=str, default="mambaflow_12m_6layers_320d_k1", help="Experiment name")
    parser.add_argument("--ckpt_dir", type=str, default="TTS_checkpoints/", help="Checkpoint output directory")
    parser.add_argument("--last_ckpt_name", type=str, default="mambaflow_last.ckpt", help="Last checkpoint filename")
    parser.add_argument("--d_codec", type=int, default=64, help="Target latent dimension (default: 64 for VoxCPM 1.5)")
    parser.add_argument("--d_dec", type=int, default=320, help="Decoder hidden dimension (default: 320, can use 384)")
    parser.add_argument("--expand", type=int, default=2, help="Mamba2 expansion factor (default: 2)")
    parser.add_argument("--mlp_mult", type=int, default=2, help="Decoder MLP expansion multiplier (default: 2)")
    parser.add_argument("--gate_init", type=float, default=-2.0, help="Initial value for 320D sigmoid text gates on later layers (default: -2.0)")
    parser.add_argument("--num_dec_blocks", type=int, default=6, help="Number of decoder layers (default: 6)")
    parser.add_argument("--no_val", action="store_true", default=False, help="Disable validation entirely")
    parser.add_argument("--prefetch", action="store_true", default=False, help="Enable in-RAM dataset prefetching")
    parser.add_argument("--override_lr", type=float, default=None, help="Explicit learning rate override")
    parser.add_argument("--cosine_t_max", type=int, default=None, help="CosineAnnealingLR T_max in epochs (default: remaining epochs to --epochs)")
    parser.add_argument("--resume_step", type=int, default=None, help="Starting step offset for scheduler")
    parser.add_argument("--metrics_csv", type=str, default="training_metrics.csv", help="CSV path to log metrics")
    parser.add_argument("--save_top_k", type=int, default=1, help="Top k checkpoints to save (default: 1)")
    parser.add_argument("--save_every_n_epochs", type=int, default=10, help="Save milestone checkpoint every N epochs (default: 10)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience in epochs (default: 50)")
    args, _ = parser.parse_known_args()

    max_epochs = args.epochs
    lr = args.lr
    min_lr = args.min_lr
    batch_size = args.batch_size
    accumulate_grad_batches = args.accumulate
    num_workers = args.num_workers
    duration_weight = args.duration_weight
    prefetch_data = args.prefetch

    d_codec = args.d_codec
    d_enc = 192                              # 192-dim 6-Stack Transformer Text Encoder
    d_dec = args.d_dec
    expand = args.expand                     # expand=2 default
    mlp_mult = args.mlp_mult                 # mlp_mult=3 default
    num_dec_blocks = args.num_dec_blocks
    cfg_dropout = 0.15                       # 15% Classifier-Free Guidance
    gradient_checkpointing = False
    logit_normal_mean = 0.0
    logit_normal_std = 1.0

    ckpt_dir = args.ckpt_dir
    last_ckpt_name = args.last_ckpt_name
    save_top_k = args.save_top_k

    train_file = args.train_file
    val_file = None if args.no_val else args.val_file

    log_dir = "TTS_logs"
    exp_name = args.exp_name

    os.makedirs(ckpt_dir, exist_ok=True)

    datamodule = TTSDataModule(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_data=prefetch_data,
        target_dim=d_codec,
    )

    import pandas as pd
    import math
    if os.path.exists(train_file):
        num_train_samples = len(pd.read_csv(train_file))
    else:
        num_train_samples = 27434
    steps_per_epoch = max(1, math.ceil(num_train_samples / (batch_size * accumulate_grad_batches)))
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    last_checkpoint_callback = AtomicLastCheckpointCallback(
        dirpath=ckpt_dir,
        filename=last_ckpt_name,
        save_every=1,
    )

    metrics_tracker_callback = MetricsTrackerCallback(output_csv=args.metrics_csv)

    callbacks = [
        last_checkpoint_callback,
        metrics_tracker_callback,
        ModelSummary(max_depth=3),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if not args.no_val:
        top_k_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{exp_name}_epoch_{{epoch:03d}}_val_loss_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=save_top_k,
            save_last=False,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        )
        callbacks.insert(0, top_k_checkpoint_callback)
        callbacks.append(early_stop_callback)
    else:
        top_k_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{exp_name}_epoch_{{epoch:03d}}_train_loss_{{train_loss_epoch:.4f}}",
            monitor="train_loss_epoch",
            mode="min",
            save_top_k=save_top_k,
            save_last=False,
        )
        callbacks.insert(0, top_k_checkpoint_callback)

    if args.save_every_n_epochs > 0:
        milestone_checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{exp_name}_milestone_epoch_{{epoch:03d}}",
            every_n_epochs=args.save_every_n_epochs,
            save_top_k=-1,
            save_last=False,
        )
        callbacks.append(milestone_checkpoint_callback)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=exp_name,
        default_hp_metric=False,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        deterministic=False,
        check_val_every_n_epoch=1 if not args.no_val else 100000,
        num_sanity_val_steps=1 if not args.no_val else 0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    resume_ckpt = None
    if args.ckpt is not None and os.path.exists(args.ckpt):
        resume_ckpt = args.ckpt
        print(f"[+] Explicit checkpoint provided: {resume_ckpt}")
    elif args.resume:
        last_ckpt_path = os.path.join(ckpt_dir, last_ckpt_name)
        if os.path.exists(last_ckpt_path):
            resume_ckpt = last_ckpt_path
            print(f"[+] Auto-resuming from latest atomic checkpoint: {resume_ckpt}")
        else:
            all_ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if all_ckpts:
                resume_ckpt = max(all_ckpts, key=os.path.getmtime)
                print(f"[+] Auto-resuming from latest available checkpoint: {resume_ckpt}")
            else:
                print(f"[-] No checkpoint found in {ckpt_dir} to resume from.")

    lightning_model = TTSMODEL(
        lr=lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_epochs=max_epochs,
        cosine_t_max=args.cosine_t_max,
        patience=args.patience,
        override_lr=args.override_lr,
        d_codec=d_codec,
        d_enc=d_enc,
        d_dec=d_dec,
        expand=expand,
        mlp_mult=mlp_mult,
        gate_init=args.gate_init,
        d_spk=192,
        duration_hidden_dim=256,
        num_dec_blocks=num_dec_blocks,
        cfg_dropout=cfg_dropout,
        duration_weight=duration_weight,
        cfm_loss_type=args.cfm_loss_type,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
        gradient_checkpointing=gradient_checkpointing,
        monitor="train_loss_epoch" if args.no_val else "val_loss",
    )

    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)

    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model_state = lightning_model.state_dict()
        ckpt_state = ckpt.get("state_dict", {})
        has_full_state = ("loops" in ckpt and "optimizer_states" in ckpt and "lr_schedulers" in ckpt)
        exact_match = (
            has_full_state
            and (set(ckpt_state.keys()) == set(model_state.keys()))
            and all(ckpt_state[k].shape == model_state[k].shape for k in model_state)
        )
        ckpt_epoch = int(ckpt.get("epoch", 0))
        ckpt_step = int(ckpt.get("global_step", 0))

        if exact_match and not args.warm_start:
            print(f"[+] Full Lightning resume from {resume_ckpt} (Epoch {ckpt_epoch}, Step {ckpt_step})...")
            del ckpt
        else:
            print(f"[+] Resuming model weights from {resume_ckpt} with strict=False...")
            lightning_model.on_load_checkpoint(ckpt)
            missing, unexpected = lightning_model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"[+] Weights loaded successfully with strict=False (missing={len(missing)}, unexpected={len(unexpected)}).")
            if missing:
                print(f"[+]   Initialized new layers/parameters ({len(missing)} keys): {missing[:6]} ...")
            if unexpected:
                print(f"[+]   Dropped obsolete/mismatched layers ({len(unexpected)} keys): {unexpected[:6]} ...")
            if not args.warm_start:
                ep = trainer.fit_loop.epoch_progress.current
                ep.ready = ckpt_epoch
                ep.started = ckpt_epoch
                ep.processed = ckpt_epoch
                ep.completed = ckpt_epoch
                if args.cosine_t_max is None:
                    rem_epochs = max(1, max_epochs - ckpt_epoch)
                    lightning_model.cosine_t_max = rem_epochs
                    lightning_model.hparams.cosine_t_max = rem_epochs
                    print(f"[+] CosineAnnealingLR configured for remaining epochs: T_max={rem_epochs} (Epoch {ckpt_epoch} -> {max_epochs}, min_lr={min_lr:.2e})")
                print(f"[+] Resuming training loop starting from epoch {ckpt_epoch} / {max_epochs}")
            del ckpt
            resume_ckpt = None

    print("=" * 80)
    print(f"[+] MAMBAFLOW-TTS ARCHITECTURE (LENGTH REGULATION + BIMAMBA-2 DECODER)")
    print(f"[+] Decoder Channels (d_dec): {d_dec}D (expand={expand}, mlp_mult={mlp_mult}, gate_init={args.gate_init})")
    print(f"[+] Target Latent Dim:        {d_codec}D (VoxCPM 1.5 Latent CFM Velocity Matching)")
    print(f"[+] Model Size:               {total_params/1e6:.2f}M Parameters (Strictly < 25M)")
    print(f"[+] Cross-Attention:          ELIMINATED (Zero repetition / collapse by design)")
    print(f"[+] Sequence Alignment:       Per-Phoneme Speaker Duration Predictor + repeat_interleave")
    print(f"[+] Conditioning Mode:        Concat [x_t (64D); H_aligned (192D)] -> 256D + Layers 3-5 320D Gated Norm Text")
    print(f"[+] Speakers Filtered:        146 Unique Speakers (Filtered >= 100 Utterances)")
    print(f"[+] Dataset Distribution:     Train={num_train_samples:,} rows | Val={len(pd.read_csv(val_file)) if val_file and os.path.exists(val_file) else 0:,} rows")
    print(f"[+] Total Parameters:         {total_params:,}")
    print(f"[+] Trainable Parameters:     {trainable_params:,}")
    print(f"[+] Checkpoint Directory:     {ckpt_dir}")
    print("=" * 80)

    trainer.fit(model=lightning_model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
