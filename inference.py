import os
import sys
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# Setup module path resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from TTS_model import MambaFlowTTSModel
    from TTSDataModule import TTSMODEL
    from preprocessing.text import text_to_sequence
except (ImportError, ValueError):
    from TTSModel.libri.TTS_model import MambaFlowTTSModel
    from TTSModel.libri.TTSDataModule import TTSMODEL
    from TTSModel.libri.preprocessing.text import text_to_sequence

try:
    from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig
except ImportError as e:
    raise ImportError("voxcpm is required. Please install it with: pip install voxcpm") from e


class TTS:
    """
    MambaFlow-Latent-v1 High-Level Multi-Speaker Text-to-Speech Inference Pipeline.
    
    Architecture:
      - Text Encoder: 6-Layer Transformer Encoder with RoPE (192D)
      - Duration Predictor: 3-Layer Speaker-Conditioned Monotonic Predictor
      - Velocity Field Decoder: 6-Layer Bidirectional Mamba-2 (320D) with Dual Modulation
      - Acoustic Codec: 44.1 kHz Continuous Latent Audio VAE (VoxCPM)
    """
    def __init__(
        self,
        checkpoint_path=None,
        spk_pt=None,
        latent_mean=None,
        latent_std=None,
        solver="midpoint",
        scheduler="sway",
        steps=16,
        cfg=1.8,
        temp=0.8,
        length_scale=0.88,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve paths with sensible defaults relative to script directory
        if checkpoint_path is None:
            candidate_ckpts = [
                os.path.join(CURRENT_DIR, "TTS_checkpoints", "mambaflow_latent_v1.ckpt"),
                os.path.join(CURRENT_DIR, "TTS_checkpoints", "mambaflow_last.ckpt"),
                os.path.join(CURRENT_DIR, "mambaflow_latent_v1.ckpt"),
            ]
            for c in candidate_ckpts:
                if os.path.exists(c):
                    checkpoint_path = c
                    break
            if checkpoint_path is None:
                checkpoint_path = candidate_ckpts[0]

        spk_pt = spk_pt or os.path.join(CURRENT_DIR, "speaker_embeddings.pt")
        latent_mean = latent_mean or os.path.join(CURRENT_DIR, "latent_mean_voxcpm.npy")
        latent_std = latent_std or os.path.join(CURRENT_DIR, "latent_std_voxcpm.npy")

        # 1. Load trained MambaFlow model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        print(f"[+] Loading MambaFlow checkpoint: {checkpoint_path}")
        loaded = TTSMODEL.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False
        )
        self.model = loaded.model if hasattr(loaded, "model") else loaded
        self.model.to(self.device).eval()

        # 2. Load VoxCPM AudioVAE
        weights_path = self._resolve_audiovae_weights()
        print(f"[+] Loading VoxCPM AudioVAE from: {weights_path}")
        vae_config = AudioVAEConfig(
            encoder_dim=64, encoder_rates=[2, 3, 6, 7, 7], latent_dim=64,
            decoder_dim=2048, decoder_rates=[7, 7, 6, 3, 2], sample_rate=44100
        )
        self.vae = AudioVAE(vae_config).to(self.device).eval()
        ckpt_vae = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.vae.load_state_dict(ckpt_vae["state_dict"], strict=True)
        del ckpt_vae
        gc.collect()

        # 3. Load Speaker Embeddings & Rankings
        print(f"[+] Loading Speaker Embeddings: {spk_pt}")
        spk_data = torch.load(spk_pt, map_location=self.device, weights_only=False)
        self.spk_embs = spk_data.get("speaker_embeddings", spk_data) if isinstance(spk_data, dict) else spk_data

        # Load Speaker Rankings metadata if available
        rankings_path = os.path.join(CURRENT_DIR, "speaker_rankings.json")
        self.speaker_rankings = []
        if os.path.exists(rankings_path):
            with open(rankings_path, "r") as f:
                self.speaker_rankings = json.load(f)

        # 4. Latent Normalization Statistics
        self.latent_mean = torch.from_numpy(np.load(latent_mean)).float().to(self.device)
        self.latent_std = torch.from_numpy(np.load(latent_std)).float().to(self.device)

        # 5. Default Hyperparameters
        self.solver = solver
        self.scheduler = scheduler
        self.steps = steps
        self.cfg = cfg
        self.temp = temp
        self.length_scale = length_scale
        self.sample_rate = 44100
        print("[+] MambaFlow TTS initialization complete.\n")

    def _resolve_audiovae_weights(self):
        """Locates cached AudioVAE weights or downloads via huggingface_hub."""
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id="openbmb/VoxCPM1.5", filename="audiovae.pth")
        except Exception:
            # Fallback local paths
            fallbacks = [
                os.path.expanduser("~/.cache/huggingface/hub/models--openbmb--VoxCPM1.5/snapshots/cdaae72f8ca2d51971c93f0a02430c4209016f6b/audiovae.pth"),
                os.path.expanduser("~/.cache/huggingface/hub/models--openbmb--VoxCPM1.5/snapshots/8cdc403854cda0e3af12252d27da038fda5982ac/audiovae.pth"),
            ]
            for p in fallbacks:
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(
                "Could not find VoxCPM audiovae.pth. Install huggingface_hub and connect to the internet, "
                "or place audiovae.pth manually."
            )

    def list_speakers(self, top_k=10):
        """Prints and returns the top-ranked validation speakers."""
        if not self.speaker_rankings:
            print("Speaker ranking metadata not loaded.")
            return []
        
        print("=" * 80)
        print(f"TOP {min(top_k, len(self.speaker_rankings))} MULTI-SPEAKER PROFILES (Ranked by Validation Benchmark):")
        print("=" * 80)
        print(f"{'Rank':<6}{'Speaker ID':<14}{'Composite Score':<18}{'Speaker Sim':<16}{'NISQA MOS':<12}{'WER':<8}")
        print("-" * 80)
        for s in self.speaker_rankings[:top_k]:
            print(f"{s['rank']:<6}{s['speaker_id']:<14}{s['composite_score']:<18.4f}{s['spk_sim']:<16.4f}{s['nisqa_mos']:<12.2f}{s['wer']:<8.2f}")
        print("=" * 80)
        print("Tip: Pass speaker_id=0 for the top-rated speaker (ID: 4362), or specify any ID directly.\n")
        return self.speaker_rankings[:top_k]

    @torch.inference_mode()
    def generate(
        self,
        text,
        speaker_id=0,
        steps=None,
        solver=None,
        schedule=None,
        time_schedule=None,
        sway_coef=-1.0,
        temp=None,
        length_scale=None,
        cfg_scale=None,
        output_path=None,
        **kwargs,
    ):
        """
        Synthesize speech audio from text.
        
        Args:
            text (str): Input English text string.
            speaker_id (int or str): Speaker rank (0 to 145) or original LibriTTS ID (e.g. 4362). Default: 0 (best).
            steps (int): ODE integration timesteps (default: 16).
            solver (str): ODE numerical solver ('midpoint' or 'euler', default: 'midpoint').
            schedule (str): Time scheduler ('sway', 'sine', 'uniform', default: 'sway').
            sway_coef (float): Sway schedule coefficient (default: -1.0).
            temp (float): Prior sampling noise temperature (default: 0.8).
            length_scale (float): Duration pacing multiplier (default: 0.88).
            cfg_scale (float): Classifier-Free Guidance scale (default: 1.8).
            output_path (str, optional): If provided, writes the 44.1 kHz WAV file to disk.
            
        Returns:
            np.ndarray: 44.1 kHz 1D audio waveform.
        """
        device = self.device
        steps = steps if steps is not None else self.steps
        solver = solver if solver is not None else self.solver
        schedule = time_schedule or schedule or self.scheduler
        temp = temp if temp is not None else self.temp
        length_scale = length_scale if length_scale is not None else self.length_scale
        cfg = cfg_scale if cfg_scale is not None else self.cfg

        # 1. Phonemize input text
        seq = text_to_sequence(text, ["english_cleaners2"])[0]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        mask = torch.ones_like(x, dtype=torch.bool)

        # 2. Retrieve & normalize speaker embedding
        spk_emb = None
        if isinstance(self.spk_embs, dict):
            # Check sequential rank or speaker id
            spk_emb = self.spk_embs.get(speaker_id, None)
            if spk_emb is None:
                spk_emb = self.spk_embs.get(str(speaker_id), None)
            if spk_emb is None and str(speaker_id).isdigit():
                spk_emb = self.spk_embs.get(int(speaker_id), None)
        elif torch.is_tensor(self.spk_embs):
            spk_emb = self.spk_embs

        if spk_emb is None:
            print(f"[!] Warning: Speaker ID '{speaker_id}' not found. Falling back to default top speaker (Rank 0 / 4362).")
            spk_emb = self.spk_embs.get(0, self.spk_embs.get("0", torch.zeros(1, 192, device=device)))

        if not torch.is_tensor(spk_emb):
            spk_emb = torch.tensor(spk_emb, device=device)
        else:
            spk_emb = spk_emb.to(device)

        spk_emb = spk_emb.float()
        if spk_emb.ndim == 1:
            spk_emb = spk_emb.unsqueeze(0)
        spk_emb = F.normalize(spk_emb, p=2, dim=-1)

        # 3. Flow Matching Velocity Field ODE Integration
        dec_out, enc_out, pred_frames = self.model.forward(
            x=x,
            target_latent=None,
            spk_emb=spk_emb,
            mask=mask,
            n_timesteps=steps,
            temperature=temp,
            solver=solver,
            time_schedule=schedule,
            sway_coef=sway_coef,
            cfg_scale=cfg,
            length_scale=length_scale,
            **kwargs,
        )

        # 4. Denormalize continuous latents & decode with VoxCPM AudioVAE
        dec_out = (dec_out * (self.latent_std + 1e-8)) + self.latent_mean
        z_slice = dec_out.transpose(1, 2)
        wav_gen = self.vae.decode(z_slice).squeeze().cpu().numpy()

        # 5. Optionally save audio
        if output_path:
            self.save(wav_gen, output_path)

        return wav_gen

    def save(self, wav: np.ndarray, filepath: str, sample_rate: int = 44100):
        """Save synthesized audio array to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        sf.write(filepath, wav, sample_rate)
        print(f"[+] Audio successfully saved to: {filepath} ({len(wav) / sample_rate:.2f}s @ {sample_rate}Hz)")


def main():
    parser = argparse.ArgumentParser(description="MambaFlow-Latent-v1 TTS Inference CLI")
    parser.add_argument("--text", "-t", type=str, default=None, help="Input English text to synthesize")
    parser.add_argument("--speaker_id", "-s", default=0, help="Speaker index (0-145) or LibriTTS ID (default: 0)")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Path to output WAV file")
    parser.add_argument("--steps", type=int, default=16, help="ODE steps (default: 16)")
    parser.add_argument("--solver", type=str, default="midpoint", choices=["midpoint", "euler"], help="ODE solver")
    parser.add_argument("--cfg", type=float, default=1.8, help="CFG scale (default: 1.8)")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (default: 0.8)")
    parser.add_argument("--length_scale", type=float, default=0.88, help="Length scale (default: 0.88)")
    parser.add_argument("--list_speakers", action="store_true", help="List top speaker profiles and exit")
    args = parser.parse_args()

    if not args.list_speakers and not args.text:
        parser.error("--text/-t is required unless --list_speakers is specified.")

    tts = TTS()
    if args.list_speakers:
        tts.list_speakers(top_k=20)
        return

    wav = tts.generate(
        text=args.text,
        speaker_id=args.speaker_id,
        steps=args.steps,
        solver=args.solver,
        temp=args.temp,
        length_scale=args.length_scale,
        cfg_scale=args.cfg,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
