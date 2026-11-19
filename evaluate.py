import os
import sys
import time
import json
import re
import glob
import gc
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import jiwer
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if "/home/monesh" not in sys.path:
    sys.path.insert(0, "/home/monesh")

# Triton memory fix
try:
    import triton.backends.nvidia.driver as triton_driver
    triton_driver.CudaDriver.get_empty_cache_for_benchmark = lambda self, *args, **kwargs: torch.empty(1024, dtype=torch.int, device='cuda')
except Exception:
    pass

try:
    from TTSDataModule import TTSMODEL
    from preprocessing.text import text_to_sequence
    from preprocessing.text.symbols import symbols
except (ImportError, ValueError):
    from TTSModel.libri.TTSDataModule import TTSMODEL
    from TTSModel.preprocessing.text import text_to_sequence
    from TTSModel.preprocessing.text.symbols import symbols
from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig

_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def get_phonemes_str(text: str) -> str:
    try:
        seq = text_to_sequence(text, ["english_cleaners2"])[0]
        return " ".join([_id_to_symbol.get(i, "") for i in seq if i in _id_to_symbol and _id_to_symbol[i] != ""])
    except Exception:
        return ""


def clean_text_for_asr(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_audio_stats(wav: np.ndarray):
    if len(wav) == 0:
        return 0.0, -100.0, 0.0
    rms = float(np.sqrt(np.mean(wav ** 2)))
    dbfs = float(20.0 * np.log10(rms + 1e-8))
    peak = float(np.max(np.abs(wav)))
    return rms, dbfs, peak


def main():
    parser = argparse.ArgumentParser(description="Full Validation Set Production Benchmark")
    parser.add_argument("--ckpt", type=str, default="TTS_checkpoints/mambaflow_last.ckpt", help="Checkpoint to test")
    parser.add_argument("--val_csv", type=str, default="val.csv", help="Validation dataset CSV")
    parser.add_argument("-n", "--num_samples", type=int, default=None, help="Number of validation samples to evaluate (default: all)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in validation set (default: 0)")
    parser.add_argument("--cfg_scale", type=float, default=1.8, help="CFG scale (default: 1.8)")
    parser.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint"], help="ODE solver (default: midpoint)")
    parser.add_argument("--time_schedule", type=str, default="sway", help="Time schedule (default: sway)")
    parser.add_argument("--timesteps", type=int, default=16, help="Timesteps (default: 16)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (default: 0.8)")
    parser.add_argument("--length_scale", type=float, default=0.88, help="Length scale (default: 0.88)")
    parser.add_argument("--save_wav_count", type=int, default=50, help="Number of audio files to write to disk (default: 50)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--out_dir", type=str, default="eval_production_full_val", help="Output directory")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    wavs_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    print("=" * 90)
    print("MAMBAFLOW-TTS FULL VALIDATION BENCHMARK SUITE")
    print(f"Checkpoint:       {args.ckpt}")
    print(f"Dataset:          {args.val_csv}")
    print(f"Sampling Setup:   Solver: {args.solver} | Schedule: {args.time_schedule} | Steps: {args.timesteps} | CFG: {args.cfg_scale} | Temp: {args.temperature} | LengthScale: {args.length_scale}")
    print(f"Device:           {device.upper()}")
    print("=" * 90)

    # 1. Load Checkpoint Metadata
    ckpt_raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    epoch_num = ckpt_raw.get("epoch", "unknown")
    global_step = ckpt_raw.get("global_step", "unknown")
    del ckpt_raw
    gc.collect()
    print(f"[+] Loaded checkpoint metadata -> Epoch: {epoch_num}, Global Step: {global_step}")

    # 2. Load Validation Data
    df_val = pd.read_csv(args.val_csv)
    # Filter to samples where latent and speaker embedding exist
    valid_mask = df_val["latent_path"].apply(os.path.exists) & df_val["speaker_emb_path"].apply(os.path.exists)
    df_val = df_val[valid_mask].copy().reset_index(drop=True)

    total_avail = len(df_val)
    start = args.start_idx
    end = total_avail if args.num_samples is None else min(start + args.num_samples, total_avail)
    eval_df = df_val.iloc[start:end].copy().reset_index(drop=True)
    print(f"[+] Evaluating {len(eval_df)} samples across {eval_df['speaker_id'].nunique()} unique speakers (from index {start} to {end}).")

    # 3. Load Models onto Device
    print("\n--- PHASE 1: Loading MambaFlow and AudioVAE ---")
    pl_mod = TTSMODEL.load_from_checkpoint(args.ckpt, map_location="cpu", strict=False)
    model = pl_mod.model.to(device).eval()
    del pl_mod
    gc.collect()

    weights_path = "/home/monesh/.cache/huggingface/hub/models--openbmb--VoxCPM1.5/snapshots/8cdc403854cda0e3af12252d27da038fda5982ac/audiovae.pth"
    vae_config = AudioVAEConfig(
        encoder_dim=64, encoder_rates=[2, 3, 6, 7, 7], latent_dim=64,
        decoder_dim=2048, decoder_rates=[7, 7, 6, 3, 2], sample_rate=44100
    )
    vae = AudioVAE(vae_config).to(device).eval()
    ckpt_vae = torch.load(weights_path, map_location=device, weights_only=True)
    vae.load_state_dict(ckpt_vae["state_dict"], strict=True)
    del ckpt_vae
    gc.collect()

    mean_np = np.load(os.path.join(CURRENT_DIR, "latent_mean_voxcpm.npy")).astype(np.float32)
    std_np = np.load(os.path.join(CURRENT_DIR, "latent_std_voxcpm.npy")).astype(np.float32)
    mean_t = torch.from_numpy(mean_np).to(device)
    std_t = torch.from_numpy(std_np).to(device)

    print(f"[+] Models loaded. Current GPU VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    # 4. Generate Audio & Measure RTF / Durations / RMS / Volume
    print("\n--- PHASE 2: Synthesizing Audio and Computing Signal / Duration Metrics ---")
    phase2_items = []
    t_synth_start = time.time()

    for idx, row in eval_df.iterrows():
        sample_id = str(row["id"])
        spk_id = str(row["speaker_id"])
        raw_text = str(row["original_text"]).strip()
        lat_path = str(row["latent_path"])
        spk_path = str(row["speaker_emb_path"])
        num_utts = int(row.get("num_speaker_utterances", 0))

        seq = text_to_sequence(raw_text, ["english_cleaners2"])[0]
        x_pad = torch.tensor([seq], dtype=torch.long, device=device)
        text_mask = torch.ones_like(x_pad, dtype=torch.bool)

        if spk_path and os.path.exists(spk_path):
            spk_emb = torch.load(spk_path, map_location="cpu", weights_only=False).float().unsqueeze(0).to(device)
            spk_emb = F.normalize(spk_emb, p=2, dim=-1)
        else:
            spk_emb = torch.zeros(1, 192, device=device)

        t0 = time.time()
        with torch.inference_mode():
            dec_out, enc_out, pred_frames = model.forward(
                x=x_pad,
                target_latent=None,
                spk_emb=spk_emb,
                mask=text_mask,
                n_timesteps=args.timesteps,
                temperature=args.temperature,
                solver=args.solver,
                time_schedule=args.time_schedule,
                cfg_scale=args.cfg_scale,
                length_scale=args.length_scale,
            )

            z_denorm = dec_out * (std_t + 1e-8) + mean_t
            z_slice = z_denorm.transpose(1, 2)
            wav_gen = vae.decode(z_slice).squeeze().cpu().numpy()

        gen_time = time.time() - t0
        audio_dur = len(wav_gen) / 44100.0
        rtf = gen_time / max(audio_dur, 1e-4)

        # Ground truth waveform via AudioVAE decode of GT latent
        wav_gt = None
        gt_dur_s = 0.0
        if lat_path and os.path.exists(lat_path):
            z_gt = torch.load(lat_path, map_location="cpu", weights_only=True)
            if z_gt.ndim == 2:
                if z_gt.shape[0] != 64 and z_gt.shape[1] == 64:
                    z_gt = z_gt.T
                z_gt = z_gt.unsqueeze(0)
            with torch.inference_mode():
                wav_gt = vae.decode(z_gt.to(device).float()).squeeze().cpu().numpy()
            gt_dur_s = len(wav_gt) / 44100.0

        dur_ratio = (audio_dur / gt_dur_s) if gt_dur_s > 0 else 1.0
        dur_diff_s = audio_dur - gt_dur_s

        # Audio level stats
        rms_gen, dbfs_gen, peak_gen = compute_audio_stats(wav_gen)
        rms_gt, dbfs_gt, peak_gt = compute_audio_stats(wav_gt) if wav_gt is not None else (0.0, -100.0, 0.0)

        # Resample to 16k for downstream evaluation
        w16_gen = librosa.resample(wav_gen, orig_sr=44100, target_sr=16000).astype(np.float32) if len(wav_gen) > 0 else np.zeros(16000, dtype=np.float32)
        w16_gt = librosa.resample(wav_gt, orig_sr=44100, target_sr=16000).astype(np.float32) if wav_gt is not None else None

        # Save audio files if within save_wav_count to conserve disk space
        gen_wav_path = None
        gt_wav_path = None
        if idx < args.save_wav_count:
            gen_wav_path = os.path.join(wavs_dir, f"{sample_id}_gen.wav")
            sf.write(gen_wav_path, wav_gen, 44100)
            if wav_gt is not None:
                gt_wav_path = os.path.join(wavs_dir, f"{sample_id}_gt.wav")
                sf.write(gt_wav_path, wav_gt, 44100)

        phase2_items.append({
            "sample_index": idx + 1,
            "id": sample_id,
            "speaker_id": spk_id,
            "num_speaker_utterances": num_utts,
            "raw_text": raw_text,
            "pred_frames": int(pred_frames.item()) if pred_frames is not None else int(dec_out.size(1)),
            "audio_dur_s": audio_dur,
            "gt_dur_s": gt_dur_s,
            "duration_ratio": dur_ratio,
            "duration_diff_s": dur_diff_s,
            "gen_time_s": gen_time,
            "rtf": rtf,
            "rms_gen": rms_gen,
            "rms_gt": rms_gt,
            "dbfs_gen": dbfs_gen,
            "dbfs_gt": dbfs_gt,
            "peak_gen": peak_gen,
            "peak_gt": peak_gt,
            "w16_gen": w16_gen,
            "w16_gt": w16_gt,
            "gen_wav_path": gen_wav_path,
            "gt_wav_path": gt_wav_path,
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_df):
            elapsed = time.time() - t_synth_start
            samples_per_sec = (idx + 1) / max(elapsed, 1e-4)
            rem_s = (len(eval_df) - (idx + 1)) / max(samples_per_sec, 1e-4)
            print(f"[{idx+1:03d}/{len(eval_df)}] Elapsed: {elapsed/60.0:.1f}m | ETA: {rem_s/60.0:.1f}m | Spk: {spk_id:<5} | Gen: {audio_dur:5.2f}s | Ratio: {dur_ratio:5.2f}x | GenRMS: {rms_gen:6.4f} | RTF: {rtf:5.3f}")

    print(f"[+] Synthesis complete in {(time.time() - t_synth_start)/60.0:.2f} min.")

    # Free MambaFlow and AudioVAE to release GPU memory
    del model, vae, mean_t, std_t
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Whisper ASR: WER, WSR, PER, PSR
    print("\n--- PHASE 3: Whisper ASR Evaluation (WER / WSR / PER / PSR) ---")
    from faster_whisper import WhisperModel
    try:
        whisper_model = WhisperModel("small.en", device="cuda", compute_type="float16")
    except Exception:
        whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

    t_asr_start = time.time()
    for idx, item in enumerate(phase2_items):
        try:
            segments, _ = whisper_model.transcribe(item["w16_gen"], language="en")
            transcription = " ".join([s.text for s in segments]).strip()
        except Exception as e:
            transcription = ""

        ref_clean = clean_text_for_asr(item["raw_text"])
        hyp_clean = clean_text_for_asr(transcription)
        wer = float(jiwer.wer(ref_clean, hyp_clean)) if ref_clean else (0.0 if not hyp_clean else 1.0)
        wsr = max(0.0, 1.0 - wer)

        ref_phonemes = get_phonemes_str(item["raw_text"])
        hyp_phonemes = get_phonemes_str(transcription)
        per = float(jiwer.wer(ref_phonemes, hyp_phonemes)) if ref_phonemes else 1.0
        psr = max(0.0, 1.0 - per)

        item["transcription"] = transcription
        item["wer"] = wer
        item["wsr"] = wsr
        item["per"] = per
        item["psr"] = psr

        if (idx + 1) % 50 == 0 or (idx + 1) == len(phase2_items):
            print(f"[ASR {idx+1:03d}/{len(phase2_items)}] Mean WER so far: {np.mean([x['wer'] for x in phase2_items[:idx+1]])*100:.2f}%")

    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[+] Whisper ASR complete in {(time.time() - t_asr_start)/60.0:.2f} min. Mean WER: {np.mean([x['wer'] for x in phase2_items])*100:.2f}%, WSR: {np.mean([x['wsr'] for x in phase2_items])*100:.2f}%")

    # 6. WavLM Speaker Similarity (SECS)
    print("\n--- PHASE 4: WavLM Speaker Similarity (SECS) ---")
    from transformers import WavLMForXVector, AutoFeatureExtractor
    wavlm_id = "microsoft/wavlm-base-plus-sv"
    wavlm_fe = AutoFeatureExtractor.from_pretrained(wavlm_id)
    wavlm_model = WavLMForXVector.from_pretrained(wavlm_id).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    t_wavlm_start = time.time()
    for idx, item in enumerate(phase2_items):
        w16_gen = item["w16_gen"]
        w16_gt = item["w16_gt"]
        spk_sim = 0.0
        if w16_gt is not None and len(w16_gt) > 0 and w16_gen is not None and len(w16_gen) > 0:
            try:
                w_gen_in = w16_gen if len(w16_gen) >= 8000 else np.pad(w16_gen, (0, 8000 - len(w16_gen)), mode="wrap")
                w_gt_in = w16_gt if len(w16_gt) >= 8000 else np.pad(w16_gt, (0, 8000 - len(w16_gt)), mode="wrap")
                with torch.inference_mode():
                    dev = next(wavlm_model.parameters()).device
                    i_gen = wavlm_fe(w_gen_in, return_tensors="pt", sampling_rate=16000).to(dev)
                    i_gt = wavlm_fe(w_gt_in, return_tensors="pt", sampling_rate=16000).to(dev)
                    e_gen = F.normalize(wavlm_model(**i_gen).embeddings, dim=-1)
                    e_gt = F.normalize(wavlm_model(**i_gt).embeddings, dim=-1)
                    sim_val = float(torch.mm(e_gen, e_gt.T).item())
                    spk_sim = max(0.0, min(1.0, sim_val)) if not np.isnan(sim_val) else 0.0
            except Exception as e:
                spk_sim = 0.0
        item["spk_sim"] = spk_sim

        if (idx + 1) % 50 == 0 or (idx + 1) == len(phase2_items):
            print(f"[WavLM {idx+1:03d}/{len(phase2_items)}] Mean SECS so far: {np.mean([x['spk_sim'] for x in phase2_items[:idx+1]])*100:.2f}%")

    del wavlm_model, wavlm_fe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[+] WavLM complete in {(time.time() - t_wavlm_start)/60.0:.2f} min. Mean Speaker Similarity: {np.mean([x['spk_sim'] for x in phase2_items])*100:.2f}%")

    # 7. DNSMOS (OVRL, SIG, BAK, P808)
    print("\n--- PHASE 5: DNSMOS Audio Quality Evaluation ---")
    from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)

    t_dns_start = time.time()
    for idx, item in enumerate(phase2_items):
        try:
            w16_t = torch.from_numpy(item["w16_gen"]).unsqueeze(0)
            scores = dnsmos_metric(w16_t)
            ovrl = float(scores[0].item())
            sig = float(scores[1].item())
            bak = float(scores[2].item())
            p808 = float(scores[3].item())
        except Exception as e:
            ovrl, sig, bak, p808 = 0.0, 0.0, 0.0, 0.0
        item["dnsmos_ovrl"] = ovrl
        item["dnsmos_sig"] = sig
        item["dnsmos_bak"] = bak
        item["dnsmos_p808"] = p808

        if (idx + 1) % 50 == 0 or (idx + 1) == len(phase2_items):
            print(f"[DNSMOS {idx+1:03d}/{len(phase2_items)}] Mean OVRL so far: {np.mean([x['dnsmos_ovrl'] for x in phase2_items[:idx+1]]):.3f}")

    del dnsmos_metric
    gc.collect()
    print(f"[+] DNSMOS complete in {(time.time() - t_dns_start)/60.0:.2f} min. Mean OVRL: {np.mean([x['dnsmos_ovrl'] for x in phase2_items]):.3f}")

    # 8. NISQA (MOS, NOI, DIS, COL, LOUD)
    print("\n--- PHASE 6: NISQA Speech Quality Evaluation ---")
    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
    nisqa_metric = NonIntrusiveSpeechQualityAssessment(fs=16000)

    t_nisqa_start = time.time()
    for idx, item in enumerate(phase2_items):
        try:
            w16_t = torch.from_numpy(item["w16_gen"]).unsqueeze(0)
            scores = nisqa_metric(w16_t)
            mos = float(scores[0].item())
            noi = float(scores[1].item())
            dis = float(scores[2].item())
            col = float(scores[3].item())
            loud = float(scores[4].item())
        except Exception as e:
            mos, noi, dis, col, loud = 0.0, 0.0, 0.0, 0.0, 0.0
        item["nisqa_mos"] = mos
        item["nisqa_noi"] = noi
        item["nisqa_dis"] = dis
        item["nisqa_col"] = col
        item["nisqa_loud"] = loud

        if (idx + 1) % 50 == 0 or (idx + 1) == len(phase2_items):
            print(f"[NISQA {idx+1:03d}/{len(phase2_items)}] Mean MOS so far: {np.mean([x['nisqa_mos'] for x in phase2_items[:idx+1]]):.3f}")

    del nisqa_metric
    gc.collect()
    print(f"[+] NISQA complete in {(time.time() - t_nisqa_start)/60.0:.2f} min. Mean MOS: {np.mean([x['nisqa_mos'] for x in phase2_items]):.3f}")

    # Remove heavy waveform objects
    for item in phase2_items:
        item.pop("w16_gen", None)
        item.pop("w16_gt", None)

    # Convert to DataFrame
    res_df = pd.DataFrame(phase2_items)

    # Speaker aggregations
    spk_agg = res_df.groupby("speaker_id").agg({
        "num_speaker_utterances": "first",
        "wer": ["mean", "count"],
        "wsr": "mean",
        "per": "mean",
        "psr": "mean",
        "spk_sim": "mean",
        "dnsmos_ovrl": "mean",
        "nisqa_mos": "mean",
        "rms_gen": "mean",
        "rms_gt": "mean",
        "dbfs_gen": "mean",
        "dbfs_gt": "mean",
        "duration_ratio": "mean",
        "audio_dur_s": "mean",
        "gt_dur_s": "mean",
    })
    spk_agg.columns = ["train_utts", "wer", "sample_count", "wsr", "per", "psr", "spk_sim", "dnsmos", "nisqa", "rms_gen", "rms_gt", "dbfs_gen", "dbfs_gt", "dur_ratio", "audio_dur_s", "gt_dur_s"]
    spk_agg = spk_agg.reset_index()

    # Composite Score
    spk_agg["composite_score"] = (
        0.30 * spk_agg["wsr"] +
        0.25 * spk_agg["spk_sim"] +
        0.20 * (spk_agg["nisqa"] / 5.0) +
        0.15 * (spk_agg["dnsmos"] / 5.0) +
        0.10 * np.clip(1.0 - np.abs(spk_agg["dur_ratio"] - 1.0), 0.0, 1.0)
    )

    # Correlations
    corr_utts_rms = float(spk_agg["train_utts"].corr(spk_agg["rms_gen"]))
    corr_utts_ratio = float(spk_agg["train_utts"].corr(spk_agg["dur_ratio"]))
    corr_gt_gen_rms = float(spk_agg["rms_gt"].corr(spk_agg["rms_gen"]))

    # Summary dictionary
    summary = {
        "checkpoint": args.ckpt,
        "epoch": epoch_num,
        "global_step": global_step,
        "solver": args.solver,
        "time_schedule": args.time_schedule,
        "timesteps": args.timesteps,
        "cfg_scale": args.cfg_scale,
        "temperature": args.temperature,
        "length_scale": args.length_scale,
        "num_samples": len(res_df),
        "num_unique_speakers": int(res_df["speaker_id"].nunique()),
        "metrics": {
            "wer_mean": float(res_df["wer"].mean()),
            "wsr_mean": float(res_df["wsr"].mean()),
            "per_mean": float(res_df["per"].mean()),
            "psr_mean": float(res_df["psr"].mean()),
            "rms_gen_mean": float(res_df["rms_gen"].mean()),
            "rms_gt_mean": float(res_df["rms_gt"].mean()),
            "dbfs_gen_mean": float(res_df["dbfs_gen"].mean()),
            "dbfs_gt_mean": float(res_df["dbfs_gt"].mean()),
            "dnsmos_ovrl_mean": float(res_df["dnsmos_ovrl"].mean()),
            "dnsmos_sig_mean": float(res_df["dnsmos_sig"].mean()),
            "dnsmos_bak_mean": float(res_df["dnsmos_bak"].mean()),
            "dnsmos_p808_mean": float(res_df["dnsmos_p808"].mean()),
            "nisqa_mos_mean": float(res_df["nisqa_mos"].mean()),
            "nisqa_noi_mean": float(res_df["nisqa_noi"].mean()),
            "nisqa_dis_mean": float(res_df["nisqa_dis"].mean()),
            "nisqa_col_mean": float(res_df["nisqa_col"].mean()),
            "nisqa_loud_mean": float(res_df["nisqa_loud"].mean()),
            "spk_sim_mean": float(res_df["spk_sim"].mean()),
            "rtf_mean": float(res_df["rtf"].mean()),
            "xrt_speed_mean": float((1.0 / res_df["rtf"].clip(lower=1e-4)).mean()),
            "duration_ratio_mean": float(res_df["duration_ratio"].mean()),
            "duration_diff_s_mean": float(res_df["duration_diff_s"].mean()),
            "gen_dur_s_mean": float(res_df["audio_dur_s"].mean()),
            "gt_dur_s_mean": float(res_df["gt_dur_s"].mean()),
        },
        "correlations": {
            "utts_vs_gen_rms": corr_utts_rms,
            "utts_vs_duration_ratio": corr_utts_ratio,
            "gt_rms_vs_gen_rms": corr_gt_gen_rms,
        },
        "speaker_aggregated": spk_agg.to_dict(orient="records"),
        "samples": res_df.to_dict(orient="records")
    }

    # Save outputs
    csv_out = os.path.join(args.out_dir, f"full_validation_metrics_n{len(res_df)}.csv")
    json_out = os.path.join(args.out_dir, f"full_validation_metrics_n{len(res_df)}.json")
    spk_csv_out = os.path.join(args.out_dir, f"speaker_aggregated_rankings_n{len(spk_agg)}.csv")
    res_df.to_csv(csv_out, index=False)
    spk_agg.to_csv(spk_csv_out, index=False)
    with open(json_out, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[+] Saved detailed sample CSV to: {csv_out}")
    print(f"[+] Saved speaker aggregated CSV to: {spk_csv_out}")
    print(f"[+] Saved detailed summary JSON to: {json_out}")

    print("\n" + "=" * 90)
    print("                    FULL VALIDATION BENCHMARK FINAL REPORT")
    print("=" * 90)
    m = summary["metrics"]
    print(f"Total Evaluated:   {len(res_df)} Samples across {int(res_df['speaker_id'].nunique())} Speakers")
    print(f"Intelligibility:   WER: {m['wer_mean']*100:5.2f}% | WSR: {m['wsr_mean']*100:5.2f}% | PER: {m['per_mean']*100:5.2f}% | PSR: {m['psr_mean']*100:5.2f}%")
    print(f"Audio Volume:      Gen RMS: {m['rms_gen_mean']:6.4f} (GT: {m['rms_gt_mean']:6.4f}) | Gen dBFS: {m['dbfs_gen_mean']:5.2f} dB (GT: {m['dbfs_gt_mean']:5.2f} dB)")
    print(f"Audio Quality:     DNSMOS OVRL: {m['dnsmos_ovrl_mean']:.3f} / 5.0 (SIG: {m['dnsmos_sig_mean']:.3f}, BAK: {m['dnsmos_bak_mean']:.3f}, P808: {m['dnsmos_p808_mean']:.3f})")
    print(f"Speech Quality:    NISQA MOS:   {m['nisqa_mos_mean']:.3f} / 5.0 (NOI: {m['nisqa_noi_mean']:.3f}, DIS: {m['nisqa_dis_mean']:.3f}, COL: {m['nisqa_col_mean']:.3f})")
    print(f"Speaker Fidelity:  SECS: {m['spk_sim_mean']*100:5.2f}% (WavLM Cosine Similarity)")
    print(f"Inference Speed:   RTF: {m['rtf_mean']:6.3f} ({m['xrt_speed_mean']:.1f}x real-time)")
    print(f"Duration Fidelity: Gen Dur: {m['gen_dur_s_mean']:.2f}s | GT Dur: {m['gt_dur_s_mean']:.2f}s | Ratio: {m['duration_ratio_mean']:.3f}x (Diff: {m['duration_diff_s_mean']:+.2f}s)")
    print("=" * 90)


if __name__ == "__main__":
    main()

