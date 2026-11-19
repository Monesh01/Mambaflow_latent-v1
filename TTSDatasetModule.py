import re
import os
import json
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
try:
    from preprocessing.text import text_to_sequence
except (ImportError, ValueError):
    from TTSModel.libri.preprocessing.text import text_to_sequence


def clean_path(path_str, mount_prefix="/home/monesh/TTSModel/libri"):
    if not isinstance(path_str, str):
        return path_str
    path_str = path_str.replace("\\", "/")
    if path_str.startswith("C:/libri/") or path_str.startswith("C:/libri"):
        path_str = re.sub(r"^C:/libri/*", "/home/monesh/TTSModel/libri/", path_str)
        return os.path.normpath(path_str)
    if path_str.startswith("/home/"):
        return os.path.normpath(path_str)
    if path_str.startswith(mount_prefix):
        return os.path.normpath(path_str)
    if re.match(r"^[a-zA-Z]:", path_str):
        path_str = re.sub(r"^[a-zA-Z]:[/]*", "", path_str)
        return os.path.normpath(os.path.join(mount_prefix, path_str.lstrip("/")))
    return os.path.normpath(path_str)


def collate_fn(batch):
    mels = [
        torch.from_numpy(x["mel"]) if isinstance(x["mel"], np.ndarray)
        else torch.as_tensor(x["mel"], dtype=torch.float32)
        for x in batch
    ]
    mels = pad_sequence(mels, batch_first=True)

    lengths = torch.tensor([x["mel"].shape[0] for x in batch], dtype=torch.long)

    text_seqs = [x["ipa_seq"] for x in batch]
    ipa = pad_sequence([torch.as_tensor(x, dtype=torch.long) for x in text_seqs], batch_first=True, padding_value=0)
    ipa_lengths = torch.tensor([len(x) for x in text_seqs], dtype=torch.long)

    dur_seqs = [x["durations"] for x in batch]
    durations = pad_sequence([torch.as_tensor(x, dtype=torch.long) for x in dur_seqs], batch_first=True, padding_value=0)

    # Stack precomputed normalized speaker embeddings [B, 192]
    spk_embs = [
        x["spk_emb"] if isinstance(x["spk_emb"], torch.Tensor)
        else torch.as_tensor(x["spk_emb"], dtype=torch.float32)
        for x in batch
    ]
    spk_emb = torch.stack(spk_embs, dim=0)

    return {
        "mel": mels,
        "latent": mels,
        "lengths": lengths,
        "ipa_lengths": ipa_lengths,
        "ipa": ipa,
        "durations": durations,
        "spk_emb": spk_emb,
    }


class MyDataset(Dataset):
    """
    Dataset Module for VoxCPM / LibriTTS-R Multi-Speaker Voice Cloning:
    - Loads continuous latents from .pt or .npy files
    - Applies channel-wise Z-Score standardization (mean=0, std=1.0)
    - Loads normalized speaker embeddings
    - Text phonemization
    - Loads and resamples per-phoneme MFA durations matching text sequence and latent frames
    """
    def __init__(self, csv_path, mount_prefix="/home/monesh/TTSModel/libri", target_dim=64, prefetch_data=False, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.target_dim = target_dim
        self.mount_prefix = mount_prefix
        self.is_train = is_train
        for col in ['latent', 'latent_path', 'kvae_latent', 'mel_path', 'mel_spectrogram_path', 'mel']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).apply(lambda p: clean_path(p, mount_prefix))
        self.phoneme_cache = {}
        self.prefetch_data = prefetch_data

        self.latent_mean = None
        self.latent_std = None

        cand_dirs = [
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.abspath(csv_path)),
            os.getcwd(),
        ]
        mean_path, std_path = None, None
        pref_names = ["latent_mean_voxcpm.npy", f"latent_mean_{target_dim}.npy", "latent_mean.npy"] if target_dim == 64 else ["latent_mean.npy"]
        for pref in pref_names:
            s_name = pref.replace("mean", "std")
            for d in cand_dirs:
                mp = os.path.join(d, pref)
                sp = os.path.join(d, s_name)
                if os.path.exists(mp) and os.path.exists(sp):
                    arr_m = np.load(mp)
                    if arr_m.shape[0] == target_dim:
                        mean_path, std_path = mp, sp
                        break
            if mean_path is not None:
                break

        if mean_path is not None and std_path is not None:
            self.latent_mean = np.load(mean_path).astype(np.float32)
            self.latent_std = np.load(std_path).astype(np.float32)
            print(f"[+] Loaded {self.latent_mean.shape[0]}-dim latent_mean and latent_std for Z-Score standardization from: {mean_path}")
        else:
            try:
                from stats import LATENT_MEAN_ARR, LATENT_STD_ARR
                if LATENT_MEAN_ARR.shape[0] == target_dim:
                    self.latent_mean = LATENT_MEAN_ARR
                    self.latent_std = LATENT_STD_ARR
                    print(f"[+] Loaded {self.latent_mean.shape[0]}-dim LATENT_MEAN and LATENT_STD from stats.py for Z-Score standardization.")
            except ImportError:
                print("[-] Warning: No matching latent mean/std statistics found. Using unnormalized latents.")

        # Latent std safety check: if dynamic range across channels is extreme (> 4.0), use global scalar std
        if self.latent_std is not None and (self.latent_std.max() / (self.latent_std.min() + 1e-8)) > 4.0:
            self.effective_latent_std = float(self.latent_std.mean())
            print(f"[!] latent_std max/min ratio > 4.0 ({self.latent_std.max() / (self.latent_std.min() + 1e-8):.2f}). Using global scalar std: {self.effective_latent_std:.4f}")
        else:
            self.effective_latent_std = self.latent_std

        # Load precomputed normalized speaker embeddings dictionary
        self.speaker_embs = {}
        spk_emb_candidates = [
            os.path.join(mount_prefix, "speaker_embeddings.pt"),
            os.path.join(mount_prefix, "processed_data/speaker_embeddings_avg_norm.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_embeddings.pt"),
            "/home/monesh/TTSModel/libri/speaker_embeddings.pt",
        ]
        for p in spk_emb_candidates:
            if os.path.exists(p):
                self.speaker_embs = torch.load(p, map_location="cpu", weights_only=False)
                print(f"[+] Loaded normalized average speaker embeddings for {len(self.speaker_embs)} entries from: {p}")
                break
        if not self.speaker_embs:
            print("[-] Warning: speaker_embeddings.pt not found! Will fallback to zero embeddings.")

        # Pre-extract paths, texts, and speakers into fast Python lists
        self.latent_paths = []
        for col in ['latent', 'latent_path', 'kvae_latent', 'mel_path', 'mel_spectrogram_path', 'mel']:
            if col in self.df.columns:
                self.latent_paths = [clean_path(str(p), mount_prefix) for p in self.df[col]]
                break

        # Utterance-level ECAPA embeddings path
        self.speaker_emb_paths = []
        for col in ['speaker_emb_path', 'spk_emb_path', 'spk_path', 'ecapa_path']:
            if col in self.df.columns:
                self.speaker_emb_paths = [clean_path(str(p), mount_prefix) for p in self.df[col]]
                print(f"[+] Loaded {len(self.speaker_emb_paths)} per-utterance speaker embedding paths from '{col}'.")
                break

        text_col = None
        for col in ['original_text', 'normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in self.df.columns:
                text_col = col
                break
        self.texts = [str(t) for t in self.df[text_col].fillna('')] if text_col is not None else [""] * len(self.df)
        self.speaker_ids = self.df['speaker_id'].tolist() if 'speaker_id' in self.df.columns else [None] * len(self.df)

        # Raw durations from CSV
        self.raw_durations = []
        if 'durations' in self.df.columns:
            for val in self.df['durations']:
                if pd.isna(val):
                    self.raw_durations.append([])
                elif isinstance(val, (list, np.ndarray)):
                    self.raw_durations.append(list(val))
                elif isinstance(val, str):
                    try:
                        self.raw_durations.append(json.loads(val))
                    except Exception:
                        try:
                            self.raw_durations.append(ast.literal_eval(val))
                        except Exception:
                            self.raw_durations.append([])
                else:
                    self.raw_durations.append([])
        else:
            self.raw_durations = [[]] * len(self.df)

        # In-Memory Pre-Tokenization of all phoneme sequences
        print(f"[+] Pre-tokenizing {len(self.df)} texts into phoneme sequences...")
        self.ipa_sequences = [
            self._get_phonemes(t) for t in self.texts
        ]
        print(f"[+] Pre-tokenization complete.")

        # In-memory cache for standardized latents
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _get_phonemes(self, text):
        if text not in self.phoneme_cache:
            seq = text_to_sequence(text, ["english_cleaners2"])[0]
            self.phoneme_cache[text] = seq
        return self.phoneme_cache[text]

    def _load_latent(self, target_path):
        if self.prefetch_data and target_path in self.cache:
            return self.cache[target_path]

        if target_path.endswith('.pt'):
            _raw = torch.load(target_path, weights_only=True, map_location="cpu")
            if isinstance(_raw, torch.Tensor):
                _raw = _raw.float().numpy()
            else:
                _raw = np.array(_raw, dtype=np.float32)
        else:
            _raw = np.squeeze(np.load(target_path, mmap_mode="r" if not self.prefetch_data else None))

        if _raw.ndim == 2:
            if _raw.shape[0] in [16, 32, 64, 72, 80, 100, 128] and _raw.shape[1] != _raw.shape[0]:
                _raw = _raw.T
            elif _raw.shape[0] < _raw.shape[1] and _raw.shape[0] <= 128:
                _raw = _raw.T

        _latent = _raw.astype(np.float32)

        # Per-dimensional Z-Score Standardization: (x - mean) / (std + 1e-8) -> mean=0, std=1.0
        if self.latent_mean is not None and self.effective_latent_std is not None:
            _latent = (_latent - self.latent_mean) / (self.effective_latent_std + 1e-8)

        if self.prefetch_data:
            self.cache[target_path] = _latent
        return _latent

    def __getitem__(self, idx):
        target_path = self.latent_paths[idx]
        _latent = self._load_latent(target_path)
        _text = self.texts[idx]
        _ipa_seq = self.ipa_sequences[idx]

        # 1. Per-utterance L2-normalized ECAPA vector
        spk_emb = None
        if self.speaker_emb_paths and idx < len(self.speaker_emb_paths):
            p_spk = self.speaker_emb_paths[idx]
            if p_spk and os.path.exists(p_spk):
                try:
                    if p_spk.endswith('.pt'):
                        raw_spk = torch.load(p_spk, weights_only=True, map_location="cpu")
                    else:
                        raw_spk = np.load(p_spk)
                    if isinstance(raw_spk, np.ndarray):
                        spk_emb = torch.from_numpy(raw_spk).float()
                    else:
                        spk_emb = raw_spk.float()
                    spk_emb = F.normalize(spk_emb, p=2, dim=-1)
                except Exception:
                    spk_emb = None

        # 2. Fall back to speaker-wide average dictionary
        if spk_emb is None:
            spk = self.speaker_ids[idx]
            if self.speaker_embs:
                if spk in self.speaker_embs:
                    spk_emb = self.speaker_embs[spk]
                elif str(spk) in self.speaker_embs:
                    spk_emb = self.speaker_embs[str(spk)]
                elif int(spk) in self.speaker_embs:
                    spk_emb = self.speaker_embs[int(spk)]
            if spk_emb is not None:
                if not isinstance(spk_emb, torch.Tensor):
                    spk_emb = torch.tensor(spk_emb, dtype=torch.float32)
                else:
                    spk_emb = spk_emb.float()
                spk_emb = F.normalize(spk_emb, p=2, dim=-1)

        # 3. Fall back to zero vector
        if spk_emb is None:
            spk_emb = torch.zeros(192, dtype=torch.float32)

        # 4. MFA phone durations directly match _ipa_seq and latent frame length
        durations = self.raw_durations[idx]
        if len(durations) != len(_ipa_seq):
            if len(durations) < len(_ipa_seq):
                durations = durations + [0] * (len(_ipa_seq) - len(durations))
            else:
                durations = durations[:len(_ipa_seq)]

        return {
            "mel": _latent,
            "latent": _latent,
            "text": _text,
            "ipa_seq": _ipa_seq,
            "durations": durations,
            "spk_emb": spk_emb,
            "speaker_id": self.speaker_ids[idx],
        }


def align_seq_len(predict, target, mask):
    min_len = min(predict.size(1), target.size(1), mask.size(1))
    return predict[:, :min_len], target[:, :min_len], mask[:, :min_len]


def masked_huber_loss(predict, target, mask, delta=1.0):
    """
    Masked Huber (Smooth L1) loss across valid sequence tokens.
    Outlier-resistant to preserve crisp high-frequency latent variance.
    """
    predict, target, mask = align_seq_len(predict, target, mask)
    if predict.ndim == 3 and target.ndim == 2:
        target = target.unsqueeze(-1)

    loss = F.huber_loss(predict, target, reduction='none', delta=delta)
    mask_float = mask.unsqueeze(-1).float() if mask.ndim < loss.ndim else mask.float()
    loss = loss * mask_float
    num_feats = predict.shape[-1] if predict.ndim > 2 else 1
    valid_elements = mask_float.sum() * num_feats
    return loss.sum() / (valid_elements + 1e-8)


def masked_mse_loss(predict, target, mask):
    """Masked Mean Squared Error loss across valid sequence tokens."""
    predict, target, mask = align_seq_len(predict, target, mask)
    if predict.ndim == 3 and target.ndim == 2:
        target = target.unsqueeze(-1)

    loss = F.mse_loss(predict, target, reduction='none')
    mask_float = mask.unsqueeze(-1).float() if mask.ndim < loss.ndim else mask.float()
    loss = loss * mask_float
    num_feats = predict.shape[-1] if predict.ndim > 2 else 1
    valid_elements = mask_float.sum() * num_feats
    return loss.sum() / (valid_elements + 1e-8)
