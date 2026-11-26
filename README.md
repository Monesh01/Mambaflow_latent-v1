# MambaFlow-Latent-v1: English Multi-Speaker TTS Flow Matching Model using LibriTTS-R

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/SSM-Mamba--2_BiDirectional-purple.svg)](https://github.com/state-spaces/mamba)
[![Sample Rate](https://img.shields.io/badge/Audio-44.1kHz_Continuous_Latent-green.svg)](https://huggingface.co/openbmb/VoxCPM1.5)

**MambaFlow-Latent-v1** is a compact, non-autoregressive closed-set multi-speaker English text-to-speech (TTS) system that couples **Continuous Flow Matching (CFM)** with **Bidirectional Mamba-2 State Space Models (SSMs)** operating directly over 64-dimensional continuous neural audio latents from VoxCPM 1.5 AudioVAE at **44.1 kHz** (**25 Hz** latent frame rate, 40 ms/frame).

Built and trained under student hardware constraints on a single **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)** using a curated **43.38-hour (146-speaker)** subset of LibriTTS-R `train-clean-100`, MambaFlow replaces quadratic $O(T^2)$ decoder self-attention with linear $O(T)$ state space decoders in a **19.95M-parameter** footprint (76.2 MiB checkpoint).

---

## Architecture Overview

```
                      +-----------------------------+
                      |     Input English Text      |
                      +-----------------------------+
                                     |
                                     v
                  +------------------------------------+
                  |  XTEncoder (6-Layer Transformer)   |  <-- 192D, RoPE, RMSNorm
                  +------------------------------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
+-----------------------+                         +--------------------------+
|  Duration Predictor   |                         |  Speaker Embedding (192D)|
| (3-Layer Speaker-Cond)|                         |  (146 Curated Speakers)  |
+-----------------------+                         +--------------------------+
           |                                                   |
           v                                                   |
+------------------------------------+                         |
| Length Regulator + SoftBoundaryConv|                         |
+------------------------------------+                         |
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                                     v
                  +------------------------------------+
                  |   Bidirectional Mamba-2 Decoder    |  <-- 6 Layers, 320D, expand=2
                  |   Flow Matching Velocity Field     |  <-- Decoupled Dual Modulation (t, spk)
                  +------------------------------------+
                                     |
                                     v  (16-step Midpoint ODE with Sway Scheduling)
                  +------------------------------------+
                  | Continuous 64D Latents @ 25 Hz     |
                  +------------------------------------+
                                     |
                                     v
                  +------------------------------------+
                  |    VoxCPM 44.1 kHz AudioVAE        |
                  +------------------------------------+
                                     |
                                     v
                  +------------------------------------+
                  |    44.1 kHz Synthesized WAV        |
                  +------------------------------------+
```

### Key Architectural Components

1. **Text Representation**: 6-stack Transformer encoder (`XTEncoder`) with Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU feed-forward blocks (`ff_mult=3`), 192 hidden dimensions, and 6 heads over a 179-token IPA/punctuation vocabulary.
2. **Deterministic Monotonic Duration Modeling**: A 3-layer speaker-conditioned Conv1D stack predicting per-phoneme continuous $\log(1+d)$ frame counts, followed by `repeat_interleave` expansion, a zero-initialized `SoftBoundaryResidualConv1d` ($k=3$), and sinusoidal frame positional embeddings—avoiding cross-attention alignment drift, repetition loops, and word skips.
3. **Linear-Time Velocity Field Decoder**: 6-layer Bidirectional Mamba-2 state space sequence mixer ($d=320$, `expand=2`, `d_state=64`, `headdim=64`, `mlp_mult=2`) with per-channel sigmoid-gated `text_proj` injection on later layers (3, 4, 5). Forward and backward SSD passes provide global context in $O(T)$ time.
4. **Decoupled Dual Modulation**: Timestep embedding $t \in [0, 1]$ and 192-dimensional L2-normalized ECAPA-TDNN speaker embeddings modulate layer-norm scaling and shift parameters (`ada_g`).
5. **Continuous Latent Space**: Predicts 64-dimensional Z-score-standardized continuous latents at 25 Hz for the pretrained VoxCPM 1.5 AudioVAE (`openbmb/VoxCPM1.5`), reconstructing 44.1 kHz audio without a separate mel vocoder.

---

## Dataset Curation & Hardware Constraints

- **Source Corpus**: LibriTTS-R `train-clean-100` (pre-extracted VoxCPM latent pool: **53.60 hours** across **247 speakers**, **33,232 utterances**, 4,823,740 frames at 25 Hz).
- **Speaker Filtering (`>= 100` utterances/speaker)**: Curated down to **146 speakers** totaling **28,018 utterances** and **43.38 hours** of speech (3,904,033 frames at 25 Hz).
- **Train / Validation Split**:
  - **Training set**: **27,434 utterances** across 146 speakers (**42.43 hours** / 3,819,053 frames).
  - **Validation set**: **584 utterances** across 146 speakers (4 utterances per speaker, **0.94 hours** / 84,980 frames).
- **Training Hardware**: Single **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** using `16-mixed` precision and `batch_size=16` (`lr=2e-4`, cosine schedule, 15% CFG dropout, `duration_weight=0.1`). Evaluated checkpoint: epoch **165** (`step 189,738`).

---

## Validation Benchmark Performance

Empirically evaluated on the full validation partition ($N=584$ utterances across all $146$ speakers; 4 per speaker) using the checkpoint at epoch 165 (`step 189,738`) with 16-step Midpoint ODE, Sway schedule, `cfg_scale=1.8`, `temperature=0.8`, and `length_scale=0.88`.

### Aggregate System Metrics

| Metric | Measured Score | Evaluation Instrument / Notes |
| :--- | :---: | :--- |
| **Word Error Rate (WER)** | **6.96%** | `faster-whisper` (`small.en`), standardized text normalization (median: **0.00%**, 348/584 exact matches) |
| **GT AudioVAE Reference WER** | **5.93%** | Identical Whisper `small.en` pipeline on AudioVAE-decoded GT val latents (median: **0.00%**, 361/584 exact; **+1.03%** gap vs GT floor) |
| **Word Success Rate (WSR)** | **93.38%** | Mean per-utterance $\max(0, 1 - \text{WER}_i)$ ($1 - \overline{\text{WER}} = 93.04\%$ due to one short `"XII."` sample with $\text{WER}=3.0$) |
| **Phoneme Error Rate (PER)** | **5.13%** | Espeak-ng G2P Levenshtein alignment |
| **Phoneme Success Rate (PSR)** | **94.87%** | $(1 - \text{PER})$ phonetic fidelity |
| **Speaker Cosine Similarity** | **0.9491 (94.9%)** | `microsoft/wavlm-base-plus-sv` vs AudioVAE-decoded GT (conditioned on per-utterance `val.csv` ECAPA vectors) |
| **NISQA Overall MOS** | **4.63 / 5.00** | Deep neural non-intrusive speech quality assessment (`torchmetrics`) |
| **NISQA Speech Distortion** | **4.62 / 5.00** | Spectral continuity & low distortion |
| **NISQA Natural Loudness** | **4.41 / 5.00** | Acoustic loudness score |
| **DNSMOS Overall Quality** | **3.65 / 5.00** | ITU-T P.835 deep neural perceptual quality metric |
| **DNSMOS Signal Distortion** | **3.35 / 5.00** | Speech clarity and articulation preservation |
| **DNSMOS Background Noise** | **3.90 / 5.00** | Clean background / high SNR score |
| **Real-Time Factor (RTF)** | **0.248** | RTX 3050 Laptop 4 GB GPU (~**4.03×** from mean RTF; **6.71×** mean per-sample inverse RTF) |
| **Model Parameter Count** | **19.95M** | 19,951,449 params (~76.2 MiB inference checkpoint) |

---

## Ranked Multi-Speaker Profile Indexing

All 146 curated speakers from LibriTTS-R `train-clean-100` have been quantitatively evaluated and indexed in `speaker_embeddings.pt` and `speaker_rankings.json`.

- **Sequential Index Access**: Speakers are sorted in descending order of composite validation quality ($0 \dots 145$).
- **Top Performer**: **Rank 0 corresponds to Speaker 4362**, which demonstrated the highest composite quality score across all validation trials.
- **Dual Querying & Key Collision Caveat**: You can query by rank (`0..145`) or LibriTTS-R speaker ID (e.g., `4362`). Note that 8 LibriTTS-R speaker IDs (`32, 40, 60, 78, 83, 87, 118, 125`) numerically overlap with ranks `0..145` in the flat `speaker_embeddings.pt` dictionary; consult `speaker_rankings.json` when targeting those specific speakers or ranks.

### Top 10 Calibrated Speakers

| Rank | Speaker ID | Composite Score | WavLM Sim | NISQA MOS | DNSMOS | WER | Training Utterances |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **4362** | **0.9466** | **0.9889** | **4.89** | **3.92** | **0.03** | 137 |
| **1** | **7505** | **0.9464** | **0.9907** | **4.71** | **3.94** | **0.02** | 204 |
| **2** | **3857** | **0.9462** | **0.9862** | **4.74** | **3.88** | **0.01** | 140 |
| **3** | **196** | **0.9446** | **0.9716** | **4.91** | **3.87** | **0.03** | 181 |
| **4** | **7447** | **0.9413** | **0.9852** | **4.76** | **3.90** | **0.04** | 183 |
| **5** | **446** | **0.9412** | **0.9610** | **4.79** | **3.98** | **0.03** | 104 |
| **6** | **3240** | **0.9396** | **0.9819** | **4.92** | **3.98** | **0.03** | 126 |
| **7** | **2092** | **0.9385** | **0.9787** | **4.77** | **4.02** | **0.04** | 128 |
| **8** | **8747** | **0.9379** | **0.9641** | **4.88** | **4.00** | **0.02** | 181 |
| **9** | **2911** | **0.9374** | **0.9833** | **4.83** | **3.86** | **0.04** | 134 |

---

## Technical Merits & Academic Limitations

### Powers & Strengths

- **Linear-Time ODE Flow Solver**: Unlike attention-based autoregressive models or quadratic diffusion decoders, Bidirectional Mamba-2 maintains $O(T)$ sequence mixing for 16-step midpoint ODE integration without quadratic sequence scaling.
- **Native 44.1 kHz Continuous Latent Representation**: Operates directly on 64D VoxCPM 1.5 AudioVAE latents, bypassing mel-spectrogram vocoder phase/aliasing artifacts and reconstructing 44.1 kHz audio.
- **Monotonic Duration Stability**: Explicit per-phoneme duration prediction + `repeat_interleave` prevents attention-based alignment drift and word looping.
- **Compact & Resource-Frugal**: 19.95M parameters (76.2 MiB checkpoint) trained from scratch on a 4 GB RTX 3050 Laptop GPU with **42.43 hours** of training speech, achieving **6.96% WER** against a **5.93%** GT AudioVAE Whisper floor.

### Weaknesses & Limitations (Why This Is an Academic Research Project, Not Production-Ready)

- **Closed-Set Multi-Speaker Only (No Zero-Shot Voice Cloning)**: Supports the **146 curated training speakers** from LibriTTS-R rather than arbitrary zero-shot reference audio cloning.
- **Limited Dataset Scale**: Trained on **42.43 hours** (**27,434 utterances**) across **146 speakers**, curated from the **53.60-hour / 247-speaker** `train-clean-100` latent pool (`>= 100` utterances/speaker), with **0.94 hours** (**584 utterances**) for validation. Production TTS systems (e.g., CosyVoice, XTTS, ElevenLabs) train on 10,000–100,000+ hours of multi-accented speech.
- **19th-Century Literary Text Domain Bias**: LibriTTS-R is derived from LibriVox public-domain audiobooks, so prosody and pronunciation reflect literary narrative English rather than spontaneous dialogue, modern slang, URLs, or technical terminology.
- **Text Normalization & Rare Short-Sequence Collapse**: Roman numerals (e.g., `XII.`, `CHAPTER XVIII`) are not expanded by `numbers.py`, and in full validation **6/584** samples (~1.0%) landing around `63–64` predicted frames collapsed to low-RMS outputs (`rms_gen < 0.02`), contributing directly to the +1.03% WER gap above ground truth.
- **Evaluation Conditioning Nuance**: `evaluate.py` uses per-utterance ECAPA embeddings from `val.csv` (`speaker_emb_path`) rather than the speaker-averaged dictionary in `speaker_embeddings.pt` used by `inference.py`.
- **Hardware Dependency for SSM Acceleration**: Fast synthesis requires CUDA hardware with `mamba-ssm` and `causal-conv1d` Triton kernels, plus `espeak-ng` installed on the system.

---

## Installation

### Prerequisites

- Linux (Ubuntu 20.04+ recommended) with system `espeak-ng` (`sudo apt-get install espeak-ng`)
- Python 3.10 or 3.12
- NVIDIA GPU with CUDA 11.8+ or 12.0+

```bash
# Clone the repository
git clone https://github.com/Monesh01/Mambaflow_latent-v1.git
cd Mambaflow_latent-v1

# Install core inference & training dependencies
pip install -r requirements.txt

# Optional: additional dependencies required only for evaluate.py
pip install faster-whisper transformers torchmetrics
```

> **Note on VoxCPM AudioVAE**: The AudioVAE weights (`audiovae.pth`) are automatically downloaded from Hugging Face Hub (`openbmb/VoxCPM1.5`) upon first inference run.

---

## Quickstart & Usage

### 1. Python API

```python
from inference import TTS

# Initialize the pipeline (auto-detects CUDA, model checkpoint, and AudioVAE)
tts = TTS()

# Synthesize using the top-performing speaker (Rank 0 -> Speaker 4362)
text = "MambaFlow represents a linear-time continuous flow matching architecture for speech synthesis."
audio = tts.generate(
    text=text,
    speaker_id=0,       # Rank 0 is top-calibrated speaker (4362)
    steps=16,           # 16-step ODE integration
    solver="midpoint",  # High-accuracy midpoint solver
    schedule="sway",    # Sway noise schedule
    temp=0.8,           # Prior sampling temperature
    length_scale=0.88,  # Natural speech cadence
    cfg_scale=1.8,      # Classifier-free guidance scale
    output_path="output_spk0.wav"
)

# List available speaker profiles
tts.list_speakers(top_k=10)

# Synthesize using an alternate speaker (e.g. LibriTTS ID 7505 or Rank 1)
audio_spk1 = tts.generate(
    text="Multi-speaker conditioning allows high voice consistency across diverse acoustic styles.",
    speaker_id=1,
    output_path="output_spk1.wav"
)
```

### 2. Command-Line Interface (CLI)

```bash
# Synthesize text with default top speaker (Rank 0 / 4362)
python inference.py --text "Hello world! This is MambaFlow speech synthesis." --output hello.wav

# Specify speaker rank and ODE steps
python inference.py --text "Linear state space decoders enable fast speech generation." --speaker_id 0 --steps 16 --output mamba.wav

# View all 146 calibrated speaker profiles
python inference.py --list_speakers
```

---

## Repository Structure

```
Mambaflow_latent-v1/
├── TTS_model.py                # Core MambaFlow-TTS architecture
├── mamba_decoder.py            # 6-Layer Bidirectional Mamba-2 velocity field ODE decoder
├── duration_predictor.py       # 3-Layer speaker-conditioned duration predictor
├── trans_encoder.py            # 6-Layer Transformer text encoder with RoPE
├── TTSDataModule.py            # PyTorch Lightning DataModule & LightningModule
├── TTSDatasetModule.py         # PyTorch Dataset loader & loss functions
├── TTSTraining.py              # Training script with atomic checkpointing
├── evaluate.py                 # Full validation benchmark suite (WER, PER, WavLM, NISQA, DNSMOS)
├── inference.py                # High-level inference API and CLI
├── speaker_embeddings.pt       # Calibrated 146 speaker embeddings (ranked 0..145 + ID keys)
├── speaker_rankings.json       # Metadata & validation metrics for all 146 speakers
├── latent_mean_voxcpm.npy      # VoxCPM latent normalization mean tensor
├── latent_std_voxcpm.npy       # VoxCPM latent normalization std tensor
├── requirements.txt            # Minimal pip dependencies
├── preprocessing/              # G2P phonemizer & text normalization
├── TTS_checkpoints/
│   └── mambaflow_latent_v1.ckpt # Standalone 76.2 MiB model checkpoint
└── eval_production_full_val/
    ├── full_validation_metrics_n584.json  # Comprehensive validation metric report
    └── speaker_aggregated_rankings_n146.csv # Per-speaker validation statistics
```

---

## Citation & Academic Acknowledgments

```bibtex
@software{monesh2026mambaflow,
  author = {Monesh, S.},
  title = {MambaFlow-Latent-v1: English Multi-Speaker TTS Flow Matching Model using LibriTTS-R},
  url = {https://github.com/Monesh01/Mambaflow_latent-v1},
  year = {2026}
}
```

- **Mamba-2**: Albert Gu and Tri Dao (State Spaces).
- **Continuous Flow Matching**: Lipman et al., Meta AI.
- **VoxCPM AudioVAE**: OpenBMB.
- **LibriTTS-R**: Google LLC & Heiga Zen et al.

