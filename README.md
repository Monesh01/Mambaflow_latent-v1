# MambaFlow-Latent-v1: English Multi-Speaker TTS Flow Matching Model using LibriTTS-R

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/SSM-Mamba--2_BiDirectional-purple.svg)](https://github.com/state-spaces/mamba)
[![Sample Rate](https://img.shields.io/badge/Audio-44.1kHz_Continuous_Latent-green.svg)](https://huggingface.co/openbmb/VoxCPM1.5)

**MambaFlow-Latent-v1** is a high-fidelity, non-autoregressive multi-speaker text-to-speech (TTS) system that couples **Continuous Flow Matching (CFM)** with **Bidirectional Mamba-2 State Space Models (SSMs)** operating directly over continuous neural audio latents from VoxCPM AudioVAE at **44.1 kHz**.

By replacing quadratic $O(T^2)$ self-attention with linear $O(T)$ state space decoders, MambaFlow achieves fast, high-quality multi-speaker speech generation with a compact model footprint of **19.95M parameters** (76.2 MB checkpoint).

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
| (3-Layer Speaker-Cond)|                         |  (146 Calibrated Spks)   |
+-----------------------+                         +--------------------------+
           |                                                   |
           v                                                   |
+------------------------------------+                         |
| Length Regulator (Monotonic Frame) |                         |
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
                  | Continuous 64D Latents @ 44.1 kHz  |
                  +------------------------------------+
                                     |
                                     v
                  +------------------------------------+
                  |    VoxCPM 44.1 kHz AudioVAE        |
                  +------------------------------------+
                                     |
                                     v
                  +------------------------------------+
                  |    High-Fidelity 44.1 kHz WAV      |
                  +------------------------------------+
```

### Key Architectural Components

1. **Text Representation**: 6-stack Transformer encoder with Rotary Position Embeddings (RoPE), RMSNorm, and feed-forward multiplier of 3 (192 hidden dimensions, 6 heads).
2. **Deterministic Monotonic Duration Modeling**: A 3-layer speaker-conditioned convolution/linear stack predicting per-phoneme continuous $\log(1+d)$ frame counts. Frames are expanded via exact repetition, completely preventing cross-attention hallucinations, word skips, or looping.
3. **Linear-Time Velocity Field Decoder**: 6-layer Bidirectional Mamba-2 state space sequence mixer ($d=320$, expand=2, MLP multiplier=2). Both forward and backward recurrent state dynamics provide global context in $O(T)$ linear time and constant $O(1)$ memory per step.
4. **Decoupled Dual Modulation**: Timestep embedding $t \in [0, 1]$ and 192-dimensional speaker vectors modulate layer norm scaling and shift parameters independently without cross-talk.
5. **Continuous Latent Space**: Operates directly in 64-dimensional continuous latents from the pretrained VoxCPM AudioVAE, bypassing mel-spectrogram vocoder artifacts and directly reconstructing 44.1 kHz master audio.

---

## Validation Benchmark Performance

Empirically evaluated on the full validation partition of **LibriTTS-R** ($N=584$ test utterances across all $146$ unique speakers) using the checkpoint at epoch 165 (step 189,738).

### Aggregate System Metrics

| Metric | Measured Score | Evaluation Instrument / Standard |
| :--- | :---: | :--- |
| **Word Error Rate (WER)** | **6.96%** | OpenAI Whisper Small (`en`), standardized text normalization |
| **Word Success Rate (WSR)** | **93.38%** | $(1 - \text{WER})$ linguistic transcription accuracy |
| **Phoneme Error Rate (PER)** | **5.13%** | Espeak-ng G2P Levenshtein alignment |
| **Phoneme Success Rate (PSR)** | **94.87%** | $(1 - \text{PER})$ phonetic fidelity |
| **Speaker Cosine Similarity** | **0.9491 (94.9%)** | WavLM-TDNN pre-trained speaker verification ceiling |
| **NISQA Overall MOS** | **4.63 / 5.00** | Deep neural non-intrusive speech quality assessment |
| **NISQA Speech Distortion** | **4.62 / 5.00** | Clean, natural spectral continuity with minimal artifacts |
| **NISQA Natural Loudness** | **4.41 / 5.00** | Balanced acoustic volume and dynamics |
| **DNSMOS Overall Quality** | **3.65 / 5.00** | ITU-T P.835 deep neural perceptual quality metric |
| **DNSMOS Signal Distortion** | **3.35 / 5.00** | Speech clarity and articulation preservation |
| **DNSMOS Background Noise** | **3.90 / 5.00** | Clean background, high SNR reconstruction |
| **Real-Time Factor (RTF)** | **0.248** | GPU synthesis latency (6.71x faster than real-time) |
| **Model Parameter Count** | **19.95M** | Compact footprint (~76.2 MB checkpoint) |

---

## Ranked Multi-Speaker Profile Indexing

All 146 speakers from the LibriTTS-R dataset have been quantitatively evaluated and indexed in `speaker_embeddings.pt` and `speaker_rankings.json`.

- **Sequential Index Access**: Speakers are sorted in descending order of composite validation quality ($0 \dots 145$).
- **Top Performer**: **Rank 0 corresponds to Speaker 4362**, which demonstrated the highest composite quality score across all validation trials.
- **Dual Querying**: You can select a speaker either by their ranking index (`0` for best) or by their original LibriTTS-R speaker ID (`4362`).

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

- **Linear-Time ODE Flow Solver**: Unlike attention-based autoregressive models or diffusion decoders with quadratic complexity, Bidirectional Mamba-2 maintains $O(T)$ sequence operations, allowing fast 16-step integration without latency degradation on long phrases.
- **Native 44.1 kHz Continuous Latent Representation**: Directly produces studio-grade 44.1 kHz audio by utilizing pre-trained continuous AudioVAE latents, avoiding the phase distortion, metallic buzz, and frequency cutoff common to 80-channel mel-spectrogram vocoders.
- **Deterministic Monotonic Duration Stability**: Monotonic phoneme alignment eliminates word repetitions, skipping, and phantom hallucinations under diverse prompts.
- **Ultra-Compact Footprint**: 19.95M parameters with an inference checkpoint size of only 76.2 MB, making it viable for on-device and edge deployment.

### Weaknesses & Limitations (Why This Is an Academic Research Project, Not Production-Ready)

- **Limited Dataset Scale**: The model was trained on the clean partition of LibriTTS-R (~100 hours across 146 speakers). Industry production TTS models (e.g., ElevenLabs, XTTS, CosyVoice) are trained on 10,000+ to 100,000+ hours of multi-accented speech.
- **19th-Century Literary Text Domain Bias**: LibriTTS-R is derived from LibriVox audiobooks, which consist primarily of 19th-century public-domain classic literature. Consequently, the model's prosody, vocabulary distributions, and phrasing patterns reflect literary, Victorian, and narrative English prose. It may demonstrate unnatural intonation, hesitation, or mispronunciation when presented with contemporary colloquial dialogue, modern technical terminology, URLs, numbers, abbreviations, or conversational internet slang.
- **Absence of Spontaneous Conversational Dynamics**: Because audiobook recordings feature deliberate, well-enunciated reading styles, the model does not capture spontaneous conversational nuances such as natural laughter, hesitations, dynamic breathing variability, or colloquial vocal fry.
- **Out-of-Distribution Phoneme Sequencing**: Non-standard phonetic tokens or irregular foreign names not present in the CMU dictionary can cause minor duration prediction variance.
- **Hardware Dependency for SSM Acceleration**: Fast real-time factor requires CUDA hardware with support for `mamba-ssm` and `causal-conv1d` Triton kernels. CPU inference is functional but significantly slower.

---

## Installation

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- Python 3.10 or 3.12
- NVIDIA GPU with CUDA 11.8+ or 12.0+

```bash
# Clone the repository
git clone https://github.com/Monesh01/Mambaflow_latent-v1.git
cd Mambaflow_latent-v1

# Install dependencies
pip install -r requirements.txt
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
│   └── mambaflow_latent_v1.ckpt # Standalone 76.2 MB model checkpoint
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

