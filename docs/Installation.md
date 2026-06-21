# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchaudio 2.0+
- CUDA (optional, for GPU acceleration)

## pip install

```bash
pip install flashaudio
pip install "flashaudio[all]"   # all optional dependencies
```

## From source

```bash
git clone https://github.com/FlashVision/FlashAudio.git
cd FlashAudio
pip install -e ".[all]"
```

## Automatic setup

```bash
bash setup_env.sh              # auto-detects GPU
bash setup_env.sh --cpu        # CPU-only
bash setup_env.sh --cuda 12.4  # specific CUDA version
```

## Verify installation

```bash
flashaudio check
flashaudio settings
```

## System dependencies

On Linux, you may need:
```bash
sudo apt install libsndfile1 ffmpeg
```
