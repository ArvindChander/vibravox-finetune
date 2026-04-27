# Environment and Run Instructions

## Python
- Python 3.12 on RunPod Linux GPU pods

## Required packages
torch>=2.7.0           # CUDA-enabled Linux build for RunPod
transformers>=4.45.0
peft>=0.13.0
datasets>=3.0.0
accelerate>=0.34.0
jiwer>=3.0.4
librosa>=0.10.0
soundfile>=0.12.1
evaluate>=0.4.0

## Hugging Face authentication
Vibravox requires accepting the dataset terms on the HF page first,
then `huggingface-cli login` with a read-capable token.

## Runtime target
- RunPod GPU pod
- Linux
- Python 3.12
- CUDA-enabled NVIDIA GPU
- Prefer a 40 GB to 48 GB GPU for Whisper-large LoRA fine-tuning
- Minimum workable GPU is 24 GB VRAM with conservative settings

## Folder layout to create
src/
  check_data.py  # environment + gated-access-aware dataset sanity check
  baseline.py    # zero-shot Whisper baseline
  data.py        # dataset loading + preprocessing
  model.py       # Whisper + LoRA setup
  train.py       # training loop entry point
  evaluate.py    # WER computation on test set
configs/
  run_v1.yaml    # all hyperparameters in one place
outputs/
  checkpoints/   # LoRA adapter saves
  logs/          # training logs
  results.md     # final WER numbers

## Run commands (target)
python src/check_data.py
python src/baseline.py
python src/train.py --config configs/run_v1.yaml
python src/evaluate.py --checkpoint outputs/checkpoints/best
