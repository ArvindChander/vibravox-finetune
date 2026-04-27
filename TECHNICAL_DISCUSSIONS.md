# Technical Decisions (locked for v1)

## Model
- Base: openai/whisper-large
- Language token: French ("fr")
- Task: transcribe (not translate)

## PEFT (LoRA)
- Library: peft (Hugging Face)
- Rank: 32
- Alpha: 64
- Dropout: 0.05
- Target modules: q_proj, v_proj on BOTH encoder and decoder
- Bias: none
- Trainable params should be ~1% of total. Print this at startup.

## Data
- Dataset: Cnam-LMSSC/vibravox
- Config: "speech_clean"
- Sensor channel for v1: audio.body_conducted.throat.larynx_microphone
  (verify exact field name when loading; adjust if HF schema differs)
- Reference channel for ceiling baseline:
  audio.airborne.mouth_headworn.reference_microphone
- Sampling rate: resample everything to 16 kHz (Whisper requirement)
- Splits: use the dataset's built-in train/validation/test

## Audio preprocessing
- Use WhisperProcessor.feature_extractor (computes 80-bin log-mel)
- Pad/truncate to 30 seconds (Whisper's fixed window)

## Text preprocessing
- Use WhisperProcessor.tokenizer with language="fr", task="transcribe"
- Lowercase + strip punctuation BEFORE tokenization for WER fairness
- Use jiwer for WER computation (also lowercase + strip punctuation)

## Training
- Optimizer: AdamW (default in Trainer)
- Learning rate: 1e-4
- LR scheduler: linear warmup 500 steps, then constant
- Batch size: 1 per device, gradient accumulation 16 (effective batch 16)
- Steps: 5,000
- Precision: bf16 (NOT fp16 — Whisper encoder has fp16 overflow issues)
- Gradient checkpointing: ON (saves memory)
- Eval every 500 steps on a 200-sample validation subset
- Save best checkpoint by validation WER

## Evaluation
- Metric: WER via jiwer
- Test on 3 conditions:
  1. Zero-shot Whisper-large on laryngophone (baseline floor)
  2. Fine-tuned Whisper-large on laryngophone (our model)
  3. Zero-shot Whisper-large on airborne reference (ceiling)
- Report all 3 numbers in a results.md file

## Runtime assumptions
- RunPod GPU pod, not local training
- Linux
- Python 3.12
- CUDA-enabled NVIDIA GPU
- Prefer 40 GB to 48 GB VRAM for smoother training and evaluation
- Minimum workable GPU target: 24 GB VRAM

## What NOT to do
- Do NOT train all 6 sensors in v1. Just laryngophone.
- Do NOT add data augmentation in v1.
- Do NOT add an enhancement front-end. Direct fine-tune only.
- Do NOT use fp16. Use bf16.
- Do NOT fall back to local 8 GB assumptions for this v1 pipeline.
