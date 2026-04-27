# Vibravox Whisper Fine-Tuning — Project Brief

## Goal
Fine-tune OpenAI Whisper on the Vibravox laryngophone channel
to transcribe French body-conducted speech. Measure WER vs.
zero-shot Whisper baseline.

## Why this dataset
Vibravox (Cnam-LMSSC, 2024) is the only open multi-sensor
body-conducted speech corpus with paired air-mic reference and
both orthographic and IPA phonetic transcripts. 188 French
speakers, ~45 hours per sensor, 6 synchronized sensors per
utterance, CC-BY-4.0 on Hugging Face.

Hugging Face: Cnam-LMSSC/vibravox
Paper: https://arxiv.org/abs/2407.11828
Project: https://vibravox.cnam.fr/

## Why this approach
1. Direct fine-tune (not enhancement cascade) keeps pipeline simple.
2. LoRA (rank 32) keeps memory footprint manageable on rented cloud GPUs
   and prevents catastrophic forgetting of Whisper's French.
3. Single sensor (laryngophone) first; later expand to all 6 for
   sensor-comparison study.

## Runtime environment
RunPod GPU pod with Linux, Python 3.12, CUDA-enabled NVIDIA GPU,
and Hugging Face authentication. This project now targets Whisper-large
on cloud hardware rather than Whisper-medium on a local 8 GB GPU.

## Success criteria for v1
- Pipeline runs end-to-end without crashing.
- Training loss decreases monotonically over 5,000 steps.
- Test-set WER on laryngophone channel is meaningfully lower
  than zero-shot Whisper-large WER on the same channel.
- All numbers logged and reproducible.
