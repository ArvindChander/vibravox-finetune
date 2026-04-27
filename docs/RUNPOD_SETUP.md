# RunPod Setup for Vibravox Whisper Fine-Tuning

This guide is for running the `vibravox-finetune` project on a RunPod GPU pod with Linux, Python 3.12, CUDA, and a Hugging Face account.

## 1. Create a RunPod account

1. Go to https://www.runpod.io/ and create an account.
2. Add a payment method before launching a pod.
3. Open the pod console at https://console.runpod.io/.

## 2. Launch a GPU pod

1. In the RunPod console, open `Pods`.
2. Click `Deploy`.
3. Choose an `On-Demand` pod for training stability.
4. Prefer `Secure Cloud` first. Use `Community Cloud` only if pricing or availability forces it.

## 3. Choose the template

RunPod’s docs recommend using an official PyTorch template for AI workloads.

Use:
- the newest official `PyTorch` pod template available in the RunPod UI
- prefer a template based on CUDA `12.8` or `12.6`
- prefer Ubuntu `24.04` or another recent Linux base if shown

If the template list shows multiple PyTorch variants, pick the newest official one rather than an older community template.

Why this choice:
- PyTorch templates come with CUDA already wired up
- they are the least error-prone starting point for Whisper fine-tuning
- Python 3.12 is supported by current PyTorch releases on Linux

If the template already exposes Python 3.12, use it directly. If it ships with another Python version by default, create a Python 3.12 virtual environment inside the pod before installing this project.

## 4. GPU sizing for this project

Whisper-large with LoRA is still much lighter than full fine-tuning, but it needs materially more VRAM than Whisper-medium. You still need enough VRAM for:
- the base model
- activations
- Whisper feature tensors
- evaluation and generation overhead
- bf16 training with gradient checkpointing

Minimum workable GPU:
- `24 GB VRAM`
- examples: RTX 4090 24 GB, L4 24 GB, A10 24 GB, A5000 24 GB

Recommended GPU:
- `40 GB to 48 GB VRAM`
- examples: A100 40 GB, A6000 48 GB, L40/L40S 48 GB

Why not 8 GB:
- this project now targets Whisper-large on RunPod
- 8 GB is not realistic for this training configuration
- generation-based evaluation and LoRA fine-tuning are far more comfortable on 24 GB+

Practical expectation for VRAM:
- minimum workable: about `24 GB` with `bf16`, batch size `1`, gradient accumulation `16`, and gradient checkpointing
- safer headroom: `32 GB+`
- recommended for fewer surprises during eval/checkpointing: `40 GB+`

## 5. Disk and volume sizing

The Vibravox dataset card reports a large download size. Plan for the dataset cache, model weights, checkpoints, logs, and temporary files.

Recommended:
- container disk: `50 GB` minimum
- persistent volume: `250 GB` recommended

Reasoning:
- the full `speech_clean` subset is large
- Hugging Face datasets cache can consume substantial disk
- keeping checkpoints on a persistent volume avoids losing work when the pod stops

If you expect to cache multiple splits and rerun experiments, `300 GB+` is safer.

## 6. Open the pod terminal or connect by SSH

After the pod is running, you can connect in either of these ways:

### Browser terminal

1. Open the pod details page.
2. Click the terminal or web shell option.

### SSH

1. In the pod page, copy the SSH connection command shown by RunPod.
2. From your local machine, run that SSH command.
3. If needed, add your SSH public key to RunPod first.

Browser terminal is simplest for first-time setup. SSH is better for long sessions and editor integration.

## 7. Clone or upload this project

Inside the pod:

```bash
cd /workspace
git clone <your-repo-url> silent-speech
cd silent-speech/vibravox-finetune
```

If the project is not in Git yet, upload it with:
- RunPod file upload in the web UI
- `scp`
- `rsync`

Recommended destination:

```bash
/workspace/silent-speech/vibravox-finetune
```

## 8. Create and activate a Python 3.12 environment

First check what Python versions already exist:

```bash
python --version
python3 --version
python3.12 --version
```

If `python3.12` already exists:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

If `python3.12` is missing, use the newest official PyTorch image you launched and install Python 3.12 in the pod before continuing. On Ubuntu-based images, that usually means using the system package manager or selecting a newer template. Do not continue with Python 3.11 for this run because this project is now targeting Python 3.12 on RunPod.

## 9. Install requirements

Upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install project dependencies:

```bash
pip install -r requirements.txt
```

Verify the GPU build of PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `False`, the pod image or CUDA runtime is wrong. Stop there and relaunch with an official PyTorch CUDA template.

## 10. Authenticate with Hugging Face

Log in before touching the dataset:

```bash
huggingface-cli login
```

Paste a Hugging Face token from https://huggingface.co/settings/tokens

Recommended token scope:
- `Read` is enough for dataset access

You can also export a token for the current shell:

```bash
export HF_TOKEN=your_token_here
```

The scripts can use either:
- the cached Hugging Face login
- the `HF_TOKEN` environment variable

## 11. Accept the Vibravox dataset terms first

Before running any project script that loads the dataset:

1. Visit the dataset page:
   https://huggingface.co/datasets/Cnam-LMSSC/vibravox
2. Sign in to Hugging Face.
3. Accept the dataset terms or gated-access prompt on that page.
4. Wait a moment for access to propagate.

Important:
- if you skip this step, `src/check_data.py` is expected to stop with a clear message
- that is not treated as a code failure

## 12. Run the project stages

Run every command from:

```bash
/workspace/silent-speech/vibravox-finetune
```

### Stage 1: environment and data sanity check

```bash
python src/check_data.py
```

Expected behavior:
- prints Python, PyTorch, Transformers, Datasets, CUDA, GPU, VRAM, and working directory info
- tries to load the first 5 examples from `Cnam-LMSSC/vibravox`, config `speech_clean`
- if access is blocked, prints exactly what to do on Hugging Face and exits cleanly
- if access works, prints split sizes, example keys, transcript text, durations, and the discovered laryngophone field

### Stage 2: zero-shot baseline

Run only after confirming the discovered laryngophone field from Stage 1:

```bash
python src/baseline.py
```

### Stage 3: training

```bash
python src/train.py --config configs/run_v1.yaml
```

### Stage 4: final evaluation

```bash
python src/evaluate.py --checkpoint outputs/checkpoints/best
```

## 13. Monitoring during runs

Useful checks:

```bash
nvidia-smi
df -h
```

If you want logs saved:

```bash
python src/check_data.py | tee outputs/logs/check_data.log
```

The same pattern works for training and evaluation.

## 14. Stop the pod to avoid billing surprises

When you are not actively using the pod:

1. Save any outputs you need.
2. Push code changes to Git or copy outputs off the pod.
3. In RunPod, stop or terminate the pod.

Use:
- `Stop` if you want to keep the attached volume and resume later
- `Terminate` only when you are done and no longer need the pod or its attached storage

Important billing reminder:
- GPU charges continue while the pod is running
- attached storage can still cost money even when the pod is stopped
- delete unused pods and unused volumes when you are finished

## Suggested first session checklist

1. Launch an official PyTorch pod.
2. Verify CUDA is available.
3. Create and activate a Python 3.12 virtual environment.
4. Install `requirements.txt`.
5. Run `huggingface-cli login`.
6. Accept the Vibravox terms on the dataset page.
7. Run `python src/check_data.py`.
8. Confirm the laryngophone field name before continuing to Stage 2.
