# RunPod Runbook

This is the shortest practical guide for running the current repo on RunPod with minimal wasted spend.

## 1. Start the pod

Use:

- RunPod Pod, not Serverless
- 1 GPU
- your existing PyTorch 2.8 / CUDA 12.8 image
- persistent `/workspace` storage

## 2. Connect to the pod

Example:

```bash
ssh 6j1rcmwlilfogy-64412036@ssh.runpod.io -i ~/.ssh/runpod_ed25519
```

## 3. Enter the project and environment

```bash
cd /workspace/vibravox-finetune
source .venv/bin/activate
```

If `.venv` does not exist yet:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 4. Pull the latest code

```bash
git pull --rebase origin main
```

## 5. Log into Hugging Face

```bash
huggingface-cli login
```

Also make sure you already accepted:

- https://huggingface.co/datasets/Cnam-LMSSC/vibravox

## 6. Disable the problematic transfer flag

```bash
unset HF_HUB_ENABLE_HF_TRANSFER
```

## 7. Install the Xet helper once

```bash
pip install hf_xet
```

This improves reliability for Xet-backed Hugging Face storage.

## 8. Run the cheap checks first

### Stage 1

```bash
python src/check_data.py
```

### Stage 2 smoke test

```bash
mkdir -p outputs/logs
python src/baseline.py --max-samples 25 | tee outputs/logs/baseline_smoke.log
```

### Stage 2 full baseline

```bash
python src/baseline.py | tee outputs/logs/baseline_full.log
```

## 9. Use tmux for any long-running job

Start a session:

```bash
tmux new -s train_v1
```

Inside that session, go to the project and activate the environment again:

```bash
cd /workspace/vibravox-finetune
source .venv/bin/activate
unset HF_HUB_ENABLE_HF_TRANSFER
```

Detach safely:

- press `Ctrl+b`
- then press `d`

Reconnect later:

```bash
tmux attach -t train_v1
```

List sessions:

```bash
tmux ls
```

## 10. Use the debug config before the real training config

### Cheap debug training pass

This is the first thing to run after any training-code change:

```bash
python src/train.py --config configs/run_debug.yaml | tee outputs/logs/train_debug.log
```

What it does:

- 300 total steps
- eval every 100 steps
- save every 100 steps
- only 25 validation samples

This is the main anti-credit-bleed guardrail.

### Full training run

Only after the debug run is stable:

```bash
python src/train.py --config configs/run_v1.yaml | tee outputs/logs/train_v1.log
```

## 11. Resume from checkpoint if the run is interrupted

Find checkpoints:

```bash
find outputs/checkpoints -maxdepth 3 -type d | sort
```

Resume example:

```bash
python src/train.py --config configs/run_v1.yaml --resume-from-checkpoint outputs/checkpoints/run_v1/checkpoint-500 | tee outputs/logs/train_v1_resume.log
```

## 12. Monitor the machine

```bash
nvidia-smi
df -h /workspace
tail -n 80 outputs/logs/train_v1.log
```

## 13. When to stop the pod

Stop the pod whenever:

- no job is actively running
- you are leaving the machine
- you are debugging locally instead

Do not leave the pod running idle. Running pods still bill compute even when nothing useful is happening.

## 14. Recommended workflow summary

Use this exact sequence for a fresh run:

1. start the pod
2. connect by SSH
3. `cd /workspace/vibravox-finetune`
4. `source .venv/bin/activate`
5. `git pull --rebase origin main`
6. `unset HF_HUB_ENABLE_HF_TRANSFER`
7. `python src/check_data.py`
8. `python src/baseline.py --max-samples 25`
9. `tmux new -s train_v1`
10. `python src/train.py --config configs/run_debug.yaml`
11. if stable, run `python src/train.py --config configs/run_v1.yaml`
12. detach from tmux
13. reattach later to inspect progress
14. stop the pod immediately when the job finishes or fails
