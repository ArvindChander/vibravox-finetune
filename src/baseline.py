"""Stage 2 zero-shot Whisper baseline for Vibravox laryngophone and airborne audio."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from datasets import Audio, Dataset, load_dataset, load_dataset_builder
from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DATASET_ID = "Cnam-LMSSC/vibravox"
CONFIG_NAME = "speech_clean"
MODEL_ID = "openai/whisper-large"
TARGET_TEXT_FIELD = "normalized_text"
LARYNGOPHONE_FIELD = "audio.throat_microphone"
REFERENCE_FIELD = "audio.headset_microphone"
RESULTS_PATH = Path("outputs/results.md")
MAX_NEW_TOKENS = 225


def print_header(title: str) -> None:
    """Print a compact section header for readable logs."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def print_environment_info() -> None:
    """Print package, platform, and GPU details for baseline reproducibility."""
    print_header("Environment")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Datasets version: {datasets.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        total_vram_gb = properties.total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(device_index) / (1024**3)
        print(f"GPU name: {properties.name}")
        print(f"GPU total VRAM (GB): {total_vram_gb:.2f}")
        print(f"GPU allocated VRAM right now (GB): {allocated_gb:.2f}")
        print(f"GPU reserved VRAM right now (GB): {reserved_gb:.2f}")
    else:
        print("GPU name: none")
        print("GPU total VRAM (GB): 0.00")


def load_test_split() -> Dataset:
    """Load the test split with only the fields needed for the zero-shot baseline."""
    print_header("Dataset")
    print(f"Loading dataset: {DATASET_ID}")
    print(f"Config: {CONFIG_NAME}")
    builder = load_dataset_builder(DATASET_ID, CONFIG_NAME)
    split_size = builder.info.splits["test"].num_examples
    print(f"Test split size from metadata: {split_size}")
    dataset = load_dataset(DATASET_ID, CONFIG_NAME, split="test")
    dataset = dataset.select_columns([LARYNGOPHONE_FIELD, REFERENCE_FIELD, TARGET_TEXT_FIELD, "duration"])
    dataset = dataset.cast_column(LARYNGOPHONE_FIELD, Audio(sampling_rate=16_000, decode=True))
    dataset = dataset.cast_column(REFERENCE_FIELD, Audio(sampling_rate=16_000, decode=True))
    print(f"Loaded test examples: {len(dataset)}")
    return dataset


def load_model_and_processor() -> tuple[WhisperProcessor, WhisperForConditionalGeneration, torch.device]:
    """Load zero-shot Whisper-large and prepare French transcription generation settings."""
    print_header("Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Requested dtype: {dtype}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="fr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    model.generation_config.language = "fr"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")

    if device.type == "cuda":
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU allocated VRAM after model load (GB): {allocated_gb:.2f}")
        print(f"GPU reserved VRAM after model load (GB): {reserved_gb:.2f}")

    return processor, model, device


def normalize_text(text: str) -> str:
    """Normalize text for fair WER computation with lowercasing and punctuation removal."""
    transform = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    return transform(text)


def transcribe_example(
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    audio_entry: dict[str, Any],
) -> str:
    """Run Whisper generation on one decoded audio example and return the transcript."""
    inputs = processor(
        audio_entry["array"],
        sampling_rate=audio_entry["sampling_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        predicted_ids = model.generate(input_features=input_features, max_new_tokens=MAX_NEW_TOKENS)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript


def run_channel_baseline(
    dataset: Dataset,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    audio_field: str,
    label: str,
) -> float:
    """Run zero-shot inference on one audio channel and return normalized WER."""
    print_header(f"Baseline: {label}")
    predictions: list[str] = []
    references: list[str] = []
    total_examples = len(dataset)
    total_duration_seconds = 0.0

    for index, example in enumerate(dataset):
        prediction = transcribe_example(processor, model, device, example[audio_field])
        reference = example[TARGET_TEXT_FIELD]
        normalized_prediction = normalize_text(prediction)
        normalized_reference = normalize_text(reference)

        predictions.append(normalized_prediction)
        references.append(normalized_reference)
        total_duration_seconds += float(example["duration"])

        if index < 3:
            print(f"Example {index} reference: {normalized_reference}")
            print(f"Example {index} prediction: {normalized_prediction}")

        if (index + 1) % 100 == 0 or index + 1 == total_examples:
            current_wer = wer(references, predictions)
            print(
                f"{label}: processed {index + 1}/{total_examples} examples | "
                f"running WER={current_wer:.4f} | audio hours={total_duration_seconds / 3600:.2f}"
            )
            if device.type == "cuda":
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"{label}: current GPU allocated VRAM (GB)={allocated_gb:.2f} | "
                    f"reserved VRAM (GB)={reserved_gb:.2f}"
                )

    final_wer = wer(references, predictions)
    print(f"{label} final WER: {final_wer:.4f}")
    return final_wer


def write_results(laryngophone_wer: float, reference_wer: float) -> None:
    """Write Stage 2 baseline results to outputs/results.md."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Results",
            "",
            "## Stage 2: Zero-Shot Baseline",
            "",
            f"- Laryngophone zero-shot floor (`{LARYNGOPHONE_FIELD}`): `{laryngophone_wer:.4f}` WER",
            f"- Airborne zero-shot approximate ceiling (`{REFERENCE_FIELD}`): `{reference_wer:.4f}` WER",
            "",
        ]
    )
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote baseline results to: {RESULTS_PATH}")


def main() -> int:
    """Run the full Stage 2 zero-shot baseline workflow and return a shell exit code."""
    print_environment_info()

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print_header("Hugging Face Transfer Mode")
        print("HF_HUB_ENABLE_HF_TRANSFER=1 detected.")
        print("If `hf_transfer` is not installed, run `unset HF_HUB_ENABLE_HF_TRANSFER` before retrying.")

    dataset = load_test_split()
    processor, model, device = load_model_and_processor()

    print_header("Task Setup")
    print(f"Target transcript field: {TARGET_TEXT_FIELD}")
    print(f"Laryngophone field: {LARYNGOPHONE_FIELD}")
    print(f"Airborne reference field: {REFERENCE_FIELD}")

    laryngophone_wer = run_channel_baseline(dataset, processor, model, device, LARYNGOPHONE_FIELD, "Laryngophone")
    reference_wer = run_channel_baseline(dataset, processor, model, device, REFERENCE_FIELD, "Airborne reference")

    print_header("Final Results")
    print(f"Laryngophone zero-shot floor WER: {laryngophone_wer:.4f}")
    print(f"Airborne zero-shot approximate ceiling WER: {reference_wer:.4f}")
    write_results(laryngophone_wer, reference_wer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
