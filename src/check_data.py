"""Stage 1 sanity checks for the Vibravox Whisper fine-tuning project."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import HfFolder, login
from huggingface_hub.errors import HfHubHTTPError


DATASET_ID = "Cnam-LMSSC/vibravox"
CONFIG_NAME = "speech_clean"
MAX_EXAMPLES = 5
REFERENCE_FIELD_HINT = "audio.headset_microphone"


def print_header(title: str) -> None:
    """Print a small section header to keep console output readable."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def print_environment_info() -> None:
    """Print Python, package, and GPU information useful for debugging."""
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


def authenticate_hugging_face() -> str | None:
    """Use an existing Hugging Face login or `HF_TOKEN`, otherwise exit cleanly."""
    print_header("Hugging Face Authentication")
    token = HfFolder.get_token()
    if token:
        print("Found an existing Hugging Face token in the local cache.")
        return token

    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            print(f"Found token in environment variable: {env_name}")
            login(token=token, add_to_git_credential=False)
            return token

    print("No Hugging Face token was found.")
    print("Run `huggingface-cli login` or export `HF_TOKEN=...`, then rerun this script.")
    return None


def is_gated_access_error(error: Exception) -> bool:
    """Return True when a dataset load error looks like gated-access or permission denial."""
    text = str(error).lower()
    markers = (
        "gated",
        "403",
        "401",
        "forbidden",
        "unauthorized",
        "accept",
        "terms",
        "license",
        "access to dataset",
        "authentication",
        "not authorized",
        "you are trying to access a gated repo",
    )
    return any(marker in text for marker in markers)


def print_gated_access_instructions(error: Exception) -> None:
    """Explain how to unlock dataset access without treating it as a code failure."""
    print_header("Dataset Access Blocked")
    print(f"Could not access {DATASET_ID}/{CONFIG_NAME}.")
    print("This is not treated as a code failure.")
    print("Most likely causes:")
    print("1. You are not logged into Hugging Face in this environment.")
    print("2. You have not accepted the Vibravox dataset terms yet.")
    print("What to do next:")
    print("1. Visit: https://huggingface.co/datasets/Cnam-LMSSC/vibravox")
    print("2. Sign in to your Hugging Face account.")
    print("3. Accept the dataset terms or gated-access prompt on that page.")
    print("4. Run `huggingface-cli login` in this pod if you have not already.")
    print("5. Rerun: python src/check_data.py")
    print(f"Original loader message: {error}")


def flatten_example_keys(example: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively flatten keys from a nested example for schema inspection."""
    keys: list[str] = []
    for key, value in example.items():
        current = f"{prefix}.{key}" if prefix else key
        keys.append(current)
        if isinstance(value, dict):
            keys.extend(flatten_example_keys(value, prefix=current))
    return keys


def find_audio_columns(column_names: list[str]) -> list[str]:
    """Return column names that look like audio fields in the dataset schema."""
    return [name for name in column_names if name.startswith("audio.")]


def discover_laryngophone_field(audio_columns: list[str]) -> str | None:
    """Infer the laryngophone field from the actual schema instead of hardcoding it."""
    keywords = ("lary", "throat", "neck")
    candidates = [name for name in audio_columns if any(word in name.lower() for word in keywords)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def pick_transcript_field(example: dict[str, Any]) -> str | None:
    """Choose a transcript-like field from the example for readable logging."""
    preferred = ("normalized_text", "raw_text", "text", "transcript", "sentence")
    for field in preferred:
        if field in example and isinstance(example[field], str):
            return field
    return None


def format_duration(example: dict[str, Any], audio_field: str | None) -> str:
    """Return a duration string using the dataset duration field or decoded audio length."""
    if "duration" in example and isinstance(example["duration"], (int, float)):
        return f"{example['duration']:.2f} seconds"

    if audio_field and audio_field in example and isinstance(example[audio_field], dict):
        audio = example[audio_field]
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if array is not None and sampling_rate:
            duration = len(array) / sampling_rate
            return f"{duration:.2f} seconds"

    return "unknown"


def print_split_sizes(dataset_dict: DatasetDict) -> None:
    """Print the number of rows in each built-in dataset split."""
    print_header("Split Sizes")
    for split_name, split_dataset in dataset_dict.items():
        print(f"{split_name}: {len(split_dataset)} examples")


def print_example_report(dataset_dict: DatasetDict) -> str | None:
    """Print the first examples, field keys, transcript text, and discovered audio schema."""
    print_header("Schema And Sample Inspection")
    train_split = dataset_dict["train"]
    decoded_train = train_split.cast_column(REFERENCE_FIELD_HINT, Audio(decode=True))
    example_zero = decoded_train[0]
    flattened_keys = flatten_example_keys(example_zero)
    audio_columns = find_audio_columns(decoded_train.column_names)
    laryngophone_field = discover_laryngophone_field(audio_columns)
    transcript_field = pick_transcript_field(example_zero)

    print("Top-level column names:")
    for name in decoded_train.column_names:
        print(f"- {name}")

    print("\nFlattened keys from the first example:")
    for key in flattened_keys:
        print(f"- {key}")

    print("\nAudio-like columns discovered from the dataset schema:")
    for column in audio_columns:
        print(f"- {column}")

    if laryngophone_field:
        print(f"\nDiscovered laryngophone field: {laryngophone_field}")
    else:
        print("\nCould not uniquely infer the laryngophone field from the schema.")

    print(f"Reference channel field observed: {REFERENCE_FIELD_HINT}")

    if transcript_field:
        print(f"Transcript field selected for preview: {transcript_field}")
    else:
        print("No obvious transcript field was found in the first example.")

    print(f"\nInspecting the first {MAX_EXAMPLES} training examples:")
    preview_split = train_split.select(range(min(MAX_EXAMPLES, len(train_split))))
    preview_audio_fields = [name for name in audio_columns if name in preview_split.column_names]
    decoded_preview = preview_split
    for audio_field in preview_audio_fields:
        decoded_preview = decoded_preview.cast_column(audio_field, Audio(decode=True))

    for index in range(len(decoded_preview)):
        example = decoded_preview[index]
        print(f"\nExample {index}")
        print("Keys:")
        for key in example.keys():
            print(f"- {key}")

        if transcript_field:
            print(f"Transcript: {example.get(transcript_field)}")

        if laryngophone_field and isinstance(example.get(laryngophone_field), dict):
            sampling_rate = example[laryngophone_field].get("sampling_rate")
            print(f"Laryngophone sampling rate: {sampling_rate}")

        print(f"Duration: {format_duration(example, laryngophone_field)}")

    return laryngophone_field


def load_vibravox_dataset() -> DatasetDict | None:
    """Load the target Vibravox config and gracefully handle gated or inaccessible access."""
    print_header("Dataset Load")
    print(f"Loading dataset: {DATASET_ID}")
    print(f"Config: {CONFIG_NAME}")
    try:
        dataset_dict = load_dataset(DATASET_ID, CONFIG_NAME)
        print("Dataset access succeeded.")
        return dataset_dict
    except HfHubHTTPError as error:
        if is_gated_access_error(error):
            print_gated_access_instructions(error)
            return None
        raise
    except Exception as error:
        if is_gated_access_error(error):
            print_gated_access_instructions(error)
            return None
        raise


def main() -> int:
    """Run Stage 1 checks and return a shell exit code."""
    print_environment_info()
    token = authenticate_hugging_face()
    if not token:
        return 0

    dataset_dict = load_vibravox_dataset()
    if dataset_dict is None:
        return 0

    print_split_sizes(dataset_dict)
    discovered_field = print_example_report(dataset_dict)

    print_header("Summary")
    if discovered_field:
        print(f"Discovered laryngophone field for Stage 2 confirmation: {discovered_field}")
    else:
        print("Laryngophone field could not be uniquely discovered. Review the printed schema before Stage 2.")
    print(f"Reference field observed from schema: {REFERENCE_FIELD_HINT}")
    print("Stage 1 completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
