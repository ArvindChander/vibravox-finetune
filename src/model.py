"""Model construction utilities for Whisper-large LoRA fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_ID = "openai/whisper-large"


def load_processor(language: str = "fr", task: str = "transcribe") -> WhisperProcessor:
    """Load the Whisper processor configured for French transcription."""
    return WhisperProcessor.from_pretrained(MODEL_ID, language=language, task=task)


def build_lora_config() -> LoraConfig:
    """Create the locked LoRA configuration for Whisper-large fine-tuning."""
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )


def _format_parameter_count(value: int) -> str:
    """Format a parameter count with commas for readable logging."""
    return f"{value:,}"


def print_parameter_summary(model: Any) -> None:
    """Print trainable and total parameter counts for the LoRA-wrapped model."""
    trainable_params = 0
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    percentage = (trainable_params / total_params) * 100 if total_params else 0.0
    print(
        "Trainable parameters: "
        f"{_format_parameter_count(trainable_params)} / {_format_parameter_count(total_params)} "
        f"({percentage:.4f}%)"
    )


def load_model(language: str = "fr", task: str = "transcribe") -> WhisperForConditionalGeneration:
    """Load Whisper-large and attach the locked LoRA adapters for training."""
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model = get_peft_model(model, build_lora_config())
    print_parameter_summary(model)
    return model
