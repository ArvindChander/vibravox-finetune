"""Training entry point for Whisper-large LoRA fine-tuning on Vibravox throat-mic audio."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
import yaml
from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data import (
    TRAIN_AUDIO_FIELD,
    TEXT_FIELD,
    WhisperDataCollator,
    StreamingTrainDataset,
    build_validation_subset,
    get_split_sizes,
)
from model import MODEL_ID, load_model, load_processor


TEXT_NORMALIZER = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])


def parse_args() -> argparse.Namespace:
    """Parse the YAML config path from the command line."""
    parser = argparse.ArgumentParser(description="Train Whisper-large LoRA on the Vibravox throat microphone.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load experiment hyperparameters from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def print_environment_info() -> None:
    """Print environment and GPU details useful for reproducible training logs."""
    print("\n==================== Environment ====================")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Datasets version: {datasets.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU name: {properties.name}")
        print(f"GPU total VRAM (GB): {properties.total_memory / (1024**3):.2f}")
        print(f"GPU allocated VRAM (GB): {torch.cuda.memory_allocated() / (1024**3):.2f}")
        print(f"GPU reserved VRAM (GB): {torch.cuda.memory_reserved() / (1024**3):.2f}")


def print_data_summary(config: dict[str, Any], validation_dataset: Any) -> None:
    """Print split sizes and training-volume estimates without caching the full dataset."""
    split_sizes = get_split_sizes()
    effective_batch = config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"]
    estimated_samples = config["training"]["max_steps"] * effective_batch

    print("\n==================== Data Summary ====================")
    print(f"Train split size: {split_sizes['train']}")
    print(f"Validation split size: {split_sizes['validation']}")
    print(f"Test split size: {split_sizes['test']}")
    print(f"Training audio field: {TRAIN_AUDIO_FIELD}")
    print(f"Target text field: {TEXT_FIELD}")
    print(f"Validation subset used for eval: {len(validation_dataset)}")
    print(f"Effective training batch size: {effective_batch}")
    print(f"Estimated streamed training examples across all steps: {estimated_samples}")


def compute_metrics_factory(processor: Any):
    """Create the WER callback used by the Trainer during validation."""

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        prediction_ids = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]

        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.batch_decode(prediction_ids, skip_special_tokens=True)
        references = processor.batch_decode(label_ids, skip_special_tokens=True)

        normalized_predictions = [TEXT_NORMALIZER(text) for text in predictions]
        normalized_references = [TEXT_NORMALIZER(text) for text in references]
        return {"wer": wer(normalized_references, normalized_predictions)}

    return compute_metrics


class LoggingSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer subclass that prints VRAM and output-path context during logging."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if torch.cuda.is_available():
            logs = dict(logs)
            logs["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            logs["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
        super().log(logs, start_time=start_time)


def main() -> int:
    """Load config, build the streaming datasets, and run LoRA fine-tuning."""
    args = parse_args()
    config = load_config(args.config)

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print("HF_HUB_ENABLE_HF_TRANSFER=1 detected. If hf_transfer is unavailable, unset it before training.")

    print_environment_info()
    processor = load_processor(language=config["model"]["language"], task=config["model"]["task"])
    model = load_model(
        language=config["model"]["language"],
        task=config["model"]["task"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
    )

    train_dataset = StreamingTrainDataset(
        processor=processor,
        audio_field=config["data"]["audio_field"],
        text_field=config["data"]["text_field"],
        shuffle_buffer_size=config["data"]["shuffle_buffer_size"],
        seed=config["seed"],
    )
    validation_dataset = build_validation_subset(
        processor=processor,
        audio_field=config["data"]["audio_field"],
        text_field=config["data"]["text_field"],
        max_samples=config["training"]["eval_subset_size"],
    )
    print_data_summary(config, validation_dataset)

    data_collator = WhisperDataCollator(processor=processor)
    output_dir = Path(config["paths"]["output_dir"])
    logging_dir = Path(config["paths"]["logging_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    print("\n==================== Paths ====================")
    print(f"Model ID: {MODEL_ID}")
    print(f"Config path: {args.config}")
    print(f"Checkpoint output dir: {output_dir}")
    print(f"Logging dir: {logging_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        eval_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        predict_with_generate=True,
        generation_max_length=config["training"]["generation_max_length"],
        remove_unused_columns=False,
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_safetensors=True,
        seed=config["seed"],
        data_seed=config["seed"],
    )

    trainer = LoggingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(processor),
        processing_class=processor.feature_extractor,
    )

    print("\n==================== Training ====================")
    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    metrics = train_result.metrics
    print(f"Training metrics: {metrics}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    if torch.cuda.is_available():
        print(f"Final GPU allocated VRAM (GB): {torch.cuda.memory_allocated() / (1024**3):.2f}")
        print(f"Final GPU reserved VRAM (GB): {torch.cuda.memory_reserved() / (1024**3):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
