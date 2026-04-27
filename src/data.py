"""Dataset loading and preprocessing utilities for Vibravox Whisper fine-tuning."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from datasets import Audio, IterableDataset as HfIterableDataset, load_dataset, load_dataset_builder
from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase
from torch.utils.data import Dataset, IterableDataset


DATASET_ID = "Cnam-LMSSC/vibravox"
CONFIG_NAME = "speech_clean"
TRAIN_AUDIO_FIELD = "audio.throat_microphone"
TEXT_FIELD = "normalized_text"
DURATION_FIELD = "duration"
ALL_COLUMNS = [
    "audio.headset_microphone",
    "audio.forehead_accelerometer",
    "audio.soft_in_ear_microphone",
    "audio.rigid_in_ear_microphone",
    "audio.temple_vibration_pickup",
    "audio.throat_microphone",
    "gender",
    "speaker_id",
    "sentence_id",
    "duration",
    "raw_text",
    "normalized_text",
    "phonemized_text",
]
TEXT_NORMALIZER = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])


def normalize_transcript_text(text: str) -> str:
    """Normalize transcript text for training targets and WER computation."""
    return TEXT_NORMALIZER(text)


def get_split_sizes() -> dict[str, int]:
    """Return the built-in train, validation, and test split sizes from dataset metadata."""
    builder = load_dataset_builder(DATASET_ID, CONFIG_NAME)
    return {split_name: split_info.num_examples for split_name, split_info in builder.info.splits.items()}


def load_streaming_split(split: str, audio_field: str = TRAIN_AUDIO_FIELD) -> HfIterableDataset:
    """Load one split in streaming mode and keep only the columns needed for the current task."""
    dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=split, streaming=True)
    columns_to_remove = [column for column in ALL_COLUMNS if column not in {audio_field, TEXT_FIELD, DURATION_FIELD}]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.cast_column(audio_field, Audio(sampling_rate=16_000, decode=True))
    return dataset


def preprocess_example(
    example: dict[str, Any],
    processor: Any,
    audio_field: str = TRAIN_AUDIO_FIELD,
    text_field: str = TEXT_FIELD,
) -> dict[str, Any]:
    """Convert one raw dataset example into Whisper input features and tokenized labels."""
    normalized_text = normalize_transcript_text(example[text_field])
    audio = example[audio_field]
    audio_inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30 * 16_000,
    )
    label_ids = processor.tokenizer(
        normalized_text,
        return_attention_mask=False,
        return_tensors="pt",
    ).input_ids[0]
    return {
        "input_features": audio_inputs.input_features[0],
        "labels": label_ids,
        "text": normalized_text,
        "duration": float(example.get(DURATION_FIELD, 0.0)),
    }


class StreamingTrainDataset(IterableDataset):
    """Endless iterable training dataset that streams only the throat-mic split on demand."""

    def __init__(
        self,
        processor: Any,
        audio_field: str = TRAIN_AUDIO_FIELD,
        text_field: str = TEXT_FIELD,
        shuffle_buffer_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.processor = processor
        self.audio_field = audio_field
        self.text_field = text_field
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __iter__(self):
        epoch = 0
        while True:
            dataset = load_streaming_split("train", audio_field=self.audio_field)
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed + epoch)
            for example in dataset:
                yield preprocess_example(
                    example,
                    processor=self.processor,
                    audio_field=self.audio_field,
                    text_field=self.text_field,
                )
            epoch += 1


class ValidationDataset(Dataset):
    """Materialized validation subset used for periodic Trainer evaluation."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def build_validation_subset(
    processor: Any,
    audio_field: str = TRAIN_AUDIO_FIELD,
    text_field: str = TEXT_FIELD,
    max_samples: int = 200,
) -> ValidationDataset:
    """Stream the validation split and materialize only a bounded number of preprocessed examples."""
    dataset = load_streaming_split("validation", audio_field=audio_field)
    items: list[dict[str, Any]] = []
    for index, example in enumerate(dataset):
        if index >= max_samples:
            break
        items.append(
            preprocess_example(
                example,
                processor=processor,
                audio_field=audio_field,
                text_field=text_field,
            )
        )
    return ValidationDataset(items)


@dataclass
class WhisperDataCollator:
    """Pad variable-length Whisper features and token labels to the longest item in the batch."""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if labels.size(1) > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
