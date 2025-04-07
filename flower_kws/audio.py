"""flower-kws: A Flower / PyTorch app."""

import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset
from torchaudio.transforms import MFCC, Resample

TARGET_LENGTH = 16000


def get_apply_transforms_fn():
    ss = 8000  # 8KHz
    n_mfcc = 40
    window_width = 40e-3  # length of window in seconds
    stride = 20e-3  # stride between windows
    n_fft = 400
    audio_transforms = torch.nn.Sequential(
        Resample(TARGET_LENGTH, ss),
        MFCC(
            sample_rate=ss,
            n_mfcc=n_mfcc,
            melkwargs={
                "win_length": int(ss * window_width),
                "hop_length": int(ss * stride),
                "n_fft": n_fft,
            },
        ),
    )

    def apply_transforms(batch):
        audio = batch["audio"]
        data = {}
        # Ensure all audio samples are of the same length
        array = np.array(audio["array"])
        length = array.shape[0]
        if length < TARGET_LENGTH:
            padded = np.pad(array, (0, TARGET_LENGTH - length), mode="constant")
        else:
            padded = array[:TARGET_LENGTH]

        # Compute MFCC
        x = torch.from_numpy(padded).float()
        data["mfcc"] = torch.unsqueeze(audio_transforms(x), 0)
        # Set targets
        # All unknown keywords are assigned label 11. The silence clips get assigned label 10
        # In this way we have 12 classes with labels 0-11
        data["target"] = (
            11
            if batch["is_unknown"]
            else (10 if batch["label"] == 35 else batch["label"])
        )
        return data

    return apply_transforms


def segment_silences(num_sements):
    def segment_audio(example):
        np.random.seed(32 + hash(example["file"]) % 1000000)  # unique per file
        audio = example["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]

        max_offset = len(array) - TARGET_LENGTH
        if max_offset <= 0:
            return []  # too short to segment

        offsets = np.linspace(0, max_offset, num_sements, dtype=int)
        segments = []
        for offset in offsets:
            segment = array[offset : offset + TARGET_LENGTH]
            segments.append(
                {
                    "audio": {
                        "array": segment.copy(),
                        "sampling_rate": sr,
                    },
                }
            )

        return Dataset.from_list(segments).to_dict()

    return segment_audio


def flatten_all_segments(batch):
    flat_batch = {key: [] for key in batch if key != "audio"}
    flat_batch["audio"] = []

    for i in range(len(batch["audio"])):
        audio_list = batch["audio"][i]
        for audio in audio_list:
            flat_batch["audio"].append(audio)
            for key in flat_batch:
                if key != "audio":
                    flat_batch[key].append(batch[key][i])

    return flat_batch


def balance_dataset(dataset, target_column="target", seed=42):
    # Get class counts
    labels = [int(t) for t in dataset[target_column]]
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    max_count = max(len(idxs) for idxs in label_to_indices.values())

    all_selected_indices = []
    rng = random.Random(seed)

    for label, indices in label_to_indices.items():
        # Resample with replacement
        needed = max_count - len(indices)
        resampled = rng.choices(indices, k=needed) if needed > 0 else []
        all_selected_indices.extend(indices + resampled)

    return dataset.select(all_selected_indices).shuffle(seed=seed)
