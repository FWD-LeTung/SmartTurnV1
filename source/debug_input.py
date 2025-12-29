import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Union
from datasets import load_from_disk, load_dataset, concatenate_datasets
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import onnx
import numpy as np
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from torch.ao.quantization import disable_fake_quant, disable_observer, FakeQuantize, FusedMovingAvgObsFakeQuantize

from transformers.trainer import Trainer
from transformers import WhisperPreTrainedModel, WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from logger import log, log_model_structure, log_dataset_statistics, log_dependencies, ProgressLoggerCallback


CONFIG = {
    "model_name": "openai/whisper-tiny",

    "datasets_training": ["PandaLT/vie_train"],
    "datasets_test": ["PandaLT/vie_test"],

    "learning_rate": 0.00002248,
    "num_epochs": 1, # overfitting starts to occur beyond 4 epochs
    "train_batch_size": 64,
    "eval_batch_size": 32,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    "eval_steps": 50,
    "save_steps": 50,
    "logging_steps": 10,

    "qat_start_epoch": 1,
    "quantization_backend": "qnnpack",

    "onnx_opset_version": 19,
    "calibration_dataset_size": 256,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def load_dataset_at(path: str):
    if path.startswith("/"):
        return load_from_disk(path)['train']
    else:
        return load_dataset(path)['train']
    
def truncate_audio_to_last_n_seconds(audio_array, n_seconds, sample_rate=16000):
    max_samples = n_seconds * sample_rate
    if (len(audio_array) > max_samples):
        return audio_array[-max_samples:]
    return audio_array

class OnDemandWhisperDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio_array = sample['train']['array']

        # Truncate audio to last 8 seconds
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        # Extract label
        label = 1 if sample["endpoint_bool"] else 0

        # Extract features
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "language": sample.get("language", "eng"),
            "midfiller": sample.get("midfiller", None),
            "endfiller": sample.get("endfiller", None),
        }

def prepare_datasets_ondemand(feature_extractor, config):
    log.info("Preparing datasets...")

    datasets_training = config["datasets_training"]
    datasets_test = config["datasets_test"]

    training_splits = []
    eval_splits = []
    test_splits = {}

    for dataset_path in datasets_training:
        dataset_name = dataset_path.split("/")[-1]
        log.info(f"Loading dataset '{dataset_name}'...")
        full_dataset = load_dataset_at(dataset_path)

        log.info("  |-> Splitting dataset into train/eval splits...")
        dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
        training_splits.append(dataset_dict["train"])
        eval_splits.append(dataset_dict["test"])
    
    print(dataset_dict['train'])
    print(training_splits[0])
    log.info("Merging datasets...")

    merged_training_dataset = concatenate_datasets(training_splits).shuffle(seed=42)
    merged_eval_dataset = concatenate_datasets(eval_splits)
    
    print(merged_training_dataset)
    log.info("Loading test datasets...")

    for dataset_path in datasets_test:
        dataset_name = dataset_path.split("/")[-1]
        test_dataset = load_dataset_at(dataset_path)
        test_splits[dataset_name] = test_dataset
    
    log.info("Wrapping datasets with OnDemandWhisperDataset...")
    wrapped_training = OnDemandWhisperDataset(merged_training_dataset, feature_extractor)
    wrapped_eval = OnDemandWhisperDataset(merged_eval_dataset, feature_extractor)
    wrapped_test_splits = {
        name: OnDemandWhisperDataset(dataset, feature_extractor)
        for name, dataset in test_splits.items()
    }
    
    print(wrapped_training['train'])
    print(wrapped_eval['train'])
    print(wrapped_test_splits['train'])
    return {
        "training": wrapped_training,
        "eval": wrapped_eval,
        "test": wrapped_test_splits,
        "raw_datasets": {
            "training": merged_training_dataset,
            "eval": merged_eval_dataset,
            "test": test_splits
        }
    }

def save_dataset_ids(datasets, output_dir):
    ids_dict = {}

    if 'id' in datasets["raw_datasets"]["training"].column_names:
        train_ids = [id for id in datasets["raw_datasets"]["training"]["id"] if id is not None]
        ids_dict["train"] = sorted(train_ids)

    if 'id' in datasets["raw_datasets"]["eval"].column_names:
        eval_ids = [id for id in datasets["raw_datasets"]["eval"]["id"] if id is not None]
        ids_dict["eval"] = sorted(eval_ids)

    ids_dict["test"] = {}
    for name, dataset in datasets["raw_datasets"]["test"].items():
        if 'id' in dataset.column_names:
            test_ids = [id for id in dataset["id"] if id is not None]
            ids_dict["test"][name] = sorted(test_ids)

    ids_path = os.path.join(output_dir, "dataset_ids.json")
    with open(ids_path, 'w') as f:
        json.dump(ids_dict, f, indent=2)

    log.info(f"Saved dataset IDs to {ids_path}")
    return ids_path

load_dataset_at("PandaLT/vie_train")
prepare_datasets_ondemand(WhisperFeatureExtractor, CONFIG)