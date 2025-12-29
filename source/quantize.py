import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import modal
import numpy as np
import onnx
import onnxruntime as ort
import seaborn as sns
import torch
import wandb
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, quant_pre_process, \
    QuantFormat, CalibrationMethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.ao.quantization import disable_fake_quant, disable_observer, FakeQuantize, FusedMovingAvgObsFakeQuantize
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, WhisperPreTrainedModel, WhisperConfig
# noinspection PyProtectedMember
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from datasets import load_dataset, concatenate_datasets, load_from_disk
from logger import log, log_model_structure, log_dataset_statistics, log_dependencies, ProgressLoggerCallback


CONFIG = {
    "model_name": "openai/whisper-tiny",

    "datasets_training": ["PandaLT/vie_train"],
    "datasets_test": ["PandaLT/vie_test4"],

    "learning_rate": 5e-5,
    "num_epochs": 4, # overfitting starts to occur beyond 4 epochs
    "train_batch_size": 384,
    "eval_batch_size": 128,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,

    "qat_start_epoch": 1,
    "quantization_backend": "fbgemm",

    "onnx_opset_version": 20,
    "calibration_dataset_size": 256,
}



def load_dataset_at(path: str):
    if path.startswith('/'):
        return load_from_disk(path)["train"]
    else:
        return load_dataset(path)["train"]
    
def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array

class CalibrationDataset:
    """Calibration dataset for ONNX quantization with stratified sampling"""

    def __init__(self, dataset, feature_extractor, max_samples=500):
        self.feature_extractor = feature_extractor

        if hasattr(dataset, 'dataset'):
            # Ensure balanced sampling of endpoint classes
            underlying = dataset.dataset
            positive_indices = [i for i, sample in enumerate(underlying) if sample["endpoint_bool"]]
            negative_indices = [i for i, sample in enumerate(underlying) if not sample["endpoint_bool"]]

            # Sample half from each class
            import random
            random.seed(42)
            pos_sample_size = min(max_samples // 2, len(positive_indices))
            neg_sample_size = min(max_samples - pos_sample_size, len(negative_indices))

            selected_indices = (random.sample(positive_indices, pos_sample_size) +
                                random.sample(negative_indices, neg_sample_size))
            self.indices = selected_indices[:max_samples]
            self.dataset = underlying

            log.info(
                f"Calibration dataset: {pos_sample_size} positive + {neg_sample_size} negative = {len(self.indices)} total samples")
        else:
            raise ValueError("Invalid dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]

        audio_array = sample["audio"]["array"]
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )
        return inputs.input_features.squeeze(0).numpy()


class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset):
        self.calibration_dataset = calibration_dataset
        self.iterator = iter(range(len(calibration_dataset)))

    def get_next(self):
        try:
            idx = next(self.iterator)
            input_data = self.calibration_dataset[idx]  # shape (80, 800)
            input_data = np.expand_dims(input_data, axis=0)  # shape (1, 80, 800)
            input_data = input_data.astype(np.float32, copy=False)
            return {"input_features": input_data}
        except StopIteration:
            return None

def quantize_onnx_model(onnx_fp32_path, calibration_dataset, output_path):
    """Quantize ONNX model using static quantization"""
    try:
        log.info("Quantizing ONNX model to INT8...")

        pre_path = output_path.replace(".onnx", "_pre.onnx")
        quant_pre_process(
            onnx_fp32_path,
            pre_path,
            skip_optimization=False,  # let it fold/clean
            disable_shape_inference=False
        )
        # Calibrate + quantize
        quantize_static(
            model_input=pre_path,
            model_output=output_path,
            calibration_data_reader=ONNXCalibrationDataReader(calibration_dataset),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"]
        )

        log.info(f"Quantized ONNX model saved to {output_path}")

        # Verify the quantized model
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)

        return output_path

    except Exception as e:
        log.error(f"Failed to quantize ONNX model: {e}")
        return None
    
class OnDemandWhisperDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio_array = sample["audio"]["array"]

        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        label = 1 if sample["endpoint_bool"] else 0

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


@dataclass
class WhisperDataCollator:
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, str, None]]]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        batch = {
            "input_features": input_features,
            "labels": labels,
        }

        if "language" in features[0]:
            batch["language"] = [f["language"] for f in features]
        if "midfiller" in features[0]:
            batch["midfiller"] = [f["midfiller"] for f in features]
        if "endfiller" in features[0]:
            batch["endfiller"] = [f["endfiller"] for f in features]

        return batch

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

    log.info("Merging datasets...")

    merged_training_dataset = concatenate_datasets(training_splits).shuffle(seed=42)
    merged_eval_dataset = concatenate_datasets(eval_splits)

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


base_model = "C:/Users/letung373/Desktop/Lumi/smart-turn-local-training/model_fp32.onnx"
output_path = 'C:?Users/letung373/Desktop/Lumi/smart-turn-local-training/output'
onnx_int8_path = os.path.join(output_path, 'model_int8.onxx')
load_dataset_at(CONFIG["datasets_test"])

calibration_dataset = CalibrationDataset(
            datasets["training"],
            feature_extractor,
            max_samples=CONFIG["calibration_dataset_size"],
        )

quantize_onnx_model(base_model, calibration_dataset, onnx_int8_path)
