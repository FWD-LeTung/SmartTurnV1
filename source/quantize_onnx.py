"""
Script để quantize ONNX model từ FP32 sang INT8
Sử dụng static quantization với calibration dataset

Usage:
    python quantize_onnx.py --input model_fp32.onnx --output model_int8.onnx --calibration_data /path/to/dataset
"""

import argparse
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, 
    CalibrationDataReader, 
    QuantType, 
    quant_pre_process,
    QuantFormat, 
    CalibrationMethod
)
from datasets import load_from_disk, load_dataset
from transformers import WhisperFeatureExtractor


class ONNXCalibrationDataReader(CalibrationDataReader):
    """Data reader cho ONNX quantization calibration"""
    
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


class CalibrationDataset:
    """Dataset cho calibration với stratified sampling"""
    
    def __init__(self, dataset, feature_extractor, max_samples=256):
        self.feature_extractor = feature_extractor
        
        # Lấy indices cân bằng từ 2 classes
        positive_indices = [i for i, sample in enumerate(dataset) if sample.get("endpoint_bool", False)]
        negative_indices = [i for i, sample in enumerate(dataset) if not sample.get("endpoint_bool", False)]
        
        # Sample một nửa từ mỗi class
        import random
        random.seed(42)
        pos_sample_size = min(max_samples // 2, len(positive_indices))
        neg_sample_size = min(max_samples - pos_sample_size, len(negative_indices))
        
        selected_indices = (
            random.sample(positive_indices, pos_sample_size) +
            random.sample(negative_indices, neg_sample_size)
        )
        self.indices = selected_indices[:max_samples]
        self.dataset = dataset
        
        print(f"Calibration dataset: {pos_sample_size} positive + {neg_sample_size} negative = {len(self.indices)} total samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]
        
        audio_array = sample["audio"]["array"]
        # Truncate audio to last 8 seconds
        audio_array = self._truncate_audio(audio_array, n_seconds=8)
        
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
    
    @staticmethod
    def _truncate_audio(audio_array, n_seconds=8, sample_rate=16000):
        """Truncate audio to last n seconds"""
        max_samples = n_seconds * sample_rate
        if len(audio_array) > max_samples:
            return audio_array[-max_samples:]
        return audio_array


def load_calibration_dataset(dataset_path, max_samples=256):
    """Load dataset cho calibration"""
    print(f"Loading calibration dataset from: {dataset_path}")
    
    # Load dataset
    if dataset_path.startswith('/') or os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
        if "train" in dataset:
            dataset = dataset["train"]
    else:
        dataset = load_dataset(dataset_path)["train"]
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Khởi tạo feature extractor
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    
    # Tạo calibration dataset
    calibration_dataset = CalibrationDataset(dataset, feature_extractor, max_samples)
    
    return calibration_dataset


def quantize_onnx_model(input_path, output_path, calibration_dataset):
    """
    Quantize ONNX model từ FP32 sang INT8 sử dụng static quantization
    
    Args:
        input_path: Đường dẫn đến model FP32 (.onnx)
        output_path: Đường dẫn để lưu model INT8 (.onnx)
        calibration_dataset: Dataset để calibration
    """
    try:
        print(f"Quantizing ONNX model: {input_path} -> {output_path}")
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Bước 1: Pre-process model (optimization + shape inference)
        print("Step 1: Pre-processing model...")
        pre_path = output_path.replace(".onnx", "_pre.onnx")
        quant_pre_process(
            input_path,
            pre_path,
            skip_optimization=False,  # Optimize model
            disable_shape_inference=False  # Infer shapes
        )
        print(f"Pre-processed model saved to: {pre_path}")
        
        # Bước 2: Quantize model với calibration
        print("Step 2: Quantizing model with calibration...")
        quantize_static(
            model_input=pre_path,
            model_output=output_path,
            calibration_data_reader=ONNXCalibrationDataReader(calibration_dataset),
            quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format
            activation_type=QuantType.QUInt8,  # 8-bit unsigned int cho activations
            weight_type=QuantType.QInt8,  # 8-bit signed int cho weights
            per_channel=True,  # Per-channel quantization cho weights
            calibrate_method=CalibrationMethod.MinMax,  # MinMax calibration
            op_types_to_quantize=["Conv", "MatMul", "Gemm"]  # Chỉ quantize các ops này
        )
        
        print(f"Quantized model saved to: {output_path}")
        
        # Bước 3: Verify model
        print("Step 3: Verifying quantized model...")
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)
        print("Model verification passed!")
        
        # Bước 4: Test inference
        print("Step 4: Testing inference...")
        session = ort.InferenceSession(output_path)
        test_input = calibration_dataset[0].reshape(1, 80, 800).astype(np.float32)
        outputs = session.run(None, {"input_features": test_input})
        print(f"Inference test passed! Output shape: {outputs[0].shape}")
        
        # So sánh kích thước file
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - output_size / input_size) * 100
        
        print(f"\n{'='*60}")
        print(f"Quantization completed successfully!")
        print(f"{'='*60}")
        print(f"Input model size:  {input_size:.2f} MB")
        print(f"Output model size: {output_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}%")
        print(f"{'='*60}\n")
        
        # Dọn dẹp file tạm
        if os.path.exists(pre_path):
            os.remove(pre_path)
            print(f"Cleaned up temporary file: {pre_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model from FP32 to INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize với dataset từ disk
  python quantize_onnx.py --input model_fp32.onnx --output model_int8.onnx --calibration_data /path/to/dataset
  
  # Quantize với dataset từ HuggingFace Hub
  python quantize_onnx.py --input model_fp32.onnx --output model_int8.onnx --calibration_data username/dataset-name
  
  # Chỉ định số lượng samples cho calibration
  python quantize_onnx.py --input model_fp32.onnx --output model_int8.onnx --calibration_data /path/to/dataset --max_samples 512
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn đến model FP32 ONNX (.onnx)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Đường dẫn để lưu model INT8 ONNX (.onnx)"
    )
    
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="Đường dẫn đến dataset cho calibration (local path hoặc HuggingFace dataset name)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=256,
        help="Số lượng samples tối đa cho calibration (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra input file
    if not os.path.exists(args.input):
        print(f"Error: Input file không tồn tại: {args.input}")
        return
    
    # Load calibration dataset
    calibration_dataset = load_calibration_dataset(args.calibration_data, args.max_samples)
    
    # Quantize model
    result = quantize_onnx_model(args.input, args.output, calibration_dataset)
    
    if result:
        print(f"\n Quantization hoàn tất thành công!")
        print(f"  Model INT8 đã được lưu tại: {result}")
    else:
        print(f"\n Quantization thất bại!")


if __name__ == "__main__":
    main()