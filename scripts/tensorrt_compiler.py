"""
TensorRT Model Compiler and Optimizer
Optimizes and compiles PyTorch SegFormer models for production GPU inference.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import numpy as np


class TensorRTCompiler:
    """
    Compiles PyTorch models to TensorRT optimized format for GPU acceleration.
    Supports FP32, FP16, and INT8 precision modes.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cuda:0",
                 precision: str = "fp16",
                 batch_size: int = 1):
        """
        Initialize TensorRT compiler.
        
        Args:
            model: PyTorch model to compile
            device: GPU device (e.g., "cuda:0")
            precision: Optimization level - 'fp32', 'fp16', or 'int8'
            batch_size: Batch size for optimization
        """
        self.model = model
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        
        # Validate precision mode
        if precision not in ['fp32', 'fp16', 'int8']:
            raise ValueError(f"Unsupported precision: {precision}. Choose from 'fp32', 'fp16', 'int8'")
    
    def compile(self, 
                input_shape: Tuple[int, ...],
                output_dir: Optional[Path] = None) -> Dict:
        """
        Compile model with TensorRT optimizations.
        
        Args:
            input_shape: Expected input shape (e.g., (1, 14, 256, 256))
            output_dir: Directory to save compiled model
            
        Returns:
            Dictionary with compilation info
        """
        try:
            import tensorrt as trt
            from torch2trt import torch2trt, TRTModule
        except ImportError:
            print("[WARNING] torch2trt not available. Falling back to TorchScript compilation.")
            return self._fallback_torchscript_compile(input_shape, output_dir)
        
        print(f"[INFO] Compiling model to TensorRT ({self.precision} precision)")
        
        self.model.eval()
        
        # Prepare input tensor
        dummy_input = torch.randn(input_shape, dtype=torch.float32).to(self.device)
        
        # Compilation options
        fp16_mode = (self.precision == 'fp16')
        int8_mode = (self.precision == 'int8')
        
        print(f"[INFO] Compiling with FP16={fp16_mode}, INT8={int8_mode}")
        
        try:
            # Convert to TensorRT
            trt_model = torch2trt(
                self.model,
                [dummy_input],
                fp16_mode=fp16_mode,
                int8_mode=int8_mode,
                log_level=trt.Logger.INFO
            )
            
            print("[INFO] TensorRT compilation successful!")
            
            # Save compiled model
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = output_dir / f"model_trt_{self.precision}.pth"
                torch.save(trt_model.state_dict(), model_path)
                print(f"[INFO] Compiled model saved to {model_path}")
                
                return {
                    'status': 'success',
                    'precision': self.precision,
                    'model_path': str(model_path),
                    'input_shape': input_shape,
                    'type': 'tensorrt'
                }
            else:
                return {
                    'status': 'success',
                    'precision': self.precision,
                    'model': trt_model,
                    'input_shape': input_shape,
                    'type': 'tensorrt'
                }
        
        except Exception as e:
            print(f"[WARNING] TensorRT compilation failed: {e}")
            print("[INFO] Falling back to TorchScript...")
            return self._fallback_torchscript_compile(input_shape, output_dir)
    
    def _fallback_torchscript_compile(self, 
                                     input_shape: Tuple[int, ...],
                                     output_dir: Optional[Path] = None) -> Dict:
        """Fallback TorchScript compilation when TensorRT unavailable."""
        print("[INFO] Compiling model to TorchScript...")
        
        self.model.eval()
        
        # Prepare dummy input
        dummy_input = torch.randn(input_shape, dtype=torch.float32).to(self.device)
        
        try:
            # Convert to TorchScript
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Optional: optimize
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = output_dir / f"model_torchscript_{self.precision}.pt"
                traced_model.save(str(model_path))
                print(f"[INFO] TorchScript model saved to {model_path}")
                
                return {
                    'status': 'success',
                    'precision': self.precision,
                    'model_path': str(model_path),
                    'input_shape': input_shape,
                    'type': 'torchscript'
                }
            else:
                return {
                    'status': 'success',
                    'precision': self.precision,
                    'model': traced_model,
                    'input_shape': input_shape,
                    'type': 'torchscript'
                }
        
        except Exception as e:
            print(f"[ERROR] TorchScript compilation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def benchmark(self, 
                  input_shape: Tuple[int, ...],
                  num_iterations: int = 100) -> Dict:
        """
        Benchmark model inference performance.
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of inference iterations
            
        Returns:
            Performance metrics
        """
        print(f"[INFO] Benchmarking model inference ({num_iterations} iterations)")
        
        self.model.eval()
        dummy_input = torch.randn(input_shape, dtype=torch.float32).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start.record()
                output = self.model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        times = np.array(times)
        
        return {
            'mean_latency_ms': float(np.mean(times)),
            'std_latency_ms': float(np.std(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'throughput_fps': float(1000 / np.mean(times)),
            'iterations': num_iterations,
            'input_shape': input_shape,
            'precision': self.precision
        }


class ModelOptimizer:
    """Optimizes models for production deployment."""
    
    @staticmethod
    def quantize_model(model: nn.Module, 
                      backend: str = 'qnnpack') -> nn.Module:
        """
        Quantize model for faster inference on CPU/GPU.
        
        Args:
            model: PyTorch model
            backend: Quantization backend ('qnnpack', 'fbgemm')
            
        Returns:
            Quantized model
        """
        print(f"[INFO] Quantizing model with {backend} backend")
        
        model.eval()
        torch.quantization.set_qconfig_training(model, 'fbgemm')
        
        # Prepare model for quantization
        torch.quantization.prepare(model, inplace=True)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        print("[INFO] Quantization complete")
        return model
    
    @staticmethod
    def prune_model(model: nn.Module, 
                   pruning_ratio: float = 0.1) -> nn.Module:
        """
        Apply structured pruning to reduce model size.
        
        Args:
            model: PyTorch model
            pruning_ratio: Fraction of channels to prune (0.1 = 10% pruning)
            
        Returns:
            Pruned model
        """
        print(f"[INFO] Applying structured pruning (ratio={pruning_ratio})")
        
        import torch.nn.utils.prune as prune
        
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
        
        print("[INFO] Pruning complete")
        return model
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Size information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate size (assuming float32 = 4 bytes per param)
        model_size_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(model_size_mb)
        }


def compile_and_save(model_path: str,
                     output_dir: str,
                     num_classes: int = 9,
                     input_channels: int = 14,
                     precision: str = 'fp16',
                     benchmark: bool = True):
    """
    Complete compilation pipeline.
    
    Args:
        model_path: Path to saved PyTorch model
        output_dir: Output directory for compiled models
        num_classes: Number of segmentation classes
        input_channels: Number of input channels
        precision: Compilation precision ('fp32', 'fp16', 'int8')
        benchmark: Whether to benchmark after compilation
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    
    from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
    
    print(f"[INFO] Loading model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SegmentationModelFactory.get_model(
        name="segformer",
        num_classes=num_classes,
        input_channels=input_channels
    )
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Get model size
    size_info = ModelOptimizer.get_model_size(model)
    print(f"[INFO] Model size: {size_info['model_size_mb']:.2f} MB")
    
    # Compile
    compiler = TensorRTCompiler(
        model=model,
        device=str(device),
        precision=precision,
        batch_size=1
    )
    
    input_shape = (1, input_channels, 256, 256)
    compile_result = compiler.compile(input_shape, output_dir)
    
    print(f"[INFO] Compilation result: {compile_result}")
    
    # Benchmark
    if benchmark:
        print("\n[INFO] Running performance benchmark...")
        benchmark_result = compiler.benchmark(input_shape, num_iterations=50)
        print(f"  Mean latency: {benchmark_result['mean_latency_ms']:.2f} ms")
        print(f"  Throughput: {benchmark_result['throughput_fps']:.2f} FPS")
    
    print("\n[SUCCESS] Model compilation complete!")
    return compile_result


if __name__ == "__main__":
    # Example usage
    from lettuce_ssl_segmentation_lab.config import LabConfig
    
    config = LabConfig()
    config.resolve()
    
    model_path = config.lab_root / "stage6_segmentation_training" / "supervised_finetune" / "best_model.pth"
    output_dir = config.lab_root / "compiled_models"
    
    compile_and_save(
        str(model_path),
        str(output_dir),
        precision='fp16',
        benchmark=True
    )
