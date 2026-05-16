"""
Validation and Testing Script
Tests all components of the inference system
"""

import sys
from pathlib import Path
import subprocess
import time

# Add project root
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))


class SystemValidator:
    """Validates complete inference system."""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
    
    def test_torch_and_cuda(self):
        """Test PyTorch and CUDA availability."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
            
            print(f"\n[OK] PyTorch {torch.__version__}")
            print(f"   CUDA Available: {cuda_available}")
            print(f"   Device Count: {device_count}")
            print(f"   Device: {device_name}")
            
            self.passed += 1
            self.results['torch_cuda'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"\n[ERROR] PyTorch/CUDA Test Failed: {e}")
            self.failed += 1
            self.results['torch_cuda'] = f'FAIL: {e}'
            return False
    
    def test_model_loading(self):
        """Test model loading."""
        try:
            from lettuce_ssl_segmentation_lab.config import LabConfig
            from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
            import torch
            
            config = LabConfig()
            config.resolve()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = SegmentationModelFactory.get_model(
                name="segformer",
                num_classes=9,
                input_channels=14
            )
            
            print(f"\n[OK] Model Loaded Successfully")
            print(f"   Architecture: SegFormer")
            print(f"   Classes: 9")
            print(f"   Input Channels: 14")
            print(f"   Device: {device}")
            
            self.passed += 1
            self.results['model_loading'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Model Loading Test Failed: {e}")
            self.failed += 1
            self.results['model_loading'] = f'FAIL: {e}'
            return False
    
    def test_fastapi(self):
        """Test FastAPI import."""
        try:
            import fastapi
            import uvicorn
            from pydantic import BaseModel
            
            print(f"\n[OK] FastAPI {fastapi.__version__}")
            print(f"   Uvicorn Ready")
            
            self.passed += 1
            self.results['fastapi'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"\n[ERROR] FastAPI Test Failed: {e}")
            self.failed += 1
            self.results['fastapi'] = f'FAIL: {e}'
            return False
    
    def test_redis(self):
        """Test Redis connection."""
        try:
            import redis
            
            # Try to connect to Redis (non-blocking attempt)
            try:
                r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
                r.ping()
                print(f"\n[OK] Redis Connected")
                print(f"   Host: localhost:6379")
                
            except:
                print(f"\n[WARN] Redis Not Running (Optional)")
                print(f"   Can start with: redis-server")
                print(f"   Or: docker run -d -p 6379:6379 redis:7.2-alpine")
            
            self.passed += 1
            self.results['redis'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Redis Test Failed: {e}")
            self.failed += 1
            self.results['redis'] = f'FAIL: {e}'
            return False
    
    def test_datasets(self):
        """Test dataset availability."""
        try:
            from lettuce_ssl_segmentation_lab.config import LabConfig
            
            config = LabConfig()
            config.resolve()
            
            test_dir = config.dataset_base / "test"
            
            if test_dir.exists():
                test_images = list(test_dir.glob("**/*.jpg")) + list(test_dir.glob("**/*.png"))
                print(f"\n[OK] Test Dataset Found")
                print(f"   Location: {test_dir}")
                print(f"   Images: {len(test_images)}")
                
                self.passed += 1
                self.results['datasets'] = 'PASS'
                return True
            else:
                print(f"\n[WARN] Test Dataset Not Found")
                print(f"   Expected: {test_dir}")
                self.passed += 1
                self.results['datasets'] = 'WARN'
                return True
                
        except Exception as e:
            print(f"\n[ERROR] Dataset Test Failed: {e}")
            self.failed += 1
            self.results['datasets'] = f'FAIL: {e}'
            return False
    
    def test_disk_space(self):
        """Test available disk space."""
        try:
            import shutil
            
            disk_usage = shutil.disk_usage(str(Path.cwd()))
            free_gb = disk_usage.free / (1024**3)
            
            print(f"\n[OK] Disk Space Check")
            print(f"   Free: {free_gb:.2f} GB")
            
            if free_gb < 5:
                print(f"   [WARN] Low disk space (<5GB)")
            
            self.passed += 1
            self.results['disk_space'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Disk Space Test Failed: {e}")
            self.failed += 1
            self.results['disk_space'] = f'FAIL: {e}'
            return False
    
    def test_directory_structure(self):
        """Test required directory structure."""
        try:
            from lettuce_ssl_segmentation_lab.config import LabConfig
            
            config = LabConfig()
            config.resolve()
            
            required_dirs = [
                config.lab_root,
                config.lab_root / "stage6_segmentation_training",
                config.dataset_base
            ]
            
            all_exist = all(d.exists() for d in required_dirs)
            
            print(f"\n[OK] Directory Structure")
            for d in required_dirs:
                exists = "✓" if d.exists() else "✗"
                print(f"   {exists} {d}")
            
            if all_exist:
                self.passed += 1
                self.results['directories'] = 'PASS'
                return True
            else:
                self.failed += 1
                self.results['directories'] = 'FAIL'
                return False
                
        except Exception as e:
            print(f"\n[ERROR] Directory Test Failed: {e}")
            self.failed += 1
            self.results['directories'] = f'FAIL: {e}'
            return False
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "="*80)
        print("LETTUCE DISEASE SEGMENTATION - SYSTEM VALIDATION")
        print("="*80)
        
        tests = [
            ("PyTorch & CUDA", self.test_torch_and_cuda),
            ("Model Loading", self.test_model_loading),
            ("FastAPI", self.test_fastapi),
            ("Redis", self.test_redis),
            ("Datasets", self.test_datasets),
            ("Disk Space", self.test_disk_space),
            ("Directory Structure", self.test_directory_structure),
        ]
        
        for name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"\n[ERROR] {name} Test Error: {e}")
                self.failed += 1
                self.results[name.lower()] = f'ERROR: {e}'
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for test_name, result in self.results.items():
            status_symbol = "✅" if result == "PASS" else "⚠️" if "WARN" in result else "❌"
            print(f"{status_symbol} {test_name.upper()}: {result}")
        
        print("\n" + "-"*80)
        print(f"PASSED: {self.passed}")
        print(f"FAILED: {self.failed}")
        print(f"TOTAL:  {self.passed + self.failed}")
        print("-"*80)
        
        if self.failed == 0:
            print("\n✅ ALL TESTS PASSED - System ready for inference!")
            return 0
        else:
            print(f"\n[WARN] {self.failed} tests failed - See above for details")
            return 1


def main():
    """Run validation."""
    validator = SystemValidator()
    exit_code = validator.run_all_tests()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. To start the inference system:")
    print("   python scripts/system_orchestrator.py --mode server")
    print("\n2. For complete pipeline (inference + dashboard):")
    print("   python scripts/system_orchestrator.py --mode all")
    print("\n3. To use Docker (production):")
    print("   docker-compose up -d")
    print("\n" + "="*80)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
