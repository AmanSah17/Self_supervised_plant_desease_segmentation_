"""
Complete Inference System Orchestrator
Manages test inference, model compilation, and system deployment
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import subprocess
import time

# Add project root
root_dir = Path(__file__).resolve().parents[0]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from lettuce_ssl_segmentation_lab.config import LabConfig


class SystemOrchestrator:
    """Orchestrates the complete inference pipeline and deployment."""
    
    def __init__(self):
        self.config = LabConfig()
        self.config.resolve()
        self.start_time = datetime.now()
        
    def run_stage9_inference(self):
        """Run test set inference (Stage 9)."""
        print("\n" + "="*80)
        print("STAGE 9: TEST SET INFERENCE")
        print("="*80 + "\n")
        
        try:
            import sys
            sys.path.insert(0, str(self.config.repo_root / "scripts"))
            from stage9_test_inference import main as stage9_main
            
            stage9_main()
            print("[SUCCESS] Stage 9 inference complete")
            return True
            
        except Exception as e:
            print(f"[ERROR] Stage 9 failed: {e}")
            return False
    
    def generate_dashboard(self):
        """Generate comprehensive dashboard reports."""
        print("\n" + "="*80)
        print("GENERATING DASHBOARD REPORTS")
        print("="*80 + "\n")
        
        try:
            import sys
            sys.path.insert(0, str(self.config.repo_root / "scripts"))
            from dashboard_report_generator import generate_dashboard
            
            inference_dir = self.config.lab_root / "stage9_test_inference"
            generate_dashboard(inference_dir)
            
            print("[SUCCESS] Dashboard generated")
            return True
            
        except Exception as e:
            print(f"[ERROR] Dashboard generation failed: {e}")
            return False
    
    def compile_tensorrt_model(self, precision: str = "fp16"):
        """Compile model to TensorRT."""
        print("\n" + "="*80)
        print(f"TENSORRT MODEL COMPILATION ({precision})")
        print("="*80 + "\n")
        
        try:
            import sys
            sys.path.insert(0, str(self.config.repo_root / "scripts"))
            from tensorrt_compiler import compile_and_save
            
            model_path = self.config.lab_root / "stage6_segmentation_training" / "supervised_finetune" / "best_model.pth"
            
            if not model_path.exists():
                print("[WARNING] best_model.pth not found, using epoch_15.pth")
                model_path = self.config.lab_root / "stage6_segmentation_training" / "epoch_15.pth"
            
            output_dir = self.config.lab_root / "compiled_models"
            
            compile_and_save(
                str(model_path),
                str(output_dir),
                precision=precision,
                benchmark=True
            )
            
            print("[SUCCESS] Model compilation complete")
            return True
            
        except Exception as e:
            print(f"[ERROR] Compilation failed: {e}")
            return False
    
    def start_backend_server(self, debug: bool = False):
        """Start FastAPI backend server."""
        print("\n" + "="*80)
        print("STARTING BACKEND SERVER")
        print("="*80 + "\n")
        
        try:
            import uvicorn
            from backend.backend_server import app
            
            print("[INFO] Backend server starting on http://0.0.0.0:8000")
            print("[INFO] API Docs: http://localhost:8000/docs")
            print("[INFO] Health: http://localhost:8000/health")
            print("[INFO] Press Ctrl+C to stop")
            
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info" if debug else "warning",
                reload=debug
            )
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Backend server failed: {e}")
            return False
    
    def generate_deployment_report(self):
        """Generate deployment and system report."""
        print("\n" + "="*80)
        print("GENERATING DEPLOYMENT REPORT")
        print("="*80 + "\n")
        
        try:
            import torch
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'system': {
                    'python_version': sys.version,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'current_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                },
                'model_paths': {
                    'best_model': str(self.config.lab_root / "stage6_segmentation_training" / "supervised_finetune" / "best_model.pth"),
                    'compiled_models': str(self.config.lab_root / "compiled_models"),
                },
                'output_paths': {
                    'inference': str(self.config.lab_root / "stage9_test_inference"),
                    'dashboard': str(self.config.lab_root / "stage9_test_inference" / "dashboard_report"),
                },
                'deployment': {
                    'backend_url': 'http://localhost:8000',
                    'frontend_url': 'http://localhost:3000 (Docker) or file:///<path>/frontend/index.html (Local)',
                    'redis_url': 'redis://localhost:6379',
                    'api_docs': 'http://localhost:8000/docs',
                }
            }
            
            report_path = self.config.lab_root / "deployment_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"[INFO] Report saved to {report_path}")
            
            print("\n📋 DEPLOYMENT CONFIGURATION:")
            print(json.dumps(report, indent=2))
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")
            return False
    
    def print_status(self, success: bool):
        """Print final status."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        if success:
            print("✅ ALL SYSTEMS OPERATIONAL")
        else:
            print("⚠️  SOME COMPONENTS FAILED - SEE ABOVE FOR DETAILS")
        print("="*80)
        print(f"Elapsed time: {elapsed:.2f} seconds")


def main():
    """Main orchestrator entry point."""
    parser = argparse.ArgumentParser(
        description="Lettuce Disease Segmentation System Orchestrator"
    )
    
    parser.add_argument(
        '--mode',
        choices=['inference', 'compile', 'server', 'all', 'dashboard'],
        default='all',
        help='Mode to run'
    )
    
    parser.add_argument(
        '--precision',
        choices=['fp32', 'fp16', 'int8'],
        default='fp16',
        help='Model compilation precision'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    orchestrator = SystemOrchestrator()
    
    print("\n" + "="*80)
    print("LETTUCE DISEASE SEGMENTATION - SYSTEM ORCHESTRATOR")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Precision: {args.precision}")
    print(f"Debug: {args.debug}")
    print("="*80)
    
    success = True
    
    if args.mode in ['inference', 'all']:
        success = success and orchestrator.run_stage9_inference()
    
    if args.mode in ['dashboard', 'all']:
        success = success and orchestrator.generate_dashboard()
    
    if args.mode in ['compile', 'all']:
        success = success and orchestrator.compile_tensorrt_model(args.precision)
    
    if args.mode == 'server':
        success = success and orchestrator.start_backend_server(args.debug)
    
    if args.mode in ['all']:
        success = success and orchestrator.generate_deployment_report()
    
    orchestrator.print_status(success)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
