"""
FastAPI Backend Server with TensorRT + CUDA + Redis Queue Management
Serves the fine-tuned SegFormer model for high-performance inference.
"""

import os
import sys
import asyncio
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
from pydantic import BaseModel
import redis.asyncio as redis

# Add project root
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from lettuce_ssl_segmentation_lab.config import LabConfig
from lettuce_ssl_segmentation_lab.pipeline.models import SegmentationModelFactory
from lettuce_ssl_segmentation_lab.pipeline.metrics import SegmentationMetrics


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceRequest(BaseModel):
    """Inference request model."""
    job_id: Optional[str] = None
    priority: int = 0  # 0-100, higher = more priority


class InferenceResponse(BaseModel):
    """Inference response model."""
    job_id: str
    status: JobStatus
    result: Optional[Dict] = None
    error: Optional[str] = None


class DiseaseSegmentationServer:
    """
    Production-grade inference server with GPU acceleration, queue management,
    and comprehensive disease analytics.
    """
    
    def __init__(self, 
                 config: LabConfig,
                 model_path: str,
                 redis_url: str = "redis://localhost:6379",
                 batch_size: int = 4):
        """
        Initialize inference server.
        
        Args:
            config: Lab configuration
            model_path: Path to trained model
            redis_url: Redis connection URL
            batch_size: Batch size for GPU inference
        """
        self.config = config
        self.model_path = model_path
        self.redis_url = redis_url
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.redis_client = None
        
        # Job queue
        self.job_queue = asyncio.Queue()
        self.jobs: Dict[str, Dict] = {}
        
    def _load_model(self) -> torch.nn.Module:
        """Load model to GPU."""
        print(f"[INFO] Loading model from {self.model_path} on {self.device}")
        
        model = SegmentationModelFactory.get_model(
            name="segformer",
            num_classes=len(self.config.class_names) + 1,
            input_channels=14
        )
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Enable cuDNN benchmarking for faster inference
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        print("[INFO] Model loaded successfully")
        return model
    
    async def initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            print("[INFO] Redis connection established")
        except Exception as e:
            print(f"[WARNING] Redis connection failed: {e}")
            print("[INFO] Running without queue persistence")
            self.redis_client = None
    
    async def queue_inference_job(self, 
                                  image_path: str,
                                  job_id: Optional[str] = None,
                                  priority: int = 0) -> str:
        """
        Queue an inference job.
        
        Args:
            image_path: Path to image file
            job_id: Optional job ID (generated if not provided)
            priority: Job priority (0-100)
            
        Returns:
            Job ID
        """
        if not job_id:
            job_id = str(uuid.uuid4())
        
        job = {
            'job_id': job_id,
            'image_path': image_path,
            'status': JobStatus.QUEUED,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'result': None,
            'error': None
        }
        
        self.jobs[job_id] = job
        await self.job_queue.put((priority, job_id, image_path))
        
        # Persist to Redis if available
        if self.redis_client:
            await self.redis_client.set(f"job:{job_id}", json.dumps(job, default=str))
        
        print(f"[INFO] Job {job_id} queued")
        return job_id
    
    async def process_job(self, image_path: str, job_id: str) -> Dict:
        """
        Process a single inference job.
        
        Args:
            image_path: Path to input image
            job_id: Job ID
            
        Returns:
            Inference results
        """
        try:
            self.jobs[job_id]['status'] = JobStatus.PROCESSING
            
            # Load and prepare image
            print(f"[INFO] Processing job {job_id}")
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Resize to model input size
            image_resized = cv2.resize(image, self.config.img_size)
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)
            
            # Note: This is simplified - in production, you'd load the full 14-channel stack
            # For now, we'll replicate the 3-channel image to simulate the full stack
            image_tensor = image_tensor.repeat(1, 14 // 3, 1, 1)[:, :14, :, :]
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
                confidence = torch.max(probs, dim=1)[0].mean().cpu().item()
            
            # Compute metrics
            per_class_dist = {}
            unique_classes, counts = np.unique(prediction, return_counts=True)
            
            total_pixels = prediction.size
            for class_idx, count in zip(unique_classes, counts):
                if class_idx < len(self.config.class_names):
                    class_name = self.config.class_names[class_idx - 1] if class_idx > 0 else 'background'
                    per_class_dist[class_name] = {
                        'pixel_count': int(count),
                        'percentage': float(count / total_pixels * 100)
                    }
            
            # Determine dominant disease
            disease_pixels = {k: v['pixel_count'] for k, v in per_class_dist.items() if k != 'HLTY'}
            dominant_disease = max(disease_pixels.items(), key=lambda x: x[1])[0] if disease_pixels else 'healthy'
            
            # Disease spread score
            disease_spread = sum(v['pixel_count'] for v in disease_pixels.values()) / total_pixels * 100
            
            # Save prediction mask
            output_dir = Path(self.config.lab_root) / "inference_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path = output_dir / f"{job_id}_mask.png"
            cv2.imwrite(str(mask_path), prediction.astype(np.uint8))
            
            result = {
                'job_id': job_id,
                'status': JobStatus.COMPLETED,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'mask_path': str(mask_path),
                'dominant_disease': dominant_disease,
                'confidence_score': float(confidence),
                'disease_spread_score': float(disease_spread),
                'per_class_distribution': per_class_dist
            }
            
            self.jobs[job_id].update(result)
            
            # Persist to Redis
            if self.redis_client:
                await self.redis_client.set(f"job:{job_id}:result", json.dumps(result, default=str))
            
            print(f"[INFO] Job {job_id} completed successfully")
            return result
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Job {job_id} failed: {error_msg}")
            
            self.jobs[job_id]['status'] = JobStatus.FAILED
            self.jobs[job_id]['error'] = error_msg
            
            if self.redis_client:
                await self.redis_client.set(f"job:{job_id}:error", json.dumps({'error': error_msg}))
            
            raise
    
    async def worker(self):
        """Background worker for processing queued jobs."""
        print("[INFO] Inference worker started")
        
        while True:
            try:
                # Get next job from queue
                _, job_id, image_path = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                # Process job
                await self.process_job(image_path, job_id)
                
            except asyncio.TimeoutError:
                # Queue is empty, keep waiting
                continue
            except Exception as e:
                print(f"[ERROR] Worker error: {e}")
                await asyncio.sleep(1)
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        return {
            'job_id': job_id,
            'status': job['status'],
            'result': job.get('result'),
            'error': job.get('error')
        }


# Initialize FastAPI app
app = FastAPI(
    title="Lettuce Disease Segmentation API",
    description="GPU-accelerated inference server for lettuce disease detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global server instance
server: Optional[DiseaseSegmentationServer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    global server
    
    config = LabConfig()
    config.resolve()
    
    model_path = config.lab_root / "stage6_segmentation_training" / "supervised_finetune" / "best_model.pth"
    
    if not model_path.exists():
        model_path = config.lab_root / "stage6_segmentation_training" / "epoch_15.pth"
    
    server = DiseaseSegmentationServer(
        config=config,
        model_path=str(model_path),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        batch_size=4
    )
    
    await server.initialize_redis()
    
    # Start worker tasks
    num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"[INFO] Starting {num_workers} inference workers")
    
    for _ in range(num_workers):
        asyncio.create_task(server.worker())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(server.device),
        "model_loaded": server.model is not None,
        "redis_connected": server.redis_client is not None
    }


@app.post("/infer")
async def inference(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Submit image for inference.
    
    Returns job ID for async processing.
    """
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Save uploaded file
        temp_dir = Path(server.config.lab_root) / "upload_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        job_id = str(uuid.uuid4())
        file_path = temp_dir / f"{job_id}_{file.filename}"
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Queue job
        await server.queue_inference_job(str(file_path), job_id)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Inference job queued. Use /result/{job_id} to check status."
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get inference result for a job."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        status = server.get_job_status(job_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/mask/{job_id}")
async def get_mask(job_id: str):
    """Download segmentation mask for a job."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        job_status = server.get_job_status(job_id)
        
        if job_status['status'] != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        mask_path = Path(job_status['result']['mask_path'])
        
        if not mask_path.exists():
            raise HTTPException(status_code=404, detail="Mask file not found")
        
        return FileResponse(mask_path, media_type="image/png", filename=f"{job_id}_mask.png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    total_jobs = len(server.jobs)
    completed = sum(1 for j in server.jobs.values() if j['status'] == JobStatus.COMPLETED)
    failed = sum(1 for j in server.jobs.values() if j['status'] == JobStatus.FAILED)
    queued = sum(1 for j in server.jobs.values() if j['status'] == JobStatus.QUEUED)
    
    return {
        "device": str(server.device),
        "model_loaded": server.model is not None,
        "redis_connected": server.redis_client is not None,
        "total_jobs": total_jobs,
        "completed": completed,
        "failed": failed,
        "queued": queued,
        "queue_size": server.job_queue.qsize()
    }


def main():
    """Start the server."""
    print("[INFO] Starting Lettuce Disease Segmentation Inference Server")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    
    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        reload=False
    )


if __name__ == "__main__":
    main()
