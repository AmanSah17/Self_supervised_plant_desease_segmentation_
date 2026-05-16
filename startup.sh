#!/bin/bash

# Lettuce Disease Segmentation - Complete Startup Script
# This script sets up and starts all components for development or production

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Lettuce Disease Segmentation - Complete System Startup       ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Configuration
ENVIRONMENT="${1:-development}"
CUDA_DEVICE="${2:-0}"

echo ""
echo "📋 Configuration:"
echo "   Environment: $ENVIRONMENT"
echo "   CUDA Device: $CUDA_DEVICE"
echo "   Python: $(python --version)"
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. CPU-only inference."
fi

echo ""

# Activate CUDA environment if on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "🪟 Windows detected. Activating CUDA environment..."
    if [ -f "d:/gemma4/Scripta/activate.bat" ]; then
        cmd /c "d:\gemma4\Scripta\activate.bat"
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p lettuce_ssl_segmentation_lab/stage9_test_inference/segmentation_masks
mkdir -p lettuce_ssl_segmentation_lab/compiled_models
mkdir -p backend/logs
mkdir -p frontend/static
mkdir -p inference_output

# Install/update dependencies
if [ "$ENVIRONMENT" = "development" ]; then
    echo "📦 Installing development dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements_backend.txt
    echo "✅ Dependencies installed"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export PYTHONUNBUFFERED=1
export LOG_LEVEL=info

echo ""
echo "🚀 Starting services..."
echo ""

if [ "$ENVIRONMENT" = "development" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "DEVELOPMENT MODE"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if Redis is available
    if command -v redis-server &> /dev/null; then
        echo "✅ Starting Redis server..."
        redis-server --daemonize yes --port 6379 --appendonly yes
        sleep 2
    else
        echo "⚠️  Redis not found. Install with: pip install redis"
        echo "   Or run: docker run -d -p 6379:6379 redis:7.2-alpine"
    fi
    
    echo ""
    echo "📊 Starting FastAPI backend server..."
    echo "   Backend will run on: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    
    export REDIS_URL="redis://localhost:6379"
    
    # Start backend in background
    python -m uvicorn backend.backend_server:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info &
    BACKEND_PID=$!
    
    sleep 3
    
    echo ""
    echo "🌐 Frontend ready at:"
    echo "   File: file:///$(pwd)/frontend/index.html"
    echo "   Or serve with: python -m http.server -d frontend 3000"
    echo ""
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "✅ Development environment ready!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Backend PID: $BACKEND_PID"
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Wait for backend
    wait $BACKEND_PID
    
elif [ "$ENVIRONMENT" = "production" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "PRODUCTION MODE (Docker)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Install Docker to run production mode."
        exit 1
    fi
    
    echo "🐳 Building Docker images..."
    docker-compose build
    
    echo ""
    echo "🚀 Starting containers..."
    docker-compose up -d
    
    echo ""
    echo "⏳ Waiting for services to be ready..."
    sleep 5
    
    # Check service health
    echo ""
    echo "📊 Service Status:"
    docker-compose ps
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "✅ Production environment ready!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Frontend:  http://localhost:3000"
    echo "Backend:   http://localhost:8000"
    echo "API Docs:  http://localhost:8000/docs"
    echo "Redis:     localhost:6379"
    echo ""
    echo "View logs with: docker-compose logs -f backend"
    echo "Stop services: docker-compose down"
    echo ""
    
else
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Usage: ./startup.sh [development|production] [CUDA_DEVICE]"
    exit 1
fi
