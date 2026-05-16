@echo off
REM Lettuce Disease Segmentation - Windows Startup Script
REM Starts all components for development

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo  Lettuce Disease Segmentation - Complete System Startup (Windows)
echo ====================================================================
echo.

set ENVIRONMENT=%1
if "!ENVIRONMENT!"=="" set ENVIRONMENT=development

set CUDA_DEVICE=%2
if "!CUDA_DEVICE!"=="" set CUDA_DEVICE=0

echo Configuration:
echo   Environment: !ENVIRONMENT!
echo   CUDA Device: !CUDA_DEVICE!
echo   Python: 
python --version
echo.

REM Check NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if !errorlevel! equ 0 (
    echo OK GPU detected:
    nvidia-smi --query-gpu=name --format=csv,noheader
) else (
    echo WARNING No NVIDIA GPU detected. CPU-only inference.
)

echo.

REM Activate CUDA environment
if exist "d:\gemma4\Scripta\activate.bat" (
    echo Activating CUDA environment...
    call d:\gemma4\Scripta\activate.bat
) else (
    echo WARNING Cannot find CUDA environment at d:\gemma4\Scripta\activate.bat
)

echo.

REM Create necessary directories
echo Creating directories...
if not exist "lettuce_ssl_segmentation_lab\stage9_test_inference\segmentation_masks" mkdir lettuce_ssl_segmentation_lab\stage9_test_inference\segmentation_masks
if not exist "lettuce_ssl_segmentation_lab\compiled_models" mkdir lettuce_ssl_segmentation_lab\compiled_models
if not exist "backend\logs" mkdir backend\logs
if not exist "frontend\static" mkdir frontend\static
if not exist "inference_output" mkdir inference_output
echo OK Created directories

echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements_backend.txt
echo OK Dependencies installed

echo.
echo ====================================================================
echo  DEVELOPMENT MODE - Starting Services
echo ====================================================================
echo.

REM Set environment variables
set CUDA_VISIBLE_DEVICES=!CUDA_DEVICE!
set PYTHONUNBUFFERED=1
set LOG_LEVEL=info
set REDIS_URL=redis://localhost:6379

echo.
echo Starting FastAPI Backend Server...
echo   URL: http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo.

REM Start Redis (if available)
where redis-server >nul 2>&1
if !errorlevel! equ 0 (
    echo Starting Redis...
    start "Redis" redis-server --port 6379 --appendonly yes
    timeout /t 2 /nobreak
) else (
    echo WARNING Redis not found. Queue management disabled.
    echo Install: pip install redis
)

echo.

REM Start backend server
start "Backend Server" cmd /k python -m uvicorn backend.backend_server:app --host 0.0.0.0 --port 8000 --reload

timeout /t 3 /nobreak

echo.
echo ====================================================================
echo OK Development environment ready!
echo ====================================================================
echo.
echo Frontend:
echo   Open: file:///%cd%/frontend/index.html
echo.
echo Backend:
echo   URL: http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo   Health: http://localhost:8000/health
echo.
echo IMPORTANT: Keep this window open. Close to stop all services.
echo.

pause
