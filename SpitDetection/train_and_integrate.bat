@echo off
echo ===== Spitting Detection Model Training and Integration =====

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python first.
    exit /b 1
)

REM Check if ultralytics is installed
python -c "import ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing ultralytics package...
    pip install ultralytics
)

REM Create dataset structure if it doesn't exist
echo Ensuring dataset structure exists...
if not exist dataset\images\train mkdir dataset\images\train
if not exist dataset\images\val mkdir dataset\images\val
if not exist dataset\labels\train mkdir dataset\labels\train
if not exist dataset\labels\val mkdir dataset\labels\val

REM Run training script
echo Starting model training...
python train.py --epochs 20 --batch 8 --device cpu

REM Check if training was successful
if not exist runs\train\spitting_detection\weights\best.pt (
    echo Training failed. Model weights not found.
    exit /b 1
)

REM Integrate the model with the server
echo Integrating trained model with server...
python model_integration.py

echo ===== Process completed =====
echo You can now run the server to use the trained model.
echo Run 'python server.py' to start the server with the new model. 