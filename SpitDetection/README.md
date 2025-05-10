# AI Spitting Detection System

A real-time system to detect and alert spitting behavior in public spaces using AI to promote hygiene and support civic enforcement.

## Features

- Real-time spitting detection using YOLOv8
- Live camera feed monitoring
- Alert system for detected incidents
- Dashboard for tracking and viewing detections
- Evidence capture and storage

## Technologies Used

- **Frontend**: React with Vite
- **Backend**: Python Flask API
- **Computer Vision**: YOLOv8 via Ultralytics
- **Video Processing**: OpenCV

## Getting Started

1. Clone the repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```
   npm install
   ```
4. Build the frontend:
   ```
   npm run build
   ```
5. Start the Python server:
   ```
   python server.py
   ```
6. Open a browser and navigate to `http://localhost:5000`

## Demo Credentials

- Username: admin
- Password: admin123

## Project Structure

- `/src` - React frontend code
- `/public` - Static assets
- `/yolo_model` - YOLO model files (downloaded on first run)
- `server.py` - Flask backend for camera and detection

## How it Works

1. The system uses your webcam to stream video to the backend server
2. YOLOv8 processes each frame to detect people
3. A specialized detection algorithm identifies spitting behavior
4. When spitting is detected, the system captures the frame and triggers an alert
5. Detection data is stored and can be viewed in the dashboard

## Note

This system is designed as a proof of concept. For production use, additional training data specific to spitting behavior would be required to improve accuracy.
