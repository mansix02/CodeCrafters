from flask import Flask, Response, render_template, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import threading
import time
import json
import os
import urllib.request
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='dist')
CORS(app)  # Enable CORS for all routes

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
camera_enabled = False
last_detection = None
yolo_model = None
model_initialized = False

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'spitting_detection_model.pt')
SAMPLE_DATASET_PATH = os.path.join(MODEL_DIR, 'sample_data')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATASET_PATH, exist_ok=True)

def download_pretrained_model():
    """Download a pre-trained YOLO model if not already present"""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading pre-trained YOLOv8 model...")
            # Using YOLOv8n as a base model as it's smaller and faster
            from ultralytics import YOLO
            
            # Download the standard YOLOv8n model
            model = YOLO('yolov8n.pt')
            
            # For this demo, we'll use the standard model directly
            # In a real implementation, you would fine-tune it on spitting behavior
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
        else:
            logger.info(f"Model already exists at {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def init_yolo_model():
    """Initialize the YOLO model for detection"""
    global yolo_model, model_initialized
    
    if model_initialized:
        return True
        
    try:
        # Make sure the model file exists
        if not os.path.exists(MODEL_PATH):
            success = download_pretrained_model()
            if not success:
                return False
                
        # Load the YOLO model
        from ultralytics import YOLO
        logger.info("Loading YOLO model...")
        yolo_model = YOLO(MODEL_PATH)
        model_initialized = True
        logger.info("YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        return False

def detect_spitting(frame):
    """
    Detect spitting behavior using YOLO.
    In a real implementation, this would use a model fine-tuned specifically for spitting detection.
    For this demo, we're using YOLOv8 to detect people and simulate spitting detection.
    """
    global yolo_model
    
    # Check if model is initialized
    if not model_initialized:
        success = init_yolo_model()
        if not success:
            # Draw text indicating model initialization failure
            cv2.putText(frame, "Model initialization failed", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, None
    
    try:
        # Run YOLOv8 inference on the frame
        results = yolo_model(frame, verbose=False)
        
        # Get the first result
        result = results[0]
        
        # Check if there are any detections
        has_person = False
        spitting_detected = False
        
        # Process detections (look for people)
        detected_objects = []
        
        # Draw bounding boxes and labels
        annotated_frame = frame.copy()
        
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = result.names[cls_id]
            
            # Store detected object
            detected_objects.append((cls_name, conf))
            
            # Check if it's a person
            if cls_name == 'person' and conf > 0.5:
                has_person = True
                
                # Simulate spitting detection with a random factor
                # In a real implementation, this would use a specialized model
                # trained specifically for spitting behavior detection
                if time.time() % 30 < 1:  # Occasional detection for demo purposes
                    spitting_detected = True
                    # Draw red box for spitting person
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"SPITTING ({conf:.2f})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw green box for normal person
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Person ({conf:.2f})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw detection boxes for other objects
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"{cls_name} ({conf:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add summary information to the frame
        cv2.putText(annotated_frame, f"Detected: {len(detected_objects)} objects", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        if spitting_detected:
            # Add ALERT at the top of the frame
            cv2.putText(annotated_frame, "SPITTING ALERT!", 
                       (frame.shape[1]//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return True, annotated_frame
            
        return False, annotated_frame
        
    except Exception as e:
        logger.error(f"Error in spitting detection: {e}")
        cv2.putText(frame, f"Detection error: {str(e)[:50]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, None

def camera_stream():
    global camera, output_frame, lock, camera_enabled, last_detection
    
    while True:
        if not camera_enabled:
            # Camera is not enabled, wait and check again
            time.sleep(0.1)
            continue
            
        if camera is None or not camera.isOpened():
            # Try to initialize the camera
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    logger.error("Error: Could not open camera.")
                    camera_enabled = False
                    time.sleep(1)  # Wait before trying again
                    continue
            except Exception as e:
                logger.error(f"Camera error: {e}")
                camera_enabled = False
                time.sleep(1)
                continue
        
        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            logger.error("Failed to read frame")
            camera.release()
            camera = None
            time.sleep(0.1)
            continue
            
        # Process the frame (detect spitting)
        spitting_detected, processed_frame = detect_spitting(frame)
        
        display_frame = processed_frame if processed_frame is not None else frame
        
        # If spitting was detected, save the detection
        if spitting_detected and processed_frame is not None:
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            detection_jpg = base64.b64encode(buffer).decode('utf-8')
            
            # Save detection data
            last_detection = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'image': f"data:image/jpeg;base64,{detection_jpg}"
            }
            logger.info(f"Spitting detected at {last_detection['timestamp']}")
        
        # Apply timestamp
        cv2.putText(display_frame, time.strftime('%Y-%m-%d %H:%M:%S'), 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
        
        # Convert to JPEG
        ret, jpeg = cv2.imencode('.jpg', display_frame)
        
        # Acquire the lock to update the output frame
        with lock:
            output_frame = jpeg.tobytes()
        
        # Control the frame rate
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    global output_frame, lock
    
    while True:
        # Wait until we have a frame
        if output_frame is None:
            time.sleep(0.1)
            continue
            
        # Acquire the lock to get the current output frame
        with lock:
            frame_data = output_frame.copy() if output_frame is not None else None
            
        if frame_data is None:
            time.sleep(0.1)
            continue
            
        # Yield the frame in multipart response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_enabled
    camera_enabled = True
    
    # Initialize YOLO model in background if not already
    if not model_initialized:
        threading.Thread(target=init_yolo_model).start()
        
    return jsonify({'success': True, 'message': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_enabled, camera
    camera_enabled = False
    
    # Release the camera if it exists
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_last_detection')
def get_last_detection():
    global last_detection
    if last_detection is None:
        return jsonify({'success': False, 'message': 'No detection available'})
    return jsonify({'success': True, 'detection': last_detection})

@app.route('/capture_detection')
def capture_detection():
    global output_frame, lock
    
    if output_frame is None:
        return jsonify({'success': False, 'message': 'No frame available'})
    
    # Acquire the lock to get the current output frame
    with lock:
        frame_data = output_frame.copy() if output_frame is not None else None
    
    if frame_data is None:
        return jsonify({'success': False, 'message': 'No frame available'})
    
    # Encode as base64
    jpg_as_text = base64.b64encode(frame_data).decode('utf-8')
    
    # Create detection data
    detection = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'image': f"data:image/jpeg;base64,{jpg_as_text}"
    }
    
    # Save as last detection
    global last_detection
    last_detection = detection
    
    return jsonify({
        'success': True, 
        'message': 'Detection captured',
        'detection': detection
    })

@app.route('/model_status')
def model_status():
    global model_initialized
    return jsonify({
        'initialized': model_initialized,
        'model_path': MODEL_PATH if os.path.exists(MODEL_PATH) else None
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Try to initialize the YOLO model
    threading.Thread(target=init_yolo_model).start()
    
    # Start the camera stream thread
    camera_thread = threading.Thread(target=camera_stream)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False) 