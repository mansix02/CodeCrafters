from flask import Flask, Response, render_template, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import threading
import time
import json
import os
import random

app = Flask(__name__, static_folder='dist')
CORS(app)  # Enable CORS for all routes

# Global variables for camera
camera = None
output_frame = None
lock = threading.Lock()
camera_enabled = False
last_detection = None
camera_init_attempted = False

def detect_objects(frame):
    """
    Basic object detection function using OpenCV
    In a real implementation, this would use YOLO or another ML model
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply some blur to reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Add visual indicator for processing
    height, width = frame.shape[:2]
    cv2.putText(frame, "Processing Feed", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw detection box (simulated detection)
    # In a real app, this would use actual detection results
    if random.random() < 0.05:  # Simulate occasional detections
        # Draw a detection box
        cv2.rectangle(frame, (width//3, height//3), 
                     (2*width//3, 2*height//3), (0, 0, 255), 2)
        
        # Add text label
        cv2.putText(frame, "SPITTING DETECTED", (width//3, height//3 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Return a copy of the frame as the detection image
        return True, frame.copy()
    
    return False, None

def release_camera():
    """Safely release the camera resource"""
    global camera, camera_enabled
    
    if camera is not None:
        camera.release()
        camera = None
    
    camera_enabled = False
    print("Camera released")

def initialize_camera():
    """Initialize the camera with multiple attempts if needed"""
    global camera, camera_init_attempted
    
    if camera is not None:
        return True
    
    camera_init_attempted = True
    
    # Try different camera indices and API backends
    for camera_index in [0, 1]:
        # Try different backends on Windows
        for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                print(f"Trying to initialize camera {camera_index} with API {api}")
                cap = cv2.VideoCapture(camera_index, api)
                
                if cap is None or not cap.isOpened():
                    print(f"Failed with camera index {camera_index} and API {api}")
                    if cap is not None:
                        cap.release()
                    continue
                    
                # Read multiple test frames to ensure camera is working
                for _ in range(5):  # Read several frames to warm up the camera
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"Couldn't read initial frame from camera {camera_index}")
                        break
                    time.sleep(0.1)  # Short delay between test frames
                
                # Read one more frame to verify
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Could not read test frame from camera {camera_index}")
                    cap.release()
                    continue
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Successfully initialized
                camera = cap
                print(f"Successfully initialized camera with index {camera_index} and API {api}")
                
                # Immediately process and save a frame to make it available
                success, first_frame = camera.read()
                if success and first_frame is not None:
                    # Add timestamp to frame
                    cv2.putText(first_frame, time.strftime('%Y-%m-%d %H:%M:%S'), 
                            (10, first_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 1)
                    
                    # Convert to JPEG
                    ret, jpeg = cv2.imencode('.jpg', first_frame)
                    
                    # Save the first frame
                    global output_frame, lock
                    with lock:
                        output_frame = jpeg.tobytes()
                
                return True
                
            except Exception as e:
                print(f"Error initializing camera {camera_index} with API {api}: {e}")
                if 'cap' in locals() and cap is not None:
                    cap.release()
    
    # Try one more method: direct camera index without API specification
    try:
        print("Trying direct camera access without API specification")
        cap = cv2.VideoCapture(0)
        
        if cap is not None and cap.isOpened():
            # Read a test frame to verify camera works
            ret, frame = cap.read()
            if ret and frame is not None:
                camera = cap
                print("Successfully initialized camera using direct access")
                return True
            else:
                cap.release()
    except Exception as e:
        print(f"Error with direct camera access: {e}")
    
    print("Could not initialize any camera")
    return False

def camera_stream():
    global camera, output_frame, lock, camera_enabled, last_detection, camera_init_attempted
    
    while True:
        # If camera isn't enabled, just wait
        if not camera_enabled:
            time.sleep(0.1)
            continue
            
        # Try to initialize camera if not already done
        if camera is None:
            if not initialize_camera():
                # Failed to initialize camera
                camera_enabled = False
                time.sleep(1)  # Wait before trying again
                continue
        
        # Read a frame from the camera
        try:
            success, frame = camera.read()
            
            # Handle frame reading failure
            if not success or frame is None:
                print("Failed to read frame, attempting to reinitialize camera")
                release_camera()
                # Don't try to initialize immediately - wait for next loop
                time.sleep(0.5)
                continue
            
            # Frame was successfully read, process it
            # Add a visual indicator that the camera is active
            cv2.rectangle(frame, (5, 5), (15, 15), (0, 255, 0), -1)  # Green square indicator
                
            # Process the frame (detect objects)
            detected, detection_image = detect_objects(frame)
            
            # If detection occurred, save it
            if detected and detection_image is not None:
                # Convert to JPEG
                _, buffer = cv2.imencode('.jpg', detection_image)
                detection_jpg = base64.b64encode(buffer).decode('utf-8')
                
                # Save detection data
                last_detection = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'image': f"data:image/jpeg;base64,{detection_jpg}"
                }
                print("Detection saved!")
            
            # Add timestamp to frame
            cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
            
            # Convert to JPEG with high quality
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ret:
                print("Failed to encode frame")
                continue
                
            # Acquire the lock to update the output frame
            with lock:
                output_frame = jpeg.tobytes()
                
        except Exception as e:
            print(f"Error in camera stream: {e}")
            release_camera()
            time.sleep(0.5)
            
        # Control the frame rate - adjust if needed for smoother video
        time.sleep(0.01)  # Higher framerate (~100 FPS max, though camera will limit)

def generate_frames():
    global output_frame, lock
    
    while True:
        # Wait until we have a frame
        if output_frame is None:
            # Create a blank frame instead of a message
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', blank_frame)
            blank_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_data + b'\r\n')
            
            # Short delay before trying again
            time.sleep(0.1)
            continue
            
        # Acquire the lock to get the current output frame
        with lock:
            frame_data = output_frame.copy()
            
        # Yield the frame in multipart response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_enabled
    camera_enabled = True
    return jsonify({'success': True, 'message': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_enabled
    camera_enabled = False
    release_camera()
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
        frame_data = output_frame.copy()
    
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

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/camera_status')
def camera_status():
    global camera, camera_enabled, camera_init_attempted
    
    status = {
        'camera_initialized': camera is not None and camera.isOpened(),
        'camera_enabled': camera_enabled,
        'init_attempted': camera_init_attempted
    }
    
    return jsonify(status)

if __name__ == '__main__':
    # Start the camera stream thread
    camera_thread = threading.Thread(target=camera_stream)
    camera_thread.daemon = True
    camera_thread.start()
    
    print("Server starting... Open your browser to http://localhost:5000")
    print("API endpoints:")
    print("- /start_camera (POST): Start the camera")
    print("- /stop_camera (POST): Stop the camera")
    print("- /video_feed (GET): Get the streaming video feed")
    print("- /get_last_detection (GET): Get the last detection")
    print("- /capture_detection (GET): Manually capture a detection")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)