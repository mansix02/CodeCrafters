#!/usr/bin/env python3
"""
Script to integrate trained spitting detection model with the server application
"""

import os
import sys
import logging
import shutil
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_integration")

def parse_args():
    parser = argparse.ArgumentParser(description="Integrate trained model with server")
    parser.add_argument('--model-path', type=str, default='runs/train/spitting_detection/weights/best.pt',
                        help='Path to the trained model weights')
    parser.add_argument('--target-path', type=str, default='yolo_model/spitting_detection_model.pt',
                        help='Target path for the model in the application')
    return parser.parse_args()

def integrate_model(model_path, target_path):
    """
    Integrate the trained model with the server application
    
    Args:
        model_path: Path to the trained model weights
        target_path: Target path for the model in the application
    """
    # Check if the source model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return False
    
    # Create the target directory if it doesn't exist
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy the model to the target path
    try:
        shutil.copy(model_path, target_path)
        logger.info(f"Successfully copied model from {model_path} to {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying model: {e}")
        return False

def modify_server_file():
    """Modify the server.py file to use the custom spitting detection model"""
    server_file = "server.py"
    
    # Check if the server file exists
    if not os.path.exists(server_file):
        logger.error(f"Server file not found at {server_file}")
        return False
    
    try:
        # Read the server file content
        with open(server_file, 'r') as f:
            content = f.read()
        
        # Check if we need to modify the file (only if not already modified)
        if "model_confidence = 0.25" not in content:
            # Find the function where we need to make changes
            if "def detect_spitting(frame):" in content:
                logger.info("Modifying server.py to enhance spitting detection")
                
                # Split at the function definition
                parts = content.split("def detect_spitting(frame):")
                before = parts[0]
                after = parts[1]
                
                # Find the function body
                lines = after.split('\n')
                indentation = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('"""'):
                        # Skip docstring
                        for j in range(i+1, len(lines)):
                            if lines[j].strip().endswith('"""'):
                                indentation = len(lines[j]) - len(lines[j].lstrip())
                                i = j
                                break
                        break
                
                # Prepare the new function implementation
                spaces = ' ' * indentation
                new_function = f'''def detect_spitting(frame):
    """
    Detect spitting behavior using YOLO.
    This is an enhanced version using our custom-trained model for spitting detection.
    """
    global yolo_model
    
    # Model confidence threshold for detections
    model_confidence = 0.25
    
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
        results = yolo_model(frame, verbose=False, conf=model_confidence)
        
        # Get the first result
        result = results[0]
        
        # Check if there are any detections
        has_person = False
        spitting_detected = False
        
        # Process detections
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
            
            # Person detection (class 0)
            if cls_id == 0 and conf > model_confidence:
                has_person = True
                # Draw green box for normal person
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Person ({conf:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Spitting detection (class 1)
            elif cls_id == 1 and conf > model_confidence:
                spitting_detected = True
                # Draw red box for spitting detection
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"SPITTING ({conf:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
'''
                
                # Replace the function
                new_content = before + new_function
                
                # Find the end of the function
                in_func = False
                func_lines = []
                for line in lines:
                    if line.strip() and not line.startswith(' '):
                        if in_func:
                            # End of function
                            break
                        in_func = True
                    if in_func:
                        func_lines.append(line)
                
                # Add the rest of the file
                new_content += '\n'.join(lines[len(func_lines):])
                
                # Write the modified content back to the file
                with open(server_file, 'w') as f:
                    f.write(new_content)
                
                logger.info("Successfully modified server.py")
                return True
            else:
                logger.error("Could not find detect_spitting function in server.py")
                return False
        else:
            logger.info("Server file already modified")
            return True
    except Exception as e:
        logger.error(f"Error modifying server file: {e}")
        return False

def main():
    args = parse_args()
    
    # Integrate the model
    success = integrate_model(args.model_path, args.target_path)
    if not success:
        logger.error("Failed to integrate model")
        return
    
    # Modify the server file
    success = modify_server_file()
    if not success:
        logger.error("Failed to modify server file")
        return
    
    logger.info("Model integration complete. You can now restart the server to use the custom model.")

if __name__ == "__main__":
    main() 