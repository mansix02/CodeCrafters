#!/usr/bin/env python3
"""
Training script for spitting detection model using YOLOv8
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import random
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("spit_detection_trainer")

def parse_args():
    """Parse command line arguments for model training"""
    parser = argparse.ArgumentParser(description="Train YOLO model for spitting detection")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='Device to train on (e.g., 0 for GPU or cpu)')
    parser.add_argument('--weights', default='yolov8n.pt', help='Initial weights path')
    parser.add_argument('--data', default='dataset/data.yaml', help='Dataset YAML path')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--no-download', action='store_true', help='Skip dataset download and preparation')
    return parser.parse_args()

def download_sample_data():
    """
    Download sample data for training if it doesn't exist
    For demo purposes, this function creates synthetic data
    In a real-world scenario, use a real dataset like Roboflow's spitting dataset
    """
    logger.info("Preparing sample training data")
    
    # Define paths
    train_img_dir = Path("dataset/images/train")
    val_img_dir = Path("dataset/images/val")
    train_label_dir = Path("dataset/labels/train")
    val_label_dir = Path("dataset/labels/val")
    
    # Create synthetic data for demonstration
    num_train_samples = 20
    num_val_samples = 5
    
    # Generate synthetic training data
    for i in range(num_train_samples):
        # Create a blank image (640x640 with 3 channels)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some random noise to make it look more realistic
        noise = np.random.randint(40, 80, img.shape, dtype=np.uint8)
        img += noise
        
        # Draw a person (rectangle)
        person_x = random.randint(100, 400)
        person_y = random.randint(100, 400)
        person_w = random.randint(100, 200)
        person_h = random.randint(200, 300)
        cv2.rectangle(img, (person_x, person_y), (person_x + person_w, person_y + person_h), (0, 0, 255), -1)
        
        # Draw a spitting action (small circle) for half of the images
        is_spitting = random.random() > 0.5
        
        if is_spitting:
            spit_x = person_x + person_w // 2
            spit_y = person_y - 20
            spit_radius = random.randint(5, 15)
            cv2.circle(img, (spit_x, spit_y), spit_radius, (0, 255, 0), -1)
        
        # Save the image
        img_path = train_img_dir / f"sample_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Create YOLO format label
        label_path = train_label_dir / f"sample_{i}.txt"
        
        # Convert bounding box to YOLO format (class x_center y_center width height)
        # Normalized to be between 0 and 1
        with open(label_path, 'w') as f:
            # Person label (class 0)
            x_center = (person_x + person_w / 2) / 640
            y_center = (person_y + person_h / 2) / 640
            width = person_w / 640
            height = person_h / 640
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Add spitting label (class 1) if applicable
            if is_spitting:
                spit_x_center = spit_x / 640
                spit_y_center = spit_y / 640
                spit_width = (spit_radius * 2) / 640
                spit_height = (spit_radius * 2) / 640
                f.write(f"1 {spit_x_center:.6f} {spit_y_center:.6f} {spit_width:.6f} {spit_height:.6f}\n")
    
    # Generate synthetic validation data (similar to training data)
    for i in range(num_val_samples):
        # Create a blank image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add noise
        noise = np.random.randint(40, 80, img.shape, dtype=np.uint8)
        img += noise
        
        # Draw a person
        person_x = random.randint(100, 400)
        person_y = random.randint(100, 400)
        person_w = random.randint(100, 200)
        person_h = random.randint(200, 300)
        cv2.rectangle(img, (person_x, person_y), (person_x + person_w, person_y + person_h), (0, 0, 255), -1)
        
        # Draw a spitting action for half of the images
        is_spitting = random.random() > 0.5
        
        if is_spitting:
            spit_x = person_x + person_w // 2
            spit_y = person_y - 20
            spit_radius = random.randint(5, 15)
            cv2.circle(img, (spit_x, spit_y), spit_radius, (0, 255, 0), -1)
        
        # Save the image
        img_path = val_img_dir / f"sample_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Create YOLO format label
        label_path = val_label_dir / f"sample_{i}.txt"
        
        # Convert bounding box to YOLO format
        with open(label_path, 'w') as f:
            # Person label (class 0)
            x_center = (person_x + person_w / 2) / 640
            y_center = (person_y + person_h / 2) / 640
            width = person_w / 640
            height = person_h / 640
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Add spitting label (class 1) if applicable
            if is_spitting:
                spit_x_center = spit_x / 640
                spit_y_center = spit_y / 640
                spit_width = (spit_radius * 2) / 640
                spit_height = (spit_radius * 2) / 640
                f.write(f"1 {spit_x_center:.6f} {spit_y_center:.6f} {spit_width:.6f} {spit_height:.6f}\n")
    
    logger.info(f"Created {num_train_samples} training samples and {num_val_samples} validation samples")

def train_model(args):
    """Train the YOLOv8 model using the prepared dataset"""
    logger.info("Starting model training")
    
    # Initialize the model with pre-trained weights
    model = YOLO(args.weights)
    
    # Train the model
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.img_size,
            device=args.device,
            workers=args.workers,
            verbose=True,
            name='spitting_detection'
        )
        logger.info(f"Training completed successfully: {results}")
        return True
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

def main():
    args = parse_args()
    
    if not args.no_download:
        # Download or prepare sample data
        try:
            download_sample_data()
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return
    
    # Train the model
    success = train_model(args)
    
    if success:
        logger.info("Model training complete. The model is saved in the 'runs/train/spitting_detection' directory.")
    else:
        logger.error("Model training failed.")

if __name__ == "__main__":
    main() 