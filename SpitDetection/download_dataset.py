#!/usr/bin/env python3
"""
Script to download the spitting behavior dataset from Roboflow
"""

import os
import sys
import requests
import zipfile
import io
import logging
import shutil
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_downloader")

def parse_args():
    parser = argparse.ArgumentParser(description="Download spitting behavior dataset")
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help='Output directory for dataset')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic data instead of downloading from Roboflow')
    return parser.parse_args()

def download_roboflow_dataset(output_dir):
    """
    Download the spitting detection dataset from Roboflow
    
    Args:
        output_dir: Directory to save the dataset
    """
    # URL for the public Spitting dataset (replace with actual URL if available)
    # For demonstration purposes, we're using a placeholder URL
    # In a real project, you would use your own Roboflow API key and dataset URL
    dataset_url = "https://app.roboflow.com/ds/8Og1B8FH2J?key=9bpGdUv20s"
    
    try:
        logger.info(f"Downloading dataset from Roboflow: {dataset_url}")
        response = requests.get(dataset_url, stream=True)
        
        if response.status_code == 200:
            # Extract the ZIP file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(output_dir)
            
            logger.info(f"Dataset downloaded and extracted to {output_dir}")
            return True
        else:
            logger.error(f"Failed to download dataset: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def create_synthetic_data(output_dir):
    """
    Create synthetic data for training and validation
    
    Args:
        output_dir: Directory to save the synthetic data
    """
    try:
        import numpy as np
        import cv2
        import random
        
        # Define paths
        train_img_dir = Path(output_dir) / "images" / "train"
        val_img_dir = Path(output_dir) / "images" / "val"
        train_label_dir = Path(output_dir) / "labels" / "train"
        val_label_dir = Path(output_dir) / "labels" / "val"
        
        # Ensure directories exist
        train_img_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)
        train_label_dir.mkdir(parents=True, exist_ok=True)
        val_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic data for demonstration
        num_train_samples = 20
        num_val_samples = 5
        
        logger.info(f"Creating {num_train_samples} synthetic training images")
        
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
        
        logger.info(f"Creating {num_val_samples} synthetic validation images")
        
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
        return True
    except Exception as e:
        logger.error(f"Error creating synthetic data: {e}")
        return False

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_synthetic:
        logger.info("Using synthetic data for training")
        success = create_synthetic_data(args.output_dir)
    else:
        logger.info("Trying to download dataset from Roboflow")
        success = download_roboflow_dataset(args.output_dir)
        
        if not success:
            logger.warning("Failed to download dataset from Roboflow, falling back to synthetic data")
            success = create_synthetic_data(args.output_dir)
    
    if success:
        logger.info(f"Dataset preparation complete. Data saved to {args.output_dir}")
    else:
        logger.error("Failed to prepare dataset")

if __name__ == "__main__":
    main() 