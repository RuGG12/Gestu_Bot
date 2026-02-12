"""
GestuBot - Data Collection Tool

Captures frames, extracts features, and builds a labeled CSV dataset.
Use HSV trackbars to tune skin detection for your lighting/skin tone.

Usage:
    python data_collector.py

Controls:
    0-5: Label and capture current gesture
    q:   Quit and save
    r:   Reset dataset
"""

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, Optional

# Import shared utilities
from utils import (
    extract_features,
    preprocess_frame,
    find_largest_contour,
    GESTURE_CLASSES
)


# --- Configuration ---

# Default HSV range for skin detection (tweak with trackbars at runtime)
DEFAULT_HSV_LOWER = (0, 30, 60)
DEFAULT_HSV_UPPER = (20, 150, 255)

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATASET_PATH = os.path.join(DATA_DIR, 'gestures.csv')

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# --- Trackbar Helpers ---

def nothing(x):
    """Dummy callback for trackbar (required by OpenCV)."""
    pass


def create_hsv_trackbars(window_name: str) -> None:
    """
    Create HSV threshold sliders. Interactive tuning is essential
    since skin detection varies a lot with lighting and skin tone.
    H range 0-179 (OpenCV halves the 0-360 range), S/V 0-255.
    """
    cv2.createTrackbar('H_Low', window_name, DEFAULT_HSV_LOWER[0], 179, nothing)
    cv2.createTrackbar('H_High', window_name, DEFAULT_HSV_UPPER[0], 179, nothing)
    cv2.createTrackbar('S_Low', window_name, DEFAULT_HSV_LOWER[1], 255, nothing)
    cv2.createTrackbar('S_High', window_name, DEFAULT_HSV_UPPER[1], 255, nothing)
    cv2.createTrackbar('V_Low', window_name, DEFAULT_HSV_LOWER[2], 255, nothing)
    cv2.createTrackbar('V_High', window_name, DEFAULT_HSV_UPPER[2], 255, nothing)


def get_hsv_values(window_name: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Read current HSV values from trackbars."""
    h_low = cv2.getTrackbarPos('H_Low', window_name)
    h_high = cv2.getTrackbarPos('H_High', window_name)
    s_low = cv2.getTrackbarPos('S_Low', window_name)
    s_high = cv2.getTrackbarPos('S_High', window_name)
    v_low = cv2.getTrackbarPos('V_Low', window_name)
    v_high = cv2.getTrackbarPos('V_High', window_name)
    
    return (h_low, s_low, v_low), (h_high, s_high, v_high)


# --- Data Collector ---

class DataCollector:
    """
    Interactive data collection for gesture training.
    
    Handles:
    - Camera capture with live preview
    - HSV threshold adjustment
    - Feature extraction and labeling
    - Dataset persistence to CSV
    """
    
    def __init__(self):
        self.samples = []
        self.feature_names = [
            'aspect_ratio', 'extent', 'solidity',  # Geometric (3)
            'defect_count',                         # Hull defects (1)
            'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6',  # Hu moments (7)
            'cx_norm', 'cy_norm',                   # Center of mass (2)
            'label'                                 # Target label
        ]
        
        # Load existing dataset if present
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing dataset to allow incremental collection."""
        if os.path.exists(DATASET_PATH):
            try:
                df = pd.read_csv(DATASET_PATH)
                self.samples = df.values.tolist()
                print(f"[INFO] Loaded {len(self.samples)} existing samples from {DATASET_PATH}")
                self._print_class_distribution()
            except Exception as e:
                print(f"[WARN] Could not load existing data: {e}")
    
    def _print_class_distribution(self) -> None:
        """Print current sample count per class."""
        if not self.samples:
            print("[INFO] No samples collected yet")
            return
        
        df = pd.DataFrame(self.samples, columns=self.feature_names)
        distribution = df['label'].value_counts().sort_index()
        
        print("\n=== Class Distribution ===")
        for cls_id, count in distribution.items():
            cls_name = GESTURE_CLASSES.get(int(cls_id), "Unknown")
            print(f"  Class {int(cls_id)} ({cls_name}): {count} samples")
        print(f"  Total: {len(self.samples)} samples\n")
    
    def add_sample(self, features: np.ndarray, label: int) -> None:
        """Add a feature vector with label to the dataset."""
        sample = features.tolist() + [label]
        self.samples.append(sample)
        
        cls_name = GESTURE_CLASSES.get(label, "Unknown")
        print(f"[CAPTURED] Class {label} ({cls_name}) - Total samples: {len(self.samples)}")
    
    def add_background_sample(self) -> None:
        """
        Add a "background" sample (class 5) using a zero feature vector.
        This is how the classifier learns to do nothing when no hand is visible.
        """
        # Create zero feature vector for background
        zero_features = np.zeros(13, dtype=np.float64)
        self.add_sample(zero_features, 5)
    
    def save(self) -> None:
        """Save dataset to CSV file."""
        if not self.samples:
            print("[WARN] No samples to save")
            return
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        df = pd.DataFrame(self.samples, columns=self.feature_names)
        df.to_csv(DATASET_PATH, index=False)
        
        print(f"\n[SAVED] Dataset saved to {DATASET_PATH}")
        self._print_class_distribution()
    
    def reset(self) -> None:
        """Clear all collected samples."""
        self.samples = []
        print("[RESET] All samples cleared")


def run_collection():
    """
    Main data collection loop.
    
    Displays live camera feed with mask overlay.
    Keyboard controls:
        0-5: Capture sample for gesture class
        q: Quit and save
        r: Reset dataset
    """
    print("\n" + "="*60)
    print("GestuBot Data Collector")
    print("="*60)
    print("\nControls:")
    print("  0-5: Capture sample for gesture class")
    print("  q: Quit and save dataset")
    print("  r: Reset/clear dataset")
    print("\nGesture Classes:")
    for cls_id, cls_name in GESTURE_CLASSES.items():
        print(f"  {cls_id}: {cls_name}")
    print("\n" + "="*60 + "\n")
    
    # Initialize
    collector = DataCollector()
    
    # Set up camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    # Create windows
    cv2.namedWindow('GestuBot - Data Collection')
    cv2.namedWindow('Mask Preview')
    
    # Create HSV trackbars on main window
    create_hsv_trackbars('GestuBot - Data Collection')
    
    print("[INFO] Camera opened. Adjust HSV sliders to isolate your hand.")
    print("[INFO] Position your hand and press 0-5 to capture samples.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Flip horizontally for mirror effect (more intuitive)
        frame = cv2.flip(frame, 1)
        
        # Get current HSV values from trackbars
        hsv_lower, hsv_upper = get_hsv_values('GestuBot - Data Collection')
        
        # Apply vision pipeline
        mask, _ = preprocess_frame(frame, hsv_lower, hsv_upper)
        
        # Find hand contour
        contour = find_largest_contour(mask)
        
        # Draw visualization
        display = frame.copy()
        
        if contour is not None:
            # Draw contour and bounding box
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(display, [hull], -1, (255, 255, 0), 1)
            
            # Extract and display feature info
            features = extract_features(contour)
            if features is not None:
                info_text = f"Features: {len(features)} dims | Defects: {int(features[3])}"
                cv2.putText(display, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No hand detected (Class 5: Background)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display sample count
        cv2.putText(display, f"Samples: {len(collector.samples)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display HSV values
        hsv_text = f"HSV: {hsv_lower} - {hsv_upper}"
        cv2.putText(display, hsv_text, (10, FRAME_HEIGHT - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show windows
        cv2.imshow('GestuBot - Data Collection', display)
        cv2.imshow('Mask Preview', mask)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            collector.save()
            break
        elif key == ord('r'):
            collector.reset()
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            label = int(chr(key))
            
            if label == 5:
                # Background class - capture even without hand
                collector.add_background_sample()
            elif contour is not None:
                features = extract_features(contour)
                if features is not None:
                    collector.add_sample(features, label)
                else:
                    print("[WARN] Could not extract features, try again")
            else:
                print("[WARN] No hand detected. For background samples, press '5'")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Data collection complete.")


if __name__ == '__main__':
    run_collection()
