"""
GestuBot - Feature Extraction Utilities

Shared feature extraction and vision pipeline logic.
13-dimensional feature vector designed for SVM classification,
with features chosen for robustness to scale and rotation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# --- Constants ---

# Gesture class definitions (6 classes including background)
GESTURE_CLASSES = {
    0: "Fist",           # Stop - release all keys
    1: "Open Palm",      # Forward - W key
    2: "Point Left",     # Left - A key
    3: "Point Right",    # Right - D key
    4: "V-Sign",         # Reverse - S key
    5: "Background"      # No hand detected - no action
}

# Key mappings for each gesture class
GESTURE_KEYS = {
    0: None,    # Fist: release all (stop)
    1: 'w',     # Open Palm: forward
    2: 'a',     # Point Left: left
    3: 'd',     # Point Right: right
    4: 's',     # V-Sign: reverse
    5: None     # Background: no action
}


# --- Feature Extraction ---

def extract_features(contour: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract 13-dim feature vector from a hand contour.

    Features:
      [0-2]  aspect_ratio, extent, solidity (geometric shape)
      [3]    hull defect count (finger-like protrusions)
      [4-10] Hu Moments, log-transformed (rotation/scale invariant)
      [11-12] normalized center of mass

    Returns None if extraction fails (bad contour, etc).
    """
    if contour is None or len(contour) < 5:
        return None
    
    try:
        features = []
        
        # --- Geometric ratios (3 features) ---
        
        # Bounding rectangle for normalization
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return None
        
        # width/height of bounding box
        # fist ~ 1.0, open palm wider ~1.2-1.5
        aspect_ratio = float(w) / h
        features.append(aspect_ratio)
        
        # how much of the bounding box the contour fills
        contour_area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(contour_area) / rect_area if rect_area > 0 else 0
        features.append(extent)
        
        # ratio of contour area to convex hull area
        # lower solidity = more concavities (spread fingers)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(contour_area) / hull_area if hull_area > 0 else 0
        features.append(solidity)
        
        # --- Hull defect count (1 feature) ---
        # Counts deep convexity defects; correlates with # of fingers
        
        defect_count = _count_significant_defects(contour)
        features.append(defect_count)
        
        # --- Hu Moments (7 features) ---
        # Scale/rotation invariant shape descriptors.
        # Log transform because raw values span ~10^-3 to 10^-7.
        
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform with sign preservation and epsilon for stability
        EPSILON = 1e-10
        log_hu = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + EPSILON)
        features.extend(log_hu.tolist())
        
        # --- Normalized center of mass (2 features) ---
        # Position relative to bounding box (0-1 range).
        # Helps distinguish left-pointing vs right-pointing.
        
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = x + w/2, y + h/2
        
        # normalize to bounding box so raw pixel coords don't dominate SVM
        cx_norm = (cx - x) / w
        cy_norm = (cy - y) / h
        features.append(cx_norm)
        features.append(cy_norm)
        
        return np.array(features, dtype=np.float64)
        
    except Exception as e:
        print(f"[WARN] Feature extraction failed: {e}")
        return None


def _count_significant_defects(contour: np.ndarray) -> int:
    """
    Count convexity defects that look like finger valleys.
    Uses a dynamic threshold (8% of contour perimeter) so it
    adapts to different hand sizes. Returns 0-5 typically.
    """
    try:
        # Need at least 5 points for convex hull with returnPoints=False
        if len(contour) < 5:
            return 0
        
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # convexityDefects requires hull indices in ascending order
        if len(hull) < 3:
            return 0
        
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return 0
        
        # Dynamic threshold: 8% of contour arc length
        # This adapts to hand size - larger hands have larger threshold
        arc_length = cv2.arcLength(contour, True)
        depth_threshold = arc_length * 0.08
        
        significant_count = 0
        for i in range(defects.shape[0]):
            # defects[i][0] = [start_idx, end_idx, farthest_idx, depth]
            # depth is in fixed-point format (divide by 256)
            depth = defects[i][0][3] / 256.0
            
            if depth > depth_threshold:
                significant_count += 1
        
        return significant_count
        
    except Exception:
        return 0


# --- Vision Pipeline ---

def preprocess_frame(
    frame: np.ndarray,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
    blur_kernel: int = 5,
    morph_kernel_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: BGR -> HSV mask -> blur -> morphology.

    HSV is way more robust to lighting changes than RGB for skin detection.
    Elliptical kernel for morphology matches hand shape better than rect.
    Opening removes small noise, closing fills small holes.

    Returns (binary_mask, hsv_frame).
    """
    # Convert to HSV color space (better for skin detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create binary mask for skin tones
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    
    # Gaussian blur to reduce noise before morphology
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    
    # Morphological operations with elliptical kernel
    # Ellipse approximates hand shape better than rectangle
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (morph_kernel_size, morph_kernel_size)
    )
    
    # Opening: erosion then dilation (removes small noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing: dilation then erosion (fills small holes)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # "Face shield" - zero out top 30% of mask to prevent face detection.
    # Faces sit in the upper third, hands enter from below.
    h, w = mask.shape
    face_shield_height = int(h * 0.30)
    cv2.rectangle(mask, (0, 0), (w, face_shield_height), 0, -1)
    
    return mask, hsv


def find_largest_contour(mask: np.ndarray, min_area: int = 5000) -> Optional[np.ndarray]:
    """
    Find the largest contour in a binary mask (assumed to be the hand).
    min_area filters noise. Returns None for empty frames -> class 5.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour by area
    largest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < min_area:
        return None
    
    return largest


def draw_debug_overlay(
    frame: np.ndarray,
    contour: Optional[np.ndarray],
    prediction: int,
    latency_ms: float,
    buffer_state: list
) -> np.ndarray:
    """
    Draw debug information overlay on frame.
    
    Displays:
    - Contour outline (green)
    - Bounding box (blue)
    - Convex hull (cyan)
    - Prediction label and confidence
    - Latency measurement
    - Buffer state visualization
    """
    output = frame.copy()
    
    if contour is not None:
        # Draw contour (green)
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        
        # Draw bounding box (blue)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw convex hull (cyan)
        hull = cv2.convexHull(contour)
        cv2.drawContours(output, [hull], -1, (255, 255, 0), 1)
    
    # Prediction label
    gesture_name = GESTURE_CLASSES.get(prediction, "Unknown")
    cv2.putText(
        output, f"Gesture: {gesture_name}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    
    # Latency display
    color = (0, 255, 0) if latency_ms < 20 else (0, 165, 255) if latency_ms < 50 else (0, 0, 255)
    cv2.putText(
        output, f"Latency: {latency_ms:.1f}ms", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    
    # Buffer state (last 5 predictions)
    buffer_str = "Buffer: [" + ", ".join(str(p) for p in buffer_state) + "]"
    cv2.putText(
        output, buffer_str, (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    return output
