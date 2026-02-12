"""
GestuBot - Real-Time Inference Engine

Live gesture recognition with keyboard control.
Target latency: <20ms per frame.

Usage:
    python inference.py              # normal mode
    python inference.py --benchmark  # headless latency benchmark

Controls:
    q: Quit

Gesture Mappings:
    0: Fist       -> Release all keys (stop)
    1: Open Palm  -> W (forward)
    2: Point Left -> A (left)
    3: Point Right -> D (right)
    4: V-Sign     -> S (reverse)
    5: Background  -> No action
"""

import cv2
import numpy as np
import joblib
import time
import os
import argparse
from collections import deque
from statistics import mode
from typing import Optional, Tuple

# Import shared utilities
from utils import (
    extract_features,
    preprocess_frame,
    find_largest_contour,
    draw_debug_overlay,
    GESTURE_CLASSES,
    GESTURE_KEYS
)

# Import WebSocket server for 3D visualization
try:
    from ws_server import start_websocket_server, send_gesture, WEBSOCKETS_AVAILABLE
    WS_ENABLED = WEBSOCKETS_AVAILABLE
except ImportError:
    WS_ENABLED = False
    print("[INFO] WebSocket server not available (ws_server.py not found)")

# Keyboard control (import with safety check)
try:
    import pyautogui
    # CRITICAL SAFETY: Enable failsafe (move mouse to corner to kill script)
    pyautogui.FAILSAFE = True
    # Reduce pyautogui pause for lower latency
    pyautogui.PAUSE = 0.0
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    print("[WARNING] pyautogui not installed. Keyboard control disabled.")
    print("Install with: pip install pyautogui")
    PYAUTOGUI_AVAILABLE = False



# --- Configuration ---

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_svm.joblib')

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# HSV defaults (should match what was used during training)
DEFAULT_HSV_LOWER = (0, 30, 60)
DEFAULT_HSV_UPPER = (20, 150, 255)

# Debouncing: 5-frame rolling buffer for mode filtering.
# Higher = more stable but slower to respond.
BUFFER_SIZE = 5


# --- Rolling Buffer Debouncer ---

class PredictionBuffer:
    """
    Rolling buffer that smooths predictions via mode filtering.
    Prevents jittery outputs from frame-to-frame noise.
    At 30 FPS with buffer_size=5, this adds ~166ms of temporal smoothing.
    """
    
    def __init__(self, size: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=size)
        self.size = size
    
    def add(self, prediction: int) -> int:
        """
        Add prediction to buffer and return filtered output.
        
        Args:
            prediction: Raw classifier output (0-5)
            
        Returns:
            Filtered prediction (mode of buffer, or latest if no clear mode)
        """
        self.buffer.append(prediction)
        
        # Need at least half the buffer filled for mode filtering
        if len(self.buffer) < self.size // 2:
            return prediction
        
        try:
            return mode(self.buffer)  # most common prediction wins
        except:
            # If no single mode (tie), use latest prediction
            return prediction
    
    def clear(self) -> None:
        """Clear the buffer (useful for state reset)."""
        self.buffer.clear()
    
    def get_state(self) -> list:
        """Get current buffer contents for visualization."""
        return list(self.buffer)


# --- Latency Profiler ---

class LatencyProfiler:
    """
    Collects per-frame latency measurements and computes summary statistics.
    Tracks total pipeline time plus per-stage breakdown (preprocess, contour,
    classify, debounce). Uses a rolling window so stats reflect recent perf.
    """
    
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.total_latencies = deque(maxlen=window_size)
        self.stage_latencies = {
            'preprocess': deque(maxlen=window_size),
            'contour': deque(maxlen=window_size),
            'classify': deque(maxlen=window_size),
            'debounce': deque(maxlen=window_size),
        }
        self.frame_count = 0
    
    def record(self, total_ms: float, stage_times: dict) -> None:
        """
        Record timing for one frame.
        
        Args:
            total_ms: Total pipeline latency in milliseconds
            stage_times: Dict mapping stage name -> latency in ms
        """
        self.total_latencies.append(total_ms)
        for stage, ms in stage_times.items():
            if stage in self.stage_latencies:
                self.stage_latencies[stage].append(ms)
        self.frame_count += 1
    
    def summary(self) -> dict:
        """
        Compute summary statistics over the recorded window.
        
        Returns:
            Dict with total and per-stage stats (mean, median, p95, p99, max, min, std)
        """
        if not self.total_latencies:
            return {}
        
        arr = np.array(self.total_latencies)
        result = {
            'frame_count': self.frame_count,
            'window_size': len(arr),
            'total': self._compute_stats(arr),
            'stages': {}
        }
        
        for stage, dq in self.stage_latencies.items():
            if dq:
                result['stages'][stage] = self._compute_stats(np.array(dq))
        
        return result
    
    @staticmethod
    def _compute_stats(arr: np.ndarray) -> dict:
        """Compute descriptive statistics for a latency array."""
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }
    
    def print_report(self) -> None:
        """
        Print a formatted latency report to stdout.
        """
        stats = self.summary()
        if not stats:
            print("[PROFILER] No frames recorded.")
            return
        
        t = stats['total']
        print("\n" + "═" * 60)
        print(f"  GestuBot Latency Report ({stats['frame_count']} frames profiled)")
        print("═" * 60)
        print(f"  Total pipeline:")
        print(f"    Mean: {t['mean']:.2f}ms | Median: {t['median']:.2f}ms")
        print(f"    p95:  {t['p95']:.2f}ms | p99:    {t['p99']:.2f}ms")
        print(f"    Min:  {t['min']:.2f}ms | Max:    {t['max']:.2f}ms | Std: {t['std']:.2f}ms")
        
        # Per-stage breakdown
        if stats['stages']:
            print(f"\n  Per-stage breakdown (mean / p95):")
            for stage, s in stats['stages'].items():
                print(f"    {stage:>12s}: {s['mean']:6.2f}ms  / {s['p95']:6.2f}ms")
        
        # Verdict vs target
        target = 20.0
        if t['p95'] < target:
            verdict = f"✔ PASS — p95 ({t['p95']:.1f}ms) < {target:.0f}ms target"
        elif t['mean'] < target:
            verdict = f"⚠ MARGINAL — mean ({t['mean']:.1f}ms) < {target:.0f}ms but p95 ({t['p95']:.1f}ms) exceeds"
        else:
            verdict = f"✘ FAIL — mean ({t['mean']:.1f}ms) ≥ {target:.0f}ms target"
        
        print(f"\n  Verdict: {verdict}")
        print("═" * 60 + "\n")
    
    def get_state(self) -> list:
        """Get current buffer contents for visualization (compat shim)."""
        return list(self.total_latencies)


# --- Keyboard Controller ---

class KeyboardController:
    """
    Maps gestures to keyboard inputs. Tracks which key is currently
    held down so we only send keyDown/keyUp on actual state changes.
    """
    
    def __init__(self):
        self.current_key: Optional[str] = None
        self.active = PYAUTOGUI_AVAILABLE
    
    def update(self, gesture_class: int) -> str:
        """
        Update keyboard state based on gesture.
        
        Args:
            gesture_class: Predicted gesture (0-5)
            
        Returns:
            Action taken ("press W", "release", "hold W", etc.)
        """
        if not self.active:
            return "disabled"
        
        target_key = GESTURE_KEYS.get(gesture_class)
        
        # Case 1: No change needed
        if target_key == self.current_key:
            if target_key:
                return f"hold {target_key.upper()}"
            else:
                return "idle"
        
        # Case 2: Release current key if one is held
        if self.current_key is not None:
            try:
                pyautogui.keyUp(self.current_key)
            except Exception as e:
                print(f"[WARN] keyUp failed: {e}")
        
        # Case 3: Press new key if target is not None
        if target_key is not None:
            try:
                pyautogui.keyDown(target_key)
                self.current_key = target_key
                return f"press {target_key.upper()}"
            except Exception as e:
                print(f"[WARN] keyDown failed: {e}")
                self.current_key = None
                return "error"
        else:
            self.current_key = None
            return "release"
    
    def release_all(self) -> None:
        """Release any held keys (cleanup on exit)."""
        if self.active and self.current_key:
            try:
                pyautogui.keyUp(self.current_key)
            except:
                pass
        self.current_key = None


# --- Inference Engine ---

class GestureBotInference:
    """Main inference engine. Ties together the vision pipeline, SVM
    classifier, prediction buffer, keyboard control, and WebSocket output."""
    
    def __init__(self):
        self.model = self._load_model()
        self.buffer = PredictionBuffer(BUFFER_SIZE)
        self.keyboard = KeyboardController()
        self.profiler = LatencyProfiler(window_size=500)
        
        # HSV thresholds (could load from config if saved during training)
        self.hsv_lower = DEFAULT_HSV_LOWER
        self.hsv_upper = DEFAULT_HSV_UPPER
        
        # Start WebSocket server for 3D visualization
        self.ws_thread = None
        self.last_broadcast_gesture = -1  # Track to avoid duplicate broadcasts
        if WS_ENABLED:
            self.ws_thread = start_websocket_server()
            if self.ws_thread:
                print("[INFO] WebSocket server started on ws://localhost:8765")
                print("[INFO] Open web/robot_arm.html in browser for 3D visualization")
    
    def _load_model(self):
        """Load trained SVM model."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}\n"
                "Please run trainer.py first to train the model."
            )
        
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Loaded model from {MODEL_PATH}")
        return model
    
    def process_frame(self, frame: np.ndarray) -> Tuple[int, float, Optional[np.ndarray], dict]:
        """
        Process a single frame through the full pipeline.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (prediction, latency_ms, contour, stage_times)
            stage_times is a dict mapping stage name -> latency in ms
        """
        stage_times = {}
        pipeline_start = time.perf_counter()
        
        # 1. Preprocessing
        t0 = time.perf_counter()
        mask, _ = preprocess_frame(frame, self.hsv_lower, self.hsv_upper)
        stage_times['preprocess'] = (time.perf_counter() - t0) * 1000
        
        # 2. Contour detection
        t0 = time.perf_counter()
        contour = find_largest_contour(mask)
        stage_times['contour'] = (time.perf_counter() - t0) * 1000
        
        # 3. Classification
        t0 = time.perf_counter()
        if contour is None:
            # No hand detected -> Class 5 (Background)
            raw_prediction = 5
        else:
            features = extract_features(contour)
            if features is None:
                raw_prediction = 5
            else:
                features_2d = features.reshape(1, -1)
                raw_prediction = int(self.model.predict(features_2d)[0])
        stage_times['classify'] = (time.perf_counter() - t0) * 1000
        
        # 4. Debouncing (mode filtering over buffer)
        t0 = time.perf_counter()
        filtered_prediction = self.buffer.add(raw_prediction)
        stage_times['debounce'] = (time.perf_counter() - t0) * 1000
        
        # Total pipeline latency
        latency_ms = (time.perf_counter() - pipeline_start) * 1000
        
        return filtered_prediction, latency_ms, contour, stage_times
    
    def run(self) -> None:
        """Main inference loop."""
        print("\n" + "="*60)
        print("GestuBot Real-Time Inference")
        print("="*60)
        print("\nGesture Mappings:")
        for cls_id, cls_name in GESTURE_CLASSES.items():
            key = GESTURE_KEYS.get(cls_id)
            key_str = key.upper() if key else "None"
            print(f"  {cls_id}: {cls_name} -> {key_str}")
        print("\nControls:")
        print("  q: Quit")
        print("\nSAFETY: Move mouse to corner to kill script")
        print("="*60 + "\n")
        
        # Initialize camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Try to set higher FPS
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return
        
        print("[INFO] Camera opened. Starting inference...")
        print("[INFO] Target latency: <20ms per frame\n")
        
        # Performance tracking
        latency_history = deque(maxlen=30)  # Rolling average over 30 frames
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                # Mirror for intuitive control
                frame = cv2.flip(frame, 1)
                
                # Process frame
                prediction, latency_ms, contour, stage_times = self.process_frame(frame)
                
                # Record in profiler
                self.profiler.record(latency_ms, stage_times)
                
                # Update keyboard
                action = self.keyboard.update(prediction)
                
                # Broadcast to 3D visualization (only on change to reduce traffic)
                if WS_ENABLED and prediction != self.last_broadcast_gesture:
                    gesture_name = GESTURE_CLASSES.get(prediction, "Unknown")
                    send_gesture(prediction, gesture_name, action, latency_ms)
                    self.last_broadcast_gesture = prediction
                
                # Track latency (rolling HUD average)
                latency_history.append(latency_ms)
                avg_latency = sum(latency_history) / len(latency_history)
                
                # Draw debug overlay
                display = draw_debug_overlay(
                    frame,
                    contour,
                    prediction,
                    latency_ms,
                    self.buffer.get_state()
                )
                
                # Additional info overlay
                cv2.putText(
                    display, f"Action: {action}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                cv2.putText(
                    display, f"Avg Latency: {avg_latency:.1f}ms", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )
                
                # Per-stage timing on HUD
                stage_y = 175
                for stage_name, stage_ms in stage_times.items():
                    cv2.putText(
                        display, f"  {stage_name}: {stage_ms:.1f}ms", (10, stage_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
                    )
                    stage_y += 18
                
                # Performance indicator
                if avg_latency < 20:
                    perf_text = "PERFORMANCE: OPTIMAL"
                    perf_color = (0, 255, 0)
                elif avg_latency < 50:
                    perf_text = "PERFORMANCE: ACCEPTABLE"
                    perf_color = (0, 165, 255)
                else:
                    perf_text = "PERFORMANCE: DEGRADED"
                    perf_color = (0, 0, 255)
                
                cv2.putText(
                    display, perf_text, (FRAME_WIDTH - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 2
                )
                
                # Show frame
                cv2.imshow('GestuBot - Inference', display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            # Cleanup
            self.keyboard.release_all()
            cap.release()
            cv2.destroyAllWindows()
            
            # Print latency profiling report on exit
            self.profiler.print_report()
            print("[INFO] Inference stopped. Keys released.")


def run_benchmark(num_frames: int = 200) -> None:
    """
    Benchmark mode: capture N frames, classify them, and print a latency report.
    
    No GUI, no keyboard control, no WebSocket — pure pipeline profiling.
    Run with: python inference.py --benchmark [--frames N]
    """
    print("\n" + "=" * 60)
    print("GestuBot Latency Benchmark")
    print("=" * 60)
    print(f"\nFrames to capture: {num_frames}")
    print("Mode: headless (no GUI / keyboard / WebSocket)\n")
    
    # Load model
    model = joblib.load(MODEL_PATH)
    buffer = PredictionBuffer(BUFFER_SIZE)
    profiler = LatencyProfiler(window_size=num_frames)
    
    hsv_lower = DEFAULT_HSV_LOWER
    hsv_upper = DEFAULT_HSV_UPPER
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera for benchmark.")
        return
    
    print("[INFO] Camera opened. Running benchmark...\n")
    
    try:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Failed to read frame {i}")
                break
            
            frame = cv2.flip(frame, 1)
            stage_times = {}
            pipeline_start = time.perf_counter()
            
            # Stage 1: Preprocessing
            t0 = time.perf_counter()
            mask, _ = preprocess_frame(frame, hsv_lower, hsv_upper)
            stage_times['preprocess'] = (time.perf_counter() - t0) * 1000
            
            # Stage 2: Contour detection
            t0 = time.perf_counter()
            contour = find_largest_contour(mask)
            stage_times['contour'] = (time.perf_counter() - t0) * 1000
            
            # Stage 3: Classification
            t0 = time.perf_counter()
            if contour is None:
                raw_prediction = 5
            else:
                features = extract_features(contour)
                if features is None:
                    raw_prediction = 5
                else:
                    features_2d = features.reshape(1, -1)
                    raw_prediction = int(model.predict(features_2d)[0])
            stage_times['classify'] = (time.perf_counter() - t0) * 1000
            
            # Stage 4: Debouncing
            t0 = time.perf_counter()
            _ = buffer.add(raw_prediction)
            stage_times['debounce'] = (time.perf_counter() - t0) * 1000
            
            latency_ms = (time.perf_counter() - pipeline_start) * 1000
            profiler.record(latency_ms, stage_times)
            
            # Progress indicator every 50 frames
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{num_frames}] last frame: {latency_ms:.2f}ms")
    
    finally:
        cap.release()
    
    # Print the full latency report
    profiler.print_report()


def main():
    """Entry point with CLI argument support."""
    parser = argparse.ArgumentParser(description="GestuBot Real-Time Inference Engine")
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run latency benchmark (no GUI/keyboard/WebSocket)'
    )
    parser.add_argument(
        '--frames', type=int, default=200,
        help='Number of frames to capture in benchmark mode (default: 200)'
    )
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            run_benchmark(num_frames=args.frames)
        else:
            engine = GestureBotInference()
            engine.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise


if __name__ == '__main__':
    main()

