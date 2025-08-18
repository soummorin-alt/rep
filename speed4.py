#!/usr/bin/env python3
"""
Comprehensive Monocular Vehicle Speed Estimation Pipeline
Based on Dubská & Herout vanishing point detection and geometric scale recovery
Requires only one known vehicle height for calibration
"""

import cv2
import numpy as np
import threading
import queue
import time
import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies for enhanced functionality
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available. Using OpenCV cascade classifier for vehicle detection.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Audio alerts disabled.")

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]], dtype=np.float32)

@dataclass
class VehicleTrack:
    """Vehicle tracking information"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    world_positions: List[np.ndarray]
    timestamps: List[float]
    speeds: List[float]
    features: List[np.ndarray]
    kalman_filter: Optional[Any] = None

class DiamondSpaceVPDetector:
    """
    Vanishing point detector using diamond space accumulation
    Based on Dubská & Herout paper
    """
    
    def __init__(self, resolution: int = 512, d: float = 1.0, D: float = 1.0, mu: float = 1.0):
        self.resolution = resolution
        self.d = d  # Distance between parallel axes in first space
        self.D = D  # Distance between parallel axes in second space  
        self.mu = mu  # Image normalization factor
        self.accumulator = np.zeros((resolution, resolution), dtype=np.float32)
        # Store which transformation was used for each accumulator cell
        self.transform_map = np.full((resolution, resolution), "", dtype='U2')
        
    def normalize_coordinates(self, points: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Normalize image coordinates to [-mu, mu] range"""
        h, w = img_shape[:2]
        normalized = np.zeros_like(points, dtype=np.float32)
        normalized[:, 0] = (2 * points[:, 0] / (w - 1) - 1) * self.mu  # x -> u
        normalized[:, 1] = (2 * points[:, 1] / (h - 1) - 1) * self.mu  # y -> v
        return normalized
    
    def diamond_space_mapping(self, points: np.ndarray) -> np.ndarray:
        """
        Map points to diamond space using the four transformations from Eq. (5)
        Returns: array of diamond space coordinates for each transformation
        """
        u, v = points[:, 0], points[:, 1]
        w = np.ones_like(u)
        
        # Four cascaded transformations from Eq. (5)
        # SS: [−dDw, −dx, −x + y − dw]
        ss = np.column_stack([
            -self.d * self.D * w,
            -self.d * u,
            -u + v - self.d * w
        ])
        
        # ST: [−dDw, −dx, −x + y + dw] 
        st = np.column_stack([
            -self.d * self.D * w,
            -self.d * u,
            -u + v + self.d * w
        ])
        
        # TS: [−dDw, −dx, x + y − dw]
        ts = np.column_stack([
            -self.d * self.D * w,
            -self.d * u,
            u + v - self.d * w
        ])
        
        # TT: [−dDw, −dx, x + y + dw]
        tt = np.column_stack([
            -self.d * self.D * w,
            -self.d * u,
            u + v + self.d * w
        ])
        
        return np.array([ss, st, ts, tt])
    
    def project_to_accumulator(self, diamond_coords: np.ndarray) -> List[Tuple[int, int]]:
        """Project diamond space coordinates to accumulator indices"""
        indices = []
        for coords in diamond_coords:
            # Normalize to [0, 1] range for accumulator indexing
            # Using the second and third coordinates (p, q space)
            valid_mask = coords[:, 0] != 0  # Avoid division by zero
            
            if np.any(valid_mask):
                p = coords[valid_mask, 1] / coords[valid_mask, 0]  # -dx / (-dDw)
                q = coords[valid_mask, 2] / coords[valid_mask, 0]  # third_coord / (-dDw)
                
                # Map to accumulator space [0, resolution-1]
                # Determine appropriate scaling based on expected range
                scale_factor = self.resolution / (4 * self.mu)  # Adjust based on normalization
                
                acc_p = ((p + 2 * self.mu) * scale_factor).astype(np.int32)
                acc_q = ((q + 2 * self.mu) * scale_factor).astype(np.int32)
                
                # Clip to valid accumulator range
                acc_p = np.clip(acc_p, 0, self.resolution - 1)
                acc_q = np.clip(acc_q, 0, self.resolution - 1)
                
                indices.extend(list(zip(acc_p, acc_q)))
        
        return indices
    
    def accumulate_edgelets(self, edgelets: np.ndarray, img_shape: Tuple[int, int]):
        """Accumulate edgelets in diamond space"""
        self.accumulator.fill(0)
        self.transform_map.fill("")
        
        if len(edgelets) == 0:
            return
            
        # Normalize coordinates to [-mu, mu] range
        h, w = img_shape[:2]
        u = (2 * edgelets[:, 0] / (w - 1) - 1) * self.mu  
        v = (2 * edgelets[:, 1] / (h - 1) - 1) * self.mu
        w_coord = np.ones_like(u)
        
        # Apply four diamond space transformations
        transforms = {
            'SS': (-self.d * self.D * w_coord, -self.d * u, -u + v - self.d * w_coord),
            'ST': (-self.d * self.D * w_coord, -self.d * u, -u + v + self.d * w_coord),
            'TS': (-self.d * self.D * w_coord, -self.d * u,  u + v - self.d * w_coord),
            'TT': (-self.d * self.D * w_coord, -self.d * u,  u + v + self.d * w_coord)
        }
        
        # Accumulate votes for each transformation
        for transform_name, (coord0, coord1, coord2) in transforms.items():
            # Convert to accumulator indices
            valid_mask = np.abs(coord0) > 1e-10  # Avoid division by zero
            
            if np.any(valid_mask):
                p = coord1[valid_mask] / coord0[valid_mask]  
                q = coord2[valid_mask] / coord0[valid_mask]
                
                # Map to accumulator coordinates [0, resolution-1]
                # Determine bounds based on expected p,q range
                p_range = 4 * self.mu  # Expected range for p
                q_range = 4 * self.mu  # Expected range for q
                
                acc_p = ((p + p_range/2) / p_range * (self.resolution - 1)).astype(np.int32)
                acc_q = ((q + q_range/2) / q_range * (self.resolution - 1)).astype(np.int32)
                
                # Clip to valid range
                acc_p = np.clip(acc_p, 0, self.resolution - 1)
                acc_q = np.clip(acc_q, 0, self.resolution - 1)
                
                # Vote in accumulator
                for i in range(len(acc_p)):
                    self.accumulator[acc_q[i], acc_p[i]] += 1.0
                    self.transform_map[acc_q[i], acc_p[i]] = transform_name
    
    def find_vanishing_points(self, num_vps: int = 3) -> List[Tuple[np.ndarray, float]]:
        """Find vanishing points by detecting peaks in accumulator"""
        vps = []
        temp_acc = self.accumulator.copy()
        temp_transform_map = self.transform_map.copy()
        
        for _ in range(num_vps):
            # Find peak
            peak_idx = np.unravel_index(np.argmax(temp_acc), temp_acc.shape)
            peak_value = temp_acc[peak_idx]
            
            if peak_value < 1.0:  # Minimum threshold
                break
            
            # Get transformation type used at this peak
            transform_type = temp_transform_map[peak_idx]
            
            if transform_type == "":
                break
                
            # Convert accumulator indices back to (p,q) coordinates
            q_acc, p_acc = peak_idx
            
            # Map back to diamond space coordinates
            p_range = 4 * self.mu
            q_range = 4 * self.mu
            
            p = (p_acc / (self.resolution - 1) * p_range) - p_range/2
            q = (q_acc / (self.resolution - 1) * q_range) - q_range/2
            
            # Apply correct inverse mapping based on transform type
            if transform_type == "SS":
                x = self.D * q
                y = self.d * p + self.D * q - self.d * self.D
                w = 1.0
            elif transform_type == "ST":
                x = self.D * q
                y = self.d * p + self.D * q + self.d * self.D
                w = 1.0
            elif transform_type == "TS":
                x = self.D * q
                y = -self.d * p + self.D * q - self.d * self.D
                w = 1.0
            else: #"TT":
                x = self.D * q
                y = -self.d * p + self.D * q + self.d * self.D
                w = 1.0
        
            
            # Form vanishing point in homogeneous coordinates
            vp = np.array([x, y, w])
            vp = vp / vp[2]
            
            # Convert back to image coordinates (denormalize)
            vp_img = np.array([vp[0]/self.mu, vp[1]/self.mu, 1.0])

            vps.append((vp_img, peak_value))
            
            # Remove peak region to find next VP
            mask_size = max(10, self.resolution // 20)
            y_start = max(0, peak_idx[0] - mask_size)
            y_end = min(temp_acc.shape[0], peak_idx[0] + mask_size + 1)
            x_start = max(0, peak_idx[1] - mask_size)
            x_end = min(temp_acc.shape[1], peak_idx[1] + mask_size + 1)
            
            temp_acc[y_start:y_end, x_start:x_end] = 0
        
        return vps

class KalmanFilter:
    """Kalman filter for velocity smoothing"""
    
    def __init__(self, dt: float = 1.0/30.0):
        self.dt = dt
        self.initialized = False
        
        # State: [X, Z, vX, vZ]
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000  # High initial uncertainty
        
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix (observe velocities)
        self.H = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        # Process noise
        q = 0.1
        self.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                          [0, dt**4/4, 0, dt**3/2],
                          [dt**3/2, 0, dt**2, 0],
                          [0, dt**3/2, 0, dt**2]], dtype=np.float32) * q
        
        # Measurement noise
        self.R = np.eye(2, dtype=np.float32) * 10.0
    
    def predict(self):
        """Predict step"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """Update step with velocity measurement [vX, vZ]"""
        if not self.initialized:
            self.state[2:] = measurement
            self.initialized = True
            return
        
        # Innovation
        y = measurement - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P
    
    def get_velocity(self) -> np.ndarray:
        """Get filtered velocity [vX, vZ]"""
        return self.state[2:].copy()

class VehicleSpeedEstimator:
    """Main vehicle speed estimation pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.vp_detector = DiamondSpaceVPDetector(
            resolution=config.get('diamond_resolution', 512),
            mu=config.get('image_normalization', 1.0)
        )
        
        # Camera parameters (will be computed from VP detection)
        self.camera_intrinsics: Optional[CameraIntrinsics] = None
        self.rotation_matrix: Optional[np.ndarray] = None
        self.homography_ground: Optional[np.ndarray] = None
        self.scale_factor: Optional[float] = None
        
        # Vehicle detection
        if YOLO_AVAILABLE:
            self.vehicle_detector = YOLO('yolov8n.pt')
        else:
            # Fallback to OpenCV cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
            if Path(cascade_path).exists():
                self.vehicle_detector = cv2.CascadeClassifier(cascade_path)
            else:
                self.vehicle_detector = None
                print("Warning: No vehicle detector available")
        
        # Tracking
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        self.max_track_age = config.get('max_track_age', 30)
        
        # Feature tracking parameters
        self.feature_params = dict(
            maxCorners=20,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Speed parameters
        self.speed_threshold = config.get('speed_threshold', 50)  # km/h
        self.reference_height = config.get('reference_vehicle_height', 1.5)  # meters
        
        # Calibration
        self.calibration_frames = 0
        self.recalibration_interval = config.get('recalibration_interval', 30)
        
        # Audio alerts
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=100)
        self.processing_active = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'vehicles_detected': 0,
            'speed_violations': 0
        }
    
    def extract_edgelets(self, image: np.ndarray) -> np.ndarray:
        """Extract edgelets from image using Canny edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find edge pixels
        edge_pixels = np.where(edges > 0)
        
        if len(edge_pixels[0]) == 0:
            return np.array([]).reshape(0, 2)
        
        # Compute gradients for orientation
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        
        # Create edgelets (center points of short line segments)
        edgelets = []
        edgelet_length = 5
        
        for y, x in zip(edge_pixels[0], edge_pixels[1]):
            if 0 <= y < grad_y.shape[0] and 0 <= x < grad_x.shape[1]:
                # Gradient orientation
                theta = np.arctan2(grad_y[y, x], grad_x[y, x])
                
                # Edgelet endpoints
                dx = edgelet_length * np.cos(theta) / 2
                dy = edgelet_length * np.sin(theta) / 2
                
                edgelets.append([x, y])  # Store center point
        
        return np.array(edgelets, dtype=np.float32) if edgelets else np.array([]).reshape(0, 2)
    
    def compute_camera_intrinsics(self, vanishing_points: List[Tuple[np.ndarray, float]], img_shape: Tuple[int, int]) -> bool:
        """Compute camera intrinsics from three vanishing points via nonlinear orthogonality solve."""
        if len(vanishing_points) < 3:
            return False
        h, w = img_shape[:2]
        (u1, v1), _ = vanishing_points[0]
        (u2, v2), _ = vanishing_points[1]
        (u3, v3), _ = vanishing_points[2]

        from scipy.optimize import least_squares

        def residuals(vars):
            cx, cy, f2 = vars
            r1 = (u1 - cx)*(u2 - cx) + (v1 - cy)*(v2 - cy) + f2
            r2 = (u1 - cx)*(u3 - cx) + (v1 - cy)*(v3 - cy) + f2
            r3 = (u2 - cx)*(u3 - cx) + (v2 - cy)*(v3 - cy) + f2
            return [r1, r2, r3]

        # Initial guess: principal point at image center, f^2 = max dimension^2
        init = [(w - 1)/2, (h - 1)/2, max(w, h)**2]
        sol = least_squares(residuals, init)
        cx, cy, f2 = sol.x
        f = np.sqrt(abs(f2))
        self.camera_intrinsics = CameraIntrinsics(f, f, cx, cy)

        # Compute rotation matrix from vanishing points
        self.compute_rotation_matrix(vanishing_points)
        return True


    def compute_ground_homography(self) -> bool:
        """Compute homography mapping the world ground plane (Y=0) to image plane."""
        if self.camera_intrinsics is None or self.rotation_matrix is None:
            return False

        K = self.camera_intrinsics.K
        R = self.rotation_matrix
        h_cam = self.config.get('camera_height', 1.7)  # meters

        # Use first two columns of R (X and Y world axes) and translation [0, h_cam, 0]
        t = np.array([0.0, h_cam, 0.0], dtype=np.float32)
        H_ground = K @ np.column_stack([R[:, 0], R[:, 1], t])

        self.homography_ground = H_ground
        return True
    
    def compute_rotation_matrix(self, vanishing_points: List[Tuple[np.ndarray, float]]):
        """Compute camera rotation matrix from vanishing points"""
        K = self.camera_intrinsics.K
        K_inv = np.linalg.inv(K)
        
        # Convert vanishing points to normalized coordinates
        dirs = []
        for vp, _ in vanishing_points[:3]:
            vp_norm = K_inv @ vp
            dirs.append(vp_norm / np.linalg.norm(vp_norm))
        
        # Orthogonalize using Gram-Schmidt
        if len(dirs) >= 2:
            d1 = dirs[0]
            d2 = dirs[1] - np.dot(dirs[1], d1) * d1
            d2 = d2 / np.linalg.norm(d2)
            d3 = np.cross(d1, d2)
            
            self.rotation_matrix = np.column_stack([d1, d2, d3])
    
    def calibrate_scale(self, reference_bbox: Tuple[int, int, int, int], 
                       reference_height: float) -> bool:
        """Calibrate metric scale using reference vehicle with least squares solver"""
        if self.camera_intrinsics is None or self.rotation_matrix is None:
            return False
        
        x1, y1, x2, y2 = reference_bbox
        
        # Use top and bottom center of bounding box
        u_top, v_top = (x1 + x2) / 2, y1
        u_bot, v_bot = (x1 + x2) / 2, y2
        
        K_inv = np.linalg.inv(self.camera_intrinsics.K)
        R = self.rotation_matrix
        
        # Back-project points to normalized camera coordinates
        p_bot = K_inv @ np.array([u_bot, v_bot, 1.0])  # K⁻¹ [u_b, v_b, 1]^T
        p_top = K_inv @ np.array([u_top, v_top, 1.0])  # K⁻¹ [u_t, v_t, 1]^T
        
        # Set up system: λ_bot * p_bot = R [X, 0, Z]^T
        #                λ_top * p_top = R [X, H, Z]^T  
        # Rearranged: λ_bot * p_bot - R @ [X, 0, Z]^T = 0
        #            λ_top * p_top - R @ [X, H, Z]^T = 0
        
        # Build 6x4 system matrix for [λ_bot, λ_top, X, Z]
        # Build 6×4 system for [λ_bot, λ_top, X, Z]:
        A = np.zeros((6,4)); b = np.zeros(6)
        # λ_bot * p_bot = R @ [X,0,Z]^T
        A[0:3,0] = p_bot
        A[0:3,2] = -R[:,0]
        A[0:3,3] = -R[:,2]
        # λ_top * p_top = R @ [X,H,Z]^T
        A[3:6,1] = p_top
        A[3:6,2] = -R[:,0]
        A[3:6,3] = -R[:,2]
        b[3:6] = R[:,1] * reference_height

        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        lam_bot, lam_top, X, Z = sol
        P_bot = lam_bot * p_bot
        P_top = lam_top * p_top
        world_height = np.linalg.norm(P_top - P_bot)
        if world_height > 1e-6:
           self.scale_factor = reference_height / world_height
           return True
        return False

    
    
    def detect_vehicles(self, image: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        detections = []
        if YOLO_AVAILABLE and hasattr(self, 'vehicle_detector'):
            # Instead of indexing, do:
            results = self.vehicle_detector(image, conf=0.4, classes=[2,5,7])
            # results is a Results object
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()
                conf = box.conf.cpu().numpy()
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.vehicle_detector is not None:
                cars = self.vehicle_detector.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in cars:
                    detections.append((x, y, x+w, y+h, 0.8))
        return detections
    
    def associate_detections_to_tracks(self, detections: List[Tuple[int, int, int, int, float]],
                                     timestamp: float) -> Dict[int, Tuple[int, int, int, int]]:
        """Associate detections to existing tracks using IoU"""
        associated = {}
        used_detections = set()
        
        # Calculate IoU between detections and existing tracks
        for track_id, track in self.tracks.items():
            if not track.bbox:
                continue
            
            best_iou = 0
            best_detection_idx = -1
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                iou = self.calculate_iou(track.bbox, detection[:4])
                
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_detection_idx = i
            
            if best_detection_idx >= 0:
                associated[track_id] = detections[best_detection_idx][:4]
                used_detections.add(best_detection_idx)
        
        # Create new tracks for unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detections and detection[4] > 0.6:  # High confidence threshold
                track_id = self.next_track_id
                self.next_track_id += 1
                
                new_track = VehicleTrack(
                    track_id=track_id,
                    bbox=detection[:4],
                    world_positions=[],
                    timestamps=[timestamp],
                    speeds=[],
                    features=[],
                    kalman_filter=KalmanFilter()
                )
                
                self.tracks[track_id] = new_track
                associated[track_id] = detection[:4]
        
        return associated
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def track_features_and_estimate_speed(self, current_frame: np.ndarray, 
                                        previous_frame: np.ndarray,
                                        track_id: int, bbox: Tuple[int, int, int, int],
                                        timestamp: float, dt: float) -> Optional[float]:
        """Track features within vehicle bbox and estimate speed"""
        if self.homography_ground is None or self.scale_factor is None:
            return None
        
        x1, y1, x2, y2 = bbox
        
        # Extract ROI
        roi_prev = previous_frame[y1:y2, x1:x2]
        roi_curr = current_frame[y1:y2, x1:x2]
        
        if roi_prev.size == 0 or roi_curr.size == 0:
            return None
        
        # Convert to grayscale
        gray_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY) if len(roi_prev.shape) == 3 else roi_prev
        gray_curr = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY) if len(roi_curr.shape) == 3 else roi_curr
        
        # Detect features
        features = cv2.goodFeaturesToTrack(gray_prev, **self.feature_params)
        
        if features is None or len(features) == 0:
            return None
        
        # Track features using optical flow
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, features, None, **self.lk_params
        )
        
        # Forward-backward consistency check
        back_features, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray_curr, gray_prev, new_features, None, **self.lk_params
        )
        
        # Filter good features
        good_mask = (status.flatten() == 1) & (status_back.flatten() == 1)
        if np.any(good_mask):
            fb_error = np.linalg.norm(features[good_mask] - back_features[good_mask], axis=2).flatten()
            good_mask[good_mask] = fb_error < 3.0
        
        if not np.any(good_mask):
            return None
        
        # Convert to image coordinates (add ROI offset)
        features_img = features[good_mask] + [x1, y1]
        new_features_img = new_features[good_mask] + [x1, y1]
        
        # Project to world coordinates
        world_displacements = []
        
        for (f1, f2) in zip(features_img, new_features_img):
            try:
                # Project to ground plane
                p1_homo = np.array([f1[0], f1[1], 1.0])
                p2_homo = np.array([f2[0], f2[1], 1.0])
                
                # Use homography to project to ground plane
                if self.homography_ground is not None:
                    H_inv = np.linalg.inv(self.homography_ground)
                    
                    world_p1 = H_inv @ p1_homo
                    world_p2 = H_inv @ p2_homo
                    
                    world_p1 = world_p1 / world_p1[2] if abs(world_p1[2]) > 1e-6 else world_p1
                    world_p2 = world_p2 / world_p2[2] if abs(world_p2[2]) > 1e-6 else world_p2
                    
                    # Apply scale factor
                    if self.scale_factor is not None:
                        displacement = (world_p2[:2] - world_p1[:2]) * self.scale_factor
                        world_displacements.append(displacement)
            
            except (np.linalg.LinAlgError, ZeroDivisionError):
                continue
        
        if not world_displacements:
            return None
        
        # Calculate speeds for each displacement
        speeds = []
        for displacement in world_displacements:
            speed_ms = np.linalg.norm(displacement) / dt  # m/s
            speed_kmh = speed_ms * 3.6  # km/h
            speeds.append(speed_kmh)
        
        # Use median speed to reduce noise
        if speeds:
            median_speed = np.median(speeds)
            
            # Update Kalman filter
            track = self.tracks.get(track_id)
            if track and track.kalman_filter:
                # Velocity in X and Z directions
                if len(world_displacements) > 0:
                    avg_displacement = np.mean(world_displacements, axis=0)
                    velocity = avg_displacement / dt
                    
                    track.kalman_filter.predict()
                    track.kalman_filter.update(velocity)
                    
                    # Get filtered velocity
                    filtered_velocity = track.kalman_filter.get_velocity()
                    filtered_speed = np.linalg.norm(filtered_velocity) * 3.6  # km/h
                    
                    return filtered_speed
            
            return median_speed
        
        return None
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     previous_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process a single frame and estimate vehicle speeds"""
        results = {
            'timestamp': timestamp,
            'vehicles': [],
            'calibrated': self.camera_intrinsics is not None,
            'vanishing_points': []
        }
        
        # Periodically recalibrate camera parameters
        if (self.calibration_frames % self.recalibration_interval == 0 or 
            self.camera_intrinsics is None):
            
            # Extract edgelets
            edgelets = self.extract_edgelets(frame)
            
            if len(edgelets) > 100:  # Minimum edgelets for reliable VP detection
                # Accumulate in diamond space
                self.vp_detector.accumulate_edgelets(edgelets, frame.shape)
                
                # Find vanishing points
                vanishing_points = self.vp_detector.find_vanishing_points(3)
                results['vanishing_points'] = [(vp.tolist(), score) for vp, score in vanishing_points]
                
                if len(vanishing_points) >= 2:
                    # Compute camera intrinsics
                    if self.compute_camera_intrinsics(vanishing_points, frame.shape):
                        # Compute ground homography
                        self.compute_ground_homography()
                        
                        # If this is the first calibration and no scale set, 
                        # we need user to provide reference vehicle
                        if self.scale_factor is None:
                            print("Camera calibrated. Provide reference vehicle for scale calibration.")
        
        self.calibration_frames += 1
        
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Associate detections to tracks
        associated = self.associate_detections_to_tracks(detections, timestamp)
        
        # Update tracks and estimate speeds
        for track_id, bbox in associated.items():
            track = self.tracks[track_id]
            track.bbox = bbox
            track.timestamps.append(timestamp)
            
            # Estimate speed if we have previous frame and calibration
            speed = None
            if (previous_frame is not None and 
                len(track.timestamps) >= 2 and
                self.camera_intrinsics is not None):
                
                dt = track.timestamps[-1] - track.timestamps[-2]
                if dt > 0:
                    speed = self.track_features_and_estimate_speed(
                        frame, previous_frame, track_id, bbox, timestamp, dt
                    )
            
            if speed is not None:
                track.speeds.append(speed)
                
                # Trigger alert if speed exceeds threshold
                if speed > self.speed_threshold:
                    self.trigger_speed_alert(track_id, speed)
                    self.stats['speed_violations'] += 1
            
            # Add to results
            vehicle_result = {
                'track_id': track_id,
                'bbox': bbox,
                'speed_kmh': speed,
                'timestamp': timestamp
            }
            results['vehicles'].append(vehicle_result)
        
        # Clean up old tracks
        self.cleanup_old_tracks(timestamp)
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['vehicles_detected'] += len(associated)
        
        return results
    
    def trigger_speed_alert(self, track_id: int, speed: float):
        """Trigger speed violation alert"""
        message = f"Vehicle {track_id} exceeding speed limit at {speed:.1f} km/h"
        print(f"ALERT: {message}")
        
        if TTS_AVAILABLE and hasattr(self, 'tts_engine'):
            try:
                self.tts_engine.say(f"Speed violation: {int(speed)} kilometers per hour")
                self.tts_engine.runAndWait()
            except:
                pass  # TTS failed, continue silently
    
    def cleanup_old_tracks(self, current_timestamp: float):
        """Remove old/inactive tracks"""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if (len(track.timestamps) == 0 or 
                current_timestamp - track.timestamps[-1] > self.max_track_age):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def annotate_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        # Draw vehicle detections and speeds
        for vehicle in results['vehicles']:
            track_id = vehicle['track_id']
            x1, y1, x2, y2 = vehicle['bbox']
            speed = vehicle.get('speed_kmh')
            
            # Draw bounding box
            color = (0, 255, 0) if speed is None else (0, 0, 255) if speed > self.speed_threshold else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and speed
            label = f"ID: {track_id}"
            if speed is not None:
                label += f" | {speed:.1f} km/h"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw calibration status
        status_text = "CALIBRATED" if results['calibrated'] else "CALIBRATING..."
        status_color = (0, 255, 0) if results['calibrated'] else (0, 255, 255)
        cv2.putText(annotated, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw statistics
        stats_text = f"Frames: {self.stats['frames_processed']} | Vehicles: {self.stats['vehicles_detected']} | Violations: {self.stats['speed_violations']}"
        cv2.putText(annotated, stats_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to CSV file"""
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'speed_kmh']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_result in results:
                timestamp = frame_result['timestamp']
                for vehicle in frame_result['vehicles']:
                    row = {
                        'timestamp': timestamp,
                        'track_id': vehicle['track_id'],
                        'bbox_x1': vehicle['bbox'][0],
                        'bbox_y1': vehicle['bbox'][1],
                        'bbox_x2': vehicle['bbox'][2],
                        'bbox_y2': vehicle['bbox'][3],
                        'speed_kmh': vehicle.get('speed_kmh', '')
                    }
                    writer.writerow(row)
    
    def calibrate_with_reference_vehicle(self, frame: np.ndarray, 
                                       bbox: Tuple[int, int, int, int],
                                       height_meters: float) -> bool:
        """Calibrate scale using a reference vehicle with known height"""
        if self.calibrate_scale(bbox, height_meters):
            print(f"Scale calibrated with reference vehicle height: {height_meters}m")
            return True
        return False

class RealTimeProcessor:
    """Real-time processing wrapper with threading"""
    
    def __init__(self, estimator: VehicleSpeedEstimator):
        self.estimator = estimator
        self.capture_thread = None
        self.process_thread = None
        self.display_thread = None
        self.running = False
        
        # Queues for threading
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        
    def start_camera_processing(self, camera_index: int = 0):
        """Start real-time camera processing"""
        self.running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_frames, args=(camera_index,))
        self.process_thread = threading.Thread(target=self._process_frames)
        self.display_thread = threading.Thread(target=self._display_results)
        
        self.capture_thread.start()
        self.process_thread.start() 
        self.display_thread.start()
        
        print("Started real-time processing. Press 'q' to quit, 'c' to calibrate scale.")
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process video file and return results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Frames: {frame_count}")
        
        results = []
        previous_frame = None
        frame_idx = 0
        
        # Setup output video writer if requested
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Process frame
                frame_result = self.estimator.process_frame(frame, timestamp, previous_frame)
                results.append(frame_result)
                
                # Annotate and save frame
                if out_writer:
                    annotated_frame = self.estimator.annotate_frame(frame, frame_result)
                    out_writer.write(annotated_frame)
                
                # Display progress
                if frame_idx % 30 == 0:  # Every second at 30fps
                    print(f"Processed frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
                
                previous_frame = frame
                frame_idx += 1
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
        
        return results
    
    def _capture_frames(self, camera_index: int):
        """Capture frames from camera"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                timestamp = time.time()
                
                try:
                    self.frame_queue.put((frame, timestamp), timeout=0.1)
                except queue.Full:
                    pass  # Skip frame if queue is full
        
        cap.release()
    
    def _process_frames(self):
        """Process captured frames"""
        previous_frame = None
        
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                result = self.estimator.process_frame(frame, timestamp, previous_frame)
                
                # Add annotated frame to result
                annotated_frame = self.estimator.annotate_frame(frame, result)
                result['annotated_frame'] = annotated_frame
                
                try:
                    self.result_queue.put(result, timeout=0.1)
                except queue.Full:
                    pass  # Skip if result queue is full
                
                previous_frame = frame
                
            except queue.Empty:
                continue
    
    def _display_results(self):
        """Display processed results"""
        calibration_mode = False
        reference_bbox = None
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)
                annotated_frame = result.get('annotated_frame')
                
                if annotated_frame is not None:
                    # Handle calibration mode
                    if calibration_mode:
                        cv2.putText(annotated_frame, "CALIBRATION MODE: Click and drag to select reference vehicle", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        if reference_bbox:
                            x1, y1, x2, y2 = reference_bbox
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(annotated_frame, "Press ENTER to confirm, ESC to cancel", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imshow('Vehicle Speed Estimation', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        self.running = False
                        break
                    elif key == ord('c'):
                        calibration_mode = True
                        print("Entering calibration mode. Click and drag to select reference vehicle.")
                    elif calibration_mode:
                        if key == 13:  # Enter key
                            if reference_bbox:
                                height = float(input("Enter reference vehicle height in meters: "))
                                if self.estimator.calibrate_with_reference_vehicle(
                                    annotated_frame, reference_bbox, height):
                                    print("Calibration successful!")
                                else:
                                    print("Calibration failed!")
                                calibration_mode = False
                                reference_bbox = None
                        elif key == 27:  # ESC key
                            calibration_mode = False
                            reference_bbox = None
                            print("Calibration cancelled.")
                
            except queue.Empty:
                continue
        
        cv2.destroyAllWindows()
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()
        if self.display_thread:
            self.display_thread.join()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monocular Vehicle Speed Estimation')
    parser.add_argument('--mode', choices=['camera', 'video'], default='camera',
                       help='Processing mode: camera or video file')
    parser.add_argument('--input', type=str, help='Input video file path (for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--csv', type=str, help='Output CSV file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--reference-height', type=float, default=1.5,
                       help='Reference vehicle height in meters')
    parser.add_argument('--speed-threshold', type=float, default=50,
                       help='Speed violation threshold in km/h')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'diamond_resolution': 512,
        'image_normalization': 1.0,
        'max_track_age': 2.0,
        'recalibration_interval': 30,
        'reference_vehicle_height': args.reference_height,
        'speed_threshold': args.speed_threshold,
        'camera_height': 1.7  # meters
    }
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        except FileNotFoundError:
            print(f"Config file not found: {args.config}")
    
    # Initialize estimator
    estimator = VehicleSpeedEstimator(config)
    
    if args.mode == 'camera':
        # Real-time camera processing
        processor = RealTimeProcessor(estimator)
        
        try:
            processor.start_camera_processing(args.camera)
            
            # Keep main thread alive
            while processor.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            processor.stop()
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: Input video file required for video mode")
            return
        
        processor = RealTimeProcessor(estimator)
        
        try:
            print(f"Processing video file: {args.input}")
            results = processor.process_video_file(args.input, args.output)
            
            # Save results to CSV
            if args.csv:
                estimator.save_results_to_csv(results, args.csv)
                print(f"Results saved to: {args.csv}")
            
            print(f"Processing complete. Processed {len(results)} frames.")
            print(f"Statistics: {estimator.stats}")
            
        except Exception as e:
            print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()