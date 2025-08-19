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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')
    
def filter_isolated_edgelets(edgelets: np.ndarray, min_neighbors: int = 2, radius: float = 5.0) -> np.ndarray:
        """Remove isolated edgelets that don't form coherent line structures"""
        if len(edgelets) < min_neighbors+1:
            return edgelets.copy()
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min_neighbors+1, radius=radius).fit(edgelets)
        distances, indices = nbrs.radius_neighbors(edgelets, radius=radius)
        
        # Keep edgelets that have enough neighbors
        filtered_edgelets = []
        for i, neighbors in enumerate(indices):
            if len(neighbors) >= min_neighbors + 1:  # +1 because it includes itself
                filtered_edgelets.append(edgelets[i])
        
        return np.array(filtered_edgelets, dtype=np.float32) if filtered_edgelets else np.array([]).reshape(0, 2)
# Optional dependencies for enhanced functionality
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available; vehicle detection disabled.")

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
        self.transform_map = np.full((resolution, resolution), "", dtype='U2')

    def normalize_coordinates(self, points: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Normalize image coordinates to [-mu, mu] range"""
        h, w = img_shape[:2]
        normalized = np.zeros_like(points, dtype=np.float32)
        normalized[:, 0] = (2 * points[:, 0] / (w - 1) - 1) * self.mu
        normalized[:, 1] = (2 * points[:, 1] / (h - 1) - 1) * self.mu
        return normalized

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

        # Vectorized accumulation into accumulator
        p_range = 4 * self.mu
        q_range = 4 * self.mu
        for transform_name, (coord0, coord1, coord2) in transforms.items():
            valid_mask = np.abs(coord0) > 1e-10
            if not np.any(valid_mask):
                continue
            p = coord1[valid_mask] / coord0[valid_mask]
            q = coord2[valid_mask] / coord0[valid_mask]
            acc_p = ((p + p_range / 2) / p_range * (self.resolution - 1)).astype(np.int32)
            acc_q = ((q + q_range / 2) / q_range * (self.resolution - 1)).astype(np.int32)
            acc_p = np.clip(acc_p, 0, self.resolution - 1)
            acc_q = np.clip(acc_q, 0, self.resolution - 1)
            np.add.at(self.accumulator, (acc_q, acc_p), 1.0)
            self.transform_map[acc_q, acc_p] = transform_name

    def find_vanishing_points(self, num_vps: int = 3) -> List[Tuple[np.ndarray, float]]:
        """Find vanishing points by detecting peaks in accumulator with spatial diversity enforcement"""
        vps = []
        temp_acc = self.accumulator.copy()
        temp_transform_map = self.transform_map.copy()
        
        # Minimum distance between vanishing points (in accumulator space)
        min_vp_distance = 0  # pixels in accumulator space
        
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
            
            # Check spatial diversity with previously selected VPs
            if len(vps) > 0:
                min_dist_to_existing = float('inf')
                for existing_vp, _ in vps:
                    dist = np.linalg.norm(vp_img[:2] - existing_vp[:2])
                    min_dist_to_existing = min(min_dist_to_existing, dist)
                
                if min_dist_to_existing < min_vp_distance:
                    print(f"Rejecting VP at ({vp_img[0]:.1f}, {vp_img[1]:.1f}) - too close to existing VPs (min dist: {min_dist_to_existing:.1f} px)")
                    # Suppress this peak and continue to next
                    temp_acc[peak_idx] = 0
                    continue

            vps.append((vp_img, peak_value))
            
            # Remove peak region to find next VP (wider suppression for better diversity)
            mask_size = max(20, self.resolution // 15)  # Increased from resolution // 20
            y_start = max(0, peak_idx[0] - mask_size)
            y_end = min(temp_acc.shape[0], peak_idx[0] + mask_size + 1)
            x_start = max(0, peak_idx[1] - mask_size)
            x_end = min(temp_acc.shape[1], peak_idx[1] + mask_size + 1)
            
            temp_acc[peak_idx] = 0
        
        # Quality check: ensure we have enough well-distributed VPs
        if len(vps) < 2:
            print(f"Warning: Only {len(vps)} vanishing points found, need at least 2 for calibration")
            return []
        
        # Final spatial diversity check
        if len(vps) >= 2:
            vp_coords = np.array([vp[:2] for vp, _ in vps])
            distances = []
            for i in range(len(vps)):
                for j in range(i+1, len(vps)):
                    dist = np.linalg.norm(vp_coords[i] - vp_coords[j])
                    distances.append(dist)
            
            min_dist = min(distances) if distances else 0
            if min_dist < min_vp_distance:
                print(f"Warning: Final VP check failed - minimum distance {min_dist:.1f} px < threshold {min_vp_distance} px")
                return []
        
        return vps

class KalmanFilter:
    """Kalman filter for velocity smoothing"""

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self.initialized = False
        # State: [X, Z, vX, vZ]
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        q = 0.1
        self.Q = np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                           [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                           [dt ** 3 / 2, 0, dt ** 2, 0],
                           [0, dt ** 3 / 2, 0, dt ** 2]], dtype=np.float32) * q
        self.R = np.eye(2, dtype=np.float32) * 10.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        if not self.initialized:
            self.state[2:] = measurement
            self.initialized = True
            return
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P

    def get_velocity(self) -> np.ndarray:
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
        
        # Vehicle detection (YOLO only)
        if YOLO_AVAILABLE:
            self.vehicle_detector = YOLO('yolov8n.pt')
        else:
            self.vehicle_detector = None
            print("Warning: YOLOv8n not available; vehicle detection disabled.")
        
        # Tracking
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 1
        self.max_track_age = config.get('max_track_age', 30)
        
        # Feature tracking parameters
        self.feature_params = dict(
            maxCorners=10,
            qualityLevel=0.001,
            minDistance=5,
            blockSize=5
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Speed parameters
        self.speed_threshold = config.get('speed_threshold', 50)
        self.reference_height = config.get('reference_vehicle_height', 1.5)
        
        # Input speed for testing (when computed speed is zero)
        self.i_s = config.get('i_s', None)
        
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
    # Enhanced Edge Detection for Manhattan World Scenes

    
    def extract_edgelets(self, image: np.ndarray) -> np.ndarray:
        """Enhanced edgelet extraction with adaptive parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection
        edgelets_all = []
        
        # Scale 1: Fine edges with adaptive threshold
        blurred1 = cv2.GaussianBlur(gray, (3, 3), 0.8)
        
        # Compute adaptive Canny thresholds
        median_intensity = np.median(blurred1)
        lower_thresh = max(0, int(0.5 * median_intensity))
        upper_thresh = min(255, int(1.5 * median_intensity))
        
        edges1 = cv2.Canny(blurred1, lower_thresh, upper_thresh)
        
        # Scale 2: Medium scale with fixed moderate thresholds
        blurred2 = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges2 = cv2.Canny(blurred2, 30, 90)  # Lower than your current 50,150
        
        # Scale 3: Coarse edges for strong structural lines
        blurred3 = cv2.GaussianBlur(gray, (7, 7), 1.4)
        edges3 = cv2.Canny(blurred3, 20, 60)  # Even lower for prominent structures
        
        # Combine multi-scale edges
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Extract edgelets with gradient information
        grad_x = cv2.Sobel(blurred2, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred2, cv2.CV_32F, 0, 1, ksize=3)
        
        edge_pixels = np.where(combined_edges > 0)
        
        if len(edge_pixels[0]) == 0:
            return np.array([]).reshape(0, 2)
        
        # Filter edgelets by gradient strength
        edgelets = []
        for y, x in zip(edge_pixels[0], edge_pixels[1]):
            if 0 <= y < grad_y.shape[0] and 0 <= x < grad_x.shape[1]:
                grad_mag = np.sqrt(grad_x[y, x]**2 + grad_y[y, x]**2)
                # Only keep strong gradients
                if grad_mag > 10.0:  # Threshold for gradient strength
                    edgelets.append([x, y])
        
        edgelets = np.array(edgelets, dtype=np.float32) if edgelets else np.array([]).reshape(0, 2)
        
        # Additional filter: remove isolated edge pixels
        if len(edgelets) > 0:
            edgelets = filter_isolated_edgelets(edgelets, min_neighbors=2, radius=5)
        
        print(f"Extracted {len(edgelets)} robust edgelets")
        return edgelets

    # Hybrid Vanishing Point Detection for Challenging Scenes
    def detect_vanishing_points_hybrid(self, image: np.ndarray, edgelets: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Hybrid VP detection: Diamond Space + Hough Lines fallback"""
        
        # Try diamond space first
        if len(edgelets) > 100:
            self.vp_detector.accumulate_edgelets(edgelets, image.shape)
            diamond_vps = self.vp_detector.find_vanishing_points(3)
            
            if len(diamond_vps) >= 2:
                print(f"Diamond space successful: {len(diamond_vps)} VPs")
                return diamond_vps
        
        print("Diamond space failed, trying Hough line-based VP detection...")
        
        # Fallback: Line-based VP detection using Hough transform
        return self.detect_vps_from_hough_lines(image, edgelets)

    def detect_vps_from_hough_lines(self, image: np.ndarray, edgelets: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Fallback VP detection using Hough lines and RANSAC clustering"""
        
        h, w = image.shape[:2]
        
        # Create edge image
        edges_img = np.zeros((h, w), dtype=np.uint8)
        for x, y in edgelets.astype(int):
            if 0 <= y < h and 0 <= x < w:
                edges_img[y, x] = 255
        
        # Detect lines with aggressive parameters for sparse scenes
        lines = cv2.HoughLinesP(edges_img, rho=1, theta=np.pi/180, 
                            threshold=max(10, len(edgelets)//200),  # More aggressive
                            minLineLength=20, maxLineGap=15)
        
        if lines is None or len(lines) < 4:
            print(f"Insufficient lines detected: {len(lines) if lines is not None else 0}")
            return []
        
        print(f"Detected {len(lines)} lines for VP clustering")
        
        # Convert lines to homogeneous form
        line_eqs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            # Normalize
            norm = np.sqrt(a*a + b*b)
            if norm > 0:
                line_eqs.append([a/norm, b/norm, c/norm])
        
        if len(line_eqs) < 4:
            return []
        
        line_eqs = np.array(line_eqs)
        
        # Find vanishing points by clustering line intersections
        vps = []
        intersections = []
        
        # Compute all line intersections
        for i in range(len(line_eqs)):
            for j in range(i+1, len(line_eqs)):
                l1, l2 = line_eqs[i], line_eqs[j]
                
                # Intersection of two lines l1 and l2
                det = l1[0] * l2[1] - l1[1] * l2[0]
                
                if abs(det) > 1e-6:  # Lines not parallel
                    x = (l1[1] * l2[2] - l1[2] * l2[1]) / det
                    y = (l1[2] * l2[0] - l1[0] * l2[2]) / det
                    
                    # Keep reasonable intersections (not too far from image)
                    max_coord = 5 * max(w, h)
                    if abs(x) < max_coord and abs(y) < max_coord:
                        intersections.append([x, y, i, j])  # Store line indices too
        
        if len(intersections) < 3:
            return []
        
        intersections = np.array(intersections)
        
        # Cluster intersections using spatial proximity
        from sklearn.cluster import DBSCAN
        
        # Use adaptive clustering radius based on image size
        eps_radius = min(w, h) * 0.1  # 10% of image dimension
        clustering = DBSCAN(eps=eps_radius, min_samples=2).fit(intersections[:, :2])
        
        cluster_labels = clustering.labels_
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise points
        
        print(f"Found {len(unique_labels)} VP clusters from {len(intersections)} intersections")
        
        # Extract vanishing points from clusters
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_intersections = intersections[cluster_mask]
            
            if len(cluster_intersections) >= 2:  # Require at least 2 supporting intersections
                # Use cluster centroid as VP
                vp_x = np.mean(cluster_intersections[:, 0])
                vp_y = np.mean(cluster_intersections[:, 1])
                
                # Score based on cluster size and compactness
                cluster_size = len(cluster_intersections)
                compactness = 1.0 / (1.0 + np.std(cluster_intersections[:, :2]))
                score = cluster_size * compactness
                
                vp = np.array([vp_x, vp_y, 1.0])  # Homogeneous coordinates
                vps.append((vp, score))
                
                if len(vps) >= 3:  # Stop after finding 3 VPs
                    break
        
        # Sort by score and apply spatial diversity check
        vps.sort(key=lambda x: x[1], reverse=True)
        
        # Apply spatial diversity filtering
        filtered_vps = []
        min_distance = min(w, h) * 0.2  # 20% of image dimension
        
        for vp, score in vps:
            too_close = False
            for existing_vp, _ in filtered_vps:
                dist = np.linalg.norm(vp[:2] - existing_vp[:2])
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_vps.append((vp, score))
            
            if len(filtered_vps) >= 3:
                break
        
        print(f"Hough-based VP detection: {len(filtered_vps)} spatially diverse VPs")
        return filtered_vps
    #def extract_edgelets(self, image: np.ndarray) -> np.ndarray:
       # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       # blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        #edges = cv2.Canny(blurred, 50, 150)
        #edge_pixels = np.where(edges > 0)
        #if len(edge_pixels[0]) == 0:
       #     return np.array([]).reshape(0, 2)
      #  grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
     #   grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    #    edgelets = []
  #      for y, x in zip(edge_pixels[0], edge_pixels[1]):
   #         if 0 <= y < grad_y.shape[0] and 0 <= x < grad_x.shape[1]:
 #               edgelets.append([x, y])
#        return np.array(edgelets, dtype=np.float32) if edgelets else np.array([]).reshape(0, 2)

    def compute_camera_intrinsics(self, vanishing_points: List[Tuple[np.ndarray, float]], img_shape: Tuple[int, int]) -> bool:
        if len(vanishing_points) < 3:
            return False
        h, w = img_shape[:2]
        (vp1, _), (vp2, _), (vp3, _) = vanishing_points[:3]
        u1, v1 = float(vp1[0]), float(vp1[1])
        u2, v2 = float(vp2[0]), float(vp2[1])
        u3, v3 = float(vp3[0]), float(vp3[1])
        from scipy.optimize import least_squares
        def residuals(vars):
            cx, cy, f2 = vars
            r1 = (u1 - cx) * (u2 - cx) + (v1 - cy) * (v2 - cy) + f2
            r2 = (u1 - cx) * (u3 - cx) + (v1 - cy) * (v3 - cy) + f2
            r3 = (u2 - cx) * (u3 - cx) + (v2 - cy) * (v3 - cy) + f2
            return [r1, r2, r3]
        init = [(w - 1) / 2, (h - 1) / 2, max(w, h) ** 2]
        sol = least_squares(residuals, init)
        cx, cy, f2 = sol.x
        f = np.sqrt(abs(f2))
        self.camera_intrinsics = CameraIntrinsics(f, f, cx, cy)
        self.compute_rotation_matrix(vanishing_points)
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
            
            # Validate rotation matrix
            R = self.rotation_matrix
            RTR = R.T @ R
            print("Rotation matrix R:")
            print(R)
            print("R^T @ R =")
            print(RTR)
            
            # Check orthogonality
            identity_diff = np.abs(RTR - np.eye(3))
            max_deviation = np.max(identity_diff)
            print(f"Max deviation from identity: {max_deviation:.2e}")
            
            if max_deviation > 1e-3:
                print("WARNING: Rotation matrix is not properly orthogonal!")
                return False
            
            # Check determinant
            det = np.linalg.det(R)
            print(f"Rotation matrix determinant: {det:.6f}")
            if abs(det - 1.0) > 1e-3:
                print("WARNING: Rotation matrix determinant is not 1.0!")
                return False
                
            return True
        return False

    def compute_ground_homography(self) -> bool:
        """Compute homography mapping the world ground plane (Y=0) to image plane."""
        if self.camera_intrinsics is None or self.rotation_matrix is None:
            return False

        K = self.camera_intrinsics.K
        R = self.rotation_matrix
        h_cam = self.config.get('camera_height', 1.7)  # meters

        # Use first two columns of R (X and Z world axes) and translation [0, h_cam, 0]
        t = np.array([0.0, h_cam, 0.0], dtype=np.float32)
        H_ground = K @ np.column_stack([R[:, 0], R[:, 1], t])

        # Strict homography validation
        det = np.linalg.det(H_ground)
        print(f"Homography determinant: {det:.2e}")
        
        if abs(det) < 1e-6:
            print("ERROR: Homography is nearly singular! Recalibration needed.")
            return False
            
        if np.any(np.isnan(H_ground)) or np.any(np.isinf(H_ground)):
            print(f"ERROR: Invalid homography computed. Camera height: {h_cam}m")
            return False

        # Test homography with sample image points
        h, w = self.config.get('image_shape', (480, 640))
        test_points = [
            (w//2, h//2),      # center
            (w//4, h//4),      # top-left quadrant
            (3*w//4, 3*h//4),  # bottom-right quadrant
            (w//2, h-50)       # near bottom center
        ]
        
        print("Testing homography with sample image points:")
        world_points = []
        for x, y in test_points:
            p_homo = np.array([x, y, 1.0])
            try:
                H_inv = np.linalg.inv(H_ground)
                world_p = H_inv @ p_homo
                world_p = world_p / world_p[2] if abs(world_p[2]) > 1e-6 else world_p
                world_points.append(world_p[:2])
                print(f"  Image ({x}, {y}) -> World ({world_p[0]:.3f}, {world_p[1]:.3f})")
            except np.linalg.LinAlgError:
                print(f"  Image ({x}, {y}) -> ERROR")
                return False
        
        # Check if world points vary (not all the same)
        if len(world_points) > 1:
            world_array = np.array(world_points)
            variance = np.var(world_array, axis=0)
            print(f"World coordinates variance: {variance}")
            if np.any(variance < 1e-6):
                print("WARNING: World coordinates show very little variation!")
                return False

        self.homography_ground = H_ground
        print(f"Ground homography computed with camera height: {h_cam}m")
        return True

    def calibrate_scale(self, reference_bbox: Tuple[int, int, int, int], reference_height: float) -> bool:
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
        
        # Verify solution is reasonable
        if abs(lam_bot) < 1e-6 or abs(lam_top) < 1e-6:
            print(f"Warning: Scale calibration failed - invalid lambda values: λ_bot={lam_bot:.2e}, λ_top={lam_top:.2e}")
            return False
            
        P_bot = lam_bot * p_bot
        P_top = lam_top * p_top
        world_height = np.linalg.norm(P_top - P_bot)
        if world_height > 1e-6:
           self.scale_factor = reference_height / world_height
           print(f"Scale calibration successful: world_height={world_height:.6f}m, scale_factor={self.scale_factor:.6f}")
           return True
        print(f"Warning: Scale calibration failed - computed world height too small: {world_height:.2e}m")
        return False

    def _detect_with_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if not YOLO_AVAILABLE or self.vehicle_detector is None:
            return []
        results = self.vehicle_detector(image, conf=0.25, classes=[2, 3, 5, 7])
        results = results if isinstance(results, list) else [results]
        detections: List[Tuple[int, int, int, int, float]] = []
        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.empty((0, 4))
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.ones((xyxy.shape[0],), dtype=np.float32)
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return detections

    def detect_vehicles(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        return self._detect_with_yolo(image)

    def associate_detections_to_tracks(self, detections: List[Tuple[int, int, int, int, float]], timestamp: float) -> Dict[int, Tuple[int, int, int, int]]:
        associated = {}
        used_detections = set()
        for track_id, track in self.tracks.items():
            if not track.bbox:
                continue
            best_iou = 0
            best_detection_idx = -1
            best_center_distance = float('inf')
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                iou = self.calculate_iou(track.bbox, detection[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = i
                    x1, y1, x2, y2 = track.bbox
                    txc, tyc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    dx1, dy1, dx2, dy2, _ = detection
                    dxc, dyc = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                    best_center_distance = np.hypot(dxc - txc, dyc - tyc)
            if best_detection_idx >= 0:
                iou_ok = best_iou >= 0.1
                x1, y1, x2, y2 = track.bbox
                bbox_size_threshold = 0.5 * min(max(x2 - x1, 1), max(y2 - y1, 1))
                distance_ok = best_center_distance <= bbox_size_threshold
                if iou_ok or distance_ok:
                    associated[track_id] = detections[best_detection_idx][:4]
                    used_detections.add(best_detection_idx)
        for i, detection in enumerate(detections):
            if i not in used_detections and detection[4] > 0.6:
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

    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
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
            prev_good = features[good_mask].reshape(-1, 2)
            back_good = back_features[good_mask].reshape(-1, 2)
            fb_error = np.linalg.norm(prev_good - back_good, axis=1)
            # Relaxed forward-backward threshold for better feature retention
            refined_mask = fb_error < 10.0
            if not np.any(refined_mask):
                return None
            prev_good = prev_good[refined_mask]
            curr_good = new_features[good_mask].reshape(-1, 2)[refined_mask]
            
            # Debug: print number of features kept
            print(f"Track {track_id}: {len(prev_good)} features kept for speed.")
        else:
            return None
        
        # Convert to image coordinates (add ROI offset)
        features_img = prev_good + np.array([x1, y1], dtype=np.float32)
        new_features_img = curr_good + np.array([x1, y1], dtype=np.float32)
        
        # Debug: print pixel displacements in ROI
        for i in range(len(prev_good)):
            pixel_disp = curr_good[i] - prev_good[i]
            print(f"Track {track_id}: pixel disp={pixel_disp} px")
        
        # Project to world coordinates
        world_displacements = []
        
        for i, (f1, f2) in enumerate(zip(features_img, new_features_img)):
            try:
                # Project to ground plane
                p1_homo = np.array([float(f1[0]), float(f1[1]), 1.0], dtype=np.float32)
                p2_homo = np.array([float(f2[0]), float(f2[1]), 1.0], dtype=np.float32)
                
                print(f"Feature {i}: img1 {f1}, img2 {f2}")
                
                # Use homography to project to ground plane
                if self.homography_ground is not None:
                    H_inv = np.linalg.inv(self.homography_ground)
                    
                    world_p1 = H_inv @ p1_homo
                    world_p2 = H_inv @ p2_homo
                    
                    world_p1 = world_p1 / world_p1[2] if abs(world_p1[2]) > 1e-6 else world_p1
                    world_p2 = world_p2 / world_p2[2] if abs(world_p2[2]) > 1e-6 else world_p2
                    
                    print(f"World p1: {world_p1}, World p2: {world_p2}, diff: {world_p2-world_p1}")
                    
                    # Apply scale factor
                    if self.scale_factor is not None:
                        print(f"scale_factor: {self.scale_factor}")
                        displacement = (world_p2[:2] - world_p1[:2]) * self.scale_factor
                        print(f"Scaled displacement: {displacement}")
                        world_displacements.append(displacement)
                    else:
                        print("Warning: scale_factor is None!")
                else:
                    print("Warning: homography_ground is None!")
            
            except (np.linalg.LinAlgError, ZeroDivisionError) as e:
                print(f"Error processing feature {i}: {e}")
                continue
        
        # Debug: print number of valid ground displacements
        print(f"Track {track_id}: {len(world_displacements)} valid ground displacements.")
        
        # Debug: validate homography and scale factor
        if self.homography_ground is not None:
            print(f"Homography matrix:\n{self.homography_ground}")
            # Check if homography is degenerate
            det = np.linalg.det(self.homography_ground)
            print(f"Homography determinant: {det}")
            if abs(det) < 1e-10:
                print("WARNING: Homography is nearly singular!")
        else:
            print("ERROR: No homography available!")
            
        if self.scale_factor is not None:
            print(f"Scale factor: {self.scale_factor}")
            if abs(self.scale_factor) < 1e-10:
                print("WARNING: Scale factor is nearly zero!")
            elif np.isnan(self.scale_factor) or np.isinf(self.scale_factor):
                print("ERROR: Scale factor is NaN or Inf!")
        else:
            print("ERROR: No scale factor available!")
        
        if not world_displacements:
            # Fallback: use center-of-bbox motion if no features available
            track = self.tracks.get(track_id)
            if track and len(track.timestamps) >= 2:
                # Get previous bbox from track history
                prev_bbox = track.bbox if hasattr(track, 'bbox') else None
                if prev_bbox is not None:
                    # Calculate center displacement
                    curr_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
                    
                    # Project centers to ground plane
                    try:
                        curr_homo = np.array([curr_center[0], curr_center[1], 1.0])
                        prev_homo = np.array([prev_center[0], prev_center[1], 1.0])
                        
                        H_inv = np.linalg.inv(self.homography_ground)
                        world_curr = H_inv @ curr_homo
                        world_prev = H_inv @ prev_homo
                        
                        world_curr = world_curr / world_curr[2] if abs(world_curr[2]) > 1e-6 else world_curr
                        world_prev = world_prev / world_prev[2] if abs(world_prev[2]) > 1e-6 else world_prev
                        
                        displacement = (world_curr[:2] - world_prev[:2]) * self.scale_factor
                        speed_ms = np.linalg.norm(displacement) / dt
                        speed_kmh = speed_ms * 3.6
                        print(f"Track {track_id}: Using fallback center motion, speed: {speed_kmh:.1f} km/h")
                        return speed_kmh
                    except (np.linalg.LinAlgError, ZeroDivisionError):
                        pass
            
            return None
        
        # Calculate speeds for each displacement
        speeds = []
        for displacement in world_displacements:
            speed_ms = float(np.linalg.norm(displacement)) / dt  # m/s
            speed_kmh = speed_ms * 3.6  # km/h
            speeds.append(speed_kmh)
            print(f"Track {track_id}: feature disp={displacement}, speed={speed_kmh:.1f} km/h")
        
        # Use median speed to reduce noise
        if speeds:
            median_speed = float(np.median(speeds))
            print(f"Track {track_id}: median_speed = {median_speed:.1f} km/h")
            
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
                    filtered_speed = float(np.linalg.norm(filtered_velocity) * 3.6)  # km/h
                    
                    print(f"Track {track_id}: Kalman filtered speed = {filtered_speed:.1f} km/h")
                    return filtered_speed
            
            return median_speed
        
        return None

    # Scene Quality Assessment for Manhattan World Detection
    def assess_scene_quality(self, image: np.ndarray, edgelets: np.ndarray) -> dict:
        """Assess if the scene is suitable for Manhattan world VP detection"""
        
        assessment = {
            'suitable': False,
            'edge_density': 0.0,
            'line_diversity': 0.0,
            'structural_strength': 0.0,
            'reason': ""
        }
        
        if len(edgelets) == 0:
            assessment['reason'] = "No edges detected"
            return assessment
        
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # 1. Edge density check
        edge_density = len(edgelets) / total_pixels
        assessment['edge_density'] = edge_density
        
        if edge_density < 0.005:  # Less than 0.5% edges
            assessment['reason'] = f"Insufficient edge density: {edge_density:.4f}"
            return assessment
        
        # 2. Line diversity assessment using Hough lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges_img = np.zeros((h, w), dtype=np.uint8)
        for x, y in edgelets.astype(int):
            if 0 <= y < h and 0 <= x < w:
                edges_img[y, x] = 255
        
        # Detect lines to assess orientation diversity
        lines = cv2.HoughLinesP(edges_img, rho=1, theta=np.pi/180, 
                            threshold=max(5, len(edgelets)//200),
                            minLineLength=10, maxLineGap=20)
        
        if lines is None or len(lines) < 3:  # Need at least 6 lines for 3 VPs
            assessment['reason'] = f"Insufficient line structures: {len(lines) if lines is not None else 0}"
            return assessment
        
        # 3. Analyze line orientations for Manhattan world suitability
        orientations = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Normalize to [0, 180)
            if angle < 0:
                angle += 180
            orientations.append(angle)
        
        orientations = np.array(orientations)
        
        # Check for three dominant orthogonal directions (Manhattan world)
        # Ideal Manhattan world has lines at ~0°, ~45°, ~90°, ~135° (or similar orthogonal sets)
        hist, bin_edges = np.histogram(orientations, bins=18, range=(0, 180))  # 10° bins
        
        # Find peaks in orientation histogram
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist, height=max(2, len(lines)//20), distance=4)  # At least 40° apart
        
        line_diversity = len(peaks) / 3.0  # Normalize by ideal 3 directions
        assessment['line_diversity'] = min(line_diversity, 1.0)
        
        if len(peaks) < 2:
            assessment['reason'] = f"Insufficient orientation diversity: {len(peaks)} dominant directions"
            return assessment
        
        # 4. Structural strength assessment
        # Check for strong, well-distributed lines
        line_lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            line_lengths.append(length)
        
        avg_line_length = np.mean(line_lengths)
        max_line_length = np.max(line_lengths)
        min_image_dim = min(h, w)
        
        structural_strength = (avg_line_length / min_image_dim) * (len(lines) / 20.0)
        assessment['structural_strength'] = min(structural_strength, 1.0)
        
        # Final suitability decision
        if (edge_density >= 0.005 and 
            line_diversity >= 0.5 and 
            structural_strength >= 0.3 and
            len(peaks) >= 2):
            assessment['suitable'] = True
            assessment['reason'] = "Scene suitable for VP detection"
        else:
            assessment['reason'] = f"Scene quality insufficient: density={edge_density:.3f}, diversity={line_diversity:.3f}, strength={structural_strength:.3f}"
        if not assessment['suitable'] and edge_density >= 0.005 and line_diversity >= 0.3:
            assessment['suitable'] = True
            assessment['reason'] = "Relaxed: enough orientation despite weak structure"
            # Strict Manhattan‐world suitability check
        if (edge_density >= 0.005 and
            line_diversity >= 0.5 and
            structural_strength >= 0.3 and
            len(peaks) >= 2):
            assessment['suitable'] = True
            assessment['reason'] = "Scene suitable for VP detection"
        else:
            assessment['suitable'] = False
            assessment['reason'] = (f"Insufficient structure: "
                                    f"density={edge_density:.3f}, "
                                    f"diversity={line_diversity:.3f}, "
                                    f"strength={structural_strength:.3f}")

        # Relaxed fallback: allow high‐edge‐density + moderate orientation even if no lines
        if (not assessment['suitable'] and
            edge_density >= 0.2 and      # e.g. >=20% pixels are edges
            line_diversity >= 0.1):      # at least one orientation peak
            assessment['suitable'] = True
            assessment['reason'] = "Relaxed fallback: dense edges + some orientation"

    
        return assessment

    def process_frame(self, frame: np.ndarray, timestamp: float, previous_frame = None):
        """Modified process_frame with scene quality assessment"""
        
        results = {
            'timestamp': timestamp,
            'vehicles': [],
            'calibrated': (self.camera_intrinsics is not None and 
                        self.homography_ground is not None and 
                        self.scale_factor is not None),
            'vanishing_points': [],
            'scene_assessment': {}
        }
        
        # Only attempt VP detection if calibration is needed
        if (self.scale_factor is None) and (self.calibration_frames % self.recalibration_interval == 0 or 
                                        self.camera_intrinsics is None):
            
            edgelets = self.extract_edgelets(frame)  # Use enhanced version
            
            # Assess scene quality BEFORE VP detection
            scene_assessment = self.assess_scene_quality(frame, edgelets)
            results['scene_assessment'] = scene_assessment
            
            print(f"Scene Assessment: {scene_assessment['reason']}")
            print(f"  Edge density: {scene_assessment['edge_density']:.4f}")
            print(f"  Line diversity: {scene_assessment['line_diversity']:.2f}")
            print(f"  Structural strength: {scene_assessment['structural_strength']:.2f}")
            
            # ALWAYS attempt VP detection when we have a sufficient number of edgelets:
            if len(edgelets) > 100:
                vps = self.detect_vanishing_points_hybrid(frame, edgelets)
                results['vanishing_points'] = [(vp.tolist(), score) for vp, score in vps]
                if vps:
                    print(f"Detected {len(vps)} vanishing points (forced bypass)")
                else:
                    print("Hybrid VP detector found no points")
            else:
                print(f"Too few edgelets ({len(edgelets)}) – skipping VP detection")

        # Continue with rest of processing (vehicle detection, etc.)
        # [Rest of your existing process_frame logic]
        
        return results
    def trigger_speed_alert(self, track_id: int, speed: float):
        message = f"Vehicle {track_id} exceeding speed limit at {speed:.1f} km/h"
        print(f"ALERT: {message}")
        if TTS_AVAILABLE and hasattr(self, 'tts_engine'):
            try:
                self.tts_engine.say(f"Speed violation: {int(speed)} kilometers per hour")
                self.tts_engine.runAndWait()
            except Exception:
                pass

    def cleanup_old_tracks(self, current_timestamp: float):
        to_remove = []
        for track_id, track in self.tracks.items():
            if (len(track.timestamps) == 0 or current_timestamp - track.timestamps[-1] > self.max_track_age):
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

    def annotate_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        annotated = frame.copy()
        for vehicle in results['vehicles']:
            track_id = vehicle['track_id']
            x1, y1, x2, y2 = vehicle['bbox']
            speed = vehicle.get('speed_kmh')
            color = (0, 255, 0) if speed is None else (0, 0, 255) if speed > self.speed_threshold else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id}"
            if speed is not None:
                label += f" | {speed:.1f} km/h"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_text = "CALIBRATED" if results['calibrated'] else "CALIBRATING..."
        status_color = (0, 255, 0) if results['calibrated'] else (0, 255, 255)
        cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        # Draw statistics
        active_tracks = len(self.tracks)
        stats_text = f"Frames: {self.stats['frames_processed']} | Active Tracks: {active_tracks} | Violations: {self.stats['speed_violations']}"
        cv2.putText(annotated, stats_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated

    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: str):
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

    def calibrate_with_reference_vehicle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], height_meters: float) -> bool:
        if self.calibrate_scale(bbox, height_meters):
            print(f"Scale calibrated with reference vehicle height: {height_meters}m")
            return True
        return False

    def get_test_speed_with_variance(self, base_speed: float, variance_range: float = 7.0) -> float:
        """Generate a random speed within ±variance_range km/h of the base speed for testing"""
        if base_speed is None or base_speed <= 0:
            return 0.0
        
        # Generate random variance within ±variance_range
        variance = np.random.uniform(-variance_range, variance_range)
        test_speed = base_speed + variance
        
        # Ensure speed doesn't go negative
        test_speed = max(0.0, test_speed)
        
        return test_speed

class RealTimeProcessor:
    """Real-time processing wrapper with threading"""

    def __init__(self, estimator: VehicleSpeedEstimator):
        self.estimator = estimator
        self.capture_thread = None
        self.process_thread = None
        self.display_thread = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)

    def start_camera_processing(self, camera_index: int = 0):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, args=(camera_index,))
        self.process_thread = threading.Thread(target=self._process_frames)
        self.display_thread = threading.Thread(target=self._display_results)
        self.capture_thread.start()
        self.process_thread.start()
        self.display_thread.start()
        print("Started real-time processing. Press 'q' to quit, 'c' to calibrate scale.")

    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Frames: {frame_count}")
        results: List[Dict[str, Any]] = []
        previous_frame = None
        frame_idx = 0
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
                timestamp = frame_idx / fps if fps > 0 else 0.0
                frame_result = self.estimator.process_frame(frame, timestamp, previous_frame)
                results.append(frame_result)
                if out_writer:
                    annotated_frame = self.estimator.annotate_frame(frame, frame_result)
                    out_writer.write(annotated_frame)
                if frame_idx % 30 == 0:
                    print(f"Processed frame {frame_idx}/{frame_count} ({(frame_idx / max(frame_count, 1)) * 100:.1f}%)")
                previous_frame = frame
                frame_idx += 1
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
        return results

    def _capture_frames(self, camera_index: int):
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
                    pass
        cap.release()

    def _process_frames(self):
        previous_frame = None
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                result = self.estimator.process_frame(frame, timestamp, previous_frame)
                annotated_frame = self.estimator.annotate_frame(frame, result)
                result['annotated_frame'] = annotated_frame
                try:
                    self.result_queue.put(result, timeout=0.1)
                except queue.Full:
                    pass
                previous_frame = frame
            except queue.Empty:
                continue

    def _display_results(self):
        calibration_mode = False
        reference_bbox = None
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)
                annotated_frame = result.get('annotated_frame')
                if annotated_frame is not None:
                    if calibration_mode:
                        cv2.putText(annotated_frame, "CALIBRATION MODE: Click and drag to select reference vehicle", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        if reference_bbox:
                            x1, y1, x2, y2 = reference_bbox
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(annotated_frame, "Press ENTER to confirm, ESC to cancel", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow('Vehicle Speed Estimation', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                    elif key == ord('c'):
                        calibration_mode = True
                        print("Entering calibration mode. Click and drag to select reference vehicle.")
                    elif calibration_mode:
                        if key == 13:  # Enter
                            if reference_bbox:
                                height = float(input("Enter reference vehicle height in meters: "))
                                if self.estimator.calibrate_with_reference_vehicle(annotated_frame, reference_bbox, height):
                                    print("Calibration successful!")
                                else:
                                    print("Calibration failed!")
                                calibration_mode = False
                                reference_bbox = None
                        elif key == 27:  # ESC
                            calibration_mode = False
                            reference_bbox = None
                            print("Calibration cancelled.")
            except queue.Empty:
                continue
        cv2.destroyAllWindows()

    def stop(self):
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
    parser.add_argument('--mode', choices=['camera', 'video'], default='camera', help='Processing mode: camera or video file')
    parser.add_argument('--input', type=str, help='Input video file path (for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--csv', type=str, help='Output CSV file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--reference-height', type=float, default=1.5, help='Reference vehicle height in meters')
    parser.add_argument('--speed-threshold', type=float, default=50, help='Speed violation threshold in km/h')
    parser.add_argument('--i-s', type=float, help='Input speed in km/h for testing when computed speed is zero')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'diamond_resolution': 512,
        'image_normalization': 1.0,
        'max_track_age': 2.0,
        'recalibration_interval': 30,
        'reference_vehicle_height': args.reference_height,
        'speed_threshold': args.speed_threshold,
        'camera_height': 1.7,
        'i_s': args.i_s
    }
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        except FileNotFoundError:
            print(f"Config file not found: {args.config}")

    estimator = VehicleSpeedEstimator(config)

    if args.mode == 'camera':
        processor = RealTimeProcessor(estimator)
        try:
            processor.start_camera_processing(args.camera)
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
            if args.csv:
                estimator.save_results_to_csv(results, args.csv)
                print(f"Results saved to: {args.csv}")
            print(f"Processing complete. Processed {len(results)} frames.")
            print(f"Statistics: {estimator.stats}")
        except Exception as e:
            print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()