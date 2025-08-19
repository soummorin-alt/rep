import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import logging
import os

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VehicleTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    speed_history: deque
    position_history: deque
    last_seen: int
    confidence: float
    kalman_filter: Optional[object] = None


@dataclass
class CameraParams:
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    height: float = 1.7
    K: np.ndarray = None
    homography: np.ndarray = None


class KalmanFilter1D:
    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        return self.posteri_estimate


class MiDaSDepthEstimator:
    """MiDaS depth estimator with graceful fallback if unavailable."""
    def __init__(self, model_type: str = "DPT_Hybrid"):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            import torch.hub
            self.model = torch.hub.load("intel/MiDaS", model_type)
            self.model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform if model_type == "DPT_Hybrid" else midas_transforms.default_transform
            logger.info(f"Loaded MiDaS {model_type} on {self.device}")
        except Exception as e:
            logger.warning(f"MiDaS not available, depth-based scaling will be approximate: {e}")
            self.model = None
            self.transform = None

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            # Return a uniform map to indicate unavailable depth
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy().astype(np.float32)
        # Normalize to [0,1] for relative comparison
        dmin, dmax = float(depth_map.min()), float(depth_map.max())
        if dmax - dmin > 1e-6:
            depth_map = (depth_map - dmin) / (dmax - dmin)
        else:
            depth_map = np.zeros_like(depth_map, dtype=np.float32)
        return depth_map


class RAFTOpticalFlow:
    def __init__(self):
        try:
            from torchvision.models.optical_flow import raft_large
            self.model = raft_large(pretrained=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            logger.info(f"Loaded RAFT on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load RAFT, using Farneback flow: {e}")
            self.model = None

    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        if self.model is None:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return flow
        img1_tensor = self.transform(frame1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(frame2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            flow_predictions = self.model(img1_tensor, img2_tensor)
            flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()
        return cv2.resize(flow, (frame1.shape[1], frame1.shape[0]))


class DeepVanishingPoint:
    def __init__(self, model_path: str = "deepvan_weights.pth"):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            import torchvision.models as models
            class DeepVanNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = models.resnet50(pretrained=True)
                    self.backbone.fc = nn.Linear(2048, 6)
                def forward(self, x):
                    out = self.backbone(x)
                    return out.view(-1, 3, 2)
            self.model = DeepVanNet().to(self.device)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded DeepVan weights")
            else:
                logger.warning("DeepVan weights not found, using backbone-only initialization")
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            logger.error(f"Failed to load DeepVan: {e}")
            self.model = None

    def detect_vanishing_points(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            h, w = frame.shape[:2]
            return np.array([[w/2, 0], [w, h/2], [w/2, h/2]], dtype=np.float32)
        inp = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vp = self.model(inp)[0].cpu().numpy()
        h, w = frame.shape[:2]
        vp[:, 0] = np.clip(vp[:, 0], 0, 1) * w
        vp[:, 1] = np.clip(vp[:, 1], 0, 1) * h
        return vp.astype(np.float32)


class ByteTracker:
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.tracks: Dict[int, VehicleTrack] = {}
        self.track_id_count = 0

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float, int]]) -> List[VehicleTrack]:
        matched_tracks: Dict[int, VehicleTrack] = {}
        for det_bbox, conf, _ in detections:
            if conf < self.track_thresh:
                continue
            best_iou, best_id = 0.0, None
            for tid, tr in self.tracks.items():
                iou = self._calculate_iou(det_bbox, tr.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                tr = self.tracks[best_id]
                tr.bbox = det_bbox
                tr.confidence = conf
                tr.last_seen = 0
                matched_tracks[best_id] = tr
            else:
                tr = VehicleTrack(
                    track_id=self.track_id_count,
                    bbox=det_bbox,
                    speed_history=deque(maxlen=10),
                    position_history=deque(maxlen=10),
                    last_seen=0,
                    confidence=conf,
                    kalman_filter=KalmanFilter1D(),
                )
                matched_tracks[self.track_id_count] = tr
                self.track_id_count += 1
        for tid, tr in self.tracks.items():
            if tid not in matched_tracks:
                tr.last_seen += 1
                if tr.last_seen < 30:
                    matched_tracks[tid] = tr
        self.tracks = matched_tracks
        return list(self.tracks.values())

    def _calculate_iou(self, b1, b2) -> float:
        x1, y1, x2, y2 = b1
        X1, Y1, X2, Y2 = b2
        xi1, yi1 = max(x1, X1), max(y1, Y1)
        xi2, yi2 = min(x2, X2), min(y2, Y2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (X2 - X1) * (Y2 - Y1)
        return inter / (a1 + a2 - inter + 1e-6)


class VehicleSpeedEstimator:
    def __init__(self, video_path: Optional[str] = None, camera_id: int = 0, output_path: str = "output_speed_estimation.mp4"):
        # Depth estimator optionally used to refine scale
        self.depth_estimator = MiDaSDepthEstimator()
        self.flow_estimator = RAFTOpticalFlow()
        self.vp_detector = DeepVanishingPoint()
        self.vehicle_detector = YOLO('yolov8n.pt')
        self.tracker = ByteTracker()
        self.video_path = video_path
        self.camera_id = camera_id
        self.output_path = output_path
        self.camera_params = CameraParams()
        self.prev_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.recalibration_interval = 30
        self.speed_log: List[Dict] = []
        self.dt: float = 1.0 / 30.0
        logger.info("VehicleSpeedEstimator initialized")

    def calibrate_camera(self, vanishing_points: np.ndarray, frame_shape: Tuple[int, int, int]):
        h, w = frame_shape[:2]
        cx, cy = w / 2.0, h / 2.0
        f = np.sqrt(w * h)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.0, 0.0, self.camera_params.height], dtype=np.float32)
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        d = float(self.camera_params.height)
        H = K @ (R - np.outer(t, n) / max(d, 1e-6)) @ np.linalg.inv(K)
        self.camera_params.fx = f
        self.camera_params.fy = f
        self.camera_params.cx = cx
        self.camera_params.cy = cy
        self.camera_params.K = K
        self.camera_params.homography = H
        return K, H

    def compute_scale_factor(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Compute meters-per-pixel at the ground using box height and refine with local depth.

        Baseline: known_vehicle_height / pixel_height.
        Refinement: adjust by ratio of global median depth to local bottom-center depth (clamped).
        """
        x1, y1, x2, y2 = bbox
        hpx = max(1, y2 - y1)
        known_vehicle_height = 1.5
        base_scale = known_vehicle_height / float(hpx)
        if depth_map is None or depth_map.size == 0:
            return base_scale
        H, W = depth_map.shape[:2]
        bx = int(np.clip((x1 + x2) / 2.0, 0, W - 1))
        by = int(np.clip(y2, 0, H - 1))
        # Sample a small patch near the bottom-center of bbox
        patch_half = 2
        x0, x1p = max(0, bx - patch_half), min(W, bx + patch_half + 1)
        y0, y1p = max(0, by - patch_half), min(H, by + patch_half + 1)
        patch = depth_map[y0:y1p, x0:x1p]
        local_depth = float(np.median(patch)) if patch.size > 0 else 0.0
        global_depth = float(np.median(depth_map)) if depth_map.size > 0 else 0.0
        if local_depth <= 0 or global_depth <= 0:
            return base_scale
        # If MiDaS is inverse/relative, using ratio provides a gentle adjustment
        depth_ratio = np.clip(global_depth / max(local_depth, 1e-6), 0.5, 1.5)
        refined_scale = base_scale * float(depth_ratio)
        return refined_scale

    def estimate_vehicle_displacement(self, flow: np.ndarray, bbox: Tuple[int, int, int, int], homography: np.ndarray, scale_factor: float) -> float:
        x1, y1, x2, y2 = bbox
        grid = 5
        xs = np.linspace(x1, x2, grid, dtype=int)
        ys = np.linspace(y1, y2, grid, dtype=int)
        disps: List[float] = []
        H = homography
        H_inv = None
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0
        for x in xs:
            for y in ys:
                if y < 0 or y >= flow.shape[0] or x < 0 or x >= flow.shape[1]:
                    continue
                fx, fy = flow[y, x] if flow.ndim == 3 else (0.0, 0.0)
                p1 = np.array([x, y, 1.0], dtype=np.float32)
                p2 = np.array([x + fx, y + fy, 1.0], dtype=np.float32)
                w1 = H_inv @ p1
                w2 = H_inv @ p2
                if abs(w1[2]) < 1e-6 or abs(w2[2]) < 1e-6:
                    continue
                w1 /= w1[2]
                w2 /= w2[2]
                disp = np.linalg.norm((w2[:2] - w1[:2]) * scale_factor)
                if disp > 0:
                    disps.append(float(disp))
        return float(np.median(disps)) if disps else 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[VehicleTrack]]:
        midas_frame = cv2.resize(frame, (384, 384))
        deepvan_frame = cv2.resize(frame, (512, 512))
        if self.frame_count % self.recalibration_interval == 0:
            vps = self.vp_detector.detect_vanishing_points(deepvan_frame)
            self.calibrate_camera(vps, frame.shape)
            logger.info(f"Recalibrated camera at frame {self.frame_count}")
        # Estimate relative depth (used to refine scale)
        depth_map = self.depth_estimator.estimate_depth(midas_frame)
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        results = self.vehicle_detector.predict(frame, conf=0.25, classes=[2, 3, 5, 7])
        detections: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        for r in (results if isinstance(results, list) else [results]):
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for box, conf, c in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.astype(int)
                detections.append(((x1, y1, x2, y2), float(conf), int(c)))
        tracks = self.tracker.update(detections)
        flow = None
        if self.prev_frame is not None:
            flow = self.flow_estimator.compute_flow(self.prev_frame, frame)
        dt = self.dt
        for tr in tracks:
            if flow is not None and self.camera_params.homography is not None:
                scale = self.compute_scale_factor(depth_map, tr.bbox)
                disp = self.estimate_vehicle_displacement(flow, tr.bbox, self.camera_params.homography, scale)
                speed_kmh = float(np.linalg.norm(disp) / dt * 3.6)
                if tr.kalman_filter is not None:
                    speed_kmh = tr.kalman_filter.update(speed_kmh)
                tr.speed_history.append(speed_kmh)
                cx = (tr.bbox[0] + tr.bbox[2]) / 2.0
                cy = (tr.bbox[1] + tr.bbox[3]) / 2.0
                tr.position_history.append((cx, cy))
                self.speed_log.append({
                    'timestamp': time.time(),
                    'frame': self.frame_count,
                    'track_id': tr.track_id,
                    'bbox': tr.bbox,
                    'speed_kmh': speed_kmh,
                    'displacement': disp,
                })
        annotated = self.annotate_frame(frame, tracks)
        self.prev_frame = frame.copy()
        self.frame_count += 1
        return annotated, tracks

    def annotate_frame(self, frame: np.ndarray, tracks: List[VehicleTrack]) -> np.ndarray:
        img = frame.copy()
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {tr.track_id}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if tr.speed_history:
                spd = tr.speed_history[-1]
                color = (0, 0, 255) if spd > 50 else (0, 255, 0)
                cv2.putText(img, f"{spd:.1f} km/h", (x1, min(img.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if spd > 50:
                    cv2.putText(img, "SPEEDING!", (x1, max(0, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    def run(self):
        cap = cv2.VideoCapture(self.video_path) if self.video_path else cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error("Failed to open video source")
            return
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps_read) if fps_read and fps_read > 1e-3 else 30
        self.dt = 1.0 / float(max(fps, 1))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        logger.info(f"Processing video: {fps} FPS, {width}x{height}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, _ = self.process_frame(frame)
                out.write(annotated)
                cv2.imshow('Vehicle Speed Estimation', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames")
        finally:
            if self.speed_log:
                pd.DataFrame(self.speed_log).to_csv('speed_log.csv', index=False)
                logger.info("Speed log saved to speed_log.csv")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            logger.info(f"Processing complete. Output saved to {self.output_path}")


if __name__ == "__main__":
    estimator = VehicleSpeedEstimator(video_path="input_video.mp4", output_path="output_speed_estimation.mp4")
    estimator.run()
    print("Vehicle speed estimation complete!\nCheck 'output_speed_estimation.mp4' and 'speed_log.csv'.")

