"""
Production Multi-Camera Application
====================================

Phiên bản tối ưu cho chạy 24/7 real-time.
Sử dụng OptimizedCameraWorker để đảm bảo:
- Low-latency capture
- Stable reconnection
- Inference statistics
- Memory efficiency
"""

import threading
import time
import os
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass

from ..config import SystemConfig, CameraConfig
from ..detection import MultiModelLoader
from .optimized_worker import OptimizedCameraWorker, WorkerConfig, WorkerStatus
from .inference_stats import get_stats_manager, InferenceStatsManager


@dataclass
class ProductionStats:
    """Thống kê production"""
    total_cameras: int = 0
    running_cameras: int = 0
    total_person_alerts: int = 0
    total_coal_alerts: int = 0
    uptime_seconds: float = 0.0
    total_frames: int = 0
    total_detections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cameras": self.total_cameras,
            "running_cameras": self.running_cameras,
            "total_person_alerts": self.total_person_alerts,
            "total_coal_alerts": self.total_coal_alerts,
            "uptime_seconds": round(self.uptime_seconds, 0),
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
        }


class ProductionMultiCameraApp:
    """
    Production-ready multi-camera application
    
    Features:
    - OptimizedCameraWorker cho mỗi camera
    - Low-latency capture với grab pattern
    - Stable reconnection với exponential backoff
    - Inference statistics tracking
    - Memory efficient
    - 24/7 operation ready
    
    Usage:
        app = ProductionMultiCameraApp(
            config=system_config,
            on_alert=lambda cam_id, type, active, val: handle_alert(...)
        )
        
        if app.load_models():
            app.start_all()
            
            # Trong GUI loop
            for cam_id, worker in app.workers.items():
                frame = worker.get_display_frame()
                if frame is not None:
                    display(cam_id, frame)
            
            app.stop_all()
    """
    
    def __init__(
        self,
        config: SystemConfig,
        on_frame: Optional[Callable[[int, Any], None]] = None,
        on_alert: Optional[Callable[[int, str, bool, float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_status_change: Optional[Callable[[int, WorkerStatus], None]] = None,
    ):
        """
        Args:
            config: SystemConfig
            on_frame: Callback(camera_id, frame) - KHÔNG dùng, lấy frame qua get_display_frame()
            on_alert: Callback(camera_id, alert_type, is_active, value)
            on_log: Callback(message)
            on_status_change: Callback(camera_id, status)
        """
        self.config = config
        self.on_frame = on_frame
        self.on_alert = on_alert
        self.on_log = on_log
        self.on_status_change = on_status_change
        
        # Model loader
        self._model_loader = MultiModelLoader.get_instance()
        self._models_loaded = False
        
        # Workers
        self._workers: Dict[int, OptimizedCameraWorker] = {}
        self._worker_locks: Dict[str, threading.Lock] = {}
        
        # Stats
        self._stats = ProductionStats()
        self._start_time: Optional[float] = None
        self._inference_stats = get_stats_manager()
        
        # Alert counters
        self._person_alerts = 0
        self._coal_alerts = 0
        
        self._log("🚀 ProductionMultiCameraApp initialized")
    
    def _log(self, msg: str) -> None:
        """Log message"""
        if self.on_log:
            try:
                self.on_log(msg)
            except:
                pass
    
    def _handle_alert(self, camera_id: int, alert_type: str, is_active: bool, value: float) -> None:
        """Handle alert từ worker"""
        if is_active:
            if alert_type == "person":
                self._person_alerts += 1
            elif alert_type == "coal":
                self._coal_alerts += 1
        
        if self.on_alert:
            try:
                self.on_alert(camera_id, alert_type, is_active, value)
            except:
                pass
    
    def load_models(self) -> bool:
        """Load tất cả YOLO models từ config
        
        Returns:
            True nếu load thành công (ít nhất 1 model)
        """
        if self._models_loaded:
            return True
        
        try:
            self._log("🔄 Loading YOLO models...")
            
            results = self._model_loader.load_from_config(self.config)
            
            success_count = sum(1 for v in results.values() if v)
            
            if success_count == 0:
                self._log("❌ Không thể load model nào!")
                return False
            
            # Log model info
            for model_id, success in results.items():
                if success:
                    info = self._model_loader.get_model_info(model_id)
                    if info:
                        self._log(f"✅ {info.name}: cameras {info.cameras}")
            
            self._models_loaded = True
            self._log(f"✅ Đã load {success_count} model(s)")
            
            # Verify và hiển thị GPU status
            try:
                self._model_loader.print_gpu_status()
            except Exception as e:
                self._log(f"⚠️ Không thể check GPU status: {e}")
            
            return True
            
        except Exception as e:
            self._log(f"❌ Lỗi load model: {str(e)}")
            return False
    
    def _create_worker_config(self, cam: CameraConfig) -> WorkerConfig:
        """Tạo WorkerConfig từ CameraConfig"""
        return WorkerConfig(
            camera_id=int(cam.camera_id.split('_')[-1]) if '_' in cam.camera_id else 1,
            rtsp_url=cam.rtsp_url,
            camera_name=cam.name,
            enabled=cam.enabled,
            roi_person=list(cam.roi.roi_person),
            roi_coal=list(cam.roi.roi_coal),
            reference_resolution=cam.roi.reference_resolution,
            enable_person=True,
            enable_coal=cam.detection.coal_detection_enabled,
            person_consecutive_threshold=cam.detection.person_consecutive_threshold,
            person_no_detection_threshold=cam.detection.person_no_detection_threshold,
            coal_ratio_threshold=cam.detection.coal_ratio_threshold,
            coal_consecutive_threshold=cam.detection.coal_consecutive_threshold,
            coal_no_blockage_threshold=cam.detection.coal_no_blockage_threshold,
            detection_confidence=cam.detection.confidence_threshold,
            # PLC config (mỗi camera có PLC riêng)
            plc_ip=cam.plc.ip,
            plc_rack=cam.plc.rack,
            plc_slot=cam.plc.slot,
            plc_db_number=cam.plc.db_number,
            plc_person_byte=cam.plc.person_alarm_byte,
            plc_person_bit=cam.plc.person_alarm_bit,
            plc_coal_byte=cam.plc.coal_alarm_byte,
            plc_coal_bit=cam.plc.coal_alarm_bit,
            # Logging config
            logs_dir=self.config.logs_dir,
            artifacts_dir=self.config.artifacts_dir,
            target_capture_fps=cam.target_fps,
            detection_interval=0.5,  # 2 FPS detection
            buffer_size=1,  # Low latency
            enable_grab_pattern=True,  # Skip old frames
        )
    
    def start_all(self) -> Dict[int, bool]:
        """Khởi động tất cả camera
        
        Returns:
            Dict {camera_id: success}
        """
        if not self._models_loaded:
            if not self.load_models():
                return {}
        
        results = {}
        self._start_time = time.time()
        
        for i, cam in enumerate(self.config.cameras):
            cam_id = i + 1
            
            if not cam.enabled:
                self._log(f"⏸️ Camera {cam_id}: Đã tắt trong cấu hình")
                results[cam_id] = False
                continue
            
            # Get model for this camera
            model_info = self._model_loader.get_model_info_for_camera(cam_id)
            if not model_info:
                self._log(f"❌ Camera {cam_id}: Không tìm thấy model")
                results[cam_id] = False
                continue
            
            # Get model and lock
            model = self._model_loader._models.get(model_info.model_id)
            model_lock = self._model_loader._inference_locks.get(model_info.model_id)
            
            if not model or not model_lock:
                self._log(f"❌ Camera {cam_id}: Model chưa load")
                results[cam_id] = False
                continue
            
            # Create worker config
            config = self._create_worker_config(cam)
            
            # Create worker
            worker = OptimizedCameraWorker(
                config=config,
                model=model,
                model_lock=model_lock,
                model_id=model_info.model_id,
                on_alert=self._handle_alert,
                on_log=self._log,
            )
            
            # Set class IDs
            worker.person_class_id = model_info.person_class_id
            worker.coal_class_id = model_info.coal_class_id
            
            # Start worker
            if worker.start():
                self._workers[cam_id] = worker
                results[cam_id] = True
                self._log(f"✅ Camera {cam_id} ({model_info.name}): Đã khởi động")
            else:
                results[cam_id] = False
                self._log(f"❌ Camera {cam_id}: Không thể khởi động")
        
        self._stats.total_cameras = len(self.config.cameras)
        self._update_stats()
        
        success_count = sum(1 for v in results.values() if v)
        self._log(f"🎥 Đã khởi động {success_count}/{len(results)} cameras")
        
        return results
    
    def stop_all(self) -> None:
        """Dừng tất cả camera"""
        self._log("⏹️ Đang dừng tất cả cameras...")
        
        for cam_id, worker in self._workers.items():
            worker.stop()
        
        self._workers.clear()
        self._update_stats()
        self._log("✅ Đã dừng tất cả cameras")
    
    def get_worker(self, camera_id: int) -> Optional[OptimizedCameraWorker]:
        """Lấy worker theo camera ID"""
        return self._workers.get(camera_id)
    
    @property
    def workers(self) -> Dict[int, OptimizedCameraWorker]:
        """Dict của tất cả workers"""
        return self._workers
    
    @property
    def is_any_running(self) -> bool:
        """Có worker nào đang chạy không"""
        return any(w.is_running for w in self._workers.values())
    
    def _update_stats(self) -> None:
        """Cập nhật statistics"""
        self._stats.running_cameras = sum(1 for w in self._workers.values() if w.is_running)
        self._stats.total_person_alerts = self._person_alerts
        self._stats.total_coal_alerts = self._coal_alerts
        
        if self._start_time:
            self._stats.uptime_seconds = time.time() - self._start_time
        
        # Aggregate from workers
        total_frames = 0
        total_detections = 0
        for worker in self._workers.values():
            total_frames += worker._frame_count
            total_detections += worker._detection_count
        
        self._stats.total_frames = total_frames
        self._stats.total_detections = total_detections
    
    def get_stats(self) -> ProductionStats:
        """Lấy statistics"""
        self._update_stats()
        return self._stats
    
    def get_inference_stats(self) -> Dict[int, Dict[str, Any]]:
        """Lấy inference statistics cho tất cả cameras"""
        return self._inference_stats.get_all_stats()
    
    def print_inference_stats(self) -> None:
        """In inference statistics ra console"""
        self._inference_stats.print_stats()
    
    def get_camera_status(self, camera_id: int) -> Dict[str, Any]:
        """Lấy trạng thái camera"""
        worker = self._workers.get(camera_id)
        if not worker:
            return {
                "status": "not_found",
                "fps": 0,
                "coal_ratio": 0,
                "person_alarm": False,
                "coal_alarm": False,
            }
        
        return {
            "status": worker.status.value,
            "fps": worker.fps_display,
            "coal_ratio": worker.last_coal_ratio,
            "person_alarm": worker.person_alarm_active,
            "coal_alarm": worker.coal_alarm_active,
        }
    
    def get_all_status(self) -> Dict[int, Dict[str, Any]]:
        """Lấy trạng thái tất cả cameras"""
        result = {}
        for i in range(len(self.config.cameras)):
            cam_id = i + 1
            result[cam_id] = self.get_camera_status(cam_id)
        return result


# Alias cho compatibility
MultiCameraAppProduction = ProductionMultiCameraApp

