"""
Optimized Camera Worker Module
==============================

Worker tối ưu cho mỗi camera, dựa trên:
- coal_6cam_v1.py: CameraWorker pattern, atomic updates
- VidGear: Thread management, queue handling
- Multi-Camera-Live-Object-Tracking: Producer-consumer pattern

Key features:
1. Atomic frame update (không queue cho display)
2. Separate queue cho detection
3. Connection status tracking
4. Inference statistics
5. Thread-safe operations
"""

import threading
import time
import queue
import gc
import numpy as np
import cv2
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .inference_stats import get_stats_manager
from ..plc import PLCClient, AlarmManager, AlarmConfig, AlarmType
from ..alerting import AlertLogger, ImageSaver


class WorkerStatus(Enum):
    """Trạng thái worker"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class WorkerConfig:
    """Cấu hình worker"""
    camera_id: int
    rtsp_url: str
    camera_name: str = ""
    enabled: bool = True
    
    # ROI config
    roi_person: List[Tuple[int, int]] = field(default_factory=list)
    roi_coal: List[Tuple[int, int]] = field(default_factory=list)
    reference_resolution: Tuple[int, int] = (1920, 1080)
    
    # Detection config
    enable_person: bool = True
    enable_coal: bool = True
    person_consecutive_threshold: int = 3
    person_no_detection_threshold: int = 5
    coal_ratio_threshold: float = 73.0
    coal_consecutive_threshold: int = 5
    coal_no_blockage_threshold: int = 5
    detection_confidence: float = 0.7
    
    # PLC config (mỗi camera có PLC riêng)
    plc_ip: str = ""
    plc_rack: int = 0
    plc_slot: int = 2
    plc_db_number: int = 300
    plc_person_byte: int = 6
    plc_person_bit: int = 0
    plc_coal_byte: int = 6
    plc_coal_bit: int = 1
    
    # Logging config
    logs_dir: str = "logs"
    artifacts_dir: str = "artifacts"
    
    # Performance config
    target_capture_fps: int = 25
    detection_interval: float = 0.5  # seconds between detection
    buffer_size: int = 1
    enable_grab_pattern: bool = True


class OptimizedCameraWorker:
    """
    Worker tối ưu cho một camera
    
    Features:
    1. Low-latency capture với grab pattern
    2. Atomic frame update cho display (không dùng queue)
    3. Separate queue cho detection
    4. Inference time tracking
    5. ROI-based detection
    6. Automatic reconnection
    
    Usage:
        worker = OptimizedCameraWorker(
            config=WorkerConfig(...),
            model=yolo_model,
            model_lock=threading.Lock(),
            on_alert=lambda cam_id, type, active, val: print(f"Alert: {type}")
        )
        
        worker.start()
        
        # Trong GUI loop:
        frame = worker.get_display_frame()
        if frame is not None:
            display(frame)
        
        # Lấy result detection:
        result = worker.get_latest_result()
        
        worker.stop()
    """
    
    # Constants
    MAX_GRAB_COUNT = 3
    MIN_RECONNECT_INTERVAL = 0.5
    MAX_RECONNECT_INTERVAL = 10.0
    
    def __init__(
        self,
        config: WorkerConfig,
        model: Any,
        model_lock: threading.Lock,
        model_id: str = "default",
        on_alert: Optional[Callable[[int, str, bool, float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            config: WorkerConfig
            model: YOLO model object
            model_lock: Lock cho thread-safe inference
            model_id: ID của model đang sử dụng
            on_alert: Callback(camera_id, alert_type, is_active, value)
            on_log: Callback(message)
        """
        self.config = config
        self.model = model
        self.model_lock = model_lock
        self.model_id = model_id
        self.on_alert = on_alert
        self.on_log = on_log
        
        # IDs
        self.camera_id = config.camera_id
        
        # State
        self._status = WorkerStatus.STOPPED
        self._is_running = False
        self._stop_event = threading.Event()
        
        # Video capture
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self._video_info: Dict[str, Any] = {"width": 0, "height": 0, "fps": 0}
        
        # ===== ATOMIC FRAME STORAGE (key optimization) =====
        self._display_frame: Optional[np.ndarray] = None
        self._display_frame_lock = threading.Lock()
        
        # Detection queue (separate from display)
        self._detection_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # Latest result
        self._latest_result: Optional[Tuple[np.ndarray, Any, bool, float]] = None
        self._latest_result_lock = threading.Lock()
        
        # Threads
        self._capture_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None
        
        # Reconnection
        self._reconnect_interval = self.MIN_RECONNECT_INTERVAL
        self._last_reconnect_time = 0.0
        self._rtsp_fail_count = 0
        
        # Statistics
        self._frame_count = 0
        self._detection_count = 0
        self._fps_display = 0.0
        self._detection_fps_display = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_counter = 0
        self._fps_detection_counter = 0
        self._fps_counter_lock = threading.Lock()  # Thread-safe cho FPS counters
        
        # Detection state
        self._person_consecutive_count = 0
        self._person_no_detection_count = 0
        self._coal_consecutive_count = 0
        self._coal_no_blockage_count = 0
        self._last_coal_ratio = 0.0
        
        # Cache ROI scaled để tránh tính toán lại mỗi frame
        self._cached_roi_person: Optional[List[Tuple[int, int]]] = None
        self._cached_roi_coal: Optional[List[Tuple[int, int]]] = None
        self._cached_frame_size: Optional[Tuple[int, int]] = None
        
        # Alarm state
        self.person_alarm_active = False
        self.coal_alarm_active = False
        self.last_person_detected = False  # Track person detection state for display
        
        # Class IDs
        self.person_class_id = 0
        self.coal_class_id = 1
        
        # Inference stats manager
        self._stats_manager = get_stats_manager()
        
        # PLC client (mỗi camera có PLC riêng)
        self._plc_client: Optional[PLCClient] = None
        self._alarm_manager: Optional[AlarmManager] = None
        
        # Logging
        self._alert_logger: Optional[AlertLogger] = None
        self._image_saver: Optional[ImageSaver] = None
        
        # Initialize PLC và logging
        self._init_plc()
        self._init_logging()
    
    # ===== PROPERTIES =====
    
    @property
    def status(self) -> WorkerStatus:
        return self._status
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def fps_display(self) -> float:
        return self._fps_display
    
    @property
    def detection_fps(self) -> float:
        return self._detection_fps_display
    
    @property
    def last_coal_ratio(self) -> float:
        return self._last_coal_ratio
    
    @property
    def video_info(self) -> Dict[str, Any]:
        return self._video_info
    
    # ===== PUBLIC METHODS =====
    
    def _init_plc(self) -> None:
        """Khởi tạo PLC client (mỗi camera có PLC riêng)"""
        if not self.config.plc_ip:
            return
        
        try:
            self._plc_client = PLCClient(
                ip=self.config.plc_ip,
                rack=self.config.plc_rack,
                slot=self.config.plc_slot,
                on_state_change=lambda state: self._log(f"PLC Cam{self.camera_id}: {state.value}"),
                on_error=lambda msg: self._log(f"PLC Cam{self.camera_id} error: {msg}")
            )
            
            # Tạo AlarmConfig
            person_alarm_config = AlarmConfig()
            person_alarm_config.db_number = self.config.plc_db_number
            person_alarm_config.byte_offset = self.config.plc_person_byte
            person_alarm_config.bit_offset = self.config.plc_person_bit
            
            coal_alarm_config = AlarmConfig()
            coal_alarm_config.db_number = self.config.plc_db_number
            coal_alarm_config.byte_offset = self.config.plc_coal_byte
            coal_alarm_config.bit_offset = self.config.plc_coal_bit
            
            # Tạo AlarmManager
            self._alarm_manager = AlarmManager(
                plc_client=self._plc_client,
                person_alarm=person_alarm_config,
                coal_alarm=coal_alarm_config,
            )
            
            # Kết nối PLC
            if self._plc_client.connect():
                self._log(f"✅ Camera {self.camera_id}: Đã kết nối PLC {self.config.plc_ip}")
            else:
                self._log(f"⚠️ Camera {self.camera_id}: Không thể kết nối PLC")
        except Exception as e:
            self._log(f"❌ Camera {self.camera_id}: Lỗi khởi tạo PLC - {str(e)}")
    
    def _init_logging(self) -> None:
        """Khởi tạo logging"""
        try:
            camera_id_str = f"camera_{self.camera_id}"
            self._alert_logger = AlertLogger(
                logs_dir=self.config.logs_dir,
                camera_id=camera_id_str,
                camera_ip=self.config.rtsp_url,
                location=self.config.camera_name,
            )
            self._image_saver = ImageSaver(
                artifacts_dir=self.config.artifacts_dir,
                camera_id=camera_id_str,
            )
        except Exception as e:
            self._log(f"❌ Camera {self.camera_id}: Lỗi khởi tạo logging - {str(e)}")
    
    def start(self) -> bool:
        """Bắt đầu worker"""
        if self._is_running:
            return True
        
        if not self.config.rtsp_url:
            self._log(f"❌ Camera {self.camera_id}: Không có RTSP URL")
            return False
        
        self._is_running = True
        self._stop_event.clear()
        self._status = WorkerStatus.STARTING
        
        # Kết nối PLC nếu chưa
        if self._plc_client and not self._plc_client.is_connected:
            self._plc_client.connect()
        
        # Start threads
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"Capture-Cam{self.camera_id}"
        )
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name=f"Detect-Cam{self.camera_id}"
        )
        
        self._capture_thread.start()
        self._detection_thread.start()
        
        self._log(f"🔄 Camera {self.camera_id}: Đang kết nối...")
        return True
    
    def stop(self) -> None:
        """Dừng worker"""
        self._is_running = False
        self._stop_event.set()
        self._status = WorkerStatus.STOPPED
        
        # Wait for threads
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)
        
        # Release capture
        with self._cap_lock:
            if self._cap:
                self._cap.release()
                self._cap = None
        
        # Disconnect PLC
        if self._plc_client:
            self._plc_client.disconnect()
        
        # TỐI ƯU MEMORY: Clear all frames và results
        with self._display_frame_lock:
            self._display_frame = None
        with self._latest_result_lock:
            self._latest_result = None
        
        # Clear queues để tránh memory buildup
        while not self._detection_queue.empty():
            try:
                self._detection_queue.get_nowait()
            except:
                break
        
        # Reset state
        self.person_alarm_active = False
        self.coal_alarm_active = False
        self._person_consecutive_count = 0
        self._coal_consecutive_count = 0
        
        # TỐI ƯU MEMORY: Force cleanup GPU và CPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()  # Force Python garbage collection
        except:
            pass
        
        self._log(f"⏹️ Camera {self.camera_id}: Đã dừng")
    
    def get_display_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        """Lấy frame mới nhất cho display (atomic, thread-safe)
        
        Args:
            copy: True để trả về copy (an toàn hơn)
            
        Returns:
            Frame hoặc None
        """
        with self._display_frame_lock:
            if self._display_frame is None:
                return None
            return self._display_frame.copy() if copy else self._display_frame
    
    def get_latest_result(self) -> Optional[Tuple[np.ndarray, Any, bool, float]]:
        """Lấy result detection mới nhất
        
        Returns:
            (frame, yolo_result, coal_blocked, coal_ratio) hoặc None
        """
        with self._latest_result_lock:
            return self._latest_result
    
    def clear_result(self) -> None:
        """Xóa result sau khi đã xử lý"""
        with self._latest_result_lock:
            self._latest_result = None
    
    def update_fps(self) -> None:
        """Cập nhật FPS (gọi từ GUI thread) - Thread-safe"""
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        
        # Cập nhật FPS mỗi 2 giây
        if elapsed >= 2.0:
            with self._fps_counter_lock:
                # Đọc và reset counters (thread-safe)
                frame_count = self._fps_frame_counter
                detection_count = self._fps_detection_counter
                self._fps_frame_counter = 0
                self._fps_detection_counter = 0
            
            # Tính FPS (chính xác: số frame / thời gian)
            self._fps_display = frame_count / elapsed if elapsed > 0 else 0.0
            self._detection_fps_display = detection_count / elapsed if elapsed > 0 else 0.0
            self._last_fps_time = current_time
    
    # ===== PRIVATE METHODS =====
    
    def _log(self, message: str) -> None:
        """Log message qua callback"""
        if self.on_log:
            try:
                self.on_log(message)
            except:
                pass
    
    def _alert(self, alert_type: str, is_active: bool, value: float = 0) -> None:
        """Trigger alert callback"""
        if self.on_alert:
            try:
                self.on_alert(self.camera_id, alert_type, is_active, value)
            except:
                pass
    
    def _connect(self) -> bool:
        """Kết nối tới RTSP stream"""
        try:
            with self._cap_lock:
                # Release old
                if self._cap:
                    self._cap.release()
                    self._cap = None
                
                # Open new với FFMPEG backend
                self._cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
                
                if not self._cap.isOpened():
                    return False
                
                # ===== KEY OPTIMIZATION: Giảm buffer =====
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                # Get video info
                self._video_info = {
                    "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920),
                    "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080),
                    "fps": float(self._cap.get(cv2.CAP_PROP_FPS) or 25),
                }
            
            self._status = WorkerStatus.RUNNING
            self._rtsp_fail_count = 0
            self._reconnect_interval = self.MIN_RECONNECT_INTERVAL
            self._log(f"✅ Camera {self.camera_id}: Đã kết nối thành công")
            return True
            
        except Exception as e:
            self._log(f"❌ Camera {self.camera_id}: Lỗi kết nối - {str(e)}")
            return False
    
    def _reconnect(self) -> None:
        """Reconnect với exponential backoff"""
        current_time = time.time()
        
        # Tránh reconnect quá nhanh
        if current_time - self._last_reconnect_time < self._reconnect_interval:
            time.sleep(0.1)
            return
        
        self._last_reconnect_time = current_time
        self._status = WorkerStatus.RECONNECTING
        
        self._log(f"🔄 Camera {self.camera_id}: Đang kết nối lại... (chờ {self._reconnect_interval:.0f}s)")
        
        # Release old
        with self._cap_lock:
            if self._cap:
                self._cap.release()
                self._cap = None
        
        time.sleep(0.3)
        
        if self._connect():
            self._reconnect_interval = self.MIN_RECONNECT_INTERVAL
        else:
            # Exponential backoff
            self._reconnect_interval = min(
                self._reconnect_interval * 1.5,
                self.MAX_RECONNECT_INTERVAL
            )
    
    def _capture_loop(self) -> None:
        """Main capture loop - chạy trong thread riêng"""
        while not self._stop_event.is_set() and self._is_running:
            # Check/connect (không giữ lock khi gọi _connect)
            need_connect = False
            with self._cap_lock:
                need_connect = (self._cap is None or not self._cap.isOpened())
            
            if need_connect:
                if not self._connect():
                    time.sleep(self._reconnect_interval)
                    continue
            
            try:
                with self._cap_lock:
                    if self._cap is None:
                        continue
                    
                    # ===== KEY OPTIMIZATION: Grab pattern =====
                    # Skip frame cũ trong buffer
                    if self.config.enable_grab_pattern:
                        for _ in range(self.MAX_GRAB_COUNT - 1):
                            self._cap.grab()
                    
                    # Đọc frame mới nhất
                    ret, frame = self._cap.read()
                
                if ret and frame is not None:
                    # Update frame count
                    self._frame_count += 1
                    # Tăng FPS counter (thread-safe)
                    with self._fps_counter_lock:
                        self._fps_frame_counter += 1
                    self._rtsp_fail_count = 0
                    
                    if self._status != WorkerStatus.RUNNING:
                        self._status = WorkerStatus.RUNNING
                    
                    # Vẽ ROI lên frame ngay cả khi chưa có detection (để ROI luôn hiển thị)
                    # Tối ưu: cache ROI scaled, chỉ copy frame khi cần
                    frame_with_roi = self._draw_roi_on_frame_optimized_cached(frame)
                    
                    # ===== ATOMIC FRAME UPDATE (với ROI) =====
                    with self._display_frame_lock:
                        self._display_frame = frame_with_roi
                    
                    # Put to detection queue (không block)
                    try:
                        if self._detection_queue.full():
                            self._detection_queue.get_nowait()
                        self._detection_queue.put_nowait(frame)
                    except:
                        pass
                else:
                    self._rtsp_fail_count += 1
                    if self._rtsp_fail_count >= 3:
                        self._status = WorkerStatus.RECONNECTING
                        self._reconnect()
                        
            except Exception as e:
                self._rtsp_fail_count += 1
                if self._rtsp_fail_count >= 3:
                    self._reconnect()
            
            # FPS limiting
            time.sleep(max(0, 0.01))  # ~100 FPS max capture
    
    def _detection_loop(self) -> None:
        """Detection loop - chạy trong thread riêng"""
        detection_interval = self.config.detection_interval
        
        while not self._stop_event.is_set() and self._is_running:
            loop_start = time.time()
            
            # TỐI ƯU MEMORY: Lấy frame mới nhất từ queue (skip frames cũ để tránh memory buildup)
            frame = None
            frames_skipped = 0
            while not self._detection_queue.empty():
                try:
                    old_frame = self._detection_queue.get_nowait()
                    frames_skipped += 1
                    frame = old_frame  # Giữ frame mới nhất
                    # Frames cũ sẽ được GC tự động (không cần del explicit để tránh overhead)
                except:
                    break
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            try:
                # ===== YOLO INFERENCE với timing =====
                inference_start = time.time()
                
                with self.model_lock:
                    # TỐI ƯU: Đảm bảo model chạy trên GPU nếu có
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    results = self.model.predict(
                        frame,
                        conf=self.config.detection_confidence,
                        verbose=False,
                        task='segment',
                        device=device  # Explicit device để đảm bảo dùng GPU
                    )
                result = results[0] if results else None
                
                inference_time_ms = (time.time() - inference_start) * 1000
                
                # Log inference time và GPU usage (mỗi 20 lần log 1 lần để tránh spam)
                if self._detection_count % 20 == 0:
                    gpu_info = ""
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_mem = torch.cuda.memory_allocated() / 1024**2
                            gpu_info = f" | GPU: {gpu_mem:.0f}MB"
                    except:
                        pass
                    self._log(f"📊 Cam {self.camera_id}: Inference {inference_time_ms:.1f}ms on {device.upper()}{gpu_info}")
                
                # Record inference stats
                self._stats_manager.record_inference(
                    camera_id=self.camera_id,
                    inference_time_ms=inference_time_ms,
                    model_id=self.model_id
                )
                
                # Vẽ segment lên frame nếu có phát hiện (TRƯỚC để có frame đúng thời điểm khi cảnh báo)
                display_frame = self._draw_segments_on_frame(frame.copy(), result)
                
                # Detect coal blockage (cần display_frame để lưu đúng frame khi cảnh báo)
                coal_blocked, coal_ratio = self._detect_coal_blockage(frame, result, display_frame)
                
                # Detect person (cần display_frame để lưu đúng frame khi cảnh báo)
                person_detected = self._detect_person(frame, result)
                self.last_person_detected = person_detected
                self._update_person_alarm_state(person_detected, frame, display_frame)
                
                # Update display frame với segment
                with self._display_frame_lock:
                    self._display_frame = display_frame
                
                # Store result (cần giữ lại cho UI/log - KHÔNG release)
                with self._latest_result_lock:
                    self._latest_result = (frame.copy(), result, coal_blocked, coal_ratio)
                
                # TỐI ƯU MEMORY: Periodic cleanup để tránh memory leak (mỗi 10 lần)
                if self._detection_count % 14 == 0:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            # Clear unused GPU memory (không ảnh hưởng model đã cache)
                            torch.cuda.empty_cache()
                        # Force Python garbage collection
                        gc.collect()
                    except:
                        pass
                
                self._detection_count += 1
                # Tăng detection FPS counter (thread-safe)
                with self._fps_counter_lock:
                    self._fps_detection_counter += 1
                
            except Exception as e:
                print(f"Camera {self.camera_id} detection error: {e}")
            
            # Rate limiting
            elapsed = time.time() - loop_start
            sleep_time = max(0, detection_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _detect_coal_blockage(self, frame: np.ndarray, result: Any, display_frame: np.ndarray = None) -> Tuple[bool, float]:
        """Phát hiện tắc than"""
        if not self.config.enable_coal:
            return False, 0.0
        
        try:
            h, w = frame.shape[:2]
            roi_coal = self._scale_roi(self.config.roi_coal, w, h)
            
            if not roi_coal:
                return False, 0.0
            
            # Create ROI mask
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_polygon = np.array(roi_coal, dtype=np.int32)
            cv2.fillPoly(roi_mask, [roi_polygon], 255)
            roi_area = cv2.countNonZero(roi_mask)
            
            if roi_area == 0 or result is None or result.masks is None:
                return False, 0.0
            
            # Create coal mask
            coal_mask_total = np.zeros((h, w), dtype=np.uint8)
            
            if result.boxes is not None:
                boxes = result.boxes
                masks = result.masks
                
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    if cls_id == self.coal_class_id and masks is not None and i < len(masks.data):
                        # TỐI ƯU MEMORY: Move GPU tensor về CPU và release GPU reference
                        mask_tensor = masks.data[i]
                        mask_data = mask_tensor.cpu().numpy()  # Move to CPU
                        # Release GPU tensor reference (Python sẽ GC khi out of scope)
                        del mask_tensor
                        
                        mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        coal_mask_total = cv2.bitwise_or(coal_mask_total, mask_binary)
                        
                        # TỐI ƯU: Release intermediate arrays (sẽ được GC)
                        del mask_data, mask_resized, mask_binary
            
            # Calculate ratio
            intersection = cv2.bitwise_and(coal_mask_total, roi_mask)
            coal_area_in_roi = cv2.countNonZero(intersection)
            coal_ratio = (coal_area_in_roi / roi_area * 100) if roi_area > 0 else 0.0
            
            # TỐI ƯU MEMORY: Release intermediate arrays (sẽ được GC tự động)
            del intersection
            # Giữ lại coal_mask_total và roi_mask để dùng nếu cần, nhưng sẽ được GC sau khi return
            
            self._last_coal_ratio = coal_ratio
            
            # Logic đếm liên tiếp
            if coal_ratio >= self.config.coal_ratio_threshold:
                self._coal_no_blockage_count = 0
                self._coal_consecutive_count += 1
                
                if self._coal_consecutive_count >= self.config.coal_consecutive_threshold:
                    # Lần đầu: Bật alarm và gửi PLC
                    if not self.coal_alarm_active:
                        self.coal_alarm_active = True
                        self._alert("coal", True, coal_ratio)
                        # Gửi PLC ON
                        if self._alarm_manager:
                            self._alarm_manager.set_coal_alarm(True)
                    
                    # Mỗi lần đạt threshold đều lưu log/ảnh (theo yêu cầu)
                    if self._alert_logger and self._image_saver:
                        # Lưu log text
                        self._alert_logger.log_coal_alert(
                            coal_ratio=coal_ratio,
                            threshold=self.config.coal_ratio_threshold,
                            force=True
                        )
                        # Lưu ảnh
                        save_frame = display_frame if display_frame is not None else frame
                        if save_frame is not None:
                            self._image_saver.save_frame_direct(
                                frame=save_frame.copy(),
                                alert_type="coal_alert",
                                force=True
                            )
                    
                    # Reset counter sau khi xử lý
                    self._coal_consecutive_count = 0
                    return True, coal_ratio
            else:
                self._coal_no_blockage_count += 1
                if self._coal_no_blockage_count >= self.config.coal_no_blockage_threshold:
                    # Reset cả 2 counter khi tắt cảnh báo
                    self._coal_consecutive_count = 0
                    if self.coal_alarm_active:
                        self.coal_alarm_active = False
                        self._alert("coal", False, coal_ratio)
                        # Tắt PLC alarm
                        self._handle_coal_alarm(frame, coal_ratio, False)
                    # Reset counter "no blockage" sau khi tắt cảnh báo để logic sạch sẽ
                    self._coal_no_blockage_count = 0
            
            return False, coal_ratio
            
        except Exception as e:
            print(f"Camera {self.camera_id} coal detection error: {e}")
            return False, 0.0
    
    def _update_person_alarm_state(self, person_detected: bool, frame: np.ndarray, display_frame: np.ndarray = None) -> None:
        """Cập nhật trạng thái alarm người"""
        if not self.config.enable_person:
            return
        
        if person_detected:
            self._person_no_detection_count = 0
            self._person_consecutive_count += 1
            
            if self._person_consecutive_count >= self.config.person_consecutive_threshold:
                # Lần đầu: Bật alarm và gửi PLC
                if not self.person_alarm_active:
                    self.person_alarm_active = True
                    self._alert("person", True, 0)
                    # Gửi PLC ON
                    if self._alarm_manager:
                        self._alarm_manager.set_person_alarm(True)
                
                # Mỗi lần đạt threshold đều lưu log/ảnh (theo yêu cầu)
                if self._alert_logger and self._image_saver:
                    # Lưu log text
                    self._alert_logger.log_person_alert(
                        frames_detected=self._person_consecutive_count,
                        threshold=self.config.person_consecutive_threshold,
                        force=True
                    )
                    # Lưu ảnh
                    save_frame = display_frame if display_frame is not None else frame
                    if save_frame is not None:
                        self._image_saver.save_frame_direct(
                            frame=save_frame.copy(),
                            alert_type="person_alert",
                            force=True
                        )
                
                # Reset counter sau khi xử lý
                self._person_consecutive_count = 0
        else:
            self._person_no_detection_count += 1
            if self._person_no_detection_count >= self.config.person_no_detection_threshold:
                # Reset cả 2 counter khi tắt cảnh báo
                self._person_consecutive_count = 0
                if self.person_alarm_active:
                    self.person_alarm_active = False
                    self._alert("person", False, 0)
                    # Tắt PLC alarm
                    self._handle_person_alarm(frame, False)
                # Reset counter "no detection" sau khi tắt cảnh báo để logic sạch sẽ
                self._person_no_detection_count = 0
    
    def _detect_person(self, frame: np.ndarray, result: Any) -> bool:
        """Phát hiện người trong ROI"""
        if not self.config.enable_person:
            return False
        
        try:
            h, w = frame.shape[:2]
            roi_person = self._scale_roi(self.config.roi_person, w, h)
            
            if not roi_person or result is None or result.boxes is None:
                return False
            
            # Create ROI mask
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_polygon = np.array(roi_person, dtype=np.int32)
            cv2.fillPoly(roi_mask, [roi_polygon], 255)
            
            boxes = result.boxes
            masks = result.masks
            
            # Check if any person is in ROI
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id == self.person_class_id:
                    if masks is not None and i < len(masks.data):
                        # TỐI ƯU MEMORY: Move GPU tensor về CPU và release GPU reference
                        mask_tensor = masks.data[i]
                        mask_data = mask_tensor.cpu().numpy()  # Move to CPU
                        # Release GPU tensor reference
                        del mask_tensor
                        
                        mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        intersection = cv2.bitwise_and(mask_binary, roi_mask)
                        
                        # Early exit nếu tìm thấy (tránh xử lý thêm)
                        person_found = cv2.countNonZero(intersection) > 0
                        
                        # TỐI ƯU MEMORY: Release intermediate arrays trước khi return
                        del mask_data, mask_resized, mask_binary, intersection
                        
                        if person_found:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Camera {self.camera_id} person detection error: {e}")
            return False
    
    def _draw_segments_on_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """Vẽ segment người và than lên frame (tham khảo coal_12_12_v1.py)"""
        if result is None or result.masks is None or result.boxes is None:
            # Vẫn vẽ ROI ngay cả khi không có detection
            return self._draw_roi_on_frame(frame)
        
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        boxes = result.boxes
        masks = result.masks
        
        # Vẽ segment cho từng detection
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            
            if i >= len(masks.data):
                continue
            
            # Lấy bounding box
            x1, y1, x2, y2 = boxes.xyxy[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(boxes.conf[i])
            
            # TỐI ƯU MEMORY: Lấy mask và release GPU tensor ngay
            mask_tensor = masks.data[i]
            mask_data = mask_tensor.cpu().numpy()  # Move to CPU
            # Release GPU tensor reference
            del mask_tensor
            
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # TỐI ƯU: Release intermediate arrays ngay sau khi dùng
            del mask_data, mask_resized
            
            # Tính contour một lần để dùng cho cả bounding box và vẽ viền
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Màu theo class
            if cls_id == self.person_class_id:
                # Người: vẽ bounding box và segment
                color = (0, 255, 0)  # Xanh lá
                alpha = 0.3
                
                # Tính bounding box từ mask để đảm bảo khớp với segment
                if contours:
                    # Lấy contour lớn nhất (object chính)
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Tính bounding rect từ mask thực tế (đảm bảo khớp với segment)
                    bx, by, bw, bh = cv2.boundingRect(largest_contour)
                    # Vẽ bounding box từ mask
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                    del largest_contour
                else:
                    # Fallback: dùng bounding box từ YOLO nếu không tìm thấy contour
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
            elif cls_id == self.coal_class_id:
                # Than: CHỈ vẽ segment (không vẽ bounding box)
                color = (0, 0, 255)  # Đỏ
                alpha = 0.3
            else:
                del mask_binary, contours  # Release nếu không dùng
                continue
            
            # Tô màu overlay cho segment
            colored_mask = np.zeros_like(display_frame)
            colored_mask[mask_binary > 0] = color
            # Blend segment với alpha
            mask_3channel = np.stack([mask_binary, mask_binary, mask_binary], axis=2) / 255.0
            display_frame = (display_frame * (1 - mask_3channel * alpha) + colored_mask * (mask_3channel * alpha)).astype(np.uint8)
            
            # Vẽ viền contour cho segment (dùng contours đã tính ở trên)
            if contours:
                cv2.drawContours(display_frame, contours, -1, color, 2)
            
            # TỐI ƯU MEMORY: Release intermediate arrays sau khi vẽ xong
            # (Arrays sẽ được GC tự động khi out of scope)
            del colored_mask, mask_3channel, mask_binary, contours
        
        # Vẽ ROI lên frame (sau khi vẽ segment)
        display_frame = self._draw_roi_on_frame(display_frame)
        
        return display_frame
    
    def _draw_roi_on_frame_optimized_cached(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ ROI lên frame (tối ưu - cache ROI scaled, chỉ copy frame)"""
        h, w = frame.shape[:2]
        current_size = (w, h)
        
        # Kiểm tra xem có cần tính lại ROI không (khi frame size thay đổi)
        if self._cached_frame_size != current_size:
            # Tính lại và cache ROI scaled
            self._cached_roi_person = self._scale_roi(self.config.roi_person, w, h)
            self._cached_roi_coal = self._scale_roi(self.config.roi_coal, w, h)
            self._cached_frame_size = current_size
        
        # Copy frame (cần thiết để không ảnh hưởng frame gốc)
        display_frame = frame.copy()
        
        # Vẽ ROI người (màu vàng) - dùng cached (không cần tính lại)
        if self._cached_roi_person and len(self._cached_roi_person) >= 3:
            pts = np.array(self._cached_roi_person, dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
        
        # Vẽ ROI than (màu đỏ) - dùng cached (không cần tính lại)
        if self._cached_roi_coal and len(self._cached_roi_coal) >= 3:
            pts = np.array(self._cached_roi_coal, dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
        
        return display_frame
    
    def _draw_roi_on_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ ROI lên frame (copy frame) - dùng cho detection loop"""
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Scale ROI theo kích thước frame
        roi_person = self._scale_roi(self.config.roi_person, w, h)
        roi_coal = self._scale_roi(self.config.roi_coal, w, h)
        
        # Vẽ ROI người (màu vàng)
        if roi_person and len(roi_person) >= 3:
            pts = np.array(roi_person, dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
        
        # Vẽ ROI than (màu đỏ)
        if roi_coal and len(roi_coal) >= 3:
            pts = np.array(roi_coal, dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
        
        return display_frame
    
    def _draw_roi_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ ROI người và than lên frame (dùng cho detection loop)"""
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Scale ROI theo kích thước frame
        roi_person = self._scale_roi(self.config.roi_person, w, h)
        roi_coal = self._scale_roi(self.config.roi_coal, w, h)
        
        # Vẽ ROI người (màu vàng)
        if roi_person and len(roi_person) >= 3:
            pts = np.array(roi_person, dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
        
        # Vẽ ROI than (màu đỏ)
        if roi_coal and len(roi_coal) >= 3:
            pts = np.array(roi_coal, dtype=np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
        
        return display_frame
    
    def _scale_roi(self, roi_points: List[Tuple[int, int]], 
                   target_w: int, target_h: int) -> List[Tuple[int, int]]:
        """Scale ROI theo kích thước frame"""
        if not roi_points:
            return roi_points
        
        ref_w, ref_h = self.config.reference_resolution
        scale_x = target_w / ref_w
        scale_y = target_h / ref_h
        
        return [(int(x * scale_x), int(y * scale_y)) for (x, y) in roi_points]
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Lấy statistics dạng dict"""
        return {
            "camera_id": self.camera_id,
            "status": self._status.value,
            "frame_count": self._frame_count,
            "detection_count": self._detection_count,
            "fps_display": round(self._fps_display, 1),
            "detection_fps": round(self._detection_fps_display, 1),
            "coal_ratio": round(self._last_coal_ratio, 1),
            "person_alarm": self.person_alarm_active,
            "coal_alarm": self.coal_alarm_active,
            "video_info": self._video_info,
        }
    
    def _handle_coal_alarm(self, frame: np.ndarray, coal_ratio: float, is_active: bool) -> None:
        """Xử lý cảnh báo than: gửi PLC và lưu log/ảnh"""
        try:
            # Gửi PLC
            if self._alarm_manager:
                if is_active:
                    self._alarm_manager.set_coal_alarm(True)
                else:
                    self._alarm_manager.set_coal_alarm(False)
            
            # Lưu log và ảnh
            if is_active and self._alert_logger and self._image_saver:
                # Lưu log text
                self._alert_logger.log_coal_alert(
                    coal_ratio=coal_ratio,
                    threshold=self.config.coal_ratio_threshold,
                    force=True
                )
                
                # Lưu ảnh: frame đã được truyền vào (đã có ROI và segment tại đúng thời điểm cảnh báo)
                if frame is not None:
                    # Lưu frame trực tiếp, không vẽ thêm gì (frame đã có ROI + segment)
                    self._image_saver.save_frame_direct(
                        frame=frame.copy(),  # Copy để an toàn
                        alert_type="coal_alert",
                        force=True
                    )
        except Exception as e:
            self._log(f"❌ Camera {self.camera_id}: Lỗi xử lý coal alarm - {str(e)}")
    
    def _handle_person_alarm(self, frame: np.ndarray, is_active: bool) -> None:
        """Xử lý cảnh báo người: gửi PLC và lưu log/ảnh"""
        try:
            # Gửi PLC
            if self._alarm_manager:
                if is_active:
                    self._alarm_manager.set_person_alarm(True)
                else:
                    self._alarm_manager.set_person_alarm(False)
            
            # Lưu log và ảnh
            if is_active and self._alert_logger and self._image_saver:
                # Lưu log text
                self._alert_logger.log_person_alert(
                    frames_detected=self._person_consecutive_count,
                    threshold=self.config.person_consecutive_threshold,
                    force=True
                )
                
                # Lưu ảnh: frame đã được truyền vào (đã có ROI và segment tại đúng thời điểm cảnh báo)
                if frame is not None:
                    # Lưu frame trực tiếp, không vẽ thêm gì (frame đã có ROI + segment)
                    self._image_saver.save_frame_direct(
                        frame=frame.copy(),  # Copy để an toàn
                        alert_type="person_alert",
                        force=True
                    )
        except Exception as e:
            self._log(f"❌ Camera {self.camera_id}: Lỗi xử lý person alarm - {str(e)}")

