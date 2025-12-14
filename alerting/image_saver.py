"""
Image Saver Module
==================

Lưu ảnh cảnh báo với thông tin chi tiết.

Features:
- Tự động tạo thư mục theo ngày
- Vẽ ROI và thông tin lên ảnh
- Thread-safe
- Throttling để tránh spam
"""

import os
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any


class ImageSaver:
    """
    Lưu ảnh cảnh báo với thông tin chi tiết
    
    Features:
    - Tự động tạo thư mục theo ngày
    - Vẽ ROI và thông tin lên ảnh
    - Thread-safe
    - Throttling
    
    Usage:
        saver = ImageSaver(
            artifacts_dir="artifacts",
            camera_id="camera_1",
            throttle_interval=5.0
        )
        
        # Lưu ảnh cảnh báo người
        saver.save_person_alert(
            frame=frame,
            roi_person=[(x1,y1), ...],
            consecutive_count=3
        )
        
        # Lưu ảnh cảnh báo than
        saver.save_coal_alert(
            frame=frame,
            roi_coal=[(x1,y1), ...],
            coal_ratio=85.5,
            threshold=73.0
        )
    """
    
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        camera_id: str = "camera_1",
        throttle_interval: float = 5.0,
        draw_roi: bool = True,
        draw_info: bool = True,
    ):
        """
        Args:
            artifacts_dir: Thư mục gốc chứa ảnh
            camera_id: ID của camera
            throttle_interval: Khoảng thời gian tối thiểu giữa các lần lưu (giây)
            draw_roi: Có vẽ ROI lên ảnh không
            draw_info: Có vẽ thông tin lên ảnh không
        """
        self.artifacts_dir = artifacts_dir
        self.camera_id = camera_id
        self.throttle_interval = throttle_interval
        self.draw_roi = draw_roi
        self.draw_info = draw_info
        
        self._lock = threading.Lock()
        self._last_save_time: Dict[str, float] = {}  # {alert_type: timestamp}
        self._save_count: Dict[str, int] = {}  # {alert_type: count}
        
        # Tạo thư mục
        os.makedirs(artifacts_dir, exist_ok=True)
    
    def _get_daily_dir(self) -> str:
        """Lấy thư mục theo ngày và camera"""
        day = datetime.now().strftime("%Y%m%d")
        day_dir = os.path.join(self.artifacts_dir, day)
        # Tạo folder cho camera này (ví dụ: camera_1, camera_2, ...)
        camera_dir = os.path.join(day_dir, self.camera_id)
        os.makedirs(camera_dir, exist_ok=True)
        return camera_dir
    
    def _should_save(self, alert_type: str) -> bool:
        """Kiểm tra có nên lưu không (throttling)"""
        current_time = time.time()
        
        with self._lock:
            last_time = self._last_save_time.get(alert_type, 0)
            
            if current_time - last_time >= self.throttle_interval:
                self._last_save_time[alert_type] = current_time
                return True
            
            return False
    
    def _generate_filename(self, alert_type: str) -> str:
        """Tạo tên file duy nhất"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{alert_type}_{self.camera_id}_{ts}.jpg"
    
    def _draw_roi_on_frame(
        self,
        frame: np.ndarray,
        roi_person: Optional[List[Tuple[int, int]]] = None,
        roi_coal: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Vẽ ROI lên frame"""
        if not self.draw_roi:
            return frame
        
        result = frame.copy()
        
        if roi_person:
            pts = np.array(roi_person, np.int32)
            cv2.polylines(result, [pts], True, (0, 255, 255), 3)  # Vàng
        
        if roi_coal:
            pts = np.array(roi_coal, np.int32)
            cv2.polylines(result, [pts], True, (0, 0, 255), 3)  # Đỏ
        
        return result
    
    def _draw_info_on_frame(
        self,
        frame: np.ndarray,
        title: str,
        info_lines: List[str],
        border_color: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """Vẽ thông tin lên frame"""
        if not self.draw_info:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Vẽ title
        cv2.putText(
            result, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2
        )
        
        # Vẽ các dòng info
        y_offset = 60
        for line in info_lines:
            cv2.putText(
                result, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_offset += 30
        
        # Vẽ khung border
        cv2.rectangle(result, (0, 0), (w-1, h-1), border_color, 3)
        
        return result
    
    def save_frame(
        self,
        frame: np.ndarray,
        filename: str,
        force: bool = False,
    ) -> Optional[str]:
        """Lưu frame raw
        
        Args:
            frame: Frame video (numpy array BGR)
            filename: Tên file
            force: Bỏ qua throttling
            
        Returns:
            Đường dẫn file đã lưu hoặc None
        """
        if frame is None:
            return None
        
        try:
            save_dir = self._get_daily_dir()
            filepath = os.path.join(save_dir, filename)
            
            with self._lock:
                success = cv2.imwrite(filepath, frame)
            
            if success:
                return filepath
            
        except Exception as e:
            print(f"Lỗi lưu ảnh: {e}")
        
        return None
    
    def save_person_alert(
        self,
        frame: np.ndarray,
        roi_person: Optional[List[Tuple[int, int]]] = None,
        roi_coal: Optional[List[Tuple[int, int]]] = None,
        consecutive_count: int = 0,
        force: bool = False,
    ) -> Optional[str]:
        """Lưu ảnh cảnh báo người
        
        Args:
            frame: Frame video (numpy array BGR)
            roi_person: ROI vùng nguy hiểm
            roi_coal: ROI vùng than
            consecutive_count: Số frame liên tiếp
            force: Bỏ qua throttling
            
        Returns:
            Đường dẫn file đã lưu hoặc None
        """
        alert_type = "person_alert"
        
        if not force and not self._should_save(alert_type):
            return None
        
        if frame is None:
            return None
        
        try:
            # Vẽ ROI
            result = self._draw_roi_on_frame(frame, roi_person, roi_coal)
            
            # Vẽ thông tin
            timestamp = datetime.now().strftime("%H:%M:%S")
            result = self._draw_info_on_frame(
                result,
                title="PERSON ALERT - DANGER ZONE",
                info_lines=[
                    f"Consecutive frames: {consecutive_count}",
                    f"Time: {timestamp}",
                    f"Camera: {self.camera_id}",
                ],
                border_color=(0, 0, 255),  # Đỏ
            )
            
            # Lưu
            filename = self._generate_filename(alert_type)
            filepath = self.save_frame(result, filename, force=True)
            
            if filepath:
                self._save_count[alert_type] = self._save_count.get(alert_type, 0) + 1
            
            return filepath
            
        except Exception as e:
            print(f"Lỗi lưu ảnh person alert: {e}")
            return None
    
    def save_coal_alert(
        self,
        frame: np.ndarray,
        roi_person: Optional[List[Tuple[int, int]]] = None,
        roi_coal: Optional[List[Tuple[int, int]]] = None,
        coal_ratio: float = 0.0,
        threshold: float = 73.0,
        force: bool = False,
    ) -> Optional[str]:
        """Lưu ảnh cảnh báo tắc than
        
        Args:
            frame: Frame video (numpy array BGR)
            roi_person: ROI vùng nguy hiểm
            roi_coal: ROI vùng than
            coal_ratio: Tỷ lệ than đo được (%)
            threshold: Ngưỡng tỷ lệ
            force: Bỏ qua throttling
            
        Returns:
            Đường dẫn file đã lưu hoặc None
        """
        alert_type = "coal_alert"
        
        if not force and not self._should_save(alert_type):
            return None
        
        if frame is None:
            return None
        
        try:
            # Vẽ ROI
            result = self._draw_roi_on_frame(frame, roi_person, roi_coal)
            
            # Vẽ thông tin
            timestamp = datetime.now().strftime("%H:%M:%S")
            result = self._draw_info_on_frame(
                result,
                title="COAL BLOCKAGE ALERT",
                info_lines=[
                    f"Coal Ratio: {coal_ratio:.2f}%",
                    f"Threshold: {threshold:.1f}%",
                    f"Time: {timestamp}",
                    f"Camera: {self.camera_id}",
                ],
                border_color=(0, 0, 255),  # Đỏ
            )
            
            # Lưu
            filename = self._generate_filename(alert_type)
            filepath = self.save_frame(result, filename, force=True)
            
            if filepath:
                self._save_count[alert_type] = self._save_count.get(alert_type, 0) + 1
            
            return filepath
            
        except Exception as e:
            print(f"Lỗi lưu ảnh coal alert: {e}")
            return None
    
    def get_save_stats(self) -> Dict[str, Any]:
        """Lấy thống kê lưu ảnh"""
        return {
            "artifacts_dir": self.artifacts_dir,
            "camera_id": self.camera_id,
            "save_count": dict(self._save_count),
            "throttle_interval": self.throttle_interval,
        }

