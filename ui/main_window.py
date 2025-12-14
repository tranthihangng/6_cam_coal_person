"""
Main Window Module - 6 Camera Monitoring
=========================================

Giao diện giám sát 6 camera với cấu hình đầy đủ.
Threading model học từ coal_6cam_v1.py.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import Optional, Dict, List
import threading
import datetime
import time
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageTk

from ..config import SystemConfig, CameraConfig, save_config, load_config


# ==================== Colors ====================
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_panel': '#16213e',
    'bg_video': '#0f0f23',
    'accent': '#e94560',
    'success': '#27ae60',
    'error': '#e74c3c',
    'warning': '#f39c12',
    'info': '#3498db',
    'text_white': '#ffffff',
    'text_gray': '#888888',
    'text_light': '#bdc3c7',
}

FONT = 'Segoe UI'


class CameraPanel(tk.Frame):
    """Panel hiển thị 1 camera - kích thước cố định"""
    
    def __init__(self, parent, cam_id: int, cam_name: str, width=420, height=280):
        super().__init__(parent, bg=COLORS['bg_panel'], width=width, height=height,
                        highlightthickness=0)
        self.pack_propagate(False)
        
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.width = width
        self.height = height
        
        # Frame storage (như coal_6cam_v1.py)
        self._display_frame = None
        self._frame_lock = threading.Lock()
        self._photo = None
        
        # State
        self.connection_status = "offline"
        self.fps = 0.0
        self.coal_ratio = 0.0
        self.person_detected = False
        self.person_alarm = False
        self.coal_alarm = False
        
        # ROI data (để vẽ)
        self.roi_person = []
        self.roi_coal = []
        self.roi_reference = (1920, 1080)
        
        self._create_ui()
    
    def _create_ui(self):
        """Tạo UI"""
        # Header
        header = tk.Frame(self, bg=COLORS['bg_panel'], height=24)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"📹 Camera {self.cam_id}", font=(FONT, 9, 'bold'),
                bg=COLORS['bg_panel'], fg=COLORS['accent']).pack(side=tk.LEFT, padx=5)
        
        self.status_lbl = tk.Label(header, text="⚪ Offline", font=(FONT, 8),
                                   bg=COLORS['bg_panel'], fg=COLORS['text_gray'])
        self.status_lbl.pack(side=tk.RIGHT, padx=5)
        
        # Video
        self.video_lbl = tk.Label(self, text=f"Camera {self.cam_id}\n{self.cam_name}\nChưa kết nối",
                                  font=(FONT, 10), bg=COLORS['bg_video'], fg=COLORS['text_gray'])
        self.video_lbl.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Footer
        footer = tk.Frame(self, bg=COLORS['bg_panel'], height=20)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)
        
        self.info_lbl = tk.Label(footer, text="FPS: -- | Than: --% | Người: --", font=(FONT, 8),
                                 bg=COLORS['bg_panel'], fg=COLORS['text_gray'])
        self.info_lbl.pack(side=tk.LEFT, padx=5)
    
    def set_roi(self, roi_person: list, roi_coal: list, reference: tuple):
        """Set ROI để vẽ"""
        self.roi_person = roi_person
        self.roi_coal = roi_coal
        self.roi_reference = reference
    
    def update_frame(self, frame: np.ndarray):
        """Cập nhật frame (thread-safe)"""
        with self._frame_lock:
            self._display_frame = frame
    
    def update_stats(self, fps: float, coal_ratio: float, person_detected: bool = False):
        """Cập nhật thống kê"""
        self.fps = fps
        self.coal_ratio = coal_ratio
        self.person_detected = person_detected
    
    def set_alarm(self, person: bool, coal: bool):
        """Set trạng thái alarm"""
        self.person_alarm = person
        self.coal_alarm = coal
        
        if person or coal:
            self.set_status("alarm")
            self.config(highlightbackground=COLORS['error'], highlightthickness=3)
        else:
            if self.connection_status == "alarm":
                self.set_status("online")
            self.config(highlightthickness=0)
    
    def set_status(self, status: str):
        """Set trạng thái kết nối"""
        self.connection_status = status
        status_map = {
            "online": ("🟢 Online", COLORS['success']),
            "connecting": ("🟡 Kết nối...", COLORS['warning']),
            "reconnecting": ("🟡 Reconnect...", COLORS['warning']),
            "alarm": ("🔴 ALARM!", COLORS['error']),
            "offline": ("⚪ Offline", COLORS['text_gray']),
        }
        text, color = status_map.get(status, ("⚪ Offline", COLORS['text_gray']))
        self.status_lbl.config(text=text, fg=color)
    
    def process_display(self):
        """Xử lý và hiển thị frame (gọi từ main thread) - tham khảo coal_12_12_v1.py"""
        with self._frame_lock:
            if self._display_frame is None:
                return
            frame = self._display_frame.copy()
            self._display_frame = None
        
        try:
            # Kiểm tra frame có hợp lệ không
            if frame is None:
                return
            
            # Lấy kích thước thực tế của widget (không hardcode)
            self.video_lbl.update_idletasks()
            parent = self.video_lbl.master  # CameraPanel frame
            parent.update_idletasks()
            
            # Lấy kích thước khung hình camera (trừ padding)
            max_width = parent.winfo_width() - 8  # Trừ padding
            max_height = parent.winfo_height() - 48  # Trừ header và footer
            
            # Nếu chưa có kích thước hợp lệ, dùng giá trị từ config
            if max_width < 100 or max_height < 100:
                max_width = self.width - 8
                max_height = self.height - 48
            
            # Lấy kích thước gốc của frame
            h, w = frame.shape[:2]
            
            # Kiểm tra frame có hợp lệ không
            if w <= 0 or h <= 0:
                return
            
            # Tính tỷ lệ scale để giữ nguyên aspect ratio
            scale_x = max_width / w
            scale_y = max_height / h
            scale = min(scale_x, scale_y)  # Lấy scale nhỏ hơn để fit hoàn toàn
            
            # Giới hạn scale để tránh quá nhỏ hoặc quá lớn
            scale = max(0.1, min(scale, 3.0))
            
            # Tính kích thước mới
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Đảm bảo kích thước mới không quá nhỏ
            if new_w < 10 or new_h < 10:
                new_w = max(10, new_w)
                new_h = max(10, new_h)
            
            # Resize frame với interpolation tốt
            if scale < 1.0:
                # Thu nhỏ - dùng INTER_AREA để tránh aliasing
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                # Phóng to - dùng INTER_LINEAR
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Tạo canvas với video ở giữa (nền đen)
            canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            
            # Căn giữa video trong khung hình
            y_off = (max_height - new_h) // 2
            x_off = (max_width - new_w) // 2
            
            # Đảm bảo offset không âm
            y_off = max(0, y_off)
            x_off = max(0, x_off)
            
            # Đặt video đã resize vào giữa canvas
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = frame_resized
            
            # Convert
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_lbl.configure(image=self._photo, text='')
            
            # Info
            person_text = "Có" if self.person_detected else "Không"
            self.info_lbl.config(text=f"FPS: {self.fps:.0f} | Than: {self.coal_ratio:.1f}% | Người: {person_text}")
            
        except Exception as e:
            pass
    
    def _draw_overlay(self, frame, orig_w, orig_h, new_w, new_h, x_off, y_off):
        """Vẽ ROI lên frame (nếu chưa có từ worker)"""
        try:
            # ROI đã được vẽ trong worker, không cần vẽ lại ở đây
            # Chỉ giữ hàm này để tương thích
            pass
        except:
            pass


class MainWindow:
    """Cửa sổ chính với tabs cấu hình"""
    
    def __init__(self, root: tk.Tk, config: SystemConfig, config_path: str = "system_config.json"):
        self.root = root
        self.config = config
        self.config_path = config_path
        
        # Window
        self.root.title(config.app_name)
        self.root.geometry("1600x920")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # App - sẽ import khi cần
        self._app = None
        self._camera_panels: Dict[str, CameraPanel] = {}
        
        # State
        self._is_monitoring = False
        self._start_time = None
        
        # Config widgets (để lưu giá trị)
        self._config_widgets = {}
        
        # Create UI
        self._create_ui()
        
        # Auto-start monitoring
        self.root.after(1000, self._auto_start_monitoring)
        
        # GUI loop
        self._update_loop()
    
    def _create_ui(self):
        """Tạo UI chính"""
        # Header
        self._create_header()
        
        # Tab bar
        self._create_tab_bar()
        
        # Tab frames
        self._tab_frames = {}
        self._main_container = tk.Frame(self.root, bg=COLORS['bg_dark'])
        self._main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._create_monitor_tab()
        self._create_camera_config_tab()
        self._create_detection_config_tab()
        self._create_plc_config_tab()
        self._create_system_tab()
        
        self._show_tab("monitor")
    
    def _create_header(self):
        """Header"""
        header = tk.Frame(self.root, bg=COLORS['bg_panel'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"🎥 {self.config.app_name}", font=(FONT, 16, 'bold'),
                bg=COLORS['bg_panel'], fg=COLORS['accent']).pack(side=tk.LEFT, padx=15, pady=10)
        
        self._global_status = tk.Label(header, text="⚪ Chưa khởi động", font=(FONT, 11),
                                       bg=COLORS['bg_panel'], fg=COLORS['text_white'])
        self._global_status.pack(side=tk.RIGHT, padx=15)
        
        self._time_lbl = tk.Label(header, text="", font=(FONT, 10),
                                  bg=COLORS['bg_panel'], fg=COLORS['text_light'])
        self._time_lbl.pack(side=tk.RIGHT, padx=15)
    
    def _create_tab_bar(self):
        """Tab bar"""
        bar = tk.Frame(self.root, bg=COLORS['bg_dark'], height=38)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        
        self._tabs = {}
        tabs = [
            ("monitor", "📺 Giám sát"),
            ("cameras", "📹 Cấu hình Camera"),
            ("detection", "🎯 Cấu hình Detection"),
            ("plc", "🔌 Cấu hình PLC"),
            ("system", "⚙️ Hệ thống"),
        ]
        
        for tid, name in tabs:
            btn = tk.Button(bar, text=name, font=(FONT, 9),
                           bg=COLORS['bg_panel'] if tid == "monitor" else COLORS['bg_dark'],
                           fg=COLORS['accent'] if tid == "monitor" else COLORS['text_gray'],
                           relief=tk.FLAT, padx=12, pady=5,
                           command=lambda t=tid: self._switch_tab(t))
            btn.pack(side=tk.LEFT, padx=2, pady=3)
            self._tabs[tid] = btn
        
        self._current_tab = "monitor"
    
    def _switch_tab(self, tid: str):
        """Chuyển tab"""
        for t, btn in self._tabs.items():
            btn.config(bg=COLORS['bg_panel'] if t == tid else COLORS['bg_dark'],
                      fg=COLORS['accent'] if t == tid else COLORS['text_gray'])
        self._current_tab = tid
        self._show_tab(tid)
    
    def _show_tab(self, tid: str):
        """Hiển thị tab"""
        for t, frame in self._tab_frames.items():
            if t == tid:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()
    
    # ==================== Monitor Tab ====================
    def _create_monitor_tab(self):
        """Tab giám sát"""
        frame = tk.Frame(self._main_container, bg=COLORS['bg_dark'])
        self._tab_frames["monitor"] = frame
        
        # Layout cố định
        frame.grid_columnconfigure(0, weight=1, minsize=950)
        frame.grid_columnconfigure(1, weight=0, minsize=280)
        frame.grid_rowconfigure(0, weight=1)
        
        # Camera grid
        cam_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        cam_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        self._create_camera_grid(cam_frame)
        
        # Right panel
        right = tk.Frame(frame, bg=COLORS['bg_panel'], width=280)
        right.grid(row=0, column=1, sticky='ns')
        right.grid_propagate(False)
        self._create_right_panel(right)
    
    def _create_camera_grid(self, parent):
        """Grid 2x3 camera"""
        num_cams = min(len(self.config.cameras), 6)
        
        for i in range(num_cams):
            row, col = i // 3, i % 3
            cam = self.config.cameras[i]
            cam_id = i + 1  # Camera ID là số 1, 2, 3...
            
            panel = CameraPanel(parent, cam_id, cam.name, width=420, height=280)
            panel.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            panel.set_roi(list(cam.roi.roi_person), list(cam.roi.roi_coal), cam.roi.reference_resolution)
            
            # Lưu panel với key là số nguyên (để khớp với production_app.workers)
            self._camera_panels[cam_id] = panel
        
        for i in range(3):
            parent.grid_columnconfigure(i, weight=1)
        for i in range(2):
            parent.grid_rowconfigure(i, weight=1)
    
    def _create_right_panel(self, parent):
        """Panel phải"""
        # Control
        ctrl = tk.LabelFrame(parent, text="Điều khiển", font=(FONT, 10, 'bold'),
                            bg=COLORS['bg_panel'], fg=COLORS['text_white'])
        ctrl.pack(fill=tk.X, padx=8, pady=8)
        
        self._start_btn = tk.Button(ctrl, text="▶ Bắt đầu giám sát", font=(FONT, 10, 'bold'),
                                    bg=COLORS['success'], fg='white', command=self._start_monitoring)
        self._start_btn.pack(fill=tk.X, padx=8, pady=4)
        
        self._stop_btn = tk.Button(ctrl, text="⏹ Dừng giám sát", font=(FONT, 10, 'bold'),
                                   bg=COLORS['error'], fg='white', command=self._stop_monitoring,
                                   state=tk.DISABLED)
        self._stop_btn.pack(fill=tk.X, padx=8, pady=4)
        
        # Status
        status = tk.LabelFrame(parent, text="Trạng thái", font=(FONT, 10, 'bold'),
                              bg=COLORS['bg_panel'], fg=COLORS['text_white'])
        status.pack(fill=tk.X, padx=8, pady=8)
        
        self._status_lbls = {}
        for key, text in [("cameras", "Cameras: 0/0"), ("person", "Cảnh báo người: 0"),
                          ("coal", "Cảnh báo than: 0"), ("uptime", "Uptime: 00:00:00")]:
            lbl = tk.Label(status, text=text, font=(FONT, 9), bg=COLORS['bg_panel'],
                          fg=COLORS['text_white'], anchor='w')
            lbl.pack(anchor=tk.W, padx=10, pady=2)
            self._status_lbls[key] = lbl
        
        # Alerts
        alerts = tk.LabelFrame(parent, text="Cảnh báo khẩn cấp", font=(FONT, 10, 'bold'),
                              bg=COLORS['bg_panel'], fg=COLORS['accent'])
        alerts.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self._log_text = scrolledtext.ScrolledText(alerts, height=12, bg=COLORS['bg_video'],
                                                   fg=COLORS['accent'], font=('Consolas', 8), wrap=tk.WORD)
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # ==================== Camera Config Tab ====================
    def _create_camera_config_tab(self):
        """Tab cấu hình camera"""
        frame = tk.Frame(self._main_container, bg=COLORS['bg_dark'])
        self._tab_frames["cameras"] = frame
        
        # Info
        info = tk.Label(frame, text="💡 Cấu hình RTSP URL và bật/tắt từng camera. Nhấn 'Lưu cấu hình' để lưu thay đổi.",
                       font=(FONT, 10), bg=COLORS['bg_dark'], fg=COLORS['text_light'])
        info.pack(pady=10)
        
        # Scrollable
        canvas = tk.Canvas(frame, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self._config_widgets['cameras'] = []
        
        for i, cam in enumerate(self.config.cameras):
            cam_frame = tk.LabelFrame(scroll_frame, text=f"Camera {i+1} - {cam.name}",
                                     font=(FONT, 10, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['accent'])
            cam_frame.pack(fill=tk.X, pady=5, padx=5)
            
            widgets = {}
            
            # Name
            row = tk.Frame(cam_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text="Tên camera:", width=15, anchor='w', bg=COLORS['bg_panel'],
                    fg=COLORS['text_white'], font=(FONT, 9)).pack(side=tk.LEFT)
            widgets['name'] = tk.Entry(row, width=30, font=(FONT, 9))
            widgets['name'].insert(0, cam.name)
            widgets['name'].pack(side=tk.LEFT, padx=5)
            
            # RTSP
            row = tk.Frame(cam_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text="RTSP URL:", width=15, anchor='w', bg=COLORS['bg_panel'],
                    fg=COLORS['text_white'], font=(FONT, 9)).pack(side=tk.LEFT)
            widgets['rtsp'] = tk.Entry(row, width=70, font=(FONT, 9))
            widgets['rtsp'].insert(0, cam.rtsp_url)
            widgets['rtsp'].pack(side=tk.LEFT, padx=5)
            
            # Enabled
            row = tk.Frame(cam_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text="Bật camera:", width=15, anchor='w', bg=COLORS['bg_panel'],
                    fg=COLORS['text_white'], font=(FONT, 9)).pack(side=tk.LEFT)
            widgets['enabled'] = tk.BooleanVar(value=cam.enabled)
            tk.Checkbutton(row, variable=widgets['enabled'], bg=COLORS['bg_panel'],
                          fg=COLORS['text_white'], selectcolor=COLORS['bg_dark'],
                          text="Bật").pack(side=tk.LEFT, padx=5)
            
            self._config_widgets['cameras'].append(widgets)
        
        # Save button
        btn_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X, pady=15)
        tk.Button(btn_frame, text="💾 Lưu cấu hình Camera", font=(FONT, 11, 'bold'),
                 bg=COLORS['success'], fg='white', padx=20, pady=8,
                 command=self._save_camera_config).pack()
    
    # ==================== Detection Config Tab ====================
    def _create_detection_config_tab(self):
        """Tab cấu hình detection"""
        frame = tk.Frame(self._main_container, bg=COLORS['bg_dark'])
        self._tab_frames["detection"] = frame
        
        info = tk.Label(frame, text="💡 Cấu hình các ngưỡng phát hiện người và tắc than. Thay đổi sẽ áp dụng ngay không cần restart.",
                       font=(FONT, 10), bg=COLORS['bg_dark'], fg=COLORS['text_light'])
        info.pack(pady=10)
        
        self._config_widgets['detection'] = {}
        
        # Person detection
        person_frame = tk.LabelFrame(frame, text="🧑 PHÁT HIỆN NGƯỜI VÀO VÙNG NGUY HIỂM",
                                    font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['success'])
        person_frame.pack(fill=tk.X, padx=20, pady=10)
        
        det = self.config.cameras[0].detection if self.config.cameras else None
        
        self._create_scale_row(person_frame, "Số frame liên tiếp để BẬT cảnh báo:",
                              "person_on", 1, 15, det.person_consecutive_threshold if det else 3)
        self._create_scale_row(person_frame, "Số frame không phát hiện để TẮT:",
                              "person_off", 1, 20, det.person_no_detection_threshold if det else 5)
        
        # Coal detection
        coal_frame = tk.LabelFrame(frame, text="⛏️ PHÁT HIỆN TẮC THAN",
                                  font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['error'])
        coal_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self._create_scale_row(coal_frame, "Ngưỡng tỷ lệ than (%):",
                              "coal_threshold", 30, 100, det.coal_ratio_threshold if det else 73)
        self._create_scale_row(coal_frame, "Số frame liên tiếp để BẬT:",
                              "coal_on", 1, 20, det.coal_consecutive_threshold if det else 5)
        self._create_scale_row(coal_frame, "Số frame để TẮT:",
                              "coal_off", 1, 20, det.coal_no_blockage_threshold if det else 5)
        
        # Model
        model_frame = tk.LabelFrame(frame, text="🤖 CẤU HÌNH MODEL",
                                   font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['warning'])
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self._create_scale_row(model_frame, "Confidence threshold:",
                              "confidence", 0.1, 1.0, det.confidence_threshold if det else 0.7, resolution=0.05)
        
        # Checkbox bật/tắt detection
        row = tk.Frame(model_frame, bg=COLORS['bg_panel'])
        row.pack(fill=tk.X, padx=15, pady=5)
        self._config_widgets['detection']['coal_enabled'] = tk.BooleanVar(value=det.coal_detection_enabled if det else True)
        tk.Checkbutton(row, text="Bật phát hiện tắc than", variable=self._config_widgets['detection']['coal_enabled'],
                      bg=COLORS['bg_panel'], fg=COLORS['text_white'], selectcolor=COLORS['bg_dark'],
                      font=(FONT, 10)).pack(side=tk.LEFT)
        
        # Save button
        btn_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X, pady=20)
        tk.Button(btn_frame, text="💾 Lưu cấu hình Detection", font=(FONT, 11, 'bold'),
                 bg=COLORS['success'], fg='white', padx=20, pady=8,
                 command=self._save_detection_config).pack()
    
    def _create_scale_row(self, parent, label: str, key: str, from_: float, to: float, 
                         default: float, resolution: float = 1):
        """Tạo row với scale"""
        row = tk.Frame(parent, bg=COLORS['bg_panel'])
        row.pack(fill=tk.X, padx=15, pady=5)
        
        tk.Label(row, text=label, width=35, anchor='w', bg=COLORS['bg_panel'],
                fg=COLORS['text_white'], font=(FONT, 10)).pack(side=tk.LEFT)
        
        scale = tk.Scale(row, from_=from_, to=to, orient=tk.HORIZONTAL, length=250,
                        resolution=resolution, bg=COLORS['bg_panel'], fg=COLORS['text_white'],
                        highlightthickness=0, troughcolor=COLORS['bg_dark'])
        scale.set(default)
        scale.pack(side=tk.LEFT, padx=10)
        
        self._config_widgets['detection'][key] = scale
    
    # ==================== PLC Config Tab ====================
    def _create_plc_config_tab(self):
        """Tab cấu hình PLC"""
        frame = tk.Frame(self._main_container, bg=COLORS['bg_dark'])
        self._tab_frames["plc"] = frame
        
        info = tk.Label(frame, text="💡 Cấu hình kết nối PLC Siemens S7 và địa chỉ tín hiệu báo động cho từng camera.",
                       font=(FONT, 10), bg=COLORS['bg_dark'], fg=COLORS['text_light'])
        info.pack(pady=10)
        
        self._config_widgets['plc'] = {}
        
        # Connection
        conn_frame = tk.LabelFrame(frame, text="🔌 KẾT NỐI PLC",
                                  font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['info'])
        conn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        plc = self.config.cameras[0].plc if self.config.cameras else None
        
        for label, key, default, width in [
            ("IP Address:", "ip", plc.ip if plc else "192.168.0.4", 20),
            ("Rack:", "rack", str(plc.rack if plc else 0), 10),
            ("Slot:", "slot", str(plc.slot if plc else 2), 10),
            ("DB Number:", "db", str(plc.db_number if plc else 300), 10),
        ]:
            row = tk.Frame(conn_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, padx=15, pady=3)
            tk.Label(row, text=label, width=15, anchor='w', bg=COLORS['bg_panel'],
                    fg=COLORS['text_white'], font=(FONT, 10)).pack(side=tk.LEFT)
            entry = tk.Entry(row, width=width, font=(FONT, 10))
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, padx=5)
            self._config_widgets['plc'][key] = entry
        
        # Test button
        tk.Button(conn_frame, text="🔗 Test kết nối PLC", font=(FONT, 10),
                 bg=COLORS['info'], fg='white', command=self._test_plc).pack(pady=10)
        
        # Signals
        signals_frame = tk.LabelFrame(frame, text="📡 ĐỊA CHỈ TÍN HIỆU (DB.Byte.Bit)",
                                     font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['warning'])
        signals_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Header
        header = tk.Frame(signals_frame, bg=COLORS['bg_panel'])
        header.pack(fill=tk.X, padx=10, pady=5)
        for i, text in enumerate(["Camera", "Cảnh báo Người (Byte.Bit)", "Cảnh báo Than (Byte.Bit)"]):
            tk.Label(header, text=text, width=25 if i > 0 else 10, font=(FONT, 9, 'bold'),
                    bg=COLORS['bg_panel'], fg=COLORS['text_white']).pack(side=tk.LEFT, padx=5)
        
        self._config_widgets['plc']['signals'] = []
        
        for i, cam in enumerate(self.config.cameras[:6]):
            row = tk.Frame(signals_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(row, text=f"Camera {i+1}", width=10, font=(FONT, 9),
                    bg=COLORS['bg_panel'], fg=COLORS['text_light']).pack(side=tk.LEFT, padx=5)
            
            person_entry = tk.Entry(row, width=20, font=(FONT, 9))
            person_entry.insert(0, f"{cam.plc.person_alarm_byte}.{cam.plc.person_alarm_bit}")
            person_entry.pack(side=tk.LEFT, padx=15)
            
            coal_entry = tk.Entry(row, width=20, font=(FONT, 9))
            coal_entry.insert(0, f"{cam.plc.coal_alarm_byte}.{cam.plc.coal_alarm_bit}")
            coal_entry.pack(side=tk.LEFT, padx=15)
            
            self._config_widgets['plc']['signals'].append((person_entry, coal_entry))
        
        # Save button
        btn_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X, pady=15)
        tk.Button(btn_frame, text="💾 Lưu cấu hình PLC", font=(FONT, 11, 'bold'),
                 bg=COLORS['success'], fg='white', padx=20, pady=8,
                 command=self._save_plc_config).pack()
    
    # ==================== System Tab ====================
    def _create_system_tab(self):
        """Tab cấu hình hệ thống"""
        frame = tk.Frame(self._main_container, bg=COLORS['bg_dark'])
        self._tab_frames["system"] = frame
        
        # System info
        sys_frame = tk.LabelFrame(frame, text="⚙️ THÔNG TIN HỆ THỐNG",
                                 font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['accent'])
        sys_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self._config_widgets['system'] = {}
        
        for label, key, default in [
            ("Tên ứng dụng:", "app_name", self.config.app_name),
            ("Công ty:", "company", self.config.company),
            ("Phiên bản:", "version", self.config.version),
        ]:
            row = tk.Frame(sys_frame, bg=COLORS['bg_panel'])
            row.pack(fill=tk.X, padx=15, pady=3)
            tk.Label(row, text=label, width=15, anchor='w', bg=COLORS['bg_panel'],
                    fg=COLORS['text_white'], font=(FONT, 10)).pack(side=tk.LEFT)
            entry = tk.Entry(row, width=50, font=(FONT, 10))
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, padx=5)
            self._config_widgets['system'][key] = entry
        
        # File operations
        file_frame = tk.LabelFrame(frame, text="📁 QUẢN LÝ CẤU HÌNH",
                                  font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['info'])
        file_frame.pack(fill=tk.X, padx=20, pady=10)
        
        row = tk.Frame(file_frame, bg=COLORS['bg_panel'])
        row.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Button(row, text="📂 Mở file cấu hình khác", font=(FONT, 10),
                 bg=COLORS['info'], fg='white', command=self._load_config_file).pack(side=tk.LEFT, padx=5)
        tk.Button(row, text="💾 Lưu cấu hình ra file mới", font=(FONT, 10),
                 bg=COLORS['warning'], fg='white', command=self._save_config_as).pack(side=tk.LEFT, padx=5)
        tk.Button(row, text="🔄 Reload cấu hình", font=(FONT, 10),
                 bg=COLORS['text_gray'], fg='white', command=self._reload_config).pack(side=tk.LEFT, padx=5)
        
        # Current config path
        self._config_path_lbl = tk.Label(file_frame, text=f"📄 File hiện tại: {self.config_path}",
                                        font=(FONT, 9), bg=COLORS['bg_panel'], fg=COLORS['text_light'])
        self._config_path_lbl.pack(pady=5)
        
        # Logs
        log_frame = tk.LabelFrame(frame, text="📋 LOGS HỆ THỐNG",
                                 font=(FONT, 11, 'bold'), bg=COLORS['bg_panel'], fg=COLORS['accent'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self._sys_log = scrolledtext.ScrolledText(log_frame, bg=COLORS['bg_video'],
                                                  fg=COLORS['text_white'], font=('Consolas', 9))
        self._sys_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        btn_row = tk.Frame(log_frame, bg=COLORS['bg_panel'])
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_row, text="🗑️ Xóa log", bg=COLORS['error'], fg='white',
                 command=lambda: self._sys_log.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
    
    # ==================== Config Save/Load Functions ====================
    def _save_camera_config(self):
        """Lưu cấu hình camera"""
        try:
            for i, widgets in enumerate(self._config_widgets['cameras']):
                if i < len(self.config.cameras):
                    self.config.cameras[i].name = widgets['name'].get()
                    self.config.cameras[i].rtsp_url = widgets['rtsp'].get()
                    self.config.cameras[i].enabled = widgets['enabled'].get()
            
            save_config(self.config, self.config_path)
            self._add_log("✅ Đã lưu cấu hình Camera!")
            messagebox.showinfo("Thành công", "Đã lưu cấu hình Camera!\nKhởi động lại để áp dụng thay đổi RTSP URL.")
        except Exception as e:
            self._add_log(f"❌ Lỗi lưu: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể lưu: {str(e)}")
    
    def _save_detection_config(self):
        """Lưu cấu hình detection"""
        try:
            widgets = self._config_widgets['detection']
            
            for cam in self.config.cameras:
                cam.detection.person_consecutive_threshold = int(widgets['person_on'].get())
                cam.detection.person_no_detection_threshold = int(widgets['person_off'].get())
                cam.detection.coal_ratio_threshold = float(widgets['coal_threshold'].get())
                cam.detection.coal_consecutive_threshold = int(widgets['coal_on'].get())
                cam.detection.coal_no_blockage_threshold = int(widgets['coal_off'].get())
                cam.detection.confidence_threshold = float(widgets['confidence'].get())
                cam.detection.coal_detection_enabled = widgets['coal_enabled'].get()
            
            save_config(self.config, self.config_path)
            self._add_log("✅ Đã lưu cấu hình Detection!")
            messagebox.showinfo("Thành công", "Đã lưu cấu hình Detection!\nThay đổi sẽ áp dụng ngay.")
        except Exception as e:
            self._add_log(f"❌ Lỗi lưu: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể lưu: {str(e)}")
    
    def _save_plc_config(self):
        """Lưu cấu hình PLC"""
        try:
            widgets = self._config_widgets['plc']
            
            for i, cam in enumerate(self.config.cameras[:6]):
                cam.plc.ip = widgets['ip'].get()
                cam.plc.rack = int(widgets['rack'].get())
                cam.plc.slot = int(widgets['slot'].get())
                cam.plc.db_number = int(widgets['db'].get())
                
                if i < len(widgets['signals']):
                    person_addr = widgets['signals'][i][0].get().split('.')
                    coal_addr = widgets['signals'][i][1].get().split('.')
                    
                    cam.plc.person_alarm_byte = int(person_addr[0])
                    cam.plc.person_alarm_bit = int(person_addr[1]) if len(person_addr) > 1 else 0
                    cam.plc.coal_alarm_byte = int(coal_addr[0])
                    cam.plc.coal_alarm_bit = int(coal_addr[1]) if len(coal_addr) > 1 else 1
            
            save_config(self.config, self.config_path)
            self._add_log("✅ Đã lưu cấu hình PLC!")
            messagebox.showinfo("Thành công", "Đã lưu cấu hình PLC!\nKhởi động lại để áp dụng.")
        except Exception as e:
            self._add_log(f"❌ Lỗi lưu: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể lưu: {str(e)}")
    
    def _test_plc(self):
        """Test kết nối PLC"""
        try:
            import snap7
            ip = self._config_widgets['plc']['ip'].get()
            rack = int(self._config_widgets['plc']['rack'].get())
            slot = int(self._config_widgets['plc']['slot'].get())
            
            client = snap7.client.Client()
            client.connect(ip, rack, slot)
            
            if client.get_connected():
                self._add_log(f"✅ Kết nối PLC thành công: {ip}")
                messagebox.showinfo("Thành công", f"Kết nối PLC thành công!\nIP: {ip}\nRack: {rack}\nSlot: {slot}")
                client.disconnect()
            else:
                raise Exception("Không thể kết nối")
        except Exception as e:
            self._add_log(f"❌ Lỗi kết nối PLC: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể kết nối PLC:\n{str(e)}")
    
    def _load_config_file(self):
        """Load file config khác"""
        path = filedialog.askopenfilename(title="Chọn file cấu hình",
                                         filetypes=[("JSON files", "*.json")])
        if path:
            try:
                self.config = load_config(path)
                self.config_path = path
                self._config_path_lbl.config(text=f"📄 File: {path}")
                self._add_log(f"✅ Đã load: {path}")
                messagebox.showinfo("Thành công", f"Đã load cấu hình từ:\n{path}\n\nKhởi động lại ứng dụng để áp dụng.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể load:\n{str(e)}")
    
    def _save_config_as(self):
        """Lưu config ra file mới"""
        path = filedialog.asksaveasfilename(title="Lưu cấu hình",
                                           defaultextension=".json",
                                           filetypes=[("JSON files", "*.json")])
        if path:
            try:
                # Update system info
                widgets = self._config_widgets.get('system', {})
                if 'app_name' in widgets:
                    self.config.app_name = widgets['app_name'].get()
                if 'company' in widgets:
                    self.config.company = widgets['company'].get()
                if 'version' in widgets:
                    self.config.version = widgets['version'].get()
                
                save_config(self.config, path)
                self.config_path = path
                self._config_path_lbl.config(text=f"📄 File: {path}")
                self._add_log(f"✅ Đã lưu ra: {path}")
                messagebox.showinfo("Thành công", f"Đã lưu cấu hình ra:\n{path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu:\n{str(e)}")
    
    def _reload_config(self):
        """Reload config từ file"""
        try:
            self.config = load_config(self.config_path)
            self._add_log(f"✅ Đã reload: {self.config_path}")
            messagebox.showinfo("Thành công", "Đã reload cấu hình!\nKhởi động lại để áp dụng.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể reload:\n{str(e)}")
    
    # ==================== Monitoring (Production Mode) ====================
    def _start_monitoring(self):
        """Bắt đầu giám sát - Production mode với OptimizedCameraWorker"""
        from ..core import ProductionMultiCameraApp
        
        self._add_log("🚀 Khởi động Production Mode (24/7)...")
        self._start_btn.config(state=tk.DISABLED)
        self._global_status.config(text="🟡 Khởi động...", fg=COLORS['warning'])
        
        for panel in self._camera_panels.values():
            panel.set_status("connecting")
        
        def start():
            try:
                # Init production app
                self._app = ProductionMultiCameraApp(
                    config=self.config,
                    on_alert=self._on_production_alert,
                    on_log=self._add_log,
                )
                
                self._add_log("🔄 Loading YOLO model(s)...")
                if self._app.load_models():
                    results = self._app.start_all()
                    success = sum(1 for r in results.values() if r)
                    
                    self._is_monitoring = True
                    self._start_time = time.time()
                    
                    self._add_log(f"✅ Production mode: {success}/{len(results)} cameras ready")
                    self.root.after(0, lambda: self._stop_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self._global_status.config(
                        text=f"🟢 {success} cameras (24/7)", fg=COLORS['success']))
                else:
                    raise Exception("Không thể load model!")
            except Exception as e:
                self._add_log(f"❌ Lỗi: {str(e)}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._start_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self._global_status.config(text="❌ Lỗi", fg=COLORS['error']))
        
        threading.Thread(target=start, daemon=True).start()
    
    def _stop_monitoring(self):
        """Dừng giám sát"""
        if self._app:
            self._add_log("⏹️ Đang dừng Production mode...")
            self._stop_btn.config(state=tk.DISABLED)
            self._app.stop_all()
            self._app = None
        
        self._is_monitoring = False
        
        for panel in self._camera_panels.values():
            panel.set_status("offline")
            panel.set_alarm(False, False)
        
        self._start_btn.config(state=tk.NORMAL)
        self._global_status.config(text="⚪ Đã dừng", fg=COLORS['text_white'])
        self._add_log("✅ Đã dừng")
    
    def _on_production_alert(self, camera_id: int, alert_type: str, is_active: bool, value: float):
        """Callback alert từ Production app"""
        # camera_id là số nguyên (1, 2, 3...)
        panel = self._camera_panels.get(camera_id)
        
        if panel:
            if alert_type == "person":
                panel.set_alarm(is_active, panel.coal_alarm)
            elif alert_type == "coal":
                panel.set_alarm(panel.person_alarm, is_active)
        
        # Log
        if is_active:
            if alert_type == "person":
                self._add_log(f"🚨 Camera {camera_id}: PHÁT HIỆN NGƯỜI trong vùng nguy hiểm!")
            elif alert_type == "coal":
                self._add_log(f"🚨 Camera {camera_id}: TẮC THAN - Tỷ lệ {value:.1f}%")
        else:
            if alert_type == "person":
                self._add_log(f"✅ Camera {camera_id}: Đã tắt cảnh báo người")
            elif alert_type == "coal":
                self._add_log(f"✅ Camera {camera_id}: Đã tắt cảnh báo than")
    
    def _add_log(self, msg: str):
        """Add log"""
        def add():
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}\n"
            
            self._log_text.insert(tk.END, line)
            self._log_text.see(tk.END)
            
            if hasattr(self, '_sys_log'):
                self._sys_log.insert(tk.END, line)
                self._sys_log.see(tk.END)
            
            # Limit
            if int(self._log_text.index('end-1c').split('.')[0]) > 100:
                self._log_text.delete('1.0', '10.0')
        
        self.root.after(0, add)
    
    def _update_loop(self):
        """GUI loop ~25 FPS - Production mode"""
        try:
            # Camera displays - lấy frame từ workers
            if self._current_tab == "monitor" and self._app and self._is_monitoring:
                self._update_camera_displays()
            
            # Status mỗi 500ms
            now = int(time.time() * 1000)
            if not hasattr(self, '_last_upd') or now - self._last_upd > 500:
                self._last_upd = now
                self._update_status()
        except Exception as e:
            pass
        
        self.root.after(40, self._update_loop)
    
    def _update_camera_displays(self):
        """Cập nhật hiển thị từ production workers"""
        if not hasattr(self._app, 'workers'):
            return
        
        from ..core import WorkerStatus
        
        for cam_id, worker in self._app.workers.items():
            # cam_id là số nguyên (1, 2, 3...)
            panel = self._camera_panels.get(cam_id)
            
            if panel is None:
                # Debug: in ra nếu không tìm thấy panel
                # print(f"[DEBUG] Panel not found for cam_id={cam_id}, available keys={list(self._camera_panels.keys())}")
                continue
            
            # Update FPS
            worker.update_fps()
            
            # Update connection status
            if worker.status == WorkerStatus.RUNNING:
                if worker.person_alarm_active or worker.coal_alarm_active:
                    panel.set_status("alarm")
                else:
                    panel.set_status("online")
            elif worker.status == WorkerStatus.RECONNECTING:
                panel.set_status("reconnecting")
            elif worker.status == WorkerStatus.STARTING:
                panel.set_status("connecting")
            else:
                panel.set_status("offline")
            
            # Get frame từ worker (atomic, low-latency)
            frame = worker.get_display_frame(copy=True)
            
            # Debug: in frame info mỗi 100 lần
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = {}
            if cam_id not in self._debug_counter:
                self._debug_counter[cam_id] = 0
            self._debug_counter[cam_id] += 1
            
            if self._debug_counter[cam_id] == 1:
                print(f"[DEBUG] Cam {cam_id}: status={worker.status.value}, frame={'OK' if frame is not None else 'None'}, frame_count={worker._frame_count}")
            
            if frame is not None:
                panel.update_frame(frame)
                panel.update_stats(worker.fps_display, worker.last_coal_ratio, worker.last_person_detected)
            
            # Process result nếu có
            result = worker.get_latest_result()
            if result is not None:
                _, yolo_result, coal_blocked, coal_ratio = result
                worker.clear_result()
            
            # Process display
            panel.process_display()
    
    def _update_status(self):
        """Update status"""
        self._time_lbl.config(text=datetime.datetime.now().strftime("%H:%M:%S"))
        
        if self._app and self._is_monitoring:
            stats = self._app.get_stats()
            self._status_lbls["cameras"].config(text=f"Cameras: {stats.running_cameras}/{stats.total_cameras}")
            self._status_lbls["person"].config(text=f"Cảnh báo người: {stats.total_person_alerts}")
            self._status_lbls["coal"].config(text=f"Cảnh báo than: {stats.total_coal_alerts}")
            
            if self._start_time and self._is_monitoring:
                up = int(time.time() - self._start_time)
                self._status_lbls["uptime"].config(text=f"Uptime: {up//3600:02d}:{(up%3600)//60:02d}:{up%60:02d}")
    
    def run(self):
        """Run"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
    
    def _auto_start_monitoring(self):
        """Tự động bắt đầu giám sát khi khởi động"""
        if not self._is_monitoring:
            self._start_monitoring()
    
    def _on_close(self):
        """Close"""
        if self._app and self._app.is_any_running:
            if messagebox.askokcancel("Xác nhận", "Dừng cameras và thoát?"):
                self._app.stop_all()
                self.root.destroy()
        else:
            self.root.destroy()
