"""
Model Loader Module
===================

Quản lý loading và caching YOLO models.
Hỗ trợ:
- Load nhiều model từ file (multi-model support)
- Singleton pattern để share models giữa nhiều camera
- Thread-safe inference
- Map camera -> model
"""

import os
import sys
import threading
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Thông tin về model đã load"""
    model_id: str
    path: str
    name: str
    class_names: Dict[int, str]
    person_class_id: int
    coal_class_id: int
    cameras: List[int]  # Camera numbers sử dụng model này
    is_loaded: bool = False


class MultiModelLoader:
    """
    Singleton để quản lý nhiều YOLO models
    
    Features:
    - Singleton pattern (share models giữa cameras)
    - Support multiple models (mỗi camera dùng model khác nhau)
    - Thread-safe
    - Lazy loading
    - Auto-detect class IDs
    
    Usage:
        # Load models
        loader = MultiModelLoader.get_instance()
        loader.load_models(system_config)
        
        # Get model for camera
        model_info = loader.get_model_for_camera(camera_number=1)
        
        # Inference
        results = loader.predict(camera_number=1, frame=frame, conf=0.7)
    """
    
    _instance: Optional['MultiModelLoader'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model loader"""
        if self._initialized:
            return
        
        # Dict[model_id, YOLO model object]
        self._models: Dict[str, Any] = {}
        # Dict[model_id, ModelInfo]
        self._model_infos: Dict[str, ModelInfo] = {}
        # Dict[camera_number, model_id] - mapping camera -> model
        self._camera_model_map: Dict[int, str] = {}
        # Lock cho mỗi model (thread-safe inference)
        self._inference_locks: Dict[str, threading.Lock] = {}
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'MultiModelLoader':
        """Lấy singleton instance"""
        return cls()
    
    @property
    def loaded_models(self) -> List[str]:
        """Danh sách model IDs đã load"""
        return list(self._models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Lấy thông tin model theo ID"""
        return self._model_infos.get(model_id)
    
    def get_model_info_for_camera(self, camera_number: int) -> Optional[ModelInfo]:
        """Lấy thông tin model cho camera
        
        Args:
            camera_number: Số thứ tự camera (1, 2, 3, ...)
            
        Returns:
            ModelInfo hoặc None
        """
        model_id = self._camera_model_map.get(camera_number)
        if model_id:
            return self._model_infos.get(model_id)
        
        # Fallback: trả về model đầu tiên
        if self._model_infos:
            return list(self._model_infos.values())[0]
        return None
    
    def load_from_config(self, system_config) -> Dict[str, bool]:
        """Load tất cả models từ SystemConfig
        
        Args:
            system_config: SystemConfig object
            
        Returns:
            Dict[model_id, success] - kết quả load cho từng model
        """
        results = {}
        
        if system_config.models:
            # Load từ config models
            for model_id, model_cfg in system_config.models.items():
                try:
                    success = self.load(
                        model_id=model_id,
                        model_path=model_cfg.path,
                        model_name=model_cfg.name,
                        cameras=model_cfg.cameras
                    )
                    results[model_id] = success
                except Exception as e:
                    print(f"[ERROR] Load model {model_id} failed: {e}")
                    results[model_id] = False
        else:
            # Backward compatible: load từ model_path
            try:
                success = self.load(
                    model_id="default",
                    model_path=system_config.model_path,
                    model_name="Default Model",
                    cameras=list(range(1, 10))
                )
                results["default"] = success
            except Exception as e:
                print(f"[ERROR] Load default model failed: {e}")
                results["default"] = False
        
        return results
    
    def load(self, model_id: str, model_path: str, 
             model_name: str = "Model", 
             cameras: List[int] = None) -> bool:
        """Load một YOLO model
        
        Args:
            model_id: ID duy nhất cho model
            model_path: Đường dẫn file model (.pt)
            model_name: Tên hiển thị
            cameras: List camera numbers sử dụng model này
            
        Returns:
            True nếu load thành công
            
        Raises:
            FileNotFoundError: Nếu không tìm thấy file model
        """
        if cameras is None:
            cameras = []
        
        # Kiểm tra đã load chưa
        if model_id in self._models:
            print(f"[INFO] Model {model_id} đã được load, bỏ qua")
            return True
        
        # Tìm đường dẫn model
        resolved_path = self._resolve_model_path(model_path)
        
        if not resolved_path:
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
        
        try:
            # Import YOLO (lazy import)
            from ultralytics import YOLO
            
            # Tạo lock cho model này
            if model_id not in self._inference_locks:
                self._inference_locks[model_id] = threading.Lock()
            
            with self._inference_locks[model_id]:
                model = YOLO(resolved_path)
                
                # TỐI ƯU GPU: Đưa model lên GPU nếu có
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cuda':
                    try:
                        # YOLO tự động detect GPU, nhưng có thể explicit set
                        model.to(device)
                        print(f"[GPU] Model {model_id} đã được load lên GPU")
                        
                        # Warm-up GPU với dummy frame (tối ưu lần inference đầu tiên)
                        try:
                            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                            _ = model.predict(
                                dummy_frame,
                                conf=0.5,
                                verbose=False,
                                task='segment',
                                device=device
                            )
                            print(f"[GPU] Model {model_id} đã warm-up GPU")
                        except Exception as warmup_error:
                            print(f"[WARN] GPU warm-up failed cho {model_id}: {warmup_error}")
                    except Exception as gpu_error:
                        print(f"[WARN] Không thể load model {model_id} lên GPU: {gpu_error}")
                        device = 'cpu'
                else:
                    print(f"[CPU] Model {model_id} chạy trên CPU")
                
                # Xác định class IDs
                class_names = self._extract_class_names(model)
                person_id = self._find_class_id(class_names, ['person', 'Person'])
                coal_id = self._find_class_id(class_names, ['coal', 'Coal', 'than'])
                
                # Lưu model
                self._models[model_id] = model
                self._model_infos[model_id] = ModelInfo(
                    model_id=model_id,
                    path=resolved_path,
                    name=model_name,
                    class_names=class_names,
                    person_class_id=person_id,
                    coal_class_id=coal_id,
                    cameras=cameras,
                    is_loaded=True,
                )
                
                # Map cameras -> model
                for cam_num in cameras:
                    self._camera_model_map[cam_num] = model_id
            
            # Log GPU memory và verify GPU nếu có GPU
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[GPU] Device: {gpu_name}")
                print(f"[GPU] Memory: {allocated:.1f} MB allocated, {cached:.1f} MB cached")
                
                # Verify model đang ở GPU
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'device'):
                        model_device = str(model.model.device)
                        print(f"[GPU] ✅ Model {model_id} đang ở: {model_device}")
                    else:
                        # Kiểm tra parameters của model
                        try:
                            first_param = next(model.model.parameters()) if hasattr(model, 'model') else None
                            if first_param is not None:
                                param_device = str(first_param.device)
                                print(f"[GPU] ✅ Model {model_id} parameters đang ở: {param_device}")
                        except:
                            pass
                except Exception as verify_error:
                    print(f"[WARN] Không thể verify model device: {verify_error}")
            else:
                print(f"[CPU] Model {model_id} chạy trên CPU (không có GPU)")
            
            print(f"[OK] Loaded model: {model_id} ({model_name}) for cameras: {cameras} on {device.upper()}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Load model {model_id} failed: {e}")
            raise
    
    def _resolve_model_path(self, model_path: str) -> Optional[str]:
        """Tìm đường dẫn model (hỗ trợ cả script và exe)"""
        # Thử đường dẫn trực tiếp
        if os.path.exists(model_path):
            return model_path
        
        # Lấy base directory
        if getattr(sys, 'frozen', False):
            # Chạy từ exe (PyInstaller)
            base_dir = os.path.dirname(sys.executable)
            meipass = getattr(sys, '_MEIPASS', base_dir)
            
            paths_to_try = [
                os.path.join(meipass, model_path),
                os.path.join(base_dir, model_path),
                os.path.join(os.getcwd(), model_path),
            ]
        else:
            # Chạy từ script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(base_dir))
            
            paths_to_try = [
                os.path.join(project_root, model_path),
                os.path.join(base_dir, model_path),
                os.path.join(os.getcwd(), model_path),
            ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                return path
        
        return None
    
    def _extract_class_names(self, model) -> Dict[int, str]:
        """Trích xuất class names từ model"""
        names = getattr(model, 'names', {})
        
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        
        return {}
    
    def _find_class_id(self, class_names: Dict[int, str], 
                       possible_names: List[str]) -> int:
        """Tìm class ID theo tên"""
        for class_id, name in class_names.items():
            if name in possible_names:
                return class_id
        
        # Mặc định
        return 0 if 'person' in str(possible_names).lower() else 1
    
    def predict(self, camera_number: int, frame, 
                conf: float = 0.7, verbose: bool = False) -> Any:
        """Chạy inference trên frame cho camera cụ thể
        
        Args:
            camera_number: Số thứ tự camera (1, 2, 3, ...)
            frame: Frame video (numpy array)
            conf: Ngưỡng confidence
            verbose: In log hay không
            
        Returns:
            YOLO Results object
            
        Raises:
            RuntimeError: Nếu không tìm thấy model cho camera
        """
        model_id = self._camera_model_map.get(camera_number)
        
        if not model_id:
            # Fallback: dùng model đầu tiên
            if self._models:
                model_id = list(self._models.keys())[0]
            else:
                raise RuntimeError(f"Không tìm thấy model cho camera {camera_number}")
        
        model = self._models.get(model_id)
        if not model:
            raise RuntimeError(f"Model {model_id} chưa được load")
        
        with self._inference_locks[model_id]:
            # TỐI ƯU: Sử dụng GPU nếu có
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Debug: Log device mỗi 100 lần để verify
            if not hasattr(self, '_predict_count'):
                self._predict_count = {}
            if model_id not in self._predict_count:
                self._predict_count[model_id] = 0
            self._predict_count[model_id] += 1
            
            if self._predict_count[model_id] % 100 == 1 and verbose:
                print(f"[DEBUG] Model {model_id} inference #{self._predict_count[model_id]} on {device.upper()}")
            
            results = model.predict(
                frame, 
                conf=conf, 
                verbose=verbose, 
                task='segment',
                device=device  # Explicit device để đảm bảo dùng GPU
            )
            return results[0] if results else None
    
    def track(self, camera_number: int, frame, 
              conf: float = 0.7, persist: bool = True, 
              verbose: bool = False) -> Any:
        """Chạy tracking trên frame cho camera cụ thể
        
        Args:
            camera_number: Số thứ tự camera (1, 2, 3, ...)
            frame: Frame video (numpy array)
            conf: Ngưỡng confidence
            persist: Giữ tracking ID qua các frame
            verbose: In log hay không
            
        Returns:
            YOLO Results object với tracking IDs
        """
        model_id = self._camera_model_map.get(camera_number)
        
        if not model_id:
            if self._models:
                model_id = list(self._models.keys())[0]
            else:
                raise RuntimeError(f"Không tìm thấy model cho camera {camera_number}")
        
        model = self._models.get(model_id)
        if not model:
            raise RuntimeError(f"Model {model_id} chưa được load")
        
        with self._inference_locks[model_id]:
            # TỐI ƯU: Sử dụng GPU nếu có
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            results = model.track(
                frame,
                conf=conf,
                persist=persist,
                verbose=verbose,
                task='segment',
                device=device  # Explicit device để đảm bảo dùng GPU
            )
            return results[0] if results else None
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Lấy thông tin GPU status và verify models đang ở GPU
        
        Returns:
            Dict với thông tin GPU:
            - gpu_available: bool
            - gpu_name: str
            - memory_allocated_mb: float
            - memory_reserved_mb: float
            - models_on_gpu: Dict[model_id, bool] - True nếu model đang ở GPU
        """
        status = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": "N/A",
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "models_on_gpu": {}
        }
        
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            status["memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            
            # Verify từng model đang ở GPU
            for model_id, model in self._models.items():
                is_on_gpu = False
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'device'):
                        is_on_gpu = 'cuda' in str(model.model.device).lower()
                    elif hasattr(model, 'model'):
                        # Kiểm tra parameters
                        first_param = next(model.model.parameters(), None)
                        if first_param is not None:
                            is_on_gpu = 'cuda' in str(first_param.device).lower()
                except:
                    pass
                
                status["models_on_gpu"][model_id] = is_on_gpu
        
        return status
    
    def print_gpu_status(self) -> None:
        """In thông tin GPU status và verify models"""
        status = self.get_gpu_status()
        
        print("\n" + "="*60)
        print("📊 GPU STATUS & MODEL VERIFICATION")
        print("="*60)
        
        if status["gpu_available"]:
            print(f"✅ GPU: {status['gpu_name']}")
            print(f"📦 Memory: {status['memory_allocated_mb']:.1f} MB / {status['memory_reserved_mb']:.1f} MB")
            print("\n🔍 Model Location:")
            for model_id, is_on_gpu in status["models_on_gpu"].items():
                model_info = self._model_infos.get(model_id)
                model_name = model_info.name if model_info else model_id
                status_icon = "✅" if is_on_gpu else "❌"
                device = "GPU" if is_on_gpu else "CPU"
                print(f"   {status_icon} {model_name} ({model_id}): {device}")
        else:
            print("❌ GPU không khả dụng - Tất cả models chạy trên CPU")
            print("⚠️  Warning: Performance sẽ chậm hơn đáng kể!")
        
        print("="*60 + "\n")
    
    def unload(self, model_id: str = None) -> None:
        """Giải phóng model khỏi memory
        
        Args:
            model_id: ID model cần unload (None = tất cả)
        """
        if model_id:
            if model_id in self._models:
                with self._inference_locks.get(model_id, threading.Lock()):
                    del self._models[model_id]
                    del self._model_infos[model_id]
                    # Remove camera mappings
                    self._camera_model_map = {
                        k: v for k, v in self._camera_model_map.items() 
                        if v != model_id
                    }
        else:
            # Unload tất cả
            self._models.clear()
            self._model_infos.clear()
            self._camera_model_map.clear()


# Backward compatible alias
ModelLoader = MultiModelLoader
