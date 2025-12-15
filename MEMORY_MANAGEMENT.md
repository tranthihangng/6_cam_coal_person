# Memory Management Best Practices
## Quản lý bộ nhớ CPU và GPU - Tránh Memory Leak

Tài liệu này tổng hợp các best practices từ các GitHub repositories và cộng đồng để quản lý memory hiệu quả.

---

## 📚 Tài liệu tham khảo

### 1. PyTorch Memory Management
- **PyTorch Forum**: [GPU memory leak discussions](https://discuss.pytorch.org/t/gpu-memory-leak-with-zombie-memory-occupation-after-job-is-killed/220338)
- **Stack Overflow**: [Fix GPU mem leak after minibatches](https://stackoverflow.com/questions/61912734/pytorch-fix-gpu-mem-leak-after-exactly-10-minibatches)
- **Clay Atlas**: [Release GPU/CPU memory in PyTorch](https://clay-atlas.com/us/blog/2024/01/09/pytorch-release-gpu-cpu-memory/)

### 2. YOLO/Ultralytics
- **Ultralytics Docs**: Memory management trong YOLO inference
- **GitHub Issues**: Các vấn đề về memory leak trong YOLO

### 3. OpenCV & NumPy
- **Stack Overflow**: OpenCV memory leak prevention
- **NumPy Docs**: Array memory management

### 4. Python Threading
- **Python Docs**: Memory management trong multi-threading
- **Gist**: [Memory leak prevention in threading](https://gist.github.com/odhondt/014e39acc31cca945d636e4b4d74e1a5)

---

## 🔧 Best Practices

### 1. GPU Memory Management (PyTorch/CUDA)

#### a) Release GPU Tensors
```python
# ❌ SAI: Giữ tensor trên GPU
tensor = model(input).cuda()

# ✅ ĐÚNG: Move về CPU và release GPU reference
tensor_gpu = model(input).cuda()
tensor_cpu = tensor_gpu.cpu().detach()
del tensor_gpu  # Release GPU memory
```

#### b) Clear GPU Cache
```python
import torch
import gc

# Sau khi xử lý xong một batch
torch.cuda.empty_cache()  # Clear unused GPU memory
gc.collect()  # Force Python garbage collection
```

#### c) Use Context Managers
```python
# Tự động release khi out of scope
with torch.no_grad():
    results = model.predict(frame)
    # Process results
    # GPU tensors sẽ được release tự động
```

#### d) Detach và Move to CPU
```python
# Khi không cần gradient
tensor = tensor.detach().cpu().numpy()
# Hoặc
tensor = tensor.cpu().detach().numpy()
```

---

### 2. CPU Memory Management (NumPy/OpenCV)

#### a) Release NumPy Arrays
```python
# ❌ SAI: Giữ nhiều copies
array1 = frame.copy()
array2 = frame.copy()
array3 = frame.copy()

# ✅ ĐÚNG: Release sau khi dùng
array1 = frame.copy()
# ... use array1 ...
del array1  # Hoặc để GC tự động

# Hoặc reuse nếu có thể
working_array = frame.copy()
# ... process ...
working_array = None  # Allow GC
```

#### b) OpenCV Memory Management
```python
# OpenCV tự động quản lý memory cho Mat objects
# Nhưng cần chú ý:
# - Copy chỉ khi cần thiết
# - Release VideoCapture/VideoWriter khi done

cap = cv2.VideoCapture(url)
# ... use cap ...
cap.release()  # Quan trọng!
```

#### c) Large Array Operations
```python
# Sử dụng in-place operations khi có thể
array += 1  # ✅ In-place
array = array + 1  # ❌ Tạo copy mới

# Hoặc
np.add(array, 1, out=array)  # ✅ In-place
```

---

### 3. Multi-threading Memory Management

#### a) Thread-local Storage
```python
import threading

class Worker:
    def __init__(self):
        self._local = threading.local()
    
    def process(self):
        # Each thread có storage riêng
        if not hasattr(self._local, 'buffer'):
            self._local.buffer = np.zeros((640, 640, 3))
        # Use self._local.buffer
```

#### b) Atomic Operations
```python
import threading

class SafeCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def get(self):
        with self._lock:
            return self._value
```

#### c) Queue Management
```python
import queue

# Giới hạn queue size để tránh memory buildup
frame_queue = queue.Queue(maxsize=2)  # ✅ Good

# Skip old frames
while not queue.empty():
    try:
        old_frame = queue.get_nowait()
        frame = old_frame  # Keep latest
        # Old frames sẽ được GC tự động
    except queue.Empty:
        break
```

---

### 4. YOLO/Ultralytics Specific

#### a) Release YOLO Results
```python
from ultralytics import YOLO

model = YOLO("model.pt")
results = model.predict(frame, device='cuda')

# ❌ SAI: Giữ toàn bộ results
processed_results = results

# ✅ ĐÚNG: Extract cần thiết và release
boxes = results[0].boxes
masks = results[0].masks

# Process boxes/masks
if masks is not None:
    for i, mask in enumerate(masks.data):
        mask_cpu = mask.cpu().numpy()  # Move to CPU
        # Process mask_cpu
        del mask  # Release GPU tensor

del results  # Release YOLO results
torch.cuda.empty_cache()
```

#### b) Batch Processing
```python
# Process từng frame thay vì batch lớn
for frame in frames:
    result = model.predict(frame, device='cuda')
    # Process immediately
    # Result sẽ được GC sau loop iteration
```

---

### 5. Periodic Cleanup

#### a) Scheduled GC
```python
import gc
import time

class PeriodicCleanup:
    def __init__(self, interval=100):
        self.interval = interval
        self.count = 0
    
    def check(self):
        self.count += 1
        if self.count >= self.interval:
            gc.collect()  # Force Python GC
            torch.cuda.empty_cache()  # Clear GPU cache
            self.count = 0

# Usage
cleanup = PeriodicCleanup(interval=100)
for frame in frames:
    process(frame)
    cleanup.check()
```

#### b) Memory Monitoring
```python
import torch
import psutil
import os

def get_memory_usage():
    """Get current memory usage"""
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**2  # MB
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        gpu_mem = 0
    
    return cpu_mem, gpu_mem

# Monitor memory
cpu_mem, gpu_mem = get_memory_usage()
if gpu_mem > 4000:  # 4GB threshold
    torch.cuda.empty_cache()
    gc.collect()
```

---

### 6. Context Managers cho Resources

#### a) Custom Context Manager
```python
from contextlib import contextmanager

@contextmanager
def gpu_memory_manager():
    """Context manager để quản lý GPU memory"""
    try:
        yield
    finally:
        import torch
        import gc
        torch.cuda.empty_cache()
        gc.collect()

# Usage
with gpu_memory_manager():
    result = model.predict(frame)
    # Process result
    # Memory sẽ được cleanup tự động
```

#### b) Resource Cleanup
```python
class ResourceManager:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        if hasattr(self, 'cap'):
            self.cap.release()
        import gc
        gc.collect()

# Usage
with ResourceManager() as rm:
    rm.cap = cv2.VideoCapture(url)
    # Use rm.cap
    # Tự động release khi done
```

---

## 🎯 Áp dụng trong Coal Monitoring

### 1. YOLO Inference
```python
# Trong optimized_worker.py
with self.model_lock:
    results = self.model.predict(frame, device=device)
result = results[0] if results else None

# Process result
# ... detection logic ...

# Release GPU tensors sau khi xử lý
if result is not None and hasattr(result, 'boxes'):
    # Extract cần thiết
    boxes = result.boxes
    masks = result.masks
    
    # Process và release từng mask
    for i, mask in enumerate(masks.data):
        mask_cpu = mask.cpu().numpy()
        # Process mask_cpu
        del mask  # Release GPU tensor
    
    # Periodically clear cache
    if self._detection_count % 10 == 0:
        torch.cuda.empty_cache()
```

### 2. Frame Processing
```python
# Skip old frames trong queue
frame = None
while not self._detection_queue.empty():
    try:
        old_frame = self._detection_queue.get_nowait()
        frame = old_frame  # Keep latest
        # Old frames được GC tự động
    except:
        break

# Process frame
if frame is not None:
    display_frame = frame.copy()  # Only copy when needed
    # ... process ...
    # frame sẽ được GC sau khi out of scope
```

### 3. ROI Masks
```python
# Cache masks nhưng release intermediate arrays
def _detect_coal_blockage(self, frame, result):
    # Create mask (cached nếu có thể)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Process
    for mask_tensor in masks.data:
        mask_cpu = mask_tensor.cpu().numpy()  # Move to CPU
        # Process mask_cpu
        del mask_tensor  # Release GPU reference
    
    # Intermediate arrays sẽ được GC
    return result
```

---

## 📊 Memory Monitoring Tools

### 1. GPU Memory Tracking
```python
import torch

def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
```

### 2. CPU Memory Tracking
```python
import psutil
import os

def log_cpu_memory():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    print(f"CPU: {mem_mb:.1f} MB")
```

### 3. Memory Profiler
```python
# Install: pip install memory-profiler

from memory_profiler import profile

@profile
def process_frame(frame):
    # Code here
    pass
```

---

## ⚠️ Common Mistakes

### 1. ❌ Giữ references không cần thiết
```python
# ❌ SAI
self.all_results = []  # Giữ tất cả results
for frame in frames:
    result = model.predict(frame)
    self.all_results.append(result)  # Memory leak!

# ✅ ĐÚNG
for frame in frames:
    result = model.predict(frame)
    # Process immediately
    process(result)
    # Result được GC tự động
```

### 2. ❌ Không release VideoCapture
```python
# ❌ SAI
cap = cv2.VideoCapture(url)
# ... use ...
# Quên release!

# ✅ ĐÚNG
try:
    cap = cv2.VideoCapture(url)
    # ... use ...
finally:
    cap.release()
```

### 3. ❌ Copy không cần thiết
```python
# ❌ SAI
frame1 = frame.copy()
frame2 = frame.copy()
frame3 = frame.copy()

# ✅ ĐÚNG
# Chỉ copy khi cần modify mà không muốn ảnh hưởng original
working_frame = frame.copy() if need_modify else frame
```

---

## 🔗 Useful Links

1. **PyTorch Memory Management**: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
2. **NumPy Memory**: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
3. **OpenCV Memory**: https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html#details
4. **Python GC**: https://docs.python.org/3/library/gc.html
5. **Threading Best Practices**: https://docs.python.org/3/library/threading.html

---

## 📝 Checklist

- [ ] Release GPU tensors sau khi xử lý (`.cpu().detach()`)
- [ ] Sử dụng `torch.cuda.empty_cache()` định kỳ
- [ ] Release VideoCapture/VideoWriter khi done
- [ ] Giới hạn queue size để tránh memory buildup
- [ ] Skip old frames thay vì accumulate
- [ ] Cache những gì có thể (ROI masks, polygon arrays)
- [ ] Release intermediate arrays sau khi dùng
- [ ] Sử dụng in-place operations khi có thể
- [ ] Monitor memory usage định kỳ
- [ ] Sử dụng context managers cho resources

---

**Last Updated**: 2025-01-XX
**Version**: 1.0

