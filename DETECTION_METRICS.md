# Các Thông Số Quan Trọng Trong Dự Đoán (Detection Metrics)

## 📊 THÔNG SỐ HIỆN TẠI ĐÃ TRACK

### 1. **Performance Metrics (Hiệu suất)**
- ✅ **FPS (Frames Per Second)**
  - Capture FPS: Tốc độ đọc frame từ camera
  - Display FPS: Tốc độ hiển thị trên UI
  - Detection FPS: Tốc độ xử lý detection

- ✅ **Inference Time**
  - Last inference time (ms): Thời gian inference lần cuối
  - Average inference time (ms): Trung bình thời gian inference
  - Min/Max inference time (ms): Giá trị min/max
  - Inference throughput (FPS): Số frame inference được mỗi giây

### 2. **Detection Metrics (Phát hiện)**
- ✅ **Total Detections**: Tổng số lần detection
- ✅ **Frame Count**: Tổng số frame đã xử lý
- ✅ **Detection Confidence**: Ngưỡng confidence threshold (0.7)

### 3. **Alert Metrics (Cảnh báo)**
- ✅ **Total Person Alerts**: Tổng số cảnh báo người
- ✅ **Total Coal Alerts**: Tổng số cảnh báo than
- ✅ **Alarm Active Status**: Trạng thái cảnh báo đang active hay không

### 4. **System Metrics (Hệ thống)**
- ✅ **Uptime**: Thời gian chạy liên tục
- ✅ **Camera Status**: Trạng thái camera (running/stopped/reconnecting)
- ✅ **GPU Memory**: Bộ nhớ GPU đang sử dụng (MB)

---

## 🎯 THÔNG SỐ BỔ SUNG NÊN QUAN TÂM

### 5. **Detection Quality Metrics (Chất lượng phát hiện)**

#### 5.1. **Detection Statistics**
- ⭐ **Objects Detected Per Frame**
  - Average number of persons per frame
  - Average number of coal objects per frame
  - Max objects detected in a single frame
  - Distribution of object counts

- ⭐ **Confidence Scores Distribution**
  - Average confidence score per detection
  - Min/Max/Average confidence cho person
  - Min/Max/Average confidence cho coal
  - Confidence histogram/percentiles (P50, P90, P95, P99)

#### 5.2. **ROI Coverage Metrics**
- ⭐ **ROI Hit Rate**
  - Percentage of detections within person ROI
  - Percentage of detections within coal ROI
  - ROI intersection over union (IoU) scores

- ⭐ **Spatial Distribution**
  - Distribution of detections across frame regions
  - Hot zones (vùng có nhiều detection)

### 6. **Performance Bottleneck Metrics (Điểm nghẽn)**

#### 6.1. **Processing Pipeline Times**
- ⭐ **Frame Processing Pipeline Breakdown**
  - Frame capture time (ms)
  - Frame preprocessing time (ms)
  - Model inference time (ms) ✅ (đã có)
  - Post-processing time (ms): mask processing, ROI checking
  - Display rendering time (ms)
  - Total latency: từ capture đến display

#### 6.2. **Queue Metrics**
- ⭐ **Detection Queue Depth**
  - Current queue size
  - Average queue size
  - Max queue size
  - Queue overflow events (frames dropped)

- ⭐ **Frame Drop Rate**
  - Frames dropped due to queue full
  - Frames skipped due to slow processing
  - Missed frame percentage

### 7. **Detection Accuracy Metrics (Độ chính xác)**

#### 7.1. **Detection Reliability**
- ⭐ **Detection Stability**
  - Frame-to-frame detection consistency
  - Detection flicker rate (appear/disappear)
  - False positive indicators (detections that appear briefly)

- ⭐ **Alarm Accuracy Metrics**
  - False alarm rate (số lần báo sai / tổng số alarm)
  - True positive rate (số lần báo đúng / tổng số sự kiện thực)
  - Missed detection rate (sự kiện thực nhưng không báo)

#### 7.2. **Temporal Consistency**
- ⭐ **Detection Duration**
  - Average duration of person detections (seconds)
  - Average duration of coal blockage (seconds)
  - Detection persistence (how long objects remain detected)

- ⭐ **State Transitions**
  - Number of alarm activations per hour
  - Number of alarm deactivations per hour
  - Average alarm duration

### 8. **Resource Utilization Metrics (Sử dụng tài nguyên)**

#### 8.1. **Memory Metrics**
- ⭐ **Memory Usage**
  - CPU memory usage (MB)
  - GPU memory usage (MB) ✅ (đã có một phần)
  - Memory leak indicators (memory growth over time)
  - Peak memory usage

#### 8.2. **GPU Metrics**
- ⭐ **GPU Utilization**
  - GPU utilization percentage (%)
  - GPU temperature (°C)
  - GPU power consumption (W)
  - CUDA kernel execution time breakdown

#### 8.3. **CPU Metrics**
- ⭐ **CPU Usage**
  - CPU utilization per core (%)
  - Thread CPU usage
  - Context switch rate

### 9. **Network & I/O Metrics (Mạng và I/O)**

#### 9.1. **Camera Stream Metrics**
- ⭐ **RTSP Stream Health**
  - Stream latency (ms)
  - Frame loss rate (%)
  - Reconnection frequency
  - Stream quality indicators (resolution drops, artifacts)

#### 9.2. **PLC Communication Metrics**
- ⭐ **PLC Metrics**
  - PLC response time (ms)
  - PLC connection status
  - PLC write success rate (%)
  - PLC communication errors count

### 10. **Business/Operational Metrics (Nghiệp vụ)**

#### 10.1. **Alert Patterns**
- ⭐ **Alert Frequency Analysis**
  - Alerts per hour/day/week
  - Peak alert times
  - Alert correlation (người và than xuất hiện cùng lúc?)

#### 10.2. **Coal Blockage Analysis**
- ⭐ **Blockage Characteristics**
  - Average blockage ratio (%)
  - Max blockage ratio (%)
  - Blockage duration distribution
  - Blockage area coverage

#### 10.3. **Person Detection Analysis**
- ⭐ **Person Presence Patterns**
  - Person detection frequency
  - Average person count per detection
  - Peak detection times
  - Detection duration statistics

---

## 📈 ĐỀ XUẤT THỨ TỰ ƯU TIÊN

### **Priority 1 (Quan trọng nhất - nên implement ngay):**
1. ⭐ **Processing Pipeline Breakdown** - Hiểu được bottleneck ở đâu
2. ⭐ **Detection Queue Depth & Frame Drop Rate** - Đảm bảo không mất frame
3. ⭐ **Confidence Scores Distribution** - Đánh giá chất lượng model
4. ⭐ **Detection Objects Per Frame** - Hiểu tải xử lý
5. ⭐ **Total Latency (Capture → Display)** - Trải nghiệm người dùng

### **Priority 2 (Quan trọng - nên có sớm):**
6. ⭐ **GPU Utilization & Temperature** - Đảm bảo GPU không quá tải
7. ⭐ **CPU Memory Usage** - Tránh memory leak
8. ⭐ **Detection Stability/Flicker Rate** - Cải thiện độ tin cậy
9. ⭐ **ROI Hit Rate** - Validate ROI configuration
10. ⭐ **PLC Communication Metrics** - Đảm bảo giao tiếp ổn định

### **Priority 3 (Hữu ích - có thể bổ sung sau):**
11. ⭐ **Alert Patterns Analysis** - Insights cho nghiệp vụ
12. ⭐ **Detection Duration Statistics** - Hiểu hành vi
13. ⭐ **False Alarm Rate** (cần ground truth)
14. ⭐ **RTSP Stream Health Metrics** - Debug network issues

---

## 🔧 IMPLEMENTATION SUGGESTIONS

### Cấu trúc Metrics nên có:
```python
@dataclass
class DetectionMetrics:
    # Performance
    fps_capture: float
    fps_display: float
    fps_detection: float
    inference_time_avg: float
    inference_time_min: float
    inference_time_max: float
    
    # Detection Quality
    avg_persons_per_frame: float
    avg_coal_per_frame: float
    avg_confidence_person: float
    avg_confidence_coal: float
    roi_hit_rate_person: float
    roi_hit_rate_coal: float
    
    # Pipeline Breakdown
    capture_time_ms: float
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    total_latency_ms: float
    
    # Queue & Drops
    queue_depth_current: int
    queue_depth_avg: int
    frames_dropped: int
    frame_drop_rate: float
    
    # Resources
    gpu_memory_mb: float
    gpu_utilization_pct: float
    cpu_memory_mb: float
    cpu_utilization_pct: float
    
    # Alerts
    person_alerts_total: int
    coal_alerts_total: int
    alerts_per_hour: float
    
    # Stability
    detection_flicker_rate: float
    alarm_activation_count: int
```

### Export/Logging:
- Export metrics mỗi 1 phút/5 phút ra CSV/JSON
- Dashboard real-time metrics
- Alert khi metrics vượt ngưỡng (VD: GPU > 90%, drop rate > 5%)

