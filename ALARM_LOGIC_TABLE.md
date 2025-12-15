# Bảng Logic Cảnh Báo Thực Tế (Theo Code)

## CẢNH BÁO NGƯỜI (Person Alarm)
**Cấu hình:**
- `person_consecutive_threshold = 3` (số frame liên tiếp để kích hoạt cảnh báo)
- `person_no_detection_threshold = 5` (số frame không phát hiện để tắt cảnh báo)

| Frame | person_in_roi | person_consecutive_count | person_no_detection_count | Alarm Active | Hiển thị UI | Hiển thị cảnh báo | Lưu ảnh | Lưu log | Tín hiệu PLC (DB300.DBX6.0) | Ghi chú |
|-------|---------------|-------------------------|--------------------------|--------------|-------------|-------------------|---------|---------|----------------------------|---------|
| F1 | ✅ True | 1 | 0 | ❌ OFF | TẮT | - | - | - | OFF | Counter tăng |
| F2 | ✅ True | 2 | 0 | ❌ OFF | TẮT | - | - | - | OFF | Counter tăng |
| F3 | ✅ True | 3 | 0 | ✅ **ON** | **BẬT** | ✅ Cảnh báo | ✅ person_alert_*.jpg | ✅ LƯU LOG | **ON (gửi)** | **Lần đầu: Bật alarm, gửi PLC, lưu log/ảnh → Reset counter về 0** |
| F4 | ✅ True | 1 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng lại |
| F5 | ✅ True | 2 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng |
| F6 | ✅ True | 3 | 0 | ✅ ON | BẬT | ✅ Cảnh báo | ✅ person_alert_*.jpg | ✅ LƯU LOG | ON (giữ) | **Đạt threshold: Chỉ lưu log/ảnh, KHÔNG gửi lại PLC → Reset counter** |
| F7 | ✅ True | 1 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng lại |
| F8 | ✅ True | 2 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng |
| F9 | ✅ True | 3 | 0 | ✅ ON | BẬT | ✅ Cảnh báo | ✅ person_alert_*.jpg | ✅ LƯU LOG | ON (giữ) | **Đạt threshold: Chỉ lưu log/ảnh → Reset counter** |
| F10 | ❌ False | 0 | 1 | ✅ ON | BẬT | - | - | - | ON (giữ) | Reset consecutive, tăng no_detection |
| F11 | ❌ False | 0 | 2 | ✅ ON | BẬT | - | - | - | ON (giữ) | No_detection tăng |
| F12 | ❌ False | 0 | 3 | ✅ ON | BẬT | - | - | - | ON (giữ) | No_detection tăng |
| F13 | ❌ False | 0 | 4 | ✅ ON | BẬT | - | - | - | ON (giữ) | No_detection tăng |
| F14 | ❌ False | 0 | 5 | ❌ **OFF** | **TẮT** | - | - | - | **OFF (gửi)** | **Tắt alarm, gửi PLC OFF → Reset cả 2 counter về 0** |
| F15 | ❌ False | 0 | 0 | ❌ OFF | TẮT | - | - | - | OFF (giữ) | Cả 2 counter đã reset |
| ... | ... | 0 | 0 | ❌ OFF | TẮT | - | - | - | OFF (giữ) | Giữ trạng thái tắt |

---

## CẢNH BÁO THAN (Coal Alarm)
**Cấu hình:**
- `coal_consecutive_threshold = 5` (số frame liên tiếp để kích hoạt cảnh báo)
- `coal_no_blockage_threshold = 5` (số frame không tắc để tắt cảnh báo)
- `coal_ratio_threshold = 73.0%` (ngưỡng tỷ lệ than trong ROI)

| Frame | coal_ratio | coal_consecutive_count | coal_no_blockage_count | Alarm Active | Hiển thị UI | Hiển thị cảnh báo | Lưu ảnh | Lưu log | Tín hiệu PLC (DB300.DBX6.X) | Ghi chú |
|-------|------------|------------------------|------------------------|--------------|-------------|-------------------|---------|---------|----------------------------|---------|
| F1 | 75% ✅ | 1 | 0 | ❌ OFF | TẮT | - | - | - | OFF | Ratio >= 73%, counter tăng |
| F2 | 76% ✅ | 2 | 0 | ❌ OFF | TẮT | - | - | - | OFF | Counter tăng |
| F3 | 74% ✅ | 3 | 0 | ❌ OFF | TẮT | - | - | - | OFF | Counter tăng |
| F4 | 77% ✅ | 4 | 0 | ❌ OFF | TẮT | - | - | - | OFF | Counter tăng |
| F5 | 75% ✅ | 5 | 0 | ✅ **ON** | **BẬT** | ✅ Cảnh báo | ✅ coal_alert_*.jpg | ✅ LƯU LOG | **ON (gửi)** | **Lần đầu: Bật alarm, gửi PLC, lưu log/ảnh → Reset counter về 0** |
| F6 | 76% ✅ | 1 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng lại |
| F7 | 74% ✅ | 2 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng |
| F8 | 75% ✅ | 3 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng |
| F9 | 77% ✅ | 4 | 0 | ✅ ON | BẬT | - | - | - | ON (giữ) | Counter tăng |
| F10 | 74% ✅ | 5 | 0 | ✅ ON | BẬT | ✅ Cảnh báo | ✅ coal_alert_*.jpg | ✅ LƯU LOG | ON (giữ) | **Đạt threshold: Chỉ lưu log/ảnh, KHÔNG gửi lại PLC → Reset counter** |
| F11 | 50% ❌ | 0 | 1 | ✅ ON | BẬT | - | - | - | ON (giữ) | Ratio < 73%, reset consecutive, tăng no_blockage |
| F12 | 45% ❌ | 0 | 2 | ✅ ON | BẬT | - | - | - | ON (giữ) | No_blockage tăng |
| F13 | 40% ❌ | 0 | 3 | ✅ ON | BẬT | - | - | - | ON (giữ) | No_blockage tăng |
| F14 | 35% ❌ | 0 | 4 | ✅ ON | BẬT | - | - | - | ON (giữ) | No_blockage tăng |
| F15 | 30% ❌ | 0 | 5 | ❌ **OFF** | **TẮT** | - | - | - | **OFF (gửi)** | **Tắt alarm, gửi PLC OFF → Reset cả 2 counter về 0** |
| F16 | 25% ❌ | 0 | 0 | ❌ OFF | TẮT | - | - | - | OFF (giữ) | Cả 2 counter đã reset |
| ... | ... | 0 | 0 | ❌ OFF | TẮT | - | - | - | OFF (giữ) | Giữ trạng thái tắt |

---

## ĐIỂM QUAN TRỌNG

### 1. **Bật cảnh báo:**
- **Lần đầu đạt threshold:** Bật alarm + Gửi PLC ON + Lưu log + Lưu ảnh + Reset counter
- **Các lần đạt threshold sau:** Chỉ Lưu log + Lưu ảnh + Reset counter (KHÔNG gửi lại PLC vì đã ON)

### 2. **Tắt cảnh báo:**
- Khi đạt `no_detection_threshold` hoặc `no_blockage_threshold`: Tắt alarm + Gửi PLC OFF + Reset **CẢ 2 COUNTER** về 0

### 3. **Counter logic:**
- Khi có phát hiện: Reset `no_detection_count` = 0, tăng `consecutive_count`
- Khi không phát hiện: Reset `consecutive_count` = 0, tăng `no_detection_count`

### 4. **Lưu log/ảnh:**
- **Mỗi lần đạt threshold** đều lưu log và ảnh (không chỉ lần đầu)
- Điều này giúp theo dõi tất cả các sự kiện quan trọng

