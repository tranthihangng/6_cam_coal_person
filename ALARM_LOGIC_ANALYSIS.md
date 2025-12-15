# Phân tích Logic Cảnh Báo

## 1. CẢNH BÁO NGƯỜI (Person Alarm)

### Cấu hình mặc định:
- `person_consecutive_threshold` = 3 frames (số frame liên tiếp để BẬT cảnh báo)
- `person_no_detection_threshold` = 5 frames (số frame để TẮT cảnh báo)

### Logic BẬT cảnh báo (person_detected = True):
```
1. Reset _person_no_detection_count = 0
2. Tăng _person_consecutive_count += 1
3. Nếu _person_consecutive_count >= 3 VÀ alarm chưa active:
   - Bật alarm
   - Gửi PLC
   - Lưu log/ảnh
   - Reset _person_consecutive_count = 0
```

### Logic TẮT cảnh báo (person_detected = False):
```
1. Tăng _person_no_detection_count += 1
2. Nếu _person_no_detection_count >= 5 VÀ alarm đang active:
   - Reset _person_consecutive_count = 0
   - Tắt alarm
   - Gửi PLC
   - KHÔNG reset _person_no_detection_count (VẤN ĐỀ!)
```

### ⚠️ VẤN ĐỀ PHÁT HIỆN:
- Sau khi tắt cảnh báo, `_person_no_detection_count` không được reset
- Nếu người xuất hiện lại ngay sau đó:
  - `_person_no_detection_count` vẫn giữ giá trị >= 5
  - Khi người xuất hiện (person_detected = True), counter này được reset về 0 (đúng)
  - NHƯNG nếu người biến mất lại trước khi đạt threshold, counter này sẽ tiếp tục từ giá trị cũ

## 2. CẢNH BÁO THAN (Coal Alarm)

### Cấu hình mặc định:
- `coal_consecutive_threshold` = 5 frames
- `coal_no_blockage_threshold` = 5 frames
- `coal_ratio_threshold` = 73.0%

### Logic BẬT cảnh báo (coal_ratio >= 73%):
```
1. Reset _coal_no_blockage_count = 0
2. Tăng _coal_consecutive_count += 1
3. Nếu _coal_consecutive_count >= 5 VÀ alarm chưa active:
   - Bật alarm
   - Gửi PLC
   - Lưu log/ảnh
   - Reset _coal_consecutive_count = 0
```

### Logic TẮT cảnh báo (coal_ratio < 73%):
```
1. Tăng _coal_no_blockage_count += 1
2. Nếu _coal_no_blockage_count >= 5 VÀ alarm đang active:
   - Reset _coal_consecutive_count = 0
   - Tắt alarm
   - Gửi PLC
   - KHÔNG reset _coal_no_blockage_count (VẤN ĐỀ!)
```

## 3. KẾT LUẬN VÀ ĐỀ XUẤT SỬA

### Vấn đề:
Sau khi tắt cảnh báo, counter "no detection/no blockage" không được reset, có thể gây logic không mong muốn.

### Đề xuất sửa:
Sau khi tắt cảnh báo, nên reset cả 2 counter để đảm bảo logic sạch sẽ:
- Reset counter "consecutive" = 0
- Reset counter "no detection/no blockage" = 0 (HIỆN TẠI THIẾU)

