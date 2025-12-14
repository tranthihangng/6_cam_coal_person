# 📚 Hướng dẫn Quản lý Version với Git & GitHub

## 🔄 Workflow Cơ bản

### 1. Khởi tạo và Push code lần đầu

```bash
cd coal_monitoring

# Khởi tạo Git repository
git init

# Thêm tất cả file (trừ các file trong .gitignore)
git add .

# Commit
git commit -m "Initial commit - Coal Monitoring System v1.0"

# Kết nối với GitHub repo (thay YOUR_USERNAME và REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push lên GitHub
git branch -M main
git push -u origin main
```

### 2. Đánh dấu Version với Tag

Sau khi push code, đánh dấu version quan trọng:

```bash
# Tag version
git tag -a v1.0 -m "Version 1.0 - Hệ thống cơ bản"

# Push tag lên GitHub
git push origin v1.0

# Hoặc push tất cả tags
git push origin --tags
```

### 3. Quay lại Version cũ để sửa đổi ⚠️

**❌ SAI - Không nên làm:**
```bash
git checkout v1.0  # Bạn sẽ ở "detached HEAD" state
# Nếu commit ở đây, code có thể bị mất!
```

**✅ ĐÚNG - Nên làm:**

#### Cách 1: Tạo Branch mới từ Tag (KHUYẾN NGHỊ)

```bash
# Tạo branch mới từ tag v1.0
git checkout -b branch-v1.0 v1.0

# Bây giờ bạn có thể sửa và commit bình thường
# ... sửa code ...
git add .
git commit -m "Fix bug trong v1.0"
git push origin branch-v1.0
```

#### Cách 2: Checkout về tag rồi tạo branch

```bash
# Checkout về tag
git checkout v1.0

# Tạo branch mới từ đây
git checkout -b fix-v1.0

# Bây giờ có thể sửa đổi
# ... sửa code ...
git add .
git commit -m "Fix bug"
git push origin fix-v1.0
```

#### Cách 3: Xem code version cũ mà không sửa

```bash
# Chỉ xem code, không sửa
git checkout v1.0

# Sau khi xem xong, quay lại branch chính
git checkout main
```

### 4. Workflow Hoàn chỉnh

```bash
# 1. Làm việc trên main branch
git checkout main
git pull origin main

# 2. Tạo branch mới cho tính năng
git checkout -b feature/new-detection

# 3. Code và commit
git add .
git commit -m "Thêm tính năng detection mới"
git push origin feature/new-detection

# 4. Merge vào main (hoặc tạo Pull Request trên GitHub)
git checkout main
git merge feature/new-detection
git push origin main

# 5. Tag version mới
git tag -a v1.1 -m "Version 1.1 - Thêm tính năng detection mới"
git push origin v1.1

# 6. Nếu cần sửa version cũ (v1.0)
git checkout -b hotfix-v1.0 v1.0
# ... sửa code ...
git commit -m "Fix critical bug trong v1.0"
git tag -a v1.0.1 -m "Version 1.0.1 - Fix bug"
git push origin v1.0.1
```

## 📋 Các Lệnh Hữu ích

### Xem tất cả tags
```bash
git tag
git tag -l "v1.*"  # Xem tags theo pattern
```

### Xem thông tin tag
```bash
git show v1.0
```

### Xóa tag (local)
```bash
git tag -d v1.0
```

### Xóa tag trên GitHub
```bash
git push origin --delete v1.0
```

### Xem lịch sử commits
```bash
git log --oneline
git log --oneline --graph --all  # Với biểu đồ
git log --oneline --graph --all --decorate  # Với tags
```

### Xem sự khác biệt giữa các version
```bash
git diff v1.0 v1.1  # So sánh 2 tags
git diff v1.0 main  # So sánh tag với branch
```

## 🎯 Best Practices

1. **Luôn tạo branch mới** khi checkout về tag cũ để sửa
2. **Đặt tên tag có ý nghĩa**: `v1.0`, `v1.1`, `v2.0-beta`
3. **Viết message rõ ràng** khi tag: `git tag -a v1.0 -m "Mô tả chi tiết"`
4. **Không commit** khi đang ở detached HEAD state (sau `git checkout v1.0`)
5. **Pull trước khi push**: `git pull origin main` trước khi push

## ⚠️ Lưu ý quan trọng

- **Detached HEAD**: Khi `git checkout v1.0`, bạn sẽ ở trạng thái "detached HEAD". Nếu commit ở đây mà không tạo branch, commit sẽ bị mất khi chuyển branch khác.
- **Luôn tạo branch** nếu muốn sửa code từ tag cũ.
- **Tag không thay đổi**: Tag trỏ đến một commit cụ thể và không di chuyển. Nếu sửa code từ tag, hãy tạo tag mới (vd: v1.0.1).

## 🔗 Tài liệu tham khảo

- [Git Tags Documentation](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [Git Branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)

