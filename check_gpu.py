"""
Script kiểm tra GPU và verify models đang chạy trên GPU

Usage:
    python check_gpu.py [model_path]
    
    Ví dụ:
    python check_gpu.py best_segment_26_11.pt
"""

import sys
import os
import torch
from detection.model_loader import MultiModelLoader

def main():
    print("="*60)
    print("🔍 KIỂM TRA GPU VÀ MODEL STATUS")
    print("="*60)
    
    # 1. Kiểm tra GPU cơ bản
    print("\n1️⃣ CUDA Availability:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA is available")
        print(f"   📱 Device: {torch.cuda.get_device_name(0)}")
        print(f"   🔢 CUDA Version: {torch.version.cuda}")
        print(f"   📦 Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("   ❌ CUDA is NOT available")
        print("   ⚠️  Models sẽ chạy trên CPU (chậm hơn 5-10x)")
        return
    
    # 2. GPU Memory Info
    print("\n2️⃣ GPU Memory:")
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"   📊 Allocated: {allocated:.1f} MB")
    print(f"   📦 Reserved: {reserved:.1f} MB")
    print(f"   💾 Total: {total:.1f} MB")
    print(f"   📈 Usage: {(allocated/total*100):.1f}%")
    
    # 3. Kiểm tra Model Loader hoặc load model từ command line
    print("\n3️⃣ Model Loader Status:")
    loader = MultiModelLoader.get_instance()
    
    # Nếu có model path từ command line, thử load
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"   🔄 Đang load model từ: {model_path}")
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(model_path):
                print(f"   ❌ File không tồn tại: {model_path}")
            else:
                memory_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                success = loader.load(
                    model_id="test_model",
                    model_path=model_path,
                    model_name="Test Model",
                    cameras=[1]
                )
                if success:
                    memory_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    print(f"   ✅ Model loaded successfully!")
                    if torch.cuda.is_available() and memory_after > memory_before:
                        print(f"   📊 GPU Memory tăng: {memory_before:.1f} MB → {memory_after:.1f} MB (+{memory_after-memory_before:.1f} MB)")
                else:
                    print(f"   ❌ Không thể load model")
        except Exception as e:
            print(f"   ❌ Lỗi khi load model: {e}")
    
    if not loader._models:
        print("   ⚠️  Chưa có model nào được load")
        print("   💡 Cách 1: Chạy ứng dụng chính để load models")
        print("   💡 Cách 2: python check_gpu.py <model_path>")
        print("   💡 Ví dụ: python check_gpu.py best_segment_26_11.pt")
    else:
        print(f"   ✅ Đã load {len(loader._models)} model(s)")
        loader.print_gpu_status()
    
    # 4. Test inference trên GPU (nếu có model)
    if loader._models:
        print("\n4️⃣ Test Inference trên GPU:")
        try:
            import numpy as np
            from ultralytics import YOLO
            
            # Lấy model đầu tiên
            model_id = list(loader._models.keys())[0]
            model = loader._models[model_id]
            
            # Tạo dummy frame
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Test inference
            import time
            start = time.time()
            results = model.predict(
                dummy_frame,
                conf=0.5,
                verbose=False,
                task='segment',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            elapsed = (time.time() - start) * 1000
            
            # Verify GPU memory tăng sau inference
            after_allocated = torch.cuda.memory_allocated() / 1024**2
            
            print(f"   ⏱️  Inference time: {elapsed:.1f}ms")
            print(f"   📊 GPU Memory sau inference: {after_allocated:.1f} MB")
            
            if after_allocated > allocated:
                print(f"   ✅ GPU được sử dụng (memory tăng {after_allocated-allocated:.1f} MB)")
            else:
                print(f"   ⚠️  GPU memory không tăng - có thể đang chạy trên CPU")
                
        except Exception as e:
            print(f"   ❌ Lỗi test inference: {e}")
    
    print("\n" + "="*60)
    print("✅ Hoàn tất kiểm tra")
    print("="*60)

if __name__ == "__main__":
    main()

