"""
Test script to verify YOLOv8 works before integrating
"""
print("Testing AI Detection System...")
print("=" * 50)

try:
    import cv2
    print("✅ OpenCV imported successfully")
except Exception as e:
    print(f"❌ OpenCV failed: {e}")
    exit(1)

try:
    from ultralytics import YOLO
    print("✅ Ultralytics imported successfully")
except Exception as e:
    print(f"❌ Ultralytics failed: {e}")
    exit(1)

try:
    # Load YOLOv8 nano model (smallest, fastest)
    print("\n📦 Downloading YOLOv8 nano model (first time only)...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8 model loaded successfully")

    # Test detection on a sample
    print("\n🔍 Running test detection...")
    results = model('https://ultralytics.com/images/bus.jpg')

    print("✅ Detection completed!")
    print(f"   Detected {len(results[0].boxes)} objects")

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"   - {class_name}: {confidence:.2%}")

    print("\n" + "=" * 50)
    print("🎉 AI System Ready! Proceeding to integration...")

except Exception as e:
    print(f"❌ Detection failed: {e}")
    exit(1)