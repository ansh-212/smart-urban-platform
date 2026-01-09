from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class HazardDetector:
    def __init__(self):
        """Initialize YOLOv8 model"""
        print("🤖 Initializing AI Detection System...")

        # Load YOLOv8 nano (fast, good for demo)
        self.model = YOLO('yolov8n.pt')

        # Classes we care about for urban monitoring
        self.hazard_classes = {
            'pothole': ['sports ball'],  # Placeholder - in production use custom model
            'vehicle': ['car', 'truck', 'bus', 'motorcycle'],
            'person': ['person'],
            'bicycle': ['bicycle']
        }

        print("✅ AI Detection System Ready!")

    def detect_objects(self, image_path):
        """
        Detect objects in an image
        Returns list of detections with bounding boxes
        """
        try:
            results = self.model(image_path, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    # Categorize detection
                    category = self._categorize_detection(class_name)

                    detections.append({
                        'type': class_name,
                        'category': category,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                        }
                    })

            return detections

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []

    def _categorize_detection(self, class_name):
        """Categorize detected object into hazard types"""
        for category, classes in self.hazard_classes.items():
            if class_name in classes:
                return category
        return 'other'

    def detect_and_annotate(self, image_path, output_path=None):
        """
        Detect objects and return annotated image
        """
        try:
            results = self.model(image_path, verbose=False)

            # Get annotated image
            annotated = results[0].plot()

            if output_path:
                cv2.imwrite(output_path, annotated)

            return annotated

        except Exception as e:
            print(f"❌ Annotation error: {e}")
            return None

    def count_by_category(self, detections):
        """Count detections by category"""
        counts = {}
        for detection in detections:
            category = detection['category']
            counts[category] = counts.get(category, 0) + 1
        return counts

# Singleton instance
_detector_instance = None

def get_detector():
    """Get or create detector instance (singleton pattern)"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HazardDetector()
    return _detector_instance