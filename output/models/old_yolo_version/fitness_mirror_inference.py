"""
AI Fitness Mirror - Unified Inference Runtime
Single entry point for YOLO + Activity Classifier + Correctness Model

Required by sponsor as unified deployment solution.
Usage: python fitness_mirror_inference.py <image_path>
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

class FitnessMirrorUnified:
    """Unified model runtime - chains YOLO, Activity Classifier, and Correctness Model"""

    def __init__(self, models_dir):
        print("Loading AI Fitness Mirror models...")

        # Load YOLO Pose model
        yolo_path = Path(models_dir) / "yolov8n-pose.onnx"
        if yolo_path.exists():
            self.yolo_session = ort.InferenceSession(str(yolo_path))
            print(f"✓ YOLO Pose loaded: {yolo_path.name}")
        else:
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

        # Load Activity Classifier
        activity_path = Path(models_dir) / "activity_classifier.onnx"
        self.activity_session = ort.InferenceSession(str(activity_path))
        print(f"✓ Activity Classifier loaded: {activity_path.name}")

        # Load Correctness Model
        correctness_path = Path(models_dir) / "correctness_model.onnx"
        self.correctness_session = ort.InferenceSession(str(correctness_path))
        print(f"✓ Correctness Model loaded: {correctness_path.name}")

        # Load metadata
        import json
        metadata_path = Path(models_dir) / "model_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.classes = self.metadata['classes']
        self.class_thresholds = self.metadata.get('class_reconstruction_errors', {})
        print(f"✓ Metadata loaded: {len(self.classes)} classes")
        print("Models ready!\n")

    def predict(self, image_path_or_array):
        """
        Single method for end-to-end inference
        Input: image path or numpy array
        Output: dict with activity, confidence, form_score
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array

        # Preprocess for YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 640))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_input = np.expand_dims(image_transposed, axis=0)

        # Step 1: YOLO Pose Detection
        yolo_outputs = self.yolo_session.run(None, {self.yolo_session.get_inputs()[0].name: image_input})

        # Extract keypoints from YOLO output
        # YOLO v8 Pose output format: [batch, num_boxes, 56]
        # Last 51 values are keypoints (17 * 3)
        predictions = yolo_outputs[0]
        if predictions.shape[1] == 0:
            return {"error": "No pose detected"}

        # Take first detection and extract keypoints
        first_detection = predictions[0, 0, :]
        keypoints_raw = first_detection[-51:]  # Last 51 values

        # Normalize keypoints
        keypoints = keypoints_raw.reshape(17, 3)
        keypoints[:, 0] /= 640.0  # Normalize x
        keypoints[:, 1] /= 640.0  # Normalize y
        keypoints_flat = keypoints.reshape(1, -1).astype(np.float32)

        # Step 2: Activity Classification
        activity_outputs = self.activity_session.run(
            None,
            {self.activity_session.get_inputs()[0].name: keypoints_flat}
        )
        class_logits = activity_outputs[0][0]

        # Softmax
        exp_logits = np.exp(class_logits - np.max(class_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

        # Step 3: Form Correctness Assessment
        class_onehot = np.zeros((1, len(self.classes)), dtype=np.float32)
        class_onehot[0, predicted_class_idx] = 1.0

        correctness_outputs = self.correctness_session.run(
            None,
            {
                self.correctness_session.get_inputs()[0].name: keypoints_flat,
                self.correctness_session.get_inputs()[1].name: class_onehot
            }
        )
        reconstructed = correctness_outputs[0]

        # Calculate MSE
        reconstruction_error = np.mean((keypoints_flat - reconstructed) ** 2)

        # Calculate form score
        if predicted_class in self.class_thresholds:
            threshold = self.class_thresholds[predicted_class]['threshold']
            form_score = max(0, min(100, 100 * (1 - reconstruction_error / threshold)))
        else:
            form_score = 50.0

        return {
            'activity': predicted_class,
            'confidence': float(confidence * 100),
            'form_score': float(form_score),
            'reconstruction_error': float(reconstruction_error),
            'probabilities': {self.classes[i]: float(probabilities[i] * 100) for i in range(len(self.classes))},
            'keypoints': keypoints_flat[0].tolist()
        }

# Example usage
if __name__ == "__main__":
    import sys

    # Initialize unified model
    models_dir = "output/models"  # Adjust path as needed
    model = FitnessMirrorUnified(models_dir)

    # Test on image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = model.predict(image_path)

        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Activity: {result['activity']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Form Score: {result['form_score']:.1f}%")
            print(f"\nAll probabilities:")
            for activity, prob in result['probabilities'].items():
                print(f"  {activity}: {prob:.1f}%")
    else:
        print("Usage: python fitness_mirror_inference.py <image_path>")
