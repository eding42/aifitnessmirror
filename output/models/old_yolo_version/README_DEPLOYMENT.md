# AI Fitness Mirror - Model Deployment Guide

## Overview
This directory contains the trained models for the AI Fitness Mirror system.

**Created:** 2026-01-31 17:15:36
**Training Accuracy:** 87.56%
**Validation Accuracy:** 92.52%

## Model Files

### 1. YOLO v8 Nano Pose (`yolov8n-pose.onnx`)
- **Purpose:** Extract skeletal keypoints from images
- **Input:** RGB image (640 × 640 × 3)
- **Output:** 17 COCO keypoints with (x, y, confidence)
- **Use:** First stage of the pipeline

### 2. Combined Model (`fitness_mirror_combined.onnx`) ⭐ MAIN MODEL
- **Purpose:** Activity classification + form assessment
- **Input:** Keypoints (51 features: 17 keypoints × 3 values)
- **Outputs:**
  - `predicted_class`: Activity class index (INT64)
  - `confidence`: Prediction confidence (FLOAT, 0-1)
  - `form_error`: Reconstruction error (FLOAT, lower is better)
  - `class_probs`: All class probabilities (FLOAT[6])
  - `reconstructed`: Reconstructed keypoints (FLOAT[51])

### Activity Classes
1. hip thrusts
2. jumping jacks
3. lunges
4. pullups
5. pushups
6. squats

## Deployment Pipeline

```
Image (any size)
    ↓
YOLO v8 Pose Model (yolov8n-pose.onnx)
    ↓
Keypoints (17 × 3)
    ↓
Normalize & Flatten → (51,)
    ↓
Combined Model (fitness_mirror_combined.onnx)
    ↓
Results: [class, confidence, form_quality]
```

## Usage Example (Python with ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

# Load models
yolo_session = ort.InferenceSession('yolov8n-pose.onnx')
combined_session = ort.InferenceSession('fitness_mirror_combined.onnx')

# Step 1: Extract keypoints from image
image = preprocess_image(your_image)  # Resize to 640x640, normalize
yolo_output = yolo_session.run(None, {'images': image})
keypoints = extract_keypoints(yolo_output)  # Get first person's keypoints

# Step 2: Normalize and flatten
h, w = original_image_height, original_image_width
keypoints[:, 0] /= w  # Normalize x
keypoints[:, 1] /= h  # Normalize y
keypoints_flat = keypoints.flatten().reshape(1, 51)

# Step 3: Run combined model
outputs = combined_session.run(None, {'keypoints': keypoints_flat})
predicted_class = outputs[0][0]
confidence = outputs[1][0]
form_error = outputs[2][0]

# Step 4: Calculate form quality
threshold = reconstruction_thresholds[class_names[predicted_class]]['threshold']
form_quality = max(0, min(100, 100 * (1 - form_error / threshold)))

print(f"Activity: {class_names[predicted_class]}")
print(f"Confidence: {confidence*100:.1f}%")
print(f"Form Quality: {form_quality:.1f}%")
```

## TFLite Conversion

To convert the combined model to TFLite format:

1. Open `tflitequantizer.ipynb`
2. Load `fitness_mirror_combined.onnx`
3. Apply quantization (INT8 or FLOAT16)
4. Export to `.tflite` format
5. Deploy to edge device (Nuvoton M55, etc.)

## Form Quality Calculation

Form quality is calculated using reconstruction error from the correctness model:

```
form_quality = 100 × (1 - error / threshold)
```

Where `threshold` is the baseline error for each activity class:

- hip thrusts: 0.007664
- jumping jacks: 0.010244
- lunges: 0.006416
- pullups: 0.010763
- pushups: 0.009775
- squats: 0.005078

## Model Architecture

### Activity Classifier
- Input: 51 features (flattened keypoints)
- Hidden: 256 → 256 → 128 → 64
- Output: 6 classes
- Activation: ReLU, Dropout, BatchNorm
- Loss: Cross-Entropy with class weights

### Correctness Model (Autoencoder)
- Input: 51 features + 6 class one-hot
- Encoder: 128 → 64 → 32
- Decoder: 32 → 64 → 128 → 51
- Loss: MSE (reconstruction error)

## Notes

- **YOLO Processing:** YOLO v8 Pose is kept separate because it requires image preprocessing
- **Combined Model:** Activity + Correctness models are merged for efficiency
- **Edge Deployment:** Total model size < 5 MB, suitable for embedded devices
- **Real-time:** Models are optimized for fast inference (<50ms on CPU)

## Support

For issues or questions, refer to the main notebook: `pose_activity_recognition_yolonano.ipynb`
