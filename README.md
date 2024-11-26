---

# **Object Detection and Pose Estimation Pipeline**

This project integrates **object detection**, **pose estimation**, and **activity recognition** into a streamlined processing pipeline. Built using **TensorFlow**, **PyTorch**, and other advanced tools, the system is designed for efficient and accurate analysis of video and image data. The modular structure allows customization for a variety of use cases.

---

## **Features**

### **Object Detection**
- Utilizes YOLO-based models and TensorFlow for bounding box and mask prediction.
- Efficient multithreaded detection processing for high performance.

### **Pose Estimation**
- Integrates MoveNet for precise human pose estimation.
- Extracts and normalizes keypoints relative to anatomical reference points (e.g., hips).

### **Activity Recognition**
- Processes sequences of frames using LSTM for temporal activity analysis.
- Captures spatial relationships and object distances for richer insights.

### **Multimodal Integration**
- Merges detection and pose data for enhanced interpretability.
- Calculates distances from keypoints to objects for detailed spatial information.

---

## **Requirements**

### **Environment**
- **Python**: 3.8+
- **TensorFlow**: 2.x
- **PyTorch**: 1.9+
- **OpenCV**: 4.x
- **Label Studio ML Backend**

### **Dependencies**
Install dependencies using `pip`:
```bash
pip install tensorflow torch torchvision opencv-python label-studio-ml numpy matplotlib imageio
```

---

## **Setup**

### **Environment Variables**
Set up the following environment variables:

| Variable                | Description                                      | Default Value                     |
|-------------------------|--------------------------------------------------|-----------------------------------|
| `MODEL_DIR`             | Path to the saved object detection model.       | `/default/model/path`             |
| `LABEL_STUDIO_TOKEN`    | Token for Label Studio integration.             | `default-token`                   |
| `PATH_TO_LABELS`        | Path to the label map for object detection.      | `/default/label/map/path`         |
| `LABEL_STUDIO_URL`      | Base URL for Label Studio API.                   | `http://localhost:8080`           |
| `CACHE_DIR`             | Cache directory for TensorFlow Hub models.      | `/mnt/cache-dir`                  |
| `LSTM_MODEL_PATH`       | Path to the LSTM model for activity recognition. | `/mnt/lstm-model-directory`       |

---

## **Usage**

### **1. Object Detection**
Run object detection on images or video frames:
```python
detection_model = load_model(MODEL_DIR)
```

### **2. Pose Estimation**
Perform pose estimation on preprocessed frames:
```python
keypoints_with_scores = run_inference(movenet, image, crop_region, [input_size, input_size])
```

### **3. Activity Recognition**
Predict temporal sequences using the LSTM model:
```python
activity_predictions = LSTM_LOADED_MODEL.predict(sequence_buffer)
```

---

## **Execution**

### **Video Processing**
To process a video for detection, pose estimation, and activity recognition:
```bash
python video_processing_pipeline.py --video /path/to/video.mp4
```

### **Single Image**
For single image analysis:
```bash
python image_processing_pipeline.py --image /path/to/image.jpg
```

---

## **Outputs**

### **1. Annotated Visualizations**
- Saved as images with bounding boxes, pose keypoints, and masks.

### **2. JSON Metadata**
- Includes:
  - Bounding box coordinates.
  - Mask data.
  - Keypoints.
  - Activity insights.

### **3. Activity Logs**
- Stores normalized keypoints and spatial relationships for each frame.

---

## **Project Structure**

```plaintext
.
├── models/                 # Model weights and configurations
├── data/                   # Label maps and datasets
├── utils/                  # Helper functions for preprocessing and postprocessing
├── video_processing_pipeline.py   # Main script for video analysis
├── image_processing_pipeline.py   # Script for image analysis
├── README.md               # Project documentation
```

---

## **Future Improvements**

### **1. Real-Time Processing**
- Enable real-time video inference with GPU acceleration.

### **2. Enhanced Metrics**
- Add mAP and precision-recall calculations for model evaluation.

### **3. Activity Insights**
- Extend LSTM model for multi-class activity recognition.

---

## **Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## **License**
This project is licensed under the **MIT License**.
