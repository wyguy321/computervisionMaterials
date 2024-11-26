README

Object Detection and Pose Estimation Pipeline

This project integrates object detection, pose estimation, and activity recognition into a streamlined processing pipeline. Using TensorFlow, PyTorch, and other tools, the system enables efficient and accurate analysis of video and image data, with modularity for various use cases.

Features

Object Detection:
Utilizes YOLO-based models and TensorFlow for bounding box and mask prediction.
Efficient postprocessing with multithreaded detection capabilities.
Pose Estimation:
Integrates MoveNet for human pose estimation.
Extracts and normalizes keypoints relative to anatomical reference points (e.g., hips).
Activity Recognition:
Processes sequences of frames using LSTM for temporal activity analysis.
Captures object distances and spatial relationships.
Multimodal Integration:
Merges detection and pose data for enhanced interpretability.
Calculates distances from keypoints to objects for detailed spatial insights.
Requirements

Environment
Python 3.8+
TensorFlow 2.x
PyTorch 1.9+
OpenCV 4.x
Label Studio ML Backend
Dependencies
Install dependencies via pip:

pip install tensorflow torch torchvision opencv-python label-studio-ml numpy matplotlib imageio
Setup

Environment Variables
Define the following environment variables:

MODEL_DIR: Path to the saved object detection model.
LABEL_STUDIO_TOKEN: Token for Label Studio integration.
PATH_TO_LABELS: Path to the label map for object detection.
LABEL_STUDIO_URL: Base URL for Label Studio API.
For pose estimation:

CACHE_DIR: Cache directory for TensorFlow Hub models.
LSTM_MODEL_PATH: Path to the LSTM model for activity recognition.
Loading Models
Object detection models can be loaded from TensorFlow SavedModel or YOLO weights. Pose estimation models are fetched from TensorFlow Hub or loaded locally.

Usage

1. Object Detection
Run object detection on images or video frames:

detection_model = load_model(MODEL_DIR)
2. Pose Estimation
Perform pose estimation on preprocessed frames:

keypoints_with_scores = run_inference(movenet, image, crop_region, [input_size, input_size])
3. Activity Recognition
Predict temporal sequences using the LSTM model:

activity_predictions = LSTM_LOADED_MODEL.predict(sequence_buffer)
Execution

Video Processing
To process a video for detection, pose estimation, and activity recognition:

python video_processing_pipeline.py --video /path/to/video.mp4
Single Image
For single image analysis:

python image_processing_pipeline.py --image /path/to/image.jpg
Outputs

Annotated Visualizations:
Saved as images with bounding boxes, pose keypoints, and masks.
JSON Metadata:
Includes bounding box coordinates, mask data, keypoints, and activity insights.
Activity Logs:
Stores normalized keypoints and spatial relationships for each frame.
Project Structure

.
├── models/                 # Model weights and configurations
├── data/                   # Label maps and datasets
├── utils/                  # Helper functions for preprocessing and postprocessing
├── video_processing_pipeline.py   # Main script for video analysis
├── image_processing_pipeline.py   # Script for image analysis
├── README.md               # Project documentation
Future Improvements

Real-Time Processing:
Enable real-time video inference with GPU acceleration.
Enhanced Metrics:
Add mAP and precision-recall calculations for model evaluation.
Activity Insights:
Extend LSTM model for multi-class activity recognition.
Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License

This project is licensed under the MIT License.

