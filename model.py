from typing import List, Dict, Optional
import tensorflow as tf
import cv2
from label_studio_ml.model import LabelStudioMLBase
import numpy as np
from object_detection.utils import label_map_util
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
import json
import uuid
import os

# Environment variables with default values
MODEL_DIR = os.getenv('MODEL_DIR', '/default/model/path')
TOKEN = os.getenv('LABEL_STUDIO_TOKEN', 'default-token')
PATH_TO_LABELS = os.getenv('PATH_TO_LABELS', '/default/label/map/path')
URL = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')


def load_model(filename: str):
    """Load a TensorFlow model from a given path."""
    return tf.saved_model.load(str(filename))


class NewModel(LabelStudioMLBase):
    """Custom Label Studio model for processing and predicting object detections."""

    def process_detections(
        self, start, end, detection_scores, score_threshold, detection_boxes,
        detection_masks, detection_classes, category_index, image,
        original_width, original_height, task_results, lock
    ):
        """Process a subset of detections and append results to task_results."""
        image_h, image_w, _ = image.shape
        local_results = []

        for i in range(start, end):
            if detection_scores[i] > score_threshold:
                box = detection_boxes[i]
                mask = detection_masks[i]
                class_id = detection_classes[i]

                # Scale box coordinates to image size
                box_coords = [
                    int(box[1] * image_w),
                    int(box[0] * image_h),
                    int(box[3] * image_w),
                    int(box[2] * image_h),
                ]
                cv2.rectangle(image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2)

                class_name = category_index[class_id]["name"]
                cv2.putText(image, f"Class: {class_name}", (box_coords[0], box_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Process mask
                mask = (mask > 0.5).astype(np.uint8) * 255
                resized_mask = cv2.resize(mask, (box_coords[2] - box_coords[0], box_coords[3] - box_coords[1]))
                contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Extract coordinates for each contour
                coordinates = []
                for cnt in contours:
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    for point in approx:
                        x, y = point[0]
                        x += box_coords[0]
                        y += box_coords[1]
                        x, y = self.convert_to_ls(x, y, original_width, original_height)
                        coordinates.append([x, y])

                result = {
                    "original_width": image_w,
                    "original_height": image_h,
                    "image_rotation": 0,
                    "score": float(detection_scores[i]),
                    "value": {
                        "points": coordinates,
                        "polygonlabels": [class_name],
                    },
                    "id": str(uuid.uuid4())[:8],
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "origin": "machine",
                }
                local_results.append(result)

        with lock:
            task_results.extend(local_results)

    def convert_to_ls(self, x: int, y: int, original_width: int, original_height: int):
        """Convert pixel coordinates to Label Studio percentage coordinates."""
        return x / original_width * 100.0, y / original_height * 100.0

    def create_category_index(self):
        """Create a category index from a label map file."""
        return label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    def preprocess_image(self, image_url: str):
        """Load and preprocess an image from a URL or local path."""
        if image_url.startswith(('http://', 'https://')):
            headers = {'Authorization': f'Token {TOKEN}'}
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_url)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_project_name(self, project_id: str) -> Optional[str]:
        """Fetch project name from Label Studio using the project ID."""
        response = requests.get(f"{URL}/api/projects/{project_id}/", headers={"Authorization": f"Token {TOKEN}"})
        if response.status_code == 200:
            return response.json().get('title')
        print(f"Failed to fetch project details: {response.status_code}")
        return None

    def preprocess_image_with_output_dict(self, image_rgb: np.ndarray, model):
        """Prepare image for inference and run it through the model."""
        image_tensor = tf.convert_to_tensor(image_rgb)[tf.newaxis, ...]
        model_fn = model.signatures['serving_default']
        return model_fn(image_tensor)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """Run predictions for given Label Studio tasks."""
        print(f"Running prediction on {len(tasks)} tasks")
        category_index = self.create_category_index()
        project_name = self.get_project_name(self.project_id)
        print(f"Project Name: {project_name}")
        
        # Check for greyscale projects (custom logic example)
        if project_name and "greyscale" in project_name.lower():
            print("Processing as greyscale project")

        # Load model
        detection_model = load_model(MODEL_DIR)

        predictions = []
        for task in tasks:
            image_url = task['data']['image']
            image = self.preprocess_image(URL + image_url)
            output_dict = self.preprocess_image_with_output_dict(image, detection_model)

            num_detections = int(output_dict['num_detections'])
            detection_classes = output_dict['detection_classes'][0].numpy().astype(np.uint8)
            detection_boxes = output_dict['detection_boxes'][0].numpy()
            detection_scores = output_dict['detection_scores'][0].numpy()
            detection_masks = output_dict['detection_masks'][0].numpy()

            score_threshold = 0.4
            original_height, original_width, _ = image.shape

            # Multithreaded detection processing
            task_results = []
            chunk_size = max(num_detections // 4, 1)
            lock = threading.Lock()

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.process_detections, i, min(i + chunk_size, num_detections),
                        detection_scores, score_threshold, detection_boxes,
                        detection_masks, detection_classes, category_index,
                        image, original_width, original_height, task_results, lock
                    ) for i in range(0, num_detections, chunk_size)
                ]
                for future in futures:
                    future.result()

            predictions.append({"result": task_results})
        return predictions

    def fit(self, event: str, data: Dict, **kwargs):
        """Handle fit events to update model state."""
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f"Previous Data: {old_data}, Previous Version: {old_model_version}")

        self.set('my_data', 'updated_data')
        self.set('model_version', 'updated_version')
        print(f"Updated Data: {self.get('my_data')}, Updated Version: {self.get('model_version')}")
