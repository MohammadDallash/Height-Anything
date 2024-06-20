import cv2
import numpy as np
from collections import defaultdict, deque
from sklearn.linear_model import RANSACRegressor

class ObjectDetectionUtils:
    def __init__(self, detection_model):
        """
        Initialize the ObjectDetectionUtils with a given detection model.

        Parameters:
        detection_model: The model to be used for object detection.
        """
        self.detection_model = detection_model

    def calculate_head_and_leg_points(self, polygon_points, threshold=0.12):
        """
        Calculate the average points for the head and leg based on the top and bottom threshold percentage of Y-coordinates.

        Parameters:
        polygon_points (np.array): The polygon points.
        threshold (float): The percentage to consider for the top and bottom points (default is 0.12).

        Returns:
        tuple: The average points for the head and leg.
        """
        poly = np.array(polygon_points, dtype=np.int32)

        # Extract Y-coordinates
        y_coords = poly[:, 1]

        # Calculate top and bottom threshold percentage
        top_threshold_indices = np.argsort(y_coords)[:max(int(threshold * len(y_coords)), 1)]
        bottom_threshold_indices = np.argsort(y_coords)[-max(int(threshold * len(y_coords)), 1):]

        # Get average points for head and leg
        head_points = poly[top_threshold_indices]
        leg_points = poly[bottom_threshold_indices]

        head_avg = np.mean(head_points, axis=0).astype(int)
        leg_avg = np.mean(leg_points, axis=0).astype(int)

        leg_avg[1] = int ((leg_avg[1] +  np.max(y_coords))/2)
        head_avg[1] = int ((head_avg[1] + np.min(y_coords))/2)

        return head_avg, leg_avg

    def infer_obj_detection(self, frame):
        """
        Perform object detection on a given frame and calculate head and leg positions for each detected object.

        Parameters:
        frame (np.array): The input video frame.

        Returns:
        tuple: Results from the model inference, dictionary of head and leg positions, and list of bounding boxes.
        """
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference using the model
        results = self.detection_model(frame_rgb, classes=[0], conf=0.45, verbose=False)[0]
        masks = results.masks
        boxes = results.boxes

        # Dictionary to store head and leg positions
        legs_and_heads = {}

        # List to store bounding boxes
        boxes_list = []

        # Iterate over each detected box and mask
        for idx, box in enumerate(boxes):
            # Convert box coordinates to list
            xyxy = box.cpu().xyxy.tolist()[0]
            boxes_list.append(xyxy)

            # Calculate head and leg positions for the current mask
            head_pos, leg_pos = self.calculate_head_and_leg_points(masks[idx].xy[0])
            legs_and_heads[idx] = (head_pos, leg_pos)

        return results, legs_and_heads, boxes_list

    def calculate_IOU(self, bbox1, bbox2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        bbox1 (list): Coordinates of the first bounding box in the format [x1, y1, x2, y2].
        bbox2 (list): Coordinates of the second bounding box in the format [x1, y1, x2, y2].

        Returns:
        float: Intersection over Union (IoU) score between the two bounding boxes.
        """
        x1, y1, x2, y2 = bbox1
        X1, Y1, X2, Y2 = bbox2

        # Calculate intersection area
        interArea = max(0, min(x2, X2) - max(x1, X1)) * max(0, min(y2, Y2) - max(y1, Y1))

        # Calculate areas of bounding boxes
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (X2 - X1) * (Y2 - Y1)

        # Calculate IoU
        iou = interArea / (bbox1_area + bbox2_area - interArea)

        return iou

    def match_best_box(self, xyxy, boxes):
        """
        Match the best bounding box based on the highest IoU.

        Parameters:
        xyxy (list): The bounding box coordinates to match [x1, y1, x2, y2].
        boxes (list): List of bounding boxes to match against.

        Returns:
        int: The index of the best matching bounding box.
        """
        best_idx = 0
        best_iou = 0.0

        for idx, box in enumerate(boxes):
            iou = self.calculate_IOU(box, xyxy)

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        return best_idx