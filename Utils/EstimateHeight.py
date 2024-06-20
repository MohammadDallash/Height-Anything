import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

class EstimateHeight:
    @staticmethod
    def estimate_height_currnetFrame(c_length_pixels, c_depth, anchors, infer_cam_calib_fuction, cam_clib_model):
        """
        Estimate the height of an object in the current frame based on anchors and camera calibration.

        Parameters:
        - c_length_pixels: Length in pixels of the object in the current frame.
        - c_depth: Depth of the object in the current frame.
        - anchors: List of anchor tuples (a_length_pixels, a_true_length, a_depth).
        - cam_clib_model: Camera calibration model.

        Returns:
        - av_c_tall: Average estimated height of the object.
        """
        n = len(anchors)
        av_c_tall = 0.0

        for a_length_pixels, a_true_length, a_depth in anchors:
            c_tranformed_length_pixels = infer_cam_calib_fuction(cam_clib_model, c_depth, a_depth - c_depth) * c_length_pixels
            c_tall = a_true_length * (c_tranformed_length_pixels / a_length_pixels)
            av_c_tall += c_tall / n

        return round (av_c_tall, 2)

    @staticmethod
    def annotate_frame(frame, head_pos, leg_pos, height):
        """
        Annotate the frame with markers for head and leg positions and display the height.

        Parameters:
        - frame: Input frame.
        - head_pos: Position of the head (x, y).
        - leg_pos: Position of the leg (x, y).
        - height: Height to be displayed.

        Returns:
        - Annotated frame.
        """
        cv2.circle(frame, head_pos, 5, (0, 255, 0), -1)  # Green circle for head
        cv2.circle(frame, leg_pos, 5, (0, 0, 255), -1)   # Red circle for leg
        cv2.line(frame, head_pos, leg_pos, (255, 0, 0), 2) 

        midpoint = ((head_pos[0] + leg_pos[0]) // 2, (head_pos[1] + leg_pos[1]) // 2)
        cv2.putText(frame, str(round(height)), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    @staticmethod
    def get_stable_height(object_heights, obj_id, height):
        """
        Estimate and smooth the height of an object using RANSAC and a moving average.
        
        Parameters:
        - object_heights: defaultdict storing height measurements for each object
        - obj_id: ID of the object
        - height: Current height measurement

        Returns:
        - stable_height: The estimated and smoothed height
        """
        object_heights[obj_id].append(height)

        if len(object_heights[obj_id]) >= 5:  # Minimum samples required for RANSAC
            X = np.arange(len(object_heights[obj_id])).reshape(-1, 1)
            y = np.array(object_heights[obj_id])
            ransac = RANSACRegressor(min_samples=2)  # Use more samples for RANSAC
            try:
                ransac.fit(X, y)
                height_ransac = ransac.predict(np.array([[len(object_heights[obj_id])-1]]))[0]
            except ValueError as e:
                print(f"RANSAC error for object {obj_id}: {e}")
                height_ransac = height  # Fall back to the simple height estimate
        else:
            height_ransac = height

        smoothed_height = np.mean(list(object_heights[obj_id])[-5:])  # Moving average over last 5 measurements
        stable_height = 0.7 * height_ransac + 0.3 * smoothed_height  # Weighted average for more stability

        return stable_height