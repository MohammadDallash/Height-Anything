import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox


class CameraCalibrateApp:
    def __init__(self, video_path):
        self.video_path = video_path
        self.lines = []
        self.drawing = False
        self.line_start = None
        self.line_end = None
        self.frame = None
        self.original_frame = None

    def init_ui(self):
        if not self.show_instructions():
            return None

        self.reset_drawing_window()
        cv2.namedWindow('Draw Line')
        cv2.setMouseCallback('Draw Line', self.draw_line)

        while True:
            display_frame = self.frame.copy()
            for start, end, _ in self.lines:
                cv2.line(display_frame, start, end, (0, 0, 255), 2)

            if self.drawing and self.line_start and self.line_end:
                cv2.line(display_frame, self.line_start, self.line_end, (0, 0, 255), 2)

            cv2.imshow('Draw Line', display_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == 13:  # Proceed on 'Enter' key
                if not self.ask_to_add_line():
                    break
            elif key & 0xFF == 27:  # Reset on 'Esc' key
                if self.lines == []:
                    break
                self.reset_drawing_window()
                cv2.imshow('Draw Line', self.frame)

        cv2.destroyAllWindows()

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.line_start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.line_end = (x, y)
                img_copy = self.frame.copy()
                for start, end, _ in self.lines:
                    cv2.line(img_copy, start, end, (0, 0, 255), 2)
                cv2.line(img_copy, self.line_start, self.line_end, (0, 0, 255), 2)
                cv2.imshow('Draw Line', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.line_end = (x, y)
            self.drawing = False
            cv2.line(self.frame, self.line_start, self.line_end, (0, 0, 255), 2)
            height = self.get_line_height()
            midpoint = ((self.line_start[0] + self.line_end[0]) // 2, (self.line_start[1] + self.line_end[1]) // 2)
            cv2.putText(self.frame, f"{height:.2f}cm", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            self.lines.append((self.line_start, self.line_end, height))
            self.line_start = None
            self.line_end = None
            cv2.imshow('Draw Line', self.frame)
    def get_line_height(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        height = simpledialog.askfloat("Input", "Enter the real height of the line in cm:")
        root.destroy()
        return height
    def show_instructions(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        response = messagebox.askokcancel("Instructions",
                                        "Instructions:\n\n"
                                        "1. Draw vertical lines on the object of known height in the video by dragging the mouse.\n"
                                        "2. Press 'Enter' after each line to add it.\n"
                                        "3. You can press 'Esc' to reset and start over.\n"
                                        "4. Make sure to choose ONLY vertical objects for accurate calibration.\n"
                                        "5. Click 'OK' to start drawing lines or 'Cancel' to exit.")
        root.destroy()
        return response


    def ask_to_add_line(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        response = messagebox.askyesno("Add Line", "Do you want to add another line?")
        root.destroy()
        return response

    def reset_drawing_window(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Couldn't open video.")
            exit()

        ret, self.original_frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Couldn't read the first frame.")
            exit()

        self.frame = self.original_frame.copy()
        self.line_start, self.line_end = None, None
        self.lines = []

    def start(self):
        self.reset_drawing_window()

        will_break = self.init_ui()
        
        if will_break == True:
            return None
        

        ground_truth_lines = []
        for start, end, real_height in self.lines:
            ground_truth_lines.append((start, end, real_height))
        return ground_truth_lines, self.original_frame
