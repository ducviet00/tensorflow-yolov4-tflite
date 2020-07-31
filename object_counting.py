from sort import Sort
from utils import COLORS, intersect, get_output_fps_height_and_width
import cv2
import numpy as np
import time

DETECTION_FRAME_THICKNESS = 1

OBJECTS_ON_FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJECTS_ON_FRAME_COUNTER_FONT_SIZE = 0.5


LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
LINE_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
LINE_COUNTER_FONT_SIZE = 1.0
LINE_COUNTER_POSITION = (20, 45)
#total in frame
FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
FRAME_COUNTER_FONT_SIZE = 1.0
FRAME_COUNTER_POSITION = (20, 100)

class ObjectCounting:

    def __init__(self, options):
        self.options = options

    def _write_quantities(self, frame, labels_quantities_dic):
        for i, (label, quantity) in enumerate(labels_quantities_dic.items()):
            class_id = [i for i, x in enumerate(labels_quantities_dic.keys()) if x == label][0]
            color = [int(c) for c in COLORS[class_id % len(COLORS)]]

            cv2.putText(
                frame,
                f"{label}: {quantity}",
                (10, (i + 1) * 35),
                OBJECTS_ON_FRAME_COUNTER_FONT,
                OBJECTS_ON_FRAME_COUNTER_FONT_SIZE,
                color,
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )
    
    def _draw_detection_results(self, frame, resutls, labels_quantities_dic):
        for start_point, end_point, label, confidence, in resutls:
            x1, y1 = start_point

            class_id = []
