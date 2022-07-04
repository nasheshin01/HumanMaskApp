import cv2

import numpy as np

class WindowOutputStreamer:

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        
    def update_window(self, frame: np.array) -> None:
        cv2.imshow(self.window_name, frame)

    def check_output_end(self) -> bool:
        return cv2.waitKey(1) & 0xFF == ord('q')