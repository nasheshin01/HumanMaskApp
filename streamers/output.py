import cv2
import os

import numpy as np

class WindowOutputStreamer:

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        
    def update_window(self, frame: np.array) -> None:
        cv2.imshow(self.window_name, frame)

    def check_output_end(self) -> bool:
        return cv2.waitKey(1) & 0xFF == ord('q')

class FolderOutputStreamer:

    def __init__(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        self.folder_path = folder_path
        self.frame_index = 0

    def get_frame_name(self):
        frame_index_string = str(self.frame_index)
        digit_index_count = len(frame_index_string)
        zero_string = ''
        for _ in range(8 - digit_index_count):
            zero_string += '0'

        return f'{zero_string}{frame_index_string}.png'

    def write_frame(self, frame: np.array) -> None:
        cv2.imwrite(os.path.join(self.folder_path, self.get_frame_name()), frame)
        self.frame_index += 1

class ImageOutputStreamer:

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def write_frame(self, frame: np.array) -> None:
        cv2.imwrite(self.output_path, frame)
        

        