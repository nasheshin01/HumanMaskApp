import cv2

class CameraVideoInputStreamer:
    
    def __init__(self) -> None:
        self.streamer = cv2.VideoCapture(0)

    def read_frame(self):
        is_read, frame = self.streamer.read()

        if is_read:
            return frame
        else:
            return None

class FileVideoInputStreamer:

    def __init__(self, file_path: str) -> None:
        self.streamer = cv2.VideoCapture(file_path)

    def read_frame(self):
        is_read, frame = self.streamer.read()

        if is_read:
            return frame
        else:
            return None

class ImageInputStreamer:

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def read_frame(self):
        try:
            return cv2.imread(self.file_path)
        except:
            return None