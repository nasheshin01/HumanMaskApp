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