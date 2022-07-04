import cv2

import numpy as np

class BlurTransformer:

    def __init__(self, ksize=(15, 15)) -> None:
        self.ksize = ksize

    def transform(self, image: np.array, mask: np.array):
        blur = cv2.blur(image, self.ksize, 0)
        out = image.copy()
        out[mask == 0] = blur[mask == 0]

        return out