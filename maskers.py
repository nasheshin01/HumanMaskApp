import cv2

import numpy as np

from training.seg_models import ASPPNet

class HumanMaskFinder:

    def __init__(self, weights_path: str, image_size=(160, 160), threshold=0.5) -> None:
        self.model = ASPPNet()
        self.model.build((1, *image_size, 3))
        self.model.load_weights(weights_path)
        self.image_size = image_size
        self.threshold = threshold

    def get_mask(self, frame: np.array) -> np.array:
        image = cv2.resize(frame, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = image / 255.0

        mask = self.model.predict(np.array([image_norm]))[0]
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        return mask > self.threshold