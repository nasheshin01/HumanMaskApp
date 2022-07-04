import cv2

import numpy as np

class BlurBackgroundTransformer:

    def __init__(self, ksize=(15, 15)) -> None:
        self.ksize = ksize

    def transform(self, image: np.array, mask: np.array):
        blur = cv2.blur(image, self.ksize, 0)
        out = image.copy()
        out[mask == 0] = blur[mask == 0]

        return out

class ImageBackgroundTransformer:

    def __init__(self, background_path) -> None:
        self.background = cv2.imread(background_path)

    def transform(self, image: np.array, mask: np.array) -> np.array:
        background_height, background_width, _ = self.background.shape
        image_height, image_width, _ = image.shape

        scale = background_width / image_width
        if image_height * scale > background_height:
            scale = background_height / image_height

        rescaled_size = (round(image_width * scale), round(image_height * scale))
        image_resized = cv2.resize(image, rescaled_size)
        mask_reshaped = (mask * 255).reshape((mask.shape[0], mask.shape[1], 1)).astype('float32')
        mask_resized = cv2.resize(mask_reshaped, rescaled_size) / 255
        background_copy = self.background.copy()

        if (rescaled_size[0] == background_width):
            background_copy[background_height-rescaled_size[1]:background_height, :, :][mask_resized == 1] = image_resized[mask_resized == 1]
        else:
            x = round(background_width / 2 - rescaled_size[0] / 2)
            x_end = round(x + rescaled_size[0])
            background_copy[:, x:x_end, :][mask_resized == 1] = image_resized[mask_resized == 1]

        return background_copy