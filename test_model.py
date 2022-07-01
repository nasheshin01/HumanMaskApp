import cv2

import numpy as np

from seg_models import SegmentationModel, UNet


def main():
    frame = cv2.imread("test.jpg")
    image = cv2.resize(frame, (96, 96))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image / 255.0

    model = UNet()
    model.build(input_shape=(1, 96, 96, 3))
    model.load_weights("model.h5")

    mask = model.predict(np.array([image_norm]))[0]
    mask = cv2.resize(mask * 256, (frame.shape[1], frame.shape[0]))
    mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("test_mask.jpg", mask_image)

main()