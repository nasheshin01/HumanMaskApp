import cv2

import numpy as np

from seg_models import UNet
  
model = UNet()
model.build(input_shape=(1, 128, 128, 3))
model.load_weights("model.h5")


# define a video capture object
vid = cv2.VideoCapture(0)


while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    image = cv2.resize(frame, (160, 160))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image / 255.0

    mask = model.predict(np.array([image_norm]))[0]
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    blur = cv2.blur(frame,(15,15),0)
    out = frame.copy()
    out[mask<0.5] = blur[mask<0.5]

    mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  
    # Display the resulting frame
    cv2.imshow('frame', mask_image)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()