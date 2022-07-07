# HumanMaskApp

## Description:
HumanMaskAppp is an application which can divide a human and a background on image and video from file or camera.
And you can choose what should replace background, blurred background or image.

## How to use:
To use this app, you should launch main.py file with python and with required arguments.

###### Required arguments:

1. input_type: camera, video or image. With help of this argument you can choose where to get frames for processing: from camera, from video or from image.
2. output_type: window, folder or image. With help of this argument you can choose where save transformed frames: to show in window, to save in folder, to save by image.
3. model_weights: with this argument you set path to model weights, for now there is the only one file with weigths - models/aspp-v1.h5.
4. background_type: blur or image. With help of this argument you can choose what to do with background: to blur it or to replace with image.

###### Optional arguments:

1. input_path: if you chose get frames from video or image, you should set path to this video or image.
2. output_path: if you chose save frames in folder or by image, you should set path where to save frames: folder or file path (for image).
3. background_path: if you chose to replace background with image, you should set the path to this image.
