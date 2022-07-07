import keras
import os

import tensorflow as tf
import pandas as pd


augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    keras.layers.RandomRotation(0.3),
    keras.layers.Resizing(160, 160)
])


def join_paths(origin_path, filenames):
    return [os.path.join(origin_path, filename) for filename in filenames]


def get_paths_dataframe(data_dirs, train_size):
    result_paths = []

    for data_dir in data_dirs:
        images_dir = os.path.join(data_dir, "images")
        masks_dir = os.path.join(data_dir, "masks")

        images_names = os.listdir(images_dir)

        images_paths = join_paths(images_dir, images_names)
        masks_paths = join_paths(masks_dir, images_names)

        for image_path, mask_path in zip(images_paths, masks_paths):
            result_paths.append([image_path, mask_path])

    df = pd.DataFrame(result_paths)
    df.columns = ['image_path', 'mask_path']

    df_train = df.sample(frac=train_size, random_state=42)
    df_val = pd.concat([df_train, df]).drop_duplicates(keep=False)

    return df_train, df_val


def read_image_and_mask_train(image_path, mask_path):
    input_size = (160, 160)

    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_png(image, channels=3)
    mask = tf.image.decode_png(mask, channels=1)

    image = tf.image.resize(image, input_size) / 255
    mask = tf.image.resize(mask, input_size) / 255

    image_and_mask = tf.concat([image, mask], axis=2)
    image_and_mask_cropped = augmentation(image_and_mask)

    image = image_and_mask_cropped[..., :3]
    mask = image_and_mask_cropped[..., 3:4]

    return image, mask

def read_image_and_mask_val(image_path, mask_path):
    input_size = (160, 160)

    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_png(image, channels=3)
    mask = tf.image.decode_png(mask, channels=1)

    image = tf.image.resize(image, input_size) / 255
    mask = tf.image.resize(mask, input_size) / 255

    return image, mask


def get_dataset(df, is_training):
    images_paths = df['image_path'].values
    masks_paths = df['mask_path'].values

    ds = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))

    if is_training:
        ds = ds.map(read_image_and_mask_train).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(read_image_and_mask_val).prefetch(tf.data.AUTOTUNE)

    return ds.batch(1)



