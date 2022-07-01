import os

import tensorflow as tf
import pandas as pd


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

    df_train = df.sample(frac=0.95, random_state=42)
    df_val = pd.concat([df_train, df]).drop_duplicates(keep=False)

    return df_train, df_val


def read_image_and_mask(image_path, mask_path):
    input_size = (160, 160)

    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_png(image, channels=3)
    mask = tf.image.decode_png(mask, channels=1)

    image = tf.image.resize(image, input_size) / 255
    mask = tf.image.resize(mask, input_size) / 255

    return image, mask


def get_dataset(df):
    images_paths = df['image_path'].values
    masks_paths = df['mask_path'].values

    ds = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))
    ds = ds.map(read_image_and_mask)

    return ds.batch(1)



