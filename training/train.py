import imp
import keras
import cv2

import numpy as np
import matplotlib.pyplot as plt

from dataset import get_paths_dataframe, get_dataset
from seg_models import ASPPBlock, ASPPNet, UNet


DATA_DIRS = [r"C:\Projects\Learning\HumanMaskApp\data\kaggle",
             r"C:\Projects\Learning\HumanMaskApp\data\kaggle2",
             r"C:\Projects\Learning\HumanMaskApp\data\kaggle3"]


def main():
    df_train, df_val = get_paths_dataframe(DATA_DIRS, train_size=0.95)
    ds_train, ds_val = get_dataset(df_train, True), get_dataset(df_val, False)

    # for elem in ds_train:
    #     img = elem[0].numpy()
    #     img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)

    #     mask = elem[1].numpy()
    #     mask = cv2.cvtColor(mask[0], cv2.COLOR_GRAY2RGB)
    #     diff_image = np.hstack([img, mask])

    #     cv2.imshow("debug", diff_image)
    #     cv2.waitKey(0)


    model = ASPPNet()
    model.build((0, 160, 160, 3))

    adam_optimizer = keras.optimizers.Adam(learning_rate=0.00008)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="model-best.h5",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam_optimizer)
    history = model.fit(ds_train, epochs=10, validation_data=ds_val, callbacks=[model_checkpoint_callback])
    model.save_weights("model.h5")

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
