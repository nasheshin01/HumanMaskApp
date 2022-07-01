import keras
import tensorflow as tf

from matplotlib import pyplot as plt

class SegmentationModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')        
        self.conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')        
        self.conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv6 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')    
        
        self.conv7 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv8 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')        
        self.conv9 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv10 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv11 = keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')
        
        self.pool = keras.layers.MaxPool2D((2, 2))
        self.unpool = keras.layers.UpSampling2D((2, 2))
                
    def call(self, x):
        
        # Encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)
        out = self.conv5(out)
        out = self.conv6(out)
        
        # Decoder        
        out = self.unpool(out)        
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.unpool(out)        
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)

        return out

class UNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.encoderConv11 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.encoderConv12 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.encoderConv21 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.encoderConv22 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.encoderConv31 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.encoderConv32 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        
        self.bottleNeckConv1 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.bottleNeckConv2 = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')

        self.decoderConv11 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.decoderConv12 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.decoderConv21 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.decoderConv22 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.decoderConv31 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.decoderConv32 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        self.decoderConvResult = keras.layers.Conv2D(1, (3, 3), padding='same', activation='relu')

        self.decoderUpconv1 = keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.decoderUpconv2 = keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.decoderUpconv3 = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')
        
        self.maxPool = keras.layers.MaxPool2D((2, 2))
                
    def call(self, x):

        # Encoder
        encoderConv11Out = self.encoderConv11(x)
        encoderConv12Out = self.encoderConv12(encoderConv11Out)
        maxPool1Out =  self.maxPool(encoderConv12Out)

        encoderConv21Out = self.encoderConv21(maxPool1Out)
        encoderConv22Out = self.encoderConv22(encoderConv21Out)
        maxPool2Out =  self.maxPool(encoderConv22Out)

        encoderConv31Out = self.encoderConv31(maxPool2Out)
        encoderConv32Out = self.encoderConv32(encoderConv31Out)
        maxPool3Out =  self.maxPool(encoderConv32Out)

        bottleNeck1Out = self.bottleNeckConv1(maxPool3Out)
        bottleNeck2Out = self.bottleNeckConv2(bottleNeck1Out)

        upConv1Out = self.decoderUpconv1(bottleNeck2Out)
        tensorConcat1 = tf.concat([encoderConv32Out, upConv1Out], axis=3)
        decoderConv11Out = self.decoderConv11(tensorConcat1)
        decoderConv12Out = self.decoderConv12(decoderConv11Out)

        upConv2Out = self.decoderUpconv2(decoderConv12Out)
        tensorConcat2 = tf.concat([encoderConv22Out, upConv2Out], axis=3)
        decoderConv21Out = self.decoderConv21(tensorConcat2)
        decoderConv22Out = self.decoderConv22(decoderConv21Out)

        upConv3Out = self.decoderUpconv3(decoderConv22Out)
        tensorConcat3 = tf.concat([encoderConv12Out, upConv3Out], axis=3)
        decoderConv31Out = self.decoderConv31(tensorConcat3)
        decoderConv32Out = self.decoderConv32(decoderConv31Out)

        decoderConvResult = self.decoderConvResult(decoderConv32Out)

        return decoderConvResult

model = UNet()
model.build((0, 256, 256, 3))
tf.keras.utils.plot_model(model, show_shapes=True)
plt.show()
