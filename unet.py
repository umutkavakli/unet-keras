
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU, concatenate


class UNet():
    def __init__(self, input_size=(224, 224, 3), n_filters=32, n_classes=10):
        self.input_size = input_size
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.model = self.build()

    def encoder_block(self, inputs, n_filters, dropout=0, max_pooling=True):
        c = Conv2D(
            filters=n_filters, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal')(inputs)
        c = Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal')(c)
        c = BatchNormalization()(c)
        c = ReLU()(c)

        if dropout > 0:
            c = Dropout(dropout)(c)
        
        if max_pooling:
            output = MaxPooling2D(pool_size=(2, 2))(c)
        else:
            output = c
        skip_connection = c

        return output, skip_connection

    def decoder_block(self, inputs, skip_connections, n_filters):
        u = Conv2DTranspose(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same')(inputs)
        merge = concatenate([u, skip_connections], axis=-1)
        c = Conv2D(
            filters=n_filters, 
            kernel_size=(3, 3), 
            padding='same', 
            kernel_initializer='he_normal')(merge)
        c = Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c)
        c = BatchNormalization()(c)
        c = ReLU()(c)

        return c

    def build(self):
        inputs = Input(self.input_size)

        cblock1, skip1 = self.encoder_block(inputs, self.n_filters)
        cblock2, skip2 = self.encoder_block(cblock1, self.n_filters * 2)
        cblock3, skip3 = self.encoder_block(cblock2, self.n_filters * 4)
        cblock4, skip4 = self.encoder_block(cblock3, self.n_filters * 8, dropout=0.3)
        
        bottleneck, _ = self.encoder_block(cblock4, self.n_filters * 16, dropout=0.3, max_pooling=False)

        ublock4 = self.decoder_block(bottleneck, skip4, self.n_filters * 8)
        ublock3 = self.decoder_block(ublock4, skip3, self.n_filters * 4)
        ublock2 = self.decoder_block(ublock3, skip2, self.n_filters * 2)
        ublock1 = self.decoder_block(ublock2, skip1, self.n_filters)

        conv = Conv2D(
            filters=self.n_filters, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal')(ublock1)
        
        outputs = Conv2D(self.n_classes, kernel_size=(1, 1), padding='same')(conv)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
    