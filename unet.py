import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU, concatenate

    
class UNet(tf.keras.Model):
    """
    UNet semantic segmentation architecture.

    Arguments:
        n_filters: Initial number of filters. Following ones will be multiplied by two ([32, 64, 128, ...]).
        n_classes: Number of output classes. 
    Returns:
        Tensorflow keras model object. 
    """

    def __init__(self, n_filters=32, n_classes=1):
        super(UNet, self).__init__()
        self.conv = DoubleConvolution(n_filters)
        self.dblock1 = DownConvolve(n_filters * 2)
        self.dblock2 = DownConvolve(n_filters * 4)
        self.dblock3 = DownConvolve(n_filters * 8)
        self.dblock4 = DownConvolve(n_filters * 16)

        self.ublock4 = UpConvolve(n_filters * 8)
        self.ublock3 = UpConvolve(n_filters * 4)
        self.ublock2 = UpConvolve(n_filters * 2)
        self.ublock1 = UpConvolve(n_filters)

        self.outputs = Conv2D(
            n_classes, 
            kernel_size=(1, 1), 
            padding='same', 
            activation='softmax' if n_classes > 1 else 'sigmoid'
        )

    def call(self, inputs):
        s1 = self.conv(inputs)
        s2 = self.dblock1(s1)
        s3 = self.dblock2(s2)
        s4 = self.dblock3(s3)
        s5 = self.dblock4(s4)
        x = self.ublock4(s5, s4)
        x = self.ublock3(x, s3)
        x = self.ublock2(x, s2)
        x = self.ublock1(x, s1)
        
        return self.outputs(x)

class DoubleConvolution(tf.keras.layers.Layer):
    """
    Sequential two CNN layer operation.

    Arguments:
        n_filters: Number of output filters.
        dropout: Dropout ratio. Default is 0 meaning that don't add dropout.
    Returns:
        Tensorflow keras sequential object with two CNN layer.
    """
    def __init__(self, n_filters, dropout=0):
        super(DoubleConvolution, self).__init__()
        self.conv_block = Sequential(
            [
                Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'),
                BatchNormalization(),
                ReLU(),
                Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'),
                BatchNormalization(),
                ReLU()
            ]
        )

        if dropout > 0:
            self.conv_block.add(Dropout(dropout))

    def __call__(self, inputs):
        return self.conv_block(inputs)
    
class DownConvolve(tf.keras.layers.Layer):
    """
    Applying max pooling following double CNN operation to reduce feature map's size.
    
    Arguments:
        n_filters: Number of output filters.
        dropout: Dropout ratio. Default is 0 meaning that don't add dropout.
    Returns:
        Tensorflow keras sequential object with one max pooling and two CNN operation.
    """

    def __init__(self, n_filters, dropout=0):
        super(DownConvolve, self).__init__()
        self.dblock = Sequential(
            [   MaxPooling2D(pool_size=(2, 2)),
                DoubleConvolution(n_filters, dropout)
            ]
        )

    def call(self, inputs):
        return self.dblock(inputs)

class UpConvolve(tf.keras.layers.Layer):
    """
    Upscaling feature map with transpose convolution operation and concatenating skip connections with the output.

    Arguments:
        n_filters: Number of output filters.
        dropout: Dropout ratio. Default is 0 meaning that don't add dropout.
    Returns:
        Tensorflow keras sequential object with one transpose convolution and two convolution operation on combination of the output and skip connections.
    """
    
    def __init__(self, n_filters, dropout=0):
        super(UpConvolve, self).__init__()
        self.up = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.conv = DoubleConvolution(n_filters, dropout)

    def call(self, inputs, skip_connections):
        u = self.up(inputs)
        merge = concatenate([u, skip_connections], axis=-1)
        return self.conv(merge)

