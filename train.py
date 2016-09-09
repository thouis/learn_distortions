from distorter import gen_distorted
import numpy as np

from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K

def residual_block(input, num_feature_maps, filter_size=3):
    conv_1 = BatchNormalization(axis=1, mode=2)(input)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = Convolution2D(num_feature_maps, filter_size, filter_size,
                           border_mode='same', bias=False)(conv_1)

    conv_2 = BatchNormalization(axis=1, mode=2)(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = Convolution2D(num_feature_maps, filter_size, filter_size,
                           border_mode='same', bias=False)(conv_2)

    return merge([input, conv_2], mode='sum')


def residual_chain(input, num_blocks, num_features_maps, filter_size=3):
    output = input
    for idx in range(num_blocks):
        output = residual_block(output, num_features_maps, filter_size)
    return output

def mean_vec_diff(y_true, y_pred):
    # th ordering is batch, channel, height, width
    diff = y_true - y_pred
    mag = K.sqrt((diff ** 2).sum(axis=1))  # drops channel axis
    mean_mag = mag.mean(axis=[1, 2])
    return K.mean(mean_mag)

if __name__ == '__main__':
    num_feature_maps = 64
    INPUT_SHAPE = (2, 96, 96)

    x = Input(shape=INPUT_SHAPE)
    pre = Convolution2D(num_feature_maps, 5, 5, bias=True, border_mode='same')(x)
    post = residual_chain(pre, 10, num_feature_maps)
    # 3 outputs per voxel, 3x3 final filter
    output = Convolution2D(2, 3, 3, activation=None, border_mode='same')(post)
    model = Model(input=x, output=output)

    model.compile(optimizer=SGD(lr=0.001, clipnorm=1., momentum=0.9),
                  loss=mean_vec_diff)

    gen = gen_distorted('images', INPUT_SHAPE, batch_size=32)
    a = next(gen)  # debugging
    model.fit_generator(gen,
                        4096, 10000, verbose=1)
