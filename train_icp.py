from distorter import gen_icp
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

def warped_icp_loss_function(image_size):
    i_indices = K.variable(np.arange(image_size[0]).reshape((-1, 1)), dtype=np.int32, name='i_indices')
    j_indices = K.variable(np.arange(image_size[1]).reshape((1, -1)), dtype=np.int32, name='j_indices')

    def warped_icp(y_true, y_pred):
        ''' Loss function for warped Iterated Closest Point.

        y_pred is size (batch, 2, H, W) - i,j offsets for warp
        y_true is size (batch, 3, H, W) -
           channel 0,1 - i,j offsets to nearest boundary in second input image
           channel 2 - boundary pixels in first image
        '''

        # find warped positions
        i_warp = K.cast(K.round(K.clip(y_pred[:, 0, :, :] + i_indices, 0, image_size[0] - 1)), 'int32')
        j_warp = K.cast(K.round(K.clip(y_pred[:, 1, :, :] + j_indices, 0, image_size[1] - 1)), 'int32')

        # Find the vector to the closest boundary pixel in the warped position in the second image.
        i_warp_delta = y_true[:, 0, i_warp, j_warp]
        j_warp_delta = y_true[:, 1, i_warp, j_warp]

        # The desired warp is our existing warp plus the deltas.
        desired_i_warp = i_warp_delta + y_pred[:, 0, :, :]
        desired_j_warp = j_warp_delta + y_pred[:, 1, :, :]

        # We disconnect gradients here because we want this to be the target value.
        desired_i_warp = K.stop_gradient(desired_i_warp)
        desired_j_warp = K.stop_gradient(desired_j_warp)

        # compute magnitude of vector error (predicted vs. desired)
        error = K.sqrt((y_pred[:, 0, :, :] - desired_i_warp) ** 2 +
                       (y_pred[:, 1, :, :] - desired_j_warp) ** 2)

        # mask to pixels we care about
        mask = y_true[:, 2, :, :]
        zero_pixel_count = mask.sum(axis=[1, 2])
        per_batch_error = (error * mask).sum(axis=[1, 2]) / zero_pixel_count

        return K.mean(per_batch_error)

    return warped_icp

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
                  loss=warped_icp_loss_function(INPUT_SHAPE[1:]))

    gen = gen_icp('images', 'labels', INPUT_SHAPE[1:], batch_size=32)
    a = next(gen)  # debugging
    model.fit_generator(gen,
                        4096, 10000, verbose=1)
