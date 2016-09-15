from distorter import gen_icp
import numpy as np

from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import SGD
from keras.engine.training import collect_trainable_weights
import keras.backend as K
import theano.tensor as T
from theano.printing import Print

def Print(m):
    return lambda x: x

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

def warped_icp_loss_function(image_size, batch_size):
    i_indices = K.variable(np.arange(image_size[0])).dimshuffle(['x', 0, 'x']).copy(name='i_indices')
    j_indices = K.variable(np.arange(image_size[1])).dimshuffle(['x', 'x', 0]).copy(name='j_indices')

    def warped_icp(y_true, y_pred):
        ''' Loss function for warped Iterated Closest Point.

        y_pred is size (batch, 2, H, W) - i,j offsets for warp
        y_true is size (batch, 3, H, W) -
           channel 0,1 - i,j offsets to nearest boundary in second input image
           channel 2 - boundary pixels in first image
        '''

        # find warped positions
        i_clipped = K.clip(y_pred[:, 0, :, :] + i_indices, 0, float(image_size[0]) - 1).copy('i_warp_clip')
        j_clipped = K.clip(y_pred[:, 1, :, :] + j_indices, 0, float(image_size[1]) - 1).copy('j_warp_clip')
        i_warp_int = K.round(i_clipped).astype('int32').copy('i_warp')
        j_warp_int = K.round(j_clipped).astype('int32').copy('j_warp')

        # Find the vector to the closest boundary pixel in the warped position in the second image.
        # Advanced indexing makes this somewhat tricky.
        batch_idx = np.arange(32).reshape((32, 1, 1))
        i_warp_delta = y_true[:, 0, ...][batch_idx, i_warp_int, j_warp_int].copy('i_warp_delta')
        j_warp_delta = y_true[:, 1, ...][batch_idx, i_warp_int, j_warp_int].copy('j_warp_delta')

        # The desired warp is our existing warp plus the deltas.
        desired_i_warp = (i_warp_delta + i_warp_int).copy('desired_i_warp')
        desired_j_warp = (j_warp_delta + j_warp_int).copy('desired_j_warp')

        # compute the correction we should apply to the predictions
        i_warp_correction = desired_i_warp.astype('float32') - (y_pred[:, 0, :, :] + i_indices)
        j_warp_correction = desired_j_warp.astype('float32') - (y_pred[:, 1, :, :] + j_indices)
        warp_correction_lengths = K.sqrt(i_warp_correction ** 2 + j_warp_correction ** 2 + 0.01)

        # compute magnitude of vector error (predicted vs. desired)
        error = K.sqrt(i_warp_correction ** 2 + j_warp_correction ** 2).copy('error')

        # mask to pixels we care about
        mask = y_true[:, 2, :, :].copy('mask')
        zero_pixel_count = mask.sum(axis=[1, 2]).copy('zerop') + 1
        per_batch_error = (error * mask).sum(axis=[1, 2]).copy('sumb') / zero_pixel_count

        normalized_corrections = T.stack([i_warp_correction, j_warp_correction], axis=1) / warp_correction_lengths.dimshuffle([0, 'x', 1, 2])
        neg_gradient = (normalized_corrections * mask.dimshuffle([0, 'x', 1, 2])) / zero_pixel_count.dimshuffle([0, 'x', 'x', 'x'])

        return (K.mean(per_batch_error),
                - neg_gradient / batch_size)

    return warped_icp

if __name__ == '__main__':
    num_feature_maps = 64
    INPUT_SHAPE = (2, 96, 96)
    batch_size = 32

    x = Input(shape=INPUT_SHAPE)
    pre = Convolution2D(num_feature_maps, 5, 5, bias=True, border_mode='same')(x)
    post = residual_chain(pre, 10, num_feature_maps)
    # 3 outputs per voxel, 3x3 final filter
    output = Convolution2D(2, 3, 3, activation=None, border_mode='same')(post)
    model = Model(input=x, output=output)

    parameters = collect_trainable_weights(model)
    ground_truth = K.placeholder(shape=(batch_size, 3) + INPUT_SHAPE[1:])
    error, d_err_d_pred = warped_icp_loss_function(INPUT_SHAPE[1:], batch_size)(ground_truth, model.output)

    # gradient descent
    grads = T.grad(None,  # we are passing a known gradient
                   wrt=parameters,
                   known_grads={model.output: d_err_d_pred})
    shapes = [K.get_variable_shape(p) for p in parameters]
    moments = [K.zeros(shape) for shape in shapes]
    updates = []

    min_lr = 0.001
    max_lr = 0.005
    momentum = 0.9

    def layer_depth(p):
        return int(p.name.split('_')[-2])
    max_depth = max([layer_depth(p) for p in parameters])
    min_depth = min([layer_depth(p) for p in parameters])

    def local_lr(d):
        return min_lr + (max_lr - min_lr) * (d - min_depth) / float(max_depth - min_depth)

    for p, g, m in zip(parameters, grads, moments):
        v = momentum * m - local_lr(layer_depth(p)) * K.clip(g, -0.1, 0.1)
        updates += [(m, v), (p, p + v)]
    stepfun = K.function([x, ground_truth],
                         [error] + grads,
                         updates=updates)

    errfun = K.function([x, ground_truth],
                        [error])

    model.load_weights('pretrained_on_warps.h5')

    gen = gen_icp('images', 'labels', INPUT_SHAPE[1:], batch_size=batch_size)
    a = next(gen)  # debugging
    while True:
        inp, gt = next(gen)
        z = gt[:, 2, ...]
        delta = gt[:, :2, ...]
        d2 = np.sqrt((delta**2).sum(axis=1))
        d2m = z * d2
        err, *gr = stepfun((inp, gt))

        rand_step = np.random.uniform(-1, 1, gr[-1].shape)
        sz = 0.00001
        expected_change = (gr[-1] * (sz * rand_step)).sum()
        parameters[-1].set_value((parameters[-1].get_value() + sz * rand_step).astype(np.float32))
        err2 = errfun((inp, gt))
        print("expected", expected_change, err - err2)

        pre_errs = np.sum(d2m, axis=(1, 2)) / (np.sum(z, axis=(1, 2)) + 1)
        print("pre", np.mean(pre_errs), "post", float(err[0]))
        print("")
