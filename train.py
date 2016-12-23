from distorter import gen_distorted
import imread
import sys
import numpy as np
import cv2
import pylab

from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K


def residual_block(input, num_feature_maps, filter_size=3):
    conv_1 = BatchNormalization(axis=1, mode=2)(input)
    conv_1 = ELU()(conv_1)
    conv_1 = Convolution2D(num_feature_maps, filter_size, filter_size,
                           border_mode='same', bias=True)(conv_1)

    conv_2 = BatchNormalization(axis=1, mode=2)(conv_1)
    conv_2 = ELU()(conv_2)
    conv_2 = Convolution2D(num_feature_maps, filter_size, filter_size,
                           border_mode='same', bias=True)(conv_2)

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

    local_diff = y_pred[:, :, 1:, 1:] - y_pred[:, :, :-1, :-1]
    diff_mag = K.sqrt((local_diff ** 2).sum(axis=1))
    mean_diff_mag = diff_mag.mean(axis=[1, 2])
    return K.mean(mean_mag + 0.05 * mean_diff_mag)

if __name__ == '__main__':
    num_feature_maps = 64
    INPUT_SHAPE = (2, None, None)

    x = Input(shape=INPUT_SHAPE)
    pre = Convolution2D(num_feature_maps, 5, 5, bias=True, border_mode='same')(x)
    post = residual_chain(pre, 15, num_feature_maps)
    # 3 outputs per voxel, 3x3 final filter
    output = Convolution2D(2, 3, 3, activation=None, border_mode='same')(post)
    model = Model(input=x, output=output)

    model.compile(optimizer=SGD(lr=0.0005, clipnorm=1., momentum=0.9),
                  loss=mean_vec_diff)

    model.load_weights('pretrained_1.hdf5')

    example1 = imread.imread(sys.argv[1]).astype(np.float32) / 255
    example2 = imread.imread(sys.argv[2]).astype(np.float32) / 255
    label1 = imread.imread(sys.argv[3])
    label2 = imread.imread(sys.argv[4])
    test_input = np.stack((example1, example2))[np.newaxis, ...]

    gen = gen_distorted('images', (2, 96, 96), batch_size=32, max_distortion=15)
    _, distortion = next(gen)  # debugging
    dmag = np.sqrt((distortion ** 2).sum(axis=1))
    print("mean D", dmag.mean(axis=(1, 2)).mean())

    epoch_idx = 1
    while True:
        model.fit_generator(gen,
                            4096 * 256, 1, verbose=1)
        model.save_weights('weights_{}.h5'.format(epoch_idx))

        predicted_warp = model.predict(test_input)
        imshape = example1.shape
        orig_i, orig_j = np.ogrid[:imshape[0], :imshape[1]]
        sample_i = orig_i + predicted_warp[0, 0, ...]
        sample_j = orig_j + predicted_warp[0, 1, ...]
        remapped = cv2.remap(example1, sample_j.astype(np.float32), sample_i.astype(np.float32), cv2.INTER_LINEAR)
        remapped_l = cv2.remap(label1, sample_j.astype(np.float32), sample_i.astype(np.float32), cv2.INTER_NEAREST)

        pylab.figure(figsize=(40, 5))
        pylab.gray()
        pylab.subplot(1, 7, 1)
        pylab.imshow(example1[200:400, 200:400])
        pylab.subplot(1, 7, 2)
        pylab.imshow(remapped[200:400, 200:400])
        pylab.subplot(1, 7, 3)
        pylab.imshow(example2[200:400, 200:400])

        pylab.subplot(1, 7, 4)
        pylab.imshow(label1[200:400, 200:400])
        pylab.jet()
        pylab.subplot(1, 7, 5)
        pylab.imshow(remapped_l[200:400, 200:400])
        pylab.subplot(1, 7, 6)
        pylab.imshow(label2[200:400, 200:400])
        pylab.subplot(1, 7, 7)
        pylab.imshow(predicted_warp[0, 0, 200:400, 200:400])
        pylab.colorbar()
        pylab.savefig("iteration_{}.png".format(epoch_idx), dpi=150)
        pylab.close()
