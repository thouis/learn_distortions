import matplotlib
# matplotlib.use('WXAgg')
import pylab

import numpy as np
from scipy.interpolate import RectBivariateSpline
import glob
import os.path
import imread
import cv2

def load_images(image_dir):
    filenames = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    for f in filenames:
        imread.imread(f)
    ims = [imread.imread(f) for f in filenames]
    print("read images", len(ims))
    return ims

def smooth_1channel_warp(shp, spacing):
    d = min(shp) // spacing
    smallwarp = np.random.uniform(-1, 1, (d, d))
    coo = np.linspace(0, 1, d)
    r = RectBivariateSpline(coo, coo, smallwarp, kx=3, ky=3)
    return r(np.linspace(0, 1, shp[0]),
             np.linspace(0, 1, shp[1]))

def random_warp(shp, max_distortion=5, spacing=50):
    warp = np.stack([smooth_1channel_warp(shp, spacing=spacing),
                     smooth_1channel_warp(shp, spacing=spacing)])
    largest = np.sqrt((warp ** 2).sum(axis=0)).max()
    warp /= largest
    warp *= max_distortion
    return warp

def random_distortion(ims, image_size, max_distortion=5, spacing=50):
    '''choose two adjacent images from ims.  return the first one unchanged, the
    second distorted, and the distortion.

    '''

    imshape = ims[0].shape

    # first image
    idx = np.random.randint(len(ims) - 1)

    # cutout random square with some padding
    padded = (image_size[0] + 2 * max_distortion + 3 * spacing, image_size[1] + 2 * max_distortion + 3 * spacing)
    base_i = np.random.randint(imshape[0] - padded[0])
    base_j = np.random.randint(imshape[1] - padded[1])
    subim1 = ims[idx][base_i:, base_j:][:padded[0], :padded[1]]
    subim2 = ims[idx + 1][base_i:, base_j:][:padded[0], :padded[1]]

    # randomly swap which is the first image
    if np.random.randint(2) == 0:
        subim1, subim2 = subim2, subim1

    # random flips, tranposes
    if np.random.randint(2) == 0:
        subim1 = subim1[::-1, :]
        subim2 = subim2[::-1, :]
    if np.random.randint(2) == 0:
        subim1 = subim1[:, ::-1]
        subim2 = subim2[:, ::-1]
    if np.random.randint(2) == 0:
        subim1 = subim1.T
        subim2 = subim2.T

    distortion = random_warp(subim2.shape, max_distortion=max_distortion, spacing=spacing)

    orig_i, orig_j = np.ogrid[:subim2.shape[0], :subim2.shape[1]]
    sample_i = orig_i + distortion[0, ...]
    sample_j = orig_j + distortion[1, ...]
    distorted = cv2.remap(subim2, sample_j.astype(np.float32), sample_i.astype(np.float32), cv2.INTER_LINEAR)

    center_cutout = ((padded[0] - image_size[0]) // 2,
                     (padded[1] - image_size[1]) // 2)

    inputs = np.stack([subim1, distorted])
    inputs = inputs[:, center_cutout[0]:, center_cutout[1]:]
    inputs = inputs[:, :image_size[0], :image_size[1]]
    inputs = inputs.astype(np.float32) / 255

    outputs = distortion[:, center_cutout[0]:, center_cutout[1]:]
    outputs = outputs[:, :image_size[0], :image_size[1]]

    return inputs, outputs

def gen_distorted(image_dir, image_size, batch_size=16, **kwargs):
    ims = load_images(image_dir)
    while True:
        examples = [random_distortion(ims, image_size[1:], **kwargs) for _ in range(batch_size)]
        distorted_ims, distortions = zip(*examples)
        yield np.stack(distorted_ims), np.stack(distortions)
