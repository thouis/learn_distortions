from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import distance_transform_edt

# import matplotlib
# matplotlib.use('WXAgg')
# import pylab

import numpy as np
from scipy.interpolate import RectBivariateSpline
import glob
import os.path
import imread
import cv2
import h5py

def load_images(image_dir):
    filenames = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    for f in filenames:
        imread.imread(f)
    ims = [imread.imread(f) for f in filenames]
    print("read images", len(ims), "from", image_dir)
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

def random_distortion(ims, subimage_size, max_distortion=5, spacing=50):
    '''choose two adjacent images from ims.  return the first one unchanged, the
    second distorted, and the distortion.

    '''

    imshape = ims[0].shape

    # first image
    idx = np.random.randint(len(ims) - 1)

    # cutout random square with some padding
    padded = (subimage_size[0] + 2 * max_distortion + 3 * spacing, subimage_size[1] + 2 * max_distortion + 3 * spacing)
    base_i = np.random.randint(imshape[0] - padded[0])
    base_j = np.random.randint(imshape[1] - padded[1])
    subim1 = ims[idx][base_i:, base_j:][:padded[0], :padded[1]]
    subim2 = ims[idx + 1][base_i:, base_j:][:padded[0], :padded[1]]

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

    inputs = np.stack([subim1, distorted])[:, :subimage_size[0], :subimage_size[1]].astype(np.float32) / 255
    outputs = distortion[:, :subimage_size[0], :subimage_size[1]]

    return inputs, outputs

def gen_distorted(image_dir, subimage_size, batch_size=16):
    ims = load_images(image_dir)
    while True:
        examples = [random_distortion(ims, subimage_size[1:]) for _ in range(batch_size)]
        distorted_ims, distortions = zip(*examples)
        yield np.stack(distorted_ims), np.stack(distortions)

def find_boundaries(labels):
    max_label = maximum_filter(labels, 3)
    min_label = minimum_filter(labels + (labels == 0) * (labels.max() + 1), 3)
    min_label[labels == 0] = 0
    difference = (max_label != min_label)
    return (difference | (labels == 0))

def find_offset_vectors(boundaries):
    dists, (i_ind, j_ind) = distance_transform_edt(~ boundaries, return_distances=True, return_indices=True)
    i_base, j_base = np.ogrid[:boundaries.shape[0], :boundaries.shape[1]]
    offsets_i = i_ind - i_base
    offsets_j = j_ind - j_base
    assert np.allclose(dists, np.sqrt(offsets_i ** 2 + offsets_j ** 2))
    return np.stack([offsets_i, offsets_j])


def gen_icp(image_dir, label_dir, subimage_size, batch_size=16):
    ims = load_images(image_dir)[:5]
    labels = load_images(label_dir)[:5]
    print("LAB", len(labels))
    zeros = [find_boundaries(l) for l in labels]
    offsets_to_zero = [find_offset_vectors(l) for l in zeros]

    for z, off in zip(zeros, offsets_to_zero[1:]):
        deltas = off[:, z > 0]
        mag = np.sqrt((deltas**2).sum(axis=0))
        print(np.mean(mag))

    f = h5py.File('debug.hdf5', 'w')
    f.create_dataset('im', data=ims[0])
    f.create_dataset('lab', data=labels[0])
    f.create_dataset('zero', data=zeros[0])
    f.create_dataset('offset', data=offsets_to_zero[0])
    f.close()
    del f

    imshape = ims[0].shape

    def image_gen():
        while True:
            # first image
            idx1 = np.random.randint(len(ims) - 1)
            # second image
            idx2 = idx1 + 1

            # random cutout window
            base_i = np.random.randint(imshape[0] - subimage_size[0])
            base_j = np.random.randint(imshape[1] - subimage_size[1])
            cutout_i = slice(base_i, base_i + subimage_size[0])
            cutout_j = slice(base_j, base_j + subimage_size[1])

            # extract subimages
            im1 = ims[idx1][cutout_i, cutout_j].astype(np.float32) / 255
            im2 = ims[idx2][cutout_i, cutout_j].astype(np.float32) / 255
            zeros1 = zeros[idx1][cutout_i, cutout_j]
            offsets2 = offsets_to_zero[idx2][:, cutout_i, cutout_j]  # 2 channels vector

            # random flips, tranposes
            allims = [im1, im2, zeros1, offsets2]
            if np.random.randint(2) == 0:
                allims = [im[..., ::-1, :] for im in allims]
            if np.random.randint(2) == 0:
                allims = [im[..., ::-1] for im in allims]
            if np.random.randint(2) == 0:
                allims = [(im.T if len(im.shape) == 2 else im.transpose([0, 2, 1]))
                          for im in allims]
            yield allims

    subimage_generator = image_gen()
    while True:
        examples = [next(subimage_generator) for _ in range(batch_size)]
        batchim1, batchim2, batchzeros, batchoffsets = [np.stack(l) for l in zip(*examples)]
        inputs = np.stack([batchim1, batchim2], axis=1)
        outputs = np.concatenate((batchoffsets, batchzeros[:, np.newaxis, ...].astype(batchoffsets.dtype)), axis=1)
        assert inputs.shape == (batch_size, 2) + subimage_size
        assert outputs.shape == (batch_size, 3) + subimage_size
        yield inputs, outputs
