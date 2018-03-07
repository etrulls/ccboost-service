import tifffile
import h5py
import os
import nrrd
import numpy as np
from scipy import ndimage
from skimage import morphology
import sys
from time import time
from datetime import datetime


def timestamp():
    ts = time()
    return datetime.fromtimestamp(ts).strftime('%H:%M:%S')


def convert(src, fmt, delete_after=False):
    '''
    Convert files between HDF5, TIF and NRRD.
    '''

    if not os.path.isfile(src):
        raise RuntimeError('Source file does not exist: {}'.format(src))

    # Current format
    curr_fmt = src.split('.')[-1]
    if curr_fmt != 'h5' and curr_fmt != 'tif' and curr_fmt != 'nrrd':
        raise RuntimeError('Unsupported format')

    # Output name
    base = '.'.join(src.split('.')[:-1])
    if fmt == 'h5':
        tgt = base + '.h5'
    elif fmt in ['tif', 'tiff']:
        tgt = base + '.tif'
    elif fmt == 'nrrd':
        tgt = base + '.nrrd'
    else:
        raise RuntimeError('Unsupported format')

    # No need to re-convert (or self-convert)
    # if os.path.isfile(tgt):
    #     return tgt

    # Do this every time, for safety
    if os.path.isfile(tgt):
        os.remove(tgt)

    # Parse input
    if curr_fmt == 'tif':
        x = tifffile.imread(src)
    elif curr_fmt == 'h5':
        f = h5py.File('{}'.format(src), 'r')
        x = f['data'].value
        f.close()
    elif curr_fmt == 'nrrd':
        x, opts = nrrd.read(src)
        x = x.transpose((2, 1, 0))

    # Convert
    if fmt == 'h5':
        with h5py.File(tgt, 'w') as f:
            f.create_dataset(name='data', data=x, chunks=True)
            # f.create_dataset(name='data', data=x, chunks=tuple(25 for i in range(len(x.shape))))
    elif fmt == 'tif':
        tifffile.imsave(tgt, x)
    elif fmt == 'nrrd':
        nrrd.write(tgt, x.transpose((2, 1, 0)))

    # Delete
    if delete_after:
        os.remove(src)

    # Return filename
    return tgt


def compute_synapse_features(data, output_folder, mirrored, force_recompute=False, verbose=False):
    '''
    Compute synapse features (hard-coded).
    '''

    if mirrored > 0:
        suffix = '-mirrored-{}'.format(mirrored)
    else:
        suffix = ''

    bin_loc = os.path.dirname(
        os.path.realpath(__file__)) + "/ccboost-v0.21/build"
    # gaussBin = 'GaussianImageFilter'
    gradBin = 'GradientMagnitudeImageFilter'
    # LoGBin = 'LoGImageFilter'
    # eigOfHessBin = 'EigenOfHessianImageFilter' 
    eigOfST = 'EigenOfStructureTensorImageFilter'
    singleEigVecHess = 'SingleEigenVectorOfHessian'
    allEigVecHess = 'AllEigenVectorsOfHessian'
    repolarizeOrient = 'RepolarizeYVersorWithGradient'

    if not os.path.isdir(output_folder):
        os.path.mkdir(output_folder)

    print(timestamp() + ' Computing eigenvectors of the structure tensor (s=0.5/r=1.0)...')
    sys.stdout.flush()
    o = '{}/stensor-s0.5-r1.0{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 0.5 1.0 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    print(timestamp() + ' Computing eigenvectors of the structure tensor (s=0.8/r=0.6)...')
    sys.stdout.flush()
    o = '{}/stensor-s0.8-r1.6{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 0.8 1.6 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    print(timestamp() + ' Computing eigenvectors of the structure tensor (s=1.8/r=3.5)...')
    sys.stdout.flush()
    o = '{}/stensor-s1.8-r3.5{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 1.8 3.5 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    print(timestamp() + ' Computing eigenvectors of the structure tensor (s=2.5/r=5.0)...')
    sys.stdout.flush()
    o = '{}/stensor-s2.5-r5.0{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 2.5 5.0 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    print(timestamp() + ' Computing single eigenvector of Hessian...')
    sys.stdout.flush()
    o = '{}/hessOrient-s3.5-highestMag{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 3.5 1.0 {} 1'.format(bin_loc, singleEigVecHess, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    print(timestamp() + ' Computing all eigenvectors of Hessian...')
    sys.stdout.flush()
    o = '{}/hessOrient-s3.5-allEigVecs{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 3.5 1.0 {} 1'.format(bin_loc, allEigVecHess, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    print(timestamp() + ' Repolarizing orientation...')
    sys.stdout.flush()
    o = '{}/hessOrient-s3.5-repolarized{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} {}/hessOrient-s3.5-allEigVecs{}.nrrd 3.5 1.0 {}'.format(bin_loc, repolarizeOrient, data, output_folder, suffix, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Command: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))
    sys.stdout.flush()

    for sigma in [1.0, 1.6, 3.5, 5.0]:
        print(timestamp() + ' Computing gradients (s={0:.1f})...'.format(sigma))
        sys.stdout.flush()
        str_sigma = '{0:.2f}'.format(sigma)
        o = '{}/gradient-magnitude-s{}{}.nrrd'.format(output_folder, str_sigma, suffix)
        s = '{}/{} {} {} 1.0 {}'.format(bin_loc, gradBin, data, str_sigma, o)
        if not os.path.isfile(o) or force_recompute:
            if verbose:
                print('Command: "{}"'.format(s))
            os.system(s)
        elif verbose:
            print('Skipping: "{}"'.format(s))
        sys.stdout.flush()


def dilate_labels(data, dilation, dims=2):
    '''
    Dilate labels to ignore boundaries.
    * dilation: tuple (inner, outer) dilation, in pixels
    * dims: 2 (2D) or 3 (3D) dilation
    * value: label for the dilated pixels
    '''

    if dilation == (0, 0):
        return data
    else:
        # Get current mask
        mask = data == 128

        # Rank: number of dimensions
        # Connectivity -> 1: conn-4, 2: conn-8, 3: all on for 3D filters
        filt = ndimage.generate_binary_structure(rank=dims, connectivity=1)

        inner = np.zeros(data.shape).astype(bool)
        outer = np.zeros(data.shape).astype(bool)
        if dims == 2:
            for i in range(data.shape[0]):
                inner[i] = ndimage.binary_erosion(
                    data[i] > 128,
                    structure=filt,
                    iterations=dilation[0],
                    border_value=True,
                )
                outer[i] = ndimage.binary_dilation(
                    data[i] > 128,
                    structure=filt,
                    iterations=dilation[1],
                    border_value=False,
                )
        elif dims ==3:
            inner = ndimage.binary_erosion(
                data > 128,
                structure=filt,
                iterations=dilation[0],
                border_value=True,
            )
            outer = ndimage.binary_dilation(
                data > 128,
                structure=filt,
                iterations=dilation[1],
                border_value=False,
            )
        else:
            raise RuntimeError('Parameter "dims" must be 2 or 3')

        if dilation[0] == 0:
            inner.fill(False)
        else:
            inner = (data > 128) ^ inner

        if dilation[1] == 0:
            outer.fill(False)
        else:
            outer = outer ^ (data > 128)

        # Copy
        dilated = data.copy()

        # Re-apply original mask
        dilated[mask] = 128

        # Assign dilated positives
        dilated[inner + outer] = 255

        return dilated


# def dilate_labels(data, ignore):
#     '''
#     Dilate labels to ignore boundaries.
#     Uses a filter size 'ignore' and connectivity-1, both inside and outside.
#     '''
# 
#     if ignore == 0:
#         return data
#     else:
#         # 1: conn-4, 2: conn-8
#         filt = ndimage.generate_binary_structure(2, 1)
# 
#         # Inner dilation
#         inner = np.zeros(data.shape).astype(np.uint8)
#         outer = np.zeros(data.shape).astype(np.uint8)
#         for i in range(data.shape[0]):
#             inner[i] = ndimage.binary_erosion(data[i] > 0, structure=filt, iterations=ignore)
#             outer[i] = ndimage.binary_dilation(data[i] > 0, structure=filt, iterations=ignore)
#         outer = outer - data
#         inner = data - inner
# 
#         # Unite
#         d = (inner + outer).astype(np.uint8) * 128
#         data[d > 0] = d[d > 0]
# 
#     return data
