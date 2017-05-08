import tifffile
import h5py
import os
import nrrd
import numpy as np


def convert(src, fmt, delete_after=False):
    '''
    Convert files between HDF5, TIF and NRRD.
    '''

    if not os.path.isfile(src):
        raise RuntimeError('Source file does not exist')

    # Current format
    curr_fmt = src.split('.')[-1]
    if curr_fmt != 'h5' and curr_fmt != 'tif' and curr_fmt != 'nrrd':
        raise RuntimeError('Unsupported format')

    # Output name
    base = '.'.join(src.split('.')[:-1])
    if fmt == 'h5':
        tgt = base + '.h5'
    elif fmt == 'tif':
        tgt = base + '.tif'
    elif fmt == 'nrrd':
        tgt = base + '.nrrd'
    else:
        raise RuntimeError('Unsupported format')

    # No need to re-convert (or self-convert)
    if os.path.isfile(tgt):
        return tgt

    # Parse input
    if curr_fmt == 'tif':
        x = tifffile.imread(src)
    elif curr_fmt == 'h5':
        f = h5py.File('{}'.format(src), 'r')
        x = f['data'].value
        f.close()
    elif curr_fmt == 'nrrd':
        x, opts = nrrd.read(src)
        x = x.transpose((2,1,0))

    # Convert
    if fmt == 'h5':
        f = h5py.File(tgt, 'w-')
        f.create_dataset(name='data', data=x)
        f.close()
    elif fmt == 'tif':
        tifffile.imsave(tgt, x)
    elif fmt == 'nrrd':
        nrrd.write(tgt, x.transpose((2,1,0)))

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

    bin_loc = dir_path = os.path.dirname(
        os.path.realpath(__file__)) + "/ccboost-v0.21/build"
    gaussBin = 'GaussianImageFilter'
    gradBin = 'GradientMagnitudeImageFilter'
    LoGBin = 'LoGImageFilter'
    eigOfHessBin = 'EigenOfHessianImageFilter' 
    eigOfST = 'EigenOfStructureTensorImageFilter'
    singleEigVecHess = 'SingleEigenVectorOfHessian'
    allEigVecHess = 'AllEigenVectorsOfHessian'
    repolarizeOrient = 'RepolarizeYVersorWithGradient'

    if not os.path.isdir(output_folder):
        os.path.mkdir(output_folder)

    o = '{}/stensor-s0.5-r1.0{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 0.5 1.0 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    o = '{}/stensor-s0.8-r1.6{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 0.8 1.6 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    o = '{}/stensor-s1.8-r3.5{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 1.8 3.5 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    o = '{}/stensor-s2.5-r5.0{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 2.5 5.0 1.0 {} 1'.format(bin_loc, eigOfST, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    o = '{}/hessOrient-s3.5-highestMag{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 3.5 1.0 {} 1'.format(bin_loc, singleEigVecHess, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    o = '{}/hessOrient-s3.5-allEigVecs{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} 3.5 1.0 {} 1'.format(bin_loc, allEigVecHess, data, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    o = '{}/hessOrient-s3.5-repolarized{}.nrrd'.format(output_folder, suffix)
    s = '{}/{} {} {}/hessOrient-s3.5-allEigVecs{}.nrrd 3.5 1.0 {}'.format(bin_loc, repolarizeOrient, data, output_folder, suffix, o)
    if not os.path.isfile(o) or force_recompute:
        if verbose:
            print('Running: "{}"'.format(s))
        os.system(s)
    elif verbose:
        print('Skipping: "{}"'.format(s))

    for sigma in [1.0, 1.6, 3.5, 5.0]:
        str_sigma = '{0:.2f}'.format(sigma)
        o = '{}/gradient-magnitude-s{}{}.nrrd'.format(output_folder, str_sigma, suffix)
        s = '{}/{} {} {} 1.0 {}'.format(bin_loc, gradBin, data, str_sigma, o)
        if not os.path.isfile(o) or force_recompute:
            if verbose:
                print('Running: "{}"'.format(s))
            os.system(s)
        elif verbose:
            print('Skipping: "{}"'.format(s))
