import os
import tifffile
from utils import convert
import numpy as np

files = {
    'hipp-train-data.tif',
    'hipp-train-mito.tif',
    'hipp-train-syn.tif',
    'hipp-test-data.tif',
    'hipp-test-mito.tif',
    'hipp-test-syn.tif',
}

src = '/cvlabdata1/cvlab/datasets_eduard/em/tif2'
for fn in files:
    print('Processing {}'.format(fn))

    # Fix labels and overwrite
    if fn == 'hipp-train-mito.tif' or fn == 'hipp-train-syn.tif':
        l = tifffile.imread(src + '/' + fn)

        # RGB
        if len(l.shape) == 4:
            l = l[:, :, :, 0]

        # uint16 and too large
        u = np.unique(l)
        if len(u) > 2:
            raise RuntimeError('Unknown labels file')
        l[l == l.max()] = 255
        tifffile.imsave(src + '/' + fn, l.astype(np.uint8))

    # Generate h5 for the full version
    convert(src + '/' + fn, 'h5')
    fn_base = '.'.join(fn.split('.')[:-1])
    # os.rename(src + '/' + fn_base + '.h5', 'input/' + fn_base + '.h5')

    # Make small stacks for testing
    v = tifffile.imread(src + '/' + fn)
    s = v.shape

    f = v[30:120, 100:600, 300:1000]
    print('Small: ', s, ' -> ', f.shape)
    tifffile.imsave(src + '/' + fn_base + '-small.tif', f)
    convert(src + '/' + fn_base + '-small.tif', 'h5')
    # os.rename(src + '/' + fn_base + '-small.h5', 'input/' + fn_base + '-small.h5')

    f = v[50:100, 200:500, 600:900]
    print('Smaller: ', s, ' -> ', f.shape)
    tifffile.imsave(src + '/' + fn_base + '-smaller.tif', f)
    convert(src + '/' + fn_base + '-smaller.tif', 'h5')
    # os.rename(src + '/' + fn_base + '-smaller.h5', 'input/' + fn_base + '-smaller.h5')

    f = v[60:90, 300:500, 600:800]
    print('Smallest: ', s, ' -> ', f.shape)
    tifffile.imsave(src + '/' + fn_base + '-smallest.tif', f)
    convert(src + '/' + fn_base + '-smallest.tif', 'h5')
    # os.rename(src + '/' + fn_base + '-smallest.h5', 'input/' + fn_base + '-smallest.h5')
