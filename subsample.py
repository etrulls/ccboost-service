import os
import tifffile
from utils import convert

for fn in os.listdir('my_raw_input'):
    if fn.endswith('-data.tif') or fn.endswith('-dil5.tif'):
        # Generate h5 for the full version
        convert('my_raw_input/' + fn, 'h5')
        fn_base = '.'.join(fn.split('.')[:-1])
        os.rename('my_raw_input/' + fn_base + '.h5', 'input/' + fn_base + '.h5')

        # Make small stacks for testing
        v = tifffile.imread('my_raw_input/' + fn)
        s = v.shape

        f = v[30:120, 100:600, 300:1000]
        print('Small: ', s, ' -> ', f.shape)
        tifffile.imsave('my_raw_input/' + fn_base + '-small.tif', f)
        convert('my_raw_input/' + fn_base + '-small.tif', 'h5')
        os.rename('my_raw_input/' + fn_base + '-small.h5', 'input/' + fn_base + '-small.h5')

        f = v[50:100, 200:500, 600:900]
        print('Smaller: ', s, ' -> ', f.shape)
        tifffile.imsave('my_raw_input/' + fn_base + '-smaller.tif', f)
        convert('my_raw_input/' + fn_base + '-smaller.tif', 'h5')
        os.rename('my_raw_input/' + fn_base + '-smaller.h5', 'input/' + fn_base + '-smaller.h5')

        f = v[60:90, 300:500, 600:800]
        print('Smallest: ', s, ' -> ', f.shape)
        tifffile.imsave('my_raw_input/' + fn_base + '-smallest.tif', f)
        convert('my_raw_input/' + fn_base + '-smallest.tif', 'h5')
        os.rename('my_raw_input/' + fn_base + '-smallest.h5', 'input/' + fn_base + '-smallest.h5')
