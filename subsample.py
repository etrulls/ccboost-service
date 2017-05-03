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
        f = tifffile.imread('my_raw_input/' + fn)
        s = f.shape

        f = f[50:100, 200:500, 600:900]
        print(s, ' -> ', f.shape)
        tifffile.imsave('my_raw_input/' + fn_base + '-smaller.tif', f)
        convert('my_raw_input/' + fn_base + '-smaller.tif', 'h5')
        os.rename('my_raw_input/' + fn_base + '-smaller.h5', 'input/' + fn_base + '-smaller.h5')
