import os
import tifffile
from utils import convert_tif_v_h5

for fn in os.listdir('my_raw_input'):
    if fn.endswith('.tif'):
        # Generate h5 for the full version
        convert_tif_v_h5('my_raw_input/' + fn, tif_to_h5=True)
        fn_base = '.'.join(fn.split('.')[:-1])
        os.rename('my_raw_input/' + fn_base + '.h5', 'input/' + fn_base + '.h5')

        # Make small stacks for testing
        f = tifffile.imread('my_raw_input/' + fn)
        s = f.shape

        f = f[30:130, 100:500, 300:900]
        print(s, ' -> ', f.shape)
        tifffile.imsave('my_raw_input/' + fn_base + '-small.tif', f)
        convert_tif_v_h5('my_raw_input/' + fn_base + '-small.tif', tif_to_h5=True)
        os.rename('my_raw_input/' + fn_base + '-small.h5', 'input/' + fn_base + '-small.h5')
