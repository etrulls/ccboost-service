import os
import argparse
from configobj import ConfigObj
from validate import Validator
from utils import convert, compute_synapse_features
import random
import string
import h5py
import tifffile
import numpy as np
import nrrd



# Command-line parameters
parser = argparse.ArgumentParser(description='ccboost service')
parser.add_argument('--train', type=str, help='Config file for training')
parser.add_argument('--test', type=str, help='Config file for testing')
parser.add_argument('--username', type=str, help='Username')
parser.add_argument('--tag', type=str, help='Data tag')
parser.add_argument('--recompute', dest='recompute', default=False, action='store_true', help='Recompute CCboost features, even if cached')
parser.add_argument('--dilate', type=int, default=0, help='Mirror data/labels (number of voxels)')

params = parser.parse_args()
username = params.username
if params.train is None and params.test is None:
    raise RuntimeError('Must specify a config file')
if params.train is not None and params.test is not None:
    raise RuntimeError('Please run separate instances for training and stand-alone testing')
if params.username is None:
    raise RuntimeError('Must specify a username')
if params.tag is None:
    raise RuntimeError('Must specify a tag')

dir_path = os.path.dirname(os.path.realpath(__file__))
username = params.username
tag = params.tag

if params.train:
    is_train = True
    config_file = params.train
else:
    is_train = False
    config_file = params.test

# Config file
if is_train:
    config = ConfigObj(config_file, configspec=dir_path + '/config/trainspec.cfg')
    print('CCBOOST Service :: config = {}'.format(config))
else:
    config = ConfigObj(config_file, configspec=dir_path + '/config/testspec.cfg')
if not config:
    raise RuntimeError(
        'Could not load the configuration file: "{}"'.format(config_file))   #BUG1

validator = Validator()
v = config.validate(validator)
if v is not True:
    raise RuntimeError(
        'Errors validating the training configuration file: {}'.format(v))


# Root folder
root = config['root'] + '/workspace/' + username
if not os.path.isdir(root):
    os.makedirs(root)

# Features folder
folder_features = root + '/runs/' + config['dataset_name'] + '/features'
if not os.path.isdir(folder_features):
    os.makedirs(folder_features)

# Scratch
scratch = root + '/runs/' + config['dataset_name'] + '/scratch'
if not os.path.isdir(scratch):
    os.makedirs(scratch)

# Results folder
folder_results = root + '/runs/' + config['dataset_name'] + '/results/' + config['model_name']
if not os.path.isdir(folder_results):
    os.makedirs(folder_results)

# Models folder
folder_model = root + '/models/' + config['model_name']
if not os.path.isdir(folder_model):
    os.makedirs(folder_model)


# Convert from h5 to tif
print('CCBOOST Service :: Converting data into TIFF')
config['stack_tif'] = convert(config['stack'], 'tif')
if is_train:
    config['labels_tif'] = convert(config['labels'], 'tif')

# Dilate training stack and labels to avoid boundary issues
# Will be deleted at the end of the run, but it should not be displayed to the user (h5 only)
if params.dilate > 0:
    print('CCBOOST Service :: Dilating stacks and labels')
    print('CCBOOST Service :: Source data file: {}'.format(config['stack_tif']))
    s = config['stack_tif'].split('/')
    fn_file = s[-1]
    fn_base = '.'.join(fn_file.split('.')[0:-1])
    fn_ext = fn_file.split('.')[-1]
    stack = '{}/{}-mirrored-{}.{}'.format(scratch, fn_base, params.dilate, fn_ext)
    if os.path.isfile(stack):
        print('CCBOOST Service :: Skipping, exists: {}'.format(stack))
    else:
        print('CCBOOST Service :: Mirrored data file: {}'.format(stack))
        x = tifffile.imread(config['stack_tif'])
        x = np.pad(x, params.dilate, 'reflect')
        tifffile.imsave(stack, x)
        x = None

    if is_train:
        s = config['labels_tif'].split('/')
        fn_file = s[-1]
        fn_base = '.'.join(fn_file.split('.')[0:-1])
        fn_ext = fn_file.split('.')[-1]
        labels = '{}/{}-mirrored-{}.{}'.format(scratch, fn_base, params.dilate, fn_ext)
        if os.path.isfile(labels):
            print('CCBOOST Service :: Skipping, exists: {}'.format(labels))
        else:
            print('CCBOOST Service :: Mirrored labels file: {}'.format(labels))
            x = tifffile.imread(config['labels_tif'])
            x = np.pad(x, params.dilate, 'reflect')
            tifffile.imsave(labels, x)
            x = None

    from time import sleep
    sleep(5)
else:
    stack = config['stack_tif']
    labels = config['labels_tif']

# Compute features
print('CCBOOST Service :: Computing features')
compute_synapse_features(
    stack,
    folder_features,
    force_recompute=params.recompute)

# Generate the configuration file to train the model
feats = [
    "gradient-magnitude-s1.00.nrrd",
    "gradient-magnitude-s1.60.nrrd",
    "gradient-magnitude-s3.50.nrrd",
    "gradient-magnitude-s5.00.nrrd",
    "stensor-s0.5-r1.0.nrrd",
    "stensor-s0.8-r1.6.nrrd",
    "stensor-s1.8-r3.5.nrrd",
    "stensor-s2.5-r5.0.nrrd"
]
if is_train:
    # Open template
    with open(dir_path + '/templates/train.cfg', 'r') as f:
        template = f.read()

    # Replace variables
    template = template.replace('VAR_NUM_STUMPS', '{}'.format(str(config['num_adaboost_stumps'])))
    template = template.replace('VAR_OUTPUT_PREFIX', '"' + folder_results + '/out"')
    template = template.replace('VAR_DATA', '"' + stack + '"')
    template = template.replace('VAR_LABELS', '"' + labels + '"')
    template = template.replace('VAR_ORIENTATION', '"' + folder_features + '/hessOrient-s3.5-repolarized.nrrd' + '"')
    for i in range(len(feats)):
        feats[i] = '"' + folder_features + '/' + feats[i] + '"'
    feats_single = ', '.join(feats)
    template = template.replace('VAR_FEATURES', feats_single)

    if template.find('VAR_') >= 0:
        raise RuntimeError('Some field was not filled in in the template? ' + template)
else:
    # Open template
    with open('templates/test.cfg', 'r') as f:
        template = f.read()

    # Replace variables
    template = template.replace('VAR_NUM_STUMPS', '{}'.format(str(config['num_adaboost_stumps'])))
    template = template.replace('VAR_OUTPUT_PREFIX', '"' + folder_results + '/out"')
    template = template.replace('VAR_MODEL', '"' + folder_model + '/stumps.cfg"')
    template = template.replace('VAR_DATA', '"' + stack + '"')
    template = template.replace('VAR_ORIENTATION', '"' + folder_features + '/hessOrient-s3.5-repolarized.nrrd' + '"')
    for i in range(len(feats)):
        feats[i] = '"' + folder_features + '/' + feats[i] + '"'
    feats_single = ', '.join(feats)
    template = template.replace('VAR_FEATURES', feats_single)

    if template.find('VAR_') >= 0:
        raise RuntimeError('Some field was not filled in in the template? ' + template)

tmp_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
tmp_name = '/tmp/' + tmp_name
print('CCBOOST Service :: Saving template to "{}"'.format(tmp_name))
f = open(tmp_name, 'w')
f.write(template)
f.close()

# Train model (if necessary) and get results
print('CCBOOST Service :: Calling ccboost binary')
cmd = dir_path + '/ccboost-v0.21/build/ccboost {}'.format(tmp_name)
print(cmd)
os.system(cmd)

# TODO we should really try to catch an error here...
# raise RuntimeError("stop")

# Remove mirrored boundaries for training
ccboost_res = folder_results + '/out-0-ab-max.nrrd'
if params.dilate > 0:
    print('CCBOOST Service :: Removing mirrored boundaries')
    print('CCBOOST Service :: File: {}'.format(ccboost_res))
    x, o = nrrd.read(ccboost_res)
    x = x[params.dilate:-params.dilate,
          params.dilate:-params.dilate,
          params.dilate:-params.dilate
          ]
    nrrd.write(ccboost_res, x, options=o)

# Convert results to tif so we can see something, and h5 so we can get them back to the server
r_tif = convert(ccboost_res, 'tif')
r_h5 = convert(ccboost_res, 'h5')
print('CCBOOST Service :: Results stored in h5 at "{}"'.format(r_h5))

# Delete the tiff files after processing
# TODO (better to keep them for debugging, for now)

# Move models to the right location
if is_train:
    os.rename(folder_results + '/out-backupstumps.cfg', folder_model + '/backup-stumps.cfg')
    os.rename(folder_results + '/out-stumps.cfg', folder_model + '/stumps.cfg')
    print('CCBOOST Service :: Trained models stored at "{}"'.format(folder_model))

# Clean up
# os.remove(config['stack_tif'])
# if os.path.isfile(stack):
#     os.remove(stack)
# if is_train:
#     os.remove(config['labels_tif'])
#     if os.path.isfile(labels):
#         os.remove(labels)

# TODO
# - Sanity-check the strings as we're running on the command line (remove ";", others?)
# - Send log to server during processing (redirect output to /tmp file?)
# - Dilate ground truth on the fly to ignore pixels around annotations
