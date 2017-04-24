import os
import argparse
from configobj import ConfigObj
from validate import Validator
from utils import convert, compute_synapse_features
import random
import string

# Command-line parameters
parser = argparse.ArgumentParser(description='ccboost service')
parser.add_argument('--train', type=str, help='Config file for training')
parser.add_argument('--test', type=str, help='Config file for testing')
params = parser.parse_args()
if params.train is None and params.test is None:
    raise RuntimeError('Must specify a config file')
if params.train is not None and params.test is not None:
    raise RuntimeError('Please run separate instances for training and stand-alone testing')

if params.train:
    is_train = True
    config_file = params.train
else:
    is_train = False
    config_file = params.test

# Config file
if is_train:
    config = ConfigObj(config_file, configspec='config/trainspec.cfg')
    print('CCBOOST Service :: config = {}'.format(config))
else:
    config = ConfigObj(config_file, configspec='config/testspec.cfg')
if not config:
    raise RuntimeError(
        'Could not load the configuration file: "{}"'.format(config_file))

validator = Validator()
v = config.validate(validator)
if v is not True:
    raise RuntimeError(
        'Errors validating the training configuration file: {}'.format(v))

# Create dataset folder
root = config['root']
if not os.path.isdir(root + '/runs'):
    os.mkdir(root + '/runs')
root += '/runs'
if not os.path.isdir(root + '/' + config['dataset_name']):
    os.mkdir(root + '/' + config['dataset_name'])
root += '/' + config['dataset_name']
folder_dataset = root

# Folders for features and results
if not os.path.isdir(folder_dataset + '/features'):
    os.mkdir(folder_dataset + '/features')
if not os.path.isdir(folder_dataset + '/results'):
    os.mkdir(folder_dataset + '/results')
if not os.path.isdir(folder_dataset + '/results/' + config['model_name']):
    os.mkdir(folder_dataset + '/results/' + config['model_name'])
folder_results = folder_dataset + '/results/' + config['model_name']

# Create model folder
root = config['root']
if not os.path.isdir(root + '/models'):
    os.mkdir(root + '/models')
root += '/models'
if not os.path.isdir(root + '/' + config['model_name']):
    os.mkdir(root + '/' + config['model_name'])
root += '/' + config['model_name']
folder_model = root

# Convert from h5 to tif
print('CCBOOST Service :: Converting data into TIFF')
config['stack_tif'] = convert(config['stack'], 'tif')
if is_train:
    config['labels_tif'] = convert(config['labels'], 'tif')

# Dilate training stack and labels to avoid boundary issues
# print('CCBOOST Service :: Dilating stacks and labels')

# Compute features
print('CCBOOST Service :: Computing features')
compute_synapse_features(config['stack_tif'], folder_dataset + '/features')

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
    with open('templates/train.cfg', 'r') as f:
        template = f.read()

    # Replace variables
    template = template.replace('VAR_NUM_STUMPS', '{}'.format(str(config['num_adaboost_stumps'])))
    template = template.replace('VAR_OUTPUT_PREFIX', '"' + folder_results + '/out"')
    template = template.replace('VAR_DATA', '"' + config['stack_tif'] + '"')
    template = template.replace('VAR_LABELS', '"' + config['labels_tif'] + '"')
    template = template.replace('VAR_ORIENTATION', '"' + folder_dataset + '/features/hessOrient-s3.5-repolarized.nrrd' + '"')
    for i in range(len(feats)):
        feats[i] = '"' + folder_dataset + '/features/' + feats[i] + '"'
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
    template = template.replace('VAR_DATA', '"' + config['stack_tif'] + '"')
    template = template.replace('VAR_ORIENTATION', '"' + folder_dataset + '/features/hessOrient-s3.5-repolarized.nrrd' + '"')
    for i in range(len(feats)):
        feats[i] = '"' + folder_dataset + '/features/' + feats[i] + '"'
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
cmd = './ccboost-v0.21/build/ccboost {}'.format(tmp_name)
print(cmd)
os.system(cmd)

# Convert results to tif so we can see something, and h5 so we can get them back to the server
r_tif = convert(folder_results + '/out-0-ab-max.nrrd', 'tif')
r_h5 = convert(folder_results + '/out-0-ab-max.nrrd', 'h5')
print('CCBOOST Service :: Results stored in h5 at "{}"'.format(r_h5))

# Move models to the right location
if is_train:
    os.rename(folder_results + '/out-backupstumps.cfg', folder_model + '/backup-stumps.cfg')
    os.rename(folder_results + '/out-stumps.cfg', folder_model + '/stumps.cfg')
    print('CCBOOST Service :: Trained models stored at "{}"'.format(folder_model))

# TODO
# - Dilate and mirror data and labels (right now stack borders are bad), and prune it from the output
# - Sanity-check the strings as we're running on the command line (remove ";", others?)
# - Send log to server during processing (redirect output to /tmp file?)
# - Actually not recompute the features if they exist
# - We may want to return an error measure while training and maybe also testing (no labels loaded while testing, for now)
# - Dilate ground truth on the fly to ignore pixels around annotations
# - Ilastik: we return multiple channels. Do we want to visualize a binary output or binarize it on the fly with a given threshold?
