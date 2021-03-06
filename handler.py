import os
import argparse
from configobj import ConfigObj
from validate import Validator
from utils import convert, compute_synapse_features, dilate_labels, timestamp
import random
import string
import tifffile
import numpy as np
import nrrd
import sys


if __name__ == "__main__":
    # Command-line parameters
    parser = argparse.ArgumentParser(description='ccboost service')
    parser.add_argument('--train', type=str, help='Config file for training')
    parser.add_argument('--test', type=str, help='Config file for testing')
    parser.add_argument('--username', type=str, help='Username')
    parser.add_argument('--recompute', dest='recompute', default=False, action='store_true', help='Recompute CCboost features, even if cached')

    params = parser.parse_args()
    username = params.username
    if params.train is None and params.test is None:
        raise RuntimeError('Must specify a config file')
    if params.train is not None and params.test is not None:
        raise RuntimeError('Please run separate instances for training and stand-alone testing')
    if params.username is None:
        raise RuntimeError('Must specify a username')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    username = params.username

    if params.train:
        is_train = True
        config_file = params.train
    else:
        is_train = False
        config_file = params.test

    # Config file
    if is_train:
        config = ConfigObj(config_file, configspec=dir_path + '/config/trainspec.cfg')
        print(timestamp() + ' config = {}'.format(config))
        sys.stdout.flush()
    else:
        config = ConfigObj(config_file, configspec=dir_path + '/config/testspec.cfg')
    if not config:
        raise RuntimeError(
            'Could not load the configuration file: "{}"'.format(config_file))

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
    print(timestamp() + ' Converting data into TIFF')
    sys.stdout.flush()
    stack = convert(config['stack'], 'tif')
    if is_train:
        labels = convert(config['labels'], 'tif')

    # Mirror training stack and labels to avoid boundary issues
    # Will be deleted at the end of the run, but it should not be displayed to the user (h5 only)
    if config['mirror'] > 0:
        print(timestamp() + ' Mirroring stacks and labels')
        print(timestamp() + ' Source data file: {}'.format(stack))
        sys.stdout.flush()
        s = stack.split('/')
        fn_file = s[-1]
        fn_base = '.'.join(fn_file.split('.')[0:-1])
        fn_ext = fn_file.split('.')[-1]
        stack_m = '{}/{}-mirrored-{}.{}'.format(scratch, fn_base, config['mirror'], fn_ext)
        if os.path.isfile(stack_m):
            print(timestamp() + ' Skipping, exists: {}'.format(stack_m))
            sys.stdout.flush()
        else:
            print(timestamp() + ' Mirrored data file: {}'.format(stack_m))
            sys.stdout.flush()
            x = tifffile.imread(stack)
            x = np.pad(x, config['mirror'], 'reflect')
            tifffile.imsave(stack_m, x)
            del x
        stack = stack_m

        if is_train:
            s = labels.split('/')
            fn_file = s[-1]
            fn_base = '.'.join(fn_file.split('.')[0:-1])
            fn_ext = fn_file.split('.')[-1]
            labels_m = '{}/{}-mirrored-{}.{}'.format(scratch, fn_base, config['mirror'], fn_ext)
            if os.path.isfile(labels_m):
                print(timestamp() + ' Skipping, exists: {}'.format(labels_m))
                sys.stdout.flush()
            else:
                print(timestamp() + ' Mirrored labels file: {}'.format(labels_m))
                sys.stdout.flush()
                x = tifffile.imread(labels)
                x = np.pad(x, config['mirror'], 'reflect')
                tifffile.imsave(labels_m, x)
                del x
            labels = labels_m

        # from time import sleep
        # sleep(5)

    # Ignore pixels around annotations
    if is_train:
        if sum(config['ignore']) > 0:
            print(timestamp() + ' Dilating labels')
            s = labels.split('/')
            fn_file = s[-1]
            fn_base = '.'.join(fn_file.split('.')[0:-1])
            fn_ext = fn_file.split('.')[-1]
            labels_d = '{}/{}-ignore-{}-{}.{}'.format(scratch, fn_base, config['ignore'][0], config['ignore'][1], fn_ext)
            x = tifffile.imread(labels)
            x = dilate_labels(x, config['ignore'])
            tifffile.imsave(labels_d, x)
            del x
            labels = labels_d

    # Compute features
    print(timestamp() + ' Computing features')
    sys.stdout.flush()
    compute_synapse_features(
        stack,
        folder_features,
        config['mirror'],
        force_recompute=params.recompute,
        verbose=False)

    # Mirroring string
    if config['mirror']> 0:
        suffix = '-mirrored-{}'.format(config['mirror'])
    else:
        suffix = ''

    # Generate the configuration file to train the model
    feats = [
        "gradient-magnitude-s1.00{}.nrrd".format(suffix),
        "gradient-magnitude-s1.60{}.nrrd".format(suffix),
        "gradient-magnitude-s3.50{}.nrrd".format(suffix),
        "gradient-magnitude-s5.00{}.nrrd".format(suffix),
        "stensor-s0.5-r1.0{}.nrrd".format(suffix),
        "stensor-s0.8-r1.6{}.nrrd".format(suffix),
        "stensor-s1.8-r3.5{}.nrrd".format(suffix),
        "stensor-s2.5-r5.0{}.nrrd".format(suffix)
    ]
    if is_train:
        # Store the number of stumps in the model
        with open(folder_model + '/num_stumps.txt', 'w') as f:
            f.write(str(config['num_adaboost_stumps']))

        # Open template
        with open(dir_path + '/templates/train.cfg', 'r') as f:
            template = f.read()

        # Replace variables
        template = template.replace('VAR_NUM_STUMPS', '{}'.format(str(config['num_adaboost_stumps'])))
        template = template.replace('VAR_OUTPUT_PREFIX', '"' + folder_results + '/out"')
        template = template.replace('VAR_DATA', '"' + stack + '"')
        template = template.replace('VAR_LABELS', '"' + labels + '"')
        template = template.replace('VAR_ORIENTATION', '"' + folder_features + '/hessOrient-s3.5-repolarized{}.nrrd'.format(suffix) + '"')
        for i in range(len(feats)):
            feats[i] = '"' + folder_features + '/' + feats[i] + '"'
        feats_single = ', '.join(feats)
        template = template.replace('VAR_FEATURES', feats_single)

        if template.find('VAR_') >= 0:
            raise RuntimeError('Some field was not filled in in the template? ' + template)
    else:
        # Retrieve the number of stumps in the model
        try:
            with open(folder_model + '/num_stumps.txt', 'r') as f:
                num_stumps = int(f.read().strip())
            print(timestamp() + ' Using trained model with {} stumps'.format(num_stumps))
        except:
            num_stumps = 2000
            print(timestamp() + ' Could not read number of stumps: using {} per default'.format(num_stumps))
        
        # Open template
        with open(dir_path + '/templates/test.cfg', 'r') as f:
            template = f.read()

        # Replace variables
        template = template.replace('VAR_NUM_STUMPS', '{}'.format(str(num_stumps)))
        template = template.replace('VAR_OUTPUT_PREFIX', '"' + folder_results + '/out"')
        template = template.replace('VAR_MODEL', '"' + folder_model + '/stumps.cfg"')
        template = template.replace('VAR_DATA', '"' + stack + '"')
        template = template.replace('VAR_ORIENTATION', '"' + folder_features + '/hessOrient-s3.5-repolarized{}.nrrd'.format(suffix) + '"')
        for i in range(len(feats)):
            feats[i] = '"' + folder_features + '/' + feats[i] + '"'
        feats_single = ', '.join(feats)
        template = template.replace('VAR_FEATURES', feats_single)

        if template.find('VAR_') >= 0:
            raise RuntimeError('Some field was not filled in in the template? ' + template)


    tmp_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
    tmp_name = '/tmp/' + tmp_name
    print(timestamp() + ' Saving template to "{}"'.format(tmp_name))
    sys.stdout.flush()
    f = open(tmp_name, 'w')
    f.write(template)
    f.close()

    # Train model (if necessary) and get results
    print(timestamp() + ' Calling ccboost binary')
    sys.stdout.flush()
    cmd = dir_path + '/ccboost-v0.21/build/ccboost {}'.format(tmp_name)
    os.system(cmd)

    # TODO we should really try to catch an error here...
    # raise RuntimeError("stop")

    # Remove mirrored boundaries for training
    ccboost_res = folder_results + '/out-0-ab-max.nrrd'
    if config['mirror']> 0:
        print(timestamp() + ' Removing mirrored boundaries')
        print(timestamp() + ' File: {}'.format(ccboost_res))
        sys.stdout.flush()
        x, o = nrrd.read(ccboost_res)
        x = x[config['mirror']:-config['mirror'],
              config['mirror']:-config['mirror'],
              config['mirror']:-config['mirror']
              ]
        nrrd.write(ccboost_res, x, options=o)

    # Convert results to tif so we can see something (for debugging)
    r_tif = convert(ccboost_res, 'tif')
    # Convert to h5 so we can get them back to the server
    r_h5 = convert(ccboost_res, 'h5')
    print(timestamp() + ' Results stored in h5 at "{}"'.format(r_h5))
    sys.stdout.flush()

    # Delete the tiff files after processing
    # TODO (better to keep them for debugging, for now)

    # Move models to the right location
    if is_train:
        print(timestamp() + ' Trained models stored at "{}"'.format(folder_model))
        sys.stdout.flush()
        os.rename(folder_results + '/out-backupstumps.cfg', folder_model + '/backup-stumps.cfg')
        os.rename(folder_results + '/out-stumps.cfg', folder_model + '/stumps.cfg')

    # Clean up
    # os.remove(config['stack_tif'])
    # if os.path.isfile(stack):
    #     os.remove(stack)
    # if is_train:
    #     os.remove(config['labels_tif'])
    #     if os.path.isfile(labels):
    #         os.remove(labels)
