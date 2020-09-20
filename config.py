# Based on:
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from utils import AttrDict
logger = logging.getLogger('root')


__C = AttrDict()
config = __C

__C.DATASET = b''
__C.OUTPUT_DIR = b''
__C.GPU_ID = 0
__C.RNG_SEED = 2
__C.LOG_PERIOD = 10
__C.SELF_ATTN = 0

# Training options
__C.TRAIN = AttrDict()
__C.TRAIN.EPOCH = -1
__C.TRAIN.STEP = 0
__C.TRAIN.BATCH_SIZE = 64
# for 3D data
__C.TRAIN.PATCH_SIZE = 512
__C.TRAIN.PATCH_DEPTH = 80

# Train model options
__C.MODEL = AttrDict()
__C.MODEL.BN_MOMENTUM = 0.9
__C.MODEL.RATIO = {'mbm': 1000, 'dcc': 500,
                   'adi': 100, 'vgg': 100, 'mbc': 1000}

# Solver
__C.SOLVER = AttrDict()
__C.SOLVER.BASE_LR = 1e-3
__C.SOLVER.WEIGHT_DECAY = 1e-3
__C.SOLVER.RESTART_STEP = 50


def print_cfg():
    import pprint
    logger.info('Training with config:')
    logger.info(pprint.pformat(__C))


def assert_and_infer_cfg():
    if __C.DATASET not in ['vgg', 'adi', 'mbm', 'dcc', 'mbc']:
        raise ValueError('Incorrect Dataset {}'.format(__C.DATASET))


def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # the types must match, too
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
            raise ValueError(
                'Type mismatch ({} vs. {}) for config key: {}'.format(
                    type(dict_b[key]), type(value), key)
            )
        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    merge_dicts(yaml_config, __C)


def cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val
