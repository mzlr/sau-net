import logging
import os
from collections import OrderedDict
import tensorflow as tf
import h5py


class AttrDict(dict):
    """ subclass dict and define getter-setter. This behaves as both dict and obj"""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def setup_custom_logger(name):
    formatter = logging.Formatter(
        '[%(levelname)s: %(filename)s:%(lineno)d]: %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])


def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    # Using None for one of the slice parameters is the same as omitting it.
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict


def save_results(file_path, file_dict):
    with h5py.File(file_path, 'w') as hf:
        for k in file_dict:
            g = hf.create_group(k)
            for i, item in enumerate(file_dict[k]):
                g.create_dataset(str(i), data=item)
