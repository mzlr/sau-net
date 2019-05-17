import logging
import math

import numpy as np
import tensorflow as tf


models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        # set default parameters
        fn.range = None
        fn.mean = None
        fn.bgr = False
        return fn
    return decorator

def get_model_fn(name):
    return models[name]

def preprocessing(inputs, labels, dataset_name, training=False):

    if training:
        labels = tf.expand_dims(labels, -1)
        inputs = tf.concat([inputs, labels], -1)

        if dataset_name == 'mbm':
            inputs = tf.image.random_crop(inputs, [520, 520, 4])
            # inputs = tf.image.random_crop(inputs, [168, 168, 4])
        elif dataset_name == 'adi':
            inputs = tf.image.random_crop(inputs, [128, 128, 4])
        elif dataset_name == 'vgg' or dataset_name == 'dcc':
            inputs = tf.image.random_crop(inputs, [224, 224, 4])
        else:
            raise ValueError('Incorrect dataset')

        inputs = tf.image.random_flip_left_right(inputs)
        inputs = tf.image.random_flip_up_down(inputs)
        inputs = tf.image.rot90(inputs, k=tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32))

        inputs, labels = inputs[:, :, :3], inputs[:, :, 3]
       
    return inputs, labels

