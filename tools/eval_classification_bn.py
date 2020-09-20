from __future__ import division
import logging
import os
import time
import math
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import adda
import shelve
import random

def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])

@click.command()
@click.argument('dataset_name')
@click.argument('split')
@click.argument('model')
@click.argument('weights', type=str)
@click.argument('seed', type=int)
@click.option('--task', default='count')
@click.option('--gpu', default='0')
def main(dataset_name, split, model, weights, seed, task, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    if dataset_name == 'mbm':
        ratio = 1000.
    elif dataset_name == 'dcc':
        ratio = 500.
    else:
        ratio = 100.

    model_fn = adda.models.get_model_fn(model)
    def helper(s, dataset_name=dataset_name, seed=seed):
        dataset = getattr(adda.data.get_dataset(dataset_name, seed, shuffle=False), s)
        if s == 'train':
            capacity = 2*len(dataset)
            batch_size = len(dataset)
        elif s == 'test':
            capacity = 2
            batch_size = 1
        else:
            raise ValueError('Incorrect split')
        im, gt = dataset.tf_ops(capacity=capacity)
        im, gt = adda.models.preprocessing(im, gt, dataset_name, training=False)
        im_batch, gt_batch = tf.train.batch(
            [im, gt], batch_size=batch_size)
        return im_batch, gt_batch

    im_train_batch, _ = helper('train')
    im_test_batch, density_map_batch = helper('test')
    im_batch = tf.concat([im_train_batch, im_test_batch], 0)

    net, _ = model_fn(im_batch, is_training=False, layers=4)
    with tf.variable_scope(task):
        net = tf.layers.conv2d(net, 1, 1, activation=tf.nn.relu)
        loss_pixel = tf.losses.mean_squared_error(
            tf.squeeze(density_map_batch*ratio , axis=0), 
            net[-1, :, :, 0])

        loss_pixel_sum = tf.losses.absolute_difference(
            tf.reduce_sum(density_map_batch), 
            tf.reduce_sum(net[-1, :, :, 0]/ratio))

    config = tf.ConfigProto(device_count=dict(GPU=1), gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    var_dict = adda.util.collect_vars(model)
    var_dict_classifier = adda.util.collect_vars(task)
    restorer = tf.train.Saver(var_list=var_dict.values()+var_dict_classifier.values())
    if os.path.isdir(weights):
        output = shelve.open(os.path.join(weights, 'result_{}_{}'.format(split, task)))
        weights = tf.train.latest_checkpoint(weights)
        # weights = os.path.join(weights, 'model-700')
    restorer.restore(sess, weights)
    logging.info('Evaluating weight {}'.format(weights))

    p = []
    l = []
    bar = tqdm(range(len(getattr(adda.data.get_dataset(dataset_name, seed, shuffle=False), split))))
    for _ in bar:
        outputs = sess.run(
            [loss_pixel, im_batch[-1, :, :, :], density_map_batch, 
            net[-1, :, :, 0], loss_pixel_sum])
        loss_val = [outputs[0],outputs[4]]
        p.append(outputs)
        l.append(loss_val)

    output['prediction'] = l
    output['results'] = p
    output.close()

    coord.request_stop()
    coord.join(threads)
    sess.close()

    logging.info('Evaluating split:{} with bn'.format(split))
    logging.info('pixel loss: {:.4f} sum loss: {:.4f}'.format(
        np.mean([i[0] for i in l]), np.mean([i[1] for i in l])))
    

if __name__ == '__main__':
    main()
