import logging
import os
import random
import math
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import adda

@click.command()
@click.argument('dataset_name')
@click.argument('split')
@click.argument('model')
@click.argument('output_dir')
@click.argument('seed', type=int)
@click.option('--task', default='count')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--weights')
@click.option('--solver', default='adamw')
def main(dataset_name, split, model, output_dir, seed, task, gpu, iterations, batch_size, display,
         lr, stepsize, snapshot, weights, solver):
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

    dataset = getattr(adda.data.get_dataset(dataset_name, seed, shuffle=True), split)
    model_fn = adda.models.get_model_fn(model)
    im, density_map = dataset.tf_ops(capacity=10*batch_size)
    im, density_map = adda.models.preprocessing(im, density_map, dataset_name, training=True)
    im_batch, density_map_batch = tf.train.batch(
        [im, density_map],
        num_threads=8,
        batch_size=batch_size,
        capacity=5*batch_size)

    net, _ = model_fn(im_batch, is_training=True, layers=4)
    with tf.variable_scope(task):
        net = tf.layers.conv2d(net, 1, 1, activation=tf.nn.relu)
        loss_pixel = tf.losses.mean_squared_error(
            density_map_batch*ratio, 
            tf.squeeze(net, axis=-1))

        loss_pixel_sum = tf.losses.absolute_difference(
            tf.reduce_sum(density_map_batch, axis=[1, 2]), 
            tf.reduce_sum(net/ratio, axis=[1, 2, 3]))

    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    lr_decayed = tf.train.cosine_decay_restarts(lr_var, 
        tf.train.get_or_create_global_step(), 50, alpha=0.0)
    wd = 1e-3 * lr_decayed / lr_var
    if solver == 'sgdw':
        optimizer = tf.contrib.opt.MomentumWOptimizer(
            learning_rate=lr_decayed,
            weight_decay=wd,
            momentum=0.9,
            use_nesterov=True)
    elif solver == 'adam':
        optimizer = tf.train.AdamOptimizer(lr_decayed)
    elif solver == 'adamw':
        optimizer = tf.contrib.opt.AdamWOptimizer(wd, learning_rate=lr_decayed)
    else: 
        raise ValueError('Incorrect optimizer')
    step_pixel = optimizer.minimize(loss_pixel, global_step=tf.train.get_or_create_global_step())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    step = tf.group([step_pixel, update_ops])

    config = tf.ConfigProto(device_count=dict(GPU=1), gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    
    var_dict = adda.util.collect_vars(model)
    var_dict_classifier = adda.util.collect_vars(task)
    saver = tf.train.Saver(
        var_list=var_dict.values()+var_dict_classifier.values()+[tf.train.get_or_create_global_step()])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        if tf.train.latest_checkpoint(output_dir) is not None:
            weights = output_dir

    if weights:            
        logging.info('Restoring weights from {}:'.format(weights))
        for tgt in var_dict.values()+var_dict_classifier.values():
            logging.info(' {:30}'.format(tgt.name))
        if os.path.isdir(weights):
            weights = tf.train.latest_checkpoint(weights)
        saver.restore(sess, weights)

    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output_dir, lr))
    for i in bar:
        loss_pixel_val, loss_sum_val, _= sess.run(
            [loss_pixel, loss_pixel_sum, step])
        if i % display == 0 or i == iterations -1:
            logging.info('{:6} per_pixel:{:6.3f}   sum:{:6.3f}'
                        .format('Step {}:'.format(i),
                                loss_pixel_val,
                                loss_sum_val))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output_dir, lr))
        if (i + 1) % snapshot == 0:
            snapshot_path = saver.save(sess, os.path.join(output_dir, 'model'),
                                       global_step=sess.run(tf.train.get_or_create_global_step()))
            logging.info('Saved snapshot to {}'.format(snapshot_path))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
