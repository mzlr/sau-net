import os
import argparse
import tensorflow as tf
import data
import model
import utils

from config import config as cfg
from config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)


def train(tf_config, logger):
    dataset = data.Dataset(cfg.DATASET, cfg.RNG_SEED)
    imgs, labels = dataset.preprocessing(
        training=True, augment=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=cfg.TRAIN.EPOCH)

    net, _ = model.unet(imgs, bn_training=True,
                        dropout_training=True, dataset=cfg.DATASET)
    with tf.variable_scope('cls'):
        net = tf.layers.conv2d(net, 1, 1, activation=tf.nn.relu)
    loss_pixel = tf.losses.mean_squared_error(
        labels * cfg.MODEL.RATIO[cfg.DATASET], net)
    loss_pixel_sum = tf.losses.absolute_difference(
        tf.reduce_sum(labels, axis=[1, 2, 3]),
        tf.reduce_sum(net / cfg.MODEL.RATIO[cfg.DATASET], axis=[1, 2, 3]))

    lr_decayed = tf.train.cosine_decay_restarts(
        cfg.SOLVER.BASE_LR, tf.train.get_or_create_global_step(), cfg.SOLVER.RESTART_STEP)
    wd = cfg.SOLVER.WEIGHT_DECAY * lr_decayed / cfg.SOLVER.BASE_LR
    optimizer = tf.contrib.opt.AdamWOptimizer(wd, learning_rate=lr_decayed)
    step_pixel = optimizer.minimize(
        loss_pixel, global_step=tf.train.get_or_create_global_step())

    tf.summary.scalar('per_pixel_mse', loss_pixel)
    tf.summary.scalar('sum_mae', loss_pixel_sum)
    merged = tf.summary.merge_all()
    # this step is optional since our bn extension does not need this
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    step = tf.group([step_pixel, update_ops])
    saver = tf.train.Saver(max_to_keep=1000)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)

    with tf.Session(config=tf_config) as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(cfg.OUTPUT_DIR, 'train'), sess.graph)
        if tf.train.latest_checkpoint(cfg.OUTPUT_DIR) is None:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            logger.info('Saving path is {}'.format(cfg.OUTPUT_DIR))
        else:
            weights_path = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
            start_step = int(weights_path.split('-')[-1])
            tf.train.Saver().restore(sess, weights_path)
            logger.info('Restoring weights from {}'.format(weights_path))
        logger.info('Training at Step {}'.format(start_step + 1))

        for i in range(start_step, cfg.TRAIN.STEP):
            if i % cfg.LOG_PERIOD == 0 or i == cfg.TRAIN.STEP - 1:
                loss_pixel_val, loss_sum_val, summary, _ = sess.run(
                    [loss_pixel, loss_pixel_sum, merged, step])
                summary_writer.add_summary(summary, i + 1)
                logger.info('Step:{}/{} per_pixel:{:6.3f}  sum:{:6.3f}'.format(
                    i + 1, cfg.TRAIN.STEP, loss_pixel_val, loss_sum_val))
            else:
                sess.run([step])
            if i == cfg.TRAIN.STEP - 1:
                weights_path = saver.save(
                    sess, os.path.join(cfg.OUTPUT_DIR, 'model'), global_step=i + 1)
                logger.info('Saving weights to {}'.format(weights_path))
    tf.reset_default_graph()


def bn_update(tf_config, logger):
    dataset = data.Dataset(cfg.DATASET, cfg.RNG_SEED)
    cfg.MODEL.BN_MOMENTUM = 0.
    assert cfg.MODEL.BN_MOMENTUM == 0., 'BN_MOMENTUM should be 0. for update step'
    imgs, _ = dataset.preprocessing(
        training=True, augment=False, batch_size=dataset.train_num, num_epochs=1)

    net, _ = model.unet(imgs, bn_training=True,
                        dropout_training=False, dataset=cfg.DATASET)
    with tf.variable_scope('cls'):
        _ = tf.layers.conv2d(net, 1, 1, activation=tf.nn.relu)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    saver = tf.train.Saver(max_to_keep=1000)
    with tf.Session(config=tf_config) as sess:
        weights_path = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
        logger.info('Restoring weights from {}'.format(weights_path))
        saver.restore(sess, weights_path)

        sess.run(update_ops)

        weights_path = saver.save(
            sess, os.path.join(cfg.OUTPUT_DIR, 'bn-model'),
            global_step=int(weights_path.split('-')[-1]))
        logger.info('Updating weights to {}'.format(weights_path))
    tf.reset_default_graph()


def test(tf_config, logger):
    dataset = data.Dataset(cfg.DATASET, cfg.RNG_SEED)
    imgs, labels = dataset.preprocessing(
        training=False, augment=False,
        batch_size=dataset.total_num - dataset.train_num, num_epochs=1)

    net, _ = model.unet(imgs, bn_training=False,
                        dropout_training=False, dataset=cfg.DATASET)
    with tf.variable_scope('cls'):
        net = tf.layers.conv2d(net, 1, 1, activation=tf.nn.relu)
    loss_pixel_sum = tf.losses.absolute_difference(
        tf.reduce_sum(labels, axis=[1, 2, 3]),
        tf.reduce_sum(net / cfg.MODEL.RATIO[cfg.DATASET], axis=[1, 2, 3]))

    summary_writer_bn = tf.summary.FileWriter(
        os.path.join(cfg.OUTPUT_DIR, 'test_bn'))
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        weights_path = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
        assert 'bn-model' in weights_path, 'check weights path, make sure BN extension is applied'

        logger.info('Restoring weights from {}'.format(weights_path))
        saver.restore(sess, weights_path)

        pred_val, loss_sum_val = sess.run([net, loss_pixel_sum])
        logger.info('sum:{:10.3f}'.format(loss_sum_val))

        summary = tf.Summary()
        summary.value.add(tag='sum_mae', simple_value=loss_sum_val)
        summary_writer_bn.add_summary(
            summary, global_step=int(weights_path.split('-')[-1]))

    utils.save_results(
        os.path.join(cfg.OUTPUT_DIR,
                     'result_seed_{}.hdf5'.format(cfg.RNG_SEED)),
        {'density': pred_val / cfg.MODEL.RATIO[cfg.DATASET]})
    tf.reset_default_graph()


def main(_):
    parser = argparse.ArgumentParser(
        description='Classification model training')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    assert_and_infer_cfg()
    print_cfg()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
    logger = utils.setup_custom_logger('root')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tf_config = tf.ConfigProto(device_count=dict(
        GPU=1), gpu_options=tf.GPUOptions(allow_growth=True))
    tf.enable_resource_variables()

    train(tf_config, logger)
    bn_update(tf_config, logger)
    test(tf_config, logger)


if __name__ == '__main__':
    tf.app.run()
