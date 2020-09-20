import os
import argparse
import tensorflow as tf
import numpy as np
import data
import model
import utils
from config import config as cfg
from config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)


def train(tf_config, logger):
    dataset = data.Dataset3D(cfg.DATASET, cfg.RNG_SEED, training=True)
    imgs, labels = dataset.preprocessing(
        augment=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=cfg.TRAIN.EPOCH)

    net, _ = model.unet_3d(imgs, bn_training=True, layers=4, features_root=32,
                           dropout_training=True, dataset=cfg.DATASET)
    with tf.variable_scope('cls'):
        net = tf.layers.conv3d(net, 1, 1, activation=tf.nn.relu)
    loss_pixel = tf.losses.mean_squared_error(
        labels * cfg.MODEL.RATIO[cfg.DATASET], net)
    loss_pixel_sum = tf.losses.absolute_difference(
        tf.reduce_sum(labels, axis=[1, 2, 3, 4]),
        tf.reduce_sum(net / cfg.MODEL.RATIO[cfg.DATASET], axis=[1, 2, 3, 4]))

    lr_decayed = tf.train.cosine_decay_restarts(
        cfg.SOLVER.BASE_LR, tf.train.get_or_create_global_step(), cfg.SOLVER.RESTART_STEP)
    wd = cfg.SOLVER.WEIGHT_DECAY * lr_decayed / cfg.SOLVER.BASE_LR
    optimizer = tf.contrib.opt.AdamWOptimizer(wd, learning_rate=lr_decayed)
    step_pixel = optimizer.minimize(
        loss_pixel, global_step=tf.train.get_or_create_global_step())

    tf.summary.scalar('per_pixel_mse', loss_pixel)
    tf.summary.scalar('sum_mae', loss_pixel_sum)
    merged = tf.summary.merge_all()
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


def test(tf_config, logger):
    cfg.TRAIN.BATCH_SIZE = 1
    dataset = data.Dataset3D(cfg.DATASET, cfg.RNG_SEED, training=False)
    imgs, labels = dataset.preprocessing(
        augment=False, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=-1)
    batch_size = tf.shape(imgs)[0]
    imgs_1, labels_1 = imgs[:batch_size // 2, :, :,
                            :, :], labels[:batch_size // 2, :, :, :, :]

    dataset = data.Dataset3D(cfg.DATASET, cfg.RNG_SEED, training=False)
    imgs, labels = dataset.preprocessing(
        augment=False, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=-1)
    batch_size = tf.shape(imgs)[0]
    imgs_2, labels_2 = imgs[batch_size // 2:, :, :,
                            :, :], labels[batch_size // 2:, :, :, :, :]

    def _predict(imgs, labels, reuse=False):
        net, _ = model.unet_3d(imgs, bn_training=False, layers=4, features_root=32,
                               dropout_training=False, dataset=cfg.DATASET, reuse=reuse)
        with tf.variable_scope('cls', reuse=reuse):
            net = tf.layers.conv3d(net, 1, 1, activation=tf.nn.relu)
        pred_sum = tf.reduce_sum(
            net / cfg.MODEL.RATIO[cfg.DATASET], axis=[0, 1, 2, 3, 4])
        label_sum = tf.reduce_sum(labels, axis=[0, 1, 2, 3, 4])
        return pred_sum, label_sum, net

    pred_sum_1, label_sum_1, net_1 = _predict(imgs_1, labels_1)
    pred_sum_2, label_sum_2, net_2 = _predict(imgs_2, labels_2, reuse=True)

    summary_writer = tf.summary.FileWriter(
        os.path.join(cfg.OUTPUT_DIR, 'test'))
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        weights_path = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
        logger.info('Restoring weights from {}'.format(weights_path))
        saver.restore(sess, weights_path)

        loss_sum_aggregate = 0.
        step_num = (dataset.total_num -
                    dataset.train_num) // cfg.TRAIN.BATCH_SIZE

        pred_val_density = []
        for i in range(step_num):
            pred_sum_1_val, label_sum_1_val, net_1_val = sess.run(
                [pred_sum_1, label_sum_1, net_1])
            pred_sum_2_val, label_sum_2_val, net_2_val = sess.run(
                [pred_sum_2, label_sum_2, net_2])
            loss_sum_val = abs(
                pred_sum_1_val + pred_sum_2_val - label_sum_1_val - label_sum_2_val)
            net_val = np.concatenate((net_1_val, net_2_val), axis=0)

            pred_val_density.append(net_val / cfg.MODEL.RATIO[cfg.DATASET])
            loss_sum_aggregate += loss_sum_val
            logger.info('#{} sum:{:6.3f}'.format(i, loss_sum_val))

        logger.info('Total Sum:{:6.3f}'.format(
            loss_sum_aggregate / step_num))
        summary = tf.Summary()
        summary.value.add(
            tag='sum_mae', simple_value=loss_sum_aggregate / step_num)
        summary_writer.add_summary(
            summary, global_step=int(weights_path.split('-')[-1]))

    utils.save_results(
        os.path.join(cfg.OUTPUT_DIR,
                     'result_seed_{}.hdf5'.format(cfg.RNG_SEED)),
        {'density': pred_val_density})
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
    test(tf_config, logger)


if __name__ == '__main__':
    tf.app.run()
