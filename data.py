import h5py
import os
import glob
import logging
import numpy as np
import tensorflow as tf
from config import config as cfg
logger = logging.getLogger('root')


data_path = {
    'vgg': 'data/VGG/VGG.hdf5',
    'dcc': 'data/DCC/DCC.hdf5',
    'mbm': 'data/MBM/MBM.hdf5',
    'adi': 'data/ADI/ADI.hdf5',
    'mbc': 'data/MBC/tfrecord', }

image_shape = {
    'vgg': (256, 256, 3),
    'dcc': (256, 256, 3),
    'mbm': (600, 600, 3),
    'adi': (152, 152, 3),
    'mbc': (512, 512), }

train_num = {
    'vgg': 64,
    'dcc': 100,
    'mbm': 15,
    'adi': 50,
    'mbc': 58, }

total_num = {
    'vgg': 200,
    'dcc': 176,
    'mbm': 44,
    'adi': 200,
    'mbc': 158, }


class Dataset(object):
    def __init__(self, dataset, seed):
        if dataset not in ['vgg', 'dcc', 'mbm', 'adi']:
            raise ValueError('Wrong dataset name: {}'.format(dataset))
        self.dataset = dataset
        self.image_shape = image_shape[dataset]
        self.seed = seed
        self.total_num = total_num[self.dataset]
        self.train_num = train_num[self.dataset]

        self.data = {}
        with h5py.File(data_path[dataset], 'r') as hf:
            self.data['imgs'] = hf.get('imgs')[()]
            self.data['counts'] = hf.get('counts')[()]
        self.label_shape = self.image_shape[:2] + (1,)

        imgs = self.data['imgs'].astype(np.float32)
        counts = self.data['counts'].astype(np.float32)[..., np.newaxis]

        imgs = imgs / 255.
        assert np.max(imgs) <= 1
        assert np.min(imgs) >= 0

        assert imgs.shape == (self.total_num,) + self.image_shape
        assert counts.shape == (self.total_num,) + self.label_shape

        np.random.seed(self.seed)
        ind = np.random.permutation(self.total_num)

        mn = np.mean(imgs[ind[:self.train_num], ...], axis=(0, 1, 2))
        std = np.std(imgs[ind[:self.train_num], ...], axis=(0, 1, 2))
        self.train = (imgs[ind[:self.train_num], ...] - mn) / \
            std, counts[ind[:self.train_num], ...]
        self.test = (imgs[ind[self.train_num:], ...] - mn) / \
            std, counts[ind[self.train_num:], ...]

    def preprocessing(self, training, augment, batch_size, num_epochs):
        def _augment(imgs, labels):
            inputs = tf.concat([imgs, labels], -1)
            assert inputs.get_shape().ndims == 3
            if self.image_shape == (600, 600, 3):
                inputs = tf.image.random_crop(inputs, [576, 576, 4])
            elif self.image_shape == (152, 152, 3):
                inputs = tf.image.random_crop(inputs, [144, 144, 4])
            elif self.image_shape == (256, 256, 3):
                inputs = tf.image.random_crop(inputs, [224, 224, 4])
            else:
                raise ValueError('Incorrect dataset')
            inputs = tf.image.random_flip_left_right(inputs)
            inputs = tf.image.random_flip_up_down(inputs)
            inputs = tf.image.rot90(
                inputs, k=tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32))
            return inputs[:, :, :3], inputs[:, :, 3:]

        if training:
            dataset = tf.data.Dataset.from_tensor_slices(self.train)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.test)

        if augment:
            dataset = dataset.map(
                _augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=self.train_num)

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_iterator = dataset.make_one_shot_iterator()
        imgs, labels = dataset_iterator.get_next()
        return imgs, labels


class Dataset3D(object):
    def __init__(self, dataset, seed, training):
        self.dataset = dataset
        self.image_shape = image_shape[dataset]
        self.seed = seed
        self.total_num = total_num[self.dataset]
        self.train_num = train_num[self.dataset]
        self.training = training

        record_files = glob.glob(os.path.join(data_path[dataset], '*'))
        assert len(record_files) == self.total_num

        np.random.seed(self.seed)
        ind = np.random.permutation(self.total_num)
        ind = ind[:self.train_num] if self.training else ind[self.train_num:]
        record_files = [record_files[i] for i in ind]

        if self.training:
            assert len(record_files) == self.train_num
            self.data = tf.data.TFRecordDataset(
                record_files,
                buffer_size=100 * 1024 * 1024,  # 100 MiB per file
                num_parallel_reads=tf.data.experimental.AUTOTUNE)
        else:
            assert len(record_files) == self.total_num - self.train_num
            self.data = tf.data.TFRecordDataset(record_files)

    def preprocessing(self, augment, batch_size, num_epochs):
        def _parse(example_proto):
            # Create a dictionary describing the features.
            image_feature_description = {
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
                'depth': tf.io.FixedLenFeature([], tf.int64),
            }
            # Parse the input tf.Example proto using the dictionary above.
            parsed = tf.io.parse_single_example(
                example_proto, image_feature_description)
            image = tf.decode_raw(parsed['image_raw'], tf.uint8)
            image = tf.reshape(
                image, (self.image_shape[0], self.image_shape[1], parsed['depth']))
            image = tf.cast(image, tf.float32)
            image = tf.image.per_image_standardization(image)

            label = tf.decode_raw(parsed['label'], tf.float32)
            label = tf.reshape(
                label, (self.image_shape[0], self.image_shape[1], parsed['depth']))
            return image, label

        def _rand_flip(data, dim):
            assert data.get_shape().ndims == 4
            assert dim in [0, 1, 2]
            flip_flag = tf.random_uniform([]) > 0.5
            return tf.cond(flip_flag, lambda: tf.reverse(data, axis=[dim]), lambda: data)

        def _rand_rot(data):
            assert data.get_shape().ndims == 4
            data = tf.transpose(data, perm=[2, 0, 1, 3])
            data = tf.image.rot90(
                data, k=tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32))
            return tf.transpose(data, perm=[1, 2, 0, 3])

        def _pad(image, label):
            data = tf.stack([image, label], axis=-1)
            assert data.get_shape().ndims == 4
            depth = tf.shape(data)[2]
            target_depth = cfg.TRAIN.PATCH_DEPTH * \
                tf.cast(tf.math.ceil(tf.math.truediv(
                    depth, cfg.TRAIN.PATCH_DEPTH)), tf.int32)
            depth_diff = tf.maximum(target_depth - depth, 0)
            paddings = tf.reshape(
                tf.stack([
                    0, 0, 0, 0, depth_diff // 2,
                    depth_diff - depth_diff // 2, 0, 0
                ]), [4, 2])
            data = tf.pad(data, paddings)
            data.set_shape(
                [512, 512, None, 2])
            return data

        def _augment(data):
            assert data.get_shape().ndims == 4
            data = tf.image.random_crop(
                data, [cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 2])
            data.set_shape(
                [cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 2])

            # flip xyz
            data = _rand_flip(data, 0)
            data = _rand_flip(data, 1)
            data = _rand_flip(data, 2)

            data = _rand_rot(data)
            image, label = tf.unstack(data, num=2, axis=-1)
            return image, label

        def _tile(data):
            assert data.get_shape().ndims == 4

            tile_size = (cfg.TRAIN.PATCH_SIZE,
                         cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH)
            data_shape = tf.shape(data)
            data = tf.transpose(
                tf.reshape(data, [data_shape[0] // tile_size[0], tile_size[0],
                                  data_shape[1] // tile_size[1], tile_size[1],
                                  data_shape[2] // tile_size[2], tile_size[2], data_shape[3]]),
                [0, 2, 4, 1, 3, 5, 6])
            data = tf.reshape(
                data, [-1, tile_size[0], tile_size[1], tile_size[2], data_shape[3]])

            image, label = tf.unstack(data, num=2, axis=-1)
            return image, label

        def _normalize(image, label):
            image = tf.image.per_image_standardization(image)
            image = tf.expand_dims(image, axis=-1)
            label = tf.expand_dims(label, axis=-1)
            return image, label

        self.data = self.data.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        self.data = self.data.map(
            _parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.data = self.data.map(
            _pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # comment this if you run out of CPU memory to store the data
        self.data = self.data.cache()
        if self.training:
            self.data = self.data.shuffle(buffer_size=self.train_num)

        self.data = self.data.repeat(num_epochs)
        if augment:
            self.data = self.data.map(
                _augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            self.data = self.data.map(
                _tile, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.data = self.data.map(
            _normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if augment:
            self.data = self.data.batch(batch_size)
        self.data = self.data.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        dataset_iterator = self.data.make_one_shot_iterator()
        imgs, labels = dataset_iterator.get_next()
        return imgs, labels
