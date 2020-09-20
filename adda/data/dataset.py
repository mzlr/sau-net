import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import FeedingQueueRunner


class ImageDataset(object):

    def __init__(self, images, labels, image_shape=None, label_shape=None,
                 shuffle=True):
        self.images = images
        self.labels = labels
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.shuffle = shuffle

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        inds = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(inds)
        for ind in inds:
            yield self.images[ind], self.labels[ind]

    def feed(self, im, label, epochs=None):
        epochs_elapsed = 0
        while epochs is None or epochs_elapsed < epochs:
            for entry in self:
                yield {im: entry[0], label: entry[1]}
            epochs_elapsed += 1

    def tf_ops(self, capacity=32):
        im = tf.placeholder(tf.float32, shape=self.image_shape)
        label = tf.placeholder(tf.float32, shape=self.label_shape)
        if self.image_shape is None or self.label_shape is None:
            shapes = None
        else:
            shapes = [self.image_shape, self.label_shape]
        queue = tf.FIFOQueue(capacity, [tf.float32, tf.float32], shapes=shapes)
        enqueue_op = queue.enqueue([im, label])
        fqr = FeedingQueueRunner(queue, [enqueue_op],
                                 feed_fns=[self.feed(im, label).next])
        tf.train.add_queue_runner(fqr)
        return queue.dequeue()


datasets = {}


def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def get_dataset(name, *args, **kwargs):
    return datasets[name](*args, **kwargs)
