from collections import OrderedDict

import tensorflow as tf
from adda.models import register_model_fn

tf.logging.set_verbosity(tf.logging.INFO)


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)


def conv2d(x, features, kernel_size, is_bn):
    conv_2d = tf.layers.conv2d(x, features, kernel_size, 
        use_bias=False, activation=tf.nn.relu, padding='same')
    conv_2d = tf.layers.batch_normalization(conv_2d, training=is_bn, renorm=False)
    return conv_2d


def deconv2d(x, features, kernel_size, stride, is_bn):
    deconv_2d = tf.layers.conv2d_transpose(x, features, kernel_size, [stride, stride], 
        use_bias=False, activation=tf.nn.relu)
    return tf.layers.batch_normalization(deconv_2d, training=is_bn, renorm=False)


def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        return tf.concat([x1, x2], 3)


def self_attn(x, is_bn, name='self_attn'):
    with tf.variable_scope(name):
        batch_num, height, width, channel = x.shape
        dim_inner = channel / 2

        max_pool = tf.layers.max_pooling2d(x, 2, 2)

        theta = tf.layers.conv2d(x, dim_inner, 1, use_bias=True, activation=None)
        phi = tf.layers.conv2d(max_pool, dim_inner, 1, use_bias=True, activation=None)
        g = tf.layers.conv2d(max_pool, dim_inner, 1, use_bias=True, activation=None)

        theta = tf.reshape(theta, [batch_num, -1, dim_inner])
        phi = tf.reshape(phi, [batch_num, -1, dim_inner])
        phi = tf.transpose(phi, perm=[0, 2, 1])
        g = tf.reshape(g, [batch_num, -1, dim_inner])

        theta_phi = tf.matmul(theta, phi)
        t = tf.matmul(tf.nn.softmax(theta_phi, axis=2), g)

        t = tf.reshape(t, [batch_num, height, width, dim_inner])
        t = tf.layers.conv2d(t, 1, 1, use_bias=False, activation=tf.nn.relu)
        t = tf.layers.batch_normalization(t, training=is_bn)
        return t
        # t = tf.layers.conv2d(t, 1, 1, use_bias=True, activation=tf.nn.relu)
        # t = tf.layers.batch_normalization(t, training=is_bn)
        # return t


@register_model_fn('unet')
def unet(in_node, is_training=True, layers=4, features_root=32, is_bn=True,
         filter_size=3, pool_size=2, scope='unet'):
    """
    Creates a new convolutional unet for the given parametrization.
    :param in_node: input tensor, shape [?,nx,ny,channels]
    :param is_training: whether training
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param scope: model scope
    """

    tf.logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    pools = OrderedDict()
    dw_h_convs = OrderedDict()

    with tf.variable_scope(scope):
        # down layers
        for layer in range(0, layers):
            with tf.name_scope("down_conv_{}".format(str(layer))):
                features = 2 ** layer * features_root

                conv1 = conv2d(in_node, features, filter_size, is_bn)
                dw_h_convs[layer] = conv2d(conv1, features, filter_size, is_bn)

                if layer == layers - 1:
                    dw_h_convs[layer] = self_attn(dw_h_convs[layer], is_bn)

                if layer > layers - 3:
                    dw_h_convs[layer] = tf.layers.dropout(dw_h_convs[layer], training=is_training)

                if layer < layers - 1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            with tf.name_scope("up_conv_{}".format(str(layer))):
                features = 2 ** layer * features_root

                h_deconv = deconv2d(in_node, features, pool_size, pool_size, is_bn)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)

                conv1 = conv2d(h_deconv_concat, features, filter_size, is_bn)
                in_node = conv2d(conv1, features, filter_size, is_bn)

    return in_node, dw_h_convs[layers - 1]


unet.default_image_size = 240
unet.num_channels = 1
unet.mean = None
unet.bgr = False
