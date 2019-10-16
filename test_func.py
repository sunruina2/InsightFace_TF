''' 测试resnet网络 '''

import collections
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers.python.layers import utils
from tensorlayer.layers import Layer, list_remove_repeat


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    ```
        https://www.tensorflow.org/api_docs/python/tf/cond
        If x < y, the tf.add operation will be executed and tf.square operation will not be executed.
        Since z is needed for at least one branch of the cond, the tf.multiply operation is always executed, unconditionally.
    ```

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    dtype : tf.float32 (default) or tf.float16
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_

    """

    def __init__(
            self,
            layer=None,
            decay=0.9,
            epsilon=2e-5,
            act=tf.identity,
            is_train=False,
            fix_gamma=True,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),  # tf.ones_initializer,
            # dtype = tf.float32,
            trainable=None,
            name='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (
        self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=tf.float32,
                                   trainable=is_train)  # , restore=restore)

            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                initializer=gamma_init,
                dtype=tf.float32,
                trainable=fix_gamma,
            )  # restore=restore)

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init, dtype=tf.float32,
                                          trainable=False)  # restore=restore)
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=tf.float32,
                trainable=False,
            )  # restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay,
                                                                           zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)  # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if trainable:
                mean, var = mean_var_with_update()
                print(mean)
                print(var)
                self.outputs = act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            else:
                self.outputs = act(
                    tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))
            variables = [beta, gamma, moving_mean, moving_variance]
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return tl.layers.MaxPool2d(inputs, [1, 1], strides=(factor, factor), name=scope)


class ElementwiseLayer(Layer):
    """
    The :class:`ElementwiseLayer` class combines multiple :class:`Layer` which have the same output shapes by a given elemwise-wise operation.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    combine_fn : a TensorFlow elemwise-merge function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`_ .
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=[],
            combine_fn=tf.minimum,
            name='elementwise_layer',
            act=None,
    ):
        Layer.__init__(self, name=name)

        if act:
            print("  [TL] ElementwiseLayer %s: size:%s fn:%s, act:%s" % (
                self.name, layer[0].outputs.get_shape(), combine_fn.__name__, act.__name__))
        else:
            print("  [TL] ElementwiseLayer %s: size:%s fn:%s" % (
                self.name, layer[0].outputs.get_shape(), combine_fn.__name__))

        self.outputs = layer[0].outputs
        # print(self.outputs._shape, type(self.outputs._shape))
        for l in layer[1:]:
            # assert str(self.outputs.get_shape()) == str(l.outputs.get_shape()), "Hint: the input shapes should be the same. %s != %s" %  (self.outputs.get_shape() , str(l.outputs.get_shape()))
            self.outputs = combine_fn(self.outputs, l.outputs, name=name)
        if act:
            self.outputs = act(self.outputs)
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


def conv2d_same(inputs, num_outputs, kernel_size, strides, rate=1, w_init=None, scope=None, trainable=None):
    '''
    Reference slim resnet
    :param inputs:
    :param num_outputs:
    :param kernel_size:
    :param strides:
    :param rate:
    :param scope:
    :return:
    '''
    if strides == 1:
        if rate == 1:
            nets = tl.layers.Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), W_init=w_init, act=None, padding='SAME', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable,
                                  name=scope + '_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable,
                                  name=scope + '_bn/BatchNorm')
        return nets
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tl.layers.PadLayer(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]],
                                    name='padding_%s' % scope)
        if rate == 1:
            nets = tl.layers.Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), W_init=w_init, act=None, padding='VALID', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable,
                                  name=scope + '_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               b_init=None,
                                               rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable,
                                  name=scope + '_bn/BatchNorm')
        return nets


def bottleneck_IR(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = tl.layers.Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv', use_cudnn_on_gpu=True)
            shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=True, trainable=trainable,
                                      name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = BatchNormLayer(inputs, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn1')
        residual = tl.layers.Conv2d(residual, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None,
                                    b_init=None,
                                    W_init=w_init, name='conv1', use_cudnn_on_gpu=True)
        residual = BatchNormLayer(residual, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn2')
        # bottleneck prelu
        residual = tl.layers.PReluLayer(residual)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2',
                               trainable=trainable)
        output = ElementwiseLayer(layer=[shortcut, residual],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=None)
        return output


a_class = collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])

base_depth = 64
num_units = 3
stride = 2
rate = 1
a_obj = a_class('block1', bottleneck_IR, [{
    'depth': base_depth * 4,
    'depth_bottleneck': base_depth,
    'stride': stride,
    'rate': rate
}] + [{
    'depth': base_depth * 4,
    'depth_bottleneck': base_depth,
    'stride': 1,
    'rate': rate
}] * (num_units - 1))
print(a_obj.scope)
print(a_obj.unit_fn)
print(a_obj.args)

# a_obj.scope >> block1
# a_obj.unit_fn >> <function bottleneck_IR>
# a_obj.args >>  [{'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1},
#                 {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1},
#                 {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}]

print(tf.variable_scope('black1', 'bottleneck_v1').default_name)