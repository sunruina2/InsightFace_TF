import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers.python.layers import utils
import collections
from tensorlayer.layers import Layer, list_remove_repeat


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
            combine_fn=tf.minimum,  # 默认是取对应元素最小值
            name='elementwise_layer',
            act=None,
    ):
        Layer.__init__(self, name=name)  # [shortcut, residual]

        if act:
            print("  [TL] ElementwiseLayer %s: size:%s fn:%s, act:%s" % (
                self.name, layer[0].outputs.get_shape(), combine_fn.__name__, act.__name__))
        else:
            print("  [TL] ElementwiseLayer %s: size:%s fn:%s" % (
                self.name, layer[0].outputs.get_shape(),
                combine_fn.__name__))  # ElementwiseLayer resnet_v1_50/block1/unit_1/bottleneck_v1/combine_layer: size:(?, 56, 56, 256) fn:add

        self.outputs = layer[0].outputs  # shortcut (?, 56, 56, 256)
        # print(self.outputs._shape, type(self.outputs._shape))
        for l in layer[1:]:  # residual (?, 56, 56, 256)
            # assert str(self.outputs.get_shape()) == str(l.outputs.get_shape()), "Hint: the input shapes should be the same. %s != %s" %  (self.outputs.get_shape() , str(l.outputs.get_shape()))
            self.outputs = combine_fn(self.outputs, l.outputs,
                                      name=name)  # shortcut (?, 56, 56, 256) + residual (?, 56, 56, 256)
        if act:
            self.outputs = act(self.outputs)
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(
            self.all_layers)  # 剔除掉重复的layers，resnet_v1_50/conv1，resnet_v1_50/bn0，resnet_v1_50/prelu0
        self.all_params = list_remove_repeat(self.all_params)


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
            # https://www.cnblogs.com/ranjiewen/articles/7748232.html 这个里面维度是(n,c,h,w)所以分通道求均值方差是axis=(0，2，3），而本项目是(n,h,w,c)，因此均值方差是axis=(0,1,2)
            layer=None,  # Last layer is: Conv2dLayer
            decay=0.9,  # 衰减系数。合适的衰减系数值接近1.0,特别是含多个9的值：0.999,0.99,0.9。如果训练集表现很好而验证/测试集表现得不好，选择小的系数（推荐使用0.9）
            epsilon=2e-5,  # 避免被零除
            act=tf.identity,  # 创建并返回一个和输入大小一样的tensor
            is_train=False,
            fix_gamma=True,
            beta_init=tf.zeros_initializer,  # bn层的参数 y = gamma * x' + beta
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            # bn层的参数 y = gamma * x' + beta，tf.ones_initializer,
            # dtype = tf.float32,
            trainable=None,  # ？和 is_train区别？
            name='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs  # Tensor("resnet_v1_50/conv1/Identity:0", shape=(?, 112, 112, 64), dtype=float32)
        print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (
            self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()  # cnn (?, 112, 112, 64) or # fc (?, 512) 全连接，一个batch会被扔到输出的512个里每个neural中去，也就是说一个batch会被扔512次，因此对每个neural都要加一组mean、var因此是(512,)个mean，(512,)个var
        params_shape = x_shape[-1:]  # cnn (64, ) 均值应该有通道数那么多个，方差也应该有通道数那么多个，一个通道上求衣蛾均值，输入有c个通道就有c个均值c个方差, fc (512, )

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:  # 在命名空间'bn0'下，创建以下下变量
            axis = list(range(len(
                x_shape) - 1))  # cnn：[0, 1, 2]，fc是[0] https://www.cnblogs.com/ranjiewen/articles/7748232.html 这个里面维度是(n,c,h,w)所以分通道求均值方差是axis=(0，2，3），而本项目是(n,h,w,c)，因此均值方差是axis=(0,1,2)
            # fc axis = [0]
            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=tf.float32,
                                   trainable=is_train)  # , restore=restore)  # <tf.Variable 'resnet_v1_50/bn0/beta:0' shape=(64,) dtype=float32_ref>

            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                initializer=gamma_init,
                dtype=tf.float32,
                trainable=fix_gamma,
            )  # restore=restore) # <tf.Variable 'resnet_v1_50/bn0/gamma:0' shape=(64,) dtype=float32_ref>

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init, dtype=tf.float32,
                                          trainable=False)  # restore=restore)  如果有则取已有'moving_mean'，没有的话init新的，# <tf.Variable 'resnet_v1_50/bn0/moving_mean:0' shape=(64,) dtype=float32_ref>
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=tf.float32,
                trainable=False,
            )  # restore=restore)  # <tf.Variable 'resnet_v1_50/bn0/moving_variance:0' shape=(64,) dtype=float32_ref>

            ## 3.
            # These ops will only be preformed when training.  # fc
            mean, variance = tf.nn.moments(self.inputs,
                                           axis)  # cnn (?,112,112,64), axis=[0,1,2] ; for i in 64 : mean(?,112,112), var(?,112,112)，out = (64:), (64:)， Tensor("resnet_v1_50/bn0/moments/Squeeze:0", shape=(64,), dtype=float32)，Tensor("resnet_v1_50/bn0/moments/Squeeze_1:0", shape=(64,), dtype=float32)
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay,
                                                                           zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay,
                    zero_debias=False)  # if zero_debias=True, has bias，如果想提高稳定性，zero_debias设为true
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
                print('bn_mean:', mean)
                print('bn_var:', var)
                self.outputs = act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            else:
                self.outputs = act(
                    tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma,
                                              epsilon))  # Tensor("resnet_v1_50/bn0/Identity:0", shape=(?, 112, 112, 64), dtype=float32)
            variables = [beta, gamma, moving_mean,
                         moving_variance]  # <class 'list'>: [<tf.Variable 'resnet_v1_50/bn0/beta:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'resnet_v1_50/bn0/gamma:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'resnet_v1_50/bn0/moving_mean:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'resnet_v1_50/bn0/moving_variance:0' shape=(64,) dtype=float32_ref>]
        self.all_layers = list(
            layer.all_layers)  # <class 'list'>: [<tf.Tensor 'resnet_v1_50/conv1/Identity:0' shape=(?, 112, 112, 64) dtype=float32>]
        self.all_params = list(
            layer.all_params)  # <class 'list'>: [<tf.Variable 'resnet_v1_50/conv1/W_conv2d:0' shape=(3, 3, 3, 64) dtype=float32_ref>]
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([
                                   self.outputs])  # <class 'list'>: [<tf.Tensor 'resnet_v1_50/conv1/Identity:0' shape=(?, 112, 112, 64) dtype=float32>, <tf.Tensor 'resnet_v1_50/bn0/Identity:0' shape=(?, 112, 112, 64) dtype=float32>]
        self.all_params.extend(variables)


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return tl.layers.MaxPool2d(inputs, [1, 1], strides=(factor, factor), name=scope)


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
        pad_total = kernel_size_effective - 1  # 2
        pad_beg = pad_total // 2  # 1
        pad_end = pad_total - pad_beg  # 1 两边各延伸一个格
        inputs = tl.layers.PadLayer(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]],
                                    name='padding_%s' % scope)  # (?, 112, 112, 64) >> (?, 114, 114, 64)
        if rate == 1:
            nets = tl.layers.Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), W_init=w_init, act=None, padding='VALID', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable,
                                  name=scope + '_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               b_init=None,
                                               rate=rate, act=None, W_init=w_init, padding='SAME',
                                               name=scope)  # (?, 114, 114, 64) >> (?, 56, 56, 256)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable,
                                  name=scope + '_bn/BatchNorm')  # (?, 56, 56, 256) >> (?, 56, 56, 256)
        return nets


def bottleneck_IR(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    '''

    :param inputs:
    :param depth: depth_bottleneck * 4 = 256
    :param depth_bottleneck: 64
    :param stride: 2
    :param rate: 1
    :param w_init:
    :param scope: block1
    :param trainable: False
    :return:
    Block(scope='block1', unit_fn=<function bottleneck_IR at 0x13f22b950>, args=[
        {'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1},
        {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}*2])
    '''
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)  # 64
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')  # ?
        else:
            shortcut = tl.layers.Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv',
                                        use_cudnn_on_gpu=True)  # (?, 112, 112, 64)>> (?, 56, 56, 256), 缩面积/2 升维*4, ksize=1感受野是1只为了变换形状以便后边对应元素相加
            shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=True, trainable=trainable,
                                      name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = BatchNormLayer(inputs, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn1')
        residual = tl.layers.Conv2d(residual, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None,
                                    b_init=None, W_init=w_init, name='conv1',
                                    use_cudnn_on_gpu=True)  # (?, 112, 112, 64)>> (?, 112, 112, 64),ksize=3感受野大吃周边信息
        residual = BatchNormLayer(residual, act=tf.identity, is_train=True, trainable=trainable,
                                  name='conv1_bn2')  # (?, 112, 112, 64)>> (?, 112, 112, 64)
        # bottleneck prelu
        residual = tl.layers.PReluLayer(residual)  # (?, 112, 112, 64)>> (?, 112, 112, 64)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2',
                               trainable=trainable)  # (?, 112, 112, 64) >> (?, 56, 56, 256) ，把吃过周边信息的residual通过cnn变换为shortcut形状，以便对应元素相加
        output = ElementwiseLayer(layer=[shortcut, residual], combine_fn=tf.add, name='combine_layer',
                                  act=None)  # # shortcut (?, 56, 56, 256) + residual (?, 56, 56, 256)
        return output


def bottleneck_IR_SE(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
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
        # squeeze
        squeeze = tl.layers.InputLayer(tf.reduce_mean(residual.outputs, axis=[1, 2]), name='squeeze_layer')
        # excitation
        excitation1 = tl.layers.DenseLayer(squeeze, n_units=int(depth / 16.0), act=tf.nn.relu,
                                           W_init=w_init, name='excitation_1')
        # excitation1 = tl.layers.PReluLayer(excitation1, name='excitation_prelu')
        excitation2 = tl.layers.DenseLayer(excitation1, n_units=depth, act=tf.nn.sigmoid,
                                           W_init=w_init, name='excitation_2')
        # scale
        scale = tl.layers.ReshapeLayer(excitation2, shape=[tf.shape(excitation2.outputs)[0], 1, 1, depth],
                                       name='excitation_reshape')

        residual_se = ElementwiseLayer(layer=[residual, scale],
                                       combine_fn=tf.multiply,
                                       name='scale_layer',
                                       act=None)

        output = ElementwiseLayer(layer=[shortcut, residual_se],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=tf.nn.relu)
        return output


def resnet(inputs, bottle_neck, blocks, w_init=None, trainable=None, reuse=False, keep_rate=None, scope=None):
    """
    :param inputs: Tensor("img_inputs:0", shape=(?, 112, 112, 3), dtype=float32)
    :param bottle_neck: True
    :param blocks: <class 'list'>: [Block(scope='block1', unit_fn=<function bottleneck_IR at 0x14a9c08c0>, args=[{'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1}, {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}, {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}]), Block(scope='block2', unit_fn=<function bottleneck_IR at 0x14a9c08c0>, args=[{'depth': 512, 'depth_bottleneck': 128, 'stride': 2, 'rate': 1}, {'depth': 512, 'depth_bottleneck': 128, 'stride': 1, 'rate': 1}, {'depth': 512, 'depth_bottleneck': 128, 'stride': 1, 'rate': 1}, {'depth': 512, 'depth_bottleneck': 128, 'stride': 1, 'rate': 1}]), Block(scope='block3', unit_fn=<function bottleneck_IR at 0x14a9c08c0>, args=[{'depth': 1024, 'depth_bottleneck': 256, 'stride': 2, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}, {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}]), Block(scope='block4', unit_fn=<function bottleneck_IR at 0x14a9c08c0>, args=[{'depth': 2048, 'depth_bottleneck': 512, 'stride': 2, 'rate': 1}, {'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'rate': 1}, {'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'rate': 1}])]
    :param w_init: <function variance_scaling_initializer.<locals>._initializer at 0x14dfa50e0>
    :param trainable: False
    :param reuse: False
    :param keep_rate: Tensor("dropout_rate:0", dtype=float32)
    :param scope: 'resnet_v1_50'
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):  # 在scope = resnet_v1_50，的命名空间中，创建一下变量：
        # inputs = tf.subtract(inputs, 127.5)
        # inputs = tf.multiply(inputs, 0.0078125)
        net_inputs = tl.layers.InputLayer(inputs, name='input_layer')
        if bottle_neck:
            net = tl.layers.Conv2d(net_inputs, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init,
                                   b_init=None, name='conv1',
                                   use_cudnn_on_gpu=True)  # (?, 112, 112, 3) > (?, 112, 112, 64)
            net = BatchNormLayer(net, act=tf.identity, name='bn0', is_train=True,
                                 trainable=trainable)  # (?, 112, 112, 64) > (?, 112, 112, 64)
            net = tl.layers.PReluLayer(net, name='prelu0')
            print('@@@@@@@@')
        else:
            raise ValueError('The standard resnet must support the bottleneck layer')
        for block in blocks:  # Block(scope='block1', unit_fn=<function bottleneck_IR at 0x14900d950>, args=[{'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1}, {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}, {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}])
            with tf.variable_scope(block.scope):  # 在'block1' 命名空间下，创建以下下内容，建立block串
                for i, var in enumerate(
                        block.args):  # 0； <class 'dict'>: {'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1}
                    with tf.variable_scope('unit_%d' % (i + 1)):  # unit_1
                        net = block.unit_fn(net, depth=var['depth'], depth_bottleneck=var['depth_bottleneck'],
                                            w_init=w_init, stride=var['stride'], rate=var['rate'], scope=None,
                                            trainable=trainable)
        net = BatchNormLayer(net, act=tf.identity, is_train=True, name='E_BN1', trainable=trainable)  # (?, 7, 7, 2048)
        # net = tl.layers.DropoutLayer(net, keep=0.4, name='E_Dropout')
        net.outputs = tf.nn.dropout(net.outputs, keep_prob=keep_rate, name='E_Dropout')
        net_shape = net.outputs.get_shape()
        net = tl.layers.ReshapeLayer(net, shape=[-1, net_shape[1] * net_shape[2] * net_shape[3]],
                                     name='E_Reshapelayer')  # 7*7*2048 = (?, 100352)
        net = tl.layers.DenseLayer(net, n_units=512, W_init=w_init, name='E_DenseLayer')  # (?, 512)
        net = BatchNormLayer(net, act=tf.identity, is_train=True, fix_gamma=False, trainable=trainable, name='E_BN2')
        return net


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def resnetse_v1_block(scope, base_depth, num_units, stride, rate=1, unit_fn=None):
    """Helper function for creating a resnet_v1 bottleneck block.

    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the first unit.
        All other units have stride=1.第一个
      args = [{'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1}, {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}, {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}]

    Returns:
      A resnet_v1 bottleneck block.
    """
    return Block(scope, unit_fn, [{
        'depth': base_depth,  # issue9去掉*4
        'depth_bottleneck': base_depth,
        'stride': stride,
        'rate': rate
    }] + [{
        'depth': base_depth,  # issue9去掉*4
        'depth_bottleneck': base_depth,
        'stride': 1,
        'rate': rate
    }] * (num_units - 1))


def get_resnet(inputs, num_layers, type=None, w_init=None, trainable=None, sess=None, reuse=False, keep_rate=None):
    if type == 'ir':
        unit_func = bottleneck_IR  # ？？？为啥不往函数里面跳呢？ ！！！只是给函数换个名字。unint_func也具有 bottleneck_IR功能
    elif type == 'se_ir':
        unit_func = bottleneck_IR_SE
    else:
        raise ValueError('the input fn is unknown')

    if num_layers == 50:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block3', base_depth=256, num_units=14, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_func)
        ]
        """
        Block(scope='block1', unit_fn=<function bottleneck_IR at 0x13f22b950>, args=[
        {'depth': 256, 'depth_bottleneck': 64, 'stride': 2, 'rate': 1},  112 112 64 > 56,56,256
        {'depth': 256, 'depth_bottleneck': 64, 'stride': 1, 'rate': 1}*2])  56,56,256 > 56,56,256; 56,56,256 > 56,56,256

        Block(scope='block2', unit_fn=<function bottleneck_IR at 0x1441e88c0>, args=[
        {'depth': 512, 'depth_bottleneck': 128, 'stride': 2, 'rate': 1},  56,56,256 > 28,28,512
        {'depth': 512, 'depth_bottleneck': 128, 'stride': 1, 'rate': 1}*3])  28,28,512 > 28,28,512; 28,28,512 > 28,28,512; 28,28,512 > 28,28,512

        Block(scope='block3', unit_fn=<function bottleneck_IR at 0x1441e88c0>, args=[
        {'depth': 1024, 'depth_bottleneck': 256, 'stride': 2, 'rate': 1},  28,28,512 > 14,14,1024
        {'depth': 1024, 'depth_bottleneck': 256, 'stride': 1, 'rate': 1}*13])  14,14,1024 > 14,14,1024; 14,14,1024 > 14,14,1024; 14,14,1024 > 14,14,1024; ... 14,14,1024 > 14,14,1024

        Block(scope='block4', unit_fn=<function bottleneck_IR at 0x1441e88c0>, args=[
        {'depth': 2048, 'depth_bottleneck': 512, 'stride': 2, 'rate': 1},  14,14,1024 > 7,7,2048
        {'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'rate': 1}*2])  7,7,2048 > 7,7,2048; 7,7,2048 > 7,7,2048; 

        """
    elif num_layers == 100:  # issue9改为100
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block2', base_depth=128, num_units=13, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block3', base_depth=256, num_units=30, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_func)
        ]
    elif num_layers == 152:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1, unit_fn=unit_func),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_func)
        ]
    else:
        raise ValueError('Resnet layer %d is not supported now.' % num_layers)
    net = resnet(inputs=inputs,
                 bottle_neck=True,
                 # T/F分别表示：resnet两种bloack单元，https://blog.csdn.net/weixin_42486685/article/details/84789740
                 blocks=blocks,
                 w_init=w_init,
                 trainable=trainable,
                 reuse=reuse,
                 keep_rate=keep_rate,
                 scope='resnet_v1_%d' % num_layers)  # 给resnet送入参数bloack和block函数单元，在resnet里面具体进行网络实例化
    return net


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_place')
    sess = tf.Session()
    # w_init = tf.truncated_normal_initializer(mean=10, stddev=5e-2)
    w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    # test resnetse
    nets = get_resnet(x, 50, type='ir', w_init=w_init, sess=sess)
    tl.layers.initialize_global_variables(sess)

    for p in tl.layers.get_variables_with_name('W_conv2d', True, True):
        print(p.op.name)
    print('##############' * 30)
    with sess:
        nets.print_params()
