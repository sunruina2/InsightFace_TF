import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers.python.layers import utils
import collections
# from tl_layers_modify import ElementwiseLayer, BatchNormLayer, Conv2d, PReluLayer, DenseLayer
# MGPU和issue9的主要区别点：
# （1）layer>>BatchNormLayer, Conv2d, PReluLayer, DenseLayer在Layer源码上修改：加上了 /cpu:0 ,并定义dtype=D_type
# （2）get_resnet不再传入reuse=False
# （3）dropout改用layer的，不用nn的了

D_TYPE = tf.float32
TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
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
        layer = [],
        combine_fn = tf.minimum,
        name ='elementwise_layer',
        act = None,
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
        print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=D_TYPE, trainable=is_train)  #, restore=restore)

                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=tf.float32,
                    trainable=fix_gamma,
                )  #restore=restore)

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init, dtype=D_TYPE, trainable=False)  #   restore=restore)
                moving_variance = tf.get_variable(
                    'moving_variance',
                    params_shape,
                    initializer=tf.constant_initializer(1.),
                    dtype=tf.float32,
                    trainable=False,
                )  #   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)  # if zero_debias=True, has bias
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
                self.outputs = act(tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))
            variables = [beta, gamma, moving_mean, moving_variance]
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


def Conv2d(
        net,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        act=None,
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args={},
        b_init_args={},
        use_cudnn_on_gpu=None,
        data_format=None,
        name='conv2d',
):
    """Wrapper for :class:`Conv2dLayer`, if you don't understand how to use :class:`Conv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_filter : number of filter.
    filter_size : tuple (height, width) for filter size.
    strides : tuple (height, width) for strides.
    act : None or activation function.
    others : see :class:`Conv2dLayer`.

    Examples
    --------
    >>> w_init = tf.truncated_normal_initializer(stddev=0.01)
    >>> b_init = tf.constant_initializer(value=0.0)
    >>> inputs = InputLayer(x, name='inputs')
    >>> conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    >>> conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    >>> pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
    >>> conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    >>> conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    >>> pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
    """
    assert len(strides) == 2, "len(strides) should be 2, Conv2d and Conv2dLayer are different."
    if act is None:
        act = tf.identity

    try:
        pre_channel = int(net.outputs.get_shape()[-1])
    except:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        pre_channel = 1
        print("[warnings] unknow input channels, set to 1")
    net = Conv2dLayer(
        net,
        act=act,
        shape=[filter_size[0], filter_size[1], pre_channel, n_filter],  # 32 features for each 5x5 patch
        strides=[1, strides[0], strides[1], 1],
        padding=padding,
        W_init=W_init,
        W_init_args=W_init_args,
        b_init=b_init,
        b_init_args=b_init_args,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        name=name)
    return net


class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints.
        The stride of the sliding window for each dimension of input.\n
        It Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    use_cudnn_on_gpu : bool, default is None.
    data_format : string "NHWC" or "NCHW", default is "NHWC"
    name : a string or None
        An optional name to attach to this layer.

    Notes
    ------
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.Conv2dLayer(network,
    ...                   act = tf.nn.relu,
    ...                   shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    ...                   strides=[1, 1, 1, 1],
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   W_init_args={},
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   b_init_args = {},
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> network = tl.layers.PoolLayer(network,
    ...                   ksize=[1, 2, 2, 1],
    ...                   strides=[1, 2, 2, 1],
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    >>> Without TensorLayer, you can implement 2d convolution as follow.
    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> b = tf.Variable(b_init(shape=[32], ), name='b_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') + b )
    """

    def __init__(
            self,
            layer=None,
            act=tf.identity,
            shape=[5, 5, 1, 100],
            strides=[1, 1, 1, 1],
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args={},
            b_init_args={},
            use_cudnn_on_gpu=None,
            data_format=None,
            name='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] Conv2dLayer %s: shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init:
                with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                    b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(
                    tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


## Special activation
class PReluLayer(Layer):
    """
    The :class:`PReluLayer` class is Parametric Rectified Linear layer.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    channel_shared : `bool`. Single weight is shared by all channels
    a_init : alpha initializer, default zero constant.
        The initializer for initializing the alphas.
    a_init_args : dictionary
        The arguments for the weights initializer.
    name : A name for this activation op (optional).

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_
    """

    def __init__(
            self,
            layer=None,
            channel_shared=False,
            a_init=tf.constant_initializer(value=0.0),
            a_init_args={},
            # restore = True,
            name="prelu_layer"):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] PReluLayer %s: channel_shared:%s" % (self.name, channel_shared))
        if channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name) as vs:
            with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=a_init, dtype=D_TYPE, **a_init_args)
            try:  ## TF 1.0
                self.outputs = tf.nn.relu(self.inputs) + tf.multiply(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5
            except:  ## TF 0.12
                self.outputs = tf.nn.relu(self.inputs) + tf.mul(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])
        self.all_params.extend([alphas])


## Dense layer
class DenseLayer(Layer):
    """
    The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    n_units : int
        The number of units of the layer.
    act : activation function
        The function that is applied to the layer activations.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable.
    b_init_args : dictionary
        The arguments for the biases tf.get_variable.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DenseLayer(
    ...                 network,
    ...                 n_units=800,
    ...                 act = tf.nn.relu,
    ...                 W_init=tf.truncated_normal_initializer(stddev=0.1),
    ...                 name ='relu_layer'
    ...                 )

    >>> Without TensorLayer, you can do as follow.
    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the input to this layer has more than two axes, it need to flatten the
    input by using :class:`FlattenLayer` in this case.
    """

    def __init__(
            self,
            layer=None,
            n_units=100,
            act=tf.identity,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args={},
            b_init_args={},
            name='dense_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  [TL] DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:
            with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init is not None:
                try:
                    with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                        b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, dtype=D_TYPE, **b_init_args)
                except:  # If initializer is a constant, do not specify shape.
                    with tf.device('/cpu:0'):  # MGPU 增加 '/cpu:0'，(以及dtype=D_TYPE)
                        b = tf.get_variable(name='b', initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, W) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, W))

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


def get_shape(input_tensor):
    static_shape = input_tensor.get_shape().as_list()
    dynamic_shape = tf.unstack(tf.shape(input_tensor))
    final_shapes = [shapes[1] if shapes[0] is None else shapes[0] for shapes in zip(static_shape, dynamic_shape)]
    return final_shapes


## Group Batch Normalization layer
class GroupNormLayer(Layer):
    """The :class:`GroupNormLayer` class is a for instance normalization.
       The implementation is based on paper [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)
    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function.
    epsilon : float
        A small float number.
    scale_init : beta initializer
        The initializer for initializing beta
    offset_init : gamma initializer
        The initializer for initializing gamma
    G: the group number
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
            self,
            layer=None,
            act=tf.identity,
            epsilon=1e-5,
            scale_init=tf.constant_initializer(1.0),
            offset_init=tf.constant_initializer(0.0),
            G=32,
            name='group_norm',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] GroupNormLayer %s: epsilon:%f act:%s" % (self.name, epsilon, act.__name__))
        inputs_shape = get_shape(layer.outputs)
        G = tf.minimum(G, inputs_shape[-1])
        # [N, H, W, C] to [N, C, H, W]
        temp_input = tf.transpose(self.inputs, [0, 3, 1, 2])
        temp_input = tf.reshape(temp_input, [inputs_shape[0], G, inputs_shape[-1]//G, inputs_shape[1], inputs_shape[2]],
                                name='group_reshape1')
        with tf.variable_scope(name) as vs:
            mean, var = tf.nn.moments(temp_input, [2, 3, 4], keep_dims=True)
            scale = tf.get_variable('scale', shape=[1, inputs_shape[-1], 1, 1], initializer=scale_init, dtype=D_TYPE)
            offset = tf.get_variable('offset', shape=[1, inputs_shape[-1], 1, 1], initializer=offset_init, dtype=D_TYPE)
            temp_input = (temp_input - mean) / tf.sqrt(var + epsilon)
            temp_input = tf.reshape(temp_input, shape=[inputs_shape[0], inputs_shape[-1], inputs_shape[1], inputs_shape[2]],
                                    name='group_reshape2')
            self.outputs = scale * temp_input + offset
            self.outputs = tf.transpose(self.outputs, [0, 2, 3, 1])
            self.outputs = act(self.outputs)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
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
            nets = Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                   strides=(strides, strides), W_init=w_init, act=None, padding='SAME', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        return nets
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tl.layers.PadLayer(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], name='padding_%s' % scope)
        if rate == 1:
            nets = Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), W_init=w_init, act=None, padding='VALID', name=scope,
                                    use_cudnn_on_gpu=True)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                              rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets = BatchNormLayer(nets, act=tf.identity, is_train=True, trainable=trainable, name=scope+'_bn/BatchNorm')
        return nets


def bottleneck_IR(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv', use_cudnn_on_gpu=True)
            shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=True, trainable=trainable, name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = BatchNormLayer(inputs, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn1')
        residual = Conv2d(residual, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None, b_init=None,
                                    W_init=w_init, name='conv1', use_cudnn_on_gpu=True)
        residual = BatchNormLayer(residual, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn2')
        # bottleneck prelu
        residual = PReluLayer(residual)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2', trainable=trainable)
        output = ElementwiseLayer(layer=[shortcut, residual],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=None)
        return output


def bottleneck_IR_SE(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv', use_cudnn_on_gpu=True)
            shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=True, trainable=trainable, name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = BatchNormLayer(inputs, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn1')
        residual = Conv2d(residual, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None, b_init=None,
                                    W_init=w_init, name='conv1', use_cudnn_on_gpu=True)
        residual = BatchNormLayer(residual, act=tf.identity, is_train=True, trainable=trainable, name='conv1_bn2')
        # bottleneck prelu
        residual = PReluLayer(residual)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2', trainable=trainable)
        # squeeze
        squeeze = tl.layers.InputLayer(tf.reduce_mean(residual.outputs, axis=[1, 2]), name='squeeze_layer')
        # excitation
        excitation1 = DenseLayer(squeeze, n_units=int(depth/16.0), act=tf.nn.relu,
                                           W_init=w_init, name='excitation_1')
        # excitation1 = tl.layers.PReluLayer(excitation1, name='excitation_prelu')
        excitation2 = DenseLayer(excitation1, n_units=depth, act=tf.nn.sigmoid,
                                           W_init=w_init, name='excitation_2')
        # scale
        scale = tl.layers.ReshapeLayer(excitation2, shape=[tf.shape(excitation2.outputs)[0], 1, 1, depth], name='excitation_reshape')

        residual_se = ElementwiseLayer(layer=[residual, scale],
                                       combine_fn=tf.multiply,
                                       name='scale_layer',
                                       act=None)

        output = ElementwiseLayer(layer=[shortcut, residual_se],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=tf.nn.relu)
        return output


# MGPU 去掉reuse=False,参数
def resnet(inputs, bottle_neck, blocks, w_init=None, trainable=None, keep_rate=None, scope=None):
    with tf.variable_scope(scope):
        net_inputs = tl.layers.InputLayer(inputs, name='input_layer')
        if bottle_neck:
            net = Conv2d(net_inputs, n_filter=64, filter_size=(3, 3), strides=(1, 1),
                                   act=None, W_init=w_init, b_init=None, name='conv1', use_cudnn_on_gpu=True)
            net = BatchNormLayer(net, act=tf.identity, name='bn0', is_train=True, trainable=trainable)
            net = PReluLayer(net, name='prelu0')
        else:
            raise ValueError('The standard resnet must support the bottleneck layer')
        for block in blocks:
            with tf.variable_scope(block.scope):
                for i, var in enumerate(block.args):
                    with tf.variable_scope('unit_%d' % (i+1)):
                        net = block.unit_fn(net, depth=var['depth'], depth_bottleneck=var['depth_bottleneck'],
                                            w_init=w_init, stride=var['stride'], rate=var['rate'], scope=None,
                                            trainable=trainable)
        net = BatchNormLayer(net, act=tf.identity, is_train=True, name='E_BN1', trainable=trainable)
        net.outputs = tf.nn.dropout(net.outputs, keep_prob=keep_rate, name='E_Dropout')  # issue_old
        # net = tl.layers.DropoutLayer(net, keep=keep_rate, name='E_Dropout')  # MGPU  placehoder报错，改回issue版本
        net_shape = net.outputs.get_shape()
        net = tl.layers.ReshapeLayer(net, shape=[-1, net_shape[1]*net_shape[2]*net_shape[3]], name='E_Reshapelayer')
        net = DenseLayer(net, n_units=512, W_init=w_init, name='E_DenseLayer')
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
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return Block(scope, unit_fn, [{
      'depth': base_depth,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'rate': rate
  }] + [{
      'depth': base_depth,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'rate': rate
  }] * (num_units - 1))


# MGPU 去掉reuse=False,参数
def get_resnet(inputs, num_layers, type=None, w_init=None, trainable=None, keep_rate=None, sess=None):
    if type == 'ir':
        unit_fn = bottleneck_IR
    elif type == 'se_ir':
        unit_fn = bottleneck_IR_SE
    else:
        raise ValueError('the input fn is unknown')

    if num_layers == 50:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=14, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    elif num_layers == 100:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=13, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=30, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    elif num_layers == 152:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    else:
        raise ValueError('Resnet layer %d is not supported now.' % num_layers)
    # MGPU 去掉reuse=False,参数
    net = resnet(inputs=inputs,
                 bottle_neck=True,
                 blocks=blocks,
                 w_init=w_init,
                 trainable=trainable,
                 keep_rate=keep_rate,
                 scope='resnet_v1_%d' % num_layers)
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
        print('##############'*30)
        with sess:
            nets.print_params()
