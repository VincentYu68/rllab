# encoding: utf-8

import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne
import lasagne.init as LI
import theano
import theano as T
import theano.tensor as TT
import numpy as np

class ElemwiseMultLayer(L.ElemwiseMergeLayer):
    """
    This layer performs an elementwise sum of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.

    Notes
    -----
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """
    def __init__(self, incomings, coeffs=1, **kwargs):
        super(ElemwiseMultLayer, self).__init__(incomings, TT.mul, **kwargs)
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)

        self.coeffs = coeffs

    def get_output_for(self, inputs, **kwargs):
        # if needed multiply each input by its coefficient
        inputs = [input * coeff if coeff != 1 else input
                  for coeff, input in zip(self.coeffs, inputs)]

        # pass scaled inputs to the super class for summing
        return super(ElemwiseMultLayer, self).get_output_for(inputs, **kwargs)

# layer that gives constant value
class ConstantLayer(L.Layer):
    def __init__(self, incoming, constant_vec, **kwargs):
        super(ConstantLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_out = len(constant_vec)
        self.constant_W = self.add_param(lasagne.init.Constant(0), (num_inputs, self.num_out), name="constW", trainable=False)
        self.constant_vec = self.add_param(constant_vec, constant_vec.shape, name='constant', trainable=False)

    def get_output_for(self, input, **kwargs):
        return TT.dot(input, self.constant_W) + self.constant_vec

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_out)

# apply sin transform to input
class PhaseLayer(L.Layer):
    def __init__(self, incoming, frequency, phase, scale = None, **kwargs):
        super(PhaseLayer, self).__init__(incoming, **kwargs)
        self.frequency = frequency
        self.phase = phase

    def get_output_for(self, input, **kwargs):
        return TT.sin(input * self.frequency + self.phase)

    def get_output_shape_for(self, input_shape):
        return input_shape


# take part of the input as output
class SplitLayer(L.Layer):
    def __init__(self, incoming, select_idx, scale = None, **kwargs):
        super(SplitLayer, self).__init__(incoming, **kwargs)
        self.select_idx = select_idx
        self.scale = scale

    def get_output_for(self, input, **kwargs):
        if self.scale is None:
            return input[:, self.select_idx]
        else:
            return input[:, self.select_idx] * self.scale

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], len(self.select_idx))


class RBFLayer(L.Layer):
    def __init__(self, incoming, num_units, bandwidth, W=lasagne.init.Normal(1), b=lasagne.init.Uniform(np.pi), **kwargs):
        super(RBFLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W', trainable=False)
        self.bandwidth = bandwidth
        self.b = self.add_param(b, (num_units,), name='b', trainable=False)

    def get_output_for(self, input, **kwargs):
        return TT.sin(TT.dot(input, self.W) / self.bandwidth + self.b)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class ParamLayer(L.Layer):

    def __init__(self, incoming, num_units, param=lasagne.init.Constant(0.),
                 trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.param = self.add_param(
            param,
            (num_units,),
            name="param",
            trainable=trainable
        )

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        ndim = input.ndim
        reshaped_param = TT.reshape(self.param, (1,) * (ndim - 1) + (self.num_units,))
        tile_arg = TT.concatenate([input.shape[:-1], [1]])
        tiled = TT.tile(reshaped_param, tile_arg, ndim=ndim)
        return tiled

class ParamLayerSplit(L.Layer):
    def __init__(self, incoming, num_units, param=lasagne.init.Constant(0.), split_num=1, init_param=None,
                 trainable=True, **kwargs):
        super(ParamLayerSplit, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.param_list = []
        for i in range(split_num):
            self.param_list.append(self.add_param(
                param,
                (num_units,),
                name="param%d"%(i),
                trainable=trainable
            ))
            if init_param is not None:
                self.get_params()[-1].set_value(init_param.get_value())
        self.split_num = split_num

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        ndim = input.ndim

        activation_mask = TT.stack([input[:, -self.split_num]]*self.num_units).T
        reshaped_param = TT.reshape(self.param_list[0], (1,) * (ndim - 1) + (self.num_units,))
        tile_arg = TT.concatenate([input.shape[:-1], [1]])
        tiled = TT.tile(reshaped_param, tile_arg, ndim=ndim) * activation_mask
        for i in range(1, self.split_num):
            activation_mask = TT.stack([input[:, -self.split_num+i]]*self.num_units).T
            reshaped_param = TT.reshape(self.param_list[i], (1,) * (ndim - 1) + (self.num_units,))
            tile_arg = TT.concatenate([input.shape[:-1], [1]])
            tiled += TT.tile(reshaped_param, tile_arg, ndim=ndim) * activation_mask

        return tiled


class OpLayer(L.MergeLayer):
    def __init__(self, incoming, op,
                 shape_op=lambda x: x, extras=None, **kwargs):
        if extras is None:
            extras = []
        incomings = [incoming] + extras
        super(OpLayer, self).__init__(incomings, **kwargs)
        self.op = op
        self.shape_op = shape_op
        self.incomings = incomings

    def get_output_shape_for(self, input_shapes):
        return self.shape_op(*input_shapes)

    def get_output_for(self, inputs, **kwargs):
        return self.op(*inputs)


class BatchNormLayer(L.Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, mode='low_mem',
    beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem', beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
                 mean=lasagne.init.Constant(0), std=lasagne.init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=False)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(std, shape, 'std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_std = TT.sqrt(input.var(self.axes) + self.epsilon)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = kwargs.get('batch_norm_use_averages',
                                  deterministic)
        if use_averages:
            mean = self.mean
            std = self.std
        else:
            mean = input_mean
            std = input_std

        # Decide whether to update the stored averages
        update_averages = kwargs.get('batch_norm_update_averages',
                                     not deterministic)
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_std.default_update = ((1 - self.alpha) *
                                              running_std +
                                              self.alpha * input_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            std += 0 * running_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        std = std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * TT.inv(std)) + beta
        return normalized


def batch_norm(layer, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` and :class:`NonlinearityLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormLayer` constructor.

    Returns
    -------
    BatchNormLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.

    Examples
    --------
    Just wrap any layer into a :func:`batch_norm` call on creating it:

    >>> from lasagne.layers import InputLayer, DenseLayer, batch_norm
    >>> from lasagne.nonlinearities import tanh
    >>> l1 = InputLayer((64, 768))
    >>> l2 = batch_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))

    This introduces batch normalization right before its nonlinearity:

    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'BatchNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    layer = BatchNormLayer(layer, **kwargs)
    if nonlinearity is not None:
        layer = L.NonlinearityLayer(layer, nonlinearity)
    return layer

class MaskedDenseLayer(L.Layer):
    def __init__(self, incoming, num_units, W_init, b_init, W=LI.GlorotUniform(), split_num=1, split_mask_W=None, split_mask_b=None,
                 b=LI.Constant(0.), nonlinearity=LN.rectify,
                 **kwargs):
        super(MaskedDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (LN.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))-split_num

        self.Ws = []
        self.bs = []
        for i in range(split_num+0):
            append='split'
            if i == 0:
                append='share'
            self.Ws.append(self.add_param(W, (num_inputs, num_units), name="W"+append+"%d"%(i)))
            if W_init is not None:
                self.get_params()[-1].set_value(W_init)
            self.bs.append(self.add_param(b, (num_units,), name="b"+append+"%d"%(i),
                                    regularizable=False))
            if b_init is not None:
                self.get_params()[-1].set_value(b_init)
        self.split_mask_W = split_mask_W
        self.share_mask_W = np.ones((num_inputs, num_units)) - split_mask_W
        self.split_mask_b = split_mask_b
        self.share_mask_b = np.ones((num_units,)) - split_mask_b

        self.input_size = num_inputs
        self.split_num = split_num

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input[:, 0:-self.split_num], self.Ws[0]*self.share_mask_W)
        activation = activation + (self.bs[0]* self.share_mask_b).dimshuffle('x', 0)
        for i in range(0, len(self.Ws)):
            activation_mask = TT.stack([input[:, -self.split_num+i-0]]*self.num_units).T
            activation += (T.dot(input[:, 0:-self.split_num], self.Ws[i]*self.split_mask_W) + (self.bs[i]* self.split_mask_b).dimshuffle('x', 0)) * activation_mask

        return self.nonlinearity(activation)

class MaskedDenseLayerCont(L.Layer):
    def __init__(self, incoming, num_units, init_layer, task_id = 0, W=LI.GlorotUniform(),
                 b=LI.Constant(0.), nonlinearity=LN.rectify, freeze_first = False,
                 **kwargs):
        super(MaskedDenseLayerCont, self).__init__(incoming, **kwargs)
        self.nonlinearity = (LN.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.Ws = []
        self.bs = []
        for i in range(2):
            append='split'
            if i == 0:
                append='share'
            trainable=False
            if i != 0:
                trainable = True
            self.Ws.append(self.add_param(W, (num_inputs, num_units), name="W"+append+"%d"%(i), trainable=trainable))

            self.get_params()[-1].set_value(init_layer.get_params()[i*task_id*2].get_value())

            self.bs.append(self.add_param(b, (num_units,), name="b"+append+"%d"%(i),
                                    regularizable=False, trainable=trainable))
            self.get_params()[-1].set_value(init_layer.get_params()[i*task_id*2+1].get_value())
        self.split_mask_W = init_layer.split_mask_W
        self.share_mask_W = np.ones((num_inputs, num_units)) - init_layer.split_mask_W
        self.split_mask_b = init_layer.split_mask_b
        self.share_mask_b = np.ones((num_units,)) - init_layer.split_mask_b
        self.input_size = num_inputs

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.Ws[0]*self.share_mask_W) + T.dot(input, self.Ws[1]*self.split_mask_W)
        activation = activation + (self.bs[0]* self.share_mask_b).dimshuffle('x', 0) + (self.bs[1]* self.split_mask_b).dimshuffle('x', 0)

        return self.nonlinearity(activation)
