
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import theano.tensor as TT
import theano
from rllab.misc import ext
from rllab.core.lasagne_layers import OpLayer, RBFLayer, SplitLayer, ElemwiseMultLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable

import numpy as np


def wrapped_conv(*args, **kwargs):
    copy = dict(kwargs)
    copy.pop("image_shape", None)
    copy.pop("filter_shape", None)
    assert copy.pop("filter_flip", False)

    input, W, input_shape, get_W_shape = args
    if theano.config.device == 'cpu':
        return theano.tensor.nnet.conv2d(*args, **kwargs)
    try:
        return theano.sandbox.cuda.dnn.dnn_conv(
            input.astype('float32'),
            W.astype('float32'),
            **copy
        )
    except Exception as e:
        print("falling back to default conv2d")
        return theano.tensor.nnet.conv2d(*args, **kwargs)




class MLP(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None, input_layer=None, input_shape=None, batch_norm=False):

        Serializable.quick_init(self, locals())

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if input_layer is None:
            l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        else:
            l_in = input_layer
        self._layers = [l_in]
        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
            if batch_norm:
                l_hid = L.batch_norm(l_hid)
            self._layers.append(l_hid)

        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        # self._input_var = l_in.input_var
        self._output = L.get_output(l_out)
        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    # @property
    # def input_var(self):
    #     return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class GRULayer(L.Layer):
    """
    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity,
                 gate_nonlinearity=LN.sigmoid, name=None,
                 W_init=LI.GlorotUniform(), b_init=LI.Constant(0.),
                 hidden_init=LI.Constant(0.), hidden_init_trainable=True):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = LN.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = LN.identity

        super(GRULayer, self).__init__(incoming, name=name)

        input_shape = self.input_shape[2:]

        input_dim = ext.flatten_shape_dim(input_shape)
        # self._name = name
        # Weights for the initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights for the reset gate
        self.W_xr = self.add_param(W_init, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W_init, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
        # Weights for the update gate
        self.W_xu = self.add_param(W_init, (input_dim, num_units), name="W_xu")
        self.W_hu = self.add_param(W_init, (num_units, num_units), name="W_hu")
        self.b_u = self.add_param(b_init, (num_units,), name="b_u", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)
        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity

    def step(self, x, hprev):
        r = self.gate_nonlinearity(x.dot(self.W_xr) + hprev.dot(self.W_hr) + self.b_r)
        u = self.gate_nonlinearity(x.dot(self.W_xu) + hprev.dot(self.W_hu) + self.b_u)
        c = self.nonlinearity(x.dot(self.W_xc) + r * (hprev.dot(self.W_hc)) + self.b_c)
        h = (1 - u) * hprev + u * c
        return h.astype(theano.config.floatX)

    def get_step_layer(self, l_in, l_prev_hidden):
        return GRUStepLayer(incomings=[l_in, l_prev_hidden], gru_layer=self)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        n_batches = input.shape[0]
        n_steps = input.shape[1]
        input = TT.reshape(input, (n_batches, n_steps, -1))
        h0s = TT.tile(TT.reshape(self.h0, (1, self.num_units)), (n_batches, 1))
        # flatten extra dimensions
        shuffled_input = input.dimshuffle(1, 0, 2)
        hs, _ = theano.scan(fn=self.step, sequences=[shuffled_input], outputs_info=h0s)
        shuffled_hs = hs.dimshuffle(1, 0, 2)
        return shuffled_hs


class GRUStepLayer(L.MergeLayer):
    def __init__(self, incomings, gru_layer, name=None):
        super(GRUStepLayer, self).__init__(incomings, name)
        self._gru_layer = gru_layer

    def get_params(self, **tags):
        return self._gru_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0]
        return n_batch, self._gru_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = x.shape[0]
        x = x.reshape((n_batch, -1))
        return self._gru_layer.step(x, hprev)


class GRUNetwork(object):
    def __init__(self, input_shape, output_dim, hidden_dim, hidden_nonlinearity=LN.rectify,
                 output_nonlinearity=None, name=None, input_var=None, input_layer=None):
        if input_layer is None:
            l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
        else:
            l_in = input_layer
        l_step_input = L.InputLayer(shape=(None,) + input_shape)
        l_step_prev_hidden = L.InputLayer(shape=(None, hidden_dim))
        l_gru = GRULayer(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                         hidden_init_trainable=False)
        l_gru_flat = L.ReshapeLayer(
            l_gru, shape=(-1, hidden_dim)
        )
        l_output_flat = L.DenseLayer(
            l_gru_flat,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
        )
        l_output = OpLayer(
            l_output_flat,
            op=lambda flat_output, l_input:
            flat_output.reshape((l_input.shape[0], l_input.shape[1], -1)),
            shape_op=lambda flat_output_shape, l_input_shape:
            (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
            extras=[l_in]
        )
        l_step_hidden = l_gru.get_step_layer(l_step_input, l_step_prev_hidden)
        l_step_output = L.DenseLayer(
            l_step_hidden,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            W=l_output_flat.W,
            b=l_output_flat.b,
        )

        self._l_in = l_in
        self._hid_init_param = l_gru.h0
        self._l_gru = l_gru
        self._l_out = l_output
        self._l_step_input = l_step_input
        self._l_step_prev_hidden = l_step_prev_hidden
        self._l_step_hidden = l_step_hidden
        self._l_step_output = l_step_output

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_hidden_layer(self):
        return self._l_step_prev_hidden

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param


class ConvNetwork(object):
    def __init__(self, input_shape, output_dim, hidden_sizes,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 # conv_W_init=LI.GlorotUniform(), conv_b_init=LI.Constant(0.),
                 hidden_nonlinearity=LN.rectify,
                 output_nonlinearity=LN.softmax,
                 name=None, input_var=None):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if len(input_shape) == 3:
            l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var)
            l_hid = L.reshape(l_in, ([0],) + input_shape)
        elif len(input_shape) == 2:
            l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var)
            input_shape = (1,) + input_shape
            l_hid = L.reshape(l_in, ([0],) + input_shape)
        else:
            l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
            l_hid = l_in
        for idx, conv_filter, filter_size, stride, pad in zip(
                range(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
        ):
            l_hid = L.Conv2DLayer(
                l_hid,
                num_filters=conv_filter,
                filter_size=filter_size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                name="%sconv_hidden_%d" % (prefix, idx),
                convolution=wrapped_conv,
            )
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var



class RBFLinear(LasagnePowered, Serializable):
    def __init__(self, output_dim, RBF_size, RBF_bandwidth, output_nonlinearity,
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None, input_layer=None, input_shape=None, batch_norm=False):

        Serializable.quick_init(self, locals())

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if input_layer is None:
            l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        else:
            l_in = input_layer
        self._layers = [l_in]
        l_hid = RBFLayer(l_in, num_units=RBF_size, bandwidth = RBF_bandwidth)
        self._layers.append(l_hid)

        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        # self._input_var = l_in.input_var
        self._output = L.get_output(l_out)
        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    # @property
    # def input_var(self):
    #     return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


# Hierarchical MLP
# Experimental for walker2d case only
# Highly Engineered!
class HMLP(LasagnePowered, Serializable):
    def __init__(self, hidden_sizes, hidden_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 subnet_size = (16,16), subnet_nonlinearity=LN.tanh, subnet_W_init=LI.GlorotUniform(), subnet_b_init=LI.Constant(0.),
                 name=None, input_shape=None, option_dim = 2, subnet_split1 = [2,3,4,11,12,13], subnet_split2=[5,6,7, 14,15,16], sub_out_dim = 3):

        Serializable.quick_init(self, locals())

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        l_in = L.InputLayer(shape=(None,) + input_shape)
        self._layers = [l_in]
        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
            self._layers.append(l_hid)

        l_leg1 = SplitLayer(l_in, subnet_split1)
        #l_leg1 = SplitLayer(l_in, [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16])
        l_option1 = L.DenseLayer(
                l_hid,
                num_units=option_dim,
                nonlinearity=hidden_nonlinearity,
                name="%soption_1" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )
        dup_rep = int(len(subnet_split1) / option_dim)
        l_dup1 = L.concat([l_option1]*dup_rep)
        l_concat1 = L.ElemwiseSumLayer([l_leg1, l_dup1])

        l_leg2 = SplitLayer(l_in, subnet_split2)
        #l_leg2 = SplitLayer(l_in, [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16])
        l_option2 = L.DenseLayer(
                l_hid,
                num_units=option_dim,
                nonlinearity=hidden_nonlinearity,
                name="%soption_2" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )
        #l_concat2 = L.concat([l_leg2, l_option2])
        l_dup2 = L.concat([l_option2]*dup_rep)
        l_concat2 = L.ElemwiseSumLayer([l_leg2, l_dup2])
        self._layers.append(l_leg1)
        self._layers.append(l_option1)
        self._layers.append(l_concat1)
        self._layers.append(l_leg2)
        self._layers.append(l_option2)
        self._layers.append(l_concat2)

        self._layers.append(l_dup1)
        self._layers.append(l_dup2)

        l_snet = l_concat1
        l_snet2 = l_concat2
        for idx, size in enumerate(subnet_size):
            l_snet = L.DenseLayer(
                l_snet,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_1_%d" % (prefix, idx),
                W=subnet_W_init,
                b=subnet_b_init,
            )
            self._layers.append(l_snet)

            l_snet2 = L.DenseLayer(
                l_snet2,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_2_%d" % (prefix, idx),
                W=l_snet.W,
                b=l_snet.b,
            )
            self._layers.append(l_snet2)
        l_out1 = L.DenseLayer(
            l_snet,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput1" % (prefix,),
            W=subnet_W_init,
            b=subnet_b_init,
        )
        self._layers.append(l_out1)

        '''l_snet = l_concat2
        for idx, size in enumerate(subnet_size):
            l_snet = L.DenseLayer(
                l_snet,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_2_%d" % (prefix, idx),
                W=subnet_W_init,
                b=subnet_b_init,
            )
            self._layers.append(l_snet)'''
        l_out2 = L.DenseLayer(
            l_snet2,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput2" % (prefix,),
            W=l_out1.W,
            b=l_out1.b,
        )
        self._layers.append(l_out2)

        l_out = L.concat([l_out1, l_out2])
        self._layers.append(l_out)

        self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        # self._input_var = l_in.input_var
        self._output = L.get_output(l_out)

        self.hlc_signal1 = L.get_output(l_option1)
        self.hlc_signal2 = L.get_output(l_option2)
        self.leg1_part = L.get_output(l_leg1)
        self.leg2_part = L.get_output(l_leg2)

        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    # @property
    # def input_var(self):
    #     return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output