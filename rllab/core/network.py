
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import theano.tensor as TT
import theano
from rllab.misc import ext
from rllab.core.lasagne_layers import OpLayer, RBFLayer, SplitLayer, ElemwiseMultLayer, PhaseLayer, MaskedDenseLayer, MaskedDenseLayerCont
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

class MLPAppend(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None, input_layer=None, input_shape=None, batch_norm=False, append_dim = 1):

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

        l_prop_in = SplitLayer(l_in, range(0, input_shape[0]-append_dim))
        l_append_in = SplitLayer(l_in, range(-append_dim, 0))

        l_hid = l_prop_in
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

            if idx == 1:
                l_hid = L.concat([l_hid, l_append_in])

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

class MLPAux(LasagnePowered, Serializable):
    def __init__(self, history_size, output_dim, output_nonlinearity, CtlNet, skip_last = 1, copy_output=False):

        Serializable.quick_init(self, locals())

        l_in = L.InputLayer(shape=(None, CtlNet.input_layer.shape[1]*history_size), input_var=None, name='aux_input')
        obs_dim = CtlNet.input_layer.shape[1]
        self._layers = [l_in]
        l_hid_out = []
        for i in range(history_size):
            l_hid = SplitLayer(l_in, np.arange(i*obs_dim,(i+1) * obs_dim))
            for h in range(len(CtlNet.layers)-2-skip_last):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=CtlNet.layers[h+1].num_units,
                    nonlinearity=CtlNet.layers[h+1].nonlinearity,
                    name="hidden_%d_%d" % (i, h),
                    W=CtlNet.layers[h+1].W,
                    b=CtlNet.layers[h+1].b,
                )
                self._layers.append(l_hid)
            l_hid_out.append(l_hid)
        l_merge = L.concat(l_hid_out)
        if not copy_output:
            l_out = L.DenseLayer(
                l_merge,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="auxoutput",
                W=LI.GlorotUniform(),
                b=LI.Constant(0.),
            )
        else:
            l_out = L.DenseLayer(
                l_merge,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="auxoutput",
                W=CtlNet.layers[-1].W,
                b=CtlNet.layers[-1].b,
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
# Experimental for bipedal model only
# Highly Engineered!
class HMLP_NonConcat(LasagnePowered, Serializable):
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
        l_hid.get_params()

        l_options = L.DenseLayer(
                l_hid,
                num_units=option_dim*2,
                nonlinearity=hidden_nonlinearity,
                name="%soptions" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_leg1 = SplitLayer(l_in, subnet_split1)
        #l_leg1 = SplitLayer(l_in, [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16])
        l_option1 = SplitLayer(l_options, range(0, option_dim))
        dup_rep = int(len(subnet_split1) / option_dim)
        l_dup1 = L.concat([l_option1]*dup_rep)
        l_concat1 = L.ElemwiseSumLayer([l_leg1, l_dup1])

        l_leg2 = SplitLayer(l_in, subnet_split2)
        #l_leg2 = SplitLayer(l_in, [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16])
        l_option2 = SplitLayer(l_options, range(option_dim, 2*option_dim))
        #l_concat2 = L.concat([l_leg2, l_option2])
        l_dup2 = L.concat([l_option2]*dup_rep)
        l_concat2 = L.ElemwiseSumLayer([l_leg2, l_dup2])
        self._layers.append(l_options)
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


class HMLP(LasagnePowered, Serializable):
    def __init__(self, hidden_sizes, hidden_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 subnet_size = (16,16), subnet_nonlinearity=LN.tanh, subnet_W_init=LI.GlorotUniform(), subnet_b_init=LI.Constant(0.),
                 name=None, input_shape=None, option_dim = 2, subnet_split1 = [2,3,4,11,12,13], subnet_split2=[5,6,7, 14,15,16], hlc_output_dim = 0, sub_out_dim = 3):

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
        l_hid.get_params()

        l_options = L.DenseLayer(
                l_hid,
                num_units=option_dim*2+hlc_output_dim,
                nonlinearity=hidden_nonlinearity,
                name="%soptions" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_leg1 = SplitLayer(l_in, subnet_split1)
        l_option1 = SplitLayer(l_options, np.arange(0, option_dim))
        l_concat1 = L.concat([l_leg1, l_option1])

        l_leg2 = SplitLayer(l_in, subnet_split2)
        l_option2 = SplitLayer(l_options, np.arange(option_dim, 2*option_dim))
        l_concat2 = L.concat([l_leg2, l_option2])
        self._layers.append(l_options)
        self._layers.append(l_leg1)
        self._layers.append(l_option1)
        self._layers.append(l_concat1)
        self._layers.append(l_leg2)
        self._layers.append(l_option2)
        self._layers.append(l_concat2)

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
            #l_s_concat1 = L.concat([l_snet, l_option1])
            self._layers.append(l_snet)
            #self._layers.append(l_s_concat1)

            l_snet2 = L.DenseLayer(
                l_snet2,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_2_%d" % (prefix, idx),
                W=l_snet.W,
                b=l_snet.b,
            )
            #l_s_concat2 = L.concat([l_snet2, l_option2])
            self._layers.append(l_snet2)
            #self._layers.append(l_s_concat2)

            #l_snet = l_s_concat1
            #l_snet2 = l_s_concat2

        l_out1 = L.DenseLayer(
            l_snet,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput1" % (prefix,),
            W=subnet_W_init,
            b=subnet_b_init,
        )
        self._layers.append(l_out1)

        l_out2 = L.DenseLayer(
            l_snet2,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput2" % (prefix,),
            W=l_out1.W,
            b=l_out1.b,
        )
        self._layers.append(l_out2)

        if not hlc_output_dim == 0:
            l_out_hlc = SplitLayer(l_options, np.arange(option_dim*2, 2*option_dim + hlc_output_dim))
            self._layers.append(l_out_hlc)
            l_out = L.concat([l_out_hlc, l_out1, l_out2])
        else:
            l_out = L.concat([l_out1, l_out2])


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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class HMLPPhase(LasagnePowered, Serializable):
    def __init__(self, hidden_sizes, hidden_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 subnet_size = (16,16), subnet_nonlinearity=LN.tanh, subnet_W_init=LI.GlorotUniform(), subnet_b_init=LI.Constant(0.),
                 name=None, input_shape=None, option_dim = 2, subnet_split1 = [2,3,4,11,12,13], subnet_split2=[5,6,7, 14,15,16], hlc_output_dim = 0, sub_out_dim = 3):

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
        l_hid.get_params()

        l_options = L.DenseLayer(
                l_hid,
                num_units=option_dim*2+hlc_output_dim,
                nonlinearity=hidden_nonlinearity,
                name="%soptions" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_time = SplitLayer(l_in, [-1])
        l_phase1 = L.concat([PhaseLayer(l_time, 2*np.pi, 0), PhaseLayer(l_time, np.pi, 0), PhaseLayer(l_time, 4*np.pi, 0)])
        l_phase2 = L.concat([PhaseLayer(l_time, 2*np.pi, np.pi), PhaseLayer(l_time, np.pi, np.pi), PhaseLayer(l_time, 4*np.pi, np.pi)])
        self._layers.append(l_phase1)
        self._layers.append(l_phase2)

        l_leg1 = SplitLayer(l_in, subnet_split1)
        l_option1 = L.concat([SplitLayer(l_options, np.arange(0, option_dim)), l_phase1])
        l_concat1 = L.concat([l_leg1, l_option1])

        l_leg2 = SplitLayer(l_in, subnet_split2)
        l_option2 = L.concat([SplitLayer(l_options, np.arange(option_dim, 2*option_dim)), l_phase2])
        l_concat2 = L.concat([l_leg2, l_option2])
        self._layers.append(l_options)
        self._layers.append(l_leg1)
        self._layers.append(l_option1)
        self._layers.append(l_concat1)
        self._layers.append(l_leg2)
        self._layers.append(l_option2)
        self._layers.append(l_concat2)

        l_snet = l_option1
        l_snet2 = l_option2
        for idx, size in enumerate(subnet_size):
            l_snet = L.DenseLayer(
                l_snet,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_1_%d" % (prefix, idx),
                W=subnet_W_init,
                b=subnet_b_init,
            )
            l_s_concat1 = L.concat([l_snet, l_option1])
            self._layers.append(l_snet)
            self._layers.append(l_s_concat1)

            l_snet2 = L.DenseLayer(
                l_snet2,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_2_%d" % (prefix, idx),
                W=l_snet.W,
                b=l_snet.b,
            )
            l_s_concat2 = L.concat([l_snet2, l_option2])
            self._layers.append(l_snet2)
            self._layers.append(l_s_concat2)

            #l_snet = l_s_concat1
            #l_snet2 = l_s_concat2

        l_out1 = L.DenseLayer(
            l_snet,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput1" % (prefix,),
            W=subnet_W_init,
            b=subnet_b_init,
        )
        self._layers.append(l_out1)

        l_out2 = L.DenseLayer(
            l_snet2,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput2" % (prefix,),
            W=l_out1.W,
            b=l_out1.b,
        )
        self._layers.append(l_out2)

        if not hlc_output_dim == 0:
            l_out_hlc = SplitLayer(l_options, np.arange(option_dim*2, 2*option_dim + hlc_output_dim))
            self._layers.append(l_out_hlc)
            l_out = L.concat([l_out_hlc, l_out1, l_out2])
        else:
            l_out = L.concat([l_out1, l_out2])


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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class HLC(LasagnePowered, Serializable):
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
        l_hid.get_params()

        l_out = L.DenseLayer(
                l_hid,
                num_units=option_dim*2,
                nonlinearity=hidden_nonlinearity,
                name="%soptions" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class LLC(LasagnePowered, Serializable):
    def __init__(self, obs_dim, hidden_sizes, hidden_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 subnet_size = (16,16), subnet_nonlinearity=LN.tanh, subnet_W_init=LI.GlorotUniform(), subnet_b_init=LI.Constant(0.),
                 name=None, input_shape=None, option_dim = 2, subnet_split1 = [2,3,4,11,12,13], subnet_split2=[5,6,7, 14,15,16], sub_out_dim = 3):

        Serializable.quick_init(self, locals())

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        l_in = L.InputLayer(shape=(None,) + input_shape)
        self._layers = [l_in]

        l_leg1 = SplitLayer(l_in, subnet_split1)
        l_option1 = SplitLayer(l_in, range(obs_dim, obs_dim+option_dim))
        l_concat1 = L.concat([l_leg1, l_option1])

        l_leg2 = SplitLayer(l_in, subnet_split2)
        l_option2 = SplitLayer(l_in, range(obs_dim+option_dim, obs_dim+2*option_dim))
        l_concat2 = L.concat([l_leg2, l_option2])
        self._layers.append(l_leg1)
        self._layers.append(l_option1)
        self._layers.append(l_concat1)
        self._layers.append(l_leg2)
        self._layers.append(l_option2)
        self._layers.append(l_concat2)

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
            l_s_concat1 = L.concat([l_snet, l_option1])
            self._layers.append(l_snet)
            self._layers.append(l_s_concat1)

            l_snet2 = L.DenseLayer(
                l_snet2,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_2_%d" % (prefix, idx),
                W=l_snet.W,
                b=l_snet.b,
            )
            l_s_concat2 = L.concat([l_snet2, l_option2])
            self._layers.append(l_snet2)
            self._layers.append(l_s_concat2)

            l_snet = l_s_concat1
            l_snet2 = l_s_concat2


        l_out1 = L.DenseLayer(
            l_snet,
            num_units=sub_out_dim,
            nonlinearity=None,
            name="%soutput1" % (prefix,),
            W=subnet_W_init,
            b=subnet_b_init,
        )
        self._layers.append(l_out1)

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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class HMLP_PROP(LasagnePowered, Serializable):
    def __init__(self, hidden_sizes, hidden_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 subnet_size = (16,16), subnet_nonlinearity=LN.tanh, subnet_W_init=LI.GlorotUniform(), subnet_b_init=LI.Constant(0.),
                 name=None, input_shape=None, option_dim = 2, subnet_split1 = [2,3,4,11,12,13], subnet_split2=[5,6,7, 14,15,16], sub_out_dim = 3):

        Serializable.quick_init(self, locals())

        self.use_proprioceptive_sensing = theano.shared(1)

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
        l_hid.get_params()

        l_options = L.DenseLayer(
                l_hid,
                num_units=option_dim*2,
                nonlinearity=hidden_nonlinearity,
                name="%soptions" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_leg1 = SplitLayer(l_in, subnet_split1, scale=self.use_proprioceptive_sensing)
        l_option1 = SplitLayer(l_options, np.arange(0, option_dim))
        l_concat1 = L.concat([l_leg1, l_option1])

        l_leg2 = SplitLayer(l_in, subnet_split2, scale=self.use_proprioceptive_sensing)
        l_option2 = SplitLayer(l_options, np.arange(option_dim, 2*option_dim))
        l_concat2 = L.concat([l_leg2, l_option2])
        self._layers.append(l_options)
        self._layers.append(l_leg1)
        self._layers.append(l_option1)
        self._layers.append(l_concat1)
        self._layers.append(l_leg2)
        self._layers.append(l_option2)
        self._layers.append(l_concat2)

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
            if idx == 0:
                p = l_snet.W.get_value(borrow=True)
                p[0:len(subnet_split1), :] = 0
                l_snet.W.set_value(p)

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

        self._l_in = l_in
        self._l_out = l_out
        # self._input_var = l_in.input_var
        self._output = L.get_output(l_out)

        self.val_contact1 = L.get_output(l_concat1)
        self.val_contact2 = L.get_output(l_concat2)

        LasagnePowered.__init__(self, [l_out])

    def set_use_propsensing(self, use_prop):
        if use_prop:
            self.use_proprioceptive_sensing.set_value(1)
        else:
            self.use_proprioceptive_sensing.set_value(0)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

class HMLPPhaseHumanoid(LasagnePowered, Serializable):
    def __init__(self, hidden_sizes, hidden_nonlinearity, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 subnet_size = (16,16), subnet_nonlinearity=LN.tanh, subnet_W_init=LI.GlorotUniform(), subnet_b_init=LI.Constant(0.),
                 name=None, input_shape=None, option_dim = 2, hlc_output_dim = 0, sub_out_dim1 = 4, sub_out_dim2 = 3,
                 subnet_split1 = [], subnet_split2=[], subnet_split3=[], subnet_split4=[]):

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
        l_hid.get_params()

        l_options = L.DenseLayer(
                l_hid,
                num_units=option_dim*4+hlc_output_dim,
                nonlinearity=hidden_nonlinearity,
                name="%soptions" % (prefix),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_time = SplitLayer(l_in, [-1])
        l_phase1 = L.concat([PhaseLayer(l_time, 2*np.pi, 0), PhaseLayer(l_time, np.pi, 0), PhaseLayer(l_time, 4*np.pi, 0)])
        l_phase2 = L.concat([PhaseLayer(l_time, 2*np.pi, np.pi), PhaseLayer(l_time, np.pi, np.pi), PhaseLayer(l_time, 4*np.pi, np.pi)])
        self._layers.append(l_phase1)
        self._layers.append(l_phase2)

        l_option1 = L.concat([SplitLayer(l_options, np.arange(0, option_dim)), l_phase1])
        l_option2 = L.concat([SplitLayer(l_options, np.arange(option_dim, 2*option_dim)), l_phase2])
        l_option3 = L.concat([SplitLayer(l_options, np.arange(option_dim*2, option_dim*3)), l_phase2])
        l_option4 = L.concat([SplitLayer(l_options, np.arange(option_dim*3, option_dim*4)), l_phase1])
        self._layers.append(l_options)
        self._layers.append(l_option1)
        self._layers.append(l_option2)
        self._layers.append(l_option3)
        self._layers.append(l_option4)

        l_snet = l_option1
        l_snet2 = l_option2
        l_snet3 = l_option3
        l_snet4 = l_option4
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

            l_snet3 = L.DenseLayer(
                l_snet3,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_3_%d" % (prefix, idx),
                W=subnet_W_init,
                b=subnet_b_init,
            )
            self._layers.append(l_snet3)

            l_snet4 = L.DenseLayer(
                l_snet4,
                num_units=size,
                nonlinearity=subnet_nonlinearity,
                name="%ssnet_4_%d" % (prefix, idx),
                W=l_snet3.W,
                b=l_snet3.b,
            )
            self._layers.append(l_snet4)


        l_out1 = L.DenseLayer(
            l_snet,
            num_units=sub_out_dim1,
            nonlinearity=None,
            name="%soutput1" % (prefix,),
            W=subnet_W_init,
            b=subnet_b_init,
        )
        self._layers.append(l_out1)

        l_out2 = L.DenseLayer(
            l_snet2,
            num_units=sub_out_dim1,
            nonlinearity=None,
            name="%soutput2" % (prefix,),
            W=l_out1.W,
            b=l_out1.b,
        )
        self._layers.append(l_out2)

        l_out3 = L.DenseLayer(
            l_snet3,
            num_units=sub_out_dim2,
            nonlinearity=None,
            name="%soutput3" % (prefix,),
            W=subnet_W_init,
            b=subnet_b_init,
        )
        self._layers.append(l_out3)

        l_out4 = L.DenseLayer(
            l_snet4,
            num_units=sub_out_dim2,
            nonlinearity=None,
            name="%soutput4" % (prefix,),
            W=l_out3.W,
            b=l_out3.b,
        )
        self._layers.append(l_out4)

        if not hlc_output_dim == 0:
            l_out_hlc = SplitLayer(l_options, np.arange(option_dim*4, 4*option_dim + hlc_output_dim))
            self._layers.append(l_out_hlc)
            l_out = L.concat([l_out_hlc, l_out1, l_out2, l_out3, l_out4])
        else:
            l_out = L.concat([l_out1, l_out2, l_out3, l_out4])


        self._layers.append(l_out)

        self._l_in = l_in
        self._l_out = l_out
        # self._input_var = l_in.input_var
        self._output = L.get_output(l_out)

        self.hlc_signal1 = L.get_output(l_option1)
        self.hlc_signal2 = L.get_output(l_option2)

        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


# With Model Parameter Selection
class MLP_PS(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, mp_dim, mp_sel_hid_dim, mp_sel_num, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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

        l_mp_in = SplitLayer(l_in, range(-mp_dim, 0))
        self._layers.append(l_mp_in)

        # selection part
        l_mp_hid = L.DenseLayer(
            l_mp_in,
            num_units=mp_sel_hid_dim,
            nonlinearity=hidden_nonlinearity,
            name="%smp_hidden" % (prefix),
            W=hidden_W_init,
            b=hidden_b_init,
        )
        l_blendweights = L.DenseLayer(
            l_mp_hid,
            num_units=mp_sel_num,
            nonlinearity=LN.softmax,
            name="%sblend_weights" % (prefix,),
            W=hidden_W_init,
            b=hidden_b_init,
        )
        self._layers.append(l_mp_hid)
        self._layers.append(l_blendweights)

        blended_input =[]
        # merge selection with input
        for i in range(mp_sel_num):
            blend_weight = SplitLayer(l_blendweights, [i])
            extended_weights = L.concat([blend_weight]*input_shape[0])
            blended_input.append(ElemwiseMultLayer([l_in, extended_weights]))

        l_blended_iputs = L.concat(blended_input)
        self._layers.append(l_blended_iputs)


        l_hid = l_blended_iputs
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
        self._blend_weights = L.get_output(l_blendweights)
        self.l_blend_weights = l_blendweights
        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

# With Model Parameter Selection
class MLP_PROJ(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, mp_dim, mp_hid_dim, mp_proj_dim, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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


        l_prop_in = SplitLayer(l_in, range(0, input_shape[0] - mp_dim))
        l_mp_in = SplitLayer(l_in, range(-mp_dim, 0))
        self._layers.append(l_prop_in)
        self._layers.append(l_mp_in)

        # projection part
        l_proj_hid = L.DenseLayer(
            l_mp_in,
            num_units=mp_hid_dim,
            nonlinearity=hidden_nonlinearity,
            name="%sproj_hidden" % (prefix),
            W=hidden_W_init,
            b=hidden_b_init,
        )
        l_proj_out = L.DenseLayer(
            l_proj_hid,
            num_units=mp_proj_dim,
            nonlinearity=hidden_nonlinearity,
            name="%sproj_out" % (prefix,),
            W=hidden_W_init,
            b=hidden_b_init,
        )
        self._layers.append(l_proj_hid)
        self._layers.append(l_proj_out)


        l_concat_input = L.concat([l_prop_in, l_proj_out])
        self._layers.append(l_concat_input)


        l_hid = l_concat_input
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
        self._projected_mp = L.get_output(l_proj_out)
        self.l_proj_out = l_proj_out
        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

class WeightConverter(LasagnePowered, Serializable):
    def __init__(self, input_dim=5):
        Serializable.quick_init(self, locals())
        intput_shape = (input_dim,)

        l_in = L.InputLayer(shape=(None,) + intput_shape, input_var=None, name='aux_input')
        self._layers = [l_in]
        l_hid_out = []

        l_hid = L.DenseLayer(
            l_in,
            num_units=128,
            nonlinearity=LN.tanh,
            name="wchid",
            W=LI.GlorotUniform(),
            b=LI.Constant(0.),
        )
        self._layers.append(l_hid)

        l_out = L.DenseLayer(
            l_hid,
            num_units=input_dim - 1,
            nonlinearity=LN.softmax,
            name="wcout",
            W=LI.GlorotUniform(),
            b=LI.Constant(0.),
        )

        self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        self._output = L.get_output(l_out)

        self.pred_weight = ext.compile_function(
            inputs=[self._l_in.input_var],
            outputs=[self._output],
        )

        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

# With Model Parameter Selection discrete
class MLP_PSD(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, mp_dim, mp_sel_hid_dim, mp_sel_num, wc_net, hidden_W_init=LI.GlorotUniform(),
                 hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None, input_layer=None, input_shape=None, batch_norm=False,
                 learn_segment=False):

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

        '''l_input = SplitLayer(l_in, range(0, input_shape[0]-1))
        l_mp_in = SplitLayer(l_in, range(-mp_dim-1, -1))'''
        l_input = SplitLayer(l_in, np.arange(0, input_shape[0] - 1))
        l_mp_in = SplitLayer(l_in, np.arange(input_shape[0] - 1 - mp_dim, input_shape[0] - 1))
        l_rand_in = SplitLayer(l_in, [input_shape[0] - 1])
        self._layers.append(l_mp_in)
        self._layers.append(l_input)
        self._layers.append(l_rand_in)

        # selection part
        l_mp_hid = L.DenseLayer(
            l_mp_in,
            num_units=mp_sel_hid_dim,
            nonlinearity=hidden_nonlinearity,
            name="%smp_hidden" % (prefix),
            W=hidden_W_init,
            b=hidden_b_init,
        )
        l_blendweights = L.DenseLayer(
            l_mp_hid,
            num_units=mp_sel_num,
            nonlinearity=LN.softmax,
            name="%sblend_weights" % (prefix,),
            W=hidden_W_init,
            b=hidden_b_init,
        )
        self._layers.append(l_mp_hid)
        self._layers.append(l_blendweights)

        l_hidwc = L.concat([l_blendweights, l_rand_in])
        for h in range(len(wc_net.layers) - 1):
            l_hidwc = L.DenseLayer(
                l_hidwc,
                num_units=wc_net.layers[h + 1].num_units,
                nonlinearity=wc_net.layers[h + 1].nonlinearity,
                name="wc_hidden_%d" % (h),
                W=wc_net.layers[h + 1].W,
                b=wc_net.layers[h + 1].b,
            )
            l_hidwc.params[l_hidwc.W].remove('trainable')
            l_hidwc.params[l_hidwc.b].remove('trainable')
            self._layers.append(l_hidwc)

        blended_input = []
        # merge selection with input
        for i in range(mp_sel_num):
            blend_weight = SplitLayer(l_hidwc, [i])
            extended_weights = L.concat([blend_weight] * (input_shape[0] - 1))
            blended_input.append(ElemwiseMultLayer([l_input, extended_weights]))

        l_blended_iputs = L.concat(blended_input)
        self._layers.append(l_blended_iputs)

        l_hid = l_blended_iputs
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
            if idx > 0 and learn_segment:
                l_hid.params[l_hid.W].remove('trainable')
                l_hid.params[l_hid.b].remove('trainable')
            self._layers.append(l_hid)

        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        if learn_segment:
            l_out.params[l_out.W].remove('trainable')
            l_out.params[l_out.b].remove('trainable')

        self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        # self._input_var = l_in.input_var
        self._output = L.get_output(l_out)
        self._blend_weights = L.get_output(l_blendweights)
        self.l_blend_weights = l_blendweights
        LasagnePowered.__init__(self, [l_out])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

# MLP that splits at one layer
class MLP_Split(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, split_layer, split_num, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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
        layer_id = 0

        l_input = SplitLayer(l_in, np.arange(0, input_shape[0] - split_num))
        l_split = SplitLayer(l_in, np.arange(input_shape[0] -split_num, input_shape[0]))
        self._layers.append(l_input)
        self._layers.append(l_split)

        l_hid = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            if layer_id in split_layer:
                split_layers = []
                for i in range(split_num):
                    l_hid_sub = L.DenseLayer(
                        l_hid,
                        num_units=hidden_size,
                        nonlinearity=hidden_nonlinearity,
                        name="%shidden_%d_%d" % (prefix, idx, i),
                        W=hidden_W_init,
                        b=hidden_b_init,
                    )
                    split_single_expand = L.concat([SplitLayer(l_split, [i])] * hidden_size)
                    l_hid_sub_mult = ElemwiseMultLayer([l_hid_sub, split_single_expand])
                    self._layers.append(l_hid_sub)
                    split_layers.append(l_hid_sub_mult)
                l_hid = L.ElemwiseSumLayer(split_layers)
            else:
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
            layer_id += 1

        if layer_id in split_layer:
            split_outputs = []
            for i in range(split_num):
                l_out_sub = L.DenseLayer(
                    l_hid,
                    num_units=output_dim,
                    nonlinearity=output_nonlinearity,
                    name="%soutput%d" % (prefix, i),
                    W=output_W_init,
                    b=output_b_init,
                )
                split_single_expand = L.concat([SplitLayer(l_split, [i])] * output_dim)
                l_output_mult = ElemwiseMultLayer([l_out_sub, split_single_expand])
                self._layers.append(l_out_sub)
                split_outputs.append(l_output_mult)

            l_out = L.ElemwiseSumLayer(split_outputs)
        else:
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="%soutput" % (prefix),
                W=output_W_init,
                b=output_b_init,
            )

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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class MLP_SplitAct(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, split_units, split_num, init_net, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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
        layer_id = 0

        l_input = SplitLayer(l_in, np.arange(0, input_shape[0] - split_num))
        l_split = SplitLayer(l_in, np.arange(input_shape[0] -split_num, input_shape[0]))
        self._layers.append(l_input)
        self._layers.append(l_split)

        l_hid = l_input
        split_indices = None
        shared_indices = None
        for idx, hidden_size in enumerate(hidden_sizes):
            initial_weights_W = init_net.layers[layer_id+1].get_params()[0].get_value()
            initial_weights_b = init_net.layers[layer_id + 1].get_params()[1].get_value()
            if layer_id-1 in np.array(split_units)[:, 0]: # permute the matrix if last layer was splitted
                original_weights_W = np.copy(initial_weights_W)
                initial_weights_W = np.vstack([original_weights_W[shared_indices, :], original_weights_W[split_indices, :]])

            if layer_id in np.array(split_units)[:, 0]:
                split_indices = (np.array(split_units)[:, 1][np.array(split_units)[:, 0]==layer_id]).tolist()
                shared_indices = list(set(np.arange(hidden_size).tolist()) - set(split_indices))
                split_layers = []
                for i in range(split_num):
                    l_hid_sub = L.DenseLayer(
                        l_hid,
                        num_units=len(split_indices),
                        nonlinearity=hidden_nonlinearity,
                        name="%shidden_split_%d_%d" % (prefix, idx, i),
                        W=hidden_W_init,
                        b=hidden_b_init,
                    )
                    l_hid_sub.get_params()[0].set_value(initial_weights_W[:, split_indices])
                    l_hid_sub.get_params()[1].set_value(initial_weights_b[split_indices])

                    split_single_expand = L.concat([SplitLayer(l_split, [i])] * len(split_indices))
                    l_hid_sub_mult = ElemwiseMultLayer([l_hid_sub, split_single_expand])
                    self._layers.append(l_hid_sub)
                    split_layers.append(l_hid_sub_mult)
                l_hid_split_sum = L.ElemwiseSumLayer(split_layers)
                if len(shared_indices) != 0:
                    l_hid_share = L.DenseLayer(
                        l_hid,
                        num_units=len(shared_indices),
                        nonlinearity=hidden_nonlinearity,
                        name="%shidden_share_%d" % (prefix, idx),
                        W=hidden_W_init,
                        b=hidden_b_init,
                    )
                    l_hid_share.get_params()[0].set_value(initial_weights_W[:, shared_indices])
                    l_hid_share.get_params()[1].set_value(initial_weights_b[shared_indices])
                    self._layers.append(l_hid_share)
                    l_hid = L.concat([l_hid_share, l_hid_split_sum])
                else:
                    l_hid = l_hid_split_sum
            else:
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="%shidden_share_%d" % (prefix, idx),
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
                if batch_norm:
                    l_hid = L.batch_norm(l_hid)
                l_hid.get_params()[0].set_value(initial_weights_W)
                l_hid.get_params()[1].set_value(initial_weights_b)
            self._layers.append(l_hid)
            layer_id += 1

        initial_weights_W = init_net.layers[layer_id + 1].get_params()[0].get_value()
        initial_weights_b = init_net.layers[layer_id + 1].get_params()[1].get_value()
        if layer_id - 1 in np.array(split_units)[:, 0]:  # permute the matrix if last layer was splitted
            original_weights_W = np.copy(initial_weights_W)
            initial_weights_W = np.vstack([original_weights_W[shared_indices, :], original_weights_W[split_indices, :]])

        if layer_id in np.array(split_units)[:, 0]:
            split_indices = (np.array(split_units)[:, 1][np.array(split_units)[:, 0] == layer_id]).tolist()
            shared_indices = list(set(np.arange(output_dim).tolist()) - set(split_indices))
            split_outputs = []
            for i in range(split_num):
                l_out_sub = L.DenseLayer(
                    l_hid,
                    num_units=len(split_indices),
                    nonlinearity=output_nonlinearity,
                    name="%soutput_split%d" % (prefix, i),
                    W=output_W_init,
                    b=output_b_init,
                )
                l_out_sub.get_params()[0].set_value(initial_weights_W[:, split_indices])
                l_out_sub.get_params()[1].set_value(initial_weights_b[split_indices])
                split_single_expand = L.concat([SplitLayer(l_split, [i])] * len(split_indices))
                l_output_mult = ElemwiseMultLayer([l_out_sub, split_single_expand])
                self._layers.append(l_out_sub)
                split_outputs.append(l_output_mult)
            l_out_split_sum = L.ElemwiseSumLayer(split_outputs)

            if len(shared_indices) != 0:
                l_out_share = L.DenseLayer(
                    l_hid,
                    num_units=len(shared_indices),
                    nonlinearity=output_nonlinearity,
                    name="%soutput_share" % (prefix),
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
                l_out_share.get_params()[0].set_value(initial_weights_W[:, shared_indices])
                l_out_share.get_params()[1].set_value(initial_weights_b[shared_indices])
                self._layers.append(l_out_share)
                new_order = shared_indices + split_indices
                recover_order = []
                for idx in range(len(new_order)):
                    recover_order.append(new_order.index(idx))
                l_out = SplitLayer(L.concat([l_out_share, l_out_split_sum]), recover_order)
            else:
                l_out = l_out_split_sum
        else:
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="%soutput_share" % (prefix),
                W=output_W_init,
                b=output_b_init,
            )
            l_out.get_params()[0].set_value(initial_weights_W)
            l_out.get_params()[1].set_value(initial_weights_b)

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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

class MLP_SoftSplit(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, split_num, init_net, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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
        layer_id = 0

        l_input = SplitLayer(l_in, np.arange(0, input_shape[0] - split_num))
        l_split = SplitLayer(l_in, np.arange(input_shape[0] -split_num, input_shape[0]))
        self._layers.append(l_input)
        self._layers.append(l_split)

        l_hid = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            initial_weights_W = init_net.layers[layer_id+1].get_params()[0].get_value()
            initial_weights_b = init_net.layers[layer_id + 1].get_params()[1].get_value()

            split_layers = []
            for i in range(split_num):
                l_hid_sub = L.DenseLayer(
                    l_hid,
                    num_units=(hidden_size),
                    nonlinearity=hidden_nonlinearity,
                    name="%shidden_split_%d_copy%d" % (prefix, idx, i),
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
                l_hid_sub.get_params()[0].set_value(initial_weights_W)
                l_hid_sub.get_params()[1].set_value(initial_weights_b)

                split_single_expand = L.concat([SplitLayer(l_split, [i])] * (hidden_size))
                l_hid_sub_mult = ElemwiseMultLayer([l_hid_sub, split_single_expand])
                self._layers.append(l_hid_sub)
                split_layers.append(l_hid_sub_mult)
            l_hid_split_sum = L.ElemwiseSumLayer(split_layers)
            l_hid = l_hid_split_sum

            self._layers.append(l_hid)
            layer_id += 1

        initial_weights_W = init_net.layers[layer_id + 1].get_params()[0].get_value()
        initial_weights_b = init_net.layers[layer_id + 1].get_params()[1].get_value()


        split_outputs = []
        for i in range(split_num):
            l_out_sub = L.DenseLayer(
                l_hid,
                num_units=(output_dim),
                nonlinearity=output_nonlinearity,
                name="%soutput_split_copy%d" % (prefix, i),
                W=output_W_init,
                b=output_b_init,
            )
            l_out_sub.get_params()[0].set_value(initial_weights_W)
            l_out_sub.get_params()[1].set_value(initial_weights_b)
            split_single_expand = L.concat([SplitLayer(l_split, [i])] * (output_dim))
            l_output_mult = ElemwiseMultLayer([l_out_sub, split_single_expand])
            self._layers.append(l_out_sub)
            split_outputs.append(l_output_mult)
        l_out_split_sum = L.ElemwiseSumLayer(split_outputs)
        l_out = l_out_split_sum

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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

    def get_split_parameter(self, split_id):
        params = self.get_params()
        return_params = []
        for param in params:
            if 'copy%d'%(split_id) in param.name:
                return_params.append(param)
        return return_params

class MLP_MaskedSplit(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, split_num, split_masks, init_net, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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
        layer_id = 0

        l_input = SplitLayer(l_in, np.arange(0, input_shape[0] - split_num))
        l_split = SplitLayer(l_in, np.arange(input_shape[0] -split_num, input_shape[0]))
        self._layers.append(l_input)
        self._layers.append(l_split)

        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):
            initial_weights_W = init_net.layers[layer_id+1].get_params()[0].get_value()
            initial_weights_b = init_net.layers[layer_id + 1].get_params()[1].get_value()

            split_layers = []
            l_hid = MaskedDenseLayer(
                l_hid,
                num_units=(hidden_size),
                nonlinearity=hidden_nonlinearity,
                W_init= initial_weights_W,
                b_init= initial_weights_b,
                split_num = split_num,
                split_mask_W = split_masks[layer_id*2],
                split_mask_b = split_masks[layer_id*2+1],
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
            self._layers.append(l_hid)
            l_hid = L.concat([l_hid, l_split])

            layer_id += 1

        initial_weights_W = init_net.layers[layer_id + 1].get_params()[0].get_value()
        initial_weights_b = init_net.layers[layer_id + 1].get_params()[1].get_value()

        split_outputs = []

        l_out = MaskedDenseLayer(
            l_hid,
            num_units=(output_dim),
            nonlinearity=output_nonlinearity,
            W_init= initial_weights_W,
            b_init= initial_weights_b,
            split_num = split_num,
            split_mask_W = split_masks[layer_id*2],
            split_mask_b = split_masks[layer_id*2+1],
            name="%soutput" % (prefix),
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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class MLP_MaskedSplitCont(LasagnePowered, Serializable):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, task_id, init_net, hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
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
        layer_id = 0

        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):
            split_layers = []
            l_hid = MaskedDenseLayerCont(
                l_hid,
                num_units=(hidden_size),
                nonlinearity=hidden_nonlinearity,
                init_layer=init_net.layers[idx+3],
                task_id=task_id,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
            self._layers.append(l_hid)

            layer_id += 1

        split_outputs = []

        l_out = MaskedDenseLayerCont(
            l_hid,
            num_units=(output_dim),
            nonlinearity=output_nonlinearity,
            init_layer=init_net.layers[-1],
            task_id=task_id,
            name="%soutput" % (prefix),
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

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output