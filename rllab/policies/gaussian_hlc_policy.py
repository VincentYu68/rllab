__author__ = 'yuwenhao'

import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP, HMLP, HLC, LLC
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT


class GaussianHLCPolicy(GaussianMLPPolicy):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            mean_network=None,
            lowlevelnetwork=None,
            std_network=None,
            dist_cls=DiagonalGaussian,
            subnet_split1=[2, 3, 4, 11, 12, 13],
            subnet_split2=[5, 6, 7, 14, 15, 16],
            sub_out_dim=3,
            option_dim=4,
    ):

        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        if mean_network is None:
            mean_network = HLC(
                hidden_sizes,
                hidden_nonlinearity,
                input_shape=(obs_dim,),
                subnet_split1=subnet_split1,
                subnet_split2=subnet_split2,
                sub_out_dim=sub_out_dim,
                option_dim=option_dim,
            )
        self._mean_network = mean_network

        if lowlevelnetwork is None:
            lowlevelnetwork = LLC(
                    obs_dim,
                    hidden_sizes,
                    hidden_nonlinearity,
                    input_shape=(obs_dim+option_dim*2,),
                    subnet_split1=subnet_split1,
                    subnet_split2=subnet_split2,
                    sub_out_dim=sub_out_dim,
                    option_dim=option_dim,
                )
        self._lowlevelnetwork = lowlevelnetwork

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_layer.input_var

        if std_network is not None:
            l_log_std = std_network.output_layer
        else:
            if adaptive_std:
                std_network = MLP(
                    input_shape=(obs_dim,),
                    input_layer=mean_network.input_layer,
                    output_dim=option_dim*2,
                    hidden_sizes=std_hidden_sizes,
                    hidden_nonlinearity=std_hidden_nonlinearity,
                    output_nonlinearity=None,
                )
                l_log_std = std_network.output_layer
            else:
                l_log_std = ParamLayer(
                    mean_network.input_layer,
                    num_units=option_dim*2,
                    param=lasagne.init.Constant(np.log(init_std)),
                    name="output_log_std",
                    trainable=learn_std,
                )

        self.min_std = min_std

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(min_std))

        self._mean_var, self._log_std_var = mean_var, log_std_var

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = dist_cls(option_dim*2)

        LasagnePowered.__init__(self, [l_mean, l_log_std, self._lowlevelnetwork.output_layer])
        super(GaussianMLPPolicy, self).__init__(env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

        self.llc_out = ext.compile_function(
            inputs=[self._lowlevelnetwork.input_layer.input_var],
            outputs=[L.get_output(self._lowlevelnetwork.output_layer)]
        )


    def lowlevel_action(self, observation, option):
        return self.llc_out([np.hstack([observation, option])])[0][0]
