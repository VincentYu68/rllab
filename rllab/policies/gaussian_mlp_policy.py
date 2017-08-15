import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer, ParamLayerSplit
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP, MLPAppend, MLP_PS, MLP_PROJ, MLP_PSD, MLP_Split, MLP_SplitAct, MLP_SoftSplit, MLP_MaskedSplit
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT
import theano as T

import joblib

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * TT.switch(x > 0, x, alpha * TT.expm1(x))

class GaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
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
            std_network=None,
            split_masks=None,
            dist_cls=DiagonalGaussian,
            mp_dim = 0,
            mp_sel_hid_dim = 0,
            mp_sel_num = 0,
            mp_projection_dim = 2,
            net_mode = 0, # 0: vanilla, 1: append mp to second layer, 2: project mp to lower space, 3: mp selection blending, 4: mp selection discrete
            split_init_net=None,
            split_units=None,
            wc_net_path = None,
            learn_segment = False,
            split_num = 1,
            split_layer=[0],
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        if mean_network is None:
            if net_mode == 1:
                mean_network = MLPAppend(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    append_dim=mp_dim,
                )
            elif net_mode == 2:
                mean_network = MLP_PROJ(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    mp_dim=mp_dim,
                    mp_hid_dim=16,
                    mp_proj_dim=mp_projection_dim,
                )
            elif net_mode == 3:
                mean_network = MLP_PS(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    mp_dim=mp_dim,
                    mp_sel_hid_dim=mp_sel_hid_dim,
                    mp_sel_num=mp_sel_num,
                )
            elif net_mode == 4:
                wc_net = joblib.load(wc_net_path)
                mean_network = MLP_PSD(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    mp_dim=mp_dim,
                    mp_sel_hid_dim=mp_sel_hid_dim,
                    mp_sel_num=mp_sel_num,
                    wc_net=wc_net,
                    learn_segment = learn_segment,
                )
            elif net_mode == 5:
                mean_network = MLP_Split(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    split_layer=split_layer,
                    split_num=split_num,
                )
            elif net_mode == 6:
                mean_network = MLP_SplitAct(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    split_num=split_num,
                    split_units=split_units,
                    init_net=split_init_net._mean_network,
                )
            elif net_mode == 7:
                mean_network = MLP_SoftSplit(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    split_num=split_num,
                    init_net=split_init_net._mean_network,
                )
            elif net_mode == 8:
                mean_network = MLP_MaskedSplit(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    split_num=split_num,
                    split_masks=split_masks,
                    init_net=split_init_net._mean_network,
                )
            else:
                mean_network = MLP(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                )
        self._mean_network = mean_network

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_layer.input_var

        if std_network is not None:
            l_log_std = std_network.output_layer
        else:
            if adaptive_std:
                std_network = MLP(
                    input_shape=(obs_dim,),
                    input_layer=mean_network.input_layer,
                    output_dim=action_dim,
                    hidden_sizes=std_hidden_sizes,
                    hidden_nonlinearity=std_hidden_nonlinearity,
                    output_nonlinearity=None,
                )
                l_log_std = std_network.output_layer
            else:
                if net_mode != 18:
                    l_log_std = ParamLayer(
                        mean_network.input_layer,
                        num_units=action_dim,
                        param=lasagne.init.Constant(np.log(init_std)),
                        name="output_log_std",
                        trainable=learn_std,
                    )
                else:
                    l_log_std = ParamLayerSplit(
                        mean_network.input_layer,
                        num_units=action_dim,
                        param=lasagne.init.Constant(np.log(init_std)),
                        name="output_log_std",
                        trainable=learn_std,
                        split_num = split_num,
                        init_param=split_init_net.get_params()[-1]
                    )
                if net_mode == 6 or net_mode == 7 or net_mode == 8:
                    l_log_std.get_params()[0].set_value(split_init_net.get_params()[-1].get_value())

        self.min_std = min_std

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(min_std))

        self._mean_var, self._log_std_var = mean_var, log_std_var

        self._l_mean = l_mean
        self._l_log_std = l_log_std
        self._dist = dist_cls(action_dim)

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMLPPolicy, self).__init__(env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

        if net_mode == 3 or net_mode == 4:
            self._f_blendweight = ext.compile_function(
                inputs = [obs_var],
                outputs=[self._mean_network._blend_weights]
            )
            entropy = -TT.mean(self._mean_network._blend_weights * TT.log(self._mean_network._blend_weights))
            self._f_weightentropy = ext.compile_function(
                inputs = [obs_var],
                outputs=[entropy]
            )
            avg_weights = TT.mean(self._mean_network._blend_weights, axis=0)
            entropy2 = -TT.mean(avg_weights * TT.log(avg_weights))
            self._f_choiceentropy = ext.compile_function(
                inputs=[obs_var],
                outputs=[entropy2]
            )


    # average entropy of the blend weight
    def bw_entropy(self, obs_var):
        blend_weights = L.get_output(self._mean_network.l_blend_weights, obs_var)
        entropy = -TT.mean(blend_weights * TT.log(blend_weights))
        return entropy

    # average entropy of the blend weight across each sample
    def bw_choice_entropy(self, obs_var):
        blend_weights = L.get_output(self._mean_network.l_blend_weights, obs_var)
        avg_weights = TT.mean(blend_weights, axis=0)
        entropy = -TT.mean(avg_weights * TT.log(avg_weights))
        return entropy

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (TT.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * TT.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
