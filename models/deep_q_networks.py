"""DeepQNetwork models for molecule generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf


class DeepQNetwork(object):
    def __init__(self, hparams, q_fn, reuse=None):
        """Creates the model function.
            Args:
            input_shape: Tuple. The shape of input.
            q_fn: A function, whose input is the observation features, and the
            output is the Q value of the observation.
            scope: String or VariableScope. Variable Scope.
            reuse: Boolean or None. Whether or not the variable should be reused.
        """
        self.hparams = hparams
        self.q_fn = q_fn
        self.input_shape = (hparams['train_param']['batch_size'],
                            hparams['ingredient_param']['fingerprint_length'] + 1)
        self.learning_rate = self.hparams['train_param']['learning_rate']
        self.learning_rate_decay_steps = self.hparams['train_param']['learning_rate_decay_steps']
        self.learning_rate_decay_rate = self.hparams['train_param']['learning_rate_decay_rate']
        self.optimizer = self.hparams['train_param']['optimizer']
        self.grad_clipping = self.hparams['train_param']['grad_clipping']
        self.num_bootstrap_heads = self.hparams['model_param']['num_bootstrap_heads']
        self.double_q = self.hparams['model_param']['double_q']
        self.epsilon = self.hparams['model_param']['epsilon']
        self.gamma = self.hparams['model_param']['gamma']
        self.scope = self.hparams['model']
        self.reuse = reuse

    def build(self):
        """Builds the computational graph and training operations."""
        self._build_graph()
        self._build_training_ops()
        # self._build_summary_ops()

    def _build_single_q_network(self, observations, head, state_t, state_tp1,
                                done_mask, reward_t, error_weight):
        """Builds the computational graph for a single Q network.
        Briefly, this part is calculating the following two quantities:
        1. q_value = q_fn(observations)
        2. td_error = q_fn(state_t) - reward_t - gamma * q_fn(state_tp1)
        The optimization target is to minimize the td_error.
        Args:
          observations: shape = [batch_size, hparams.fingerprint_length].
            The input of the Q function.
          head: shape = [1].
            The index of the head chosen for decision in bootstrap DQN.
          state_t: shape = [batch_size, hparams.fingerprint_length].
            The state at time step t.
          state_tp1: a list of tensors, with total number of batch_size,
            each has shape = [num_actions, hparams.fingerprint_length].
            Note that the num_actions can be different for each tensor.
            The state at time step t+1, tp1 is short for t plus 1.
          done_mask: shape = [batch_size, 1]
            Whether state_tp1 is the terminal state.
          reward_t: shape = [batch_size, 1]
            the reward at time step t.
          error_weight: shape = [batch_size, 1]
            weight for the loss.
        Returns:
          q_values: Tensor of [batch_size, 1]. The q values for the observations.
          td_error: Tensor of [batch_size, 1]. The TD error.
          weighted_error: Tensor of [batch_size, 1]. The TD error weighted by error_weight.
          q_fn_vars: List of tf.Variables. The variables of q_fn when computing the q_values of state_t
          q_fn_vars: List of tf.Variables. The variables of q_fn when computing the q_values of state_tp1
        """
        with tf.variable_scope('q_fn'):
            # q_value have shape [batch_size, 1].
            q_values = tf.gather(self.q_fn(observations), head, axis=-1)

        # calculating q_fn(state_t)
        # The Q network shares parameters with the action graph.
        with tf.variable_scope('q_fn', reuse=True):
            q_t = self.q_fn(state_t, reuse=True)
        q_fn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_fn')

        # calculating q_fn(state_tp1)
        with tf.variable_scope('q_tp1', reuse=tf.AUTO_REUSE):
            q_tp1 = [self.q_fn(s_tp1, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1]
        q_tp1_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1')

        if self.double_q:
            with tf.variable_scope('q_fn', reuse=True):
                q_tp1_online = [self.q_fn(s_tp1, reuse=True) for s_tp1 in state_tp1]
            if self.num_bootstrap_heads:
                num_heads = self.num_bootstrap_heads
            else:
                num_heads = 1
            # determine the action to choose based on online Q estimator.
            q_tp1_online_idx = [tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1)
                                for q in q_tp1_online]
            # use the index from max online q_values to compute the value
            # function
            v_tp1 = tf.stack([tf.gather_nd(q, idx) for q, idx in zip(q_tp1, q_tp1_online_idx)], axis=0)
        else:
            v_tp1 = tf.stack([tf.reduce_max(q) for q in q_tp1], axis=0)

        # if s_{t+1} is the terminal state, we do not evaluate the Q value of the state.
        q_tp1_masked = (1.0 - done_mask) * v_tp1

        q_t_target = reward_t + self.gamma * q_tp1_masked

        # stop gradient from flowing to the computating graph which computes the Q value of s_{t+1}.
        # td_error has shape [batch_size, 1]
        td_error = q_t - tf.stop_gradient(q_t_target)

        # If use bootstrap, each head is trained with a different subset of the
        # training sample. Like the idea of dropout.
        if self.num_bootstrap_heads:
            head_mask = tf.keras.backend.random_binomial(shape=(1, self.num_bootstrap_heads), p=0.6)
            td_error = tf.reduce_mean(td_error * head_mask, axis=1)

        # The loss function : Huber loss
        # The l2 loss has the disadvantage that it has the tendency to be dominated by outliers.
        # In terms of estimation theory, the asymptotic relative efficiency of the l1 loss estimator is better
        # for heavy-tailed distributions.

        errors = tf.where(tf.abs(td_error) < 1.0, tf.square(td_error) * 0.5,
                          1.0 * (tf.abs(td_error) - 0.5))
        weighted_error = tf.reduce_mean(error_weight * errors)

        return q_values, td_error, weighted_error, q_fn_vars, q_tp1_vars

    def _build_input_placeholder(self):
        """Creates the input placeholders.
           Input placeholders created:
           observations: shape = [batch_size, hparams.fingerprint_length].
           The input of the Q function.
           head: shape = [1].
           The index of the head chosen for decision.
           state_t: shape = [batch_size, hparams.fingerprint_length].
           The state at time step t.
           state_tp1: a list of tensors,
           each has shape = [num_actions, hparams.fingerprint_length].
           Note that the num_actions can be different for each tensor.
           The state at time step t+1.
           done_mask: shape = [batch_size, 1]
           Whether state_tp1 is the terminal state.
           error_weight: shape = [batch_size, 1]
           weight for the loss.
        """
        batch_size, fingerprint_length = self.input_shape

        with tf.variable_scope(self.scope, reuse=self.reuse):
            # Build the action graph to choose an action.
            # The observations, which are the inputs of the Q function.
            self.observations = tf.placeholder(
                tf.float32, [None, fingerprint_length], name='observations')
            # head is the index of the head we want to choose for decison.
            # See https://arxiv.org/abs/1703.07608
            self.head = tf.placeholder(tf.int32, [], name='head')

            # When sample from memory, the batch_size can be fixed,
            # as it is possible to sample any number of samples from memory.
            # state_t is the state at time step t
            self.state_t = tf.placeholder(
                tf.float32, self.input_shape, name='state_t')
            # state_tp1 is the state at time step t + 1, tp1 is short for t plus 1.
            self.state_tp1 = [tf.placeholder(tf.float32, [None, fingerprint_length], name='state_tp1_%i' % i)
                              for i in range(batch_size)]
            # done_mask is a {0, 1} tensor indicating whether state_tp1 is the
            # terminal state.
            self.done_mask = tf.placeholder(
                tf.float32, (batch_size, 1), name='done_mask')

            self.error_weight = tf.placeholder(
                tf.float32, (batch_size, 1), name='error_weight')

    def _build_graph(self):
        """Builds the computational graph.
           Input placeholders created:
           reward_t: shape = [batch_size, 1]
           the reward at time step t.
           Instance attributes created:
           q_values: the q values of the observations.
           q_fn_vars: the variables in q function.
           q_tp1_vars: the variables in q_tp1 function.
           td_error: the td_error.
           weighted_error: the weighted td error.
           action: the action to choose next step.
        """
        batch_size, _ = self.input_shape
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self._build_input_placeholder()
            self.reward_t = tf.placeholder(
                tf.float32, (batch_size, 1), name='reward_t')
            # The Q network shares parameters with the action graph.
            # tenors start with q or v have shape [batch_size, 1] when not using bootstrap.
            # When using bootstrap, the shapes are [batch_size, num_bootstrap_heads]
            (self.q_values, self.td_error, self.weighted_error, self.q_fn_vars, self.q_tp1_vars) \
                = self._build_single_q_network(self.observations, self.head, self.state_t, self.state_tp1,
                                               self.done_mask, self.reward_t, self.error_weight)
            self.action = tf.argmax(self.q_values)

    def _build_training_ops(self):
        """Creates the training operations.
           Instance attributes created:
           optimization_op: the operation of optimize the loss.
           update_op: the operation to update the q network.
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.optimization_op = tf.contrib.layers.optimize_loss(
                loss=self.weighted_error,
                global_step=tf.train.get_or_create_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                clip_gradients=self.grad_clipping,
                learning_rate_decay_fn=functools.partial(tf.train.exponential_decay,
                                                         decay_steps=self.learning_rate_decay_steps,
                                                         decay_rate=self.learning_rate_decay_rate),
                variables=self.q_fn_vars)

            self.update_op = []
            for var, target in zip(sorted(self.q_fn_vars, key=lambda v: v.name),
                                   sorted(self.q_tp1_vars, key=lambda v: v.name)):
                self.update_op.append(target.assign(var))
            self.update_op = tf.group(*self.update_op)

    def _build_summary_ops(self):
        """Creates the summary operations.
           Input placeholders created:
           smiles: the smiles string.
           reward: the reward.
           Instance attributes created:
           error_summary: the operation to log the summary of error.
           episode_summary: the operation to log the smiles string and reward.
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.name_scope('summaries'):
                # The td_error here is the difference between q_t and q_t_target.
                # Without abs(), the summary of td_error is actually underestimated.
                self.error_summary = tf.summary.scalar(
                    'td_error', tf.reduce_mean(tf.abs(self.td_error)))
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.reward = tf.placeholder(tf.float32, [], 'summary_reward')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summary = tf.summary.scalar('reward', self.reward)
                self.episode_summary = tf.summary.merge(
                    [smiles_summary, reward_summary])

    def log_result(self, smiles, reward):
        """Summarizes the SMILES string and reward at the end of an episode.
           Args:
           smiles: String. The SMILES string.
           reward: Float. The reward.
           Returns:
           the summary protobuf
        """
        return tf.get_default_session().run(
            self.episode_summary,
            feed_dict={
                self.smiles: smiles,
                self.reward: reward
            })

    def _run_action_op(self, observations, head):
        """Function that runs the op calculating an action given the observations.
           Args:
           observations: np.array. shape = [num_actions, fingerprint_length].
           Observations that can be feed into the Q network.
           head: Integer. The output index to use.
           Returns:
           Integer. which action to be performed.
        """
        return np.asscalar(tf.get_default_session().run(
            self.action,
            feed_dict={
                self.observations: observations,
                self.head: head
            }))

    def get_action(self, observations, stochastic=True,
                   head=0, update_epsilon=None):
        """Function that chooses an action given the observations.
           Args:
           observations: np.array. shape = [num_actions, fingerprint_length].
           Observations that can be feed into the Q network.
           stochastic: Boolean. If set to False all the actions are always
           deterministic (default True).
           head: Integer. The output index to use.
           update_epsilon: Float or None. update epsilon a new value,
           if None no update happens (default: no update).
           Returns:
           Integer. which action to be performed.
        """
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        if stochastic and np.random.uniform() < self.epsilon:
            return np.random.randint(0, observations.shape[0])
        else:
            return self._run_action_op(observations, head)

    def train(self, states, rewards, next_states, done, weight, summary=True):
        """Function that takes a transition (s,a,r,s') and optimizes Bellman error.
           Args:
           states: object, a batch of observations.
           rewards: np.array, immediate reward attained after executing those actions
           dtype must be float32 and shape must be (batch_size,).
           next_states: object, observations that followed states.
           done: np.array, 1 if obs_t was the last observation in the episode and 0
           otherwise obs_tp1 gets ignored, but must be of the valid shape.
           dtype must be float32 and shape must be (batch_size,).
           weight: np.array, importance sampling weights for every element of the
           batch. dtype must be float32 and shape must be (batch_size,).
           summary: Boolean, whether to get summary.
           Returns:
           td_error: np.array. a list of differences between Q(s,a) and the target in Bellman's equation.
           dtype is float32 and shape is (batch_size,).
        """
        if summary:
            ops = [self.td_error, self.error_summary, self.optimization_op]
        else:
            ops = [self.td_error, self.optimization_op]
        feed_dict = {
            self.state_t: states,
            self.reward_t: rewards,
            self.done_mask: done,
            self.error_weight: weight
        }
        for i, next_state in enumerate(next_states):
            feed_dict[self.state_tp1[i]] = next_state
        return tf.get_default_session().run(ops, feed_dict=feed_dict)


def Q_fn_neuralnet_model(inputs, hparams, reuse=None):
    """Multi-layer model for q learning.
    Args:
    inputs: Tensor. The input.
    hparams: tf.HParameters. The hyper-parameters.
    reuse: Boolean. Whether the parameters should be reused.
    Returns:
    Tensor. shape = [batch_size, hparams.num_bootstrap_heads]. The output.
    """
    output = inputs
    for i, units in enumerate(hparams['model_param']['dense_layers']):
        output = tf.layers.dense(output, units, name='dense_%i' % i, reuse=reuse)
        output = getattr(tf.nn, hparams['train_param']['activation'])(output)
        if hparams['train_param']['batch_norm']:
            output = tf.layers.batch_normalization(
                output, fused=True, name='bn_%i' % i, reuse=reuse)
    if hparams['model_param']['num_bootstrap_heads']:
        output_dim = hparams['model_param']['num_bootstrap_heads']
    else:
        output_dim = 1
    output = tf.layers.dense(output, output_dim, name='final', reuse=reuse)
    return output