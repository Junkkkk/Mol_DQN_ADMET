from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from deep_q_networks import DeepQNetwork

class MultiObjectiveDeepQNetwork(DeepQNetwork):
    """Multi Objective Deep Q Network.
    Briefly, the difference between this Multi Objective Deep Q Network and
    a naive Deep Q Network is that this one uses one Q network for approximating
    each of the objectives.
    And a weighted sum of those Q values are used for decision making.
    The loss is the summation of the losses of each Q network.
    """
    def __init__(self, objective_weight, **kwargs):
        """Creates the model function.
           Args:
           objective_weight: np.array with shape [num_objectives, 1].
           The weight vector for the objectives.
           **kwargs: arguments for the DeepQNetworks class.
        """
        # Normalize the sum to 1.
        self.objective_weight = objective_weight / np.sum(objective_weight)
        self.num_objectives = objective_weight.shape[0]
        super(MultiObjectiveDeepQNetwork, self).__init__(**kwargs)

    def _build_graph(self):
        """Builds the computational graph.
           Input placeholders created:
           observations: shape = [batch_size, hparams.fingerprint_length].
           The input of the Q function.
           head: shape = [1].
           The index of the head chosen for decision.
           objective_weight: shape = [num_objectives, 1].
           objective_weight is the weight to scalarize the objective vector:
           reward = sum (objective_weight_i * objective_i)
           state_t: shape = [batch_size, hparams.fingerprint_length].
           The state at time step t.
           state_tp1: a list of tensors,
                each has shape = [num_actions, hparams.fingerprint_length].
           Note that the num_actions can be different for each tensor.
           The state at time step t+1.
           done_mask: shape = [batch_size, 1]
           Whether state_tp1 is the terminal state.
           reward_t: shape = [batch_size, num_objectives]
           the reward at time step t.
           error weight: shape = [batch_size, 1]
           weight for the loss.
           Instance attributes created:
           q_values: List of Tensors of [batch_size, 1]. The q values for the observations.
           td_error: List of Tensor of [batch_size, 1]. The TD error.
           weighted_error: List of Tensor of [batch_size, 1].
           The TD error weighted by importance sampling weight.
           q_fn_vars: List of tf.Variables. The variables of q_fn when computing the q_values of state_t
           q_fn_vars: List of tf.Variables. The variables of q_fn when computing the q_values of state_tp1
        """
        batch_size, _ = self.input_shape
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self._build_input_placeholder()
            self.reward_t = tf.placeholder(tf.float32, (batch_size, self.num_objectives), name='reward_t')
            # objective_weight is the weight to scalarize the objective vector:
            # reward = sum (objective_weight_i * objective_i)
            self.objective_weight_input = tf.placeholder(
                  tf.float32, [self.num_objectives, 1], name='objective_weight')

            # split reward for each q network
            rewards_list = tf.split(self.reward_t, self.num_objectives, axis=1)
            q_values_list = []
            self.td_error = []
            self.weighted_error = 0
            self.q_fn_vars = []
            self.q_tp1_vars = []

            # build a Q network for each objective
            for obj_idx in range(self.num_objectives):
                with tf.variable_scope('objective_%i' % obj_idx):
                    (q_values, td_error, weighted_error,
                     q_fn_vars, q_tp1_vars) = self._build_single_q_network(
                        self.observations, self.head, self.state_t, self.state_tp1,
                        self.done_mask, rewards_list[obj_idx], self.error_weight)
                    q_values_list.append(tf.expand_dims(q_values, 1))
                    # td error is for summary only.
                    # weighted error is the optimization goal.
                    self.td_error.append(td_error)
                    self.weighted_error += weighted_error / self.num_objectives
                    self.q_fn_vars += q_fn_vars
                    self.q_tp1_vars += q_tp1_vars
            q_values = tf.concat(q_values_list, axis=1)
            # action is the one that leads to the maximum weighted reward.
            self.action = tf.argmax(
                  tf.matmul(q_values, self.objective_weight_input), axis=0)

    def _build_summary_ops(self):
        """Creates the summary operations.
        Input placeholders created:
          smiles: the smiles string.
          rewards: the rewards.
          weighted_reward: the weighted sum of the rewards.
        Instance attributes created:
          error_summary: the operation to log the summary of error.
          episode_summary: the operation to log the smiles string and reward.
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.name_scope('summaries'):
                # The td_error here is the difference between q_t and q_t_target.
                # Without abs(), the summary of td_error is actually underestimated.
                error_summaries = [tf.summary.scalar
                                   ('td_error_%i' % i, tf.reduce_mean(tf.abs(self.td_error[i])))
                                   for i in range(self.num_objectives)]
                self.error_summary = tf.summary.merge(error_summaries)
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.rewards = [
                    tf.placeholder(tf.float32, [], 'summary_reward_obj_%i' % i)
                    for i in range(self.num_objectives)
                ]
                # Weighted sum of the rewards.
                self.weighted_reward = tf.placeholder(tf.float32, [],
                                                      'summary_reward_sum')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summaries = [
                    tf.summary.scalar('reward_obj_%i' % i, self.rewards[i])
                    for i in range(self.num_objectives)
                ]
                reward_summaries.append(
                    tf.summary.scalar('sum_reward', self.rewards[-1]))

                self.episode_summary = tf.summary.merge([smiles_summary] +
                                                        reward_summaries)

    def log_result(self, smiles, reward):
        """Summarizes the SMILES string and reward at the end of an episode.
           Args:
           smiles: String. The SMILES string.
           reward: List of Float. The rewards for each objective.
           Returns:
           the summary protobuf.
        """
        feed_dict = {
            self.smiles: smiles,
        }
        for i, reward_value in enumerate(reward):
            feed_dict[self.rewards[i]] = reward_value
        # calculated the weighted sum of the rewards.
        feed_dict[self.weighted_reward] = np.asscalar(
            np.array([reward]).dot(self.objective_weight))
        return tf.get_default_session().run(
            self.episode_summary, feed_dict=feed_dict)

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
                self.objective_weight_input: self.objective_weight,
                self.head: head}))