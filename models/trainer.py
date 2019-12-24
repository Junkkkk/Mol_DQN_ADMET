"""Executor for deep Q network models."""

from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/home/junyoung/workspace/Mol_DQN')

import os
import time
import numpy as np
import pickle
import tensorflow as tf
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from baselines.common import schedules
from baselines.deepq import replay_buffer
from models.utils import get_logger

class Trainer():
    """Runs the training procedure.
    Briefly, the agent runs the action network to get an action to take in
    the environment. The state transition and reward are stored in the memory.
    Periodically the agent samples a batch of samples from the memory to
    update(train) its Q network. Note that the Q network and the action network
    share the same set of parameters, so the action network is also updated by
    the samples of (state, action, next_state, reward) batches.
    Args:
        hparams: tf.contrib.training.HParams. The hyper parameters of the model.
        environment: molecules.Molecule. The environment to run on.
        dqn: An instance of the DeepQNetwork class.
    Returns:
        None
    """
    def __init__(self, hparams, environment, model):
        self.hparams = hparams
        self.environment = environment
        self.dqn = model

        self.model_name = self.hparams['model_name']
        self.num_episodes = self.hparams['train_param']['num_episodes']
        self.max_steps_per_episode = self.hparams['train_param']['max_steps_per_episode']
        self.learning_frequency = self.hparams['train_param']['learning_frequency']
        self.update_frequency = self.hparams['train_param']['update_frequency']
        self.batch_size = self.hparams['train_param']['batch_size']

        self.prioritized = self.hparams['model_param']['prioritized']
        self.replay_buffer_size = self.hparams['model_param']['replay_buffer_size']
        self.prioritized_alpha = self.hparams['model_param']['prioritized_alpha']
        self.prioritized_beta = self.hparams['model_param']['prioritized_beta']
        self.num_bootstrap_heads = self.hparams['model_param']['num_bootstrap_heads']
        self.prioritized_epsilon = self.hparams['model_param']['prioritized_epsilon']

        self.fingerprint_length = self.hparams['ingredient_param']['fingerprint_length']
        self.fingerprint_radius = self.hparams['ingredient_param']['fingerprint_radius']

        self.model_path = self.hparams['save_param']['model_path']
        self.log_path = self.hparams['save_param']['log_path']
        self.max_num_checkpoints = self.hparams['save_param']['max_num_checkpoints']
        self.save_frequency = self.hparams['save_param']['save_frequency']
        self.logger = get_logger(self.model_name, self.log_path)

    def run_training(self):
        # self.summary_writer = tf.summary.FileWriter(self.model_path)
        tf.reset_default_graph()
        with tf.Session() as sess:
            self.dqn.build()
            model_saver = tf.train.Saver(max_to_keep=self.max_num_checkpoints)
            # The schedule for the epsilon in epsilon greedy policy.
            self.exploration = schedules.PiecewiseSchedule(
                [(0, 1.0), (int(self.num_episodes / 2), 0.1),
                 (self.num_episodes, 0.01)],
                outside_value=0.01)
            if self.prioritized:
                self.memory = replay_buffer.PrioritizedReplayBuffer(self.replay_buffer_size,
                                                                    self.prioritized_alpha)
                self.beta_schedule = schedules.LinearSchedule(
                    self.num_episodes, initial_p=self.prioritized_beta, final_p=0)
            else:
                self.memory = replay_buffer.ReplayBuffer(self.replay_buffer_size)
                self.beta_schedule = None
            sess.run(tf.global_variables_initializer())
            sess.run(self.dqn.update_op)
            self.global_step = 0
            mol_path=[]
            for self.episode in range(self.num_episodes):
                self.global_step = self._episode()
                if (self.episode + 1) % self.update_frequency == 0:
                    sess.run(self.dqn.update_op)
                if (self.episode + 1) % self.save_frequency == 0:
                    model_saver.save(
                        sess,
                        os.path.join(self.model_path, self.model_name+'_ckpt'),
                        global_step=self.episode+1)
                if self.num_episodes < 20 or self.num_episodes - self.episode < 101:
                    mol_path.append(self.environment._path)
                    with open(self.model_path +'/record_mol.pkl', 'wb') as f:
                        pickle.dump(mol_path, f)

    def _episode(self):
        """Runs a single episode.
        Args:
            environment: molecules.Molecule; the environment to run on.
            dqn: DeepQNetwork used for estimating rewards.
            memory: ReplayBuffer used to store observations and rewards.
            episode: Integer episode number.
            global_step: Integer global step; the total number of steps across all episodes.
            summary_writer: FileWriter used for writing Summary protos.
            exploration: Schedule used for exploration in the environment.
            beta_schedule: Schedule used for prioritized replay buffers.
        Returns:
            Updated global_step.
        """
        episode_start_time = time.time()
        init_mol = self.environment.initialize()
        self.logger.warning('##########################################')
        self.logger.warning('The SMILES of init mol : %s', init_mol)
        if self.num_bootstrap_heads:
            self.head = np.random.randint(self.num_bootstrap_heads)
        else:
            self.head = 0
        for step in range(self.max_steps_per_episode):
            try:
                self.result = self._step()
            except Exception:
                continue
            if step == self.max_steps_per_episode - 1:
                # episode_summary = self.dqn.log_result(self.result.state, self.result.reward)
                # self.summary_writer.add_summary(episode_summary, self.global_step)
                self.logger.warning('Episode %d/%d took %gs', self.episode + 1, self.num_episodes,
                                    time.time() - episode_start_time)
                self.logger.warning('The SMILES of last mol : %s', self.result.state)
                # Use %s since reward can be a tuple or a float number.
                self.logger.warning('The reward is: %s', str(self.result.reward))
                self.logger.warning('##########################################')
            if (self.episode > min(100, self.num_episodes / 10)) and (self.global_step % self.learning_frequency == 0):
                if self.prioritized:
                    (state_t, _, reward_t, state_tp1, done_mask, weight,
                     indices) = self.memory.sample(
                      self.batch_size, beta=self.beta_schedule.value(self.episode))
                else:
                    (state_t, _, reward_t, state_tp1,
                     done_mask) = self.memory.sample(self.batch_size)
                    weight = np.ones([reward_t.shape[0]])
                # np.atleast_2d cannot be used here
                # because a new dimension will be always added in the front and there is no way of changing this.
                if reward_t.ndim == 1:
                    reward_t = np.expand_dims(reward_t, axis=1)
                td_error, _ = self.dqn.train(
                    states=state_t,
                    rewards=reward_t,
                    next_states=state_tp1,
                    done=np.expand_dims(done_mask, axis=1),
                    weight=np.expand_dims(weight, axis=1),
                    summary=False)
                # self.summary_writer.add_summary(error_summary, self.global_step)
                self.logger.warning('The SMILES of %d steps mol : %s', step, self.result.state)
                self.logger.warning('Current reward: %.4f', self.result.reward)
                if self.prioritized:
                    self.memory.update_priorities(
                        indices,
                        np.abs(np.squeeze(td_error) + self.prioritized_epsilon).tolist())
            self.global_step += 1
        return self.global_step

    def _step(self):
        """Runs a single step within an episode.
           Args:
            environment: molecules.Molecule; the environment to run on.
            dqn: DeepQNetwork used for estimating rewards.
            memory: ReplayBuffer used to store observations and rewards.
            episode: Integer episode number.
            hparams: HParams.
            exploration: Schedule used for exploration in the environment.
            head: Integer index of the DeepQNetwork head to use.
           Returns:
            molecules.Result object containing the result of the step.
        """
        # Compute the encoding for each valid action from the current state.
        self.steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
        self.valid_actions = list(self.environment.get_valid_actions())
        self.observations = np.vstack([self.get_fingerprint_with_steps_left(act, self.steps_left)
                                       for act in self.valid_actions])
        self.action = self.valid_actions[self.dqn.get_action(
            self.observations, head=self.head, update_epsilon=self.exploration.value(self.episode))]
        self.action_t_fingerprint = self.get_fingerprint_with_steps_left(self.action, self.steps_left)
        self.result = self.environment.step(self.action)
        self.steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
        self.action_fingerprints = np.vstack([self.get_fingerprint_with_steps_left(act, self.steps_left)
                                              for act in self.environment.get_valid_actions()])

        # we store the fingerprint of the action in obs_t so action does not matter here.
        self.memory.add(
            obs_t=self.action_t_fingerprint,
            action=0,
            reward=self.result.reward,
            obs_tp1=self.action_fingerprints,
            done=float(self.result.terminated))
        return self.result

    def get_fingerprint_with_steps_left(self, smiles, steps_left):
        if smiles is None:
            return np.append(np.zeros((self.fingerprint_length,)), steps_left)
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.append(np.zeros((self.fingerprint_length,)), steps_left)

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule, self.fingerprint_radius, self.fingerprint_length)
        arr = np.zeros((1,))
        # ConvertToNumpyArray takes ~ 0.19 ms, while
        # np.asarray takes ~ 4.69 ms
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return np.append(arr, steps_left)