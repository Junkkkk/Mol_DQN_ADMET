"""Defines the Markov decision process of generating a molecule.
The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

from rdkit import Chem
from rdkit.Chem import Draw

from models import molecules_rules
from models.molecules_action import Molecules_Action


class Result(collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
    """A namedtuple defines the result of a step for the molecule class.
      The namedtuple contains the following fields:
      state: Chem.RWMol. The molecule reached after taking the action.
      reward: Float. The reward get after taking the action.
      terminated: Boolean. Whether this episode is terminated.
    """

class Molecule_MDP(object):
    """Defines the Markov decision process of generating a molecule."""
    def __init__(self, hparams, scaffold, target_fn=None, record_path=False):
        """Initializes the parameters for the MDP.
           Internal state will be stored as SMILES strings.
           Args:
           atom_types: The set of elements the molecule may contain.
           allow_removal: Boolean. Whether to allow removal of a bond.
           allow_no_modification: Boolean. If true, the valid action set will
           include doing nothing to the current molecule, i.e., the current
           molecule itself will be added to the action set.
           allow_bonds_between_rings: Boolean. If False, new bonds connecting two
           atoms which are both in rings are not allowed.
           DANGER Set this to False will disable some of the transformations eg.
           c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
           But it will make the molecules generated make more sense chemically.
           allowed_ring_sizes: Set of integers or None. The size of the ring which
           is allowed to form. If None, all sizes will be allowed. If a set is
           provided, only sizes in the set is allowed.
           max_steps: Integer. The maximum number of steps to run.
           target_fn: A function or None. The function should have Args of a
           String, which is a SMILES string (the state), and Returns as
           a Boolean which indicates whether the input satisfies a criterion.
           If None, it will not be used as a criterion.
           record_path: Boolean. Whether to record the steps internally.
        """
        self.hparams = hparams
        self.scaffold = scaffold

        self.atom_types = self.hparams['action_param']['atom_types']
        self.allow_removal = self.hparams['action_param']['allow_removal']
        self.allow_no_modification = self.hparams['action_param']['allow_no_modification']
        self.allow_bonds_between_rings = self.hparams['action_param']['allow_bonds_between_rings']
        self.allowed_ring_sizes = self.hparams['action_param']['allowed_ring_sizes']
        self.max_steps = self.hparams['train_param']['max_steps_per_episode']
        self.discount_factor = self.hparams['model_param']['discount_factor']
        self._state = None
        self.NumRingPenalty = 0
        self._valid_actions = []
        # The status should be 'terminated' if initialize() is not called.
        self._counter = self.max_steps
        self._target_fn = target_fn
        self.record_path = record_path
        self._path = []
        self._max_bonds = 4
        self._max_new_bonds = dict(
            zip(self.atom_types, molecules_rules.atom_valences(self.atom_types)))

    @property
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self._counter

    def get_path(self):
        return self._path

    def initialize(self):
        """Resets the MDP to its initial state."""
        pass

    def get_valid_actions(self, state=None, force_rebuild=False):
        """Gets the valid actions for the state.
          In this design, we do not further modify a aromatic ring. For example,
          we do not change a benzene to a 1,3-Cyclohexadiene. That is, aromatic
          bonds are not modified.
          Args:
          state: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
          considered as the SMILES string. The state to query. If None, the
          current state will be considered.
          force_rebuild: Boolean. Whether to force rebuild of the valid action set.
          Returns:
          A set contains all the valid actions for the state. Each action is a
          SMILES string. The action is actually the resulting state.
        """
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)
        self._valid_actions = Molecules_Action(self.hparams, self.scaffold).get_valid_actions(state=state)
        return copy.deepcopy(self._valid_actions)

    def _reward(self):
        """Gets the reward for the state.
           A child class can redefine the reward function if reward other than
           zero is desired.
           Returns:
           Float. The reward for the current state.
        """
        pass

    def _goal_reached(self):
        """Sets the termination criterion for molecule Generation.
           A child class can define this function to terminate the MDP before
           max_steps is reached.
           Returns:
           Boolean, whether the goal is reached or not. If the goal is reached,
           the MDP is terminated.
        """
        if self._target_fn is None:
            return False
        return self._target_fn(self._state)

    def step(self, action):
        """Takes a step forward according to the action.
           Args:
           action: Chem.RWMol. The action is actually the target of the modification.
           Returns:
           results: Namedtuple containing the following fields:
            * state: The molecule reached after taking the action.
            * reward: The reward get after taking the action.
            * terminated: Whether this episode is terminated.
           Raises:
           ValueError: If the number of steps taken exceeds the preset max_steps, or
           the action is not in the set of valid_actions.
        """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError('This episode is terminated.')
        if action not in self._valid_actions:
            raise ValueError('Invalid action.')

        ####For Bond_addition, if Num of Ring increases more than two before,
        ####consider fused ring -> so we want to penalty reward -1
        # print("current state", self._state)
        # current_mol = Chem.MolFromSmiles(self._state)
        # next_mol = Chem.MolFromSmiles(action)
        # print("next state", action)
        # if next_mol.GetRingInfo().NumRings() >= current_mol.GetRingInfo().NumRings() + 2:
        #     self.NumRingPenalty = 1
        # else:
        #     self.NumRingPenalty = 0

        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1

        result = Result(
            state=self._state,
            reward=self._reward(),
            terminated=(self._counter >= self.max_steps) or self._goal_reached())
        return result

    def visualize_state(self, state=None, **kwargs):
        """Draws the molecule of the state.
           Args:
           state: String, Chem.Mol, or Chem.RWMol. If string is prov ided, it is
           considered as the SMILES string. The state to query. If None, the
           current state will be considered.
           **kwargs: The keyword arguments passed to Draw.MolToImage.
           Returns:
           A PIL image containing a drawing of the molecule.
        """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)