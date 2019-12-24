"""Optimizes QED of a molecule with DQN.
This experiment tries to find the molecule with the highest QED
starting from a given molecule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED


from models import molecules_mdp
from models import molecules_rules
from models import admet

class ADMET_QED_Molecule(molecules_mdp.Molecule_MDP):
    """
    Defines the subclass of generating a molecule with a specific reward.
    The reward is defined as a scalar
    reward = weight * similarity_score + (1 - weight) *  qed_score
    """

    def __init__(self, molecules, SA_model, logS_model, caco_model, cyp3a4_model,t_model, ld50_model,  **kwargs):
        """Initializes the class.
        Args:
          target_molecule: SMILES string. The target molecule against which we
            calculate the similarity.
          similarity_weight: Float. The weight applied similarity_score.
          discount_factor: Float. The discount factor applied on reward.
          **kwargs: The keyword arguments passed to the parent class.
        """
        super(ADMET_QED_Molecule, self).__init__(**kwargs)
        self.molecules = molecules
        self.SA_model = SA_model
        self.logS = logS_model
        self.caco = caco_model
        self.cyp3a4 = cyp3a4_model
        self.t = t_model
        self.ld50 = ld50_model

    def initialize(self):
        self._state = random.choice(self.molecules)
        self._target_mol_fingerprint = self.get_fingerprint(
            Chem.MolFromSmiles(self._state))
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0
        return self._state

    def get_fingerprint(self, molecule):
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, smiles):
        """Gets the similarity between the current molecule and the target molecule.
        Args:
          smiles: String. The SMILES string for the current molecule.
        Returns:
          Float. The Tanimoto similarity.
        """
        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                              fingerprint_structure)

    def _reward(self):
        """Calculates the reward of the current state.
        The reward is defined as a tuple of the similarity and QED value.
        Returns:
          A tuple of the similarity and qed value
        """
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        #similarity_score = self.get_similarity(self._state)
        # calculate QED
        #qed_value = QED.qed(mol)

        # Calculate ADMET
        all = admet.ADMET(self._state)
        caco = all.Get_caco(self.caco)
        cyp3a4 = all.Get_cyp3a4(self.cyp3a4)
        t = all.Get_t(self.t)
        LD50 = all.Get_ld50(self.ld50)

        reward = caco + t + LD50 + cyp3a4

        # calculate Penalty (Synthesis ability / logS / logP)
        sa_score = molecules_rules.SA_score(mol, self.SA_model)
        log_p = Chem.Descriptors.MolLogP(mol)
        logS = all.Get_logS(self.logS)

        #logS pealty
        if logS < -4:
            logS_penalty = 1
        else:
            logS_penalty = 0

        #logP penalty
        if log_p > 3 or log_p < 0:
            logP_penalty = 1
        else:
            logP_penalty = 0

        reward = reward - logS_penalty - logP_penalty - sa_score

        discount = self.discount_factor**(self.max_steps - self._counter)

        return reward * discount