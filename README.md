# Mol-DQN-ADMET

Multi-objective **lead optimization** with deep reinforcement learning. This project
adapts **MolDQN** ([Zhou et al., 2019](https://arxiv.org/abs/1810.08678)) to optimize
drug candidates against a set of **ADMET** (Absorption, Distribution, Metabolism,
Excretion, Toxicity) properties, while preserving the lead compound's core ring
scaffold throughout generation.

Unlike the original MolDQN — which optimizes simple physicochemical targets such as
QED or penalized logP — this work redefines the reward around pharmacokinetic and
toxicity endpoints that matter in real lead optimization, and constrains the action
space so that the optimized molecule retains the structural backbone of the starting
lead.

## Method overview

Molecule generation is framed as a Markov Decision Process (MDP):

- **State** — a molecule represented as a SMILES string.
- **Action** — a single valid graph edit (atom addition, bond addition/removal),
  filtered to keep the starting lead's ring scaffold intact.
- **Reward** — a multi-objective ADMET score (below), evaluated at each step and
  discounted toward the end of the episode.
- **Agent** — a Deep Q-Network (single- and multi-objective variants) that learns
  which edits maximize the cumulative reward.

### Reward design

The reward combines six ADMET-related objectives with structural and physicochemical
penalties:

**Objectives (maximized)**

| Property | Type | Meaning |
|---|---|---|
| Caco-2 | regression | Intestinal permeability |
| Half-life (T½) | regression | Metabolic stability |
| LD50 | regression | Acute toxicity |
| CYP3A4 | classification | Metabolic liability |

**Penalties (subtracted)**

- **logS** penalty if aqueous solubility is too low (logS < −4)
- **logP** penalty if lipophilicity is out of range (logP < 0 or > 3)
- **Synthetic accessibility (SA)** penalty to discourage hard-to-synthesize molecules

```
reward = (Caco-2 + T½ + LD50 + CYP3A4)
         − logS_penalty − logP_penalty − SA_score
reward = reward × discount_factor ** (max_steps − step)
```

ADMET property predictors are pre-trained models (loaded from `ADMETlab`-style
pickled regressors/classifiers); the RL agent treats them as fixed reward oracles.

### Scaffold-constrained action space

At each step, candidate actions that would destroy the starting lead's ring scaffold
are filtered out (`molecules_action.py`). This keeps optimization within the
chemical neighborhood of the lead — the defining requirement of lead optimization
as opposed to de novo generation.

## Repository structure

```
models/
  molecules_mdp.py            # MDP definition (state / action / reward / step)
  molecules_action.py         # Valid action enumeration + scaffold constraint
  molecules_rules.py          # Chemistry utilities, SA score, penalized logP
  deep_q_networks.py          # Deep Q-Network
  multiobj_deep_q_networks.py # Multi-objective DQN (one Q-net per objective)
  trainer.py                  # Training loop (replay buffer, ε-greedy, checkpoints)
  admet.py                    # ADMET descriptor computation + predictor interface
  utils.py                    # Logging utilities
  admet_qed_model/
    optimize_multi_obj_all.py # ADMET reward environment
    config_1 / config_2 / config_3   # Hyperparameter configs
train_1.py / train_2.py / train_3.py # Entry points (different starting leads)
```

## Usage

Each `train_*.py` script optimizes a different starting lead compound (given as a
SMILES string in the script). To run:

```bash
python train_1.py
```

Each script:
1. Loads hyperparameters from a `config_*` file.
2. Loads the pre-trained ADMET predictor models.
3. Initializes the scaffold-constrained molecule environment from the starting lead.
4. Trains the DQN agent to optimize the multi-objective ADMET reward.
5. Logs generated molecules and rewards, and checkpoints the model.

## Requirements

- Python 3
- TensorFlow 1.x
- RDKit
- OpenAI Baselines (replay buffer, schedules)
- NumPy

> **Note:** This project was developed in 2019–2020 and uses TensorFlow 1.x
> (`tf.Session` / `tf.placeholder`-style graphs).

## Reference

- Zhou, Z., Kearnes, S., Li, L., Zare, R. N., & Riley, P. (2019).
  *Optimization of Molecules via Deep Reinforcement Learning.*
  Scientific Reports, 9, 10752. [arXiv:1810.08678](https://arxiv.org/abs/1810.08678)
