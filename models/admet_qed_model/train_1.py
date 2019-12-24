from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import sys
import warnings
sys.path.append('/home/junyoung/workspace/Mol_DQN_ADMET_v2')

from absl import app

import pickle
from rdkit import Chem

from models import deep_q_networks
from models import trainer
from models import molecules_rules
from models.admet_qed_model.optimize_multi_obj_all import ADMET_QED_Molecule


def main(argv):
    del argv  # unused.
    config_name = '/home/junyoung/workspace/Mol_DQN_ADMET_v2/models/admet_qed_model/config_1'
    # all_cid = '/home/junyoung/workspace/Mol_DQN_ADMET/Config/all_cid'

    logS_model_path = '/home/junyoung/workspace/ADMET/ADMETlab/regression_model/logS/logS_Model_v3.pkl'
    caco_model_path = '/home/junyoung/workspace/ADMET/ADMETlab/regression_model/caco2/caco2_Model_v3.pkl'
    cyp3a4_model_path = '/home/junyoung/workspace/ADMET/ADMETlab/classification_model/CYP3A4-inhibitor/CYP_inhibitor_3A4_SVC_ecfp4_model_v3.pkl'
    # ppb_model_path = '/home/junyoung/workspace/ADMET/ADMETlab/regression_model/PPB/PPB_Model_v3.pkl'
    t_model_path = '/home/junyoung/workspace/ADMET/ADMETlab/regression_model/T/T_Model_v3.pkl'
    ld50_model_path = '/home/junyoung/workspace/ADMET/ADMETlab/regression_model/LD50/LD50_Model_v3.pkl'


    with open(config_name) as f:
        hparams = json.load(f)

    # with open(all_cid) as f:
    #     all_mols = json.load(f)

    with open(logS_model_path, 'rb') as f:
        logS_model = pickle.load(f, encoding="latin1")

    # with open(ppb_model_path, 'rb') as f:
    #     ppb_model = pickle.load(f, encoding="latin1")

    with open(caco_model_path, 'rb') as f:
        caco_model = pickle.load(f, encoding="latin1")

    with open(cyp3a4_model_path, 'rb') as f:
        cyp3a4_model = pickle.load(f, encoding="latin1")

    with open(t_model_path, 'rb') as f:
        t_model = pickle.load(f, encoding="latin1")

    with open(ld50_model_path, 'rb') as f:
        ld50_model = pickle.load(f, encoding="latin1")


    ##To calculate SA score##
    SA_model = molecules_rules.readSAModel()

    all_mols = ["C1=CC(=C(C(=C1)[H])C2=C(C=CC=C2)C=NN(C3=C(C=NC=C3Cl)Cl)[H])[H]"]

    mol = Chem.MolFromSmiles(all_mols[0])
    rings = mol.GetRingInfo().BondRings()
    ring_scaffolds = [Chem.MolToSmiles(Chem.PathToSubmol(mol, ring)) for ring in rings]

    environment = ADMET_QED_Molecule(hparams=hparams,
                                     molecules=all_mols,
                                     SA_model=SA_model,
                                     logS_model=logS_model,
                                     caco_model=caco_model,
                                     cyp3a4_model=cyp3a4_model,
                                     t_model=t_model,
                                     ld50_model=ld50_model,
                                     scaffold=ring_scaffolds,
                                     record_path=True)

    dqn = deep_q_networks.DeepQNetwork(
        hparams=hparams,
        q_fn=functools.partial(
            deep_q_networks.Q_fn_neuralnet_model,
            hparams=hparams))

    Trainer =trainer.Trainer(
        hparams=hparams,
        environment=environment,
        model=dqn)

    Trainer.run_training()

    # config.write_hparams(hparams, os.path.join(hparams['save_param']['model_path'], 'config.json'))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    app.run(main)