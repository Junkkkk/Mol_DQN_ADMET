"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig
import numpy as np
import time
import pickle
import gzip
import math
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.
    Note that this is not a count of valence electrons, but a count of the
    maximum number of bonds each element will make. For example, passing
    atom_types ['C', 'H', 'O'] will return [4, 1, 2].
    Args:
      atom_types: List of string atom types, e.g. ['C', 'H', 'O'].
    Returns:
      List of integer atom valences.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.
    Args:
      mol: RDKit Mol.
    Returns:
      String scaffold SMILES.
    """
    return Chem.MolToSmiles(
        MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.
    NOTE: This is more advanced than simply computing scaffold equality (i.e.
    scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
    be a subset of the (possibly larger) scaffold in mol.
    Args:
      mol: RDKit Mol.
      scaffold: String scaffold SMILES.
    Returns:
      Boolean whether scaffold is found in mol.
    """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    Args:
      molecule: Chem.Mol. A molecule.
    Returns:
      Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    See Junction Tree Variational Autoencoder for Molecular Graph Generation
    https://arxiv.org/pdf/1802.04364.pdf
    Section 3.2
    Penalized logP is defined as:
     y(m) = logP(m) - SA(m) - cycle(m)
     y(m) is the penalized logP,
     logP(m) is the logP of a molecule,
     SA(m) is the synthetic accessibility score,
     cycle(m) is the largest ring size minus by six in the molecule.
    Args:
      molecule: Chem.Mol. A molecule.
    Returns:
      Float. The penalized logP value.
    """
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score


def readSAModel(filename='./SA_score.pkl.gz'):
    # print("mol_metrics: reading SA model ...")
    # start = time.time()
    # if filename == 'SA_score.pkl.gz':
    #     filename = os.path.join(os.path.dirname(organ.__file__), filename)
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    # end = time.time()
    # print("loaded in {}".format(end - start))
    return SA_model

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def SA_score(mol, SA_model):
    # fragment score
    fp = Chem.AllChem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # SA_model = readSAModel()
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    val = remap(sascore, 5, 1.5)
    val = np.clip(val, 0.0, 1.0)
    val = 1 - val
    #hard 1 easy 0
    return val


# def num_long_cycles(mol):
#   """Calculate the number of long cycles.
#   Args:
#     mol: Molecule. A molecule.
#   Returns:
#     negative cycle length.
#   """
#   cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
#   if not cycle_list:
#     cycle_length = 0
#   else:
#     cycle_length = max([len(j) for j in cycle_list])
#   if cycle_length <= 6:
#     cycle_length = 0
#   else:
#     cycle_length = cycle_length - 6
#   return -cycle_length
#
#
# def penalized_logp(molecule):
#   log_p = Descriptors.MolLogP(molecule)
#   sas_score = SA_Score.sascorer.calculateScore(molecule)
#   cycle_score = num_long_cycles(molecule)
#   return log_p - sas_score + cycle_score