"""
Tools for creating graph inputs from molecule data
"""

import itertools
import os
import sys
from collections import deque
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Union

import numpy as np
from pymatgen.core import Element, Molecule
from pymatgen.io.babel import BabelMolAdaptor

try:
    import pybel  # type: ignore
except ImportError:
    try:
        from openbabel import pybel
    except ImportError:
        pybel = None

try:
    from rdkit import Chem  # type: ignore
except ImportError:
    Chem = None


class MolecularGraph():

    def __init__(self):

        # Check if openbabel and RDKit are installed
        if Chem is None or pybel is None:
            raise RuntimeError("RDKit and openbabel must be installed")
        self.atom_features = ['atomic_num', 'pos']

    def convert(self, mol, state_attributes: List = None) -> Dict:  # type: ignore
        """
        Compute the representation for a molecule

        Args：
            mol (pybel.Molecule): Molecule to generate features for
            state_attributes (list): State attributes. Uses average mass and number of bonds per atom as default
        """

        # Get the features features for all atoms and bonds
        atom_features = []

        for idx, atom in enumerate(mol.atoms):
            f = self.get_atom_feature(mol, atom)
            atom_features.append(f)
        atom_features = sorted(atom_features, key=lambda x: x["coordid"])

        atoms = []
        pos_list = []
        for atom in atom_features:
            z, pos = self._create_atom_feature_vector(atom)
            atoms.append(z)
            pos_list.append(pos)
            
        # Generate the state attributes (that describe the whole network)
        state_attributes = state_attributes 

        return {"atom": atoms, "pos": pos_list, "state": state_attributes}


    def get_atom_feature(
        self, mol, atom  # type: ignore
    ) -> Dict:  # type: ignore
        """
        Generate all features of a particular atom

        Args:
            mol (pybel.Molecule): Molecule being evaluated
            atom (pybel.Atom): Specific atom being evaluated
        Return:
            (dict): All features for that atom
        """

        # Get the link to the OpenBabel representation of the atom
        obatom = atom.OBAtom
        atom_idx = atom.idx - 1  # (pybel atoms indices start from 1)

        # Get the fast-to-compute properties
        output = {
            "atomic_num": obatom.GetAtomicNum(),
            "coordid": atom.coordidx,
            "pos": list(atom.coords)
        }
        
        return output

    def _create_atom_feature_vector(self, atom: dict):
        """Generate the feature vector from the atomic feature dictionary

        Handles the binarization of categorical variables, and transforming the ring_sizes to a list

        Args:
            atom (dict): Dictionary of atomic features
        Returns:
            ([int]): Atomic feature vector
        """
        
        for i in self.atom_features:
            if i == "atomic_num":
                z=int(atom[i])
            elif i == "pos":
                pos=atom[i]

        return z, pos

    @staticmethod
    def _get_rdk_mol(mol, format: str = "smiles"):
        """
        Return: RDKit Mol (w/o H)
        """
        if format == "pdb":
            return Chem.rdmolfiles.MolFromPDBBlock(mol.write("pdb"))
        if format == "smiles":
            return Chem.rdmolfiles.MolFromSmiles(mol.write("smiles"))
        return None


def mol_from_smiles(smiles: str):
    """
    load molecule object from smiles string
    Args:
        smiles (string): smiles string

    Returns:
        openbabel molecule
    """
    mol = pybel.readstring(format="smi", string=smiles)
    mol.make3D()
    return mol


def mol_from_pymatgen(mol: Molecule):
    """
    Args:
        mol(Molecule)
    """
    mol = pybel.Molecule(BabelMolAdaptor(mol).openbabel_mol)
    mol.make3D()
    return mol


def mol_from_file(file_path: str, file_format: str = "xyz"):
    """
    Args:
        file_path(str)
        file_format(str): allow formats that open babel supports
    """
    mol = list(pybel.readfile(format=file_format, filename=file_path))[0]
    return mol


def _convert_mol(mol: str, molecule_format: str, converter: MolecularGraph) -> Dict:
    """Convert a molecule from string to its graph features

    Utility function used in the graph generator.

    The parse and convert operations are both in this function due to Pybel objects
    not being serializable. By not using the Pybel representation of each molecule
    as an input to this function, we can use multiprocessing to parallelize conversion
    over molecules as strings can be passed as pickle objects to the worker threads but
    but Pybel objects cannot.

    Args:
        mol (str): String representation of a molecule
        molecule_format (str): Format of the string representation
        converter (MolecularGraph): Tool used to generate graph representation
    Returns:
        (dict): Graph representation of the molecule
    """

    # Convert molecule into pybel format
    if molecule_format == "smiles":
        mol = mol_from_smiles(mol)  # Used to generate 3D coordinates/H atoms
    else:
        mol = pybel.readstring(molecule_format, mol)

    return converter.convert(mol)

