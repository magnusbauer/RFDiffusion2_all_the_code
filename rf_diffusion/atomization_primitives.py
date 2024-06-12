from collections import namedtuple

from dataclasses import dataclass
import torch


AtomizedLabel = namedtuple('AtomizedLabel', ['coarse_idx0', 'aa', 'atom_name', 'pdb_idx', 'terminus_type'])
'''
Human readable definition of where an atomized atom came from.
    coarse_idx0: The residue index before atomization
    aa: Integer represenation of the residues/element.
    atom_name: The PDB name of the atomized atom. "None" means the residue
        is "coarse grained" and not atomized.
    pdb_idx: The index of the residue in the original pdb file before atomization.
    terminus_type: Stores the terminus type of the residue before atomization.
'''

@dataclass
class AtomizerSpec:
    '''
    Hold all the data needed to instantiate the AtomizeResidues class.
    '''
    deatomized_state: list[AtomizedLabel]
    residue_to_atomize: torch.Tensor
