from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import io
import numpy as np
from Bio import PDB
import networkx as nx
import assertpy
import copy
import torch
from icecream import ic
import Bio.PDB.SASA
import tree
np.int = np.int64

import aa_model

p = PDBParser(PERMISSIVE=0, QUIET=1)

def get_sasa_indep(indep, probe_radius=1.4):
    buffer = io.StringIO()
    names = indep.write_pdb_file(buffer)
    buffer.seek(0)
    struct = p.get_structure('none', buffer)
    sr = ShrakeRupley(probe_radius=probe_radius)
    sr.compute(struct, level="A")
    return struct, names


def small_molecules(indep):
    e = indep.bond_feats.detach().cpu().clone().numpy()
    e[~indep.is_sm] = 0
    e[:,~indep.is_sm] = 0
    e *= (e != aa_model.GP_BOND)
    G = nx.from_numpy_matrix(e)
    cc = list(nx.connected_components(G))
    cc.sort(key=min)
    o = [np.array([], dtype=np.int32)]
    for c in cc:
        c = np.array(list(c))
        is_sm = indep.is_sm[c]
        assertpy.assert_that(is_sm.unique()).is_length(1)
        if is_sm[0]:
            o.append(c)
    return o

def get_max_sasa(atom, probe_radius=1.4):
    return 4 * np.pi * (Bio.PDB.SASA.ATOMIC_RADII[atom.element] + probe_radius)**2

def get_relative_sasa_sm(indep):
    sm_sasa, sm_atoms, sm = get_sm_sasa(indep, probe_radius=1.40)
    sm_max_sasa = tree.map_structure(get_max_sasa, sm_atoms)
    assertpy.assert_that(len(sm_max_sasa)).is_equal_to(len(sm_sasa))
    sm_relative_sasa = []
    for sasa, max_sasa in zip(sm_sasa, sm_max_sasa):
        sm_relative_sasa.append(sasa / max_sasa)
    return sm, sm_atoms, sm_relative_sasa, sm_sasa, sm_max_sasa

    
def get_sm_sasa(indep, probe_radius=1.40):
    struct, names = get_sasa_indep(indep, probe_radius=probe_radius)
    atomList = PDB.Selection.unfold_entities(struct, target_level='A')
    atom_by_id = {atom.serial_number:atom for atom in atomList}
    sm = small_molecules(indep)
    sm_atom_ids = []
    for sm_i in sm:
        sm_names = names[sm_i]
        for n in sm_names:
            assertpy.assert_that(n).starts_with('id ')
        sm_atom_ids.append([int(n.split()[1]) for n in sm_names])
    sm_atoms = []
    for atom_ids in sm_atom_ids:
        sm_atoms.append([atom_by_id[i] for i in atom_ids])
    
    sm_atom_sasa = []
    for atoms in sm_atoms:
        sm_atom_sasa.append(np.array([atom.sasa for atom in atoms]))
    return sm_atom_sasa, sm_atoms, sm

def get_relative_sasa(indep):
    sm, _, sm_relative_sasa, _, _ = get_relative_sasa_sm(indep)
    flat_sm_indices = np.concatenate(sm)
    flat_relative_sasa = np.concatenate(sm_relative_sasa)
    relative_sasa = torch.full((indep.length(),), -10.0)
    relative_sasa[flat_sm_indices] = torch.tensor(flat_relative_sasa).float()
    return relative_sasa

def noised_relative_sasa(indep, std_std):
    sasa = get_relative_sasa(indep)
    std = torch.zeros((indep.length(),))
    if np.random.random() < 0.5:
        std[:] = torch.abs(torch.normal(0.0, std_std, (1,)))
    else:
        std = torch.abs(torch.normal(0.0, std_std, std.shape))
    sasa = torch.normal(sasa, std)
    sasa[~indep.is_sm] = -10
    return sasa, std
