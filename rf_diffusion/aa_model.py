from __future__ import annotations
import logging
import torch
import contextlib
from functools import wraps
import assertpy
from collections import defaultdict, OrderedDict
from typing import Iterable
import torch.nn.functional as F
import dataclasses
from icecream import ic
from assertpy import assert_that
from rf_diffusion.chemical import ChemicalData as ChemData
import rf2aa.util
from rf2aa.data import parsers
from dataclasses import dataclass
from rf2aa.kinematics import get_chirals
from rf2aa.util_module import XYZConverter
from omegaconf import DictConfig

import rf2aa.tensor_util
import copy
import numpy as np
import os 
import pickle
import rf_diffusion.kinematics
import rf_diffusion.util as util
from rf_diffusion.parsers import parse_pdb_lines_target
import networkx as nx
nx.from_numpy_matrix = nx.from_numpy_array
import random
import rf_diffusion.rotation_conversions as rotation_conversions
import rf_diffusion.atomize as atomize
from rf_diffusion import write_file
from rf_diffusion.contigs import ContigMap


from rf_diffusion.atomization_primitives import AtomizedLabel, AtomizerSpec  # noqa: F401
from rf_diffusion import build_coords

import rf_diffusion.contigs as contigs

import rf_diffusion.frame_diffusion.data.utils as du
from rf_diffusion.frame_diffusion.data import all_atom

import ipd

logger = logging.getLogger(__name__)

import rf_diffusion.nucleic_compatibility_utils as nucl_utils

# CONSTANTS
NINDEL=1
"""Number of indel channels"""

NTERMINUS=2
"""Number of terminus channels (N-terminal and C-terminal)"""

N_TERMINUS = 1
"""Integer used to indicate the N-terminal"""

C_TERMINUS = 2
"""Integer used to indicate the C-terminal"""

UNIQUE_LIGAND="__UNIQUE_LIGAND"
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

CA_ONLY = 'CA_ONLY'

def chain_letters_from_same_chain(same_chain):
    L = same_chain.shape[0]
    G = nx.from_numpy_array(same_chain.cpu().numpy())
    cc = list(nx.connected_components(G))
    cc.sort(key=min)
    chain_letters = np.chararray((L,), unicode=True)

    for ch_i, ch_name in zip(cc, alphabet):
        chain_letters[list(ch_i)] = ch_name

    return chain_letters

def same_chain_from_chain_letters(chains):
    return torch.tensor(chains[:, None] == chains[None, :]).bool()

def same_chain_with_covale(same_chain, covale_bonds):
    same_chain = same_chain.clone()
    for (res_i, _), sm_i, _ in covale_bonds:
        same_chain[res_i, sm_i] = True
    chains_after_covale = chain_letters_from_same_chain(same_chain)
    same_chain = same_chain_from_chain_letters(chains_after_covale)
    return same_chain

def chain_start_end_from_hal(hal):
    chains = []

    chain_letters, resi_idxs = zip(*hal)

    previous_chain_letter = chain_letters[0]
    chain_start = 0
    total_element_count = 0

    for current_chain_letter in chain_letters:
        total_element_count += 1

        if current_chain_letter != previous_chain_letter:
            chains.append((chain_start, total_element_count - 1))
            previous_chain_letter = current_chain_letter
            chain_start = total_element_count - 1
        
    chains.append((chain_start, total_element_count))

    return chains

@dataclass
class Indep:
    # WARNING: Be careful with the order of the fields as this will change the instantiation order 
    #   where indeps are created without named arguments.
    seq: torch.Tensor # [L]
    xyz: torch.Tensor # [L, 36, 3]
    idx: torch.Tensor # [L] 

    # SM specific
    bond_feats: torch.Tensor # [L, L, ?]
    chirals: torch.Tensor # [n_chiral, 5]
    same_chain: torch.Tensor # [L, L]
    terminus_type: torch.Tensor # [L]

    # Conditioning specific
    extra_t1d: torch.Tensor = dataclasses.field(default_factory=lambda: None)  # [L, ?]
    extra_t2d: torch.Tensor = dataclasses.field(default_factory=lambda: None)  # [L, L, ?]
    is_gp: torch.Tensor = None  # [L]

    def __post_init__(self):
        '''
        Runs after the implicit __init__ from @dataclass
        '''
        # is_gp needs to be the correct size if it was not specified
        if self.is_gp is None and self.seq is not None:
            self.is_gp = torch.zeros(self.length(), dtype=bool)
        self.add_tensor_dim_names() # make sure names are ok
        self.remove_tensor_dim_names()

    def add_tensor_dim_names(self):
        '''used to temporarially adds dimension names to the tensors needed for symmetry'''
        if self.seq           is not None: self.seq          .rename_(*'L'.split())
        if self.xyz           is not None: self.xyz          .rename_(*'L Atom XYZ'.split())
        if self.idx           is not None: self.idx          .rename_(*'L'.split())
        if self.bond_feats    is not None: self.bond_feats   .rename_(*'L1 L2'.split())
        if self.chirals       is not None: self.chirals      .rename_(*'Lsparse IdxAll0'.split())
        if self.same_chain    is not None: self.same_chain   .rename_(*'L1 L2'.split())
        if self.terminus_type is not None: self.terminus_type.rename_(*'L'.split())
        if self.extra_t1d     is not None: self.extra_t1d    .rename_(*'L D'.split())
        if self.extra_t2d     is not None: self.extra_t2d    .rename_(*'L1 L2 D'.split())
        if self.is_gp         is not None: self.is_gp        .rename_(*'L'.split())

    def remove_tensor_dim_names(self):
        '''used to remove dimension names from the tensors'''
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if torch.is_tensor(val): val.rename_(None)

    def print1d(self, compact=True):
        '''Pretty prints the indeps bacis 1D information'''
        chirals = {int(_) for _ in self.chirals[:,0].reshape(-1)}
        stuff1d = zip(self.is_gp,self.seq,self.idx,self.human_readable_seq(),self.type())
        chain, prev, skipped, first = 0, None, False, True
        spacer = ' ⋮   ⋮  ⋮  ⋮  ⋮   ⋮   ⋮'
        for i, (gp, si, idx, sn, typ) in enumerate(stuff1d):
            if i > 0 and not self.same_chain[i,i-1]: chain += 1
            crl = "CRL" if i in chirals else  "   "
            if compact and prev == (gp, chain, crl, typ):
                skipped = True
                continue
            if skipped: print(spacer)
            if not prev or prev[1] != chain: print('='*23 if first else '-'*23)
            print(f'CH{chain} {"GP" if gp else "  "} {crl} {typ} {si:2} {idx:3} {sn:5}')
            prev, skipped, first = (gp, chain, crl, typ), False, False
        if skipped:     print(spacer)
        print('='*23, flush=True)


    def clone(self):
        '''
        Make of copy of the object. Needed because 
        copy.deepcopy can only copy leaf tensors and the 
        fields are not always leaves.
        '''

        _deepcopy_detached = lambda x: copy.deepcopy(x.detach()) if hasattr(x, 'detach') else copy.deepcopy(x)  # noqa: E731

        indep_copy = self.__class__(
            seq=_deepcopy_detached(self.seq),
            xyz=_deepcopy_detached(self.xyz),
            idx=_deepcopy_detached(self.idx),
            bond_feats=_deepcopy_detached(self.bond_feats),
            chirals=_deepcopy_detached(self.chirals),
            same_chain=_deepcopy_detached(self.same_chain),
            terminus_type=_deepcopy_detached(self.terminus_type),
            extra_t1d=_deepcopy_detached(self.extra_t1d),
            extra_t2d=None if not hasattr(self, 'extra_t2d') else _deepcopy_detached(self.extra_t2d), # Only doesn't exist in tests
            is_gp=_deepcopy_detached(self.is_gp) # None only occurs in unit tests
        )

        # Special extra fields
        if hasattr(self, 'origin'):
            indep_copy.origin = _deepcopy_detached(self.origin)

        if hasattr(self, 'extra_t2d'):
            indep_copy.extra_t2d = copy.deepcopy(self.extra_t2d.detach())

        return indep_copy


    @property
    def device(self):
        return self.xyz.device

    @property
    def is_sm(self) -> torch.Tensor:
        return rf2aa.util.is_atom(self.seq).to(self.device)

    @property
    def atom_frames(self) -> torch.Tensor:
        return self.get_atom_frames()
    
    def get_atom_frames(self, omit_permutation=False):
        is_sm = rf2aa.util.is_atom(self.seq).to('cpu')

        # Make a graph of the small molecule elements
        is_covalently_bonded = (self.bond_feats >= 1) & (self.bond_feats <= 4)
        is_covalently_bonded = is_covalently_bonded[is_sm][:, is_sm]
        G = nx.from_numpy_matrix(is_covalently_bonded.numpy())

        # Get the implied atom_frames
        sm_seq = self.seq[is_sm].cpu()
        atom_frames = rf2aa.util.get_atom_frames(
            sm_seq, 
            G, 
            omit_permutation=omit_permutation
            )

        # If atom_frames is empty, make it the "right" size
        if atom_frames.numel() == 0:
            atom_frames = torch.empty((0, 3, 2), dtype=torch.int64)

        return atom_frames.to(self.device)

    def chains(self):
        return chain_letters_from_same_chain(self.same_chain)
    
    def chain_masks(self):
        chain_i = []
        chains =  self.chains()
        for ch in sorted(np.unique(chains)):
            chain_i.append(ch == chains)
        return chain_i

    def write_pdb(self, path, **kwargs):
        with open(path, kwargs.pop('file_mode', 'w')) as fh:
            return self.write_pdb_file(fh, **kwargs)
    
    def write_pdb_file(self, fh, **kwargs):
        seq = self.seq
        seq = torch.where(seq == 20, 0, seq)
        seq = torch.where(seq == 21, 0, seq)
        chain_letters = self.chains()
        return write_file.writepdb_file(fh,
            atoms=torch.nan_to_num(self.xyz[:,:ChemData().NHEAVY]), 
            seq=seq, 
            idx_pdb=self.idx, 
            chain_letters=chain_letters, 
            bond_feats=self.bond_feats[None], 
            **kwargs
        )

    def ca_dists(self):
        xyz_ca = self.xyz[:,1]
        ca_sm = xyz_ca[self.is_sm]
        ca_prot = xyz_ca[~self.is_sm]
        dist = torch.norm(ca_prot[:,None] - ca_sm[None], dim=-1)
        dist = dist.min(dim=-1)[0]
        return dist
    
    def center_of_mass(self, mask=None):
        xyz_ca = self.xyz[:,1]
        if mask is not None:
            xyz_ca = xyz_ca[mask]
        return torch.mean(xyz_ca, dim=0)
    
    def length(self):
        return self.seq.shape[0]
      
    def has_c_terminal_residue(self):
        return is_monotonic(self.idx)

    def has_n_terminal_residue(self):
        return torch.flip(is_monotonic(-torch.flip(self.idx,(0,))),(0,))
    
    def human_readable_seq(self):
        return human_readable_seq(self.seq)
    
    def has_heavy_atoms_and_seq(self, atom_mask):
        want_atom_mask = ChemData().allatom_mask[self.seq]
        has_all_heavy_atoms = (want_atom_mask[:,:ChemData().NHEAVYPROT] == atom_mask[:,:ChemData().NHEAVYPROT]).all(dim=-1)
        has_sequence = torch.logical_or(self.seq < ChemData().UNKINDEX, torch.logical_and(self.seq >= ChemData().NPROTAAS, self.seq <= ChemData().MASKINDEXRNA))
        return has_all_heavy_atoms * ~self.is_sm * has_sequence
    
    def is_valid_for_atomization(self, atom_mask):
        return self.has_c_terminal_residue() * self.has_n_terminal_residue() * self.has_heavy_atoms_and_seq(atom_mask)

    def human_readable_atom_frames(self):
        atom_frames_absolute = self.atom_frames[:,:,0].clone()
        atom_frames_absolute += (torch.arange(self.length())[self.is_sm])[:,None]
        o = []

        def atom_label(i):
            return (i, ChemData().num2aa[self.seq[i]])
        for a,b,c in atom_frames_absolute.tolist():
            o.append(
                (atom_label(a), atom_label(b), atom_label(c))
            )
        return o
    
    def type(self):
        """Returns the type of each residue/atom in the indep.
        -1: unassigned
        0: protein
        1: ligand
        2: atomized covalent
        """
        chains = self.chains()
        chains_with_prot = np.unique(chains[~self.is_sm])
        is_on_same_chain_as_prot = torch.tensor(np.isin(chains, chains_with_prot))
        is_atomized_cov = self.is_sm * is_on_same_chain_as_prot
        is_ligand = self.is_sm * ~is_on_same_chain_as_prot
        metadata = {'type': torch.zeros(self.length())}
        metadata['type'][:] = -1
        metadata['type'][~self.is_sm] = TYPE_PROT
        metadata['type'][is_ligand] = TYPE_LIGAND
        metadata['type'][is_atomized_cov] = TYPE_ATOMIZED_COV
        return metadata['type']

    def get_connected(self, i):
        G = nx.from_numpy_matrix(self.bond_feats.detach().cpu().numpy())
        ic(G.nodes, i)
        connected_idx0 = fetch_connected_nodes(G, i)
        return torch.tensor(list(connected_idx0))

    def assert_types(self):
        assertpy.assert_that(self.same_chain.dtype).is_equal_to(torch.bool)

    def atom_label(self, i):
        return (i, ChemData().num2aa[self.seq[i]])

    def human_readable_2d_mask(self, mask):
        return [(self.atom_label(i), self.atom_label(j)) for i, j in mask.nonzero()]

    def human_readable_2d_symmetric_mask(self, mask):
        assertpy.assert_that((mask.T == mask).all()).is_true()
        return self.human_readable_2d_mask(torch.triu(mask))

    def is_contiguous_protein_monomer(self):
        idx_jump = self.idx[1:] - self.idx[:-1]
        contiguous = (idx_jump == 1).all().item()
        return (self.terminus_type[0] == N_TERMINUS and
                self.terminus_type[self.length()-1] == C_TERMINUS and
                not self.is_sm.any() and
                contiguous
                )

    def set_device(self, device):
        rf2aa.tensor_util.to_device(self, device)

    def chain_index(self):
        '''
        Returns a list of chain and pdb index like: [('A', 11), ('A', 12), ...]
        '''
        return list(zip(self.chains(), self.idx))

def what_is_diffused(indep, is_diffused, atomizer):
    point_ids = get_point_ids(indep, atomizer)
    return list(zip(point_ids, is_diffused.tolist()))

def human_readable_seq(seq):
    return [ChemData().num2aa[s] for s in seq]

def is_monotonic(idx):
    idx_pad = torch.concat([idx, torch.tensor([9999])])
    return (idx_pad[1:] - idx_pad[:-1] == 1)

def assert_valid_seq_mask(indep, is_masked_seq):
    if is_masked_seq[indep.is_sm].any():
        ic(
            list(zip(
                human_readable_seq(indep.seq[indep.is_sm]),
                is_masked_seq[indep.is_sm],
            ))
        )
        raise Exception('Sequence mask is invalid: atom indices are sequence masked.')

def get_atom_names(seq_token):
    atom_names = ChemData().aa2long[seq_token][:ChemData().NHEAVYPROT]
    return [a.strip() for a in atom_names if a is not None]

def make_is_motif14(seq, atom_name_by_res_idx):
    '''
    Converts a mapping of residue_index:atom_names to an LxN_HEAVY binary mask.

    Params:
        seq [L]: sequence tokens
        atom_name_by_res_idx: Dict[int]-> list<string> like {2:['CA', 'N']}
            The character CA_ONLY refers to index 1 in the final dimension of the output mask.
    '''
    within_res_atom_idx = {
        i: {
            atom_name.strip(): j
            for j, atom_name in enumerate(atom_names)
            if atom_name is not None
        }
        for i, atom_names in enumerate(ChemData().aa2long)
    }
    is_motif14 = torch.zeros((len(seq), ChemData().NHEAVY)).bool()
    for res_idx, atom_names in atom_name_by_res_idx.items():
        if atom_names == 'CA_ONLY':
            is_motif14[res_idx, 1] = True
        else:
            aa = seq[res_idx].item()
            for atom_name in atom_names:
                atom_i  = within_res_atom_idx[aa][atom_name.strip()]
                is_motif14[res_idx, atom_i] = True
    return is_motif14


backbone_atoms = ['N', 'C', 'CA', 'O']
POINT_LIGAND = 'L'
POINT_RESIDUE = 'R'
POINT_ATOMIZED_BACKBONE = 'AB'
POINT_ATOMIZED_SIDECHAIN = 'AS'
def get_point_types(indep, atomizer):
    t = np.empty(indep.length(), dtype='object')
    t[indep.is_sm.cpu().detach()] = POINT_LIGAND
    t[~indep.is_sm.cpu().detach()] = POINT_RESIDUE
    res_atom_by_i = {}
    if atomizer:
        res_atom_by_i = atomize.get_res_atom_name_by_atomized_idx(atomizer)
        for i, (_,  atom_name) in res_atom_by_i.items():
            if atom_name in backbone_atoms:
                t[i] = POINT_ATOMIZED_BACKBONE
            else:
                t[i] = POINT_ATOMIZED_SIDECHAIN
    return t

def get_point_ids(indep, atomizer):
    t = np.empty(indep.length(), dtype='object')
    seq_hr = indep.human_readable_seq()
    seq_hr = np.array(seq_hr, dtype='object')
    t[indep.is_sm.cpu().detach()] = [f'L-{element}' for element in seq_hr[indep.is_sm.cpu().detach()]]
    t[~indep.is_sm.cpu().detach()] = [f'R-{element}' for element in indep.idx[~indep.is_sm.cpu().detach()]]
    if atomizer:
        res_atom_by_i = atomize.get_res_atom_name_by_atomized_idx(atomizer)
        for i, (res_i,  atom_name) in res_atom_by_i.items():
            t[i] = f'A{res_i}-{atom_name}'
    return t

@dataclass
class RFI:
    msa_latent: torch.Tensor
    msa_full: torch.Tensor
    seq: torch.Tensor
    seq_unmasked: torch.Tensor
    xyz: torch.Tensor
    sctors: torch.Tensor
    idx: torch.Tensor
    bond_feats: torch.Tensor
    dist_matrix: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    t1d: torch.Tensor
    t2d: torch.Tensor
    xyz_t: torch.Tensor
    alpha_t: torch.Tensor
    mask_t: torch.Tensor
    same_chain: torch.Tensor
    is_motif: torch.Tensor
    msa_prev: torch.Tensor
    pair_prev: torch.Tensor
    state_prev: torch.Tensor

    def __post_init__(self):
        '''sanity check and tensor dimension naming'''
        self.add_tensor_dim_names()
        self.remove_tensor_dim_names()

    def add_tensor_dim_names(self):
        '''Adds dimension names to the tensors; needed for symmetry'''
        self.msa_latent  .rename_(*'B T L D'.split())
        self.msa_full    .rename_(*'B T L D'.split())
        self.seq         .rename_(*'B L'.split())
        self.seq_unmasked.rename_(*'B L'.split())
        self.xyz         .rename_(*'B L Atom XYZ'.split())
        self.sctors      .rename_(*'B L AA D'.split())
        self.idx         .rename_(*'B L'.split())
        self.bond_feats  .rename_(*'B L1 L2'.split())
        self.dist_matrix .rename_(*'B L1 L2'.split())
        self.chirals     .rename_(*'B Lsparse IdxAll0'.split())
        self.atom_frames .rename_(*'B Lsparse IdxAll0 D'.split())
        self.t1d         .rename_(*'B T L D'.split())
        self.t2d         .rename_(*'B T L1 L2 Pair'.split())
        self.xyz_t       .rename_(*'B T L XYZ'.split())
        self.alpha_t     .rename_(*'B T L D'.split())
        self.mask_t      .rename_(*'B T L1 L2'.split())
        self.same_chain  .rename_(*'B L1 L2'.split())
        self.is_motif    .rename_(*'L'.split())
        # self.msa_prev    .rename_('B T L D'.split())
        # self.pair_prev   .rename_('B T L D'.split())
        # self.state_prev  .rename_('B T L D'.split())

    def remove_tensor_dim_names(self):
        '''Removes dimension names from the tensors'''
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if torch.is_tensor(val): val.rename_(None)

class SymAdaptRFI(ipd.sym.SymAdaptDataClass):
    "Symmetrize RFI with the default dataclass SymAdaptor; uses add_tensor_dim_names"
    adapts = RFI

@dataclass
class RFO:
    logits: torch.Tensor      # ([1, 61, L, L], [1, 61, L, L], [1, 37, L, L], [1, 19, L, L])
    logits_aa: torch.Tensor   # [1, 80, 115]
    logits_pae: torch.Tensor  # [1, 64, L, L]
    logits_pde: torch.Tensor  # [1, 64, L, L]
    p_bind: torch.Tensor      # [1,1]
    xyz: torch.Tensor         # [40, 1, L, 3, 3]
    alpha_s: torch.Tensor     # [40, 1, L, 20, 2]
    xyz_allatom: torch.Tensor # [1, L, 36, 3]
    lddt: torch.Tensor        # [1, 50, L]
    msa: torch.Tensor
    pair: torch.Tensor
    state: torch.Tensor
    quat: torch.Tensor

    def __post_init__(self):
        '''sanity check and tensor dimension naming'''
        self.add_tensor_dim_names()
        self.remove_tensor_dim_names()

    def add_tensor_dim_names(self):
        '''Adds dimension names to the tensors; needed for symmetry'''
        for l in self.logits: l.rename_(*'B D L1 L2'.split())
        self.logits_aa  .rename_(*'B D L'.split())
        self.logits_pae .rename_(*'B D L1 L2'.split())
        self.logits_pde .rename_(*'B D L1 L2'.split())
        # self.p_bind     .rename_(*'D'.split())
        self.xyz        .rename_(*'R B L Atom XYZ'.split())
        self.alpha_s    .rename_(*'R B L AA D'.split())
        self.xyz_allatom.rename_(*'B L Atom XYZ'.split())
        self.lddt       .rename_(*'B D L'.split())
        if self.pair  is not None: self.pair .rename_(*'B L1 L2 Pair'.split())
        if self.msa   is not None: self.msa  .rename_(*'B L D'.split())
        if self.quat  is not None: self.quat .rename_(*'B D L Quat'.split())
        if self.state is not None: self.state.rename_(*'B L D'.split())

    def remove_tensor_dim_names(self):
        '''Removes dimension names from the tensors'''
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if torch.is_tensor(val): val.rename_(None)
            elif isinstance(val, tuple):
                for l in val: l.rename_(None)

    # dataclass.astuple returns a deepcopy of the dataclass in which
    # gradients of member tensors are detached, so we define a
    # custom unpacker here.
    def unsafe_astuple(self):
        return tuple(self.__dict__[field.name] for field in dataclasses.fields(self))

    def get_seq_logits(self):
        return self.logits_aa.permute(0,2,1)

    def get_xyz(self):
        return self.xyz_allatom[0]


class SymAdaptRFO(ipd.sym.SymAdaptDataClass):
    "Symmetrize RFO with the default dataclass SymAdaptor; uses add_tensor_dim_names"
    adapts = RFO


def get_refinement_metadata(meta:dict, conf:DictConfig):
    """
    Retrives information about the original diffusion run for refinement. 
    E.g., grab original motif coordinates, ij_visible info, etc. 

    Args: 
        meta: metadata from the original diffusion run
        conf: run config
    """
    pdb = conf.inference.input_pdb
    trb = pdb.replace('.pdb', '.trb')
    assert os.path.exists(trb), f"Must have .trb file for refinement, but trb file not found: {trb}"
    with open(trb, 'rb') as fp: 
        trb_dict = pickle.load(fp)

    og_conf = trb_dict['config']
    ref_dict = {'config'         : trb_dict['config'],
                'con_hal_idx0'   : trb_dict['con_hal_idx0'],
                'con_ref_idx0'   : trb_dict['con_ref_idx0'],
                'ij_visible'     : trb_dict['config']['inference']['ij_visible']
                }

    # get the original motif coordinates/seq
    motif_pdb = og_conf['inference']['input_pdb']
    motif_ligand = og_conf['inference']['ligand']
    motif_indep = make_indep(motif_pdb, motif_ligand, return_metadata=False)

    ref_dict['motif_indep'] = motif_indep
    meta['ref_dict'] = ref_dict

    # Make a contig string which encodes a contig map that 
    # has motifs in the same location that diffusion run had them. 
    L_nolig = len(trb_dict['mask_1d'])
    ref = trb_dict['contigmap_ref'][:L_nolig]
    hal = trb_dict['contigmap_hal'][:L_nolig]
    conf.contigmap.contigs = contigs.get_refinement_contigs_from_ref_and_hal(ref, hal)

    # reset the output prefix as the pdb with '_refined' tag
    conf.inference.output_prefix = pdb.replace('.pdb','_refined')

    # reset ij_visible to whatever it was in diffusion 
    conf.inference.ij_visible = og_conf['inference']['ij_visible']


def get_ligands(pdb_lines):
    ligands = set()
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        curr_ligand = l[17:17+4].strip()
        ligands.add(curr_ligand)
    return ligands

def get_only_ligand_or_none(pdb_lines):
    ligands = get_ligands(pdb_lines)
    assertpy.assert_that(len(ligands), description=ligands).is_less_than_or_equal_to(1)
    if len(ligands) == 0:
        return None
    return list(ligands)[0]

def get_only_ligand(pdb_lines):
    ligands = get_ligands(pdb_lines)
    assertpy.assert_that(len(ligands), description=ligands).is_equal_to(1)
    return list(ligands)[0]

def get_non_target_hetatm_ids(pdb_lines, ligands):
    non_target_hetatm_ids = []
    for l in pdb_lines:
        if l.startswith('HETATM'):
            curr_ligand = l[17:17+4].strip()
            if curr_ligand not in ligands:
                non_target_hetatm_ids.append(int(l[6:6+5].strip()))
                continue
    return set(non_target_hetatm_ids)

def filter_connect(l, invalid_ids):
    return filter_connect_by_func(l, lambda d, r: d in invalid_ids or r in invalid_ids)

def filter_connect_by_func(l, is_invalid):
    '''
    is_invalid: (donor:int, receptor:int) --> bool
    l: PDB CONECT line
    '''
    ids = [int(e.strip()) for e in l[6:].strip().split()]
    implied_bonds = []
    d = ids[0]
    R = ids[1:]

    implied_bonds = [(d, r) for r in R]

    valid_bonds = []
    invalid_bonds = []
    for d, r in implied_bonds:
        if is_invalid(d,r):
            invalid_bonds.append((d,r))
        else:
            valid_bonds.append((d,r))

    if not valid_bonds:
        return '', invalid_bonds

    d = valid_bonds[0][0]
    R = [b[1] for b in valid_bonds]
    new_l = ['CONECT']
    for atom_id in [d] + R:
        new_l.append(f'{atom_id:>5}')

    return ''.join(new_l), invalid_bonds

def remove_non_target_ligands(pdb_lines: list[str], ligands: list[str]) -> list[str]:
    """
    Removes non-target ligands from the PDB lines

    Args:
        pdb_lines (list): list of PDB lines from a PDB parser
        ligands (list): list of ligands to be removed.

    Returns:
        list[str]: list of PDB lines
    """
    non_target_hetatm_ids = get_non_target_hetatm_ids(pdb_lines, ligands)
    lines = []
    all_invalid_bonds = []
    all_invalid_hetatms = []
    for l in pdb_lines:
        if 'HETATM' in l:
                atom_id = int(l[6:6+5].strip())
                if atom_id in non_target_hetatm_ids:
                    all_invalid_hetatms.append(atom_id)
                    continue
        if 'CONECT' in l:
            l, invalid_bonds = filter_connect(l, non_target_hetatm_ids)
            all_invalid_bonds.extend(invalid_bonds)
        lines.append(l)

    # Uncomment to see ignored atoms/bonds
    # if all_invalid_bonds or all_invalid_hetatms:
    #     atom_by_serial_number = get_atom_by_atom_serial_number(pdb_lines)
    #     name = lambda x: str(atom_by_serial_number.get(x, 'UNKNOWN'))
    #     print(f'Removing hetatms:\n' + '\n'.join([name(atom_id) for atom_id in all_invalid_hetatms]))
    #     print(f'Removing the following bonds:' + '\n'.join(f'{name(d)} --> {name(d)}' for d,r in all_invalid_bonds))

    return lines

def filter_het(pdb_lines, ligands, covale_allowed=False, keep_conect=True):
    """
    Args:
        keep_conect (bool): If True, keep CONECT lines that reference atoms in the target ligands
    """
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        curr_ligand = l[17:17+4].strip()
        if curr_ligand not in ligands:
            continue
        lines.append(l)
        hetatm_ids.append(int(l[7:7+5].strip()))

    violations = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        if all(i in hetatm_ids for i in ids):
            if keep_conect:
                lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            ligand_atms_bonded_to_protein = [i for i in ids if i in hetatm_ids]
            violations.append(f'line {l} references atom ids in the target ligands {ligands}: {ligand_atms_bonded_to_protein} and another atom')
    if violations and not covale_allowed:
        raise Exception('\n'.join(violations))
    return lines

def get_hetatm_ids(pdb_lines, ligands):
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        curr_ligand = l[17:17+4].strip()
        if curr_ligand not in ligands:
            continue
        lines.append(l)
        hetatm_ids.append(int(l[7:7+5].strip()))
    return hetatm_ids

def get_bonds(pdb_lines):
    from_to = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        from_to.extend(tuple(sorted((ids[0], r))) for r in ids[1:])
    return list(set(from_to))

import io
from Bio import PDB
from Bio.PDB import PDBParser
p = PDBParser(PERMISSIVE=0, QUIET=1)
def get_atom_by_atom_serial_number(pdb_lines):
    buffer = io.StringIO()
    for l in pdb_lines:
        buffer.write(l)
    buffer.seek(0)
    struct = p.get_structure('none', buffer)
    atomList = PDB.Selection.unfold_entities(struct, target_level='A')
    atom_by_id = {atom.serial_number:atom for atom in atomList}
    return atom_by_id

def find_covale_bonds(pdb_lines, ligand):
    # res_atom_name_by_atom_id = get_res_atom_name_by_atom_id(pdb_lines)
    # covale_bonds = filter_het(pdb_lines, ligand)

    hetatm_ids = get_hetatm_ids(pdb_lines, ligand)
    bonds = get_bonds(pdb_lines)

    hetatm_id_set = set(hetatm_ids)

    protein_ligand_bonds = []
    for d, r in bonds:
        if (d in hetatm_id_set) != (r in hetatm_id_set):
            protein_ligand_bonds.append(sorted((d,r), key=lambda x: x in hetatm_id_set))

    atom_by_serial_number = get_atom_by_atom_serial_number(pdb_lines)
    for i, (d,r) in enumerate(protein_ligand_bonds):
        protein_ligand_bonds[i] = (
            atom_by_serial_number[d],
            atom_by_serial_number[r],
        )

    return protein_ligand_bonds

def get_atom_uid(a):
    _, _, ch, (ligand_name, res_idx, _), (atom_name, _) = a.get_full_id()
    return (res_idx, atom_name)


def parse_ligand(pdb, ligand, pdb_stream=None):
    '''
    Parse a ligand from a pdb file

    Args:
        pdb (str or None): Path to pdb file
        ligand (str): The ligand to parse
        pdb_stream (list[str] or None): If present the pdb will not be read and this will be assumed to be its contents
    '''
    if pdb_stream is None:
        with open(pdb, 'r') as fh:
            pdb_stream = fh.read_lines()
    stream = [l for l in pdb_stream if "HETATM" in l or "CONECT" in l]
    if ligand == UNIQUE_LIGAND:
        raise NotImplementedError

    stream = filter_het(stream, [ligand], covale_allowed=True)
    if not len(stream):
        raise Exception(f'ligand {ligand} not found in pdb: {pdb}')

    mol, seq_sm, _, xyz_sm, _ = parsers.parse_mol("".join(stream), filetype="pdb", string=True, find_automorphs=False)
    G = rf2aa.util.get_nxgraph(mol)
    atom_frames = rf2aa.util.get_atom_frames(seq_sm, G, omit_permutation=False)
    chirals = get_chirals(mol, xyz_sm[0])
    bond_feats = rf2aa.util.get_bond_feats(mol)

    atom_names = []
    for line in stream:
        if line.startswith('HETATM'):
            atom_type = line[76:78].strip()
            if atom_type == 'H':
                continue
            atom_names.append(line[12:16])
    atom_names = np.array(atom_names, dtype='<U4')

    return xyz_sm[0], seq_sm, atom_frames, chirals, bond_feats, atom_names
    # if chirals.numel() !=0:
    #     chirals[:,:-1] += protein_L

def get_atom_mask(pdb):
    with open(pdb, 'r') as fh:
        stream = fh.readlines()
    target_feats = parse_pdb_lines_target(stream, parse_hetatom=True)
    _, mask_prot, _, _ = target_feats['xyz'], target_feats['mask'], target_feats['idx'], target_feats['seq']
    mask_prot[:,ChemData().NHEAVY:] = False
    return mask_prot

def make_indep(pdb, ligand='', return_metadata=False, pdb_stream=None):
    """
    Creates an Indep from a pdb file for use in inference, or file rewriting.
    NOTE: this function does not center anything, which means coordiantes can be large

    Args:
        pdb (str or None): Path to pdb file
        ligand (str): Ligand names to include in Indep separated by commas (',')
        return_metadata (bool): whether or not to return metadata
        pdb_stream (list[str] or None): If present the pdb will not be read and this will be assumed to be its contents
    """
    # self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
    # init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False, init_ligand_xyz=False,
    #     parse_hetatm=False, n_cycle=10, random_noise=5.0)
    chirals = torch.zeros((0, 5))
    atom_frames = torch.zeros((0,3,2), dtype=torch.int64)

    # xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb, seq=True, parse_hetatom=True)

    if pdb_stream is None:
        with open(pdb, 'r') as fh:
            pdb_stream = fh.readlines()
    stream = pdb_stream

    ligands = []
    if ligand:
        ligands = ligand.split(',')

    stream = remove_non_target_ligands(stream, ligands)
    target_feats = parse_pdb_lines_target(stream, parse_hetatom=True)
    het_atom_uids = [(e['res_idx'], e['atom_id'].strip()) for e in target_feats['info_het']]
    prot_atom_uids = [(idx, 'CA') for idx in target_feats['idx']]
    uids = prot_atom_uids + het_atom_uids
    xyz_polymer, mask_polymer, idx_polymer, seq_polymer = target_feats['xyz'], target_feats['mask'], target_feats['idx'], target_feats['seq']
    is_nuc = nucl_utils.get_resi_type_mask(seq_polymer, 'na')
    is_pro = nucl_utils.get_resi_type_mask(seq_polymer, 'prot')

    xyz_polymer[is_pro,ChemData().NHEAVYPROT:] = 0 # remove hydrogens
    mask_polymer[is_pro,ChemData().NHEAVYPROT:] = False
    xyz_polymer[is_nuc,ChemData().NHEAVY:] = 0 # remove hydrogens
    mask_polymer[is_nuc,ChemData().NHEAVY:] = False
    xyz_polymer = torch.tensor(xyz_polymer)
    mask_polymer = torch.tensor(mask_polymer)
    protein_L, nprotatoms, _ = xyz_polymer.shape
    seq_polymer = torch.tensor(seq_polymer).long()
    covale_bonds = []

    #Ls = [seq_polymer.shape[0]]
    Ls, is_protein, is_protein_chain = nucl_utils.find_protein_dna_chains(target_feats['pdb_idx'], seq_polymer)
    N_poly_resi = sum(Ls)
    N_poly = len(Ls)
    seq_sm = torch.zeros((0,)).long()
    if len(ligands):
        protein_ligand_bonds_atoms = find_covale_bonds(stream, ligands)
        # Hack, no way to detect bond types in PDB
        bond_type = 1 # Single bond
        # Debugging
        # msg = []
        # for d,r in protein_ligand_bonds_atoms:
        #     msg.append(f'{d.get_full_id()} : {r.get_full_id()}')
        # msg = '\n'.join(msg)
        # if msg:
        #     print(f'Protein-ligand bonds:\n{msg}')
        # else:
        #     print(f'No protein-ligand bonds')

        for protein_atom, ligand_atom in protein_ligand_bonds_atoms:
            prot_res_idx, prot_atom_name = get_atom_uid(protein_atom)
            res_i = uids.index((prot_res_idx, 'CA'))
            ligand_atom_uid = get_atom_uid(ligand_atom)
            atom_i = uids.index((ligand_atom_uid))
            covale_bonds.append(
                ((res_i, prot_atom_name), atom_i, bond_type)
            )

        xyz_sm_stack = []
        seq_sm_stack = []
        atom_frames_stack = []
        chirals_stack = []
        bond_feats_stack = []
        sm_atom_names_stack = []
        # covale_bonds_stack = []
        for ligand in ligands:
            o = parse_ligand(pdb, ligand, pdb_stream=pdb_stream)
            xyz_sm, seq_sm, atom_frames, chirals, bond_feats, atom_names = o

            chirals[:, :-1] += sum(Ls)
            xyz_sm_stack.append(xyz_sm)
            seq_sm_stack.append(seq_sm)
            atom_frames_stack.append(atom_frames)
            chirals_stack.append(chirals)
            bond_feats_stack.append(bond_feats)
            sm_atom_names_stack.append(atom_names)
            # covale_bonds_stack.append(covale_bonds)

            Ls.append(seq_sm.shape[0])

        xyz_sm = torch.cat(xyz_sm_stack)
        seq_sm = torch.cat(seq_sm_stack)
        atom_frames = torch.cat(atom_frames_stack)
        chirals = torch.cat(chirals_stack)
        bond_feats_sm = torch.block_diag(*bond_feats_stack)
        sm_atom_names = np.concatenate(sm_atom_names_stack)
    else:
        Ls.append(0)

    xyz = torch.full((sum(Ls), ChemData().NTOTAL, 3), np.nan).float()
    xyz[:N_poly_resi, :nprotatoms, :] = xyz_polymer
    if ligand:
        xyz[N_poly_resi:, 1, :] = xyz_sm
    idx_sm = torch.arange(max(idx_polymer),max(idx_polymer)+Ls[N_poly])+200 # Access last element of Ls, which is ligand
    idx_sm_stack = []
    last_idx = max(idx_polymer)
    for l in Ls[N_poly:]:
        new_idx = torch.arange(l) + 200 + last_idx
        idx_sm_stack.append(new_idx)
        if len(new_idx):
            last_idx = max(new_idx)
    idx_sm = torch.cat(idx_sm_stack)

    idx_pdb = torch.concat([torch.tensor(idx_polymer), idx_sm])
    seq = torch.cat((seq_polymer, seq_sm))
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    # Assign protein bond feats
    l_sum = 0
    for i, l in enumerate(Ls[:N_poly]):
        if is_protein_chain[i]:
            bond_feats[l_sum:l_sum+l, l_sum:l_sum+l] = rf2aa.util.get_protein_bond_feats(Ls[i])
        l_sum += l
    # Assign ligand bond feats
    if ligand:
        bond_feats[sum(Ls[:N_poly]):, sum(Ls[:N_poly]):] = bond_feats_sm

    #same_chain = torch.zeros((sum(Ls), sum(Ls))).bool()
    #same_chain[:Ls[0], :Ls[0]] = True
    #for i in range(N_poly):
    #    same_chain[Ls[i]:Ls[i+1], Ls[i]:Ls[i+1]] = True
    #same_chain[Ls[N_poly-1]:, Ls[N_poly-1]:] = True
    chains = []
    for i, l in enumerate(Ls):
        chains.extend([alphabet[i]] * l)
    same_chain = same_chain_from_chain_letters(np.array(chains))
    # Amend same chain for the covalently linked small molecule case
    same_chain = same_chain_with_covale(same_chain, covale_bonds)

    is_sm = torch.zeros(sum(Ls)).bool()
    is_sm[sum(Ls[:N_poly]):] = True
    # assert len(Ls) <= 2, 'multi chain inference not implemented yet'
    terminus_type = torch.zeros(sum(Ls))
    # Assign proteins to terminus type
    l_sum = 0
    for i, l in enumerate(Ls[:N_poly]):
        if is_protein_chain[i]:
            terminus_type[l_sum] = N_TERMINUS
            terminus_type[l_sum+l-1] = C_TERMINUS
        l_sum += l

    ###TODO: currently network needs values at 0,2 indices of tensor, need to remove this reliance
    xyz[is_sm, 0] = 0
    xyz[is_sm, 2] = 0
    indep = Indep(
        seq=seq,
        xyz=xyz,
        idx=idx_pdb,
        # SM specific
        bond_feats=bond_feats,
        chirals=chirals,
        same_chain=same_chain,
        terminus_type=terminus_type,
    )

    if return_metadata:
        ligand_name_arr = []
        for l, name in zip(Ls, N_poly * [''] + ligands):
            ligand_name_arr.extend(l * [name])
        ligand_atom_names_arr = ['']*sum(Ls[:N_poly])
        if ligand:
            ligand_atom_names_arr.extend(sm_atom_names)

        metadata = {
            'covale_bonds': covale_bonds,
            'ligand_names': np.array(ligand_name_arr,  dtype='<U3'),
            'ligand_atom_names': np.array(ligand_atom_names_arr,  dtype='<U4'),
        }
        return indep, metadata
    
    return indep

def add_fake_frame_legs(xyz, is_atom, generator=None):
    # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
    # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
    xyz = xyz.clone()
    atom_xyz = xyz[is_atom, 1]
    xyz[is_atom,:3] = atom_xyz[...,None,:]
    xyz[is_atom, 0] += torch.normal(torch.zeros_like(xyz[is_atom, 0]), std=1.0, generator=generator)
    xyz[is_atom, 2] += torch.normal(torch.zeros_like(xyz[is_atom, 2]), std=1.0, generator=generator)
    return xyz


def indep_from_sequence(sequence_str=None, sequence_numeric=None):
    '''
    Build an indep with extended backbone from sequence

    Works exactly like pyrosetta.pose_from_sequence()

    This isn't a quick function. If you are relying on this to be fast, maybe we should rethink how this works...

    Args:
        sequence_str (str): A string of the sequence ("ACDEFG")
        sequence_numeric (torch.tensor[int]): A tensor of the sequence like found in indep ([0, 1, 2, 3, 4])

    Returns:
        indep (indep): An indep with the sequence you specified
    '''

    assert (sequence_str is None) != (sequence_numeric is None), 'One of sequence_str and sequence_numeric should be specified'

    if sequence_numeric is None:
        sequence_numeric = torch.tensor([ChemData().one_letter.index(s) for s in sequence_str])

    xyz, atom_mask = build_coords.extended_ideal_xyz_from_seq(sequence_numeric, include_hydrogens=False)

    # So this is definitely not the most elegant way to make an indep
    # BUT! It is definitely the way to ensure we do this in the most accurate way possible
    fh = io.StringIO()
    write_file.writepdb_file(fh, xyz, sequence_numeric)
    fh.seek(0)

    return make_indep('indep_from_sequence', pdb_stream=fh.readlines())



class Model:

    def __init__(self, conf):
        self.conf = conf
        self.NTOKENS = ChemData().NAATOKENS
        self.atomizer = None
        self.converter = XYZConverter()

    def forward(self, rfi, **kwargs):
        rfi_dict = dataclasses.asdict(rfi)
        return RFO(*self.model(**{**rfi_dict, **kwargs}))

    def insert_contig_pre_atomization(self, indep: Indep, contig_map: ContigMap, metadata: dict = None, for_partial_diffusion: bool = False):
        """
        Performs all inference time contig insertion tasks for Indep that are upstream to transform_indep

        Args:
            indep (Indep): the holy Indep
            contig_map (ContigMap): the contig map prior to any indep insertion
            metadata (dict): ligand metadata
            for_partial_diffusion (bool): whether or not partial diffusion is to be used
        """
        if metadata is None:
            print("warning, insert contig with no metadata is not handled gracefully")
            metadata = defaultdict(dict)
            metadata['ligand_names'] = np.array([])
            metadata['ligand_atom_names'] = np.array([])
        o = copy.deepcopy(indep)

        # Insert small mol into contig_map
        all_chains = {ch for ch,_ in contig_map.hal}
        next_unused_chain = (e for e in contig_map.chain_order if e not in all_chains)
        n_sm = indep.is_sm.sum()
        is_sm_idx0 = torch.nonzero(indep.is_sm, as_tuple=True)[0].tolist()
        contig_map.ref_idx0.extend(is_sm_idx0)
        n_protein_hal = len(contig_map.hal)
        contig_map.hal_idx0 = np.concatenate((contig_map.hal_idx0, np.arange(n_protein_hal, n_protein_hal+n_sm)))
        max_hal_idx = max(i for _, i  in contig_map.hal)
        new_chains = {}
        for ch in np.unique(indep.chains()[indep.is_sm]):
            new_chains[ch] = next(next_unused_chain)

        new_chain_arr = np.array([new_chains[ch] for ch in indep.chains()[indep.is_sm]])
        new_idx_arr = np.full((n_sm), np.nan, dtype=np.int64)
        chs, n_chs = np.unique(new_chain_arr, return_counts=True)
        for ch, n_ch in zip(chs, n_chs):
            new_idx_ch = np.arange(n_ch) + max_hal_idx+200
            new_idx_arr[new_chain_arr == ch] = new_idx_ch
            max_hal_idx = np.max(new_idx_ch)

        chain_start_end = chain_start_end_from_hal(contig_map.hal)

        contig_map.hal.extend(zip(new_chain_arr, new_idx_arr))
        contig_map.ref.extend(zip(metadata['ligand_names'][indep.is_sm],metadata['ligand_atom_names'][indep.is_sm]))
        chain_id = np.array([c for c, _ in contig_map.hal])
        L_mapped = len(contig_map.hal)
        n_prot = L_mapped - n_sm
        L_in, NATOMS, _ = indep.xyz.shape
        if not for_partial_diffusion:
            o.xyz = torch.full((L_mapped, NATOMS, 3), np.nan)

        o.xyz[contig_map.hal_idx0] = indep.xyz[contig_map.ref_idx0]
        o.seq = torch.full((L_mapped,), ChemData().MASKINDEX)

        assert len(contig_map.mol_classes)==len(chain_start_end), f"contig error: number of mol_classes ({len(contig_map.mol_classes)}) must match number of chains ({len(chain_start_end)})."
        
        for mol_class, (chain_start, chain_end) in zip(contig_map.mol_classes, chain_start_end):
            o.seq[chain_start:chain_end] = nucl_utils.mask_ind_by_class[mol_class]

        o.seq[contig_map.hal_idx0] = indep.seq[contig_map.ref_idx0]
        o.same_chain = torch.tensor(chain_id[None, :] == chain_id[:, None])
        if not for_partial_diffusion:
            # Required for se3 optimal transport aligner to work
            o.xyz = rf_diffusion.kinematics.get_init_xyz(o.xyz[None, None], o.is_sm, center=False).squeeze()

        o.bond_feats = torch.full((L_mapped, L_mapped), 0).long()
        o.bond_feats[:n_prot, :n_prot] = rf2aa.util.get_protein_bond_feats(n_prot)
        n_prot_ref = L_in-n_sm
        o.bond_feats[n_prot:, n_prot:] = indep.bond_feats[n_prot_ref:, n_prot_ref:]

        hal_by_ref_d = dict(zip(contig_map.ref_idx0, contig_map.hal_idx0))
        def hal_by_ref(ref):
            return hal_by_ref_d[ref]

        hal_by_ref = np.vectorize(hal_by_ref, otypes=[float])
        o.chirals[...,:-1] = torch.tensor(hal_by_ref(o.chirals[...,:-1]))

        o.idx = torch.tensor([i for _, i in contig_map.hal])

        o.terminus_type = torch.zeros(L_mapped)

        assert len(self.conf.contigmap.has_termini) == contig_map.n_inpaint_chains, "Please specify in contigmap.has_termini for which chains you want to have the termini present."

        for use_termini, (chain_start, chain_end) in zip(self.conf.contigmap.has_termini,chain_start_end):
            if chain_end < L_mapped:
                o.bond_feats[chain_end][chain_end - 1] = 0
                o.bond_feats[chain_end - 1][chain_end] = 0

            if use_termini:
                o.terminus_type[chain_start] = N_TERMINUS
                o.terminus_type[chain_end-1] = C_TERMINUS

        # this line looks wrong but it's because inpaint_str is defined backwards. inpaint_str[i] == False means diffuse this residue's structure
        is_res_str_shown_prot = torch.from_numpy(contig_map.inpaint_str)
        is_res_str_shown_sm = torch.ones(n_sm, dtype=bool)

        partially_fixed_ligand = self.conf.inference.partially_fixed_ligand or {}
        assert not (self.conf.inference.flexible_ligand and len(partially_fixed_ligand)), 'inference.flexible_ligand and inference.partially_fixed_ligand and mutually exclusive'
        if self.conf.inference.flexible_ligand:
            is_res_str_shown_sm[:] = False
        
        if partially_fixed_ligand:
            for ligand_name, atom_ids in self.conf.inference.partially_fixed_ligand.items():
                if isinstance(ligand_name, int):
                    raise Exception(f'bad parse of {self.conf.inference.partially_fixed_ligand=}: hydra does not allow int keys: {ligand_name=} {type(ligand_name)=}')

            is_res_str_shown_sm = torch.zeros(n_sm).bool()
            within_sm_index_by_name_atom = defaultdict(dict)
            for i, (ligand_name, atom_id) in enumerate(zip(metadata['ligand_names'][indep.is_sm], metadata['ligand_atom_names'][indep.is_sm])):
                atom_id = atom_id.strip()
                within_sm_index_by_name_atom[ligand_name][atom_id] = i
            violations_by_ligand = defaultdict(list)
            for ligand_name, atom_ids in self.conf.inference.partially_fixed_ligand.items():
                for atom_id in atom_ids:
                    if atom_id not in within_sm_index_by_name_atom[ligand_name]:
                        violations_by_ligand[ligand_name].append(atom_id)
                        continue
                    i = within_sm_index_by_name_atom[ligand_name][atom_id]
                    is_res_str_shown_sm[i] = True
            if len(violations_by_ligand) > 0:
                msg = []
                for ligand_name, atom_ids in violations_by_ligand.items():
                    msg.append(f'{ligand_name} has {len(within_sm_index_by_name_atom[ligand_name])} ids: {within_sm_index_by_name_atom[ligand_name]}, missing {len(atom_ids)}/{len(self.conf.inference.partially_fixed_ligand[ligand_name])} partially fixed atoms: {atom_ids}')
                msg = '\n'.join(msg)
                raise Exception(f'atom_ids not found in ligand: {msg}')

        is_res_str_shown = torch.cat((is_res_str_shown_prot, is_res_str_shown_sm))
        is_atom_str_shown = contig_map.atomize_indices2atomname
        inpaint_seq = torch.from_numpy(contig_map.inpaint_seq)
        # Although the "shown" state of atomized residues is not used for the generated atoms, transforms like Center use this information.
        #   The following lines must remain here so that the regression tests continue to pass
        if is_atom_str_shown:
            is_res_str_shown[list(is_atom_str_shown.keys())] = False
            inpaint_seq[list(is_atom_str_shown.keys())] = False # counter-intuitively, False here means mask this position

        # this line looks wrong but it's because inpaint_seq is defined backwards. inpaint_seq[i] == False means mask this residue's sequence
        is_res_seq_shown = torch.cat((inpaint_seq, is_res_str_shown_sm))

        # If the user hasn't specified only_guidepost_positions then all motif residues and atomized residues can be guideposts
        if self.conf.inference.only_guidepost_positions is None:
            can_be_gp = is_res_str_shown & is_res_seq_shown
            if is_atom_str_shown:
                can_be_gp[list(is_atom_str_shown)] = True
        else:
            assert isinstance(self.conf.inference.only_guidepost_positions, str), (
                f'inference.only_guidepost_positions must be a string! You passed "{self.conf.inference.only_guidepost_positions}"')
            can_be_gp = contig_map.res_list_to_mask(self.conf.inference.only_guidepost_positions)

        for i, ((res_i, atom_name), sm_i, bond_type) in enumerate(metadata['covale_bonds']):
            res_i = hal_by_ref_d[res_i]
            sm_i = hal_by_ref_d[sm_i]
            metadata['covale_bonds'][i] = ((res_i, atom_name), sm_i, bond_type)

        o.same_chain = same_chain_with_covale(o.same_chain, metadata['covale_bonds'])

        # Handle is_gp the same way that sequence was handled
        o.is_gp = torch.zeros(L_mapped, dtype=bool)
        o.is_gp[contig_map.hal_idx0] = indep.is_gp[contig_map.ref_idx0]

        masks_1d = {
            'input_str_mask': is_res_str_shown,
            'input_seq_mask': is_res_seq_shown,
            'is_atom_motif': is_atom_str_shown,
            'can_be_gp': can_be_gp
        }
        return o, masks_1d


    def prepro(self, indep: Indep, t: float, is_diffused: torch.Tensor) -> RFI:
        """
        Function to prepare inputs to diffusion model

        Parameters: 
            - indep (aa_model.Indep): Indep dataclass
            - t (float): current timestep 
            - is_diffused (torch.Tensor): mask of which atoms are diffused 

        Returns:
            - rfi (aa_model.RFI): RFI dataclass with features ready for input to model
                - seq (L,22) one-hot sequence
                - msa_masked (1,1,L,48)
                - msa_full (1,1,L,25)
                - xyz_t (L,14,3) template crds (diffused)
                - t1d (1,L,28) this is the t1d before tacking on the chi angles:
                    - seq + unknown/mask (21)
                    - global timestep (1-t/T if not motif else 1) (1)
                    - contacting residues: for ppi. Target residues in contact with biner (1)
                    - chi_angle timestep (1)
                    - ss (H, E, L, MASK) (4)
                - t2d (1, L, L, 45)
                    - last plane is block adjacency
        """
        xyz_t = indep.xyz  # [L, N_ATOMS_PER_TOKEN, 3] here (although only N CA C CB are used?)
        seq_one_hot = torch.nn.functional.one_hot(
                indep.seq, 
                num_classes=self.NTOKENS
        ).float()  # <-- used to fabricate fake MSAs
        
        # Set basic RF2AA input shape variables
        B = 1  # batch size (always 1 here as not used in diffusion)
        R = 1  # number of recycles (always 1 here as not used in diffusion)
        L = seq_one_hot.shape[0]  # number of tokens (residues + atoms)
        NUM_AA_TOKENS = ChemData().NAATOKENS  # number of AA tokens
        NUM_TERMINI = 2  # number of terminus channels
        NUM_INDEL = 1  # number of indel channels

        ##########################################
        ## Create standard RF2AA input features ##
        ##########################################

        ### msa_masked ###
        ##################
        # `msa_masked` has channels: N_AA_TOKENS, N_AA_TOKENS, N_INDEL, N_INDEL, N_TERMINUS
        msa_masked = torch.zeros((B,R,L,2*NUM_AA_TOKENS + 2*NUM_INDEL + NUM_TERMINI))

        msa_masked[..., :NUM_AA_TOKENS] = seq_one_hot[None, None]
        msa_masked[..., NUM_AA_TOKENS:2*NUM_AA_TOKENS] = seq_one_hot[None, None]

        if self.conf.preprocess.annotate_termini:
            # ... annotate N-terminus
            MSAMASKED_N_TERM = 2*NUM_AA_TOKENS + 2*NUM_INDEL
            msa_masked[..., MSAMASKED_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
            # ... annotate C-terminus
            MSAMASKED_C_TERM = MSAMASKED_N_TERM + 1
            msa_masked[..., MSAMASKED_C_TERM] = (indep.terminus_type == C_TERMINUS).float()


        ### msa_full ###
        ################
        # `msa_full` has channels: N_AA_TOKENS, N_INDEL, N_TERMINUS
        msa_full = torch.zeros((B,R,L,NUM_AA_TOKENS + NUM_INDEL + NUM_TERMINI))
        msa_full[..., :NUM_AA_TOKENS] = seq_one_hot[None, None]
        
        if self.conf.preprocess.annotate_termini:
            # ... annotate N-terminus
            MSAFULL_N_TERM = NUM_AA_TOKENS + NUM_INDEL
            msa_full[..., MSAFULL_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
            # ... annotate C-terminus
            MSAFULL_C_TERM = MSAFULL_N_TERM + 1
            msa_full[..., MSAFULL_C_TERM] = (indep.terminus_type == C_TERMINUS).float()


        ### t1d ###
        ###########
        # Here we need to go from one hot with 22 classes to one hot with 21 classes
        # If sequence is masked, it becomes unknown
        # t1d = torch.zeros((B,R,L,NAATOKENS-1))

        #seqt1d = torch.clone(seq)
        seq_cat_shifted = seq_one_hot.argmax(dim=-1)
        seq_cat_shifted[seq_cat_shifted >= ChemData().MASKINDEX] -= 1
        t1d = torch.nn.functional.one_hot(seq_cat_shifted, num_classes=ChemData().NAATOKENS-1)
        t1d = t1d[None, None] # [L, NAATOKENS-1] --> [B,R,L, NAATOKENS-1]

        ## Str Confidence (structure confidence)
        # ... set confidence to 1 where diffusion mask is True, else 1-t/T
        strconf = torch.zeros((L,)).float().to(is_diffused.device)
        strconf[~is_diffused] = 1.
        strconf[is_diffused] = 1. - t/self.conf.diffuser.T  # [L]
        strconf = strconf[None,None,...,None]  # [B,R,L,1]

        # ... and append as last channel in t1d
        t1d = torch.cat((t1d, strconf), dim=-1) # [B,R,L,79+1]
        t1d = t1d.float()


        ### xyz_t ###
        #############
        if self.conf.preprocess.sidechain_input:
            raise NotImplementedError
        else:
            # ... set all coordinates except those for the backbone 
            #     `N`, `CA`, `C` atoms for nucleotides or the
            #     `O4'`, `C1'` , `C2'` atoms for nucleotides or 
            #     the atomized atom coordinates (in the CA channel) to `nan`
            xyz_t[is_diffused, 3:, :] = float('nan')  # < if sidechain wasn't gone before, it's gone now


        # ... can either have shortened form, or full form
        assert_that(tuple(xyz_t.shape)).is_in(*[(L,ChemData().NHEAVY,3),
                                                (L, ChemData().NTOTAL, 3)])
        # ... chop off hydrogens (without this, there is a bug due to torsions)
        xyz_t = xyz_t[:,:ChemData().NHEAVY]
        xyz_t = xyz_t[None, None]

        # ... pad dimension if necessary
        xyz_t = torch.cat(
            (
                xyz_t, 
                torch.full((B,1,L,ChemData().NTOTAL-xyz_t.shape[3],3), float('nan')).to(xyz_t.device)
            ), 
            dim=3
        )  # [B(1), T(1), L, N_ATOMS_PER_TOKEN(36), 3]


        ### t2d ###
        ###########
        t2d = None
        # t2d = xyz_to_t2d(xyz_t)
        # B = 1
        # zeros = torch.zeros(B,1,L,36-3,3).float().to(px0_xyz.device)
        # xyz_t = torch.cat((px0_xyz.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]
        # t2d, mask_t_2d_remade = get_t2d(
        #     xyz_t[0], mask_t[0], seq_scalar[0], same_chain[0], atom_frames[0])
        # t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]


        ### idx ###
        ###########
        """
        idx = torch.arange(L)[None]
        if ppi_design:
            idx[:,binderlen:] += 200
        """
        # JW Just get this from the contig_mapper now. This handles chain breaks
        #idx = torch.tensor(self.contig_map.rf)[None]


        # ### alpha_t ###
        # ###############
        # ... get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)  # ... (everything up to last dim which is timestep embedding)
        alpha, _, alpha_mask, _ = self.converter.get_torsions(xyz_t.reshape(-1,L,ChemData().NTOTAL,3), seq_tmp)
            #rf2aa.util.torsion_indices, rf2aa.util.torsion_can_flip, rf2aa.util.reference_angles)
        # ... set torsion angles that are nan to 0 and mask them out
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        # ... reshape 
        alpha = alpha.reshape(-1,L,ChemData().NTOTALDOFS,2)  # [n, L, N_TOTAL_DOFS(10), 2]
        alpha_mask = alpha_mask.reshape(-1,L,ChemData().NTOTALDOFS,1)  # [n, L, N_TOTAL_DOFS(10), 1]
        # ... concatenate `alpha` and `alpha_mask` along last dimension
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 3*ChemData().NTOTALDOFS)  # [n, L, 3*N_TOTAL_DOFS(30)]
        # ... add template dimension
        alpha_t = alpha_t.unsqueeze(1) # [n,T,L,3*N_TOTAL_DOFS(30)]
        alpha_t = alpha_t.tile((1,2,1,1))

        ### bond_feats (distance matrix) ###
        ####################################
        # ... get bond distance matrix
        dist_matrix = rf2aa.data.data_loader.get_bond_distances(indep.bond_feats)


        ######################################
        ## Create additional input features ##
        ######################################
        if self.conf.preprocess.d_t1d == 24: # add hotpot residues
            raise NotImplementedError

        # Uncomment to see categorical extra_t1d_v2
        # ic(
        #     indep.extra_t1d.shape,
        #     indep.extra_t1d[:,:10].argmax(dim=-1),
        #     indep.extra_t1d[:,10:].argmax(dim=-1),
        # )

        ### extra_tXd ###
        #################
        t1d = torch.cat((t1d, indep.extra_t1d[None, None, ...]), dim=-1)  # [B, T, L, t1d+extra_t1d]
        # ... add t1d for the self-conditioning template (template_idx=1)
        t1d = torch.tile(t1d, (1,2,1,1))  # Repeat t1d accross the template dimension  # [B(1), T(2), L, t1d+extra_t1d]
        # X_t template is at index 0, Self-conditioning template is at index 1 (if self conditioning is active)
        t1d[0,1,:,ChemData().NAATOKENS-1] = -1 # This distiniguishes the templates to the model.
        # ic(t1d[0,:,:4,NAATOKENS-1]) # Will look like [[conf, -1], [conf, -1], ...], 0 < conf < 1
        
        # return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        mask_t = torch.ones(1,2,L,L).bool()  # [B, T, L, L]
        sctors = torch.zeros((1,L,ChemData().NTOTALDOFS,2))

        xyz = torch.squeeze(xyz_t, dim=0)

        # NO SELF COND
        xyz_t = torch.zeros(1,2,L,3)  # < Q(Woody): This overwrites `xyz_t` from above (which is now in `xyz` somewhat confusingly)
        t2d = torch.zeros(1,2,L,L,68)

        use_cb = self.conf.preprocess.use_cb_to_get_pair_dist

        if not self.conf.preprocess.get('omit_atom_frame_permutation', False): 
            atom_frames_in = indep.atom_frames # without omit permutation
        else: 
            atom_frames_in = indep.get_atom_frames(True) # with omit permutation

        t2d_xt, mask_t_2d_remade = util.get_t2d(
            xyz, indep.is_sm, atom_frames_in, use_cb=use_cb)

        t2d[0,0] = t2d_xt[0]
        xyz_t[0,0] = xyz[0,:,1]

        # ... concatenate `extra_t2d` channels [B(1),T(2),L,L,t2d_pre] -> [B,T,L,L,t2d+extra_t2d]
        t2d = torch.cat((t2d, torch.tile(indep.extra_t2d[None, None, ...], (1,2,1,1,1))), dim=-1)  # [B(1), T(2), L, L, t2d+extra_t2d]


        # Perform `nan_to_num` style replacement for the hydrogens (which follow the heavy atoms here)
        #ic(xyz.shape)
        # ic(
        #     xyz[0, is_diffused][0][:,0], # nan 3:
        #     xyz[0, indep.is_sm][0][:,0], # nan 14:
        #     xyz[0, ~is_diffused * ~indep.is_sm][0][:,0], # nan 14:
        # )

        is_protein_motif = ~is_diffused * ~indep.is_sm * nucl_utils.get_resi_type_mask(indep.seq, 'prot_and_mask')
        is_nucleic_motif = ~is_diffused * ~indep.is_sm * nucl_utils.get_resi_type_mask(indep.seq, 'na')
        # idx_diffused = torch.nonzero(is_diffused)
        # idx_protein_motif  = torch.nonzero(is_protein_motif)
        # idx_sm = torch.nonzero(indep.is_sm)

        # ic(
        #     idx_diffused,
        #     idx_protein_motif,
        #     idx_sm
        # )

        # xyz = torch.nan_to_num(xyz)
        # ... perform `nan_to_num` style replacement for the hydrogens (which follow the heavy atoms here)
        xyz[0, is_diffused*~indep.is_sm,3:] = torch.nan
        xyz[0, indep.is_sm,ChemData().NHEAVYPROT:] = 0
        xyz[0, is_protein_motif, ChemData().NHEAVYPROT:] = 0
        xyz[0, is_nucleic_motif, ChemData().NHEAVY:] = 0

        # Note: should be batched
        rfi = RFI(
            # ... msa related features
            msa_latent = msa_masked,  # [B(1), R(1), L, 164]
            msa_full = msa_full,  # [B(1), R(1), L, 83]
            seq = indep.seq[None],  # [B(1), L]
            seq_unmasked = indep.seq[None],  # [B(1), L]
            # ... structure related features
            xyz =xyz,  # [B(1), L, 36, 3]
            sctors = sctors,  # [1, L, 20, 2]
            idx = indep.idx[None], # [L]
            bond_feats = indep.bond_feats[None],  # [B(1), L, L]
            dist_matrix = dist_matrix[None],  # [B(1), L, L]
            chirals = indep.chirals[None],  # [n_chirals, 5]
            atom_frames = indep.atom_frames[None],  # [n_frames, 3, 2]
            # ... template related features 
            t1d = t1d,  # [B(1), T(2), L, x1d] (x1d = 114 for example)
            t2d = t2d,  # [B(1), T(2), L, L, x2d] (0 = Xt, 1 = self-conditioning, x2d = 68 for example)
            xyz_t = xyz_t,  # [B(1), T(2), L, 3]  (0 = Xt, 1 = self-conditioning)
            alpha_t = alpha_t,  # [B(1), T(2), L, 60]  (0 = Xt, 1 = self-conditioning)
            mask_t = mask_t,  # [B(1), T(2), L, L]
            # ... other features
            same_chain = indep.same_chain[None],  # [B(1), L, L]
            is_motif = ~is_diffused,  # [L] # TODO:(smathis) Swap to `is_motif` annotation
            msa_prev = None,
            pair_prev = None,
            state_prev = None)
        return rfi
    

def assert_has_coords(xyz, indep):
    assert len(xyz.shape) == 3
    missing_backbone = torch.isnan(xyz).any(dim=-1)[...,:3].any(dim=-1)
    prot_missing_bb = missing_backbone[~indep.is_sm]
    sm_missing_ca = torch.isnan(xyz).any(dim=-1)[...,1]
    try:
        assert not prot_missing_bb.any(), f'prot_missing_bb {prot_missing_bb}'
        assert not sm_missing_ca.any(), f'sm_missing_ca {sm_missing_ca}'
    except Exception as e:
        print(e)


def has_coords(xyz, indep):
    assert len(xyz.shape) == 3
    missing_backbone = torch.isnan(xyz).any(dim=-1)[...,:3].any(dim=-1)
    prot_missing_bb = missing_backbone[~indep.is_sm]
    sm_missing_ca = torch.isnan(xyz).any(dim=-1)[...,1]
    try:
        assert not prot_missing_bb.any(), f'prot_missing_bb {prot_missing_bb}'
        assert not sm_missing_ca.any(), f'sm_missing_ca {sm_missing_ca}'
    except Exception as e:
        print(e)


def pad_dim(x, dim, new_l, value=0):
    padding = [0]*2*x.ndim
    padding[2*dim] = new_l - x.shape[dim]
    padding = padding[::-1]
    return F.pad(x, pad=tuple(padding), value=value)

def write_traj(path, xyz_stack, seq, bond_feats, **kwargs):
    '''
    Write a trajectory to pdb

    Args:
        path (str or None): The place to write (or None if you don't want to write)
        xyz_stack (torch.Tensor[float]): The xyz of the indep to write [B,L,?,3]
        seq (torch.Tensor[float]): The sequence of the indep to write [L]
        bond_feats (torch.Tensor[float,float] or None): The bond_feats of the indep [L,L]
        **kwargs: Additional args for write_file.write_pdb_file

    Returns:
        pdb_stream (list[str]): The contents of the pdb file
    '''
    xyz23 = pad_dim(xyz_stack, 2, ChemData().NHEAVY)
    if bond_feats is not None:
        bond_feats = bond_feats[None]
    fh = io.StringIO()
    for i, xyz in enumerate(xyz23):
        write_file.writepdb_file(fh, xyz, seq, bond_feats=bond_feats, modelnum=i, **kwargs)

    if path is not None:
        with open(path, 'w') as f:
            f.write(fh.getvalue())

    fh.seek(0)
    return fh.readlines()

def minifier(argument_map):
    argument_map['out_9'] = None
    argument_map['out_0'] = None
    argument_map['out_2'] = None
    argument_map['out_3'] = None
    argument_map['out_5'] = None
    argument_map['t2d'] = None

TYPE_PROT = 0
"""Integer used to indicate a protein residue"""

TYPE_LIGAND = 1
"""Integer used to indicate a small molecule ligand"""

TYPE_ATOMIZED_COV = 2
"""Integer used to indicate an atomized covalent bond"""


@dataclass
class Bond:
    a: str
    b: str
    order: int
    aromatic: bool


@dataclass
class Atom:
    element: int

def get_obmol(xyz_sm, seq_sm, bond_feats_sm):
    atomnumbyatomtype = {v:k for k,v in ChemData().atomnum2atomtype.items()}
    akeys = []
    atoms = []
    bonds = []
    for i, (_, seq) in enumerate(zip(xyz_sm, seq_sm)):
        atom_name = ChemData().num2aa[seq]
        atomnum = atomnumbyatomtype[atom_name]
        # xyz.append(xyz_i)
        atoms.append(Atom(atomnum))
        akeys.append(i)

    bonds.extend(
        Bond(i, j, bond_feats_sm[i, j].item(), False)
        for i, j in bond_feats_sm.nonzero()
        if i <= j
    )
    mol, bond_feats = rf2aa.util.cif_ligand_to_obmol(xyz_sm, akeys, atoms, bonds)
    return mol, bond_feats

def adaptor_fix_bb_indep(out):
    """
    Adapts the outputs of RF2-allatom phase 3 dataloaders into fixed bb outputs

    Paramters:
        out: RF2-allatom phase 3 dataloader outputs
    Returns:
        indep: Indep
        atom_mask: torch.Tensor
    """
    assert len(out) == 24, f"found {len(out)} elements in RF2-allatom output"
    (seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev,
        mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, dist_matrix, chirals, ch_label, symm_group,
         dataset_name, item) = out
    true_crds = torch.nan_to_num(true_crds)
    assert symm_group=="C1", f"example with {symm_group} found, symmetric training not set up for aa-diffusion"
    #remove permutation symmetry dimension if present
    if len(true_crds.shape) == 4 and len(atom_mask.shape) == 3:
        true_crds = true_crds[0]
        atom_mask = atom_mask[0]
    
    # our dataloaders return torch.zeros(L...) for atom frames and chirals when there are none, this updates it to use common shape 
    if torch.all(atom_frames == 0):
        atom_frames = torch.zeros((0,3,2)).long()
    if torch.all(chirals == 0):
        chirals = torch.zeros((0,5))

    MSAFULL_N_TERM = ChemData().NAATOKENS+NINDEL
    MSAFULL_C_TERM = MSAFULL_N_TERM+1
    is_n_terminus = msa_full[0, 0, :, MSAFULL_N_TERM].bool()
    is_c_terminus = msa_full[0, 0, :, MSAFULL_C_TERM].bool()
    terminus_type = torch.zeros(msa_masked.shape[2], dtype=int)
    terminus_type[is_n_terminus] = N_TERMINUS
    terminus_type[is_c_terminus] = C_TERMINUS

    indep = Indep(
        seq=rf2aa.tensor_util.assert_squeeze(seq), # [L]
        xyz=true_crds[:,:ChemData().NHEAVY], # [L, N_HEAVY, 3]  # N_HEAVY = 23
        idx=idx_pdb,
        # SM specific
        bond_feats=bond_feats,
        chirals=chirals,
        same_chain=same_chain.bool(),
        terminus_type=terminus_type,
    )
    return indep, atom_mask

def deatomize_covales(indep, atom_mask):
    """
    Removes atomized sidechains created in the structure prediction dataloader and
    parses the small-molecule:residue-atom bonds out into the metadata dictionary.

    Paramters:
        indep: Indep
        atom_mask: torch.Tensor
    Returns:
        indep: Indep
        atom_mask: torch.Tensor
        metadata:
            {
                covale_bonds: [((res_idx0, atom_name), lig_idx0, bond_type),...]
            }
    """
    # Clear out peptide bonds from covale atomizations
    indep.assert_types()
    is_peptide_bond = indep.bond_feats == 6
    indep.bond_feats = indep.bond_feats * ~is_peptide_bond

    metadata = {'type': indep.type()}
    assertpy.assert_that(metadata['type']).does_not_contain(-1)

    ca = indep.xyz[:,1]
    L = indep.length()
    ca_dist = torch.cdist(ca[None,...], ca[None,...], p=2.0)[0]
    is_res_to_atomized_ca = (ca_dist < 1e-4) * \
        (metadata['type'] == TYPE_PROT)[: None] * \
        (metadata['type'] == TYPE_ATOMIZED_COV)[None, :]
    is_ca_close = ca_dist < 1e-4

    is_res_to_atomized_ca =  ((metadata['type'] == TYPE_PROT)[:,None]) * ((metadata['type'] == TYPE_ATOMIZED_COV)[None, :])
    is_ca_close.fill_diagonal_(0)

    metadata['covale_correspondence'] = {}
    is_res_to_atomized_ca_correspondence = is_res_to_atomized_ca * is_ca_close
    for res_idx0, atomized_ca_idx0 in is_res_to_atomized_ca_correspondence.nonzero():
        original_aa = indep.seq[res_idx0]
        atom_names = ChemData().aa2long[original_aa]
        a = indep.xyz[res_idx0][None,...]
        b = indep.xyz[(metadata['type'] == TYPE_ATOMIZED_COV), 1][None,...]
        dist = torch.cdist(a,b)
        dist = dist[0]
        covale_idx0_by_local = torch.arange(L)[(metadata['type'] == TYPE_ATOMIZED_COV)]
        corresponding_atom_names = []
        corresponding_idx0 = []
        for res_local, covale_local in (dist < 1e-1).nonzero():
            corresponding_atom_names.append(atom_names[res_local])
            corresponding_idx0.append(covale_idx0_by_local[covale_local])
        corresponding_idx0 = np.array(corresponding_idx0)

        G = nx.from_numpy_matrix(indep.bond_feats.detach().cpu().numpy())
        connected_idx0 = fetch_connected_nodes(G, corresponding_idx0[0])
        # for idx in nx.connected_components(G):
        metadata['covale_correspondence'][res_idx0.item()] = {
            'atom_names': corresponding_atom_names,
            'idx0': corresponding_idx0,
            'connected_idx0': np.array(list(connected_idx0))
        }

    # Detect cross bonds
    # i.e. bond features between atomized and covalent ligand
    resi_atom_name_by_atomized_idx = {}
    for res_idx0, d in metadata['covale_correspondence'].items():
        for atom_name, atomized_idx0 in zip(d['atom_names'], d['idx0']):
            resi_atom_name_by_atomized_idx[atomized_idx0] = (res_idx0, atom_name)

    atom_identifiers = [] # TODO
    for i in range(indep.length()):
        if i in resi_atom_name_by_atomized_idx:
            atom_identifiers.append(resi_atom_name_by_atomized_idx[i])
            continue
        atom_identifiers.append(i)

    # HACK: assumes only one covalent ligand
    all_corresponding = [np.array([])]
    for v in metadata['covale_correspondence'].values():
        all_corresponding.append(v['idx0'])
    all_corresponding = np.concatenate(all_corresponding)
    is_corresponding = torch.zeros(indep.length()).bool()
    is_corresponding[all_corresponding] = True
    is_covale_sm = (metadata['type'] == TYPE_ATOMIZED_COV) * ~is_corresponding
    is_covale_bond = is_corresponding[...,None] * is_covale_sm[None, ...]
    covale_bonds = indep.bond_feats * is_covale_bond
    bonds = []
    for atomized_i, covale_i in covale_bonds.nonzero():
        bond_type = covale_bonds[atomized_i, covale_i]
        bonds.append(
            (atom_identifiers[atomized_i], atom_identifiers[covale_i], bond_type)
        )
    metadata['covale_bonds'] = bonds

    pop_mask(indep, ~is_corresponding, break_chirals=True)
    atom_mask = atom_mask[~is_corresponding]

    new_i_from_old_i = (~is_corresponding).cumsum(dim=0) - 1
    for i, (a, b, bond_type) in enumerate(metadata['covale_bonds']):
        new_b = new_i_from_old_i[b].item()
        metadata['covale_bonds'][i] = (a, new_b, bond_type)

    metadata = {
        'covale_bonds': metadata['covale_bonds'],
    }

    return indep, atom_mask, metadata

def missing_atom_names(indep, atom_mask, res_i):
    seq = indep.seq[res_i]
    all_atom_mask = ChemData().allatom_mask[seq]
    all_atom_names = np.array(ChemData().aa2long[seq][:ChemData().NHEAVYPROT], dtype=np.str_)
    all_atom_names = all_atom_names[all_atom_mask[:ChemData().NHEAVYPROT]]
    have_atom_mask = atom_mask[res_i]
    have_atom_names = np.array(ChemData().aa2long[seq][:ChemData().NHEAVYPROT], dtype=np.str_)
    have_atom_names = have_atom_names[have_atom_mask[:ChemData().NHEAVYPROT]]
    return [a for a in all_atom_names if a not in have_atom_names]

def fetch_connected_nodes(G, node, seen = None):
    if seen is None:
        seen = {node}
    for neighbor in G.neighbors(node):
        if neighbor not in seen:
            seen.add(neighbor)
            fetch_connected_nodes(G, neighbor, seen)
    return seen

def is_occupied(indep, atom_mask):
    """
    Returns a boolean mask which is:
        False for ligand atoms which are not present in the atom mask.
        False for residues which do not have N,C,Ca in the atom mask.
    """
    pop = rf2aa.util.get_prot_sm_mask(atom_mask, indep.seq)
    return pop

def reindex_dict(d, pop: torch.tensor):
    '''
    Reindexes a dictionary keyed on index after popping a mask.

    Params:
        d: Dictionary of {index: ....}
        pop: binary mask
    Returns:
        Dictionary with keys not present in pop removed and the remaining reindexed.
    '''
    new_indices = (pop.cumsum(dim=0) - 1).tolist()
    def shift(k):
        return new_indices[k]
    return {shift(k): v for k, v in d.items() if pop[k]}


def reindex_covales(covales, pop: torch.tensor):
    '''
    Reindex covalent bonds after popping a mask.

    Params:
        covales: list of tuples: [((index, atom_name), small_molecule_index, bond_type), ...]
        pop: binary mask
    Returns:
        list of tuples: [((index, atom_name), small_molecule_index, bond_type), ...]
    '''
    for (i, atom_name), sm_i, _ in covales:
        assert pop[i]
        assert pop[sm_i]
    new_indices = (pop.cumsum(dim=0) - 1).tolist()
    def shift(k):
        return new_indices[k]
    return [((shift(i), atom_name), shift(sm_i), bond_type) for (i, atom_name), sm_i, bond_type in covales]

def pop_mask(indep: Indep, pop: torch.Tensor, break_chirals: bool = False) -> None:
    n_atoms = indep.is_sm.sum()
    assertpy.assert_that(len(indep.atom_frames)).is_equal_to(n_atoms)

    # ASSERT REFERENCES CHECK OUT

    N     = pop.sum()
    pop2d = pop[None,:] * pop[:,None]

    indep.seq           = indep.seq[pop]
    indep.xyz           = indep.xyz[pop]
    indep.idx           = indep.idx[pop]
    indep.bond_feats    = indep.bond_feats[pop2d].reshape(N,N)
    indep.same_chain    = indep.same_chain[pop2d].reshape(N,N)
    indep.terminus_type = indep.terminus_type[pop]
    indep.is_gp         = indep.is_gp[pop]

    pop_i = pop.nonzero()
    is_chiral_popped = torch.isin(indep.chirals[:,:-1].type(torch.DoubleTensor), pop_i)
    
    # assertpy.assert_that(cmp_pretty(any_chi))
    any_is_chiral_popped = torch.any(is_chiral_popped, dim=1)
    all_is_chiral_popped = torch.all(is_chiral_popped, dim=1)
    if not break_chirals:
        assertpy.assert_that((any_is_chiral_popped == all_is_chiral_popped).all()).is_true()
    indep.chirals = indep.chirals[all_is_chiral_popped]

    if indep.chirals.numel():
        n_shift = (~pop).cumsum(dim=0)
        chiral_indices = indep.chirals[:,:-1]
        chiral_shift = n_shift[chiral_indices.long()]
        indep.chirals[:,:-1] = chiral_indices - chiral_shift

def slice_indep(indep: Indep, pop: torch.Tensor, break_chirals: bool = False) -> tuple[Indep, torch.Tensor]:
    """Subset an indep by a `pop` mask, which is true for atoms to keep."""
    indep = copy.deepcopy(indep)
    cross_bonds = indep.bond_feats[pop][:, ~pop]
    # ic(cross_bonds)
    # assert_that(cross_bonds.sum()).is_equal_to(0)
    pop_mask(indep, pop, break_chirals=break_chirals)
    return indep, cross_bonds
 
def cat_indeps(indeps: list[Indep], same_chain: torch.Tensor) -> Indep:
    """Concatenate a list of indeps, assuming no inter-indep bonds."""
    indep = Indep(
        seq=None,
        xyz=None,
        idx=None,
        bond_feats=None,
        chirals=None,
        same_chain=None,
        terminus_type=None,
    )
    indep.seq = torch.cat([i.seq for i in indeps])
    indep.xyz = torch.cat([i.xyz for i in indeps])
    indep.idx = torch.cat([i.idx for i in indeps])
    # ... assume no inter-indep bonds (i.e. bond_feats is block diagonal)
    indep.bond_feats = torch.block_diag(*(i.bond_feats for i in indeps))
    indep.same_chain = same_chain
    indep.terminus_type = torch.cat([i.terminus_type for i in indeps])
    indep.is_gp = torch.cat([i.is_gp for i in indeps])
    assert len(indep.is_gp) == indep.length()

    L = 0
    all_chirals = []
    for i in indeps:
        chirals = i.chirals.clone()
        chirals[:,:-1] += L
        all_chirals.append(chirals.clone())
        L += i.length()
    indep.chirals = torch.cat(all_chirals)
    return indep

def cat_indeps_same_chain(indeps: list[Indep]) -> Indep:
    """Concatenate a list of indeps, putting all indeps in the same chain."""
    L = sum(i.length() for i in indeps)
    same_chain = torch.ones((L,L)).bool()
    return cat_indeps(indeps, same_chain)

def cat_indeps_separate_chains(indeps: list[Indep]) -> Indep:
    """Concatenate a list of indeps, putting each indep in its own chain."""
    same_chain = torch.block_diag(*(i.same_chain for i in indeps))
    return cat_indeps(indeps, same_chain)


def rearrange_indep(indep, from_i):
    # from_i = torch.tensor(from_i)
    assert_that(sorted(from_i.tolist())).is_equal_to(list(range(indep.length())))
    to_i = torch.argsort(from_i)
    indep.seq = indep.seq[from_i]
    indep.xyz = indep.xyz[from_i]
    indep.idx = indep.idx[from_i]
    indep.bond_feats = indep.bond_feats[from_i,:][:, from_i]
    indep.same_chain = indep.same_chain[from_i, :][:, from_i]
    indep.terminus_type = indep.terminus_type[from_i]
    indep.chirals[:,:-1] = indep.chirals[:,:-1].type(torch.LongTensor).apply_(lambda i: to_i[i])
    indep.is_gp = indep.is_gp[from_i]
    is_sm_new = indep.is_sm[from_i]
    is_sm_old = indep.is_sm
    n_sm = is_sm_old.sum()
    sm_i_relative = torch.arange(n_sm)
    a = torch.zeros(indep.length()).type(torch.LongTensor)
    a[:] = 9999
    a[is_sm_old] = sm_i_relative
    relative_from_absolute_new = torch.zeros(indep.length()).type(torch.LongTensor)
    a[:] = 9999
    relative_from_absolute_new[is_sm_new] = torch.arange(n_sm)
    


def diffuse(conf, diffuser, indep, is_diffused, t):
    """
    conf.diffuser.preserve_motif_sidechains, when set to true, diffusion and frames addition are only performed on positions that are set to diffuse via the is_diffused boolean mask 
    """
    
    indep = copy.deepcopy(indep)
    
    indep.xyz = add_fake_frame_legs(indep.xyz, indep.is_sm)
    rigids_0 = du.rigid_frames_from_atom_14(indep.xyz)  # [L, 23, 3] # not 14?
    diffuser_out = diffuser.forward_marginal(
        rigids_0,
        t=t/conf.diffuser.T,
        diffuse_mask=is_diffused.float(),
        as_tensor_7=False
    )
    diffuser_out['rigids_0'] = rigids_0.to_tensor_7()[None]
    diffuser_out['rigids_0_raw'] = rigids_0
    xT = all_atom.atom37_from_rigid(diffuser_out['rigids_t'])
    # Only update heavy atoms from diffused positions
    if conf.diffuser.preserve_motif_sidechains:
        # Only 'is_diffused' motifs get replaced, should be more correct in general
        # Training datasets are returning 23 atoms, so this line breaks.
        # TODO: unify this such that we are always in either atom36 or atom23
        # indep.xyz[is_diffused,:ChemData().NTOTAL] = xT[is_diffused,:ChemData().NTOTAL]
        indep_n_atoms = indep.xyz.shape[1]
        assert indep_n_atoms in [23, 36, 37]
        indep.xyz[is_diffused,:indep_n_atoms] = xT[is_diffused,:indep_n_atoms]
        indep.xyz = indep.xyz[:, :ChemData().NTOTAL]
    else:
        # In this case, all motif atoms are replaced with idealized frames from rigids
        indep.xyz = xT[:,:ChemData().NTOTAL] # WARNING: this goes from (AF's)atom37 to (our) 36. Looks like not a problem for now since we only use N CA C CB?
    return indep, diffuser_out


def mask_seq(indep: Indep, is_seq_masked: torch.Tensor) -> torch.Tensor:
    """
    Diffuses the sequence by applying masks to each polymer type.

    Args:
        indep (torch.Tensor): The input sequence tensor.
        is_seq_masked (torch.Tensor): The mask indicating which positions in the sequence are masked.

    Returns:
        indep with sequence diffused
    """
    indep = indep.clone()
    mask_indep(indep, is_seq_masked)
    return indep



def add_fake_peptide_frame(indep, generator=None):
    indep = copy.deepcopy(indep)
    indep.xyz = add_fake_frame_legs(indep.xyz, indep.is_sm, generator=generator)
    return idealize_peptide_frames(indep, generator=generator)

def idealize_peptide_frames(indep, generator=None):
    indep = copy.deepcopy(indep)
    rigids = du.rigid_frames_from_atom_14(indep.xyz)
    atom37 = all_atom.atom37_from_rigid(rigids, generator=generator)
    # Not sure if this clone is necessary
    atom37 = torch.clone(atom37)
    # Keeping these comments here in case this breaks tests.
    # This is incorrect, do not return only :14, now pad to 36 full size 
    # orig_dim = indep.xyz.shape[1]
    # if orig_dim > ChemData().NHEAVY:
    #     indep.xyz = torch.cat([atom37[:,:ChemData().NHEAVY], 
    #                            torch.zeros(indep.xyz.shape[0], orig_dim - ChemData().NHEAVY, 3)], 
    #                            dim=1)
    # else:
    indep.xyz = atom37[:,:ChemData().NTOTAL]
    return indep

def diffuse_then_add_conditional(conf, diffuser, indep, is_diffused, t):
    """
    Args:
        diffuser (Diffuser): Diffuser model, can be None, in which case diffusion is not performed
    """
    if diffuser is None:
        return indep, indep

    # Make the unconditional indep    
    indep_uncond, diffuser_out = diffuse(conf, diffuser, indep, torch.ones_like(is_diffused).bool(), t)

    # In cases with large ~is_diffused. The is_diffused cloud can be rather off-origin
    if conf.diffuser.independently_center_diffuseds and t == conf.diffuser.T:
        if is_diffused.sum() > 0:
            indep_uncond.xyz[is_diffused] -= indep_uncond.xyz[is_diffused,1].mean(axis=0)[None,None,:]
        if (~is_diffused).sum() > 0:
            indep_uncond.xyz[~is_diffused] -= indep_uncond.xyz[~is_diffused,1].mean(axis=0)[None,None,:]


    # Make the conditional portion
    indep = copy.deepcopy(indep)
    indep = add_fake_peptide_frame(indep)
    indep_cond = copy.deepcopy(indep_uncond)
    indep_cond.xyz[~is_diffused] = indep.xyz[~is_diffused]

    return indep_uncond, indep_cond


def forward(model, rfi, **kwargs):
    rfi_dict = dataclasses.asdict(rfi)
    return RFO(*model(**{**rfi_dict, **kwargs}))

def mask_indep(indep, is_diffused):
    fully_masked_seq = nucl_utils.get_full_mask_seq(indep.seq)
    indep.seq[is_diffused] = fully_masked_seq[is_diffused]

def self_cond_new(indep, rfi, rfo, use_cb=False):
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
    xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
    t2d, mask_t_2d_remade = util.get_t2d(
        xyz_t[0], indep.is_sm, rfi.atom_frames[0], use_cb=use_cb)
    base_d_t2d = t2d.shape[-1]
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    rfi_sc.xyz_t[0,1] = xyz_t[0, 0, :, 1]
    rfi_sc.t2d[0, 1,:,:,:base_d_t2d] = t2d[0, 0]
    return rfi_sc

def self_cond(indep, rfi, rfo, use_cb=False):
    xyz_last = rfo.xyz[-1:]
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
    xyz_t = torch.cat((xyz_last, zeros), dim=-2) # [B,T,L,27,3]
    t2d, mask_t_2d_remade = util.get_t2d(
        xyz_t[0], indep.is_sm, rfi.atom_frames[0], use_cb=use_cb)
    base_d_t2d = t2d.shape[-1]
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    rfi_sc.xyz_t[0,1] = xyz_t[0, 0, :, 1]
    rfi_sc.t2d[0, 1,:,:,:base_d_t2d] = t2d[0, 0]
    return rfi_sc

def diagnose_xyz(xyz):
    '''Returns a string describing where the coordinates are NaN'''
    has_ca = torch.isnan(xyz[..., 1, :]).any()
    has_backbone = torch.isnan(xyz[..., :3, :]).any()
    # has_heavy = torch.isnan(xyz[..., :3, :]).any()
    return f'diagnosis: nan-CA: {has_ca}    nan-BB: {has_backbone}'

def get_atomization_state(indep):
    return [
        AtomizedLabel(
            coarse_idx0=i,
            aa=int(indep.seq[i]),
            atom_name=None,
            pdb_idx=int(indep.idx[i]),
            terminus_type=int(indep.terminus_type[i]),
        )
        for i in range(indep.length())
    ]

def insert_tensor(body: torch.tensor, fill, index: int, dim: int=0) -> torch.tensor:
    """
    Inserts a tensor or a single value 'fill' into tensor 'body' at the specified 'index' 
    along dimension 'dim', after checking shape compatibility.

    Args:
    body (torch.tensor): The original tensor.
    fill: The tensor or value to be inserted.
    index (int): The index at which to insert 'fill'.
    dim (int): The dimension along which to insert.

    Returns:
    torch.tensor: The resulting tensor after insertion.

    Raises:
    ValueError: If the shapes of 'body' and 'fill' are not compatible for insertion.
    """
    # Coerce fill into the right shape
    fill_shape = list(body.shape)
    fill_shape[dim] = 1
    
    # If fill is not a tensor, create a tensor filled with the value
    if not torch.is_tensor(fill):
        fill = torch.full(fill_shape, fill, dtype=body.dtype, device=body.device)

    if list(fill.shape) != fill_shape:
        fill = fill.unsqueeze(dim)
                
    for d in range(body.dim()):
        if (d != dim) and (body.shape[d] != fill.shape[d]):
            raise ValueError(
                f"Dimensions of 'body' and 'fill' must match in all dimensions except "
                f"dimension {dim}, but are {body.shape=} and {fill.shape=}."
            )

    # Split the original tensor into two parts
    body1 = body.narrow(dim, 0, index)
    body2 = body.narrow(dim, index, body.size(dim) - index)

    # Concatenate the three parts
    return torch.cat([body1, fill, body2], dim=dim)


class AtomizeResidues:
    def __init__(self, deatomized_state: list[AtomizedLabel], residue_to_atomize: torch.Tensor):
        assert len(deatomized_state) == len(residue_to_atomize)

        self.deatomized_state = deatomized_state
        self.residue_to_atomize = residue_to_atomize

    @property
    def atomized_state(self):
        idx0s = torch.where(self.residue_to_atomize)[0]
        return self._atomize_state(self.deatomized_state, idx0s)
 
    def atomize(self, indep_deatomized):
        # Length check
        assert indep_deatomized.length() == len(self.deatomized_state)
        
        indep_deatomized = indep_deatomized.clone()
        
        seq_atomize_all,  xyz_atomize_all, chirals_atomize_all, res_idx_atomize_all, gp_atomize_all = \
            self._generate_atomize_protein_features(indep_deatomized, self.residue_to_atomize)
        
        if not res_idx_atomize_all: # no residues atomized, just return the inputs
            indep_atomized = indep_deatomized
            return indep_atomized
        
        # update msa feats, xyz, mask, frames and chirals
        indep_atomized = self._update_rf_features_w_atomized_features(
            indep_deatomized,
            seq_atomize_all,
            xyz_atomize_all,
            chirals_atomize_all,
            gp_atomize_all
        ) 

        indep_atomized = self._pop_protein_feats(indep_atomized, res_idx_atomize_all)
        
        return indep_atomized
    
    def _deatomize_residue(self, indep_atomized, coarse_idx0: int, atomization_state):
        '''
        Deatomize a single residue
        
        Inputs
            indep_atomized: Indep containing an atomized residue
            coarse_idx0: The idx0 for the residue before it was atomized.
                Hint: You can look at self.atomization_state to find this.
                
        Returns
            indep_deatomized: Indep with the specified residue deatomized.
        '''
        # Length check
        assert indep_atomized.length() == len(atomization_state)

        indep_deatomized = indep_atomized.clone()
        deatomization_state, deatomized_labels, deatomized_idx0 = self._deatomize_state(atomization_state, coarse_idx0)        
        
        # Insert 1D info for the new residue
        ## Annoyingly, the "fill" value for nonexistent atoms is not zero. Need to infer it here.
        atom_mask = ChemData().allatom_mask[indep_deatomized.seq]
        n_atoms = indep_deatomized.xyz.shape[1]
        atom_mask = atom_mask[:, :n_atoms]  # clip to the number of atoms that are in xyz
        xyz_fill = indep_deatomized.xyz[~atom_mask]
        xyz_fill = xyz_fill[~torch.isnan(xyz_fill).any(-1)]
        if xyz_fill.numel() > 0:
            xyz_fill = xyz_fill[0]
        else:
            xyz_fill = torch.full((3,), torch.nan)

        xyz_deatomized = torch.full((n_atoms, 3), np.nan)
        xyz_deatomized[:ChemData().NHEAVYPROT] = xyz_fill
        xyz_deatomized[:len(deatomized_idx0)] = indep_deatomized.xyz[deatomized_idx0, 1]
        
        indep_deatomized.xyz = insert_tensor(
            body = indep_deatomized.xyz,
            fill = xyz_deatomized,
            index = coarse_idx0
        )
        indep_deatomized.seq = insert_tensor(
            body = indep_deatomized.seq, 
            fill = deatomized_labels[0].aa,
            index = coarse_idx0
        )
        indep_deatomized.idx = insert_tensor(
            body = indep_deatomized.idx,
            fill = deatomized_labels[0].pdb_idx,
            index = coarse_idx0
        )
        indep_deatomized.terminus_type = insert_tensor(
            body = indep_deatomized.terminus_type,
            fill = deatomized_labels[0].terminus_type,
            index = coarse_idx0
        )
        
        # Insert 2D info for the new residue
        indep_deatomized.bond_feats = insert_tensor(
            body = indep_deatomized.bond_feats,
            fill = 0.,
            index = coarse_idx0,
            dim = 0
        )
        indep_deatomized.bond_feats = insert_tensor(
            body = indep_deatomized.bond_feats,
            fill = 0.,
            index = coarse_idx0,
            dim = 1
        )
        indep_deatomized.is_gp = insert_tensor(
            body = indep_deatomized.is_gp,
            fill = indep_deatomized.is_gp[deatomized_idx0].any(), # if any atoms were gp, the full residue is gp. Right?
            index = coarse_idx0,
            dim = 0
        )
        assert len(indep_deatomized.is_gp) == indep_deatomized.length()

        # Connect new residue to neighbors it is bonded to
        ## Get chain numbers
        chain_signatures = indep_deatomized.same_chain.unique(dim=0).flip(0)
        chain_nums = chain_signatures.float().argmax(dim=0)
        
        chain_nums = insert_tensor(
            body = chain_nums,
            fill = -1,  # Provisionally treated as a new chain
            index = coarse_idx0,
        )
        
        ## Adjust by 1 because of newly inserted residue
        deatomized_idx0 = torch.tensor(deatomized_idx0)
        deatomized_idx0 += 1
        
        ## Resolve atom-residue bonds
        for i, j in zip(*torch.where(indep_deatomized.bond_feats==6)):
            if i in deatomized_idx0:
                indep_deatomized.bond_feats[coarse_idx0, j] = 5
                
            if j in deatomized_idx0:
                indep_deatomized.bond_feats[i, coarse_idx0] = 5
                
        ## Resolve any N-C or C-N bonds
        for i, j in zip(*torch.where(indep_deatomized.bond_feats==1)):
            if (i in deatomized_idx0):
                indep_deatomized.bond_feats[coarse_idx0, j] = 6
                
            if (j in deatomized_idx0):
                indep_deatomized.bond_feats[i, coarse_idx0] = 6
                  
        ## What chain number is the deatomized residue on?
        covalently_bonded = (indep_deatomized.bond_feats >= 1) & (indep_deatomized.bond_feats <= 6)
        G = nx.from_numpy_array(covalently_bonded.numpy())
        
        for nbunch in nx.connected_components(G):
            if coarse_idx0 in nbunch:
                nbunch.remove(coarse_idx0)
                is_ptn = rf2aa.util.is_protein(indep_deatomized.seq)
                is_connected = torch.zeros_like(is_ptn)
                is_connected[list(nbunch)] = True
                connected_chain_nums = chain_nums[is_ptn & is_connected]
                if connected_chain_nums.numel() > 0:
                    # Connect to an existing chain
                    chain_nums[coarse_idx0] = connected_chain_nums.unique()[0]
                else:
                    # Guarantees a new chain is formed
                    chain_nums[coarse_idx0] = -1
                    
        ## Make same_chain
        indep_deatomized.same_chain = chain_nums[:, None] == chain_nums[None, :]
                    
        # The inserted residue can cause the chiral indices to advance by one
        chiral_idxs_to_shift = indep_deatomized.chirals[:, :4] > coarse_idx0
        indep_deatomized.chirals[:, :4] += chiral_idxs_to_shift.float()
        
        # Remove leftover atoms that were deatomized
        atoms_to_remove = torch.zeros(indep_deatomized.length(), dtype=bool)
        atoms_to_remove[deatomized_idx0] = True
        pop_mask(indep_deatomized, ~atoms_to_remove, break_chirals=True)
        
        return indep_deatomized, deatomization_state
    
    def deatomize(self, indep_atomized):
        '''
        Deatomize all atomized residues in an Indep
        
        Inputs
            indep_atomized: Indep containing atomized residue(s)

        Returns
            indep_deatomized: Indep with all residue(s) deatomized.
        '''
        # Length check
        assert indep_atomized.length() == len(self.atomized_state), f'{indep_atomized.length()=}, {len(self.atomized_state)=}'

        indep_deatomized = indep_atomized.clone()
        atomization_state = self.atomized_state
        for coarse_idx0 in torch.where(self.residue_to_atomize)[0].tolist():
            indep_deatomized, atomization_state = self._deatomize_residue(indep_deatomized, coarse_idx0, atomization_state)

        return indep_deatomized
        
    def _generate_atomize_protein_features(self, indep_deatomized, residue_to_atomize):        
        L = indep_deatomized.length()
        is_protein_motif = residue_to_atomize * ~indep_deatomized.is_sm
        allatom_mask = ChemData().allatom_mask.clone() # 1 if that atom index exists for that residue and 0 if it doesnt
        allatom_mask[:, ChemData().NHEAVY:] = False # no Hs
        atom_mask = allatom_mask[indep_deatomized.seq]

        # make xyz the right dimensions for atomization
        xyz = torch.full((L, ChemData().NTOTAL, 3), np.nan).float()
        xyz[:, :ChemData().NHEAVY] = indep_deatomized.xyz[:, :ChemData().NHEAVY]

        # iterate through residue indices in the structure to atomize
        seq_atomize_all = []
        xyz_atomize_all = []
        chirals_atomize_all = [indep_deatomized.chirals]
        res_idx_atomize_all = [] # indices of residues that end up getting atomized (0-indexed, contiguous)
        res_atomize_all = [] # residues (integer-coded) that end up getting atomized 
        idx_atomize_all = [] # PDB indices of residues that end up getting atomized
        gp_atomize_all = [] # is_gp for each new atom
        
        # track the res_idx and absolute index of the previous C to draw peptide bonds between contiguous residues
        prev_res_idx = -2
        prev_C_index = -2
        for res_idx in is_protein_motif.nonzero():
            residue = indep_deatomized.seq[res_idx] # number representing the residue in the token list, can be used to index aa2long
            # check if all the atoms in the residue are resolved, if not dont atomize
            if not torch.all(allatom_mask[residue]==atom_mask[res_idx]):
                raise Exception(f'not all atoms resolved for {residue} at 0-indexed position {res_idx}')
            N_term = res_idx == 0
            C_term = res_idx == indep_deatomized.idx.shape[0]-1
            N_resolved = False
            C_resolved = False

            C_resolved = (not C_term) and indep_deatomized.idx[res_idx+1]-indep_deatomized.idx[res_idx] == 1
            N_resolved = (not N_term) and indep_deatomized.idx[res_idx]-indep_deatomized.idx[res_idx-1] == 1

            seq_atomize, _, xyz_atomize, _, frames_atomize, bond_feats_atomize, ra, chirals_atomize = \
                nucl_utils.atomize_prot_na(res_idx, indep_deatomized.seq[None], xyz, atom_mask, n_res_atomize=1)
            # rf2aa.util.atomize_protein(res_idx, indep_deatomized.seq[None], xyz, atom_mask, n_res_atomize=1)
            r,a = ra.T
            C_index = torch.all(ra==torch.tensor([r[-1],2]),dim=1).nonzero()
            assert len(C_index) == 1
            C_index = int(C_index[0])
            
            natoms = seq_atomize.shape[0]
            # update the chirals to be after all the other residues
            chirals_atomize[:, :-1] += L

            seq_atomize_all.append(seq_atomize)
            xyz_atomize_all.append(xyz_atomize)
            chirals_atomize_all.append(chirals_atomize)
            res_idx_atomize_all.append(res_idx)
            res_atomize_all.append(residue)
            idx_atomize_all.append(indep_deatomized.idx[res_idx])
            gp_atomize_all.append(torch.full((natoms,), bool(indep_deatomized.is_gp[res_idx]), dtype=bool))
            
            # update bond_feats every iteration, update all other features at the end 
            bond_feats_new = torch.zeros((L+natoms, L+natoms))
            bond_feats_new[:L, :L] = indep_deatomized.bond_feats
            bond_feats_new[L:, L:] = bond_feats_atomize

            # add bond between protein and atomized N
            if N_resolved:
                bond_feats_new[res_idx-1, L] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L, res_idx-1] = 6 # protein (backbone)-atom bond 
            # add bond between protein and C, assumes every residue is being atomized one at a time (eg n_res_atomize=1)
            if C_resolved:
                bond_feats_new[res_idx+1, L+C_index] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L+C_index, res_idx+1] = 6 # protein (backbone)-atom bond 
            # handle drawing peptide bond between contiguous atomized residues
            res_is_na = nucl_utils.get_resi_type_mask(residue, 'na')  # Do not include if it is a nucleic acid
            if indep_deatomized.idx[res_idx]-indep_deatomized.idx[prev_res_idx] == 1 and prev_res_idx > -1 and not res_is_na:
                bond_feats_new[prev_C_index, L] = 1 # single bond
                bond_feats_new[L, prev_C_index] = 1 # single bond
            prev_res_idx = res_idx
            prev_C_index =  L+C_index
            #update same_chain every iteration
            same_chain_new = torch.zeros((L+natoms, L+natoms))
            same_chain_new[:L, :L] = indep_deatomized.same_chain
            residues_in_prot_chain = indep_deatomized.same_chain[res_idx].squeeze().nonzero()

            same_chain_new[L:, residues_in_prot_chain] = 1
            same_chain_new[residues_in_prot_chain, L:] = 1
            same_chain_new[L:, L:] = 1

            indep_deatomized.bond_feats = bond_feats_new
            indep_deatomized.same_chain = same_chain_new
            L = indep_deatomized.bond_feats.shape[0]
        
        return seq_atomize_all, xyz_atomize_all, chirals_atomize_all, res_idx_atomize_all, gp_atomize_all

    def _get_atomized_residue_info(self):
        '''
        Get the coarse_idx0, sequence and pdb_idx of residues that are already atomized.
        '''
        atomized_residues = {x.coarse_idx0: x for x in self.atomized_state if x.atom_name is not None}.values()
        if atomized_residues:
            atomized_res_idx, atomized_res, _, atomized_idx, _ = zip(*atomized_residues)
        else:
            atomized_res_idx = []
            atomized_res = []
            atomized_idx = []
        
        return atomized_res_idx, atomized_res, atomized_idx
    
    def _update_rf_features_w_atomized_features(self, \
                                        indep_deatomized,
                                        seq_atomize_all, \
                                        xyz_atomize_all, \
                                        chirals_atomize_all,
                                        gp_atomize_all
                                        ):
        """
        adds the msa, xyz, frame and chiral features from the atomized regions to the rosettafold input tensors
        """
        indep_atomized = copy.deepcopy(indep_deatomized)
        
        # Handle MSA feature updates
        seq_atomize_all = torch.cat(seq_atomize_all)
        
        atomize_L = seq_atomize_all.shape[0]
        indep_atomized.seq = torch.cat((indep_deatomized.seq, seq_atomize_all), dim=0)

        # handle coordinates, need to handle permutation symmetry
        orig_L, natoms = indep_deatomized.xyz.shape[:2]
        total_L = orig_L + atomize_L
        xyz_atomize_all = rf2aa.util.cartprodcat(xyz_atomize_all)
        N_symmetry = xyz_atomize_all.shape[0]
        xyz = torch.full((N_symmetry, total_L, ChemData().NTOTAL, 3), np.nan).float()
        xyz[:, :orig_L, :natoms, :] = indep_deatomized.xyz.expand(N_symmetry, orig_L, natoms, 3)
        xyz[:, orig_L:, 1, :] = xyz_atomize_all
        xyz[:, orig_L:, :3, :] += ChemData().INIT_CRDS[:3]
        #ignoring permutation symmetry for now, network should learn permutations at low T
        indep_atomized.xyz = xyz[0]
        
        #handle idx_pdb 
        last_res = indep_deatomized.idx[-1]
        idx_atomize = torch.arange(atomize_L) + last_res
        indep_atomized.idx = torch.cat((indep_deatomized.idx, idx_atomize))
        
        # handle sm specific features- atom_frames, chirals
        indep_atomized.chirals = torch.cat(chirals_atomize_all)
        indep_atomized.terminus_type = torch.cat((indep_deatomized.terminus_type, torch.zeros(atomize_L).long()))
        indep_atomized.is_gp = torch.cat([indep_deatomized.is_gp, torch.cat(gp_atomize_all)])
        assert len(indep_atomized.is_gp) == indep_atomized.length()

        return indep_atomized

    def _pop_protein_feats(self, indep_atomized, res_idx_atomize_all):
        """
        after adding the atom information into the tensors, remove the associated protein sequence and template information
        """
        is_atomized_residue = torch.tensor(res_idx_atomize_all) # indices of residues that have been atomized and need other feats removed
        pop = torch.ones(indep_atomized.length())
        pop[is_atomized_residue] = 0
        pop = pop.bool()
        indep_atomized.seq = indep_atomized.seq[pop]
        indep_atomized.xyz = indep_atomized.xyz[pop]
        indep_atomized.idx = indep_atomized.idx[pop]
        indep_atomized.same_chain  = indep_atomized.same_chain[pop][:, pop]
        indep_atomized.bond_feats = indep_atomized.bond_feats[pop][:, pop].long()
        n_shift = (~pop).cumsum(dim=0)
        chiral_indices = indep_atomized.chirals[:,:-1]
        chiral_shift = n_shift[chiral_indices.long()]
        indep_atomized.chirals[:,:-1] = chiral_indices - chiral_shift
        indep_atomized.terminus_type = indep_atomized.terminus_type[pop]
        indep_atomized.is_gp = indep_atomized.is_gp[pop]
        assert len(indep_atomized.is_gp) == indep_atomized.length()
        
        return indep_atomized
    
    @staticmethod
    def _atomize_state(atomization_state: list[AtomizedLabel], idx0s: Iterable[int]) -> list[AtomizedLabel]:
        '''
        Changes the atomization_state for proper bookkeeping when
        atomizing residues.
        
        !! Note !!: You typically want to atomize in one shot, otherwise the
        idx0s could point to different residues after atomization.
        
        Inputs
            idx0s: 0-indices of coarse grained residues you wish to atomize.
        '''
        atomization_state = copy.deepcopy(atomization_state)
        new_atomization_states = []
        for idx0 in idx0s:
            removed_residue = atomization_state[idx0]
            atom_names = (name for name in ChemData().aa2long[removed_residue.aa][:ChemData().NHEAVY] if name is not None)
            new_atomization_states += [
                AtomizedLabel(
                    coarse_idx0=removed_residue.coarse_idx0, 
                    aa=removed_residue.aa, 
                    atom_name=atom_name, 
                    pdb_idx=removed_residue.pdb_idx,
                    terminus_type=removed_residue.terminus_type
                ) 
                for atom_name in atom_names]
        
        # Remove residues that were atomized...
        atomization_state = [x for i, x in enumerate(atomization_state) if i not in idx0s]
        
        # ...and append their atomized representations
        atomization_state += new_atomization_states

        return atomization_state
        
    @staticmethod
    def _deatomize_state(atomization_state: list[AtomizedLabel], coarse_idx0: int):
        '''
        Changes the atomization_state for proper bookkeeping when
        deatomizing a residue.
        
        Input
            coarse_idx0: The idx0 of where the residue would have been pre-atomization.
                This can be found be found in self.atomization_state.
        
        Returns
            deatomized_labels: The AtomizedLabels that were removed
            deatomized_idx0: The idx0 location of the removed AtomizedLabels
        '''
        atomization_state = copy.deepcopy(atomization_state)

        new_atomization_state = []
        deatomized_labels = []
        deatomized_idx0 = []
        for idx0, atomized_label in enumerate(atomization_state):
            if atomized_label.coarse_idx0 == coarse_idx0:
                deatomized_labels.append(atomized_label)
                deatomized_idx0.append(idx0)
            else:
                new_atomization_state.append(atomized_label)
        
        if deatomized_labels:
            atomized_label = AtomizedLabel(
                coarse_idx0=deatomized_labels[0].coarse_idx0,
                aa=deatomized_labels[0].aa,
                atom_name=None,
                pdb_idx=deatomized_labels[0].pdb_idx,
                terminus_type=deatomized_labels[0].terminus_type,
            )
            
            new_atomization_state.insert(coarse_idx0, atomized_label)
            
        return new_atomization_state, deatomized_labels, deatomized_idx0
            
    def get_atomized_res_idx_from_res(self) -> dict[int, int]:
        """Maps pre-atomized residue indices to post-atomized residue indices for non-atomized residues only.
        
        Returns a dictionary mapping original residue indices to their new indices after atomization, but only for 
        residues that were not atomized. For atomized residues, see get_atom_idx_by_res().

        Example:
            If residues 0 and 140 are atomized, the returned dict would be:
            {
                1: 0,    # Original residue 1 is now at index 0
                2: 1,    # Original residue 2 is now at index 1
                ...
                139: 138,  # Original residue 139 is now at index 138
                141: 139,  # Original residue 141 is now at index 139
                ...
            }
            Note that residues 0 and 140 are not in the mapping since they were atomized.

        Returns:
            dict[int, int]: Mapping from original residue indices to new indices after atomization
        """
        atomized_res_idx_from_res = {
            atomized_label.coarse_idx0: atomized_res_idx for atomized_res_idx, atomized_label 
            in enumerate(self.atomized_state) if atomized_label.atom_name is None
        }
        return atomized_res_idx_from_res
    
    def get_atom_idx_by_res(self) -> dict[int, torch.Tensor]:
        """Maps pre-atomized residue indices to post-atomized atom indices for atomized residues only.
        
        For residues that were atomized during transformation, returns a mapping from their original residue index to the 
        indices of their constituent atoms in the transformed structure. For non-atomized residues, 
        see get_atomized_res_idx_from_res().

        Returns:
            dict[int, torch.Tensor]: A dictionary mapping original residue indices to tensors containing the indices of 
                their constituent atoms after atomization. Only includes residues that were atomized.

        Example:
            If residue 0 and 140 are atomized and put at the end of a 253-residue sequence, the returned dict would be:
            {
                0: tensor([251, 252, 253, 254]),  # 251 because 0 and 140 are moved to end
                140: tensor([255, 256, 257, 258, 259, 260, 261, 262])
            }
        """

        atom_idx_by_res = defaultdict(list)
        for atomized_res_idx, atomized_label in enumerate(self.atomized_state):
            if atomized_label.atom_name is not None:
                atom_idx_by_res[atomized_label.coarse_idx0].append(atomized_res_idx)
                
        for k, v in atom_idx_by_res.items():
            atom_idx_by_res[k] = torch.tensor(v)
            
        return dict(atom_idx_by_res)

def choose_random_atom_motif(natoms, p=0.5):
    """
    selects each atom to be in the motif with a probability p 
    """
    return torch.rand((natoms)) > p

def choose_sm_contact_motif(indep, xyz_atomize):
    """
    chooses atoms to be the motif based on the atoms that are closest to the small molecule
    """
    dist = torch.cdist(indep.xyz[indep.is_sm, 1, :], xyz_atomize)
    closest_sm_atoms = torch.min(dist, dim=-2)[0][0] # min returns a tuple of values and indices, we want the values
    contacts = closest_sm_atoms < 4
    # if no atoms are closer than 4 angstroms, choose the closest three atoms
    if torch.all(contacts == 0):
        min_indices = torch.argsort(closest_sm_atoms)[:3]
        contacts[min_indices] = 1
    return contacts

def choose_contiguous_atom_motif(bond_feats_atomize):
    """
    chooses a contiguous 3 or 4 atom motif
    """
    natoms = bond_feats_atomize.shape[0]
    # choose atoms to be given as the motif 
    is_atom_motif = torch.zeros((natoms),dtype=bool)
    bond_graph = nx.from_numpy_matrix(bond_feats_atomize.numpy())
    paths = rf2aa.util.find_all_paths_of_length_n(bond_graph, 2)
    paths.extend(rf2aa.util.find_all_paths_of_length_n(bond_graph, 3))
    chosen_path = random.choice(paths)
    is_atom_motif[torch.tensor(chosen_path)] = 1
    return is_atom_motif

GP_BOND = 7
"""Integer used to indicate a guideposted bond (bond between a guidepost and the rest of the protein) in `bond_feats`."""

BACKBONE_BOND = 5
"""Integer used to indicate a backbone bond in `bond_feats`."""

def make_guideposts(indep: Indep, is_motif: torch.Tensor) -> tuple[Indep, dict[int, int]]:
    """
    Creates guidepost residues by duplicating selected residues from the input structure. Guidepost residues are placed 
    on separate chains and connected to their original residues via special guidepost bonds.

    Args:
    indep (Indep): Input structure containing sequence, coordinates, and other features
    is_motif (torch.Tensor[bool]): Boolean mask of length L indicating which residues should be duplicated as 
        guideposts. True indicates the residue will be duplicated.

    Returns:
        tuple[Indep, dict[int, int]]:
            - indep (Indep): Modified structure with guidepost residues added. Length is L + n_gp where n_gp is the 
                number of guideposts added. Guidepost residues are appended at the end.
            - gp_to_ptn_idx0 (dict[int, int]): Mapping from guidepost residue indices to their original residue 
                indices. Keys are indices in range [L, L+n_gp), values are indices in range [0, L).

    Note:
        - Guidepost residues are placed CHAIN_GAP (33) residues apart. With a 32-length relative sequence encoding this
            will put them on separate chains.
        - Guidepost bonds are indicated by GP_BOND (7) in the bond_feats matrix
        - Small molecules (indep.is_sm == True) cannot be made into guideposts
    """
    # If there is no motif, return the indep unchanged
    if is_motif.sum() == 0:
        return indep, {}
    
    # Else, prepare the guideposts
    pre_gp_len = indep.length()  # L
    n_motif = is_motif.sum()  # n_gp
    chains = indep.chains()  # e.g. ['A', 'A', 'B', 'B', ...]
    motif_chains = indep.chains()[is_motif]  # e.g. ['A', 'B', ...]
    indep_motif, cross_bonds = slice_indep(indep, is_motif) # cross_bonds: [n_gp, L-n_gp] (int - 5 = covalent bond)

    # Break peptide bonds
    indep_motif.bond_feats *= (indep_motif.bond_feats != BACKBONE_BOND) # [n_gp, n_gp] (int)
    gp_i = range(pre_gp_len, pre_gp_len + n_motif)
    
    # Position all guidepost atoms/residues on their individual chain, and on a different chain to the original indep
    CHAIN_GAP = 33
    indep_motif.idx = torch.arange(n_motif) * CHAIN_GAP + CHAIN_GAP + indep.idx.max()
    indep_motif.is_gp[:] = True

    # Concatenate the original indep with the guideposts
    indep_cat = cat_indeps_separate_chains((indep, indep_motif))
    chains_cat =  np.concatenate((chains, motif_chains))
    indep_cat.same_chain = same_chain_from_chain_letters(chains_cat)
    gp_to_ptn_idx0 = dict(zip(gp_i, is_motif.nonzero()[:,0].tolist()))  # dict[guidepost_idx, original_idx]

    is_inter_gp = indep_cat.is_gp[None, :] != indep_cat.is_gp[:, None]  # [L+n_gp, L+n_gp] (bool)
    has_sm = indep_cat.is_sm[None, :] + indep_cat.is_sm[:, None]  # [L+n_gp, L+n_gp] (bool)
    indep_cat.bond_feats[is_inter_gp * ~has_sm] = GP_BOND  # [L+n_gp, L+n_gp] (int) <-- set guidepost bonds to GP_BOND (7)
    return indep_cat, gp_to_ptn_idx0

def transform_indep(
        *,
        indep: Indep,
        is_res_str_shown: torch.Tensor,
        is_res_seq_shown: torch.Tensor,
        is_atom_str_shown: dict[int, list[str]],
        can_be_gp: torch.Tensor,
        use_guideposts: bool,
        guidepost_bonds: bool = True,
        metadata: dict | None = None
    ) -> tuple[Indep, torch.Tensor, torch.Tensor, AtomizeResidues | None, dict[int, int]]:
    '''
    Add guidepost residues (duplicates of residues set to be guideposted), atomize residues, and prepare is_diffused and is_seq_diffused

    Some definitions:
        - is_res_str_shown: Is the backbone of this residue/atom shown during diffusion? These residues will not be XYZ noised and de-noised
            Aliases: is_motif
            Anti-aliases: is_diffused
        - is_res_seq_shown: Is the sequence of this residue/atom shown during diffusion? This is independent of XYZ noising
            Anti-aliases: is_masked_seq
        - is_atom_str_shown: Should this residue be atomized? If so only atoms present in this list will be used

    Inputs:
        indep (Indep): indep
        is_res_str_shown (torch.Tensor[bool]): Is this residue/atom part of the motif that will not have XYZ noising? [L]
        is_res_seq_shown (torch.Tensor[bool]): Is the sequence of this residue/atom known during calls to RF2 (True) or is it mask (False)? [L]
        is_atom_str_shown (dict[int,list[str]]): Should this residue be atomized? dict(key=residue_idx, value=list(atom_name1, atom_name2, ...))
        can_be_gp (torch.Tensor[bool]): Is this residue elgible to be a guidepost? [L]
        use_guideposts (bool): Should motif residues (any residue or atom with str_shown) be duplicated into a guidepost residue?
        guidepost_bonds (bool): Whether to also guidepost the bonds (i.e. in the RF2AA bond matrix feats), using the special GP_BOND value (7)
        metadata (dict): Extra data about indep ligands

    Returns:
        indep (Indep): The indep but with guidepost residues added and residues atomized [new_L] = [L + n_gp]
        is_diffused (torch.Tensor[bool]): Which residues will have their xyz noised and denoised [new_L]
        is_masked_seq (torch.Tensor[bool]): Which residues will have their residue type set to mask [new_L]
        atomizer (Atomizer or None): The atomizer used to atomize the indep
        gp_to_ptn_idx0 (dict[int,int] or None): Lookup original motif residue from guidepost residue dict(key=guidepost_residue_idx, value=original_residue_idx)
    '''
    indep = copy.deepcopy(indep)
    # is_atom_str_shown: dict[int, list[str]], e.g. {97: [' N  ', ' CA ', ' C  ', ' CB '], 58: [' CD1', ' CG ', ' CB ', ' CD2'], 56: [' CA ', ' N  ', ' C  ', ' CB '], 127: [' N  '], 135: [' NH1'], 94: [' CD2', ' CG '], 161: [' NZ ', ' CE ', ' CD ']}
    is_res_str_shown = is_res_str_shown.clone()  # e.g. tensor([False, False, True, False, ...]), shape [L]
    is_res_seq_shown = is_res_seq_shown.clone()  # e.g. tensor([False, False, True, False, ...]), shape [L]
    use_atomize = is_atom_str_shown is not None
    atomizer = None
    gp_to_ptn_idx0 = None

    if (~is_res_seq_shown[indep.is_sm]).any():
        print('WARNING! Atoms may not have masked seq! See assert_valid_seq_mask()')
        is_res_seq_shown[indep.is_sm] = True

    # It really doesn't matter either way how these are set but set them to true such that asserts make sense
    #  if they happen to be guideposted
    is_res_str_shown[list(is_atom_str_shown.keys())] = True
    is_res_seq_shown[list(is_atom_str_shown.keys())] = True

    if use_guideposts:
        # All motif residues are turned into guideposts (for now...)
        mask_gp = is_res_str_shown.clone()
        mask_gp[indep.is_sm] = False  # Small molecules can never be guideposts
        if use_atomize:
            atomized_residues = list(is_atom_str_shown.keys())  # e.g. [97, 58, 56, ...]
            mask_gp[atomized_residues] = True  # show sequence of atomized residues
        mask_gp &= can_be_gp

        assert not (mask_gp & ~is_res_seq_shown).any(), "If a residue is set to be a guidepost, it's sequence must be shown."


        # Actually add guideposts to indep ( [L] -> [L + n_gp], guideposts are added at end, each on its own chain )
        indep, gp_to_ptn_idx0 = make_guideposts(indep, mask_gp)
        n_gp = len(gp_to_ptn_idx0)

        # The original copies of guidepost residues are diffused and sequence masked
        #  The duplicated residues (guideposts) are shown and have their sequences shown
        is_res_str_shown[list(gp_to_ptn_idx0.values())] = False
        is_res_str_shown = torch.cat((is_res_str_shown, torch.ones((n_gp,), dtype=bool)))
        is_res_seq_shown[list(gp_to_ptn_idx0.values())] = False
        is_res_seq_shown = torch.cat((is_res_seq_shown, torch.ones((n_gp,), dtype=bool)))

        if use_atomize:
            # Create mapping from original motif residue -> guidepost residue dict(key=original_residue_idx, value=guidepost_residue_idx)
            gp_from_ptn_idx0 = {v:k for k,v in gp_to_ptn_idx0.items()}

            # Translate is_atom_str_shown to be relative to the guideposts, not the original residues. If residue wasn't guideposted key remains the same
            is_atom_str_shown = {gp_from_ptn_idx0.get(k, k):v for k,v in is_atom_str_shown.items()}

            # Mark to-be-atomized duplicated guidepost residues to be diffused? This was to overcome a shortcoming in atomization. Delete this someday
            # is_res_str_shown[list(is_atom_str_shown.keys())] = False  # By all accounts this should be true. But these residues get dropped anyways
            # is_res_seq_shown[list(is_atom_str_shown.keys())] = False  # Same

            cov_resis = [gp_from_ptn_idx0[res_i] for (res_i, _), _, _ in metadata['covale_bonds']]
            for i, (a, b, t) in enumerate(metadata['covale_bonds']):
                res_i, atom_name = a
                assertpy.assert_that(gp_from_ptn_idx0).described_as('residues participating in covalent bonds to small molecules must be made into guideposts').contains(res_i)
                res_i = gp_from_ptn_idx0[res_i]
                assertpy.assert_that(is_atom_str_shown).described_as('residues participating in covalent bonds to small molecules must be atomized').contains(res_i)
                metadata['covale_bonds'][i] = ((res_i, atom_name), b, t)

    if len(metadata['covale_bonds']):
        # ... ensure that we atomize, if we have residues covalently bonded to a small molecule
        assert use_atomize, "Atomization is required for covalent bonds to small molecules"

    if use_atomize:
        is_covale_ligand = indep.type() == TYPE_ATOMIZED_COV
        is_ligand = indep.is_sm

        # convert is_atom_str_shown keys to strings if they are tensors
        is_atom_str_shown = {k.item() if hasattr(k, 'item') else k :v for k,v in is_atom_str_shown.items()}
        # actually atomize
        indep, tmp_is_diffused, tmp_is_masked_seq, atomizer = atomize.atomize_and_mask(
            indep=indep,
            is_res_str_shown=is_res_str_shown,
            is_res_seq_shown=is_res_seq_shown,
            is_atom_str_shown=is_atom_str_shown
        )
        # convert back to "shown" notation
        is_res_str_shown, is_res_seq_shown = ~tmp_is_diffused, ~tmp_is_masked_seq

        # TODO: Handle sequence masking more elegantly
        is_res_seq_shown[indep.is_sm] = True
        assertpy.assert_that(indep.is_sm.sum()).is_equal_to(indep.atom_frames.shape[0])
        ligand_idx = atomize.atomized_indices_res(atomizer, is_ligand)
        covale_ligand_idx = atomize.atomized_indices_res(atomizer, is_covale_ligand)
        
        if use_guideposts:
            is_inter_gp = indep.is_gp[None, :] != indep.is_gp[:, None]
            is_inter_gp[ligand_idx] = False
            is_inter_gp[:,ligand_idx] = False
            indep.bond_feats[is_inter_gp] = GP_BOND

    # Find the indices of atomized covalent residues
    for_join = []
    cov_resis = [res_i for (res_i, _), _, _ in metadata['covale_bonds']]
    for i in cov_resis:
        cov_atomized_idx0 = atomize.atomized_indices_from_preatomized_res_indices(atomizer, [i])
        for_join.append(cov_atomized_idx0)
    for_join.append(covale_ligand_idx)
    
    is_covale = torch.zeros(indep.length()).bool()
    for idx0 in for_join:
        is_covale[idx0] = True
    
    if not guidepost_bonds:
        # ... if false, remove guidepost bonds
        is_gp_bond = indep.bond_feats == GP_BOND  # [L+n_gp, L+n_gp] (bool)
        indep.bond_feats = indep.bond_feats * ~is_gp_bond  # [L+n_gp, L+n_gp] (int) <-- set guidepost bonds to 0 (non-bond)
    
    # Add back in bond feats
    atom_names_by_res = OrderedDict()
    for a, _, _ in metadata['covale_bonds']:
        res_i, atom_name = a
        res_i = int(res_i)
        if res_i not in atom_names_by_res:
            atom_names_by_res[res_i] = []
        atom_names_by_res[res_i].append(atom_name)
    atomized_i = atomize.atomized_indices_atoms(atomizer, atom_names_by_res)
    ligand_bond_recipient = torch.tensor([b for _, b, _ in metadata['covale_bonds']])
    ligand_bond_recipient = atomize.atomized_indices_res_i(atomizer, ligand_bond_recipient)

    ligand_bond_type = [c for _, _, c in metadata['covale_bonds']]
    for atom_i, ligand_i, bond_type in zip(atomized_i, ligand_bond_recipient, ligand_bond_type):
        # Uncomment to view the covale bond added here.
        # ic(
        #     'debug',
        #     atom_i, ligand_i,
        #     human_readable_seq(indep.seq[atom_i:atom_i+1]),
        #     human_readable_seq(indep.seq[ligand_i:ligand_i+1]),
        # )
        indep.bond_feats[atom_i, ligand_i] = bond_type
        indep.bond_feats[ligand_i, atom_i] = bond_type

    with open_indep(indep, is_covale) as covale:
        # Get obmol of combined atomized residues and covalent sm
        obmol, _ = get_obmol(covale.xyz[:,1], covale.seq, covale.bond_feats)
        covale.chirals = rf2aa.kinematics.get_chirals(obmol, covale.xyz[:, 1])

    is_diffused = ~is_res_str_shown
    is_masked_seq = ~is_res_seq_shown
    assertpy.assert_that(len(is_diffused)).is_equal_to(indep.length())
    # assertpy.assert_that(indep.xyz.shape[1]).is_equal_to(input_indep_xyz_shape[1])
    return indep, is_diffused, is_masked_seq, atomizer, gp_to_ptn_idx0


def generate_pre_to_post_transform_indep_mapping(indep, atomizer=None, gp_to_ptn_idx0=None):
    '''
    Generates a list of lists describing where each residue ended up post transform_indep

    Args:
        indep (indep): Indep
        atomizer (AtomizeResidues): The atomizer used to make this indep
        gp_to_ptn_idx0 (dict[int,int]): Maps the index of a guidepost residue (key) to the index that it came from (value), can be None

    Returns:
        pre_to_post_transform (list[list[int]]): A list mapping the original indep's numbering (key) to a list of where those residues are post (value) [L pre]
        is_atomized_res (torch.Tensor[bool]): In the post-transform indep, was this now atom originally a residue that was atomized
    '''


    # in transform indep, the process looks like this:
    # 1. Add gp generating gp_to_ptn_idx0
    # 2. Atomize residues generating atomizer

    # atomizer_mapping might be moving guideposts around
    atomizer_mapping, is_atomized_res = atomize.get_idx0_post_atomization_from_pre_atomization(indep.length(), atomizer)

    if is_atomized_res.sum() > 0:
        assert indep.is_sm[is_atomized_res].all(), "Atoms were atomized but aren't is_sm"

    # If guideposts weren't added, then atomizer_mapping would be what we want to return.
    #  However, some residues from the pre-transform have been duplicated into guideposts
    #    These residues are given by the keys of gp_to_ptn_idx0

    # Figure out what the is_gp vector looked like before atomization
    is_gp_before_atomization = torch.zeros(len(atomizer_mapping), dtype=bool)
    for i_before, is_after in enumerate(atomizer_mapping):
        gp_statuses = indep.is_gp[is_after]
        assert gp_statuses.all() or (~gp_statuses).all(), 'A guide-posted residue was atomized and because partially un-guideposted'
        is_gp_before_atomization[i_before] = gp_statuses[0]

    n_non_guidepost = (~is_gp_before_atomization).sum()
    if n_non_guidepost < len(is_gp_before_atomization):
        assert is_gp_before_atomization[n_non_guidepost:].all(), 'Guide-posted residues were not contiguously at the end pre-atomization'


    # Stack all of the guidepost residues into their original locations
    gp_from_ptn_idx0 = {v:k for k,v in gp_to_ptn_idx0.items()} if gp_to_ptn_idx0 is not None else {}
    pre_to_post_transform = []
    for i in range(n_non_guidepost):
        pre_to_post_transform.append(atomizer_mapping[i])
        if i in gp_from_ptn_idx0:
            pre_to_post_transform[i] += atomizer_mapping[gp_from_ptn_idx0[i]]


    return pre_to_post_transform, is_atomized_res


def hetatm_names(pdb):
    d = defaultdict(list)
    with open(pdb) as f:
        for line in f:
            if line.startswith('HETATM'):
                lig_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                element_name = line[76:78].strip()
                d[lig_name].append((atom_name, element_name))
    return d

def without_H(atom_elem_by_lig):
    ''' Drops Hs from a dictionary like {'LG1': [('CB', 'C'), ('H2', 'H')]}'''
    out = {}
    for lig, atom_names in atom_elem_by_lig.items():
        out[lig] = [(atom_name, element) for atom_name, element in atom_names if element != 'H']
    return out

def rename_ligand_atoms(ref_fn, out_fn, pdb_stream=None):
    '''
    Copies names of ligand residue and ligand heavy atoms from input pdb
    into output (design) pdb.

    Args:
        ref_fn (str): The file to grab names from
        out_fn (str or None): The file to modify, None if you don't want to write
        pdb_stream (list[str]): If present, assume this is the contents of out_fn

    Returns:
        pdb_stream (list[str]): The modified pdb data
    '''

    ref_atom_names_by_lig = hetatm_names(ref_fn)
    ref_atom_names_by_lig = without_H(ref_atom_names_by_lig)
    if pdb_stream is None:
        with open(out_fn) as f:
            pdb_stream = f.readlines()
    lines = [line.strip() for line in pdb_stream]

    lines2 = []
    ligand_counters = defaultdict(lambda: 0)
    for line in lines:
        if line.startswith('HETATM'):
            lig_name = line[17:20].strip()
            element_name = line[76:78].strip()
            assertpy.assert_that(ref_atom_names_by_lig).contains(lig_name)
            assertpy.assert_that(element_name).is_not_equal_to('H')
            ref_atom_name, ref_element_name = ref_atom_names_by_lig[lig_name][ligand_counters[lig_name]]
            assertpy.assert_that(element_name.upper()).is_equal_to(ref_element_name.upper())
            ligand_counters[lig_name] += 1
            line = line[:12] + ref_atom_name.ljust(4, ' ') + line[16:]
            line = line[:76] + ref_element_name.rjust(2, ' ') + line[78:]
        if line.startswith('MODEL'):
            ligand_counters = defaultdict(lambda: 0)
        lines2.append(line)

    if out_fn is not None:
        with open(out_fn,'w') as f:
            for line in lines2:
                print(line, file=f)

    return [line + '\n' for line in lines2]

def randomly_rotate_frames(xyz):
    L, _, _ = xyz.shape
    R_rand = rotation_conversions.random_rotations(L, dtype=xyz.dtype)
    frame_origins = xyz[:,1:2,:]
    xyz_centered = xyz - frame_origins
    rotated = torch.einsum('lab,lib->...lia', R_rand, xyz_centered)
    rotated += frame_origins
    return rotated

def eye_frames(xyz, recenter=False):
    """
    replaces frames in xyz with identity frames, this version simply using chemical.INIT_CRDS as frame.
    """
    L, _, _ = xyz.shape

    T = xyz[:,1,:] # CA coords

    init = ChemData().INIT_CRDS[:3] # (3,3)
    init = init[None].expand(L,3,3) # (L,3,3), replaces N,CA,C such that frame is identity

    xyz[:,:3,:] = init + T[:,None,:].expand(L,3,3)

    if recenter:
        xyz = xyz - xyz[:,1:2,:].mean(dim=0, keepdim=True)
        
    return xyz


def functionalize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create deep copies of the arguments to prevent modification
        args_copy = copy.deepcopy(args)
        kwargs_copy = copy.deepcopy(kwargs)

        # Call the original function with the copied arguments
        return func(*args_copy, **kwargs_copy)

    return wrapper


def standardize_frames(atom_frames):
    o = atom_frames.clone()
    for i, f in enumerate(atom_frames):
        if f[0,0] < f[2,0]:
            continue
        o[i, 0, 0] = atom_frames[i, 2, 0]
        o[i, 2, 0] = atom_frames[i, 0, 0]
    return o

def make_mask(i, L):
    mask = torch.zeros((L,)).bool()
    mask[i] = True
    return mask

@contextlib.contextmanager
def open_indep(indep, is_open, break_chirals=False):
    assertpy.assert_that(indep.length()).is_equal_to(len(is_open))
    indep_closed, _ = slice_indep(indep, ~is_open, break_chirals=break_chirals)
    indep_open, _ = slice_indep(indep, is_open, break_chirals=break_chirals)
    yield indep_open
    i = torch.arange(len(is_open))
    i_r = torch.cat([i[~is_open], i[is_open]])
    i_inv = torch.argsort(i_r)
    indep_cat = cat_indeps_separate_chains((indep_closed, indep_open))
    rearrange_indep(indep_cat, i_inv)
    is_cross_term = ~(is_open[:, None] == is_open[None, :])
    indep_cat.bond_feats[is_cross_term] = indep.bond_feats[is_cross_term]
    indep_cat.same_chain[is_cross_term] = indep.same_chain[is_cross_term]
    for key, value in dataclasses.asdict(indep_cat).items():
        setattr(indep, key, value)

def residue_atoms(res):
    return [n.strip() for n in ChemData().aa2long[res][:ChemData().NHEAVY] if n is not None]

def make_conditional_indep(indep, indep_original, is_diffused, sm_assertion=True):
    if sm_assertion:
        assert indep.is_sm[~is_diffused].all(), 'sequence unmasking not yet implemented, only coordinate conditioning, so only atomized/small molecule motifs are allowed'
    indep = copy.deepcopy(indep)
    indep.xyz[~is_diffused, :ChemData().NHEAVY] = indep_original.xyz[~is_diffused, :ChemData().NHEAVY]
    return indep
