import functools
import torch
import contextlib
from functools import wraps
import assertpy
from collections import defaultdict, OrderedDict
import torch.nn.functional as F
import dataclasses
from icecream import ic
from assertpy import assert_that
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVYPROT, UNKINDEX
import rf2aa.util
from rf2aa import parsers
from dataclasses import dataclass
from rf2aa.data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from rf2aa.util_module import XYZConverter
import rf2aa.tensor_util
import torch
import copy
import numpy as np
from kinematics import get_init_xyz
# import chemical
from rf2aa.chemical import MASKINDEX
import rf2aa.chemical
from rf2aa import chemical
import util
import inference.utils
import networkx as nx
nx.from_numpy_matrix = nx.from_numpy_array
import itertools
import random
import guide_posts as gp
import rotation_conversions
import atomize

import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from data import utils as du
from data import all_atom
import show
import features


NINDEL=1
NTERMINUS=2
NMSAFULL=NAATOKENS+NINDEL+NTERMINUS
NMSAMASKED=NAATOKENS+NAATOKENS+NINDEL+NINDEL+NTERMINUS

MSAFULL_N_TERM = NAATOKENS+NINDEL
MSAFULL_C_TERM = MSAFULL_N_TERM+1

MSAMASKED_N_TERM = 2*NAATOKENS + 2*NINDEL
MSAMASKED_C_TERM = 2*NAATOKENS + 2*NINDEL + 1

N_TERMINUS = 1
C_TERMINUS = 2

UNIQUE_LIGAND="__UNIQUE_LIGAND"


def chain_letters_from_same_chain(same_chain):
    L = same_chain.shape[0]
    G = nx.from_numpy_array(same_chain.numpy())
    cc = list(nx.connected_components(G))
    cc.sort(key=min)
    chain_letters = np.chararray((L,), unicode=True)

    for ch_i, ch_name in zip(cc, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
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

@dataclass
class Indep:
    seq: torch.Tensor # [L]
    xyz: torch.Tensor # [L, 36?, 3]
    idx: torch.Tensor

    # SM specific
    bond_feats: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    same_chain: torch.Tensor
    is_sm: torch.Tensor
    terminus_type: torch.Tensor
    extra_t1d: torch.Tensor = dataclasses.field(default_factory=lambda: None)

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
        return rf2aa.util.writepdb_file(fh,
            torch.nan_to_num(self.xyz[:,:14]), seq, idx_pdb=self.idx, chain_letters=chain_letters, bond_feats=self.bond_feats[None], **kwargs)

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
        # return np.chararray([rf2aa.chemical.num2aa[s] for s in self.seq], unicode=False)
    
    def has_heavy_atoms_and_seq(self, atom_mask):
        want_atom_mask = rf2aa.util.allatom_mask[self.seq]
        has_all_heavy_atoms = (want_atom_mask[:,:rf2aa.chemical.NHEAVYPROT] == atom_mask[:,:rf2aa.chemical.NHEAVYPROT]).all(dim=-1)
        has_sequence = self.seq < 20
        return has_all_heavy_atoms * ~self.is_sm * has_sequence
    
    def is_valid_for_atomization(self, atom_mask):
        return self.has_c_terminal_residue() * self.has_n_terminal_residue() * self.has_heavy_atoms_and_seq(atom_mask)

    def human_readable_atom_frames(self):
        atom_frames_absolute = self.atom_frames[:,:,0].clone()
        atom_frames_absolute += (torch.arange(self.length())[self.is_sm])[:,None]
        o = []

        def atom_label(i):
            return (i, rf2aa.chemical.num2aa[self.seq[i]])
        for a,b,c in atom_frames_absolute.tolist():
            o.append(
                (atom_label(a), atom_label(b), atom_label(c))
            )
        return o
    
    def type(indep):
        chains = indep.chains()
        chains_with_prot = np.unique(chains[~indep.is_sm])
        is_on_same_chain_as_prot = torch.tensor(np.isin(chains, chains_with_prot))
        is_atomized_cov = indep.is_sm * is_on_same_chain_as_prot
        is_ligand = indep.is_sm * ~is_on_same_chain_as_prot
        metadata = {}
        metadata['type'] = torch.zeros((indep.length()))
        metadata['type'][:] = -1
        metadata['type'][~indep.is_sm] = TYPE_PROT
        metadata['type'][is_ligand] = TYPE_LIGAND
        metadata['type'][is_atomized_cov] = TYPE_ATOMIZED_COV
        return metadata['type']
    

    def get_connected(self, i):
        G = nx.from_numpy_matrix(self.bond_feats.detach().cpu().numpy())
        ic(G.nodes, i)
        connected_idx0 = fetch_connected_nodes(G, i)
        return torch.tensor(list(connected_idx0))
    
    def assert_types(indep):
        assertpy.assert_that(indep.same_chain.dtype).is_equal_to(torch.bool)

    def atom_label(self, i):
        return (i, rf2aa.chemical.num2aa[self.seq[i]])
    
    def human_readable_2d_mask(self, mask):
        o = []
        for i, j in mask.nonzero():
            o.append((self.atom_label(i), self.atom_label(j)))
        return o
    
    def human_readable_2d_symmetric_mask(self, mask):
        assertpy.assert_that((mask.T == mask).all()).is_true()
        return self.human_readable_2d_mask(torch.triu(mask))

def human_readable_seq(seq):
    return [rf2aa.chemical.num2aa[s] for s in seq]

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

    # dataclass.astuple returns a deepcopy of the dataclass in which
    # gradients of member tensors are detached, so we define a 
    # custom unpacker here.
    def unsafe_astuple(self):
        return tuple([self.__dict__[field.name] for field in dataclasses.fields(self)])

    def get_seq_logits(self):
        return self.logits_aa.permute(0,2,1)
    
    def get_xyz(self):
        return self.xyz_allatom[0]

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
    ligand = list(ligands)[0]
    return ligand

def get_only_ligand(pdb_lines):
    ligands = get_ligands(pdb_lines)
    assertpy.assert_that(len(ligands), description=ligands).is_equal_to(1)
    ligand = list(ligands)[0]
    return ligand

def get_non_target_hetatm_ids(pdb_lines, ligands):
    non_target_hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' in l:
            curr_ligand = l[17:17+4].strip()
            if curr_ligand not in ligands:
                non_target_hetatm_ids.append(int(l[6:6+5].strip()))
                continue
    return set(non_target_hetatm_ids)

def remove_non_target_ligands(pdb_lines, ligands, cautious=False):
    non_target_hetatm_ids = get_non_target_hetatm_ids(pdb_lines, ligands)
    lines = []
    violations = []
    for l in pdb_lines:
        if 'HETATM' in l:
                atom_id = int(l[6:6+5].strip())
                if atom_id in non_target_hetatm_ids:
                    continue
        if 'CONECT' in l:
            ids = [int(e.strip()) for e in l[6:].strip().split()]
            if all(i in non_target_hetatm_ids for i in ids):
                continue
            if any(i in non_target_hetatm_ids for i in ids):
                target_to_non_target = [i for i in ids if i in non_target_hetatm_ids]
                violations.append(f'line {l} references atom ids in a non-target ligand, but also target atoms: {target_to_non_target}')
                new_l = ['CONECT'] 
                for i in ids:
                    if i not in non_target_hetatm_ids:
                        new_l.append(f'{123:>4}')
                l = ''.join(new_l)
        lines.append(l)
    if len(violations) and cautious:
        raise Exception('\n'.join(violations))
    return lines

def filter_het(pdb_lines, ligand, covale_allowed=False):
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        curr_ligand = l[17:17+4].strip()
        if curr_ligand != ligand:
            continue
        lines.append(l)
        hetatm_ids.append(int(l[7:7+5].strip()))

    violations = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        if all(i in hetatm_ids for i in ids):
            lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            ligand_atms_bonded_to_protein = [i for i in ids if i in hetatm_ids]
            violations.append(f'line {l} references atom ids in the target ligand {ligand}: {ligand_atms_bonded_to_protein} and another atom')
    if violations and not covale_allowed:
        raise Exception('\n'.join(violations))
    return lines

def get_hetatm_ids(pdb_lines, ligand):
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        curr_ligand = l[17:17+4].strip()
        if curr_ligand != ligand:
            continue
        lines.append(l)
        hetatm_ids.append(int(l[7:7+5].strip()))
    return hetatm_ids

def get_bonds(pdb_lines):
    ligand_atom_pdb_atom = []
    from_to = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        for r in ids[1:]:
            from_to.append(tuple(sorted((ids[0], r))))
    
    # ic(from_to)
    from_to = list(set(from_to))
    return from_to

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


def make_indep(pdb, ligand=None, center=True, return_metadata=False):
    # self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
    # init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False, init_ligand_xyz=False,
    #     parse_hetatm=False, n_cycle=10, random_noise=5.0)
    chirals = torch.zeros((0, 5))
    atom_frames = torch.zeros((0,3,2), dtype=torch.int64)

    # xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb, seq=True, parse_hetatom=True)

    with open(pdb, 'r') as fh:
        stream = fh.readlines()
    stream = remove_non_target_ligands(stream, [ligand])
    target_feats = inference.utils.parse_pdb_lines(stream, parse_hetatom=True)
    het_atom_uids = [(e['res_idx'], e['atom_id'].strip()) for e in target_feats['info_het']]
    prot_atom_uids = [(idx, 'CA') for idx in target_feats['idx']]
    uids = prot_atom_uids + het_atom_uids
    xyz_prot, mask_prot, idx_prot, seq_prot = target_feats['xyz'], target_feats['mask'], target_feats['idx'], target_feats['seq']
    xyz_prot[:,14:] = 0 # remove hydrogens
    mask_prot[:,14:] = False
    xyz_prot = torch.tensor(xyz_prot)
    mask_prot = torch.tensor(mask_prot)
    protein_L, nprotatoms, _ = xyz_prot.shape
    msa_prot = torch.tensor(seq_prot)[None].long()
    ins_prot = torch.zeros(msa_prot.shape).long()
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    covale_bonds = []
    if ligand:
        protein_ligand_bonds_atoms = find_covale_bonds(stream, ligand)
        print('Protein-ligand bonds:')
        for i, (d,r) in enumerate(protein_ligand_bonds_atoms):
            print(f'{d.get_full_id()} : {r.get_full_id()}')
        for protein_atom, ligand_atom in protein_ligand_bonds_atoms:
            prot_res_idx, prot_atom_name = get_atom_uid(protein_atom)
            res_i = uids.index((prot_res_idx, 'CA'))
            ligand_atom_uid = get_atom_uid(ligand_atom)
            atom_i = uids.index((ligand_atom_uid))
            # Hack, no way to detect bond types in PDB
            bond_type = 1 # Single bond
            covale_bonds.append(
                ((res_i, prot_atom_name), atom_i, bond_type)
            )
        ic(covale_bonds)

        with open(pdb, 'r') as fh:
            stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
        if ligand == UNIQUE_LIGAND:
            raise Exception('not implemented')
            ligand = get_only_ligand(pdb_lines)

        stream = filter_het(stream, ligand, covale_allowed=True)
        if not len(stream):
            raise Exception(f'ligand {ligand} not found in pdb: {pdb}')

        mol, msa_sm, ins_sm, xyz_sm, _ = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
        assertpy.assert_that(len(het_atom_uids)).is_equal_to(xyz_sm.shape[1])
        a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
        G = rf2aa.util.get_nxgraph(mol)
        atom_frames = rf2aa.util.get_atom_frames(msa_sm, G)
        N_symmetry, sm_L, _ = xyz_sm.shape
        Ls = [protein_L, sm_L]
        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
        msa = a3m['msa'].long()
        chirals = get_chirals(mol, xyz_sm[0])
        if chirals.numel() !=0:
            chirals[:,:-1] += protein_L
    else:
        Ls = [msa_prot.shape[-1], 0]
        N_symmetry = 1
        msa = msa_prot

    xyz = torch.full((N_symmetry, sum(Ls), NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
    if ligand:
        xyz[:, Ls[0]:, 1, :] = xyz_sm
    xyz = xyz[0]
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, Ls[0], nprotatoms)
    idx_sm = torch.arange(max(idx_prot),max(idx_prot)+Ls[1])+200
    idx_pdb = torch.concat([torch.tensor(idx_prot), idx_sm])
    
    seq = msa[0]
    
    # seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, 
    #     p_mask=0.0, params={'MAXLAT': 128, 'MAXSEQ': 1024, 'MAXCYCLE': n_cycle}, tocpu=True)
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = rf2aa.util.get_protein_bond_feats(Ls[0])
    if ligand:
        bond_feats[Ls[0]:, Ls[0]:] = rf2aa.util.get_bond_feats(mol)

    same_chain = torch.zeros((sum(Ls), sum(Ls))).bool()
    same_chain[:Ls[0], :Ls[0]] = True
    same_chain[Ls[0]:, Ls[0]:] = True
    # Amend same chain for the covalently linked small molecule case
    same_chain = same_chain_with_covale(same_chain, covale_bonds)

    is_sm = torch.zeros(sum(Ls)).bool()
    is_sm[Ls[0]:] = True
    assert len(Ls) <= 2, 'multi chain inference not implemented yet'
    terminus_type = torch.zeros(sum(Ls))
    terminus_type[0] = N_TERMINUS
    terminus_type[Ls[0]-1] = C_TERMINUS

    if center:
        xyz = get_init_xyz(xyz[None, None], is_sm).squeeze()
    ###TODO: currently network needs values at 0,2 indices of tensor, need to remove this reliance
    xyz[is_sm, 0] = 0
    xyz[is_sm, 2] = 0
    indep = Indep(
        seq,
        xyz,
        idx_pdb,
        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain,
        is_sm,
        terminus_type)
    if return_metadata:
        metadata = {'covale_bonds': covale_bonds}
        return indep, metadata
    return indep

def add_fake_frame_legs(xyz, is_atom):
    # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
    # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
    xyz = xyz.clone()
    atom_xyz = xyz[is_atom, 1]
    xyz[is_atom,:3] = atom_xyz[...,None,:]
    xyz[is_atom, 0] += torch.normal(torch.zeros_like(xyz[is_atom, 0]), std=1.0)
    xyz[is_atom, 2] += torch.normal(torch.zeros_like(xyz[is_atom, 2]), std=1.0)
    return xyz

class Model:

    def __init__(self, conf):
        self.conf = conf
        self.NTOKENS = rf2aa.chemical.NAATOKENS
        self.atomizer = None
        self.converter = XYZConverter()

    def forward(self, rfi, **kwargs):
        # ipdb.set_trace()
        rfi_dict = dataclasses.asdict(rfi)
        # assert set(rfi_dict.keys()) - set()
        return RFO(*self.model(**{**rfi_dict, **kwargs}))


    def insert_contig(self, indep, contig_map, partial_T=False, metadata=None):
        metadata = metadata or defaultdict(dict)
        o = copy.deepcopy(indep)

        # Insert small mol into contig_map
        all_chains = set(ch for ch,_ in contig_map.hal)
        # Not yet implemented due to index shifting
        assert_that(len(all_chains)).is_equal_to(1)
        next_unused_chain = next(e for e in contig_map.chain_order if e not in all_chains)
        n_sm = indep.is_sm.sum()
        is_sm_idx0 = torch.nonzero(indep.is_sm, as_tuple=True)[0].tolist()
        contig_map.ref_idx0.extend(is_sm_idx0)
        n_protein_hal = len(contig_map.hal)
        contig_map.hal_idx0 = np.concatenate((contig_map.hal_idx0, np.arange(n_protein_hal, n_protein_hal+n_sm)))
        max_hal_idx = max(i for _, i  in contig_map.hal)
        contig_map.hal.extend(zip([next_unused_chain]*n_sm, range(max_hal_idx+200,max_hal_idx+200+n_sm)))
        chain_id = np.array([c for c, _ in contig_map.hal])
        L_mapped = len(contig_map.hal)
        n_prot = L_mapped - n_sm
        L_in, NATOMS, _ = indep.xyz.shape
        o.xyz = torch.full((L_mapped, NATOMS, 3), np.nan)

        o.xyz[contig_map.hal_idx0] = indep.xyz[contig_map.ref_idx0]
        o.seq = torch.full((L_mapped,), MASKINDEX)
        o.seq[contig_map.hal_idx0] = indep.seq[contig_map.ref_idx0]
        o.is_sm = torch.full((L_mapped,), 0).bool()
        o.is_sm[contig_map.hal_idx0] = indep.is_sm[contig_map.ref_idx0]
        o.same_chain = torch.tensor(chain_id[None, :] == chain_id[:, None])
        o.xyz = get_init_xyz(o.xyz[None, None], o.is_sm).squeeze()

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
        o.terminus_type[0] = N_TERMINUS
        o.terminus_type[n_prot-1] = C_TERMINUS

        is_diffused_prot = ~torch.from_numpy(contig_map.inpaint_str)
        is_diffused_sm = torch.zeros(n_sm).bool()
        is_diffused = torch.cat((is_diffused_prot, is_diffused_sm))
        is_atom_str_shown = contig_map.atomize_indices2atomname
        # The motifs for atomization are double-counted.
        if is_atom_str_shown:
            is_diffused[list(is_atom_str_shown.keys())] = True
        is_res_str_shown = ~is_diffused
        for i, ((res_i, atom_name), sm_i, bond_type) in enumerate(metadata['covale_bonds']):
            res_i = hal_by_ref_d[res_i]
            sm_i = hal_by_ref_d[sm_i]
            metadata['covale_bonds'][i] = ((res_i, atom_name), sm_i, bond_type)

        o.same_chain = same_chain_with_covale(o.same_chain, metadata['covale_bonds'])

        use_guideposts = self.conf.dataloader.USE_GUIDE_POSTS
        pre_transform_length = o.length()
        o, is_diffused, is_seq_masked, self.atomizer, contig_map.gp_to_ptn_idx0 = transform_indep(o, is_res_str_shown, is_atom_str_shown, use_guideposts, 'anywhere', self.conf.guidepost_bonds, metadata=metadata)
        # o.extra_t1d = torch.zeros((o.length(),0))
        # HACK: gp indices may be lost during atomization, so we assume they are at the end of the protein.
        is_gp = torch.full((o.length(),), True)
        is_gp[:pre_transform_length] = False
        extra_t1d = getattr(self.conf, 'extra_t1d', [])
        o.extra_t1d = features.get_extra_t1d_inference(o, extra_t1d, self.conf.extra_t1d_params, self.conf.inference.conditions, is_gp=is_gp)
        # for i, e in enumerate(o.extra_t1d.T):
        #     ic(i, e)
        # ic(self.conf.inference.conditions, extra_t1d, o.extra_t1d.shape)

        # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
	    # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
        sm_ca = o.xyz[o.is_sm, 1]
        o.xyz[o.is_sm,:3] = sm_ca[...,None,:]
        o.xyz[o.is_sm] += chemical.INIT_CRDS

        # To see the shapes of the indep struct with contig inserted
        # print(rf2aa.tensor_util.info(rf2aa.tensor_util.to_ordered_dict(o)))
        return o, is_diffused, is_seq_masked


    def prepro(self, indep, t, is_diffused):
        """
        Function to prepare inputs to diffusion model
        
            seq (L,22) one-hot sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)
        
            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)
                - contacting residues: for ppi. Target residues in contact with biner (1)
                - chi_angle timestep (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (1, L, L, 45)
                - last plane is block adjacency
        """
        xyz_t = indep.xyz
        seq_one_hot = torch.nn.functional.one_hot(
                indep.seq, num_classes=self.NTOKENS).float()
        L = seq_one_hot.shape[0]


        '''
        msa_full:   NSEQ,NINDEL,NTERMINUS,
        msa_masked: NSEQ,NSEQ,NINDEL,NINDEL,NTERMINUS
        '''
        NTERMINUS = 2
        NINDEL = 1
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,2*NAATOKENS+NINDEL*2+NTERMINUS))

        msa_masked[:,:,:,:NAATOKENS] = seq_one_hot[None, None]
        msa_masked[:,:,:,NAATOKENS:2*NAATOKENS] = seq_one_hot[None, None]
        if self.conf.preprocess.annotate_termini:
            msa_masked[:,:,:,MSAMASKED_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
            msa_masked[:,:,:,MSAMASKED_C_TERM] = (indep.terminus_type == C_TERMINUS).float()

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,NAATOKENS+NINDEL+NTERMINUS))
        msa_full[:,:,:,:NAATOKENS] = seq_one_hot[None, None]
        if self.conf.preprocess.annotate_termini:
            msa_full[:,:,:,MSAFULL_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
            msa_full[:,:,:,MSAFULL_C_TERM] = (indep.terminus_type == C_TERMINUS).float()

        ### t1d ###
        ########### 
        # Here we need to go from one hot with 22 classes to one hot with 21 classes
        # If sequence is masked, it becomes unknown
        # t1d = torch.zeros((1,1,L,NAATOKENS-1))

        #seqt1d = torch.clone(seq)
        seq_cat_shifted = seq_one_hot.argmax(dim=-1)
        seq_cat_shifted[seq_cat_shifted>=MASKINDEX] -= 1
        t1d = torch.nn.functional.one_hot(seq_cat_shifted, num_classes=NAATOKENS-1)
        t1d = t1d[None, None] # [L, NAATOKENS-1] --> [1,1,L, NAATOKENS-1]
        # for idx in range(L):
            
        #     if seqt1d[idx,MASKINDEX] == 1:
        #         seqt1d[idx, MASKINDEX-1] = 1
        #         seqt1d[idx,MASKINDEX] = 0
        # t1d[:,:,:,:NPROTAAS+1] = seqt1d[None,None,:,:NPROTAAS+1]
        
        ## Str Confidence
        # Set confidence to 1 where diffusion mask is True, else 1-t/T
        strconf = torch.zeros((L,)).float()
        strconf[~is_diffused] = 1.
        strconf[is_diffused] = 1. - t/self.conf.diffuser.T
        strconf = strconf[None,None,...,None]

        t1d = torch.cat((t1d, strconf), dim=-1)
        t1d = t1d.float()

        ### xyz_t ###
        #############
        if self.conf.preprocess.sidechain_input:
            raise Exception('not implemented')
            xyz_t[torch.where(seq_one_hot == 21, True, False),3:,:] = float('nan')
        else:
            xyz_t[is_diffused,3:,:] = float('nan')
        #xyz_t[:,3:,:] = float('nan')

        assert_that(xyz_t.shape).is_equal_to((L,NHEAVYPROT,3))
        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,NTOTAL-NHEAVYPROT,3), float('nan'))), dim=3)

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
        # seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        # alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        # alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        # alpha[torch.isnan(alpha)] = 0.0
        # alpha = alpha.reshape(1,-1,L,10,2)
        # alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        # alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)


        # get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)

        alpha, _, alpha_mask, _ = self.converter.get_torsions(xyz_t.reshape(-1,L,rf2aa.chemical.NTOTAL,3), seq_tmp)
            #rf2aa.util.torsion_indices, rf2aa.util.torsion_can_flip, rf2aa.util.reference_angles)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(-1,L,rf2aa.chemical.NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(-1,L,rf2aa.chemical.NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 3*rf2aa.chemical.NTOTALDOFS) # [n,L,30]

        alpha_t = alpha_t.unsqueeze(1) # [n,I,L,30]
        alpha_t = alpha_t.tile((1,2,1,1))



        # #put tensors on device
        # msa_masked = msa_masked.to(self.device)
        # msa_full = msa_full.to(self.device)
        # seq = seq.to(self.device)
        # xyz_t = xyz_t.to(self.device)
        # #idx = idx.to(self.device)
        # t1d = t1d.to(self.device)
        # # t2d = t2d.to(self.device)
        # alpha_t = alpha_t.to(self.device)
        
        ### added_features ###
        ######################
        # NB the hotspot input has been removed in this branch. 
        # JW added it back in, using pdb indexing

        if self.conf.preprocess.d_t1d == 24: # add hotpot residues
            raise Exception('not implemented')
            if self.ppi_conf.hotspot_res is None:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots. If you're doing monomer diffusion this is fine")
                hotspot_idx=[]
            else:
                hotspots = [(i[0],int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx=[]
                for i,res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
            hotspot_tens = torch.zeros(L).float()
            hotspot_tens[hotspot_idx] = 1.0
            t1d=torch.cat((t1d, hotspot_tens[None,None,...,None].to(self.device)), dim=-1)
        # Uncomment to see categorical extra_t1d_v2
        # ic(
        #     indep.extra_t1d.shape,
        #     indep.extra_t1d[:,:10].argmax(dim=-1),
        #     indep.extra_t1d[:,10:].argmax(dim=-1),
        # )
        t1d = torch.cat((t1d, indep.extra_t1d[None, None, ...]), dim=-1)
        
        # return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        mask_t = torch.ones(1,2,L,L).bool()
        sctors = torch.zeros((1,L,rf2aa.chemical.NTOTALDOFS,2))

        xyz = torch.squeeze(xyz_t, dim=0)

        # NO SELF COND
        xyz_t = torch.zeros(1,2,L,3)
        t2d = torch.zeros(1,2,L,L,68)

        t2d_xt, mask_t_2d_remade = util.get_t2d(
            xyz, indep.is_sm, indep.atom_frames)
        t2d[0,0] = t2d_xt[0]
        xyz_t[0,0] = xyz[0,:,1]

        #ic(xyz.shape)
        # ic(
        #     xyz[0, is_diffused][0][:,0], # nan 3:
        #     xyz[0, indep.is_sm][0][:,0], # nan 14:
        #     xyz[0, ~is_diffused * ~indep.is_sm][0][:,0], # nan 14:
        # )

        is_protein_motif = ~is_diffused * ~indep.is_sm
        # idx_diffused = torch.nonzero(is_diffused)
        # idx_protein_motif  = torch.nonzero(is_protein_motif)
        # idx_sm = torch.nonzero(indep.is_sm)

        # ic(
        #     idx_diffused,
        #     idx_protein_motif,
        #     idx_sm
        # )

        # xyz = torch.nan_to_num(xyz)
        xyz[0, is_diffused*~indep.is_sm,3:] = torch.nan
        xyz[0, indep.is_sm,14:] = 0
        xyz[0, is_protein_motif, 14:] = 0
        dist_matrix = rf2aa.data_loader.get_bond_distances(indep.bond_feats)
        
        t1d = torch.tile(t1d, (1,2,1,1))
        # This is messed up for extra_t1d
        t1d[0,1,:,NAATOKENS-1] = -1
        # ic(t1d[0,:,:4,NAATOKENS-1]) # Should look like [[conf, -1], [conf, -1], ...], 0 < conf < 1

        # Note: should be batched
        rfi = RFI(
            msa_masked,
            msa_full,
            indep.seq[None],
            indep.seq[None],
            xyz,
            sctors,
            indep.idx[None],
            indep.bond_feats[None],
            dist_matrix[None],
            indep.chirals[None],
            indep.atom_frames[None],
            t1d,
            t2d,
            xyz_t,
            alpha_t,
            mask_t,
            indep.same_chain[None],
            ~is_diffused,
            None,
            None,
            None)
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

def write_traj(path, xyz_stack, seq, bond_feats, natoms=23, **kwargs):
    xyz23 = pad_dim(xyz_stack, 2, natoms)
    if bond_feats is not None:
        bond_feats = bond_feats[None]
    with open(path, 'w') as fh:
        for i, xyz in enumerate(xyz23):
            rf2aa.util.writepdb_file(fh, xyz, seq, bond_feats=bond_feats, modelnum=i, **kwargs)

def minifier(argument_map):
    argument_map['out_9'] = None
    argument_map['out_0'] = None
    argument_map['out_2'] = None
    argument_map['out_3'] = None
    argument_map['out_5'] = None
    argument_map['t2d'] = None

TYPE_PROT = 0
TYPE_LIGAND = 1
TYPE_ATOMIZED_COV = 2


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
    atomnumbyatomtype = {v:k for k,v in chemical.atomnum2atomtype.items()}
    akeys = []
    atoms = []
    bonds = []
    for i, (_, seq) in enumerate(zip(xyz_sm, seq_sm)):
        atom_name = chemical.num2aa[seq]
        atomnum = atomnumbyatomtype[atom_name]
        # xyz.append(xyz_i)
        atoms.append(Atom(atomnum))
        akeys.append(i)

    for i, j in bond_feats_sm.nonzero():
        if i > j:
            continue
        bonds.append(
            Bond(i, j, bond_feats_sm[i,j].item(), False)
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

    is_sm = rf2aa.util.is_atom(seq)

    is_n_terminus = msa_full[0, 0, :, MSAFULL_N_TERM].bool()
    is_c_terminus = msa_full[0, 0, :, MSAFULL_C_TERM].bool()
    terminus_type = torch.zeros(msa_masked.shape[2], dtype=int)
    terminus_type[is_n_terminus] = N_TERMINUS
    terminus_type[is_c_terminus] = C_TERMINUS

    indep = Indep(
        rf2aa.tensor_util.assert_squeeze(seq), # [L]
        true_crds[:,:14], # [L, 14, 3]
        idx_pdb,

        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain.bool(),
        rf2aa.tensor_util.assert_squeeze(is_sm),
        terminus_type)
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
    
    metadata = {}
    metadata['type'] = indep.type()
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
        atom_names = rf2aa.chemical.aa2long[original_aa]
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
    all_atom_mask = rf2aa.util.allatom_mask[seq]
    all_atom_names = np.array(rf2aa.chemical.aa2long[seq][:rf2aa.chemical.NHEAVYPROT], dtype=np.str_)
    all_atom_names = all_atom_names[all_atom_mask[:rf2aa.chemical.NHEAVYPROT]]
    have_atom_mask = atom_mask[res_i]
    have_atom_names = np.array(rf2aa.chemical.aa2long[seq][:rf2aa.chemical.NHEAVYPROT], dtype=np.str_)
    have_atom_names = have_atom_names[have_atom_mask[:rf2aa.chemical.NHEAVYPROT]]
    return [a for a in all_atom_names if a not in have_atom_names]

def fetch_connected_nodes(G, node, seen = None):
    if seen == None:
        seen = set([node])
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

def pop_mask(indep, pop, break_chirals=False):
    n_atoms = indep.is_sm.sum()
    assertpy.assert_that(len(indep.atom_frames)).is_equal_to(n_atoms)

    pop_sm = pop[indep.is_sm]
    # ASSERT REFERENCES CHECK OUT
    indep.atom_frames = indep.atom_frames[pop_sm]


    N     = pop.sum()
    pop2d = pop[None,:] * pop[:,None]

    indep.seq           = indep.seq[pop]
    indep.xyz           = indep.xyz[pop]
    indep.idx           = indep.idx[pop]
    indep.bond_feats    = indep.bond_feats[pop2d].reshape(N,N)
    indep.same_chain    = indep.same_chain[pop2d].reshape(N,N)
    indep.is_sm         = indep.is_sm[pop]
    indep.terminus_type = indep.terminus_type[pop]

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

def slice_indep(indep, pop):
    indep = copy.deepcopy(indep)
    cross_bonds = indep.bond_feats[pop][:, ~pop]
    # ic(cross_bonds)
    # assert_that(cross_bonds.sum()).is_equal_to(0)
    pop_mask(indep, pop)
    return indep, cross_bonds
 
def cat_indeps(indeps, same_chain):
    indep = Indep(None, None, None, None, None, None, None, None, None)
    indep.seq = torch.cat([i.seq for i in indeps])
    indep.xyz = torch.cat([i.xyz for i in indeps])
    indep.idx = torch.cat([i.idx for i in indeps])
    indep.bond_feats = torch.block_diag(*(i.bond_feats for i in indeps))
    indep.same_chain = same_chain
    indep.is_sm = torch.cat([i.is_sm for i in indeps])
    indep.terminus_type = torch.cat([i.terminus_type for i in indeps])
    L = 0
    all_chirals = []
    for i in indeps:
        chirals = i.chirals.clone()
        chirals[:,:-1] += L
        all_chirals.append(chirals.clone())
        L += i.length()
    indep.chirals = torch.cat(all_chirals)
    indep.atom_frames = torch.cat([i.atom_frames for i in indeps])
    return indep

def cat_indeps_same_chain(indeps):
    L = sum(i.length() for i in indeps)
    same_chain = torch.ones((L,L)).bool()
    return cat_indeps(indeps, same_chain)

def cat_indeps_separate_chains(indeps):
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
    is_sm_new = indep.is_sm[from_i]
    is_sm_old = indep.is_sm
    sm_i_old = is_sm_old.nonzero()[:, 0]
    absolute_from_relative_old = sm_i_old
    sm_i_new = is_sm_new.nonzero()[:, 0]
    n_sm = is_sm_old.sum()
    sm_i_relative = torch.arange(n_sm)
    a = torch.zeros(indep.length()).type(torch.LongTensor)
    a[:] = 9999
    a[is_sm_old] = sm_i_relative
    from_i_sm_relative = a[from_i[is_sm_new]]
    relative_from_absolute_new = torch.zeros(indep.length()).type(torch.LongTensor)
    a[:] = 9999
    relative_from_absolute_new[is_sm_new] = torch.arange(n_sm)


    atom_frames_relative_i = indep.atom_frames[:, :, 0]
    atom_frames_sm_absolute = atom_frames_relative_i + torch.arange(n_sm)[:, None]
    absolute_from_sm = indep.is_sm.nonzero()[:, 0]
    atom_frames_absolute = atom_frames_sm_absolute.apply_(lambda i: absolute_from_sm[i])
    atom_frames_absolute = atom_frames_absolute[from_i_sm_relative]
    atom_frames_absolute_i_new = atom_frames_absolute.apply_(lambda i: to_i[i])
    atom_frames_sm_absolute = atom_frames_absolute_i_new.apply_(lambda i: relative_from_absolute_new[i])
    atom_frames_relative_i_new = atom_frames_sm_absolute - torch.arange(n_sm)[:, None]

    indep.atom_frames[:,:,0] = atom_frames_relative_i_new
    indep.is_sm = is_sm_new

    
def centre(indep, is_diffused):
    xyz = indep.xyz
    #Centre unmasked structure at origin, as in training (to prevent information leak)
    if torch.sum(~is_diffused) != 0:
        motif_com=xyz[~is_diffused,1,:].mean(dim=0) # This is needed for one of the potentials
        xyz = xyz - motif_com
    elif torch.sum(~is_diffused) == 0:
        xyz = xyz - xyz[:,1,:].mean(dim=0)
    indep.xyz = xyz


def diffuse(conf, diffuser, indep, is_diffused, t):
    indep = copy.deepcopy(indep)
    indep.xyz = add_fake_frame_legs(indep.xyz, indep.is_sm)
    rigids_0 = du.rigid_frames_from_atom_14(indep.xyz)
    diffuser_out = diffuser.forward_marginal(
        rigids_0,
        t=t/conf.diffuser.T,
        diffuse_mask=is_diffused.float(),
        as_tensor_7=False
    )
    diffuser_out['rigids_0'] = rigids_0.to_tensor_7()[None]
    xT = all_atom.atom37_from_rigid(diffuser_out['rigids_t'])
    indep.xyz = xT[:,:14]
    return indep, diffuser_out

def forward(model, rfi, **kwargs):
    rfi_dict = dataclasses.asdict(rfi)
    return RFO(*model(**{**rfi_dict, **kwargs}))

def mask_indep(indep, is_diffused):
    indep.seq[is_diffused] = MASKINDEX

# def get_xyz_t_t2d(atom_frames, is_sm, xyz):
#     '''
#     Params:
#         xyz: [T, L, 36, 3]
#         is_sm: [L]
#         atom_frames: [F, 3, 2]
#     '''
#     assertpy.assert_that(is_sm.ndim).is_equal_to(3) # [F, 3, 2]
#     assertpy.assert_that(atom_frames.ndim).is_equal_to(3) # [F, 3, 2]
#     assertpy.assert_that(xyz.ndim).is_equal_to(3) # [L, n_atoms, 3]
#     L, = is_sm.shape
#     assertpy.assert_that(xyz.shape).is_equal_to((L, 36, 3))
#     t2d, mask_t_2d_remade = util.get_t2d(
#         xyz, is_sm, atom_frames)
#     t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
#     return


def self_cond_new(indep, rfi, rfo):
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
    xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
    ic(xyz_t[0].shape, rfi.atom_frames.shape)
    t2d, mask_t_2d_remade = util.get_t2d(
        xyz_t[0], indep.is_sm, rfi.atom_frames[0])
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    ic(rfi_sc.xyz_t.shape, rfi_sc.t2d.shape)
    rfi_sc.xyz_t[0,1] = xyz_t[0, 0, :, 1]
    rfi_sc.t2d[0, 1] = t2d[0, 0]
    return rfi_sc

def self_cond(indep, rfi, rfo):
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
    xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
    t2d, mask_t_2d_remade = util.get_t2d(
        xyz_t[0], indep.is_sm, rfi.atom_frames[0])
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    rfi_sc.xyz_t[0,1] = xyz_t[0, 0, :, 1]
    rfi_sc.t2d[0, 1] = t2d[0, 0]
    return rfi_sc

def diagnose_xyz(xyz):
    '''Returns a string describing where the coordinates are NaN'''
    has_ca = torch.isnan(xyz[..., 1, :]).any()
    has_backbone = torch.isnan(xyz[..., :3, :]).any()
    # has_heavy = torch.isnan(xyz[..., :3, :]).any()
    return f'diagnosis: nan-CA: {has_ca}    nan-BB: {has_backbone}'

class AtomizeResidues:
    def __init__(
        self,
        indep,
        input_str_mask # dict
        ) -> None:

        self.indep_initial_copy = copy.deepcopy(indep)
        self.indep = copy.deepcopy(indep)
        self.input_str_mask = input_str_mask
        self.input_seq_mask = input_str_mask

        # bookkeeping for de-atomization
        self.has_been_atomized = False
        self.atomized_res = [] # integer-coded AA identities of atomized residues
        self.atomized_res_idx = [] # tensor indices of atomized residues (0-indexed contiguous)
        self.atomized_idx = [] # PDB indices of atomized residues
    
    def featurize_atomized_residues(
        self,
        atom_mask, 
    ):
        """
        this function takes outputs of the RF2aa dataloader and the generated masks and refeaturizes the example where the 
        portion of the structure that is provided as context is treated as atoms instead of residues. the mask will be updated so 
        that only the tip atoms of the context residues are context and the rest is diffused
        """
        is_motif = self.input_str_mask
        seq_atomize_all,  xyz_atomize_all, frames_atomize_all, chirals_atomize_all, res_idx_atomize_all = \
            self.generate_atomize_protein_features(is_motif, atom_mask)
        if not res_idx_atomize_all: # no residues atomized, just return the inputs
            return
        
        # update msa feats, xyz, mask, frames and chirals
        total_L = self.update_rf_features_w_atomized_features(seq_atomize_all, \
                                                            xyz_atomize_all, \
                                                            frames_atomize_all, 
                                                            chirals_atomize_all
                                                            ) 

        pop = self.pop_protein_feats(res_idx_atomize_all, total_L)
        #handle diffusion specific things such as masking
        # self.construct_diffusion_masks(is_motif, is_atom_motif_all, pop)
    
    def generate_atomize_protein_features(self, is_motif, atom_mask):
        """
        given a motif, generate "atomized" features for the residues in that motif
        skips residues that have any unresolved atoms or neighbors with unresolved atoms
        """
        L = self.indep.bond_feats.shape[0]
        orig_L = L
        is_protein_motif = is_motif * ~self.indep.is_sm
        allatom_mask = rf2aa.util.allatom_mask.clone() # 1 if that atom index exists for that residue and 0 if it doesnt
        allatom_mask[:, 14:] = False # no Hs

        # make xyz the right dimensions for atomization
        xyz = torch.full((L, rf2aa.chemical.NTOTAL, 3), np.nan).float()
        xyz[:, :14] = self.indep.xyz[:, :14]

        aa2long_ = [[x.strip() if x is not None else None for x in y] for y in rf2aa.chemical.aa2long]
        # iterate through residue indices in the structure to atomize
        seq_atomize_all = []
        xyz_atomize_all = []
        frames_atomize_all = [self.indep.atom_frames]
        chirals_atomize_all = [self.indep.chirals]
        res_idx_atomize_all = [] # indices of residues that end up getting atomized (0-indexed, contiguous)
        res_atomize_all = [] # residues (integer-coded) that end up getting atomized 
        idx_atomize_all = [] # PDB indices of residues that end up getting atomized
        # track the res_idx and absolute index of the previous C to draw peptide bonds between contiguous residues
        prev_res_idx = -2
        prev_C_index = -2 
        for res_idx in is_protein_motif.nonzero():
            residue = self.indep.seq[res_idx] # number representing the residue in the token list, can be used to index aa2long
            # check if all the atoms in the residue are resolved, if not dont atomize
            if not torch.all(allatom_mask[residue]==atom_mask[res_idx]):
                raise Exception(f'not all atoms resolved for {residue} at 0-indexed position {res_idx}')
            N_term = res_idx == 0
            C_term = res_idx == self.indep.idx.shape[0]-1
            N_resolved = False
            C_resolved = False

            C_resolved = (not C_term) and self.indep.idx[res_idx+1]-self.indep.idx[res_idx] == 1
            N_resolved = (not N_term) and self.indep.idx[res_idx]-self.indep.idx[res_idx-1] == 1

            seq_atomize, _, xyz_atomize, _, frames_atomize, bond_feats_atomize, C_index, chirals_atomize = \
                            rf2aa.util.atomize_protein(res_idx, self.indep.seq[None], xyz, atom_mask, n_res_atomize=1)
            natoms = seq_atomize.shape[0]
            # update the chirals to be after all the other residues
            chirals_atomize[:, :-1] += L

            seq_atomize_all.append(seq_atomize)
            xyz_atomize_all.append(xyz_atomize)
            frames_atomize_all.append(frames_atomize)
            chirals_atomize_all.append(chirals_atomize)
            res_idx_atomize_all.append(res_idx)
            res_atomize_all.append(residue)
            idx_atomize_all.append(self.indep.idx[res_idx])

            # update bond_feats every iteration, update all other features at the end 
            bond_feats_new = torch.zeros((L+natoms, L+natoms))
            bond_feats_new[:L, :L] = self.indep.bond_feats
            bond_feats_new[L:, L:] = bond_feats_atomize
            # add bond between protein and atomized N
            if N_resolved:
                bond_feats_new[res_idx-1, L] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L, res_idx-1] = 6 # protein (backbone)-atom bond 
            # add bond between protein and C, assumes every residue is being atomized one at a time (eg n_res_atomize=1)
            if C_resolved:
                bond_feats_new[res_idx+1, L+int(C_index.numpy())] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L+int(C_index.numpy()), res_idx+1] = 6 # protein (backbone)-atom bond 
            # handle drawing peptide bond between contiguous atomized residues
            if self.indep.idx[res_idx]-self.indep.idx[prev_res_idx] == 1 and prev_res_idx > -1:
                bond_feats_new[prev_C_index, L] = 1 # single bond
                bond_feats_new[L, prev_C_index] = 1 # single bond
            prev_res_idx = res_idx
            prev_C_index =  L+int(C_index.numpy())
            #update same_chain every iteration
            same_chain_new = torch.zeros((L+natoms, L+natoms))
            same_chain_new[:L, :L] = self.indep.same_chain
            residues_in_prot_chain = self.indep.same_chain[res_idx].squeeze().nonzero()

            same_chain_new[L:, residues_in_prot_chain] = 1
            same_chain_new[residues_in_prot_chain, L:] = 1
            same_chain_new[L:, L:] = 1

            self.indep.bond_feats = bond_feats_new
            self.indep.same_chain = same_chain_new
            L = self.indep.bond_feats.shape[0]

        # save atomized position info needed for deatomization
        self.atomized_res = res_atomize_all
        self.atomized_res_idx = res_idx_atomize_all
        self.atomized_idx = idx_atomize_all

        return seq_atomize_all, xyz_atomize_all, frames_atomize_all, chirals_atomize_all, res_idx_atomize_all

    def update_rf_features_w_atomized_features(self, \
                                        seq_atomize_all, \
                                        xyz_atomize_all, \
                                        frames_atomize_all, \
                                        chirals_atomize_all, 
                                        ):
        """
        adds the msa, xyz, frame and chiral features from the atomized regions to the rosettafold input tensors
        """
        # Handle MSA feature updates
        seq_atomize_all = torch.cat(seq_atomize_all)
        
        atomize_L = seq_atomize_all.shape[0]
        self.indep.seq = torch.cat((self.indep.seq, seq_atomize_all), dim=0)

        # handle coordinates, need to handle permutation symmetry
        orig_L, natoms = self.indep.xyz.shape[:2]
        total_L = orig_L + atomize_L
        xyz_atomize_all = rf2aa.util.cartprodcat(xyz_atomize_all)
        N_symmetry = xyz_atomize_all.shape[0]
        xyz = torch.full((N_symmetry, total_L, NTOTAL, 3), np.nan).float()
        xyz[:, :orig_L, :natoms, :] = self.indep.xyz.expand(N_symmetry, orig_L, natoms, 3)
        xyz[:, orig_L:, 1, :] = xyz_atomize_all
        xyz[:, orig_L:, :3, :] += rf2aa.chemical.INIT_CRDS[:3]
        #ignoring permutation symmetry for now, network should learn permutations at low T
        self.indep.xyz = xyz[0]
        
        #handle idx_pdb 
        last_res = self.indep.idx[-1]
        idx_atomize = torch.arange(atomize_L) + last_res
        self.indep.idx = torch.cat((self.indep.idx, idx_atomize))
        
        # handle sm specific features- atom_frames, chirals
        self.indep.atom_frames = torch.cat(frames_atomize_all)
        self.indep.chirals = torch.cat(chirals_atomize_all)
        self.indep.terminus_type = torch.cat((self.indep.terminus_type, torch.zeros(atomize_L).long()))
        return total_L

    def pop_protein_feats(self, res_idx_atomize_all, total_L):
        """
        after adding the atom information into the tensors, remove the associated protein sequence and template information
        """
        is_atomized_residue = torch.tensor(res_idx_atomize_all) # indices of residues that have been atomized and need other feats removed
        pop = torch.ones((total_L))
        pop[is_atomized_residue] = 0
        pop = pop.bool()
        self.indep.seq         = self.indep.seq[pop]
        self.indep.xyz         = self.indep.xyz[pop]
        self.indep.idx     = self.indep.idx[pop]
        self.indep.same_chain  = self.indep.same_chain[pop][:, pop]
        self.indep.bond_feats = self.indep.bond_feats[pop][:, pop].long()
        n_shift = (~pop).cumsum(dim=0)
        chiral_indices = self.indep.chirals[:,:-1]
        chiral_shift = n_shift[chiral_indices.long()]
        self.indep.chirals[:,:-1] = chiral_indices - chiral_shift
        self.indep.terminus_type = self.indep.terminus_type[pop]
        return pop

    def get_atomized_res_idx_from_res(self, expect_H=False):

        atom_idx_by_res = {}
        
        N_atoms = rf2aa.chemical.NHEAVYPROT
        if expect_H:
            N_atoms = rf2aa.chemical.NTOTAL
        atomized_mask = rf2aa.util.allatom_mask[torch.tensor(self.atomized_res, dtype=int)][:,:N_atoms]
        atomized_res_natoms = atomized_mask.sum(dim=1)
        N_atomized_res = len(self.atomized_res)
        N_atomized_atoms = sum(atomized_res_natoms)

        L = self.indep.seq.shape[0] # length of atomized features
        L_base = L - N_atomized_atoms # length of non-atomized region in atomized features
        L_new = L_base + N_atomized_res # length of deatomized features

        idx_nonatomized = np.setdiff1d(np.arange(L_new), torch.tensor(self.atomized_res_idx))
        # map residue indices in atomized features to indices in deatomized features
        return dict(zip(idx_nonatomized, np.arange(L_base)))


    def get_atom_idx_by_res(self, expect_H=False):

        atom_idx_by_res = {}
        
        N_atoms = rf2aa.chemical.NHEAVYPROT
        if expect_H:
            N_atoms = rf2aa.chemical.NTOTAL

        atomized_mask = rf2aa.util.allatom_mask[torch.tensor(self.atomized_res, dtype=int)][:,:N_atoms]
        atomized_res_natoms = atomized_mask.sum(dim=1)
        N_atomized_res = len(self.atomized_res)
        assertpy.assert_that(len(self.atomized_res)).is_equal_to(len(self.atomized_res_idx))
        N_atomized_atoms = sum(atomized_res_natoms)

        L = self.indep.seq.shape[0] # length of atomized features
        L_base = L - N_atomized_atoms # length of non-atomized region in atomized features
        L_new = L_base + N_atomized_res # length of deatomized features

        # indices of non-atomized positions in deatomized features
        idx_nonatomized = np.setdiff1d(np.arange(L_new), torch.tensor(self.atomized_res_idx))
        assert(len(idx_nonatomized)==L_base)

        # deatomize the previously atomized residues
        for i in range(N_atomized_res):
            res_idx = self.atomized_res_idx[i]
            # assumes atomized atoms were in standard order
            atom_idx_range = L_base + sum(atomized_res_natoms[:i]) + np.arange(atomized_res_natoms[i])
            atom_idx_by_res[res_idx.item()] = atom_idx_range
            # xyz[res_idx, :len(atom_idx_range)] = xyz_pred[atom_idx_range,1]
        return atom_idx_by_res


    def get_deatomized_features(self, seq_pred, xyz_pred, expect_H=False):
        """Converts previously atomized residues back into residue representation and
        returns features for PDB output. Does not update instance variables.

        NOTE: This only generates features needed for output, NOT for input into RF (i.e.
        chirals, frames, etc, are ignored here).

        Args:
            seq_pred: torch.Tensor (L, NAATOKENS), Sequence (1-hot) output by the network
            xyz_pred: torch.Tensor (L, N_atoms, 3), Coordinates output by the network,
            without the batch dimension. Should be the same shape as the input coordinates
            (i.e. `self.indep.xyz`)

        Returns:
            Deatomized sequence (1-hot), coordinates, PDB indices, bond & same_chain features.
            seq: torch.Tensor (L_new, NAATOKENS)
            xyz: torch.Tensor (L_new, N_atoms, 3)
            idx: torch.Tensor (L_new,)
            bond_feats: torch.Tensor (L_new, L_new)
            same_chain: torch.Tensor (L_new, L_New)
        """
        N_atoms = rf2aa.chemical.NHEAVYPROT
        if expect_H:
            N_atoms = rf2aa.chemical.NTOTAL
        assert(seq_pred.shape[0] == self.indep.seq.shape[0])
        
        # no atomization was done, just return unmodified features
        if len(self.atomized_res)==0:
            return seq_pred, xyz_pred, self.indep.idx, self.indep.bond_feats, self.indep.same_chain

        # assumes all heavy atoms are present
        atomized_mask = rf2aa.util.allatom_mask[torch.tensor(self.atomized_res)][:,:N_atoms]
        atomized_res_natoms = atomized_mask.sum(dim=1)
        N_atomized_res = len(self.atomized_res)
        N_atomized_atoms = sum(atomized_res_natoms)

        L = seq_pred.shape[0] # length of atomized features
        L_base = L - N_atomized_atoms # length of non-atomized region in atomized features
        L_new = L_base + N_atomized_res # length of deatomized features

        # deatomized features
        seq = torch.full((L_new,), UNKINDEX).long()
        seq = torch.nn.functional.one_hot(seq, num_classes=NAATOKENS) # (L_new, NAATOKENS)
        xyz = torch.full((L_new, rf2aa.chemical.NTOTAL, 3), np.nan).float()
        idx = torch.full((L_new,), np.nan).long()
        bond_feats = torch.zeros((L_new, L_new)).long()
        same_chain = torch.zeros((L_new, L_new)).long()

        # indices of non-atomized positions in deatomized features
        idx_nonatomized = np.setdiff1d(np.arange(L_new), torch.tensor(self.atomized_res_idx))
        assert(len(idx_nonatomized)==L_base)

        # map residue indices in atomized features to indices in deatomized features
        idxmap = dict(zip(np.arange(L_base), idx_nonatomized))

        # copy over features of positions that were never atomized
        seq[idx_nonatomized] = seq_pred[:L_base].long()
        xyz[idx_nonatomized] = xyz_pred[:L_base]
        idx[idx_nonatomized] = self.indep.idx[:L_base]
        for i_src, i_dest in enumerate(idx_nonatomized):
            bond_feats[i_dest,idx_nonatomized] = self.indep.bond_feats[i_src, :L_base]
            same_chain[i_dest,idx_nonatomized] = self.indep.same_chain[i_src, :L_base].long()

        # residue indices of residue-atom bonds in atomized features
        idx_atomize_bonds = torch.where(self.indep.bond_feats==6)

        # deatomize the previously atomized residues
        for i in range(N_atomized_res):
            res_idx = self.atomized_res_idx[i] 
            seq[res_idx, self.atomized_res[i]] = 1
            idx[res_idx] = self.atomized_idx[i]

            # assumes atomized atoms were in standard order
            atom_idx_range = L_base + sum(atomized_res_natoms[:i]) + np.arange(atomized_res_natoms[i])
            xyz[res_idx, :len(atom_idx_range)] = xyz_pred[atom_idx_range,1] 

            # bonds between atoms of this atomized residue and nonatomized residues
            for i,j in zip(*idx_atomize_bonds):
                if i in atom_idx_range: # assume bond features are symmetrical
                    bond_feats[res_idx, idxmap[int(j)]] = 5
                    bond_feats[idxmap[int(j)], res_idx] = 5

        # assign atomized residues to whichever chain their atomized atoms were bonded to
        same_chain = rf2aa.util.same_chain_from_bond_feats(bond_feats | same_chain)

        return seq, xyz, idx, bond_feats, same_chain

    @staticmethod
    def choose_random_atom_motif(natoms, p=0.5):
        """
        selects each atom to be in the motif with a probability p 
        """
        return torch.rand((natoms)) > p

    def choose_sm_contact_motif(self, xyz_atomize):
        """
        chooses atoms to be the motif based on the atoms that are closest to the small molecule
        """
        dist = torch.cdist(self.indep.xyz[self.indep.is_sm, 1, :], xyz_atomize)
        closest_sm_atoms = torch.min(dist, dim=-2)[0][0] # min returns a tuple of values and indices, we want the values
        contacts = closest_sm_atoms < 4
        # if no atoms are closer than 4 angstroms, choose the closest three atoms
        if torch.all(contacts == 0):
            min_indices = torch.argsort(closest_sm_atoms)[:3]
            contacts[min_indices] = 1
        return contacts
    
    @staticmethod
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

    def return_input_tensors(self):
        return self.indep, self.input_str_mask, self.input_seq_mask

GP_BOND = 7
BACKBONE_BOND = 5
def make_guideposts(indep, is_motif):
    pre_gp_len = indep.length()
    n_motif = is_motif.sum()
    chains = indep.chains()
    motif_chains = indep.chains()[is_motif]
    indep_motif, cross_bonds = slice_indep(indep, is_motif)
    # Break peptide bonds
    indep_motif.bond_feats *= (indep_motif.bond_feats != BACKBONE_BOND)
    gp_i = range(pre_gp_len, pre_gp_len + n_motif)
    CHAIN_GAP = 33
    indep_motif.idx = torch.arange(n_motif) * CHAIN_GAP + CHAIN_GAP + indep.idx.max()
    indep_cat = cat_indeps_separate_chains((indep, indep_motif))
    chains_cat =  np.concatenate((chains, motif_chains))
    indep_cat.same_chain = same_chain_from_chain_letters(chains_cat)
    gp_to_ptn_idx0 = {i:j for i,j in zip(gp_i, is_motif.nonzero()[:,0].tolist())}

    is_gp = torch.zeros(indep_cat.length()).bool()
    is_gp[-n_motif:] = True
    is_inter_gp = is_gp[None, :] != is_gp[:, None]
    has_sm = indep_cat.is_sm[None, :] + indep_cat.is_sm[:, None]
    indep_cat.bond_feats[is_inter_gp * ~has_sm] = GP_BOND
    return indep_cat, gp_to_ptn_idx0

def transform_indep(indep, is_res_str_shown, is_atom_str_shown, use_guideposts, guidepost_placement='anywhere', guidepost_bonds=True, metadata=None):
    indep = copy.deepcopy(indep)
    use_atomize = is_atom_str_shown is not None
    # use_atomize = is_atom_str_shown is not None and len(is_atom_str_shown) > 0
    is_diffused = is_masked_seq = ~is_res_str_shown
    atomizer = None
    gp_to_ptn_idx0 = None

    if use_guideposts:
        mask_gp = is_res_str_shown.clone()
        mask_gp[indep.is_sm] = False
        if use_atomize:
            atomized_residues = list(is_atom_str_shown.keys())
            mask_gp[atomized_residues] = True
        if mask_gp.sum() == 0:
            is_masked_seq[indep.is_sm] = False
            return indep, is_diffused, is_masked_seq, atomizer, {}


        indep, gp_to_ptn_idx0 = make_guideposts(indep, mask_gp)
        is_diffused[list(gp_to_ptn_idx0.values())] = True
        n_gp = len(gp_to_ptn_idx0)
        is_diffused = torch.cat((is_diffused, torch.zeros((n_gp,)).bool()))
        if use_atomize:
            gp_from_ptn_idx0 = {v:k for k,v in gp_to_ptn_idx0.items()}
            is_atom_str_shown = {gp_from_ptn_idx0[k]: v for k,v in is_atom_str_shown.items()}
            # Remove redundancy
            is_diffused[list(is_atom_str_shown.keys())] = True
            cov_resis = [gp_from_ptn_idx0[res_i] for (res_i, _), _, _ in metadata['covale_bonds']]
            for i, (a, b, t) in enumerate(metadata['covale_bonds']):
                res_i, atom_name = a
                assertpy.assert_that(gp_from_ptn_idx0).described_as('residues participating in covalent bonds to small molecules must be made into guideposts').contains(res_i)
                res_i = gp_from_ptn_idx0[res_i]
                assertpy.assert_that(is_atom_str_shown).described_as('residues participating in covalent bonds to small molecules must be atomized').contains(res_i)
                metadata['covale_bonds'][i] = ((res_i, atom_name), b, t)

        is_masked_seq = is_diffused.clone()
    if len(metadata['covale_bonds']):
        assert use_atomize
    if use_atomize:
        is_covale_ligand = indep.type() == TYPE_ATOMIZED_COV
        is_ligand = indep.is_sm
        # is_diffused[indep.is_sm] = False
        is_atom_str_shown = {k.item() if hasattr(k, 'item') else k :v for k,v in is_atom_str_shown.items()}
        indep, is_diffused, is_masked_seq, atomizer = atomize.atomize_and_mask(indep, ~is_diffused, is_atom_str_shown)
        # TODO: Handle sequence masking more elegantly
        is_masked_seq[indep.is_sm] = False
        assertpy.assert_that(indep.is_sm.sum()).is_equal_to(indep.atom_frames.shape[0])
        ligand_idx = atomize.atomized_indices_res(atomizer, is_ligand)
        covale_ligand_idx = atomize.atomized_indices_res(atomizer, is_covale_ligand)
        if use_guideposts:
            # HACK: use is_masked_seq as gp_idx0.
            is_gp = ~is_masked_seq
            is_inter_gp = is_gp[None, :] != is_gp[:, None]
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
        indep.bond_feats = indep.bond_feats * ~(indep.bond_feats == GP_BOND)
    
    # Add back in bond feats
    atom_names_by_res = OrderedDict()
    for a, _, _ in metadata['covale_bonds']:
        res_i, atom_name = a
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
        G = rf2aa.util.get_nxgraph(obmol)
        covale.atom_frames = rf2aa.util.get_atom_frames(covale.seq, G)
        covale.chirals = rf2aa.kinematics.get_chirals(obmol, covale.xyz[:, 1])
    assertpy.assert_that(len(is_diffused)).is_equal_to(indep.length())
    return indep, is_diffused, is_masked_seq, atomizer, gp_to_ptn_idx0


def hetatm_names(pdb):
    d = defaultdict(list)
    with open(pdb) as f:
        for line in f.readlines():
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

def rename_ligand_atoms(ref_fn, out_fn):
    """Copies names of ligand residue and ligand heavy atoms from input pdb
    into output (design) pdb."""

    ref_atom_names_by_lig = hetatm_names(ref_fn)
    ref_atom_names_by_lig = without_H(ref_atom_names_by_lig)
    with open(out_fn) as f:
        lines = [line.strip() for line in f.readlines()]

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

    with open(out_fn,'w') as f:
        for line in lines2:
            print(line, file=f)

def randomly_rotate_frames(xyz):
    L, _, _ = xyz.shape
    R_rand = rotation_conversions.random_rotations(L, dtype=xyz.dtype)
    frame_origins = xyz[:,1:2,:]
    xyz_centered = xyz - frame_origins
    rotated = torch.einsum('lab,lib->...lia', R_rand, xyz_centered)
    rotated += frame_origins
    return rotated


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
def open_indep(indep, is_open):
    assertpy.assert_that(indep.length()).is_equal_to(len(is_open))
    indep_closed, _ = slice_indep(indep, ~is_open)
    indep_open, _ = slice_indep(indep, is_open)
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
