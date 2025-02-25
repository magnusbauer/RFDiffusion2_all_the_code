from __future__ import annotations
import random
import logging
import sys
from collections.abc import Mapping

import torch
import scipy.stats
from rf_diffusion import kinematics
import numpy as np
from icecream import ic
import rf2aa.util
import networkx as nx
nx.from_numpy_matrix = nx.from_numpy_array
from functools import wraps
import assertpy
from collections import OrderedDict
import rf_diffusion.aa_model as aa_model
from functools import partial

from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion import error
from rf_diffusion import tip_atoms
import rf_diffusion.ppi as ppi
from typing import Literal, Callable


import rf_diffusion.nucleic_compatibility_utils as nucl_utils

logger = logging.getLogger(__name__)

class InvalidMaskException(Exception):
    pass

def get_invalid_mask(*args, **kwargs):
    '''
    This is for a unit test
    '''
    raise InvalidMaskException('This mask always throws InvalidMaskException')

def make_covale_compatible(get_mask):
    '''
    This decorator is used to make a mask generator compatible with covalently modified residues.

    Covalently modified residues must be atomized.
    
    If a covalently modified residue is NOT motif:
        - It is atomized but not made motif
    If a covalently modified residue IS motif
        - It is atomized and all of its atoms are made motif.
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        covale_res_i = torch.tensor([res_i for (res_i, atom_name), lig_i, _ in indep.metadata['covale_bonds']]).tolist()
        is_atom_motif = is_atom_motif or {}
        for res_i in covale_res_i:
            if res_i not in is_atom_motif:
                is_atom_motif[res_i] = []
        motif_idx = is_motif.nonzero()[:,0].numpy()
        covalently_modified_res_motif = set(motif_idx).intersection(set(covale_res_i))
        for res_i in covalently_modified_res_motif:
            seq_token = indep.seq[res_i]
            atom_names = ChemData().aa2long[seq_token][:ChemData().NHEAVY]
            atom_names = [a if a is None else a.strip() for a in atom_names]
            atom_names = np.array(atom_names, dtype=np.str_)
            n_atoms_expected = (atom_names != 'None').sum()
            n_atoms_occupied = atom_mask[res_i].sum()
            if n_atoms_expected != n_atoms_occupied:
                # TODO: Make this an expected exception type that can be caught by the fallback dataloader
                # for a less scary warning.
                raise Exception(f'residue {res_i} should have {n_atoms_expected} but has {n_atoms_occupied}')
            atom_mask_i = atom_mask[res_i].numpy()
            atom_names = atom_names[atom_mask_i]
            is_atom_motif[res_i] = atom_names.tolist()
            is_motif[res_i] = False
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, **ret)
    return out_get_mask

#####################################
# Misc functions for mask generation
#####################################

def get_masks(L, min_length, max_length, min_flank, max_flank):
    """
    Makes a random contiguous mask, with (or without) flanking residues masked.
    """
    flank_width = random.randint(min_flank, max_flank)
    max_length = min([max_length, L-2*flank_width - 20]) #require at least 20 visible residues in any masking regime.
    central_width = random.randint(min_length,max_length)
    assert central_width > min_length - 1
    assert max_length > min_length
    
    start = random.randint(flank_width,L-flank_width-central_width-1)
    return (start,start+central_width),flank_width


def get_diffusion_pos(L,min_length, max_length=None):
    """
    Random contiguous mask generation to denote which residues are being diffused 
    and which are not. 

    TODO: This does not support multi-chain diffusion training at the moment 

    Returns:

        start,end : indices between which residues are allowed to be diffused. 
                    Otherwise, residues are held fixed and revealed 
    """
    if (max_length is None) or (max_length > L):
        max_length = L 

    assert min_length <= max_length 

    # choose a length to crop 
    chosen_length = np.random.randint(min_length, max_length)

    # choose a start position - between 0 (inclusive) and L-chosen_length (exclusive)
    start_idx = random.randint(0, L-chosen_length)
    end_idx   = start_idx + chosen_length

    return start_idx, end_idx 

def get_cb_distogram(xyz):
    Cb = kinematics.get_Cb(xyz)
    dist = kinematics.get_pair_dist(Cb, Cb)
    return dist

def get_contacts(xyz, xyz_less_than=5, seq_dist_greater_than=10):
    L = xyz.shape[0]
    dist = get_cb_distogram(xyz)

    is_close_xyz = dist < xyz_less_than

    seq_dist = torch.abs(torch.arange(L)[None] - torch.arange(L)[:,None])
    is_far_seq = torch.abs(seq_dist) > seq_dist_greater_than

    contacts = is_far_seq * is_close_xyz
    return contacts

def sample_around_contact(L, indices, len_low, len_high):
    diffusion_mask = torch.zeros(L).bool()
    for anchor in indices:
        mask_length = int(np.floor(random.uniform(len_low, len_high)))
        l = anchor - mask_length // 2
        r = anchor + (mask_length - mask_length//2)
        l = max(0, l)
        r = min(r, L)
        diffusion_mask[l:r] = True
    return dict(is_motif=diffusion_mask)


def _get_double_contact(xyz, low_prop, high_prop, broken_prop, xyz_less_than=5, seq_dist_greater_than=25, len_low=5, len_high=10):
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop)
    contact_idxs = contacts.nonzero()
    contact_idx = np.random.choice(np.arange(len(contact_idxs)))
    indices = contact_idxs[contact_idx]
    L = xyz.shape[0]
    return sample_around_contact(L, indices, len_low, len_high)

def find_third_contact(contacts):
    contact_idxs = contacts.nonzero()
    contact_idxs = contact_idxs[torch.randperm(len(contact_idxs))]
    for i,j in contact_idxs:
        if j < i:
            continue
        K = (contacts[i,:] * contacts[j,:]).nonzero()
        if len(K):
            K = K[torch.randperm(len(K))]
            for k in K:
                return torch.tensor([i,j,k])
    return None

def _get_sm_contacts(
        indep, atom_mask,
    d_beyond_closest = 1.5,
    n_beyond_closest = 2,
    n_sample_low = 1,
    n_sample_high = 8, **kwargs):

    xyz, is_sm = indep.xyz, indep.is_sm

    assert len(xyz.shape) == 3
    assert is_sm.any()

    L = xyz.shape[0]
    L_prot = (~is_sm).sum()
    n_sample = np.random.randint(n_sample_low, n_sample_high)

    crds = torch.clone(xyz)
    crds[~atom_mask] = torch.nan
    prot_crds = crds[~is_sm]
    sm_crds = crds[is_sm]
    dist = (prot_crds[:, None] - sm_crds[ None]).pow(2).sum(dim=-1).sqrt()
    dist = dist.nan_to_num(99999)
    dist = dist.min(dim=-1)[0].min(dim=-1)[0]
    dist_cutoff = dist.min() + d_beyond_closest

    is_sampled = torch.zeros(L_prot).bool()
    _, closest_idx = torch.topk(dist, n_sample + n_beyond_closest, largest=False)
    is_sampled[closest_idx] = True
    is_sampled[dist < dist_cutoff] = True

    is_sampled_het = torch.zeros(L).bool()
    is_sampled_het[~is_sm] = is_sampled

    candidate_indices = is_sampled_het.nonzero().flatten()
    indices = np.random.choice(candidate_indices, n_sample, replace=False)
    is_motif = torch.zeros(L).bool()
    is_motif[is_sm] = True
    is_motif[indices] = True

    # Verification
    picked = crds[is_motif]
    dist_conf = (picked[:, None] - sm_crds[ None]).pow(2).sum(dim=-1).sqrt()
    dist_conf = dist_conf.nan_to_num(9999)
    #ic(is_motif, n_sample, picked_distances, dist_cutoff, indices)

    return dict(is_motif=is_motif, is_atom_motif={})

def get_triple_contact_atomize(*args, **kwargs):
    raise Exception('not implemented')

# TODO: fix
@make_covale_compatible
def _get_closest_tip_atoms(indep, atom_mask,
    d_beyond_closest = 1.0,
    n_beyond_closest = 1,
    n_sample_low = 1,
    n_sample_high = 5, **kwargs):

    assert len(indep.xyz.shape) == 3
    assert indep.is_sm.any()

    L = indep.length()
    L_prot = (~indep.is_sm).sum()
    n_sample = np.random.randint(n_sample_low, n_sample_high)

    crds = torch.clone(indep.xyz)
    crds[~atom_mask] = torch.nan
    prot_crds = crds[~indep.is_sm]
    sm_crds = crds[indep.is_sm][:, 1]
    dist_res_sidechain_ligand = (prot_crds[:,:, None,...] - sm_crds[ None,None,...]).pow(2).sum(dim=-1).sqrt()
    dist_res_sidechain_ligand = dist_res_sidechain_ligand.nan_to_num(9999)
    dist_res_sidechain = dist_res_sidechain_ligand.min(dim=-1)[0]
    dist = dist_res_sidechain.min(dim=-1)[0]
    is_valid_for_atomization = indep.is_valid_for_atomization(atom_mask)[~indep.is_sm]
    if not is_valid_for_atomization.any():
        ic('No valid residues for atomization, falling back to unconditional generation')
        return dict(is_motif=torch.zeros(L).bool(), is_atom_motif=None)
    dist[~is_valid_for_atomization] = 9999

    # Calculate distance cutoff
    dist_cutoff = dist.min() + d_beyond_closest
    is_sampled = torch.zeros(L_prot).bool()
    _, closest_idx = torch.topk(dist, n_sample + n_beyond_closest, largest=False)
    is_sampled[closest_idx] = True
    is_sampled[dist < dist_cutoff] = True
    is_sampled[~is_valid_for_atomization] = False
    #ic(f'After removing residue contacts with unresolved heavy atoms: {n_contacts_before} --> {n_contacts_after}')

    is_sampled_het = torch.zeros(L).bool()
    is_sampled_het[~indep.is_sm] = is_sampled
    candidate_indices = is_sampled_het.nonzero().flatten()

    n_sample = min(n_sample, len(candidate_indices))
    # print(f'choosing {n_sample} out of {len(candidate_indices)}')
    indices = np.random.choice(candidate_indices, n_sample, replace=False)

    # Verification for debugging
    if False:
        picked = crds[indices]
        dist_conf = (picked[:, None] - sm_crds[ None]).pow(2).sum(dim=-1).sqrt()
        dist_conf = dist_conf.nan_to_num(9999)
        picked_distances = dist_conf.min(-1)[0].min(-1)[0]
        ic(picked_distances, dist_cutoff, indices)

    is_atom_diffused = {}
    sm_prot_transition_types = (indep.is_sm[1:].int() - indep.is_sm[:-1].int()).unique().tolist()
    # If the ligands do not come in a single block after the protein, using dist_sc_to_sm will provide incorrect indices,
    assertpy.assert_that(sm_prot_transition_types).is_equal_to([0,1])
    # prot_by_het = torch.full((indep.length(),), torch.nan)
    # prot_by_het[~indep.is_sm] = torch.arange((~indep.is_sm).sum())
    # torch.nonzero(~indep.is_sm).flatten()
    for het_i in indices:
        # prot_i = prot_by_het[het_i]
        prot_i = het_i
        closest_atom = torch.argmin(dist_res_sidechain[prot_i]).item()
        n_bonds = np.random.randint(1, 3)
        is_atom_diffused[het_i] = get_atom_names_within_n_bonds(indep.seq[het_i], closest_atom, n_bonds)
    is_motif = torch.zeros(L).bool()
    is_motif[indep.is_sm] = True

    return dict(is_motif=is_motif, is_atom_motif=is_atom_diffused)


def get_atom_names_within_n_bonds(res, source_node, n_bonds):
    bond_feats = get_residue_bond_feats(res)
    bond_graph = nx.from_numpy_matrix(bond_feats.numpy())
    paths = nx.single_source_shortest_path_length(bond_graph, source=source_node,cutoff=n_bonds)
    atoms_within_n_bonds = paths.keys()
    atom_names = [ChemData().aa2long[res][i] for i in atoms_within_n_bonds]
    return atom_names

def tip_crd(indep, i):
    '''Returns the internal index of the tip atom of residue index i'''
    tip_idx_within_res = tip_idx(indep, i)
    return indep.xyz[i, tip_idx_within_res]

def tip_idx(indep, i):
    '''Returns the coordinates of the tip atom of residue index i'''
    aa = indep.seq[i]
    tip_atom_name = ChemData().aa2tip[aa].strip()
    tip_idx_within_res = next(i for i, atom_name in enumerate(ChemData().aa2long[aa]) if atom_name.strip() == tip_atom_name)
    return tip_idx_within_res

def _get_tip_gaussian_mask(indep, atom_mask, *args, std_dev=8, show_tip=False, **kwargs):
    '''
    Params:
        indep: aa_model.Indep, a description of a protein complex
        atom_mask: [L, 36] mask of whether an atom is resolved in indep
        std_dev: standard deviation of the multivariate gaussian (see below)
        *args: ignored, necessary to match masker function signature
        **kwargs: ignored, necessary to match masker function signature
    Returns:
        is_motif: binary mask that is True where a non-atomized residue is motif
        is_atom_motif: dictionary mapping residue indices to the atom names which are motif
    
    This masking function provides a few partial sidechains as motif.

    The protocol for selecting those sidechains is as follows:
        1. Find all atomizable residue
        2. Select one at random, call it origin
        3. Sample 1-6 atomizable residues with probabilities given by evaluation of a
            multivariate gaussian centered at origin at the tips of the atomizable residues
        4. Select a random atom in each residue, weighted towards selecting the tip
        5. Expand the mask starting from each selected atom by traversing 1-3 bonds within the residue.
    '''
    ic(indep.xyz.shape)
    ic(atom_mask.shape)
    assert not indep.is_sm.any()
    is_valid_for_atomization = indep.has_heavy_atoms_and_seq(atom_mask)
    if not is_valid_for_atomization.any():
        ic('No valid residues for atomization in tip_gaussian_mask, falling back to unconditional generation')
        is_motif = torch.zeros(indep.length()).bool()
        is_motif[indep.is_sm] = True
        return dict(is_motif=is_motif, is_atom_motif=None)
    valid_idx = is_valid_for_atomization.nonzero()[:,0]
    
    origin_i = np.random.choice(valid_idx, 1)[0]
    origin_tip_crd = tip_crd(indep, origin_i)
    tip_crds = [tip_crd(indep, i) for i in valid_idx]
    tip_crds = np.stack(tip_crds, axis=0)
    gaussian = scipy.stats.multivariate_normal(origin_tip_crd, std_dev)
    probs = gaussian.pdf(tip_crds)
    probs /= probs.sum()
    n_atomize = random.randint(1, 6)
    n_atomize = min(n_atomize, len(valid_idx))
    atomize_i = np.random.choice(valid_idx, n_atomize, p=probs, replace=False)

    is_atom_motif = {}
    for i in atomize_i:
        atom_crds = indep.xyz[i][atom_mask[i]]
        closest_atom_i = torch.argmin(torch.norm(atom_crds - origin_tip_crd), axis=-1)
        n_atoms = len(atom_crds)
        prob_non_closest = 0.5 / (n_atoms-1)
        probs = np.full((n_atoms,), prob_non_closest)
        probs[closest_atom_i] = 0.5
        p_tip_only = 0.5
        if np.random.rand() < p_tip_only:
            probs[:4] = 1e-6
        probs = probs.astype('float64')
        probs /= probs.sum()
        if show_tip:
            seed_atom = n_atoms - 1
        else:
            seed_atom = np.random.choice(np.arange(n_atoms), 1, p=probs)[0]
        n_bonds = np.random.randint(1, 3)
        atom_names = get_atom_names_within_n_bonds(indep.seq[i], seed_atom, n_bonds)
        assertpy.assert_that(atom_names).does_not_contain(None)
        is_atom_motif[i] = atom_names

    is_motif = torch.zeros(indep.length()).bool()
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

def _get_tip_mask(indep, atom_mask, *args,
                  n_atomize_min=1,
                  n_atomize_max=8,
                  p_tip=0.8,
                  bond_inclusion_p=0.5,
                  unconditional=False,
                  can_be_tip=None,
                   **kwargs):
    """
    Generate a tip atom `motif` for protein residues.

    This function selects residues for atomization and creates a mask specifying which atoms
    should be included in the motif for the case of atomized motifs.

    Args:
        indep: An object representing the independent variables of the structure.
        atom_mask (torch.Tensor): A mask indicating which atoms are present in each residue, [L, N_heavy(23)].
        *args: ignored
        n_atomize_min (int): Minimum number of residues to atomize. Default: 1.
        n_atomize_max (int): Maximum number of residues to atomize. Default: 8.
        p_tip (float): Probability of selecting the tip atom as the seed atom. Default: 0.8.
        bond_inclusion_p (float): Probability of including an additional bond in the motif. Default: 0.5.
            The `n_bonds` hop-distance around the seed atom that are included in the atom motif fragment 
            is sampled from a geometric distribution with parameter `1-bond_inclusion_p`.
        unconditional (bool): If True, generate an unconditional mask (empty atom list for each residue). Default: False.
        can_be_tip (torch.Tensor, optional): Boolean mask indicating which residues can be selected for tip atom masking.
        **kwargs: ignored

    Returns:
        dict: A dictionary containing:
            - 'is_motif' (torch.Tensor): Boolean tensor indicating motif tokens.
            - 'is_atom_motif' (dict): Dictionary mapping residue indices to lists of atom names in the motif.

    Notes:
        - If no valid residues are found for atomization, it falls back to unconditional generation.
        - The function selects between tip atom conditioning and general atom conditioning based on `p_tip`.
    """ 
    # assert not indep.is_sm.any()
    is_valid_for_atomization = indep.has_heavy_atoms_and_seq(atom_mask)
    if can_be_tip is not None:
        is_valid_for_atomization &= can_be_tip
    if not is_valid_for_atomization.any():
        ic('No valid residues for atomization in _get_tip_mask, falling back to unconditional generation')
        is_motif = torch.zeros(indep.length()).bool()
        is_motif[indep.is_sm] = True
        return dict(is_motif=is_motif, is_atom_motif=None)
    valid_idx = is_valid_for_atomization.nonzero()[:,0]
    n_valid_targets = is_valid_for_atomization.sum()
    
    n_atomize = random.randint(n_atomize_min, min(n_atomize_max, n_valid_targets))
    atomize_i = np.random.choice(valid_idx, n_atomize, replace=False)

    is_atom_motif = {}  # dict of (res_idx, [atom_names]) where atom_names is list of atom names constituting the motif
    for i in atomize_i:
        if unconditional:
            atom_names = []
        else:
            if np.random.rand() < p_tip:
                # ... tip atom conditioning: choose seed atom as the furthest from 
                #     the backbone oxygen
                seed_atom = tip_atoms.choose_furthest_from_oxygen(indep.seq[i])
            else:
                # ... general atom conditioning: choose random seed atom
                n_atoms = atom_mask[i].sum()
                seed_atom = np.random.choice(np.arange(n_atoms), 1)[0]
            
            # sample bonded fragment to show as motif from geom. distribution
            n_bonds = np.random.geometric(p=1-bond_inclusion_p) - 1

            # get atom names within n_bonds which will constitute the motif
            atom_names = get_atom_names_within_n_bonds(indep.seq[i], seed_atom, n_bonds)

        assertpy.assert_that(atom_names).does_not_contain(None)
        is_atom_motif[i] = atom_names

    is_motif = torch.zeros(indep.length()).bool()
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

def atomize_all_res(indep, atom_mask, *args, **kwargs):
    is_motif = torch.zeros(indep.length()).bool()
    is_atom_motif = {}
    for i in torch.where(indep.has_heavy_atoms_and_seq(atom_mask))[0]:
        if not indep.is_sm[i]:
            is_atom_motif[i] = []

    is_motif = torch.zeros(indep.length()).bool()
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

def _get_entirely_atomized(indep, atom_mask, crop=9999, *args, **kwargs):
    pop = indep.is_sm.clone()
    is_motif = torch.zeros(indep.length()).bool()
    is_atom_motif = {}
    points_used = pop.sum()

    covale_res_i = torch.tensor([res_i for (res_i, atom_name), lig_i, _ in indep.metadata['covale_bonds']]).tolist()
    for i in covale_res_i:
        points_used += len(aa_model.get_atom_names(indep.seq[i]))
        is_atom_motif[i] = []

    for i in torch.where(indep.has_heavy_atoms_and_seq(atom_mask))[0]:
        if not indep.is_sm[i]:
            points_used += len(aa_model.get_atom_names(indep.seq[i]))
            if points_used > crop:
                break
            is_atom_motif[i] = []

    is_motif = torch.zeros(indep.length()).bool()
    for k in is_atom_motif.keys():
        pop[k] = True
    if pop.sum() == 0:
        # Fall back to unconditional if none are atomizable
        pop = torch.ones_like(indep.is_sm)
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, pop=pop)

def _get_triple_contact(xyz, low_prop, high_prop, broken_prop, xyz_less_than=6, seq_dist_greater_than=10, len_low=1, len_high=3):
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop)
    indices = find_third_contact(contacts)
    if indices is None:
        return _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop)
    L = xyz.shape[0]
    return sample_around_contact(L, indices, len_low, len_high)

def _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop, crop=None, **kwargs):
    """
    Function to make a diffusion mask.
    Options:
        low_prop - lower bound on the proportion of the protein masked
        high_prop - upper bound on the proportion of the protein masked
        broken_prop - proportion of the time the mask is in the middle (broken motif), vs at the ends
    Output:
        1D diffusion mask. True is unmasked, False is masked/diffused
    """
    L = xyz.shape[0]
    diffusion_mask = torch.ones(L).bool()
    if L <= 3:
        # Too small to mask
        return dict(is_motif=torch.zeros(L).bool())
    mask_length = int(np.floor(random.uniform(low_prop, high_prop) * L))
    # decide if mask goes in the middle or the ends
    if random.uniform(0,1) < broken_prop or mask_length < 3:
        high_start = L-mask_length-1
        start = random.randint(0, high_start)
        diffusion_mask[start:start+mask_length] = False
    else:
        # split mask in two
        split = random.randint(1, mask_length-2)
        diffusion_mask[:split] = False
        diffusion_mask[-(mask_length-split):] = False
    return dict(is_motif=diffusion_mask)

def _get_prot_diffusion_mask_islands_from_na_contacts(indep, atom_mask, *args, search_radius_min=1e-2, search_radius_max=20, n_islands_min=1, n_islands_max=4, max_resi_atomize=20, **kwargs):
    # Initializations
    L = indep.xyz.shape[0]
    is_motif = torch.zeros(L).bool()

    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    if torch.sum(is_nucl) < 1:
        # No nucleic acids, no atomization
        return is_motif

    normal_contacts, base_contacts = nucl_utils.get_nucl_prot_contacts(indep, dist_thresh=4.5, ignore_prot_bb=True)  # NOTE: is_gp variable is not used (guide posting compatability)
    if len(normal_contacts) == 0 and len(base_contacts) == 0:
        # No nucleic acid contacts, no atomization
        return is_motif
    elif len(base_contacts) == 0:
        # Default on the only available contacts
        contacts = normal_contacts
    else:
        # Select which contacts to use
        if np.random.rand() < 0.2:
            contacts = normal_contacts
        else:
            contacts = base_contacts

    # Sample the number of islands constrainted by the number of available contacts
    n_islands_min_eff = min([n_islands_min, len(contacts)])
    n_islands_max_eff = min([n_islands_max, len(contacts)])    
    if n_islands_min_eff == n_islands_max_eff:
        # Not enough potential islands 
        n_islands = n_islands_min_eff
    else:
        n_islands = np.random.randint(n_islands_min_eff, n_islands_max_eff)

    # Sample contact points, expand aroundd random radii, and place into motif mask
    na_contact_residues = np.random.choice(contacts, n_islands, replace=False)
    for resi_i in na_contact_residues:
        # Get search radius, then sample residues
        search_radius = np.random.rand() * (search_radius_max - search_radius_min) + search_radius_min
        mask_i = get_neighboring_residues(indep.xyz, atom_mask, resi_i, r=search_radius)
        mask_i = torch.logical_and(is_nucl, mask_i)  # Only consider nucleic acids for atomization
        is_motif[mask_i] = True
    
    motif_idx = torch.nonzero(is_motif).flatten()
    if len(motif_idx) > max_resi_atomize:
        # Too many residues to atomize, in this case, subsample randomly
        is_motif = torch.zeros(L).bool()
        # Sample motif_idx
        motif_idx = np.random.choice(motif_idx, max_resi_atomize, replace=False)
        is_motif[motif_idx] = True

    # Prevent mask tokens from being atomized
    is_motif[torch.logical_or(indep.seq == 26, indep.seq == 31)] = False

    # Prevents the entire thing from being motif, as this is disallowed.
    if is_motif.all():
        is_motif[np.random.randint(L)] = False
        
    is_atom_motif = None  # Do not have any atomization motif
    return is_motif, is_atom_motif

def _protein_motif_scaffold_dna(indep, atom_mask, expand_max=6, *args, **kwargs):
    prot_contact_indices = nucl_utils.get_dna_contacts(indep, 3.5)
    if prot_contact_indices is None or len(prot_contact_indices) == 0:
        return dict(is_motif=None, is_atom_motif=None)
    
    # pick a contact and make motif out of random number of neighbor residues
    selection = np.random.randint(len(prot_contact_indices))
    motif_residues = [prot_contact_indices[selection].item()]

    right_max_expand = np.random.randint(0, expand_max)
    left_max_expand = np.random.randint(0, expand_max)

    bonds = [bond.item() for bond in torch.nonzero(indep.bond_feats[motif_residues[0]]).flatten()]
    
    if len(bonds) == 1:
        to_expand = []
        to_visit = [bonds[0]]
        visited = set(motif_residues)
        while len(to_visit) > 0 and len(to_expand) < max(right_max_expand, left_max_expand):
            current_index = to_visit.pop()
            visited.add(current_index)
            to_expand.append(current_index)

            to_visit.extend([bond.item() for bond in torch.nonzero(indep.bond_feats[current_index]).flatten() if bond.item() not in visited])
    
    elif len(bonds) == 2:
        right_expand = []
        to_visit = [bonds[0]]
        visited = set(motif_residues)

        while len(to_visit) > 0 and len(right_expand) < right_max_expand:
            current_index = to_visit.pop()
            visited.add(current_index)
            right_expand.append(current_index)

            to_visit.extend([bond.item() for bond in torch.nonzero(indep.bond_feats[current_index]).flatten() if bond.item() not in visited])
        
        left_expand = []
        to_visit = [bonds[1]]
        while len(to_visit) > 0 and len(left_expand) < left_max_expand:
            current_index = to_visit.pop()
            visited.add(current_index)
            left_expand.append(current_index)

            to_visit.extend([bond.item() for bond in torch.nonzero(indep.bond_feats[current_index]).flatten() if bond.item() not in visited])

        to_expand = left_expand + right_expand
    
    else:
        return dict(is_motif=None, is_atom_motif=None)

    motif_residues.extend(to_expand)
    is_motif = torch.zeros(len(indep.seq)).bool()
    is_motif[motif_residues] = True

    return dict(is_motif=is_motif, is_atom_motif=None)

def _protein_motif_scaffold_dna_wrapper(protein_motif_scaffold_dna):
    # so this one is gonna wrap protein_motif_scaffold_dna and either return is_motif or run unconditional mask and return that
    @wraps(protein_motif_scaffold_dna)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = protein_motif_scaffold_dna(indep, atom_mask, *args, **kwargs)
        is_motif = ret['is_motif']
        if is_motif is None:
            ret = _get_unconditional_diffusion_mask(indep.xyz, atom_mask, *args, **kwargs)
            ret['is_atom_motif'] = None
        return ret
    return out_get_mask

def _inverse_protein_motif_scaffold_dna(protein_motif_scaffold_dna):
    # this one is gonna wrap protein_motif_scaffold_dna and either return inverse of is_motif or run unconditional mask and return that.
    @wraps(protein_motif_scaffold_dna)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = protein_motif_scaffold_dna(indep, atom_mask, expand_mask=10)
        is_motif = ret.pop('is_motif')
        if is_motif is None:
            ret = _get_unconditional_diffusion_mask(indep.xyz, atom_mask, *args, **kwargs)
        else:
            # invert the motif
            is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot')
            for i in range(len(is_motif)):
                if is_prot[i]:
                    is_motif[i] = ~is_motif[i]
            ret['is_motif'] = is_motif
        ret['is_atom_motif'] = None
        return ret
    return out_get_mask
                

def _get_diffusion_mask_islands(xyz, *args, island_len_min=1, island_len_max=15, n_islands_min=1, n_islands_max=4, p_island_can_be_gp=1, **kwargs):
    L = xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    can_be_gp = torch.zeros(L).bool()
    n_islands = np.random.randint(n_islands_min, n_islands_max)
    for _ in range(n_islands):
        mask_length = np.random.randint(island_len_min, island_len_max)
        mask_length = min(mask_length, L)
        high_start = L - mask_length
        start = random.randint(0, high_start)
        is_motif[start:start+mask_length] = True
        if p_island_can_be_gp >= 1:
            can_be_gp[start:start+mask_length] = True
        else:
            can_be_gp[start:start+mask_length] = torch.rand(1) < p_island_can_be_gp
    
    # Prevents the entire thing from being motif, as this is disallowed.
    if is_motif.all():
        is_motif[np.random.randint(L)] = False
    return dict(is_motif=is_motif, can_be_gp=can_be_gp)

def add_tips(get_mask):
    '''
    Add tip-atom conditions to another mask.
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif, can_be_gp, is_res_seq_shown = ret.pop('is_motif'), ret.pop('is_atom_motif'), ret.pop('can_be_gp'), ret.pop('is_res_seq_shown')
        is_atom_motif = is_atom_motif or {}

        # We only want to turn non-motif residues into tips. The idea being we're adding more constraints to a mask
        can_be_tip = ~is_motif
        can_be_tip[list(is_atom_motif)] = False

        if 'pop' in ret:
            can_be_tip &= ret['pop']

        if can_be_tip.any():

            tip_ret = _get_tip_mask(indep, atom_mask, *args, can_be_tip=can_be_tip, **kwargs)
            tip_atom_motifs = tip_ret['is_atom_motif']
            if tip_atom_motifs is not None:
                for key, value in tip_atom_motifs.items():
                    is_atom_motif[key] = value
                    can_be_gp[key] = True
                    is_res_seq_shown[key] = True # this value is unused but oh well
                    is_motif[key] = False # this value is unused but oh well
        else:
            print("WARNING: add_tips() can't find any region allowed to be tip atoms")

        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, can_be_gp=can_be_gp, is_res_seq_shown=is_res_seq_shown, **ret)
    return out_get_mask

def add_seq_islands(get_mask, mode='anywhere', p_islands_means_shown=0.5, seq_island_len_min=1, seq_island_len_max=15, seq_n_islands_min=1, seq_n_islands_max=4):
    '''
    Modify the is_res_seq_shown vector to be different from is_motif

    if get_mask has is_res_seq_shown:
        - Only the mode region is modified
    else:
        - is_res_seq_shown is defaulted to is_motif then mode region is modified

    Modes:
        - anywhere - the islands have no correlation with is_motif
        - is_motif - the islands are only inside is_motif
        - is_diffused - the islands are only inside is_diffused

    Inputs:
        get_mask (mask_generator func): A function that returns at least is_motif, is_atom_motif, and can_be_gp
        mode (str): One of the modes listed above. Defines where islands can be
        p_islands_means_shown (float): 0-1. Probability that islands are True and the rest is False
        seq_island_len_min (int): Minimum size of island
        seq_island_len_max (int): Maximum size of island
        seq_n_islands_min (int): Min number of islands
        seq_n_islands_max (int): Max number of islands


    '''
    assert mode in ['anywhere', 'is_motif', 'is_diffused'], f'Unknown add_seq_islands mode: {mode}'

    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif, can_be_gp = ret.pop('is_motif'), ret.pop('is_atom_motif'), ret.pop('can_be_gp')
        is_atom_motif = is_atom_motif or {}
        if 'is_res_seq_shown' in ret:
            is_res_seq_shown = ret.pop('is_res_seq_shown')
        else:
            is_res_seq_shown = is_motif.clone()
            is_res_seq_shown[list(is_atom_motif)] = True


        # Figure out where we are going to put the islands
        if mode == 'anywhere':
            can_be_island = torch.ones(len(is_motif), dtype=bool)
        elif mode == 'is_motif':
            can_be_island = is_motif.clone()
        elif mode == 'is_diffused':
            can_be_island = ~is_motif.clone()
        else:
            assert False, "This shouldn't be possible."

        can_be_island[list(is_atom_motif)] = False
        if 'pop' in ret:
            can_be_island &= ret['pop']


        if can_be_island.sum() > 0:

            # A sea of False with islands of True? (means_shown = True) Or a sea of True with islands of False (means_shown = False)
            if p_islands_means_shown < 1e-8:
                means_shown = False
            elif p_islands_means_shown > 1 - 1e-8:
                means_shown = True
            else:
                means_shown = torch.rand(1) < p_islands_means_shown

            # Use _diffusion_mask_islands to do the island calculation
            island_ret = _get_diffusion_mask_islands(indep.xyz[can_be_island], atom_mask,
                            island_len_min=seq_island_len_min, island_len_max=seq_island_len_max, n_islands_min=seq_n_islands_min, n_islands_max=seq_n_islands_max)

            # Create the sea
            is_res_seq_shown[can_be_island] = not means_shown
            # Create the islands
            wh_can_be_island = torch.where(can_be_island)[0]
            island_indices = wh_can_be_island[island_ret['is_motif']]
            is_res_seq_shown[island_indices] = means_shown

            # can_be_gp needs to be set to false for anything we just masked the sequence of
            if means_shown:
                we_masked_seq_of = ~island_ret['is_motif']
            else:
                we_masked_seq_of = island_ret['is_motif']
            can_be_gp[wh_can_be_island[we_masked_seq_of]] = False

        else:
            print("WARNING: add_seq_islands() can't find any region allowed to be sequence islands")

        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, is_res_seq_shown=is_res_seq_shown, can_be_gp=can_be_gp, **ret)
    return out_get_mask


def _get_unconditional_diffusion_mask(xyz, *args, **kwargs):
    """
    unconditional generation of proteins, if a small molecule is present it will be given as context
    """
    L = xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    return dict(is_motif=is_motif)

def make_sm_compatible(get_mask):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        L = indep.length()
        diffusion_mask = torch.ones(L).bool()
        diffusion_mask_prot = get_mask(indep.xyz[~indep.is_sm], *args, **kwargs).pop('is_motif')
        diffusion_mask[~indep.is_sm] = diffusion_mask_prot
        return dict(is_motif=diffusion_mask, is_atom_motif=None)
    return out_get_mask

def _get_prot_unconditional_atomize_na_contacts_mask(indep, atom_mask, *args, **kwargs):
    # Initializations
    L = indep.xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    is_atom_motif = None

    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot')
    if torch.sum(is_nucl) < 1 or torch.sum(is_prot) < 1:
        # No nucleic acids, or no proteins, then no atomization
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

    prot_contacts, na_contacts = nucl_utils.protein_dna_sidechain_base_contacts(indep, contact_distance=4.5, expand_prot=False)
    # NOTE: is_gp variable is not used (guide posting compatability)
    if prot_contacts is None:
        # No nucleic acid contacts, no atomization
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

    # Sample contact points, expand aroundd random radii, and place into motif mask
    for contact_index in na_contacts:
        is_motif[contact_index] = True

    # Prevent mask tokens from being atomized
    is_motif[torch.logical_or(indep.seq == 26, indep.seq == 31)] = False

    # Prevents the entire thing from being motif, as this is disallowed.
    if is_motif.all():
        is_motif[np.random.randint(L)] = False
        
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

def _get_prot_contactmotif_atomize_na_contacts_mask(indep, atom_mask, expand_prot_prob=0.25, *args, **kwargs):
    # Initializations
    L = indep.xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    is_atom_motif = None  # Do not have any atomization motif    

    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot')
    if torch.sum(is_nucl) < 1 or torch.sum(is_prot) < 1:
        # No nucleic acids, or no proteins, then no atomization
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)


    # Some fraction of the time (a quarter by default) we will expand the guideposted region to attempt to include the entire recognition helix
    if random.random() < expand_prot_prob:
        prot_contacts, na_contacts = nucl_utils.protein_dna_sidechain_base_contacts(indep, contact_distance=4.5, expand_prot=True)
    else:
        # otherwise only guidepost the residues making actual contacts
        prot_contacts, na_contacts = nucl_utils.protein_dna_sidechain_base_contacts(indep, contact_distance=4.5, expand_prot=False)
    # NOTE: is_gp variable is not used (guide posting compatability)
    if prot_contacts is None:
        # No nucleic acid contacts, no atomization
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

    # Sample contact points, expand aroundd random radii, and place into motif mask
    for contact_index in na_contacts:
        is_motif[contact_index] = True

    for contact_index in prot_contacts:
        is_motif[contact_index] = True

    # Prevent mask tokens from being atomized
    is_motif[torch.logical_or(indep.seq == 26, indep.seq == 31)] = False

    # Prevents the entire thing from being motif, as this is disallowed.
    if is_motif.all():
        is_motif[np.random.randint(L)] = False
        
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)


def _get_na_prot_contact_mask(indep, atom_mask, *args, **kwargs):
    # Initializations
    L = indep.xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    is_atom_motif = None  # Do not have any atomization motif    

    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot')
    if torch.sum(is_nucl) < 1 or torch.sum(is_prot) < 1:
        # No nucleic acids, or no proteins, then no atomization
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

    prot_contacts, na_contacts = nucl_utils.protein_dna_sidechain_base_contacts(indep, contact_distance=4.5, expand_prot=False)

    # NOTE: is_gp variable is not used (guide posting compatability)
    if prot_contacts is None:
        # No nucleic acid contacts, no atomization
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

    # Sample contact points, expand aroundd random radii, and place into motif mask
    for contact_index in na_contacts:
        is_motif[contact_index] = True

    for contact_index in prot_contacts:
        is_motif[contact_index] = True

    # Prevent mask tokens from being atomized
    is_motif[torch.logical_or(indep.seq == 26, indep.seq == 31)] = False

    # Prevents the entire thing from being motif, as this is disallowed.
    if is_motif.all():
        is_motif[np.random.randint(L)] = False
        
    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

def _get_tipatom_mask_proteinonly_anywhere(indep, atom_mask, sample_min=1, sample_max=6, *args, **kwargs):
    L = indep.xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    is_atom_motif = None  # Do not have any atomization motif    

    is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot')

    n_sample = np.random.randint(sample_min, sample_max)
    prot_indices = torch.where(is_prot)[0]
    sampled_indices = np.random.choice(prot_indices, n_sample, replace=False)

    is_motif[sampled_indices] = True

    return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)

def make_atomized(get_mask, min_atomized_residues=1, max_atomized_residues=5, sample=True):
    """
    Args:
        sample (bool): whether or not to sample positions to get a random subset
    """
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        assert is_atom_motif is None, 'attempting to atomize a masking function that is already returning atomization masks'
        can_be_atomized = is_motif * indep.is_valid_for_atomization(atom_mask)
        if not can_be_atomized.any():
            return dict(is_motif=is_motif, is_atom_motif=None, **ret)
        atomize_indices = torch.nonzero(can_be_atomized).flatten()

        # Only sample when allowed (default behavior)
        if sample:
            n_sample = random.randint(min_atomized_residues, max_atomized_residues)
            n_sample = min(len(atomize_indices), n_sample)
            atomize_indices = np.random.choice(atomize_indices, n_sample, replace=False)
        is_atom_motif = {i:choose_contiguous_atom_motif(indep.seq[i]) for i in atomize_indices}
        is_motif[atomize_indices] = False
        return is_motif, is_atom_motif, *extra_ret
    return out_get_mask


def make_atomized_complete(get_mask, min_atomized_residues=1, max_atomized_residues=8, max_size=384):
    """
    Decorator function that atomizes a masking function by adding completely atomized residues until the maximum size is reached.

    Args:
        get_mask (function): The masking function to be atomized.
        min_atomized_residues (int): The minimum number of atomized residues to atomize.
        max_atomized_residues (int): The maximum number of atomized residues to atomize
        max_size (int): The maximum size of the indep. Defaults to 384.

    Returns:
        function: The atomized masking function.

    Raises:
        AssertionError if the input exceeds the max_size
    """
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        assert is_atom_motif is None, 'attempting to atomize a masking function that is already returning atomization masks'
        can_be_atomized = is_motif * indep.is_valid_for_atomization(atom_mask)
        if not can_be_atomized.any():
            return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
        atomize_indices = torch.nonzero(can_be_atomized).flatten()

        N_resi = len(is_motif)
        assert N_resi <= max_size, f'Protein is too large for atomization: {N_resi} > {max_size}'

        # Only sample when allowed (default behavior)
        n_sample = random.randint(min_atomized_residues, max_atomized_residues)
        n_sample = min(len(atomize_indices), n_sample)
        atomize_indices = np.random.choice(atomize_indices, n_sample, replace=False)
        is_atom_motif = {}  
        # Randomize order
        atomize_indices = [atomize_indices[i].item() for i in np.random.permutation(len(atomize_indices))]
        for i in atomize_indices:
            # Add new completely atomized atoms until the max size is reached
            atoms_new = [atom for atom in ChemData().aa2long[indep.seq[i]]
                                if atom is not None and atom.find('H') == -1]
            if len(atoms_new) + N_resi > max_size:
                break
            is_atom_motif[i] = atoms_new
            N_resi += len(atoms_new)
        is_motif[atomize_indices] = False
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
    return out_get_mask

def make_atomized_complete_noprotein(get_mask, min_atomized_residues=1, max_atomized_residues=8, max_size=384):
    """
    Decorator function that atomizes a masking function by adding completely atomized residues until the maximum size is reached.
    Does not atomize protein residues

    Args:
        get_mask (function): The masking function to be atomized.
        min_atomized_residues (int): The minimum number of atomized residues to atomize.
        max_atomized_residues (int): The maximum number of atomized residues to atomize
        max_size (int): The maximum size of the indep. Defaults to 384.

    Returns:
        function: The atomized masking function.

    Raises:
        AssertionError if the input exceeds the max_size
    """
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs): 
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        assert is_atom_motif is None, 'attempting to atomize a masking function that is already returning atomization masks'
        can_be_atomized = is_motif * indep.is_valid_for_atomization(atom_mask) * ~nucl_utils.get_resi_type_mask(indep.seq, 'prot')
        if not can_be_atomized.any():
            return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
        atomize_indices = torch.nonzero(can_be_atomized).flatten()

        N_resi = len(is_motif)
        assert N_resi <= max_size, f'Protein is too large for atomization: {N_resi} > {max_size}'

        # Only sample when allowed (default behavior)
        n_sample = random.randint(min_atomized_residues, max_atomized_residues)
        n_sample = min(len(atomize_indices), n_sample)
        atomize_indices = np.random.choice(atomize_indices, n_sample, replace=False)
        is_atom_motif = {}  
        # Randomize order
        atomize_indices = [atomize_indices[i].item() for i in np.random.permutation(len(atomize_indices))]
        for i in atomize_indices:
            # Add new completely atomized atoms until the max size is reached
            atoms_new = [atom for atom in ChemData().aa2long[indep.seq[i]]
                                if atom is not None and atom.find('H') == -1]
            if len(atoms_new) + N_resi > max_size:
                break
            is_atom_motif[i] = atoms_new
            N_resi += len(atoms_new)
        is_motif[atomize_indices] = False
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
    return out_get_mask


def make_tip_protein_for_na_compl(get_mask, max_size=384):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        protein_tip_atoms = {
            1: [' NH1', ' NH2', ' CZ '],
            2: [' ND2', ' OD1', ' CG '],
            3: [' OD2', ' CG ', ' OD1'],
            5: [' OE1', ' NE2', ' CD '],
            6: [' OE1', ' CD ', ' OE2'],
            8: [' CG ', ' ND1', ' CE1', ' NE2', ' CD2'],
            9: [' CD1', ' CG1'],
            10: [' CD1', ' CG ', ' CD2'],
            11: [' CE ', ' NZ '],
            13: [' CG ', ' CD1', ' CE1', ' CZ ', ' CE2', ' CD2'],
            15: [' OG ', ' CB '],
            16: [' OG1', ' CB '],
            17: [' CG ', ' CD1', ' NE1', ' CE2', ' CZ2', ' CH2', ' CZ3', ' CE3', ' CD2'],
            18: [' CG ', ' CD1', ' CE1', ' CZ ', ' OH ', ' CE2', ' CD2'],
            19: [' CG1', ' CG2', ' CB ']
            }
        
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        
        if not is_motif.any():
            ret = _get_tipatom_mask_proteinonly_anywhere(indep, atom_mask)
            is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')

        can_be_atomized_prot = is_motif * indep.is_valid_for_atomization(atom_mask) * nucl_utils.get_resi_type_mask(indep.seq, 'prot')
        # can_be_atomized_na = is_motif * indep.is_valid_for_atomization(atom_mask) * ~nucl_utils.get_resi_type_mask(indep.seq, 'prot')

        if not can_be_atomized_prot.any(): # and not can_be_atomized_na.any():
            return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
        prot_atomize_indices = torch.nonzero(can_be_atomized_prot).flatten()
        # na_atomize_indices = torch.nonzero(can_be_atomized_na).flatten()

        N_resi = len(is_motif)
        assert N_resi <= max_size, 'Protein is too large for atomization'
        is_atom_motif = {}

        # tip atomize all proteins
        for i in prot_atomize_indices:
            if indep.seq[i].item() not in protein_tip_atoms.keys():
                continue
            atoms_new = protein_tip_atoms[indep.seq[i].item()]
            if len(atoms_new) + N_resi > max_size:
                break
            is_atom_motif[i.item()] = atoms_new
            N_resi += len(atoms_new)

        # n_sample = random.randint(min_atomized_residues, max_atomized_residues)
        # n_sample = min(len(na_atomize_indices), n_sample)
        # na_atomize_indices = np.random.choice(na_atomize_indices, n_sample, replace=False)
        # na_atomize_indices = [na_atomize_indices[i] for i in np.random.permutation(len(na_atomize_indices))]
        # for i in na_atomize_indices:
            # Add new completely atomized atoms until the max size is reached
            # atoms_new = [atom for atom in ChemData().aa2long[indep.seq[i]]
                                # if atom is not None and atom.find('H') == -1]
            # if len(atoms_new) + N_resi > max_size:
                # break
            # is_atom_motif[i] = atoms_new
            # N_resi += len(atoms_new)

        # is_motif[na_atomize_indices] = False
        is_motif[prot_atomize_indices] = False

        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
    return out_get_mask


def make_atomized_dna_tip_protein_for_na_compl(get_mask, min_atomized_residues=1, max_atomized_residues=8, max_size=384):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        protein_tip_atoms = {
            1: [' NH1', ' NH2', ' CZ '],
            2: [' ND2', ' OD1', ' CG '],
            3: [' OD2', ' CG ', ' OD1'],
            5: [' OE1', ' NE2', ' CD '],
            6: [' OE1', ' CD ', ' OE2'],
            8: [' CG ', ' ND1', ' CE1', ' NE2', ' CD2'],
            9: [' CD1', ' CG1'],
            10: [' CD1', ' CG ', ' CD2'],
            11: [' CE ', ' NZ '],
            13: [' CG ', ' CD1', ' CE1', ' CZ ', ' CE2', ' CD2'],
            15: [' OG ', ' CB '],
            16: [' OG1', ' CB '],
            17: [' CG ', ' CD1', ' NE1', ' CE2', ' CZ2', ' CH2', ' CZ3', ' CE3', ' CD2'],
            18: [' CG ', ' CD1', ' CE1', ' CZ ', ' OH ', ' CE2', ' CD2'],
            19: [' CG1', ' CG2', ' CB ']
            }
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')

        can_be_atomized_prot = is_motif * indep.is_valid_for_atomization(atom_mask) * nucl_utils.get_resi_type_mask(indep.seq, 'prot')
        can_be_atomized_na = is_motif * indep.is_valid_for_atomization(atom_mask) * ~nucl_utils.get_resi_type_mask(indep.seq, 'prot')

        if not can_be_atomized_prot.any() and not can_be_atomized_na.any():
            return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
        prot_atomize_indices = torch.nonzero(can_be_atomized_prot).flatten()
        na_atomize_indices = torch.nonzero(can_be_atomized_na).flatten()

        N_resi = len(is_motif)
        assert N_resi <= max_size, 'Protein is too large for atomization'
        is_atom_motif = {}

        # tip atomize all proteins
        for i in prot_atomize_indices:
            if indep.seq[i].item() not in protein_tip_atoms.keys():
                continue
            atoms_new = protein_tip_atoms[indep.seq[i].item()]
            if len(atoms_new) + N_resi > max_size:
                break
            is_atom_motif[i.item()] = atoms_new
            N_resi += len(atoms_new)

        n_sample = random.randint(min_atomized_residues, max_atomized_residues)
        n_sample = min(len(na_atomize_indices), n_sample)
        na_atomize_indices = np.random.choice(na_atomize_indices, n_sample, replace=False)
        na_atomize_indices = [na_atomize_indices[i] for i in np.random.permutation(len(na_atomize_indices))]
        for i in na_atomize_indices:
            # Add new completely atomized atoms until the max size is reached
            atoms_new = [atom for atom in ChemData().aa2long[indep.seq[i].item()]
                                if atom is not None and atom.find('H') == -1]
            if len(atoms_new) + N_resi > max_size:
                break
            is_atom_motif[i.item()] = atoms_new
            N_resi += len(atoms_new)

        is_motif[na_atomize_indices] = False
        is_motif[prot_atomize_indices] = False

        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif)
    return out_get_mask


def atomize_and_diffuse_motif(get_mask):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        is_motif[indep.is_sm] = False
        motif_idx = is_motif.nonzero()[:,0].tolist()
        is_atom_motif = {}
        is_valid_for_atomization = indep.has_heavy_atoms_and_seq(atom_mask)
        for res_i in motif_idx + list(is_atom_motif.keys()):
            if is_valid_for_atomization[res_i]:
                is_atom_motif[res_i] = []
        is_motif[:] = False
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, **ret)
    return out_get_mask


def partially_mask_ligand(get_mask, ligand_mask_low=0.0, ligand_mask_high=1.0):
    '''
    Only show a contiguous portion of a ligand.
    The fraction to show is sampled from Uniform(ligand_mask_low, ligand_mask_high).
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif = ret.pop('is_motif')
        is_motif[indep.is_sm] = False
        abs_from_sm_i = indep.is_sm.nonzero()[:, 0]
        G = nx.from_numpy_matrix(indep.bond_feats[indep.is_sm,:][:,indep.is_sm].detach().cpu().numpy())
        cc = list(nx.connected_components(G))
        for component in cc:
            n_atoms = len(component)
            mask_frac = np.random.uniform(low=ligand_mask_low, high=ligand_mask_high)
            random_node = np.random.choice(list(component), 1)[0]
            component_sorted = [random_node]
            for depth, nodes_at_depth in nx.bfs_successors(G, random_node):
                component_sorted.extend(nodes_at_depth)
            n_closest = int(np.floor(mask_frac*n_atoms))
            to_show = component_sorted[:n_closest]

            to_show_abs = abs_from_sm_i[to_show]
            if to_show_abs.any():
                assertpy.assert_that(indep.is_sm[to_show_abs].all()).is_true()
            is_motif[to_show_abs] = True
        return dict(is_motif=is_motif, **ret)
    return out_get_mask

show_whole_ligand = partial(partially_mask_ligand, ligand_mask_low=1.0, ligand_mask_high=1.0)

def completely_mask_ligand(get_mask):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif = ret.pop('is_motif')
        is_motif[indep.is_sm] = False
        return dict(is_motif=is_motif, **ret)
    return out_get_mask

def clean_mask(get_mask):
    '''
    Cleans a mask so that is_motif is False for atom-motif residues.
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        for k in is_atom_motif.keys():
            assert not indep.is_sm[k]
            is_motif[k] = False
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, **ret)
    return out_get_mask


def no_pop(get_mask):
    '''
    Cleans a mask so that is_motif is False for atom-motif residues.
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        pop = torch.ones(indep.length()).bool()
        return dict(pop=pop, **ret)
    return out_get_mask

def motif_gp(get_mask, overwrite=False):
    '''
    Adapter for old masks that don't return can_be_gp
    Applies default behavior of can_be_gp = is_motif + list(is_atom_motif)
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)

        if overwrite:
            ret.pop('can_be_gp', None)
        else:
            assert 'can_be_gp' not in ret, 'can_be_gp has already been computed, set overwrite=True to overwrite'
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        can_be_gp = is_motif.clone()
        if is_atom_motif is not None:
            can_be_gp[list(is_atom_motif)] = True

        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, can_be_gp=can_be_gp, **ret)
    return out_get_mask

def bb_only(get_mask):
    '''
    Adapter that converts all atomized residues to backbone motif.
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        for residue_index in is_atom_motif.keys():
            is_motif[residue_index] = True
        is_atom_motif = {}
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, **ret)
    return out_get_mask

def no_gp(get_mask):
    '''
    Forbids guideposting everywhere
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        can_be_gp = torch.zeros(len(indep.seq)).bool()
        return dict(can_be_gp=can_be_gp, **ret)
    return out_get_mask

def protein_gp_only(get_mask):
    '''
    Allows guideposting only at protein residues
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        can_be_gp = is_motif.clone()
        if is_atom_motif is not None and is_atom_motif:
            can_be_gp[list(is_atom_motif)] = True
        can_be_gp = can_be_gp * nucl_utils.get_resi_type_mask(indep.seq, 'prot')
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, can_be_gp=can_be_gp, **ret)
    
    return out_get_mask

def motif_shows_seq(get_mask, overwrite=False):
    '''
    Adapter for old masks that don't return is_res_seq_shown
    Applies default behavior of is_res_seq_shown = is_motif + list(is_atom_motif) + indep.is_sm
    '''
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        ret = get_mask(indep, atom_mask, *args, **kwargs)
        if overwrite:
            ret.pop('is_res_seq_shown', None)
        else:
            assert 'is_res_seq_shown' not in ret, 'is_res_seq_shown has already been computed, set overwrite=True to overwrite'
        is_motif, is_atom_motif = ret.pop('is_motif'), ret.pop('is_atom_motif')
        is_res_seq_shown = is_motif.clone()
        if is_atom_motif is not None and is_atom_motif:
            is_res_seq_shown[list(is_atom_motif)] = True
        is_res_seq_shown[indep.is_sm] = True # small molecules must have their sequence shown
        return dict(is_motif=is_motif, is_atom_motif=is_atom_motif, is_res_seq_shown=is_res_seq_shown, **ret)
    return out_get_mask


def _PPI_fully_diffused(indep, *args, only_first_chain_ppi_binders=False, **kwargs):

    is_target = ppi.decide_target(indep, use_first_chain=only_first_chain_ppi_binders)
    if is_target is None:
        raise InvalidMaskException('_PPI_fully_diffused requires a binder/target pair.')

    is_motif = is_target.clone()
    can_be_gp = torch.zeros(indep.length(), dtype=bool)

    return dict(is_motif=is_motif, is_target=is_target, can_be_gp=can_be_gp)

def _PPI_interface_motif_scaffolding(indep, *args, only_first_chain_ppi_binders=False, max_frac_ppi_motifs=0.8, max_ppi_motif_trim_frac=0.4, **kwargs):

    is_target = ppi.decide_target(indep, use_first_chain=only_first_chain_ppi_binders)
    if is_target is None:
        raise InvalidMaskException('_PPI_interface_motif_scaffolding requires a binder/target pair.')
    is_ppi_motif = ppi.training_extract_ppi_motifs(indep, is_target, max_frac_ppi_motifs, max_ppi_motif_trim_frac)

    is_motif = is_target | is_ppi_motif

    return dict(is_motif=is_motif, is_target=is_target, can_be_gp=is_ppi_motif)

def _PPI_random_motif_scaffolding(indep, *args, only_first_chain_ppi_binders=False, max_frac_ppi_motifs=0.8, max_ppi_motif_trim_frac=0.4, **kwargs):

    is_target = ppi.decide_target(indep, use_first_chain=only_first_chain_ppi_binders)
    if is_target is None:
        raise InvalidMaskException('_PPI_random_motif_scaffolding requires a binder/target pair.')
    is_ppi_motif = ppi.training_extract_ppi_motifs(indep, is_target, max_frac_ppi_motifs, max_ppi_motif_trim_frac, dist=10000)

    is_motif = is_target | is_ppi_motif

    return dict(is_motif=is_motif, is_target=is_target, can_be_gp=is_ppi_motif)

def _PPI_no_crop(get_mask):

    @wraps(get_mask)
    def out_get_mask(indep, *args, **kwargs):
        ret = get_mask(indep, *args, **kwargs)
        ret.pop('is_target')
        pop = torch.ones(indep.length()).bool()
        return dict(is_atom_motif=None, pop=pop, **ret)
    return out_get_mask

def _PPI_radial_crop(get_mask):

    @wraps(get_mask)
    def out_get_mask(indep, *args, ppi_radial_crop_low=10, ppi_radial_crop_high=25, **kwargs):
        ret = get_mask(indep, *args, **kwargs)
        is_motif, is_target = ret.pop('is_motif'), ret.pop('is_target')
        is_hotspot, is_antihotspot = ppi.find_hotspots_antihotspots(indep)
        pop = ppi.radial_crop(indep, ~is_motif, is_hotspot, is_target, distance=random.uniform(ppi_radial_crop_low,ppi_radial_crop_high))
        return dict(is_motif=is_motif, is_atom_motif=None, pop=pop, **ret)
    return out_get_mask

def _PPI_planar_crop(get_mask):

    @wraps(get_mask)
    def out_get_mask(indep, *args, ppi_planar_crop_low=10, ppi_planar_crop_high=25, **kwargs):
        ret = get_mask(indep, *args, **kwargs)
        is_motif, is_target = ret.pop('is_motif'), ret.pop('is_target')
        is_hotspot, is_antihotspot = ppi.find_hotspots_antihotspots(indep)
        pop = ppi.planar_crop(indep, ~is_motif, is_hotspot, is_target, distance=random.uniform(ppi_planar_crop_low,ppi_planar_crop_high))
        return dict(is_motif=is_motif, is_atom_motif=None, pop=pop, **ret)
    return out_get_mask

get_sm_contacts = motif_shows_seq(motif_gp(no_pop(_get_sm_contacts)))
get_diffusion_mask_simple = motif_shows_seq(motif_gp(no_pop(make_covale_compatible(make_sm_compatible(_get_diffusion_mask_simple)))))
get_diffusion_mask_islands = motif_shows_seq(motif_gp(no_pop(make_covale_compatible(make_sm_compatible(_get_diffusion_mask_islands)))))
get_triple_contact = motif_shows_seq(motif_gp(no_pop(make_sm_compatible(_get_triple_contact))))
get_double_contact = motif_shows_seq(motif_gp(no_pop(make_sm_compatible(_get_double_contact))))
atomize_get_triple_contact = make_atomized(get_triple_contact)
atomize_get_double_contact = make_atomized(get_double_contact)
get_unconditional_diffusion_mask = motif_shows_seq(motif_gp(no_pop(make_covale_compatible(make_sm_compatible(_get_unconditional_diffusion_mask)))))
get_tip_gaussian_mask = motif_shows_seq(motif_gp(no_pop(_get_tip_gaussian_mask)))
get_tip_mask = motif_shows_seq(motif_gp(no_pop(make_covale_compatible(_get_tip_mask))))
get_tip_mask_unconditional = motif_shows_seq(motif_gp(no_pop(make_covale_compatible(partial(_get_tip_mask, unconditional=True)))))
get_tip_mask_unconditional_free_ligand = motif_shows_seq(motif_gp(no_pop(completely_mask_ligand(make_covale_compatible(partial(_get_tip_mask, unconditional=True))))))
get_tip_mask_unconditional_partial_ligand = motif_shows_seq(motif_gp(no_pop(partially_mask_ligand(make_covale_compatible(_get_tip_mask)))))
get_closest_tip_atoms = motif_shows_seq(motif_gp(no_pop(_get_closest_tip_atoms)))

get_tip_mask_free_ligand = get_tip_mask
get_tip_mask_partial_ligand = partially_mask_ligand(get_tip_mask_free_ligand)
get_tip_mask_whole_ligand = show_whole_ligand(get_tip_mask_free_ligand)

get_tip_mask_bb_only_free_ligand = motif_shows_seq(motif_gp(bb_only(get_tip_mask), overwrite=True), overwrite=True)
get_tip_mask_bb_only_partial_ligand = partially_mask_ligand(get_tip_mask_bb_only_free_ligand)
get_tip_mask_bb_only_whole_ligand = show_whole_ligand(get_tip_mask_bb_only_free_ligand)

get_atomized_islands = motif_gp(no_pop(make_covale_compatible(atomize_and_diffuse_motif(make_sm_compatible(
        partial(_get_diffusion_mask_islands, n_islands_max=2, island_len_min=10, island_len_max=15))))))

get_unconditional_diffusion_mask_free_ligand = completely_mask_ligand(get_unconditional_diffusion_mask)
get_diffusion_mask_islands_partial_ligand = partially_mask_ligand(get_diffusion_mask_islands)
get_diffusion_mask_islands_free_ligand = completely_mask_ligand(get_diffusion_mask_islands)
get_tip_gaussian_mask_partial_ligand = motif_shows_seq(motif_gp(no_pop(partially_mask_ligand(_get_tip_gaussian_mask))))
get_closest_tip_atoms_partial_ligand = motif_shows_seq(motif_gp(no_pop(partially_mask_ligand(_get_closest_tip_atoms))))
get_unconditional_diffusion_mask_partial_ligand = partially_mask_ligand(get_unconditional_diffusion_mask)
get_entirely_atomized = motif_shows_seq(motif_gp(make_covale_compatible(_get_entirely_atomized)))
get_tip_gaussian_mask.name = 'get_tip_gaussian_mask'
get_tip_gaussian_mask_partial_ligand.name = 'get_tip_gaussian_mask_partial_ligand'

get_PPI_fully_diffused_no_crop = motif_shows_seq(make_covale_compatible(_PPI_no_crop(_PPI_fully_diffused)))
get_PPI_fully_diffused_radial_crop = motif_shows_seq(make_covale_compatible(_PPI_radial_crop(_PPI_fully_diffused)))
get_PPI_fully_diffused_planar_crop = motif_shows_seq(make_covale_compatible(_PPI_planar_crop(_PPI_fully_diffused)))
get_PPI_interface_motif_no_crop = motif_shows_seq(make_covale_compatible(_PPI_no_crop(_PPI_interface_motif_scaffolding)))
get_PPI_interface_motif_radial_crop = motif_shows_seq(make_covale_compatible(_PPI_radial_crop(_PPI_interface_motif_scaffolding)))
get_PPI_interface_motif_planar_crop = motif_shows_seq(make_covale_compatible(_PPI_planar_crop(_PPI_interface_motif_scaffolding)))
get_PPI_random_motif_no_crop = motif_shows_seq(make_covale_compatible(_PPI_no_crop(_PPI_random_motif_scaffolding)))
get_PPI_random_motif_radial_crop = motif_shows_seq(make_covale_compatible(_PPI_radial_crop(_PPI_random_motif_scaffolding)))
get_PPI_random_motif_planar_crop = motif_shows_seq(make_covale_compatible(_PPI_planar_crop(_PPI_random_motif_scaffolding)))


get_diffusion_mask_islands_w_tip = add_tips(get_diffusion_mask_islands)
get_diffusion_mask_islands_partial_ligand_w_tip = add_tips(get_diffusion_mask_islands_partial_ligand)
get_diffusion_mask_islands_free_ligand_w_tip = add_tips(get_diffusion_mask_islands_free_ligand)
get_PPI_fully_diffused_no_crop_w_tip = add_tips(get_PPI_fully_diffused_no_crop)
get_PPI_fully_diffused_radial_crop_w_tip = add_tips(get_PPI_fully_diffused_radial_crop)
get_PPI_fully_diffused_planar_crop_w_tip = add_tips(get_PPI_fully_diffused_planar_crop)
get_PPI_interface_motif_no_crop_w_tip = add_tips(get_PPI_interface_motif_no_crop)
get_PPI_interface_motif_radial_crop_w_tip = add_tips(get_PPI_interface_motif_radial_crop)
get_PPI_interface_motif_planar_crop_w_tip = add_tips(get_PPI_interface_motif_planar_crop)
get_PPI_random_motif_no_crop_w_tip = add_tips(get_PPI_random_motif_no_crop)
get_PPI_random_motif_radial_crop_w_tip = add_tips(get_PPI_random_motif_radial_crop)
get_PPI_random_motif_planar_crop_w_tip = add_tips(get_PPI_random_motif_planar_crop)


get_diffusion_mask_islands_w_seq_islands = add_seq_islands(get_diffusion_mask_islands)
get_diffusion_mask_islands_w_tip_w_seq_islands = add_seq_islands(get_diffusion_mask_islands_w_tip)
get_PPI_fully_diffused_no_crop_w_binder_seqshown_islands = add_seq_islands(get_PPI_fully_diffused_no_crop, mode='is_diffused', p_islands_means_shown=1)
get_PPI_fully_diffused_no_crop_w_target_seqhidden_islands = add_seq_islands(get_PPI_fully_diffused_no_crop, mode='is_motif', p_islands_means_shown=0)

get_PPI_fully_diffused_no_crop_w_binder_w_tip_seqshown_islands = add_seq_islands(get_PPI_fully_diffused_no_crop_w_tip, mode='is_diffused', p_islands_means_shown=1)
get_PPI_fully_diffused_no_crop_w_target_w_tip_seqhidden_islands = add_seq_islands(get_PPI_fully_diffused_no_crop_w_tip, mode='is_motif', p_islands_means_shown=0)




# NA specific mask generators, may generalize to other systems, but not guaranteed currently
get_na_contacting_atomized_islands = no_pop(make_covale_compatible(make_atomized_complete(
        partial(_get_prot_diffusion_mask_islands_from_na_contacts, n_islands_max=3, search_radius_min=1e-2, search_radius_max=10, max_resi_atomize=8))))

get_na_motif_scaffold = motif_shows_seq(no_gp(no_pop(make_covale_compatible(_protein_motif_scaffold_dna_wrapper(_protein_motif_scaffold_dna)))))
get_na_inverse_motif_scaffold = motif_shows_seq(no_gp(no_pop(make_covale_compatible(_inverse_protein_motif_scaffold_dna(_protein_motif_scaffold_dna)))))

get_prot_unconditional_atomize_na_contacts = motif_shows_seq(no_gp(no_pop(make_covale_compatible(make_atomized_complete(_get_prot_unconditional_atomize_na_contacts_mask, max_atomized_residues=20)))))

get_prot_contactmotif_atomize_na_contacts = motif_shows_seq(no_gp(no_pop(make_covale_compatible(make_atomized_complete_noprotein(_get_prot_contactmotif_atomize_na_contacts_mask, max_atomized_residues=20)))))

# guideposted tipatoms on the protein motif contacting residues, na_contacts are atomized motif
get_prot_tipatom_guidepost_atomize_na_contacts = motif_shows_seq(protein_gp_only(no_pop(make_covale_compatible(make_atomized_dna_tip_protein_for_na_compl(_get_na_prot_contact_mask, max_atomized_residues=20)))))

get_prot_tipatom_guidepost_na_contacts = motif_shows_seq(protein_gp_only(no_pop(make_covale_compatible(make_tip_protein_for_na_compl(_get_na_prot_contact_mask)))))

get_prot_tipatom_guidepost_anywhere = motif_shows_seq(protein_gp_only(no_pop(make_covale_compatible(make_tip_protein_for_na_compl(_get_tipatom_mask_proteinonly_anywhere)))))

sm_mask_fallback = {
    get_closest_tip_atoms: get_tip_gaussian_mask,
    get_closest_tip_atoms_partial_ligand: get_tip_gaussian_mask_partial_ligand,
}
"""Defines a dict of fallback masking functions to use when no atomized bits (sm) are present."""

def get_diffusion_mask(
        indep, 
        atom_mask: torch.Tensor, 
        low_prop: float, 
        high_prop: float, 
        broken_prop: float,
        diff_mask_probs: dict[Callable, float],  # dict of (masking_function, probability) tuples
        **kwargs
        ) -> dict:
    """
    Sample a `motif` mask for training diffusion models.

    This function selects a masking function based on provided probabilities and applies it to generate a mask.
    If a selected mask is incompatible with the input, it tries other masks until a valid one is found.

    Args:
        indep: Independent variable containing structural information.
        atom_mask (torch.Tensor): Mask indicating which atoms to consider.
        low_prop (float): Lower bound for the proportion of the structure to mask.
        high_prop (float): Upper bound for the proportion of the structure to mask.
        broken_prop (float): Probability of generating a broken (non-contiguous) mask.
        diff_mask_probs (dict[Callable, float]): Dictionary mapping masking functions to their selection probabilities.
        **kwargs: Additional keyword arguments to pass to the masking function.

    Returns:
        dict: A dictionary containing the generated mask and associated information. Keys that are 
            always present include:
            - 'is_motif' (torch.Tensor): Boolean tensor indicating the motif region. [L]
            - 'is_res_seq_shown' (torch.Tensor): Boolean tensor indicating which residue sequences are shown. [L]
            - 'is_atom_motif' (dict[int, list]): Dictionary indicating atomized motif regions. The key
                is the residue index, the value is the list of atom names that are part of the motif.
            - 'pop' (torch.Tensor): Mask of which atoms are resolved. [L]
            - 'can_be_gp' (torch.Tensor): Boolean tensor indicating which tokens can serve as guide-posts. [L]
            - 'mask_name': Name of the masking function used.

    Raises:
        Exception: If no valid mask can be generated after trying all available masking functions.
    """

    mask_probs = list(diff_mask_probs.items())  # list of (masking_function, probability) tuples
    logger.debug(f'{mask_probs=}')
    logger.debug(f'{[(m.name, p) for m,p in mask_probs]=}')
    logger.debug(f'{sum([p for _, p in mask_probs])=}')

    # Masks can declare that they are incompatible with the example given by throwing InvalidMaskException
    #   Incompatible masks are removed for the next iteration
    for attempt in range(len(mask_probs)):

        probs = np.array([p for _, p in mask_probs])
        if probs.sum() == 0:
            raise Exception('No valid mask found. Remaining probabilities sum to 0.')
        probs /= probs.sum()

        # Sample a masking function
        i_mask = np.random.choice(np.arange(len(probs)), p=probs)
        get_mask, _ = mask_probs.pop(i_mask)

        # Use fallback mask if no small molecule present.
        if not indep.is_sm.any():
            get_mask = sm_mask_fallback.get(get_mask, get_mask)

        logger.debug(f'{get_mask.name=}')
        with error.context(f'mask - {get_mask.name}'):
            try:
                # Call the mask function to get the mask
                ret = get_mask(indep, atom_mask, low_prop=low_prop, high_prop=high_prop, broken_prop=broken_prop, **kwargs)

                # Make sure we have all the return types
                assert isinstance(ret, Mapping), 'Mask functions now return a dictionary'
                required_keys = ['is_motif', 'is_res_seq_shown', 'is_atom_motif', 'pop', 'can_be_gp']
                for key in required_keys:
                    assert key in ret, f'Mask: {get_mask.name} failed to return "{key}"'

                # ... add mask name to output dict
                ret['mask_name'] = get_mask.name
                return ret
            except InvalidMaskException as e:
                logger.debug(f'Mask {get_mask.name} incompatible with example: {e}')

    raise Exception('No valid mask found. Tried all of them.')


def generate_sm_mask(prot_masks, is_sm):
    # Not currently used, but may become part of a better way to do this
    L = is_sm.shape[0]
    input_seq_mask = torch.ones(L).bool()
    input_str_mask = torch.ones(L).bool()
    input_floating_mask = -1
    input_t1d_str_conf_mask = torch.ones(L)
    input_t1d_seq_conf_mask = torch.ones(L)
    loss_seq_mask = torch.ones(L).bool()
    loss_str_mask = torch.ones(L).bool()
    loss_str_mask_2d = torch.ones(L,L).bool()

    mask_dict = {'input_seq_mask':input_seq_mask,
                'input_str_mask':input_str_mask,
                'input_floating_mask':input_floating_mask,
                'input_t1d_str_conf_mask':input_t1d_str_conf_mask,
                'input_t1d_seq_conf_mask':input_t1d_seq_conf_mask,
                'loss_seq_mask':loss_seq_mask,
                'loss_str_mask':loss_str_mask,
                'loss_str_mask_2d':loss_str_mask_2d}
    #is_motif = torch.ones(L).bool()
    #is_motif_prot = mask_dict['input_str_mask']
    for k, v in mask_dict.items():
        if type(v) is not torch.Tensor:
            continue
        if k == 'loss_str_mask_2d':
            continue
        #ic(k, v.shape, prot_masks[k].shape, is_sm.shape)
        v[~is_sm] = prot_masks[k]
        mask_dict[k] = v
    mask_dict['input_seq_mask']
    
    return mask_dict

###################
# Functions for making a mask for nearby contigs - DT
###################
def closest_distance(group1: torch.Tensor, group2: torch.Tensor, 
                     include_point1: torch.Tensor, include_point2: torch.Tensor) -> torch.Tensor:
    '''
    Given two groups of points, how close are the closest pair of points?
    
    Args
        group1 (batch1, n_points1, 3)
        group2 (batch2, n_points2, 3)
        include_point1 (batch1, n_points1): True = the coordinates should be considered in the distance calculation.
    
    Returns
        closest_dist: (batch1, batch2)
    '''
    assert group1.shape[:-1] == include_point1.shape
    assert group2.shape[:-1] == include_point2.shape
    
    # Expand shapes so we can broadcast
    group1 = group1[:,:,None,None,:]
    group2 = group2[None,None,:,:,:]
    include_point1 = include_point1[:,:,None,None]
    include_point2 = include_point2[None,None,:,:]

    # Distance calc
    dists = torch.linalg.norm(group1 - group2, ord=2, dim=-1)
    
    # Both points must be "included" to consider the dist between them
    include_dist = torch.logical_and(include_point1, include_point2)
    dists[~include_dist] = torch.inf

    # find min over all pairs of atom in each group. Would be clearner to do with a "topk_any_dims" like function.
    closest_dist = dists.min(dim=1)[0]
    closest_dist = closest_dist.min(dim=2)[0]
    
    return closest_dist

def get_neighboring_residues(xyz: torch.Tensor, atom_mask: torch.Tensor, 
                             i: int, r: float) -> torch.Tensor:
    '''
    Args
        xyz (L, 14, 3): Atom coordinates in the protien
        atom_mask (L, 14): True = atom is "really" there.
        i: Index of the central residue
        r: Contact radius.
            
    Returns
        neighboring_residues (L,): Boolean mask. True if any atom in the central residue is 
            closer than r to any atom in another residue, they are considered neighbors.
            DOES NOT INCLUDE THE CENTRAL RESIDUE! This is a mask of the *neighbors*.
    '''
    res_xyz = xyz[[i]]
    res_atom_mask = atom_mask[[i]]
    closest_dist = closest_distance(
        group1=res_xyz, 
        group2=xyz,
        include_point1=res_atom_mask,
        include_point2=atom_mask
    )[0]
    neighboring_residues = closest_dist < r
    return neighboring_residues

def dilate_1d(mask: torch.Tensor) -> torch.Tensor:
    '''
    Args
        mask: A 1D boolean mask
        
    Returns
        dilated: A boolean mask where True values have "spread" one space
            to the left and right.
    '''
    
    mask = mask[None,None].float()
    kernel = torch.ones(1,1,3).float()
    dilated = torch.nn.functional.conv1d(mask, kernel, padding=1)
    dilated = torch.clamp(dilated, 0, 1)
    return dilated[0,0].bool()

def erode_1d(mask: torch.Tensor) -> torch.Tensor:
    '''
    Args
        mask: A 1D boolean mask
        
    Returns
        eroded: A boolean mask where True values have "contracted" one space
            from the left and right. Isolated islands of True are removed.
    '''
    return ~dilate_1d(~mask)

def merge_islands(mask: torch.Tensor, n: int=1) -> torch.Tensor:
    '''
    If two Trues are separated by 2*n or fewer spaces,
    the interviening spaces are set to True.
    
    Ex for n=2.
        in:  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        out: [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    
    Args
        mask: A 1D boolean mask        
    '''
    
    for _ in range(n):
        mask = dilate_1d(mask)
    for _ in range(n):
        mask = erode_1d(mask)
        
    return mask

def remove_small_islands(mask: torch.Tensor, n: int = 1) -> torch.Tensor:
    '''
    If a contiguous chunk has less than or equal to 2*n Trues, it is removed.
    
    Ex for n=2.
        in:  [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        out: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    
    Args
        mask: A 1D boolean mask        
    '''
    for _ in range(n):
        mask = erode_1d(mask)
    for _ in range(n):
        mask = dilate_1d(mask)
        
    return mask

def get_contigs_around_residue(xyz: torch.Tensor, atom_mask: torch.Tensor,
                                i: int, r: float) -> torch.Tensor:
    '''
    Given a residue in a protein, find contigs that have residues with at least
    one atom within r Angstroms of any atom in the central residue. Essentially
    it selects residues in a sphere around a central residue, then joins isolated
    residues into contigs. Small contigs are then removed.
    
    Args
        xyz (L, 14, 3): Atom coordinates in the protien
        atom_mask = True = Atom is "really" there.
        i: Index of the central residue
        r: Contact radius.
       
    Returns
        mask (L,): True = residue is in the motif.
    '''
    mask = get_neighboring_residues(xyz, atom_mask, i, r)
    mask[i] = True  # include the central resiude in the motif
    mask = merge_islands(mask, n=1)
    mask = remove_small_islands(mask, n=2)
    
    return mask

def get_nearby_contigs(indep, atom_mask, low_prop, high_prop, broken_prop):
    '''
    Randomly samples a central residue and radius, and returns a contig mask
    of residues in that radius. 
    
    Args: NOTE: These must match the call signature of "get_mask", hence the unused args.
    
    Return
        mask: True = residue is in the contig(s)
        is_atom_motif: Currently this contig selector only works for proteins.
            This is spoofed to match the "get_mask" output signature.
    '''
    max_tries = 100
    xyz = indep.xyz
    L_ptn = xyz.shape[0]
    
    for _ in range(max_tries):
        # Get nearby contig mask
        i = int(torch.randint(high=L_ptn, size=(1,)))
        r = float(torch.rand(size=(1,))) * 15. + 5.
        mask = get_contigs_around_residue(xyz, atom_mask, i, r)
        
        # Do the contigs cover enough/too much of the protein?
        prop = mask.sum() / L_ptn
        if low_prop <= prop <= high_prop:
            break

    # Spoof is_atom_motif output
    is_atom_motif = None

    return mask, is_atom_motif

#####################################
# Main mask generator function
#####################################

def generate_masks(
        indep,
        task: Literal["diff"],
        loader_params: dict,
        chosen_dataset: Literal["complex", "negative", "pdb_aa", "compl", "sm_compl", "sm_complex"],
        full_chain: tuple[int, int] | None = None,
        atom_mask=None,
        metadata=None,
        datahub_config=None
    ) -> dict: #full_chain is for complexes, to signify which chain is complete
    '''
    Slimmed down function that outputs 1D masks for inputs and loss calculations.
    Input masks are defined as True=(unmasked)/False=masked (except for input_t1dconf, which is a scalar value, and seq2str_mask which is the msa mask for the seq2str task)
    Loss masks are defined as True=(loss applied)/False=(no loss applied)
    
    Input masks:
        -input_seq
        -input_str
        -input_floating = points to be represented as floating points (structure present but side chains masked out)
        -input_t1d_str_conf = scalar to multiply input str t1d confidences by
        -input_t1d_seq_conf = scalar to multiply input seq t1d confidences by

    Output masks:
        -loss_seq
        -loss_str
        -loss_str_2d = additional coordinate pair masking to be applied on top of loss_str 1d masking.
    '''
    assert task == "diff", "All tasks except 'diff' are deprecated and not supported anymore."

    L = indep.length()

    # Initialize output variables
    mask_dict = {}
    # ... special variables that will be overwritten in `mask_dict` at the end
    mask_name = None
    input_seq_mask = torch.ones(L).bool()  # by default, all positions "shown"
    input_str_mask = torch.ones(L).bool()  # by default, all positions "shown"
    is_atom_motif = None
    can_be_gp = torch.ones(L).bool()  # by default, all positions "can_be_gp"

    # ... task & dataset specific processing 
    if chosen_dataset not in ["complex", "negative"]:
        """
        Hal task but created for the diffusion-based training. 
        """ 
        thismodule = sys.modules[__name__]
        mask_probs = OrderedDict()
        
        # mask probabilities from datahub config
        if datahub_config and chosen_dataset in datahub_config and 'mask_probabilities' in datahub_config[chosen_dataset]:
            mask_probabilities = datahub_config[chosen_dataset]['mask_probabilities']
        else:
            mask_probabilities = loader_params['DIFF_MASK_PROBS']
        
        for k,v in mask_probabilities.items():
            f = getattr(thismodule, k)
            f.name = k
            mask_probs[f] = float(v)
        # Plumbing hack
        indep.metadata = metadata
    
        mask_dict = get_diffusion_mask(
            indep,
            atom_mask,  # [L, N_heavy(23)]
            low_prop=loader_params['MASK_MIN_PROPORTION'],  # e.g. 0.2
            high_prop=loader_params['MASK_MAX_PROPORTION'], # e.g. 1.0
            broken_prop=loader_params['MASK_BROKEN_PROPORTION'], # e.g. 0.5
            crop=loader_params['CROP']-20, # crop size of original loader, -20 for buffer. Q(Woody): Does anything bad happen if we exceed the crop size?
            diff_mask_probs=mask_probs, # dict of (masking_function, probability) tuples
            show_tip=loader_params.get('show_tip', False),
            **loader_params.mask,
            ) 
        
        # Unpack required items in mask dict (Needed for now for compatibility with old code) 
        # TODO:(smathis) Possibly deprecate this
        input_str_mask = mask_dict.pop('is_motif')
        input_seq_mask = mask_dict.pop('is_res_seq_shown')
        is_atom_motif = mask_dict.pop('is_atom_motif')
        pop = mask_dict.pop('pop')
        can_be_gp = mask_dict.pop('can_be_gp')
        mask_name = mask_dict.pop('mask_name')
    elif chosen_dataset == 'complex':
        '''
        Diffusion task for complexes. Default is to diffuse the whole of the complete chain.
        Takes full_chain as input, which is [full_chain_start_idx, full_chain_end_idx]
        '''
        assert full_chain[1] is not None
        
        input_str_mask = torch.clone(full_chain)
        input_seq_mask = torch.clone(input_str_mask)
        can_be_gp = input_str_mask.clone()  # by default, all positions for which structure is shown `can_be_gp`
    else:
        sys.exit(f"Masks cannot be generated for the {task} task!. NOTE: All tasks except 'diff' are deprecated and not supported anymore.")

    # ... sanity check
    assert torch.sum(~input_seq_mask) > 0, f'Task = {task}, dataset = {chosen_dataset}, full chain = {full_chain}'

    # ... log task, dataset & mask selection
    logger.info(f'Mask selection: Task = {task}, dataset = {chosen_dataset}, mask = {mask_name}')

    # ... final processing of output mask_dict
    # ... make dictionary keys integers, not torch scalars.
    if is_atom_motif:
        is_atom_motif = {maybe_item(res_idx): atom_name_list for res_idx, atom_name_list in is_atom_motif.items()}
    # ... update specialised keys
    mask_dict.update({
        'input_str_mask': input_str_mask,  # whether a token's structure is shown
        'input_seq_mask': input_seq_mask,  # whether a token's sequence is shown
        'is_atom_motif': is_atom_motif or {},  # whether and which specific atoms in a token are shown as motif (will require atomization later)
        'pop': pop,  # whether a token is occupied (i.e. resolved in the structure) 
        'can_be_gp': can_be_gp,  # whether a token can be a gp
        'mask_name': mask_name,  # name of the mask
    })

    return mask_dict

def maybe_item(i: torch.Tensor | int | float) -> int | float:
    if hasattr(i, 'item'):
        return i.item()
    return i

def choose_contiguous_atom_motif(res):
    """
    chooses a contiguous 3 or 4 atom motif
    """
    bond_feats = get_residue_bond_feats(res)
    # choose atoms to be given as the motif 
    bond_graph = nx.from_numpy_matrix(bond_feats.numpy())
    paths = rf2aa.util.find_all_paths_of_length_n(bond_graph, 2)
    paths.extend(rf2aa.util.find_all_paths_of_length_n(bond_graph, 3))
    chosen_path = random.choice(paths)
    atom_names = [ChemData().aa2long[res][i] for i in chosen_path]
    return atom_names


def get_residue_bond_feats(res, include_H=False):
    bond_feats = torch.zeros((ChemData().NTOTAL, ChemData().NTOTAL))
    for j, bond in enumerate(ChemData().aabonds[res]):
        start_idx = ChemData().aa2long[res].index(bond[0])
        end_idx = ChemData().aa2long[res].index(bond[1])

        # maps the 2d index of the start and end indices to btype
        bond_feats[start_idx, end_idx] = ChemData().aabtypes[res][j]
        bond_feats[end_idx, start_idx] = ChemData().aabtypes[res][j]
    
    if not include_H:
        bond_feats = bond_feats[:ChemData().NHEAVY, :ChemData().NHEAVY]
    return bond_feats
