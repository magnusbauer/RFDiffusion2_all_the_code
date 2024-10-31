from __future__ import annotations  # allows circular imports for type hinting

from collections import defaultdict

import torch
import assertpy

import rf_diffusion.aa_model as aa_model
from rf_diffusion.chemical import ChemicalData as ChemData

def set_nonexistant_atoms_to_nan(xyz, seq, H_exists=False):
    atom_mask = ChemData().allatom_mask[seq]
    if not H_exists:
        atom_mask[:, ChemData().NHEAVYPROT:] = False # no Hs
    xyz = xyz.clone()
    xyz[~atom_mask] = torch.nan
    return xyz

def atomized_indices_atoms(atomizer: aa_model.AtomizeResidues, atom_names_by_res: dict[int, list[str]]) -> list[int]:
    atom_idx_by_res = atomizer.get_atom_idx_by_res()
    named_i = []
    for res_i, atom_names in atom_names_by_res.items():
        assert isinstance(res_i, int), res_i
        atomized_residue_idxs = atom_idx_by_res[res_i]
        original_aa = atomizer.deatomized_state[res_i].aa
        within_res_atom_idxs = {atom_name:i for i,atom_name in enumerate(e for e in ChemData().aa2long[original_aa] if e is not None)}

        # Strip whitespace
        within_res_atom_idxs = {atom_name.strip():i for atom_name,i in within_res_atom_idxs.items()}
        atom_names = [a.strip() for a in atom_names]

        for atom_name in atom_names:
            try:
                within_res_atom_idx = within_res_atom_idxs[atom_name]
            except KeyError as e:
                raise KeyError(f'{atom_name} not one of the known atoms for residue {res_i} with seq {ChemData().num2aa[original_aa]}: {list(within_res_atom_idxs.keys())}') from e
            atom_i = atomized_residue_idxs[within_res_atom_idx].item()
            named_i.append(atom_i)

    return named_i

def atomized_indices_res_i(atomizer, idx):
    atomized_res_idx = []
    atomized_res_idx_from_res = atomizer.get_atomized_res_idx_from_res()
    for i in idx:
        atomized_res_idx.append(atomized_res_idx_from_res[i.item()])
    return atomized_res_idx


def atomized_indices_res(
        atomizer: aa_model.AtomizeResidues, 
        mask: torch.Tensor, 
        allow_missing_residue: bool=False
    ) -> list[int]:
    """
    Converts residue indices from a pre-atomized mask to their corresponding indices in the atomized protein structure.

    Args:
        - atomizer (AtomizeResidues): Object that handles the mapping between atomized and non-atomized protein 
            representations.
        - mask (torch.Tensor): Binary mask tensor for the pre-atomized protein, where True indicates residues to convert.
        - allow_missing_residue (bool, optional): If True, skips residues that don't have a mapping in the atomized 
            structure. Defaults to False.

    Returns:
        - list[int]: List of indices in the atomized protein structure corresponding to the True positions in the input 
            mask.

    Raises:
        - KeyError: If allow_missing_residue is False and a residue index is not found in the atomized structure.
    """
    # ... get mapping of pre-atomized residue idx to post-atomized residue idx (for non-atomized residues only!)
    atomized_res_idx_from_res: dict[int, int] = atomizer.get_atomized_res_idx_from_res()

    atomized_res_idx = []
    mask_idx = torch.nonzero(mask)[:,0]
    for i in mask_idx:
        if allow_missing_residue and i.item() not in atomized_res_idx_from_res:
            continue
        atomized_res_idx.append(atomized_res_idx_from_res[i.item()])
    return atomized_res_idx

def get_res_atom_name_by_atomized_idx(atomizer) -> dict[int, tuple[int, str]]:
    '''
    Returns a dictionary mapping the index of an atom in the atomized protein
    to the original (0-index residue, atom_name) from pre-atomization.
    '''
    atomized_res_idx_from_res = atomizer.get_atom_idx_by_res()
    res_idx_atom_name_by_atomized_idx = {}
    for res_idx, atomized_res_idx in atomized_res_idx_from_res.items():
        original_aa = atomizer.deatomized_state[res_idx].aa
        atom_name_by_within_res_idx = {i:atom_name for i,atom_name in enumerate(e for e in ChemData().aa2long[original_aa] if e is not None)}
        for within_res_atom_idx, atom_idx in enumerate(atomized_res_idx):
            res_idx_atom_name_by_atomized_idx[atom_idx.item()] = (
                # f'{rf2aa.chemical.num2aa[original_aa]}{atomizer.indep_initial_copy.idx[res_idx]}'
                res_idx, atom_name_by_within_res_idx[within_res_atom_idx].strip()
            )
    return res_idx_atom_name_by_atomized_idx

def res_atom_name(atomizer: aa_model.AtomizeResidues, atomized_idx: torch.Tensor) -> list[tuple[int, str]]:
    '''
    Params:
        Indices of atoms in the atomized protein
    Returns:
        List of (0-index residue, atom_name)-tuples from pre-atomization.
    '''
    res_idx_atom_name_by_atomized_idx = get_res_atom_name_by_atomized_idx(atomizer)
    return [res_idx_atom_name_by_atomized_idx[i.item()] for i in atomized_idx]

def convert_atomized_mask(atomizer, mask):
    '''
    Params:
        atomizer: aa_model.AtomizeResidues
        mask: binary mask, the length of an atomized protein
    Returns:
        Dictionary mapping deatomized 0-indexed residues to the atom names corresponding to True in the mask, i.e.
            {0: ['CB', 'CG], 1: ['ALL'], ...}
    '''
    atomized_idx = mask.nonzero()[:,0]
    atomized_res_idx_from_res = atomizer.get_atomized_res_idx_from_res()
    res_idx_from_atomized_res_idx = {v:k for k,v in atomized_res_idx_from_res.items()}

    res_idx_atom_name_by_atomized_idx = get_res_atom_name_by_atomized_idx(atomizer)
    o = defaultdict(list)
    for atomized_i in atomized_idx.tolist():
        if atomized_i in res_idx_atom_name_by_atomized_idx:
            deatomized_i, atom_name = res_idx_atom_name_by_atomized_idx[atomized_i]
            o[deatomized_i].append(atom_name)

        elif atomized_i in res_idx_from_atomized_res_idx:
            deatomized_i = res_idx_from_atomized_res_idx[atomized_i]
            o[deatomized_i].append('ALL')
        else:
            raise Exception(f'{atomized_i} not found')
    return o

def atomized_indices_from_preatomized_res_indices(atomizer, res_indices) -> torch.Tensor:
    res_idx_atom_name_by_atomized_idx = get_res_atom_name_by_atomized_idx(atomizer)
    o = []
    for atomized_i, (res_i, atom_name) in res_idx_atom_name_by_atomized_idx.items():
        if res_i in res_indices:
            o.append(atomized_i)

    return torch.tensor(o)

def atom_indices(
        atomizer: aa_model.AtomizeResidues, 
        res_mask: torch.Tensor, 
        atom_names_by_res: dict[int, list[str]], 
        allow_missing_residue: bool=False
    ) -> torch.Tensor:
    res_i = atomized_indices_res(atomizer, res_mask, allow_missing_residue=allow_missing_residue)
    atom_i = atomized_indices_atoms(atomizer, atom_names_by_res)
    assert set(res_i).isdisjoint(set(atom_i))
    return res_i + atom_i

def create_masks(
        atomizer: aa_model.AtomizeResidues, 
        is_res_str_shown: torch.Tensor,  # e.g. [True, False, True, ...], shape [L]
        is_res_seq_shown: torch.Tensor,  # e.g. [True, False, True, ...], shape [L]
        is_atom_str_shown: dict[int, list[str]]  # e.g. {23: [" N " , " CA "], ...}
) -> tuple[torch.Tensor, torch.Tensor]:

    # Show all atoms for residues which have an atom motif
    is_atom_seq_shown = {res_i: [elt for elt in ChemData().aa2long[atomizer.deatomized_state[res_i].aa][:ChemData().NHEAVYPROT] if elt is not None]
                            for res_i in is_atom_str_shown.keys()}
    
    return create_masks_str_seq(atomizer, is_res_str_shown, is_res_seq_shown, is_atom_str_shown, is_atom_seq_shown)

def create_masks_str_seq(
        atomizer: aa_model.AtomizeResidues, 
        is_res_str_shown: torch.Tensor, 
        is_res_seq_shown: torch.Tensor, 
        is_atom_str_shown: dict[int, list[str]], 
        is_atom_seq_shown: dict[int, list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
    L = len(atomizer.atomized_state)
    str_shown_indices = atom_indices(atomizer, is_res_str_shown, is_atom_str_shown, allow_missing_residue=True)
    seq_shown_indices = atom_indices(atomizer, is_res_seq_shown, is_atom_seq_shown, allow_missing_residue=True)
    is_diffused = torch.ones(L).bool()
    is_masked_seq = torch.ones(L).bool()
    is_diffused[str_shown_indices] = False
    is_masked_seq[seq_shown_indices] = False

    return is_diffused, is_masked_seq

def atomize_and_mask(
        *,
        indep: aa_model.Indep, 
        is_res_str_shown: torch.Tensor, 
        is_res_seq_shown: torch.Tensor, 
        is_atom_str_shown: dict[int, list[str]]
    ) -> tuple[aa_model.Indep, torch.Tensor, torch.Tensor, aa_model.AtomizeResidues]:
    """
    Atomizes and masks residues in a protein structure.

    Args:
        indep (aa_model.Indep): The input protein structure.
        is_res_str_shown (torch.Tensor): Boolean mask indicating which residues have their structure shown.
        is_res_seq_shown (torch.Tensor): Boolean mask indicating which residues have their sequence shown.
        is_atom_str_shown (dict): Dictionary mapping residue indices to lists of atom names to be shown.
            Example: {0: ['OD1', 'CB'], 4: ['CA']}

    Returns:
        tuple:
            - atomized_indep (aa_model.Indep): The atomized protein structure.
            - is_diffused (torch.Tensor): Boolean mask indicating which atoms are diffused.
            - is_masked_seq (torch.Tensor): Boolean mask indicating which atoms have masked sequences.
            - atomizer (aa_model.AtomizeResidues): The atomizer object used for atomization.
    """
    assertpy.assert_that(len(is_res_str_shown)).is_equal_to(indep.length())
    is_atomized = torch.zeros(indep.length()).bool()
    for k in is_atom_str_shown.keys():
        is_atomized[k] = True

    deatomized_state = aa_model.get_atomization_state(indep)
    atomizer = aa_model.AtomizeResidues(deatomized_state, is_atomized)  # These types of inports are confusing, can this be revised? Maybe AtomizeResidues should be in this module
    indep = atomizer.atomize(indep)

    indep.same_chain = indep.same_chain.bool()
    is_diffused, is_masked_seq = create_masks(atomizer, is_res_str_shown, is_res_seq_shown, is_atom_str_shown)
    return indep, is_diffused, is_masked_seq, atomizer

def deatomize_mask(atomizer: aa_model.AtomizeResidues, indep, mask: torch.Tensor) -> torch.Tensor:
    '''
    Takes a mask (of boolean values) for a protein with atomized resiudes
    and deatomizes it. If any atom in the deatomized residue was True, 
    that residue's value will be True in the deatomized mask.

    Inputs
        atomizer: Object to map between atomized and deatomized protein representions.
        mask (L_atomized,): An array of boolean values for the atomized protein.

    Returns
        mask_deatomized (L_deatomized,): An array of boolean values for the deatomized protein.
    '''
    # Don't want to change the input objects
    indep = indep.clone()

    # Shape checks
    assert len(atomizer.atomized_state) == indep.length()
    assert len(atomizer.atomized_state) == len(mask)

    # dtype check
    assert mask.dtype == torch.bool

    # Spoof the mask values as the CA x-coordinate
    indep.xyz = torch.zeros_like(indep.xyz)
    indep.xyz[:, 1, 0] = mask.float()

    # "Deatomize"
    indep_deatomized = atomizer.deatomize(indep)

    # If any atom of a residue is True, so is the whole residue
    mask_deatomized = indep_deatomized.xyz[..., 0].bool().any(-1)

    return mask_deatomized


def get_idx0_post_atomization_from_pre_atomization(L, atomizer=None, consistency_check=True):
    '''
    Returns a list of lists that maps where a pre-atomized residue resides within the post-atomized state
    For non-atomized residues, the result is a 1-element list. But the residue may have changed index!
    For atomized residues, the result is a list of idx0 atom indices (the atoms that belong to the residue)
    Args:
        L (int): Length of indep in the atomized state (which is length of indep in deatomized state if atomizer is None)
        atomizer (AtomizeResidues): The atomizer used, can be None
        consistency_check (bool): Assert that all pre-atomized residues are present post-atomization
    Returns:
        post_from_pre_list (list[list[int]]): Maps location pre-atomized residue to location after atomization [L preatomized]
        post_is_atomized (torch.Tensor[bool]): Is this residue in the atomized indep an atomized atom [L atomized]
    '''

    if atomizer:
        if consistency_check:
            assert L == len(atomizer.atomized_state)

        # Assign post_idx0 to a dictionary of the original positions (coarse_idx0)
        post_from_pre = defaultdict(list)
        for post_idx0, atomized_label in enumerate(atomizer.atomized_state):
            post_from_pre[atomized_label.coarse_idx0].append(post_idx0)

        # Turn that dictionary into a list
        post_from_pre_list = []
        for pre_idx0 in range(len(atomizer.deatomized_state)):
            assert not consistency_check or pre_idx0 in post_from_pre, 'Pre-atomized residue idx0 {pre_idx0} was dropped during atomization!'
            post_from_pre_list.append(post_from_pre[pre_idx0])

        # Mark all atomized residues as atomized
        post_is_atomized = torch.zeros(len(atomizer.atomized_state), dtype=bool)
        for pre_idx0 in torch.where(atomizer.residue_to_atomize)[0]:
            post_is_atomized[post_from_pre_list[pre_idx0]] = True

        return post_from_pre_list, post_is_atomized
    else:
        # Not atomization, the result is a simple list of lists
        return [[x] for x in range(L)], torch.zeros(L, dtype=bool)

    
