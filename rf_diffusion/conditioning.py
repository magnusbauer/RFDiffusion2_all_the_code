import numpy as np
import torch
import logging

from rf_diffusion import aa_model


def add_conditional_inputs(indep, metadata, masks_1d, conditioning_cfg):

    aa_model.pop_mask(indep, masks_1d['pop'])
    # atom_mask = atom_mask[masks_1d['pop']]
    masks_1d['input_str_mask'] = masks_1d['input_str_mask'][masks_1d['pop']]
    masks_1d['is_atom_motif'] = aa_model.reindex_dict(masks_1d['is_atom_motif'], masks_1d['pop'])
    metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], masks_1d['pop'])

    is_res_str_shown = masks_1d['input_str_mask']
    is_atom_str_shown = masks_1d['is_atom_motif']

    # Cast to non-tensor
    is_atom_str_shown = is_atom_str_shown or {}
    def maybe_item(i):
        if hasattr(i, 'item'):
            return i.item()
        return i
    if is_atom_str_shown:
        is_atom_str_shown = {maybe_item(res_i):v for res_i, v in is_atom_str_shown.items()}
    
    logging.debug(
        f'{is_atom_str_shown=}, {is_res_str_shown=}'
    )

    pre_transform_length = indep.length()
    use_guideposts = (torch.rand(1) < conditioning_cfg["P_IS_GUIDEPOST_EXAMPLE"]).item()
    masks_1d['use_guideposts'] = use_guideposts
    indep, is_diffused, is_masked_seq, atomizer, _ = aa_model.transform_indep(indep, is_res_str_shown, is_atom_str_shown, use_guideposts, guidepost_bonds=conditioning_cfg.guidepost_bonds, metadata=metadata)

    # HACK: gp indices may be lost during atomization, so we assume they are at the end of the protein.
    is_gp = torch.full((indep.length(),), True)
    is_gp[:pre_transform_length] = False
    aa_model.assert_valid_seq_mask(indep, is_masked_seq)
    return indep, is_diffused, is_masked_seq, is_gp, atomizer


def center_motif(indep, metadata, masks_1d, conditioning_cfg):
    pass

def center_motif_even_unconditional(indep, metadata, masks_1d, conditioning_cfg):
    pass