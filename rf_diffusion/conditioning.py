import numpy as np
import torch
import logging

from rf_diffusion import aa_model

def get_center_of_mass(xyz14, mask):
    assert mask.any()
    points = xyz14[mask]
    return points.mean(dim=0)

class PopMask:
    def __call__(self, indep, metadata, masks_1d, **kwargs):
        aa_model.pop_mask(indep, masks_1d['pop'])
        masks_1d['input_str_mask'] = masks_1d['input_str_mask'][masks_1d['pop']]
        masks_1d['is_atom_motif'] = aa_model.reindex_dict(masks_1d['is_atom_motif'], masks_1d['pop'])
        metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], masks_1d['pop'])
        is_atom_str_shown = masks_1d['is_atom_motif']
        is_atom_str_shown
        return dict(
            indep=indep,
            metadata=metadata,
            masks_1d=masks_1d,
            **kwargs
        )

class Center:
    def __call__(self, indep, masks_1d, **kwargs):

        is_res_str_shown = masks_1d['input_str_mask']
        is_atom_str_shown = masks_1d['is_atom_motif']

        # For debugging
        is_sm_shown = indep.is_sm[is_res_str_shown.nonzero()[:, 0]]
        n_atomic_motif = is_sm_shown.sum()
        n_residue_motif = (~is_sm_shown).sum()
        logging.debug(
            f'{n_atomic_motif=} {n_residue_motif=} {is_atom_str_shown=} {is_res_str_shown.nonzero()[:, 0]=}', 
        )

        motif_atom_name_by_res_idx = {}
        for i in is_res_str_shown.nonzero()[:,0]:
            motif_atom_name_by_res_idx[i] = aa_model.CA_ONLY
        motif_atom_name_by_res_idx.update(is_atom_str_shown)
        is_motif14 = aa_model.make_is_motif14(indep.seq, motif_atom_name_by_res_idx)
        center_of_mass_mask = is_motif14
        if not center_of_mass_mask.any():
            # Unconditional case
            center_of_mass_mask[:, 1] = True

        indep.xyz -= get_center_of_mass(indep.xyz, center_of_mass_mask)
        return dict(
            indep=indep,
            masks_1d=masks_1d,
            **kwargs
        )


class AddConditionalInputs:
    def __init__(self, p_is_guidepost_example, guidepost_bonds):
        self.p_is_guidepost_example = p_is_guidepost_example
        self.guidepost_bonds = guidepost_bonds

    def __call__(self, indep, metadata, masks_1d, **kwargs):
        '''
        Duplicates/masks parts of a protein to create a conditional input.
        i.e. creates guideposts, performs atomization, applies masks.
        '''
        is_res_str_shown = masks_1d['input_str_mask']
        is_atom_str_shown = masks_1d['is_atom_motif']

        pre_transform_length = indep.length()
        use_guideposts = (torch.rand(1) < self.p_is_guidepost_example).item()
        masks_1d['use_guideposts'] = use_guideposts
        indep, is_diffused, is_masked_seq, atomizer, _ = aa_model.transform_indep(indep, is_res_str_shown, is_atom_str_shown, use_guideposts, guidepost_bonds=self.guidepost_bonds, metadata=metadata)

        # HACK: gp indices may be lost during atomization, so we assume they are at the end of the protein.
        is_gp = torch.full((indep.length(),), True)
        is_gp[:pre_transform_length] = False
        aa_model.assert_valid_seq_mask(indep, is_masked_seq)
        
        return dict(
            indep=indep,
            is_diffused=is_diffused,
            is_masked_seq=is_masked_seq,
            is_gp=is_gp,
            atomizer=atomizer,
            metadata=metadata,
            masks_1d=masks_1d,
            **kwargs
        )
