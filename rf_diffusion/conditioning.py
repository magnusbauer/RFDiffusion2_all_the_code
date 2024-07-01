import types

import numpy as np
import torch
import logging

from rf_diffusion import aa_model
from rf_diffusion.aa_model import Indep
from rf_diffusion.contigs import ContigMap

from rf2aa.chemical import ChemicalData as ChemData

from typing import Union

logger = logging.getLogger(__name__)

def get_center_of_mass(xyz14, mask):
    assert mask.any(), f'{mask=}'
    points = xyz14[mask]
    return points.mean(dim=0)

LEGACY_TRANSFORMS_TO_IGNORE = ['PopMask']

class PopMask:
    def __call__(self, indep, metadata, masks_1d, **kwargs):
        
        aa_model.pop_mask(indep, masks_1d['pop'])
        masks_1d['input_str_mask'] = masks_1d['input_str_mask'][masks_1d['pop']]
        masks_1d['is_atom_motif'] = aa_model.reindex_dict(masks_1d['is_atom_motif'], masks_1d['pop'])
        metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], masks_1d['pop'])
        metadata['ligand_names'] = np.array(['LIG']*indep.length(),  dtype='<U3')
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

        indep.xyz -= get_center_of_mass(indep.xyz[:,:ChemData().NHEAVYPROT], center_of_mass_mask)
        return dict(
            indep=indep,
            masks_1d=masks_1d,
            **kwargs
        )


class CenterPostTransform:
    """
    A class recentering around the diffused frames. Allows jittering of the center to prevent overfitting
    and memorization of exact placement of ligands / fixed motifs relative to center of mass.
    Must be used after AddConditionalInputs
    Attributes:
        jitter (float): The expected average distance between the center point and the origin
        jitter_clip (float): The maximum amount of distance between the center point and origin. Set this depending on
            the size of jitter, possibly 3 * jitter
        center_type (str): The type of centering to apply. Options are 'is_diffused' and 'is_not_diffused'    
    """    
    def __init__(self, 
                 jitter: float = 0.0,
                 jitter_clip: float = 50.0,
                 center_type: str = 'is_diffused'):
        """
            Centering around diffused atoms for traning stability and design control during inference.
        could solve the problem of the proteins drifting off and ligands being centered. Need to pair with
        extra flags and new parser at inference to specify diffusion origin. This code reduces the requirement
        for the model to learn large center of mass translations. However, it is more prone to memorization 
        of the training data if there are not many examples since data leak occurs under this training 
        regime. This is because the model can memorize the exact placement of the ligands and fixed motifs

        Args:
            jitter (float): The expected average distance between the center point and the origin
            jitter_clip (float): The maximum amount of distance between the center point and origin. Set this depending on
                the size of jitter, possibly 3 * jitter
            center_type (str): The mode of centering. Can be 'is_diffused' or 'is_not_diffused'
        """

        self.jitter = jitter
        self.jitter_clip = jitter_clip
        self.center_type = center_type
        assert center_type in ['is_diffused', 'is_not_diffused'], "must use 'is_diffused' or 'is_not_diffused' for center_type"

    def __call__(self,
                     indep: Indep, 
                     is_diffused: torch.Tensor,
                     origin: torch.Tensor = None,
                     **kwargs) -> dict:
        """
        Computes centering for the indep. Must happen post transform_indep

        Args:
            indep (Indep): the holy Indep
            for_partial_diffusion (bool): whether the model is for partial diffusion
            is_diffused (torch.Tensor): the diffused residues as a boolean mask
            origin (torch.Tensor): the origin to center around. If None, the center of mass is calculated
        """
        if not ((origin is not None) and (self.center_type == 'is_diffused')):
            # Default behavior: calculate center of mass for default case where ground truth of the protein and other targets are available
            center_of_mass_mask = torch.zeros(indep.xyz.shape[:2], dtype=torch.bool)
            if self.center_type == 'is_diffused':
                # CA atoms (position 1) of each frame forms center of rigid translations
                center_of_mass_mask[is_diffused,1] = True 
            elif self.center_type == 'is_not_diffused':
                # In this case center on not diffused if there are any not diffused atoms, otherwise center on diffused
                if torch.sum(~is_diffused) != 0:
                    center_of_mass_mask[~is_diffused,1] = True
                elif torch.sum(~is_diffused) == 0:
                    center_of_mass_mask[is_diffused,1] = True  
                              
            # Calculate center of mass
            origin = get_center_of_mass(indep.xyz, center_of_mass_mask)

        # Calculate jitter amount and add to the origin
        if self.jitter > 0:
            gauss_norm_3d_mean = 1.5956947  # Expected L2 norm of a 3d unit gaussian
            jitter_amount = torch.randn_like(origin) / gauss_norm_3d_mean * self.jitter
            if torch.norm(jitter_amount).item() > self.jitter_clip:
                jitter_amount = jitter_amount / torch.norm(jitter_amount) * self.jitter_clip 
            origin += jitter_amount
                
        # Perform centering
        indep.xyz = indep.xyz - origin[None, None, :]
        
        return kwargs | dict(
            indep=indep,
            is_diffused=is_diffused,
        )


class AddConditionalInputs:
    def __init__(self, p_is_guidepost_example: Union[float,bool], guidepost_bonds):
        """
        Args:
            p_is_guidepost_example (Union[float,bool]): The probability of using guideposts. If a boolean is given, guideposts are always used or never used
        """
        self.p_is_guidepost_example = p_is_guidepost_example
        self.guidepost_bonds = guidepost_bonds

    def __call__(self, indep, metadata, masks_1d, contig_map=types.SimpleNamespace(), **kwargs):
        '''
        Duplicates/masks parts of a protein to create a conditional input.
        i.e. creates guideposts, performs atomization, applies masks.
        '''
        is_res_str_shown = masks_1d['input_str_mask']
        is_atom_str_shown = masks_1d['is_atom_motif']
        is_diffused = masks_1d.get('is_diffused', ~is_res_str_shown)

        pre_transform_length = indep.length()
        # Sample guide posts with probability p_is_guidepost_example, or simply set true or false if p_is_guidepost_example is a boolean
        use_guideposts = self.p_is_guidepost_example if isinstance(self.p_is_guidepost_example, bool) else (torch.rand(1) < self.p_is_guidepost_example).item()
        masks_1d['use_guideposts'] = use_guideposts
        indep, is_diffused, is_masked_seq, atomizer, contig_map.gp_to_ptn_idx0 = aa_model.transform_indep(indep, is_diffused, is_res_str_shown, is_atom_str_shown, use_guideposts, guidepost_bonds=self.guidepost_bonds, metadata=metadata)

        masks_1d['is_masked_seq']=is_masked_seq
        # The previous code here was wrong. All is_gp are not necessarily contiguously at the end.
        #  Atomized residues come after gp residues (and aren't necessarily gp themselves)
        aa_model.assert_valid_seq_mask(indep, is_masked_seq)
        
        return kwargs | dict(
            indep=indep,
            is_diffused=is_diffused,
            is_masked_seq=is_masked_seq,
            atomizer=atomizer,
            metadata=metadata,
            masks_1d=masks_1d,
            contig_as_guidepost=use_guideposts,
            contig_map=contig_map,
        )

def get_contig_map(indep, input_str_mask, is_atom_motif):

    motif_resis = sorted(list(set(
        indep.idx[input_str_mask].tolist() +
        indep.idx[list(is_atom_motif.keys())].tolist()
    )))

    contigs = []
    for ch, i in zip(indep.chains(), indep.idx):
        if i in motif_resis:
            contigs.append(f'{ch}{i}-{i}')
        else:
            contigs.append('1-1')
    contig_atoms = {}
    for i, atom_names in is_atom_motif.items():
        contig_atoms[f'{indep.chains()[i]}{indep.idx[i]}'] = atom_names

    contig_map_args = {
        'parsed_pdb': {
            'seq': indep.seq.numpy(),
            'pdb_idx': [(ch,int(i)) for ch, i in zip(indep.chains(), indep.idx)],
        },
        'contigs': [','.join(contigs)],
        'contig_atoms': str({idx: ','.join(atom_names) for idx, atom_names in contig_atoms.items()}),
    }
    return ContigMap(**contig_map_args)

class ReconstructContigMap:

    def __call__(self, indep, masks_1d, **kwargs):
        contig_map = get_contig_map(indep, masks_1d['input_str_mask'], masks_1d['is_atom_motif'])
        return dict(
            contig_map=contig_map,
            indep=indep,
            masks_1d=masks_1d,
        ) | kwargs
