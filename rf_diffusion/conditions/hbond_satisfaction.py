import torch

from rf_diffusion import chemical
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion import calc_hbonds
from rf_diffusion.calc_hbonds import HBMAP_OTHER_IDX0, HBMAP_OTHER_IATOM, HBMAP_OUR_IATOM, HBMAP_WE_ARE_DONOR
from rf_diffusion.nucleic_compatibility_utils import get_resi_type_mask
from rf_diffusion import aa_model
from rf_diffusion.conditions.util import expand_1d_atomized_ok_gp_not

from rf_diffusion.ppi import downsample_bool_mask

import random


class HBondSatisfactionApplierBase:
    '''
    Base class for the appliers in this file

    Basically, we need a way to store state in conditions dict that then performs somewhat complex logic in ExpandConditionsDict
    This base class gives the framework to do such a thing
    '''

    def __init__(self):
        pass

    def pop_mask(self, pop):
        '''
        Called from pop_conditions_dict() to tell us that the indep is getting smaller

        Args:
            pop (torch.Tensor[bool]): The residues of indep that will remain after the operation [pre_L]
        '''
        pass

    def generate_conditions_on_atomized_indep(self, indep, post_idx_from_pre_idx, is_atomized):
        '''
        Not to be overriden. The catcher function that gets called from ExpandConditionsDict
        Adding more args to this function is fine

        Args:
            indep (Indep): atomized indep with guideposts
            pre_to_post_transform (list[list[int]]): A list mapping the original indep's numbering (key) to a list of where those residues are post (value) [L pre]
            is_atomized_res (torch.Tensor[bool]): In the post-transform indep, was this now atom originally a residue that was atomized

        Returns:
            new_conditions (dict): The new conditions
        '''

        return self.generate_conditions(indep, post_idx_from_pre_idx, is_atomized)


    def generate_conditions(self, indep, post_idx_from_pre_idx, is_atomized):
        '''
        Override this!

        Generates the actual conditions during transform_indep. Receives an atomized indep with guideposts

        Args:
            indep (Indep): atomized indep with guideposts
            pre_to_post_transform (list[list[int]]): A list mapping the original indep's numbering (key) to a list of where those residues are post (value) [L pre]
            is_atomized_res (torch.Tensor[bool]): In the post-transform indep, was this now atom originally a residue that was atomized

        Returns:
            new_conditions (dict): The new conditions
        '''

        assert False, 'HBondSatisfactionApplierBase.generate_conditions() was called. You should overwrite this!'



class HBondTargetSatisfactionInferenceTransform:
    '''
    Transform to do the target_satisfaction_condition during inference

    See HBondTargetSatisfactionInferenceApplier
    '''

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def __call__(self, indep, conditions_dict, contig_map, **kwargs):
        conditions_dict['target_hbond_satisfaction'] = HBondTargetSatisfactionInferenceApplier(indep, contig_map, **self.init_kwargs)

        return dict(
            indep=indep,
            conditions_dict=conditions_dict,
            contig_map=contig_map,
            **kwargs
            )


class HBondTargetSatisfactionInferenceApplier(HBondSatisfactionApplierBase):
    '''
    The Applier that does the target_hbond_satisfaction at inference time

    Allows users to specify which residues on the target they want specific kinds of binder h-bonds to
    '''

    def __init__(
            self,
            indep,
            contig_map,
            binder_bb_donates_to_target_bb='',
            binder_bb_accepts_from_target_bb='',
            binder_sc_donates_to_target_bb='',
            binder_sc_accepts_from_target_bb='',
            binder_bb_donates_to_target_sc='',
            binder_bb_accepts_from_target_sc='',
            binder_sc_donates_to_target_sc='',
            binder_sc_accepts_from_target_sc='',
            binder_bb_donates_to_target_atom='',
            binder_bb_accepts_from_target_atom='',
            binder_sc_donates_to_target_atom='',
            binder_sc_accepts_from_target_atom='',
            binder_HIS_accepts_from_target_bb='',
            binder_HIS_accepts_from_target_sc='',
            binder_HIS_accepts_from_target_atom='',
        ):
        '''
        Args:
            indep (indep): Indep before atomization
            contig_map (ContigMap): Contig map
            ... (str): The individual kinds of h-bonds you can ask for. Ones that say _atom accept atomized selectors too
        '''
        # Doing it in this wacky way so that people will get errors if they type something on the commandline wrong
        self.binder_bb_donates_to_target_bb = binder_bb_donates_to_target_bb
        self.binder_bb_accepts_from_target_bb = binder_bb_accepts_from_target_bb
        self.binder_sc_donates_to_target_bb = binder_sc_donates_to_target_bb
        self.binder_sc_accepts_from_target_bb = binder_sc_accepts_from_target_bb
        self.binder_bb_donates_to_target_sc = binder_bb_donates_to_target_sc
        self.binder_bb_accepts_from_target_sc = binder_bb_accepts_from_target_sc
        self.binder_sc_donates_to_target_sc = binder_sc_donates_to_target_sc
        self.binder_sc_accepts_from_target_sc = binder_sc_accepts_from_target_sc
        self.binder_bb_donates_to_target_atom = binder_bb_donates_to_target_atom
        self.binder_bb_accepts_from_target_atom = binder_bb_accepts_from_target_atom
        self.binder_sc_donates_to_target_atom = binder_sc_donates_to_target_atom
        self.binder_sc_accepts_from_target_atom = binder_sc_accepts_from_target_atom
        self.binder_HIS_accepts_from_target_bb = binder_HIS_accepts_from_target_bb
        self.binder_HIS_accepts_from_target_sc = binder_HIS_accepts_from_target_sc
        self.binder_HIS_accepts_from_target_atom = binder_HIS_accepts_from_target_atom

        self.polymer_keys = [
            'binder_bb_donates_to_target_bb',
            'binder_bb_accepts_from_target_bb',
            'binder_sc_donates_to_target_bb',
            'binder_sc_accepts_from_target_bb',
            'binder_bb_donates_to_target_sc',
            'binder_bb_accepts_from_target_sc',
            'binder_sc_donates_to_target_sc',
            'binder_sc_accepts_from_target_sc',
            'binder_HIS_accepts_from_target_bb',
            'binder_HIS_accepts_from_target_sc',
        ]

        self.atom_keys = [
            'binder_bb_donates_to_target_atom',
            'binder_bb_accepts_from_target_atom',
            'binder_sc_donates_to_target_atom',
            'binder_sc_accepts_from_target_atom',
            'binder_HIS_accepts_from_target_atom',
        ]

        self.setup_with_indep(indep, contig_map)


    def setup_with_indep(self, indep, contig_map):
        '''
        Called to initialize the applier with the pre-atomized indep

        Args:
            indep (indep): Indep before atomization
            contig_map (ContigMap): Contig map
        '''

        self.polymer_masks = {}
        for polymer_key in self.polymer_keys:
            value = getattr(self, polymer_key)
            if value != '':
                mask = contig_map.res_list_to_mask(value)
                assert not indep.is_sm[mask].any(), f'{key} cannot be specified for small molecules'
                self.polymer_masks[polymer_key] = mask

        self.atom_masks = {}
        for atom_key in self.atom_keys:
            value = getattr(self, atom_key)
            if value != '':
                mask, atomized_dict = contig_map.res_list_to_mask(value, allow_atomized_residues=True)
                assert indep.is_sm[mask].all(), f'{key} may only be specified for small molecules and atomized residues'
                atomized_dict = convert_atomized_dict_to_iatoms(indep, atomized_dict)
                self.atom_masks[atom_key] = (mask, atomized_dict)


    def pop_mask(self, pop):
        '''
        Called from pop_conditions_dict() to tell us that the indep is getting smaller

        Args:
            pop (torch.Tensor[bool]): The residues of indep that will remain after the operation [pre_L]
        '''

        for key in list(self.polymer_masks):
            self.polymer_masks[key] = self.polymer_masks[key][pop]

        old_index_to_new_index = torch.cumsum(pop.long(), axis=0)-1
        for key in list(self.atom_masks):
            mask, atomized_dict = self.polymer_masks[key]
            mask = mask[pop]
            atomized_dict = {old_index_to_new_index[k]:v for k,v in atomized_dict.items()}
            self.atom_masks[key] = (mask, atomized_dict)


    def generate_conditions(self, indep, post_idx_from_pre_idx, is_atomized):
        '''
        Called in ExpandConditionsDict to make the actual conditions with the atomized indep

        Args:
            indep (Indep): atomized indep with guideposts
            pre_to_post_transform (list[list[int]]): A list mapping the original indep's numbering (key) to a list of where those residues are post (value) [L pre]
            is_atomized_res (torch.Tensor[bool]): In the post-transform indep, was this now atom originally a residue that was atomized

        Returns:
            new_conditions (dict[str,torch.Tensor[float]]): The new conditions. See __init__ for the full list
        '''
        
        return_dict = {}

        for key, mask in self.polymer_masks.items():
            to_store = torch.full((len(mask),), torch.nan)
            to_store[mask] = 1.0

            return_dict[key] = expand_1d_atomized_ok_gp_not(indep, to_store, post_idx_from_pre_idx, null_value=torch.nan, key=key)

        for key, (mask, atomized_dict) in self.atom_masks.items():
            to_store = torch.full((len(mask),), torch.nan)
            to_store[mask] = 1.0
            to_store[list(atomized_dict)] = torch.nan

            to_store = expand_1d_atomized_ok_gp_not(indep, to_store, post_idx_from_pre_idx, null_value=torch.nan, key=key)

            for atomized_old_idx0, iatoms in atomized_dict.items():
                new_indices = torch.tensor(post_idx_from_pre_idx[atomized_old_idx0])
                new_indices = new_indices[~indep.is_gp[new_indices]]
                assert len(new_indices) > 1, f"You tried to specify a residue for {key} that didn't actually get atomized. pre-atomized idx0: {atomized_old_idx0}"
                starting_offset = new_indices.min()
                assert new_indices.max() - starting_offset >= iatoms.max(), 'You caught a rare bug. Please tell Brian. A residue was only partially atomized'
                to_store[starting_offset + iatoms] = 1.0
                assert is_atomized[starting_offset + iatoms].all(), 'You caught a rare bug. Please tell Brian. A residue was only partially atomized'

            return_dict[key] = to_store

        return return_dict


def convert_atomized_dict_to_iatoms(indep, atomized_dict):
    '''
    Util function to convert a dictionary of {idx0:[atom_name, atom_name]} to {idx0:[iatom, iatom]}

    Args:
        indep (Indep): pre-atomized indep
        atomized_dict (dict[int,List[str]]): A dictionary of {idx0:[atom_name, atom_name]} to convert

    Returns:
        new_dict (dict[int,torch.Tensor[int]]): A dictionary of {idx0:[iatom, iatom]}
    '''
    new_dict = {}
    for idx0, atoms in atomized_dict.items():
        atoms = [x.strip() for x in atoms]

        atom_names = [x.strip() if x is not None else None for x in ChemData().aa2long[indep.seq[idx0]][:ChemData().NHEAVY]]

        iatoms = []
        for atom in atoms:
            assert atom in atom_names, f'Could not find atom {atom} in residue type {ChemData().one_letter[indep.seq[idx0]]}'
            iatoms.append(atom_names.index(atom))
        new_dict[idx0] = torch.tensor(iatoms)
    return new_dict




class HBondTargetSatisfactionTrainingTransform:
    '''
    Transform to do the target_satisfaction_condition during training

    See HBondTargetSatisfactionTrainingApplier
    '''

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def __call__(self, conditions_dict, **kwargs):
        conditions_dict['target_hbond_satisfaction'] = HBondTargetSatisfactionTrainingApplier(**self.init_kwargs)

        return dict(
            conditions_dict=conditions_dict,
            **kwargs
            )


class HBondTargetSatisfactionTrainingApplier(HBondSatisfactionApplierBase):
    '''
    Applier to generate the target_hbond_satisfaction features during training

    Does all of the actual work inside ExpandConditionsDict (since we need an atomized indep)

    Notes:
        Atomized residues on chain 0 are ignored
        Atomized residues on other chains are counted as small molecules
        Guideposts are ignored
    '''

    def __init__(self, p_bb_sc_cats_shown=0, hb_score_to_count=-0.25, p_tiny_labeling=0.25, dont_label_binder=True, debug_dont_downsample=False):
        '''
        Args:
            p_bb_sc_cats_shown (float): Probability we show the conditions
            hb_score_to_count (float): HBond score (out of -2) for a h-bond to count
            p_tiny_labeling (float): Probability we do an alternate downsampling method where between 1-5 hbonds (and no not-hbonds) are shown
            dont_label_binder (bool): Mark the whole binder as nan
            debug_dont_downsample (bool): For show_dataset. Don't do any downsampling
        '''
        self.p_bb_sc_cats_shown = p_bb_sc_cats_shown
        self.p_tiny_labeling = p_tiny_labeling
        self.hb_score_to_count = hb_score_to_count
        self.dont_label_binder = dont_label_binder
        self.debug_dont_downsample = debug_dont_downsample


    def generate_conditions(self, indep, post_idx_from_pre_idx, is_atomized):
        '''
        Called in ExpandConditionsDict to make the actual conditions with the atomized indep

        Args:
            indep (Indep): atomized indep with guideposts
            pre_to_post_transform (list[list[int]]): A list mapping the original indep's numbering (key) to a list of where those residues are post (value) [L pre]
            is_atomized_res (torch.Tensor[bool]): In the post-transform indep, was this now atom originally a residue that was atomized

        Returns:
            new_conditions (dict[str,torch.Tensor[float]]): The new conditions. Scroll down for the full list
        '''

        return_dict = {}
        if torch.rand(1) < self.p_bb_sc_cats_shown:

            # First we strip out the guideposts and positions with missing sidechain atoms
            dont_calculate_positions = indep.is_gp | chemical.missing_heavy_atoms(indep.xyz, indep.seq).any(axis=-1)
            indep_no_gp, _ = aa_model.slice_indep(indep, ~dont_calculate_positions)

            # Build some util vectors
            L = indep_no_gp.length()
            L_range = torch.arange(L, dtype=int)
            binder_mask = torch.tensor(indep_no_gp.chain_masks()[0])

            # get the hbond_map
            hbond_map, hbond_scores, polar_ret = calc_hbonds.calculate_hbond_map(indep_no_gp, hbond_threshold=self.hb_score_to_count, return_polar_info=True)

            is_sm = indep_no_gp.is_sm
            is_nucl = get_resi_type_mask(indep_no_gp.seq, 'na')
            is_prot = ~is_sm & ~is_nucl

            bb_for_idx0 = torch.zeros((L, ChemData().NHEAVY), dtype=bool)
            bb_for_idx0[is_prot,:4] = True
            bb_for_idx0[is_nucl,:11] = True # 11 backbone atoms for DNA?. The order depends on use_phosephate_frames

            # HBMAP_OTHER_IDX0, HBMAP_OTHER_IATOM, HBMAP_OUR_IATOM, HBMAP_WE_ARE_DONOR

            valid_hbond = hbond_map[:,:,HBMAP_OTHER_IDX0] > -1

            # Generate the primary masks
            our_is_binder = binder_mask.clone()[:,None]
            other_is_binder = valid_hbond & binder_mask[hbond_map[:,:,HBMAP_OTHER_IDX0]]

            our_is_backbone = ~indep_no_gp.is_sm[:,None] & valid_hbond & bb_for_idx0[L_range[:,None],hbond_map[:,:,HBMAP_OUR_IATOM]]
            other_is_backbone = ~indep_no_gp.is_sm[:,None] & valid_hbond & bb_for_idx0[hbond_map[:,:,HBMAP_OTHER_IDX0],hbond_map[:,:,HBMAP_OTHER_IATOM]]

            our_is_atom = indep_no_gp.is_sm[:,None] & valid_hbond
            other_is_atom = valid_hbond & indep_no_gp.is_sm[hbond_map[:,:,HBMAP_OTHER_IDX0]]

            # Still primary masks but defined in terms of NOT one of the other masks
            our_is_target = valid_hbond & ~our_is_binder
            other_is_target = valid_hbond & ~other_is_binder

            our_is_sidechain = valid_hbond & ~our_is_backbone & ~our_is_atom
            other_is_sidechain = valid_hbond & ~other_is_backbone & ~other_is_atom

            # donor acceptor shorthand
            our_is_don = valid_hbond & hbond_map[:,:,HBMAP_WE_ARE_DONOR].bool()
            our_is_acc = valid_hbond & ~hbond_map[:,:,HBMAP_WE_ARE_DONOR].bool()
            other_is_don = ~our_is_don
            other_is_acc = ~our_is_acc

            # histidine
            res_is_HIS = indep_no_gp.seq == ChemData().one_letter.index('H')
            our_is_HIS = res_is_HIS.clone()[:,None]
            other_is_HIS = valid_hbond & res_is_HIS[hbond_map[:,:,HBMAP_OTHER_IDX0]]


            # Secondary masks
            our_binder_bb = our_is_binder & our_is_backbone
            other_binder_bb = other_is_binder & other_is_backbone

            our_binder_sc = our_is_binder & our_is_sidechain
            other_binder_sc = other_is_binder & other_is_sidechain

            our_target_bb = our_is_target & our_is_backbone
            other_target_bb = other_is_target & other_is_backbone

            our_target_sc = our_is_target & our_is_sidechain
            other_target_sc = other_is_target & other_is_sidechain

            # our_binder_atom = our_is_binder & our_is_atom
            # other_binder_atom = other_is_binder & other_is_atom
            our_target_atom = our_is_binder & our_is_atom
            other_target_atom = other_is_target & other_is_atom


            # Tertiary masks
            binder_bb_donating_target_bb =  (our_binder_bb & other_target_bb & our_is_don) | (other_binder_bb & our_target_bb & other_is_don)
            binder_bb_accepting_target_bb = (our_binder_bb & other_target_bb & our_is_acc) | (other_binder_bb & our_target_bb & other_is_acc)

            binder_sc_donating_target_bb =  (our_binder_sc & other_target_bb & our_is_don) | (other_binder_sc & our_target_bb & other_is_don)
            binder_sc_accepting_target_bb = (our_binder_sc & other_target_bb & our_is_acc) | (other_binder_sc & our_target_bb & other_is_acc)

            binder_bb_donating_target_sc =  (our_binder_bb & other_target_sc & our_is_don) | (other_binder_bb & our_target_sc & other_is_don)
            binder_bb_accepting_target_sc = (our_binder_bb & other_target_sc & our_is_acc) | (other_binder_bb & our_target_sc & other_is_acc)

            binder_sc_donating_target_sc =  (our_binder_sc & other_target_sc & our_is_don) | (other_binder_sc & our_target_sc & other_is_don)
            binder_sc_accepting_target_sc = (our_binder_sc & other_target_sc & our_is_acc) | (other_binder_sc & our_target_sc & other_is_acc)

            binder_bb_donating_target_atom =  (our_binder_bb & other_target_atom & our_is_don) | (other_binder_bb & our_target_atom & other_is_don)
            binder_bb_accepting_target_atom = (our_binder_bb & other_target_atom & our_is_acc) | (other_binder_bb & our_target_atom & other_is_acc)

            binder_sc_donating_target_atom =  (our_binder_sc & other_target_atom & our_is_don) | (other_binder_sc & our_target_atom & other_is_don)
            binder_sc_accepting_target_atom = (our_binder_sc & other_target_atom & our_is_acc) | (other_binder_sc & our_target_atom & other_is_acc)

            binder_HIS_accepting_target_bb =  (our_binder_sc & other_target_bb & our_is_acc & our_is_HIS) | (other_binder_sc & our_target_bb & other_is_acc & other_is_HIS)
            binder_HIS_accepting_target_sc = (our_binder_sc & other_target_sc & our_is_acc & our_is_HIS) | (other_binder_sc & our_target_sc & other_is_acc & other_is_HIS)
            binder_HIS_accepting_target_atom =  (our_binder_sc & other_target_atom & our_is_acc & our_is_HIS) | (other_binder_sc & our_target_atom & other_is_acc & other_is_HIS)

            # Accumulate
            binder_bb_donates_to_target_bb = any_hbond_map(hbond_map, binder_bb_donating_target_bb).float()
            binder_bb_accepts_from_target_bb = any_hbond_map(hbond_map, binder_bb_accepting_target_bb).float()
            binder_sc_donates_to_target_bb = any_hbond_map(hbond_map, binder_sc_donating_target_bb).float()
            binder_sc_accepts_from_target_bb = any_hbond_map(hbond_map, binder_sc_accepting_target_bb).float()
            binder_bb_donates_to_target_sc = any_hbond_map(hbond_map, binder_bb_donating_target_sc).float()
            binder_bb_accepts_from_target_sc = any_hbond_map(hbond_map, binder_bb_accepting_target_sc).float()
            binder_sc_donates_to_target_sc = any_hbond_map(hbond_map, binder_sc_donating_target_sc).float()
            binder_sc_accepts_from_target_sc = any_hbond_map(hbond_map, binder_sc_accepting_target_sc).float()
            binder_bb_donates_to_target_atom = any_hbond_map(hbond_map, binder_bb_donating_target_atom).float()
            binder_bb_accepts_from_target_atom = any_hbond_map(hbond_map, binder_bb_accepting_target_atom).float()
            binder_sc_donates_to_target_atom = any_hbond_map(hbond_map, binder_sc_donating_target_atom).float()
            binder_sc_accepts_from_target_atom = any_hbond_map(hbond_map, binder_sc_accepting_target_atom).float()
            binder_HIS_accepts_from_target_bb = any_hbond_map(hbond_map, binder_HIS_accepting_target_bb).float()
            binder_HIS_accepts_from_target_sc = any_hbond_map(hbond_map, binder_HIS_accepting_target_sc).float()
            binder_HIS_accepts_from_target_atom = any_hbond_map(hbond_map, binder_HIS_accepting_target_atom).float()

            # Drop labels on binder side so we don't confuse the network
            #   For one, the donor-acceptor labeling will be backwards
            if self.dont_label_binder:
                binder_bb_donates_to_target_bb[binder_mask] = torch.nan
                binder_bb_accepts_from_target_bb[binder_mask] = torch.nan
                binder_sc_donates_to_target_bb[binder_mask] = torch.nan
                binder_sc_accepts_from_target_bb[binder_mask] = torch.nan
                binder_bb_donates_to_target_sc[binder_mask] = torch.nan
                binder_bb_accepts_from_target_sc[binder_mask] = torch.nan
                binder_sc_donates_to_target_sc[binder_mask] = torch.nan
                binder_sc_accepts_from_target_sc[binder_mask] = torch.nan
                binder_bb_donates_to_target_atom[binder_mask] = torch.nan
                binder_bb_accepts_from_target_atom[binder_mask] = torch.nan
                binder_sc_donates_to_target_atom[binder_mask] = torch.nan
                binder_sc_accepts_from_target_atom[binder_mask] = torch.nan
                binder_HIS_accepts_from_target_bb[binder_mask] = torch.nan
                binder_HIS_accepts_from_target_sc[binder_mask] = torch.nan
                binder_HIS_accepts_from_target_atom[binder_mask] = torch.nan
            
            
            # combine for masking
            together = torch.stack([
                binder_bb_donates_to_target_bb,
                binder_bb_accepts_from_target_bb,
                binder_sc_donates_to_target_bb,
                binder_sc_accepts_from_target_bb,
                binder_bb_donates_to_target_sc,
                binder_bb_accepts_from_target_sc,
                binder_sc_donates_to_target_sc,
                binder_sc_accepts_from_target_sc,
                binder_bb_donates_to_target_atom,
                binder_bb_accepts_from_target_atom,
                binder_sc_donates_to_target_atom,
                binder_sc_accepts_from_target_atom,
                binder_HIS_accepts_from_target_bb,
                binder_HIS_accepts_from_target_sc,
                binder_HIS_accepts_from_target_atom,
            ], axis=0)

            wh_valid = torch.where(~torch.isnan(together))
            N_valid = len(wh_valid[0])

            if self.debug_dont_downsample:
                wh_keep = wh_valid
                print("Not downsampling target hbond satisfaction")
            else:
                if torch.rand(1) < self.p_tiny_labeling:
                    # First, we reduce wh_valid to only positions that actually define a h-bond
                    has_annotation = together[wh_valid].bool()
                    wh_valid = tuple([wh_valid[0][has_annotation], wh_valid[1][has_annotation]])
                    N_valid = len(wh_valid[0])

                    # Now pick up to 5 from within that set
                    N_to_keep = random.randint(min(1, N_valid), min(5, N_valid))
                else:
                    # Otherwise we choose a fraction between 0 and 1 of the whole thing to keep
                    N_to_keep = random.randint(min(1, N_valid), N_valid)

                # Figure out which measurements to keep
                keep_this_one = torch.ones(N_valid, dtype=bool)
                keep_this_one = downsample_bool_mask(keep_this_one, n_to_keep=N_to_keep)
                wh_keep = tuple([wh_valid[0][keep_this_one], wh_valid[1][keep_this_one]])

            # Actually keep the elements we want
            final_together = torch.full(together.shape, torch.nan)
            final_together[wh_keep] = together[wh_keep]

            # Expand back up to gp size
            final_together_w_gp = torch.full((together.shape[0], indep.length()), torch.nan)
            final_together_w_gp[:,~dont_calculate_positions] = final_together

            # Store the results
            (
                final_binder_bb_donates_to_target_bb,
                final_binder_bb_accepts_from_target_bb,
                final_binder_sc_donates_to_target_bb,
                final_binder_sc_accepts_from_target_bb,
                final_binder_bb_donates_to_target_sc,
                final_binder_bb_accepts_from_target_sc,
                final_binder_sc_donates_to_target_sc,
                final_binder_sc_accepts_from_target_sc,
                final_binder_bb_donates_to_target_atom,
                final_binder_bb_accepts_from_target_atom,
                final_binder_sc_donates_to_target_atom,
                final_binder_sc_accepts_from_target_atom,
                final_binder_HIS_accepts_from_target_bb,
                final_binder_HIS_accepts_from_target_sc,
                final_binder_HIS_accepts_from_target_atom,
            ) = final_together_w_gp

            return_dict['binder_bb_donates_to_target_bb'] = final_binder_bb_donates_to_target_bb
            return_dict['binder_bb_accepts_from_target_bb'] = final_binder_bb_accepts_from_target_bb
            return_dict['binder_sc_donates_to_target_bb'] = final_binder_sc_donates_to_target_bb
            return_dict['binder_sc_accepts_from_target_bb'] = final_binder_sc_accepts_from_target_bb
            return_dict['binder_bb_donates_to_target_sc'] = final_binder_bb_donates_to_target_sc
            return_dict['binder_bb_accepts_from_target_sc'] = final_binder_bb_accepts_from_target_sc
            return_dict['binder_sc_donates_to_target_sc'] = final_binder_sc_donates_to_target_sc
            return_dict['binder_sc_accepts_from_target_sc'] = final_binder_sc_accepts_from_target_sc
            return_dict['binder_bb_donates_to_target_atom'] = final_binder_bb_donates_to_target_atom
            return_dict['binder_bb_accepts_from_target_atom'] = final_binder_bb_accepts_from_target_atom
            return_dict['binder_sc_donates_to_target_atom'] = final_binder_sc_donates_to_target_atom
            return_dict['binder_sc_accepts_from_target_atom'] = final_binder_sc_accepts_from_target_atom
            return_dict['binder_HIS_accepts_from_target_bb'] = final_binder_HIS_accepts_from_target_bb
            return_dict['binder_HIS_accepts_from_target_sc'] = final_binder_HIS_accepts_from_target_sc
            return_dict['binder_HIS_accepts_from_target_atom'] = final_binder_HIS_accepts_from_target_atom

        return return_dict

def any_hbond_map(hbond_map, map_mask):
    '''
    Util function to find the idx0 of the true elements of map_mask within hbond_map

    Args:
        hbond_map (torch.Tensor[int]): A standard hbond_map [L,?,4]
        map_mask (torch.Tensor[bool]): A mask of which hbonds to look at [L,?]

    Returns:
        output_vec (torch.Tensor[bool]): The idx0 where there were hbonds in the mask
    '''

    output_vec = torch.zeros(len(hbond_map), dtype=bool)

    wh_map_mask = torch.where(map_mask)
    our_idx0 = wh_map_mask[0]
    other_idx0 = hbond_map[wh_map_mask][:,HBMAP_OTHER_IDX0]
            
    # torch handles this operation just fine despite there being multiple our_idx0 with the same value
    output_vec[our_idx0] = True
    output_vec[other_idx0] = True
            
    return output_vec



def get_target_hbond_satisfaction_keys_for_t1d():
    '''
    Get the list of keys for the target_hbond_satisfaction_cond

    Return:
        keys (list[str]): The list of keys
    '''
    keys = [
        'binder_bb_donates_to_target_bb',
        'binder_bb_accepts_from_target_bb',
        'binder_sc_donates_to_target_bb',
        'binder_sc_accepts_from_target_bb',
        'binder_bb_donates_to_target_sc',
        'binder_bb_accepts_from_target_sc',
        'binder_sc_donates_to_target_sc',
        'binder_sc_accepts_from_target_sc',
        'binder_bb_donates_to_target_atom',
        'binder_bb_accepts_from_target_atom',
        'binder_sc_donates_to_target_atom',
        'binder_sc_accepts_from_target_atom',

        'binder_HIS_accepts_from_target_bb',
        'binder_HIS_accepts_from_target_sc',
        'binder_HIS_accepts_from_target_atom',
    ]
    return keys


def get_hbond_target_satisfaction_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    '''
    See get_hbond_target_satisfaction_conditioning()
    '''
    return get_hbond_target_satisfaction_conditioning(indep, feature_conf, **kwargs)


def get_hbond_target_satisfaction_conditioning(indep, feature_conf, target_hbond_satisfaction=None, **kwargs):
    '''
    Generates the hotspot and antihotspot features for training and inference

    Args:
        indep (Indep): indep
        feature_conf (OmegaConf): The configuration for this feature
        target_hbond_satisfaction (dict[str,torch.Tensor]): The conditions_dict object created by the appliers

    Returns:
        dict:
            t1d (torch.Tensor[bool]): mask_key, value_key for all 15 keys [L,30]
    '''

    L = indep.length()

    keys = get_target_hbond_satisfaction_keys_for_t1d()

    if target_hbond_satisfaction is None:
        target_hbond_satisfaction = {}

    # Even if none of these are present we still need to make extra_t1d
    for key in keys:
        if key not in target_hbond_satisfaction:
            target_hbond_satisfaction[key] = torch.full((L,), torch.nan)
        else:
            assert len(target_hbond_satisfaction[key]) == L, f'{key} length does not match indep.length(): {len(target_hbond_satisfaction[key])} {L}'

    ret_stack = []
    for key in keys:
        values = target_hbond_satisfaction[key]
        valid_mask = ~torch.isnan(values)
        to_store = torch.zeros(L)
        to_store[valid_mask] = values[valid_mask]

        ret_stack.append(valid_mask.float())
        ret_stack.append(to_store.float())

    return {'t1d':torch.stack(ret_stack, axis=-1)}







