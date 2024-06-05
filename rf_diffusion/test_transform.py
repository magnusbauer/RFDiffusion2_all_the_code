import copy
import unittest

import assertpy
import torch
from icecream import ic

import inference.utils
import contigs
import atomize
from rf2aa import tensor_util
import aa_model
import test_utils
import numpy as np
from rf_diffusion.chemical import ChemicalData as ChemData
import rf_diffusion.kinematics
ic.configureOutput(includeContext=True)
from unittest import mock

class TestTransform(unittest.TestCase):
    
    def test_atomized_placement_agnostic(self):
        testcases = [
            (
                {'contigs': ['A508-511']},
                None,
                {2: ['CB', 'CG1', 'CG2']},
                [],
                ['ASP', 'GLN', 'VAL', 'PRO', 'N', 'C', 'C', 'O', 'C', 'C', 'C']
            ),
            (
                {'contigs': ['A508-511']},
                'LG1',
                {2: ['CB', 'CG1', 'CG2']},
                [1],
                ['ASP', 'GLN', 'VAL', 'PRO', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'GLN', 'N', 'C', 'C', 'O', 'C', 'C', 'C']
            ),
        ]
        for contig_kwargs, ligand, is_atom_str_shown, res_str_shown_idx, want_seq in testcases:
            test_pdb = 'benchmark/input/gaa.pdb'
            target_feats = inference.utils.process_target(test_pdb)
            contig_map =  contigs.ContigMap(target_feats,
                                        **contig_kwargs
                                        )
            indep, metadata = aa_model.make_indep(test_pdb, ligand=ligand, return_metadata=True)
            conf = test_utils.construct_conf(inference=True)             

            # Legacy insert_contig requires centering using a non-standard centering strategy, so we recreate for comptability, in the future, can be removed if goldens are overwritten
            def get_init_xyz_id(xyz_t, *args, **kwargs):
                return xyz_t
            with mock.patch.object(rf_diffusion.kinematics, 'get_init_xyz', new=get_init_xyz_id):
                adaptor = aa_model.Model(conf)
                o, masks_1d = adaptor.insert_contig_pre_atomization(indep, contig_map, metadata, for_partial_diffusion=False)
            # Create the init xyz values with the legacy centering
            o.xyz = rf_diffusion.kinematics.get_init_xyz(o.xyz[None, None], o.is_sm, center=True).squeeze() 

            pre_transform_length = o.length()
            o, is_diffused, is_seq_masked, atomizer, contig_map.gp_to_ptn_idx0 = aa_model.transform_indep(o, masks_1d['is_diffused'], masks_1d['input_str_mask'], masks_1d['is_atom_motif'], conf.inference.contig_as_guidepost, 'anywhere', conf.guidepost_bonds, metadata=metadata)

            # HACK: gp indices may be lost during atomization, so we assume they are at the end of the protein.
            is_gp = torch.full((o.length(),), True)
            is_gp[:pre_transform_length] = False
            o.is_gp = is_gp

            sm_ca = o.xyz[o.is_sm, 1]
            o.xyz[o.is_sm,:3] = sm_ca[...,None,:]
            o.xyz[o.is_sm] += ChemData().INIT_CRDS 

            contig_map.ligand_names = np.full(o.length(), '', dtype='<U3')
            contig_map.ligand_names[contig_map.hal_idx0.astype(int)] = metadata['ligand_names'][contig_map.ref_idx0] 
            indep = o

            # Now check atomization

            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)
            is_res_str_shown = torch.zeros(indep.length()).bool()
            is_res_str_shown[res_str_shown_idx] = True
            n_res_shown = is_res_str_shown.sum()
            indep, is_diffused, is_masked_seq, atomizer, _ = aa_model.transform_indep(indep, ~is_res_str_shown, is_res_str_shown, is_atom_str_shown, True, metadata=metadata)

            n_ligand = indep_init.is_sm.sum()
            n_motif = sum(len(v) for v in is_atom_str_shown.values()) + n_res_shown
            n_valine = 7
            assertpy.assert_that(indep.length()).is_equal_to(n_valine + indep_init.length() + n_res_shown)
            assertpy.assert_that((~is_diffused).sum()).is_equal_to(n_motif)
            assertpy.assert_that((~is_masked_seq).sum()).is_equal_to(n_valine + n_res_shown + n_ligand)
            assertpy.assert_that(indep.is_sm.sum()).is_equal_to(n_valine + n_ligand)
            tensor_util.assert_equal(indep.bond_feats, indep.bond_feats.T)
            n_gp_atomized = n_valine + n_res_shown
            is_gp = torch.zeros(indep.length()).bool()
            is_gp[-n_gp_atomized:] = True
            is_gp_receptor = ~is_gp.clone() * ~indep.is_sm
            other_bonds = indep.bond_feats[is_gp][:, is_gp_receptor]
            assert torch.all(other_bonds == 7), other_bonds
            CB = 4
            CG1 = 5
            assertpy.assert_that(indep.human_readable_seq()).is_equal_to(want_seq)
            atomized_res_bonds = indep.bond_feats[indep.is_sm * is_gp][:, indep.is_sm * is_gp]
            assertpy.assert_that(atomized_res_bonds[CB, CG1].item()).is_equal_to(1)
            
            # Deatomize
            indep_deatomized = atomizer.deatomize(indep)
            n_gp = len(is_atom_str_shown) + n_res_shown
            is_gp = torch.zeros(indep_deatomized.length()).bool()
            is_gp[-n_gp:] = True
            aa_model.pop_mask(indep_deatomized, ~is_gp)
            delattr(indep_init, 'is_gp')

            diff = test_utils.cmp_pretty(indep_deatomized, indep_init)
            if diff:
                print(diff)
                self.fail(f'{contig_kwargs=} {diff=}')


if __name__ == '__main__':
        unittest.main()
