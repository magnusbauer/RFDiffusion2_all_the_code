import copy
import unittest

import assertpy
import torch
from icecream import ic

import rf_diffusion.inference.utils as iu
from rf_diffusion import contigs
from rf_diffusion import atomize
from rf2aa import tensor_util
from rf_diffusion import aa_model
from rf_diffusion import test_utils
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
            target_feats = iu.process_target(test_pdb)
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

            o, is_diffused, is_seq_masked, atomizer, contig_map.gp_to_ptn_idx0 = aa_model.transform_indep(o, masks_1d['input_str_mask'], masks_1d['input_seq_mask'], masks_1d['is_atom_motif'], masks_1d['can_be_gp'], conf.inference.contig_as_guidepost, conf.guidepost_bonds, metadata=metadata)

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
            is_res_seq_shown = is_res_str_shown.clone()
            is_res_seq_shown[indep.is_sm] = True
            can_be_gp = is_res_str_shown.clone()
            can_be_gp[list(is_atom_str_shown)] = True
            n_res_shown = is_res_str_shown.sum()
            indep, is_diffused, is_masked_seq, atomizer, gp_to_ptn_idx0 = aa_model.transform_indep(indep, is_res_str_shown, is_res_seq_shown, is_atom_str_shown, can_be_gp, True, metadata=metadata)

            n_ligand = indep_init.is_sm.sum()
            n_motif = sum(len(v) for v in is_atom_str_shown.values()) + n_res_shown
            n_valine = 7
            assertpy.assert_that(indep.length()).is_equal_to(n_valine + indep_init.length() + n_res_shown)
            assertpy.assert_that((~is_diffused).sum()).is_equal_to(n_motif)
            assertpy.assert_that((~is_masked_seq).sum()).is_equal_to(n_valine + n_res_shown + n_ligand)
            assertpy.assert_that(indep.is_sm.sum()).is_equal_to(n_valine + n_ligand).described_as('Number of atom tokens in indep not equal to number of atoms in atomized residue + ligand')
            tensor_util.assert_equal(indep.bond_feats, indep.bond_feats.T)
            n_gp_atomized = n_valine + n_res_shown
            check_is_gp = torch.zeros(indep.length()).bool()
            check_is_gp[-n_gp_atomized:] = True
            assert (check_is_gp == indep.is_gp).all()
            is_gp_receptor = ~indep.is_gp * ~indep.is_sm
            other_bonds = indep.bond_feats[indep.is_gp][:, is_gp_receptor]
            assert torch.all(other_bonds == 7), other_bonds
            CB = 4
            CG1 = 5
            assertpy.assert_that(indep.human_readable_seq()).is_equal_to(want_seq)
            atomized_res_bonds = indep.bond_feats[indep.is_sm * indep.is_gp][:, indep.is_sm * indep.is_gp]
            assertpy.assert_that(atomized_res_bonds[CB, CG1].item()).is_equal_to(1)

            self.check_slicing(indep)
            
            # Deatomize
            indep_deatomized = atomizer.deatomize(indep)
            n_gp = len(is_atom_str_shown) + n_res_shown
            check_is_gp = torch.zeros(indep_deatomized.length()).bool()
            check_is_gp[-n_gp:] = True
            assert (check_is_gp == indep_deatomized.is_gp).all()
            aa_model.pop_mask(indep_deatomized, ~indep_deatomized.is_gp)

            diff = test_utils.cmp_pretty(indep_deatomized, indep_init)
            if diff:
                print(diff)
                self.fail(f'{contig_kwargs=} {diff=}')


            # Check mapping
            post_from_pre, is_atomized = aa_model.generate_pre_to_post_transform_indep_mapping(indep, atomizer, gp_to_ptn_idx0)
            for i_pre, is_post in enumerate(post_from_pre):
                for i_post in is_post:
                    if is_atomized[i_post]:
                        assert not indep_deatomized.is_sm[i_pre], 'A ligand atom got marked as atomized'
                        assert indep.is_sm[i_post], 'A residue was marked as atomized but is not an atom now'
                    else:
                        assert indep.seq[i_post] == indep_deatomized.seq[i_pre], 'Mapping function error'


    def check_slicing(self, indep):
        '''
        This would be best as a separate test but generating indeps that are both atomized and have gp residues is non-trivial
        Called from test_atomized_placement_agnostic

        Args:
            indep (indep): Any indep to check for slicing and catting compatibility
        '''
        indep = indep.clone()

        # checker slice is literally every other residue goes left/right
        checker_slice = torch.zeros(indep.length(), dtype=bool)
        checker_slice[::2] = True

        left, _ = aa_model.slice_indep(indep, checker_slice, break_chirals=True)
        right, _ = aa_model.slice_indep(indep, ~checker_slice, break_chirals=True)

        seq = copy.deepcopy(indep.seq)
        seq[checker_slice] = left.seq
        seq[~checker_slice] = right.seq
        assert torch.allclose(seq, indep.seq), 'failed checker slice'

        xyz = copy.deepcopy(indep.xyz)
        xyz[checker_slice] = left.xyz
        xyz[~checker_slice] = right.xyz
        assert torch.allclose(xyz, indep.xyz, equal_nan=True), 'failed checker slice'

        idx = copy.deepcopy(indep.idx)
        idx[checker_slice] = left.idx
        idx[~checker_slice] = right.idx
        assert torch.allclose(idx, indep.idx), 'failed checker slice'

        assert torch.allclose(indep.bond_feats[checker_slice][:,checker_slice], left.bond_feats), 'failed checker slice'
        assert torch.allclose(indep.bond_feats[~checker_slice][:,~checker_slice], right.bond_feats), 'failed checker slice'

        # chirals are messed up when you slice like this

        terminus_type = copy.deepcopy(indep.terminus_type)
        terminus_type[checker_slice] = left.terminus_type
        terminus_type[~checker_slice] = right.terminus_type
        assert torch.allclose(terminus_type, indep.terminus_type), 'failed checker slice'

        # extra_t1d isn't sliced
        
        is_gp = copy.deepcopy(indep.is_gp)
        is_gp[checker_slice] = left.is_gp
        is_gp[~checker_slice] = right.is_gp
        assert torch.allclose(is_gp, indep.is_gp), 'failed checker slice'


        # The half slice cuts it in half so we can use cat_indep
        half_slice = torch.zeros(indep.length(), dtype=bool)
        half_slice[:indep.length()//2] = True

        left, _ = aa_model.slice_indep(indep, half_slice, break_chirals=True)
        right, _ = aa_model.slice_indep(indep, ~half_slice, break_chirals=True)

        new_indep = aa_model.cat_indeps([left, right], indep.same_chain)
        new_indep.bond_feats = indep.bond_feats
        new_indep.chirals = indep.chirals


        diff = test_utils.cmp_pretty(indep, new_indep)
        if diff:
            print(diff)
            self.fail(f'{diff=}')


        # Also test open_indep()
        open_indep = indep.clone()
        with aa_model.open_indep(open_indep, checker_slice, break_chirals=True) as _:
            pass
        open_indep.bond_feats = indep.bond_feats
        open_indep.chirals = indep.chirals

        diff = test_utils.cmp_pretty(indep, open_indep)
        if diff:
            print(diff)
            self.fail(f'{diff=}')



if __name__ == '__main__':
        unittest.main()
