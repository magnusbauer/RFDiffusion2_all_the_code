import unittest
import pytest
from unittest import mock
import subprocess
from pathlib import Path
from inspect import signature
from io import StringIO
import assertpy

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from icecream import ic
import torch
import numpy as np
import pickle

import test_utils
import run_inference
from functools import partial
from rf2aa import tensor_util
import rf2aa.chemical
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
import rf2aa.loss
from rf_diffusion.inference import model_runners
from rf_diffusion import aa_model
from rf_diffusion import inference
import rf_se3_diffusion.data.utils as du
ic.configureOutput(includeContext=True)

REWRITE = False
def infer(overrides):
    conf = construct_conf(overrides)
    run_inference.main(conf)
    p = Path(conf.inference.output_prefix + '_0-atomized-bb-True.pdb')
    return p, conf

def construct_conf(overrides):
    overrides = overrides + ['inference.cautious=False', 'inference.design_startnum=0']
    initialize(version_base=None, config_path="config/inference", job_name="test_app")
    conf = compose(config_name='aa_small.yaml', overrides=overrides, return_hydra_config=True)
    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
    HydraConfig.instance().set_config(conf)
    conf = compose(config_name='aa_small.yaml', overrides=overrides)
    return conf

def get_trb(conf):
    path = conf.inference.output_prefix + '_0-atomized-bb-True.trb'
    return np.load(path,allow_pickle=True)

class TestRegression(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Example regression test.
    @pytest.mark.slow
    @pytest.mark.nondeterministic
    @pytest.mark.generates_golden
    def test_t2(self):
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'T2', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)
    
    @pytest.mark.generates_golden
    def test_inference_rfi(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=True',
        ])

        func_sig = signature(RoseTTAFoldModule.forward)
        fake_forward = mock.patch.object(RoseTTAFoldModule, "forward", autospec=True)

        def side_effect(self, *args, **kwargs):
            ic("mock forward", type(self), side_effect.call_count)
            side_effect.call_count += 1
            return fake_forward.temp_original(self, *args, **kwargs)
        side_effect.call_count = 0

        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect
            run_inference.main(conf)

            mapped_calls = []
            for args, kwargs in mock_forward.call_args_list:
                args = (None,) + args[1:]
                argument_binding = func_sig.bind(*args, **kwargs)
                argument_map = argument_binding.arguments
                argument_map = tensor_util.cpu(argument_map)
                mapped_calls.append(argument_map)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)
        
    @pytest.mark.slow
    @pytest.mark.generates_golden
    def test_partial_sidechain(self):
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_3',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        # The network exhibits chaotic behavior when coordinates corresponding to chiral gradients are updated,
        # so this primarily checks that inference runs and produces the expected shapes, rather than coordinate
        # values, which vary wildly across CPU architectures.
        cmp = partial(tensor_util.cmp, atol=2, rtol=0)
        test_utils.assert_matches_golden(self, 'partial_sc', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)
    
    @pytest.mark.slow
    @pytest.mark.nondeterministic
    @pytest.mark.generates_golden
    def test_10res_self_conditioning(self):
        '''
        This test should be used to write the golden 10res_self_conditioning, which can then be picked up by
        the following test which uses the refactored model_runner.Sampler: model_runners.FlowMatching, to assert
        that it is close to a pure refactor.
        '''
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_10res_self_conditioning',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
        test_utils.assert_matches_golden(self, '10res_self_conditioning', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_10res_flow_matching(self):
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_10res_flow_matching',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=FlowMatching",
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=0.25, rtol=0)
        test_utils.assert_matches_golden(self, '10res_self_conditioning', pdb_contents, rewrite=False, custom_comparator=cmp)

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    @pytest.mark.generates_golden
    def test_10res_batch_optimal_transport_false(self):
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_10res_batch_optimal_transport_false',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=FlowMatching",
            "+diffuser.batch_optimal_transport=False",
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
        test_utils.assert_matches_golden(self, '10res_batch_optimal_transport_false', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)


#     @pytest.mark.slow
#     @pytest.mark.nondeterministic
#     def test_10res_batch_optimal_transport_false_make_conditional(self):
#         run_inference.make_deterministic()
#         pdb, _ = infer([
#             'diffuser.T=10',
#             'inference.num_designs=1',
#             'inference.output_prefix=tmp/test_10res_batch_optimal_transport_false_make_conditional',
#             "contigmap.contigs=['9,A518-518,1']",
#             "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
#             "inference.model_runner=FlowMatching_make_conditional",
#             "+diffuser.batch_optimal_transport=False",
#         ])
#         pdb_contents = inference.utils.parse_pdb(pdb)
#         cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
#         test_utils.assert_matches_golden(self, '10res_batch_optimal_transport_false', pdb_contents, rewrite=False, custom_comparator=cmp)

# # #  TO UNCOMMENT WHEN WORKING
#     @pytest.mark.slow
#     @pytest.mark.nondeterministic
#     def test_10res_batch_optimal_transport_false_classifier_guidance(self):
#         run_inference.make_deterministic()
#         pdb, _ = infer([
#             'diffuser.T=10',
#             'inference.num_designs=1',
#             'inference.output_prefix=tmp/test_10res_batch_optimal_transport_false_classifier_guidance',
#             "contigmap.contigs=['9,A518-518,1']",
#             "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
#             "+diffuser.batch_optimal_transport=False",
#             "inference.model_runner=ClassifierFreeGuidance",
#         ])
#         pdb_contents = inference.utils.parse_pdb(pdb)
#         cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
#         test_utils.assert_matches_golden(self, '10res_batch_optimal_transport_false', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)
    

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    @pytest.mark.generates_golden
    def test_guidepost(self):
        '''
        Tests that predictions with guide posts flag are correct.
        '''
        run_inference.make_deterministic()
        pdb, conf = infer([
            'diffuser.T=10',
            'inference.input_pdb=test_data/1qys.pdb',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_gp',
            'inference.contig_as_guidepost=True',
            "contigmap.contigs=['20,A62-68,A88-92']",
            'contigmap.length=null',
        ])

        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'guidepost', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

class TestModelRunners(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_add_fake_peptide_frame_idempotency(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "+diffuser.batch_optimal_transport=False",
            "inference.model_runner=FlowMatching",
        ])
        sampler = model_runners.sampler_selector(conf)
        indep = sampler.sample_init()
        run_inference.make_deterministic()
        indep_fake_frame = model_runners.add_fake_peptide_frame(indep)
        run_inference.make_deterministic()
        indep_fake_frame_2 =  model_runners.add_fake_peptide_frame(indep_fake_frame)
        diff = test_utils.cmp_pretty(indep_fake_frame_2.xyz, indep_fake_frame.xyz, atol=1e-4)
        if diff:
            print(diff)
            self.fail(f'{diff=}')
    
    def test_idealize_peptide_frames_idempotency(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "+diffuser.batch_optimal_transport=False",
            "inference.model_runner=FlowMatching",
        ])
        sampler = model_runners.sampler_selector(conf)
        indep = sampler.sample_init()
        run_inference.make_deterministic()
        indep_fake_frame = model_runners.idealize_peptide_frames(indep)
        run_inference.make_deterministic()
        indep_fake_frame_2 =  model_runners.idealize_peptide_frames(indep_fake_frame)
        diff = test_utils.cmp_pretty(indep_fake_frame_2.xyz, indep_fake_frame.xyz, atol=1e-4)
        ic(
            torch.linalg.vector_norm(indep_fake_frame.xyz - indep_fake_frame_2.xyz, dim=-1),
            torch.linalg.vector_norm(indep_fake_frame.xyz - indep_fake_frame_2.xyz, dim=-1).max(),
        )
        if diff:
            print(diff)
            self.fail(f'{diff=}')

    def test_rigid_from_atom_14(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "+diffuser.batch_optimal_transport=False",
            "inference.model_runner=FlowMatching",
        ])
        sampler = model_runners.sampler_selector(conf)
        indep = sampler.sample_init()
        # run_inference.make_deterministic()
        rigid_1 = du.rigid_frames_from_atom_14(indep.xyz)
        # run_inference.make_deterministic()
        rigid_2 =  du.rigid_frames_from_atom_14(indep.xyz)
        rots_1 = rigid_1.get_rots().get_rot_mats()
        rots_2 = rigid_2.get_rots().get_rot_mats()
        diff = test_utils.cmp_pretty(rots_1, rots_2, atol=1e-6)
        if diff:
            ic(
                (rots_1 - rots_2).max(),
            )
            print(diff)
            self.fail(f'{diff=}')

    def test_make_conditional(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "+diffuser.batch_optimal_transport=False",
            "inference.model_runner=FlowMatching",
        ])
        sampler = model_runners.sampler_selector(conf)
        indep = sampler.sample_init()

        indep_cond = aa_model.make_conditional_indep(indep, sampler.indep_cond, sampler.is_diffused)

        diff = test_utils.cmp_pretty(indep_cond.xyz[~sampler.is_diffused, :3], indep.xyz[~sampler.is_diffused, :3], atol=1e-6)
        if diff:
            print(diff)
            ic(
                indep_cond.xyz[~sampler.is_diffused] - indep.xyz[~sampler.is_diffused]
            )
            self.fail(f'Backbone disagreement: {diff=}')
        

        diff = test_utils.cmp_pretty(indep_cond.xyz[~sampler.is_diffused], indep.xyz[~sampler.is_diffused], atol=1e-6)
        if diff:
            print(diff)
            ic(
                indep_cond.xyz[~sampler.is_diffused] - indep.xyz[~sampler.is_diffused]
            )
            # TODO: uncomment
            self.fail(f'Some disagreement: {diff=}')

        diff = test_utils.cmp_pretty(indep_cond, indep, atol=1e-6)
        if diff:
            print(diff)
            self.fail(f'{diff=}')

class TestInference(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Test that the motif remains fixed throughout inference.
    def test_motif_remains_fixed(self):
        T = 2
        conf = construct_conf([
            f'diffuser.T={T}',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_1',
            'inference.contig_as_guidepost=False',
            'inference.str_self_cond=0',
        ])

        func_sig = signature(RoseTTAFoldModule.forward)
        fake_forward = mock.patch.object(RoseTTAFoldModule, "forward", autospec=True)

        def side_effect(self, *args, **kwargs):
            ic("mock forward", type(self), side_effect.call_count)
            side_effect.call_count += 1
            return fake_forward.temp_original(self, *args, **kwargs)
        side_effect.call_count = 0

        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect
            run_inference.main(conf)

            mapped_calls = []
            for args, kwargs in mock_forward.call_args_list:
                args = (None,) + args[1:]
                argument_binding = func_sig.bind(*args, **kwargs)
                argument_map = argument_binding.arguments
                argument_map = tensor_util.cpu(argument_map)
                mapped_calls.append(argument_map)
        
        is_motif = 10
        def constant(mapped_call):
            c = {}
            # Technically only the first 3 indices are used by the network, but helpful to check to understand
            # inconsistencies in the alpha tensor computed from these coordinates.
            c['xyz'] = mapped_call['xyz'][0,is_motif]
            is_sidechain_torsion = torch.ones(3*rf2aa.chemical.NTOTALDOFS).bool()
            is_sidechain_torsion[0:2] = False
            is_sidechain_torsion[3:5] = False
            c['alpha'] = mapped_call['alpha_t'][0,0,is_motif]
            # Remove backbone torsions
            c['alpha'][~is_sidechain_torsion] = torch.nan
            c['sctors'] = mapped_call['sctors'][0, is_motif]
            # Remove backbone torsions
            c['sctors'][0:2] = torch.nan
            return c
        
        constants = []
        for mapped_call in mapped_calls:
            constants.append(constant(mapped_call))
        
        self.assertEqual(len(constants), T)
        cmp = partial(tensor_util.cmp, atol=1e-9, rtol=1e-4)
        for i in range(1, T):
            test_utils.assertEqual(self, cmp, constants[0], constants[i])
    
    def test_motif_fixed_in_output(self):
        '''
        Tests that motif N, CA and C atoms, as well as implicit side chain atoms
        remain fixed during inference.
        '''
        output_pdb, conf = infer([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_2',
            "contigmap.contigs=['1,A518-519,1']",
            'contigmap.length=4-4',
            'inference.guidepost_xyz_as_design_bb=[True]',
        ])

        input_feats = inference.utils.parse_pdb(conf.inference.input_pdb)
        output_feats = inference.utils.parse_pdb(output_pdb)

        trb = get_trb(conf)
        is_motif = torch.tensor(trb['con_hal_idx0'])
        is_motif_ref = torch.tensor(trb['con_ref_idx0'])
        n_motif = len(is_motif)
        
        input_motif_xyz = input_feats['xyz'][is_motif_ref]
        output_motif_xyz = output_feats['xyz'][is_motif]
        atom_mask = input_feats['mask'][is_motif_ref] # rk. should this be is_motif_ref?
        self.assertEqual(n_motif, 2)

        # Backbone only
        backbone_atom_mask = torch.zeros((n_motif, 14)).bool()
        backbone_atom_mask[:,:3] = True
        backbone_rmsd = rf2aa.loss.calc_crd_rmsd(
                torch.tensor(input_motif_xyz)[None],
                torch.tensor(output_motif_xyz)[None],
                backbone_atom_mask[None])
        # The motif gets rotated and translated, so the accuracy is somewhat limited
        # due to the precision of coordinates in a PDB file.
        self.assertLess(backbone_rmsd, 0.04)

        # All atoms
        atom_mask[:, 3] = False  # Exclude bb O because it can move around depending on non-motif residue placement.
        rmsd = rf2aa.loss.calc_crd_rmsd(
                torch.tensor(input_motif_xyz)[None],
                torch.tensor(output_motif_xyz)[None],
                torch.tensor(atom_mask)[None])
        self.assertLess(rmsd, 0.04)

    # TODO create function for regenerating the golden here
    # def test_convert_motif_to_guide_posts(self):
        # '''
        # Test that the function guide_posts.convert_motif_to_guide_posts modifies `indep` appropriately.
        # '''
        # import pickle
        # with open('test_data/pkl/indep_pre_guide_post_conversion.pkl', 'rb') as f:
        #     indep_pre = pickle.load(f)
        # with open('test_data/pkl/indep_post_guide_post_conversion.pkl', 'rb') as f:
        #     indep_post = pickle.load(f)

        # # Process indep_pre through gp.convert_motif_to_guide_posts
        # gp_to_ptn_idx0 = {
        #     113: 0,
        #     114: 1,
        #     115: 2,
        #     116: 3,
        #     117: 4,
        #     118: 5,
        #     119: 6,
        #     120: 7,
        #     121: 8,
        #     122: 9,
        #     123: 10,
        #     124: 11
        # }
        # indep_test, _ = gp.convert_motif_to_guide_posts(
        #     gp_to_ptn_idx0=gp_to_ptn_idx0,
        #     indep=indep_pre,
        #     placement='anywhere'
        # )

        # for k in indep_post.__dataclass_fields__.keys():
        #     v_post = getattr(indep_post, k)
        #     v_test = getattr(indep_test, k)
        #     self.assertTrue( torch.isclose(v_post, v_test, equal_nan=True).all(), k)

    def test_heme_no_lig(self):
        """
        test that network atomizes protein
        """
        output_pdb, conf = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_3',
            'inference.input_pdb=benchmark/input/1yzr_no_covalent.pdb',
            'inference.zero_weights=True',
            "contigmap.contigs=['1-1,A173-173,1-1']",
            "+contigmap.contig_atoms=\"{'A173':'CD2,ND1,NE2,CE1'}\"",
            'contigmap.length=3-3'
        ])

    def test_heme_lig(self):
        """
        test that network atomizes protein
        """
        output_pdb, conf = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_4',
            'inference.input_pdb=benchmark/input/1yzr_no_covalent.pdb',
            'inference.ligand=HEM',
            'inference.zero_weights=True',
            "contigmap.contigs=['1-1,A173-173,1-1']",
            "+contigmap.contig_atoms=\"{'A173':'CD2,ND1,NE2,CE1'}\"",
            'contigmap.length=3-3'
        ])
    
    def test_guidepost_removal(self):
        '''
        Tests that tip atom guideposts are removed.
        '''
        run_inference.make_deterministic()
        pdb, conf = infer([
            'diffuser.T=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'inference.contig_as_guidepost=True',
            "contigmap.contigs=['5,A62-63,5']",
            'contigmap.length=null',
            'inference.guidepost_xyz_as_design=True'
        ])

        pdb_contents = inference.utils.parse_pdb(pdb)
        assertpy.assert_that(pdb_contents['xyz'].shape[0]).is_equal_to(12)


if __name__ == '__main__':
        unittest.main()
