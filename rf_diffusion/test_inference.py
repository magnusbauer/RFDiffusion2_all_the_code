import os
import unittest
import pytest
from unittest import mock
from pathlib import Path
from inspect import signature
import assertpy

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from icecream import ic
import torch
import numpy as np

import test_utils
import run_inference
from functools import partial
from rf2aa import tensor_util

from rf_diffusion.chemical import ChemicalData as ChemData
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
import rf2aa.loss.loss
from rf_diffusion.inference import model_runners
from rf_diffusion import aa_model
from rf_diffusion import inference
import rf_diffusion.frame_diffusion.data.utils as du
import rf_diffusion
from rf_diffusion.dev import analyze
from rf_diffusion.dev import show_bench
from rf_diffusion.inference.model_runners import Sampler
from rf_diffusion import noisers
from rf_diffusion.frame_diffusion.rf_score.model import RFScore
from omegaconf import OmegaConf


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

class StopForwardCall(Exception):
    pass

def fake_load_model(self):
    """
    Mock loader that creates models and noisers based on the fm_tip yaml but does not actually load the weights
    nor create the real model
    """
    # Mock a simple model with as few parameters as possible
    toy_model_config = ['rf.model.refiner_topk=1',
    'rf.model.n_extra_block=1',
    'rf.model.n_main_block=1',
    'rf.model.n_ref_block=1',
    'rf.model.d_msa=1',
    'rf.model.d_msa_full=1',
    'rf.model.d_pair=1',
    'rf.model.d_templ=1',
    'rf.model.n_head_msa=1',
    'rf.model.n_head_pair=1',
    'rf.model.n_head_templ=1',
    'rf.model.d_hidden=1',
    'rf.model.d_hidden_templ=1',
    'rf.model.SE3_param.num_channels=1',
    'rf.model.SE3_param.l0_in_features=1',
    'rf.model.SE3_param.l0_out_features=1',
    'rf.model.SE3_param.l1_in_features=1',
    'rf.model.SE3_param.l1_out_features=1',
    'rf.model.SE3_param.num_degrees=2',
    'rf.model.SE3_param.div=1',
    'rf.model.SE3_param.num_edge_features=1',
    'rf.model.SE3_param.n_heads=1',]

    # Create the new config
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra().clear()
    with initialize(version_base=None, config_path="config/training", job_name="test_app_train"):
        weights_conf = compose(config_name='fm_tip.yaml', overrides=toy_model_config, return_hydra_config=True)

    # Merge with the old conf from inferene time
    OmegaConf.set_struct(self._conf, False)
    OmegaConf.set_struct(weights_conf, False)
    self._conf = OmegaConf.merge(
        weights_conf, self._conf)        

    # Create the diffuser and model
    self.diffuser = noisers.get(self._conf.diffuser)
    self.model = RFScore(self._conf.rf.model, self.diffuser, self.device)       

def get_rfi(conf):
    run_inference.make_deterministic()    

    # Mock the forward pass of the model and the model loader
    func_sig = signature(LegacyRoseTTAFoldModule.forward)
    fake_forward = mock.patch.object(LegacyRoseTTAFoldModule, "forward", autospec=True)
    fake_loader = mock.patch.object(Sampler, 'load_model', new=fake_load_model)

    def side_effect(self, *args, **kwargs):
        ic("mock forward", type(self), side_effect.call_count)
        side_effect.call_count += 1
        raise StopForwardCall()
    side_effect.call_count = 0

    with fake_loader:
        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect
            try:
                run_inference.main(conf)
            except StopForwardCall:
                pass

            mapped_calls = []
            for args, kwargs in mock_forward.call_args_list:
                args = (None,) + args[1:]
                argument_binding = func_sig.bind(*args, **kwargs)
                argument_map = argument_binding.arguments
                argument_map = tensor_util.cpu(argument_map)
                mapped_calls.append(argument_map)
    return mapped_calls

def NA_adaptor(pdb_contents):
    # For NA, can be removed if goldens are rewritten    
    pdb_contents['xyz'] = pdb_contents['xyz'][:,:ChemData().NHEAVYPROT]
    pdb_contents['mask'] = pdb_contents['mask'][:,:ChemData().NHEAVYPROT]    
    return pdb_contents

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
        pdb_contents = NA_adaptor(pdb_contents)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'T2', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_ori_cm(self):
        """
        Tests that sampling from a fixed center works as expected
        """
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=1',
            'inference.input_pdb=test_data/1yzr_no_covalent_ORI_cm1.pdb',            
            'inference.ligand=HEM',
            'inference.num_designs=1',
            '++transforms.names=["AddConditionalInputs","CenterPostTransform"]',  # Simulate behavior on different model
            '++transforms.configs.CenterPostTransform.center_type="is_diffused"',
            'inference.output_prefix=tmp/test_ori_cm',
            "contigmap.contigs=['10']",
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        pdb_contents = NA_adaptor(pdb_contents)
        cmp = partial(tensor_util.cmp, atol=6e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'ori_cm', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_ori_partial_diffusion(self):
        """
        Tests that sampling using partial diffusion with ORI models work
        """
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=20',
            'diffuser.partial_T=1',
            'inference.input_pdb=test_data/1yzr_no_covalent_ORI_cm1.pdb',            
            'inference.ligand=HEM',
            'inference.num_designs=1',
            '++transforms.names=["AddConditionalInputs","CenterPostTransform"]',  # Simulate behavior on different model
            '++transforms.configs.CenterPostTransform.center_type="is_diffused"',
            'inference.output_prefix=tmp/test_ori_partial_diffusion',
            "contigmap.contigs=['A1-10']",
            "contigmap.inpaint_str=['A1-10']",
            "contigmap.inpaint_seq=['A1-10']",
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        pdb_contents = NA_adaptor(pdb_contents)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'ori_pd', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=True',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_ori_cm(self):
        """
        Tests that sampling from a fixed center works as expected
        """
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.input_pdb=test_data/1yzr_no_covalent_ORI_cm1.pdb',            
            'inference.ligand=HEM',
            'inference.num_designs=1',
            '++transforms.names=["AddConditionalInputs","CenterPostTransform"]',  # Simulate behavior on different model
            '++transforms.configs.CenterPostTransform.center_type="is_diffused"',
            'inference.output_prefix=tmp/test_ori_cm',
            "contigmap.contigs=['10']",
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression_ori_cm', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_parses_covale(self):
        covale = 'benchmark/input/3l0f_covale.pdb'
        noncovale = 'benchmark/input/3l0f.pdb'

        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=True',
            'guidepost_bonds=True',            
            "contigmap.contigs=['2-2,A84-87,2-2']",
            "contigmap.contig_atoms=\"{'A84':'CA,C,N,O,CB,SG'}\"",
            "contigmap.length='8-8'",
            'inference.ligand=CYC',  
            '++inference.zero_weights=True',                      
        ])

        for name, input_pdb in [
            ('noncovale', noncovale),
            ('covale', covale),
        ]:
            # Set pdb
            conf.inference.input_pdb = input_pdb
            mapped_calls = get_rfi(conf)
            cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
            test_utils.assert_matches_golden(self, f'rfi_regression_parses_covale_{name}', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)        

    @pytest.mark.generates_golden
    def test_inference_rfi_parses_multiligand_2(self):
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=False',
            'guidepost_bonds=False',            
            "contigmap.contigs=['3-3']",
            "contigmap.length='3-3'",
            '++inference.zero_weights=True',            
        ])

        for name, ligand_name, input_pdb in [
            ('single', 'UDX', 'test_data/ec1_M0092_same_resn.pdb'),
            ('multi', 'NAD,UDX', 'test_data/ec1_M0092.pdb'),
        ]:
            # Set ligand name and pdb
            conf.inference.ligand = ligand_name
            conf.inference.input_pdb = input_pdb
            mapped_calls = get_rfi(conf)
            cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
            test_utils.assert_matches_golden(self, f'rfi_regression_parses_multiligand_2_{name}', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)        

    @pytest.mark.generates_golden
    def test_inference_rfi_two_chain_aa_model(self):
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=False',
            'guidepost_bonds=False',
            "contigmap.contigs=['A441-450_10']",
            "contigmap.has_termini=[True,True]",
            'inference.ligand=ANP',
            'inference.input_pdb=benchmark/input/2j0l.pdb', 
            '++inference.zero_weights=True',                           
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression_two_chain', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)       

    @pytest.mark.generates_golden
    def test_inference_rfi_3_chain_indep(self):
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=False',
            'guidepost_bonds=False',
            "contigmap.contigs=['3_3_A441-443,A445-447']",
            "contigmap.has_termini=[True,True,False]",
            'inference.ligand=ANP',
            'inference.input_pdb=benchmark/input/2j0l.pdb',   
            '++inference.zero_weights=True',                         
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression_3_chain_indep', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)            

    @pytest.mark.generates_golden
    def test_inference_rfi_3_chain_binder_middle_indep(self):
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=False',
            'guidepost_bonds=False',
            "contigmap.contigs=['A441-443_3_A445-447']",
            "contigmap.has_termini=[True,True,False]",
            'inference.ligand=ANP',
            'inference.input_pdb=benchmark/input/2j0l.pdb',      
            '++inference.zero_weights=True',                      
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression_3_chain_binder_middle_indep', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)            

    @pytest.mark.generates_golden
    def test_inference_rfi_inpaint_str(self):
        ''' 
        Binder design without specifying the structure of a peptide a priori at inference. inpaint_str is set for the residues in the motif that should keep their sequence identity but have their structure diffused.
        This test makes sure that is_diffused is set to True for the residues indicated in inpaint_str and is_seq_masked is set to False to keep the sequence unchanged.
        '''
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.contig_as_guidepost=True',
            'guidepost_bonds=True',
            "contigmap.contigs=['A676-677_2-2']",
            "contigmap.has_termini=[True,True]",
            "contigmap.inpaint_str=['A676-677']",
            'inference.ligand=ANP',
            'inference.input_pdb=benchmark/input/2j0l.pdb',   
            '++inference.zero_weights=True',                         
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_regression_inpaint_str', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)            

    @pytest.mark.generates_golden
    def test_inference_rfi_pd(self):
        # Tests partial diffusion 
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=2',
            'diffuser.partial_T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/partial_T_example_mini.pdb',
            'contigmap.contigs=["A1-9"]',
            'contigmap.inpaint_str=["A1-9"]',
            'contigmap.inpaint_seq=["A1-9"]',
            'inference.ligand="FMN"',
            'inference.output_prefix=tmp/test_pd',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_pd_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_sm(self):
        # Tests small molecule input preparation
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/1yzr_no_covalent.pdb',
            'contigmap.contigs=["5-10"]',
            'inference.ligand="HEM"',
            'inference.output_prefix=tmp/test_sm',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_sm_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_tip(self):
        # Tests tip diffusion input preparation
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa.pdb',
            'contigmap.contigs=["A518-518,10,A616-616"]',
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2','A616':'CG,OD1,OD2'}\"",
            'inference.ligand=LG1',
            'inference.output_prefix=tmp/test_tip',
            '++inference.zero_weights=True',
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_tip_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_tip_w_motif(self):
        # Tests tip diffusion input preparation
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa.pdb',
            'contigmap.contigs=["A518-518,10,A616-620"]',
            'inference.contig_as_guidepost=True',
            'inference.only_guidepost_positions="A518-518,A616-617,LG1:C1-C16"',
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2','A616':'CG,OD1,OD2'}\"",
            'inference.ligand=LG1',
            'inference.output_prefix=tmp/test_tip',
            '++inference.zero_weights=True',
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_tip_regression_w_motif', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_atomize(self):
        # Tests atomization of ligand
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa.pdb',
            'contigmap.contigs=["10"]',
            'inference.ligand=LG1',
            'inference.output_prefix=tmp/test_atomize',
            '++inference.zero_weights=True',
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_atomize_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_motif_sm(self):
        # Tests motif in the presence of ligand
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa.pdb',
            'contigmap.contigs=["10,A518-618,10"]',
            'inference.ligand=LG1',
            'inference.output_prefix=tmp/test_motif_sm',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_motif_sm_regression', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

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
        # Add xyz mask for now until goldens are remade
        pdb_contents['xyz'] = pdb_contents['xyz'][:,:ChemData().NHEAVYPROT]
        pdb_contents['mask'] = pdb_contents['mask'][:,:ChemData().NHEAVYPROT]
        # The network exhibits chaotic behavior when coordinates corresponding to chiral gradients are updated,
        # so this primarily checks that inference runs and produces the expected shapes, rather than coordinate
        # values, which vary wildly across CPU architectures.
        cmp = partial(tensor_util.cmp, atol=5, rtol=0)
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
        # NA Compatability, only check first 14 atoms, which is consistent with all previous tests
        pdb_contents['xyz'] = pdb_contents['xyz'][:,:14]  # This isn't correct still!
        test_utils.assert_matches_golden(self, '10res_self_conditioning', pdb_contents, rewrite=False, custom_comparator=cmp)

class TestWriteTrajs(unittest.TestCase):

    T=2

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    def assert_generates(self, overrides, golden, output_dir='cfg/write_trajs'):
        run_inference.make_deterministic(ignore_if_cuda=True)
        test_name=os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        pdb, _ = infer([
            f'diffuser.T={self.T}',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{output_dir}/{test_name}',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=FlowMatching",
            "+diffuser.batch_optimal_transport=False",
            'inference.write_trajectory=True'
        ] + overrides)
        row = show_bench.get_sdata(str(pdb)).iloc[0]
        x0_path = analyze.get_traj_path(row, 'pX0')
        got = parse_traj(x0_path)
        x0_path = analyze.get_traj_path(row, 'px0_cond')
        got_2 = parse_traj(x0_path)
        cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
        test_utils.assertEqual(self, cmp, got, got_2)
    
    def test_write_base(self):
        self.assert_generates(
                [
                    "inference.model_runner=ClassifierFreeGuidance",
                    "inference.classifier_free_guidance_scale=1",
                ],
                'cfg_cond_trajs'
        )

def parse_traj(traj_path):
    traj_pdb_lines = inference.utils.get_pdb_lines_traj(traj_path)
    indeps = [rf_diffusion.parsers.parse_pdb_lines_target(lines) for lines in traj_pdb_lines]
    return indeps

class InferenceTestBase(unittest.TestCase):

    guidance_scales_T = 20

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    def assert_generates(self, overrides, golden, output_dir='cfg'):
        run_inference.make_deterministic(ignore_if_cuda=True)
        test_name=os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{output_dir}/{test_name}',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=FlowMatching",
            "+diffuser.batch_optimal_transport=False",
        ] + overrides)
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
        test_utils.assert_matches_golden(self, golden, pdb_contents, rewrite=REWRITE, custom_comparator=cmp)


class TestCFG(unittest.TestCase):

    guidance_scales_T = 20

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    def assert_generates(self, overrides, golden, output_dir='cfg'):
        run_inference.make_deterministic(ignore_if_cuda=True)
        test_name=os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{output_dir}/{test_name}',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=FlowMatching",
            "+diffuser.batch_optimal_transport=False",
        ] + overrides)
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
        test_utils.assert_matches_golden(self, golden, pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    ###################### Conditional generation ######################

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    @pytest.mark.generates_golden
    def test_cond_base(self):
        self.assert_generates([], 'cfg_cond_base')
    
    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_cond_make_conditional(self):
        self.assert_generates(
                [
                        "inference.model_runner=FlowMatching_make_conditional",
                ],
                'cfg_cond_base',
        )

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_cond_make_conditional_diffuse_all(self):
        self.assert_generates(
                [
                        "inference.model_runner=FlowMatching_make_conditional_diffuse_all",
                ],
                'cfg_cond_base',
        )

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_cond_make_conditional_diffuse_all_xt_unfrozen(self):
        self.assert_generates(
                [
                        "inference.model_runner=FlowMatching_make_conditional_diffuse_all_xt_unfrozen",
                ],
                'cfg_cond_base',
        )

    
    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_cond_cfg(self):
        self.assert_generates(
                [
                        "inference.model_runner=ClassifierFreeGuidance",
                        "inference.classifier_free_guidance_scale=1",
                ],
                'cfg_cond_base',
        )
    

    # @pytest.mark.slow
    # @pytest.mark.nondeterministic
    # def test_cond_cfg_calculate_full_grads(self):
    #     self.assert_generates(
    #             [
    #                     "inference.model_runner=ClassifierFreeGuidance",
    #                     "inference.classifier_free_guidance_scale=1",
    #                     "+inference.calculate_full_grads=True",
    #             ],
    #             'cfg_cond_base',
    #             output_dir='cfg/garbo',
    #     )


    ###################### Unconditional generation ######################
    
    @pytest.mark.slow
    @pytest.mark.nondeterministic
    @pytest.mark.generates_golden
    def test_uncond_base(self):
        self.assert_generates(
                [
                        "++contigmap.contig_atoms=\"{'A518':''}\"",
                ],
                'cfg_uncond_base'
        )
    
    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_uncond_make_conditional_diffuse_all(self):
        self.assert_generates(
                [
                        "++contigmap.contig_atoms=\"{'A518':''}\"",
                        "inference.model_runner=FlowMatching_make_conditional_diffuse_all",
                ],
                'cfg_uncond_base'
        )

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_uncond_cfg(self):
        self.assert_generates(
                [
                    "inference.model_runner=ClassifierFreeGuidance",
                    "inference.classifier_free_guidance_scale=0",
                ],
                'cfg_uncond_base'
        )

    ###################### Xt tests ######################

    def assert_generates_xt(self, overrides, golden):
        run_inference.make_deterministic(ignore_if_cuda=True)
        test_name=os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        pdb, _ = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/cfg/xt/{test_name}',
            "contigmap.contigs=['9,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=FlowMatching",
            "+diffuser.batch_optimal_transport=False",
        ] + overrides)
        row = show_bench.get_sdata(str(pdb)).iloc[0]
        xt_path = analyze.get_traj_path(row, 'Xt')
        first_denoised_pdb_lines = inference.utils.get_pdb_lines_traj(xt_path)[-1]
        first_denoised = rf_diffusion.parsers.parse_pdb_lines_target(first_denoised_pdb_lines)
        cmp = partial(tensor_util.cmp, atol=1e-2, rtol=0)
        test_utils.assert_matches_golden(self, golden, first_denoised, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.slow
    @pytest.mark.generates_golden
    @pytest.mark.nondeterministic
    def test_xt_cond_unfrozen(self):
        self.assert_generates_xt(
                [
                    "inference.model_runner=FlowMatching_make_conditional_diffuse_all_xt_unfrozen",
                ],
                'cfg_cond_xt'
        )

    @pytest.mark.slow
    @pytest.mark.nondeterministic
    def test_xt_cfg(self):
        self.assert_generates_xt(
                [
                    "inference.model_runner=ClassifierFreeGuidance",
                    "inference.classifier_free_guidance_scale=1",
                ],
                'cfg_cond_xt'
        )

    # ###################### Non-trivial guidance ######################
    
    @pytest.mark.slow
    @pytest.mark.generates_golden
    @pytest.mark.nondeterministic
    def test_uncentered_xt_w_8(self):
        self.assert_generates(
                [
                        "diffuser.T=20",
                        "inference.str_self_cond=False",
                        "inference.model_runner=ClassifierFreeGuidance",
                        "inference.classifier_free_guidance_scale=8",
                ],
                'cfg_uncentered_xt_w_8',
        )
    
    @pytest.mark.slow
    @pytest.mark.generates_golden
    @pytest.mark.nondeterministic
    @pytest.mark.debug
    def test_centered_xt_w_1(self):
        self.assert_generates(
                [
                        "diffuser.T=20",
                        "inference.str_self_cond=False",
                        "inference.model_runner=ClassifierFreeGuidance",
                        "inference.classifier_free_guidance_scale=1",
                        "+inference.classifier_free_guidance_recenter_xt=True",
                ],
                'cfg_centered_xt_w_1',
        )

    @pytest.mark.slow
    @pytest.mark.generates_golden
    @pytest.mark.nondeterministic
    def test_centered_xt_w_8(self):
        self.assert_generates(
                [
                        "diffuser.T=20",
                        "inference.str_self_cond=False",
                        "inference.model_runner=ClassifierFreeGuidance",
                        "inference.classifier_free_guidance_scale=8",
                        "+inference.classifier_free_guidance_recenter_xt=True",
                ],
                'cfg_centered_xt_w_8',
        )
    

    @pytest.mark.slow
    @pytest.mark.generates_golden
    @pytest.mark.nondeterministic
    @pytest.mark.debug
    def test_centered_xt_w_8_self_cond(self):
        self.assert_generates(
                [
                        "diffuser.T=20",
                        "inference.str_self_cond=True",
                        "inference.model_runner=ClassifierFreeGuidance",
                        "inference.classifier_free_guidance_scale=8",
                        "+inference.classifier_free_guidance_recenter_xt=True",
                ],
                'cfg_centered_xt_w_8_self_cond',
        )

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
        indep, contig_map, atomizer, t_step_input = sampler.sample_init()
        run_inference.make_deterministic()
        indep_fake_frame = aa_model.add_fake_peptide_frame(indep)
        run_inference.make_deterministic()
        indep_fake_frame_2 =  aa_model.add_fake_peptide_frame(indep_fake_frame)
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
        indep, contig_map, atomizer, t_step_input = sampler.sample_init()
        run_inference.make_deterministic()
        indep_fake_frame = aa_model.idealize_peptide_frames(indep)
        run_inference.make_deterministic()
        indep_fake_frame_2 =  aa_model.idealize_peptide_frames(indep_fake_frame)
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
        indep, contig_map, atomizer, t_step_input = sampler.sample_init()
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
        indep, contig_map, atomizer, t_step_input = sampler.sample_init()

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

    def test_uncond_init_ori(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa_ORI_cm1.pdb',            
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+diffuser.batch_optimal_transport=False",
            "+contigmap.contig_atoms=\"{'A518':''}\"",
            "inference.model_runner=ClassifierFreeGuidance",
            '++transforms.names=["AddConditionalInputs","CenterPostTransform"]',  # Simulate behavior on different model
            '++transforms.configs.CenterPostTransform.center_type="is_diffused"',          
        ])
        sampler = model_runners.sampler_selector(conf)
        indep_uncond, contig_map, atomizer, t_step_input = sampler.sample_init()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa_ORI_cm1.pdb',            
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+diffuser.batch_optimal_transport=False",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.model_runner=ClassifierFreeGuidance",
            "inference.classifier_free_guidance_scale=0",
            '++transforms.names=["AddConditionalInputs","CenterPostTransform"]',  # Simulate behavior on different model
            '++transforms.configs.CenterPostTransform.center_type="is_diffused"',             
        ])        
        sampler = model_runners.sampler_selector(conf) 
        indep_uncond_cfg, contig_map, atomizer, t_step_input = sampler.sample_init()
        xyzd = torch.isnan(indep_uncond_cfg.xyz) != torch.isnan(indep_uncond.xyz)
        assert xyzd.sum() == 0

        diff = test_utils.cmp_pretty(indep_uncond_cfg, indep_uncond, atol=1e-5)
        if diff:
            xyzd = indep_uncond_cfg.xyz - indep_uncond.xyz
            ic(
                xyzd,
                torch.linalg.vector_norm(xyzd, dim=-1),
                torch.linalg.vector_norm(xyzd, dim=(-1,-2)),
                indep_uncond.xyz[-1,:3],
                indep_uncond_cfg.xyz[-1,:3],
                indep_uncond.xyz[-1,:3]-indep_uncond_cfg.xyz[-1,:3],
            )
            print(diff)
            self.fail('indep unequal')


    def test_uncond_init_fm(self):
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa.pdb',
            'inference.output_prefix=tmp/test_no_batch_ot',
            "contigmap.contigs=['9,A518-518,1']",
            "+diffuser.batch_optimal_transport=False",            
            "+contigmap.contig_atoms=\"{'A518':''}\"",
            "inference.model_runner=ClassifierFreeGuidance",
            "+diffuser.type='flow_matching'",
        ])
        sampler = model_runners.sampler_selector(conf)
        indep_uncond, contig_map, atomizer, t_step_input = sampler.sample_init()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=10',
            'inference.num_designs=1',
            'inference.input_pdb=benchmark/input/gaa.pdb',            
            'inference.output_prefix=tmp/test_no_batch_ot',
            "+diffuser.batch_optimal_transport=False",            
            "contigmap.contigs=['9,A518-518,1']",
            "inference.model_runner=ClassifierFreeGuidance",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            "inference.classifier_free_guidance_scale=0",
            "+diffuser.type='flow_matching'",
        ])        
        sampler = model_runners.sampler_selector(conf) 
        indep_uncond_cfg, contig_map, atomizer, t_step_input = sampler.sample_init()
        xyzd = torch.isnan(indep_uncond_cfg.xyz) != torch.isnan(indep_uncond.xyz)
        assert xyzd.sum() == 0

        diff = test_utils.cmp_pretty(indep_uncond_cfg, indep_uncond, atol=1e-5)
        if diff:
            xyzd = indep_uncond_cfg.xyz - indep_uncond.xyz
            ic(
                xyzd,
                torch.linalg.vector_norm(xyzd, dim=-1),
                torch.linalg.vector_norm(xyzd, dim=(-1,-2)),
                indep_uncond.xyz[-1,:3],
                indep_uncond_cfg.xyz[-1,:3],
                indep_uncond.xyz[-1,:3]-indep_uncond_cfg.xyz[-1,:3],
            )
            print(diff)
            self.fail('indep unequal')


class TestInference(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    


    def test_checkpoint_for_training(self):
        '''
        Test that ensures that the checkpoint.checkpoint path through the model doesn't have bugs

        This was an issue around May 23, 2024

        If this test doesn't crash, we're good to go.
        '''
        T = 1
        conf = construct_conf([
            f'diffuser.T={T}',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_1',
            'inference.contig_as_guidepost=False',
            'inference.str_self_cond=0',
        ])

        fake_forward = mock.patch.object(LegacyRoseTTAFoldModule, "forward", autospec=True)

        def side_effect(self, *args, **kwargs):
            ic("mock forward", type(self))
            kwargs['use_checkpoint'] = True # force the use_checkpoint path (typically used in training)
            return fake_forward.temp_original(self, *args, **kwargs)

        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect
            run_inference.main(conf)


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

        func_sig = signature(LegacyRoseTTAFoldModule.forward)
        fake_forward = mock.patch.object(LegacyRoseTTAFoldModule, "forward", autospec=True)

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
            is_sidechain_torsion = torch.ones(3*ChemData().NTOTALDOFS).bool()
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
        ic(trb.keys())
        is_motif = torch.tensor(trb['con_hal_idx0'])
        is_motif_ref = torch.tensor(trb['con_ref_idx0'])
        n_motif = len(is_motif)
        
        input_motif_xyz = input_feats['xyz'][is_motif_ref]
        output_motif_xyz = output_feats['xyz'][is_motif]
        atom_mask = input_feats['mask'][is_motif_ref] # rk. should this be is_motif_ref?
        self.assertEqual(n_motif, 2)

        # Backbone only
        backbone_atom_mask = torch.zeros((n_motif, ChemData().NHEAVY)).bool()
        backbone_atom_mask[:,:3] = True
        backbone_rmsd = rf2aa.loss.loss.calc_crd_rmsd(
                torch.tensor(input_motif_xyz)[None],
                torch.tensor(output_motif_xyz)[None],
                backbone_atom_mask[None])
        # The motif gets rotated and translated, so the accuracy is somewhat limited
        # due to the precision of coordinates in a PDB file.
        self.assertLess(backbone_rmsd, 0.04)

        # All atoms
        atom_mask[:, 3] = False  # Exclude bb O because it can move around depending on non-motif residue placement.
        rmsd = rf2aa.loss.loss.calc_crd_rmsd(
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
