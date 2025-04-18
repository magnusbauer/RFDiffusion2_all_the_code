import os
import unittest
import pytest
from unittest import mock
from pathlib import Path
from inspect import signature
import assertpy
import time

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from icecream import ic
import torch
import numpy as np
import pandas as pd

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
from rf_diffusion.conditions import hbond_satisfaction
import rf_diffusion.frame_diffusion.data.utils as du
import rf_diffusion
from rf_diffusion.dev import analyze
from rf_diffusion.dev import show_bench
from rf_diffusion.inference.model_runners import Sampler
from rf_diffusion import noisers
from rf_diffusion.frame_diffusion.rf_score.model import RFScore
from rf_diffusion.conditions.ss_adj.sec_struct_adjacency import SS_HELIX, SS_STRAND, SS_LOOP, SS_MASK, N_SS
from omegaconf import OmegaConf
from rf_diffusion.inference.filters import TestFilter
from rf_diffusion.inference.t_setup import setup_t_arrays
import rf_diffusion.inference.mid_run_modifiers as mrm


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

def get_rfi(conf, spy_on_call=0):
    run_inference.make_deterministic()    

    # Mock the forward pass of the model and the model loader
    func_sig = signature(LegacyRoseTTAFoldModule.forward)
    fake_forward = mock.patch.object(LegacyRoseTTAFoldModule, "forward", autospec=True)
    fake_loader = mock.patch.object(Sampler, 'load_model', new=fake_load_model)

    def side_effect(self, *args, **kwargs):
        ic("mock forward", type(self), side_effect.call_count)
        if side_effect.call_count == spy_on_call:
            raise StopForwardCall()
        side_effect.call_count += 1
        return fake_rosettafold(*args, **kwargs)
    side_effect.call_count = 0

    with fake_loader:
        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect
            try:
                run_inference.main(conf)
            except StopForwardCall:
                mapped_calls = []
                for args, kwargs in mock_forward.call_args_list:
                    args = (None,) + args[1:]
                    argument_binding = func_sig.bind(*args, **kwargs)
                    argument_map = argument_binding.arguments
                    argument_map = tensor_util.cpu(argument_map)
                    mapped_calls.append(argument_map)
                return mapped_calls
            else:
                raise Exception('LegacyRoseTTAFoldModule was called {side_effect.call_count} times, but we are attempting to spy on call {spy_on_call}')

def fake_rosettafold(*args, **kwargs):
    '''
    Mock for LegacyRoseTTAFoldModule.forward
    '''
    if 'xyz' in kwargs:
        xyz = kwargs['xyz']
    else:
        xyz = args[4] # [1, L, 36 3]
    L = xyz.shape[1]
    px0_xyz = xyz[None,:,:,:3].repeat(40, 1, 1, 1, 1)
    px0_xyz = torch.normal(0, 1, px0_xyz.shape) + px0_xyz
    logits = (
        torch.normal(0, 1, (1, 61, L, L)),
        torch.normal(0, 1, (1, 61, L, L)),
        torch.normal(0, 1, (1, 37, L, L)),
        torch.normal(0, 1, (1, 19, L, L)),
    )
    logits_aa = torch.normal(0, 1, (1, 80, L))
    logits_pae = torch.normal(0, 1, (1, 64, L, L))
    logits_pde = torch.normal(0, 1, (1, 64, L, L))
    p_bind = torch.normal(0, 1, (1, 1))
    alpha_s = torch.normal(0, 1, (40, 1, L, 20, 2))
    xyz_allatom = torch.normal(0, 1, xyz.shape) + xyz[:]
    lddt = torch.normal(0, 1, (1, 50, L))
    xyz_allatom.requires_grad = True
    ca_y = xyz[None,:,:,1:2, 1]
    quat = torch.normal(0, 1, (1, 40, L, 4)) + ca_y.repeat(1, 40, 1, 4)
    quat = quat / quat.norm(dim=-1)[...,None]
    rfo = aa_model.RFO(logits, logits_aa, logits_pae, logits_pde, p_bind, px0_xyz, alpha_s, xyz_allatom, lddt, None, None, None, quat)
    rfo = tensor_util.to_ordered_dict(rfo)
    return rfo.values()

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
        cmp = partial(tensor_util.cmp, atol=1e-1, rtol=0)
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


    @pytest.mark.generates_golden
    def test_inference_rfi_ss_adj_monomer_nonspec_glob(self):
        # Tests ss_adj where the ss/adj generates a monomer and the contig is specified non-specifically
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'contigmap.contigs=["1-100"]',
            'inference.output_prefix=tmp/test_ss_adj',
            '++inference.zero_weights=True',
            'scaffoldguided.scaffold_dir=test_data/ss_adj/HHH_b1_00001',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',
            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',
        ])
        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 85, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_adj_monomer_nonspec_glob', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


    @pytest.mark.generates_golden
    def test_inference_rfi_ss_adj_monomer_motif_graft_arc(self):
        # Tests ss_adj where the ss/adj generates a monomer with a motif
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'contigmap.contigs=["22,A38-47,20"]',
            'inference.output_prefix=tmp/test_ss_adj',
            '++inference.zero_weights=True',
            'scaffoldguided.scaffold_arc=test_data/ss_adj/HHH_b1_00001_arc.pt',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',
            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',
        ])
        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 85, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_adj_monomer_motif_graft_arc', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


    @pytest.mark.generates_golden
    def test_inference_rfi_ss_adj_binder_target_full_arc(self):
        # Tests ss/adj where it's a binder/target situation and the whole system is specified
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'contigmap.contigs=["42,0_A38-47"]',
            'inference.output_prefix=tmp/test_ss_adj',
            '++inference.zero_weights=True',
            'scaffoldguided.scaffold_arc=test_data/ss_adj/HHH_b1_00001_arc.pt',
            'contigmap.has_termini=[True,True]',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',
            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',
        ])
        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 85, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_adj_binder_target_full_arc', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_inference_rfi_ss_adj_binder_and_target_auto_glob(self):
        # Tests ss/adj where the binder has the ss/adj specified and the target is autogenerated + all scaffoldguided options
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'contigmap.contigs=["100,0_A38-77"]',
            'inference.output_prefix=tmp/test_ss_adj',
            '++inference.zero_weights=True',
            'scaffoldguided.scaffold_dir=test_data/ss_adj/HHH_b1_00001',
            'scaffoldguided.sampled_insertion=2',
            'scaffoldguided.sampled_N=2',
            'scaffoldguided.sampled_C=2',
            'scaffoldguided.ss_mask=2',
            'scaffoldguided.mask_loops=True',
            'scaffoldguided.autogenerate_target_ss_adj=True',
            'contigmap.has_termini=[True,True]',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',
            '++upstream_inference_transforms.names=[AutogenerateTargetSSADJTransform]',
            '++upstream_inference_transforms.configs.AutogenerateTargetSSADJTransform={}',
            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',
        ])
        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 85, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_adj_binder_and_target_auto_glob', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


    @pytest.mark.generates_golden
    def test_inference_rfi_ss_adj_binder_motif_graft_target_ss_adj(self):
        # Tests ss/adj where the binder is a motif graft situation and we're pulling the contig from the scaffold_list
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/two_chain.pdb',
            'contigmap.contigs=["error if used"]',
            'inference.output_prefix=tmp/test_ss_adj',
            '++inference.zero_weights=True',
            'scaffoldguided.scaffold_dir=test_data/ss_adj/HHH_b1_00001',
            'scaffoldguided.scaffold_list=test_data/ss_adj/HHH_b1_00001.txt',
            'scaffoldguided.target_ss=test_data/ss_adj/two_chain_B_ss.pt',
            'scaffoldguided.target_adj=test_data/ss_adj/two_chain_B_adj.pt',
            'contigmap.has_termini=[True,True]',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',
            '++upstream_inference_transforms.names=[LoadTargetSSADJTransform]',
            '++upstream_inference_transforms.configs.LoadTargetSSADJTransform={}',
            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',
        ])
        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 85, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_adj_binder_motif_graft_target_ss_adj', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)



    @pytest.mark.generates_golden
    def test_inference_rfi_ppi_hotspots_antihotspots(self):
        # Tests ppi hotspots, antihotspots, ExposedTerminusTransform, and RenumberCroppedInput
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/two_chain.pdb',
            'contigmap.contigs=["30-30,0_B121-130"]',
            'inference.output_prefix=tmp/test_ppi_hotspots',
            '++inference.zero_weights=True',
            'contigmap.has_termini=[True,True]',
            '+extra_tXd=["ppi_hotspots_antihotspots"]',
            '+extra_tXd_params.ppi_hotspots_antihotspots={}',
            '++diffuser.independently_center_diffuseds=True',

            '++upstream_training_transforms.names=[FindHotspotsTrainingTransform]',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.p_is_hotspot_example=0.9',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.p_is_antihotspot_example=0.05',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.hotspot_values_mean=tenA_neighbors',

            '++upstream_inference_transforms.names=[HotspotAntihotspotResInferenceTransform,ExposedTerminusTransform,RenumberCroppedInput]',
            '++upstream_inference_transforms.configs.HotspotAntihotspotResInferenceTransform={}',
            '++upstream_inference_transforms.configs.ExposedTerminusTransform={}',
            '++upstream_inference_transforms.configs.RenumberCroppedInput={}',

            '++transforms.names=["AddConditionalInputs","CenterPostTransform","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++transforms.configs.CenterPostTransform.center_type=target_hotspot',

            '++ppi.hotspot_res="B122,B128"',
            '++ppi.super_hotspot_res="B124,B125"',
            '++ppi.antihotspot_res="B123,B129"',
            '++ppi.exposed_N_terminus=5',
            '++ppi.exposed_C_terminus=6',
        ])
        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 84, "If this throws, the hotspots arent being written to t1d"

        hotspots = mapped_calls[0]['t1d'][0,0,:,-4]
        super_hotspots = mapped_calls[0]['t1d'][0,0,:,-3]
        antihotspots = mapped_calls[0]['t1d'][0,0,:,-2]

        assert (torch.where(hotspots)[0] == torch.tensor([31,37])).all()
        assert (torch.where(super_hotspots)[0] == torch.tensor([33,34])).all()
        assert (torch.where(antihotspots)[0] == torch.tensor([0,1,2,3,4,24,25,26,27,28,29,32,38])).all()


        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ppi_hotspots_antihotspots', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


    @pytest.mark.generates_golden
    def test_inference_rfi_ideal_ss(self):
        # Tests ppi hotspots, antihotspots, ExposedTerminusTransform, and RenumberCroppedInput
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/two_chain.pdb',
            'contigmap.contigs=["30-30,0_B121-130"]',
            'inference.output_prefix=tmp/test_ideal_ss',
            '++inference.zero_weights=True',
            'contigmap.has_termini=[True,True]',
            '+extra_tXd=["ideal_ss_cond"]',
            "+extra_tXd_params.ideal_ss_cond.topo_spec_choices=['HH','HHH','HHHH','HHHHH']",

            '++upstream_training_transforms.names=[AddIdealSSTrainingTransform]',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_ideal_ss=1',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_loop_frac=True',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_avg_scn=True',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_topo_spec=True',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_min_value=1.5',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_max_value=2.5',
            "++upstream_training_transforms.configs.AddIdealSSTrainingTransform.topo_spec_choices=['HH','HHH','HHHH','HHHHH']",

            '++upstream_inference_transforms.names=[AddIdealSSInferenceTransform]',
            '++upstream_inference_transforms.configs.AddIdealSSInferenceTransform.only_first_chain=True',

            '++transforms.names=["AddConditionalInputs","CenterPostTransform","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++transforms.configs.CenterPostTransform.center_type=is_not_diffused',

            '++ideal_ss.ideal_value=0.9',
            '++ideal_ss.ideal_std=0.01',
            '++ideal_ss.avg_scn=1.5',
            '++ideal_ss.scn_std=0.01',
            '++ideal_ss.loop_frac=0.15',
            '++ideal_ss.topo_spec={HHH:1}',
        ])
        mapped_calls = get_rfi(conf)
        print(mapped_calls[0]['t1d'].shape[-1])
        assert mapped_calls[0]['t1d'].shape[-1] == 91, "If this throws, the ideal_ss isn't being written to t1d"

        chain_A_size = 30
        ideal_ss_mask = mapped_calls[0]['t1d'][0,0,:,-11]
        ideal_ss = mapped_calls[0]['t1d'][0,0,:,-10]
        avg_scn_mask = mapped_calls[0]['t1d'][0,0,:,-9]
        avg_scn = mapped_calls[0]['t1d'][0,0,:,-8]
        loop_frac_mask = mapped_calls[0]['t1d'][0,0,:,-7]
        loop_frac = mapped_calls[0]['t1d'][0,0,:,-6]
        is_3h = mapped_calls[0]['t1d'][0,0,:,-5+1]

        assert ideal_ss_mask[:chain_A_size].all()
        assert not ideal_ss_mask[chain_A_size:].any()
        assert avg_scn_mask[:chain_A_size].all()
        assert not avg_scn_mask[chain_A_size:].any()
        assert loop_frac_mask[:chain_A_size].all()
        assert not loop_frac_mask[chain_A_size:].any()
        assert is_3h[:chain_A_size].all()
        assert not is_3h[chain_A_size:].any()

        assert ideal_ss[:chain_A_size].mean() > 0.8 and ideal_ss[:chain_A_size].max() <= 1
        assert avg_scn[:chain_A_size].mean() < 0.2 and avg_scn[:chain_A_size].min() >= 0
        assert torch.allclose(loop_frac[:chain_A_size], torch.tensor(0.15))


        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ideal_ss', mapped_calls, rewrite=False, custom_comparator=cmp)



    @pytest.mark.generates_golden
    def test_inference_rfi_target_hbond_satisfaction(self):
        '''
        Test the 15 features of target_hbond_satisfaction
        '''
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'contigmap.contigs=["A49-60"]',
            "+contigmap.contig_atoms=\"{'A49':'all','A59':'N,C,OE1,OE2'}\"",
            'inference.output_prefix=tmp/test_hbond_target_satisfaction',
            '++inference.zero_weights=True',
            'contigmap.has_termini=[True]',
            '+extra_tXd=["target_hbond_satisfaction_cond"]',
            '+extra_tXd_params.target_hbond_satisfaction_cond={}',
            '++upstream_inference_transforms.names=[HBondTargetSatisfactionInferenceTransform]',
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_bb_donates_to_target_bb=A50',#0
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_bb_accepts_from_target_bb=A52',#1
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_sc_donates_to_target_bb=A54',#2
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_sc_accepts_from_target_bb=A60',#3
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_bb_donates_to_target_sc=A51',#4
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_bb_accepts_from_target_sc=A53',#5
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_sc_donates_to_target_sc=A56',#6
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_sc_accepts_from_target_sc=A57',#7
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_bb_donates_to_target_atom=A59:OE2',#8
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_bb_accepts_from_target_atom=A49:NE',#9
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_sc_donates_to_target_atom=A59:OE1',#A
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_sc_accepts_from_target_atom=A49:NH1',#B
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_HIS_accepts_from_target_bb=A55',#C
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_HIS_accepts_from_target_sc=A58',#D
            '++upstream_inference_transforms.configs.HBondTargetSatisfactionInferenceTransform.binder_HIS_accepts_from_target_atom=A49:NH2',#E
 
            '++transforms.names=["AddConditionalInputs","CenterPostTransform","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++transforms.configs.CenterPostTransform.center_type=is_not_diffused',
        ])
        N_cond = len(hbond_satisfaction.get_target_hbond_satisfaction_keys_for_t1d())
        assert N_cond == 15
        mapped_calls = get_rfi(conf)
        print(mapped_calls[0]['t1d'].shape[-1])
        assert mapped_calls[0]['t1d'].shape[-1] == 80 + N_cond * 2, "If this throws, the target_hbond_satisfaction_cond isn't being written to t1d"

        # 0 'binder_bb_donates_to_target_bb',
        # 1 'binder_bb_accepts_from_target_bb',
        # 2 'binder_sc_donates_to_target_bb',
        # 3 'binder_sc_accepts_from_target_bb',
        # 4 'binder_bb_donates_to_target_sc',
        # 5 'binder_bb_accepts_from_target_sc',
        # 6 'binder_sc_donates_to_target_sc',
        # 7 'binder_sc_accepts_from_target_sc',
        # 8 'binder_bb_donates_to_target_atom',
        # 9 'binder_bb_accepts_from_target_atom',
        # A 'binder_sc_donates_to_target_atom',
        # B 'binder_sc_accepts_from_target_atom',
        # C 'binder_HIS_accepts_from_target_bb',
        # D 'binder_HIS_accepts_from_target_sc',
        # E 'binder_HIS_accepts_from_target_atom',
 
        # 45         6
        # 9012345678901234567890
        # RISITARTKKEAEKFAAILIKVFA
        # 904152C67DA3
        # B         8
        # E

        n = None
        conditions_in_order = [
            #atomized
            0, 4, 1, 5, 2, 12, 6, 7, 13,
            #atomized
            3,
            # N,CA,C,O,CB,CG,CD,NE,CZ,NH1,NH2 # arg
              n, n,n,n, n, n, n, 9, n, 11, 14,
            # N,CA,C,O,CB,CG,CD,OE1,OE2 #glu
              n, n,n,n, n, n, n, 10,  8
        ]

        assert len(conditions_in_order) == mapped_calls[0]['t1d'].shape[-2]

        t1d = mapped_calls[0]['t1d'][0,0].clone()
        offset = t1d.shape[-1] - N_cond *2

        for icond in range(N_cond):
            idx0 = conditions_in_order.index(icond)

            mask = t1d[:,offset + icond*2]
            value = t1d[:,offset + icond*2 + 1]

            assert torch.isclose(mask[idx0], torch.tensor(1.0))
            mask[idx0] = 0
            assert torch.allclose(mask, torch.tensor(0.0))

            assert torch.isclose(value[idx0], torch.tensor(1.0)) 
            value[idx0] = 0
            assert torch.allclose(mask, torch.tensor(0.0))

        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'target_hbond_sat', mapped_calls, rewrite=False, custom_comparator=cmp)

        
    @pytest.mark.generates_golden
    def test_inference_rfi_SSSprinkle(self):
        # Tests the SSSprinkle inference condition
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/two_chain.pdb',
            'contigmap.contigs=["30-30,0_B121-130"]',
            'inference.output_prefix=tmp/test_ss_sprinkle',
            '++inference.zero_weights=True',
            'contigmap.has_termini=[True,True]',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',

            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',

            '++upstream_inference_transforms.names=[SSSprinkleTransform]',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.active=True',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.background=HELIX',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.helix_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.strand_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.loop_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.mask_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.min_strand=2',
            "++upstream_inference_transforms.configs.SSSprinkleTransform.max_strand=2",
            '++upstream_inference_transforms.configs.SSSprinkleTransform.min_loop=3',
            "++upstream_inference_transforms.configs.SSSprinkleTransform.max_loop=3",
            '++upstream_inference_transforms.configs.SSSprinkleTransform.min_mask=4',
            "++upstream_inference_transforms.configs.SSSprinkleTransform.max_mask=4",

            '++transforms.names=["AddConditionalInputs","CenterPostTransform","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++transforms.configs.CenterPostTransform.center_type=is_not_diffused',

        ])

        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 80 + N_SS, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"

        ss_t1d = mapped_calls[0]['t1d'][0,0,:,-N_SS:]

        assert ss_t1d[:,SS_STRAND].sum() == 2
        assert ss_t1d[:,SS_HELIX].sum() == 30 - 2 - 3 - 4
        assert ss_t1d[:,SS_LOOP].sum() == 3
        assert ss_t1d[:,SS_MASK].sum() == 4 + 10

        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_sprinkle', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


    @pytest.mark.generates_golden
    def test_inference_rfi_SSSprinkle_explicit(self):
        # Tests the SSSprinkle inference condition when using use_this_ss_ordering
        run_inference.make_deterministic()
        conf = construct_conf([
            'diffuser.T=1',
            'inference.num_designs=1',
            'inference.input_pdb=test_data/two_chain.pdb',
            'contigmap.contigs=["30-30,0_B121-130"]',
            'inference.output_prefix=tmp/test_ss_sprinkle',
            '++inference.zero_weights=True',
            'contigmap.has_termini=[True,True]',
            '++extra_tXd=[ss_adj_cond]',
            '++extra_tXd_params.ss_adj_cond={}',

            '++upstream_training_transforms.names=[GenerateSSADJTrainingTransform]',
            '++upstream_training_transforms.configs.GenerateSSADJTrainingTransform={p_is_ss_example:1,p_is_adj_example:1}',

            '++upstream_inference_transforms.names=[SSSprinkleTransform]',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.active=True',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.background=MASK',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.helix_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.strand_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.loop_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.mask_chunk_size=1',
            '++upstream_inference_transforms.configs.SSSprinkleTransform.use_this_ss_ordering="HEEHEL"',

            '++transforms.names=["AddConditionalInputs","CenterPostTransform","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++transforms.configs.CenterPostTransform.center_type=is_not_diffused',

        ])

        mapped_calls = get_rfi(conf)
        assert mapped_calls[0]['t1d'].shape[-1] == 80 + N_SS, "If this throws, the ss_adj isnt being written to t1d"
        assert mapped_calls[0]['t2d'].shape[-1] == 72, "If this throws, the ss_adj isnt being written to t2d"

        ss_t1d = mapped_calls[0]['t1d'][0,0,:,-N_SS:]

        assert ss_t1d[:,SS_STRAND].sum() == 3
        assert ss_t1d[:,SS_HELIX].sum() == 2
        assert ss_t1d[:,SS_LOOP].sum() == 1
        assert ss_t1d[:,SS_MASK].sum() == 30 - 6 + 10

        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'rfi_ss_sprinkle_explicit', mapped_calls, rewrite=REWRITE, custom_comparator=cmp)



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

    def test_only_guidepost_positions(self):
        '''
        Tests that only_guidepost_positions works and that the inference code still assigns motifs correctly
        '''

        leading_free = 5
        run_inference.make_deterministic()
        pdb, conf = infer([
            'diffuser.T=1',
            'inference.input_pdb=test_data/1qys.pdb',
            'inference.contig_as_guidepost=True',
            'inference.only_guidepost_positions="A62,A64"',
            "+contigmap.contig_atoms=\"{'A64':'CB','A65':'CB'}\"",
            f"contigmap.contigs=['{leading_free},A62-65,5']",
            'contigmap.length=null',
            'inference.guidepost_xyz_as_design=True'
        ])

        trb = get_trb(conf)
        assert assertpy.assert_that(trb['indep']['is_gp'].sum()).is_equal_to(2)

        # The guideposts could technically be swapped, but 2 residues were motifs and we know they can't swap
        assert trb['con_hal_idx0'][1] == leading_free + 1
        assert trb['con_hal_idx0'][3] == leading_free + 3

        # We know this guy got atomized so it must be in this array
        assert leading_free + 3 in trb['atomize_indices2atomname']


    def test_filters_output(self):
        '''
        Tests that the filter system actually works at inference

        We do this by defining a filter that reports the call count
        The filter will fail until it has been called twice at which point we check the scorefile to ensure it
        was indeed called twice
        '''

        # We have to construct our own conf so we can figure out the scorefile name
        conf = construct_conf([
                'diffuser.T=1',
                'inference.input_pdb=test_data/1qys.pdb',
                "contigmap.contigs=['1-1,A64-64']",
                'filters.names=["MyFilter:TestFilter"]',
                '+filters.configs.MyFilter.t=1',
                '+filters.configs.MyFilter.test_value_threshold=2',
                '+filters.configs.MyFilter.suffix=_suff',
                '+filters.configs.MyFilter.prefix=pre_',
                '+filters.configs.MyFilter.verbose=True',
            ])

        scorefile_name = conf.inference.output_prefix + '_out.sc'
        if os.path.exists(scorefile_name):
            os.remove(scorefile_name)


        # TestFilter requires us to overwrite it's get_test_value() function
        fake_forward = mock.patch.object(TestFilter, "get_test_value", autospec=True)

        def side_effect(self, *args, **kwargs):
            side_effect.call_count += 1
            return side_effect.call_count
        side_effect.call_count = 0

        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect

            # Do inference
            run_inference.make_deterministic()
            run_inference.main(conf)

        df = pd.read_csv(scorefile_name, sep='\s+')
        assert 'pre_t-1_test_value_suff' in list(df)
        assert len(df) == 1
        assert df['pre_t-1_test_value_suff'].iloc[0] == 2

        trb = get_trb(conf)
        assert trb['scores']['pre_t-1_test_value_suff'] == 2

    def test_filters_fail(self):
        '''
        Tests the behavior when the filters fail to produce an output

        A pdb should not be generated and there should be an empty .trb to prove we tried
        '''

        output_prefix = 'debug/aa_small' + str(os.getpid()) + '_%.4f'%(time.time() % 100)

        for actual_fail_criterion in ['++filters.max_attempts_per_design=2', '++filters.max_steps_per_design=2']:

            # We have to construct our own conf so we can figure out the names

            if hydra.core.global_hydra.GlobalHydra().is_initialized():
                hydra.core.global_hydra.GlobalHydra().clear()

            conf = construct_conf([
                    'diffuser.T=1',
                    f'inference.output_prefix={output_prefix}',
                    'inference.input_pdb=test_data/1qys.pdb',
                    "contigmap.contigs=['1-1,A64-64']",
                    'filters.names=["MyFilter:TestFilter"]',
                    '+filters.configs.MyFilter.t=1',
                    '+filters.configs.MyFilter.test_value_threshold=10',
                    '+filters.configs.MyFilter.suffix=_suff',
                    '+filters.configs.MyFilter.prefix=pre_',
                    actual_fail_criterion,
                ])

            # Delete them ahead of time
            long_trb_name = conf.inference.output_prefix + '_0-atomized-bb-True.trb'
            short_trb_name = conf.inference.output_prefix + '_0.trb'
            long_pdb_name = conf.inference.output_prefix + '_0-atomized-bb-True.pdb'
            if os.path.exists(long_trb_name):
                os.remove(long_trb_name)
            if os.path.exists(short_trb_name):
                os.remove(short_trb_name)
            if os.path.exists(long_pdb_name):
                os.remove(long_pdb_name)


            # Overwrite the get_test_value() function so we can keep track of how many times this is called
            fake_forward = mock.patch.object(TestFilter, "get_test_value", autospec=True)

            def side_effect(self, *args, **kwargs):
                side_effect.call_count += 1
                return side_effect.call_count
            side_effect.call_count = 0

            with fake_forward as mock_forward:
                mock_forward.side_effect = side_effect

                run_inference.make_deterministic()
                run_inference.main(conf)


            assert side_effect.call_count == 2
            assert not os.path.exists(long_pdb_name)
            assert not os.path.exists(long_trb_name)
            assert os.path.exists(short_trb_name)

            trb = np.load(short_trb_name,allow_pickle=True)
            assert len(trb) == 0

            os.remove(short_trb_name)



    def test_mid_run_modifiers(self):
        '''
        These things are so "protocol level" that there's not really a way to assert they are working

        Instead we make sure they don't crash

        I've tried several times and this comes out with different xyz on different machines
        '''

        output_prefix = 'debug/aa_small' + str(os.getpid()) + '_%.4f'%(time.time() % 100)

        # We have to construct our own conf so we can figure out the names

        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()

        conf = construct_conf([
                'diffuser.T=2',
                f'inference.output_prefix={output_prefix}',
                'inference.input_pdb=test_data/1qys.pdb',
                "contigmap.contigs=['1-1,A64-64']",
                'inference.contig_as_guidepost=True',

                'inference.custom_t_range=[2,-2,1]',
                'inference.final_step=1',
                'inference.start_str_self_cond_at_t=1',
                'inference.write_extra_ts=[2]',
                'inference.ORI_guess=True',
                'inference.fast_partial_trajectories=[[1,2]]',
                'inference.fpt_drop_guideposts=True',
                'inference.fpt_diffuse_chains=0',

            ])

        expected_suffixes = [
            '_0-atomized-bb-True_fpt-1-2.pdb',
            '_0-atomized-bb-True.pdb',
            '_0-atomized-bb-True_t2.pdb',
            '_0-atomized-bb-True.trb',
        ]

        # Delete them ahead of time
        for suffix in expected_suffixes:
            fname = conf.inference.output_prefix + suffix
            if os.path.exists(fname):
                os.remove(fname)

        run_inference.make_deterministic()
        run_inference.main(conf)

        # Make sure no one accidentally breaks these things
        # indep = aa_model.make_indep(conf.inference.output_prefix + expected_suffixes[0])

        # cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        # test_utils.assert_matches_golden(self, 'inference_mid_run_modifiers', indep, rewrite=False, custom_comparator=cmp)

        # Make sure all the expected files are there
        for suffix in expected_suffixes:
            fname = conf.inference.output_prefix + suffix
            assert os.path.exists(fname)
            os.remove(fname)



    def test_ORI_guess(self):
        '''
        The ORI guess causes a massive difference in output with tiny changes in input structure
        So instead let's just make sure it's working correctly rather than golden it
        '''

        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()

        conf = construct_conf([
                'diffuser.T=1',
                'inference.input_pdb=test_data/1qys.pdb',
                "contigmap.contigs=['1-1,A64-64']",
                'inference.ORI_guess=2',
                '++transforms.names=["AddConditionalInputs","CenterPostTransform"]',
                '++transforms.configs.CenterPostTransform.center_type="is_not_diffused"',
            ])

        # Overwrite the get_test_value() function so we can keep track of how many times this is called
        og_func = model_runners.NRBStyleSelfCond.sample_step
        fake_forward = mock.patch.object(model_runners.NRBStyleSelfCond, "sample_step", autospec=True)

        origins = []
        px0_diffused_locs = []
        not_diffused_locs = []

        def side_effect(self, *args, **kwargs):

            indep_in = args[1]
            not_diffused_locs.append(indep_in.xyz[~self.is_diffused,1,:].mean(axis=0))
            origins.append(self.extra_transform_kwargs['origin'])

            ret = og_func(self, *args, **kwargs)

            px0 = ret[0]
            px0_diffused_locs.append(px0[self.is_diffused,1,:].mean(axis=0))

            return ret

        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect

            run_inference.make_deterministic()
            run_inference.main(conf)


        assert len(origins) == 3, 'Something went wrong in the setup of ORI_guess'
        origins = torch.stack(origins)
        px0_diffused_locs = torch.stack(px0_diffused_locs)
        not_diffused_locs = torch.stack(not_diffused_locs)

        actual_diffused_com = origins[0] + px0_diffused_locs[0]
        assert torch.allclose(actual_diffused_com, origins[1]), "The origin_override kwarg to the dataloader didn't make it down all the way"

        actual_diffused_com1 = origins[1] + px0_diffused_locs[1]
        assert torch.allclose(actual_diffused_com1, origins[2]), "The origin_override kwarg to the dataloader didn't make it down all the way"

        assert torch.linalg.norm(not_diffused_locs[0]) < 0.1, 'CenterPostTransform failed'
        assert torch.linalg.norm(not_diffused_locs[1]) > 0.1, "Either you got incredibly unlucky or the origin_override kwarg didn't make it down"


class TestInferenceSetup(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    

    def test_t_setups(self):
        '''
        Blanket test of all the t_setups added in the first iteration
        '''
        conf = construct_conf([
                'diffuser.T=8',
                'inference.custom_t_range=[8,-7,6,5,4,1]',
                'inference.final_step=1',
                'inference.start_str_self_cond_at_t=4',
                'inference.write_extra_ts=[7,6,5]',
                'inference.ORI_guess=2',
                'inference.fast_partial_trajectories=[[1,8,2],[4,7]]',
                'inference.fpt_drop_guideposts=True',
                'inference.fpt_diffuse_chains=1',
            ])

        (
            ts,
            n_steps,
            self_cond,
            final_it,
            addtl_write_its,
            mid_run_modifiers,
        ) = setup_t_arrays(conf, 8)

        goal_ts           = torch.tensor([8,8,8,7,6,5,4,1,8,2,7])
        goal_n_steps      = torch.tensor([1,1,1,1,1,1,1,3,1,1,1])
        goal_self_cond    = torch.tensor([0,0,0,0,0,0,1,1,0,0,0]).bool()
        
        assert torch.allclose(ts, goal_ts), f'{ts} {goal_ts}'
        assert torch.allclose(n_steps, goal_n_steps), f'{n_steps} {goal_n_steps}'
        assert torch.allclose(self_cond, goal_self_cond), f'{self_cond} {goal_self_cond}'
        assert int(final_it) == 7

        look_for = [3,4,5,9,10]
        found = torch.zeros(len(look_for), dtype=bool)
        for it, suffix in addtl_write_its:
            it = int(it)
            assert it in look_for
            found[look_for.index(it)] = True

        assert isinstance(mid_run_modifiers[0][0], mrm.ReinitializeWithCOMOri)
        assert isinstance(mid_run_modifiers[1][0], mrm.ReinitializeWithCOMOri)
        assert isinstance(mid_run_modifiers[2][0], mrm.PartiallyDiffusePx0Toxt)
        assert int(mid_run_modifiers[2][0].t) == 7
        assert isinstance(mid_run_modifiers[7][0], mrm.RemoveGuideposts)
        assert isinstance(mid_run_modifiers[7][1], mrm.DiffuseChains)
        assert len(mid_run_modifiers[7][1].diffused_chains) == 1
        assert mid_run_modifiers[7][1].diffused_chains[0] == 1

        for it, from_it in zip([7, 8, 9], [7, 8, 6]):
            assert isinstance(mid_run_modifiers[it][-2], mrm.ReplaceXtWithPx0), f'{it}'
            assert isinstance(mid_run_modifiers[it][-1], mrm.PartiallyDiffusePx0Toxt), f'{it}'
            assert int(mid_run_modifiers[it][-1].t) == int(ts[it+1]), f'{it} {from_it}, {mid_run_modifiers[it][-1].t} {ts[it+1]}'





if __name__ == '__main__':
        unittest.main()
