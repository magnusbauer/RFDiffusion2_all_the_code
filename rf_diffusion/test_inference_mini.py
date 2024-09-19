'''
This file contains a simplified version of test_inference.py that
uses the same inference task to run a variety of different tests,
checking the inference outputs or model inputs at various points.

The intention is for it to be more interpretable than test_inference.py
and serve as a template for how test_inference.py can be improved.

This test should be deterministic on CPU.  Several different CPU
architectures were able to pass it with 0-tolerance, but if you're
unsure, try getting a CPU with: qlogin -p cpu --mem=10g -c 1
'''
import unittest
import pytest

import hydra
from icecream import ic

import test_utils
import run_inference
from functools import partial
from rf2aa import tensor_util
from rf_diffusion import inference
from rf_diffusion.test_inference import get_rfi, infer, NA_adaptor, construct_conf

ic.configureOutput(includeContext=True)

REWRITE = False
class HydraTest(unittest.TestCase):

    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()

class TestInferenceOutputPDB(HydraTest):
    @pytest.mark.generates_golden
    def test_t1(self):
        test_name = 'T01'
        output_suffix = 'current'
        if REWRITE:
            output_suffix = 'golden'
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=1',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{test_name}_{output_suffix}',
            'inference.write_trajectory=True',
            'inference.write_trb_indep=True',
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        pdb_contents = NA_adaptor(pdb_contents)
        cmp = partial(tensor_util.cmp, atol=0, rtol=0)
        test_utils.assert_matches_golden(self, test_name, pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_t2(self):
        test_name = 'T02'
        output_suffix = 'current'
        if REWRITE:
            output_suffix = 'golden'
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{test_name}_{output_suffix}',
            'inference.write_trajectory=True',
            'inference.write_trb_indep=True',
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        pdb_contents = NA_adaptor(pdb_contents)
        cmp = partial(tensor_util.cmp, atol=0, rtol=0)
        test_utils.assert_matches_golden(self, test_name, pdb_contents, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_t10(self):
        test_name = 'T10'
        output_suffix = 'current'
        if REWRITE:
            output_suffix = 'golden'
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=10',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{test_name}_{output_suffix}',
            'inference.write_trajectory=True',
            'inference.write_trb_indep=True',
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        pdb_contents = NA_adaptor(pdb_contents)
        cmp = partial(tensor_util.cmp, atol=0, rtol=0)
        test_utils.assert_matches_golden(self, test_name, pdb_contents, rewrite=REWRITE, custom_comparator=cmp)


class TestInferenceNetworkInput(HydraTest):

    @pytest.mark.generates_golden
    def test_rfi_t1(self):
        test_name = 'test_rfi_t1'
        run_inference.make_deterministic()
        conf = construct_conf([ 
            'diffuser.T=2',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{test_name}',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf, 0)
        cmp = partial(tensor_util.cmp, atol=0, rtol=0)
        test_utils.assert_matches_golden(self, test_name, mapped_calls, rewrite=REWRITE, custom_comparator=cmp)

    @pytest.mark.generates_golden
    def test_rfi_t2(self):
        test_name = 'test_rfi_t2'
        run_inference.make_deterministic()
        conf = construct_conf([ 
            'diffuser.T=2',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{test_name}',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf, 1)
        cmp = partial(tensor_util.cmp, atol=0, rtol=0)
        test_utils.assert_matches_golden(self, test_name, mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


    @pytest.mark.generates_golden
    def test_rfi_t3(self):
        test_name = 'test_rfi_t3'
        run_inference.make_deterministic()
        conf = construct_conf([ 
            'diffuser.T=3',
            'inference.num_designs=1',
            f'inference.output_prefix=tmp/{test_name}',
            '++inference.zero_weights=True',            
        ])
        mapped_calls = get_rfi(conf, 2)
        cmp = partial(tensor_util.cmp, atol=0, rtol=0)
        test_utils.assert_matches_golden(self, test_name, mapped_calls, rewrite=REWRITE, custom_comparator=cmp)


if __name__ == '__main__':
        unittest.main()
