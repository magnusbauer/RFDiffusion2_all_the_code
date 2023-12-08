import copy
from collections import defaultdict
import shutil
import os
from functools import partial
import sys
import torch
import assertpy
import unittest
import numpy as np
from icecream import ic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
import rf2aa
import aa_model
from aa_model import AtomizeResidues, Indep, Model, make_indep
import atomize
import test_utils
from aa_model import Model, make_indep
import inference.utils
import rf_diffusion.contigs as contigs
import pytest
from rf2aa import tensor_util
import run_inference
ic.configureOutput(includeContext=True)

class TestContigs(unittest.TestCase):

    def test_two_chains_swapped(self):
        input_pdb = './benchmark/input/2j0l.pdb'

        conf = test_utils.construct_conf(inference=True) 
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                    contigs=['10_A441-450'],
                                    has_termini=[True, True]
                                    )

        contig_map_attributes = {
            "hal": contig_map.hal,
            "chain_order": contig_map.chain_order,
            "hal_idx0": contig_map.hal_idx0,
            "ref_idx0": contig_map.ref_idx0,
            "atomize_indices2atomname": contig_map.atomize_indices2atomname,
            "inpaint_str": contig_map.inpaint_str
        }
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, 'contigs_two_chains_swapped', contig_map_attributes, rewrite=REWRITE, custom_comparator=cmp)

                
    def test_one_chain_covale(self):
        input_pdb = 'benchmark/input/3l0f_covale.pdb'
        ligand_name = 'CYC'

        conf = test_utils.construct_conf(inference=True) 
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                contigs=['10-60,A84-87,10-60'],
                                contig_atoms="{'A84':'CA,C,N,O,CB,SG'}",
                                length='100-100'
                                )   

        contig_map_attributes = {
            "hal": contig_map.hal,
            "chain_order": contig_map.chain_order,
            "hal_idx0": contig_map.hal_idx0,
            "ref_idx0": contig_map.ref_idx0,
            "atomize_indices2atomname": contig_map.atomize_indices2atomname,
            "inpaint_str": contig_map.inpaint_str
        }
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, 'contigs_one_chain_covale', contig_map_attributes, rewrite=REWRITE, custom_comparator=cmp)

    def test_two_chains(self):
        input_pdb = './benchmark/input/2j0l.pdb'

        conf = test_utils.construct_conf(inference=True) 
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                    contigs=['A441-450_10'],
                                    has_termini=[True, True]
                                    )

        contig_map_attributes = {
            "hal": contig_map.hal,
            "chain_order": contig_map.chain_order,
            "hal_idx0": contig_map.hal_idx0,
            "ref_idx0": contig_map.ref_idx0,
            "atomize_indices2atomname": contig_map.atomize_indices2atomname,
            "inpaint_str": contig_map.inpaint_str
        }
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, 'contigs_two_chains', contig_map_attributes, rewrite=REWRITE, custom_comparator=cmp)

    def test_two_chains_swapped(self):
        input_pdb = './benchmark/input/2j0l.pdb'

        conf = test_utils.construct_conf(inference=True) 
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                    contigs=['10_A441-450'],
                                    has_termini=[True, True]
                                    )

        contig_map_attributes = {
            "hal": contig_map.hal,
            "chain_order": contig_map.chain_order,
            "hal_idx0": contig_map.hal_idx0,
            "ref_idx0": contig_map.ref_idx0,
            "atomize_indices2atomname": contig_map.atomize_indices2atomname,
            "inpaint_str": contig_map.inpaint_str
        }
        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
        test_utils.assert_matches_golden(self, 'contigs_two_chains_swapped', contig_map_attributes, rewrite=REWRITE, custom_comparator=cmp)

    def test_3_chain_termini(self):
        input_pdb = './benchmark/input/2j0l.pdb'

        conf = test_utils.construct_conf(inference=True)
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map = contigs.ContigMap(target_feats,
                                    contigs=['A441-443,A445-447_3_3'],
                                    has_termini=[True, True, False])

        # Extract the required chain termini from the chains attribute
        extracted_chain_termini = [(chain[1], chain[2]) for chain in contig_map.chains]

        # We expect the following termini in the format (N-terminus, C-terminus)
        expected_termini = [(0, 6), (6, 9), (9, 12)]

        # Assert that each expected terminus is in the extracted termini
        for terminus in expected_termini:
            assertpy.assert_that(extracted_chain_termini).contains(terminus)

    def test_3_chain_termini_swapped(self):
        input_pdb = './benchmark/input/2j0l.pdb'

        conf = test_utils.construct_conf(inference=True)
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map = contigs.ContigMap(target_feats,
                                    contigs=['3_3_A441-443,A445-447'],
                                    has_termini=[True, True, False])

        # Extract the required chain termini from the chains attribute
        extracted_chain_termini = [(chain[1], chain[2]) for chain in contig_map.chains]

        # We expect the following termini in the format (N-terminus, C-terminus)
        expected_termini = [(0, 3), (3, 6), (6, 12)]

        # Assert that each expected terminus is in the extracted termini
        for terminus in expected_termini:
            assertpy.assert_that(extracted_chain_termini).contains(terminus)

    def test_3_contig_map_hal(self):
        input_pdb = './benchmark/input/2j0l.pdb'

        conf = test_utils.construct_conf(inference=True)
        adaptor = aa_model.Model(conf)

        run_inference.seed_all()
        target_feats = inference.utils.process_target(input_pdb)
        contig_map = contigs.ContigMap(target_feats,
                                    contigs=['3_3_A441-443,A445-447'],
                                    has_termini=[True, True, False])

        # Extract chain letters and indices from contig_map.hal
        extracted_chain_letters, extracted_indices = zip(*contig_map.hal)

        print(contig_map.has_termini)

        # Expected chain letters and indices
        expected_chain_letters = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C']
        expected_indices = [1, 2, 3, 36, 37, 38, 71, 72, 73, 74, 75, 76]

        # Assert that the extracted chain letters and indices match the expected values
        assertpy.assert_that(extracted_chain_letters).is_equal_to(tuple(expected_chain_letters))
        assertpy.assert_that(extracted_indices).is_equal_to(tuple(expected_indices))

REWRITE = False
if __name__ == '__main__':
        unittest.main()
