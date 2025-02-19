# tests for ca RFdiffusion inference
import torch 
import numpy as np 
import unittest 
import pickle 
import hydra 
from hydra import initialize, compose 
from unittest import mock 

import run_inference 
from test_ca_rfd_sm_train import ExitMockCall

def get_ca_config(overrides=[]):
    """
    Create a config for testing.
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    initialize(config_path="config/inference")
    conf = compose(config_name='test_ca_rfd_inference.yaml', overrides=overrides, return_hydra_config=True)
    return conf


class TestCARFDFeaturization(unittest.TestCase):
    """
    Tests for CA RFdiffusion inference.
    """

    @classmethod
    # @mock.patch.object('aa_model.Model', 'insert_contig_pre_atomization', side_effect=insert_contig_side_effect)
    def setUpClass(cls): 
        
        cls.conf = get_ca_config()

        try: 
            run_inference.main(cls.conf)
        except ExitMockCall as e: 
            pass 


    def test_inference(self):
        pass