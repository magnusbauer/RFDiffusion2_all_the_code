# tests for ca RFdiffusion refinement 
import torch 
import numpy as np 
import unittest 
import pickle 
import hydra 
from hydra import initialize, compose 
from unittest import mock 

# rfdiffusion imports 
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
import run_inference 
from test_ca_rfd_sm_train import ExitMockCall

import pdb

def rfold_side_effect(*args, **kwargs): 
    # mocks the fwd in LegacyRosettaFoldModule 
    raise ExitMockCall('Captured forward inputs')


def get_ca_config(overrides=[]):
    """
    Create a config for testing.
    """
    hydra.core.global_hydra.GlobalHydra().clear()
    initialize(config_path="config/inference")
    conf = compose(config_name='test_ca_rfd_refinement.yaml', overrides=overrides, return_hydra_config=True)
    return conf

#### Side effects and data catching #### 
PATCH_SAVE_DICT = {} # save data here 


class TestCARFDFeaturization(unittest.TestCase):
    """
    Tests for CA RFdiffusion inference.
    """

    @classmethod
    @mock.patch.object(LegacyRoseTTAFoldModule, 'forward', side_effect=rfold_side_effect)
    def setUpClass(cls, mock_rfold_fwd): 
        """
        Load goldens, get config, run inference all the way up to fwd pass. 
        Then, compare inputs at fwd pass with goldens.
        """

        cls.load_goldens() # load goldens 
        cls.conf = get_ca_config() # get config 

        try: 
            run_inference.main(cls.conf)
        except ExitMockCall as e: 
            pass 

        # save inputs to live fwd pass
        assert len(mock_rfold_fwd.call_args_list) == 1
        cls.kall = mock_rfold_fwd.call_args_list[0]

        # all the things we caught with side effects 
        cls.patch_caught = PATCH_SAVE_DICT

    @classmethod
    def load_goldens(cls):
        """
        Load goldens to compare with.
        """
        goldens_path = './goldens/refinement_rfi_dict_before_fwd.pt'
        data = torch.load(goldens_path)
        cls.golden_rfi = data

    def test_msas(self): 

        want_latent = self.golden_rfi['msa_latent']
        want_full = self.golden_rfi['msa_full']
        
        got_latent = self.kall.kwargs['msa_latent']
        got_full = self.kall.kwargs['msa_full']

        want_tmp = want_latent.squeeze().argmax(-1)
        got_tmp = got_latent.squeeze().argmax(-1)
        pdb.set_trace()
        torch.testing.assert_allclose(want_latent.squeeze().argmax(-1), 
                                      got_latent.squeeze().argmax(-1))

        torch.testing.assert_allclose(want_full, got_full)