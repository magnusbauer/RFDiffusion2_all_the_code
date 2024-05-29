import shutil
import os
import unittest

import torch
from icecream import ic
import numpy as np

from rf_diffusion.inference import utils as iu
from rf_diffusion.benchmark.util import hydra_utils
from rf_diffusion.benchmark import mpnn_designs_v2
from rf2aa import loss

class TestMPNN(unittest.TestCase):

    def test_motif_static(self):

        # Setup
        test_input_dir = 'test_data/mpnn_test/test_0'
        test_pdb = 'test_data/mpnn_test/test_0/run_train_0_cond0_0-atomized-bb-False.pdb'
        tmp_dir = 'test_data/mpnn_test/tmp/test_0'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(test_input_dir, tmp_dir)
        conf = hydra_utils.construct_conf(
            yaml_path='benchmark/configs/pipeline.yaml',
            overrides=[
                f'outdir={tmp_dir}',
                'start_step=mpnn',
                'stop_step=mpnn',
                'in_proc=True',
                'mpnn.num_seq_per_target=1',
            ])
        
        # Act
        mpnn_designs_v2.main(conf.mpnn)

        # Assert
        output_pdb = 'test_data/mpnn_test/tmp/test_0/mpnn/packed/run_train_0_cond0_0-atomized-bb-False_0_1.pdb'

        input_feats = iu.parse_pdb(test_pdb)
        output_feats = iu.parse_pdb(output_pdb)
        
        trb = np.load('test_data/mpnn_test/tmp/test_0/run_train_0_cond0_0-atomized-bb-False.trb', allow_pickle=True)
        is_motif = torch.tensor(trb['con_hal_idx0'])
        
        input_motif_xyz = input_feats['xyz'][is_motif]
        output_motif_xyz = output_feats['xyz'][is_motif]
        atom_mask = input_feats['mask'][is_motif]

        motif_sidechain_rmsd = loss.calc_crd_rmsd(
                torch.tensor(input_motif_xyz)[None],
                torch.tensor(output_motif_xyz)[None],
                torch.tensor(atom_mask)[None])
        ic(motif_sidechain_rmsd)
        self.assertLess(motif_sidechain_rmsd, 0.1)

if __name__ == '__main__':
        unittest.main()
