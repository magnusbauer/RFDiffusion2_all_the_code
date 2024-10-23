import pytest
import unittest
import copy
import time
import torch
from omegaconf import OmegaConf
import pickle
import io

import rf_diffusion
PKG_DIR = rf_diffusion.__path__[0]
from rf_diffusion import metrics
from rf_diffusion import test_utils


class TestIdealizedResidueRMSD(unittest.TestCase):
    def setUp(self):
        super().setUp()
        info = test_utils.read(f'{PKG_DIR}/test_data/metrics_inputs.pkl')
        self.metrics_inputs = {
            'indep': info['metrics_inputs']['indep'],
            'pred_crds': info['metrics_inputs']['pred_crds'],
            'true_crds': info['metrics_inputs']['true_crds'],
            'input_crds': info['metrics_inputs']['input_crds'],
            't': info['metrics_inputs']['t'],
            'is_diffused': info['metrics_inputs']['is_diffused'],
            'point_types': info['metrics_inputs']['point_types'],
            'pred_crds_stack': info['metrics_inputs']['pred_crds_stack'],
            'atomizer_spec': info['metrics_inputs']['atomizer_spec'],
            'contig_as_guidepost': False,
        }
        conf = OmegaConf.create({'idealization_metric_n_steps': 100})
        self.idealized_residue_rmsd = metrics.IdealizedResidueRMSD(conf)
        
    @pytest.mark.noparallel
    def test_call_speed(self):
        '''
        Idealizing residues shouldn't take more than ~5 seconds.
        If it takes longer, something is slowing down the calculation.
        In the past, it has been slow if `torch.is_anomaly_enabled() ==  True`.
        Sometimes it seems to be CPU dependent. :/
        '''
        time1 = time.time()
        self.idealized_residue_rmsd(**self.metrics_inputs)
        time2 = time.time()
        run_time = time2 - time1

        msg = (
            f'It took {run_time:.2f} second to idealize a residue, '
            f'but should take less than 5. '
        )
        if torch.is_anomaly_enabled():
            msg += 'Hint: torch.is_anomaly_enabled() is True. Try turning it off.'

        self.assertLess(run_time, 7, msg)

    def test_reached_minimum(self):
        '''
        Check that the idealizer took enough steps to reach a minima
        '''
        metrics_inputs = copy.deepcopy(self.metrics_inputs)

        # 100 steps is the default
        rmsd_100 = self.idealized_residue_rmsd(**metrics_inputs)['rmsd_mean']

        metrics_inputs['steps'] = 200
        rmsd_200 = self.idealized_residue_rmsd(**metrics_inputs)['rmsd_mean']

        metrics_inputs['steps'] = 300
        rmsd_300 = self.idealized_residue_rmsd(**metrics_inputs)['rmsd_mean']
        
        # All rmsds should be about the same
        rmsds = torch.tensor([rmsd_100, rmsd_200, rmsd_300])
        rmsd_range = rmsds.max() - rmsds.min()

        self.assertLess(rmsd_range, 0.1, 'Idealizing a residue for 100 steps did not reach a minima.')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        else: return super().find_class(module, name)

# WIP
# class TestAllAtomRigid(unittest.TestCase):
#     def test_bond_lengths(self):
#         with open('epoch-0_step-0_i-0_DATASET-pdb_aa_MASK-get_tip_mask_POSITIONED-False_nSM-14_nMOTIF-4_r3t-0.979796_so3t-0.979796_info.pkl', 'rb') as fh:
#             info = CPU_Unpickler(fh).load()
        
#         metrics_inputs = info['metrics_inputs']

#         out = metrics.all_atom_rigid(**metrics_inputs)
#         ic(out)
#         ic(out['any_any:any_any'])

if __name__ == '__main__':
        unittest.main()
