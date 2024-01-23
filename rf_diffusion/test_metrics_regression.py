import unittest
import copy
import time
import torch

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
        }
        self.idealized_residue_rmsd = metrics.IdealizedResidueRMSD(None)
        
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
            msg += f'Hint: torch.is_anomaly_enabled() is True. Try turning it off.'

        self.assertLess(run_time, 5, msg)

    def test_reached_minimum(self):
        '''
        Check that the idealizer took enough steps to reach a minima
        '''
        metrics_inputs = copy.deepcopy(self.metrics_inputs)

        # 100 steps is the default
        rmsd_100 = self.idealized_residue_rmsd(**metrics_inputs)

        metrics_inputs['steps'] = 200
        rmsd_200 = self.idealized_residue_rmsd(**metrics_inputs)

        metrics_inputs['steps'] = 300
        rmsd_300 = self.idealized_residue_rmsd(**metrics_inputs)
        
        # All rmsds should be about the same
        rmsds = torch.tensor([rmsd_100, rmsd_200, rmsd_300])
        rmsd_range = rmsds.max() - rmsds.min()

        self.assertLess(rmsd_range, 0.1, 'Idealizing a residue for 100 steps did not reach a minima.')

