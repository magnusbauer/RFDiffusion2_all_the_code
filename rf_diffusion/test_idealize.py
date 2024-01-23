import unittest
import torch

import rf_diffusion
PKG_DIR = rf_diffusion.__path__[0]
from rf_diffusion.inference.utils import parse_pdb
from rf_diffusion import idealize


class TestSideChainIdealization(unittest.TestCase):
    def test_input_noise_tolerance(self):
        '''
        Tests that...
            1. Idealizing an experimental structure yields a very low rmsd.
            2. If noise is added to corrupt an experimental structure, the rmsd 
                between the idealized and the original experimental structure 
                should be less than the amount of corrupting noise.
        '''
        # Prepare an experimental structure
        parsed_pdb = parse_pdb(f'{PKG_DIR}/test_data/1qys.pdb')

        is_resolved = parsed_pdb['mask']

        xyz14 = torch.tensor(parsed_pdb['xyz'])
        xyz14[~is_resolved] = torch.nan
        xyz = torch.full((1, 91, 36, 3), torch.nan)
        xyz[0, :, :14] = xyz14

        seq = torch.tensor(parsed_pdb['seq'])[None]

        # Corrupt and idealize at different noise scales
        rmsds_to_true_structure = []
        noise_scales = torch.linspace(0, 0.2, 10)
        for noise_scale in noise_scales:
            # Noise only side chains
            noise = noise_scale * torch.randn_like(xyz)
            noise[0, :, :5] = 0.  # No noise added to bb atoms
            xyz_noised = xyz + noise
            
            # Idealize
            xyz_ideal = idealize.idealize_pose(xyz_noised, seq)[0]
            
            # Calculate rmsd to the true structure
            rmsds_to_true_structure.append(idealize.calc_residue_rmsds(xyz, xyz_ideal, seq)[0])

        rmsds_to_true_structure = torch.concat(rmsds_to_true_structure)

        # There should only be a small rmsd when idealizing an experimental structure
        rmsd_to_true_structure_no_noise = rmsds_to_true_structure[0]
        self.assertLess(rmsd_to_true_structure_no_noise, 0.04)

        # The rmsd to the true structure roughly scales with the noise scale
        self.assertTrue(( rmsds_to_true_structure <= (noise_scales + rmsd_to_true_structure_no_noise) ).all())
