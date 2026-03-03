import unittest
from unittest import mock

import torch

from rf_diffusion import ppi


class DummyIndep:
    def __init__(self, xyz, same_chain):
        self.xyz = xyz
        self.same_chain = same_chain

    def length(self):
        return self.xyz.shape[0]


class TestPPI(unittest.TestCase):
    def test_normal_to_target_hotspot_falls_back_to_diffused_target_ca(self):
        # [L, N/Ca/C, xyz]
        xyz = torch.zeros((5, 3, 3), dtype=torch.float32)
        # Target chain A: residues 0,1,2. Binder chain B: residues 3,4.
        xyz[0, 1] = torch.tensor([0.0, 0.0, 0.0])   # hotspot residue (diffused)
        xyz[1, 1] = torch.tensor([5.0, 0.0, 0.0])   # diffused target residue near hotspot
        xyz[2, 1] = torch.tensor([25.0, 0.0, 0.0])  # fixed target residue far from hotspot
        xyz[3, 1] = torch.tensor([1.0, 0.0, 0.0])   # diffused binder residue near hotspot
        xyz[4, 1] = torch.tensor([30.0, 0.0, 0.0])  # fixed binder residue far from hotspot

        same_chain = torch.zeros((5, 5), dtype=torch.bool)
        same_chain[:3, :3] = True
        same_chain[3:, 3:] = True
        indep = DummyIndep(xyz=xyz, same_chain=same_chain)

        is_diffused = torch.tensor([True, True, False, True, False], dtype=torch.bool)
        conditions_dict = {'is_hotspot': torch.tensor([True, False, False, False, False], dtype=torch.bool)}

        cb = torch.zeros((5, 3), dtype=torch.float32)
        cb[0] = torch.tensor([0.0, 0.0, 0.0])

        with mock.patch.object(ppi, 'Cb_or_atom', return_value=cb):
            origin = ppi.get_origin_normal_to_target_hotspot(indep, conditions_dict, is_diffused, normal_extension=10)

        # fallback near set is residues 0 and 1 (same chain as hotspot); COM is [2.5, 0, 0],
        # so direction is [-1, 0, 0] and origin is hotspot + 10 * direction.
        want = torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float32)
        assert torch.allclose(origin, want), f'{origin=} {want=}'
        assert torch.isfinite(origin).all(), origin

    def test_normal_to_target_hotspot_asserts_when_no_nearby_ca(self):
        xyz = torch.zeros((4, 3, 3), dtype=torch.float32)
        xyz[:, 1] = torch.tensor([
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ])

        same_chain = torch.zeros((4, 4), dtype=torch.bool)
        same_chain[:2, :2] = True
        same_chain[2:, 2:] = True
        indep = DummyIndep(xyz=xyz, same_chain=same_chain)

        is_diffused = torch.tensor([True, True, False, False], dtype=torch.bool)
        conditions_dict = {'is_hotspot': torch.tensor([True, False, False, False], dtype=torch.bool)}

        cb = torch.zeros((4, 3), dtype=torch.float32)
        cb[0] = torch.tensor([0.0, 0.0, 0.0])

        with mock.patch.object(ppi, 'Cb_or_atom', return_value=cb):
            with self.assertRaisesRegex(AssertionError, 'no CA within 10A'):
                ppi.get_origin_normal_to_target_hotspot(indep, conditions_dict, is_diffused, normal_extension=10)

    def test_normal_to_target_hotspot_asserts_on_degenerate_vector(self):
        xyz = torch.zeros((1, 3, 3), dtype=torch.float32)
        xyz[0, 1] = torch.tensor([0.0, 0.0, 0.0])
        same_chain = torch.ones((1, 1), dtype=torch.bool)
        indep = DummyIndep(xyz=xyz, same_chain=same_chain)

        is_diffused = torch.tensor([False], dtype=torch.bool)
        conditions_dict = {'is_hotspot': torch.tensor([True], dtype=torch.bool)}

        cb = torch.zeros((1, 3), dtype=torch.float32)
        cb[0] = torch.tensor([0.0, 0.0, 0.0])

        with mock.patch.object(ppi, 'Cb_or_atom', return_value=cb):
            with self.assertRaisesRegex(AssertionError, 'degenerate normal vector'):
                ppi.get_origin_normal_to_target_hotspot(indep, conditions_dict, is_diffused, normal_extension=10)


if __name__ == '__main__':
    unittest.main()
