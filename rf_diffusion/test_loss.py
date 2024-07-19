
import unittest
from icecream import ic
import torch

import aa_model
import bond_geometry
import perturbations
import atomize
import metrics
import test_utils
import rf_diffusion.inference.data_loader

# def is_se3_invariant(loss, true, pred):

class TestLoss(unittest.TestCase):

    def test_atom_bond_loss(self):
        conf = test_utils.construct_conf(inference=True, overrides=["contigmap.contigs=['2,A518-518']",
                                                                    "contigmap.contig_atoms=\"{'A518':'CA,C,N,O,CB,CG,OD1,OD2'}\"",
                                                                    "contigmap.length='3-3'",
                                                                    "inference.input_pdb='benchmark/input/gaa.pdb'"])

        # Load the input data during inference
        dataset = rf_diffusion.inference.data_loader.InferenceDataset(conf)
        _, _, indep_contig, _, is_diffused, atomizer, contig_map, t_step_input, _ = next(iter(dataset))

        point_types = aa_model.get_point_types(indep_contig, atomizer)
        true = indep_contig.xyz

        def simplify(bond_losses):
            for k, v in list(bond_losses.items()):
                for e in k.split(':'):
                    if e.startswith('any') or e.endswith('any') or e.endswith('atomized_backbone') or e.endswith('atomized_sidechain') or e.endswith('ligand'):
                        bond_losses.pop(k)
                        break
        
        def assert_all_zero(bond_losses):
            zero_losses = {k:v for k,v in bond_losses.items() if torch.isnan(v) or v < 1e-6}
            assert len(zero_losses) == len(bond_losses)

        perturbed = perturbations.se3_perturb(true)
        bond_losses = bond_geometry.calc_atom_bond_loss(indep_contig, perturbed, indep_contig.xyz, is_diffused, point_types)
        assert_all_zero(bond_losses)
        
        perturbed = true.clone()
        T = torch.tensor([1,1,1])
        perturbed[-1,1,:] += T
        bond_losses = bond_geometry.calc_atom_bond_loss(indep_contig, perturbed, indep_contig.xyz, is_diffused, point_types)
        simplify(bond_losses)
        should_change = 'motif_atomized:motif_atomized'
        bond_loss = bond_losses.pop(should_change)
        self.assertGreater(bond_loss, 0.1)
        assert_all_zero(bond_losses)

    def test_rigid_loss(self):
        conf = test_utils.construct_conf(inference=True, overrides=["contigmap.contigs=['2,A518-518']",
                                                                    "contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
                                                                    "contigmap.length='3-3'",
                                                                    "inference.input_pdb='benchmark/input/gaa.pdb'"])
        # Load the input data during inference
        dataset = rf_diffusion.inference.data_loader.InferenceDataset(conf)
        _, _, indep_contig, _, is_diffused, atomizer, contig_map, t_step_input, _ = next(iter(dataset))

        true = indep_contig.xyz
        perturbed = perturbations.se3_perturb(true)
        T = torch.tensor([1,1,1])
        cg_atomized_idx = atomize.atomized_indices_atoms(atomizer, {2: ['CG']})
        perturbed[cg_atomized_idx,1,:] += T

        point_ids = aa_model.get_point_ids(indep_contig, atomizer)
        point_types = aa_model.get_point_types(indep_contig, atomizer)
        rigid_groups = bond_geometry.find_all_rigid_groups_human_readable(indep_contig.bond_feats, point_ids)
        ic(rigid_groups)

        rigid_losses = bond_geometry.calc_rigid_loss(indep_contig, perturbed, indep_contig.xyz, is_diffused, point_types)
        ic(rigid_losses)

        self.assertLess(rigid_losses['max']['fine.diffused_atomized'], 1e-6)
        self.assertGreater(rigid_losses['max']['fine.diffused_atomized:motif_atomized'], 1)
        self.assertEqual(
            rigid_losses['max']['fine.diffused_atomized:motif_atomized'],
            rigid_losses['max']['determined'])
        
        bond_lengths = metrics.true_bond_lengths(indep_contig, indep_contig.xyz)
        ic(bond_lengths)



if __name__ == '__main__':
        unittest.main()

