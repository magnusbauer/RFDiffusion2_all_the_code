import itertools
import sys
import os

import addict
import unittest
from icecream import ic
import torch
import numpy as np

import inference.utils
import aa_model
import contigs
import bond_geometry
import perturbations
import atomize

# def is_se3_invariant(loss, true, pred):

class TestLoss(unittest.TestCase):

    def test_atom_bond_loss(self):
        test_pdb = 'benchmark/input/gaa.pdb'
        
        target_feats = inference.utils.process_target(test_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                       contigs=['2,A518-518'],
                                       contig_atoms="{'A518':'CA,C,N,O,CB,CG,OD1,OD2'}",
                                       length='3-3',
                                       )
        indep = aa_model.make_indep(test_pdb)
        adaptor = aa_model.Model({})
        indep_contig,is_diffused,_ = adaptor.insert_contig(indep, contig_map)

        true = indep_contig.xyz

        perturbed = perturbations.se3_perturb(true)

        expected_losses = list(f'{a}:{b}' for a,b in itertools.combinations_with_replacement(
            ['diffused_residue', 'motif_atom'], 2))
        bond_losses = bond_geometry.calc_atom_bond_loss(indep_contig, perturbed, is_diffused)
        for k in expected_losses:
            self.assertLess(bond_losses.pop(k), 1e-6, msg=k)
        for k, v in bond_losses.items():
            self.assertTrue(torch.isnan(v), msg=k)
        
        perturbed = true.clone()
        T = torch.tensor([1,1,1])
        perturbed[-1,1,:] += T
        bond_losses = bond_geometry.calc_atom_bond_loss(indep_contig, perturbed, is_diffused)
        should_change = 'motif_atom:motif_atom'
        bond_loss = bond_losses.pop(should_change)
        self.assertGreater(bond_loss, 0.1)
        for k in expected_losses:
            if k == should_change:
                continue
            self.assertLess(bond_losses.pop(k), 1e-6, msg=k)
        for k, v in bond_losses.items():
            self.assertTrue(torch.isnan(v), msg=k)

    def test_rigid_loss(self):
        test_pdb = 'benchmark/input/gaa.pdb'
        
        target_feats = inference.utils.process_target(test_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                       contigs=['2,A518-518'], # Asp518
                                       contig_atoms="{'A518':'CG,OD1,OD2'}",
                                       length='3-3',
                                       )
        indep, metadata = aa_model.make_indep(test_pdb, return_metadata=True)

        conf = addict.Dict()
        adaptor = aa_model.Model(conf)
        indep_contig,is_diffused,_ = adaptor.insert_contig(indep, contig_map, metadata=metadata)

        true = indep_contig.xyz
        perturbed = perturbations.se3_perturb(true)
        T = torch.tensor([1,1,1])
        cg_atomized_idx = atomize.atomized_indices_atoms(adaptor.atomizer, {2: ['CG']})
        perturbed[cg_atomized_idx,1,:] += T

        point_ids = aa_model.get_point_ids(indep_contig, adaptor.atomizer)
        point_types = aa_model.get_point_types(indep_contig, adaptor.atomizer)
        rigid_groups = bond_geometry.find_all_rigid_groups_human_readable(indep_contig.bond_feats, point_ids)
        ic(rigid_groups)

        rigid_losses = bond_geometry.calc_rigid_loss(indep_contig, perturbed, indep_contig.xyz, is_diffused, point_types)
        ic(rigid_losses)

        self.assertLess(rigid_losses['max']['fine.diffused_atomized'], 1e-6)
        self.assertGreater(rigid_losses['max']['fine.diffused_atomized:motif_atomized'], 1)
        self.assertEqual(
            rigid_losses['max']['fine.diffused_atomized:motif_atomized'],
            rigid_losses['max']['determined'])



if __name__ == '__main__':
        unittest.main()

