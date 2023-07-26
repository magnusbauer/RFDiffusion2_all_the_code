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
import contigs
import pytest
from rf2aa import tensor_util
import run_inference
ic.configureOutput(includeContext=True)

class TestRearrange(unittest.TestCase):

    def test_swap(self):
        indep = make_indep('benchmark/input/gaa.pdb', 'LG1')
        L = indep.length()
        input_copy = copy.deepcopy(indep)
        i = torch.arange(L)
        i[L-1] = L -2
        i[L-2] = L -1
        aa_model.rearrange_indep(indep, i)
        aa_model.rearrange_indep(indep, i)
 
        diff = test_utils.cmp_pretty(indep, input_copy)
        if diff:
            print(diff)
            self.fail(f'{diff=}')


    def test_all(self):
        indep = make_indep('benchmark/input/gaa.pdb', 'LG1')
        L = indep.length()
        input_copy = copy.deepcopy(indep)
        i = torch.randperm(L)
        i_inv = torch.argsort(i)
        aa_model.rearrange_indep(indep, i)
        aa_model.rearrange_indep(indep, i_inv)


        diff = test_utils.cmp_pretty(indep, input_copy)
        if diff:
            print(diff)
            self.fail(f'{diff=}')


    def test_move_sm(self):
        indep = make_indep('benchmark/input/gaa.pdb', 'LG1')
        L = indep.length()
        input_copy = copy.deepcopy(indep)
        i = np.arange(L)
        i = np.concatenate((i[indep.is_sm], i[~indep.is_sm]))
        aa_model.rearrange_indep(indep, i)

        diff = test_utils.cmp_pretty(indep.atom_frames, input_copy.atom_frames)
        if diff:
            print(diff)
            self.fail(f'{diff=}')
    

    def test_rearrange_sm(self):
        indep = make_indep('benchmark/input/gaa.pdb', 'LG1')
        L = indep.length()
        input_copy = copy.deepcopy(indep)
        i = np.arange(L)
        i = np.concatenate((i[indep.is_sm][-1:], i[indep.is_sm][:-1], i[~indep.is_sm]))
        aa_model.rearrange_indep(indep, i)

        frame_has_0 = (input_copy.atom_frames[:,:,0] == 0).any(dim=1)
        assertpy.assert_that(frame_has_0.all()).is_true()

        frame_has_0 = (indep.atom_frames[:,:,0] == 0).any(dim=1)
        assertpy.assert_that(frame_has_0.all()).is_true()


    def test_split_sm(self):
        indep = make_indep('benchmark/input/gaa.pdb', 'LG1')
        L = indep.length()
        input_copy = copy.deepcopy(indep)

        sm_i = indep.is_sm.nonzero()[:,0]
        sm_i_a = sm_i[:len(sm_i)//2]
        sm_i_b = sm_i[len(sm_i)//2:]
        non_sm_i = (~indep.is_sm).nonzero()[:,0]
        i = np.concatenate((sm_i_a, non_sm_i, sm_i_b))
        aa_model.rearrange_indep(indep, i)

        diff = test_utils.cmp_pretty(indep.atom_frames, input_copy.atom_frames)
        if diff:
            print(diff)
            self.fail(f'{diff=}')    

class TestOpenIndep(unittest.TestCase):
    def test_open_table(self):
        for l, is_open in [
            (6, torch.arange(6) < 2),
            (50, torch.arange(50) > 20),
            (50, torch.arange(50) < 20),
            (6, (torch.arange(6)%2).bool()),
        ]:
            with self.subTest(l=l, is_open=is_open):
                indep = make_indep('benchmark/input/gaa.pdb')
                L = indep.length()
                i = torch.arange(L)
                ic(indep.chirals.shape)
                indep, _ = aa_model.slice_indep(indep, i < l)
                indep.idx = torch.arange(indep.length())
                indep.assert_types()
                input_copy = copy.deepcopy(indep)
                L = indep.length()
                with aa_model.open_indep(indep, is_open) as indep_open:
                    pass
                
                
                diff = test_utils.cmp_pretty(indep, input_copy)
                ic(
                    indep.idx,
                    input_copy.idx
                )
                if diff:
                    print(diff)
                    assert(False)


    def test_open_sm(self):
        indep = make_indep('benchmark/input/gaa.pdb', 'LG1')
        indep.assert_types()
        input_copy = copy.deepcopy(indep)
        with aa_model.open_indep(indep, indep.is_sm) as indep_open:
            indep_open.bond_feats[0,1] = 4
            ic(indep_open.bond_feats[0,1])
            ic(indep_open.bond_feats[1,0])
            indep_open.bond_feats[0,1] = 1
            indep_open.bond_feats[1,0] = 1
        
        bond_feats_diff = (indep.bond_feats != input_copy.bond_feats).nonzero()
        ic(bond_feats_diff)
        assertpy.assert_that(bond_feats_diff.shape).is_equal_to((2,2))

class AAModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model =  Model(conf=None)
        self.indep = make_indep(pdb="benchmark/input/gaa.pdb")
        self.L = self.indep.xyz.shape[0]
        return super().setUp()

    def test_atomize_residues_Nerm(self):
        # Test N term motif 
        input_str_mask = torch.zeros(self.L).bool()
        input_str_mask[:3] = 1

        self.check_protein_atomization_constraints(self.indep, input_str_mask, terminus=True, discontiguous=False)

    def test_atomize_residues_Cterm(self):
        # Test C term motif 
        # indep modified in place need to recreate (should move to setUp)
        
        input_str_mask = torch.zeros(self.L).bool()
        input_str_mask[-3:] = 1

        self.check_protein_atomization_constraints(self.indep, input_str_mask, terminus=True, discontiguous=False)

    def test_atomize_residues_discontig_motif(self):
        # Test motif in the middle of the protein
        input_str_mask = torch.zeros(self.L).bool()
        input_str_mask[[20,55,88]] = 1

        self.check_protein_atomization_constraints(self.indep, input_str_mask, terminus=False, discontiguous=True)

    def check_protein_atomization_constraints(self, indep, input_str_mask, terminus=False, discontiguous=True):
        """
        checks the output values of indep and masks1d given an input indep and str_mask
        """
        indep.xyz = indep.xyz[:, :14]
        L = indep.xyz.shape[0]
        input_indep = copy.deepcopy(indep)
        input_mask = copy.deepcopy(input_str_mask)

        atom_mask = rf2aa.util.allatom_mask[indep.seq]
        atom_mask[:, 14:] = False # no Hs
        indep, atomizer = atomize.atomize(indep, input_str_mask)
        num_atoms = torch.sum(atom_mask[input_mask])

        # original length, remove the residue nodes that are popped and add new atom nodes for those residues
        new_protein_L = L - torch.sum(input_mask)
        correct_new_L = new_protein_L + num_atoms  
        self.assertEqual(indep.seq.shape[0], correct_new_L.item(), msg="atomized indep has the wrong size seq")
        self.assertEqual(indep.xyz.shape[0], correct_new_L.item(), msg="atomized indep has the wrong size xyz")
        self.assertEqual(indep.idx.shape[0], correct_new_L.item(), msg="atomized indep has the wrong size idx")
        self.assertEqual(indep.bond_feats.shape[:2], (correct_new_L.item(), correct_new_L.item()), msg="atomized indep has the wrong size bond_feats")
        self.assertEqual(indep.atom_frames.shape[0], torch.sum(indep.is_sm), msg="atom frames dimension does not match number of atom nodes")
        self.assertEqual(indep.same_chain.shape[:2], (correct_new_L.item(), correct_new_L.item()), msg="atomized indep has the wrong size same_chain")
        self.assertEqual(torch.sum(indep.is_sm), torch.sum(input_indep.is_sm)+num_atoms, msg="atomized indep has the wrong size number of atom nodes in is_sm")

        # assert some edge cases are handled correctly in same_chain and bond_feats
        residue_atom_bonds = (indep.bond_feats == 6)
        residue_atom_bonds_indices = residue_atom_bonds.nonzero()
        atomized_residue_lengths = torch.sum(atom_mask[input_mask], dim=-1)
        if terminus:
            self.assertEqual(residue_atom_bonds_indices.shape[0], 2, msg="terminal contiguous motifs should only have 1 bond to protein ((i, j), (j, i))")
        elif discontiguous:
            num_residue_atom_bonds = 4*atomized_residue_lengths.shape[0] # 2 bonds for each atomized residue and then two orderings (i, j), (j, i)
            self.assertEqual(residue_atom_bonds_indices.shape[0], num_residue_atom_bonds, msg="nonterminal dicontiguous motif has correct bonds to protein")
        rf2aa.tensor_util.assert_equal(residue_atom_bonds, residue_atom_bonds.T)
        
        if not discontiguous:
            for i in range(atomized_residue_lengths.shape[0]-1):
                # should be one bond between contiguous residues
                # indexes out the bonds between each atomized residue and its following neighbor to assert there is a bond placed
                self.assertEqual(torch.sum(indep.bond_feats[new_protein_L+ torch.sum(atomized_residue_lengths[:i]): \
                                new_protein_L+ torch.sum(atomized_residue_lengths[:i+1]), \
                                new_protein_L+ torch.sum(atomized_residue_lengths[:i+1]): \
                                new_protein_L+ torch.sum(atomized_residue_lengths[:i+2])]), 1, msg="Incorrect bond placement between neighboring residues")
        
        self.assertTrue(torch.all(indep.same_chain == 1), msg="all nodes should be on the same chain because there is no small molecule")

    def test_ligand_renaming_pdb(self):
        ligand_name = 'LLK'
        input_pdb = 'benchmark/input/ra_5an7_no_cov.pdb'
        atoms_by_ligand = aa_model.hetatm_names(input_pdb)
        atoms_by_ligand = aa_model.without_H(atoms_by_ligand)
        want_atoms = atoms_by_ligand[ligand_name]
        indep = make_indep(input_pdb, ligand_name)
        out_pdb = 'tmp/out.pdb'
        indep.write_pdb(out_pdb, lig_name=ligand_name)
        aa_model.rename_ligand_atoms(input_pdb, out_pdb)
        atoms_by_ligand = aa_model.hetatm_names(out_pdb)
        got_atoms = atoms_by_ligand[ligand_name]
        assertpy.assert_that(want_atoms).is_equal_to(got_atoms)

    
    def test_ligand_renaming_traj(self):
        ligand_name = 'LLK'
        input_pdb = 'benchmark/input/ra_5an7_no_cov.pdb'
        atoms_by_ligand = aa_model.hetatm_names(input_pdb)
        atoms_by_ligand = aa_model.without_H(atoms_by_ligand)
        want_atoms = atoms_by_ligand[ligand_name]
        n_models = 3
        traj_pdb = 'test_data/traj_2.pdb'
        out_pdb = 'tmp/traj.pdb'
        shutil.copy(traj_pdb, out_pdb)
        aa_model.rename_ligand_atoms(input_pdb, out_pdb)
        atoms_by_ligand = aa_model.hetatm_names(out_pdb)
        got_atoms = atoms_by_ligand[ligand_name]
        assertpy.assert_that(want_atoms * n_models).is_equal_to(got_atoms)
    
    def test_ligand_renaming(self):
        for input_pdb, ligand_name in [
                ('benchmark/input/1yzr_no_covalent.pdb', 'HEM')
            ]:
            atoms_by_ligand = aa_model.hetatm_names(input_pdb)
            atoms_by_ligand = aa_model.without_H(atoms_by_ligand)
            want_atoms = atoms_by_ligand[ligand_name]
            indep = make_indep(input_pdb, ligand_name)
            out_pdb = 'tmp/out.pdb'
            indep.write_pdb(out_pdb, lig_name=ligand_name)
            aa_model.rename_ligand_atoms(input_pdb, out_pdb)
            atoms_by_ligand = aa_model.hetatm_names(out_pdb)
            got_atoms = atoms_by_ligand[ligand_name]
            ic(got_atoms)
            ic(want_atoms)
            assertpy.assert_that(want_atoms).is_equal_to(got_atoms)


    def test_parses_covale(self):
        covale = '/home/ahern/campaigns/bilin/inputs/covale/3l0f.pdb'
        noncovale = '/home/ahern/campaigns/bilin/inputs/raw/3l0f.pdb'
        ligand_name = 'CYC'

        conf = test_utils.construct_conf(inference=True) 
        adaptor = aa_model.Model(conf)
        d = defaultdict(dict)
        for name, input_pdb in [
            ('noncovale', noncovale),
            ('covale', covale),
        ]:
            print(f'----------{name}-------------------')
            run_inference.seed_all()
            indep, metadata = aa_model.make_indep(input_pdb, ligand_name, return_metadata=True)
            target_feats = inference.utils.process_target(input_pdb)
            contig_map =  contigs.ContigMap(target_feats,
                                        contigs=['10-60,A84-87,10-60'],
                                        contig_atoms="{'A84':'CA,C,N,O,CB,SG'}",
                                        length='100-100',
                                        )
            indep_contig,is_diffused,_ = adaptor.insert_contig(indep, contig_map, metadata=metadata)
            d[name]['indep_contig'] = indep_contig
            # d[name]['rfi'] = adaptor.prepro(indep_contig, 100, is_diffused)


        cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)        
        indep_covale, indep_noncovale = d['covale']['indep_contig'], d['noncovale']['indep_contig']
        test_utils.assert_matches_golden(self, 'inference_bilin_noncovale', indep_noncovale, rewrite=REWRITE, custom_comparator=cmp)
        diff = test_utils.cmp_pretty(indep_covale, indep_noncovale)
        if not diff:
            self.fail('expected difference between covale and noncovale input')
        ic(diff)
        test_utils.assert_matches_golden(self, 'inference_bilin_covale', indep_covale, rewrite=REWRITE, custom_comparator=cmp)

        new_bonds = indep_covale.bond_feats != indep_noncovale.bond_feats
        new_bonds = indep_covale.human_readable_2d_symmetric_mask(new_bonds)
        ic(new_bonds)
        new_bonded_elements = tuple(set([element for _, element in b]) for b in new_bonds)
        assertpy.assert_that(new_bonded_elements).is_equal_to((set(['C', 'S']),))

REWRITE = False
if __name__ == '__main__':
        unittest.main()
