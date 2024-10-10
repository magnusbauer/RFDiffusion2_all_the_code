import torch
import unittest

from rf_diffusion import aa_model
import rf_diffusion.structure as structure
from rf_diffusion import sasa


class TestSSComp(unittest.TestCase):

    def test_pydssp(self):
        input_pdb = './benchmark/input/1AYA.pdb'
        ligand_name = ''

        indep, metadata = aa_model.make_indep(input_pdb, ligand_name, return_metadata=True)
        coord_correct = structure.read_pdbtext_with_checking(open('./benchmark/input/1AYA.pdb').read())

        bb_pydssp, is_prot = structure.get_bb_pydssp(indep)
        # Strip H
        torch.all(torch.eq(bb_pydssp[:,:4], torch.Tensor(coord_correct)))

        prot_ss = structure.get_dssp_string( structure.get_dssp(indep)[0] )
        prot_ss_correct = 'LLLLEELLLLHHHHHHHHHHLLLLLEEEEEELLLLLLLEEEEEEELLEEEEEELEELLLLEELLLLLLELLHHHHHHHHHHLLLLLEELLLLELLLLEELLLLLLLELL'
        len(prot_ss) == len(prot_ss_correct)
        prot_ss == prot_ss_correct


    def test_sasa_per_res(self):
        '''
        What we are actually testing is the parsing code of biopython vs diffusion
        Someday Magnus is going to change the output and this test might fail
        '''
        input_pdb = './test_data/ec1_M0092_same_resn.pdb'
        ligand_name = 'UDX'

        # Load an indep that has 2 ligands
        indep, metadata = aa_model.make_indep(input_pdb, ligand_name, return_metadata=True)

        protein_mask, ligands_mask = [torch.tensor(x) for x in indep.chain_masks()]

        # The ligands get loaded as 1 molecule
        protein, _ = aa_model.slice_indep(indep, protein_mask)
        ligands, _ = aa_model.slice_indep(indep, ligands_mask)

        # Split out the ligands individually
        _, wh_l1, wh_l2 = sasa.small_molecules(ligands)

        l1_mask = torch.zeros(ligands.length(), dtype=bool)
        l1_mask[wh_l1] = True
        l1, _ = aa_model.slice_indep(ligands, l1_mask)

        l2_mask = torch.zeros(ligands.length(), dtype=bool)
        l2_mask[wh_l2] = True
        l2, _ = aa_model.slice_indep(ligands, l2_mask)

        # Initial sasa numeric checks
        protein_sasa = sasa.get_indep_sasa_per_res(protein)
        l1_sasa = sasa.get_indep_sasa_per_res(l1)
        l2_sasa = sasa.get_indep_sasa_per_res(l2)

        assert torch.isclose( torch.sum(protein_sasa), torch.tensor(19723.), atol=200. ) # as measured in pymol
        assert torch.isclose( torch.sum(l1_sasa), torch.tensor(824.), atol=100. )
        assert torch.isclose( torch.sum(l2_sasa), torch.tensor(659.), atol=100. )

        # Split the protein in half
        p1_mask = torch.zeros(protein.length(), dtype=bool)
        p1_mask[:100] = True
        p1, _ = aa_model.slice_indep(protein, p1_mask)
        p2, _ = aa_model.slice_indep(protein, ~p1_mask)

        p1_sasa = sasa.get_indep_sasa_per_res(p1)
        p2_sasa = sasa.get_indep_sasa_per_res(p2)

        # Move the parts so they don't touch
        l1.xyz += torch.tensor([0., -100., 0.])
        l2.xyz += torch.tensor([0., 100., 0.])
        p1.xyz += torch.tensor([100., 0., 0.])
        p2.xyz += torch.tensor([-100., 0., 0.])

        # Check the sasa of a bunch of different permutations to test parsing code
        who_is = [p1, p2, l1, l2]
        who_sasa = [p1_sasa, p2_sasa, l1_sasa, l2_sasa]

        orderings = [
            # (0, 1, 2, 3), # You can uncomment these if you're debugging. This whole thing is kinda slow though
            # (0, 2, 1, 3),
            (2, 0, 1, 3),
            (3, 0, 2, 1),
        ]

        for order in orderings:
            same_chain = torch.zeros((indep.length(), indep.length()), dtype=bool)

            new_indeps = [who_is[i] for i in order]
            new_sasas = [who_sasa[i] for i in order]
            Ls = torch.tensor([x.length() for x in new_indeps])
            cs_Ls = torch.cumsum(Ls, 0)

            # All separate chains
            cur_L = 0
            for L in Ls:
                ub = cur_L + L
                same_chain[cur_L:ub,cur_L:ub] = True
                cur_L = ub

            new_indep = aa_model.cat_indeps(new_indeps, same_chain)

            # Test forward and reverse indexing of indep
            for forwards_order in [True, False]:
                new_indep.idx = torch.arange(new_indep.length())
                if forwards_order:
                    for L_start in cs_Ls:
                        new_indep.idx[L_start:] += 500
                else:
                    new_indep.idx += 2000
                    for L_start in cs_Ls:
                        new_indep.idx[L_start:] -= 500

                # print(order, forwards_order)
                sasa_per_res = sasa.get_indep_sasa_per_res(new_indep)

                # Make sure we got the SASA of every chain correct despite moving stuff around
                for mask, goal_sasa in zip(new_indep.chain_masks(), new_sasas):
                    mask = torch.tensor(mask)
                    saw_sasa = sasa_per_res[mask]
                    assert torch.allclose(saw_sasa, goal_sasa, atol=3.), f'Order: {order} forwards_order: {forwards_order}'


REWRITE = False
if __name__ == '__main__':
        unittest.main()
