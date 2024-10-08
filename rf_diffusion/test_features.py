import torch
import unittest

from rf_diffusion import aa_model
import rf_diffusion.structure as structure


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

REWRITE = False
if __name__ == '__main__':
        unittest.main()
