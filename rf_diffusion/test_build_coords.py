import torch
import unittest
import io
from functools import partial

from rf2aa import tensor_util
from rf2aa.chemical import ChemicalData as ChemData
from rf_diffusion import aa_model
from rf_diffusion import write_file
from rf_diffusion import test_utils
from rf_diffusion import build_coords


class TestBuildCoords(unittest.TestCase):
    '''
    Tests functions that build coordinates from nothing
    '''

    def test_indep_from_sequence(self):
        '''
        Makes sure that indep_from_sequence works

        If you rewrite this by changing how the default rotamers work that's fine (as long as you made them better)
        '''
        sequence = 'ACDEFGHIKLMNPQRSTVWY'

        indep = aa_model.indep_from_sequence(sequence)

        assert ''.join(ChemData().one_letter[s] for s in indep.seq) == sequence

        debug = False
        if debug:
            indep.write_pdb('test_indep_from_sequence.pdb')

        cmp = partial(tensor_util.cmp, atol=1e-4, rtol=1e-5)
        test_utils.assert_matches_golden(self, 'indep_from_sequence', indep, rewrite=False, custom_comparator=cmp)


    def test_building_with_hydrogens(self):
        '''
        Makes sure that the extended_ideal_xyz_from_seq with include_hydrogens=True actually works

        If you rewrite this, dump the pdb and make sure that the H atoms look good and that glycine's 2 CA Hs look good
        '''

        sequence = 'ACDEFGHIKLMNPQRSTVWY'
        sequence_numeric = torch.tensor([ChemData().one_letter.index(s) for s in sequence])

        xyz, atom_mask = build_coords.extended_ideal_xyz_from_seq(sequence_numeric, include_hydrogens=True)

        debug = False
        if debug:
            with open('test_building_with_hydrogens.pdb', 'w') as fh:
                write_file.writepdb_file(fh, xyz, sequence_numeric)

        cmp = partial(tensor_util.cmp, atol=1e-4, rtol=1e-5)
        test_utils.assert_matches_golden(self, 'building_with_hydrogens', xyz, rewrite=False, custom_comparator=cmp)


    def test_fixing_corrupt_sidechains(self):
        '''
        This test makes sure that if you write an indep that has null coordinates that the situation is handled gracefully
        '''

        sequence = 'ACDEFGHIKLMNPQRSTVWY'
        indep = aa_model.indep_from_sequence(sequence)
        indep.xyz[0,4:] = 0.0 # Set ALA to 0 because this is the only way to catch the fault
        indep.xyz[1:,4:] = 10.0 # Set the rest to non-0 so that they don't trigger the easy-catch

        fh = io.StringIO()
        indep.write_pdb_file(fh, fix_corrupt_sidechains=True)
        fh.seek(0)

        indep2 = aa_model.make_indep('test', pdb_stream=fh.readlines())
        indep2.xyz = indep2.xyz[:,:ChemData().NHEAVY]

        assert (indep.seq == indep2.seq).any()

        atom_mask = ChemData().allatom_mask[indep2.seq][:,:ChemData().NHEAVY]
        atom_mask[:,:4] = False

        all_sidechain_atoms = indep2.xyz[atom_mask].reshape(-1,3)
        assert not torch.isnan(all_sidechain_atoms).any()
        all_by_d2 = torch.sum( torch.square( all_sidechain_atoms[:,None] - all_sidechain_atoms[None,:] ), axis=-1 )
        all_by_d2[torch.eye(len(all_sidechain_atoms), dtype=bool)] = 100
        closest = torch.sqrt( all_by_d2.min() )

        assert closest > 0.5



REWRITE = False
if __name__ == '__main__':
        unittest.main()
