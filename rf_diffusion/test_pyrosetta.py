import torch
import unittest
import os

from functools import partial
from rf2aa import tensor_util

from rf_diffusion import aa_model
from rf_diffusion import silent_files
from rf_diffusion import import_pyrosetta
from rf_diffusion import test_utils


class TestPyrosetta(unittest.TestCase):
    '''
    Tests that involve pyrosetta
    '''

    def setUp(self):
        conf = test_utils.construct_conf()
        import_pyrosetta.prepare_pyrosetta(conf)

    def test_silent_accuracy(self):
        '''
        Ensure that the silent file machinery generates a pdb that is equivalent
        '''
        input_pdb = './test_data/two_chain.pdb'
        ligand_name = ''
        silent_name = 'tmp/out.silent'
        pdb_name = 'tmp/out.pdb'
        tag = 'my_tag'

        indep = aa_model.make_indep(input_pdb, ligand_name)

        if os.path.exists(silent_name):
            os.remove(silent_name)

        silent_files.add_indep_to_silent(silent_name, tag, indep)
        pose, _ = silent_files.read_pose_from_silent(silent_name, tag)
        pose.dump_pdb(pdb_name)

        indep2 = aa_model.make_indep(pdb_name)
        
        cmp = partial(tensor_util.cmp, atol=0.0011, rtol=0)
        diff = cmp(indep, indep2)

        assert not diff



if __name__ == '__main__':
        unittest.main()
