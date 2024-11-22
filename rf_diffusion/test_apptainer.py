import os
import subprocess
import unittest

from icecream import ic

import rf_diffusion as rfd

class TestApptainer(unittest.TestCase):

    def test_rf_diffusion_aa_spec(self):
        '''
        This test ensures that rf_diffusion/exec/rf_diffusion_aa.spec stays up-to-date with
         rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif
        Further, if this test passes at the IPD, it means that rf_diffusion/exec/rf_diffusion_aa.spec
         will build a valid apptainer for rf_diffusion (since all the other tests presumably pass)

        If you are not at the IPD, this test will simply return

        If you are at the IPD, the following must be true:
         1. This test must be run from the apptainer pointed to by rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif
         2. That .sif must have been built from rf_diffusion/exec/rf_diffusion_aa.spec
        '''

        at_ipd_file = '/software/containers/versions/rf_diffusion_aa/ipd.txt'
        bakerlab_sif_symlink = 'exec/bakerlab_rf_diffusion_aa.sif'
        spec_file = 'exec/rf_diffusion_aa.spec'
        internal_spec = '/.singularity.d/Singularity'

        if not os.path.exists(at_ipd_file):
            ic('TestApptainer:test_rf_diffusion_aa_spec is not run because you are not at the IPD (sorry!)')
            return

        if 'APPTAINER_CONTAINER' not in os.environ:
            print('You are not running this test from inside an apptainer')
            return

        symlink_target = os.path.realpath(bakerlab_sif_symlink)
        if not os.path.islink(bakerlab_sif_symlink):
            if os.path.exists(bakerlab_sif_symlink):
                assert False, f"{bakerlab_sif_symlink} isn't a symlink!"
            else:
                assert False, f"{bakerlab_sif_symlink} doesn't exist!"


        symlink_target = os.path.realpath(bakerlab_sif_symlink)

        assert os.path.exists(symlink_target), f"{bakerlab_sif_symlink} target {symlink_target} doesn't exist!"

        our_apptainer = os.path.realpath(os.environ['APPTAINER_CONTAINER'])

        assert symlink_target == our_apptainer, (f'You are not running this test from the apptainer pointed to by {bakerlab_sif_symlink}.'
            f'{bakerlab_sif_symlink}: {symlink_target} Current apptainer: {our_apptainer}. '
            f'Your command should look something like this `apptainer exec exec/bakerlab_rf_diffusion_aa.sif pytest --disable-warnings -s -m "not nondeterministic"`')

        assert os.path.exists(internal_spec), 'This test may be broken. Ask bcov or Luki'
        apptainer_spec = open(internal_spec).read()

        assert os.path.exists(spec_file), f"{spec_file} doesn't exist!"
        comitted_spec = open(spec_file).read()

        if apptainer_spec != comitted_spec:
            with open('apptainer.spec', 'w') as f:
                f.write(apptainer_spec)
            assert False, f"{spec_file} doesn't match internal spec of {bakerlab_sif_symlink}. Run this: diff apptainer.spec {spec_file}"

    def test_shebang(self):
        cmd = fr'{rfd.projdir}/run_inference.py inference.num_designs=0 inference.input_pdb={rfd.projdir}/test_data/1qys.pdb'
        out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        assert out.returncode == 0


    def test_apptainer_apptainer(self):
        out = subprocess.run('apptainer --version', shell=True)
        assert out.returncode == 0, 'apptainer must have apptainer installed!'
        
if __name__ == '__main__':
        unittest.main()
