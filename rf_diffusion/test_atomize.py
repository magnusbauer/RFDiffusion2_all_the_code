import copy
import os
import unittest
import json

import torch
from icecream import ic

from rf_diffusion.aa_model import AtomizeResidues, get_atomization_state
import rf_diffusion.inference.utils
from rf_diffusion import atomize
from rf2aa import tensor_util
from rf_diffusion import test_utils
import rf_diffusion.inference.data_loader

class TestAtomization(unittest.TestCase):
    testcases = [
        (
            ["contigmap.contigs=['A517-518']",],
            [0],
            'LG1',
        ),
        (
            ["contigmap.contigs=['A518-527']"],
            [0,2,4],
            'LG1',
        ),
        (
            ["contigmap.contigs=['A126-127']"],
            [0],
            None,
        ),
        (
            ["contigmap.contigs=['A126-127']", "contigmap.inpaint_str=['A126-126']"],
            [0],
            None,
        ),
        # Should fail as you cannot atomize C-terminal residues
        # (
        #     {'contigs': ['A517-518']},
        #     [1]
        # ),
    ]

    def _make_input_indep(self, contig_kwargs, ligand):
        test_pdb = 'benchmark/input/gaa.pdb'

        conf = test_utils.construct_conf(inference=True, overrides=contig_kwargs)
        conf.inference.input_pdb = test_pdb
        dataset = rf_diffusion.inference.data_loader.InferenceDataset(conf)
        _, _, indep_cond, _, is_diffused, atomizer, contig_map, t_step_input, _ = next(iter(dataset))
        indep = indep_cond

        indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)

        return indep

    def test_atomization_table(self):
        for contig_kwargs, atomize_indices, ligand in self.testcases:
            indep = self._make_input_indep(contig_kwargs, ligand)
            indep_init = copy.deepcopy(indep)

            L = indep.seq.shape[0]
            is_residue_atomized = torch.zeros(L).bool()
            is_residue_atomized[atomize_indices] = True

            # Atomize
            deatomized_state = get_atomization_state(indep)
            atomizer = AtomizeResidues(deatomized_state, is_residue_atomized)
            indep_atomized = atomizer.atomize(indep)

            # Deatomize
            indep_deatomized = atomizer.deatomize(indep_atomized)

            # Compare original indep with deatomize(atomize(indep))
            diff = cmp_pretty(indep_deatomized, indep_init)
            if diff:
                self.fail(f'{contig_kwargs=} {atomize_indices=} {diff=}')
            
            # For visual inspection
            os.makedirs('tmp', exist_ok=True)
            indep_deatomized.write_pdb('tmp/test_deatomized.pdb')

    def test_deatomization_via_new_atomizer(self):
        '''
        Tests that the atomizer is state independent. One atomizer should
        be able to deatomize an indep_atomized, even if that atomizer instance 
        didn't make it.
        '''
        for contig_kwargs, atomize_indices, ligand in self.testcases:
            indep = self._make_input_indep(contig_kwargs, ligand)
            indep_init = copy.deepcopy(indep)

            L = indep.seq.shape[0]
            is_residue_atomized = torch.zeros(L).bool()
            is_residue_atomized[atomize_indices] = True

            # Atomize
            deatomized_state = get_atomization_state(indep)
            atomizer1 = AtomizeResidues(deatomized_state, is_residue_atomized)
            indep_atomized = atomizer1.atomize(indep)

            # Deatomize with a new atomizer
            atomizer2 = AtomizeResidues(deatomized_state, is_residue_atomized)
            indep_deatomized = atomizer2.deatomize(indep_atomized)

            # Compare original indep with deatomize(atomize(indep))
            diff = cmp_pretty(indep_deatomized, indep_init)
            if diff:
                self.fail(f'{contig_kwargs=} {atomize_indices=} {diff=}')
            
            # For visual inspection
            os.makedirs('tmp', exist_ok=True)
            indep_deatomized.write_pdb('tmp/test_deatomization_via_atomized_labels.pdb')

def cmp_pretty(got, want, **kwargs):
    diff = tensor_util.cmp(got, want, **kwargs)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        return json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        return json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 

if __name__ == '__main__':
        unittest.main()
