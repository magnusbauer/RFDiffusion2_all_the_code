import copy
import os
import sys
import unittest
import json

import addict
import torch
from icecream import ic

from rf_diffusion.aa_model import Model, make_indep
import rf_diffusion.inference.utils
from rf_diffusion import contigs
from rf_diffusion import atomize
from rf2aa import tensor_util

class TestAtomization(unittest.TestCase):

    def test_atomization_table(self):
        testcases = [
            (
                {'contigs': ['A517-518']},
                [0],
                'LG1',
            ),
            (
                {'contigs': ['A518-527']},
                [0,2,4],
                'LG1',
            ),
            (
                {'contigs': ['A126-127']},
                [0],
                None,
            ),
            # Should fail as you cannot atomize C-terminal residues
            # (
            #     {'contigs': ['A517-518']},
            #     [1]
            # ),
        ]

        for contig_kwargs, atomize_indices, ligand in testcases:
            test_pdb = 'benchmark/input/gaa.pdb'
            target_feats = rf_diffusion.inference.utils.process_target(test_pdb)
            contig_map =  contigs.ContigMap(target_feats,
                                        **contig_kwargs
                                        )
            # indep = make_indep(test_pdb, 'LG1')
            indep, metadata = make_indep(test_pdb, ligand, return_metadata=True)
            conf = addict.Dict()
            adaptor = Model(conf)
            indep, _, _ = adaptor.insert_contig(indep, contig_map, metadata=metadata)
            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)

            L = indep.seq.shape[0]
            is_residue_atomized = torch.zeros(L).bool()
            is_residue_atomized[atomize_indices] = True

            # Atomize
            indep_atomized, atomizer = atomize.atomize(indep, is_residue_atomized)

            # Deatomize
            indep_deatomized = atomize.deatomize(atomizer, indep_atomized)

            # Compare original indep with deatomize(atomize(indep))
            diff = cmp_pretty(indep_deatomized, indep_init)
            if diff:
                self.fail(f'{contig_kwargs=} {atomize_indices=} {diff=}')
            
            # For visual inspection
            os.makedirs('tmp', exist_ok=True)
            indep_deatomized.write_pdb(f'tmp/test_deatomized.pdb')

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
