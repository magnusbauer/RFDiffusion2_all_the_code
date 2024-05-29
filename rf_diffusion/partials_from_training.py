from rf_diffusion.benchmark import pipeline

import os
import shutil
from datetime import date
import json
import time
import itertools

from icecream import ic
import fire
import glob
from hydra import compose, initialize
import numpy as np

from rf_diffusion import aa_model


def get_datetime():
    return str(date.today()) + '_' + str(time.time())

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def format_for_benchmark(pdb, atom_names_by_res_idx, total_length=10):
    motif_length = max(atom_names_by_res_idx.keys()) - min(atom_names_by_res_idx.keys())+1
    ic(motif_length, total_length)
    motif_resis = sorted(atom_names_by_res_idx.keys())
    motif_resis_strings = [f'A{i}-{i}' for i in motif_resis]
    contig_str = []
    for i, e in enumerate(motif_resis_strings):
        contig_str.append(e)
    contigs = f"[\\'{','.join(contig_str)}\\']"
    contig_atoms = {f"\\'A{i}\\'": f"\\'{','.join(v)}\\'" for i,v in atom_names_by_res_idx.items()}
    contig_atoms = ",".join(f"{k}:{v}" for k,v in contig_atoms.items())
    contig_atoms = '{' + contig_atoms + '}'
    
    contig_atoms = f"\"'{contig_atoms}'\""
    s = f'inference.input_pdb={pdb} contigmap.contigs={contigs} contigmap.contig_atoms={contig_atoms}'
    return s

def is_good_true_pdb(pdb):
    indep = aa_model.make_indep(pdb)
    idx_len = indep.idx.max() - indep.idx.min() + 1
    idx_expected_len = indep.length()
    if idx_len != idx_expected_len:
        return False
    
    info_path = pdb[:-(len('_true_deatomized.pdb'))] + '_info.pkl'
    info = np.load(info_path, allow_pickle=True)
    n_atomized = len(info['masks_1d']['is_atom_motif'])
    if n_atomized != idx_expected_len:
        return False
    return True
    

def main(
        restart_dir='/net/scratch/ahern/se3_diffusion/training/aa/toy_bonds/continuous_time_10res_t5_prod/restart/2023-10-23_1698098708.852064_igkuerjj/',
        n_pdbs=10,
        in_proc=False,
        ):
    '''
    Finds n true pdbs from a training run and kicks off partial diffusion trajectories for them, in which they are
    fully atomized using the config benchmark/partials_from_training.
    '''
    true_pdbs = glob.glob(os.path.join(restart_dir, 'rank_0/training_pdbs/*_true_deatomized.pdb'))
    ic(len(true_pdbs))
    inference_dir = os.path.join(restart_dir, 'inference', get_datetime())
    inference_pdb_dir = os.path.join(inference_dir, 'input')
    os.makedirs(inference_pdb_dir, exist_ok=False)
    input_pdbs = []
    for pdb in true_pdbs:
        if not is_good_true_pdb(pdb):
            print(f'Skipping the following pdb as it is "not good": {pdb}')
            continue

        tail = os.path.basename(pdb).split('.')[0]
        to_path = os.path.join(inference_pdb_dir, tail + '.pdb')
        shutil.copy(pdb, to_path)
        input_pdbs.append(to_path)

        if len(input_pdbs) == n_pdbs:
            break

    bm = {}
    for pdb in input_pdbs:
        tail = os.path.basename(pdb).split('.')[0]
        indep = aa_model.make_indep(pdb)
        contig_atoms = {k:[''] for k in indep.idx.tolist()}
        formatted = format_for_benchmark(
            pdb,
            contig_atoms,
            10)
        bm[tail] = formatted
    
    print(json.dumps(bm, indent=4))

    benchmark_json = os.path.join(inference_pdb_dir, 'benchmark.json')
    with open(benchmark_json, 'w') as fh:
        json.dump(bm, fh, indent=4)
    ic(benchmark_json)
    
    initialize(version_base=None, config_path="benchmark/configs", job_name="inference_partials")
    conf = compose(config_name='partials_from_training', return_hydra_config=True)
    conf.outdir = os.path.join(inference_dir, "out")

    conf.sweep.benchmark_json = benchmark_json
    conf.sweep.benchmarks = 'all'
    if in_proc:
        conf.in_proc = True

    ic(conf.outdir)
    pipeline.main(conf)


if __name__ == '__main__':
    fire.Fire(main)