#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Takes a folder of pdb & trb files, generates MPNN features (fixing AAs at
# contig positions), makes list of MPNN jobs on batches of those designs,
# and optionally submits slurm array job and outputs job ID
# 
from collections import defaultdict
import sys
import os
import glob
import numpy as np
import copy
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
from rf_diffusion.benchmark.util import slurm_tools

@hydra.main(version_base=None, config_path='configs/', config_name='mpnn_designs')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    datadir:                Folder of designs to score.
    chunk:                  How many designs to process in each job.
    num_seq_per_target:     How many mpnn sequences per design.
    use_ligand:             Use ligandMPNN.
    cautious:               Skip design if output file exists.

    slurm:
        J:          Job name
        p:          Partition
        gres:       Gres specification
        submit:     False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:    Run slurm array job on the current node? <True, False>
        keep_logs:  Keep the slurm logs? <True, False>
    '''
    filenames = glob.glob(conf.datadir+'/*.pdb')
    
    if not conf.use_ligand:
        return run_mpnn(conf, filenames)

    filenames_by_ligand_presence = defaultdict(list)
    for fn in filenames:
        trb_path = os.path.splitext(fn)[0] + '.trb'
        trb = np.load(trb_path,allow_pickle=True)
        has_ligand = bool(trb['config']['inference']['ligand'])
        filenames_by_ligand_presence[has_ligand].append(fn)
    
    for use_ligand, filenames in filenames_by_ligand_presence.items():
        conf_for_mpnn_flavor = copy.deepcopy(conf)
        conf_for_mpnn_flavor.use_ligand = use_ligand
        return run_mpnn(conf_for_mpnn_flavor, filenames)

def get_binary(in_proc):
    in_apptainer = os.path.exists('/.singularity.d/Singularity')
    if in_apptainer and in_proc:
        return 'python -u'
    return '/net/software/containers/users/dtischer/rf_se3_diffusion.sif -u'

def run_mpnn(conf, filenames):

    mpnn_flavor = 'mpnn'
    if conf.use_ligand:
        mpnn_flavor = 'ligmpnn'
    mpnn_folder = conf.datadir+f'/{mpnn_flavor}/'
    os.makedirs(mpnn_folder, exist_ok=True)

    
    # skip designs that have already been done
    if conf.cautious:
        filenames = [fn for fn in filenames 
            if not os.path.exists(mpnn_folder+'/seqs/'+os.path.basename(fn).replace('.pdb','.fa'))]

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    # run parser script
    job_fn = conf.datadir + f'/jobs.{mpnn_flavor}.parse.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
    if conf.use_ligand:
        parse_script = f'{script_dir}/util/parse_multiple_chains_ligand.py'
    else:
        parse_script = f'{script_dir}/util/parse_multiple_chains.py'

    for i in range(0, len(filenames), conf.chunk):
        with open(mpnn_folder+f'parse_multiple_chains.list.{i}','w') as outf:
            for fn in filenames[i:i+conf.chunk]:
                print(fn,file=outf)
        print(f'{parse_script} --input_files {mpnn_folder}/parse_multiple_chains.list.{i} '\
              f'--datadir {conf.datadir} '\
              f'--output_parsed {mpnn_folder}/pdbs_{i}.jsonl '\
              f'--output_fixed_pos {mpnn_folder}/pdbs_position_fixed_{i}.jsonl', file=job_list_file)
    if conf.slurm.submit: job_list_file.close()

    # submit to slurm
    job_ids = []
    if conf.slurm.submit:
        pre = 'ligmpnn_pre' if conf.use_ligand else 'mpnn_pre'
        job_id, proc = slurm_tools.array_submit(job_fn, p='cpu', gres=None, J=pre, log=conf.slurm.keep_logs, in_proc=conf.slurm.in_proc)
        if job_id > 0:
            job_ids.append(job_id)
        print(f'Submitted array job {job_id} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to preprocess {len(filenames)} designs for MPNN')

        prev_job = job_id
    else:
        prev_job = None

    job_fn = conf.datadir + f'/jobs.{mpnn_flavor}.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
    if conf.use_ligand:
        mpnn_script = '/net/databases/mpnn/github_repo/ligandMPNN/protein_mpnn_run.py'
        model_name = 'v_32_020'
    else:
        mpnn_script = '/net/databases/mpnn/github_repo/protein_mpnn_run.py'
        model_name = 'v_48_020'

    for i in range(0, len(filenames), conf.chunk):
        print(f'{get_binary(conf.slurm.in_proc)} {mpnn_script} '\
              f'--model_name "{model_name}" '\
              f'--jsonl_path {mpnn_folder}pdbs_{i}.jsonl '\
              f'--fixed_positions_jsonl {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
              f'--out_folder {mpnn_folder} '\
              f'--num_seq_per_target  {conf.num_seq_per_target} '\
              f'--sampling_temp="0.1" '\
              f'--batch_size {8 if conf.num_seq_per_target > 8 else conf.num_seq_per_target} '\
              f'--omit_AAs XC',
              file=job_list_file)
    if conf.slurm.submit: job_list_file.close()

    # submit job
    if conf.slurm.submit:
        if conf.slurm.J is not None:
            job_name = conf.slurm.J
        else:
            job_name = 'mpnn_' + os.path.basename(conf.datadir.strip('/'))
        pre = 'ligand_' if conf.use_ligand else 'protein_'
        job_name = pre + job_name
        job_id, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, wait_for=[prev_job], in_proc=conf.slurm.in_proc)
        if job_id > 0:
            job_ids.append(job_id)
        print(f'Submitted array job {job_id} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to MPNN {len(filenames)} designs')

    return job_ids

if __name__ == "__main__":
    main()
