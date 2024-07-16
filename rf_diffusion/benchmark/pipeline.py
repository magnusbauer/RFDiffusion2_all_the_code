#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Runs the benchmarking pipeline, given arguments for a hyperparameter sweep
#

import sys
import os
import re
import subprocess
import time
import glob
import copy

from icecream import ic
import hydra
from hydra.core.hydra_config import HydraConfig
from rf_diffusion.benchmark.sweep_hyperparam import main as main_sweep
from rf_diffusion.benchmark.cluster_pipeline_outputs import main as main_cluster
from rf_diffusion.benchmark.chunkify_foldseek_pdb import main as main_foldseek
from rf_diffusion.benchmark import mpnn_designs
from rf_diffusion.benchmark import mpnn_designs_v2
from rf_diffusion.benchmark.score_designs import main as main_score
from rf_diffusion.benchmark import add_metrics
script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
IN_PROC = False
    
@hydra.main(version_base=None, config_path='configs/', config_name='pipeline')
def main(conf: HydraConfig) -> None:
    '''
    ### Expected conf keys ###
    outdir:         Dir to output generated backbones.
    start_step:     Pipeline step to start at. <'sweep', 'foldseek', 'mpnn','thread_mpnn', 'score', 'compile'>
    use_ligand:     Use LigandMPNN instead of regular MPNN. <True, False>
    slurm_submit:   False = Do not submit slurm array job, only generate job list. <True, False>
    in_proc:        True = Do not submit slurm array job, run on current node. <True, False>
    af2_unmpnned:   Run Alphafold on the raw sequences made during backbone generation. <True, False>

    sweep:          Conf for the sweep_hyperparam step.
    cluster:        Conf for clustering generated backbones.
    foldseek:       Conf for using foldseek to compare generated backbones to the PDB.
    mpnn:           Conf for the mpnn_designs step.
    score:          Conf for the score_designs step.
    '''
    os.makedirs(conf.outdir, exist_ok=True)
    cwd = os.path.dirname(conf.outdir.rstrip('/'))
    os.chdir(cwd)

    global IN_PROC
    IN_PROC = conf.in_proc
    ic(conf.outdir)

    if step_in_scope(conf.start_step, conf.stop_step, 'sweep'):
        jobid_sweep = main_sweep(conf.sweep)
        print('Waiting for design jobs to finish...', jobid_sweep)
        wait_for_jobs(jobid_sweep)

    if step_in_scope(conf.start_step, conf.stop_step, 'foldseek') and 'foldseek' not in conf.skip_steps:
        # Move "orphan" pdbs that somehow lack a trb file
        orphan_dir = f'{conf.outdir}/orphan_pdbs'
        os.makedirs(orphan_dir, exist_ok=True)
        pdb_set = {os.path.basename(x.replace('.pdb', '')) for x in glob.glob(f'{conf.outdir}/*.pdb')}
        trb_set = {os.path.basename(x.replace('.trb', '')) for x in glob.glob(f'{conf.outdir}/*.trb')}
        orphan_pdbs = pdb_set - trb_set
        for basename in orphan_pdbs:
            os.rename(f'{conf.outdir}/{basename}.pdb', f'{orphan_dir}/{basename}.pdb')

        # Cluster designs within each condition
        jobid_cluster = main_cluster(conf.cluster)
        print(f'Running foldseek in parallel ({jobid_cluster})to cluster generated backbones by condition. The pipeline will continue forward.')

        # Compute similarity of generated backbones to the PDB
        jobid_foldseek = main_foldseek(conf.foldseek)
        print(f'Running foldseek in parallel ({jobid_foldseek})to compare the similarity of the generated backbones to the PDB. The pipeline will continue forward.')

    if step_in_scope(conf.start_step, conf.stop_step, 'graft'):
        if conf.graft:
            ic(script_dir)
            run_pipeline_step(f'{os.path.join(script_dir, "../dev/graft_native_motif.py")} {conf.outdir} {conf.outdir}')
            run_pipeline_step(f'{os.path.join(script_dir, "../dev/renumber_chains.py")} {conf.outdir} {conf.outdir} --cautious=False')

    if step_in_scope(conf.start_step, conf.stop_step, 'mpnn') and 'mpnn' not in conf.skip_steps:

        if conf.mpnn.v2:
            main_mpnn = mpnn_designs_v2.main
        else:
            main_mpnn = mpnn_designs.main
            if conf.use_ligand:
                job_id_prepare_ligandmpnn_params = run_pipeline_step(f'{script_dir}/pdb_to_params.py {conf.outdir}')
                wait_for_jobs(job_id_prepare_ligandmpnn_params)
        jobid_mpnn = main_mpnn(conf.mpnn)

        print('Waiting for MPNN jobs to finish...', jobid_mpnn)
        wait_for_jobs(jobid_mpnn)

    if step_in_scope(conf.start_step, conf.stop_step, 'thread_mpnn') and 'thread_mpnn' not in conf.skip_steps:
        ic(conf.mpnn.v2)
        if conf.mpnn.v2:
            # raise Exception('do we thread here?')
            print('Skipping threading since mpnn v2 is used')
        else:
            print('Threading MPNN sequences onto design models...')
            if conf.use_ligand:
                run_pipeline_step(f'{script_dir}thread_mpnn.py --use_ligand {conf.outdir}')
            else:
                run_pipeline_step(f'{script_dir}thread_mpnn.py {conf.outdir}')

    if step_in_scope(conf.start_step, conf.stop_step, 'score') and 'score' not in conf.skip_steps:
        print('Initiating scoring')
        if conf.af2_unmpnned:
            conf_score = copy.deepcopy(conf.score)
            conf_score.datadir = conf.outdir
            jobid_score = main_score(conf_score)
        
        mpnn_dirs = []
        for mpnn_flavor in ['mpnn', 'ligmpnn']:
            mpnn_dirs.append(f'{conf.outdir}/{mpnn_flavor}')
        
        assert any(os.path.exists(d) for d in mpnn_dirs)
        jobid_score_mpnn = []
        for d in mpnn_dirs:
            if os.path.exists(d):
                ic(d, conf.score.datadir)
                conf_score = copy.deepcopy(conf.score)
                if conf.mpnn.v2:
                    d = os.path.join(d, 'packed')
                    # add_h_dir = os.path.join(d, 'addh')
                    # if os.path.exists(add_h_dir):
                    #     d = add_h_dir
                conf_score.trb_dir = conf.outdir
                conf_score.datadir = d
                # conf_score.input_dir = os.path.join(conf.outdir, 'input')
                jobid_score_mpnn += main_score(conf_score)

        print('Waiting for scoring jobs to finish...', jobid_score_mpnn)
        if conf.af2_unmpnned:
            wait_for_jobs(jobid_score)
        wait_for_jobs(jobid_score_mpnn)

    if step_in_scope(conf.start_step, conf.stop_step, 'metrics'):
        conf.metrics.datadir = conf.outdir
        jobid_add_metrics = add_metrics.main(conf.metrics)
        wait_for_jobs(jobid_add_metrics)

    if step_in_scope(conf.start_step, conf.stop_step, 'compile'):
        print('Compiling metrics...')
        run_pipeline_step(f'{script_dir}compile_metrics.py {conf.outdir} --cached_trb_df')

    print('Done.')

def get_mpnn_dirs(outdir, v2):
    mpnn_dirs = []
    for mpnn_flavor in ['mpnn', 'ligmpnn']:
        mpnn_dirs.append(f'{outdir}/{mpnn_flavor}')
    
    out_mpnn_dirs = []
    for d in mpnn_dirs:
        if os.path.exists(d):
            if v2:
                d = os.path.join(d, 'packed')
            out_mpnn_dirs.append(d)
    return out_mpnn_dirs

def run_pipeline_step(cmd):
    '''Runs a script in shell, prints its output, quits if there's an error,
    and returns list of slurm ids that appear in its output'''

    if IN_PROC:
        print(f'RUNNING: {cmd}')
        proc = subprocess.run(cmd, shell=True)
        out = ''
        if proc.returncode != 0:
            raise Exception(f'FAILED: {cmd}')
    else:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.stdout.decode()
        print(out)
        if proc.returncode != 0: 
            sys.exit(proc.stderr.decode())

    jobids = re.findall(r'array job (\d+)', out)

    return jobids

def is_running(job_ids):
    '''Returns list of bools corresponding to whether each slurm ID in input
    list corresponds to a currently queued/running job.'''

    idstr = ','.join(map(str,job_ids))

    proc = subprocess.run(f'squeue -j {idstr}', shell=True, stdout=subprocess.PIPE)
    stdout = proc.stdout.decode()

    out = [False]*len(job_ids)
    for line in stdout.split('\n'):
        for i,id_ in enumerate(job_ids):
            if id_ == -1 or line.startswith(str(id_)):
                out[i] = True

    return out

def wait_for_jobs(job_ids, interval=60):
    '''Returns when all the SLURM jobs given in `job_ids` aren't running
    anymore.'''
    if job_ids:
        while True:
            if any(is_running(job_ids)):
                time.sleep(interval)
            else:
                break
        return 

def step_in_scope(start_step, stop_step, current_step):
    all_steps = ['sweep', 'foldseek', 'graft', 'mpnn', 'thread_mpnn', 'score', 'metrics', 'compile', 'end']
    steps_to_run = all_steps[all_steps.index(start_step):all_steps.index(stop_step)+1]
    do_run = current_step in steps_to_run
    print(f'{"Running" if do_run else "Skipping"} step: {current_step}')
    return do_run

if __name__ == "__main__":
    main()
