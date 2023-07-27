#!/net/scratch/ahern/shebangs/shebang_rf_se3_diffusion.sh
#
# Runs the benchmarking pipeline, given arguments for a hyperparameter sweep
#

import sys, os, re, subprocess, time, argparse, glob, json
import hydra
from hydra.core.hydra_config import HydraConfig
from util.hydra_utils import command_line_overrides
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
    inpaint:        Use sweep_hyperparam_inpaint.py to generate the backbones.
    af2_unmpnned:   Run Alphafold on the raw sequences made during backbone generation. <True, False>

    sweep:          Conf for the hyperparameter sweep step.
    mpnn:           Conf for the mpnn_designs step.
    score:          Conf for the score_designs step.
    '''
    global IN_PROC
    IN_PROC = conf.in_proc

    if conf.start_step == 'sweep':
        if conf.inpaint:
            script = f'{script_dir}sweep_hyperparam_inpaint.py'
        else:
            script = f'{script_dir}sweep_hyperparam.py'
        print('run pipeline step')
        jobid_sweep = run_pipeline_step(f'{script} {command_line_overrides(conf.sweep)}')

        print('Waiting for design jobs to finish...')
        wait_for_jobs(jobid_sweep)

    if conf.start_step in ['sweep', 'foldseek']:
        # Move "orphan" pdbs that somehow lack a trb file
        orphan_dir = f'{conf.outdir}/orphan_pdbs'
        os.makedirs(orphan_dir, exist_ok=True)
        pdb_set = {os.path.basename(x.replace('.pdb', '')) for x in glob.glob(f'{conf.outdir}/*pdb')}
        trb_set = {os.path.basename(x.replace('.trb', '')) for x in glob.glob(f'{conf.outdir}/*trb')}
        orphan_pdbs = pdb_set - trb_set
        for basename in orphan_pdbs:
            os.rename(f'{conf.outdir}/{basename}.pdb', f'{orphan_dir}/{basename}.pdb')

        # Cluster designs within each condition
        jobid_cluster = run_pipeline_step(f'{script_dir}/cluster_pipeline_outputs.py --pipeline_outdir {outdir} --in_proc')
        print('Running foldseek in parallel to cluster generated backbones by condition. The pipeline will continue forward.')

        # Compute similarity of generated backbones to the PDB
        jobid_foldseek = run_pipeline_step(f'{script_dir}/chunkify_foldseek_pdb.py --pdb_dir {outdir} --chunk {args.foldseek_chunk} --in_proc')
        print('Running foldseek in parallel to compare the similarity of the generated backbones to the PDB. The pipeline will continue forward.')

    if conf.start_step in ['sweep', 'foldseek', 'mpnn']:
        if conf.use_ligand:
            job_id_prepare_ligandmpnn_params = run_pipeline_step(f'{script_dir}/pdb_to_params.py {conf.outdir}')
            wait_for_jobs(job_id_prepare_ligandmpnn_params)
        jobid_mpnn = run_pipeline_step(f'{script_dir}mpnn_designs.py {command_line_overrides(conf.mpnn)}')

        print('Waiting for MPNN jobs to finish...')
        wait_for_jobs(jobid_mpnn)

    if conf.start_step in ['sweep', 'foldseek', 'mpnn', 'thread_mpnn']:
        print('Threading MPNN sequences onto design models...')
        if conf.use_ligand:
            run_pipeline_step(f'{script_dir}thread_mpnn.py --use_ligand {conf.outdir}')
        else:
            run_pipeline_step(f'{script_dir}thread_mpnn.py {conf.outdir}')

    if conf.start_step in ['sweep', 'foldseek', 'mpnn', 'thread_mpnn', 'score']:
        print('Initiating scoring')
        if conf.af2_unmpnned:
            jobid_score = run_pipeline_step(
                f'{script_dir}score_designs.py {command_line_overrides(conf.score)} conf.score.datadir={conf.outdir}'
            )
        
        mpnn_dirs = []
        for mpnn_flavor in ['mpnn', 'ligmpnn']:
            mpnn_dirs.append(f'{conf.outdir}/{mpnn_flavor}')
        
        assert any(os.path.exists(d) for d in mpnn_dirs)
        jobid_score_mpnn = []
        for d in mpnn_dirs:
            if os.path.exists(d):
                jobid_score_mpnn.extend(run_pipeline_step(
                    f'{script_dir}score_designs.py {command_line_overrides(conf.score)}'
                ))

    print('Waiting for scoring jobs to finish...')
    if conf.af2_unmpnned:
        wait_for_jobs(jobid_score)
    wait_for_jobs(jobid_score_mpnn)

    print('Compiling metrics...')
    run_pipeline_step(f'{script_dir}compile_metrics.py {conf.outdir}')

    print('Done.')
    
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
    while True:
        if any(is_running(job_ids)):
            time.sleep(interval)
        else:
            break
    return 

if __name__ == "__main__":
    main()
