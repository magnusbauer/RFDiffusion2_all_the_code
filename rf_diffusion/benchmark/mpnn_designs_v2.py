#!/net/software/containers/users/dtischer/shebang_rf_se3_diffusion_dev.sh
#
# Takes a folder of pdb & trb files, generates MPNN features (fixing AAs at
# contig positions), makes list of MPNN jobs on batches of those designs,
# and optionally submits slurm array job and outputs job ID
# 
from collections import defaultdict
import sys, os, argparse, itertools, json, glob
import numpy as np
import tqdm
import copy
from icecream import ic
import pickle
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir,'util'))
import slurm_tools
package_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
mpnn_script = os.path.join(package_dir, 'fused_mpnn/run.py')

def memoize_to_disk(file_name):
    file_name = file_name + '.memo'
    def decorator(func):
        def new_func(*args, **kwargs):
            if os.path.exists(file_name):
                with open(file_name, 'rb') as fh:
                    return pickle.load(fh)
            o = func(*args, **kwargs)
            with open(file_name, 'wb') as fh:
                pickle.dump(o, fh)
            return o
        return new_func
    return decorator

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


    @memoize_to_disk(os.path.join(conf.datadir, 'filenames_by_ligand_presence'))
    def categorize_by_ligand_presence():
        filenames_by_ligand_presence = defaultdict(list)
        print(f'Categorizing {len(filenames)} PDBs by presence/absence of ligand')
        for fn in tqdm.tqdm(filenames):
            trb_path = os.path.splitext(fn)[0] + '.trb'
            trb = np.load(trb_path,allow_pickle=True)
            has_ligand = bool(trb['config']['inference']['ligand'])
            filenames_by_ligand_presence[has_ligand].append(fn)
        return filenames_by_ligand_presence
        
    filenames_by_ligand_presence = categorize_by_ligand_presence()
    for use_ligand, filenames in filenames_by_ligand_presence.items():
        conf_for_mpnn_flavor = copy.deepcopy(conf)
        conf_for_mpnn_flavor.use_ligand = use_ligand
        return run_mpnn(conf_for_mpnn_flavor, filenames)
    
    

def get_binary(in_proc):
    in_apptainer = os.path.exists('/.singularity.d/Singularity')
    ic(in_apptainer)
    # if in_apptainer and in_proc:
    #     return 'python -u'
    return '/usr/bin/apptainer exec --nv --bind /databases:/databases --bind /net/software/:/net/software/ --bind /projects:/projects /software/containers/mlfold.sif python -u'

def run_mpnn(conf, filenames):


    model_type = 'protein_mpnn'
    mpnn_flavor = 'mpnn'
    if conf.use_ligand:
        mpnn_flavor = 'ligmpnn'
        model_type = 'ligand_mpnn'

    mpnn_folder = conf.datadir+f'/{mpnn_flavor}/'
    os.makedirs(mpnn_folder, exist_ok=True)

    
    # skip designs that have already been done
    ic(conf.cautious, conf.chunk)
    if conf.cautious:
        filtered = [fn for fn in filenames 
            if not os.path.exists(mpnn_folder+'/seqs/'+os.path.basename(fn).replace('.pdb','.fa'))]
        
        completed = set(filenames).difference(filtered)
        print(f'{len(completed)}/{len(filtered)} already complete, skipping these')

        if conf.unsafe_skip_parsing and len(completed):
            raise Exception('do not combine unsafe_skip_parsing with cautious')

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    prev_job = None
    if not conf.unsafe_skip_parsing:
        # run parser script
        job_fn = conf.datadir + f'/jobs.{mpnn_flavor}.parse.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        # if conf.use_ligand:
        #     parse_script = f'{script_dir}/util/parse_multiple_chains_ligand.py'
        # else:
        #     parse_script = f'{script_dir}/util/parse_multiple_chains.py'
        parse_script = f'{script_dir}/util/parse_multiple_chains_v2.py'
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

    job_fn = conf.datadir + f'/jobs.{mpnn_flavor}.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout

    for i in range(0, len(filenames), conf.chunk):
        # print(f'{get_binary(conf.slurm.in_proc)} {mpnn_script} '\
        #       f'--model_name "{model_name}" '\
        #       f'--jsonl_path {mpnn_folder}pdbs_{i}.jsonl '\
        #       f'--fixed_positions_jsonl {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
        #       f'--out_folder {mpnn_folder} '\
        #       f'--num_seq_per_target  {conf.num_seq_per_target} '\
        #       f'--sampling_temp="0.1" '\
        #       f'--batch_size {8 if conf.num_seq_per_target > 8 else conf.num_seq_per_target} '\
        #       f'--omit_AAs XC',
        #       file=job_list_file)

        print(f'{get_binary(conf.slurm.in_proc)} {mpnn_script} '\
            f'--pdb_path_multi {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
            f'--fixed_residues_multi {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
            # f'--fixed_residues "{" ".join(fixed_pos)}" '\
            f'--model_type {model_type} '\
            f'--pack_side_chains 1 '\
            f'--out_folder {mpnn_folder} '\
            f'--temperature="0.1" '\
            f'--batch_size {conf.num_seq_per_target} '\
            f'--ligand_mpnn_use_side_chain_context 1 '\
            f'--zero_indexed 1 '\
            f'--packed_suffix "" '\
            f'--omit_AA XC ',
            f'--force_hetatm 1',
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
