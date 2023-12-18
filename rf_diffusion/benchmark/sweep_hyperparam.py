#!/net/software/containers/users/dtischer/shebang_rf_se3_diffusion_dev.sh
#
# Generates and slurm array jobs for hyperparameter sweeps on design
# scripts, optionally submits array job and outputs slurm job ID
#

import sys, os, argparse, itertools, json, shutil, re
import numpy as np
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig

import rf_diffusion
PKG_DIR = rf_diffusion.__path__[0]

from icecream import ic 
from rf_diffusion.benchmark.util import slurm_tools
import assertpy

def split_string_with_parentheses(string, delimiter=None):
    '''
    Splits a string using on delimiter (or whitespace if delimiter is None),
    ignoring delimiters in between pairs of parentheses.
    '''
    if delimiter is None:
        is_delimiter = lambda x: x.isspace()
    else:
        is_delimiter = lambda x: x==delimiter
    result = []
    current_word = ''
    paren_count = 0
    
    for char in string:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        
        if is_delimiter(char) and paren_count == 0:
            if current_word:
                result.append(current_word)
            current_word = ''
        else:
            current_word += char
    
    if current_word:
        result.append(current_word)
    
    return result

def remove_whitespace(arg_str):
    arg_str = re.sub(r'\s*\|\s*', '|', arg_str)
    arg_str = re.sub(r'\s*=\s*', '=', arg_str)
    return arg_str

def get_arg_combos(arg_str):
    '''
    Params:
        arg_str: key=value string like `
            c=5
            (a=1 b=2)|(a=3 b=4)
        `
    
    Returns:
        List of dictionaries like:
        `[
            {'c':'5','a':'1', 'b':'2'},
            {'c':'5','a':'3', 'b':'4'},
        ]`
    '''
    all_arg_dicts = []
    arg_str = remove_whitespace(arg_str)
    for arg in split_string_with_parentheses(arg_str):
        if arg.startswith('('):
            arg_dicts = []
            for c in split_string_with_parentheses(arg, '|'):
                assertpy.assert_that(c).starts_with('(')
                assertpy.assert_that(c).ends_with(')')
                arg_dicts.extend(get_arg_combos(c[1:-1]))
        else:
            # base case
            k, vs = arg.split('=')
            arg_dicts = []
            for v in vs.split('|'):
                arg_dicts.append({k:v})
        all_arg_dicts.append(arg_dicts)
            
    arg_dicts = [dict()]
    for sub_arg_dicts in all_arg_dicts:
        next_arg_dicts = []
        for d1 in arg_dicts:
            for d2 in sub_arg_dicts:
                next_arg_dicts.append(dict(d1, **d2))
        arg_dicts = next_arg_dicts

    return arg_dicts

def process_post(input_str):
    parts = input_str.split("POST(")
    if len(parts) == 1:
        return input_str
    if len(parts) > 2:
        raise Exception('invalid input format')
    
    before_post = parts[0]
    rest = parts[1]
    
    # Extracting substrings enclosed in parentheses
    in_post = ""
    stack = ["("]

    after_post = ""
    done = False
    for char in rest:
        if done:
            after_post += char
        else:
            if char == '(':
                stack.append('(')
            elif char == ')':
                if stack:
                    stack.pop()
                    if not stack:
                        done = True
                else:
                    raise Exception("invalid POST parentheses")
        
            if stack:
                in_post += char
    
    return ' '.join([before_post, after_post, in_post])

@hydra.main(version_base=None, config_path='configs/', config_name='sweep_hyperparam')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    out:                Path prefix for output files.
    command:            Design script to run.
    command_args:       string with all arguments to pass to the command, 
                        with pipe (|)-delimited value options for each.
    benchmark_json:     Path to non-standard custom json file of benchmarks.
    benchmarks:         List of benchmark names, as defined in "benchmarks.json". Also accepts the input "all".
    num_per_condition:  Number of designs to make for each condition.
    num_per_job:        Split runs for each condition into this many designs per job.
    pilot: 
    pilot_single:

    slurm:
        J:          Job name
        t:          Time limit
        p:          Partition
        gres:       Gres specification
        submit:     False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:    Run slurm array job on the current node? <True, False>
        keep_logs:  Keep the slurm logs? <True, False>
    '''
    if conf.num_per_job > conf.num_per_condition:
        sys.exit('ERROR: --num_per_job cannot be greater than --num_per_condition '\
                 '(different conditions can\'t be in the same job.)')
    
    if conf.pilot:
        conf.num_per_condition = 1
        conf.num_per_job = 1

    args_vals = [] # argument names and lists of values for passing to design script

    # default design script
    if conf.command is None:
        conf.command = f'{PKG_DIR}/run_inference.py'

    # parse pre-defined benchmarks
    print('This is benchmarks json')
    print(conf.benchmark_json)
    if not conf.benchmark_json.startswith('/'):
        conf.benchmark_json =f'{PKG_DIR}/benchmark/{conf.benchmark_json}'
    with open(conf.benchmark_json) as f: 
        benchmarks = json.load(f)
    input_path = f'{PKG_DIR}/benchmark/input/' # prepend path to input pdbs in current repo
    benchmark_list = []
    if conf.benchmarks is not None:
        if conf.benchmarks == 'all':
            to_run = benchmarks
        else:
            to_run = conf.benchmarks.split(',')
        for bm in to_run:
            benchmark_list.append([
                f'inference.output_prefix={bm}',
                re.sub(r'inference.input_pdb=(?!/)', f'inference.input_pdb={input_path}', benchmarks[bm])
            ])

    # parse names of arguments and their value options to be passed into the design script
    arg_str = ''.join(conf.command_args)
    if '--config-name' in arg_str.split():
        raise Exception('config names must be passed like: --config-name=name_here')

    if len(benchmark_list) > 0:
        benchmark_arg_groups = []
        for benchmark in benchmark_list: # [output path, input pdb, contig spec]
            benchmark_arg_groups.append(f"({' '.join(benchmark)})")
        arg_str += ' ' + '|'.join(benchmark_arg_groups)

    # Process POST groups to be applied AFTER benchmarks, so that they may override benchmark defaults.
    arg_str = process_post(arg_str)

    arg_dicts = get_arg_combos(arg_str)

    df = pd.DataFrame.from_dict(arg_dicts, dtype=str)

    # make output folder
    os.makedirs(os.path.dirname(conf.out), exist_ok=True) 
    os.makedirs(os.path.dirname(conf.out)+'/input', exist_ok=True)

    def get_input_copy_path(input_pdb):
        return os.path.join(os.path.dirname(conf.out), 'input', os.path.basename(input_pdb))
    if 'inference.input_pdb' in df:
        for input_pdb in df['inference.input_pdb'].unique():
            if not input_pdb.startswith('/'):
                input_pdb = os.path.join(input_path, input_pdb)
            shutil.copyfile(input_pdb, get_input_copy_path(input_pdb))
        df['inference.input_pdb'] = df['inference.input_pdb'].apply(get_input_copy_path)

    out_dir, basename = os.path.split(conf.out)
    def get_output_path(row):
        output_path_components = []
        if basename != '':
            output_path_components.append(basename)
        existing_prefix = row.get('inference.output_prefix', '')
        if existing_prefix and not pd.isna(existing_prefix):
            output_path_components.append(os.path.basename(existing_prefix))

        output_path_components.append(f'cond{row.name}')
        return os.path.join(out_dir, '_'.join(output_path_components))
    df['inference.output_prefix'] = df.apply(get_output_path, axis=1)

    # output commands with all combos of argument values
    job_fn = os.path.dirname(conf.out) + '/jobs.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
    for _, arg_row in df.iterrows():
        arg_dict = arg_row.dropna().to_dict()
        combo = []
        for k,v in arg_dict.items():
            combo.append(f'{k}={v}')
        extra_args = ' '.join(combo)

        for istart in np.arange(0, conf.num_per_condition, conf.num_per_job):
            log_fn = f'{arg_row["inference.output_prefix"]}_{istart}.log'
            print(f'{conf.command} {extra_args} '\
                  f'inference.num_designs={conf.num_per_job} inference.design_startnum={istart} >> {log_fn}', file=job_list_file)

    if conf.slurm.submit or conf.slurm.in_proc:
        job_list_file.close()
    # submit job
    job_ids = []
    if conf.slurm.submit:
        job_fn = prune_jobs_list(job_fn)
        if conf.pilot:
            job_fn = pilot_jobs_list(job_fn, conf.pilot_single)

        if conf.slurm.J is not None:
            job_name = conf.slurm.J
        else:
            job_name = 'sweep_hyp_'+os.path.basename(os.path.dirname(conf.out))
        if conf.slurm.p == 'cpu':
            conf.slurm.gres = ""
        slurm_job, proc = slurm_tools.array_submit(job_fn, p=conf.slurm.p, gres=conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, t=conf.slurm.t, in_proc=conf.slurm.in_proc)
        if slurm_job > 0:
            job_ids.append(slurm_job)
        print(f'Submitted array job {slurm_job} with {len(df)*conf.num_per_condition/conf.num_per_job} jobs to make {len(df)*conf.num_per_condition} designs for {len(df)} conditions')

    return job_ids

def pilot_jobs_list(jobs_path, single=False):
    pilot_path = os.path.join(os.path.split(jobs_path)[0], 'jobs.list.pilot')
    with open(jobs_path, 'r') as fh:
        jobs = fh.readlines()
    job_by_input_pdb = {}
    for job in jobs:
        input_pdb = re.match('.*inference\.input_pdb=(\S+).*', job).groups()[0]
        if input_pdb not in job_by_input_pdb:
            job_by_input_pdb[input_pdb] = job
    jobs = list(job_by_input_pdb.values())
    with open(pilot_path, 'w') as fh:
        if single:
            jobs = jobs[0:1]
        fh.writelines(jobs)
    ic(f'running {len(jobs)} pilot jobs for PDBS: {list(job_by_input_pdb.keys())}')
    return pilot_path
        
def prune_jobs_list(jobs_path):
    pruned_path = os.path.join(os.path.split(jobs_path)[0], 'jobs.list.pruned')
    pruned = []
    with open(jobs_path, 'r') as fh:
        jobs = fh.readlines()
    for i, job in enumerate(jobs):
        want_outs = expected_outputs(job)
        def has_output(want_out):
            want_out = want_out[:-4]
            for suffix in ['', '-atomized-bb-False', '-atomized-bb-True']:
                possible_path = want_out + suffix + '.pdb'
                if os.path.exists(possible_path):
                    return True
            return False
        has_outs = [has_output(want) for want in want_outs]
        if not all(has_outs):
            pruned.append(job)
    if len(pruned) != len(jobs):
        print(f'{len(jobs)} jobs described, pruned to {len(pruned)} because all expected outputs exist for {len(jobs)-len(pruned)} jobs')
    with open(pruned_path, 'w') as fh:
        fh.writelines(pruned)
    return pruned_path

import re
def expected_outputs(job):
    output_prefix = re.match('.*inference\.output_prefix=(\S+).*', job).groups()[0]
    design_startnum = re.match('.*inference\.design_startnum=(\S+).*', job).groups()[0]
    num_designs = re.match('.*inference\.num_designs=(\S+).*', job).groups()[0]

    design_startnum = int(design_startnum)
    num_designs = int(num_designs)

    des_i_start = design_startnum
    des_i_end = design_startnum + num_designs
    return [f'{output_prefix}_{i}.pdb' for i in range(des_i_start, des_i_end)]

if __name__ == "__main__":
    main()
