import fire
import glob
from icecream import ic

import os
from dev import show_tip_pa

cmd = show_tip_pa.cmd

from collections import defaultdict
import inference.utils
import numpy as np
import itertools

import os
from dev import show_tip_pa
from dev import show_tip_row
from dev import analyze

def model_generator(traj_path, seq=False):
    with open(traj_path) as f:
        s = f.read()
        models = s.strip().split('ENDMDL')
        parsed = []
        seqs = []
        for i, m in enumerate(models):
            if not m:
                continue
            yield m
    #         # o = parsers.parse_pdb_lines(m.split('\n'), False, False, lambda x: 0)
    #         o = inference.utils.parse_pdb_lines(m.split('\n'), True)
    #         xyz = o[0]
    #         if seq:
    #             seqs.append(o[-1])
    #         parsed.append(xyz)
    #     # parsed = torch.concat(parsed)
    # parsed = torch.tensor(np.array(parsed))
    # if seqs:
    #     return parsed, seqs
    # return parsed

def parse_traj(traj_path, n=None):
    
    # d = defaultdict(list)
    d = []
    for pdb_lines in itertools.islice(model_generator(traj_path), n):
        o = inference.utils.parse_pdb_lines(pdb_lines.split('\n'), True)
        d.append(o)
        # print(list(o.keys()))
        # print(list((k, type(v)) for k,v in o.items()))
        # for k, v in o.items():
        #     if isinstance(v, np.ndarray):
        #         d[k].append(v)
    
    # for k, v in d.items():
    #     d[k] = np.stack(v)
    
    return d
        
# traj_path = os.path.join(row['rundir'], f'traj/{row["name"]}_pX0_traj.pdb')
# parsed = parse_traj(traj_path)
# parsed['xyz'].shape

# def motif_backbone_dists(px0, inferred_atom_names_by_i, gp_atom_names_by_i):
def motif_backbone_dists(px0_xyz, inferred_i, gp_i):
    # backbone_crds = []
    # motif_ca
    prot_bb = px0_xyz[inferred_i, :3]
    gp_bb = px0_xyz[gp_i, :3]
    # print(f'{inferred_i=}, {gp_i=}')
    # print(f'{prot_bb=}')
    # print(f'{gp_bb=}')
    d = np.linalg.norm(prot_bb - gp_bb, axis=-1) # L, 3
    mean_d = np.mean(d, axis=0)
    return {f'dist_backbone_gp_{k}':v for k,v in enumerate(mean_d)}

def motif_backbone_dists_row(row):
    trb = analyze.get_trb(row)
    is_sm = trb['indep']['is_sm']
    sm_i = is_sm.nonzero()[0]
    gp_i = list(trb['motif'].keys())
    gp_i = [i for i in gp_i if i not in sm_i]
    inferred_i = trb['con_hal_idx0']
    px0_xyz = trb['px0_xyz_stack'][0]
    assert len(inferred_i) == len(gp_i), f'{gp_i=}, {inferred_i=}'
    # Need to get ordering to do this properly
    # return motif_backbone_dists(px0_xyz, inferred_i, gp_i)
    dists = []
    for c in itertools.permutations(inferred_i, len(inferred_i)):
        dists.append(motif_backbone_dists(px0_xyz, c, gp_i))
    
    dist = min(dists, key=lambda x: sum(map(np.abs, x.values())))
    return dist
        

from tqdm.notebook import trange, tqdm
def apply_dict(df, f, safe=True):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            metrics_dict = f(row)
        except Exception as e:
            print(safe)
            if safe:
                print(f'Caught exception at row {i}: {row["name"]}: {str(e)}')
                continue
            else:
                raise e
        for k, v in metrics_dict.items():
            df.loc[i, k] = v.item()
    return metrics_dict.keys()

def get_design_df(df):
    groupers = ['name']
    if 'source' in df.columns:
        groupers = ['source'] + groupers
    designs = df.drop_duplicates(groupers, ignore_index=True).reset_index(drop=True)
    return designs

def apply_dict_design(df, f, **kwargs):
    designs = get_design_df(df)
    print(f'{df.shape=}, {designs.shape=}')
    keys = apply_dict(designs, f, **kwargs)
    groupers = ['name']
    if 'source' in df.columns:
        groupers = ['source'] + groupers
    return df.merge(designs[groupers + list(keys)], on=groupers, how='inner')

# smol_df = df.sample(1).reset_index(drop=True)

# data = df[df['self_consistent']]
# data = data[data['model'] == '8_1681815206.1173074/models/BFF_8.pt']
# data = data[data['benchmark'] == 'tip_2_lysozyme_rigid']
# data = data[data['contig_rmsd_af2_atomized'] == data['contig_rmsd_af2_atomized'].min()]
# df = analyze.read_metrics('/home/ahern/benchmarks/aa_diffusion/tip_atoms/220420_tip_cmp/out/compiled_metrics.csv')
# df = apply_dict_design(df, motif_backbone_dists_row, safe=False)


def is_self_consistent(row):
    return (row['rmsd_af2_des'] < 2) and (row['af2_pae_mean'] < 5)
# df['self_consistent'] = df.apply(is_self_consistent, axis=1)

import glob
import pandas as pd

import re

def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)

# '/mnt/home/ahern/projects/dev_rf_diffusion/debug/no_so3_0_*.pdb'a
def get_sdata(path, pattern=None):
    traj_paths = glob.glob(path)
    if pattern:
        traj_paths = glob_re(pattern, traj_paths)
    traj_paths = [p[:-4] for p in traj_paths]
    traj_paths = sorted(traj_paths)
    srows = [analyze.make_row_from_traj(traj_path) for traj_path in traj_paths]
    data = pd.DataFrame.from_dict(srows)
    data['des_color'] = pd.NA
    return data

import re

def transform_file_path(file_path):
    # Extract date and epoch from the file path using regular expressions
    pattern = r'train_session(\d{4}-\d{2}-\d{2})_\d+\.\d+/models/BFF_(\d+)\.pt'
    match = re.search(pattern, file_path)
    
    if not match:
        raise Exception('not match')
    # Extract the date and epoch from the matched groups
    date = match.group(1)
    epoch = match.group(2)

    # Rearrange the extracted date and epoch to the desired format
    formatted_date = '-'.join(date.split('-')[1:])  # Extract month and day from the date
    formatted_epoch = f'epoch_{epoch:>02}'

    # Return the transformed string
    return f'{formatted_date}_{formatted_epoch}'
    
def get_epoch(row):
    ckpt = row['score_model.weights_path']
    ckpt = ckpt.split('_')[-1]
    ckpt = ckpt[:-3]
    return float(ckpt)

def load_df(metrics_path):
    df = analyze.read_metrics(metrics_path)
    df['seed'] = df.name.apply(lambda x: int(x.split('_cond')[1].split('_')[1].split('-')[0]))
    try:
        df['model_number'] = df['model'].apply(lambda x: int(x.split('.')[0]))
    except Exception as e:
        print(e)
    # df['epoch'] = df.name.apply(lambda x: int(x.split('cond')[1].split('_')[1].split('-')[0]))
    df['des_color'] = pd.NA
    try:
        df['model'] = df['inference.ckpt_path'].apply(transform_file_path)
    except Exception as e:
        pass
    try:
        df['epoch'] = df.apply(get_epoch, axis=1)
    except Exception as e:
        print(f'caught exception {e}')
    return df


def show_df(data, structs={'X0'}, af2=False, des=False, **kwargs):
    cmd.set('grid_mode', 1)
    all_pymol = []
    for i, (_, row) in enumerate(data.iterrows(), start=1):
        # print(row[['benchmark', 'name', 'dist_backbone_gp_sum', 'contig_rmsd_af2_des', 'contig_rmsd_af2_atomized']])
        # print(f'{(row['rmsd_af2_des'] < 2)=}, {row['contig_rmsd_af2_des'] < 2=} and {row['af2_ligand_dist'] > 2=}')
        # print(row[['benchmark', 'name', 'rmsd_af2_des', 'contig_rmsd_af2_des_atomized','contig_rmsd_af2_des', 'dist_backbone_gp',  'contig_rmsd_af2_atomized', 'inference.guidepost_xyz_as_design_bb']]) 
        des_color = None
        if not pd.isna(row['des_color']):
            des_color = row['des_color']
        pymol_objs, _ = show_tip_pa.show(row, structs=structs, af2=af2, des=des, des_color=des_color, **kwargs)
        for v in pymol_objs.values():
            cmd.set('grid_slot', i, v)
        all_pymol.append(pymol_objs)
    return all_pymol
        
            
def write_png(path):
    cmd.png(path, 0, 0, 100, 0)
    
def add_pymol_name(data, keys):
    def f(row):
        pymol_prefix = []
        for k in keys:
            k_str = k.replace('.', '_')
            pymol_prefix.append(f"{k_str}-{row[k]}")
        pymol_prefix = '_'.join(pymol_prefix)
        return pymol_prefix
    data['pymol'] = data.apply(f, axis=1)

def get_sweeps(data):
    uniques = {}
    for k in data.keys():
        try:
            uniques[k] = data[k].unique()
        except Exception as e:
            continue
        

    sweeps = {k:v for k,v in uniques.items() if len(v) > 1}

    for k in ['name', 'inference.output_prefix']:
        _ = sweeps.pop(k, None)

    return sweeps

def main(path, name=None, clear=False, structs='X0', pymol_keys=None, pymol_url='http://calathea.dhcp.ipd:9123'):
    ic(pymol_url)
    # cmd = analyze.get_cmd(pymol_url)
    # analyze.cmd = cmd
    # show_tip_pa.cmd = cmd
    # show_tip_row.cmd = cmd
    ic('before show pro3')
    cmd.fragment('pro')
    ic('after show pro')
    ic.configureOutput(includeContext=True)
    data = get_sdata(path)
    print(data.shape)
    if name:
        data['pymol'] = name
    # if pymol_keys:
    #     ic(pymol_keys, structs)
    #     pymol_keys = pymol_keys.split(',')
    #     add_pymol_name(data, pymol_keys)
    sweeps = get_sweeps(data)
    ic(sweeps)
    if len(sweeps):
        add_pymol_name(data, sweeps.keys())
    if clear:
        show_tip_pa.clear()
    all_pymol = show_df(data, structs=structs)
    cmd.do('mass_paper_rainbow')

# TODO: make this monadic
cmd = analyze.get_cmd('http://10.64.100.67:9123')
analyze.cmd = cmd
show_tip_pa.cmd = cmd
show_tip_row.cmd = cmd

if __name__ == '__main__':
    fire.Fire(main)