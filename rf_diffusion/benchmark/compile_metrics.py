#!/usr/bin/env python
#
# Compiles metrics from scoring runs into a single dataframe CSV
#

import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
from icecream import ic
from tqdm import tqdm

def flatten_dictionary(dictionary, parent_key='', separator='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dictionary(value, new_key, separator))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

def sorted_value_counts(df, cols):
    return pd.DataFrame(df.value_counts(cols, dropna=False)).sort_values(cols).rename(columns={0: 'count'})

def count_of_counts(df, cols):
    return sorted_value_counts(
        sorted_value_counts(df, cols).reset_index().rename(columns={0: 'count'}),
        ['count']
    )

def main():
    ic.configureOutput(includeContext=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs')
    parser.add_argument('--outcsv',type=str,default='compiled_metrics.csv',help='Output filename')
    parser.add_argument('--cached_trb_df', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print('finding trbs')
    filenames = glob.glob(args.datadir+'/*.trb')


    print('loading run metadata (base metrics)')
    df_trb_path = os.path.join(args.datadir, 'trb_compiled_metrics.csv')
    cache_hit = False
    if os.path.exists(df_trb_path) and args.cached_trb_df:
        print('loading run metadata (base metrics) from cached csv')
        df_base = pd.read_csv(df_trb_path)
        if len(df_base) == len(filenames):
            cache_hit = True
    
    if not cache_hit:
        print('loading run metadata (base metrics) from individual trbs, if re-compiling consider passing --cached_trb_df=1 to use the cacheed trb compilation df')
        records = []
        for fn in tqdm(filenames):
            name = os.path.basename(fn).replace('.trb','')
            trb = np.load(fn, allow_pickle=True)

            record = {'name':name}
            if 'lddt' in trb:
                record['lddt'] = trb['lddt'].mean()
            if 'inpaint_lddt' in trb:
                record['inpaint_lddt'] = np.mean(trb['inpaint_lddt'])

            if 'plddt' in trb:
                plddt = trb['plddt'].mean(1)
                record.update(dict(
                    plddt_start = plddt[0],
                    plddt_mid = plddt[len(plddt)//2],
                    plddt_end = plddt[-1],
                    plddt_mean = plddt.mean()
                ))
            if 'sampled_mask' in trb:
                record['sampled_mask'] = trb['sampled_mask']
            if 'config' in trb:
                flat = flatten_dictionary(trb['config'])
                record.update(flat)
            records.append(record)

        df_base = pd.DataFrame.from_records(records)
        print('writing run metadata (base metrics) to cached csv')
        df_base.to_csv(df_trb_path, index=None)

    # load computed metrics, if they exist
    print('loading computed metrics')
    # accumulate metrics for: no mpnn, mpnn, ligand mpnn
    df_all_list = [pd.DataFrame(dict(name=[]))]

    # metrics of "no mpnn" designs
    df_nompnn = df_base.copy()
    for path in [
        args.datadir+'/af2_metrics.csv.*',
        args.datadir+'/pyrosetta_metrics.csv.*',
    ]:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        df_nompnn = df_nompnn.merge(tmp, on='name', how='outer')

    if df_nompnn.shape[1] > df_base.shape[1]: # were there designs that we added metrics for?
        df_nompnn['mpnn'] = False
        df_nompnn['ligmpnn'] = False
        df_all_list.append(df_nompnn)

    # MPNN and LigandMPNN metrics
    def _load_mpnn_df(mpnn_dir, df_base):
        def strip_packing_suffix(name):
            has_packing_suffix = bool(re.match(r'.*-atomized-bb-(False|True)_\d+_\d+$', name)) or bool(re.match(r'.*cond\d+_\d+_\d+_\d+$', name))
            if has_packing_suffix:
                return re.sub(r'_\d+$', '', name)
            return name
        df_accum = pd.DataFrame(dict(name=[]))
        for path in [
            mpnn_dir+'/af2_metrics.csv.*',
            mpnn_dir+'/pyrosetta_metrics.csv.*',
            mpnn_dir+'/rosetta_gen_ff.csv.*',
        ]:
            df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
            tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
            n_unique_names = len(set(tmp['name']))
            n_names = len(tmp)
            if n_unique_names < n_names:
                print('Dropping {n_names - n_unique_names}/{n_names} duplicates from {path}')
                tmp.drop_duplicates('name', inplace=True)
            if len(tmp):
                tmp.name = tmp.name.map(strip_packing_suffix)
            df_accum = df_accum.merge(tmp, on='name', how='outer')

        # chemnet
        chemnet_dfs = [pd.read_csv(fn,index_col=None) for fn in glob.glob(mpnn_dir+'/chemnet_scores.csv.*')]
        tmp = pd.concat(chemnet_dfs) if len(chemnet_dfs)>0 else pd.DataFrame(dict(name=[]))
        if len(tmp)>0:
            chemnet1 = tmp.groupby('label',as_index=False).max()[['label','plddt','plddt_lp','lddt']]
            chemnet2 = tmp.groupby('label',as_index=False).min()[['label','lrmsd','kabsch']]
            chemnet3 = tmp.groupby('label',as_index=False).mean()[['label','lrmsd','kabsch']]
            colnames = tmp.columns[1:]
            chemnet = chemnet1.merge(chemnet2, on='label').rename(
                columns={col:'cn_'+col+'_best' for col in colnames})
            chemnet = chemnet.merge(chemnet3, on='label').rename(
                columns={col:'cn_'+col+'_mean' for col in colnames})
            chemnet = chemnet.rename(columns={'label':'name'})
            df_accum = df_accum.merge(chemnet, on='name', how='outer')

        # rosetta ligand
        df_s = [pd.read_csv(fn,index_col=None) for fn in glob.glob(mpnn_dir+'/rosettalig_scores.csv.*')]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        if len(tmp)>0:
            df_accum = df_accum.merge(tmp, on='name', how='outer')

        # mpnn likelihoods
        for seq_dir in [
            os.path.join(mpnn_dir, 'seqs'),
            os.path.join(mpnn_dir, '../seqs')
        ]:
            if os.path.exists(seq_dir):
                break
        mpnn_scores = load_mpnn_scores(seq_dir)
        df_accum = df_accum.merge(mpnn_scores, on='name', how='outer')
            
        df_accum['name'] = df_accum.name.map(strip_packing_suffix)
        df_accum['mpnn_index'] = df_accum.name.map(lambda x: int(x.split('_')[-1]))
        df_accum['mpnn_index'] = df_accum.name.map(lambda x: int(x.split('_')[-1]))
        df_accum['name'] = df_accum.name.map(lambda x: '_'.join(x.split('_')[:-1]))
        df_out = df_base.copy().merge(df_accum, on='name', how='right')
        return df_out

    # MPNN metrics
    for flavor in ['mpnn', 'ligmpnn']:
        mpnn_dir = f'{args.datadir}/{flavor}/'
        if os.path.exists(mpnn_dir):
            packed_dir = os.path.join(mpnn_dir,'packed')
            if os.path.exists(packed_dir):
                mpnn_dir = packed_dir
            df_ligmpnn = _load_mpnn_df(mpnn_dir, df_base)
            if df_ligmpnn.shape[1] > df_base.shape[1]: # were there designs that we added metrics for?
                df_ligmpnn['mpnn'] = False
                df_ligmpnn['ligmpnn'] = False
                df_ligmpnn[flavor] = True
                df_all_list.append(df_ligmpnn)

    # concatenate all designs into one list
    df = pd.concat(df_all_list)
    if len(df) == 0:
        df = df_base

    # add seq/struc clusters (assumed to be the same for mpnn designs as non-mpnn)
    backbone_metric_dirs = [os.path.join(d, 'csv.*') for d in glob.glob(args.datadir+'/metrics/per_design/*/')]
    for path in [
        args.datadir+'/tm_clusters.csv',
        args.datadir+'/blast_clusters.csv',
    ] + backbone_metric_dirs:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        ic(
            path,
            tmp.shape,
            tmp['name'][:3],
        )
        df = df.merge(tmp, on='name', how='outer')
        design_id_counts = sorted_value_counts(tmp, ['name'])
        assert (design_id_counts['count'] == 1).all(), f'{path} has duplicate metrics for some designs, re-run pipeline with metrics.invalidate_cache=1'

    # add seq/struc clusters (assumed to be the same for mpnn designs as non-mpnn)
    sequence_metric_dirs = [os.path.join(d, 'csv.*') for d in glob.glob(args.datadir+'/metrics/per_sequence/*/')]
    for path in sequence_metric_dirs:
        ic('sequence metrics path', path, len(glob.glob(path)))
        if len(glob.glob(path)) == 0:
            continue

        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        if 'catalytic_constraints.mpnn_packed.name' in tmp.columns:
            prefix = 'catalytic_constraints.mpnn_packed.'
            ic(prefix)
        elif 'catalytic_constraints.raw.name' in tmp.columns:
            prefix = 'catalytic_constraints.raw.'
        else:
            prefix = ''
        tmp['name'] = tmp[f'{prefix}name']
        tmp['mpnn_index'] = tmp[f'{prefix}mpnn_index']
        merge_keys = ['name']
        if 'mpnn_index' in tmp.columns:
            merge_keys.append('mpnn_index')
        df = df.merge(tmp, on=merge_keys, how='left', suffixes=(False, False))

    df.to_csv(args.datadir+'/'+args.outcsv, index=None)
    print(f'Wrote metrics dataframe {df.shape} to "{args.datadir}/{args.outcsv}"')

def load_mpnn_scores(folder):

    filenames = glob.glob(os.path.join(folder, '*.fa'))

    records = []
    for fn in filenames:
        scores = []
        with open(fn) as f:
            lines = f.readlines()
            for header in lines[2::2]:
                scores.append(float(header.split(',')[2].split('=')[1]))

            for i, score in enumerate(scores):
                records.append(dict(
                    name = os.path.basename(fn).replace('.fa','') + f'_{i}',
                    mpnn_score = score
                ))

    df = pd.DataFrame.from_records(records)
    return df

if __name__ == "__main__":
    main()
