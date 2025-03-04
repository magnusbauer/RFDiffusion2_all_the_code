import glob
import itertools
import math
import os
import re

from rf_diffusion.dev import analyze
# analyze.cmd = analyze.set_remote_cmd('10.64.100.67')
cmd = analyze.cmd

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rf_diffusion import metrics
from omegaconf import OmegaConf
from rf_diffusion.benchmark import compile_metrics

from rf_diffusion import aa_model
from rf_diffusion.dev import show_bench
from rf_diffusion.dev import show_tip_pa
import tree
import torch

try:
   api = wandb.Api(timeout=150)
except Exception as e:
   print(f'Warning: wandb API failed to instantiate: {e}')

def get_history(run_id, api, n_samples):
    run = api.run(f"bakerlab/fancy-pants/{run_id}")
    hist = run.history(samples=n_samples)
    hist = hist.set_index('_step')
    hist = hist.sort_values('_step')
    print(f'{n_samples=}, {len(hist)=}')
    return hist


def remove_prefix(str, prefix):
    return str.lstrip(prefix)

def get_loss_names(hist):
    loss_names = set()
    for k in hist.keys():
        if k.startswith('loss_weights'):
            loss_names.add(k[len('loss_weights.'):])
    return list(loss_names)

def get_loss_weights(hist):
    loss_names = get_loss_names(hist)
    loss_weights = {}
    row = hist.iloc[0]
    for loss_name in loss_names:
        loss_weights[loss_name] = row[f'loss_weights.{loss_name}']
    return loss_weights

def get_loss_df(hist):
    loss_names = get_loss_names(hist)
    # hist = hist[['_step'] + list(loss_names)]
    hist = hist[list(loss_names)]
    return hist


def plot_rolling_mean(hist, n_steps = 500):
    loss_weights = get_loss_weights(hist)
    losses = get_loss_df(hist)
    for loss, weight in sorted(loss_weights.items(), key=lambda x: x[1]):
        time_series_df = losses[[loss]]
        line = time_series_df.rolling(n_steps).mean()
        # line_deviation = time_series_df.rolling(n_steps).std()
        # under_line = (line - line_deviation)[loss]
        # over_line = (line + line_deviation)[loss]
        plt.plot(line, linewidth=2)
        # plt.fill_between(line_deviation.index, under_line,
        #                   over_line, color='red', alpha=.3)

        plt.title(f'{loss=} * {weight=}')
        plt.show()
        
sns.set(font_scale=0.8)
sns.set_context("paper")
sns.set_style("white")
plt.rcParams['axes.linewidth'] = 0.75

# Define custom color scheme
hex_codes = [
    "#4FB9AF",
    "#FFE0AC",
    "#FFC6B2",
    "#6686C5",

    "#FFACB7",
    "#4B5FAA",
    "#D59AB5",
    "#9596C6",
]
# Set color palette
sns.set_palette(hex_codes)
cm = 1/2.54

def melt_only(df, melt_vars, variable_renamer=None, var_name='variable', value_name='value', **kwargs):
    id_vars = df.columns
    id_vars = [v for v in id_vars if v not in melt_vars]
    melted = df.melt(id_vars, var_name=var_name, value_name=value_name, **kwargs)
    if variable_renamer:
        melted[var_name] = melted[var_name].map(variable_renamer)
    return melted


def extract_metric(x):
    if isinstance(x, torch.Tensor) and x.numel() == 0: return None
    return x.item() if hasattr(x, 'cpu') else x

def get_metrics(conf, metrics_inputs_list, metric_names=None):
    conf = OmegaConf.create(conf)
    if metric_names:
        OmegaConf.set_struct(conf, False)
        conf.metrics = metric_names
    # print(f'{conf.metric_names=}')
    # raise Exception('stopr')
    manager = metrics.MetricManager(conf)
    # print(f'{manager.metric_callables=}')
        
    all_metrics = []
    for metrics_inputs in metrics_inputs_list:
        m=compile_metrics.flatten_dictionary(dict(metrics=manager.compute_all_metrics(**metrics_inputs)))
        m['t'] = metrics_inputs['t']
        m=tree.map_structure(extract_metric, m)
        # m = {'metrics':m}
        all_metrics.append(m)
    return all_metrics


def numpy_to_tensor(a):
    if isinstance(a, np.ndarray):
        return torch.tensor(a)
    return a

def get_conf(trb_path):
    trb = np.load(trb_path,allow_pickle=True)
    return trb['config']

def get_inference_metrics_inputs(trb_path, only_x0=True):
    trb = np.load(trb_path,allow_pickle=True)
    n_t = trb['denoised_xyz_stack'].shape[0]
    all_metrics_inputs = []
    indep_true_dict = tree.map_structure(numpy_to_tensor, trb['indep_true'])
    # print(indep_true_dict)
    # indep_true_dict = tree.map_structure(torch.tensor, trb['indep_true'])
    indep_true = aa_model.Indep(**indep_true_dict)
    # indep_true = aa_model.Indep(**trb['indep_true'])
    for i in range(n_t):
        metrics_inputs = dict(
            indep=indep_true,
            pred_crds=torch.tensor(trb['px0_xyz_stack'][i][:,:3]),
            input_crds=torch.tensor(trb['denoised_xyz_stack'][i][:,:3]),
            true_crds=indep_true.xyz[:,:3],
            t=trb['t'][i],
            point_types=trb['point_types'],
            is_diffused=trb['is_diffused'],
            atomizer_spec=trb['atomizer_spec'],
            contig_as_guidepost=trb['config']['inference']['contig_as_guidepost'],
        )
        all_metrics_inputs.append(metrics_inputs)
        if only_x0:
            break
    return (trb['config'], all_metrics_inputs)

def get_inference_metrics(trb_path,
                            metric_names=[
                                # 'atom_bonds_permutations',
                                # 'rigid_loss',
                                # 'rigid_loss_input',
                                # 'VarianceNormalizedPredTransMSE',
                                # 'VarianceNormalizedInputTransMSE',
                                # 'displacement_permutations'
                            ],
                           **kwargs):
    dfi = get_inference_metrics_base(trb_path, metric_names=metric_names, **kwargs)
    conf = get_conf(trb_path)
    conf_flat = compile_metrics.flatten_dictionary(conf)
    conf_df = pd.DataFrame.from_records([conf_flat])
    dfi = dfi.merge(conf_df, how='cross').reset_index(drop=True)
    dfi['training_run'] = dfi['score_model.weights_path'].map(lambda x: x.split('/rank_')[0])
    # dfi = drop_nan_string_columns(dfi)
    return dfi

# Uncached
def _get_inference_metrics_base(trb_path, metric_names=None):
    # Example for one trajectory
    conf, metrics_inputs_list = get_inference_metrics_inputs(trb_path)
    m = get_metrics(conf, metrics_inputs_list, metric_names=metric_names)
    dfi = pd.DataFrame.from_records(m)
    dfi['trb_path'] = trb_path
    return dfi
    return dfi

from tqdm import tqdm
def get_inference_metrics_multi(pattern, metric_names=None):
    trb_paths = glob.glob(pattern)
    # print(pattern, trb_paths)
    metrics_dfs = []
    for trb in tqdm(trb_paths):
        metrics_dfs.append(get_inference_metrics(trb, metric_names=metric_names))
    # metrics_dfs = [get_inference_metrics(trb, metric_names=metric_names) for trb in trb_paths]
    return pd.concat(metrics_dfs)

def drop_nan_string_columns(df):
    # Get a list of column names that meet the criteria
    # columns_to_drop = [col for col in df.columns if (df[col].dtype == 'O') and (df[col] == 'NaN').all()]
    columns_to_drop = [col for col in df.columns if df[col].isna().all() or (df[col].dtype == 'O') and (df[col] == 'NaN').all()]
    
    # Drop the selected columns from the DataFrame
    df = df.drop(columns=columns_to_drop).reset_index(drop=True)

    return df

def get_training_metrics(wandb_id, n=9999, floor_every = 1/2000):
    TRAINING_T = 200
    N_EPOCH = 25600
    hist = get_history(wandb_id, api, n)
    hist = drop_nan_string_columns(hist)
    hist['t_cont'] = hist['t'] / TRAINING_T
    hist['t_cont_binned'] = hist['t_cont'].map(lambda x: (x // floor_every) * floor_every)
    hist['epoch'] = hist['total_examples'].map(lambda x: x // N_EPOCH)
    return hist

def get_frozen_training_for_inference(inference_dir):
    restart_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(inference_dir))))
    wandb_run_id = restart_dir.split('_')[-1]
    return get_training_metrics(wandb_run_id)

def get_training_metrics_multi(runid_by_name, n=9999):
    tmp = []
    for name, runid in runid_by_name.items():
        hist = get_training_metrics(runid, n=n)
        hist['name'] = name
        hist['run_name'] = name
        if name.startswith('ep'):
            epoch = int(name[2:])
            hist['epoch'] = epoch
        tmp.append(hist)
    return pd.concat(tmp)

def sorted_value_counts(df, cols):
    return pd.DataFrame(df.value_counts(cols, dropna=False)).sort_values(cols)

def get_inference_metrics_sweep(hp_sweep_run:str, regenerate_cache=False):
    metrics_path = os.path.join(hp_sweep_run, 'metrics_1.csv')
    print(f'loading {hp_sweep_run}')
    if os.path.exists(metrics_path) and not regenerate_cache:
        print('found cached metrics')
        dfi = pd.read_csv(metrics_path)
    else:
        print('recomputing metrics')
        dfi = get_inference_metrics_multi(hp_sweep_run + '*.trb', metric_names=[
                'atom_bonds_permutations',
                'rigid_loss',
                'rigid_loss_input',
                'VarianceNormalizedPredTransMSE',
                'VarianceNormalizedInputTransMSE',
                'displacement_permutations'
        ])
        dfi.to_csv(metrics_path)

    return dfi

def get_inference_metrics_sweep_multi(hp_sweep_runs: list[str], regenerate_cache=False, **kwargs):
    tmp = []
    for hp_sweep_run in hp_sweep_runs:
        tmp.append(get_inference_metrics_sweep(hp_sweep_run), **kwargs)
    return pd.concat(tmp)


def bin_metric(df, metric, bin_width):
    binned_metric = f'{metric}_bin_{bin_width}'
    df[binned_metric] = df[metric].map(lambda x: (x//bin_width) * bin_width)
    return binned_metric

def get_runid_by_name(group):
    runid_by_group = {}
    for run in api.runs("bakerlab/fancy-pants", filters={'group':group}):
        runid_by_group[run.name] = run.id
    print(f'{group=}, {runid_by_group=}')
    return runid_by_group

def get_training_metrics_groups(groups, n=9999, first_n_runs=999):

    tmp = []
    for group in groups:
        runid_by_name = get_runid_by_name(group)
        runid_by_name = {k:v for i, (k,v) in enumerate(runid_by_name.items()) if i < first_n_runs}
        print(f'{runid_by_name=}')
        df = get_training_metrics_multi(runid_by_name, n=n).copy()
        df['group'] = group
        tmp.append(df)
    return pd.concat(tmp)


def strip_group_timestamp(df):
    def f(group):
        return group.split('202')[0]
    df['group'] = df['group'].map(f)

def plot_self_consistency_ecdf(ax, df, **kwargs):

    ax = sns.ecdfplot(ax=ax, data=df, x="rmsd_af2_des", hue='method', **kwargs)
    xmin = 0
    xmax = 20
    ax.set(xlim=(xmin,xmax))
    ax.set_ylabel("Proportion")
    
    x_special = 2.0
    for i, line in enumerate(ax.get_lines()):
        x, y = line.get_data()
        ind = np.argwhere(x >= x_special)[0, 0]  # first index where y is larger than y_special
        y_int = y[ind]
        # j = {0:0, 1:2, 2:1}[i]
        j=i
        ax.text(xmax-1.4*j-0.2, y_int-0.015, f' {y_int:.2f}', ha='right', va='top', fontsize=6,bbox={'facecolor': line.get_color(), 'alpha': 0.5, 'pad': 1})
        ax.axhline(y_int, xmax=1, xmin=x_special/xmax, linestyle='--', color='#cfcfcf', alpha=0.95, lw=0.5)
    ax.axvline(x=2, color='grey', linestyle='-')


def plot_self_consistency_ecdf_benchmark(bench, **kwargs):
    benchmarks = bench['benchmark'].unique()

    print(f'{benchmarks=}')

    fig, axes = plt.subplots(nrows=1,ncols=len(benchmarks),
                            figsize=(19.5*cm,7*cm),
                            # constrained_layout=True,
                            dpi=300, squeeze=0)
    print(f'{axes}')
    for ax, benchmark in zip(axes[0,:], benchmarks):
        df_bench = bench[bench['benchmark'] == benchmark]
        plot_self_consistency_ecdf(ax, df_bench, **kwargs)

def show_percents(g):
    # iterate through axes
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()*100):.1f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
        _ =ax.tick_params(axis='x', rotation=90)
        
def add_metrics_sc(df):
    df['self_consistent'] = df['rmsd_af2_des'] < 2.0
    df['self_consistent_and_motif'] = df['self_consistent'] & (np.isnan(df['contig_rmsd_af2_des']) | (df['contig_rmsd_af2_des'] < 1.0))

def get_best(df, motif_first=True):
    other_sorts = ['contig_rmsd_af2_des', 'rmsd_af2_des']
    if not motif_first:
        other_sorts = ['rmsd_af2_des', 'contig_rmsd_af2_des']

    data = df.groupby(["design_id"], dropna=False).apply(lambda grp: grp.sort_values(['self_consistent_and_motif'] + other_sorts, ascending=[False, True, True]).head(1)).reset_index(drop=True)
    return data

def get_most_sc_designs_in_group(df, groups, n=1):
    return get_best_n_designs_in_group(df, groups, n, ['self_consistent_and_motif', 'contig_rmsd_af2_des', 'rmsd_af2_des'], ascendings=[False, True, True])

def get_best_n_designs_in_group(df, groups, n, columns, ascendings):
    assert max(df['design_id'].value_counts()) == 1
    data = df.groupby(groups,  dropna=False).apply(lambda grp: grp.sort_values(columns, ascending=ascendings).head(n)).reset_index(drop=True)
    return data

def get_least_in_group_single(df, column, ascending=True, groups=['design_id']):
    return get_least_in_group(df, groups, 1, [column], ascendings=[ascending])

def get_least_in_group(df, groups, n, columns, ascendings):
    data = df.groupby(groups,  dropna=False).apply(lambda grp: grp.sort_values(columns, ascending=ascendings).head(n)).reset_index(drop=True)
    return data

def get_least_in_group_indexed(df, groups, n, columns, ascendings):
    index_name = df.index.name
    df.reset_index(inplace=True)
    df.set_index(groups, inplace=True)
    data = df.groupby(groups,  dropna=False).apply(lambda grp: grp.sort_values(columns, ascending=ascendings).head(n)).reset_index(drop=True)
    df.set_index(index_name, inplace=True)
    return data

def get_training_id(bench):
    return bench['score_model.weights_path'].map(lambda x: x.split('/')[-4].split('2023')[0])

def only_latest_epoch(df):
    # latest_epoch = df.groupby('training_id')
    if 'training_id' not in df:
        df['training_id'] = get_training_id(df)
    highest_epoch = get_least_in_group(df[['training_id', 'epoch']], ['training_id'], 1, ['epoch'], ascendings=[False])
    # highest_epoch = highest_epoch.set_
    return df.merge(highest_epoch, on=['training_id', 'epoch'])
    # return get_best_n_designs_in_group(df, ['training_id'], n=999999, 

def plot_self_consistency(bench, x='epoch', hue='benchmark', **kwargs):
    # x = 'score_model.weights_path'
    # if 'method' in bench.columns:
    #     x='method'
    add_metrics_sc(bench)
    data = get_best(bench)
    data['motif RMSD < 1 & RMSD < 2'] = data['self_consistent_and_motif']
    # g = sns.catplot(data=data, y='motif RMSD < 1 & RMSD < 2', x=x, hue='benchmark', kind='bar', orient='v', height=8.27, aspect=11.7/8.27, legend_out=True, ci=None, **kwargs)
    sns.catplot(data=data, y='motif RMSD < 1 & RMSD < 2', x=x, hue=hue, kind='bar', orient='v', legend_out=True, ci=None, **kwargs)
    _ = plt.xticks(rotation=90)
    # show_percents

def plot_self_consistency_no_motif(bench, x='epoch', hue='benchmark', **kwargs):
    # x = 'score_model.weights_path'
    # if 'method' in bench.columns:
    #     x='method'
    add_metrics_sc(bench)
    data = get_least_in_group_single(bench, 'rmsd_af2_des')
    data['RMSD < 2'] = data['rmsd_af2_des'] < 2.0
    # g = sns.catplot(data=data, y='motif RMSD < 1 & RMSD < 2', x=x, hue='benchmark', kind='bar', orient='v', height=8.27, aspect=11.7/8.27, legend_out=True, ci=None, **kwargs)
    sns.catplot(data=data, y='RMSD < 2', x=x, hue=hue, kind='bar', orient='v', legend_out=True, ci=None, **kwargs)
    _ = plt.xticks(rotation=90)
    # show_percents


def autobench_df(training_dirs):
    metrics_csvs = []
    for d in training_dirs:
        metrics_csvs.extend(metrics_paths(d))
    bench = analyze.combine(*metrics_csvs)
    bench['training'] = bench['score_model.weights_path'].map(lambda x: x.split('/')[-4].split('2023')[0])
    bench['epoch'] = bench.apply(show_bench.get_epoch, axis=1)
    bench['design_id'] = bench['training'] + '_ep' + bench['epoch'].astype(str) + '_' + bench['name']
    return bench

def metrics_paths(training_dir):
    o = []
    autobench_dir_pattern = os.path.join(training_dir, 'rank_0/models/auto_benchmark/*/out/compiled_metrics.csv')
    for d in glob.glob(autobench_dir_pattern):
        o.append(d)
    return o

def show_unconditional_performance_over_epochs(bench):
    bench['training'] = bench['score_model.weights_path'].map(lambda x: x.split('/')[-4].split('2023')[0])
    bench['epoch'] = bench.apply(show_bench.get_epoch, axis=1)
    # get_epoch  = lambda x: float(re.match('.*_(\w+).*', x).groups()[0])
    # bench['epoch'] = bench['score_model.weights_path'].apply(get_epoch)
    bench['method'] = bench['epoch']
    bench['method'] = bench['method'].astype(int)
    bench['ema'] = bench['inference.state_dict_to_load'] == 'model_state_dict'
    bench = bench.sort_values('method')

    print(f'{bench.shape=}')

    # show = bench[bench['inference.state_dict_to_load'] == 'model_state_dict']
    show = bench
    show = show[show['benchmark'] == 'unconditional']
    print(f'{show.shape=}')
    plot_self_consistency(show, row='ema', col='training')
    plt.title("Self consistency of unconditonal generation")

def show_performance_over_epochs(bench, col='training', row='ema', **kwargs):
    # bench['training'] = bench['score_model.weights_path'].map(lambda x: x.split('/')[-4].split('2023')[0])
    # get_epoch  = lambda x: float(re.match('.*_(\w+).*', x).groups()[0])
    # bench['epoch'] = bench['score_model.weights_path'].apply(get_epoch)
    bench['method'] = bench['epoch']
    bench['method'] = bench['method'].astype(int)
    bench['ema'] = bench['inference.state_dict_to_load'] == 'model_state_dict'
    bench = bench.sort_values('method')

    print(f'{bench.shape=}')

    # show = bench[bench['inference.state_dict_to_load'] == 'model_state_dict']
    show = bench
    # show = show[show['benchmark'] == 'unconditional']
    print(f'{show.shape=}')
    plot_self_consistency(show, col=col, row=row, **kwargs)

def get_trb_path(row):
    return os.path.join(row['rundir'], f'{row["name"]}.trb')

# def get_autobench_trajectory_metrics(bench, **kwargs):
#     trbs = bench.apply(get_trb_path, axis=1)
#     trbs =list(set(trbs)) 
#     tmp = []
#     for trb_path in tqdm(trbs):
#         tmp.append(get_inference_metrics(trb_path, **kwargs))
#     return pd.concat(tmp)

def get_inference_metrics_base(trb_path:str,
                              metric_names=[
                                    # 'atom_bonds_permutations',
                                    # 'rigid_loss',
                                    # 'rigid_loss_input',
                                    # 'VarianceNormalizedPredTransMSE',
                                    # 'VarianceNormalizedInputTransMSE',
                                    'IdealizedResidueRMSD',
                                    # 'displacement_permutations'
                                ],
                                regenerate_cache=True):
    trb_dir, trb_name = os.path.split(trb_path)
    trb_name, _ = os.path.splitext(trb_name)
    cache_dir = os.path.join(trb_dir, 'metrics_cache')
    os.makedirs(cache_dir, exist_ok=True)
    metrics_path = os.path.join(cache_dir, trb_name + '.csv')
    if os.path.exists(metrics_path) and not regenerate_cache:
        # print('found cached metrics')
        dfi = pd.read_csv(metrics_path)
    else:
        # print('recomputing metrics')
        dfi = _get_inference_metrics_base(trb_path, metric_names)
        dfi.to_csv(metrics_path)

    # dfi['training_run'] = dfi['score_model.weights_path'].map(lambda x: x.split('/')[-4])
    # dfi['training'] = dfi['training_run'].map(lambda x: x.split('2023')[0])
    return dfi


# def pymol_best_from_each_epoch(bench, unique_keys = ['training', 'epoch', 'benchmark'], n=1):
#     # show = bench[bench['benchmark'] == '10_res_atomized_1']
#     show = bench[~bench['benchmark'].isin(['10_res_atomized_1', '10_res_atomized_2', '10_res_atomized_3'])].copy()
#     add_metrics_sc(show)
#     show = get_best(show)
#     show = get_most_sc_designs_in_group(show, unique_keys, n=n)
#     show_bench.add_pymol_name(show, unique_keys + ['seed', 'rmsd_af2_des', 'contig_rmsd_af2_des'])
#     show_tip_pa.clear()

#     print(f'showing {len(show)} designs')
#     all_entities = show_bench.show_df(
#         show,
#         # structs={},
#         structs={'X0'},
#         des=0,
#         # pair_seeds=pair_seeds,
#         # af2=af2,
#         # mpnn_packed=mpnn_packed,
#         # ga_lig=ga_lig,
#         # rosetta_lig=rosetta_lig,
#         # hydrogenated=hydrogenated,
#         return_entities=True)
    
#     return all_entities

def get_autobench_trajectory_metrics(bench, **kwargs):
    trbs = bench.apply(get_trb_path, axis=1)
    trbs =list(set(trbs)) 
    tmp = []
    for trb_path in tqdm(trbs):
        x = get_inference_metrics(trb_path, **kwargs)
        x['trb_path'] = trb_path
        x['name'], _ = os.path.splitext(os.path.basename(trb_path))
        n = x.iloc[0]['name']
        x['benchmark'] = n[n.index('_')+1:n.index('_cond')]
        tmp.append(x)
    return pd.concat(tmp)

def pymol_best_from_each_epoch(
        bench,
        unique_keys = ['training', 'epoch', 'benchmark'],
        n=1, 
        structs={'X0'},
        mpnn_packed=False,
        af2=False,
        **kwargs):
    # show = bench[bench['benchmark'] == '10_res_atomized_1']
    # show = bench[~bench['benchmark'].isin(['10_res_atomized_1', '10_res_atomized_2', '10_res_atomized_3'])].copy()
    show = bench
    add_metrics_sc(show)
    show = get_best(show)
    show = get_most_sc_designs_in_group(show, unique_keys, n=n)
    show_bench.add_pymol_name(show, unique_keys + ['seed', 'rmsd_af2_des', 'contig_rmsd_af2_des'])
    show = show.sort_values(unique_keys)
    show_tip_pa.clear()

    print(f'showing {len(show)} designs')
    all_entities = show_bench.show_df(
        show,
        # structs={},
        structs=structs,
        des=0,
        # pair_seeds=pair_seeds,
        af2=af2,
        mpnn_packed=mpnn_packed,
        # ga_lig=ga_lig,
        # rosetta_lig=rosetta_lig,
        # hydrogenated=hydrogenated,
        return_entities=True)
    # cmd.do(f'mass_paper_rainbow')
    # cmd.show('licorice')
    return all_entities

def show_by_seed(
        bench,
        unique_keys = ['training', 'epoch', 'benchmark'],
        n=1, 
        structs={'X0'},
        mpnn_packed=False,
        des=0,
        af2=False):
    # show = bench[bench['benchmark'] == '10_res_atomized_1']
    # show = bench[~bench['benchmark'].isin(['10_res_atomized_1', '10_res_atomized_2', '10_res_atomized_3'])].copy()
    show = bench
    add_metrics_sc(show)
    show = get_best(show)
    show = show[show['seed'] < n]
    print(show.shape)
    show_bench.add_pymol_name(show, unique_keys + ['seed', 'rmsd_af2_des', 'contig_rmsd_af2_des'])
    show = show.sort_values(unique_keys)
    show_tip_pa.clear()

    print(f'showing {len(show)} designs')
    all_entities = show_bench.show_df(
        show,
        structs=structs,
        des=des,
        af2=af2,
        mpnn_packed=mpnn_packed,
        return_entities=True)
    return all_entities

def isnan(x):
    return isinstance(x, float) and math.isnan(x)

def add_cc_columns(df):

    add_metrics_sc(df)
    df['seq_id'] = df['name'] + '_' + df['mpnn_index'].astype('str')
    
    for subtype in [
        'raw',
        'mpnn_packed',
    ]:
        prefix = f'catalytic_constraints.{subtype}.'
        df[f'{prefix}all'] = (
            df[f'{prefix}criterion_1'] &
            df[f'{prefix}criterion_2'] &
            df[f'{prefix}criterion_3'] &
            df[f'{prefix}criterion_4'] &
            df[f'{prefix}criterion_5'] &
            df[f'{prefix}criterion_6']
        )


def best_in_group(df, group_by=['design_id'], cols=['catalytic_constraints.raw.criterion_1'], ascending=[False], unique_column='seq_id', n=1):
    assert df[unique_column].is_unique
    df_small  = df[group_by + cols + [unique_column]]
    grouped = df_small.groupby(group_by).apply(lambda grp: grp.sort_values(cols, ascending=ascending).head(n))
    return pd.merge(df, grouped[unique_column], on=unique_column, how='inner')


def best_in_group_fast(df, group_by=['design_id'], cols=['catalytic_constraints.raw.criterion_1'], ascending=True):
    return df.sort_values(by=cols, ascending=ascending).drop_duplicates(group_by, keep='first')

def get_best_in_group_for_each_metric(
        df,
        melt_vars,
        variable_renamer=None,
        var_name='variable',
        value_name='value',
        group_by=['design_id'],
        ascending=True):

    # This helps for bookkeeping later
    assert df.index.is_unique
    df_small = df[melt_vars + group_by]
    df_small.reset_index(inplace=True)

    melted = melt_only(df_small, melt_vars,
            var_name=var_name, value_name=value_name,
            variable_renamer=variable_renamer)
    
    best = best_in_group_fast(melted, group_by=group_by + [var_name], cols=[value_name], ascending=ascending)
    return best, melted

def get_cc_passing(df, subtypes=('raw',)):
    all_melted = {}
    for subtype in subtypes:
        prefix = f'catalytic_constraints.{subtype}.'
        filter_names = [f'{prefix}criterion_{i}' for i in range(1,7)] + [f'{prefix}all']
        filter_names_no_prefix = [f'criterion_{i}' for i in range(1,7)] + ['criterion_all']
        df_remapped = df.rename(columns=dict(zip(filter_names, filter_names_no_prefix)))
        df_remapped['pack'] = subtype
        filter_names = filter_names_no_prefix
        melts = []
        for filter_union in filter_names:
            best_filter_passers = best_in_group(df_remapped,
                                                cols=[filter_union, 'contig_rmsd_af2_full_atom'],
                                                ascending=[False, True]
            )
            melted = analyze.melt_filters(best_filter_passers, [filter_union]).copy()
            melted['filter_set'] = filter_union
            melts.append(melted)
        melted = pd.concat(melts)
        all_melted[subtype] = melted.copy()
    by_pack = pd.concat(all_melted.values())
    return by_pack

def columns_with_substring(df, substring):
    o = []
    for c in df.columns:
        if substring in c:
            o.append(c)
    return o

def columns_with_substring_value(df, substring):
    '''
    Same as above, but include an example value for that column
    '''
    o = columns_with_substring(df, substring)
    return {c: df.iloc[0][c] for c in o}


def get_sweep_df(df, sweeps):
    '''
        Arguments:
            sweeps: list of swept parameters
        Returns:
            A dataframe where column `sweep` details which sweep the row comes from.
    '''
    sweep_dfs = []
    modes = df[sweeps].mode()
    for sweep in sweeps:
        non_swept_columns = [s for s in sweeps if s != sweep]
        other_modes = modes[non_swept_columns]
        tmp = df.merge(other_modes, on=non_swept_columns, how='inner')
        tmp['sweep'] = sweep
        sweep_dfs.append(tmp)

    sweep_df = pd.concat(sweep_dfs)
    print(f"{sweep_df.shape=}")
    print(f"{sweep_df.value_counts('sweep')=}")
    return sweep_df

def grouped_palette(
        hue_groups, # [(a,b,c), (d,e)]
        palettes=itertools.cycle(['flare', 'mako', 'vlag'])):
    hues_flat = []
    palette_flat = []
    for hues, sub_pallete in zip(hue_groups, palettes):
        hues_flat.extend(hues)
        palette_flat.extend(sns.color_palette(sub_pallete, n_colors=len(hues)))
    return hues_flat, palette_flat

def get_n_motif(row):
    return len(eval(row['contigmap.contig_atoms']))

def get_n_contiguous_motif(row):
    # print(f"{row['contigmap.contigs']=}")
    contigs = eval(row['contigmap.contigs'])
    assert len(contigs) == 1
    contigs =  contigs[0]
    # print(f'{contigs=}')
    contigs = contigs.split(',')
    # print(f'{contigs=}')

    n_contiguous = 0
    for c in contigs:
        if c[0].isalpha():
            n_contiguous += 1
    return n_contiguous

def get_n_motif_contig_residues(row):
    contigs = eval(row['contigmap.contigs'])
    assert len(contigs) == 1
    contigs =  contigs[0]
    # print(f'{contigs=}')
    contigs = contigs.split(',')
    # print(f'{contigs=}')

    n_res = 0
    for c in contigs:
        if c[0].isalpha():
            # re.match(r'[A-Z]([0-9]+)-([0-9]+)', c)
            m = re.match(r'[A-Z]([0-9]+)-([0-9]+)', c)
            contig_start, contig_end = int(m.group(1)), int(m.group(2))
            n_res += contig_end - contig_start + 1
    return n_res

def motif_has_backbone_atom(row):
    backbone_atoms = ['N', 'CA', 'C', 'O']
    d = eval(row['contigmap.contig_atoms'])
    d = {k: v.split(',') for k,v in d.items()}
    for v in d.values():
        assert isinstance(v, list), f'{v=} {type(v)=}'
        if any(a in v for a in backbone_atoms):
            return True
    return False


def label_bar_heights(ax):
    def round_half_up(n, decimals=0):
        multiplier = 10 ** decimals
        return np.floor(n * multiplier + 0.5) / multiplier

    # show the mean
    label_y_offset = 0.002
    for p in ax.patches:
        h, w, x = p.get_height(), p.get_width(), p.get_x()
        xy = (x + w / 2., h + label_y_offset)

        h = round_half_up(h, 3)
        
        text = f'{h*100:0.1f}%'
        _ = ax.annotate(text=text, xy=xy, ha='center', va='center')

    max_height = max(p.get_height() for p in ax.patches)
    _ = plt.ylim(0, max_height + 2 * label_y_offset + 0.01)

def set_seq_id(df):
    df['seq_id'] = analyze.fast_apply(df, lambda x: f'{x["design_id"]}_{x["mpnn_index"]}', ['design_id', 'mpnn_index'])

def set_sequence_id_unique_index(df):
    drop = df.index.name == 'seq_id'
    df.reset_index(inplace=True, drop=drop)
    set_seq_id(df)
    df.set_index('seq_id', inplace=True, verify_integrity=True)

def designs_with_all_sequences_scored(
        df,
        score_column,
        expected_sequences_per_design=8,
        verbose=True):

    df.reset_index(inplace=True)
    df.set_index('design_id', inplace=True)

    df[f'has_{score_column}'] = df[score_column].notna()
    designs = df.groupby('design_id').agg({f'has_{score_column}': 'sum'})

    has_all_sequences_scored = designs[f'has_{score_column}'] == expected_sequences_per_design
    good_design_ids = designs[has_all_sequences_scored].index
    df_good = df.loc[good_design_ids]

    assert len(df_good) == (len(good_design_ids) * expected_sequences_per_design)

    if verbose:
        print(f'{len(df_good)}/{len(df)} = {len(df_good)/len(df)}  designs with all {expected_sequences_per_design} sequences having non-NA {score_column}')
    return df_good

pocket_aligned_prefix = 'chai_pocket_aligned_'
def is_raw_chai1_pocket_column(column):
    return column.startswith('pocket_aligned_model_')

def is_processed_chai1_pocket_column(column):
    return column.startswith(pocket_aligned_prefix)

def rename_pocket_aligned_columns(df):
    def mapper(column):
        if is_raw_chai1_pocket_column(column):
            m = re.match(r'^pocket_aligned_model_(\d+)_(.*)', column)
            return f'{pocket_aligned_prefix}{m.group(2)}-{m.group(1)}'
        return column
    df.rename(
        columns=mapper,
        inplace=True
    )


def melt_chai_models(df, pocket_aligned_prefix=pocket_aligned_prefix):

    rename_pocket_aligned_columns(df)

    stubname_cols = df.columns[df.columns.str.startswith(pocket_aligned_prefix)]
    stubnames = [re.match(rf'^({pocket_aligned_prefix}.*)-\d+', col).group(1) for col in stubname_cols]
    stubnames = list(set(stubnames))

    # df.reset_index(inplace=True, drop=True)
    assert df.index.is_unique
    df['id'] = df.index

    return pd.wide_to_long(
        df=df,
        i='id',
        j='chai_model_idx',
        stubnames=stubnames,
        sep='-',
        suffix=r'.*'
    )

def safe_merge(*args, merge_assert='both', **kwargs):
    out = pd.merge(*args, **kwargs, indicator=True)
    if merge_assert == 'both':
        assert out['_merge'].unique() == [merge_assert], 'corresponding row not found in both dataframes'
    out.drop(['_merge'], axis=1, inplace=True)
    return out


def join_on_seq_id(df, additional, additional_suffix=None, one_to_one=True, **kwargs):
    print(f'{df.index.name=}')
    print(f'{additional.index.name=}')
    assert df.index.name == 'seq_id'

    assert additional.index.name == 'seq_id'
    assert additional.index.is_unique
    if one_to_one:
        assert df.index.is_unique

    m = df.merge(additional, how='left', left_on='seq_id', right_index=True, indicator=True, suffixes=(None, additional_suffix))
    assert m['_merge'].unique() == ['both'], 'corresponding row in df not found for seq_id in additoinal'
    return m

### Chai melting utilities

def trim_all_sym_resolved_suffixes(df):
    '''
    For any column ending in _sym_resolved, remove the suffix if the column without the suffix is not present already in the dataframe.
    '''
    for col in df.columns:
        if col.endswith('_sym_resolved'):
            base = col[:-len('_sym_resolved')]
            if base not in df.columns:
                df.rename(columns={col: base}, inplace=True)

chaimodel = 'chaimodel'
def normalize_chai_columns(df):
    '''
    Put the model suffix at the end
    '''

    def pocket_metrics_mapper(column):
        if column.startswith('pocket_aligned_model_'):
            m = re.match(r'^pocket_aligned_model_(\d+)_(.*)', column)
            return f'{pocket_aligned_prefix}{m.group(2)}_{chaimodel}_{m.group(1)}'
        return column
    
    for mapper in [pocket_metrics_mapper]:
        df.rename(
            columns=mapper,
            inplace=True
        )

def melt_all_chai_models(df):

    normalize_chai_columns(df)

    stubname_cols = df.columns[df.columns.str.contains(chaimodel)]
    # print(f'{stubname_cols=}')

    # stubnames = []

    stubnames = []
    for col in stubname_cols:
        pattern = rf'^(.*_{chaimodel})_\d+'
        m = re.match(pattern, col)
        if not m:
            raise Exception(f'column {col} does not match the regex: {pattern}')
        stubnames.append(m.group(1))

    # # stubnames = [re.match(rf'^({pocket_aligned_prefix}.*)-\d+', col).group(1) for col in stubname_cols]
    stubnames = list(set(stubnames))
    # print(f'{stubnames=}')

    # df.reset_index(inplace=True, drop=True)
    assert df.index.is_unique
    df['id'] = df.index

    out = pd.wide_to_long(
        df=df,
        i='id',
        j='chai_model_idx',
        stubnames=stubnames,
        sep='_',
        suffix=r'\d'
        # suffix=r'.*'
    )
    out.reset_index('chai_model_idx', inplace=True)

    # Remove the suffix chaimodel from all column names
    out.columns = [re.sub(rf'_{chaimodel}$', '', col) for col in out.columns]
    return out


### Subprocess compiled_metrics.csv loading utilities

import psutil
import multiprocessing

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024**3

def print_memory_usage():
    print(f'{get_memory_usage()} GB')


def load_data(path_by_name):
    print('START load_data')
    print_memory_usage()

    df = analyze.combine(
        *list(path_by_name.values()),
        names=list(path_by_name.keys()),
        low_memory=False
    )

    # Drop the expensive columns -- the residue specific RMSDs
    uneeded_columns = columns_with_substring(df, '_residue_')
    df.drop(columns=uneeded_columns, inplace=True)

    print(f'pre-dict {df.shape=}')
    df = pd.DataFrame.from_dict(df.reset_index(drop=True).to_dict())
    print(f'post-dict {df.shape=}')

    print_memory_usage()
    print('END load_data')
    return df

def load_data_multiprocess(path_by_name):    
        
    print_memory_usage()
    p = multiprocessing.Pool(1) 
    df = p.map(load_data, [path_by_name])[0] 
    # df = df.convert_dtypes()
    p.terminate() 
    p.join() 
    print_memory_usage()
    print(f'{df.shape=}')
    return df

### Chai melting utilities


def add_motif_length_metrics(df):
    # per_benchmark = df[df['is_rfd']].drop_duplicates(['benchmark'])
    per_benchmark = df[df['source'] == 'rfd_shuffled'].drop_duplicates(['benchmark'])
    per_benchmark['n_contiguous_motif'] = per_benchmark.apply(lambda r: get_n_contiguous_motif(r), axis=1)

    n_before = len(df)
    df_tmp = safe_merge(
        df,
        per_benchmark[['benchmark', 'n_contiguous_motif']],
        how='left',
        left_on='benchmark',
        right_on='benchmark',
        validate='m:1',
        suffixes=('_left', None),
        merge_assert=None,
    )
    n_after = len(df_tmp)
    assert n_before == n_after
    print(f"AFTER:  {df_tmp.iloc[0]['benchmark']=}")
    assert not df_tmp['benchmark'].isna().any()
    return df_tmp

def add_n_motif_metrics(df):
    per_benchmark = df[~df['is_rfd']].drop_duplicates(['benchmark'])
    per_benchmark['n_motif'] = per_benchmark.apply(lambda r: get_n_motif(r), axis=1)

    n_before = len(df)
    df_tmp = safe_merge(
        df,
        per_benchmark[['benchmark', 'n_motif']],
        how='left',
        left_on='benchmark',
        right_on='benchmark',
        validate='m:1',
        suffixes=('_left', None),
        merge_assert=None,
    )
    n_after = len(df_tmp)
    assert n_before == n_after
    print(f"AFTER:  {df_tmp.iloc[0]['benchmark']=}")
    assert not df_tmp['benchmark'].isna().any()
    return df_tmp

def add_per_benchmark_metric(df, metric_names, per_benchmark):
    n_before = len(df)
    df_tmp = safe_merge(
        df,
        per_benchmark[['benchmark'] + metric_names],
        how='left',
        left_on='benchmark',
        right_on='benchmark',
        validate='m:1',
        suffixes=('_left', None),
        merge_assert=None,
    )
    n_after = len(df_tmp)
    assert n_before == n_after
    print(f"AFTER:  {df_tmp.iloc[0]['benchmark']=}")
    assert not df_tmp['benchmark'].isna().any()
    return df_tmp

def filter_incomplete_benchmarks(df, assert_fraction_complete=1.0):

    # Drop any incomplete benchmarls
    val_counts = sorted_value_counts(df, 'benchmark')
    n_benchmarks = len(val_counts)
    n_rows_expected = val_counts.max().values[0]
    # len(val_counts[val_counts[0] == n_rows_expected])
    df_filt = df[df['benchmark'].isin(val_counts[val_counts[0] == n_rows_expected].index)]

    n_benchmarks_filtered = len(df_filt['benchmark'].drop_duplicates())
    assert n_benchmarks_filtered >= assert_fraction_complete * n_benchmarks
    assert df_filt.shape[0] == n_rows_expected * n_benchmarks_filtered
    return df_filt

def add_motif_metrics(df, n_contiguous_method = 'same_within_benchmark'):
    if n_contiguous_method == 'same_within_benchmark':
        for benchmark, group in df.groupby('benchmark'):
            # contig_strings = df[df['benchmark'] == benchmark]['contigmap.contigs']
            contigs = group['contigmap.contigs'].drop_duplicates()
            print(f'{benchmark=} {contigs=}')
            assert len(contigs) == 1
        per_benchmark = df.drop_duplicates(['benchmark'])
    elif n_contiguous_method == 'from_rfd':
        per_benchmark = df[df['is_rfd']].drop_duplicates(['source', 'benchmark'])
        # per_benchmark['n_contiguous_motif'] = per_benchmark.apply(lambda r: get_n_contiguous_motif(r), axis=1)
        # per_benchmark[['benchmark', 'source', 'n_contiguous_motif']].sort_values(['benchmark', 'source', 'n_contiguous_motif'], ascending=False)
    else:
        raise Exception(f'Unknown n_contiguous_method: {n_contiguous_method=}')

    per_benchmark['n_contiguous_motif'] = per_benchmark.apply(lambda r: get_n_contiguous_motif(r), axis=1)
    per_benchmark['n_motif'] = per_benchmark.apply(lambda r: get_n_motif_contig_residues(r), axis=1)
    return add_per_benchmark_metric(df, ['n_motif', 'n_contiguous_motif'], per_benchmark)

def index_check(df):
    print(f'{"id" in df.columns=}')
    print(f'{"design_id" in df.columns=}')
    print(f'{"seq_id" in df.columns=}')
    print(f'{"chai_model_idx" in df.columns=}')
    print(f'{df.index.name=}')

def get_chai_melt_dataframe(compiled_metrics_df, n_contiguous_method='same_within_benchmark'):
    df = compiled_metrics_df

    # Normalize inverse rotamer benchmark names
    df['benchmark_raw'] = df['benchmark']
    sorted_value_counts(df, ['source', 'benchmark'])
    df['benchmark'] = df['benchmark'].apply(lambda x: '_'.join(x.split('_')[:2]))
    sorted_value_counts(df, ['source', 'benchmark']).groupby('source').count()

    # Set 'is_rfd' (whether the design comes from vanilla RFDiffusion)
    df['is_rfd'] = df.apply(analyze.is_rfd, axis=1)

    # # Set 'n_contiguous_motif' and 'n_motif'
    # df = add_motif_length_metrics(df)
    # df = add_n_motif_metrics(df)
    # Set 'n_contiguous_motif' and 'n_motif'
    df = add_motif_metrics(df, n_contiguous_method=n_contiguous_method)

    # SHOULD BE UNNEEDED
    # df_filt = get_scored_designs(df)
    df = filter_incomplete_benchmarks(df)

    # Add the sequence ID index
    set_sequence_id_unique_index(df)

    # Drop sym resolved suffixes
    df = df.copy()
    trim_all_sym_resolved_suffixes(df)

    # # SHOULD BE UNNEEDED
    # # Assert indexes are as expected.
    # index_check(df)

    # Create a row for each chai model idx
    print('BEFORE MELT')
    index_check(df)
    chai_melt = melt_all_chai_models(df)
    print('AFTER MELT 0')
    index_check(chai_melt)

    # Assert all rows turned into 'n_chai_models' rows
    n_chai_models = 5 # 5 chai models per sequence
    assert chai_melt.shape[0] == df.shape[0] * n_chai_models
    assert 'chai_model_idx' in chai_melt.columns

    # Assert that the reference motif is fully present in the design.
    # This is important for the validity of following metrics that measure
    # rmsd between the design and the prediction.

    # DEBUG:
    # return chai_melt
    # set_chai_motif_goal(chai_melt, 
    if (~chai_melt['is_rfd']).any():
        keys_to_check = [
            ('metrics_meta_rmsd_to_input', 1e-3),
            ('all_minus_backbone_oxygen_rmsd_to_input', 0.1)
        ]
        for key, threshold in keys_to_check:
            motif_rmsds_to_input = chai_melt[~chai_melt['is_rfd']][key]
            assert not motif_rmsds_to_input.isna().any()
            assert motif_rmsds_to_input.max() < threshold, f'for {key=}: {motif_rmsds_to_input.max()=} > {threshold}'
            del motif_rmsds_to_input
        # motif_rmsds_to_input = chai_melt[~chai_melt['is_rfd']]['motif_rmsd_to_input']
        # assert not motif_rmsds_to_input.isna().any()
        # assert motif_rmsds_to_input.max() < 0.05
        # del motif_rmsds_to_input
    
    # Make the RMSD to input the goal metric for RFD, and the RMSD to the design the goal metric for the rest.
    # This is valid due to the preceding step in which we assert that the motif is fully present in the design for non-RFD designs.
    # The comparable metric created here is chai_motif_goal.
    chai1_motif_goal_rfflow = 'allatom_aligned_allatom_rmsd_chai_unideal'
    chai1_motif_goal_rfd = 'allatom_aligned_allatom_rmsd_chai_ref'
    chai1_motif_goal = 'allatom_aligned_allatom_rmsd_chai_motif'
    # chai1_motif_goal_rfflow = 'backbone_aligned_allatom_rmsd_chai_unideal_all'
    # chai1_motif_goal_rfd = 'backbone_aligned_allatom_rmsd_chai_ref_all'
    # chai1_motif_goal = 'backbone_aligned_allatom_rmsd_chai_motif'
    def normalized_chai_motif_goal(row):
        if row['is_rfd']:
            return row[chai1_motif_goal_rfd]
        else:
            return row[chai1_motif_goal_rfflow]

    chai_melt[chai1_motif_goal] = analyze.fast_apply(chai_melt, normalized_chai_motif_goal, ['is_rfd', chai1_motif_goal_rfflow, chai1_motif_goal_rfd])

    # Define the ligand-in-pocket RMSD goal metric
    chai1_largest_ligand_goal = 'chai_pocket_aligned_ligand_0_rmsd'
    chai1_all_ligands_goal = 'chai_pocket_aligned_rmsd_max'
    ligand_columns = columns_with_substring(chai_melt, 'chai_pocket_aligned_ligand_')
    ligand_columns = [c for c in chai_melt.columns if re.match(r'^chai_pocket_aligned_ligand_\d+_rmsd$', c)]
    max_ligands = len(ligand_columns)

    def f(row):
        return max(row[f'chai_pocket_aligned_ligand_{i}_rmsd'] for i in range(max_ligands))

    cols = [f'chai_pocket_aligned_ligand_{i}_rmsd' for i in range(max_ligands)]
    chai_melt[chai1_all_ligands_goal] = analyze.fast_apply(chai_melt, f, cols)

    print(chai_melt.info(memory_usage='deep'))

    index_check(chai_melt)
    set_seq_id(chai_melt)
    chai_melt = chai_melt.set_index(['seq_id', 'chai_model_idx'])

    fa_rmsd_cutoff = 1.5
    pocket_rmsd_cutoff = 2.5
    def chai_motif_pass(df):
        return df[chai1_motif_goal] < fa_rmsd_cutoff

    def chai_motif_pass_constellation(df):
        return df[chai1_motif_goal] < fa_rmsd_cutoff

    def no_clash(df, thresh=1.5):
        return df['ligand_dist_des_ncac_min'] > thresh

    def chai_motif_pass_and_no_clash(df, thresh=1.5):
        return df[chai_motif_pass.__name__] & df[no_clash.__name__]

    def chai_largest_ligand_pass(df):
        return df[chai1_largest_ligand_goal] < pocket_rmsd_cutoff

    def chai_all_ligands_pass(df):
        return df[chai1_all_ligands_goal] < pocket_rmsd_cutoff

    def chai_motif_and_largest_ligand_pass(df):
        return df[chai_motif_pass.__name__] & df[chai_largest_ligand_pass.__name__]

    def chai_motif_and_all_ligands_pass(df):
        return df[chai_motif_pass.__name__] & df[chai_all_ligands_pass.__name__]

    filters = []
    for filter in [
        chai_motif_pass,
        chai_motif_pass_constellation,
        no_clash,
        chai_motif_pass_and_no_clash,
        chai_largest_ligand_pass,
        chai_all_ligands_pass,
        chai_motif_and_largest_ligand_pass,
        chai_motif_and_all_ligands_pass,
    ]:
        chai_melt[filter.__name__] = filter(chai_melt)
        filters.append(filter.__name__)

    # Set a unique index.
    if not chai_melt.index.names == ['seq_id', 'chai_model_idx']:
        set_seq_id(chai_melt)
        chai_melt.set_index(['seq_id', 'chai_model_idx'], verify_integrity=True, inplace=True)

    return chai_melt, filters


def get_best_chai_model_per_design_per_filter(chai_melt, filters):
    # Find the number of designs passing each metric
    melt_vars = filters
    var_name='filter'
    value_name='pass'
    # variable_namer = {
    #                         'chai_motif_pass': 'AF2',
    #                         chai1_goal: 'Chai1'
    # }.__getitem__
    # id_col = 'seq_id'
    id_col = 'design_id'

    print(f'{chai_melt.shape=}')
    best, _ = get_best_in_group_for_each_metric(
        chai_melt,
        melt_vars,
        var_name=var_name,
        value_name=value_name,
        ascending=False,
        # variable_renamer=variable_namer,
    )

    n_unique = len(chai_melt[id_col].drop_duplicates()) # number of unique designs
    assert best.shape[0] == n_unique * len(melt_vars), f'{best.shape[0]=} != {n_unique * len(melt_vars)=} [{n_unique=}*{len(melt_vars)=}]'

    best = safe_merge(
        best,
        # df[['benchmark', 'is_rfd', 'n_contiguous_motif', 'source', 'seed', 'mpnn_index']],
        # df[['benchmark', 'is_rfd', 'n_contiguous_motif', 'n_motif', 'source', 'seed', 'mpnn_index']],
        chai_melt[['benchmark', 'is_rfd', 'n_contiguous_motif', 'n_motif', 'source', 'seed', 'mpnn_index']].reset_index().drop_duplicates('seq_id').set_index('seq_id').drop(columns='chai_model_idx'),
        how='left',
        left_on='seq_id',
        right_index=True,
        suffixes=(None, None),
        validate='m:1',
    )
    return best

def add_columns_to_best(best, chai_melt, columns):
    assert chai_melt.index.names == ['seq_id', 'chai_model_idx'], f"{chai_melt.index.names=} != ['seq_id', 'chai_model_idx']"
    return safe_merge(
        best,
        # df[['benchmark', 'is_rfd', 'n_contiguous_motif', 'source', 'seed', 'mpnn_index']],
        # df[['benchmark', 'is_rfd', 'n_contiguous_motif', 'n_motif', 'source', 'seed', 'mpnn_index']],
        chai_melt[columns].reset_index().drop_duplicates('seq_id').set_index('seq_id').drop(columns='chai_model_idx'),
        how='left',
        left_on='seq_id',
        right_index=True,
        suffixes=(None, None),
        validate='m:1',
    )

def default_metrics_melts(path_by_name):
    '''
    Parameters:
        path_by_name: dict mapping run name to path of compiled_metrics.csv
    '''

    df = load_data_multiprocess(path_by_name)
    chai_melt, filters = get_chai_melt_dataframe(df)
    best = get_best_chai_model_per_design_per_filter(chai_melt, filters)
    return chai_melt, best


def default_metrics_melts_assert_no_leak(path_by_name):
    '''
    Parameters:
        path_by_name: dict mapping run name to path of compiled_metrics.csv
    '''

    print_memory_usage()
    chai_melt, best = default_metrics_melts(path_by_name)
    print_memory_usage()
    chai_melt.info(memory_usage='deep')
    best.info(memory_usage='deep')
    return chai_melt, best
