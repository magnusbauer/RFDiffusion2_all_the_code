import glob
import math
import os
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

api = wandb.Api(timeout=150)

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

def melt_only(df, melt_vars):
    id_vars = df.columns
    id_vars = [v for v in id_vars if v not in melt_vars]
    return df.melt(id_vars)


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
        m=tree.map_structure(lambda x: x.item() if hasattr(x, 'cpu') else x, m)
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

def get_inference_metrics_inputs(trb_path):
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
            is_diffused=trb['is_diffused']
        )
        all_metrics_inputs.append(metrics_inputs)
    return (trb['config'], all_metrics_inputs)

def get_inference_metrics(trb_path,
                            metric_names=[
                            'atom_bonds_permutations',
                            'rigid_loss',
                            # 'rigid_loss_input',
                            'VarianceNormalizedPredTransMSE',
                            'VarianceNormalizedInputTransMSE',
                            'displacement_permutations'],
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
                                'atom_bonds_permutations',
                                'rigid_loss',
                                # 'rigid_loss_input',
                                'VarianceNormalizedPredTransMSE',
                                'VarianceNormalizedInputTransMSE',
                                'displacement_permutations'],
                                regenerate_cache=False):
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


def best_in_group(df, group_by=['design_id'], cols=['catalytic_constraints.raw.criterion_1'], ascending=[False], unique_column='seq_id'):
    df_small  = df[group_by + cols + [unique_column]]
    grouped = df_small.groupby(group_by).apply(lambda grp: grp.sort_values(cols, ascending=ascending).head(1))
    return pd.merge(df, grouped[unique_column], on=unique_column, how='inner')

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