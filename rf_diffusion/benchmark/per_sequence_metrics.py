#!/net/software/containers/users/dtischer/shebang_rf_se3_diffusion_dev.sh

import os
import sys
from collections import defaultdict

from functools import partial
import warnings

# Hack for autobenching
PKG_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SE3_DIR = os.path.join(PKG_DIR, 'lib/se3_flow_matching')
sys.path.append(SE3_DIR)

import pandas as pd
import fire
from icecream import ic
from tqdm import tqdm
import torch
import numpy as np
import scipy

from rf_diffusion import aa_model
from rf_diffusion import atomize
from rf_diffusion.dev import analyze
from rf_diffusion.inference import utils
import rf_diffusion.dev.analyze
import analysis.metrics
from rf_diffusion import aa_model
from rf_diffusion import atomize
from rf2aa.chemical import ChemicalData as ChemData
from rf_diffusion import bond_geometry
from rf_diffusion.dev import benchmark as bm
from rf_diffusion import loss

def main(pdb_names_file, outcsv=None, metric='default'):
    ic(__name__)
    thismodule = sys.modules[__name__]
    metric_f = getattr(thismodule, metric)

    with open(pdb_names_file, 'r') as fh:
        pdbs = [pdb.strip() for pdb in fh.readlines()]
    
    df = get_metrics(pdbs, metric_f)

    print(f'Outputting computed metrics to {outcsv}')
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv)

def get_metrics(pdbs, metric):
    records = []
    for pdb in tqdm(pdbs):
        record = metric(pdb)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df

def get_aligner(
        f, # from [L, 3]
        t, # to [L, 3]
        ):
    T = analyze.register_full_atom(f[:, None, :], t[:, None, :])
    def T_flat(f):
        f_aligned = T(f[:, None, :])
        return f_aligned[:, 0, :]

    return T_flat

def rmsd(V, W, eps=1e-6):
    assert V.ndim == 2, V.ndim
    assert W.ndim == 2, V.ndim
    L = V.shape[0]
    # dist = np.linalg.norm(V-W, axis=1)
    # ic( torch.sqrt(torch.sum((V-W)*(V-W), dim=1) / L + eps))
    # ic(torch.sum((V-W)*(V-W), dim=(1,2)).shape)
    return torch.sqrt(torch.sum((V-W)*(V-W), dim=(0,1)) / L + eps)

def catalytic_constraints_mpnn_packed(pdb):
    out = catalytic_constraints_inner(pdb, mpnn_packed=True)
    out = {f'catalytic_constraints.mpnn_packed.{k}': v for k,v in out.items()}
    return out

def catalytic_constraints(pdb):
    out = catalytic_constraints_inner(pdb, mpnn_packed=False)
    out = {f'catalytic_constraints.raw.{k}': v for k,v in out.items()}
    return out

def make_ligand_pids_unique(point_ids):
    new_pids = []
    ligand_pid_counts = defaultdict(int)
    for pid in point_ids:
        new_pid = pid
        if pid.startswith('L'):
            new_pid = f'{pid}_{ligand_pid_counts[pid]}'
            ligand_pid_counts[pid] += 1
        new_pids.append(new_pid)
    return new_pids


def catalytic_constraints_inner(pdb, mpnn_packed: bool):
    out = {}
    row = analyze.make_row_from_traj(pdb[:-4])
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']
    af2_pdb = analyze.get_af2(row)
    if mpnn_packed:
        des_pdb = analyze.get_design_pdb(row)
    else:
        des_pdb = analyze.get_diffusion_pdb(row)
    ref_pdb = analyze.get_input_pdb(row)

    indeps = {}
    indeps_a = {}
    atomizers = {}
    point_ids = {}
    ligand = row['inference.ligand']
    for name, pdb in [
        ('af2', af2_pdb),
        ('ref', ref_pdb),
        ('des', des_pdb),
    ]:
        if not os.path.exists(pdb):
            warnings.warn(f'{name} pdb: {pdb} for design {des_pdb} does not exist')
            return {}
        # f['name'] = utils.process_target(pdb, parse_hetatom=True, center=False)
        indeps[name] = aa_model.make_indep(pdb, ligand=None if name == 'af2' else ligand)
        is_atomized = ~indeps[name].is_sm
        atomization_state = aa_model.get_atomization_state(indeps[name])
        atomizers[name] = aa_model.AtomizeResidues(atomization_state, is_atomized)
        indeps_a[name] = atomizers[name].atomize(indeps[name])
        point_ids[name] = aa_model.get_point_ids(indeps_a[name], atomizers[name])
        point_ids[name] = make_ligand_pids_unique(point_ids[name])
    
    trb = analyze.get_trb(row)
    heavy_motif_atoms = {}
    for ref_idx0, (ref_chain, ref_idx_pdb) in zip(trb['con_ref_idx0'], trb['con_ref_pdb_idx']):
        aa = indeps['ref'].seq[ref_idx0]
        heavy_atom_names = aa_model.get_atom_names(aa)
        heavy_motif_atoms[f'{ref_chain}{ref_idx_pdb}'] = heavy_atom_names

    contig_atoms = row['contigmap.contig_atoms']
    if contig_atoms is not None:
        contig_atoms = eval(contig_atoms)
        contig_atoms = {k:v.split(',') for k,v in contig_atoms.items()}
    else:
        contig_atoms = heavy_motif_atoms
    
    def get_pids(name, *getters):
        pids = []
        for g in getters:
            pids.extend(g(point_ids[name]))
        return np.array(pids)
    
    def get_ii(name, *getters):
        pids = get_pids(name, *getters)
        i_by_pid = {pid: i for i, pid in enumerate(point_ids[name])}
        i_by_pid_v = np.vectorize(i_by_pid.__getitem__, otypes=[int])
        ii = i_by_pid_v(pids)
        return ii
    
    def xyz_by_id(name, ii):
        return indeps_a[name].xyz[ii, 1]
    
    def xyz(name, *getters):
        ii = get_ii(name, *getters)
        return xyz_by_id(name, ii)
    
    def zip_safe(*args):
        assert len(set(map(len, args))) == 1
        return zip(*args)

    def get_motif(_, ref: bool, contig_atoms=contig_atoms):
        pids = []
        idx0 = trb[f'con_{"ref" if ref else "hal"}_idx0']
        for (chain, ref_pdb_i), hal_i in zip_safe(
            trb['con_ref_pdb_idx'],
            idx0,
        ):
            atom_names = contig_atoms[f'{chain}{ref_pdb_i}']
            for a in atom_names:
                pids.append(f'A{hal_i}-{a}')
        return pids
    
    get_ref_motif = partial(get_motif, ref=True)
    get_des_motif = partial(get_motif, ref=False)
    
    def get_ligand(pids):
        return [pid for pid in pids if pid.startswith('L')]
    
    # 1. All-atom RMSD of scaffolded region, diffusion backbone to input motif < 1.0 Å
    # NOTE: Aligned on all heavy motif atoms, RMSD calculated on heavy motif atoms + ligand
    motif_des = xyz('des', get_des_motif)
    motif_ref = xyz('ref', get_ref_motif)
    T_motif_des_to_ref = get_aligner(motif_des, motif_ref)
    motif_and_ligand_des = xyz('des', get_des_motif, get_ligand)
    motif_and_ligand_ref = xyz('ref', get_ref_motif, get_ligand)
    ref_aligned_motif_and_ligand_des = T_motif_des_to_ref(motif_and_ligand_des)
    out['des_ref_motif_aligned_motif_ligand_rmsd'] = rmsd(ref_aligned_motif_and_ligand_des, motif_and_ligand_ref).item()
    out['criterion_1_metric'] = out['des_ref_motif_aligned_motif_ligand_rmsd']
    out['criterion_1_cutoff'] = 1.
    out['criterion_1'] = out['des_ref_motif_aligned_motif_ligand_rmsd'] < 1

    # 2. All-atom RMSD of scaffolded region, AF2 prediction to input motif < 1.0 Å
    motif_af2 = xyz('af2', get_des_motif)
    T = get_aligner(motif_des, motif_af2)
    af2_aligned_motif_des = T(motif_des)
    out['af2_des_motif_aligned_motif_rmsd'] = rmsd(af2_aligned_motif_des, motif_af2).item()
    out['criterion_2_metric'] = out['af2_des_motif_aligned_motif_rmsd']
    out['criterion_2_cutoff'] = 1.
    out['criterion_2'] = out['af2_des_motif_aligned_motif_rmsd'] < 1

    # 3. Cα RMSD of design model to AF2 prediction < 1.5 A
    def get_ca(pids):
        return [pid for pid in pids if pid.startswith('A') and pid.endswith('-CA')]
    def get_backbone(pids):
        return [pid for pid in pids if pid.startswith('A') and
                (pid.endswith('-CA') or pid.endswith('-C') or pid.endswith('-N') or pid.endswith('-O'))]
    
    ca_af2 = xyz('af2', get_ca)
    ca_des = xyz('des', get_ca)
    T = get_aligner(ca_des, ca_af2)
    af2_aligned_ca_des = T(ca_des)
    out['af2_des_ca_aligned_ca_rmsd'] = rmsd(af2_aligned_ca_des, ca_af2).item()
    out['criterion_3_metric'] = out['af2_des_ca_aligned_ca_rmsd']
    out['criterion_3_cutoff'] = 1.5
    out['criterion_3'] = out['af2_des_ca_aligned_ca_rmsd'] < 1.5

    # 4. No backbone-ligand clashes* in diffusion output
    # In diffusion model: Scaffolded region will be aligned (all atom) to input motif. If any backbone (N,CA,C)
    #    atom is < 2.0 Å from a ligand atom, the diffusion model is clashing.
    ligand_ref = xyz('ref', get_ligand)
    T = get_aligner(motif_ref, motif_des)
    des_aligned_ref_ligand = T(ligand_ref)
    backbone_des = xyz('des', get_backbone)

    ligand_bb_dist = scipy.spatial.distance.cdist(des_aligned_ref_ligand, backbone_des)
    out['des_ref_motif_aligned_ligand_bb_dist'] = ligand_bb_dist.min()
    out['criterion_4_metric'] = out['des_ref_motif_aligned_ligand_bb_dist']
    out['criterion_4_cutoff'] = 2.
    out['criterion_4'] = (ligand_bb_dist > 2).all()

    # 5. No backbone-ligand clashes* in AF2 prediction
    # In AF2 prediction: Design model will be aligned (Cα) to AF2 prediction. If any backbone atom is < 2.0 Å 
    #    from a ligand atom, the AF2 prediction is clashing.
    T = get_aligner(motif_des, motif_af2)
    ligand_des =xyz('des', get_ligand)
    af2_aligned_ligand_des = T(ligand_des)
    ligand_bb_dist = scipy.spatial.distance.cdist(af2_aligned_ligand_des, backbone_des)
    out['des_af2_motif_aligned_ligand_bb_dist'] = ligand_bb_dist.min()
    out['criterion_5_metric'] = out['des_af2_motif_aligned_ligand_bb_dist']
    out['criterion_5_cutoff'] = 2.
    out['criterion_5'] = (ligand_bb_dist > 2).all()

    # 6. AF2 mean plDDT > 0.7
    af2_npz = af2_pdb[:-4] + '.npz'
    af2_metrics = np.load(af2_npz)
    out['criterion_6_metric'] = np.mean(af2_metrics['plddt'])
    out['criterion_6_cutoff'] = 0.7
    out['criterion_6'] = np.mean(af2_metrics['plddt']) > 0.7

    # Extras
    # get_ref_motif_all_heavy = partial(get_motif, ref=True, contig_atoms=heavy_motif_atoms)
    get_des_motif_all_heavy = partial(get_motif, ref=False, contig_atoms=heavy_motif_atoms)
    motif_des_all_heavy = xyz('des', get_des_motif_all_heavy)
    motif_af2_all_heavy = xyz('af2', get_des_motif_all_heavy)
    T = get_aligner(motif_des_all_heavy, motif_af2_all_heavy)
    af2_aligned_motif_des_all_heavy = T(motif_des_all_heavy)
    out['af2_des_motif_aligned_motif_rmsd_all_heavy'] = rmsd(af2_aligned_motif_des_all_heavy, motif_af2_all_heavy).item()

    # # Testing
    # motif_des = xyz('des', get_des_motif)
    # motif_ref = xyz('ref', get_ref_motif)
    # T = get_aligner(motif_des, motif_ref)
    # motif_and_ligand_des = xyz('des', get_des_motif)
    # motif_and_ligand_ref = xyz('ref', get_ref_motif)
    # aligned_motif_and_ligand_des = T(motif_and_ligand_des)
    # out['des_ref_motif_aligned_motif_ligand_rmsd'] = rmsd(aligned_motif_and_ligand_des, motif_and_ligand_ref)
    # ic(pdb, des_pdb)
    # ic(out)
    # raise Exception('storp')
    return out

def default(pdb):
    record = {}
    row = analyze.make_row_from_traj(pdb[:-4])
    # ic(row['mpnn_index'])
    record['name'] = row['name']
    record['mpnn_index'] = row['mpnn_index']

    design_pdb = analyze.get_design_pdb(row)
    design_info = utils.process_target(design_pdb, center=False, parse_hetatom=True)
    des = design_info['xyz_27']
    mask = design_info['mask_27']
    des[~mask] = torch.nan
    dgram = torch.sqrt(torch.sum((des[None, None,:,:,:] - des[:,:,None,None, :]) ** 2, dim=-1))
    dgram = torch.nan_to_num(dgram, 999)

    # Ignore backbone-backbone distance, as ligandmpnn is not responsible for this.
    bb2bb = torch.full(dgram.shape, False)
    bb2bb[:, :4, :, :4] = True
    # ic(bb2bb.dtype)
    dgram[bb2bb] = 999

    dgram = dgram.min(dim=3)[0]
    dgram = dgram.min(dim=1)[0]
    # ic(dgram)
    # ic(dgram.shape)

    dgram.fill_diagonal_(999)
    min_dist = dgram.min()
    record['res_to_res_min_dist'] = min_dist.item()
    is_dist = torch.ones_like(dgram).bool()
    is_dist = torch.triu(is_dist, diagonal=1)
    dists = dgram[is_dist]
    clash_dist = 2 # roughly 2 VDW radii
    
    n_pair_clash = torch.sum(dists < clash_dist).item()
    record['n_pair_clash'] = n_pair_clash
    res_clash = (dgram < clash_dist).any(dim=-1)
    record['n_res_clash'] = res_clash.sum().item()
    record['fraction_res_clash'] = res_clash.float().mean().item()
    record['res_clash'] = res_clash.tolist()
    pair_clashes = (dgram < clash_dist)

    diffusion_pdb = analyze.get_diffusion_pdb(row)
    diffusion_info = utils.process_target(diffusion_pdb, center=False, parse_hetatom=True)
    diff = diffusion_info['xyz_27']
    diff[~mask] = torch.nan

    trb = analyze.get_trb(row)
    des_motif = des[trb['con_hal_idx0']]
    diff_motif = diff[trb['con_hal_idx0']]
    flat_des = des_motif[~des_motif.isnan().any(dim=-1)]
    flat_diff = diff_motif[~diff_motif.isnan().any(dim=-1)]

    mpnn_motif_dist = ((flat_des - flat_diff) ** 2).sum(dim=-1) ** 0.5
    record['motif_ideality_diff'] = mpnn_motif_dist.mean().item()
    return record



def rigid_loss(r):
    trb = rf_diffusion.dev.analyze.get_trb(r)
    
    def get_motif_indep(pdb, motif_i):
        indep = aa_model.make_indep(pdb)
        is_motif = aa_model.make_mask(motif_i, indep.length())
        indep_motif, _ = aa_model.slice_indep(indep, is_motif)
        return indep_motif
    
    
    indep_motif_native = get_motif_indep(r['inference.input_pdb'], trb['con_ref_idx0'])
    indep_motif_des = get_motif_indep(rf_diffusion.dev.analyze.get_design_pdb(r), trb['con_hal_idx0'])
    i_inv = torch.argsort(torch.tensor(trb['con_hal_idx0']))
    i_inv = torch.argsort(i_inv)
    aa_model.rearrange_indep(indep_motif_des, i_inv)
    
    # ic(
    #     indep_motif_des.seq,
    #     indep_motif_native.seq,
    # )
    
    is_atom_str_shown = {}
    for i in range(indep_motif_des.length()):
        res = indep_motif_des.seq[i]
        atom_names = [n.strip() for n in ChemData().aa2long[res][:14] if n is not None]
        is_atom_str_shown[i] = atom_names
    is_res_str_shown = torch.zeros((indep_motif_des.length(),)).bool()
    true_atomized, is_diffused, is_masked_seq, atomizer = atomize.atomize_and_mask(indep_motif_native, is_res_str_shown, is_atom_str_shown)
    pred_atomized, _, _, _                              = atomize.atomize_and_mask(indep_motif_des, is_res_str_shown, is_atom_str_shown)

    rigid_losses = bond_geometry.calc_rigid_loss(true_atomized, pred_atomized.xyz, is_diffused)

    return rigid_losses.get('motif_atom_determined', torch.tensor(float('nan'))).item()

def invert(d):
   return {v: k for k,v in d.items()}

def guidepost(pdb):
    row = analyze.make_row_from_traj(pdb[:-4])
    o = {}
    trb = analyze.get_trb(row)
    gp_by_motif = invert(trb['motif_by_gp'])
    motif_by_placed = {k:v for k,v in zip(
            trb['con_hal_idx0'],
            trb['con_hal_idx0_literal'],
    )}
    gp_i = []
    bb_i = []
    for bb_ii in trb['con_hal_idx0']:
        bb_i.append(bb_ii)
        gp_i.append(gp_by_motif[motif_by_placed[bb_ii]])

    gp_i = np.array(gp_i)
    bb_i = np.array(bb_i)
    gp_motif = trb['indep']['xyz'][gp_i]
    bb_motif = trb['indep']['xyz'][bb_i]
    ca_dist = np.linalg.norm(gp_motif[:, 1] - bb_motif[:, 1], axis=-1)
    o['ca_dist.max'] = np.max(ca_dist)
    o['ca_dist.min'] = np.min(ca_dist)
    o['ca_dist.mean'] = np.mean(ca_dist)
    o = {f'guidepost.{k}':v for k,v in o.items()}
    o['name'] = row['name']
    return o

def junction_bond_len(xyz, is_motif, idx):
    '''
        Args:
            xyz: [L, 14, 3] protein only xyz
            is_motif: [L] boolean motif mask
            idx: [L] pdb index
    '''
    sig_len=0.02
    sig_ang=0.05
    ideal_NC=1.329
    blen_CN  = loss.length(xyz[:-1,2], xyz[1:,0])
    CN_loss = torch.clamp( torch.abs(blen_CN - ideal_NC) - sig_len, min=0.0 )

    pairsum = is_motif[:-1].double() + is_motif[1:].double()
    pairsum[idx[:-1] - idx[1:] != -1] = -1

    junction = pairsum == 1
    intra_motif = pairsum == 2
    intra_diff = pairsum == 0

    return {
        'junction_CN_loss': CN_loss[junction].mean().item(),
        'intra_motif_CN_loss': CN_loss[intra_motif].mean().item(),
        'intra_diff_CN_loss': CN_loss[intra_diff].mean().item()
    }


def junction_cn(pdb):
    row = analyze.make_row_from_traj(pdb[:-4])
    trb = analyze.get_trb(row)
    des_pdb = analyze.get_diffusion_pdb(row)
    indep = aa_model.make_indep(des_pdb, row['inference.ligand'])
    is_motif = torch.zeros(indep.length()).bool()
    is_motif[trb['con_hal_idx0']] = True
    ic(is_motif)
    o = junction_bond_len(
        indep.xyz[~indep.is_sm],
        is_motif[~indep.is_sm],
        indep.idx[~indep.is_sm])
    o['name'] = row['name']
    return o

def backbone(pdb):
    record = {}
    row = rf_diffusion.dev.analyze.make_row_from_traj(pdb[:-4])
    record['name'] = row['name']

    traj_metrics = bm.get_inference_metrics_base(bm.get_trb_path(row), regenerate_cache=False)
    traj_t0_metrics = traj_metrics[traj_metrics.t==traj_metrics.t.min()]
    assert len(traj_t0_metrics) == 1
    traj_t0_metrics = traj_t0_metrics.iloc[0].to_dict()
    record.update(traj_t0_metrics)

    # Ligand distance
    if row['inference.ligand']:
        for af2, c_alpha in [
            (False, True),
            # (True, True),
            # (True, True)
        ]:
            dgram = rf_diffusion.dev.analyze.get_dist_to_ligand(row, af2=af2, c_alpha=c_alpha) # [P, L]
            maybe_af2 = 'af2' if af2 else 'des'
            maybe_c_alpha = 'c-alpha' if c_alpha else 'all-atom'
            record[f'ligand_dist_{maybe_af2}_{maybe_c_alpha}'] = dgram.min(-1)[0].tolist() # [P]
            record[f'ligand_dist_{maybe_af2}_{maybe_c_alpha}_min'] = dgram.min().item()

    # Secondary structure and radius of gyration
    record.update(analysis.metrics.calc_mdtraj_metrics(pdb))
    # Broken due to residue indexing.
    # record['rigid_loss'] = rigid_loss(row)
    return record

# For debugging, can be run like:
# python -m fire /home/ahern/projects/aa/rf_diffusion_flow/rf_diffusion/benchmark/per_sequence_metrics.py single --metric guidepost --pdb=/net/scratch/ahern/se3_diffusion/benchmarks/2023-12-18_20-48-06_cc_sh_schedule_sweep/run_siteD_troh1_cond1_0-atomized-bb-False.pdb
def single(metric, pdb, **kwargs):
    metric_f = globals()[metric]
    df = get_metrics([pdb], metric_f)
    ic(df)

if __name__ == '__main__':
    fire.Fire(main)
