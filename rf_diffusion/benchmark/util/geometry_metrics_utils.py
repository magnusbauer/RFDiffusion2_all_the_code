"""
Util functions for per_sequence_metrics.py and geometry

Author: JB
"""
import numpy as np
import torch
import itertools
import ast
import os

from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.dev import analyze
from rf_diffusion.dev.show_bench import get_last_px0
from rf_diffusion import loss

from rf_diffusion.benchmark import dihedral_calculation
from rf2aa.chemical import th_ang_v
from rf2aa.util_module import XYZConverter
xyz_converter = XYZConverter()

ideal_coords_by_aa = [
    torch.tensor([list(xyz) for (atom, _, xyz) in ChemData().ideal_coords[aa] if atom in ChemData().aa2long[aa][:ChemData().NHEAVYPROT]])
    for aa in range(ChemData().NPROTAAS)
]
ref_angles_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_angles.npy')
if os.path.exists(ref_angles_path):
    ref_angles = np.load(ref_angles_path)  # should be : 20, 20, 2 for dihedrals of an example aa. Useful since most angles are conserved. NB: chis are not, ALA + GLY are zeroed.
    ref_angles = torch.from_numpy(ref_angles)
else:
    print("WARNING: failed to load reference angles. File not found at", ref_angles_path)
    ref_angles = None

aa2long_stripped = [ [a.strip() if a is not None else None for a in aa] for aa in ChemData().aa2long]
aa2longalt_stripped = [ [a.strip() if a is not None else None for a in aa] for aa in ChemData().aa2longalt]

alignment_atoms_default={
    'ALA' : None,
    'ARG' : "NE,CZ,NH1", 
    'ASN' : "OD1,CG,ND2",
    'ASP' : "OD1,CG,OD2", 
    'CYS' : "CA,CB,SG",
    'GLN' : "NE2,CD,OE1",
    'GLU' : "CG,CD,OE1",
    'GLY' : None,
    'HIS' : "CB,CG,ND1,CD2,CE1,NE2",
    'ILE' : "CB,CG1,CD1",
    'LEU' : None,
    'LYS' : "CD,CE,NZ",
    'MET' : "CG,SD,CE",
    'PHE' : "CE2,CZ,CE1",
    'PRO' : "CB,CG,CD",
    'SER' : "CA,CB,OG",
    'THR' : "CB,OG1,CG2",
    'TRP' : "CZ2,CZ3,CH2",
    'TYR' : "CZ,OH",
    'VAL' : None,

}
alignment_atoms_post = {
    'TYR': 'CE2,CZ,OH',
}
DEFAULT_ROTAMER_PROB_THRESHOLD = -14.5
ROTAMER_PROB_THRESHOLDS = {
    'THR': -4.964161092981123,
    'SER': -4.976023922825927,
    'HIS': -6.813713459950055,
    'ARG': -12.52241880442645,
    'GLN': -9.175096415074679,
    'ASN': -5.571420605307308,
    'VAL': -13.673717051354595,
    'MET': -11.94377910612243,
    'ILE': -8.342745669087149,
    'LEU': -12.530387717684619,
    'GLU': -28.33234053897388,
    'ASP': -34.792404890156256,
    'TYR': -5.605853226806332,
    'LYS': -15.260760556925096,
    'TRP': -5.7102165539264105,
    'PHE': -6.43329732594977,
    'CYS': -2.9821400946080394
}

def rmsd(V, W, eps=1e-6):
    assert V.ndim == 2, V.ndim
    assert W.ndim == 2, V.ndim
    L = V.shape[0]
    return torch.sqrt(torch.sum((V-W)*(V-W), dim=(0,1)) / L + eps)
    
def np_kabsch(A,B):
    """
    Numpy version of kabsch algorithm. Superimposes B onto A

    Parameters:
        (A,B) np.array - shape (N,3) arrays of xyz crds of points


    Returns:
        rms - rmsd between A and B
        R - rotation matrix to superimpose B onto A
        rB - the rotated B coordinates
    """
    A = np.copy(A)
    B = np.copy(B)

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    def rmsd(V,W, eps=0):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return np.sqrt(np.sum((V-W)*(V-W), axis=(-2,-1)) / N + eps)

    N, ndim = A.shape

    # move to centroid
    A = A - centroid(A)
    B = B - centroid(B)

    # computation of the covariance matrix
    C = np.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U,S,Vt = np.linalg.svd(C)

    # ensure right handed coordinate system
    d = np.eye(3)
    d[-1,-1] = np.sign(np.linalg.det(Vt.T@U.T))

    # construct rotation matrix
    R = Vt.T@d@U.T

    # get rotated coords
    rB = B@R

    # calculate rmsd
    rms = rmsd(A,rB)

    return rms, rB, R

def pad_coords_to(n, coords):
    """Bads the atom dim of coords to n
    example: N, 14, 3 -> N, 20, 3 (n=20)
    """
    if n <= coords.shape[-2]: return coords
    return torch.cat([coords, torch.zeros(coords.shape[0], n - coords.shape[-2], 3)], dim=-2)

def get_reference_xyz(seq):
    """
    seq: B,

    NB: Does NOT produce meaningful coordinates, just the base frame (first 4 atoms) have valid relative positions

    returns: B, 14, 3
    """
    stack = [pad_coords_to(ChemData().NHEAVYPROT, ideal_coords_by_aa[a][None, : , :]) for a in seq]
    return torch.concatenate(stack, dim=0)

def xyz_to_angles(xyz, seq):
    """
    xyz: B, 14, 3
    seq: B,

    angles: B, 20, 2
    """
    xyz = pad_coords_to(ChemData().NTOTAL, xyz)
    torsions, _, _, _ = xyz_converter.get_torsions(
        xyz[:,None],
        seq[:,None]
    )
    torsions = torsions.squeeze(1)
    return torsions # N, 20, 2

def angles_to_xyz(angles, seq):
    """
    angles: B, 20, 2
    seq: B,

    returns: B, 14, 3
    """
    xyz_ref = get_reference_xyz(seq)
    xyz_ref = pad_coords_to(ChemData().NTOTAL, xyz_ref)
    
    _, xyz = xyz_converter.compute_all_atom(
        seq[:,None],
        xyz_ref[:,None],
        angles[:,None]
    )
    xyz = xyz[...,0,:14,:]
    return xyz

def dih_to_deg(dih):
    return (torch.rad2deg(torch.atan2(dih[..., 1], dih[..., 0])))
def deg_to_dih(deg):
    return torch.stack([torch.cos(torch.deg2rad(deg)), torch.sin(torch.deg2rad(deg))], dim=-1)

def chis_to_xyz(chis, seq):
    """
    chis: B, n_chi, 2
    seq: B,
    
    returns: B, 14, 3
    """
    n_chi = chis.shape[1]
    # if chis.shape[-2] != 4:
    #     chis = pad_coords_to(4, chis)
    angles = ref_angles[seq]
    angles[:, 3:(3+n_chi)] = chis
    xyz = angles_to_xyz(angles, seq)
    return xyz

def resolve_symmetry(coords_ref, coords_target, seq):
    """
    Resolve symmetry between coords_ref and coords_target, returns fixed coords_ref
    coords_ref: (batch, 14, 3)
    coords_target: (batch, 14, 3)
    seq: (batch,)
    
    returns: (batch, 14, 3)
    """
    coords_ref = pad_coords_to(ChemData().NTOTAL, coords_ref)
    coords_target = pad_coords_to(ChemData().NTOTAL, coords_target)
    assert coords_ref.shape == coords_target.shape, f'{coords_ref.shape} != {coords_target.shape}'

    def make_alternate_xyz_indexes():
        n_tokens = len(ChemData().aa2long)
        n_atoms = len(ChemData().aa2long[0])
        out = torch.zeros((n_tokens, n_atoms), dtype=int)
        for i, atom_names in enumerate(ChemData().aa2long[:ChemData().NPROTAAS]):
            for j, atom_name in enumerate(atom_names):
                out[i, j] = ChemData().aa2longalt[i].index(atom_name)
        return out

    alternate_xyz_indexes = make_alternate_xyz_indexes()

    def get_alternate_xyz(xyz, seq):
        return xyz[torch.arange(xyz.size(0)).unsqueeze(1), alternate_xyz_indexes[seq]]
    
    coords_ref_alt = get_alternate_xyz(coords_ref, seq)

    # sym_agrees = [[ChemData().aa2long[a][i] == ChemData().aa2longalt[a][i] for i in
    #                 range(14)] for a in range(21)]
    # sym_agrees = torch.tensor(sym_agrees)
    # has_alt = ~sym_agrees.all(dim=-1)

    atm_mask = ChemData().allatom_mask[seq]
    atm_mask[:, 14:] = False
    distances_true_to_pred = (coords_target - coords_ref)**2
    distances_alt_to_pred = (coords_target - coords_ref_alt)**2
    distances_true_to_pred[~atm_mask] = 0.
    distances_alt_to_pred[~atm_mask] = 0.

    assert not distances_true_to_pred[0].isnan().any()
    assert not distances_true_to_pred.isnan().any()

    distance_scores_true_to_pred = torch.sum(distances_true_to_pred, dim=(1, 2))
    distance_scores_alt_to_pred = torch.sum(distances_alt_to_pred, dim=(1, 2))

    is_better_alt = distance_scores_alt_to_pred < distance_scores_true_to_pred
    is_better_alt_crds = is_better_alt[:, None, None].repeat(1, ChemData().NTOTAL, 3)

    symmetry_resolved_true_crds = torch.where(is_better_alt_crds, coords_ref_alt, coords_ref)
    symmetry_resolved_true_crds[...,~atm_mask, :] = 0
    symmetry_resolved_true_crds = symmetry_resolved_true_crds[...,:14,:]
    return symmetry_resolved_true_crds

sym_agrees = torch.tensor([[ChemData().aa2long[a][i] == ChemData().aa2longalt[a][i] for i in
                range(14)] for a in range(21)])  # 21, 14 boolean array of whether each atom is symmetric
has_alt = ~sym_agrees.all(dim=-1)  # 21 boolean array of whether any atom is symmetric
n_to_permutation_idxs = lambda n : list(itertools.permutations(list(range(n)))) # 2 -> [(0,1), (1,0)] etc.
def multiply_symmetric_xyz_stack(xyz, a):
    # xyz: L, 14, 3
    # a: atom type to multiply the stack as (must be the same for all L)
    # returns: L*n_perms, 14, 3
    L = xyz.shape[0]
    has_alt = ~sym_agrees[a].all(dim=-1)
    if not has_alt.any():
        return xyz

    _, sym_idx = torch.where(~sym_agrees[a])
    n_permuting_atoms = sym_idx.shape[0]  # 2, 2 & 24, 4 etc.
    perms = torch.tensor(n_to_permutation_idxs(n_permuting_atoms))
    sym_perm_idx = sym_idx[perms]
    n_perms = perms.shape[0]
    assert sym_idx.shape[0] == n_permuting_atoms

    xyz_multiplied = xyz[:,None].repeat(1, n_perms, 1, 1)  # L, n_perms, 14, 3
    xyz_multiplied[:, :, sym_idx, :] =  xyz[:, sym_perm_idx, :]  # picks: L, n_perms, n_perm_at, 3 <- L, n_perms, n_perm_at, 3
    return xyz_multiplied.reshape(L*n_perms, 14, 3)

def align_rotamer_to_tip(xyz_ref, mask, xyz_rot, tip_idxs=None):
    # Aligns stack of rotamer coordinates to reference xyz along tip indices
    # ref_xyz: 14, 3
    # mask: 14 (bool)
    # rot_xyz: B, 14, 3
    # tip_idxs: list of atom indices to align to. Default: None (will use full atom repr.)
    # Returns: B, 14, 3
    if tip_idxs is None: 
        tip_idxs = [i for i in range(mask.sum(-1))]
    N_rotamers = xyz_rot.shape[0]
    xyz_rot_aligned = torch.zeros_like(xyz_rot)
    xyz_rot_tip = xyz_rot[..., tip_idxs, :]  # B, 3, 3
    xyz_ref_tip = xyz_ref[..., tip_idxs, :]  # 3, 3
    Rs = []
    for i in range(N_rotamers):
        rmsd_, _, R = np_kabsch(xyz_ref_tip, xyz_rot_tip[i]) # tgt, mobile
        # if rmsd_ > 5e-1:
        #     print(f"WARNING: found high rmsd - {rmsd_} - for tips: {tip_idxs}")
        Rs.append(R.tolist())
    Rs = torch.tensor(Rs).float()  # (N_rotamers, 3, 3)

    centroid = lambda X: X.mean(axis=-2)[...,None,:]
    xyz_rot_aligned = xyz_rot - centroid(xyz_rot_tip)  # subtract centroid from self
    xyz_rot_aligned = xyz_rot_aligned @ Rs # rotate
    xyz_rot_aligned = xyz_rot_aligned + centroid(xyz_ref_tip) # add centroid to other
    xyz_rot_aligned[:,~mask] = 0

    return xyz_rot_aligned

def select_rotamer_from_aligned_set(src_xyz, src_mask, rotamer_xyz, a, target='allatom', prune=False, verbose=False):
    """
    Finds the best rotamer from a set of stacked rotamers according to rmsd (backbone)

    Args:
        src_xyz: 14, 3
        src_mask: 14,
        rotamer_xyz: B, 14, 3  # the rotamers to choose from
        a: B  # the amino acid (Int)
        target: str  # whether to minimize the allatom or backbone atom rmsd
        prune: bool  # whether to prune rotamers with bad angles
        verbose: bool  # whether to print out information about selection
    """
    angle_threshold=17.5
    aa = ChemData().num2aa[a]
    
    rmsd_mask = src_mask
    rmsd_mask[5:] = False   # backbone rmsd

    # Calculate rmsds
    rmsds = np.linalg.norm(rotamer_xyz[:, rmsd_mask] - src_xyz[None, rmsd_mask], axis=(1,2))
    allatom_rmsds = np.linalg.norm(rotamer_xyz[:, src_mask] - src_xyz[None, src_mask], axis=(1,2))
    
    # Reorder sets by increasing RMSD
    if target == 'allatom':
        idxs = np.argsort(allatom_rmsds)
    elif target == 'bb':
        idxs = np.argsort(rmsds)
    else:
        raise ValueError(f"Invalid target: {target}")

    rotamer_xyz = rotamer_xyz[idxs]
    rmsds = rmsds[idxs]
    allatom_rmsds = allatom_rmsds[idxs]

    if prune:
        angle_orig, good_angle_orig = get_angle_dev_single(a, src_xyz, cut=angle_threshold)

    for j in range(rotamer_xyz.shape[0]):
        rotamer_xyz_ideal = rotamer_xyz[j]
        rmsd = rmsds[j]
        allatom_rmsd = allatom_rmsds[j]
        
        if prune:
            prob = dihedral_calculation.get_rotamer_probability(rotamer_xyz_ideal, aa, phi=60, psi=60)
            score = float(np.log10(np.array(prob)) / ROTAMER_PROB_THRESHOLDS[aa])
            angle, good_angle = get_angle_dev_single(a, rotamer_xyz_ideal, cut=angle_threshold)
            if j > 0 or verbose:
                print(f'Selected {aa} rotamer | {j} of {rotamer_xyz.shape[0]} | RMSD: {rmsds[j]:.2f} | Score: {score:.2f} | Angle: {angle:.2f} from {angle_orig:.2f}') if verbose else None
                break
            else:
                print(f'Ignoring {aa} rotamer | {j} of {rotamer_xyz.shape[0]} | RMSD: {rmsds[j]:.2f} | Score: {score:.2f} | Angle: {angle:.2f} from {angle_orig:.2f}') if verbose else None
        else:
            break
    else:
        if verbose:
            print(f'WARNING: Failed to find good rotamer for {aa}')
        rotamer_xyz_ideal = rotamer_xyz[0]
        rmsd = rmsds[0]
        allatom_rmsd = allatom_rmsds[0]
    if (rmsd) > 3 and verbose:
        print(f"WARNING: very high RMSD {rmsd} for {aa}")
    return rotamer_xyz_ideal, rmsd, allatom_rmsd


def find_ideal_irots(
    parsed, 
    motif_idxs, 
    return_stack=False, 
    skip_good_residues=True, 
    verbose=False, 
    n_samples=2, 
    atomize_indices2atomname={}, 
    return_as_metric=False,
    target='allatom'
):
    """
    Generates idealised residues (aligned to the tip atom of motif) based on the parsed pdb.

    Args:
        parsed: dict of parsed pdb
        motif_idxs: indices of residues to idealise (0-indexed)
        return_stack: whether to return the stack of rotamers
        atomize_indices2atomname: dictionary of motif indices to ensure alignment to upon idealization. Default runs 1 alignment to default tips. For residues like TYR an additional one is run to ensure the correct alignment
            key: hal_i
            value: list of atom names to align to
        return_as_metric: whether to return as a metric dict (better for parsing into dataframes) or as idealized xyz
        target: whether to minimize the allatom or backbone atom rmsd

    Returns: parsed, rmsds(, xyz_stack)
    """
    assert target in ['allatom', 'bb'], f"Invalid target: {target}"
    angle_threshold = 10
    motif_metadata_dict = {'xyz_stack': []}
    default_result = lambda : {'idealized_xyz': None, 'allatom_rmsd': 0, 'bb_rmsd': 0, 'has_changed': False}

    for i, (a, hal_i) in enumerate((zip(parsed['seq'][motif_idxs], motif_idxs))):
        hal_i = int(hal_i)
        aa = ChemData().num2aa[a]
        src_xyz = parsed['xyz'][hal_i][:14]
        src_mask = parsed['mask'][hal_i][:14]

        if alignment_atoms_default.get(aa) is None:
            motif_metadata_dict[hal_i] = default_result()
            continue
        
        # 3) Get phi-psi angles for backbone-dependent sampling of rotamer library
        if hal_i == 0 or hal_i == len(parsed['xyz']) - 1:
            phi, psi = -40, -60
        else:
            phi, psi = dihedral_calculation.calculate_phi_psi(
                prev_residue_xyz=parsed['xyz'][hal_i - 1][:14], 
                curr_residue_xyz=parsed['xyz'][hal_i    ][:14], 
                next_residue_xyz=parsed['xyz'][hal_i + 1][:14],
            )
        
        # 3.5) Check if good residue - skip if so
        if skip_good_residues:
            # angle check (fast)
            _, good_angle = get_angle_dev_single(a, src_xyz, cut=angle_threshold)
            if not good_angle:
                # probability check (slower)
                probs = dihedral_calculation.get_rotamer_probability(src_xyz, aa, phi=phi, psi=psi)
                if probs is None or aa not in ROTAMER_PROB_THRESHOLDS:
                    motif_metadata_dict[hal_i] = default_result()
                    continue
                probs = np.array([probs])
                high_prob = (np.log10(probs) / ROTAMER_PROB_THRESHOLDS[aa] < 1.0)
                if sum(high_prob):
                    print(f"Skipping residue {aa} as it is already in a high-probability conformation") if verbose else None
                    motif_metadata_dict[hal_i] = default_result()
                    continue

        # 4) Sample high probability rotamer chi angles (based on bb)
        flexible_residues = ['ARG', 'GLN', 'GLU', 'LYS', 'MET']
        chis = dihedral_calculation.sample_bbdep_rotamers(src_xyz, aa, phi, psi, n_samples = n_samples * (int(aa in flexible_residues)+1))
        if chis is None or alignment_atoms_default[aa] is None:
            rmsds_all.append(0)
            print('Skipping residue', aa) if verbose else None
            motif_metadata_dict[hal_i] = default_result()
            continue
        chis=deg_to_dih(torch.tensor(chis))
        rotamer_xyz = chis_to_xyz(chis, torch.tensor(parsed['seq'][hal_i]).repeat(chis.shape[0]))
        rotamer_xyz = multiply_symmetric_xyz_stack(rotamer_xyz, torch.tensor([int(a),]))

        # 6) Align to tip atoms
        tip_atoms = alignment_atoms_default[aa].split(',')
        tip_idxs = [aa2long_stripped[int(a)].index(atom) for atom in tip_atoms]
        tip_idxs = torch.tensor(tip_idxs).to(int)
        rotamer_xyz = align_rotamer_to_tip(src_xyz, src_mask, rotamer_xyz, tip_idxs)

        # 6.5) Post-align to guideposts or fine-tuned atoms
        for fine_atoms in [alignment_atoms_post.get(aa), atomize_indices2atomname.get(hal_i)]:
            if fine_atoms is None: continue
            if isinstance(fine_atoms, str): fine_atoms = fine_atoms.split(',')
            tip_idxs = [aa2long_stripped[int(a)].index(atom.strip()) for atom in fine_atoms]
            tip_idxs = torch.tensor(tip_idxs).to(int)
            rotamer_xyz = align_rotamer_to_tip(src_xyz, src_mask, rotamer_xyz, tip_idxs)

        # 7) Use RMSD to find ideal rotamer
        rotamer_xyz_ideal, rmsd, allatom_rmsd = select_rotamer_from_aligned_set(src_xyz, src_mask, rotamer_xyz, a, target=target, verbose=verbose)

        # 8) Update parsed coordinates with new rotamers
        parsed['xyz'][hal_i][:14] = rotamer_xyz_ideal[..., :14, :].numpy()

        motif_metadata_dict[hal_i] = {
            'idealized_xyz': rotamer_xyz_ideal,
            'allatom_rmsd': allatom_rmsd,
            'bb_rmsd': rmsd,
            'has_changed': True,
        }
        motif_metadata_dict['xyz_stack'].append((rotamer_xyz if return_stack else None))

    if sum([d['has_changed'] for k, d in motif_metadata_dict.items() if k != 'xyz_stack']) == 0:
        print('No residues idealised.')

    if return_as_metric:
        motif_metadata_dict = {k: v for k, v in motif_metadata_dict.items() if k != 'xyz_stack'}
        o = {  # allatom for all HA atoms in residue, target for the rmsd the rotamer idealization is minimzed towards (default: backbone atom rmsds)
            'idealization_rmsd_allatom': [float(d['allatom_rmsd']) for k, d in motif_metadata_dict.items()],
            'idealization_rmsd_bb': [float(d['bb_rmsd']) for k, d in motif_metadata_dict.items()],
        }
        o['idealization_rmsd_allatom_mean'] = np.mean(o['idealization_rmsd_allatom'])
        o['idealization_rmsd_allatom_max']  = np.max(o['idealization_rmsd_allatom'])
        o['idealization_rmsd_bb_min']   = np.min(o['idealization_rmsd_bb'])
        o['idealization_rmsd_bb_mean']  = np.mean(o['idealization_rmsd_bb'])
        o['idealization_rmsd_bb_max']   = np.max(o['idealization_rmsd_bb'])
        o['n_residues_to_idealize'] = sum([d['has_changed'] for k, d in motif_metadata_dict.items() if k != 'xyz_stack'])
        return o
    else:
        if not return_stack:
            motif_metadata_dict = {k: v for k, v in motif_metadata_dict.items() if k != 'xyz_stack'}
        return parsed, motif_metadata_dict

def idealise_design_tipatoms(pdb, pdb_out=None, motif_idxs=None, assert_tip_atoms=False):
    """
    Prepares a design pdb for renoising by aligning high-probability rotamers to tip atoms.
    """

    row = analyze.make_row_from_traj(pdb)
    trb = analyze.get_trb(row)
    motif_idxs = trb['con_hal_idx0'] if motif_idxs is None else motif_idxs
    config = trb['config']

    # Parse pdb
    indep = aa_model.make_indep(pdb, ligand=config['inference']['ligand'])
    parsed = parse_pdb_lines_target(open(pdb, 'r').readlines(), parse_hetatom=True)
    
    # Should you include a second alignment step to assert the tip atoms stay the same?
    gps = trb.get('atomize_indices2atomname', {}) if assert_tip_atoms else {}
    feats_ideal, metadata = find_ideal_irots(parsed, motif_idxs, atomize_indices2atomname=gps, target='bb')
    changed_mask = np.array([metadata[k]['has_changed'] for k in metadata.keys() if k != 'xyz_stack'])

    motif_idxs = torch.tensor(motif_idxs).to(int)
    changed_mask = torch.tensor(changed_mask).to(bool)

    # Ensure the ligand is written by going via indep
    indep.xyz[motif_idxs[changed_mask], :ChemData().NHEAVY] = torch.tensor(feats_ideal['xyz'][motif_idxs[changed_mask]])

    if pdb_out is not None:
        indep.write_pdb(pdb_out, lig_name=config['inference']['ligand'])
    
    return indep, metadata, changed_mask

def bonds_list_to_triads(bonds):
    triads = []
    first, second = itertools.tee(bonds)
    next(second, None)
    for (a1, a2), (b1, b2) in zip(first, second):
        if a2 == b1:
            triads.append((a1, a2, b2))
    return triads

def get_res_angles(xyz, aa):
    # xyz: N, 3

    xyz = torch.tensor(xyz) if not isinstance(xyz, torch.Tensor) else xyz
    bonds = ChemData().aabonds[aa]
    HAs = ChemData().aa2long[aa][:ChemData().NHEAVY]
    BBAs = np.array(ChemData().aa2long[aa])[[0, 2, 3]] # backbone atoms (except CA)
    UpstreamAs = [ at for at in HAs if at not in BBAs]
    bonds = [
        (src, tgt) for (src, tgt) in bonds if 
        (src in UpstreamAs and tgt in UpstreamAs)
    ]
    triads = bonds_list_to_triads(bonds)

    angles = []
    for (a, b, c) in triads:
        ab = (xyz[HAs.index(b)] - xyz[HAs.index(a)])
        bc =-(xyz[HAs.index(c)] - xyz[HAs.index(b)])
        dih = th_ang_v(ab, bc)
        angle = float(torch.rad2deg(torch.atan2(dih[..., 1], dih[..., 0])))
        # print(a, b, c, angle)
        angles.append(angle)
    return angles


def get_angle_deviations(parsed, motif_idx, return_dict=True):
    motif_xyz = parsed['xyz'][motif_idx]
    motif_seq = parsed['seq'][motif_idx]

    motif_xyz = torch.tensor(motif_xyz) if not isinstance(motif_xyz, torch.Tensor) else motif_xyz
    motif_seq = torch.tensor(motif_seq) if not isinstance(motif_seq, torch.Tensor) else motif_seq

    motif_xyz = motif_xyz[None] if len(motif_xyz.shape) == 2 else motif_xyz

    angles_all=[]
    deviations_all=[]
    for i in range(len(motif_idx)):
        angles = get_res_angles(
            motif_xyz[i,:,:], int(motif_seq[i])
        )        
        aa = ChemData().num2aa[int(motif_seq[i])]
        allowed_angles = [109.5, 120] if aa != 'MET' else [100., 109.5, 120]  # also allow a degree of bending due to sulfur
        deviations = np.abs(
            np.array(angles)[...,None] % 180 - np.array(allowed_angles)[None] # deviation from either 109.5 or 120
        ).min(-1).tolist()
        angles_all.append(angles)
        deviations_all.append(deviations)
    if not return_dict:
        return angles_all, deviations_all
    o = {}
    o['max_angle_deviations'] = [max(ds or [0]) for ds in deviations_all] # maximum angle violation per residue
    o['total_angle_deviation'] = sum([sum(ds) for ds in deviations_all]) # total angle violation
    return o

#######################################################################
# Inner functions for evaluation; are same as per_sequence_metrics.py 
# will need preprocessing of the input pdb
#######################################################################

def get_angle_dev_single(a, xyz, cut=10):
    o = get_angle_deviations({'xyz': xyz[None], 'seq': torch.tensor([a])}, torch.tensor([0]))
    angle = o['total_angle_deviation']
    good_angle = angle < cut
    return angle, good_angle

def clashes_inner(parsed, is_diffusion_output=True, clash_dist=1.3):
    o = {}

    des = torch.tensor(parsed['xyz'])
    mask = torch.tensor(parsed['mask'])
    des[~mask] = torch.nan
    dgram = torch.sqrt(torch.sum((des[None, None,:,:,:] - des[:,:,None,None, :]) ** 2, dim=-1))
    dgram = torch.nan_to_num(dgram, 999)

    # Ignore backbone-backbone distance, as ligandmpnn is not responsible for this.
    bb2bb = torch.full(dgram.shape, False)
    bb2bb[:, :4, :, :4] = True
    if is_diffusion_output: bb2bb[:, 4, :, 4] = True # Cb-Cb distances may be mpnn'd out later
    dgram[bb2bb] = 999

    dgram = dgram.min(dim=3)[0]
    dgram = dgram.min(dim=1)[0]
    dgram.fill_diagonal_(999)
    min_dist = dgram.min()
    o['res_to_res_min_dist'] = min_dist.item()
    is_dist = torch.ones_like(dgram).bool()
    is_dist = torch.triu(is_dist, diagonal=1)
    dists = dgram[is_dist]
    
    n_pair_clash = torch.sum(dists < clash_dist).item()
    o['n_pair_clash'] = n_pair_clash
    res_clash = (dgram < clash_dist).any(dim=-1)
    o['n_res_clash'] = res_clash.sum().item()
    o['fraction_res_clash'] = res_clash.float().mean().item()
    o['res_clash'] = res_clash.tolist()
    
    return o

def ca_cb_deviations(parsed, motif_idxs):
    xyz = torch.tensor(parsed['xyz'][motif_idxs])
    is_gly = torch.tensor([parsed['seq'][motif_idxs] == 4]).to(bool).view(xyz.shape[0])

    # # three anchor atoms
    N  = xyz[...,0,:]
    Ca = xyz[...,1,:]
    C  = xyz[...,2,:]

    # # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # print(Cb.shape, xyz.shape) # torch.Size([3, 3]) torch.Size([3, 23, 3])
    dists = torch.norm(xyz[..., 4, :] - Cb, dim=-1)
    dists[is_gly] = 0
    
    o = {'ca_cb_deviations': dists.tolist()}
    return o


def get_tyr_dihedral(
    xyz
):
    xyz = torch.tensor(xyz)
    torsion_indices = torch.tensor([4, 5, 10, 11])  # (" CB "," CG "," CZ "," OH ")
    atom_xyz = xyz[:, torsion_indices]
    dih = th_ang_v(
        atom_xyz[..., 1, :] - atom_xyz[..., 0, :], # CG <- CB
        -(atom_xyz[..., 2, :] - atom_xyz[..., 1, :]), # CZ <- CG
    )
    angle_radians_in = torch.atan2(dih[..., 1], dih[..., 0])
    
    dih = th_ang_v(
        atom_xyz[..., 2, :] - atom_xyz[..., 3, :], # CZ <- OH
        -(atom_xyz[..., 1, :] - atom_xyz[..., 2, :]), # CG <- CZ
    )
    angle_radians_out = torch.atan2(dih[..., 1], dih[..., 0])
    
    angle_in = torch.rad2deg(angle_radians_in).numpy()
    angle_out = torch.rad2deg(angle_radians_out).numpy()

    return angle_in, angle_out

def tyr_angle_evaluation(
    parsed,
    motif_idxs,
    MAX_TYR_TORSION = 22.5, # degrees from ideal, 20 is not much, 25 is looser
):
    torsions = []
    distortions = []
    for i, motif_idx in enumerate(motif_idxs):
        aa = parsed['seq'][motif_idx]
        aa = ChemData().num2aa[aa]
        if aa not in ['TYR']:
            continue
        
        ang_in, ang_out = get_tyr_dihedral(
            parsed['xyz'][motif_idx][None]
        )

        distortion = np.abs(180 - np.array([ang_in, ang_out])).sum() # Combined angle from ideal
        torsions.append([float(ang_in), float(ang_out)])
        distortions.append(float(distortion))

    has_tyrosines = bool(len(torsions) > 0)
    
    o = {}
    o['has_tyrosines'] = has_tyrosines
    o['torsions'] = torsions  # angles between ring and ingoing / outgoing atoms (CB or OH)
    o['total_distortion'] = sum(distortions) if has_tyrosines else 0
    o['num_contorted'] = sum([(
        dist > MAX_TYR_TORSION
    ) for dist in distortions]) if has_tyrosines else 0
    o['has_contortion'] = bool(o['num_contorted'] > 0) if has_tyrosines else False
    o = {f'tyr.{k}':v for k,v in o.items() if k not in ['name']}
    return o

def get_motif_rotamer_probability_inner(parsed, motif_idxs, return_all=False):
    o = {}

    xyz = torch.tensor(parsed['xyz']) # L, 14, 3
    pdb_idxs = parsed['pdb_idx'] # L,

    scores=[]
    for i, hal_i in enumerate(motif_idxs):
        if hal_i == 0 or hal_i == xyz.shape[0] - 1:
            continue  # Skip first and last residues as they don't have both neighbors
        prev_residue_xyz = xyz[hal_i - 1][:14]
        curr_residue_xyz = xyz[hal_i    ][:14]
        next_residue_xyz = xyz[hal_i + 1][:14]
        phi, psi = dihedral_calculation.calculate_phi_psi(prev_residue_xyz, curr_residue_xyz, next_residue_xyz)
        aaa = ChemData().num2aa[parsed['seq'][hal_i]]

        amplitude = dihedral_calculation.get_rotamer_probability(curr_residue_xyz, aaa, phi, psi)

        if isinstance(amplitude, tuple) or amplitude is None:
            amplitude = 1.0
        score = float(np.log10(amplitude) / ROTAMER_PROB_THRESHOLDS.get(aaa, DEFAULT_ROTAMER_PROB_THRESHOLD))
        
        idx = "".join([str(ii) for ii in pdb_idxs[hal_i]])
        o[f'rot_prob_{aaa}_{idx}'] = amplitude
        o[f'rot_prob_score_{aaa}_{idx}'] = score
        scores.append(score)

        if return_all:
            o[f'phi_{hal_i}'] = phi
            o[f'psi_{hal_i}'] = psi
            o[f'{hal_i}'] = aaa
    o['max_rot_prob_score'] = max(scores) if scores else 0
    o['mean_rot_prob_score'] = sum(scores) / len(scores) if scores else 0
    return o

def chainbreaks(parsed):
    xyz = torch.tensor(parsed['xyz'])[:,:14,:3]
    xyz = xyz[...,1,:] # L, 3
    ca_dists = torch.norm(xyz[1:] - xyz[:-1], dim=-1)
    # breaks = ca_dists > 4.1
    deviation = torch.abs(ca_dists - 3.8)

    return {
        'ca_dists': ca_dists.tolist(),
        # 'num_chainbreaks': float(breaks.sum()),
        'avg_caca_dist': float(ca_dists.mean()),
        'mean_caca_deviation': float(deviation.mean()),
        'max_caca_deviation': float(deviation.max()),
    }


def junction_bond_len(parsed, motif_idxs):
    xyz = torch.tensor(parsed['xyz'])   # L, 14, 3
    idx = torch.tensor(parsed['idx'])  # L, 14
    
    motif_mask = torch.zeros(xyz.shape[0]).bool() # L,
    motif_mask[motif_idxs] = True

    L = len(idx)
    assert xyz.shape[0] == L, f'{xyz.shape}[0] != {L}'
    assert motif_mask.shape[0] == L, f'{motif_mask.shape}[0] != {L}'
    
    return junction_bond_len_inner(xyz, motif_mask, idx)

def junction_bond_len_inner(xyz, is_motif, idx):
    '''
    Args:
        xyz: [L, 14, 3] protein only xyz
        is_motif: [L] boolean motif mask
        idx: [L] pdb index
    '''
    sig_len=0.02
    ideal_NC=1.329
    blen_CN  = loss.length(xyz[:-1,2], xyz[1:,0])
    CN_loss = torch.clamp( torch.abs(blen_CN - ideal_NC) - sig_len, min=0.0 )

    pairsum = is_motif[:-1].double() + is_motif[1:].double()
    pairsum[idx[:-1] - idx[1:] != -1] = -1

    junction = pairsum == 1
    intra_motif = pairsum == 2
    intra_diff = pairsum == 0
    
    try:
        if len(CN_loss[junction]) % 2 == 1:  # remove any motif at termini
            junction[0] = False
            junction[-1] = False
        per_res_loss = CN_loss[junction].reshape(-1, 2).mean(-1)
    except Exception as e:
        print('WARNING: Could not calculate per_res_loss', e)
        if len(CN_loss[junction]) % 2 == 0:
            per_res_loss = CN_loss[junction].reshape(-1, 2).mean(-1)
        else:
            per_res_loss = torch.zeros(len(CN_loss[junction]))  # TODO: handle error cases properly (when pigs fly)
    return {
        'junction_CN_loss': CN_loss[junction].mean().item(),
        'intra_motif_CN_loss': CN_loss[intra_motif].mean().item(),
        'intra_diff_CN_loss': CN_loss[intra_diff].mean().item(),
        'max_junction_CN_loss': per_res_loss.max().item(),
    }


def ligand_closest_contacts(parsed, motif_idxs):
    """
    Simple function to calculate ligand clashes with protein

    NB masks out: 
        - motif residues at idxs specified 
        - any residues which are not alanine.

    Returns:
        - ligand_closest_contacts: list of closest contacts between atoms in ligand and protein
    """


    o = {}
    if 'xyz_het' not in parsed or len(parsed['xyz_het']) == 0: 
        return {'ligand_closest_contacts': [999]} # no ligand loaded
    xyz_het = torch.tensor(parsed['xyz_het']) # L_het, 3
    xyz_het = xyz_het.reshape(-1, 3)
    xyz = torch.tensor(parsed['xyz'])   # L, 14, 3
    mask = torch.tensor(parsed['mask']) # L, 14, 3
    mask[:, 4] = False  # Remove Cb's from mask (mpnn might remove them by replacing ala -> gly)

    # mask out motif
    motif_mask = torch.zeros(xyz.shape[0]).bool() # L,
    motif_mask[motif_idxs] = True
    motif_mask[parsed['seq'] != 0] = True  # also include any part of the protein which is not an alanine.
    mask[motif_mask] = False
    xyz = xyz[mask].reshape(-1, 3)  # L-n_motif, 14, 3 -> n_atoms_prot, 3

    dists = torch.norm(xyz_het[:,None] - xyz[None], dim=-1)  # L_het, n_atoms_prot
    dists, _ = torch.min(dists, dim=-1)  # L_het,
    o['ligand_closest_contacts'] = dists.tolist()

    return o


# Extra function (but only works if pdb is a design + has logged trajectories)
def dislocated_ca(pdb):
    o = {}
    row = analyze.make_row_from_traj(pdb)
    trb = analyze.get_trb(row)
    config = trb['config']

    pdb_ids = [''.join([str(ii) for ii in i]) for i in trb['con_hal_pdb_idx']]
    pdb_ids = np.arange(len(pdb_ids))  # can just select id in motif
    if config['inference'].get('contig_as_guidepost', False):
        bb_i = np.array(trb['con_hal_idx0'])
        gp_i = np.array(list(trb['motif'].keys()))
        deatomized_xyz, is_het = get_last_px0(row)
        het_idx = is_het.nonzero()[0]
        gp_i = gp_i[~np.isin(gp_i, het_idx)]
        return dislocated_ca_inner(deatomized_xyz, bb_i, gp_i)
    else:
        o['gp.ca_dists'] = [0.0] * len(pdb_ids)
        o['gp.ca_dist_max'] = 0
        o['gp.ca_dist_mean'] = 0
    return o

def dislocated_ca_inner(xyz, bb_i, gp_i):
    o = {}
    bb_motif = torch.tensor(xyz)[bb_i]
    gp_motif = torch.tensor(xyz)[gp_i]
    ca_dist = np.linalg.norm(gp_motif[:, 1] - bb_motif[:, 1], axis=-1)

    if np.isnan(ca_dist).any():  # paranoia.
        print('CA dist has NaNs')
        o['gp.ca_dists'] = [999]*len(pdb_ids)
        o['gp.ca_dist_max'] = 999
        o['gp.ca_dist_mean'] = 999
    else:
        o['gp.ca_dists'] = ca_dist.astype(float).tolist()
        o['gp.ca_dist_max'] = np.max(ca_dist)
        o['gp.ca_dist_mean'] = np.mean(ca_dist)
    return o


def geometry_inner(parsed, motif_idxs, cache_rots=True):
    """
    Computes geometry metrics for a parsed pdb

    Args:
        parsed: the parsed pdb dictionary
        motif_idxs: idx0 of motif residues
        cache_rots: whether to cache rotamer probability calcualtion for speedup
    Returns:
        o: dictionary of geometry metrics. Keys prefixed with 'geometry.' and categorized as 'bb.', 'gp.', 'rot.'

    TODO: Add bond length deviations
    TODO: Add exact angle deviations
    TODO: Add ligand clashes
    """
    o = {}
    out_as = lambda c, o : {f'{c}.{k}':v for k,v in o.items() if k != 'name'}

    def add_bb():
        o = {}
        o = o | clashes_inner(parsed, clash_dist=1.50)
        o = o | chainbreaks(parsed)
        o = o | ligand_closest_contacts(parsed, motif_idxs)
        return out_as('bb', o)
    o = o | add_bb()

    def add_gp():
        o = {}
        o = o | junction_bond_len(parsed, motif_idxs)
        o = o | ca_cb_deviations(parsed, motif_idxs)
        # ca-ca dislocations are added outside (as aren't always possible to compute)
        return out_as('gp', o)
    o = o | add_gp()

    def add_rot():
        o = {}
        o = o | get_motif_rotamer_probability_inner(parsed, motif_idxs)
        o = o | get_angle_deviations(parsed, motif_idxs)
        o = o | find_ideal_irots(parsed, motif_idxs, skip_good_residues=False, return_as_metric=True)
        o = o | tyr_angle_evaluation(parsed, motif_idxs)
        return out_as('rot', o)
    o = o | add_rot()

    return o

def compile_geometry_dict(o: dict,
    # GP:
    JUNCTION_CN_LOSS_THRESHOLD = 0.7, # 0.6; std of C-N bond length deviation allowed from ideal (between consecutive residues)
    CB_DEV_THRESHOLD = 0.20,  # deviation from ideal CB placement based on N-Ca-C
    CA_DISLOCATION_THRESHOLD = 0.5, # Ca (gp) - Ca (backbone) distance
    
    # ROT:
    ROTAMER_SCORE_CUT = 1.2,  # higher values will relax the rotamer score threshold. 1.0 will pass ~99% of native rotamers.
    RELAX_PROB = -0.2,  # set to 0 if you want to be absolutely sure your rotamers are as good as natives
    ANGLE_DEVIATION_CUT = 15, # angles that the upstream HA bond angles can deviate from 109.5 or 120 (whichever is smallest)
    ROTAMER_IDEALISATION_RMSD = 4.1, # RMSD to idealization. See geometry_metrics_utils.py 

    # BB:
    CHAINBREAK_DEVIATION_THRESHOLD = 1.0, # Contiguous Ca-Ca chain distance deviations (Ang.) from 3.8 allowed up to
    NUM_CLASHES_ALLOWED = 0,
    LIGAND_CLOSEST_CONTACT = 0.5, # Angstroms, minimum distance of heteroatom to protein backbone - excludes motif from calculation
):
    """
    Compile computed values from geometry() to determine whether the design passes.

    The failure modes are represented by three categories:
    - `bb.fails` if any of the following:
        - `bb.has_res_clash` - inter-residue clashes
        - `bb.has_chainbreak` - Ca-Ca distances 1 Å above 3.6 Å
        - `bb.has_ligand_clash` - for any atoms within the backbone (i've left any residues not equal to alanine & motif out of the calculation for simplicity).
    - `gp.fails` if any of the following
        - `gp.has_dislocated_ca` - A substantial disagreement between the Ca of the diffused backbone and the sidechain (NB only can be calculated if `inference.write_trajectory=True`) 
        - `gp.has_cb_deviation` - Deviation from ideal Cb placement based on N-Ca-C frame (NB most informative if `inference.write_gp_bb_as_diffused=True`).
        - `gp.has_bad_junction` - Failure for the incoming and outgoing `N-C` peptide-bond lengths (NB will be most informative if `inference.guidepost_xyz_as_design_bb=False`, 
            since that will represent the guidepost's backbone coordinates and not just the diffused backbone which is usually self-consistent)
    - `rot.fails` if any of the following:
        - `rot.has_angle_deviation` measures whether any (bend) angles in the sidechains (`Ca-Cb...` onwards) deviate significantly from either 109.5 or 120 degs
            (or 100 degs if histidine).
        - `rot.has_bad_rot_prob` whether any rotamers have a low backbone-dependent Dunbrack rotamer probability - these are calibrated to pass 99.9% of native 
            rotamers.
        - `rot.tyr.has_contortion` if there are any tyrosines in the residue and the ring is sufficiently bent (will be ~30-40% fail rate depending on number of tyrosines).
        - `rot.has_nonideal_rotamer` this will align a large set (>300) of residues sampled from the Dunbrack library to each of the catalytic residues (at the 
            tips). It then selects the closest backbone fit and gives the RMSD. If the maximum RMSD for any catalytic residue is > 3 angstrom it will flag the 
            motif as having a nonideal rotamer. 

    Whether to look for `gp.has_bad_junction` or `gp.has_cb_deviation` depends on whether you are using `inference.guidepost_xyz_as_design_bb=True` or 
        `False` (respectively) --- in either case, `gp.fails` should trigger if it's looking bad. The metric `gp.has_dislocated_ca` will capture all flying 
        rotamers *but* requires trajectory files. I recommend using `inference.guidepost_xyz_as_design_bb=True` as this will give 2 axes where punted motif 
        failures can be detected -- `gp.has_bad_junction` and `gp.has_chainbreak` -- instead of just through CB dev alone.

    *.score: continuous-value score for failure mode
    *.fails: boolean for whether any failure mode is present within category
    """
    name = o.get('name', None)
    o = { k.replace('geometry.', ''): v for k, v in o.items() if k not in ['name'] }
    
    get_failure_modes = lambda o_compiled: [k for k, v in o_compiled.items() if k.split('.')[-1].startswith('has_') and v]
    # fails_any = lambda o_compiled: any([v for k, v in o_compiled.items() if k.split('.')[-1].startswith('has_')])
    
    def compile_bb(o):
        o_compiled = {}

        # Failure modes
        o_compiled['bb.has_res_clash'] = o['bb.n_res_clash'] > NUM_CLASHES_ALLOWED
        o_compiled['bb.has_chainbreak'] = o['bb.max_caca_deviation'] > CHAINBREAK_DEVIATION_THRESHOLD
        o_compiled['bb.num_ligand_clash'] = sum([d <= LIGAND_CLOSEST_CONTACT for d in o['bb.ligand_closest_contacts']])
        o_compiled['bb.has_ligand_clash'] = o_compiled['bb.num_ligand_clash'] > 0

        o_compiled['bb.failure_modes'] = get_failure_modes(o_compiled)
        o_compiled['bb.fails'] = len(o_compiled['bb.failure_modes']) > 0

        # Score
        o_compiled['bb.score.mean_chainbreak_distance_error'] = float(np.mean(o['bb.mean_caca_deviation'])) / 1.0
        o_compiled['bb.score.number_of_clashes'] = int(o['bb.n_pair_clash']) / 1.0
        o_compiled['bb.score'] = (o_compiled['bb.score.mean_chainbreak_distance_error'] + o_compiled['bb.score.number_of_clashes'] )

        return o_compiled
    o = o | compile_bb(o)

    def compile_gp(o):
        o_compiled = {}

        # Failure modes
        o_compiled['gp.has_bad_junction'] = o['gp.max_junction_CN_loss'] > JUNCTION_CN_LOSS_THRESHOLD
        o_compiled['gp.num_dislocated_ca'] = sum([d > CA_DISLOCATION_THRESHOLD for d in o.get('gp.ca_dists', [0.])])
        o_compiled['gp.has_dislocated_ca'] = o_compiled['gp.num_dislocated_ca'] > 0
        o_compiled['gp.num_cb_deviation'] = sum([d > CB_DEV_THRESHOLD for d in o['gp.ca_cb_deviations']])
        o_compiled['gp.has_cb_deviation'] = o_compiled['gp.num_cb_deviation'] > 0

        o_compiled['gp.failure_modes'] = get_failure_modes(o_compiled)
        o_compiled['gp.fails'] = len(o_compiled['gp.failure_modes']) > 0

        # Score
        o_compiled['gp.score.max_ca_ca_dislocation'] = max(o.get('gp.ca_dists', [0.])) / 6
        o_compiled['gp.score.mean_max_junction_loss'] = o['gp.max_junction_CN_loss'] / 3
        o_compiled['gp.score.mean_cb_deviation'] = max(o['gp.ca_cb_deviations']) / 6
        o_compiled['gp.score'] = (o_compiled['gp.score.max_ca_ca_dislocation'] + o_compiled['gp.score.mean_max_junction_loss'] + o_compiled['gp.score.mean_cb_deviation'])

        return o_compiled
    o = o | compile_gp(o)

    def compile_rot(o):
        o_compiled = {}

        # Failure modes
        o_compiled['rot.tyr.has_contortion'] = o['rot.tyr.num_contorted'] > 0
        o_compiled['rot.has_bad_rot_prob'] = o['rot.max_rot_prob_score'] > ROTAMER_SCORE_CUT
        o_compiled['rot.num_angle_deviation'] = sum([a > ANGLE_DEVIATION_CUT for a in o['rot.max_angle_deviations']])
        o_compiled['rot.has_angle_deviation'] = o_compiled['rot.num_angle_deviation'] > 0
        o_compiled['rot.num_nonideal_rotamer'] = sum([d > ROTAMER_IDEALISATION_RMSD for d in o['rot.idealization_rmsd_allatom']])
        o_compiled['rot.has_nonideal_rotamer'] = o_compiled['rot.num_nonideal_rotamer'] > 0

        o_compiled['rot.failure_modes'] = get_failure_modes(o_compiled)
        o_compiled['rot.fails'] = len(o_compiled['rot.failure_modes']) > 0

        # Score
        o_compiled['rot.score.rot_prob'] = o['rot.mean_rot_prob_score'] / 2
        o_compiled['rot.score.angle_deviation'] = o['rot.total_angle_deviation'] / 80
        o_compiled['rot.score'] = (o_compiled['rot.score.rot_prob'] + o_compiled['rot.score.angle_deviation'])

        return o_compiled
    o = o | compile_rot(o)

    o['score'] = o['bb.score'] + o['gp.score'] + o['rot.score']
    o['failure_modes'] = o['bb.failure_modes'] + o['gp.failure_modes'] + o['rot.failure_modes']
    o['fails'] = len(o['failure_modes']) > 0

    o = {f'geometry.{k}':v for k,v in o.items()}
    o = {'name': name} | o
    return o

def geometry_from_parsed(parsed, idx, t_step=0):
    o = {}

    # Default values
    o['ca_dist'] = [0.] * len(idx) if idx is not None else [0.]
    o['ca_dist_max'] = 0
    o['ca_dist_mean'] = 0
    o['t_step'] = t_step # timestep where design is evaluated
    o['t'] = 0.
    o['1_minus_t'] = 1.

    o['list_catres_posi'] = []
    o['avg_cart_bonded'] = 0
    o['avg_fa_dun'] = 0
    o['max_cart_bonded'] = 0
    o['max_fa_dun'] = 0

    o = o | geometry_inner(parsed, idx)
    o = compile_geometry_dict(o)
    return o

def compile_geometry_from_rows(row):
    parsed = { k: v for k, v in row.to_dict().items() if k.startswith('geometry.') }

    for k, v in parsed.items():
        if isinstance(v, str):
            parsed[k] = ast.literal_eval(v)
    o = compile_geometry_dict(parsed)
    for k, v in o.items():
        if k == 'name': continue
        row[k] = v
    return row
