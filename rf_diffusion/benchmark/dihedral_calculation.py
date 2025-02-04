"""
Module for calculating dihedral angles and rotamer probabilities.
Uses backbone-dependent Dunbrack library for calculation.

Author(s): Jie Chen, Jasper Butcher
"""

import pandas as pd
import numpy as np
import torch
from rf_diffusion.chemical import ChemicalData 
from rf_diffusion.frame_diffusion.data.residue_constants import chi_angles_atoms
from scipy.special import logsumexp
import pathlib
from pathlib import Path

symmetric_chi_angles = {
    'ASP': [2], 'ASN': [2], 'PHE': [2], 'TYR': [2],
    'HIS': [2], 'TRP': [2], 'GLU': [3], 'GLN': [3],
    'ARG': [4], 'LYS': [4], 'MET': [3],
}

def dihedral_calculation(p: np.ndarray) -> float:
    """
    Calculate the dihedral angle.
    
    Parameters:
    p (np.ndarray): An array of shape (4, 3) containing the coordinates of four points.
    
    Returns:
    float: The dihedral angle in degrees.
    """
    # Convert input to numpy array if it's a torch tensor
    if isinstance(p, torch.Tensor):
        p = p.numpy()
    elif isinstance(p, (list, tuple)):
        # Convert list of tensors/arrays to numpy array
        p = np.array([x.numpy() if isinstance(x, torch.Tensor) else x for x in p])
    
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1
    b1 /= (np.linalg.norm(b1))

    # vector rejections
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

rot_df_cache = {}
def get_rotamer_probability(
    residue_xyz: np.ndarray, 
    residue_name: str, 
    phi: float, psi: float, 
    return_best_rotamer: bool = False, 
    allow_caching: bool = True
) -> float:
    '''
    Get the probability of a rotamer given the residue coordinates, residue name, phi, and psi angles.
    The function finds the corresponding dunbrack rotamer library file, narrow down the rotamers based on phi and psi angles,
    deals with residues with symmetric chi angles, and calculates the probability of the rotamer.

    Parameters:
    residue_xyz (np.ndarray): The coordinates of the residue atoms.
    residue_name (str): The residue name.
    phi (float): The phi angle.
    psi (float): The psi angle.

    Returns:
    float: The probability of the rotamer.
    '''
    residue_name = residue_name.upper()

    if residue_name in ['GLY', 'ALA']:
        return None

    chem_data = ChemicalData()
    residue_index = chem_data.aa2num[residue_name]
    atom_names = chem_data.aa2long[residue_index]
    residue_name_lower = residue_name.lower()
    folder = pathlib.Path(__file__).parent.resolve()
    file_path = f'{folder}/rotamer_library/{residue_name_lower}.bbdep.rotamers.lib'

    #find and read the rotamer library file
    try:
        rotamer_df = rot_df_cache.get(residue_name, None)
        if not allow_caching or rotamer_df is None:
            rotamer_df = pd.read_csv(file_path, skiprows=38, header=None, delim_whitespace=True)
            rotamer_df.columns = ["T", "Phi", "Psi", "Count", "r1", "r2", "r3", "r4", "Probabil",
                                "chi1Val", "chi2Val", "chi3Val", "chi4Val",
                                "chi1Sig", "chi2Sig", "chi3Sig", "chi4Sig"]
            print(f'Loaded {residue_name} rotamer csv') if allow_caching else None
        if allow_caching:
            rot_df_cache[residue_name] = rotamer_df

    except FileNotFoundError:
        print(f'Rotamer library for {residue_name} not found')
        return None

    phi = int(np.round(phi / 10.0) * 10)
    psi = int(np.round(psi / 10.0) * 10)

    #narrow down the rotamers based on phi and psi angles
    rotamer_df = rotamer_df[(rotamer_df['Phi'] == phi) & (rotamer_df['Psi'] == psi)]
    if rotamer_df.empty:
        print(f'Backbone for {residue_name} not found in rotamer library')
        return None

    # Convert residue_xyz to numpy if it's a tensor
    if isinstance(residue_xyz, torch.Tensor):
        residue_xyz = residue_xyz.numpy()

    # Get each atom position of the residue and store it with corresponding chi angle
    atom_info = {atom_name.strip(): coord for atom_name, coord in zip(atom_names, residue_xyz) if atom_name}
    chi_angles = chi_angles_atoms[residue_name.upper()]
    dihedral_angles = {}

    for chi_num, chi in enumerate(chi_angles):
        dihedral_coords = []
        for atom in chi:
            if atom in atom_info:
                dihedral_coords.append(atom_info[atom])
            else:
                print(f'Atom {atom} not found in residue')
                return None
        
        # Convert coordinates to numpy array explicitly with dtype=float
        dihedral_coords = np.array(dihedral_coords, dtype=float)
        angle = dihedral_calculation(dihedral_coords)

        # Deal with residues with symmetric chi angles since dunbrack library only covers until 180 degrees
        if (chi_num + 1) in symmetric_chi_angles.get(residue_name, []):
            angle = abs(angle) % 180
        else:
            angle = angle % 360
        dihedral_angles[chi_num] = angle

    total_log_prob = []
    # Calculate the probability of each rotamer using bayesian by recreating the probability density function through the chi mean and std 
    # then summing the log probabilities of each angle
    for idx, entry in rotamer_df.iterrows():
        #prior probability
        log_prob = np.log(entry['Probabil']) if entry['Probabil'] != 0 else 0 # gives warning if entry['Probabil'] is 0
        #calculate the log probability of each chi angle
        for i in range(len(chi_angles)):
            observed_angle = dihedral_angles[i]
            mean_angle = entry[f'chi{i+1}Val']
            sigma = entry[f'chi{i+1}Sig']
            #take into account the circularity of the angles
            delta_angle = (observed_angle - mean_angle + 180) % 360 - 180
            log_prob_chi = -0.5 * (delta_angle / sigma) ** 2 - np.log(np.sqrt(2 * np.pi) * sigma)
            log_prob += log_prob_chi
        total_log_prob.append(log_prob)

    total_log_prob = np.array(total_log_prob)
    #overall probability
    log_prob = logsumexp(total_log_prob)
    probability = np.exp(log_prob)

    if return_best_rotamer:
        best_rotamer_idx = np.argmax(total_log_prob)
        best_rotamer = rotamer_df.iloc[best_rotamer_idx]
        return probability, best_rotamer

    return probability

def calculate_phi_psi(prev_residue_xyz, curr_residue_xyz, next_residue_xyz):
    """
    Calculate the phi and psi angles for a given set of coordinates.

    Parameters:
    prev_residue_xyz (np.ndarray): The coordinates of the previous residue atoms.
    curr_residue_xyz (np.ndarray): The coordinates of the current residue atoms.
    next_residue_xyz (np.ndarray): The coordinates of the next residue atoms.

    Returns:
    tuple: A tuple containing the phi and psi angles in degrees
    """
    # Convert tensors to numpy arrays
    if isinstance(prev_residue_xyz, torch.Tensor):
        prev_residue_xyz = prev_residue_xyz.numpy()
    if isinstance(curr_residue_xyz, torch.Tensor):
        curr_residue_xyz = curr_residue_xyz.numpy()
    if isinstance(next_residue_xyz, torch.Tensor):
        next_residue_xyz = next_residue_xyz.numpy()

    C_prev = prev_residue_xyz[2]
    N = curr_residue_xyz[0]
    CA = curr_residue_xyz[1]
    C = curr_residue_xyz[2]
    N_next = next_residue_xyz[0]

    # Convert coordinates to numpy arrays explicitly
    phi_coords = np.array([C_prev, N, CA, C], dtype=float)
    psi_coords = np.array([N, CA, C, N_next], dtype=float)

    phi = dihedral_calculation(phi_coords)
    psi = dihedral_calculation(psi_coords)

    return phi, psi


def sample_bbdep_rotamers(residue_xyz: np.ndarray, residue_name: str, phi: float, psi: float, n_samples=10, filtered=True) -> float:
    '''
    Parameters:
    residue_xyz (np.ndarray): The coordinates of the residue atoms.
    residue_name (str): The residue name.
    phi (float): The phi angle.
    psi (float): The psi angle.

    '''
    residue_name = residue_name.upper()

    if residue_name in ['GLY', 'ALA']:
        return None

    residue_name_lower = residue_name.lower()
    folder = Path(__file__).parent.resolve()
    file_path = f'{folder}/rotamer_library/{residue_name_lower}.bbdep.rotamers.lib'

    #find and read the rotamer library file
    try:
        rotamer_df = pd.read_csv(file_path, skiprows=38, header=None, delim_whitespace=True)
        rotamer_df.columns = ["T", "Phi", "Psi", "Count", "r1", "r2", "r3", "r4", "Probabil",
                              "chi1Val", "chi2Val", "chi3Val", "chi4Val",
                              "chi1Sig", "chi2Sig", "chi3Sig", "chi4Sig"]
    except FileNotFoundError:
        print(f'WARNING: Rotamer library for {residue_name} not found')
        return None

    phi = int(np.round(phi / 10.0) * 10)
    psi = int(np.round(psi / 10.0) * 10)

    #narrow down the rotamers based on phi and psi angles
    rotamer_df = rotamer_df[(rotamer_df['Phi'] == phi) & (rotamer_df['Psi'] == psi)]
    if rotamer_df.empty:
        print(f'Backbone for {residue_name} not found in rotamer library')
        return None

    chi_angles = chi_angles_atoms[residue_name.upper()]
    all_sampled_chis = []
    for idx, entry in rotamer_df.iterrows():
        sampled_chis = [
            [entry[f'chi{i+1}Val'] 
            for i in range(len(chi_angles))]
        ] + [
            [entry[f'chi{i+1}Val'] + 
            # float(np.random.normal(loc=0, scale=entry[f'chi{i+1}Sig'])) # normal 
            float(np.random.normal(loc=0, scale=entry[f'chi{i+1}Sig'] / np.sqrt(2))) # exponentiated normal (tighter around 0) 
            for i in range(len(chi_angles))]
            for _ in range(n_samples-1)
        ]


        if residue_name in symmetric_chi_angles:
            symmetric_chi_idx = symmetric_chi_angles[residue_name][0]
            sampled_chis = sampled_chis + [
                [((180 - chi) if chi_idx == symmetric_chi_idx else chi) 
                for chi_idx, chi in enumerate(chis)]
                for chis in sampled_chis
            ]

        all_sampled_chis.extend(sampled_chis)
    all_sampled_chis = np.stack(all_sampled_chis)
    return all_sampled_chis

