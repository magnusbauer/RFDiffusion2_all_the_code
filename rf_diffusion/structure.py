from __future__ import annotations  # Fake Import for type hinting, must be at beginning of file

"""
Adapted from PyDSSP for ss conditioning
"""
from einops import repeat, rearrange
import torch
import numpy as np
from typing import Union, Tuple
from typing import Literal
from rf2aa.chemical import ChemicalData as ChemData
from rf_diffusion.kinematics import generate_H
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rf_diffusion.aa_model import Indep

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0
atomnum = {' N  ':0, ' CA ': 1, ' C  ': 2, ' O  ': 3}

C3_ALPHABET = np.array(['H', 'E', 'L', '?'])

HELIX = 0
STRAND = 1
LOOP = 2
ELSE = 3


def _check_input(coord):
    # Validates input
    org_shape = coord.shape
    assert (len(org_shape)==3) or (len(org_shape)==4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = repeat(coord, '... -> b ...', b=1) if len(org_shape)==3 else coord
    return coord, org_shape


# Don't use this function. It doesn't handle chainbreaks well at all
# def _get_hydrogen_atom_position(coord: torch.Tensor) -> torch.Tensor:
#     # A little bit lazy (but should be OK) definition of H position here.
#     vec_cn = coord[:,1:,0] - coord[:,:-1,2]
#     vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
#     vec_can = coord[:,1:,0] - coord[:,1:,1]
#     vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
#     vec_nh = vec_cn + vec_can
#     vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
#     return coord[:,1:,0] + 1.01 * vec_nh


def get_hbond_map(
    coord: torch.Tensor,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    return_e: bool=False
    ) -> torch.Tensor:
    """
    Calculate the hydrogen bond map based on the given coordinates.

    Args:
        coord (torch.Tensor): The input coordinates
        cutoff (float, optional): The cutoff distance for hydrogen bond interactions.
        margin (float, optional): The margin value for the hydrogen bond map.
        return_e (bool, optional): Whether to return the electrostatic interaction energy.

    Returns:
        torch.Tensor: The hydrogen bond map

    Raises:
        AssertionError: If the number of atoms is not 5 (N, CA, C, O, H).

    """
    # check input
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a==5), "Number of atoms should 5 (N,CA,C,O,H)"
    h = coord[:,1:,4]
    # distance matrix
    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)
    d_on = torch.linalg.norm(omap - nmap, dim=-1)
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)
    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [0,1,1,0])
    if return_e: return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool)
    local_mask *= ~torch.diag(torch.ones(l-1, dtype=bool), diagonal=-1)
    local_mask *= ~torch.diag(torch.ones(l-2, dtype=bool), diagonal=-2)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map/margin*torch.pi/2)+1.)/2
    hbond_map = hbond_map * repeat(local_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
    # return h-bond map
    hbond_map = hbond_map.squeeze(0) if len(org_shape)==3 else hbond_map
    return hbond_map


def assign_torch(coord: torch.Tensor) -> torch.Tensor:
    """
    Assigns secondary structure elements (SSEs) to a given coordinate tensor.

    Args:
        coord (torch.Tensor): The input coordinate tensor.

    Returns:
        torch.Tensor: The tensor representing the assigned SSEs.

    """
    # check input
    coord, org_shape = _check_input(coord)
    # get hydrogen bond map
    hbmap = get_hbond_map(coord)
    hbmap = rearrange(hbmap, '... l1 l2 -> ... l2 l1') # convert into "i:C=O, j:N-H" form
    # identify turn 3, 4, 5
    turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3) > 0.
    turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4) > 0.
    turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = torch.nn.functional.pad(turn3[:,:-1] * turn3[:,1:], [1,3])
    h4 = torch.nn.functional.pad(turn4[:,:-1] * turn4[:,1:], [1,4])
    h5 = torch.nn.functional.pad(turn5[:,:-1] * turn5[:,1:], [1,5])
    # helix4 first
    helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
    h3 = h3 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    h5 = h5 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
    helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1) > 0.
    unfoldmap_rev = unfoldmap.transpose(-4,-3)
    p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
    p_bridge = torch.nn.functional.pad(p_bridge, [1,1,1,1])
    a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
    a_bridge = torch.nn.functional.pad(a_bridge, [1,1,1,1])
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # H, E, L of C3
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    onehot = torch.stack([helix, strand, loop], dim=-1) # modified from pydssp
    onehot = onehot.squeeze(0) if len(org_shape)==3 else onehot
    return onehot


def read_pdbtext_with_checking(pdbstring: str):
    """
    Reads the coordinates from a PDB string and returns them as a numpy array.
    Only takes C, CA, N, O atoms. Attempts to validate inputs 

    Args:
        pdbstring (str): The PDB string containing the coordinates.

    Returns:
        numpy.ndarray: The coordinates extracted from the PDB string.

    """    
    lines = pdbstring.split("\n")
    coords, atoms, resid_old, check = [], None, None, []
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16], None)
            resid = l[21:26]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                    check.append(atom_check)
                atoms, resid_old, atom_check = [], resid, []
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
                atom_check.append(iatom)
    if atoms is not None:
        coords.append(atoms)
        check.append(atom_check)
    coords = np.array(coords)
    # check
    assert len(coords.shape) == 3, "Some required atoms [N,CA,C,O] are missing in the input PDB file"
    check = np.array(check)
    assert np.all(check[:,0]==0), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,1]==1), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,2]==2), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,3]==3), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    # output
    return coords


def assign(
    coord: Union[torch.Tensor, np.ndarray],
    out_type: Literal['onehot', 'index', 'c3'] = 'index'
    ) -> torch.Tensor:
    """
    Assigns secondary structure labels to a given set of coordinates.

    Args:
        coord (Union[torch.Tensor, np.ndarray]): The input coordinates.
        out_type (Literal['onehot', 'index', 'c3'], optional): The type of output to return. 
            Defaults to 'c3'.

    Returns:
        np.ndarray: The assigned secondary structure labels.

    Raises:
        AssertionError: If the input type is not torch.Tensor or np.ndarray.
        AssertionError: If the output type is not 'onehot', 'index', or 'c3'.
    """
    assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    assert out_type in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"
    # main calculation
    onehot = assign_torch(coord)
    # output one-hot
    if out_type == 'onehot':
        return onehot
    # output index
    index = torch.argmax(onehot.to(torch.long), dim=-1)
    if out_type == 'index':
        return index
    # output c3
    c3 = C3_ALPHABET[index.cpu().numpy()]
    return c3

def read_pdbtext_no_checking(pdbstring: str):
    """
    Reads the coordinates from a PDB string and returns them as a numpy array.
    Only takes C, CA, N, O atoms

    Args:
        pdbstring (str): The PDB string containing the coordinates.

    Returns:
        numpy.ndarray: The coordinates extracted from the PDB string.

    """
    lines = pdbstring.split("\n")
    coords, atoms, resid_old = [], None, None
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16], None)
            resid = l[21:26]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                atoms, resid_old = [], resid
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
    if atoms is not None:
        coords.append(atoms)
    coords = np.array(coords)
    return coords


def get_bb_pydssp(indep: Indep, is_gp: Union[bool, torch.Tensor] = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rearranges indep.xyz into a format for PyDSSP.

    Args:
        indep (Indep): The input object containing the data.
        is_gp (bool, Tensor[bool]): tensor describing if each residue is guide post or not
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The rearranged data in the format required by PyDSSP. [L, 4, 3]
    """
    if isinstance(is_gp, bool):
        is_gp = torch.ones(indep.length(), dtype=bool) * is_gp
    is_prot = (indep.seq <= 20) * ~is_gp * ~indep.is_sm
    # is_prot = nucl_utils.get_resi_type_mask(indep.seq, 'prot') * ~is_gp * ~indep.is_sm
    N_idx = ChemData().aa2long[0].index(" N  ")
    CA_idx = ChemData().aa2long[0].index(" CA ")
    C_idx = ChemData().aa2long[0].index(" C  ")
    O_idx = ChemData().aa2long[0].index(" O  ")
    N = indep.xyz[is_prot, N_idx]
    CA = indep.xyz[is_prot, CA_idx]
    C = indep.xyz[is_prot, C_idx]
    O = indep.xyz[is_prot, O_idx]
    H = generate_H(N, CA, C)
    bb = torch.stack([N, CA, C, O, H], dim=0)
    bb_pydssp = torch.transpose(bb, 0, 1)
    return bb_pydssp, is_prot



def get_dssp_string(dssp_output: torch.Tensor) -> str:
    '''
    Convert the output from structure.get_dssp() to a human readable string

    Args:
        dssp_output (Tensor[long]): the output tensor from structure.get_dssp()

    Returns:
        str: The human readable dssp string (ex: 'LLLHHHHHLLLLEEEELLLL')
    '''
    return ''.join(C3_ALPHABET[dssp_output])


def get_dssp(indep: Indep, is_gp: Union[bool, torch.Tensor] = False) -> torch.Tensor:
    '''
    Get the DSSP assignemt of indep using PyDSSP.

    Note! PyDSSP labels beta bulges as loops!

    structure.HELIX = 0
    structure.STRAND = 1
    structure.LOOP = 2
    structure.ELSE = 3

    Args:
        indep (Indep): The input object containing the data.
        is_gp (bool, Tensor[bool]): tensor describing if each residue is guide post or not

    Returns:
        torch.Tensor: The DSSP assignment [L]

    '''

    bb_pydssp, is_prot = get_bb_pydssp( indep, is_gp )
    dssp = torch.full((indep.length(),), ELSE, dtype=int)
    dssp[is_prot] = assign(bb_pydssp, out_type='index')

    return dssp











