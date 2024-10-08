import torch
import random
import numpy as np
import rf_diffusion.structure as structure
from rf2aa.kinematics import generate_Cbeta
import conditions.ss_adj.sec_struct_adjacency as sec_struct_adjacency

def Cb_or_atom(indep):
    '''
    Generates the virtual CB atom for protein residues and returns atom for small molecule atoms

    Args:
        indep (indep): indep

    Returns:
        Cb (torch.Tensor): The xyz of virtual Cb atom for protein atoms and atom for small molecule atoms [L,3] 
    '''
    N = indep.xyz[~indep.is_sm,0]
    Ca = indep.xyz[~indep.is_sm,1]
    C = indep.xyz[~indep.is_sm,2]

    # small molecules will use atom 1 for neighbor calculations
    Cb = indep.xyz[:,1].clone()
    Cb[~indep.is_sm] = generate_Cbeta(N,Ca,C)
    return Cb

def find_hotspots_antihotspots(indep, max_hotspot_frac, max_antihotspot_frac, dist_cutoff=10, return_all=False):
    '''
    Finds all possible hotspots and antihotspots for training and then downsamples to a smaller number.
        A hotspot is any motif residue within 10A (CB-dist) of a residue on a different chain.
        An antihotspot is same the criteria but greater than 10A from all residues.

    Args:
        indep (Indep): An indep
        max_hotspot_frac (float): What fraction of hotspots found should be returned
        max_antihotspot_frac (float): What fraction of antihotspots found should be returned
        dist_cutoff (float): What is the distance cutoff for hotspots (you can't change this without retraining)
        return_all (bool): If true, return allhotspots

    Returns:
        is_hotspot (torch.Tensor[bool]): Is this position a hotspot [L]
        is_antihotspot (torch.Tensor[bool]): Is this position a hotspot [L]

    '''
    Cb = Cb_or_atom(indep)

    d2_map = torch.sum( torch.square( Cb[:,None] - Cb[None,:] ), axis=-1 )
    are_close = d2_map <= dist_cutoff**2


    # Can these pairs be hotspot/antihotspot?
    hotspot_mask = are_close & ~indep.same_chain
    antihotspot_mask = ~are_close & ~indep.same_chain

    is_hotspot_full = hotspot_mask.any(axis=-1)
    # antihotspots additionally need to be not hotspots
    #  (just because you're 10A away from 1 residue doesn't mean your 10A away from all of them)
    is_antihotspot_full = antihotspot_mask.any(axis=-1) & ~is_hotspot_full

    if return_all:
        return is_hotspot_full, is_antihotspot_full

    # now to randomly donwample the hotspots
    n_hotspots = is_hotspot_full.sum()
    n_antihotspots = is_antihotspot_full.sum()

    keep_n_hotspots = random.randint(0, int(torch.ceil( n_hotspots * max_hotspot_frac )))
    keep_n_antihotspots = random.randint(0, int(torch.ceil( n_antihotspots * max_antihotspot_frac )))

    wh_hotspots_full = torch.where(is_hotspot_full)[0]
    wh_antihotspots_full = torch.where(is_antihotspot_full)[0]

    # basically np.random.choice()
    wh_hotspots = wh_hotspots_full[torch.randperm(n_hotspots)[:keep_n_hotspots]]
    wh_antihotspots = wh_antihotspots_full[torch.randperm(n_antihotspots)[:keep_n_antihotspots]]

    is_hotspot = torch.zeros(indep.length(), dtype=bool)
    is_antihotspot = torch.zeros(indep.length(), dtype=bool)

    is_hotspot[wh_hotspots] = True
    is_antihotspot[wh_antihotspots] = True

    return is_hotspot, is_antihotspot


def radial_crop(indep, is_diffused, is_hotspot, is_target, distance=25):
    '''
    Crop anything on is_target that isn't within distance(=25) of the hotspots

    Args:
        indep (Indep): indep
        is_diffused (torch.Tensor[bool]): Whether or not this residue is diffused (these won't be cropped) [L]
        is_hotspot (torch.Tensor[bool]): Is this residue a hotspot [L]
        is_target (torch.Tensor[bool]): Is this residue part of the target [L]
        distance (float): Distance from hotspot residues beyond which target residues are removed

    Returns:
        torch.Tensor[bool]: Which residues should remain after cropping [L]
    '''

    if is_hotspot.sum() == 0:
        print("Warning! radial_crop didn't receive any hotspot residues. Not cropping")
        return torch.ones(indep.length(), dtype=bool)

    Ca = indep.xyz[:,1]

    hotspot_Ca = Ca[is_hotspot]

    d2_to_hotspot = torch.sum( torch.square( hotspot_Ca[None,:] - Ca[:,None] ), axis=-1).min( axis=-1 ).values
    close_enough = d2_to_hotspot < distance**2

    crop_residues = is_target & ~close_enough
    assert not (crop_residues & is_hotspot).any()
    assert not (crop_residues & is_diffused).any()

    return ~crop_residues


def random_unit_vector():
    '''
    Returns a unit vector uniformly oriented in any direction

    Returns:
        torch.Tensor: Unit vector [3]
    '''
    unit = torch.normal( torch.zeros(3), torch.ones(3) )
    unit /= torch.linalg.norm(unit)
    return unit


def planar_crop(indep, is_diffused, is_hotspot, is_target, distance=10):
    '''
    Crops everything extending beyone a plane from the hotspots. Similar to shift-highlighting in pymol
        The plane orientation is randomly chosen and must remove at least 1 target residue
        This method gives up after 100 tries if it can't find a valid plane

    Args:
        indep (Indep): indep
        is_diffused (torch.Tensor[bool]): Whether or not this residue is diffused (these won't be cropped) [L]
        is_hotspot (torch.Tensor[bool]): Is this residue a hotspot [L]
        is_target (torch.Tensor[bool]): Is this residue part of the target [L]
        distance (float): How far away from the plane the closest hotspot can be

    Returns:
        torch.Tensor[bool]: Which residues should remain after cropping [L]
    '''

    if is_hotspot.sum() == 0:
        print("Warning! planar_crop didn't receive any hotspot residues. Not cropping")
        return torch.ones(indep.length(), dtype=bool)

    Ca = indep.xyz[:,1]
    hotspot_Ca = Ca[is_hotspot]
    target_Ca = Ca[is_target]

    use_upper_crop = None

    # Since we're picking random unit vectors. It's possible some won't work
    #  Try 100 times and if that doesn't work give up
    for attempt in range(100):

        unit = random_unit_vector()

        # t is how far along the unit the points are
        #  if you multiplied t * unit you'd get the projection
        hotspot_projection_t = (hotspot_Ca * unit[None]).sum(axis=-1)
        target_projection_t = (target_Ca * unit[None]).sum(axis=-1)

        upper_t = hotspot_projection_t.max() + distance
        lower_t = hotspot_projection_t.min() - distance

        cropped_upper = (target_projection_t > upper_t).sum()
        cropped_lower = (target_projection_t < lower_t).sum()

        if cropped_upper > 0 or cropped_lower > 0:
            use_upper_crop = cropped_upper > cropped_lower
            break

    if use_upper_crop is None:
        # couldn't find a crop, just dont crop (because the hotspots might envelop the target or something)
        print("Warning! Couldn't find a planar crop!")
        return torch.ones(indep.length(), dtype=bool)


    projection_t = (Ca * unit[None]).sum(axis=-1)

    if use_upper_crop:
        potentially_cropped = projection_t > upper_t
    else:
        potentially_cropped = projection_t < lower_t

    crop_residues = potentially_cropped & is_target

    assert not (crop_residues & is_hotspot).any()
    assert not (crop_residues & is_diffused).any()

    return ~crop_residues



def decide_target(indep):
    '''
    Decides which chain/s shall be the target for PPI training examples.

    The binder must be a single chain but the target can be arbitrarily many chains

    Args:
        indep (Indep): indep

    Returns:
        is_target (torch.Tensor[bool]): Which residues are the target [L]
    '''

    if indep.same_chain.all():
        return None

    chain_masks = torch.tensor(np.array(indep.chain_masks()))

    chain_is_elgible_for_binder = torch.zeros(len(chain_masks), dtype=bool)

    for imask, chain_mask in enumerate(chain_masks):
        
        # They say we can diffuse small molecules
        # if (chain_mask & indep.is_sm).any():
        #     continue
        chain_is_elgible_for_binder[imask] = True

    if chain_is_elgible_for_binder.sum() == 0:
        return None

    wh_elgible = torch.where(chain_is_elgible_for_binder)[0]
    i_wh = random.randint(0, len(wh_elgible)-1)

    i_binder_chain = wh_elgible[i_wh]

    # the opposite of the binder is the target
    return ~chain_masks[i_binder_chain]



def training_extract_ppi_motifs(indep, is_target, max_frac_ppi_motifs=0.8, max_ppi_motif_trim_frac=0.4, dist=8):
    '''
    Simulates a motif-graft case
        First we extract all ss elements that have a CB within 8Å of target
        Then we randomly trim them and delete some

    Args:
        indep (Indep): indep
        is_target (torch.Tensor[bool]): Which residues are the target [L]
        max_frac_ppi_motifs (float): What's the max fraction of motif chunks should be kept?
        max_ppi_motif_trim_frac (float): Whats the max fraction of a motif that should be trimmed away?
        dist (float): How close must a single CB atom on the motif be to count as a motif?

    Returns:
        is_ppi_motif (torch.Tensor[bool]): Is this residue part of a PPI motif?
    '''

    is_binder = ~is_target

    Cb = Cb_or_atom(indep)
    full_dssp, _ = structure.get_dssp(indep)

    binder_Cb = Cb[is_binder]
    target_Cb = Cb[is_target]

    binder_dssp = full_dssp[is_binder]

    # Segments are all the secondary structral elements
    segments = sec_struct_adjacency.ss_to_segments(binder_dssp, is_dssp=True)

    res_close_enough = torch.sum( torch.square( binder_Cb[:,None] - target_Cb[None,:] ), axis=-1) < dist**2

    # Go through segments and store them as motifs if any of the Cbs in that segment are within 8A of a Cb on the target
    motifs = []
    for typ, start, end in segments:
        if typ == structure.ELSE:    # not small molecule motifs for now
            continue
        if res_close_enough[start:end+1].any():
            motifs.append((typ, start, end))

    # Downsample the motifs
    n_keep = random.randint(min(1, len(motifs)), int(np.ceil(len(motifs)*max_frac_ppi_motifs)) )
    keep_idx = torch.randperm(len(motifs))[:n_keep]

    # Work through the motifs that we're keeping and trim from both N and C termini
    final_motifs = []
    for idx in keep_idx:
        typ, start, end = motifs[idx]

        length = end - start + 1
        n_trim_start = int(random.uniform(0, max_ppi_motif_trim_frac) * length)
        n_trim_end = int(random.uniform(0, max_ppi_motif_trim_frac) * length)

        start = start + n_trim_start
        end = end - n_trim_end
        # if we've trimmed to nothing, then convert to 1aa motif
        if start > end:
            start = end
        
        final_motifs.append((typ, start, end))

    # Store the motifs into a is_ppi_motif mask
    is_ppi_motif = torch.zeros(indep.length(), dtype=bool)
    wh_binder = torch.where(is_binder)[0]

    for tp, start, end in final_motifs:
        motif_indices = wh_binder[start:end+1]
        is_ppi_motif[motif_indices] = True

    return is_ppi_motif

