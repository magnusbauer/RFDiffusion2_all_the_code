import torch
import random
import numpy as np
import rf_diffusion.structure as structure
from rf2aa.kinematics import generate_Cbeta
import conditions.ss_adj.sec_struct_adjacency as sec_struct_adjacency
from rf_diffusion.train_data.exceptions import NextExampleException
from rf_diffusion import aa_model

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

def torch_rand_choice_noreplace(arr, n):
    '''
    Exactly the same as np.random.choice(arr, n, replace=False)

    Args:
        arr (torch.Tensor): The array
        n (int): How many of the array you want
    '''
    return arr[torch.randperm(len(arr))[:n]]

def downsample_bool_mask(mask, n_to_keep=None, max_frac_to_keep=None):
    '''
    Given a bool mask, return another bool mask with only a subsample of the True positions

    Only specify one of n_to_keep or max_frac_to_keep

    Args:
        mask (torch.Tensor[bool]): The mask you wish to downsample
        n_to_keep (int): How many True should remain in the end
        max_frac_to_keep (float): Between 0 and max_frac_to_keep * mask.sum() True wil remain in the end
    '''
    assert not (n_to_keep is None and max_frac_to_keep is None), 'You have to specify one of them'
    assert not (n_to_keep is not None and max_frac_to_keep is not None), "You can't specify both of them"

    total = mask.sum()
    if n_to_keep is None:
        n_to_keep = random.randint(0, int(torch.ceil( total * max_frac_to_keep )))

    # Take a subset of the True positions
    wh_full = torch.where(mask)[0]
    wh = torch_rand_choice_noreplace(wh_full, n_to_keep)

    # Fill the new output mask
    mask_out = torch.zeros(len(mask), dtype=bool)
    mask_out[wh] = True

    return mask_out

def find_hotspots_antihotspots(indep, max_hotspot_frac, max_antihotspot_frac, dist_cutoff=10, return_all=False, only_of_first_chain=False):
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
        only_of_first_chain (bool): If true, hotspots will only be considered if the first chain is near them

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

    # Do this separate so we don't mess up antihotspots
    if only_of_first_chain:
        first_chain_mask = torch.tensor(indep.chain_masks()[0])

        # Get the distances to the first chain
        # [binderpos, full indep]
        d2_first_chain = d2_map[first_chain_mask]
        are_close_first = d2_first_chain <= dist_cutoff**2

        # Set all binder positions such that they are not close (can't be hotspot of self)
        are_close_first[:,first_chain_mask] = False

        # Run any on axis 0 to say "Is full indep position near any binder positions"
        new_is_hotspot_full = are_close_first.any(axis=0)

        # The new set of hotspots should be a subset of the old one
        if new_is_hotspot_full.sum() > 0:
            assert is_hotspot_full[new_is_hotspot_full].all()
            assert not new_is_hotspot_full[first_chain_mask].any()

        is_hotspot_full = new_is_hotspot_full

    if return_all:
        return is_hotspot_full, is_antihotspot_full

    is_hotspot = downsample_bool_mask(is_hotspot_full, max_frac_to_keep=max_hotspot_frac)
    is_antihotspot = downsample_bool_mask(is_antihotspot_full, max_frac_to_keep=max_antihotspot_frac)

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



def decide_target(indep, use_first_chain=False):
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
    if use_first_chain:
        i_wh = 0

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


# rosetta/main/source/src/core/select/util/SelectResiduesByLayer.cc
def sidechain_neighbors(binder_Ca, binder_Cb, else_Ca):

    conevect = binder_Cb - binder_Ca
    conevect /= torch.sqrt(torch.sum(torch.square(conevect), axis=-1))[:,None]

    vect = else_Ca[:,None] - binder_Cb[None,:]
    vect_lengths = torch.sqrt(torch.sum(torch.square(vect), axis=-1))
    vect_normalized = vect / vect_lengths[:,:,None]

    dist_term = 1 / ( 1 + torch.exp( vect_lengths - 9  ) )

    angle_term = (((conevect[None,:] * vect_normalized).sum(axis=-1) + 0.5) / 1.5).clip(0, None)

    sc_neigh = (dist_term * np.square( angle_term )).sum(axis=0)

    return sc_neigh




class PPITrimTailsChain0ComplexTransform:
    '''
    A transform that trims long disordered tails from chain0 in training examples
    Should be called before GenerateMasks

    Args:
        operate_on_datasets (list[str]): Which datasets should this be applied to?
        scn_unfolded_thresh (float): SideChainNeighbor threshold for something to be considered folded
        can_remove_hotspots (bool): Can disordered parts be removed even if they are touching the target?
        all_but_1_cutoff (int): An nmer must contain at least this many residues to be eliminated if all but 1 is unfolded
        all_but_2_cutoff (int): An nmer must contain at least this many residues to be eliminated if all but 2 are unfolded
        nmer (int): Look through the tails in chunks of this size for stretches of nearly-entirely unfolded residues
        verbose (bool): Print what this Transform does
    '''

    def __init__(self,
        operate_on_datasets=['all'],
        scn_unfolded_thresh=1.0,
        can_remove_hotspots=True,
        all_but_1_cutoff=4,
        all_but_2_cutoff=8,
        n_mer=9,
        verbose=True
    ):
        self.operate_on_datasets = operate_on_datasets
        self.scn_unfolded_thresh = scn_unfolded_thresh
        self.can_remove_hotspots = can_remove_hotspots
        self.all_but_1_cutoff = all_but_1_cutoff
        self.all_but_2_cutoff = all_but_2_cutoff
        self.n_mer = n_mer
        self.verbose = verbose


    def __call__(self, indep, atom_mask, chosen_dataset, metadata, **kwargs):

        # The input arguments unchanged if we decide not to do anthing
        do_nothing_return = dict(indep=indep, atom_mask=atom_mask, chosen_dataset=chosen_dataset, metadata=metadata, **kwargs)

        # Only operate on the datasets we're told to
        if not ('all' in self.operate_on_datasets or chosen_dataset in self.operate_on_datasets):
            return do_nothing_return


        chain_masks = indep.chain_masks()
        if len(chain_masks) == 1:
            print('PPITrimTailsChain0ComplexTransform got passed a single chain')
            return do_nothing_return

        # Get the basic backbone data
        Ca = indep.xyz[:,1]
        Cb = Cb_or_atom(indep)
        is_binder = torch.tensor(chain_masks[0])
        binderlen = is_binder.sum()
        if not is_binder[:binderlen].any():
            raise NextExampleException("PPITrimTailsChain0ComplexTransform: Binder isn't contiguous") # need to see if this ever happens

        # Find sidechain neighbors and hotspots
        sc_neigh = sidechain_neighbors(Ca[is_binder], Cb[is_binder], Ca[is_binder])

        is_hotspots_both, _ = find_hotspots_antihotspots(indep, 1, 1 ,return_all=True, only_of_first_chain=False)
        hotspot_on_binder = is_hotspots_both & is_binder
        if hotspot_on_binder.sum() == 0:
            raise NextExampleException("PPITrimTailsChain0ComplexTransform: Chains aren't touching?")
        wh_hotspot_on_binder = torch.where(hotspot_on_binder)[0]

        # Mark trimming bounds
        first_hotspot = wh_hotspot_on_binder[0]
        last_hotspot = wh_hotspot_on_binder[-1]

        # Find unfolded residues
        is_unfolded = sc_neigh < self.scn_unfolded_thresh

        # If we can remove hotspots, we first start at the hotspot bounds and work into the binder
        if self.can_remove_hotspots:
            # Move the first hotspot into the binder
            for i_slice in range(first_hotspot-1, binderlen):
                all_unfolded = self.my_unfolded(is_unfolded[max(i_slice-self.n_mer+1, 0):i_slice+1])
                if not all_unfolded:
                    break
            if i_slice >= first_hotspot:
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Trimming {i_slice - first_hotspot + 1} residues into binder start')
            first_hotspot = i_slice + 1

            # Move the last hotspot into the binder
            for i_slice in range(last_hotspot+1, -1, -1):
                all_unfolded = self.my_unfolded(is_unfolded[i_slice:min(i_slice+self.n_mer, binderlen)])
                if not all_unfolded:
                    break
            if i_slice < last_hotspot:
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Trimming {last_hotspot - i_slice + 1} residues into binder end')
            last_hotspot = i_slice - 1


        if first_hotspot >= last_hotspot:
            raise NextExampleException("PPITrimTailsChain0ComplexTransform: The whole binder got trimmed!")


        # Starting from the first and last residues on the binder that are near the target
        #  Move outwards looking for the first and last n_mers that are totally unfolded
        keep_mask = torch.ones(indep.length(), dtype=bool)

        # i_slice is that last residue that will be removed
        for i_slice in range(first_hotspot-1, -1, -1):
            all_unfolded = self.my_unfolded( is_unfolded[max(i_slice-self.n_mer+1, 0):i_slice+1] )
            if all_unfolded:
                keep_mask[:i_slice+1] = False
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Removing first {i_slice+1} residues')
                break

        # i_slice is that first residue that will be removed
        for i_slice in range(last_hotspot+1, binderlen):
            all_unfolded = self.my_unfolded( is_unfolded[i_slice:min(i_slice+self.n_mer, binderlen)] )
            if all_unfolded:
                keep_mask[i_slice:binderlen] = False
                if self.verbose:
                    print(f'PPITrimTailsChain0ComplexTransform: Removing last {binderlen-i_slice} residues')
                break

        # This implies a bug in this code
        assert (keep_mask[~is_binder]).all(), 'A target residue was going to get removed'

        if keep_mask.all():
            return do_nothing_return

        # Remove the residues we said we wanted to remove
        aa_model.pop_mask(indep, keep_mask)
        atom_mask = atom_mask[keep_mask]
        metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], keep_mask)

        return dict(
            indep=indep,
            atom_mask=atom_mask,
            chosen_dataset=chosen_dataset,
            metadata=metadata,
            **kwargs
            )


    def my_unfolded(self, is_unfolded):
        '''
        Determine whether a stretch of residues counts as "totally unfolded"

        Args:
            is_unfolded (torch.Tensor[bool]): A stretch of residues with their folded states
        '''

        # Check how many are folded
        n_folded = (~is_unfolded).sum()
        total = len(is_unfolded)

        # If its a short segment, they all have to be folded
        if total <= self.all_but_1_cutoff:
            return n_folded <= 0

        # A little longer and 1 can be folded
        if total <= self.all_but_2_cutoff:
            return n_folded <= 1

        # Otherwise it's max length and up to 2 can be folded
        return n_folded <= 2



class PPIRejectUnfoldedInterfacesTransform:
    '''
    A transform that rejects training examples where the binder is too unfolded

    Uses Fraction Boundary by SideChainNeighbors as the metric

    '''

    def __init__(self, operate_on_datasets=['all'], binder_fbscn_cut=0.12, binder_fbscn_at_interface_cut=0.13, verbose=True):
        '''
        Args:
            operate_on_datasets (list[str]): Which datasets should this be applied to?
            binder_fbscn_cut (float): Fraction Boundary by SideChainNeighbor cut for the binder as a whole
            binder_fbscn_at_interface_cut (float): Fraction Boundary by SideChainNeighbor cut for the binder parts at the interface
            verbose (bool): Print something when this filter fails
        '''

        self.operate_on_datasets = operate_on_datasets
        self.binder_fbscn_cut = binder_fbscn_cut
        self.binder_fbscn_at_interface_cut = binder_fbscn_at_interface_cut
        self.verbose = verbose

    def __call__(self, indep, chosen_dataset, **kwargs):

        # The input arguments unchanged if we decide not to do anthing
        do_nothing_return = dict(indep=indep, chosen_dataset=chosen_dataset, **kwargs)

        # Only operate on the datasets we're told to
        if not ('all' in self.operate_on_datasets or chosen_dataset in self.operate_on_datasets):
            return do_nothing_return


        chain_masks = indep.chain_masks()
        if len(chain_masks) == 1:
            print('PPIRejectUnfoldedInterfacesTransform got passed a single chain')
            return do_nothing_return

        # Get the basic backbone data
        Ca = indep.xyz[:,1]
        Cb = Cb_or_atom(indep)
        is_binder = torch.tensor(chain_masks[0])

        # Get sidechain neighbors and hotspots
        sc_neigh = sidechain_neighbors(Ca[is_binder], Cb[is_binder], Ca[is_binder])

        is_hotspots_both, _ = find_hotspots_antihotspots(indep, 1, 1, return_all=True, only_of_first_chain=False)
        hotspot_on_binder = is_hotspots_both & is_binder
        if hotspot_on_binder.sum() == 0:
            raise NextExampleException("PPIRejectUnfoldedInterfacesTransform: Chains aren't touching?")

        # The standard scn cutoff for boundary is 4.0
        binder_fbscn = (sc_neigh > 4.0).float().mean()
        interface_fbscn = (sc_neigh[hotspot_on_binder[is_binder]] > 4.0).float().mean()

        if binder_fbscn < self.binder_fbscn_cut:
            raise NextExampleException(f'PPIRejectUnfoldedInterfacesTransform: Failed binder_fbscn_cut: {binder_fbscn} < {self.binder_fbscn_cut}',
                                                                                                                            quiet=not self.verbose)
        if interface_fbscn < self.binder_fbscn_at_interface_cut:
            raise NextExampleException('PPIRejectUnfoldedInterfacesTransform: Failed binder_fbscn_at_interface_cut:' +
                                        f' {interface_fbscn} < {self.binder_fbscn_at_interface_cut}', quiet=not self.verbose)

        return do_nothing_return



class PPIJoeNateDatasetRadialCropTransform:
    '''
    An interesting trick used by Joe and Nate.

    Ensure that the binder is less than a certain size uncropped, then radially crop the target around a random hotspot
    Binder can be ensured to be smaller than that by using data_loader.fast_filters.*.reject_chain0_longer_than 
    '''

    def __init__(self, operate_on_datasets=['all'], CROP=300):
        '''
        Args:
            operate_on_datasets (list[str]): Which datasets should this be applied to?
            CROP (int): How many residues should this be cropped to?
        '''
        self.operate_on_datasets = operate_on_datasets
        self.CROP = CROP

    def __call__(self, indep, atom_mask, chosen_dataset, metadata, **kwargs):

        # The input arguments unchanged if we decide not to do anthing
        do_nothing_return = dict(indep=indep, atom_mask=atom_mask, chosen_dataset=chosen_dataset, metadata=metadata, **kwargs)

        # Only operate on the datasets we're told to
        if not ('all' in self.operate_on_datasets or chosen_dataset in self.operate_on_datasets):
            return do_nothing_return

        chain_masks = indep.chain_masks()
        if len(chain_masks) == 1:
            print('PPIJoeNateDatasetRadialCropTransform got passed a single chain')
            return do_nothing_return

        # No need to crop if it's already small enough
        if indep.length() <= self.CROP:
            return do_nothing_return

        # Get the basic backbone data
        Cb = Cb_or_atom(indep)
        is_binder = torch.tensor(chain_masks[0])
        binderlen = is_binder.sum()

        if binderlen > self.CROP:
            raise NextExampleException('PPIJoeNateDatasetRadialCropTransform: Do you have data_loader.fast_filters.*.reject_chain0_longer_than ' + 
                                                                                                            f'set up? Binderlen: {binderlen}')


        # Find hotspots
        is_hotspots_both, _ = find_hotspots_antihotspots(indep, 1, 1 ,return_all=True, only_of_first_chain=False)
        hotspot_on_binder = is_hotspots_both & is_binder
        if hotspot_on_binder.sum() == 0:
            raise NextExampleException("PPIJoeNateDatasetRadialCropTransform: Chains aren't touching?")
        wh_hotspot_on_binder = torch.where(hotspot_on_binder)[0]

        # Randomly pick a single hotspot
        chosen_hotspot = torch_rand_choice_noreplace(wh_hotspot_on_binder, 1)[0]
        hotspot_Cb = Cb[chosen_hotspot]

        # Calculate distance squared to hotspot and set binders to all have distance 0
        dist2_hotspot = torch.sum( torch.square( Cb - hotspot_Cb ), axis=-1 )
        dist2_hotspot[is_binder] = 0

        # Find the closest residues to the hotspot
        _, keep_idx = torch.topk(dist2_hotspot, self.CROP, largest=False)

        keep_mask = torch.zeros(indep.length(), dtype=bool)
        keep_mask[keep_idx] = True

        # This implies a bug in this code
        assert (keep_mask[is_binder]).all(), 'A binder residue was going to get removed'

        aa_model.pop_mask(indep, keep_mask)
        atom_mask = atom_mask[keep_mask]
        metadata['covale_bonds'] = aa_model.reindex_covales(metadata['covale_bonds'], keep_mask)

        return dict(
            indep=indep,
            atom_mask=atom_mask,
            chosen_dataset=chosen_dataset,
            metadata=metadata,
            **kwargs
            )

