import torch


def expand_1d_atomized_ok_gp_not(indep, input_mask, post_idx_from_pre_idx, null_value=0, key='unknown'):
    '''
    For use with ExpandConditionsDict

    Expands a 1d mask (of any type) to the full length of the transformed indep
    Atomized copies of the original residues get the same value as the original
    is_gp copies of the original residues get null_value

    Args:
        indep (Indep): indep
        input_mask (torch.Tensor): The mask from the pre-transformed indep that we are expanding [L pre-transform]
        post_idx_from_pre_idx (list[list[int]]): Mapping from pre-transform to post-transform [L pre-transform]
        null_value (any): The value to store if input_mask is invalid for this position
        key (str): The name of this value in conditions_dict (for error messages)

    Returns:
        new_mask (torch.Tensor): The input_mask but expanded to the post-transformed indep [L post-transform]
    '''
    assert len(input_mask) == len(post_idx_from_pre_idx), f'{key} is not the same length as pre-transform_indep indep'
    new_mask = torch.full((indep.length(),), null_value, dtype=input_mask.dtype)
    for pre_idx, post_idxs in enumerate(post_idx_from_pre_idx):
        new_mask[post_idxs] = input_mask[pre_idx]
    new_mask[indep.is_gp] = null_value
    return new_mask


def expand_2d_atomized_ok_gp_not(indep, input_mask, post_idx_from_pre_idx, null_value=0, key='unknown'):
    '''
    For use with ExpandConditionsDict

    Expands a 2d mask (of any type) to the full length of the transformed indep
    Atomized copies of the original residues maintain their interactions as in the original
    All is_gp residues have all interactions set to null_value

    Args:
        indep (Indep): indep
        input_mask (torch.Tensor): The mask from the pre-transformed indep that we are expanding [L pre-transform, L pre-transform]
        post_idx_from_pre_idx (list[list[int]]): Mapping from pre-transform to post-transform [L pre-transform]
        null_value (any): The value to store if input_mask is invalid for this position
        key (str): The name of this value in conditions_dict (for error messages)

    Returns:
        new_mask (torch.Tensor): The input_mask but expanded to the post-transformed indep [L post-transform, L post-transform]
    '''
    assert input_mask.shape[0] == len(post_idx_from_pre_idx), f'{key} is not the same length as pre-transform_indep indep'
    assert input_mask.shape[1] == len(post_idx_from_pre_idx), f'{key} is not the same length as pre-transform_indep indep'
    new_mask = torch.full((indep.length(),indep.length(),), null_value, dtype=input_mask.dtype)

    post_idx_from_pre_idx_torch = [torch.tensor(x) for x in post_idx_from_pre_idx]

    # No good way to do this besides all-by-all
    for pre_idx_a, post_idxs_a in enumerate(post_idx_from_pre_idx_torch):
        for pre_idx_b, post_idxs_b in enumerate(post_idx_from_pre_idx_torch):
            # The singular value from pre gets expanded to the rectangular-matrix from post
            new_mask[post_idxs_a[:,None],post_idxs_b[None,:]] = input_mask[pre_idx_a,pre_idx_b]

    # Mask out any interaction with a gp residue
    new_mask[indep.is_gp,:] = null_value
    new_mask[:,indep.is_gp] = null_value

    return new_mask
