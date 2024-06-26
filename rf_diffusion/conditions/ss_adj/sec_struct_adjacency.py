
import itertools
import rf_diffusion.structure as structure


# must match structure.HELIX, STRAND, LOOP
SS_HELIX = 0
SS_STRAND = 1
SS_LOOP = 2
SS_MASK = 3
SS_SM = 4 # SS_SM != structure.ELSE so that we maintain compatibility with old ss files
N_SS = 5

ADJ_FAR = 0
ADJ_CLOSE = 1
ADJ_MASK = 2
N_ADJ = 3


def repeating_regions(vector, only_keep=None):
    '''
    Returns regions of repeating elements
        (value, start, stop)
        stop is the last element of the region. For slicing using stop+1

    Args:
        vector (iterable): The vector to find repeating regions within
        only_keep (any): If set, only keep regions that match this value

    Returns:
        A list of regions
    '''
    offset = 0
    regions = []
    for value, group in itertools.groupby(vector):
        this_len = len(list(group))
        next_offset = offset + this_len
        if ( only_keep is None or only_keep == value ):
            regions.append( [value, offset, next_offset-1])
        offset = next_offset

    return regions


def ss_to_segments(ss, is_dssp=False):
    '''
    Split up a secondary string into secondary structural elements
        Every atom of a small molecule is a different element

    Args:
        ss (torch.Tensor): The secondary structure, either from structure.get_dssp() or SS files
        is_dssp (bool): True if this ss is directly from structure.get_dssp()

    Returns:
        segments (list[tuple[int, int, int]]): (ss_type, start, end) for each segment in the secondary structure
                                                to slice this element, use ss[start:end+1]

    '''
    SM = structure.ELSE if is_dssp else SS_SM

    # In the original from Nate + Joe, the "end" value of each segment is 1 past (like python slicing)
    # In this version, the "end" value is the final position with that ss type
    pre_segments = repeating_regions(ss)

    # We are calling every small molecule residue a different SS segment
    segments = []
    for seg in pre_segments:
        if seg[0] != SM:
            segments.append(seg)
        else:
            for i in range(seg[1], seg[2]+1):
                segments.append((SM, i, i))

    return segments




