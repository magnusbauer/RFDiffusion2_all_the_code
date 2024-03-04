import random

import torch
import networkx as nx

from rf_diffusion.chemical import ChemicalData as ChemData

O_INDEX = 3

def choose_furthest_from_oxygen(res):
    '''
    Params:
        res (int): sequence token
    '''
    bond_feats = get_residue_bond_feats(res)
    bond_graph = nx.from_numpy_matrix(bond_feats.numpy())
    at_dist = nodes_at_distance(bond_graph, O_INDEX)
    furthest = at_dist[-1]
    return random.choice(furthest)


def get_residue_bond_feats(res, include_H=False):
    bond_feats = torch.zeros((ChemData().NTOTAL, ChemData().NTOTAL))
    for j, bond in enumerate(ChemData().aabonds[res]):
        start_idx = ChemData().aa2long[res].index(bond[0])
        end_idx = ChemData().aa2long[res].index(bond[1])

        # maps the 2d index of the start and end indices to btype
        bond_feats[start_idx, end_idx] = ChemData().aabtypes[res][j]
        bond_feats[end_idx, start_idx] = ChemData().aabtypes[res][j]
    
    if not include_H:
        bond_feats = bond_feats[:ChemData().NHEAVYPROT, :ChemData().NHEAVYPROT]
    return bond_feats

def nodes_at_distance(G, start):
    """
    Generate a list of nodes at different distances from a specified starting node in a graph.

    Parameters:
    - G (networkx.Graph): The input graph.
    - start: The starting node for distance calculation.

    Returns:
    list: A list where at_dist[i] contains all nodes that are i edges away from the starting node.
    """

    shortest_paths = nx.single_source_shortest_path_length(G, source=start)
    
    at_dist = [[] for _ in range(max(shortest_paths.values()) + 1)]

    for node, distance in shortest_paths.items():
        at_dist[distance].append(node)

    return at_dist