import itertools
from collections import OrderedDict, defaultdict
import networkx as nx
import torch
from icecream import ic
import numpy as np
from rf2aa import chemical
from typing import Iterable
from rf_diffusion.graph_utils import ListDict, find_subgraphs

def calc_atom_bond_loss(indep, pred_xyz, is_diffused):
    """
    Loss on distances between bonded atoms
    """
    # Uncomment in future to distinguish between ligand / atomized_residue
    # is_residue = ~indep.is_sm
    # is_atomized = indep.is_sm & (indep.seq < rf2aa.chemical.NPROTAAS)
    # is_ligand = indep.is_sm & ~(indep.seq < rf2aa.chemical.NPROTAAS)
    mask_by_name = {}
    for k, v in {
        'residue': ~indep.is_sm,
        'atom': indep.is_sm,
    }.items():
        for prefix, mask in {
            'diffused': is_diffused,
            'motif': ~is_diffused
        }.items():
            mask_by_name[f'{prefix}_{k}'] = v*mask
    mask_by_name['all'] = torch.ones_like(is_diffused).bool()

    bond_losses = {}
    true_xyz = indep.xyz
    is_bonded = torch.triu(indep.bond_feats > 0)
    for (a, a_mask), (b, b_mask) in itertools.combinations_with_replacement(mask_by_name.items(), 2):
        is_pair = a_mask[..., None] * b_mask[None, ...]
        is_pair = torch.triu(is_pair)
        is_bonded_pair = is_bonded * is_pair
        i, j = torch.where(is_bonded_pair)
        
        true_dist = torch.norm(true_xyz[i,1]-true_xyz[j,1],dim=-1)
        pred_dist = torch.norm(pred_xyz[i,1]-pred_xyz[j,1],dim=-1)
        bond_losses[f'{a}:{b}'] = torch.mean(torch.abs(true_dist - pred_dist))
    return bond_losses


def find_all_rigid_groups(bond_feats):
    """
    Params:
        bond_feats: torch.tensor([N, N])
    
    Returns:
        list of tensors, where each tensor contains the indices of atoms within the same rigid group.
    """
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    any_atom_bonds = bond_feats != 0
    rigid_edges = (rigid_atom_bonds.int() @ any_atom_bonds.int()).bool() + rigid_atom_bonds
    rigid_atom_bonds_np = rigid_edges.cpu().numpy()
    G = nx.from_numpy_array(rigid_atom_bonds_np)
    connected_components = nx.connected_components(G)
    connected_components = [cc for cc in connected_components if len(cc)>1]
    connected_components = [torch.tensor(list(cc)) for cc in connected_components]
    return connected_components

def align(xyz1, xyz2, eps=1e-6):

    # center to CA centroid
    xyz1_mean = xyz1.mean(0)
    xyz1 = xyz1 - xyz1_mean
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    return xyz2_ + xyz1_mean

def calc_rigid_loss(indep, pred_xyz, is_diffused, atomizer=None):
    '''
    Params:
        indep: atomized aa_mode.Indep corresponding to the true structure
        pred_xyz: atomized xyz coordinates [L, A, 3]
    Returns:
        Dictionary mapping the composition of a rigid group to the maximum aligned RMSD of the group to the true 
        coordinates in indep.
        i.e. {'diffused_atom_motif_atom': 5.5} implies that the worst predicted rigid group which has
        at least 1 diffused atom and at least 1 motif atom has an RMSD to the corresponding set of atoms
        in the true structure of 5.5.

        The suffix '_determined' is added to groups which contain >=3 motif atoms, as all DoFs of these
        groups are determined by the motif atoms.
    '''
    rigid_groups = find_all_rigid_groups(indep.bond_feats)
    if atomizer:
        named_rigid_groups = [atomize.res_atom_name(atomizer, g) for g in rigid_groups]

    mask_by_name = OrderedDict()
    for k, v in {
        'residue': ~indep.is_sm,
        'atom': indep.is_sm,
    }.items():
        for prefix, mask in {
            'diffused': is_diffused,
            'motif': ~is_diffused
        }.items():
            mask_by_name[f'{prefix}_{k}'] = v*mask


    atom_types = torch.full((indep.length(),), -1)
    for i, (mask_name, mask) in enumerate(mask_by_name.items()):
        atom_types[mask] = i
    mask_names = list(mask_by_name)

    is_motif_key = np.char.startswith(np.array(mask_names), 'motif')
    motif_keys = np.nonzero(is_motif_key)[0]

    dist_by_composition =  defaultdict(list)
    for rigid_idx in rigid_groups:
        composition = atom_types[rigid_idx].tolist()
        n_motif = np.in1d(np.array(composition), motif_keys).sum()
        composition = tuple(sorted(list(set(composition))))
        composition_string = ':'.join(mask_names[e] for e in composition)
        if n_motif >= 3:
            composition_string += '_determined'
        true_ca = indep.xyz[rigid_idx, 1]
        pred_ca = pred_xyz[rigid_idx, 1]
        # motif_rmsd = benchmark.util.af2_metrics.calc_rmsd(true_ca, pred_ca)
        # pred_ca_aligned = rf2aa.util.superimpose(pred_ca[None], true_ca[None])[0]
        pred_ca_aligned = align(true_ca, pred_ca)
        dists = torch.norm(pred_ca_aligned - true_ca, dim=-1)
        dist_by_composition[composition_string].append(dists.max())

    out = {k:max(v) for k,v in dist_by_composition.items()}
    return out


#######################################
# Tools for doing regex-like searches on molecular graphs
#######################################
class HashableDict(dict):
    '''Can use as nodes in networkx graphs.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = id(self)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, HashableDict) and self._id == other._id

class IsMember():
    def __init__(self, members):
        assert isinstance(members, Iterable)
        self.members = members
        
    def __call__(self, val):
        '''Is val in members?'''
        return val in self.members

class RegexTests(HashableDict):
    '''
    Store functions that evaluate the query node's attributes.
    '''
    def __init__(self):
        super().__init__()
    
    @property
    def attr_to_test(self):
        attrs = {k.replace('_okay', '') for k in self.keys() if '_okay' in k}
        return attrs
    
    def pass_all(self, test_results):
        '''
        test_results: 
            keys: Names of query node attributes we're testing
            values (bool): Passed "<attr>_okay" function stored in self?
        '''
        if test_results.keys() != self.attr_to_test:
            raise Exception(f'The `test_results` dict must have keys {self.attr_to_test} '
                        f'but had {test_results.keys()} instead.')
            
        return all(test_results.values())

class ChemGraphMatcher(nx.algorithms.isomorphism.GraphMatcher):
    '''
    Class to help find isomorphic chemical subgraphs.
    '''
    def __init__(self, G_chem, G_re):
        super().__init__(G_chem, G_re)
        self.G_chem = G_chem
        self.G_re = G_re
    
    def semantic_feasibility(self, node_chem, node_re):
        test1 = super().semantic_feasibility(node_chem, node_re)
        test2 = self.node_object_match(node_chem, node_re)
        return test1 and test2
    
    def node_object_match(self, node_chem, node_re):
        '''
        Defines if two nodes are equivalent.
        '''
        matching_tests = self.G_re.nodes()[node_re]['tests']
        test_results = {}
        for attr_name in matching_tests.attr_to_test:
            test_func = matching_tests[f'{attr_name}_okay']
            attr_val = self.G_chem.nodes()[node_chem][attr_name]
            test_results[attr_name] = test_func(attr_val)

        return matching_tests.pass_all(test_results)

def find_and_relabel_subgraphs(G: nx.Graph, H: nx.Graph, ChemGraphMatcher=ChemGraphMatcher, one_isomorphism: bool=True):
    '''
    Returns subgraphs of G that match H (a regex graph) and relabeld the 
    subgraph's nodes to the corresponding label in H.
    '''
    subgraphs = []
    for SG, H_to_SG_node in find_subgraphs(G, H, ChemGraphMatcher):
        SG_to_H_node = {v: k for k, v in H_to_SG_node.items()}
        SG = nx.relabel_nodes(SG, SG_to_H_node)
        subgraphs.append(SG)
        
    return subgraphs

#######################################
# Premade regex graphs for canonical amino acids
#######################################
def is_light(atom_name: str):
    return atom_name[1] == 'H'

def is_heavy(atom_name: str):
    return (atom_name is not None) and (not is_light(atom_name))

atom_names_to_int = []
for line in chemical.aa2long:
    mapping = {name: i for i, name in enumerate(line) if name and (not is_light(name))}
    atom_names_to_int.append(mapping)
        
# Make aa chemical graphs
aa_graphs = []
for line in chemical.aabonds:
    G = nx.Graph()
    for a_name, b_name in line:
        if is_heavy(a_name) and is_heavy(b_name):
            G.add_edge(a_name, b_name)
    aa_graphs.append(G)

def aa_to_regex_graph(aa_int: int) -> nx.Graph:
    '''
    Make a regex graph for the given aa.
    '''
    G_mol = aa_graphs[aa_int]
    G_re = nx.Graph()

    for atom_name in G_mol:
        atom_int = atom_names_to_int[aa_int][atom_name]

        tests = RegexTests()
        
        # Has correct element
        tests['element_okay'] = IsMember(chemical.aa2elt[aa_int][atom_int])
        
        # Has correct degree
        if atom_name == ' N  ':
            degree = {1, 2}
        elif atom_name == ' C  ':
            degree = {2, 3}
        else:
            degree = {nx.degree(G_mol, atom_name)}
            
        tests['degree_okay'] = IsMember(degree)

        G_re.add_node(atom_name, tests=tests)

    for u, v in G_mol.edges():
        G_re.add_edge(u, v)
        
    return G_re


#######################################
# Functions to calculate bond geometry metrics
#######################################
def get_bond_dists(G: nx.Graph) -> OrderedDict:
    '''
    Get the distances between all bonded atoms.
    Nodes on G must have the node attribute 'xyz'.
    '''
    bond_dists = OrderedDict()
    for a, b in G.edges:
        dist = torch.linalg.norm(G.nodes[a]['xyz'] - G.nodes[b]['xyz'])
        bond_dists[frozenset({a, b})] = float(dist)

    return bond_dists

def gather_aa_geometries(G: nx.Graph, geo_fun) -> list:
    '''
    Searches G for all 20 canonical amino acids and applies
    the `geo_fun` to each to calculate the desire geometry.

    Inputs
    G: Graph representing a molecule. Nodes are atoms and edges
        are covalent bonds. Currently does not support different bond types.
        Nodes must have these attributes:
            element: The element's symbol on the periodic table.
            xyz: Torch tensor of the xyz coordinates of the atom.
    geo_fun: The function used to calculate the desired geometric stat.

    Return a list (len 20), whose entries are ListDicts. One for each aa.
        keys: atoms involved in the geometry
        vals: value of the dist, angle, dihedral etc
    '''
    # Find subgraphs for each amino acid
    all_geo = []
    for aa_int in range(20):
        geo = ListDict()
        G_re_aa = aa_to_regex_graph(aa_int)
        for SG in find_and_relabel_subgraphs(G, G_re_aa):
            geo.update(geo_fun(SG))
            
        all_geo.append(geo)
        
    return all_geo


#######################################
# Functions for comparing bond geoemetries between two structures
#######################################
def collate_and_flatten_dicts(*ds):
    '''
    ds (dictionaries) must have the same keys. 
    This function returns lists of the values in each dictionary, 
    added to each list in the same key order.
    '''
    vals = tuple([] for _ in ds)
    for k in ds[0]:
         for val, d in zip(vals, ds):
                val += d[k]

    return vals

def collate_and_flatten_bond_geo(*geos):
    '''
    geos are a list (len 20) of bond geometry dictionaries.
    Return the values from each `geo` in the same order.
    '''
    vals = tuple([] for _ in geos)
    for aa_int in range(20):
        ds = [geo[aa_int] for geo in geos]
        for val, new_val in zip(vals, collate_and_flatten_dicts(*ds)):
            val += new_val

    return vals

def split_bb_and_sc_bonds(geo):
    bb_bonds = (
        frozenset({' CA ', ' N  '}),
        frozenset({' C  ', ' CA '}),
        frozenset({' C  ', ' O  '}),
    )

    bb_geo = []
    sc_geo = []

    for d in geo:
        bb_bond = ListDict()
        sc_bond = ListDict()

        for k, v in d.items():
            if k in bb_bonds:
                bb_bond.update({k: v})
            else:
                sc_bond.update({k: v})

        bb_geo.append(bb_bond)
        sc_geo.append(sc_bond)
        
    return bb_geo, sc_geo
