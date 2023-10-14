import networkx as nx
from typing import Iterable


class ListDict(dict):
    """
    A dictionary subclass that stores multiple values for a key as a list.

    If a key is seen more than once, the corresponding value is appended to a list
    containing the previous values for that key. If a key is seen only once, its value
    is stored in a list with a single element.

    Examples:
    >>> d = ListDict()
    >>> d['a'] = 1
    >>> d['b'] = 2
    >>> d['c'] = 3
    >>> d['a'] = 4
    >>> d['b'] = 5
    >>> d['a'] = 6
    >>> print(d)
    {'a': [1, 4, 6], 'b': [2, 5], 'c': [3]}
    """
    def __setitem__(self, key, value):
        if key in self:
            if isinstance(self[key], list):
                self[key].append(value)
            else:
                self[key] = [self[key], value]
        else:
            super().__setitem__(key, [value])

    def update(self, d2):
        for k, v in d2.items():
            if isinstance(v, Iterable):
                for v_ in v:
                    self[k] = v_
            else:
                self[k] = v

def find_subgraphs(G: nx.Graph, H: nx.Graph, 
                   GraphMatcher: nx.algorithms.isomorphism.GraphMatcher=nx.algorithms.isomorphism.GraphMatcher,
                   one_isomorphism: bool=True):
    '''
    Collect views of all of the sugraphs of G that are
    isomorphic ("match") the graph H. Matches are defined by the 
    `GraphMatcher` class.
    
    Returns
        isomorphic_subgraphs: A subgraph (view) of G that is isomorphic to H.
        H_to_subgraph_nodes: Dictionary with H nodes as keys and subgraph nodes as values.
    '''    
    # Instantiate the graph matcher object
    gm = GraphMatcher(G, H)
    
    # Collect matches
    matches = ListDict()
    for mapping in gm.subgraph_isomorphisms_iter():
        G_nodes, H_nodes = zip(*mapping.items())
        isomorphic_subgraph = G.subgraph(G_nodes)
        H_to_subgraph_nodes = {v: k for k, v in mapping.items()}        
        matches[frozenset(G_nodes)] = (isomorphic_subgraph, H_to_subgraph_nodes)

    if one_isomorphism:
        out = [v[0] for v in matches.values()]
    else:
        out = [item for sublist in matches.values() for item in sublist]
        
    return out