import numpy as np
import biotite
import biotite.structure.io as strucio


def standardize_element_names(element_names):
    return np.char.upper(element_names)

def atom_array_from_pdb(pdb_path):
    atom_array = strucio.load_structure(pdb_path, extra_fields=["b_factor"])
    atom_array.element = standardize_element_names(atom_array.element)
    return atom_array

def pdb_from_atom_array(pdb_path, atom_array):
    strucio.save_structure(pdb_path, atom_array)

def without_hydrogens(atom_array):
    return atom_array[atom_array.element != 'H']

def map_pdb(
    input_pdb,
    output_pdb,
    f,
):
    '''
    Reads a PDB to an atom array, transforms it with `f`, and saves it to disk.
    Params:
        input_pdb [string]
        output_pdb [string]
        f: [atom_array -> atom_array]
    '''

    atom_array = strucio.load_structure(input_pdb, extra_fields=["b_factor"])
    atom_array = f(atom_array)

    assert len(atom_array) > 0, f'{input_pdb=}'
    strucio.save_structure(output_pdb, atom_array)


def get_intra_chain_discontinuities(atom_array):

    all_discont = []
    atom_array.set_annotation('enclosing_idx', np.arange(len(atom_array)))
    for atom_array_chain in biotite.structure.chain_iter(atom_array):
        discontinuities = biotite.structure.check_backbone_continuity(atom_array_chain)
        discontinuities = atom_array_chain[discontinuities]
        if len(discontinuities) > 0:
            all_discont.extend(discontinuities.enclosing_idx)

    return all_discont