from biotite.structure import AtomArray
import numpy as np
import torch

from datahub.transforms.base import Transform
from datahub.transforms.atomize import AtomizeResidues, FlagNonPolymersForAtomization
from datahub.transforms.bonds import AddTokenBondAdjacency, AddRF2AABondFeaturesMatrix
from datahub.utils.token import get_token_starts
from datahub.transforms.encoding import atom_array_to_encoding
from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import AddProteinTerminiAnnotation
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)

from aa_model import Indep, N_TERMINUS, C_TERMINUS, is_occupied, pop_mask, deatomize_covales
class BackwardCompatibleDataLoaderProcessOut(Transform):
    requires_previous_transforms = [
        FlagNonPolymersForAtomization,
        AddProteinTerminiAnnotation,
        AtomizeResidues,
        AddTokenBondAdjacency,
        AddRF2AABondFeaturesMatrix
    ]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "rf2aa_bond_features_matrix", "extra_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            ["is_polymer", "pn_unit_iid", "molecule_iid", "molecule_entity", "chain_entity", "chain_iid", "atomize", "is_N_terminus", "is_C_terminus"],
        )

    def __init__(self, loader_params, sel_item_keys=['example_id', 'pdb_id', 'assembly_id', 'cluster'], task='diff'):
        self.loader_params = loader_params
        self.sel_item_keys = sel_item_keys
        self.task = task

    def _get_indep_and_atom_mask(self, data):
        encoded = atom_array_to_encoding(data['atom_array'], RF2AA_ATOM36_ENCODING)

        token_starts = get_token_starts(data['atom_array'])
        token_wise_atom_array = data['atom_array'][token_starts]

        _make_is_same_matrix = lambda x: x.unsqueeze(0) == x.unsqueeze(0).T  # noqa
        is_same_molecule_iid = _make_is_same_matrix(
            torch.from_numpy(token_wise_atom_array.molecule_iid.astype(np.int64))
        )

        terminus_type = torch.zeros(len(encoded['seq']), dtype=int)
        terminus_type[token_wise_atom_array.is_N_terminus] = N_TERMINUS
        terminus_type[token_wise_atom_array.is_C_terminus] = C_TERMINUS

        indep = Indep(
            torch.tensor(encoded['seq']),
            torch.tensor(encoded['xyz']),
            torch.arange(len(encoded['seq'])),
            data['rf2aa_bond_features_matrix'].to(torch.int64),
            torch.zeros((0,5)),
            is_same_molecule_iid,
            terminus_type
        )

        return indep, torch.tensor(encoded['mask'])

    def forward(self, data: dict) -> dict:
        mask_gen_seed = np.random.randint(0, 99999999)

        indep, atom_mask = self._get_indep_and_atom_mask(data)

        pop = is_occupied(indep, atom_mask)
        # For now, do not pop unoccupied small molecule atoms, exit instead, as popping them can lose covale information.
        unoccupied_sm = (~pop) * indep.is_sm
        if unoccupied_sm.any():
            raise Exception(f'there are small molecule atoms that are unoccupied at indices:  {unoccupied_sm.nonzero()[:,0]}')
        
        pop_mask(indep, pop)
        atom_mask = atom_mask[pop]

        # if self.conf.dataloader.max_residues > -1:
        #     # Kind of hacky as it may break covales.  Only for debugging.
        #     pop = indep.is_sm.clone()
        #     residue_indices = torch.where(~indep.is_sm)[0]
        #     residue_indices = residue_indices[:self.conf.dataloader.max_residues]
        #     pop[residue_indices] = True
        #     pop_mask(indep, pop)
        #     atom_mask = atom_mask[pop]

        indep, atom_mask, metadata = deatomize_covales(indep, atom_mask)

        sel_item = {key: data['extra_info'][key] for key in self.sel_item_keys}
        
        return {
            'indep': indep,
            'atom_mask': atom_mask,
            'metadata': metadata,
            'sel_item': sel_item,
            'task': self.task,
            'mask_gen_seed': mask_gen_seed,
            'params': self.loader_params,
            'conditions_dict': {}
        }
