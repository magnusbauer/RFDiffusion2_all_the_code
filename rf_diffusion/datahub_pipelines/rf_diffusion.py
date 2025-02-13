import numpy as np
from cifutils.constants import AF3_EXCLUDED_LIGANDS
import omegaconf

from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinPolyResIdxAnnotation,
    SortLikeRF2AA,
)
from datahub.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from datahub.transforms.base import Compose, ConvertToTorch, RandomRoute
from datahub.transforms.bonds import (
    AddRF2AABondFeaturesMatrix,
    AddTokenBondAdjacency,
)
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.encoding import EncodeAtomArray
from datahub.transforms.filters import (
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
    RemoveUnsupportedChainTypes
)

from datahub.transforms.symmetry import AddPostCropMoleculeEntityToFreeFloatingLigands
from rf_diffusion.datahub_dataset_interface import BackwardCompatibleDataLoaderProcessOut

# def make_forward_compatible_with_datahub(object):
"""
I'm leaving this code in in case we want to in the future migrate to fully datahub based dataloading pipeline
"""
#     class CompatibleTransform(Transform):
#         def forward(self, data: dict):
#             return object.__call__(**data)

#         def check_input(self, data: dict[str, Any]) -> None:
#             if hasattr(object, 'check_input'):
#                 return object.check_input(self, data)
#             else:
#                 pass  # Default implementation that does nothing

#     return CompatibleTransform()

def build_rf_diffusion_transform_pipeline(
    *,
    # Cropping parameters
    crop_size: int = 256,  # Paper: 256
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_probability: float = 0.5,
    crop_contiguous_probability: float = 0.5,
    # Filtering parameters
    unresolved_ligand_atom_limit: int | float | None = 0.1,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    # Atomization parameters
    res_names_to_atomize: list[str] = None,
    # Diffusion parameters
    loader_params = None,
    # Miscellaneous parameters
    extra_pre_crop_transforms: omegaconf.dictconfig.DictConfig | dict = {},
    conditions_dict_atom_array_keys: list[str] = [],
) -> Compose:
    """
    Creates a transformation pipeline for the RF2AA model, applying a series of transformations to the input data.

    Args:
        - crop_size (int, optional): Size of the crop for spatial and contiguous cropping (in number of tokens).
            Defaults to 384.
        - crop_center_cutoff_distance (float, optional): Cutoff distance for the center of the crop (in Angstroms).
            Defaults to 15.0.
        - crop_spatial_probability (float, optional): Probability of performing spatial cropping. Defaults to 0.5.
        - crop_contiguous_probability (float, optional): Probability of performing contiguous cropping. Defaults to 0.5.
        - unresolved_ligand_atom_limit (int | float, optional): Limit for above which a ligand is considered unresolved.
            many unresolved atoms has its atoms removed. If None, all atoms are kept, if between 0 and 1, the number of
            atoms is capped at that percentage of the crop size. If an integer >= 1, the number of unresolved atoms is
            capped at that number. Defaults to 0.1.
        - undesired_res_names (list[str]): res_names to drop. Defaults to AF3_EXCLUDED_LIGANDS
        - res_names_to_atomize (list[str], optional): List of residue names to *always* atomize. Note that RF2AA already
            atomizes all residues that are not in the encoding (i.e. that are not standard AA, RNA, DNA or special masks).
            Therefore only specify this if you want to deterministically atomize certain standard tokens. Defaults to None.
        - loader_params: unused
        - extra_pre_crop_transforms (omegaconf.dictconfig.DictConfig | dict ): Additional Transforms to run before the crop
            Should be initialized like all the other datahub stuff with _target_
        - conditions_dict_atom_array_keys (list[str]): Fields from AtomArray to put into conditions_dict

    Returns:
        Compose: A composed transformation pipeline.
    """
    if crop_contiguous_probability > 0 or crop_spatial_probability > 0:
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    if unresolved_ligand_atom_limit is None:
        unresolved_ligand_atom_limit = 1_000_000
    elif unresolved_ligand_atom_limit < 1:
        unresolved_ligand_atom_limit = np.ceil(crop_size * unresolved_ligand_atom_limit)

    encoding = RF2AA_ATOM36_ENCODING

    transforms = [
        # ============================================
        # 1. Prepare the structure
        # ============================================
        # ...remove hydrogens for efficiency
        RemoveHydrogens(),  # * (already cached from the parser)
        RemoveTerminalOxygen(),  # RF2AA does not encode terminal oxygen for AA residues.
        RemoveUnresolvedPNUnits(),  # Remove PN units that are unresolved early (and also after cropping)
        # ...remove unsupported chain types
        RemoveUnsupportedChainTypes(),  # e.g., DNA_RNA_HYBRID, POLYPEPTIDE_D, etc.
        # RaiseIfTooManyAtoms(max_atoms=max_allowed_num_atoms),
        HandleUndesiredResTokens(undesired_res_names),  # e.g., non-standard residues
        # ...filtering
        # RemoveUnresolvedLigandAtomsIfTooMany(
        #     unresolved_ligand_atom_limit=unresolved_ligand_atom_limit
        # ),  # Crop size * 10%
        # ...add an annotation that is a unique atom ID across the entire structure, and won't change as we crop or reorder the AtomArray
        AddGlobalAtomIdAnnotation(),
        # ...add additional annotations that we'll use later
        AddProteinTerminiAnnotation(),  # e.g., N-terminus, C-terminus
        AddWithinPolyResIdxAnnotation(),  # add annotation relevant for matching MSA and template info
        # ============================================
        # 2. Perform relevant atomizations to arrive at final tokens
        # ============================================
        # ...sample residues to atomize (in RF2AA, with some probability, we atomize protein residues randomly)
        # TODO: SampleResiduesToAtomize
        # ...handle covalent modifications by atomizing and attaching the bonded residue to the non-polymer
        FlagAndReassignCovalentModifications(),
        # ...flag non-polymers for atomization (in case there are polymer tokens outside of a polymer)
        FlagNonPolymersForAtomization(),
        # ...atomize
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_atomize=res_names_to_atomize,
            res_names_to_ignore=encoding.tokens,
            move_atomized_part_to_end=True,
        ),
        SortLikeRF2AA(),
        AddGlobalTokenIdAnnotation(),
    ]

    for name, transform in extra_pre_crop_transforms.items():
        transforms += [transform]

    if crop_contiguous_probability > 0 or crop_spatial_probability > 0:
        contiguous_crop_transform = CropContiguousLikeAF3(crop_size=crop_size, keep_uncropped_atom_array=True)
        spatial_crop_transform = CropSpatialLikeAF3(
            crop_size=crop_size, crop_center_cutoff_distance=crop_center_cutoff_distance, keep_uncropped_atom_array=True
        )
        if crop_contiguous_probability > 0 and crop_spatial_probability > 0:
            transforms += [
                # ...crop around our query pn_unit(s) early, since we don't need the full structure moving forward
                RandomRoute(
                    transforms=[
                        contiguous_crop_transform,
                        spatial_crop_transform,
                    ],
                    probs=[crop_contiguous_probability, crop_spatial_probability],
                ),
            ]
        elif crop_contiguous_probability > 0:
            transforms.append(contiguous_crop_transform)
        elif crop_spatial_probability > 0:
            transforms.append(spatial_crop_transform)

    transforms += [
        AddPostCropMoleculeEntityToFreeFloatingLigands(),
        EncodeAtomArray(encoding),
        AddTokenBondAdjacency(),
        AddRF2AABondFeaturesMatrix(),
        # ============================================
        # 7. Convert to torch and featurize
        # ============================================
        ConvertToTorch(
            keys=[
                "encoded",
                "rf2aa_bond_features_matrix",
            ]
        ),
        # ============================================
        # 9. Aggregate features into RF_Diffusion Indep and atom mask
        # ============================================
        BackwardCompatibleDataLoaderProcessOut(conditions_dict_atom_array_keys=conditions_dict_atom_array_keys),

    ]
    
    return Compose(transforms, track_rng_state=True)
