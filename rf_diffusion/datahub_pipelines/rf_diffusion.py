
import numpy as np
from cifutils.constants import AF3_EXCLUDED_LIGANDS

from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinPolyResIdxAnnotation,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
    RemoveUnsupportedChainTypes,
    # RemoveUnresolvedLigandAtomsIfTooMany,
    SortLikeRF2AA,
)
from datahub.transforms.atomize import AtomizeResidues, FlagNonPolymersForAtomization
from datahub.transforms.base import Compose, ConvertToTorch, RandomRoute
from datahub.transforms.bonds import (
    AddRF2AABondFeaturesMatrix,
    AddTokenBondAdjacency,
)
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.encoding import EncodeAtomArray

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
) -> Compose:
    """
    Creates a transformation pipeline for the RF2AA model, applying a series of transformations to the input data.

    Args:
        - protein_msa_dirs (list[dict]): The directories containing the protein MSAs and their associated file types,
            as a list of dictionaries. If multiple directories are provided, we will search all of them. Note that:
            (a) the directory structure must be flat (i.e., no subdirectories), (b) the files must be named using the
            SHA-256 hash of the sequence (see `hash_sequence` in `utils/misc`), and (c) order matters - we will search the
            directories in the order they are provided, and return the first match
        - rna_msa_dirs (list[dict]): The directories containing the RNA MSAs and their associated file types, as a list
            of dictionaries. See `protein_msa_dirs` for directory structure details.
        - n_recycles (int, optional): Number of recycles for the MSA featurization. Defaults to 5.
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
        - res_names_to_atomize (list[str], optional): List of residue names to *always* atomize. Note that RF2AA already
            atomizes all residues that are not in the encoding (i.e. that are not standard AA, RNA, DNA or special masks).
            Therefore only specify this if you want to deterministically atomize certain standard tokens. Defaults to None.
        - max_msa_sequences (int, optional): Maximum number of MSA sequences to load. Defaults to 10,000.
        - dense_msa (bool, optional): Whether to use dense MSA pairing. Defaults to True.
        - n_msa_cluster_representatives (int, optional): Number of MSA cluster representatives to select. Defaults to 100.
        - msa_n_extra_rows (int, optional): Number of extra rows for MSA. Defaults to 100.
        - msa_mask_probability (float, optional): Probability of masking MSA sequences according to `msa_mask_behavior_probs`.
            Defaults to 0.15.
        - msa_mask_behavior_probs (dict[str, float], optional): Probabilities for different MSA mask behaviors.
            Defaults to {"replace_with_random_aa": 0.1, "replace_with_msa_profile": 0.1, "do_not_replace": 0.1},
            which is the BERT style masking.
        - order_independent_atom_frame_prioritization (bool, optional): Whether to prioritize order-independent atom frames.
            Defaults to True.
        - n_template (int, optional): Number of templates to use. Defaults to 5.
        - pick_top_templates (bool, optional): Whether to pick the top templates if there are more than `n_template`. If
            False, the templates are selected randomly among all templates. Defaults to False.
        - template_max_seq_similarity (float, optional): Maximum sequence similarity cutoff for templates.
            Defaults to 60.0.
        - template_min_seq_similarity (float, optional): Minimum sequence similarity cutoff for templates.
            Defaults to 10.0.
        - template_min_length (int, optional): Minimum length cutoff for templates. Defaults to 10.
        - max_automorphs (int, optional): Maximum number of automorphs after which to cap small molecule ligand
            symmetry resolution. Defaults to 1,000.
        - max_isomorphs (int, optional): Maximum number of polymer isomorphs after which to cap symmetry resolution.
            Defaults to 1,000.
        - use_negative_interface_examples (bool, optional): Whether to use negative interface examples. Defaults to False.
        - unclamp_loss_probability (float, optional): Probability of unclamping the loss during training. Defaults to 0.1.
        - black_hole_init (bool, optional): Whether to use black hole initialization. Defaults to True.
        - black_hole_init_noise_scale (float, optional): Noise scale for black hole initialization. Defaults to 5.0.
        - msa_cache_dir (PathLike | str | None, optional): Directory to cache the MSAs. Defaults to None.
        - assert_rf2aa_assumptions (bool, optional): Whether to assert the RF2AA assumptions that need to be true
            to guarantee a successful forward & backward pass. Defaults to True.
        - convert_feats_to_rf2aa_input_tuple (bool, optional): Whether to convert the features to the RF2AAInputs format.
            Defaults to True.
        - p_is_ss_example (float): Probability we show any secondary structure at all.
        - p_is_adj_example (float): Probability we show any adjacency matrix at all.
        - ss_min_mask (float): Minimum fraction of ss that will be converted to SS_MASK.
        - ss_max_mask (float): Maximum fraction of ss that will be converted to SS_MASK.
        - adj_min_mask (float): Minimum fraction of adj that will be directly converted to ADJ_MASK.
        - adj_max_mask (float): Maximum fraction of adj that will be directly converted to ADJ_MASK.


    For more details on the parameters, see the RF2AA paper and the documentation for the respective Transforms.

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
        AtomizeResidues(
            atomize_by_default=True,
            res_names_to_atomize=res_names_to_atomize,
            res_names_to_ignore=encoding.tokens,
            move_atomized_part_to_end=True,
        ),
        SortLikeRF2AA(),
        AddGlobalTokenIdAnnotation(),
    ]

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
        BackwardCompatibleDataLoaderProcessOut(),

    ]
    
    return Compose(transforms, track_rng_state=True)
