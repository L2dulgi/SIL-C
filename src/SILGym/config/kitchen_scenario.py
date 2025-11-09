"""
Kitchen scenario configuration module.

This module provides configuration functions for creating training scenarios
in the kitchen domain, supporting various task partitioning strategies and
phase ordering modes.
"""

import os
import re
import math
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from enum import Enum

import numpy as np
try:
    from scipy.cluster.hierarchy import linkage, fcluster
except ImportError:
    raise ImportError("This module requires scipy. Please install via: pip install scipy")

from SILGym.config.skill_stream_config import (
    SkillPhaseConfig,
    SkillStreamConfig,
)
from SILGym.config.data_paths import (
    skill_dataset_path,
    RAW_SKILL_DATASET_PATH,
    DEFAULT_KITCHEN_VIS_MODEL,
    KITCHEN_VIS_DATASET_ROOT,
    KITCHEN_STUDIO_DATASET_ROOT,
    get_kitchen_vis_dataset_root,
    get_kitchenstudio_vis_dataset_root,
    convert_skill_path_to_embed,
)
from SILGym.config.variant_registry import (
    resolve_kitchen_vis_variant,
    KITCHEN_VIS_ENV_MAP,
    kitchen_vis_registry,
)


# ============================================================================
# Constants and Configuration
# ============================================================================

# Use variant registry for embedding dimensions and model mappings
KITCHEN_VIS_MODEL_EMBEDDINGS: Dict[str, int] = kitchen_vis_registry.VARIANTS
KITCHEN_VIS_MODEL_TO_DINOV3: Dict[str, str] = kitchen_vis_registry.VARIANT_TO_DINOV3

# Proprioceptive state features included alongside visual embeddings
DEFAULT_KITCHEN_PROPRIO_KEYS: Tuple[str, ...] = ("robot_states", "ee_states", "gripper_states")
KITCHEN_PROPRIO_KEY_DIMS: Dict[str, int] = {
    "joint_states": 7,
    "ee_states": 6,
    "gripper_states": 2,
    "robot_states": 9,
    "object_states": 0,
    "oracle_state": 60,  # oracle fallback; overridden when oracle proprio is enabled
}
DEFAULT_KITCHEN_EMBED_CAMERA_KEYS: Tuple[str, ...] = ("agentview_rgb_dinov3", "eye_in_hand_rgb_dinov3")
DEFAULT_KITCHENSTUDIO_EMBED_CAMERA_KEYS: Tuple[str, ...] = (
    "agentview_rgb_dinov3",
    "eye_in_hand_rgb_dinov3",
    "ovens_rgb_dinov3",
    "pannels_rgb_dinov3",
    "cabinets_rgb_dinov3",
)
KITCHEN_VIS_DEFAULT_IMAGE_SIZE = 224


def _resolve_proprio_keys(
    requested: Optional[Tuple[str, ...]],
) -> Tuple[str, ...]:
    """Normalize the requested proprio key tuple (falling back to defaults)."""
    if requested is None:
        return DEFAULT_KITCHEN_PROPRIO_KEYS
    normalized = tuple(str(key) for key in requested if str(key))
    if not normalized:
        raise ValueError("Proprio key tuple cannot be empty.")
    unknown = [key for key in normalized if key not in KITCHEN_PROPRIO_KEY_DIMS]
    if unknown:
        raise ValueError(
            f"Unknown proprio keys {unknown}. Supported keys: {sorted(KITCHEN_PROPRIO_KEY_DIMS.keys())}"
        )
    return normalized


def _compute_proprio_dim(keys: Tuple[str, ...], *, use_oracle: bool = False) -> int:
    """
    Compute the concatenated proprio dimensionality for the provided key tuple.

    When oracle proprio is enabled, defer to the oracle dimension regardless of the
    requested keys (the dataloader/evaluator collapse to a single oracle channel).
    """
    if use_oracle:
        # Oracle proprio is handled separately downstream; keep dimension consistent.
        return KITCHEN_PROPRIO_KEY_DIMS["oracle_state"]
    return sum(int(KITCHEN_PROPRIO_KEY_DIMS.get(key, 0)) for key in keys)

# Example task definitions
EXAMPLE_TASKS = [
    'microwave-bottom burner-light switch-slide cabinet.pkl',
    'kettle-top burner-light switch-slide cabinet.pkl',
    'microwave-kettle-bottom burner-hinge cabinet.pkl',
    'bottom burner-top burner-light switch-slide cabinet.pkl',
    'microwave-kettle-top burner-hinge cabinet.pkl',
    'kettle-bottom burner-top burner-hinge cabinet.pkl',
    'kettle-bottom burner-slide cabinet-hinge cabinet.pkl',
    'kettle-bottom burner-light switch-hinge cabinet.pkl',
    'kettle-bottom burner-top burner-light switch.pkl',
    'microwave-bottom burner-top burner-light switch.pkl',
    'microwave-bottom burner-slide cabinet-hinge cabinet.pkl',
    'kettle-light switch-slide cabinet-hinge cabinet.pkl',
    'microwave-kettle-top burner-light switch.pkl',
    'microwave-light switch-slide cabinet-hinge cabinet.pkl',
    'microwave-bottom burner-top burner-slide cabinet.pkl',
    'bottom burner-top burner-slide cabinet-hinge cabinet.pkl',
    'microwave-top burner-light switch-hinge cabinet.pkl',
    'microwave-kettle-slide cabinet-hinge cabinet.pkl',
    'kettle-bottom burner-top burner-slide cabinet.pkl',
    'kettle-bottom burner-light switch-slide cabinet.pkl',
    'microwave-kettle-light switch-hinge cabinet.pkl',
    'microwave-bottom burner-top burner-hinge cabinet.pkl',
    'microwave-kettle-bottom burner-slide cabinet.pkl',
    'microwave-kettle-light switch-slide cabinet.pkl'
]

# Example skill definitions for incremental learning
EXAMPLE_SKILLS = [
    ["microwave.pkl"],
    ["kettle.pkl", "bottom burner.pkl"],
    ["top burner.pkl", "light switch.pkl"],
    ["slide cabinet.pkl", "hinge cabinet.pkl"]
]

# Permutation lists for scenario variations
PERMUTATION_LISTS = [
    [3, 2, 1, 0],
    [2, 3, 0, 1],
    [1, 0, 3, 2],
]


# ============================================================================
# Enums for Configuration Options
# ============================================================================

class CompatibilityMode(Enum):
    """Phase ordering modes for scenario assembly."""
    DEFAULT = "default"  # Interleaved order
    SYNCHRONIZATION = "Synchronization"  # Decoder, policies, then repeat for each decoder


class TaskPartitionOption(Enum):
    """Options for task partitioning."""
    CLUSTER = "cluster"  # Cluster by similarity
    OVERLAP = "overlap"  # Include previous chunk
    ACCUMULATE = "accumulate"  # Include all previous chunks


# ============================================================================
# Utility Functions
# ============================================================================

def abbreviate_task_name(filename: str) -> str:
    """Convert task filename to abbreviation: 'bottom burner-top burner.pkl' -> 'bt'"""
    return "".join(sg.strip()[0] for sg in filename.replace(".pkl", "").split("-"))


def parse_subgoals(task_name: str) -> Set[str]:
    """Extract subgoals from task name."""
    return set(p.strip() for p in task_name.replace('.pkl', '').split('-'))


def jaccard_distance(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard distance between two sets."""
    if not set_a and not set_b:
        return 0.0
    return 1.0 - len(set_a & set_b) / len(set_a | set_b)


# ============================================================================
# Task Clustering Functions
# ============================================================================

def cluster_tasks_by_jaccard(all_tasks: List[str], n_clusters: int) -> List[List[str]]:
    """Cluster tasks based on Jaccard distance of their subgoals."""
    subgoal_sets = [parse_subgoals(t) for t in all_tasks]
    
    # Build distance array
    dist_list = [jaccard_distance(subgoal_sets[i], subgoal_sets[j]) 
                 for i in range(len(all_tasks)) 
                 for j in range(i + 1, len(all_tasks))]
    
    # Cluster and group
    cluster_labels = fcluster(linkage(np.array(dist_list), method='average'), 
                            t=n_clusters, criterion='maxclust')
    
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_dict[label].append(all_tasks[idx])
    
    return [cluster_dict[label] for label in sorted(cluster_dict.keys())]


# ============================================================================
# Scenario Assembly Functions
# ============================================================================

def assemble_scenario_phases(
    group_phases: List[Tuple[SkillPhaseConfig, List[SkillPhaseConfig]]], 
    compatibility: str = None
) -> List[SkillPhaseConfig]:
    """Assemble final training scenario from grouped phases."""
    import copy
    
    # Assign unique policy IDs
    policy_id = 0
    for _, policies in group_phases:
        for phase in policies:
            policy_id += 1
            phase.unique_policy_id = policy_id
    
    # Extract lists
    decoders = [dec for dec, _ in group_phases]
    all_policies = [p for _, policies in group_phases for p in policies]
    policy_refs = [f"policy_{p.unique_policy_id}" for p in all_policies]
    
    # Build scenario based on mode
    mode = CompatibilityMode(compatibility) if compatibility else CompatibilityMode.DEFAULT
    
    if mode == CompatibilityMode.SYNCHRONIZATION:
        # Most common case - synchronization mode
        scenario = []
        if group_phases:
            scenario.append(group_phases[0][0])
            scenario.extend(all_policies)
            for dec, _ in group_phases[1:]:
                dec.eval_ref_policies = policy_refs
                scenario.append(dec)
                scenario.extend([copy.deepcopy(p) for p in all_policies])
    elif mode == CompatibilityMode.DEFAULT:
        # Interleaved mode
        scenario = []
        for dec, policies in group_phases:
            scenario.extend([dec] + policies)
    else:
        # Other modes not used in current codebase
        raise NotImplementedError(f"Mode {mode} not currently used")
    
    # Update policy phase names
    current_skill_id = None
    for phase in scenario:
        if 'decoder' in phase.train_targets and phase.train_tasks:
            current_skill_id = phase.phase_name
        elif 'policy' in phase.train_targets:
            phase.phase_name = f"policy_{phase.unique_policy_id}/{current_skill_id}"
    
    return scenario


# ============================================================================
# Main Scenario Creation Functions
# ============================================================================

def create_kitchen_scenario(
    all_tasks: List[str],
    skill_dataset_path: str,
    phase_num: int = 4,
    options: List[str] = None,
    compatibility: str = None,
    permute_list: List[int] = None
) -> List[SkillPhaseConfig]:
    """
    Create a training scenario for the kitchen domain.
    
    Args:
        all_tasks: List of task filenames
        skill_dataset_path: Path to skill datasets
        phase_num: Number of phases to create
        options: List of options ('cluster', 'overlap', 'accumulate')
        compatibility: Phase ordering mode
        permute_list: Optional permutation for group ordering
        
    Returns:
        List of phase configurations
    """
    if options is None:
        options = []
    
    # Partition tasks
    if TaskPartitionOption.CLUSTER.value in options:
        groups = cluster_tasks_by_jaccard(all_tasks, n_clusters=phase_num)
    else:
        # Index-based chunking
        chunk_size = math.ceil(len(all_tasks) / phase_num)
        groups = []
        for i in range(phase_num):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_tasks))
            groups.append(all_tasks[start_idx:end_idx])
    
    # Apply permutation if specified
    if permute_list is not None:
        if len(permute_list) != len(groups):
            raise ValueError("Permutation list length must match number of groups")
        groups = [groups[i] for i in permute_list]
    
    # Helper to get decoder dataset for a group
    def get_decoder_dataset(idx: int) -> List[str]:
        if TaskPartitionOption.ACCUMULATE.value in options:
            return [task for g in groups[:idx + 1] for task in g]
        elif TaskPartitionOption.OVERLAP.value in options and idx > 0:
            return groups[idx - 1] + groups[idx]
        return groups[idx]
    
    # Build phases for each group
    group_phases = []
    for group_idx, group_tasks in enumerate(groups):
        # Decoder phase
        decoder_dataset = get_decoder_dataset(group_idx)
        dec_paths = [os.path.join(skill_dataset_path, t) for t in decoder_dataset]
        
        dec_phase = SkillPhaseConfig(
            phase_name=f"pre_{group_idx}",
            train_targets=['decoder', 'interface'],
            dataset_paths=dec_paths,
            train_tasks=[abbreviate_task_name(os.path.basename(p)) for p in dec_paths]
        )
        
        # Policy phases
        policy_phases = [
            SkillPhaseConfig(
                phase_name=f"task_{group_idx}_{i}",
                train_targets=['policy'],
                dataset_paths=[os.path.join(skill_dataset_path, task_name)],
                train_tasks=[abbreviate_task_name(task_name)],
                eval_tasks=[{'data_name': abbreviate_task_name(task_name)}]
            )
            for i, task_name in enumerate(group_tasks)
        ]
        
        group_phases.append((dec_phase, policy_phases))
    
    # Assemble final scenario
    return assemble_scenario_phases(group_phases, compatibility)


def create_skill_incremental_scenario(
    example_skills: List[List[str]],
    example_tasks: List[str]
) -> List[SkillPhaseConfig]:
    """Create an incremental learning scenario with skill-based phases."""
    fixed_policy_ids = [f"policy_{i}" for i in range(1, 25)]
    final_scenario = []
    
    for idx, skills_for_phase in enumerate(example_skills):
        # Decoder phase
        skill_paths = [os.path.join(RAW_SKILL_DATASET_PATH, skill) for skill in skills_for_phase]
        
        dec_phase = SkillPhaseConfig(
            phase_name=f"pre_{idx}",
            train_targets=['decoder', 'interface'],
            dataset_paths=skill_paths,
            train_tasks=[abbreviate_task_name(os.path.basename(skill)) for skill in skills_for_phase],
            eval_ref_policies=[] if idx == 0 else fixed_policy_ids[:]
        )
        final_scenario.append(dec_phase)
        
        # Policy phases
        for j, task_file in enumerate(example_tasks):
            policy_phase = SkillPhaseConfig(
                phase_name=f"policy_{j + 1}/{dec_phase.phase_name}",
                train_targets=['policy'],
                dataset_paths=[os.path.join(skill_dataset_path, task_file)],
                train_tasks=[abbreviate_task_name(task_file)],
                eval_tasks=[{'data_name': abbreviate_task_name(task_file)}]
            )
            policy_phase.unique_policy_id = j + 1
            final_scenario.append(policy_phase)
    
    return final_scenario


# ============================================================================
# Predefined Scenarios
# ============================================================================

def _create_sync_scenario(tasks=EXAMPLE_TASKS, phase_num=4, permute_idx=None):
    """Helper to create synchronized scenarios."""
    return create_kitchen_scenario(
        all_tasks=tasks,
        skill_dataset_path=skill_dataset_path,
        phase_num=phase_num,
        options=['cluster'],
        compatibility="Synchronization",
        permute_list=PERMUTATION_LISTS[permute_idx] if permute_idx is not None else None
    )

# Create scenarios lazily to improve import time
KITCHEN_SCENARIOS = {
    'kitchenem': lambda: _create_sync_scenario(),
    'debug': lambda: _create_sync_scenario(EXAMPLE_TASKS[:4], phase_num=4),
    'debugmini': lambda: _create_sync_scenario(EXAMPLE_TASKS[:2], phase_num=2),
    'objective_p1': lambda: _create_sync_scenario(permute_idx=0),
    'objective_p2': lambda: _create_sync_scenario(permute_idx=1),
    'objective_p3': lambda: _create_sync_scenario(permute_idx=2),
    'kitchenex': lambda: create_skill_incremental_scenario(EXAMPLE_SKILLS, EXAMPLE_TASKS)
}

# For backward compatibility - create scenarios on first import
def __getattr__(name):
    """Lazy loading for backward compatibility."""
    mapping = {
        'KITCHEN_SCENARIO_OBJ_SYNC': 'kitchenem',
        'KITCHEN_SCENARIO_OBJ_DEBUG': 'debug',
        'KITCHEN_SCENARIO_OBJ_DEBUG': 'debugmini',
        'KITCHEN_SCENARIO_OBJ_SYNC_P1': 'objective_p1',
        'KITCHEN_SCENARIO_OBJ_SYNC_P2': 'objective_p2',
        'KITCHEN_SCENARIO_OBJ_SYNC_P3': 'objective_p3',
        'KITCHEN_SCENARIO_SKILL_INCREMENTAL': 'kitchenex'
    }
    if name in mapping:
        scenario = KITCHEN_SCENARIOS[mapping[name]]()
        globals()[name] = scenario
        return scenario
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ============================================================================
# Main Scenario Factory Function
# ============================================================================

def kitchen_scenario(
    scenario_type: str = 'objective',
    sync_type: str = 'sync',
    *,
    use_embeddings: bool = False,
    embed_variant: Optional[str] = None,
    env_alias: Optional[str] = None,
    proprio_keys: Optional[Tuple[str, ...]] = ('robot_states',), # only robot state is good...
    camera_keys: Optional[Tuple[str, ...]] = None,
) -> SkillStreamConfig:
    """Create a SkillStreamConfig based on scenario, sync type, and dataset format."""
    logger = logging.getLogger(__name__)
    
    # Normalize inputs
    sync_type = sync_type.lower()
    scenario_type = scenario_type.lower()
    
    logger.info(f"[KitchenScenario] Initializing with sync_type='{sync_type}', scenario_type='{scenario_type}'")
    
    # Support sync mode with optional modifiers (e.g., 'sync-oracle')
    _sync_tokens = [tok for tok in re.split(r'[-_]+', sync_type) if tok]
    _sync_core = _sync_tokens[0] if _sync_tokens else 'sync'
    _sync_mods = set(_sync_tokens[1:]) if len(_sync_tokens) > 1 else set()
    if _sync_core != 'sync':
        raise ValueError(f"Only 'sync' mode is supported, got '{sync_type}'")
    
    # Get scenario (allow modifiers like '<name>-oracle')
    _scn_tokens = [tok for tok in re.split(r'[-_]+', scenario_type) if tok]
    _scn_core = _scn_tokens[0] if _scn_tokens else scenario_type
    _scn_mods = set(_scn_tokens[1:]) if len(_scn_tokens) > 1 else set()
    if _scn_core not in KITCHEN_SCENARIOS:
        if _scn_core == 'quality':
            raise NotImplementedError("Quality scenario is not implemented.")
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    oracle_requested = ('oracle' in _sync_mods) or ('oracle' in _scn_mods)

    # Create scenario lazily
    datastream = KITCHEN_SCENARIOS[_scn_core]()
    logger.info("Kitchen scenario successfully created.")

    metadata: Dict[str, object] = {
        "base_environment": "kitchen",
        "dataset_format": "pkl",
        "dataset_root": skill_dataset_path,
    }

    resolved_proprio_keys = _resolve_proprio_keys(proprio_keys)
    metadata["kitchen_proprio_keys"] = resolved_proprio_keys
    metadata.setdefault("eval_proprio_keys", resolved_proprio_keys)
    proprio_dim = _compute_proprio_dim(resolved_proprio_keys, use_oracle=oracle_requested)
    metadata["kitchen_proprio_dim"] = proprio_dim

    resolved_alias = (env_alias or ("kitchen_vis" if use_embeddings else "kitchen")).lower()
    metadata["env_alias"] = resolved_alias

    if use_embeddings:
        variant = (embed_variant or resolve_kitchen_vis_variant(resolved_alias)).lower()

        # Detect kitchenstudio_vis environment
        is_studio = "studio" in resolved_alias or "kitchenstudio" in resolved_alias
        if is_studio:
            dataset_root = get_kitchenstudio_vis_dataset_root(variant)
            metadata["is_kitchenstudio"] = True
        else:
            dataset_root = get_kitchen_vis_dataset_root(variant)
            metadata["is_kitchenstudio"] = False

        if camera_keys is not None:
            resolved_camera_keys = tuple(str(key) for key in camera_keys if str(key))
            if not resolved_camera_keys:
                raise ValueError("Camera key tuple cannot be empty when provided.")
        else:
            existing = metadata.get("kitchen_camera_keys")
            if isinstance(existing, (list, tuple)) and existing:
                resolved_camera_keys = tuple(str(key) for key in existing)
            else:
                # Use studio cameras if kitchenstudio_vis, otherwise default 2 cameras
                resolved_camera_keys = DEFAULT_KITCHENSTUDIO_EMBED_CAMERA_KEYS if is_studio else DEFAULT_KITCHEN_EMBED_CAMERA_KEYS
        metadata["kitchen_camera_keys"] = resolved_camera_keys
        metadata.setdefault(
            "kitchen_obs_modalities",
            resolved_camera_keys + metadata["kitchen_proprio_keys"],
        )

        # Update dataset paths to point to embedded HDF5 files
        for phase in datastream:
            if not phase.dataset_paths:
                continue
            phase.dataset_paths = [
                path if (("kitchen_lerobot_embed" in path or "kitchenstudio_embed" in path) and path.endswith(".hdf5"))
                else convert_skill_path_to_embed(path, dataset_root)
                for path in phase.dataset_paths
            ]

        embedding_dim = KITCHEN_VIS_MODEL_EMBEDDINGS[variant]
        # Calculate obs_dim based on number of cameras
        num_cameras = len(resolved_camera_keys)
        metadata.update({
            "dataset_format": "hdf5",
            "dataset_root": dataset_root,
            "kitchen_embedding_dim": embedding_dim,
            "kitchen_obs_dim": num_cameras * embedding_dim + proprio_dim,
            "kitchen_vis_variant": variant,
            "kitchen_dinov3_model": KITCHEN_VIS_MODEL_TO_DINOV3.get(variant, "ViT-B/16"),
            "kitchen_embed_image_size": KITCHEN_VIS_DEFAULT_IMAGE_SIZE,
        })

        # If sync_type or scenario_type contains '-oracle', enable oracle proprio options for kitchen_vis
        if oracle_requested:
            metadata["kitchen_use_oracle_proprio"] = True
            metadata["eval_use_oracle_proprio"] = True
            if "kitchen_oracle_key_name" not in metadata:
                metadata["kitchen_oracle_key_name"] = "states"

    return SkillStreamConfig(
        datastream=datastream,
        environment='kitchen',
        scenario_type=_scn_core,
        sync_type=sync_type,
        metadata=metadata,
    )


if __name__ == "__main__" :
    kitchen_scenario(scenario_type='debug').print_stream()
