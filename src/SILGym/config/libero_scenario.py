import os
import json
from typing import List, Tuple, Dict
from SILGym.config.skill_stream_config import SkillPhaseConfig, SkillStreamConfig
from SILGym.config.kitchen_scenario import assemble_scenario_phases
from SILGym.utils.logger import get_logger
from SILGym.config.data_paths import (
    LIBERO_SKILL_DATASET_PATH,
    DEFAULT_LIBERO_MODEL,
    get_libero_skill_dataset_path,
)
from SILGym.config.variant_registry import (
    libero_registry,
    LIBERO_ENV_MODEL_MAP,
    resolve_libero_variant,
)

# Model name to embedding dimension mapping (from DINOv3 models)
# Use variant registry for consistent definitions
LIBERO_MODEL_EMBEDDINGS = libero_registry.VARIANTS
LIBERO_MODEL_TO_DINOV3 = libero_registry.VARIANT_TO_DINOV3

LIBERO_EMBED_PROPRIO_DIM = 15  # proprioceptive features concatenated with vision embeddings

# libero_90 pretraining task groups by scene
LIBERO_90_KITCHEN_TASKS = list(range(46))  # tasks 0-45: KITCHEN_SCENE1-10
LIBERO_90_LIVING_ROOM_TASKS = list(range(46, 73))  # tasks 46-72: LIVING_ROOM_SCENE1-6
LIBERO_90_STUDY_TASKS = list(range(73, 90))  # tasks 73-89: STUDY_SCENE1-4
LIBERO_90_ALL_TASKS = list(range(90))  # all 90 tasks


def get_embedding_size(model_name: str) -> int:
    """Get the embedding dimension for a model name."""
    if model_name not in LIBERO_MODEL_EMBEDDINGS:
        raise ValueError(f"Unknown model: {model_name}. Must be one of: {list(LIBERO_MODEL_EMBEDDINGS.keys())}")
    return LIBERO_MODEL_EMBEDDINGS[model_name]

def get_libero_observation_dim(model_name: str) -> int:
    """Return the single-frame observation dimension for embedded Libero data."""
    embedding_dim = get_embedding_size(model_name)
    return 2 * embedding_dim + LIBERO_EMBED_PROPRIO_DIM

def create_custom_chunk_scenario(
    benchmark_name: str,
    chunk_sizes: List[int],
    model_name: str = DEFAULT_LIBERO_MODEL,
    compatibility: str = "Synchronization",
    mapping_dir: str = "data/libero/task_map"
) -> List[SkillPhaseConfig]:
    """
    Create a custom scenario with arbitrary chunk sizes for a specific benchmark.

    Args:
        benchmark_name: Name of benchmark ('libero_goal', 'libero_spatial', 'libero_object', 'libero_10')
        chunk_sizes: List of chunk sizes (e.g., [3, 3, 4] for 334 scenario)
        model_name: Model name ('base', 'large', 'smallplus')
        compatibility: Synchronization type
        mapping_dir: Directory containing task mapping files

    Returns:
        List of SkillPhaseConfig phases for the custom scenario

    Example:
        # Create a 2-3-5 scenario for libero_goal
        scenario = create_custom_chunk_scenario('libero_goal', [2, 3, 5])
    """
    if sum(chunk_sizes) != 10:
        raise ValueError(f"Chunk sizes must sum to 10, got {sum(chunk_sizes)}")

    chunks = []
    start_idx = 0
    for chunk_size in chunk_sizes:
        chunk = [(benchmark_name, i) for i in range(start_idx, start_idx + chunk_size)]
        chunks.append(chunk)
        start_idx += chunk_size

    return build_scenario_from_chunks(
        pre_train_chunks=chunks,
        task_seq_chunks=chunks,
        model_name=model_name,
        compatibility=compatibility,
        mapping_dir=mapping_dir
    )

def create_all_benchmarks_chunk_scenarios(
    chunk_sizes: List[int],
    model_name: str = DEFAULT_LIBERO_MODEL,
    compatibility: str = "Synchronization",
    mapping_dir: str = "data/libero/task_map"
) -> Dict[str, List[SkillPhaseConfig]]:
    """
    Create scenarios with the same chunk pattern for all benchmarks.

    Args:
        chunk_sizes: List of chunk sizes (e.g., [3, 3, 4] for 334 scenario)
        model_name: Model name ('base', 'large', 'smallplus')
        compatibility: Synchronization type
        mapping_dir: Directory containing task mapping files

    Returns:
        Dictionary mapping benchmark names to their scenario configurations

    Example:
        # Create 3-3-4 scenarios for all benchmarks
        all_334_scenarios = create_all_benchmarks_chunk_scenarios([3, 3, 4])
    """
    benchmarks = ['libero_goal', 'libero_spatial', 'libero_object', 'libero_10']
    scenarios = {}

    for benchmark in benchmarks:
        scenarios[benchmark] = create_custom_chunk_scenario(
            benchmark_name=benchmark,
            chunk_sizes=chunk_sizes,
            model_name=model_name,
            compatibility=compatibility,
            mapping_dir=mapping_dir
        )

    return scenarios

def build_scenario_from_chunks(
    pre_train_chunks: List[List[Tuple[str, int]]],
    task_seq_chunks: List[List[Tuple[str, int]]],
    skill_dataset_path: str | None = None,
    model_name: str = DEFAULT_LIBERO_MODEL,
    compatibility: str | None = None,
    mapping_dir: str | None = None
) -> List[SkillPhaseConfig]:
    """
    Given lists of pre-training and task-sequence chunks (as tuples of benchmark_name and task_id),
    construct a flat list of SkillPhaseConfig objects matching the Libero scenario structure.

    Args:
        pre_train_chunks: [[(benchmark_name, id), ...], ...]
        task_seq_chunks: same shape as pre_train_chunks
        skill_dataset_path: base path for the .pkl skill datasets (if None, uses model_name)
        model_name: model name ('base', 'large', or 'smallplus'), used if skill_dataset_path is None
        compatibility: if provided, applies assemble_scenario_phases to the groups
        mapping_dir: optional directory path containing JSON files named <benchmark_name>.json
                     each mapping file provides bddl_file names per task_id

    Returns:
        List of SkillPhaseConfig phases (decoder/interface then policy phases).
    """
    # Use model_name to construct path if not explicitly provided
    if skill_dataset_path is None:
        skill_dataset_path = get_libero_skill_dataset_path(model_name)
    # Prepare mapping for each benchmark
    bddl_map_by_benchmark: Dict[str, Dict[int, str]] = {}
    if mapping_dir:
        for chunk in pre_train_chunks + task_seq_chunks:
            for bm_name, _ in chunk:
                if bm_name in bddl_map_by_benchmark:
                    continue
                json_path = os.path.join(mapping_dir, f"{bm_name}.json")
                mapping: Dict[int, str] = {}
                if os.path.isfile(json_path):
                    with open(json_path, "r", encoding="utf-8") as mf:
                        loaded = json.load(mf)
                    # support dict-of-dicts or list-of-dicts
                    if isinstance(loaded, dict):
                        for key_str, entry in loaded.items():
                            try:
                                tid = int(key_str)
                            except ValueError:
                                continue
                            fname = entry.get('bddl_file', '')
                            mapping[tid] = os.path.splitext(fname)[0]
                    elif isinstance(loaded, list):
                        for entry in loaded:
                            tid = entry.get('task_id')
                            if isinstance(tid, int) and 'bddl_file' in entry:
                                mapping[tid] = os.path.splitext(entry['bddl_file'])[0]
                bddl_map_by_benchmark[bm_name] = mapping

    group_phases: List[Tuple[SkillPhaseConfig, List[SkillPhaseConfig]]] = []
    for phase_idx, pre_chunk in enumerate(pre_train_chunks):
        dec_paths: List[str] = []
        # build decoder+interface dataset paths
        for bm_name, task_id in pre_chunk:
            mapping = bddl_map_by_benchmark.get(bm_name, {})
            base = mapping.get(task_id, f"{bm_name}-{task_id}")
            dec_paths.append(os.path.join(skill_dataset_path, f"{bm_name}/{base}_demo.hdf5"))

        dec = SkillPhaseConfig(
            phase_name=f"pre_{phase_idx}",
            train_targets=["decoder", "interface"],
            dataset_paths=dec_paths,
            train_tasks=[f"{bm}-{tid}" for bm, tid in pre_chunk],
            
        )

        policies: List[SkillPhaseConfig] = []
        # build policy dataset paths
        for task_idx, (bm_name, task_id) in enumerate(task_seq_chunks[phase_idx]):
            mapping = bddl_map_by_benchmark.get(bm_name, {})
            base = mapping.get(task_id, f"{bm_name}-{task_id}")
            path = os.path.join(skill_dataset_path, f"{bm_name}/{base}_demo.hdf5")
            tag = f"{bm_name}-{task_id}"
            policies.append(
                SkillPhaseConfig(
                    phase_name=f"task_{phase_idx}_{task_idx}",
                    train_targets=["policy"],
                    dataset_paths=[path],
                    train_tasks=[tag],
                    eval_tasks=[{"data_name": tag}],
                    
                )
            )

        group_phases.append((dec, policies))

    if compatibility:
        return assemble_scenario_phases(group_phases, compatibility)
    else:
        scenario: List[SkillPhaseConfig] = []
        for dec_phase, policy_list in group_phases:
            scenario.append(dec_phase)
            scenario.extend(policy_list)
        return scenario

# ==============================================================================
# Pre-built scenarios for all three models (base, large, smallplus)
# ==============================================================================

# Helper function to build all model variants of a scenario
def _build_scenario_variants(
    pre_train_chunks: List[List[Tuple[str, int]]],
    task_seq_chunks: List[List[Tuple[str, int]]],
    mapping_dir: str = "data/libero/task_map",
    compatibility: str = "Synchronization"
) -> Dict[str, List[SkillPhaseConfig]]:
    """Build scenario for all three model sizes."""
    return {
        model_name: build_scenario_from_chunks(
            pre_train_chunks=pre_train_chunks,
            task_seq_chunks=task_seq_chunks,
            model_name=model_name,
            mapping_dir=mapping_dir,
            compatibility=compatibility
        )
        for model_name in LIBERO_MODEL_EMBEDDINGS.keys()
    }

def _build_pretraining_scenario_variants(
    benchmark_name: str,
    task_ids: List[int],
    mapping_dir: str = "data/libero/task_map"
) -> Dict[str, List[SkillPhaseConfig]]:
    """Build pretraining scenario for all three model sizes.

    Pretraining scenarios only train decoder+interface without policy,
    and have no evaluation tasks.

    Args:
        benchmark_name: Name of benchmark (e.g., 'libero_90')
        task_ids: List of task IDs to include in pretraining
        mapping_dir: Directory containing task mapping files

    Returns:
        Dictionary mapping model names to their scenario configurations
    """
    scenarios = {}

    for model_name in LIBERO_MODEL_EMBEDDINGS.keys():
        skill_dataset_path = get_libero_skill_dataset_path(model_name)

        # Prepare mapping
        bddl_map: Dict[int, str] = {}
        json_path = os.path.join(mapping_dir, f"{benchmark_name}.json")
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as mf:
                loaded = json.load(mf)
            if isinstance(loaded, dict):
                for key_str, entry in loaded.items():
                    try:
                        tid = int(key_str)
                    except ValueError:
                        continue
                    fname = entry.get('bddl_file', '')
                    bddl_map[tid] = os.path.splitext(fname)[0]

        # Build dataset paths for all tasks
        dec_paths: List[str] = []
        train_tasks: List[str] = []
        for task_id in task_ids:
            base = bddl_map.get(task_id, f"{benchmark_name}-{task_id}")
            dec_paths.append(os.path.join(skill_dataset_path, f"{benchmark_name}/{base}_demo.hdf5"))
            train_tasks.append(f"{benchmark_name}-{task_id}")

        # Create single decoder+interface phase with no evaluation
        dec_phase = SkillPhaseConfig(
            phase_name="pre_0",
            train_targets=["decoder", "interface"],
            dataset_paths=dec_paths,
            train_tasks=train_tasks,
            eval_tasks=None  # No evaluation for pretraining
        )

        scenarios[model_name] = [dec_phase]

    return scenarios

# ---------- TEST SCENARIOS ----------
_test_scenarios = _build_scenario_variants(
    pre_train_chunks=[[("libero_goal", i) for i in range(10)]],
    task_seq_chunks=[[("libero_goal", i) for i in range(10)]]
)
LIBERO_TEST_SCENARIO_BASE = _test_scenarios["base"]
LIBERO_TEST_SCENARIO_BASE[0].eval_tasks = None
LIBERO_TEST_SCENARIO_LARGE = _test_scenarios["large"]
LIBERO_TEST_SCENARIO_LARGE[0].eval_tasks = None
LIBERO_TEST_SCENARIO_SMALLPLUS = _test_scenarios["smallplus"]
LIBERO_TEST_SCENARIO_SMALLPLUS[0].eval_tasks = None

# ---------- MULTI SCENARIOS ----------
# Object Multi
_object_multi = _build_scenario_variants(
    pre_train_chunks=[[("libero_object", i) for i in range(10)]],
    task_seq_chunks=[[("libero_object", i) for i in range(10)]]
)
LIBERO_OBJECT_SCENARIO_MULTI_BASE = _object_multi["base"]
LIBERO_OBJECT_SCENARIO_MULTI_LARGE = _object_multi["large"]
LIBERO_OBJECT_SCENARIO_MULTI_SMALLPLUS = _object_multi["smallplus"]

# Goal Multi
_goal_multi = _build_scenario_variants(
    pre_train_chunks=[[("libero_goal", i) for i in range(10)]],
    task_seq_chunks=[[("libero_goal", i) for i in range(10)]]
)
LIBERO_GOAL_SCENARIO_MULTI_BASE = _goal_multi["base"]
LIBERO_GOAL_SCENARIO_MULTI_LARGE = _goal_multi["large"]
LIBERO_GOAL_SCENARIO_MULTI_SMALLPLUS = _goal_multi["smallplus"]

# Spatial Multi
_spatial_multi = _build_scenario_variants(
    pre_train_chunks=[[("libero_spatial", i) for i in range(10)]],
    task_seq_chunks=[[("libero_spatial", i) for i in range(10)]]
)
LIBERO_SPATIAL_SCENARIO_MULTI_BASE = _spatial_multi["base"]
LIBERO_SPATIAL_SCENARIO_MULTI_LARGE = _spatial_multi["large"]
LIBERO_SPATIAL_SCENARIO_MULTI_SMALLPLUS = _spatial_multi["smallplus"]

# Long Multi
_long_multi = _build_scenario_variants(
    pre_train_chunks=[[("libero_10", i) for i in range(10)]],
    task_seq_chunks=[[("libero_10", i) for i in range(10)]]
)
LIBERO_LONG_SCENARIO_MULTI_BASE = _long_multi["base"]
LIBERO_LONG_SCENARIO_MULTI_LARGE = _long_multi["large"]
LIBERO_LONG_SCENARIO_MULTI_SMALLPLUS = _long_multi["smallplus"]

# ---------- INCREMENTAL (SYNC) SCENARIOS ----------
# Goal Inc
_goal_inc = _build_scenario_variants(
    pre_train_chunks=[[("libero_goal", i)] for i in range(10)],
    task_seq_chunks=[[("libero_goal", i)] for i in range(10)]
)
LIBERO_GOAL_SCENARIO_INC_BASE = _goal_inc["base"]
LIBERO_GOAL_SCENARIO_INC_LARGE = _goal_inc["large"]
LIBERO_GOAL_SCENARIO_INC_SMALLPLUS = _goal_inc["smallplus"]

# Object Inc
_object_inc = _build_scenario_variants(
    pre_train_chunks=[[("libero_object", i)] for i in range(10)],
    task_seq_chunks=[[("libero_object", i)] for i in range(10)]
)
LIBERO_OBJECT_SCENARIO_INC_BASE = _object_inc["base"]
LIBERO_OBJECT_SCENARIO_INC_LARGE = _object_inc["large"]
LIBERO_OBJECT_SCENARIO_INC_SMALLPLUS = _object_inc["smallplus"]

# Spatial Inc
_spatial_inc = _build_scenario_variants(
    pre_train_chunks=[[("libero_spatial", i)] for i in range(10)],
    task_seq_chunks=[[("libero_spatial", i)] for i in range(10)]
)
LIBERO_SPATIAL_SCENARIO_INC_BASE = _spatial_inc["base"]
LIBERO_SPATIAL_SCENARIO_INC_LARGE = _spatial_inc["large"]
LIBERO_SPATIAL_SCENARIO_INC_SMALLPLUS = _spatial_inc["smallplus"]

# Long Inc
_long_inc = _build_scenario_variants(
    pre_train_chunks=[[("libero_10", i)] for i in range(10)],
    task_seq_chunks=[[("libero_10", i)] for i in range(10)]
)
LIBERO_LONG_SCENARIO_INC_BASE = _long_inc["base"]
LIBERO_LONG_SCENARIO_INC_LARGE = _long_inc["large"]
LIBERO_LONG_SCENARIO_INC_SMALLPLUS = _long_inc["smallplus"]

# ---------- 7111 SCENARIOS ----------
# Object 7111
object_7111 = [[("libero_object", i) for i in range(7)]]
for i in range(8, 10):
    object_7111.append([("libero_object", i)])

_object_7111 = _build_scenario_variants(
    pre_train_chunks=object_7111,
    task_seq_chunks=[[("libero_object", i)] for i in range(7,10)]
)
LIBERO_OBJECT_SCENARIO_7111_BASE = _object_7111["base"]
LIBERO_OBJECT_SCENARIO_7111_LARGE = _object_7111["large"]
LIBERO_OBJECT_SCENARIO_7111_SMALLPLUS = _object_7111["smallplus"]

# Spatial 7111
spatial_7111 = [[("libero_spatial", i) for i in range(7)]]
for i in range(8, 10):
    spatial_7111.append([("libero_spatial", i)])

_spatial_7111 = _build_scenario_variants(
    pre_train_chunks=spatial_7111,
    task_seq_chunks=[[("libero_spatial", i)] for i in range(7,10)]
)
LIBERO_SPATIAL_SCENARIO_7111_BASE = _spatial_7111["base"]
LIBERO_SPATIAL_SCENARIO_7111_LARGE = _spatial_7111["large"]
LIBERO_SPATIAL_SCENARIO_7111_SMALLPLUS = _spatial_7111["smallplus"]

# Goal 7111
goal_7111 = [[("libero_goal", i) for i in range(7)]]
for i in range(8, 10):
    goal_7111.append([("libero_goal", i)])

_goal_7111 = _build_scenario_variants(
    pre_train_chunks=goal_7111,
    task_seq_chunks=[[("libero_goal", i)] for i in range(7,10)]
)
LIBERO_GOAL_SCENARIO_7111_BASE = _goal_7111["base"]
LIBERO_GOAL_SCENARIO_7111_LARGE = _goal_7111["large"]
LIBERO_GOAL_SCENARIO_7111_SMALLPLUS = _goal_7111["smallplus"]

# Long 7111
long_7111 = [[("libero_10", i) for i in range(7)]]
for i in range(8, 10):
    long_7111.append([("libero_10", i)])

_long_7111 = _build_scenario_variants(
    pre_train_chunks=long_7111,
    task_seq_chunks=[[("libero_10", i)] for i in range(7,10)]
)
LIBERO_LONG_SCENARIO_7111_BASE = _long_7111["base"]
LIBERO_LONG_SCENARIO_7111_LARGE = _long_7111["large"]
LIBERO_LONG_SCENARIO_7111_SMALLPLUS = _long_7111["smallplus"]

# ---------- 334 SCENARIOS ----------
# Object 334 (3-3-4 chunks)
object_334 = [
    [("libero_object", i) for i in range(3)],      # First chunk: tasks 0-2
    [("libero_object", i) for i in range(3, 6)],   # Second chunk: tasks 3-5
    [("libero_object", i) for i in range(6, 10)]   # Third chunk: tasks 6-9
]

_object_334 = _build_scenario_variants(
    pre_train_chunks=object_334,
    task_seq_chunks=object_334
)
LIBERO_OBJECT_SCENARIO_334_BASE = _object_334["base"]
LIBERO_OBJECT_SCENARIO_334_LARGE = _object_334["large"]
LIBERO_OBJECT_SCENARIO_334_SMALLPLUS = _object_334["smallplus"]

# Spatial 334 (3-3-4 chunks)
spatial_334 = [
    [("libero_spatial", i) for i in range(3)],      # First chunk: tasks 0-2
    [("libero_spatial", i) for i in range(3, 6)],   # Second chunk: tasks 3-5
    [("libero_spatial", i) for i in range(6, 10)]   # Third chunk: tasks 6-9
]

_spatial_334 = _build_scenario_variants(
    pre_train_chunks=spatial_334,
    task_seq_chunks=spatial_334
)
LIBERO_SPATIAL_SCENARIO_334_BASE = _spatial_334["base"]
LIBERO_SPATIAL_SCENARIO_334_LARGE = _spatial_334["large"]
LIBERO_SPATIAL_SCENARIO_334_SMALLPLUS = _spatial_334["smallplus"]

# Goal 334 (3-3-4 chunks)
goal_334 = [
    [("libero_goal", i) for i in range(3)],      # First chunk: tasks 0-2
    [("libero_goal", i) for i in range(3, 6)],   # Second chunk: tasks 3-5
    [("libero_goal", i) for i in range(6, 10)]   # Third chunk: tasks 6-9
]

_goal_334 = _build_scenario_variants(
    pre_train_chunks=goal_334,
    task_seq_chunks=goal_334
)
LIBERO_GOAL_SCENARIO_334_BASE = _goal_334["base"]
LIBERO_GOAL_SCENARIO_334_LARGE = _goal_334["large"]
LIBERO_GOAL_SCENARIO_334_SMALLPLUS = _goal_334["smallplus"]

# Long 334 (3-3-4 chunks)
long_334 = [
    [("libero_10", i) for i in range(3)],      # First chunk: tasks 0-2
    [("libero_10", i) for i in range(3, 6)],   # Second chunk: tasks 3-5
    [("libero_10", i) for i in range(6, 10)]   # Third chunk: tasks 6-9
]

_long_334 = _build_scenario_variants(
    pre_train_chunks=long_334,
    task_seq_chunks=long_334
)
LIBERO_LONG_SCENARIO_334_BASE = _long_334["base"]
LIBERO_LONG_SCENARIO_334_LARGE = _long_334["large"]
LIBERO_LONG_SCENARIO_334_SMALLPLUS = _long_334["smallplus"]

# ---------- SEQ SCENARIOS (Pure Sequential: decoder→policy per task) ----------
# Helper function to build seq scenarios with default compatibility
def _build_seq_scenario_variants(
    benchmark_name: str,
    mapping_dir: str = "data/libero/task_map"
) -> Dict[str, List[SkillPhaseConfig]]:
    """Build fully sequential scenario for all model sizes.

    Uses 'default' compatibility mode which interleaves decoder and policy phases
    while properly setting up phase names for evaluation tracing.
    """
    pre_train_chunks = [[(benchmark_name, i)] for i in range(10)]
    task_seq_chunks = [[(benchmark_name, i)] for i in range(10)]

    return {
        model_name: build_scenario_from_chunks(
            pre_train_chunks=pre_train_chunks,
            task_seq_chunks=task_seq_chunks,
            model_name=model_name,
            mapping_dir=mapping_dir,
            compatibility="default"  # Use default mode for proper phase name formatting
        )
        for model_name in LIBERO_MODEL_EMBEDDINGS.keys()
    }

# Goal Seq
_goal_seq = _build_seq_scenario_variants("libero_goal")
LIBERO_GOAL_SCENARIO_SEQ_BASE = _goal_seq["base"]
LIBERO_GOAL_SCENARIO_SEQ_LARGE = _goal_seq["large"]
LIBERO_GOAL_SCENARIO_SEQ_SMALLPLUS = _goal_seq["smallplus"]

# Object Seq
_object_seq = _build_seq_scenario_variants("libero_object")
LIBERO_OBJECT_SCENARIO_SEQ_BASE = _object_seq["base"]
LIBERO_OBJECT_SCENARIO_SEQ_LARGE = _object_seq["large"]
LIBERO_OBJECT_SCENARIO_SEQ_SMALLPLUS = _object_seq["smallplus"]

# Spatial Seq
_spatial_seq = _build_seq_scenario_variants("libero_spatial")
LIBERO_SPATIAL_SCENARIO_SEQ_BASE = _spatial_seq["base"]
LIBERO_SPATIAL_SCENARIO_SEQ_LARGE = _spatial_seq["large"]
LIBERO_SPATIAL_SCENARIO_SEQ_SMALLPLUS = _spatial_seq["smallplus"]

# Long Seq
_long_seq = _build_seq_scenario_variants("libero_10")
LIBERO_LONG_SCENARIO_SEQ_BASE = _long_seq["base"]
LIBERO_LONG_SCENARIO_SEQ_LARGE = _long_seq["large"]
LIBERO_LONG_SCENARIO_SEQ_SMALLPLUS = _long_seq["smallplus"]

# ---------- PRETRAINING SCENARIOS (libero_90 by scene) ----------
# Kitchen pretraining (46 tasks)
_kitchen_pretrain = _build_pretraining_scenario_variants("libero_90", LIBERO_90_KITCHEN_TASKS)
LIBERO_90_PRETRAIN_KITCHEN_BASE = _kitchen_pretrain["base"]
LIBERO_90_PRETRAIN_KITCHEN_LARGE = _kitchen_pretrain["large"]
LIBERO_90_PRETRAIN_KITCHEN_SMALLPLUS = _kitchen_pretrain["smallplus"]

# Living room pretraining (27 tasks)
_living_pretrain = _build_pretraining_scenario_variants("libero_90", LIBERO_90_LIVING_ROOM_TASKS)
LIBERO_90_PRETRAIN_LIVING_BASE = _living_pretrain["base"]
LIBERO_90_PRETRAIN_LIVING_LARGE = _living_pretrain["large"]
LIBERO_90_PRETRAIN_LIVING_SMALLPLUS = _living_pretrain["smallplus"]

# Study pretraining (17 tasks)
_study_pretrain = _build_pretraining_scenario_variants("libero_90", LIBERO_90_STUDY_TASKS)
LIBERO_90_PRETRAIN_STUDY_BASE = _study_pretrain["base"]
LIBERO_90_PRETRAIN_STUDY_LARGE = _study_pretrain["large"]
LIBERO_90_PRETRAIN_STUDY_SMALLPLUS = _study_pretrain["smallplus"]

# Kitchen + Living pretraining (73 tasks)
_kitchen_living_pretrain = _build_pretraining_scenario_variants(
    "libero_90", LIBERO_90_KITCHEN_TASKS + LIBERO_90_LIVING_ROOM_TASKS
)
LIBERO_90_PRETRAIN_KITCHEN_LIVING_BASE = _kitchen_living_pretrain["base"]
LIBERO_90_PRETRAIN_KITCHEN_LIVING_LARGE = _kitchen_living_pretrain["large"]
LIBERO_90_PRETRAIN_KITCHEN_LIVING_SMALLPLUS = _kitchen_living_pretrain["smallplus"]

# Kitchen + Study pretraining (63 tasks)
_kitchen_study_pretrain = _build_pretraining_scenario_variants(
    "libero_90", LIBERO_90_KITCHEN_TASKS + LIBERO_90_STUDY_TASKS
)
LIBERO_90_PRETRAIN_KITCHEN_STUDY_BASE = _kitchen_study_pretrain["base"]
LIBERO_90_PRETRAIN_KITCHEN_STUDY_LARGE = _kitchen_study_pretrain["large"]
LIBERO_90_PRETRAIN_KITCHEN_STUDY_SMALLPLUS = _kitchen_study_pretrain["smallplus"]

# Living + Study pretraining (44 tasks)
_living_study_pretrain = _build_pretraining_scenario_variants(
    "libero_90", LIBERO_90_LIVING_ROOM_TASKS + LIBERO_90_STUDY_TASKS
)
LIBERO_90_PRETRAIN_LIVING_STUDY_BASE = _living_study_pretrain["base"]
LIBERO_90_PRETRAIN_LIVING_STUDY_LARGE = _living_study_pretrain["large"]
LIBERO_90_PRETRAIN_LIVING_STUDY_SMALLPLUS = _living_study_pretrain["smallplus"]

# All scenes pretraining (90 tasks)
_all_pretrain = _build_pretraining_scenario_variants("libero_90", LIBERO_90_ALL_TASKS)
LIBERO_90_PRETRAIN_ALL_BASE = _all_pretrain["base"]
LIBERO_90_PRETRAIN_ALL_LARGE = _all_pretrain["large"]
LIBERO_90_PRETRAIN_ALL_SMALLPLUS = _all_pretrain["smallplus"]

# Clean up temporary variables
del _test_scenarios, _object_multi, _goal_multi, _spatial_multi, _long_multi
del _goal_inc, _object_inc, _spatial_inc, _long_inc
del _object_7111, _spatial_7111, _goal_7111, _long_7111
del _object_334, _spatial_334, _goal_334, _long_334
del _goal_seq, _object_seq, _spatial_seq, _long_seq
del _kitchen_pretrain, _living_pretrain, _study_pretrain
del _kitchen_living_pretrain, _kitchen_study_pretrain, _living_study_pretrain, _all_pretrain





def libero_scenario(
    scenario_type: str = 'easy',
    sync_type: str = 'sync',
    model_name: str = DEFAULT_LIBERO_MODEL,
    requested_env: str | None = None
) -> SkillStreamConfig:
    """
    Get a Libero scenario configuration for a specific model.

    Args:
        scenario_type: Type of scenario
                      - Standard: 'test', 'object', 'goal', 'spatial', 'long'
                      - Pretraining (libero_90 scenes): 'kitchen', 'living', 'study', 'all',
                        'kitchen_living', 'kitchen_study', 'living_study'
        sync_type: Synchronization type
                   - Standard: 'sync', 'multi', 'seq', '7111', '334', 'task0', 'task1', ..., 'task9'
                   - Pretraining: 'pre' (decoder+interface only, no evaluation)
        model_name: Model name ('base', 'large', 'smallplus')
        requested_env: Full environment string requested by the user (e.g., 'libero-l')

    Returns:
        SkillStreamConfig for the requested scenario and model

    Examples:
        # Standard training
        libero_scenario(scenario_type='goal', sync_type='sync', model_name='base')

        # Pretraining on kitchen scenes only
        libero_scenario(scenario_type='kitchen', sync_type='pre', model_name='base')

        # Pretraining on all 90 tasks
        libero_scenario(scenario_type='all', sync_type='pre', model_name='large')
    """
    sync_type = sync_type.lower()
    scenario_type = scenario_type.lower()

    # Validate model name
    if model_name not in LIBERO_MODEL_EMBEDDINGS:
        raise ValueError(f"Unknown model: {model_name}. Must be one of: {list(LIBERO_MODEL_EMBEDDINGS.keys())}")

    # Check if sync_type is a single task pattern (task0, task1, ..., task9)
    import re
    task_match = re.match(r'task(\d+)', sync_type)
    if task_match:
        task_id = int(task_match.group(1))
        if task_id < 0 or task_id > 9:
            raise ValueError(f"Task ID must be between 0 and 9, got {task_id}")

        # Map scenario_type to benchmark_name
        benchmark_map = {
            'test': 'libero_goal',
            'goal': 'libero_goal',
            'spatial': 'libero_spatial',
            'object': 'libero_object',
            'long': 'libero_10'
        }

        if scenario_type not in benchmark_map:
            raise ValueError(f"Unknown scenario_type: {scenario_type}. Must be one of: {list(benchmark_map.keys())}")

        benchmark_name = benchmark_map[scenario_type]

        # Create single-task scenario dynamically
        datastream = build_scenario_from_chunks(
            pre_train_chunks=[[(benchmark_name, task_id)]],
            task_seq_chunks=[[(benchmark_name, task_id)]],
            model_name=model_name,
            mapping_dir="data/libero/task_map",
            compatibility="Synchronization"
        )

        env_alias = (requested_env or 'libero').lower()
        embedding_dim = get_embedding_size(model_name)
        obs_dim = get_libero_observation_dim(model_name)
        base_environment = "libero"

        return SkillStreamConfig(
            scenario_id=f"libero_{scenario_type}_{sync_type}_{model_name}",
            datastream=datastream,
            environment=base_environment,
            scenario_type=scenario_type,
            sync_type=sync_type,
            metadata={
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "libero_variant": model_name,
                "libero_obs_dim": obs_dim,
                "env_alias": env_alias,
                "base_environment": base_environment,
                "task_id": task_id,
            },
        )

    # Build the global variable name for the pre-built scenario
    # Format: LIBERO_{SCENARIO}_SCENARIO_{SYNCTYPE}_{MODEL}
    # Special cases: 'sync' -> 'INC', 'test' has no MULTI/INC suffix

    scenario_var_map = {
        # Test scenarios
        ('test', 'sync', 'base'): LIBERO_TEST_SCENARIO_BASE,
        ('test', 'sync', 'large'): LIBERO_TEST_SCENARIO_LARGE,
        ('test', 'sync', 'smallplus'): LIBERO_TEST_SCENARIO_SMALLPLUS,

        # Multi scenarios
        ('object', 'multi', 'base'): LIBERO_OBJECT_SCENARIO_MULTI_BASE,
        ('object', 'multi', 'large'): LIBERO_OBJECT_SCENARIO_MULTI_LARGE,
        ('object', 'multi', 'smallplus'): LIBERO_OBJECT_SCENARIO_MULTI_SMALLPLUS,

        ('goal', 'multi', 'base'): LIBERO_GOAL_SCENARIO_MULTI_BASE,
        ('goal', 'multi', 'large'): LIBERO_GOAL_SCENARIO_MULTI_LARGE,
        ('goal', 'multi', 'smallplus'): LIBERO_GOAL_SCENARIO_MULTI_SMALLPLUS,

        ('spatial', 'multi', 'base'): LIBERO_SPATIAL_SCENARIO_MULTI_BASE,
        ('spatial', 'multi', 'large'): LIBERO_SPATIAL_SCENARIO_MULTI_LARGE,
        ('spatial', 'multi', 'smallplus'): LIBERO_SPATIAL_SCENARIO_MULTI_SMALLPLUS,

        ('long', 'multi', 'base'): LIBERO_LONG_SCENARIO_MULTI_BASE,
        ('long', 'multi', 'large'): LIBERO_LONG_SCENARIO_MULTI_LARGE,
        ('long', 'multi', 'smallplus'): LIBERO_LONG_SCENARIO_MULTI_SMALLPLUS,

        # Incremental (sync) scenarios
        ('goal', 'sync', 'base'): LIBERO_GOAL_SCENARIO_INC_BASE,
        ('goal', 'sync', 'large'): LIBERO_GOAL_SCENARIO_INC_LARGE,
        ('goal', 'sync', 'smallplus'): LIBERO_GOAL_SCENARIO_INC_SMALLPLUS,

        ('object', 'sync', 'base'): LIBERO_OBJECT_SCENARIO_INC_BASE,
        ('object', 'sync', 'large'): LIBERO_OBJECT_SCENARIO_INC_LARGE,
        ('object', 'sync', 'smallplus'): LIBERO_OBJECT_SCENARIO_INC_SMALLPLUS,

        ('spatial', 'sync', 'base'): LIBERO_SPATIAL_SCENARIO_INC_BASE,
        ('spatial', 'sync', 'large'): LIBERO_SPATIAL_SCENARIO_INC_LARGE,
        ('spatial', 'sync', 'smallplus'): LIBERO_SPATIAL_SCENARIO_INC_SMALLPLUS,

        ('long', 'sync', 'base'): LIBERO_LONG_SCENARIO_INC_BASE,
        ('long', 'sync', 'large'): LIBERO_LONG_SCENARIO_INC_LARGE,
        ('long', 'sync', 'smallplus'): LIBERO_LONG_SCENARIO_INC_SMALLPLUS,

        # 7111 scenarios
        ('goal', '7111', 'base'): LIBERO_GOAL_SCENARIO_7111_BASE,
        ('goal', '7111', 'large'): LIBERO_GOAL_SCENARIO_7111_LARGE,
        ('goal', '7111', 'smallplus'): LIBERO_GOAL_SCENARIO_7111_SMALLPLUS,

        ('object', '7111', 'base'): LIBERO_OBJECT_SCENARIO_7111_BASE,
        ('object', '7111', 'large'): LIBERO_OBJECT_SCENARIO_7111_LARGE,
        ('object', '7111', 'smallplus'): LIBERO_OBJECT_SCENARIO_7111_SMALLPLUS,

        ('spatial', '7111', 'base'): LIBERO_SPATIAL_SCENARIO_7111_BASE,
        ('spatial', '7111', 'large'): LIBERO_SPATIAL_SCENARIO_7111_LARGE,
        ('spatial', '7111', 'smallplus'): LIBERO_SPATIAL_SCENARIO_7111_SMALLPLUS,

        ('long', '7111', 'base'): LIBERO_LONG_SCENARIO_7111_BASE,
        ('long', '7111', 'large'): LIBERO_LONG_SCENARIO_7111_LARGE,
        ('long', '7111', 'smallplus'): LIBERO_LONG_SCENARIO_7111_SMALLPLUS,

        # 334 scenarios
        ('object', '334', 'base'): LIBERO_OBJECT_SCENARIO_334_BASE,
        ('object', '334', 'large'): LIBERO_OBJECT_SCENARIO_334_LARGE,
        ('object', '334', 'smallplus'): LIBERO_OBJECT_SCENARIO_334_SMALLPLUS,

        ('spatial', '334', 'base'): LIBERO_SPATIAL_SCENARIO_334_BASE,
        ('spatial', '334', 'large'): LIBERO_SPATIAL_SCENARIO_334_LARGE,
        ('spatial', '334', 'smallplus'): LIBERO_SPATIAL_SCENARIO_334_SMALLPLUS,

        ('goal', '334', 'base'): LIBERO_GOAL_SCENARIO_334_BASE,
        ('goal', '334', 'large'): LIBERO_GOAL_SCENARIO_334_LARGE,
        ('goal', '334', 'smallplus'): LIBERO_GOAL_SCENARIO_334_SMALLPLUS,

        ('long', '334', 'base'): LIBERO_LONG_SCENARIO_334_BASE,
        ('long', '334', 'large'): LIBERO_LONG_SCENARIO_334_LARGE,
        ('long', '334', 'smallplus'): LIBERO_LONG_SCENARIO_334_SMALLPLUS,

        # Seq scenarios (pure sequential: decoder→policy per task)
        ('goal', 'seq', 'base'): LIBERO_GOAL_SCENARIO_SEQ_BASE,
        ('goal', 'seq', 'large'): LIBERO_GOAL_SCENARIO_SEQ_LARGE,
        ('goal', 'seq', 'smallplus'): LIBERO_GOAL_SCENARIO_SEQ_SMALLPLUS,

        ('object', 'seq', 'base'): LIBERO_OBJECT_SCENARIO_SEQ_BASE,
        ('object', 'seq', 'large'): LIBERO_OBJECT_SCENARIO_SEQ_LARGE,
        ('object', 'seq', 'smallplus'): LIBERO_OBJECT_SCENARIO_SEQ_SMALLPLUS,

        ('spatial', 'seq', 'base'): LIBERO_SPATIAL_SCENARIO_SEQ_BASE,
        ('spatial', 'seq', 'large'): LIBERO_SPATIAL_SCENARIO_SEQ_LARGE,
        ('spatial', 'seq', 'smallplus'): LIBERO_SPATIAL_SCENARIO_SEQ_SMALLPLUS,

        ('long', 'seq', 'base'): LIBERO_LONG_SCENARIO_SEQ_BASE,
        ('long', 'seq', 'large'): LIBERO_LONG_SCENARIO_SEQ_LARGE,
        ('long', 'seq', 'smallplus'): LIBERO_LONG_SCENARIO_SEQ_SMALLPLUS,

        # Pretraining scenarios (decoder+interface only, no evaluation)
        # scenario_type selects the scene(s), sync_type='pre' for pretraining
        ('kitchen', 'pre', 'base'): LIBERO_90_PRETRAIN_KITCHEN_BASE,
        ('kitchen', 'pre', 'large'): LIBERO_90_PRETRAIN_KITCHEN_LARGE,
        ('kitchen', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_KITCHEN_SMALLPLUS,

        ('living', 'pre', 'base'): LIBERO_90_PRETRAIN_LIVING_BASE,
        ('living', 'pre', 'large'): LIBERO_90_PRETRAIN_LIVING_LARGE,
        ('living', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_LIVING_SMALLPLUS,

        ('study', 'pre', 'base'): LIBERO_90_PRETRAIN_STUDY_BASE,
        ('study', 'pre', 'large'): LIBERO_90_PRETRAIN_STUDY_LARGE,
        ('study', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_STUDY_SMALLPLUS,

        ('kitchen_living', 'pre', 'base'): LIBERO_90_PRETRAIN_KITCHEN_LIVING_BASE,
        ('kitchen_living', 'pre', 'large'): LIBERO_90_PRETRAIN_KITCHEN_LIVING_LARGE,
        ('kitchen_living', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_KITCHEN_LIVING_SMALLPLUS,

        ('kitchen_study', 'pre', 'base'): LIBERO_90_PRETRAIN_KITCHEN_STUDY_BASE,
        ('kitchen_study', 'pre', 'large'): LIBERO_90_PRETRAIN_KITCHEN_STUDY_LARGE,
        ('kitchen_study', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_KITCHEN_STUDY_SMALLPLUS,

        ('living_study', 'pre', 'base'): LIBERO_90_PRETRAIN_LIVING_STUDY_BASE,
        ('living_study', 'pre', 'large'): LIBERO_90_PRETRAIN_LIVING_STUDY_LARGE,
        ('living_study', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_LIVING_STUDY_SMALLPLUS,

        ('all', 'pre', 'base'): LIBERO_90_PRETRAIN_ALL_BASE,
        ('all', 'pre', 'large'): LIBERO_90_PRETRAIN_ALL_LARGE,
        ('all', 'pre', 'smallplus'): LIBERO_90_PRETRAIN_ALL_SMALLPLUS,
    }

    key = (scenario_type, sync_type, model_name)
    if key not in scenario_var_map:
        raise ValueError(
            f"Unsupported combination: scenario_type='{scenario_type}', "
            f"sync_type='{sync_type}', model_name='{model_name}'"
        )

    datastream = scenario_var_map[key]
    env_alias = (requested_env or 'libero').lower()
    embedding_dim = get_embedding_size(model_name)
    obs_dim = get_libero_observation_dim(model_name)
    base_environment = "libero"

    return SkillStreamConfig(
        scenario_id=f"libero_{scenario_type}_{sync_type}_{model_name}",
        datastream=datastream,
        environment=base_environment,
        scenario_type=scenario_type,
        sync_type=sync_type,
        metadata={
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "libero_variant": model_name,
            "libero_obs_dim": obs_dim,
            "env_alias": env_alias,
            "base_environment": base_environment,
        },
    )



# --- Example usage for libero_object tasks ---
if __name__ == '__main__':
    logger = get_logger(__name__)
    pre_chunks = [[("libero_object", 0), ("libero_object", 1), ("libero_object", 2)]]
    task_chunks = pre_chunks
    mapping_directory = "data/libero/task_map"  # only directory provided

    # phases = LIBERO_TEST_SCENARIO

    # example_config = SkillStreamConfig(
    #     scenario_id="libero_object_example",
    #     datastream=phases,
    #     environment="libero",
    #     scenario_type="example",
    #     sync_type=None
    # )

    # for phase in example_config.datastream:
    #     logger.info(f"{phase.phase_name} {phase.train_targets}")
    #     logger.info(f"{phase.dataset_paths}")
    #     logger.info(f"{phase.train_tasks}")
