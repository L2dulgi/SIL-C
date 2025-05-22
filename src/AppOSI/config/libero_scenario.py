import os
import json
from typing import List, Tuple, Dict, Any
from AppOSI.config.skill_stream_config import SkillPhaseConfig, DEFAULT_SCHEMA, SkillStreamConfig
from AppOSI.config.kitchen_scenario import assemble_and_rename_scenario

LIBERO_SKILL_DATASET_PATH = "./data/libero"


def build_scenario_from_chunks(
    pre_train_chunks: List[List[Tuple[str, int]]],
    task_seq_chunks: List[List[Tuple[str, int]]],
    skill_dataset_path: str = LIBERO_SKILL_DATASET_PATH,
    compatibility: str | None = None,
    mapping_dir: str | None = None
) -> List[SkillPhaseConfig]:
    """
    Given lists of pre-training and task-sequence chunks (as tuples of benchmark_name and task_id),
    construct a flat list of SkillPhaseConfig objects matching the Libero scenario structure.

    Args:
        pre_train_chunks: [[(benchmark_name, id), ...], ...]
        task_seq_chunks: same shape as pre_train_chunks
        skill_dataset_path: base path for the .pkl skill datasets
        compatibility: if provided, applies assemble_and_rename_scenario to the groups
        mapping_dir: optional directory path containing JSON files named <benchmark_name>.json
                     each mapping file provides bddl_file names per task_id

    Returns:
        List of SkillPhaseConfig phases (decoder/interface then policy phases).
    """
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
            dec_paths.append(os.path.join(skill_dataset_path, f"{bm_name}/{base}_demo.pkl"))

        dec = SkillPhaseConfig(
            phase_name=f"pre_{phase_idx}",
            train_targets=["decoder", "interface"],
            dataset_paths=dec_paths,
            train_tasks=[f"{bm}-{tid}" for bm, tid in pre_chunk],
            schema=DEFAULT_SCHEMA
        )

        policies: List[SkillPhaseConfig] = []
        # build policy dataset paths
        for task_idx, (bm_name, task_id) in enumerate(task_seq_chunks[phase_idx]):
            mapping = bddl_map_by_benchmark.get(bm_name, {})
            base = mapping.get(task_id, f"{bm_name}-{task_id}")
            path = os.path.join(skill_dataset_path, f"{bm_name}/{base}_demo.pkl")
            tag = f"{bm_name}-{task_id}"
            policies.append(
                SkillPhaseConfig(
                    phase_name=f"task_{phase_idx}_{task_idx}",
                    train_targets=["policy"],
                    dataset_paths=[path],
                    train_tasks=[tag],
                    eval_tasks=[{"data_name": tag}],
                    schema=DEFAULT_SCHEMA
                )
            )

        group_phases.append((dec, policies))

    if compatibility:
        return assemble_and_rename_scenario(group_phases, compatibility)
    else:
        scenario: List[SkillPhaseConfig] = []
        for dec_phase, policy_list in group_phases:
            scenario.append(dec_phase)
            scenario.extend(policy_list)
        return scenario

LIBERO_TEST_SCENARIO = build_scenario_from_chunks(
    # pre_train_chunks=[[("libero_90", 0), ("libero_90", 6), ("libero_90", 11)]],
    # pre_train_chunks=[[("libero_object", i) for i in range(10)]],
    pre_train_chunks=[[("libero_goal", i) for i in range(10)]],
    # pre_train_chunks=[[("libero_goal", i)] for i in range(3)],
    # task_seq_chunks=[[("libero_90", 0), ("libero_90", 6), ("libero_90", 11)]],
    task_seq_chunks=[[("libero_goal", i) for i in range(10)]],
    # task_seq_chunks=[[("libero_goal", i)] for i in range(3)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)
LIBERO_TEST_SCENARIO[0].eval_tasks = None

LIBERO_OBJECT_SCENARIO_MULTI = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_object", i) for i in range(10)]],
    task_seq_chunks=[[("libero_object", i) for i in range(10)]],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

LIBERO_GOAL_SCENARIO_MULTI = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_goal", i) for i in range(10)]],
    task_seq_chunks=[[("libero_goal", i) for i in range(10)]],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

LIBERO_SPATIAL_SCENARIO_MULTI = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_spatial", i) for i in range(10)]],
    task_seq_chunks=[[("libero_spatial", i) for i in range(10)]],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

LIBERO_LONG_SCENARIO_MULTI = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_10", i) for i in range(10)]],
    task_seq_chunks=[[("libero_10", i) for i in range(10)]],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

# incremental scenario
LIBERO_GOAL_SCENARIO_INC = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_goal", i)] for i in range(10)],
    task_seq_chunks=[[("libero_goal", i)] for i in range(10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

LIBERO_OBJECT_SCENARIO_INC = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_object", i)] for i in range(10)],
    task_seq_chunks=[[("libero_object", i)] for i in range(10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

LIBERO_SPATIAL_SCENARIO_INC = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_spatial", i)] for i in range(10)],
    task_seq_chunks=[[("libero_spatial", i)] for i in range(10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

LIBERO_LONG_SCENARIO_INC = build_scenario_from_chunks(
    pre_train_chunks=[[("libero_10", i)] for i in range(10)],
    task_seq_chunks=[[("libero_10", i)] for i in range(10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)
# --- Example usage for libero_scenario ---

object_7111 = [[("libero_object", i) for i in range(7)]]
for i in range(8, 10):
    object_7111.append([("libero_object", i)])

LIBERO_OBJECT_SCENARIO_7111 = build_scenario_from_chunks(
    pre_train_chunks=object_7111,
    task_seq_chunks=[[("libero_object", i)] for i in range(7,10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

spatial_7111 = [[("libero_spatial", i) for i in range(7)]]
for i in range(8, 10):
    spatial_7111.append([("libero_spatial", i)])
LIBERO_SPATIAL_SCENARIO_7111 = build_scenario_from_chunks(
    pre_train_chunks=spatial_7111,
    task_seq_chunks=[[("libero_spatial", i)] for i in range(7,10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

goal_7111 = [[("libero_goal", i) for i in range(7)]]
for i in range(8, 10):
    goal_7111.append([("libero_goal", i)])
LIBERO_GOAL_SCENARIO_7111 = build_scenario_from_chunks(
    pre_train_chunks=goal_7111,
    task_seq_chunks=[[("libero_goal", i)] for i in range(7,10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)

long_7111 = [[("libero_10", i) for i in range(7)]]
for i in range(8, 10):
    long_7111.append([("libero_10", i)])
LIBERO_LONG_SCENARIO_7111 = build_scenario_from_chunks(
    pre_train_chunks=long_7111,
    task_seq_chunks=[[("libero_10", i)] for i in range(7,10)],
    skill_dataset_path=LIBERO_SKILL_DATASET_PATH,
    mapping_dir="data/libero/task_map",
    compatibility="Synchronization"
)





def libero_scenario(scenario_type: str = 'easy', sync_type: str = 'sync') -> SkillStreamConfig:
    datastream = None

    sync_type = sync_type.lower()
    scenario_type = scenario_type.lower()

    scenario_mapping = {
        ('test', 'sync'): LIBERO_TEST_SCENARIO,
        ('object', 'multi') : LIBERO_OBJECT_SCENARIO_MULTI,
        ('goal', 'multi') : LIBERO_GOAL_SCENARIO_MULTI,
        ('spatial', 'multi') : LIBERO_SPATIAL_SCENARIO_MULTI,
        ('long', 'multi') : LIBERO_LONG_SCENARIO_MULTI,

        ('goal', 'sync') : LIBERO_GOAL_SCENARIO_INC,
        ('object', 'sync') : LIBERO_OBJECT_SCENARIO_INC,
        ('spatial', 'sync') : LIBERO_SPATIAL_SCENARIO_INC,
        ('long', 'sync') : LIBERO_LONG_SCENARIO_INC,

        ('goal', '7111') : LIBERO_GOAL_SCENARIO_7111,
        ('object', '7111') : LIBERO_OBJECT_SCENARIO_7111,
        ('spatial', '7111') : LIBERO_SPATIAL_SCENARIO_7111,
        ('long', '7111') : LIBERO_LONG_SCENARIO_7111,
    }
    
    key = (scenario_type, sync_type)
    if key in scenario_mapping:
        datastream = scenario_mapping[key]
    else:
        raise ValueError(f"Unsupported scenario type: {scenario_type} with sync type: {sync_type}")
    
    return SkillStreamConfig(
        scenario_id=f"libero_{scenario_type}_{sync_type}",
        datastream=datastream,
        environment='libero',
        scenario_type=scenario_type,
        sync_type=sync_type,
    )


# --- Example usage for libero_object tasks ---
if __name__ == '__main__':
    pre_chunks = [[("libero_object", 0), ("libero_object", 1), ("libero_object", 2)]]
    task_chunks = pre_chunks
    mapping_directory = "data/libero/task_map"  # only directory provided

    phases = LIBERO_TEST_SCENARIO

    example_config = SkillStreamConfig(
        scenario_id="libero_object_example",
        datastream=phases,
        environment="libero",
        scenario_type="example",
        sync_type=None
    )

    for phase in example_config.datastream:
        print(phase.phase_name, phase.train_targets)
        print(phase.dataset_paths)
        print(phase.train_tasks)
