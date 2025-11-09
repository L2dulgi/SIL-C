"""
Metaworld scenario configuration module.

This module provides configuration functions for creating training scenarios
in the Metaworld domain, supporting various task combinations and difficulty levels.
"""

import os
import math
from typing import List, Dict, Optional
from itertools import permutations, product

from SILGym.config.skill_stream_config import SkillPhaseConfig, SkillStreamConfig
from SILGym.config.kitchen_scenario import assemble_scenario_phases
from SILGym.config.data_paths import (
    MMWORLD_DATASET_PATH,
    MMWORLD_SKILL_DATASET_PATH,
)

# Task sampling configuration
TASK_SAMPLING = {
    'full': (list(range(12)), 12),
    'half': ([0, 3, 4, 7, 8, 11], 12),
    'third': ([0, 4, 8], 12),
    'quarter': ([0], 4),
    'sixth': ([0, 10, 13, 23], 24),
}

# ============================================================================
# Task List Generation
# ============================================================================

def _get_task_list(task_sets: List[List[str]], sampling: str = 'full') -> List[Dict]:
    """Generic task list generator from task sets."""
    ss, bound = TASK_SAMPLING.get(sampling, TASK_SAMPLING['half'])
    
    task_shuffled = []
    for task_set in task_sets:
        for i, perm in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({
                    'skill_list': list(task_set),
                    'skill_seq': list(perm)
                })
    return task_shuffled

def get_task_list_equal_easy(sampling: str = 'full') -> List[Dict]:
    """Easy mode: single task set."""
    return _get_task_list([['puck', 'drawer', 'button', 'door']], sampling)

def get_task_list_equal_normal(sampling: str = 'full', only_normal: bool = False) -> List[Dict]:
    """Normal mode: filtered combinations."""
    task_sets = []
    for combo in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        if 'box' in combo or 'stick' in combo:
            continue
        if only_normal and 'handle' not in combo and 'lever' not in combo:
            continue
        task_sets.append(list(combo))
    return _get_task_list(task_sets, sampling)

def get_task_list_equal_hard(sampling: str = 'full') -> List[Dict]:
    """Hard mode: all combinations."""
    task_sets = [list(combo) for combo in 
                 product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick'))]
    return _get_task_list(task_sets, sampling)

# ============================================================================
# Scenario Building
# ============================================================================

def build_scenario_from_chunks(
    pre_train_chunks: List[List[str]],
    task_seq_chunks: List[List[str]],
    skill_dataset_path: str = MMWORLD_SKILL_DATASET_PATH,
    dataset_path: str = MMWORLD_DATASET_PATH,
    compatibility: Optional[str] = None
) -> List[SkillPhaseConfig]:
    """Build scenario from pre-training and task sequence chunks."""
    group_phases = []
    
    for phase_idx, (pre_chunk, task_chunk) in enumerate(zip(pre_train_chunks, task_seq_chunks)):
        # Build dataset paths for pre-training
        dataset_paths = [
            os.path.join(skill_dataset_path if '-' not in name else dataset_path, name) + ".pkl"
            for name in pre_chunk
        ]
        
        # Decoder phase
        dec = SkillPhaseConfig(
            phase_name=f"pre_{phase_idx}",
            train_targets=["decoder", "interface"],
            dataset_paths=dataset_paths,
            train_tasks=pre_chunk
        )
        
        # Policy phases
        policies = [
            SkillPhaseConfig(
                phase_name=f"task_{phase_idx}_{task_idx}",
                train_targets=["policy"],
                dataset_paths=[os.path.join(dataset_path, name) + ".pkl"],
                train_tasks=[name],
                eval_tasks=[{"data_name": name}]
            )
            for task_idx, name in enumerate(task_chunk)
        ]
        
        group_phases.append((dec, policies))
    
    return assemble_scenario_phases(group_phases, compatibility) if compatibility else \
           [phase for dec, policies in group_phases for phase in [dec] + policies]

def create_mmworld_scenario(
    difficulty: str = 'easy',
    phase_num: int = 4,
    pre_train_chunks: Optional[List] = None,
    dataset_path: str = MMWORLD_DATASET_PATH,
    compatibility: str = "Synchronization"
) -> List[SkillPhaseConfig]:
    """Create an MMWORLD training scenario."""
    # Get task list based on difficulty
    task_funcs = {
        'easy': lambda: get_task_list_equal_easy('full'),
        'normal': lambda: get_task_list_equal_normal('full'),
        'hard': lambda: get_task_list_equal_hard('full')
    }
    
    if difficulty not in task_funcs:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    
    raw_tasks = task_funcs[difficulty]()
    data_names = ['-'.join(t['skill_seq']) for t in raw_tasks]
    
    # Build chunks
    chunk_size = math.ceil(len(data_names) / phase_num)
    chunks = [data_names[i*chunk_size:(i+1)*chunk_size] for i in range(phase_num)]
    
    # Use provided pre_train_chunks or default to task chunks
    if not pre_train_chunks:
        pre_train_chunks = chunks
    
    return build_scenario_from_chunks(pre_train_chunks, chunks,
                                    dataset_path=dataset_path,
                                    compatibility=compatibility)

# ============================================================================
# Predefined Scenarios (Lazy Loading)
# ============================================================================

def _create_easy_scenario():
    """Create easy scenario with default settings."""
    return create_mmworld_scenario('easy', 4, compatibility='Synchronization')

def _create_easy_explicit_scenario():
    """Create easy scenario with explicit skill chunks."""
    skill_chunks = [
        ['puck'], ['drawer'], ['button'], ['door']
    ]
    pre_chunks = [
        [f'./data/evolving_world/skill_segments/easy/{skill[0]}.pkl'] 
        for skill in skill_chunks
    ]
    return create_mmworld_scenario('easy', 4, pre_chunks, compatibility='Synchronization')

# Lazy loading for backward compatibility
MMWORLD_SCENARIOS = {
    'easy': _create_easy_scenario,
    'easy_explicit': _create_easy_explicit_scenario,
}

def __getattr__(name):
    """Lazy loading for backward compatibility."""
    # For scenarios used in other modules
    if name == 'MMWORLD_SCENARIO_EASY_SYNC':
        scenario = _create_easy_scenario()
        globals()[name] = scenario
        return scenario
    elif name == 'MMWORLD_SCENARIO_EASY_EXPLICIT_SYNC':
        scenario = _create_easy_explicit_scenario()
        globals()[name] = scenario
        return scenario
    elif name == 'MMWORLD_SCENARIO_N1_SYNC':
        # Create N1 scenario for lazySI.py
        n1_env_set = [
            ['puck', 'drawer', 'button', 'door'],
            ['puck', 'handle', 'lever', 'door'],
            ['box', 'handle', 'button', 'door'],
            ['box', 'drawer', 'lever', 'door'],
        ]
        env_chunks = []
        for env_spec in n1_env_set:
            tasks = _get_task_list([env_spec], 'quarter')
            env_chunks.append(['-'.join(t['skill_seq']) for t in tasks])
        
        pre_chunks = []
        for env_spec in n1_env_set:
            tasks = _get_task_list([env_spec], 'full')
            pre_chunks.append(['-'.join(t['skill_seq']) for t in tasks])
            
        scenario = build_scenario_from_chunks(pre_chunks, env_chunks,
                                            dataset_path=MMWORLD_DATASET_PATH,
                                            compatibility='Synchronization')
        globals()[name] = scenario
        return scenario
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# ============================================================================
# Main Scenario Factory
# ============================================================================

def mmworld_scenario(scenario_type: str = 'easy', sync_type: str = 'sync') -> SkillStreamConfig:
    """Create a SkillStreamConfig based on scenario type."""
    sync_type = sync_type.lower()
    scenario_type = scenario_type.lower()
    
    if sync_type != 'sync':
        raise ValueError(f"Only 'sync' mode is supported, got '{sync_type}'")
    
    # Map scenario types
    scenario_mapping = {
        'mmworldem': 'easy',
        'mmworldex': 'easy_explicit',
    }
    
    if scenario_type not in scenario_mapping:
        raise ValueError(f"Unsupported scenario type: {scenario_type}")
    
    # Create scenario lazily
    scenario_key = scenario_mapping[scenario_type]
    datastream = MMWORLD_SCENARIOS[scenario_key]()
    
    return SkillStreamConfig(
        scenario_id=f"mmworld_{scenario_type}_{sync_type}",
        datastream=datastream,
        environment='mmworld',
        scenario_type=scenario_type,
        sync_type=sync_type,
    )