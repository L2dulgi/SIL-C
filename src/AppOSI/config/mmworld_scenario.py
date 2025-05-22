import os
import math
from AppOSI.config.skill_stream_config import dataset_path as MMWORLD_DATASET_PATH, SkillPhaseConfig, DEFAULT_SCHEMA, SkillStreamConfig
from itertools import permutations, product
from AppOSI.config.kitchen_scenario import assemble_and_rename_scenario

MMWORLD_DATASET_PATH = './data/evolving_world/raw/hard'
MMWORLD_SKILL_DATASET_PATH = './data/evolving_world/raw_skill/hard'

def get_task_list_equal_easy(all_task_flag='full'):
    task_sets = [['puck', 'drawer', 'button', 'door']]
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12
    if all_task_flag == 'full':
        ss = list(range(12))
    if all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'third':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]
        bound = 24
    task_shuffled = []
    for task_set in task_sets:
        for i, d in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({'skill_list': list(task_set), 'skill_seq': list(d)})
    return task_shuffled


def get_task_list_equal_normal(all_task_flag='full', only_normal=False):
    task_sets = []
    for combo in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        if 'box' in combo or 'stick' in combo:
            continue
        if only_normal and 'handle' not in combo and 'lever' not in combo:
            continue
        task_sets.append(list(combo))
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12
    if all_task_flag == 'full':
        ss = list(range(12))
    if all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'third':
        ss = [0, 3, 4, 7, 8, 11]
    if all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]
        bound = 24
    task_shuffled = []
    for task_set in task_sets:
        for i, d in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({'skill_list': list(task_set), 'skill_seq': list(d)})
    return task_shuffled


def get_task_list_equal_hard(all_task_flag='full'):
    task_sets = []
    for combo in product(('box', 'puck'), ('handle', 'drawer'), ('button', 'lever'), ('door', 'stick')):
        task_sets.append(list(combo))

    # default picks
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12

    if all_task_flag == 'full':
        ss = list(range(bound))        # keep all
        bound = 12
    elif all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]      # 6/12 = ½
        bound = 12
    elif all_task_flag == 'third':
        # if you really want 1/3, you could pick 4 out of 12:
        ss = [0, 4, 8]                # 3/12 = ¼ (but you could choose [0,3,6,9] for 4/12=⅓)
        bound = 12
    elif all_task_flag == 'quarter':
        ss = [0]                       # pick every 4th
        bound = 4
    elif all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]           # 4/24 = ⅙
        bound = 24
    else:
        raise ValueError(f"Unsupported task flag: {all_task_flag}")

    task_shuffled = []
    for task_set in task_sets:
        for i, d in enumerate(permutations(task_set)):
            if i % bound in ss:
                task_shuffled.append({
                    'skill_list': list(task_set),
                    'skill_seq': list(d)
                })
    return task_shuffled

def mmworld_scenario_per_chunk_decoder(
    all_task_flag: str = 'easy',
    phase_num: int = 4,
    only_normal: bool = False,
    pre_train_chunks: list = None,
    dataset_path: str = MMWORLD_DATASET_PATH,
    compatibility: str = None  # 'Forward', 'Backward', 'Bidirectional', 'Synchronization'
) -> list:
    """
    Generates an MMWORLD training scenario by chunking tasks per phase,
    with optional sync modes like the kitchen scenario.
    """
    if all_task_flag == 'easy':
        raw_tasks = get_task_list_equal_easy('full')
    elif all_task_flag == 'normal':
        raw_tasks = get_task_list_equal_normal('full', only_normal)
    elif all_task_flag == 'hard':
        raw_tasks = get_task_list_equal_hard('full')
    else:
        raise ValueError(f"Unsupported task flag: {all_task_flag}")

    # Build group phases
    data_names = ['-'.join(t['skill_seq']) for t in raw_tasks]
    total = len(data_names)
    chunk = math.ceil(total / phase_num)
    group_phases = []
    for idx in range(phase_num):
        names = data_names[idx*chunk:(idx+1)*chunk]
        if pre_train_chunks is not None:
            dataset_paths = pre_train_chunks[idx]
        else :
            dataset_paths = [os.path.join(dataset_path, n) + ".pkl" for n in names]
        dec = SkillPhaseConfig(
            phase_name=f"pre_{idx}",
            train_targets=['decoder','interface'],
            dataset_paths=dataset_paths,
            train_tasks=names,
            schema=DEFAULT_SCHEMA
        )
        policies = []
        for j,n in enumerate(names):
            policies.append(
                SkillPhaseConfig(
                    phase_name=f"task_{idx}_{j}",
                    train_targets=['policy'],
                    dataset_paths=[os.path.join(dataset_path,n)+".pkl"],
                    train_tasks=[n],
                    eval_tasks=[{'data_name':n}],
                    schema=DEFAULT_SCHEMA
                )
            )
        group_phases.append((dec,policies))

    # Assemble
    print(f"MMWORLD scenario with {compatibility} compatibility")
    print(f"MMWORLD scenario with {phase_num} phases")
    if compatibility != None:
        scenario = assemble_and_rename_scenario(group_phases, compatibility)
    else:
        scenario = []
        for d,ps in group_phases:
            scenario.append(d)
            scenario.extend(ps)
    return scenario

# Custom Chunk for the MMWORLD scenario
def get_task_list_by_env( env_spec : list = None, all_task_flag : str = 'full') :
    """
    Generates a list of tasks based on the provided environment specification.
    """
    if env_spec is None:
        env_spec = environment_set[0]
    
    # default picks
    ss = [0, 3, 4, 7, 8, 11]
    bound = 12

    if all_task_flag == 'full':
        ss = list(range(bound))        # keep all
        bound = 12
    elif all_task_flag == 'half':
        ss = [0, 3, 4, 7, 8, 11]      # 6/12 = ½
        bound = 12
    elif all_task_flag == 'third':
        # if you really want 1/3, you could pick 4 out of 12:
        ss = [0, 4, 8]                # 3/12 = ¼ (but you could choose [0,3,6,9] for 4/12=⅓)
        bound = 12
    elif all_task_flag == 'quarter':
        ss = [0]                       # pick every 4th
        bound = 4
    elif all_task_flag == 'sixth':
        ss = [0, 10, 13, 23]           # 4/24 = ⅙
        bound = 24
    else:
        raise ValueError(f"Unsupported task flag: {all_task_flag}")

    task_shuffled = []
    for i, d in enumerate(permutations(env_spec)):
        if i % bound in ss:
            task_shuffled.append({
                'skill_list': list(env_spec),
                'skill_seq': list(d)
            })
    return task_shuffled

def build_chunks_from_envs(
    env_set: list[list[str]],
    all_task_flag: str = 'full'
) -> list[list[str]]:
    """
    env_set: a list of environment specs, each a list of skills, e.g.
      [
        ['puck','drawer','button','door'],
        ['puck','handle','lever','door'],
        …
      ]
    all_task_flag: one of 'full', 'half', 'third', 'quarter', 'sixth'

    Returns:
      chunks: list of chunks, where each chunk is a list of
      "<skill1>-<skill2>-…-<skillN>" strings for that environment.
    """
    chunks = []
    for env_spec in env_set:
        task_dicts = get_task_list_by_env(env_spec, all_task_flag)
        names = ['-'.join(d['skill_seq']) for d in task_dicts]
        chunks.append(names)
    return chunks

def build_scenario_from_chunks(
    pre_train_chunks: list[list[str]],
    task_seq_chunks: list[list[str]],
    skill_dataset_path: str = MMWORLD_SKILL_DATASET_PATH,
    dataset_path: str = MMWORLD_DATASET_PATH,
    compatibility: str | None = None
) -> list[SkillPhaseConfig]:
    """
    Given a list of task‐sequence‐name chunks, produce the same
    pre_ / task_ phases as mmworld_scenario_per_chunk_decoder.

    Args:
        task_seq_chunks: e.g. [
            ["puck-drawer-button-door", "puck-handle-lever-door"],
            ["box-handle-button-door",   "box-drawer-lever-door"],
            ...
        ]
        dataset_path: base path where “<name>.pkl” lives
        compatibility: if not None, passed through to
                       assemble_and_rename_scenario

    Returns:
        A flat list of SkillPhaseConfig’s, in decoder+policy order.
    """
    group_phases = []
    for phase_idx, chunk_pre in enumerate(pre_train_chunks):
        dataset_paths = []
        for name in chunk_pre:
            if '-' not in name :
                dataset_paths.append(os.path.join(skill_dataset_path, name) + ".pkl")
            else:
                dataset_paths.append(os.path.join(dataset_path, name) + ".pkl")

        # decoder+interface on the whole chunk
        dec = SkillPhaseConfig(
            phase_name=f"pre_{phase_idx}",
            train_targets=["decoder", "interface"],
            dataset_paths=dataset_paths,
            train_tasks=chunk_pre,
            schema=DEFAULT_SCHEMA
        )
        chunk = task_seq_chunks[phase_idx]
        # one small policy phase per name
        policies = []
        for task_idx, name in enumerate(chunk):
            policies.append(
                SkillPhaseConfig(
                    phase_name=f"task_{phase_idx}_{task_idx}",
                    train_targets=["policy"],
                    dataset_paths=[os.path.join(dataset_path, name) + ".pkl"],
                    train_tasks=[name],
                    eval_tasks=[{"data_name": name}],
                    schema=DEFAULT_SCHEMA
                )
            )

        group_phases.append((dec, policies))

    if compatibility:
        # rename & assemble all at once
        return assemble_and_rename_scenario(group_phases, compatibility)
    else:
        # just flatten
        scenario = []
        for dec, policies in group_phases:
            scenario.append(dec)
            scenario.extend(policies)
        return scenario

n1_environment_set = [
    ['puck', 'drawer', 'button', 'door'],
    ['puck', 'handle', 'lever', 'door'],
    ['box', 'handle', 'button', 'door'],
    ['box', 'drawer', 'lever', 'door'],
]
MMWORLD_SCENARIO_N1_SYNC = build_scenario_from_chunks(
    pre_train_chunks=build_chunks_from_envs(n1_environment_set, all_task_flag='full'),
    task_seq_chunks=build_chunks_from_envs(n1_environment_set, all_task_flag='quarter'),
    dataset_path=MMWORLD_DATASET_PATH,
    compatibility='Synchronization',
)


n2_environment_set = [
    ['puck', 'drawer', 'button', 'stick'],
    ['puck', 'handle', 'lever', 'stick'],
    ['box', 'handle', 'button', 'door'],
    ['box', 'drawer', 'lever', 'door'],
]
MMWORLD_SCENARIO_N2_SYNC = build_scenario_from_chunks(
    pre_train_chunks=build_chunks_from_envs(n2_environment_set, all_task_flag='full'),
    task_seq_chunks=build_chunks_from_envs(n2_environment_set, all_task_flag='quarter'),
    dataset_path=MMWORLD_DATASET_PATH,
    compatibility='Synchronization',
)

# Prebuilt variants
MMWORLD_SCENARIO_EASY_SYNC = mmworld_scenario_per_chunk_decoder('easy',4,False,compatibility='Synchronization')

explicit_skill_set = [
    ['box', 'puck'],
    ['handle', 'drawer'],
    ['button', 'lever'],
    ['door', 'stick'],
]

MMWORLD_SCENARIO_EXPLICIT_SYNC = build_scenario_from_chunks(
    pre_train_chunks=explicit_skill_set,
    task_seq_chunks=build_chunks_from_envs(n1_environment_set, all_task_flag='quarter'),
    dataset_path=MMWORLD_DATASET_PATH,
    compatibility='Synchronization',
)

MMWORLD_SCENARIO_N2EXPLICIT_SYNC = build_scenario_from_chunks(
    pre_train_chunks=explicit_skill_set,
    task_seq_chunks=build_chunks_from_envs(n2_environment_set, all_task_flag='quarter'),
    skill_dataset_path='./data/evolving_world/skill_segments/n2',
    dataset_path=MMWORLD_DATASET_PATH,
    compatibility='Synchronization',
)


explicit_skill_set_easy = [
    ['puck'],
    ['drawer'],
    ['button'],
    ['door'],
]

explicit_skill_set_easy_chunks = [
    [f'./data/evolving_world/skill_segments/easy/{skill[0]}.pkl'] for skill in explicit_skill_set_easy
]

MMWORLD_SCENARIO_EASY_EXPLICIT_SYNC = mmworld_scenario_per_chunk_decoder(
    all_task_flag='easy',
    phase_num=4,
    pre_train_chunks=explicit_skill_set_easy_chunks,
    dataset_path=MMWORLD_DATASET_PATH,
    compatibility='Synchronization'
)

def mmworld_scenario(scenario_type: str = 'easy', sync_type: str = 'sync') -> SkillStreamConfig:
    """
    Returns the scenario configuration based on the provided type.
    """
    datastream = None

    sync_type = sync_type.lower()
    scenario_type = scenario_type.lower()

    scenario_mapping = {
        ('mmworldem', 'sync'): MMWORLD_SCENARIO_EASY_SYNC,
        ('mmworldem', 'sync'): MMWORLD_SCENARIO_EASY_EXPLICIT_SYNC,
        # ('n1', 'sync'): MMWORLD_SCENARIO_N1_SYNC,
        # ('n2', 'sync'): MMWORLD_SCENARIO_N2_SYNC,
        # ('explicit', 'sync'): MMWORLD_SCENARIO_EXPLICIT_SYNC,
        # ('n2exp', 'sync'): MMWORLD_SCENARIO_N2EXPLICIT_SYNC,
    }
    
    key = (scenario_type, sync_type)
    if key in scenario_mapping:
        datastream = scenario_mapping[key]
    else:
        raise ValueError(f"Unsupported scenario type: {scenario_type} with sync type: {sync_type}")
    
    
    return SkillStreamConfig(
        scenario_id=f"mmworld_{scenario_type}_{sync_type}",
        datastream=datastream,
        environment='mmworld',
        scenario_type=scenario_type,
        sync_type=sync_type,
    )




# Export

if __name__ == "__main__":
    for phase in MMWORLD_SCENARIO_EASY_EXPLICIT_SYNC:
        print(phase.phase_name, phase.train_targets)
        print(phase.dataset_paths)
        print(phase.train_tasks)
    # for k , i in enumerate(get_task_list_by_env(['puck', 'drawer', 'button', 'door'], 'quarter')):
    #     print(k, i['skill_seq'])
    

    # suppose you’ve already split into two chunks:
    # chunks = [
    #     ["puck-drawer-button-door", "puck-handle-lever-door"],
    #     ["box-handle-button-door",   "box-drawer-lever-door"]
    # ]

    # # build a straight sequence of phases:
    # # flat_scenario = build_scenario_from_chunks(chunks)

    # # or, if you want to apply “Synchronization” renaming:
    # # sync_scenario = build_scenario_from_chunks(chunks, compatibility="Synchronization")

    # # then wrap into a SkillStreamConfig exactly as mmworld_scenario does

    # n1_scenario = mmworld_scenario('n1', 'sync')
    # for phase in n1_scenario.datastream:
    #     print(phase.phase_name, phase.train_targets)
    #     print(phase.dataset_paths)
    #     print(phase.train_tasks)
