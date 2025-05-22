import os
import math
from typing import List
from collections import defaultdict

import numpy as np
try:
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster
except ImportError:
    raise ImportError("This example requires scipy. Please install via: pip install scipy")

# Import your own skill stream config items
from AppOSI.config.skill_stream_config import skill_dataset_path, SkillPhaseConfig, DEFAULT_SCHEMA, SkillStreamConfig

def abbreviate_task_name(filename: str) -> str:
    """
    Converts a full task filename (e.g. "bottom burner-top burner-light switch-slide cabinet.pkl")
    into a short abbreviation (e.g. "btls").
    This function removes ".pkl" and splits the remaining string by "-",
    then extracts the first letter of each sub-goal.
    """
    core_name = filename.replace(".pkl", "")
    sub_goals = core_name.split("-")
    abbreviation = "".join(sg.strip()[0] for sg in sub_goals)
    return abbreviation

def parse_subgoals(task_name: str) -> set:
    """
    Removes '.pkl', splits by '-', and returns a set of sub-goals.
    Example: "microwave-bottom burner-light switch-slide cabinet.pkl"
         -> {"microwave", "bottom burner", "light switch", "slide cabinet"}
    """
    core = task_name.replace('.pkl', '')
    parts = core.split('-')
    return set(p.strip() for p in parts)

def jaccard_distance(set_a: set, set_b: set) -> float:
    """
    Computes the Jaccard distance (1 - Jaccard similarity) 
    between two sets of subgoals.
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return 1.0 - float(intersection) / float(union)

def cluster_tasks_by_jaccard(all_tasks: List[str], n_clusters: int) -> List[List[str]]:
    """
    Clusters the given tasks into n_clusters based on Jaccard distance of their subgoal sets.
    Returns a list of clusters, where each cluster is a list of task filenames.
    """
    subgoal_sets = [parse_subgoals(t) for t in all_tasks]
    num_tasks = len(all_tasks)

    # Build the condensed distance array for pdist
    dist_list = []
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            dist_list.append(jaccard_distance(subgoal_sets[i], subgoal_sets[j]))
    dist_array = np.array(dist_list)

    # Hierarchical clustering
    Z = linkage(dist_array, method='average')

    # Use 'maxclust' to enforce exactly n_clusters
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Group tasks by cluster label
    cluster_dict = defaultdict(list)
    for idx, c_label in enumerate(cluster_labels):
        cluster_dict[c_label].append(all_tasks[idx])
    
    # Sort clusters by label for consistency
    sorted_labels = sorted(cluster_dict.keys())

    # Return a list of clusters, each cluster is a list of tasks
    return [cluster_dict[label] for label in sorted_labels]

def assemble_and_rename_scenario(group_phases, compatibility: str = None):
    """
    Assembles the final training scenario from group_phases based on the compatibility option 
    and renames policy phases.
    
    For the original policy training, each policy phase is pre-assigned a unique ID.
    If the same policy phases are repeated (e.g. in Bidirectional mode), their pre-assigned IDs are reused.
    
    The naming format for a policy phase is "policy_{unique_policy_id}/{skill_id}",
    where the skill_id is derived from the most recent decoder phase's phase_name.
    
    Additionally, for decoder phases (except the first group), instead of populating eval_tasks,
    the field eval_ref_policies is set with the list of all policy IDs from trained policies.
    
    Compatibility modes:
      - "Forward": All decoder phases come first, then all policy phases.
      - "Backward": The first decoder phase is executed first, then all policy phases,
                    and then remaining decoder phases are appended with eval_ref_policies updated.
      - "Bidirectional": Same as Backward, with an extra appended copy of all policy phases for additional training.
      - "Synchronization": The first decoder phase is executed first, then all policy phases, and for each
                           remaining decoder phase, eval_ref_policies is updated and the policies are repeated.
      - Otherwise: Default interleaved order (each group: decoder phase followed by its policy phases).
    
    Args:
      group_phases (list): A list of tuples (decoder_phase, list of policy_phases) for each group.
      compatibility (str): The scenario assembly mode.
      
    Returns:
      list: A list of SkillPhaseConfig objects representing the final training scenario.
    """
    import os
    import copy

    # Preassign unique policy IDs for each policy phase.
    policy_id_counter = 0
    for (_, policies) in group_phases:
        for phase in policies:
            policy_id_counter += 1
            phase.unique_policy_id = policy_id_counter

    # Build common lists for decoder phases and policy phases.
    dec_list = [dec for (dec, _) in group_phases]
    policy_list = []
    for (_, policies) in group_phases:
        policy_list.extend(policies)
    
    # Build a combined list of policy IDs (e.g. "policy_1", "policy_2", etc.).
    combined_eval_ref_policies = [f"policy_{p.unique_policy_id}" for p in policy_list]
    
    scenario = []
    if compatibility == "Forward":
        scenario = dec_list + policy_list

    elif compatibility == "Backward":
        # Use the first decoder phase, then policies, then remaining decoders with updated references.
        if group_phases:
            scenario.append(group_phases[0][0])
        scenario.extend(policy_list)
        if len(group_phases) > 1:
            for (dec, _) in group_phases[1:]:
                dec.eval_ref_policies = combined_eval_ref_policies
                scenario.append(dec)

    elif compatibility == "Bidirectional":
        # Same as Backward, plus an extra copy of policy phases.
        if group_phases:
            scenario.append(group_phases[0][0])
        scenario.extend(policy_list)
        if len(group_phases) > 1:
            for (dec, _) in group_phases[1:]:
                dec.eval_ref_policies = combined_eval_ref_policies
                scenario.append(dec)
        scenario.extend([copy.deepcopy(p) for p in policy_list])

    elif compatibility == "Synchronization":
        # Execute first decoder phase, then policies.
        if group_phases:
            scenario.append(group_phases[0][0])
        scenario.extend(policy_list)
        if len(group_phases) > 1:
            for (dec, _) in group_phases[1:]:
                dec.eval_ref_policies = combined_eval_ref_policies
                scenario.append(dec)
                scenario.extend([copy.deepcopy(p) for p in policy_list])
    else:
        # Default interleaved order: each group is appended (decoder followed by its policies).
        for (dec, policies) in group_phases:
            scenario.append(dec)
            scenario.extend(policies)

    # Rename policy phases: for each policy, update its phase_name using its unique_policy_id 
    # and the current skill id, which is the phase_name of the most recent decoder.
    current_skill_id = None
    for phase in scenario:
        if 'decoder' in phase.train_targets:
            if phase.train_tasks:
                current_skill_id = phase.phase_name
        elif 'policy' in phase.train_targets:
            phase.phase_name = f"policy_{phase.unique_policy_id}/{current_skill_id}"
    
    return scenario

def kitchen_scenario_per_chunk_decoder(
    all_tasks: list,
    skill_dataset_path: str,
    phase_num: int = 4,
    option: list = None,
    compatibility: str = None,  # "Forward" or "Backward"
    permute_list: list[int] = None
):
    """
    Creates a training scenario for the kitchen domain.
    
    The function partitions tasks either by:
      - Index-based chunking (default), or 
      - Clustering by similarity if 'cluster' is specified in option.
      
    Additional options:
      - 'overlap': The decoder+interface phase for chunk i uses tasks from 
                   chunk i plus chunk i-1 (if i > 0).
      - 'accumulate': The decoder+interface phase for chunk i uses tasks 
                      from all previous chunks (0 through i). 
                      (If both 'overlap' and 'accumulate' are specified, 'accumulate' takes precedence.)
    
    Original design interleaves a "decoder+interface" phase followed by policy phases for each group.
    
    With the new compatibility argument:
      - If compatibility == "Forward": All decoder phases (one per group) are placed first,
        and then a single, combined policy phase is appended at the end (using all tasks from all groups).
      - If compatibility == "Backward": A single, combined policy phase is placed first,
        followed by all decoder phases in order.
      - Otherwise, the original interleaved order is used.
      
    Args:
        all_tasks (list): A list of task filenames.
        skill_dataset_path (str): The file path where the skill datasets are stored.
        phase_num (int): The number of chunks or clusters to split the task list into.
        option (list): Options such as 'cluster', 'overlap', or 'accumulate'.
        compatibility (str): If "Forward" or "Backward", the order of phases is rearranged accordingly.
    
    Returns:
        list: A list of SkillPhaseConfig objects representing the entire training scenario.
    """
    if option is None:
        option = []

    # Partition tasks: cluster vs. index-based chunking
    if 'cluster' in option:
        groups = cluster_tasks_by_jaccard(all_tasks, n_clusters=phase_num)
    else:
        chunk_size = math.ceil(len(all_tasks) / phase_num)
        groups = []
        start_idx = 0
        for _ in range(phase_num):
            end_idx = min(start_idx + chunk_size, len(all_tasks))
            group = all_tasks[start_idx:end_idx]
            groups.append(group)
            start_idx = end_idx

    # sequence mix
    if permute_list is not None:
        if len(permute_list) != len(groups):
            raise ValueError("Length of permute_list must match the number of groups.")
        groups = [groups[i] for i in permute_list]

    # Helper for building the decoder dataset for a given group index.
    def get_decoder_dataset(idx: int) -> list:
        current_group = groups[idx]
        if 'accumulate' in option:
            dec_tasks = []
            for k in range(idx + 1):  # accumulate groups 0 through idx
                dec_tasks.extend(groups[k])
            return dec_tasks
        if 'overlap' in option and idx > 0:
            prev_group = groups[idx - 1]
            return prev_group + current_group
        return current_group

    # Build group-wise phases: each group produces a decoder phase and a list of policy phases.
    group_phases = []  # List of tuples: (decoder_phase, list of policy_phases)
    for group_idx, group_tasks in enumerate(groups):
        # Create the decoder+interface phase for this group.
        decoder_dataset = get_decoder_dataset(group_idx)
        dec_dataset_paths = [os.path.join(skill_dataset_path, t) for t in decoder_dataset]
        # Automatically generate train_tasks as the abbreviation from the filenames in dataset_paths
        dec_train_tasks = [abbreviate_task_name(os.path.basename(p)) for p in dec_dataset_paths]
        dec_phase = SkillPhaseConfig(
            phase_name=f"pre_{group_idx}",
            train_targets=['decoder', 'interface'],
            dataset_paths=dec_dataset_paths,
            train_tasks=dec_train_tasks,
            schema=DEFAULT_SCHEMA
        )
        # Create policy phases for each task in the group.
        policy_phases = []
        for i, task_name in enumerate(group_tasks):
            policy_dataset_path = os.path.join(skill_dataset_path, task_name)
            policy_train_tasks = [abbreviate_task_name(os.path.basename(policy_dataset_path))]
            eval_tasks = [{'data_name': abbreviate_task_name(task_name)}]
            policy_phase = SkillPhaseConfig(
                phase_name=f"task_{group_idx}_{i}",
                train_targets=['policy'],
                dataset_paths=[policy_dataset_path],
                train_tasks=policy_train_tasks,
                eval_tasks=eval_tasks,
                schema=DEFAULT_SCHEMA
            )
            policy_phases.append(policy_phase)
        group_phases.append((dec_phase, policy_phases))

    # Assemble the final scenario using the extracted function.
    scenario = assemble_and_rename_scenario(group_phases, compatibility)
    return scenario

# ------------------------------
# Scenario definition
# ------------------------------

# Example tasks (24)
example_tasks = [
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

KITCHEN_SCENARIO_ALLPRE = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=1,
    option=[]
)

KITCHEN_SCENARIO_DEFAULT = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=[]
)

KITCHEN_SCENARIO_OVERLAP = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['overlap']
)

KITCHEN_SCENARIO_ACCUMULATE = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['accumulate']
)

KITCHEN_SCENARIO_CLUSTER_OVERLAP = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['cluster', 'overlap']
)

# ------------------------------
# FWC Scenario definition
# ------------------------------

KITCHEN_SCENARIO_DEFAULT_FWC = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=[],
    compatibility="Forward"
)

KITCHEN_SCENARIO_DEFAULT_BWC = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=[],
    compatibility="Backward"
)

KITCHEN_SCENARIO_DEFAULT_BIC = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=[],
    compatibility="Bidirectional"
)

KITCHEN_SCENARIO_DEFAULT_SYNC = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=[],
    compatibility="Synchronization"
)

KITCHEN_SCENARIO_OBJ_SYNC = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['cluster'],
    compatibility="Synchronization"
)

KITCHEN_SCENARIO_OBJ_ASYNC = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['cluster'],
    compatibility="Bidirectional"
)


# ------------------------------
# permuted scenario
# ------------------------------
permute_lists = [
    [3,2,1,0],
    [2,3,0,1],
    [1,0,3,2],
]
KITCHEN_SCENARIO_OBJ_SYNC_P1 = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['cluster'],
    compatibility="Synchronization",
    permute_list=permute_lists[0]
)

KITCHEN_SCENARIO_OBJ_SYNC_P2 = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['cluster'],
    compatibility="Synchronization",
    permute_list=permute_lists[1]
)

KITCHEN_SCENARIO_OBJ_SYNC_P3 = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=['cluster'],
    compatibility="Synchronization",
    permute_list=permute_lists[2]
)

# ----------------------------------
# Joint Learning Scenario
# ----------------------------------

KITCHEN_SCENARIO_OBJ_JOINT = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks,
    skill_dataset_path=skill_dataset_path,
    phase_num=1,
    option=[],
    compatibility="Bidirectional"
)

# etc.
KITCHEN_DEBUG_SCENARIO = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks[:4],
    skill_dataset_path=skill_dataset_path,
    phase_num=4,
    option=[],
    compatibility="Synchronization"
)

KITCHEN_DEBUG_MIN_SCENARIO = kitchen_scenario_per_chunk_decoder(
    all_tasks=example_tasks[:2],
    skill_dataset_path=skill_dataset_path,
    phase_num=2,
    option=[],
    compatibility="Synchronization"
)

# Kitchen Skill Incremental Learning Scenario
raw_skill_dataset_path = './data/evolving_kitchen/skill_segments'

example_skills = [
    ["microwave.pkl"],
    ["kettle.pkl", "bottom burner.pkl"],
    ["top burner.pkl", "light switch.pkl"],
    ["slide cabinet.pkl","hinge cabinet.pkl"]
]

def kitchen_scenario_skill_incrementall(
        example_skills: list,
        example_tasks: list,
        phase_num: int = None,
        option: list = None,
        compatibility: str = "Synchronization"  # Only "Synchronization" mode is supported
):
    """
    Function to generate an incremental learning scenario.
    
    - Each decoder phase uses the skill files assigned to that phase in raw_skill_dataset_path for training.
      (e.g., phase 0 uses ["microwave.pkl"], phase 1 uses ["kettle.pkl", "bottom burner.pkl"], etc.)
    - The policy phase uses all example_tasks as is.
    - The first decoder phase has no evaluation targets, and from subsequent phases a fixed set of 24 policy slots (with fixed IDs)
      are registered in eval_ref_policies.
    
    Structure:
        Decoder training -> 24 policy trainings -> Decoder training -> 24 policy trainings -> ... 
        (Reusing fixed policy IDs 1~24, e.g., policy_1/pre_0, policy_1/pre_1, etc.)
    
    Args:
        example_skills (list): A list of lists of skill filenames to use for training in each phase 
                               (e.g., [["microwave.pkl"], ["kettle.pkl", "bottom burner.pkl"], ...])
        example_tasks (list): A list of task filenames to be used for policy training and evaluation.
        phase_num (int): Number of incremental stages. If not specified, the length of example_skills is used.
        option (list): Additional options (currently unused).
        compatibility (str): Only "Synchronization" mode is supported.
        
    Returns:
        list: The final scenario composed of a list of SkillPhaseConfig objects.
    """
    import os

    # If option is not provided, initialize it as an empty list
    if option is None:
        option = []

    # If phase_num is not specified, use the length of example_skills (each corresponding to a phase)
    num_groups = len(example_skills) if phase_num is None else phase_num

    # ----------------------
    # Generate policy phases: use example_tasks as is (no chunking)
    # ----------------------
    policy_groups = [example_tasks for _ in range(num_groups)]

    # ----------------------
    # For each phase, construct group-wise phases
    # ----------------------
    group_phases = []  # Each element: (decoder_phase, [list of policy_phases])
    for i in range(num_groups):
        # The skill files corresponding to the current phase 
        # (e.g., phase 0: ["microwave.pkl"], phase 1: ["kettle.pkl", "bottom burner.pkl"], …)
        skills_for_phase = example_skills[i]
        # Generate paths and abbreviations for the given skill files
        skill_paths = [os.path.join(raw_skill_dataset_path, skill) for skill in skills_for_phase]
        skill_abbrevs = [abbreviate_task_name(os.path.basename(skill)) for skill in skills_for_phase]

        # Decoder phase: use only the skill files of the current phase
        dec_phase = SkillPhaseConfig(
            phase_name=f"pre_{i}",
            train_targets=['decoder', 'interface'],
            dataset_paths=skill_paths,
            train_tasks=skill_abbrevs,
            schema=DEFAULT_SCHEMA
        )

        # Policy phase: use the entire example_tasks for the current phase
        policy_phases = []
        group_policy_tasks = policy_groups[i]
        for j, task_file in enumerate(group_policy_tasks):
            policy_dataset_path = os.path.join(skill_dataset_path, task_file)
            policy_train_tasks = [abbreviate_task_name(os.path.basename(task_file))]
            eval_tasks = [{'data_name': abbreviate_task_name(task_file)}]
            policy_phase = SkillPhaseConfig(
                phase_name=f"task_{i}_{j}",  # Will be renamed later
                train_targets=['policy'],
                dataset_paths=[policy_dataset_path],
                train_tasks=policy_train_tasks,
                eval_tasks=eval_tasks,
                schema=DEFAULT_SCHEMA
            )
            policy_phases.append(policy_phase)
        group_phases.append((dec_phase, policy_phases))

    # ----------------------
    # Assemble the final scenario (Synchronization mode)
    # ----------------------
    final_scenario = []
    # The unique id of policy phases in each phase is fixed to 1 ~ 24 (not cumulative)
    fixed_policy_ids = [f"policy_{i}" for i in range(1, 25)]
    
    for idx, (dec_phase, policy_phase_list) in enumerate(group_phases):
        # The first phase's decoder has no eval_ref_policies.
        if idx == 0:
            dec_phase.eval_ref_policies = []
        else:
            # For subsequent phases, assign a fixed set of 24 policy ids to eval_ref_policies
            dec_phase.eval_ref_policies = fixed_policy_ids[:]
    
        # Append the decoder phase to the final scenario first
        final_scenario.append(dec_phase)
    
        # For each phase, the policy phases reuse fixed ids (1~24)
        for j, p in enumerate(policy_phase_list):
            p.unique_policy_id = j + 1  # Fixed id
            p.phase_name = f"policy_{p.unique_policy_id}/{dec_phase.phase_name}"
            final_scenario.append(p)

    return final_scenario

# Example usage
KITCHEN_SCENARIO_SKILL_INCREMENTAL = kitchen_scenario_skill_incrementall(
    example_skills=example_skills,
    example_tasks=example_tasks,
    phase_num=None,
    option=[],
    compatibility="Synchronization"
)

# ------------------------------
# Environment Scenario Creator
# ------------------------------
import logging
def kitchen_scenario(scenario_type: str = 'objective', sync_type: str = 'sync') -> SkillStreamConfig:
    """
    Creates a SkillStreamConfig object based on the provided scenario type and synchronization type.

    Args:
        scenario_type (str): The type of scenario. Expected values are 'default', 'objective', or 'quality'.
        sync_type (str): The synchronization type. Expected values are 'sync' or 'async'.

    Returns:
        SkillStreamConfig: A configuration object with the selected datastream.

    Raises:
        NotImplementedError: If the 'quality' scenario is requested, as it is not implemented.
        ValueError: If an invalid scenario configuration is provided.
    """
    logger = logging.getLogger(__name__)
    
    # Normalize the input strings to lower case for consistency
    sync_type = sync_type.lower()
    scenario_type = scenario_type.lower()
    print(f"[KitchenScenario] Initializing kitchen scenario with sync_type='{sync_type}', scenario_type='{scenario_type}'")
    
    # Create a dictionary mapping for (sync_type, scenario_type) to their corresponding datastream
    scenario_mapping = {
        # sync scenarios
        # ('sync', 'default'): KITCHEN_SCENARIO_DEFAULT_SYNC,
        ('sync', 'kitchenem'): KITCHEN_SCENARIO_OBJ_SYNC,
        ('sync', 'kitchenex'): KITCHEN_SCENARIO_SKILL_INCREMENTAL,

        ('sync', 'objective_p1'): KITCHEN_SCENARIO_OBJ_SYNC_P1,
        ('sync', 'objective_p2'): KITCHEN_SCENARIO_OBJ_SYNC_P2,
        ('sync', 'objective_p3'): KITCHEN_SCENARIO_OBJ_SYNC_P3,
        # Async scenarios
        # ('async', 'default'): KITCHEN_SCENARIO_DEFAULT_BIC,
        # ('async', 'objective'): KITCHEN_SCENARIO_OBJ_ASYNC,

        # # Joint Learning scenarios
        # ('joint', 'objective'): KITCHEN_SCENARIO_OBJ_JOINT,
        # ('joint', 'objective'): KITCHEN_SCENARIO_OBJ_JOINT,
    }
    
    key = (sync_type, scenario_type)
    # Check if the key exists in the mapping
    if key not in scenario_mapping:
        logger.error(f"Invalid scenario configuration: sync_type='{sync_type}', scenario_type='{scenario_type}'")
        if scenario_type == 'quality':
            raise NotImplementedError("Quality scenario is not implemented.")
        else:
            raise ValueError(f"Invalid scenario configuration: {key}")
    
    # Retrieve the datastream from the dictionary mapping
    datastream = scenario_mapping[key]
    logger.info("Kitchen scenario successfully created.")
    
    # Return the SkillStreamConfig object with the selected datastream
    return SkillStreamConfig(datastream=datastream, scenario_type=scenario_type, sync_type=sync_type)


if __name__ == "__main__":
    exit()

    # Example visualization of synchronization scenario
    my_stream_config_overlap = SkillStreamConfig('overlap_kitchen', datastream=KITCHEN_SCENARIO_OBJ_SYNC)
  
    # for idx, phase_cfg in enumerate(KITCHEN_SCENARIO_DEFAULT_BIC):
    #     print(f"[{idx}] Phase Name: {phase_cfg.phase_name}")
    #     print(f"    Train Targets: {phase_cfg.train_targets}")
    #     print(f"    Dataset Paths: {len(phase_cfg.dataset_paths)}")
    #     print(f"    Train Tasks  : {phase_cfg.train_tasks}")
    #     print(f"    Eval Tasks   : {phase_cfg.eval_tasks}")
    #     print(f"    eval ref policies   : {phase_cfg.eval_ref_policies}\n")

    from pprint import pprint

    for idx, phase_cfg in enumerate(KITCHEN_SCENARIO_DEFAULT_BIC):
        print(f"[{idx}] Phase Config:")
        pprint(vars(phase_cfg))  # or use asdict(phase_cfg) if it's a dataclass
        print("\n" + "-"*50 + "\n")

    from AppOSI.config.skill_stream_config import EvaluationTracer

    tracer = EvaluationTracer()

    traced_data = tracer.get_eval_tasks_by_reference(  # to do update
        my_stream_config_overlap,
        50,  # phase
        # 0, # phase
    )
    print(len(traced_data))
    for data in traced_data:
        print(data)

    for idx, phase_cfg in enumerate(KITCHEN_SCENARIO_SKILL_INCREMENTAL):
        # Uncomment the condition if needed to filter phases
        if True:
            print(f"[{idx}] Phase Config:")
            pprint(vars(phase_cfg))  # or use asdict(phase_cfg) if it's a dataclass
            print("\n" + "-"*50 + "\n")
