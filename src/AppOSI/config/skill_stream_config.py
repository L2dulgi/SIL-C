import os

# Path where the dataset files are located
skill_dataset_path = './data/evolving_kitchen/raw'

DEFAULT_SCHEMA = {
    'columns': ['skill_id', 'skill_name', 'skill_description'],
    'types': ['int', 'str', 'str']
}

class SkillPhaseConfig:
    def __init__(
            self, 
            phase_name: str = 'default', 
            train_targets: list[str] = ['decoder', 'interface', 'policy'],
            dataset_paths: list[str] = [], 
            train_tasks : list[str] = [],   
            eval_tasks : list[str] = [],
            eval_ref_policies: list[str] = [],
            schema: dict = DEFAULT_SCHEMA,
        ) -> None:
        self.phase_name = phase_name
        self.train_targets = train_targets  
        self.dataset_paths = dataset_paths
        self.schema = schema
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks  # contains the evaluation tasks or evaluation reference policy id for the current phase 
        self.eval_ref_policies = eval_ref_policies  # contains the reference policy id for the current phase

    def _validate_targets(self):
        # Ensure that the targets are valid
        possible_targets = ['decoder', 'interface', 'policy']
        for target in self.train_targets:
            if target not in possible_targets:
                raise ValueError(f"Invalid target '{target}'. Must be one of {possible_targets}")
            

# Read all .pkl files in the specified directory and sort them alphabetically
task_files = [f for f in os.listdir(skill_dataset_path) if f.endswith('.pkl')]
task_files.sort()


DEFAULT_DATASTREAM = [
    SkillPhaseConfig(
        phase_name='task0',
        dataset_paths=[
            f"{skill_dataset_path}/bottom burner-top burner-light switch-slide cabinet.pkl",
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task1',
        dataset_paths=[
            f"{skill_dataset_path}/microwave-bottom burner-light switch-slide cabinet.pkl",
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task2',
        dataset_paths=[
            f"{skill_dataset_path}/microwave-kettle-bottom burner-hinge cabinet.pkl",
        ],
        schema=DEFAULT_SCHEMA
    )
]

DEFAULT_ASYNC_STREAM = [
    SkillPhaseConfig(
        phase_name='pre',
        train_targets=['decoder', 'interface'],
        dataset_paths=[
            f"{skill_dataset_path}/{task}" for task in os.listdir(skill_dataset_path)
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task0',
        train_targets=['policy'],
        dataset_paths=[
            f"{skill_dataset_path}/bottom burner-top burner-light switch-slide cabinet.pkl",
        ],
        eval_tasks=[
            {'data_name': 'btls'},
            # {'data_name': 'mbls'},
            # {'data_name': 'mkbh'},
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task1',
        train_targets=['policy'],
        dataset_paths=[
            f"{skill_dataset_path}/microwave-bottom burner-light switch-slide cabinet.pkl",
        ],
        eval_tasks=[
            {'data_name': 'btls'},
            {'data_name': 'mbls'},
            # {'data_name': 'mkbh'},
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task2',
        train_targets=['policy'],
        dataset_paths=[
            f"{skill_dataset_path}/microwave-kettle-bottom burner-hinge cabinet.pkl",
        ],
        eval_tasks=[
            {'data_name': 'btls'},
            {'data_name': 'mbls'},
            {'data_name': 'mkbh'},
        ],
        schema=DEFAULT_SCHEMA
    )
]

VIS_ASYNC_STREAM = [
    SkillPhaseConfig(
        phase_name='pre',
        train_targets=['decoder', 'interface'],
        dataset_paths=[
            f"{skill_dataset_path}/{task}" for task in os.listdir(skill_dataset_path)
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task0',
        train_targets=['policy'],
        dataset_paths=[
            f"{skill_dataset_path}/bottom burner-top burner-light switch-slide cabinet.pkl",
        ],
        eval_tasks=[
            {'data_name': 'btls'},
            {'data_name': 'mbls'},
            {'data_name': 'mkbh'},
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task1',
        train_targets=['policy'],
        dataset_paths=[
            f"{skill_dataset_path}/microwave-bottom burner-light switch-slide cabinet.pkl",
        ],
        eval_tasks=[
            {'data_name': 'btls'},
            {'data_name': 'mbls'},
            {'data_name': 'mkbh'},
        ],
        schema=DEFAULT_SCHEMA
    ),
    SkillPhaseConfig(
        phase_name='task2',
        train_targets=['policy'],
        dataset_paths=[
            f"{skill_dataset_path}/microwave-kettle-bottom burner-hinge cabinet.pkl",
        ],
        eval_tasks=[
            {'data_name': 'btls'},
            {'data_name': 'mbls'},
            {'data_name': 'mkbh'},
        ],
        schema=DEFAULT_SCHEMA
    ),
]

# Create a phase for each file and add it to the FULL_DATASTREAM list
FULL_DATASTREAM = []
for idx, file_name in enumerate(task_files, start=1):
    phase_name = f"task{idx}"
    dataset_path = os.path.join(skill_dataset_path, file_name)
    FULL_DATASTREAM.append(SkillPhaseConfig(phase_name=phase_name, dataset_paths=[dataset_path]))

# ------------------------------------------------
# ---------- Skill Stream Configuration ----------
# ------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
from copy import copy, deepcopy 

def _connect_sigmoid(ax, xA, yA, xB, yB, lw=3, color='gray'):
    """
    Draw a sigmoid-like curved line from (xA, yA) to (xB, yB) using a cubic Bézier curve.
    The control points are chosen to provide horizontal tangents at the endpoints.
    """
    cp1 = (xA + (xB - xA) / 3, yA)
    cp2 = (xA + 2 * (xB - xA) / 3, yB)
    Path = mpath.Path
    vertices = [(xA, yA), cp1, cp2, (xB, yB)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = mpath.Path(vertices, codes)
    patch = mpatches.PathPatch(path, fc="none", ec=color, lw=lw)
    ax.add_patch(patch)

class SkillStreamConfig:
    def __init__(
            self, 
            scenario_id: str = 'default', 
            datastream=None,
            environment: str = 'kitchen',
            scenario_type: str = 'objective',
            sync_type: str = 'sync',
        ):
        self.scenario_id = scenario_id
        self.environment = environment # For whole scenario (input and output space define)
        self.scenario_type = scenario_type # For whole scenario
        self.sync_type = sync_type # For whole scenario 
        self.datastream = datastream if datastream else []

    def visualize_stream(self, figsize=(12, 10), save_path=None, table=False):
        """
        Demonstrates a timeline in the top subplot and a styled table of evaluation tasks
        in the bottom subplot. Row height is increased using two approaches:
        1) the_table.scale(xscale, yscale)
        2) Manually setting each cell's height in a loop
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        # Create two subplots: top for timeline, bottom for table
        fig, (ax_timeline, ax_table) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )

        # --------------------- TIMELINE SUBPLOT ---------------------
        target_positions = {'policy': 2, 'interface': 1, 'decoder': 0}
        color_map = {'policy': 'limegreen', 'interface': 'dodgerblue', 'decoder': 'gold'}
        edge_color_map = {'policy': 'green', 'interface': 'blue', 'decoder': 'darkgoldenrod'}

        num_phases = len(self.datastream)

        # Alternate background color
        for i, phase in enumerate(self.datastream):
            phase_color = "#f5f5f5" if i % 2 == 0 else "#e0e0e0"
            ax_timeline.axvspan(i, i+1, color=phase_color, alpha=0.5, ec='none', lw=0)

        last_pos = {'policy': None, 'interface': None, 'decoder': None}

        # Plot markers and connect lines for each phase
        for i, phase in enumerate(self.datastream):
            x_center = i + 0.5
            for target in ['policy', 'interface', 'decoder']:
                if target in phase.train_targets:
                    new_pos = (x_center, target_positions[target], i)
                    if last_pos[target] is not None:
                        last_x, last_y, last_phase = last_pos[target]
                        style = 'solid' if i == last_phase + 1 else '--'
                        ax_timeline.plot(
                            [last_x, x_center],
                            [last_y, target_positions[target]],
                            color='gray', linewidth=3, linestyle=style, zorder=1
                        )
                    ax_timeline.plot(
                        x_center, target_positions[target],
                        marker='o', markersize=14,
                        color=color_map[target],
                        markeredgecolor=edge_color_map[target],
                        markeredgewidth=3, linestyle='None', zorder=2
                    )
                    last_pos[target] = new_pos

            # Display phase name
            ax_timeline.text(
                x_center, -0.7, phase.phase_name,
                ha='center', va='top', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7),
                zorder=3
            )

        # Extend dashed line if a target is not present in the final phase
        final_x = num_phases - 0.5
        for target in ['policy', 'interface', 'decoder']:
            if last_pos[target] is not None:
                lx, ly, lphase = last_pos[target]
                if lphase < (num_phases - 1):
                    ax_timeline.plot(
                        [lx, final_x],
                        [ly, ly],
                        color='gray', linewidth=3, linestyle='--', zorder=1
                    )

        # Interface-based arcs
        for i, phase in enumerate(self.datastream):
            if 'interface' in phase.train_targets:
                # Draw arcs within the same phase if applicable
                if 'policy' in phase.train_targets:
                    _connect_sigmoid(ax_timeline, i + 0.5, 1, i + 0.5, 2)
                if 'decoder' in phase.train_targets:
                    _connect_sigmoid(ax_timeline, i + 0.5, 1, i + 0.5, 0)
                # Draw arcs to the next phase only if needed.
                if i + 1 < num_phases:
                    next_phase = self.datastream[i + 1]
                    if 'policy' in next_phase.train_targets:
                        _connect_sigmoid(ax_timeline, i + 0.5, 1, (i + 1) + 0.5, 2)
                    # If the current phase already contains 'decoder', do not draw an arc for the next phase's decoder.
                    if 'decoder' in next_phase.train_targets and 'decoder' not in phase.train_targets:
                        _connect_sigmoid(ax_timeline, i + 0.5, 1, (i + 1) + 0.5, 0)

        # Configure timeline axes
        ax_timeline.set_xlim(0, num_phases)
        ax_timeline.set_ylim(-1.5, 2.5)
        ax_timeline.set_yticks([2, 1, 0])
        ax_timeline.set_yticklabels(['Policy', 'Interface', 'Decoder'])
        xtick_positions = [i + 0.5 for i in range(num_phases)]
        xtick_labels = [str(i) for i in range(num_phases)]
        ax_timeline.set_xticks(xtick_positions)
        ax_timeline.set_xticklabels(xtick_labels)
        ax_timeline.set_xlabel('Phase index')
        ax_timeline.set_title(f"Skill Stream Visualization: {self.scenario_id}", fontweight='bold')

        for spine in ax_timeline.spines.values():
            spine.set_visible(False)
        ax_timeline.grid(False)
        ax_timeline.tick_params(axis='y', length=0)

        # Legend
        policy_marker = mlines.Line2D(
            [], [], color='limegreen', marker='o', markersize=14,
            label='Policy', markeredgecolor='green', markeredgewidth=3, linestyle='None'
        )
        interface_marker = mlines.Line2D(
            [], [], color='dodgerblue', marker='o', markersize=14,
            label='Interface', markeredgecolor='blue', markeredgewidth=3, linestyle='None'
        )
        decoder_marker = mlines.Line2D(
            [], [], color='gold', marker='o', markersize=14,
            label='Decoder', markeredgecolor='darkgoldenrod', markeredgewidth=3, linestyle='None'
        )
        ax_timeline.legend(
            handles=[policy_marker, interface_marker, decoder_marker],
            loc='upper center', bbox_to_anchor=(0.5, -0.15),
            ncol=3, frameon=False
        )

        # --------------------- TABLE SUBPLOT ---------------------
        if table == True:
            ax_table.axis('off')  # Hide axes for the table subplot

            # Build table data: for each phase, show the phase name, task count, and evaluation tasks.
            table_data = []
            for phase in self.datastream:
                if phase.eval_tasks:
                    names = [str(t.get('data_name', t)) for t in phase.eval_tasks]
                    task_count = len(names)
                    # If 10 or fewer, show all names; otherwise, show first 10 and append " ..."
                    if task_count <= 10:
                        eval_tasks_str = ", ".join(names)
                    else:
                        eval_tasks_str = ", ".join(names[:10]) + " ..."
                else:
                    eval_tasks_str = "N/A"
                    task_count = 0
                table_data.append([phase.phase_name, str(task_count), eval_tasks_str])

            # Create the table only if there's at least one evaluation task entry.
            if any(row[2] != "N/A" for row in table_data):
                # Column order: Phase, Task Count, Evaluation Tasks
                the_table = ax_table.table(
                    cellText=table_data,
                    colLabels=["Phase", "Task Count", "Evaluation Tasks"],
                    loc='center',
                    cellLoc='center'
                )

                # Styling: set fixed heights and adjust column widths.
                # the_table.auto_set_font_size(False)
                # the_table.set_fontsize(10)
                for (row, col), cell in the_table.get_celld().items():
                    cell.set_height(0.1)  # Consistent row height
                    cell.set_linewidth(0.1)  # Thinner borders
                    if col == 0:  # Phase column: narrow
                        cell.set_width(0.15)
                    elif col == 1:  # Task Count column: narrow
                        cell.set_width(0.1)
                    elif col == 2:  # Evaluation Tasks column: wider
                        cell.set_width(0.75)
                    if row == 0:  # Header row
                        cell.set_facecolor("#40466e")
                        cell.set_text_props(color="white", weight="bold")
                    else:  # Data rows
                        cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "#ffffff")

        plt.tight_layout()

        # Save the figure to a file or display it
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to: {save_path}")
            plt.close(fig)
        else:
            # plt.show()
            pass

class EvaluationTracer:
    # ----------------------
    # Evaluation tracking
    # ----------------------
    def get_eval_tasks_by_reference(
        self,
        stream_config,        # Instance of SkillStreamConfig.
        phase,                # Phase index.
        reference_policy_id=None  # Optional reference policy ID for filtering.
    ):
        """
        Returns evaluation configurations for a given phase and reference policy.

        For a policy phase (when train_targets includes 'policy'):
          - Extracts the learned skill id from phase_name (e.g., "policy_1/pre_0" -> "pre_0").
          - Searches for the corresponding learned skill phase among previous phases (phase_name == learned id).
          - Uses that learned phase info for decoder and interface, and returns the current policy phase info.

        For a decoder/interface phase:
          - Iterates over the evaluation reference policies and, if a reference_policy_id is provided,
            only considers matching policies.
          - Searches among previous phases (index < current phase) for a candidate whose phase_name contains
            the reference policy string.
          - Returns the candidate policy checkpoint info along with the current phase info.

        Returns:
            list[dict]: Each dictionary contains:
              - 'agent_config': A dict with keys 'decoder', 'interface', and 'policy'
                              mapping to the corresponding checkpoint information.
              - 'eval_tasks': The evaluation tasks.
        """
        current_phase = stream_config.datastream[phase]
        is_policy_phase = 'policy' in current_phase.train_targets
        eval_configs = []

        if is_policy_phase:
            # Extract learned skill id from the phase name (e.g. "policy_1/pre_0" -> "pre_0")
            parts = current_phase.phase_name.split('/')
            learned_id = parts[1] if len(parts) == 2 else current_phase.phase_name

            # Search only among phases with an index less than the current phase.
            learned_phase_info = None
            for idx in range(phase):
                phase_obj = stream_config.datastream[idx]
                if phase_obj.phase_name == learned_id:
                    learned_phase_info = (idx, phase_obj.phase_name)
                    break
            if learned_phase_info is None:
                learned_phase_info = (None, None)

            eval_config = {
                'agent_config': {
                    'decoder': learned_phase_info,
                    'interface': learned_phase_info,
                    'policy': (phase, current_phase.phase_name),
                },
                'eval_tasks': current_phase.eval_tasks,
            }
            eval_configs.append(eval_config)
        else:
            # For decoder/interface evaluation phases:
            for ref_policy in current_phase.eval_ref_policies:
                if reference_policy_id and ref_policy != reference_policy_id:
                    continue

                candidate_policy_idx = None
                candidate_policy_phase = None

                # Search only among phases with an index less than the current phase.
                for idx in range(phase):
                    phase_obj = stream_config.datastream[idx]
                    if ref_policy in phase_obj.phase_name:
                        candidate_policy_idx = idx
                        candidate_policy_phase = phase_obj
                        break

                if candidate_policy_phase is None:
                    continue

                eval_config = {
                    'agent_config': {
                        'decoder': (phase, current_phase.phase_name),
                        'interface': (phase, current_phase.phase_name),
                        'policy': (candidate_policy_idx, candidate_policy_phase.phase_name),
                    },
                    'eval_tasks': candidate_policy_phase.eval_tasks,
                }
                eval_configs.append(eval_config)
        return eval_configs
 
DEFAULT_SKILL_STREAM_CONFIG = SkillStreamConfig()

if __name__ == "__main__":
    # my_stream_config = SkillStreamConfig('my_skill_stream', datastream=VIS_ASYNC_STREAM *3)
    # # To save to a file (e.g., 'my_skill_stream.png'):
    # my_stream_config.visualize_stream(save_path='my_skill_stream.png')

    eval_tracer = EvaluationTracer()
    trace_stream = SkillStreamConfig(datastream=VIS_ASYNC_STREAM)