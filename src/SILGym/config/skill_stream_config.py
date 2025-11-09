"""
Skill stream configuration module.

This module defines the core configuration classes for skill-based training phases
and provides visualization capabilities for training streams.
"""

from typing import List, Dict, Optional, Any
from SILGym.utils.logger import get_logger
from SILGym.config.data_paths import skill_dataset_path

# ============================================================================
# Core Configuration Classes
# ============================================================================

class SkillPhaseConfig:
    """Configuration for a single training phase."""
    
    VALID_TARGETS = ['decoder', 'interface', 'policy']
    
    def __init__(
        self,
        phase_name: str = 'default',
        train_targets: List[str] = None,
        dataset_paths: List[str] = None,
        train_tasks: List[str] = None,
        eval_tasks: List[Dict] = None,
        eval_ref_policies: List[str] = None,
    ):
        self.phase_name = phase_name
        self.train_targets = train_targets or ['decoder', 'interface', 'policy']
        self.dataset_paths = dataset_paths or []
        self.train_tasks = train_tasks or []
        self.eval_tasks = eval_tasks or []
        self.eval_ref_policies = eval_ref_policies or []
        
        # Validate targets on initialization
        self._validate_targets()
    
    def _validate_targets(self):
        """Ensure that the targets are valid."""
        for target in self.train_targets:
            if target not in self.VALID_TARGETS:
                raise ValueError(f"Invalid target '{target}'. Must be one of {self.VALID_TARGETS}")


class SkillStreamConfig:
    """Configuration for a complete skill training stream."""
    
    def __init__(
        self,
        scenario_id: str = 'default',
        datastream: List[SkillPhaseConfig] = None,
        environment: str = 'kitchen',
        scenario_type: str = 'objective',
        sync_type: str = 'sync',
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.scenario_id = scenario_id
        self.environment = environment
        self.scenario_type = scenario_type
        self.sync_type = sync_type
        self.datastream = datastream or []
        self.metadata = metadata or {}
    
    def print_stream(self):
        """Print a text-based representation of the skill stream configuration."""
        print(f"\n{'='*80}")
        print(f"Skill Stream Configuration: {self.scenario_id}")
        print(f"{'='*80}")
        print(f"Environment: {self.environment}")
        print(f"Scenario Type: {self.scenario_type}")
        print(f"Sync Type: {self.sync_type}")
        print(f"Total Phases: {len(self.datastream)}")
        print(f"{'='*80}\n")
        
        for i, phase in enumerate(self.datastream):
            print(f"Phase {i}: {phase.phase_name}")
            print(f"{'-'*40}")
            
            # Training targets
            print(f"  Training Targets: {', '.join(phase.train_targets)}")
            
            # Dataset paths
            if phase.dataset_paths:
                print(f"  Dataset Paths ({len(phase.dataset_paths)}):")
                for path in phase.dataset_paths:
                    # Extract just the filename for readability
                    filename = path.split('/')[-1] if '/' in path else path
                    print(f"    - {filename}")
            else:
                print(f"  Dataset Paths: None")
            
            # Training tasks
            if phase.train_tasks:
                print(f"  Training Tasks ({len(phase.train_tasks)}): {', '.join(phase.train_tasks[:5])}")
                if len(phase.train_tasks) > 5:
                    print(f"    ... and {len(phase.train_tasks) - 5} more")
            else:
                print(f"  Training Tasks: None")
            
            # Evaluation tasks
            if phase.eval_tasks:
                print(f"  Evaluation Tasks ({len(phase.eval_tasks)}):")
                for j, task in enumerate(phase.eval_tasks[:3]):
                    task_name = task.get('data_name', str(task)) if isinstance(task, dict) else str(task)
                    print(f"    - {task_name}")
                if len(phase.eval_tasks) > 3:
                    print(f"    ... and {len(phase.eval_tasks) - 3} more")
            else:
                print(f"  Evaluation Tasks: None")
            
            # Evaluation reference policies
            if phase.eval_ref_policies:
                print(f"  Eval Reference Policies: {', '.join(phase.eval_ref_policies)}")
            else:
                print(f"  Eval Reference Policies: None")
            
            print()
        
        print(f"{'='*80}\n")
    
    def visualize_stream(self, figsize=(12, 10), save_path=None, table=False):
        """
        Visualize the skill training stream as a timeline.

        This method delegates to the visualization module to avoid
        matplotlib dependency in the configuration module.

        Args:
            figsize: Figure size as (width, height) tuple.
            save_path: Path to save the figure. If None, displays interactively.
            table: If True, include evaluation tasks table.
        """
        from SILGym.visualization.stream_viz import visualize_skill_stream
        visualize_skill_stream(self, figsize=figsize, save_path=save_path, table=table)


class EvaluationTracer:
    """Handles evaluation configuration tracking for skill streams."""
    
    def get_eval_tasks_by_reference(
        self,
        stream_config: SkillStreamConfig,
        phase: int,
        reference_policy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
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
        
        if is_policy_phase:
            return self._get_policy_eval_config(stream_config, phase, current_phase)
        else:
            return self._get_decoder_eval_config(stream_config, phase, current_phase, 
                                                reference_policy_id)
    
    def _get_policy_eval_config(self, stream_config, phase, current_phase):
        """Get evaluation config for policy phases."""
        # Extract learned skill ID from phase name
        parts = current_phase.phase_name.split('/')
        learned_id = parts[1] if len(parts) == 2 else current_phase.phase_name
        
        # Find matching decoder/interface phase
        learned_phase_info = (None, None)
        for idx in range(phase):
            if stream_config.datastream[idx].phase_name == learned_id:
                learned_phase_info = (idx, stream_config.datastream[idx].phase_name)
                break
        
        return [{
            'agent_config': {
                'decoder': learned_phase_info,
                'interface': learned_phase_info,
                'policy': (phase, current_phase.phase_name),
            },
            'eval_tasks': current_phase.eval_tasks,
        }]
    
    def _get_decoder_eval_config(self, stream_config, phase, current_phase, reference_policy_id):
        """Get evaluation config for decoder/interface phases."""
        eval_configs = []
        
        for ref_policy in current_phase.eval_ref_policies:
            if reference_policy_id and ref_policy != reference_policy_id:
                continue
            
            # Find matching policy phase
            policy_info = None
            for idx in range(phase):
                if ref_policy in stream_config.datastream[idx].phase_name:
                    policy_info = (idx, stream_config.datastream[idx].phase_name)
                    break
            
            if policy_info:
                eval_configs.append({
                    'agent_config': {
                        'decoder': (phase, current_phase.phase_name),
                        'interface': (phase, current_phase.phase_name),
                        'policy': policy_info,
                    },
                    'eval_tasks': stream_config.datastream[policy_info[0]].eval_tasks,
                })
        
        return eval_configs


# ============================================================================
# Default Configurations
# ============================================================================

DEFAULT_DATASTREAM = [
    SkillPhaseConfig(
        phase_name='task0',
        dataset_paths=[f"{skill_dataset_path}/bottom burner-top burner-light switch-slide cabinet.pkl"]
    ),
    SkillPhaseConfig(
        phase_name='task1',
        dataset_paths=[f"{skill_dataset_path}/microwave-bottom burner-light switch-slide cabinet.pkl"]
    ),
    SkillPhaseConfig(
        phase_name='task2',
        dataset_paths=[f"{skill_dataset_path}/microwave-kettle-bottom burner-hinge cabinet.pkl"]
    )
]

DEFAULT_SKILL_STREAM_CONFIG = SkillStreamConfig(datastream=DEFAULT_DATASTREAM)
