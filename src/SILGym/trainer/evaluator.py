import os
import json
import cloudpickle
import numpy as np
from SILGym.environment.remote import KitchenRemoteEvaluator
from SILGym.models.agent.base import PTGMAgent, IsCiLAgent
from SILGym.config.skill_stream_config import EvaluationTracer
from SILGym.utils.logger import get_logger
from SILGym.config.experiment_config import SkillExperimentConfig
import jax
from copy import deepcopy
import gc
from SILGym.utils.metadata import build_phase_metadata
class SkillEvaluator:
    def __init__(self, exp_save_path=None, experiment_config=None, remote_eval_host='127.0.0.1', remote_eval_port=9999,
                 eval_num_episodes=3, eval_max_steps=280, eval_noise_enabled=False,
                 eval_noise_scale=0.01, eval_noise_clip=None, eval_noise_seed=None, output_suffix="", output_postfix="post",
                 force_static=False):
        """
        Initializes the Evaluator's basic environment.

        Parameters:
            exp_save_path: Path where the experiment's models are saved.
            experiment_config: Optional SkillExperimentConfig object for loading configurations.
            remote_eval_host: Host for remote evaluation.
            remote_eval_port: Port for remote evaluation.
            eval_num_episodes: Number of episodes for evaluation.
            eval_max_steps: Maximum steps for evaluation.
            eval_noise_enabled: Whether to add Gaussian noise during evaluation.
            eval_noise_scale: Scale/magnitude of Gaussian noise to add.
            eval_noise_clip: Optional clipping range for noisy observations.
            eval_noise_seed: Random seed for noise reproducibility.
            output_suffix: Suffix to add to output filenames.
            output_postfix: Postfix for the output filename (e.g., 'post' results in 'eval_results_post.json').
            force_static: Force use of static agent instead of default agent class.
        """
        self.logger = get_logger(__name__)
        
        # Load experiment config if provided or from file
        if experiment_config is not None:
            self.experiment_config = experiment_config
            self.exp_save_path = self.experiment_config.exp_save_path
        elif exp_save_path is not None:
            self.exp_save_path = exp_save_path
            # Try to load experiment config from file
            config_path = os.path.join(exp_save_path, "experiment_config.pkl")
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    self.experiment_config = cloudpickle.load(f)
                self.logger.info(f"Loaded experiment config from {config_path}")
            else:
                self.experiment_config = None
                self.logger.warning(f"No experiment config found at {config_path}")
        else:
            raise ValueError("Either exp_save_path or experiment_config must be provided")
        
        self.remote_eval_host = remote_eval_host
        self.remote_eval_port = remote_eval_port
        self.eval_num_episodes = eval_num_episodes
        self.eval_max_steps = eval_max_steps

        # Noise settings (Gaussian)
        self.eval_noise_enabled = eval_noise_enabled
        self.eval_noise_scale = eval_noise_scale
        self.eval_noise_clip = eval_noise_clip
        self.eval_noise_seed = eval_noise_seed
        self.output_suffix = output_suffix
        self.output_postfix = output_postfix
        self.force_static = force_static
        
        # Initialize evaluation tracking similar to skill_trainer
        self.eval_results = {}
        
        # Model caching for efficient loading
        self.eval_decoder = None
        self.eval_interface = None
        self.eval_policy = None
        self.prev_agent_config = {
            'decoder': (0, None),
            'interface': (0, None),
            'policy': (0, None),
        }
        
        # Get agent class from experiment config
        if self.experiment_config is not None:
            self.agent_cls = self.experiment_config.agent_cls
            self.scenario_config = self.experiment_config.scenario_config
            self.datastream = self.scenario_config.datastream if hasattr(self.scenario_config, 'datastream') else None
            
            # Override agent class if force_static is enabled
            if self.force_static:
                # Check the algorithm type to determine the appropriate static agent
                algo_type = getattr(self.experiment_config, 'algo_type', None)
                if algo_type in ['lazysi' , 'silc']:
                    # TODO: Implement LazySIStaticAgent if needed
                    self.logger.warning(f"Static agent not yet implemented for {algo_type}, using PTGMAgent")
                    self.agent_cls = PTGMAgent
                else:
                    # For PTGM and others, keep the default PTGMAgent (which is already static)
                    self.agent_cls = PTGMAgent
                    self.logger.info(f"Using PTGMAgent for static evaluation (algo_type: {algo_type})")
        else:
            self.agent_cls = PTGMAgent  # Default agent
            self.scenario_config = None
            self.datastream = None

        # Validation: Check for environment variant mismatches
        self._validate_evaluator_config()

    def _validate_evaluator_config(self):
        """
        Validate that the evaluator configuration matches the environment variant.
        Warns if using wrong evaluator for studio/vision variants.
        """
        # Check if path contains studio/vision indicators
        path_lower = self.exp_save_path.lower()
        is_studio_path = 'studio' in path_lower or 'kitchenstudio' in path_lower
        is_vision_path = 'vis' in path_lower or 'vision' in path_lower

        if self.experiment_config is None:
            if is_studio_path or is_vision_path:
                self.logger.warning("="*80)
                self.logger.warning("CONFIGURATION WARNING")
                self.logger.warning("="*80)
                self.logger.warning(f"Experiment path suggests studio/vision variant: {self.exp_save_path}")
                self.logger.warning("But experiment_config.pkl is missing!")
                self.logger.warning("This may cause evaluation to fail for vision-based environments.")
                self.logger.warning("Ensure experiment_config.pkl exists in the experiment directory.")
                self.logger.warning("="*80)
            return

        # Check evaluator class matches environment
        evaluator_cls = getattr(self.experiment_config, 'evaluator_cls', None)
        if evaluator_cls is not None:
            evaluator_name = evaluator_cls.__name__

            # Warn if using basic evaluator for studio environment
            if is_studio_path and evaluator_name == 'KitchenRemoteEvaluator':
                self.logger.warning("="*80)
                self.logger.warning("EVALUATOR MISMATCH WARNING")
                self.logger.warning("="*80)
                self.logger.warning(f"Path suggests kitchenstudio environment: {self.exp_save_path}")
                self.logger.warning(f"But using basic evaluator: {evaluator_name}")
                self.logger.warning("Expected: KitchenStudioEmbedRemoteEvaluator")
                self.logger.warning("This may cause evaluation to fail!")
                self.logger.warning("="*80)

            # Log successful configuration for studio/vision
            if (is_studio_path or is_vision_path) and 'Studio' in evaluator_name:
                self.logger.info(f"✓ Validated: Using {evaluator_name} for studio/vision environment")
                remote_kwargs = getattr(self.experiment_config, 'remote_eval_kwargs', {})
                if remote_kwargs and 'camera_keys' in remote_kwargs:
                    self.logger.info(f"✓ Vision config: {len(remote_kwargs['camera_keys'])} camera(s)")

    def _load_decoder(self, decoder_id):
        """
        Load the decoder model saved for the given phase (identified by decoder_id).
        """
        skill_dir = os.path.join(self.exp_save_path, "skills")
        file_path = os.path.join(skill_dir, f"decoder_{decoder_id}.pkl")
        if os.path.exists(file_path):
            self.logger.info(f"Loading decoder model from {file_path}")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            self.logger.warning(f"Decoder model file {file_path} does not exist.")
            return None

    def _load_interface(self, interface_id):
        """
        Load the interface model saved for the given phase (identified by interface_id).
        """
        skill_dir = os.path.join(self.exp_save_path, "skills")
        file_path = os.path.join(skill_dir, f"interface_{interface_id}.pkl")
        if os.path.exists(file_path):
            self.logger.info(f"Loading interface model from {file_path}")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            self.logger.warning(f"Interface model file {file_path} does not exist.")
            return None

    def _load_policy(self, policy_id):
        """
        Load the policy model saved for the given phase (identified by policy_id).
        """
        policy_parent_dir = os.path.join(self.exp_save_path, "policy")
        file_path = os.path.join(policy_parent_dir, f"{policy_id}.pkl")
        if os.path.exists(file_path):
            self.logger.info(f"Loading policy model from {file_path}")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            self.logger.warning(f"Policy model file {file_path} does not exist.")
            return None

    def _build_agent(self, agent_config):
        """
        Build the agent using trace information from the scenario configuration.
        The trace information indicates the last phase where each target (decoder, interface, policy)
        was trained. This method loads the corresponding models (using custom loading functions)
        and constructs a new agent.

        # Parameters :
            - 'agent_config': A dict with keys 'decoder', 'interface', and 'policy'
                              mapping to the corresponding checkpoint information.
        # E.g  {'decoder': (25, 'pre_1'), 'interface': (25, 'pre_1'), 'policy': (24, 'policy_24/pre_0')}
        """
        # Check if decoder model needs to be reloaded
        if self.prev_agent_config['decoder'][1] != agent_config['decoder'][1]:
            self.logger.info(f"Loading decoder model for phase {agent_config['decoder'][1]}")
            # Clear previous decoder from memory if it exists
            if hasattr(self, 'eval_decoder') and self.eval_decoder is not None:
                del self.eval_decoder
                jax.clear_caches()  # Clear JAX caches to prevent memory leakage
            self.eval_decoder = self._load_decoder(agent_config['decoder'][1])
        
        # Check if interface model needs to be reloaded
        if self.prev_agent_config['interface'][1] != agent_config['interface'][1]:
            self.logger.info(f"Loading interface model for phase {agent_config['interface'][1]}")
            # Clear previous interface from memory if it exists
            if hasattr(self, 'eval_interface') and self.eval_interface is not None:
                del self.eval_interface
                jax.clear_caches()  # Clear JAX caches to prevent memory leakage
            self.eval_interface = self._load_interface(agent_config['interface'][1])
        
        # Always reload policy (based on your original code)
        # Clear previous policy from memory if it exists
        # NOTE for experiment we assume all policy models share the same architecture
        if self.prev_agent_config['policy'][1] != agent_config['policy'][1]:
            self.logger.info(f"Loading policy model for phase {agent_config['policy'][1]}")
            loaded_policy = self._load_policy(agent_config['policy'][1])
            if self.eval_policy is None:
                self.eval_policy = loaded_policy
            else:
                self.eval_policy.train_state = self.eval_policy.train_state.replace(
                    params=loaded_policy.train_state.params
                )
                if self.experiment_config and self.experiment_config.algo_type in ['lazysi', 'silc']:
                    self.logger.info(f"Loading policy prototype for phase {agent_config['policy'][1]}")
                    self.eval_policy.set_subtask_prototype(loaded_policy.subtask_prototypes)

        # Build and return the agent using the loaded models.
        agent = self.agent_cls(
            decoder=self.eval_decoder,
            interface=self.eval_interface,
            policy=self.eval_policy,
        )
        # update prev_agent_config
        self.prev_agent_config = agent_config
        return agent
    
    def configure_noise(self, enabled=None, noise_scale=None,
                       noise_clip=None, noise_seed=None):
        """
        Configure Gaussian noise settings for evaluation.

        Parameters:
            enabled: Whether to enable Gaussian noise
            noise_scale: Scale/magnitude of Gaussian noise
            noise_clip: Optional clipping range
            noise_seed: Random seed for reproducibility
        """
        if enabled is not None:
            self.eval_noise_enabled = enabled
        if noise_scale is not None:
            self.eval_noise_scale = noise_scale
        if noise_clip is not None:
            self.eval_noise_clip = noise_clip
        if noise_seed is not None:
            self.eval_noise_seed = noise_seed

        if self.eval_noise_enabled:
            self.logger.info(f"Gaussian noise configured: scale={self.eval_noise_scale}")
    
    def _get_noise_scale(self):
        """
        Get noise scale if noise is enabled.
        
        Returns:
            Noise scale if enabled, None otherwise
        """
        if self.eval_noise_enabled:
            return self.eval_noise_scale
        return None

    @staticmethod
    def _extract_phase_name_from_agent(agent_config):
        if not agent_config:
            return None
        policy_info = agent_config.get('policy')
        if isinstance(policy_info, (list, tuple)) and len(policy_info) >= 2:
            return policy_info[1]
        if isinstance(policy_info, str):
            return policy_info
        return None

    def evaluate_direct_config(self, agent_config, eval_tasks):
        """
        Performs evaluation in direct config mode.

        Parameters:
            agent_config: A dict with keys 'decoder', 'interface', 'policy' specifying the checkpoint information.
                          (Filenames should be provided without the .pkl extension.)
            eval_tasks: A list of evaluation tasks (each task can be a dict specifying task details).
        """
        agent = self._build_agent(agent_config)
        jax.clear_caches()

        overall_rewards = []
        detailed_results = {}

        # Get noise scale if enabled
        noise_scale = self._get_noise_scale()

        # Get evaluator configuration from experiment_config if available
        if self.experiment_config:
            evaluator_cls = getattr(self.experiment_config, 'evaluator_cls', KitchenRemoteEvaluator)
            extra_remote_kwargs = getattr(self.experiment_config, 'remote_eval_kwargs', {}) or {}
            base_obs_helper = getattr(self.experiment_config, 'remote_obs_helper', None)
            self.logger.info(f"Using evaluator from config: {evaluator_cls.__name__}")
            if extra_remote_kwargs:
                self.logger.info(f"Using remote_eval_kwargs: {list(extra_remote_kwargs.keys())}")
        else:
            evaluator_cls = KitchenRemoteEvaluator
            extra_remote_kwargs = {}
            base_obs_helper = None
            self.logger.warning("No experiment_config available, using default KitchenRemoteEvaluator")

        for task in eval_tasks:
            self.logger.info(f"Evaluating task: {task}")
            remote_evaluator = evaluator_cls(
                host=self.remote_eval_host,
                port=self.remote_eval_port,
                obs_helper=base_obs_helper,
                eval_fn=agent.eval,
                **extra_remote_kwargs,
            )
            metadata_payload = build_phase_metadata(
                agent_config=agent_config,
                phase_name=self._extract_phase_name_from_agent(agent_config),
                environment=getattr(self.scenario_config, "environment", None) if self.scenario_config else None,
            )
            # Add extra metadata from experiment config (critical for vision/studio modes)
            if self.experiment_config:
                extra_metadata = getattr(self.experiment_config, 'remote_eval_metadata', None) or {}
                if extra_metadata:
                    metadata_payload = metadata_payload or {}
                    for key, value in extra_metadata.items():
                        if value is None:
                            continue
                        metadata_payload[key] = str(value)
                    self.logger.info(f"Added remote_eval_metadata: {list(extra_metadata.keys())}")
            remote_evaluator.set_task(
                [task],
                noise_scale=noise_scale,
                metadata=metadata_payload or None,
            )
            eval_rewards, _ = remote_evaluator.evaluate(
                num_episodes=self.eval_num_episodes,
                max_steps=self.eval_max_steps
            )
            remote_evaluator.close()
            eval_rewards_list = [float(r) for r in eval_rewards] if eval_rewards is not None else []
            avg_reward = np.mean(eval_rewards_list) if eval_rewards_list else 0.0
            self.logger.info(f"Task {task} average reward: {avg_reward:.2f}")
            detailed_results[str(task)] = {
                "rewards": eval_rewards_list,
                "avg_reward": avg_reward
            }
            overall_rewards.extend(eval_rewards_list)
        overall_avg_reward = np.mean(overall_rewards) if overall_rewards else 0.0
        results = {
            "agent_config": agent_config,
            "eval_tasks": eval_tasks,
            "overall_rewards": overall_rewards,
            "overall_avg_reward": overall_avg_reward,
            "detailed": detailed_results,
        }
        # Build filename with optional suffix
        filename = "evaluation_results_direct"
        if self.output_suffix:
            filename += f"_{self.output_suffix}"
        filename += ".json"
        results_save_path = os.path.join(self.exp_save_path, filename)
        with open(results_save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Direct evaluation results saved to {results_save_path}.")
        return results

    def evaluate_from_datastream(self, skill_stream_config=None):
        """
        Performs evaluation from a skill stream configuration.
        
        This function iterates through each phase in the datastream and uses the EvaluationTracer
        to obtain the evaluation configurations. It then evaluates the models just like in the trainer.
        
        Parameters:
            skill_stream_config: An instance of SkillStreamConfig (must have a 'datastream' attribute).
                               If None, uses self.scenario_config from experiment_config.
        """
        # Use provided config or default to scenario_config from experiment
        if skill_stream_config is None:
            if self.scenario_config is None:
                raise ValueError("No skill_stream_config provided and no scenario_config found in experiment_config")
            skill_stream_config = self.scenario_config
            
        if not hasattr(skill_stream_config, "datastream"):
            raise ValueError("The provided skill_stream_config does not have a datastream attribute.")
            
        datastream = skill_stream_config.datastream
        eval_tracer = EvaluationTracer()
        
        # Get evaluation settings from experiment config if available
        if self.experiment_config:
            num_eval_episodes = getattr(self.experiment_config, 'eval_num_episodes', self.eval_num_episodes)
            max_eval_steps = getattr(self.experiment_config, 'eval_max_steps', self.eval_max_steps)
            evaluator_cls = getattr(self.experiment_config, 'evaluator_cls', KitchenRemoteEvaluator)
            remote_eval_host = getattr(self.experiment_config, 'remote_eval_host', self.remote_eval_host)
            remote_eval_port = getattr(self.experiment_config, 'remote_eval_port', self.remote_eval_port)
            base_obs_helper = getattr(self.experiment_config, 'remote_obs_helper', None)
            extra_remote_kwargs = getattr(self.experiment_config, 'remote_eval_kwargs', {}) or {}
            self.logger.info(f"Using evaluator from config: {evaluator_cls.__name__}")
        else:
            num_eval_episodes = self.eval_num_episodes
            max_eval_steps = self.eval_max_steps
            evaluator_cls = KitchenRemoteEvaluator
            remote_eval_host = self.remote_eval_host
            remote_eval_port = self.remote_eval_port
            base_obs_helper = None
            extra_remote_kwargs = {}
            self.logger.warning("No experiment_config available, using default KitchenRemoteEvaluator")
        
        # Get noise scale if enabled
        noise_scale = self._get_noise_scale()
            
        num_phases = len(datastream)
        for phase in range(num_phases):
            self.logger.info(f"Starting remote evaluation for phase {phase}...")
            eval_traced_list = eval_tracer.get_eval_tasks_by_reference(
                stream_config=skill_stream_config,
                phase=phase,
            )
            self.logger.info(f"Evaluation tasks for phase {phase}: {eval_traced_list}")
            
            if len(eval_traced_list) == 0:
                self.logger.info(f"No evaluation tasks provided for phase {phase}. Skipping remote evaluation.")
                self.eval_results[phase] = {
                    "train_targets": datastream[phase].train_targets,
                    "eval_tasks": None,
                    "overall_rewards": None,
                    "overall_avg_reward": None,
                    "detailed": None,
                }
                # Build filename with postfix and optional suffix
                filename = f"eval_results_{self.output_postfix}"
                if self.output_suffix:
                    filename += f"_{self.output_suffix}"
                filename += ".json"
                eval_save_path = os.path.join(self.exp_save_path, filename)
                with open(eval_save_path, "w", encoding="utf-8") as f:
                    json.dump(self.eval_results, f, indent=4)
                continue
                
            overall_rewards = []
            detailed_results = {}
            
            # Evaluate each task individually.
            for eval_trace in eval_traced_list:
                agent_config = eval_trace.get('agent_config', None)
                eval_tasks = eval_trace.get('eval_tasks', None)
                agent = self._build_agent(agent_config)
                self.logger.info(f"Building agent for task '{eval_tasks}' with config: {agent_config}")
                
                # Create a remote evaluator instance for this task.
                remote_evaluator = evaluator_cls(
                    host=remote_eval_host,
                    port=remote_eval_port,
                    obs_helper=base_obs_helper,
                    eval_fn=agent.eval,
                    **extra_remote_kwargs,
                )
                
                self.logger.info(f"Remote evaluator initialized for task '{eval_tasks}'")
                # Set the remote evaluator to use only this eval task.
                metadata_payload = build_phase_metadata(
                    agent_config=agent_config,
                    phase_name=datastream[phase].phase_name if phase < len(datastream) else None,
                    environment=getattr(skill_stream_config, "environment", None),
                )
                # Add extra metadata from experiment config (critical for vision/studio modes)
                if self.experiment_config:
                    extra_metadata = getattr(self.experiment_config, 'remote_eval_metadata', None) or {}
                    if extra_metadata:
                        metadata_payload = metadata_payload or {}
                        for key, value in extra_metadata.items():
                            if value is None:
                                continue
                            metadata_payload[key] = str(value)
                remote_evaluator.set_task(
                    eval_tasks,
                    noise_scale=noise_scale,
                    metadata=metadata_payload or None,
                )
                
                # Evaluate on the current task.
                eval_rewards, eval_dict = remote_evaluator.evaluate(
                    num_episodes=num_eval_episodes, max_steps=max_eval_steps
                )
                remote_evaluator.close()
                
                # Process the evaluation rewards.
                eval_rewards_list = [float(r) for r in eval_rewards] if eval_rewards is not None else []
                avg_reward = np.mean(eval_rewards_list) if eval_rewards_list else 0.0
                self.logger.info(f"Evaluation for task '{eval_tasks}' in phase {phase}: Average Reward: {avg_reward:.2f}")
                
                # Save the results for this task.
                for task_configures in eval_tasks:
                    task_name = task_configures.get('data_name', 'unknown_task')
                    self.logger.info(f"Save configures '{task_name}' in phase {phase}: Average Reward: {avg_reward:.2f}")
                    detailed_results[task_name] = {
                        "rewards": eval_rewards_list,
                        "avg_reward": avg_reward,
                    }
                overall_rewards.extend(eval_rewards_list)
                
            # Compute overall average reward across all evaluated tasks.
            overall_avg_reward = np.mean(overall_rewards) if overall_rewards else 0.0
            
            # Save the evaluation results for the phase.
            eval_results = {
                "phase_name": datastream[phase].phase_name,
                "train_targets": datastream[phase].train_targets,
                "eval_tasks": eval_tasks,  # This will be the last eval_tasks from the loop
                "overall_rewards": overall_rewards,
                "overall_avg_reward": overall_avg_reward,
                "detailed": detailed_results,
            }
            self.eval_results[phase] = eval_results
            
            # Write the accumulated evaluation results to a JSON file.
            # Build filename with postfix and optional suffix
            filename = f"eval_results_{self.output_postfix}"
            if self.output_suffix:
                filename += f"_{self.output_suffix}"
            filename += ".json"
            eval_save_path = os.path.join(self.exp_save_path, filename)
            with open(eval_save_path, "w", encoding="utf-8") as f:
                json.dump(self.eval_results, f, indent=4)
                
            self.logger.info(f"Evaluation results saved to {eval_save_path}")
            
            # Memory cleanup after each phase
            gc.collect()
            jax.clear_caches()
            self.logger.debug(f"JAX caches cleared.")
            
        return self.eval_results

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    exp_save_path = "./logs/kitchen/kitchenem/sync/lazysi_conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4/0725seed6maha/"  # Actual experiment path

    # Example 1: Direct config evaluation.
    evaluator = SkillEvaluator(exp_save_path)
    # Prepare a direct agent config (filenames provided without .pkl extension)
    if False :
        direct_agent_config = {
            'decoder': (0, 'pre_0'),
            'interface': (0, 'pre_0'),
            'policy': (0, 'policy_2/pre_1')
        }
        # Define evaluation tasks directly (e.g., a list of task dicts)
        direct_eval_tasks = [
            {'data_name': 'mbls'} 
        ]
        evaluator.evaluate_direct_config(direct_agent_config, direct_eval_tasks)

    # Example 2: Evaluation from datastream.
    # The evaluator will automatically load the experiment_config.pkl from exp_save_path
    # and use the scenario_config from it.
    evaluator.evaluate_from_datastream()
