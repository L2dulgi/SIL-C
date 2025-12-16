import os
import cloudpickle
import numpy as np
from tqdm import tqdm
import json
import time
import datetime
import jax
import random
from SILGym.utils.logger import get_logger
# Example imports (adjust as needed to match your code/project)
from SILGym.config.experiment_config import SkillExperimentConfig
from SILGym.dataset.dataloader import MemoryBuffer, GPUDataLoader
from SILGym.config.skill_stream_config import  EvaluationTracer
from SILGym.utils.metadata import build_phase_metadata
import gc

class BaseTrainer:
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    # Core function for continual training
    def continual_train(self):
        for phase in range(self.continual_scenario.phase_num):
            self.logger.info(f'[phase {phase}] start' + 'V'*20)
            self.phase_train(phase)
            self.logger.info(f'[phase {phase}] ended' + '^'*20)

    # Function to train one phase
    def phase_train(self, phase: int):
        self.load_phase_data(phase)
        self.init_phase_model(phase)
        self.train_phase_model(phase)
        self.save_phase_model(phase)
        self.process_phase_data(phase)
    
    def load_phase_data(self, phase: int):
        raise NotImplementedError
    
    def init_phase_model(self, phase: int):
        raise NotImplementedError
    
    def train_phase_model(self, phase: int):
        raise NotImplementedError
    
    def save_phase_model(self, phase: int):
        raise NotImplementedError
    
    def process_phase_data(self, phase: int):
        raise NotImplementedError

from rich.console import Console
from rich.table import Table
import numpy as np

def display_unique_reward_logits(stacked_data):
    """
    For each unique skill, prints the unique reward_logits and their frequencies.
    
    Parameters:
        stacked_data (dict): A dictionary containing the keys:
            - 'skills' (shape: (B,))
            - 'reward_logits' (shape: (B,1))
    """
    # 'skills' is shaped (B,), and 'reward_logits' is (B,1), so we flatten reward_logits
    skills = np.array(stacked_data['skills'])
    reward_logits = np.array(stacked_data['reward_logits']).flatten()  # shape: (B,)

    unique_skills = np.unique(skills)
    console = Console()

    for skill in unique_skills:
        # Select indices corresponding to the current skill
        indices = np.where(skills == skill)[0]
        rewards = reward_logits[indices]
        # Compute unique reward_logits and their frequencies
        unique_rewards, counts = np.unique(rewards, return_counts=True)

        # Create a rich table for each skill
        table = Table(title=f"Skill: {skill}")
        table.add_column("Reward Logit", justify="center")
        table.add_column("Count", justify="center")

        for r, count in zip(unique_rewards, counts):
            table.add_row(str(r), str(count))

        console.print(table)


class SkillTrainer(BaseTrainer):
    def __init__(self, experiment_config: SkillExperimentConfig = None, do_eval: bool = False):
        """
        SkillTrainer handles a multi-phase training process.
        The scenario_config holds dataset information per phase.
        The experiment_config contains settings for the decoder, dataloader, and memory buffer.
        The 'do_eval' flag determines whether to perform remote evaluation after each phase.

        Now includes:
         - Instantiation of a PrintLogger (logs to exp_save_path/skill_trainer.log).
         - Saves the experiment_config to exp_save_path/experiment_config.pkl.
        """
        super().__init__()
        if experiment_config is None:
            raise ValueError("[SkillTrainer] Missing experiment_config")
        self.experiment_config = experiment_config
        self.experiment_config.print_and_save_config()
        self.do_eval = do_eval
        
        # Initialize logger specifically for SkillTrainer
        self.logger = get_logger(__name__)

        # -------------------------------------------------
        # 1) Save the experiment_config for reproducibility
        # -------------------------------------------------
        config_save_path = os.path.join(self.experiment_config.exp_save_path, "experiment_config.pkl")
        with open(config_save_path, 'wb') as f:
            cloudpickle.dump(self.experiment_config, f)
        self.logger.info(f"Experiment config saved to {config_save_path}")

        # Set the random seed for reproducibility
        np.random.seed(self.experiment_config.seed)
        random.seed(self.experiment_config.seed)
        jax.random.PRNGKey(self.experiment_config.seed)
        self.logger.info(f"Random seed set to {self.experiment_config.seed}")

        # Visualize the scenario configuration
        if hasattr(self.experiment_config.scenario_config, 'visualize_stream'):
            save_path = os.path.join(self.experiment_config.exp_save_path, 'scenario_config.png')
            self.experiment_config.scenario_config.visualize_stream(save_path=save_path)
            self.logger.info(f"Scenario stream visualization saved to {save_path}")

        # Initialize a dictionary to store evaluation rewards per phase
        self.eval_results = {}
  
        # Process scenario configuration
        self.scenario_config = self.experiment_config.scenario_config
        self.datastream = self.scenario_config.datastream
        self.phase_length = len(self.datastream)

        # DataLoader for the current phase
        self.dataloader_cls = self.experiment_config.dataloader_cls
        self.current_dataloader = None

        # Memory buffer to accumulate data across phases
        self.memory_buffer = MemoryBuffer(data_paths=[])

        # Decoder configuration (formerly self.model)
        self.decoder_config = self.experiment_config.decoder_config
        self.decoder = None
        self.decoder_pretrained : bool = False

        # Interface configuration
        self.interface_config = self.experiment_config.interface_config
        self.interface = None

        # Policy configuration
        self.policy_config = self.experiment_config.policy_config
        self.policy = None      

        # Agent (Policy - Interface - Decoder) initialization  
        self.eval_decoder = None
        self.eval_interface = None
        self.eval_policy = None

        self.agent_cls = self.experiment_config.agent_cls
        self.agent = None 

        if self.agent_cls is None:
            raise ValueError(f"[SkillTrainer] Invalid algorithm type: {self.experiment_config.algo_type}")
        # -------------------------------------------------
        # Agent evaluation optimization
        # -------------------------------------------------
        self.prev_agent_config = {
            'decoder': (0, None),
            'interface': (0, None),
            'policy': (0, None),
        }

        self.logger.info("SkillTrainer initialization complete.")

    def close_logger(self):
        """
        Placeholder for logger cleanup.
        """
        pass

    def continual_train(self):
        """
        Multi-phase training process:
        1) Load phase data
        2) Load or initialize decoder (skill model)
        3) Train the decoder and policy (via separated steps)
        4) Optionally perform remote evaluation
        5) Save the decoder
        6) Update the memory buffer
        """
        total_start_time = time.time()
        for phase in range(self.phase_length):
            phase_start_time = time.time()
            self.logger.info(f"\nStarting phase {phase}")

            self.load_phase_data(phase)
            self.load_phase_model(phase)
            self.phase_train(phase)
            self.save_phase_model(phase)
            self.update_buffer(phase)
            if self.do_eval:
                self.eval_phase_model(phase)
            
            phase_elapsed = time.time() - phase_start_time
            self.logger.info(f"Finished phase {phase} : Phase elapsed time: {phase_elapsed:.2f} seconds")

            # Calculate total elapsed time so far and estimate the remaining time
            total_elapsed = time.time() - total_start_time
            average_phase_time = total_elapsed / (phase + 1)
            remaining_phases = self.phase_length - (phase + 1)
            estimated_remaining_time = average_phase_time * remaining_phases
            
            # Format the estimated remaining time as HH:MM:SS
            formatted_eta = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
            self.logger.info(f"Estimated remaining time for experiment: {formatted_eta}")

            # Memory leackage handling for jax.
            gc.collect()
            jax.clear_caches()
            self.logger.debug(f"JAX caches cleared.")
            
        
        total_elapsed = time.time() - total_start_time
        # Format total elapsed time as HH:MM:SS
        formatted_total_time = str(datetime.timedelta(seconds=int(total_elapsed)))
        self.logger.info(f"\nTotal experiment elapsed time: {formatted_total_time}")

    def load_phase_data(self, phase):
        """
        Load data for the given phase and prepare the dataloader.
        """
        phase_data_paths = self.datastream[phase].dataset_paths 
        if phase_data_paths is not None:
            if 'policy' in self.datastream[phase].train_targets:
                self.logger.info(f"Loading data for policy training in phase {phase}")
                self.current_dataloader = self.dataloader_cls(
                    phase_data_paths, **self.experiment_config.dataloader_kwargs_policy
                )
            else :
                self.current_dataloader = self.dataloader_cls(
                    phase_data_paths, **self.experiment_config.dataloader_kwargs
                )
            self.logger.info(f"Phase {phase} data loaded from: {phase_data_paths}")
        else:
            raise ValueError(f"[SkillTrainer] No data paths found for phase {phase}")

    def reset_phase_model(self, mode):
        """
        Reset the decoder and policy models for the given phase.
        """
        if 'decoder' in mode and self.decoder is not None:
            # Check if we should reset decoder from scratch (ftscratch algorithm)
            if getattr(self.experiment_config, 'reset_decoder_each_phase', False):
                self.logger.info(f"Resetting decoder from scratch (ftscratch algorithm)")
                new_decoder = self.decoder_config['model_cls'](
                    **self.decoder_config['model_kwargs']
                )
                self.decoder.train_state = self.decoder.train_state.replace(
                    step=new_decoder.train_state.step,
                    params=new_decoder.train_state.params,
                    tx=new_decoder.train_state.tx,
                    opt_state=new_decoder.train_state.opt_state,
                )
            # NOTE 0405 only reinitialize the decoder for append model Added On
            elif isinstance(self.decoder, self.experiment_config.appender_cls) == True:
                self.decoder.reinit_optimizer()
            else :
                self.logger.info(f"Do not reset the model for FT and ER")

        if 'policy' in mode and self.policy is not None:
            new_policy = self.policy_config['model_cls'](
                **self.policy_config['model_kwargs']
            )

            self.policy.train_state = self.policy.train_state.replace(
                step=new_policy.train_state.step,
                params=new_policy.train_state.params,
                tx=new_policy.train_state.tx,
                opt_state=new_policy.train_state.opt_state,
            )

        self.logger.info(f"Phase model reset for mode {mode}")

    def load_phase_model(self, phase):
        """
        Load or initialize the decoder and policy for the given phase.
        """
        if self.interface is None:
            self.interface = self.interface_config['interface_cls'](
                **self.interface_config['interface_kwargs']
            )
        self.logger.info(f"Interface {self.interface_config['interface_cls']} ready for phase {phase}")

        if self.decoder is None:
            self.decoder = self.decoder_config['model_cls'](
                **self.decoder_config['model_kwargs']
            )
        self.logger.info(f"Decoder {self.decoder_config['model_cls']} ready for phase {phase}")

        if self.policy is None:
            self.policy = self.policy_config['model_cls'](
                **self.policy_config['model_kwargs']
            )
        self.logger.info(f"Policy {self.policy_config['model_cls']} ready for phase {phase}")


        self.agent = self.agent_cls(
            decoder=self.decoder,
            interface=self.interface,
            policy=self.policy,
        )

    def phase_train(self, phase):
        """
        Train the decoder and policy for the given phase using the current dataloader.
        Training is divided into three sequential steps:
          1. Interface initialization/training
          2. Decoder training
          3. Policy training
        """
        if not self.current_dataloader:
            self.logger.warning(f"No dataloader for phase {phase}. Skipping training.")
            return 
        
        # Clear the phase model if needed.
        if self.experiment_config.phase_reset is True :
            self.reset_phase_model(self.datastream[phase].train_targets)

        def wrap_appender_model():
            # Decoder appending after initial pre-training
            if (
                    self.decoder_pretrained == True 
                    and isinstance(self.decoder, self.experiment_config.appender_cls) == False
                ):
                # Load the decoder model for the current phase
                self.decoder = self.experiment_config.appender_cls(
                    base_model=self.decoder,
                    append_config=self.experiment_config.appender_config,
                )
                self.logger.info(f"Decoder Appended {phase}: {self.decoder}")
            else:
                self.logger.info(f"Decoder already appended {phase}: {self.decoder}")

        mode = getattr(self.datastream[phase], "train_targets", ['interface', 'decoder', 'policy'])
        if 'interface' in mode:
            self._train_interface(phase)
        if 'decoder' in mode:
            self._train_decoder(phase)
            if self.experiment_config.is_appendable == True : 
                wrap_appender_model()
        if 'policy' in mode:
            self._train_policy(phase)

        self.logger.info(f"Finished training for phase {phase}")

    def _train_interface(self, phase):
        """
        Step 1: Interface initialization/training.
        Update the current dataloader using the interface if available.
        """
        self.logger.info(f"Starting interface initialization for phase {phase}.")
        if self.interface is not None:
            start_time = time.time()
            self.current_dataloader = self.interface.update_interface(self.current_dataloader)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Interface initialization completed for phase {phase} in {elapsed_time:.2f} seconds: {self.interface}")
        else:
            self.logger.warning(f"No interface available for phase {phase}. Skipping interface initialization.")

    def _train_decoder(self, phase):
        """
        Step 2: Decoder training.
        Train the decoder model using the current dataloader.
        """
        self.decoder_pretrained = True
        batch_size = self.experiment_config.batch_size
        phase_epochs = self.experiment_config.phase_epochs
        dataloader_kwargs = {
            'batch_size': batch_size,
        }
        appender_mode = isinstance(self.decoder, self.experiment_config.appender_cls)
        if appender_mode == True:
            dataloader_kwargs['pool_key'] = 'decoder_id'

        if self.memory_buffer.is_empty() == False:
            self.logger.info(f"Using memory buffer for decoder training in phase {phase}")
            decoder_dataloader = self.experiment_config.dataloader_mixer_cls(
                [self.current_dataloader,self.memory_buffer],
                mixing_ratios=[1,1]
            )
            phase_epochs = phase_epochs * 2
        else :
            self.logger.info(f"Using current dataloader for decoder training in phase {phase}")
            decoder_dataloader = self.current_dataloader

        # Optionally wrap with GPU dataloader for acceleration
        use_gpu_dataloader = getattr(self.experiment_config, 'use_gpu_dataloader', False)
        if use_gpu_dataloader:
            self.logger.info(f"Wrapping decoder dataloader with GPUDataLoader")
            decoder_dataloader = GPUDataLoader(
                base_dataloader=decoder_dataloader,
                mode=getattr(self.experiment_config, 'gpu_dataloader_mode', 'auto'),
                gpu_memory_threshold=getattr(self.experiment_config, 'gpu_memory_threshold', 0.3),
                prefetch_buffer_size=getattr(self.experiment_config, 'prefetch_buffer_size', 2)
            )

        self.logger.info(f"Starting decoder training for phase {phase}: epochs={phase_epochs}, batch_size={batch_size}")
        pbar = tqdm(range(phase_epochs), desc=f"Phase {phase} Decoder Training")
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_recon = 0.0
            batch_count = 0    
            eval_count = 0
            for batch in decoder_dataloader.get_all_batch(**dataloader_kwargs):
                compute_eval = (batch_count % 100 == 0)
                decoder_kwargs = {
                    'cond': batch['observations'],
                    'x': batch['actions'],
                    'compute_eval_loss': compute_eval,
                }
                if appender_mode == True:
                    decoder_id = batch.get('decoder_id', None)
                    decoder_kwargs['decoder_id'] = decoder_id

                metrics = self.decoder.train_model(
                    **decoder_kwargs
                )

                loss = metrics[1]['train/loss']
                epoch_loss += loss
                if compute_eval and 'train/eval_mse' in metrics[1]:
                    epoch_recon += metrics[1]['train/eval_mse']
                    eval_count += 1
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            
            # Log training metrics
            phase_name = self.scenario_config.datastream[phase].phase_name
            metrics = {"decoder_loss": avg_epoch_loss}
            if eval_count > 0:
                avg_epoch_recon = epoch_recon / eval_count
                metrics["decoder_recon"] = avg_epoch_recon
                pbar.set_postfix({
                    "decoder_avg_loss": f"{avg_epoch_loss:.4f}",
                    "decoder_avg_recon": f"{avg_epoch_recon:.4f}"
                })
            else:
                pbar.set_postfix({
                    "decoder_avg_loss": f"{avg_epoch_loss:.4f}"
                })

    def _train_policy(self, phase):
        """
        Step 3: Policy training.
        Rollback the dataloader (if updated by the interface) and train the policy model.
        """
        if self.interface is not None:
            self.current_dataloader = self.interface.rollback_dataloader(self.current_dataloader)
        policy_batch_size = (
            self.experiment_config.policy_batch_size
            if hasattr(self.experiment_config, 'policy_batch_size')
            else self.experiment_config.batch_size
        )
        policy_epochs = (
            self.experiment_config.policy_epochs
            if hasattr(self.experiment_config, 'policy_epochs')
            else self.experiment_config.phase_epochs
        )

        self.logger.info(f"Sampling the logits for policy from the decoder and interface")
        # reward_logits_list = []
        # for batch in tqdm(self.current_dataloader.get_all_batch(batch_size=policy_batch_size, shuffle=False),
        #                   desc="Preparing policy data", ncols=100):
        #     input_batch = {
        #         'inputs': batch['observations'],
        #         'labels': batch['actions'],
        #         'skills': batch['skills']
        #     }
        #     reward_logits, _ = self.agent.sample_logit(input_batch)
        #     reward_logits_list.append(reward_logits)
        # flatten_logits = np.concatenate(reward_logits_list, axis=0)

        # print(f"[SkillTrainer] Flattened reward logits variants: {np.unique(flatten_logits)}")
        # desired_length = self.current_dataloader.stacked_data['observations'].shape[0]
        # flatten_logits = flatten_logits[:desired_length]
        # self.current_dataloader.stacked_data['reward_logits'] = flatten_logits
        
        self.current_dataloader = self.agent.create_skill_labels(self.current_dataloader)

        if self.experiment_config.algo_type in ['lazysi', 'silc']:
            self.logger.info(f"Creating policy prototypes for phase {phase}")
            subtask_prototypes = self.interface.create_subtask_prototype(self.current_dataloader)
            self.policy.set_subtask_prototype(subtask_prototypes)
            self.agent.check_skill_labels(self.current_dataloader)

        if self.experiment_config.algo_type == 'iscil' :
            display_unique_reward_logits(self.current_dataloader.stacked_data)

        self.logger.info(f"Policy training data ready for phase {phase}\n")

        # Optionally wrap with GPU dataloader for acceleration
        policy_dataloader = self.current_dataloader
        use_gpu_dataloader = getattr(self.experiment_config, 'use_gpu_dataloader', False)
        if use_gpu_dataloader:
            self.logger.info(f"Wrapping policy dataloader with GPUDataLoader")
            policy_dataloader = GPUDataLoader(
                base_dataloader=policy_dataloader,
                mode=getattr(self.experiment_config, 'gpu_dataloader_mode', 'auto'),
                gpu_memory_threshold=getattr(self.experiment_config, 'gpu_memory_threshold', 0.3),
                prefetch_buffer_size=getattr(self.experiment_config, 'prefetch_buffer_size', 2)
            )

        self.logger.info(f"Starting policy training for phase {phase}: epochs={policy_epochs}, batch_size={policy_batch_size}")
        pbar_policy = tqdm(range(policy_epochs), desc=f"Phase {phase} Policy Training")
        for epoch in pbar_policy:
            epoch_loss = 0.0
            batch_count = 0
            for batch in policy_dataloader.get_all_batch(batch_size=policy_batch_size):
                input_batch = {
                    'inputs': batch['observations'],
                    'labels': batch.get('reward_logits', batch.get('skill_id'))
                }
                loss = self.policy.train_model(input_batch)
                epoch_loss += loss
                batch_count += 1
            avg_policy_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            pbar_policy.set_postfix({"policy_avg_loss": f"{avg_policy_loss:.4f}"})

    def _load_decoder(self, decoder_id):
        """
        Load the decoder model saved for the given phase (identified by decoder_id).
        """
        skill_dir = os.path.join(self.experiment_config.exp_save_path, "skills")
        file_path = os.path.join(skill_dir, f"decoder_{decoder_id}.pkl")
        if os.path.exists(file_path):
            self.logger.info(f"Loading decoder model from {file_path}")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            self.logger.warning(f"Decoder model file {file_path} does not exist. Using current decoder.")
            return self.decoder

    def _load_interface(self, interface_id):
        """
        Load the interface model saved for the given phase (identified by interface_id).
        """
        skill_dir = os.path.join(self.experiment_config.exp_save_path, "skills")
        file_path = os.path.join(skill_dir, f"interface_{interface_id}.pkl")
        if os.path.exists(file_path):
            self.logger.info(f"Loading interface model from {file_path}")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            self.logger.warning(f"Interface model file {file_path} does not exist. Using current interface.")
            return self.interface

    def _load_policy(self, policy_id):
        """
        Load the policy model saved for the given phase (identified by policy_id).
        """
        policy_parent_dir = os.path.join(self.experiment_config.exp_save_path, "policy")
        file_path = os.path.join(policy_parent_dir, f"{policy_id}.pkl")
        if os.path.exists(file_path):
            self.logger.info(f"Loading policy model from {file_path}")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            self.logger.warning(f"Policy model file {file_path} does not exist. Using current policy.")
            return self.policy

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
            else :
                self.eval_policy.train_state = self.eval_policy.train_state.replace(
                    params=loaded_policy.train_state.params
                )
                if self.experiment_config.algo_type in ['lazysi', 'silc']:
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

    def eval_phase_model(self, phase):
        """
        Evaluate the agent using remote environment evaluation.
        For each evaluation task defined in the current phase,
        build the agent using trace information and evaluate it.
        If no eval_tasks are provided, the evaluation is skipped.
        The evaluation results for all tasks are stored in self.eval_results.
        """
        self.logger.info(f"Starting remote evaluation for phase {phase}...")
        eval_tracer = EvaluationTracer()
        eval_traced_list = eval_tracer.get_eval_tasks_by_reference(
            stream_config=self.scenario_config,
            phase=phase,
        )
        self.logger.info(f"Evaluation tasks for phase {phase}: {eval_traced_list}")

        if len(eval_traced_list) == 0:
            self.logger.info(f"No evaluation tasks provided for phase {phase}. Skipping remote evaluation.")
            self.eval_results[phase] = {
                "train_targets": self.datastream[phase].train_targets,
                "eval_tasks": None,
                "overall_rewards": None,
                "overall_avg_reward": None,
                "detailed": None,
            }
            eval_save_path = os.path.join(self.experiment_config.exp_save_path, "eval_results.json")
            with open(eval_save_path, "w", encoding="utf-8") as f:
                json.dump(self.eval_results, f, indent=4)
            return

        num_eval_episodes = getattr(self.experiment_config, 'eval_num_episodes', 3)
        max_eval_steps = getattr(self.experiment_config, 'eval_max_steps', 280)
        evaluator_cls = getattr(self.experiment_config, 'evaluator_cls', None)
        overall_rewards = []
        detailed_results = {}

        # Evaluate each task individually.
        for eval_trace in eval_traced_list:
            agent_config = eval_trace.get('agent_config', None)
            eval_tasks = eval_trace.get('eval_tasks', None)
            self.agent = self._build_agent(agent_config)
            self.logger.info(f"Building agent for task '{eval_tasks}' with config: {agent_config}")
            # Create a remote evaluator instance for this task.
            remote_kwargs = getattr(self.experiment_config, 'remote_eval_kwargs', {}) or {}
            # Add action_chunk_size to remote_kwargs
            remote_kwargs = dict(remote_kwargs)  # Make a copy
            remote_kwargs['action_chunk_size'] = getattr(self.experiment_config, 'action_chunk', 1)
            remote_evaluator = evaluator_cls(
                host=getattr(self.experiment_config, 'remote_eval_host', '127.0.0.1'),
                port=getattr(self.experiment_config, 'remote_eval_port', 9999),
                obs_helper=getattr(self.experiment_config, 'remote_obs_helper', None),
                eval_fn=self.agent.eval,
                **remote_kwargs,
            )

            self.logger.info(f"Remote evaluator initialized for task '{eval_tasks}'")
            # Set the remote evaluator to use only this eval task.
            metadata_payload = build_phase_metadata(
                agent_config=agent_config,
                phase_name=self.datastream[phase].phase_name if hasattr(self, "datastream") else None,
                environment=getattr(self.scenario_config, "environment", None) if hasattr(self, "scenario_config") else None,
            )
            extra_metadata = getattr(self.experiment_config, 'remote_eval_metadata', None) or {}
            if extra_metadata:
                metadata_payload = metadata_payload or {}
                for key, value in extra_metadata.items():
                    if value is None:
                        continue
                    metadata_payload[str(key)] = str(value)
            remote_evaluator.set_task(
                eval_tasks,
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
            "phase_name" : self.datastream[phase].phase_name,
            "train_targets": self.datastream[phase].train_targets,
            "eval_tasks": eval_tasks,
            "overall_rewards": overall_rewards,
            "overall_avg_reward": overall_avg_reward,
            "detailed": detailed_results,
        }
        self.eval_results[phase] = eval_results

        # Write the accumulated evaluation results to a JSON file.
        eval_save_path = os.path.join(self.experiment_config.exp_save_path, "eval_results.json")
        with open(eval_save_path, "w", encoding="utf-8") as f:
            json.dump(self.eval_results, f, indent=4)
        
        self.logger.info(f"Evaluation results saved to {eval_save_path}")

    def save_phase_model(self, phase):
        """
        Save the decoder, interface, and policy models following the directory structure:
        
        {skill_scenario_name}/{skill_model_type}/{exp_id_or_date}/
        ├─ skills/
        │    ├─ decoder_{version}.pkl
        │    ├─ interface_{version}.pkl
        ├─ policy/
        │    ├─ policy_{version}/{Skill_pretraining_id}.pkl

        Here, the phase's skill scenario name (version) is used for naming.
        """
        # Base experiment directory (e.g., {skill_scenario_name}/{skill_model_type}/{exp_id_or_date}/)
        base_dir = self.experiment_config.exp_save_path
        skill_scenario_name = self.datastream[phase].phase_name
        version = f"{skill_scenario_name}"  # Use the phase's skill scenario name as version

        # 1) Save Skill models (decoder and interface)
        skill_dir = os.path.join(base_dir, "skills")
        os.makedirs(skill_dir, exist_ok=True)
        
        if 'decoder' in self.datastream[phase].train_targets:
            decoder_filename = f"decoder_{version}.pkl"
            save_path_decoder = os.path.join(skill_dir, decoder_filename)
            self.logger.info(f"Saving decoder for phase {phase} to {save_path_decoder}")
            with open(save_path_decoder, 'wb') as f:
                cloudpickle.dump(self.decoder, f)
        
        if 'interface' in self.datastream[phase].train_targets:
            interface_filename = f"interface_{version}.pkl"
            save_path_interface = os.path.join(skill_dir, interface_filename)
            if self.interface is not None:
                self.logger.info(f"Saving interface for phase {phase} to {save_path_interface}")
                with open(save_path_interface, 'wb') as f:
                    cloudpickle.dump(self.interface, f)
            else:
                self.logger.warning(f"No interface available to save for phase {phase}.")
        
        # 2) Save Policy model directly inside the "policy" folder (without creating an extra subfolder)
        if 'policy' in self.datastream[phase].train_targets:
            policy_parent_dir = os.path.join(base_dir, "policy")
            os.makedirs(policy_parent_dir, exist_ok=True)
            if '/' in version:
                version_dir = version.split('/')[0]  # "policy_{version}/decoder_id" format
                os.makedirs(os.path.join(policy_parent_dir, version_dir), exist_ok=True)
            # Save the policy model
            policy_filename = f"{version}.pkl"
            save_path_policy = os.path.join(policy_parent_dir, policy_filename)
            if self.policy is not None:
                self.logger.info(f"Saving policy for phase {phase} to {save_path_policy}")
                with open(save_path_policy, 'wb') as f:
                    cloudpickle.dump(self.policy, f)
            else:
                self.logger.warning(f"No policy available to save for phase {phase}.")

    def update_buffer(self, phase):
        """
        Update the memory buffer with data from the current phase.
        """
        # if train target update the decoder and interface then, update the buffer
        if 'decoder' in self.datastream[phase].train_targets or 'interface' in self.datastream[phase].train_targets:
            keep_ratio = self.experiment_config.buffer_keep_ratio
            if not self.current_dataloader:
                self.logger.warning(f"Phase {phase} has no dataloader; skipping buffer update.")
                return
            self.logger.info(f"Updating memory buffer after phase {phase} with keep_ratio={keep_ratio}")
            self.memory_buffer.add_new_dataset(
                self.current_dataloader, 
                keep_ratio=keep_ratio,
                sample_function=getattr(self.interface, "sample_rehearsal", None)
            )
            self.logger.info(f"Memory buffer updated after phase {phase}")
        else:
            self.logger.info(f"Phase {phase} has no decoder or interface training; skipping buffer update.")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from SILGym.config.experiment_config import DEFAULT_DECODER_CONFIG
    from SILGym.config.skill_stream_config import SkillStreamConfig
    from SILGym.config.kitchen_scenario import *
    from SILGym.config.baseline_config import PTGMConfig
    import warnings
    warnings.filterwarnings("ignore")

    # Construct the experiment config
    exp_config = PTGMConfig(
        # scenario_config=SkillStreamConfig(datastream=KITCHEN_SCENARIO_DEFAULT),
        # scenario_config=SkillStreamConfig(datastream=KITCHEN_SCENARIO_DEFAULT_FWC),
        # scenario_config=SkillStreamConfig(datastream=KITCHEN_SCENARIO_DEFAULT_SYNC),
        # scenario_config=SkillStreamConfig(datastream=KITCHEN_DEBUG_SCENARIO),
        # scenario_config=SkillStreamConfig(datastream=KITCHEN_DEBUG_MIN_SCENARIO),
        decoder_config=DEFAULT_DECODER_CONFIG.copy()
    )

    # Create the trainer (with do_eval=True for remote evaluation after each phase)
    trainer = SkillTrainer(experiment_config=exp_config, do_eval=True)
    try:
        trainer.continual_train()
        # Using logger after close_logger() call - use print here
        print("[Test] Continual training finished. Check any saved models or logs.")
    finally:
        # When training is complete, restore stdout and close the log file
        trainer.close_logger()
