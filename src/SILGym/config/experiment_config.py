from SILGym.config.skill_stream_config import SkillStreamConfig
from SILGym.config.variant_registry import resolve_kitchen_vis_variant, LIBERO_ENV_MODEL_MAP
from SILGym.config.data_paths import DEFAULT_LIBERO_MODEL
from SILGym.config.kitchen_scenario import (
    KITCHEN_VIS_MODEL_TO_DINOV3,
    KITCHEN_VIS_DEFAULT_IMAGE_SIZE,
    DEFAULT_KITCHEN_PROPRIO_KEYS,
    DEFAULT_KITCHENSTUDIO_EMBED_CAMERA_KEYS,
    KITCHEN_PROPRIO_KEY_DIMS,
)
from SILGym.config.libero_scenario import (
    get_libero_observation_dim,
    LIBERO_MODEL_TO_DINOV3,
)
from SILGym.dataset.dataloader import BaseDataloader, PoolDataLoader, LeRobotDataLoader
import os
import shutil
import typing
import copy
import optax
from datetime import datetime, timedelta
import pytz
import pprint
from SILGym.utils.logger import get_logger
# Define the log base path
LOGBASEPATH = os.environ.get('LOGBASEPATH', 'logs')

# Import model classes
from SILGym.models.skill_decoder.diffusion_base import CondDDPMDecoder
from SILGym.models.skill_decoder.fql import FQLDecoder
from SILGym.models.basic.base import DenoisingMLP
from SILGym.models.skill_interface.ptgm import ContinualPTGMInterface, PTGMInterfaceConfig
from SILGym.models.skill_decoder.appender import ModelAppender, AppendConfig, ModelAppenderV2
from SILGym.models.skill_decoder.appenderv3 import ModelAppenderV3
from SILGym.dataset.dataloader import DataloaderMixer
from SILGym.models.agent.base import *

# Default optimizer configuration
DEFAULT_OPTIM_CONFIG = {
    'optimizer_cls': optax.adam,
    'optimizer_kwargs': {
        'learning_rate': 2e-5,
        'b1': 0.9,
    },
}

# Default model configuration
DEFAULT_DECODER_CONFIG = {
    "model_cls": CondDDPMDecoder,  # Replace with a real Flax model class
    "model_kwargs": {
        "input_config": {
            "x": (1, 1, 9), # action space
            "cond": (1, 1, 60), # state space
        },
        "optimizer_config": DEFAULT_OPTIM_CONFIG,
        "model_config": {
            "model_cls": DenoisingMLP,
            "model_kwargs": {
                "dim": 512,
                "n_blocks": 4,
                "context_emb_dim": 512,
                "dropout": 0.0,
            }
        },
        "clip_denoised": True,
        "decoding_iter_ratio": 1.0,  # 1.0 = full sampling, 0.1 = 10x faster (DDIM-style)
    }
}

# FQL decoder configuration (Flow Q-Learning)
DEFAULT_FQL_DECODER_CONFIG = {
    "model_cls": FQLDecoder,
    "model_kwargs": {
        "input_config": {
            "x": (1, 1, 9),        # action space
            "cond": (1, 1, 60),    # state space
        },
        "optimizer_config": {
            'optimizer_cls': optax.adam,
            'optimizer_kwargs': {
                'learning_rate': 5e-6, # higher overfitting to dataset
                'b1': 0.9,
            },
        },
        "model_config": {
            "model_cls": DenoisingMLP,
            "model_kwargs": {
                "dim": 512,
                "n_blocks": 4,
                "context_emb_dim": 512,
                "dropout": 0.0,
            }
        },
        "clip_actions": True,
        "flow_steps": 10,              # Number of Euler integration steps
        "use_onestep_flow": False,     # Enable one-step distillation
        "eval_use_onestep": False,     # Use one-step model for evaluation by default
        "use_q_loss": False,           # Enable Q-learning (disabled by default)
        "alpha": 10.0,                 # Distillation coefficient
    }
}

# Decoder configuration mapping
DECODER_CONFIG_MAP = {
    "ddpm": DEFAULT_DECODER_CONFIG,
    "diffusion": DEFAULT_DECODER_CONFIG,
    "fql": DEFAULT_FQL_DECODER_CONFIG,
    "flow": DEFAULT_FQL_DECODER_CONFIG,
}

# Appender class mapping
APPENDER_CLASS_MAP = {
    "v1": ModelAppender,
    "v2": ModelAppenderV2,
    "v3": ModelAppenderV3,
}

from SILGym.models.task_policy.mlp_base import MLPPolicy, HighLevelPolicy
from SILGym.models.task_policy.flow_policy import FlowMatchingPolicy
DEFAULT_POLICY_CONFIG = {
    'model_cls': HighLevelPolicy,
    'model_kwargs': {
        'model_config': {
            'hidden_size': 512,
            # 'hidden_size': 2048,
            'out_shape': 400, # shape of the INTERFACE action space 20 per skills
            'num_hidden_layers': 4, 
            # 'dropout': 0.1, 
            'dropout': 0.0, 
        },
        'input_config': { # for MLP Policy
            'x': (1, 1, 60), # State (condition) input shape
        },
        'optimizer_config': {
            'optimizer_cls': optax.adam,
            'optimizer_kwargs': {
                'learning_rate': 1e-4,
                'b1': 0.9,
            },
        },
    }
}

DEFAULT_FLOW_POLICY_CONFIG = {
    'model_cls': FlowMatchingPolicy,
    'model_kwargs': {
        'out_shape': 400,
        'input_config': {
            'x': (1, 1, 60),
        },
        'optimizer_config': {
            'optimizer_cls': optax.adam,
            'optimizer_kwargs': {
                'learning_rate': 1e-4,
                'b1': 0.9,
            },
        },
        'flow_model_config': {
            'model_cls': DenoisingMLP,
            'model_kwargs': {
                'dim': 512,
                'n_blocks': 4,
                'context_emb_dim': 512,
                'dropout': 0.0,
            },
        },
        'flow_steps': 10,
        'use_onestep_flow': False,
        'eval_use_onestep': False,
        'clip_logits': False,
    }
}

POLICY_CONFIG_MAP = {
    'mlp': DEFAULT_POLICY_CONFIG,
    'flow': DEFAULT_FLOW_POLICY_CONFIG,
}

# -------------------------------------
# Default interface configuration
# -------------------------------------
DEFAULT_PTGM_INTERFACE_CONFIG = {
    "interface_cls": ContinualPTGMInterface,
    "interface_kwargs": {
        "ptgm_config": PTGMInterfaceConfig(
            cluster_num=20, 
            goal_offset=20, 
            tsne_dim=2, 
            tsne_perplexity=50 # state space assumption.
            # tsne_perplexity=20
        ),
        "update_mode" : "expand", # 'overwrite' or 'expand'
    }
}

from SILGym.models.skill_interface.buds import ContinualBUDSInterface, BUDSInterfaceConfig
DEFAULT_BUDS_INTERFACE_CONFIG = {   
    "interface_cls": ContinualBUDSInterface,
    "interface_kwargs": {
        "config": BUDSInterfaceConfig(
            window_size=5,
            min_length=30,
            target_num_segments=10,
            max_k=20,
            goal_offset=20,
            verbose=True,
        ),
        "update_mode" : "expand", #  'expand'
    }
}

from SILGym.models.skill_interface.semantic_based import SemanticInterface, ImanipInterfaceConfig
DEFAULT_IMANIP_INTERFACE_CONFIG = {
    "interface_cls": SemanticInterface, 
    "interface_kwargs": {
        'config' : ImanipInterfaceConfig(),
        'semantic_emb_path' : "exp/instruction_embedding/kitchen_clip/512.pkl",
    }
}

from SILGym.models.skill_interface.semantic_based import PrototypeInterface, IsCiLInterfaceConfig
DEFAULT_ISCIL_INTERFACE_CONFIG = {
    "interface_cls": PrototypeInterface,
    "interface_kwargs": {
        'config' : IsCiLInterfaceConfig(
            bases_num=100, # skill bases updated to 100
        ),
        'semantic_emb_path' : "exp/instruction_embedding/kitchen_clip/512.pkl",
    }
}

from SILGym.models.skill_interface.lazySI import LazySIInterface, LazySIInterfaceConfig
DEFAULT_LAZYSI_INTERFACE_CONFIG ={
    "interface_cls": LazySIInterface,
    "interface_kwargs": {
        "config": LazySIInterfaceConfig(
            decoder_algo= "ptgm",
            decoder_algo_config=PTGMInterfaceConfig(
                cluster_num= 20,
                goal_offset= 20,
                tsne_perplexity= 50,
            ),
            skill_prototype_bases= 4,
            # policy side
            policy_algo= "ptgm",
            policy_algo_config=PTGMInterfaceConfig(
                cluster_num= 20,
                goal_offset= 20,
                tsne_perplexity= 50,
            ),
            subtask_prototype_bases= 4,
            # confidence interval
            confidence_interval = 99.0,
            threshold_type = 'chi2',
            distance_type = 'maha',
        ),
    }
}

from SILGym.models.skill_interface.silc import SILCInterface, SILCInterfaceConfig
DEFAULT_SILC_INTERFACE_CONFIG ={
    "interface_cls": SILCInterface,
    "interface_kwargs": {
        "config": SILCInterfaceConfig(
            decoder_algo= "ptgm",
            decoder_algo_config=PTGMInterfaceConfig(
                cluster_num= 20,
                goal_offset= 20,
                tsne_perplexity= 50,
            ),
            skill_prototype_bases= 4,
            # policy side
            policy_algo= "ptgm",
            policy_algo_config=PTGMInterfaceConfig(
                cluster_num= 20,
                goal_offset= 20,
                tsne_perplexity= 50,
            ),
            subtask_prototype_bases= 4,
            # confidence interval
            confidence_interval = 0.99,
            threshold_type = 'chi2',
            distance_type = 'maha',
        ),
    }
}

# -------------------------------------

def get_date_string():
    korea_tz = pytz.timezone('Asia/Seoul')
    now_korea = datetime.now(korea_tz)
    return now_korea.strftime('%m%d')

# Supported algorithm types
DEFAULT_ALGO_TYPES = ['ptgm', 'SILGym']

from SILGym.dataset.dataloader import libero_obs_obs_hook, history_state_obs_hook, history_state_three_obs_hook, dropthe_traj_hook
from SILGym.environment.remote import (
    KitchenRemoteEvaluator,
    KitchenEmbedRemoteEvaluator,
    MMWorldRemoteEvaluator,
    LiberoRemoteEvaluator,
    LiberoEmbedRemoteEvaluator,
    HistoryEvalHelper,
    HistoryEvalHelperTriple,
    NoiseEvaluationHelper,
)
class SkillExperimentConfig:
    '''
    ABC for Experiment Configurations.
    This class is responsible for setting up the experiment configurations
    '''
    def __init__(
            self,
            # important parameters
            scenario_config: SkillStreamConfig,
            decoder_config: dict = None,  # External model config can be provided
            interface_config: dict = None,
            policy_config: dict = None,
            # type configurations
            exp_id : str = 'DEFAULT',
            lifelong_algo: str = '',
            seed: int = 0,
            decoder_type: str = 'ddpm',
            # action chunking
            action_chunk: int = 1,
            action_chunk_padding: str = 'repeat_last',
        ) -> None:
        # Basic experiment settings
        self.scenario_config = scenario_config
        self.scenario_metadata = getattr(self.scenario_config, "metadata", {}) or {}
        self.scenario_id = self.scenario_config.scenario_id
        self.algo_type = 'ptgm'  # Default: ptgm, spirl, SILGym # Override only in subclasses
        self.lifelong_algo = lifelong_algo # ['ft', 'er', 'append', 'expand'] # expand is depend on subclass method
        self.seed = seed
        self.decoder_type = decoder_type.lower()  # Store decoder type for path naming
        self.remote_eval_kwargs: dict[str, typing.Any] = {}
        self.remote_eval_metadata: dict[str, typing.Any] = {}

        # Action chunking configuration
        self.action_chunk = max(1, int(action_chunk))
        self.action_chunk_padding = str(action_chunk_padding)
        if self.action_chunk_padding not in ['repeat_last', 'zero']:
            get_logger(__name__).warning(
                f"Invalid action_chunk_padding '{self.action_chunk_padding}'. Using 'repeat_last'."
            )
            self.action_chunk_padding = 'repeat_last'

        self.exp_id = get_date_string() + exp_id + f"seed{self.seed}" # format of '{date}{env}{scenarioType}_{auxiliary}{seed}'

        # ----------------------------
        # Scenario Environment related settings
        # ----------------------------
        requested_env = (scenario_config.environment or '').lower()
        env_alias = (self.scenario_metadata.get("env_alias") or requested_env or '').lower()
        base_env_metadata = self.scenario_metadata.get("base_environment")
        if base_env_metadata:
            base_env = base_env_metadata.lower()
        elif env_alias in LIBERO_ENV_MODEL_MAP or env_alias.startswith('libero'):
            base_env = 'libero'
        else:
            base_env = env_alias

        if not base_env:
            base_env = requested_env

        self.base_environment = base_env
        self.environment = base_env
        self.environment_alias = env_alias or base_env
        env_lookup_key = self.environment_alias

        self.libero_model_name = self.scenario_metadata.get("model_name")
        if self.base_environment == 'libero' and not self.libero_model_name:
            self.libero_model_name = LIBERO_ENV_MODEL_MAP.get(env_lookup_key, DEFAULT_LIBERO_MODEL)

        self.libero_obs_dim = self.scenario_metadata.get("libero_obs_dim")
        if self.base_environment == 'libero' and self.libero_obs_dim is None and self.libero_model_name:
            try:
                self.libero_obs_dim = get_libero_observation_dim(self.libero_model_name)
            except ValueError:
                self.libero_obs_dim = get_libero_observation_dim(DEFAULT_LIBERO_MODEL)

        self.scenario_type = scenario_config.scenario_type
        self.sync_type = scenario_config.sync_type

        self.evaluator_cls = KitchenRemoteEvaluator
        self.remote_eval_host = '127.0.0.1'
        self.remote_eval_port = 9999
        self.remote_obs_helper = None

        self.eval_num_episodes = 3
        self.eval_max_steps = 280
        
        # ----------------------------
        # Evaluation noise settings (Gaussian)
        # ----------------------------
        self.eval_noise_enabled = False
        self.eval_noise_scale = 0.01
        self.eval_noise_clip = None
        self.eval_noise_seed = None
        
        # ----------------------------
        # Interface related settings
        # ----------------------------
        self.interface_config = interface_config

        # ----------------------------
        # Decoder Model related settings
        # ----------------------------
        self.pretrained_model_path = 'data/pretrained_model'
        self.init_model_path = None
        self.decoder_config = decoder_config if decoder_config is not None else DEFAULT_DECODER_CONFIG

        # ----------------------------
        # Policy Model related settings
        # ----------------------------
        # Determine policy model type (default to 'mlp')
        self.policy_type = str(self.scenario_metadata.get("policy_model_type", "mlp")).lower()
        self.pretrained_policy_path = 'data/pretrained_policy'
        self.init_policy_path = None
        if policy_config is not None:
            self.policy_config = policy_config
        else:
            base_policy_config = POLICY_CONFIG_MAP.get(self.policy_type, DEFAULT_POLICY_CONFIG)
            if self.policy_type not in POLICY_CONFIG_MAP:
                get_logger(__name__).warning(
                    f"Unknown policy_model_type '{self.policy_type}'. Falling back to MLP policy."
                )
                self.policy_type = 'mlp'
            self.policy_config = copy.deepcopy(base_policy_config)

        # ----------------------------
        # Exp configurations
        # ----------------------------
        self.algo_config = None  # To be defined if needed.
        self.dataloader_cls = PoolDataLoader
        self.dataloader_kwargs = {} 
        self.dataloader_kwargs_policy = {} 
        self.batch_size = 1024        
        self.phase_epochs = 5_000       

        self.buffer_keep_ratio = 0.0 # ER parameter
        self.dataloader_mixer_cls = DataloaderMixer

        # ----------------------------
        # * NOTE TODO GPU Dataloader settings
        # ----------------------------
        self.use_gpu_dataloader = False  # Enable GPU-accelerated dataloader
        self.gpu_dataloader_mode = 'auto'  # 'auto', 'gpu', 'prefetch', or 'cpu'
        self.gpu_memory_threshold = 0.3  # Fraction of GPU memory for full loading
        self.prefetch_buffer_size = 2  # Number of batches to prefetch

        # ----------------------------
        # * Append-based baseline settings
        # ----------------------------
        self.is_appendable = False # Append-based baseline
        self.appender_version = "v3"  # Options: "v1", "v2", "v3"
        self.appender_cls = APPENDER_CLASS_MAP[self.appender_version]
        self.appender_config = AppendConfig(
            lora_dim=4,
            pool_length=10,
        )

        # ----------------------------
        # Ablation settings
        # ----------------------------
        self.phase_reset = True # DEFAULT for FALSE
        self.reset_decoder_each_phase = False  # Set to True for ftscratch algorithm

        self.agent_dict = {
            'ptgm': PTGMAgent,
            'iscil': IsCiLAgent, 
            'imanip': IsCiLAgent,
            'buds': BUDSAgent,
            'lazysi': LazySIAgent,
            'silc': SILCAgent,
        }

        # ----------------------------
        # process the scenario 
        # ----------------------------
        self.update_config_by_env()

    @property
    def agent_cls(self) :
        """
        Returns the agent class based on the algorithm type.
        This is used to create the appropriate agent for the experiment.
        """
        if self.algo_type in self.agent_dict:
            return self.agent_dict[self.algo_type]
        else:
            raise ValueError(f"Unsupported algo_type: {self.algo_type}. Must be one of {list(self.agent_dict.keys())}.")

    @property
    def ll_postfix(self):
        """
        Returns the postfix for the lifelong learning algorithm.
        This is used to differentiate between different lifelong learning strategies.
        Includes action_chunk and decoder_type information before the lifelong_algo.
        """
        postfix_parts = []

        # Add action chunk info if enabled
        if self.action_chunk > 1:
            postfix_parts.append(f"ac{self.action_chunk}")

        # Add flow decoder info if using flow/fql
        if self.decoder_type in ['flow', 'fql']:
            postfix_parts.append("flow")

        # Combine the parts with underscores
        prefix = "_" + "_".join(postfix_parts) if postfix_parts else ""

        # Add lifelong algo postfix
        ll_algo_suffix = f"_{self.lifelong_algo}" if self.lifelong_algo else ""

        return prefix + ll_algo_suffix

    @property
    def exp_save_path(self):
        """
        Returns the experiment save path.
        This is used to store the results and models of the experiment.
        """
        log_env = getattr(self, "environment_alias", self.environment) or self.environment
        return f'{LOGBASEPATH}/{log_env}/{self.scenario_type}/{self.sync_type}/{self.algo_type}{self.ll_postfix}/{self.exp_id}'
    # ------------------------------
    # Update Model Configurations
    # ------------------------------

    def update_config_by_env(self):
        logger = get_logger(__name__)
        logger.info(
            f"Updating model configurations for environment: {self.environment} "
            f"(alias={self.environment_alias}, base={self.base_environment})"
        )
        self.remote_eval_kwargs = {}
        # Default to lightweight hidden widths; specific envs opt into vision scaling.
        self._configure_model_width(use_vision=False)

        if self.base_environment == 'kitchen':
            dataset_paths = [
                path
                for phase in (self.scenario_config.datastream or [])
                for path in getattr(phase, "dataset_paths", []) or []
            ]
            alias = (self.environment_alias or '').lower()
            normalized_alias = alias.replace('_', '-')
            uses_embeddings = (
                normalized_alias.startswith('kitchen-vis')
                or normalized_alias.startswith('kitchenvis')
                or 'studio' in normalized_alias
                or 'kitchenstudio' in normalized_alias
                or any('kitchen_lerobot_embed' in str(path) for path in dataset_paths)
                or any('kitchenstudio_embed' in str(path) for path in dataset_paths)
            )
            is_studio = (
                'studio' in normalized_alias
                or 'kitchenstudio' in normalized_alias
                or any('kitchenstudio_embed' in str(path) for path in dataset_paths)
            )
            uses_history = (
                (alias.startswith('kitchen_vis') or normalized_alias.startswith('kitchen-vis') or normalized_alias.startswith('kitchenvis'))
                and not is_studio  # kitchenstudio doesn't use history by default
            )

            if uses_embeddings:
                self._configure_model_width(use_vision=True)
                self.dataloader_cls = LeRobotDataLoader
                self.dataloader_kwargs.setdefault('pre_process_hooks_kwargs', [])
                self.dataloader_kwargs.setdefault('post_process_hooks_kwargs', [])
                self.dataloader_kwargs_policy.setdefault('pre_process_hooks_kwargs', [])
                self.dataloader_kwargs_policy.setdefault('post_process_hooks_kwargs', [])

                if uses_history:
                    history_hook = (history_state_obs_hook, {'N': 10})
                    if history_hook not in self.dataloader_kwargs['pre_process_hooks_kwargs']:
                        self.dataloader_kwargs['pre_process_hooks_kwargs'].append(history_hook)
                    if history_hook not in self.dataloader_kwargs_policy['pre_process_hooks_kwargs']:
                        self.dataloader_kwargs_policy['pre_process_hooks_kwargs'].append(history_hook)
                    self.remote_obs_helper = HistoryEvalHelper(10)
                    logger.info("Using history hooks with N=10 for kitchen vision")
                else:
                    # Explicitly disable history hooks and helper (e.g., for kitchenstudio)
                    self.remote_obs_helper = None
                    if is_studio:
                        logger.info("Kitchenstudio detected: history hooks and helper disabled")

                # Optional: restrict observation modalities via metadata
                obs_modalities = self.scenario_metadata.get("kitchen_obs_modalities")
                if obs_modalities:
                    self.dataloader_kwargs['obs_modality_keys'] = tuple(obs_modalities)
                    self.dataloader_kwargs_policy['obs_modality_keys'] = tuple(obs_modalities)

                # Optional: replace proprio with oracle 'state' from HDF5
                use_oracle_proprio = bool(self.scenario_metadata.get("kitchen_use_oracle_proprio", False))
                oracle_key_name = self.scenario_metadata.get("kitchen_oracle_key_name", "states")
                if use_oracle_proprio:
                    self.dataloader_kwargs['replace_proprio_with_state'] = True
                    self.dataloader_kwargs_policy['replace_proprio_with_state'] = True
                    self.dataloader_kwargs['oracle_key_name'] = oracle_key_name
                    self.dataloader_kwargs_policy['oracle_key_name'] = oracle_key_name

                obs_dim = self._compute_kitchen_obs_dim(
                    dataset_paths=dataset_paths,
                    obs_modalities=obs_modalities,
                    use_oracle_proprio=use_oracle_proprio,
                    oracle_key_name=oracle_key_name,
                )
                self.scenario_metadata["kitchen_obs_dim"] = obs_dim
                
                obs_dim = int(obs_dim)
                history_multiplier = 2 if uses_history else 1
                self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, obs_dim * history_multiplier)
                self.decoder_config['model_kwargs']['input_config']['cond'] = (1, 1, obs_dim * history_multiplier)
                # Kitchen action dimension: 9 * action_chunk
                self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 9 * self.action_chunk)

                variant = self.scenario_metadata.get("kitchen_vis_variant") or resolve_kitchen_vis_variant(alias)
                dino_model = KITCHEN_VIS_MODEL_TO_DINOV3.get(str(variant), "ViT-B/16")
                image_size = self.scenario_metadata.get("kitchen_embed_image_size", KITCHEN_VIS_DEFAULT_IMAGE_SIZE)

                # Import evaluator class for studio mode
                if is_studio:
                    from SILGym.environment.remote import KitchenStudioEmbedRemoteEvaluator
                    self.evaluator_cls = KitchenStudioEmbedRemoteEvaluator
                else:
                    self.evaluator_cls = KitchenEmbedRemoteEvaluator

                # Eval-time proprio selection via metadata (and oracle override)
                eval_proprio_keys = self.scenario_metadata.get("eval_proprio_keys")
                eval_include_state = self.scenario_metadata.get("eval_include_state")
                eval_use_oracle_proprio = bool(self.scenario_metadata.get("eval_use_oracle_proprio", False)) or use_oracle_proprio
                if isinstance(eval_proprio_keys, (list, tuple)):
                    proprio_tuple = tuple(str(k) for k in eval_proprio_keys)
                elif eval_include_state is False:
                    proprio_tuple = tuple()
                else:
                    default_proprio = self.scenario_metadata.get("kitchen_proprio_keys")
                    if isinstance(default_proprio, (list, tuple)) and default_proprio:
                        proprio_tuple = tuple(str(k) for k in default_proprio)
                    else:
                        proprio_tuple = ("robot_states", "ee_states", "gripper_states")

                eval_oracle_key = self.scenario_metadata.get("kitchen_oracle_key_name", "states")
                if eval_use_oracle_proprio:
                    # When using oracle proprio at eval time, prefer a single 'oracle_state' channel
                    proprio_tuple = ("oracle_state",)

                # Get camera keys from metadata for studio mode
                camera_keys_eval = self.scenario_metadata.get("kitchen_camera_keys")
                if not camera_keys_eval:
                    camera_keys_eval = DEFAULT_KITCHENSTUDIO_EMBED_CAMERA_KEYS if is_studio else None

                self.remote_eval_kwargs = {
                    "model_name": dino_model,
                    "proprio_keys": proprio_tuple,
                    "embed_image_size": image_size,
                    "use_oracle_proprio": eval_use_oracle_proprio,
                    "oracle_key_name": eval_oracle_key,
                }
                # Add camera_keys for studio mode
                if is_studio and camera_keys_eval:
                    self.remote_eval_kwargs["camera_keys"] = camera_keys_eval

                self.remote_eval_metadata = {
                    "obs_mode": "vision",
                    "kitchen_vis_variant": variant,
                }
                if is_studio:
                    self.remote_eval_metadata["is_studio"] = True
                    # Set phase_epochs for kitchenstudio_vis
                    self.phase_epochs = 2500
                    logger.info(f"Kitchen Studio with vision embeddings: phase_epochs set to {self.phase_epochs}")
            else:
                self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, 60) # state space
                # Kitchen action dimension: 9 * action_chunk
                self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 9 * self.action_chunk) # action space
                self.decoder_config['model_kwargs']['input_config']['cond'] = (1, 1, 60) # state space
                self.remote_obs_helper = None
                # For regular kitchen state space, use higher perplexity for PTGM interface
                if self.interface_config and self.interface_config.get('interface_cls') == ContinualPTGMInterface:
                    self.interface_config['interface_kwargs']['ptgm_config'].tsne_perplexity = 50

            if not uses_embeddings:
                self.evaluator_cls = KitchenRemoteEvaluator
                self.remote_eval_kwargs = {}
                self.remote_eval_metadata = {}

            self.remote_eval_host = '127.0.0.1'
            self.remote_eval_port = 9999

            self.eval_num_episodes = 3
            self.eval_max_steps = 280
        
        elif self.base_environment == 'mmworld':
            # self.dataloader_kwargs['pre_process_hooks_kwargs'] =[
            #     (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
            # ]
            # self.remote_obs_helper = HistoryEvalHelper(10)

            self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, 140)
            # MMWorld action dimension: 4 * action_chunk
            self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 4 * self.action_chunk) # action space
            self.decoder_config['model_kwargs']['input_config']['cond'] = (1, 1, 140) # state space

            self.evaluator_cls = MMWorldRemoteEvaluator
            self.remote_eval_host = '127.0.0.1'
            self.remote_eval_port = 8888

            self.eval_num_episodes = 3
            self.eval_max_steps = 600

            if self.scenario_type == 'n1' : # about 50k 
                # self.dataloader_kwargs['pre_process_hooks_kwargs'].append(
                #     (dropthe_traj_hook, {'per_drop' : 2}),
                # )
                self.phase_epochs = 1_000
                self.policy_epochs = 5_000
                get_logger(__name__).info(f"MMWorld n1 scenario: {self.phase_epochs} epochs")

            if self.scenario_type in  ['explicit', 'n2exp'] : # about 250k
                # self.dataloader_kwargs['pre_process_hooks_kwargs'] = [
                #     (dropthe_traj_hook, {'per_drop' : 2}),
                # ]
                self.phase_epochs = 1_000
                get_logger(__name__).info(f"MMWorld explicit scenario: {self.phase_epochs} epochs")
                self.policy_epochs = 5_000
        elif self.base_environment == 'libero':
            # Libero always uses vision-scale model width (hidden_size=2048)
            self._configure_model_width(use_vision=True)
            get_logger(__name__).info("Libero environment: setting model width for vision (hidden_size=2048)")

            # Check if we should use LeRobotDataLoader (for embedded data)
            # This is determined by whether the dataset path contains 'libero_embed'
            if 'libero_embed' in str(self.scenario_config.datastream[0].dataset_paths[0] if self.scenario_config.datastream else ''):
                # Use LeRobotDataLoader for embedded HDF5 data
                self.dataloader_cls = LeRobotDataLoader
                # For embedded data, we don't need the libero_obs_hook
                # but may still want history augmentation
                self.dataloader_kwargs['pre_process_hooks_kwargs'] = [
                    (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
                ]
                self.dataloader_kwargs_policy['pre_process_hooks_kwargs'] = [
                    (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
                ]
                # Note: obs dimension will be determined by the actual embedded data size
                # For DINOv3 embeddings + proprioceptive state, this is typically ~1551
                obs_dim = self.libero_obs_dim or get_libero_observation_dim(self.libero_model_name or DEFAULT_LIBERO_MODEL)
                evaluator_cls = LiberoEmbedRemoteEvaluator
                embedder_variant = LIBERO_MODEL_TO_DINOV3.get(
                    (self.libero_model_name or DEFAULT_LIBERO_MODEL),
                    "ViT-B/16",
                )
                self.remote_eval_kwargs = {
                    "model_variant": self.libero_model_name or DEFAULT_LIBERO_MODEL,
                    "embedder_kwargs": {
                        "model_name": embedder_variant,
                    }
                }
            else:
                # Use PoolDataLoader for regular pickle data
                # Note: model width already configured at libero section start with use_vision=True
                self.dataloader_kwargs['pre_process_hooks_kwargs'] = [
                    (libero_obs_obs_hook, {'obs_dim': 130}),
                    (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
                    # (history_state_three_obs_hook, {'N': 15}),  # it triples the last dimension
                ]

                self.dataloader_kwargs_policy['pre_process_hooks_kwargs'] = [
                    (libero_obs_obs_hook, {'obs_dim': 130}),
                    (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
                    # (history_state_three_obs_hook, {'N': 15}),  # it triples the last dimension
                ]
                obs_dim = 130
                evaluator_cls = LiberoRemoteEvaluator
                self.remote_eval_kwargs = {}

            self.remote_obs_helper = HistoryEvalHelper(10)

            # Set input dimensions based on observation size
            self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, obs_dim*2)
            self.policy_config['model_kwargs']['optimizer_config']['optimizer_kwargs']['learning_rate'] = 5e-4

            # Libero action dimension: 7 * action_chunk
            self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 7 * self.action_chunk) # action space
            self.decoder_config['model_kwargs']['input_config']['cond'] = (1, 1, obs_dim*2) # state space
            self.decoder_config['model_kwargs']['optimizer_config']['optimizer_kwargs']['learning_rate'] = 5e-4
            
            self.evaluator_cls = evaluator_cls
            self.remote_eval_host = '127.0.0.1'
            self.remote_eval_port = 7777

            self.phase_epochs = 2_500
            # self.policy_epochs = 2_500

            self.eval_num_episodes = 10
            self.eval_max_steps = 600

        # Adjust phase_epochs for flow-based decoders
        # self._adjust_epochs_for_flow_decoder()

        # Apply action chunking hook if enabled
        self._apply_action_chunk_hook()

        # Apply noise configuration if enabled
        self._apply_noise_config()

    def _configure_model_width(self, *, use_vision: bool) -> None:
        """
        Align decoder and flow policy widths with the active observation modality.

        Vision-heavy setups benefit from wider networks, whereas state-only runs can
        default to more lightweight widths. This helper keeps the two configurations
        in sync.
        """
        target_dim = 2048 if use_vision else 512

        decoder_model_cfg = (
            self.decoder_config
            .get("model_kwargs", {})
            .get("model_config", {})
            .get("model_kwargs")
        )
        if isinstance(decoder_model_cfg, dict):
            # decoder_model_cfg["dim"] = target_dim
            decoder_model_cfg["context_emb_dim"] = target_dim

        # Set policy hidden_size (for MLP-based policies)
        policy_model_cfg = (
            self.policy_config
            .get("model_kwargs", {})
            .get("model_config", {})
        )
        if isinstance(policy_model_cfg, dict) and "hidden_size" in policy_model_cfg:
            policy_model_cfg["hidden_size"] = target_dim

        if self.policy_type == "flow":
            flow_model_cfg = (
                self.policy_config
                .get("model_kwargs", {})
                .get("flow_model_config", {})
                .get("model_kwargs")
            )
            if isinstance(flow_model_cfg, dict):
                # flow_model_cfg["dim"] = target_dim
                flow_model_cfg["context_emb_dim"] = target_dim

    def _adjust_epochs_for_flow_decoder(self):
        """
        Adjust phase_epochs for flow-based decoders.

        Flow matching models converge faster than diffusion models, so we reduce
        the decoder training epochs (phase_epochs) proportionally to the number of flow steps.
        For example, if phase_epochs=5000 and flow_steps=100, we set phase_epochs=50.

        NOTE: This only adjusts phase_epochs (decoder training), NOT policy_epochs (policy training).
              Policy training is unaffected as it doesn't use flow models.
        """
        # Check if decoder is FQLDecoder (flow-based)
        if self.decoder_config.get('model_cls') == FQLDecoder:
            flow_steps = self.decoder_config.get('model_kwargs', {}).get('flow_steps', 100)

            if flow_steps > 0:
                original_epochs = self.phase_epochs
                self.policy_epochs = self.phase_epochs
                self.phase_epochs = int(self.phase_epochs / flow_steps)

                # Ensure at least 1 epoch
                if self.phase_epochs < 1:
                    self.phase_epochs = 1

                logger = get_logger(__name__)
                logger.info(f"Adjusted decoder phase_epochs for FQL: {original_epochs} / {flow_steps} = {self.phase_epochs}")

                # Log policy_epochs if it exists (only for certain environments like MMWorld)
                if hasattr(self, 'policy_epochs'):
                    logger.info(f"Policy epochs remain unchanged: {self.policy_epochs}")

    def _apply_action_chunk_hook(self):
        """Apply action chunking preprocessing hook if enabled."""
        if self.action_chunk > 1:
            from SILGym.dataset.dataloader import action_chunk_hook

            chunk_hook = (action_chunk_hook, {
                'chunk_size': self.action_chunk,
                'padding_mode': self.action_chunk_padding
            })

            # Add to post_process_hooks for both decoder and policy dataloaders
            if 'post_process_hooks_kwargs' not in self.dataloader_kwargs:
                self.dataloader_kwargs['post_process_hooks_kwargs'] = []
            if chunk_hook not in self.dataloader_kwargs['post_process_hooks_kwargs']:
                self.dataloader_kwargs['post_process_hooks_kwargs'].append(chunk_hook)

            if 'post_process_hooks_kwargs' not in self.dataloader_kwargs_policy:
                self.dataloader_kwargs_policy['post_process_hooks_kwargs'] = []
            if chunk_hook not in self.dataloader_kwargs_policy['post_process_hooks_kwargs']:
                self.dataloader_kwargs_policy['post_process_hooks_kwargs'].append(chunk_hook)

            get_logger(__name__).info(
                f"Action chunking enabled: chunk_size={self.action_chunk}, "
                f"padding_mode={self.action_chunk_padding}"
            )

    def _apply_noise_config(self):
        """Apply Gaussian noise configuration to the remote observation helper if enabled."""
        if self.eval_noise_enabled:
            get_logger(__name__).info(f"Applying evaluation Gaussian noise: scale={self.eval_noise_scale}")
            # Create noise helper, potentially wrapping existing helper
            self.remote_obs_helper = NoiseEvaluationHelper(
                noise_type='gaussian',
                noise_scale=self.eval_noise_scale,
                noise_clip=self.eval_noise_clip,
                base_helper=self.remote_obs_helper,  # Wrap existing helper if any
                seed=self.eval_noise_seed
            )

    def _compute_kitchen_obs_dim(
        self,
        *,
        dataset_paths: typing.List[str],
        obs_modalities: typing.Optional[typing.Iterable[str]],
        use_oracle_proprio: bool,
        oracle_key_name: str,
    ) -> int:
        """
        Resolve the kitchen observation dimension given current metadata.

        This method first attempts to reuse cached metadata, then tries to infer the
        dimension directly from the dataset files, and finally falls back to metadata
        defaults when required.
        """

        logger = get_logger(__name__)

        def infer_from_datasets(
            paths: typing.Iterable[str],
            obs_key_filter: typing.Optional[typing.Iterable[str]],
        ) -> typing.Optional[int]:
            """
            Infer kitchen observation dimensionality from embedded HDF5 files.

            If obs_key_filter is provided, only the matching keys are summed. The first
            demo that yields a positive dimension is considered sufficient.
            """
            for path in paths:
                if not path or not str(path).endswith(".hdf5"):
                    continue
                if not os.path.exists(path):
                    continue
                try:
                    import h5py  # type: ignore
                except ImportError:
                    logger.warning(
                        "h5py is not installed; cannot infer kitchen observation dimension automatically."
                    )
                    return None

                try:
                    with h5py.File(path, "r") as f:
                        data_group = f.get("data")
                        if not data_group:
                            continue
                        # Pick the first demonstration with observations
                        for demo_key in data_group.keys():
                            demo = data_group[demo_key]
                            obs_group = demo.get("obs")
                            if not obs_group:
                                continue
                            default_keys = (
                                'agentview_rgb_dinov3',
                                'eye_in_hand_rgb_dinov3',
                                'agentview_rgb',
                                'eye_in_hand_rgb',
                                'joint_states',
                                'ee_states',
                                'gripper_states',
                                'robot_states',
                                'state',
                                'proprio',
                            )
                            if obs_key_filter is not None:
                                obs_keys_priority = list(obs_key_filter)
                            else:
                                obs_keys_priority = list(default_keys)
                            total_dim = 0
                            for obs_key in obs_keys_priority:
                                dataset = None
                                if obs_key in obs_group:
                                    dataset = obs_group[obs_key]
                                elif obs_key in demo:
                                    dataset = demo[obs_key]
                                if dataset is None:
                                    continue
                                shape = dataset.shape
                                if not shape:
                                    continue
                                if len(shape) == 1:
                                    feature_dim = 1
                                else:
                                    feature_dim = 1
                                    for size in shape[1:]:
                                        feature_dim *= int(size)
                                total_dim += int(feature_dim)
                            if total_dim > 0:
                                return total_dim
                except Exception as exc:
                    logger.warning(
                        f"Failed to infer kitchen observation dimension from '{path}': {exc}"
                    )
                    continue
            return None

        if not use_oracle_proprio:
            cached = self.scenario_metadata.get("kitchen_obs_dim")
            if cached is not None:
                return int(cached)
        else:
            self.scenario_metadata.pop("kitchen_obs_dim", None)

        dim_keys: typing.Optional[typing.List[str]] = None
        if obs_modalities is not None:
            modality_list = list(obs_modalities)
            if use_oracle_proprio:
                proprio_keys = {"joint_states", "ee_states", "gripper_states", "robot_states"}
                dim_keys = [key for key in modality_list if key not in proprio_keys]
                if oracle_key_name not in dim_keys:
                    dim_keys.append(oracle_key_name)
            else:
                dim_keys = modality_list

        if use_oracle_proprio is False:
            inferred_dim = infer_from_datasets(dataset_paths, obs_key_filter=dim_keys)
            if inferred_dim is not None:
                return int(inferred_dim)

        embedding_dim = self.scenario_metadata.get("kitchen_embedding_dim")
        if use_oracle_proprio:
            proprio_dim = 60  # oracle state size
        else:
            proprio_dim = self.scenario_metadata.get("kitchen_proprio_dim")
            if proprio_dim is None:
                fallback_keys = self.scenario_metadata.get("kitchen_proprio_keys")
                if isinstance(fallback_keys, (list, tuple)) and fallback_keys:
                    proprio_dim = sum(int(KITCHEN_PROPRIO_KEY_DIMS.get(str(key), 0)) for key in fallback_keys)
                else:
                    proprio_dim = sum(int(KITCHEN_PROPRIO_KEY_DIMS.get(key, 0)) for key in DEFAULT_KITCHEN_PROPRIO_KEYS)
            proprio_dim = int(proprio_dim)
        if embedding_dim is not None:
            if obs_modalities:
                cam_keys = {"agentview_rgb_dinov3", "eye_in_hand_rgb_dinov3"}
                cam_count = sum(1 for key in obs_modalities if key in cam_keys)
                include_proprio = True if use_oracle_proprio else any(
                    key in {"joint_states", "ee_states", "gripper_states", "robot_states", "state", "states", "proprio"}
                    for key in obs_modalities
                )
                return int(cam_count * int(embedding_dim) + (int(proprio_dim) if include_proprio else 0))
            return int(2 * int(embedding_dim) + int(proprio_dim))

        logger.warning(
            "Unable to determine kitchen visual observation dimension; falling back to legacy dimension 60."
        )
        return 60

    def update_config(self, **kwargs):
        """
        Updates the configuration attributes of the instance with the provided keyword arguments.
        This allows for dynamic updates to the configuration after initialization.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid attribute '{key}' for this configuration class.")

        # Update appender_cls if appender_version was changed
        if 'appender_version' in kwargs:
            if self.appender_version in APPENDER_CLASS_MAP:
                self.appender_cls = APPENDER_CLASS_MAP[self.appender_version]
            else:
                raise ValueError(f"Invalid appender_version '{self.appender_version}'. Must be one of {list(APPENDER_CLASS_MAP.keys())}")

        # Re-adjust epochs if phase_epochs was manually updated and using flow decoder
        if 'phase_epochs' in kwargs:
            self._adjust_epochs_for_flow_decoder()

        # Re-apply noise config if noise settings were updated
        noise_keys = ['eval_noise_enabled', 'eval_noise_scale',
                      'eval_noise_clip', 'eval_noise_seed']
        if any(key in kwargs for key in noise_keys):
            self._apply_noise_config()
    
    def _update_decoder_config(self, decoder_config, algo_type):
        """
        Updates the model configuration based on the selected algorithm type.
        
        For 'ptgm', if the current condition input shape is given as (1, 1, x),
        it doubles the last dimension, resulting in (1, 1, 2*x).
        For 'SILGym', it sets the condition input shape to (1, 1, 60).
        Args:
            decoder_config (dict): The current model configuration dictionary.
            algo_type (str): The algorithm type identifier ('ptgm' or 'SILGym').
            
        Returns:
            dict: The updated model configuration.
            
        Raises:
            ValueError: If the condition input shape is not in the expected format or
                        if an unsupported algo_type is provided.
        """
        if algo_type in ['ptgm', 'ptgmu', 'buds', 'lazysi', 'silc']:
            # Retrieve the current condition shape from the model configuration
            current_shape = decoder_config['model_kwargs']['input_config']['cond']
            # Ensure current_shape is in the expected format: (1, 1, x)
            if not (isinstance(current_shape, tuple) and len(current_shape) == 3 and current_shape[0] == 1 and current_shape[1] == 1):
                raise ValueError("Expected condition input shape to be in the format (1, 1, x)")
            x = current_shape[2]
            # Double the last dimension
            new_shape = (1, 1, 2 * x)
            decoder_config['model_kwargs']['input_config']['cond'] = new_shape
        else:
            raise ValueError(f"Unsupported algo_type: {algo_type}")
        return decoder_config
    
    def _validate_algo_type(self):
        """Validates that the selected algorithm type is supported."""
        valid_algo_types = ['ptgm', 'lotus', 'iscil' ,'imanip', 'buds', 'lazysi', 'silc']
        if self.algo_type not in valid_algo_types:
            raise ValueError(f"Unsupported algo_type: {self.algo_type}. Must be one of {valid_algo_types}.")
    
    # ------------------------------
    # Model Path Utils
    # ------------------------------
    def _validate_model_path(self):
        """
        Creates the model path if it does not exist.
        If it exists, asks the user if it should be deleted.
        """
        path = self.exp_save_path
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(f'{path}/skills')
            os.makedirs(f'{path}/policy')
        else:
            user_input = input(f"Model path {path} already exists. Do you want to delete it? (Y/N): ").strip().lower()
            if user_input == 'y':
                shutil.rmtree(path)
                os.makedirs(path)
                os.makedirs(f'{path}/skills')
                os.makedirs(f'{path}/policy')
                get_logger(__name__).info(f"Deleted and recreated model path: {path}")
            else:
                raise ValueError(f"Model path {path} already exists and was not deleted.")

    def skill_decoder_path(self, phase=0):
        """
        Returns the file path for saving the skill decoder model for a given phase.
        """
        return f'{self.exp_save_path}/skills/decoder_{phase}.pkl'

    def skill_interface_path(self, phase=0):
        """
        Returns the file path for saving the skill interface for a given phase.
        """
        return f'{self.exp_save_path}/skills/interface_{phase}.pkl'

    def skill_policy_path(self, phase=0):
        """
        Returns the file path for saving the skill policy model for a given phase.
        """
        return f'{self.exp_save_path}/policy/policy_{phase}_{0}.pkl'

    def print_and_save_config(self, file_path=None):
        # must be called in the main function NOTE
        """
        Prints and saves all configuration attributes of the instance in a human-readable text format.
        This includes attributes added in subclasses.

        Args:
            file_path (str, optional): The path to the file where the configuration should be saved.
                                       If not provided, the configuration will be saved to 
                                       self.exp_save_path/config.txt.
        """
        # Convert all instance attributes to a readable string
        config_text = pprint.pformat(self.__dict__, indent=4)
        get_logger(__name__).info(f"Configuration:\n{config_text}")
        
        # Set default file path if none is provided
        if file_path is None:
            file_path = os.path.join(self.exp_save_path, 'experiment_config.txt')
        
        # Save the configuration text to the file
        with open(file_path, 'w') as f:
            f.write(config_text)
        get_logger(__name__).info(f"Configuration saved to {file_path}")
