from AppOSI.config.skill_stream_config import SkillStreamConfig
from AppOSI.dataset.dataloader import BaseDataloader, PoolDataLoader
import os
import shutil
import optax
from datetime import datetime, timedelta
import pytz
import pprint
# Define the log base path
LOGBASEPATH = os.environ.get('LOGBASEPATH', 'logs')

# Import model classes
from AppOSI.models.skill_decoder.diffusion_base import CondDDPMDecoder
from AppOSI.models.basic.base import DenoisingMLP
from AppOSI.models.skill_interface.ptgm import ContinualPTGMInterface, PTGMInterfaceConfig
from AppOSI.models.skill_decoder.appender import ModelAppender, AppendConfig
from AppOSI.dataset.dataloader import DataloaderMixer, DataloaderExtender
from AppOSI.models.agent.base import *

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
    }
}

from AppOSI.models.task_policy.mlp_base import MLPPolicy, HighLevelPolicy
DEFAULT_POLICY_CONFIG = {
    'model_cls': HighLevelPolicy,
    'model_kwargs': {
        'model_config': {
            'hidden_size': 512,
            'out_shape': 400, # shape of the INTERFACE action space 20 per skills
            'num_hidden_layers': 4, 
            'dropout': 0.1, 
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
            tsne_perplexity=50
        ),
        "update_mode" : "expand", # 'overwrite' or 'expand'
    }
}

from AppOSI.models.skill_interface.buds import ContinualBUDSInterface, BUDSInterfaceConfig
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
        # 0420 original
        # "config": BUDSInterfaceConfig(
        #     window_size=5,
        #     min_length=30,
        #     target_num_segments=10,
        #     max_k=40,
        #     goal_offset=20,
        #     verbose=True,
        # ),
        "update_mode" : "expand", #  'expand'
    }
}

from AppOSI.models.skill_interface.semantic_based import SemanticInterface, ImanipInterfaceConfig
DEFAULT_IMANIP_INTERFACE_CONFIG = {
    "interface_cls": SemanticInterface, 
    "interface_kwargs": {
        'config' : ImanipInterfaceConfig(),
        'semantic_emb_path' : "exp/instruction_embedding/kitchen_clip/512.pkl",
    }
}

from AppOSI.models.skill_interface.semantic_based import PrototypeInterface, IsCiLInterfaceConfig
DEFAULT_ISCIL_INTERFACE_CONFIG = {
    "interface_cls": PrototypeInterface,
    "interface_kwargs": {
        'config' : IsCiLInterfaceConfig(
            bases_num=100, # skill bases updated to 100
        ),
        'semantic_emb_path' : "exp/instruction_embedding/kitchen_clip/512.pkl",
    }
}

from AppOSI.models.skill_interface.assil import AsSILInterface, AsSILInterfaceConfig
DEFAULT_ASSIL_INTERFACE_CONFIG ={
    "interface_cls": AsSILInterface,
    "interface_kwargs": {
        "config": AsSILInterfaceConfig(
            cluster_num=20,
            goal_offset=20,
            prototype_bases=3,
            # policy side
            cluster_num_policy=10,
            # tsne side
            tsne_dim=2,
            tsne_perplexity=50,
        ),
    }
}

from AppOSI.models.skill_interface.lazySI import LazySIInterface, LazySIInterfaceConfig
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
            decoder_prototype_bases= 4,
            # policy side
            policy_algo= "ptgm",
            policy_algo_config=PTGMInterfaceConfig(
                cluster_num= 20,
                goal_offset= 20,
                tsne_perplexity= 50,
            ),
            policy_prototype_bases= 4,
            # confidence interval
            confidence_interval = 99.0,
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
DEFAULT_ALGO_TYPES = ['ptgm', 'apposi']

from AppOSI.dataset.dataloader import libero_obs_obs_hook, history_state_obs_hook, history_state_three_obs_hook, dropthe_traj_hook
from AppOSI.environment.remote import KitchenRemoteEvaluator, MMWorldRemoteEvaluator, LiberoRemoteEvaluator, HistoryEvalHelper, HistoryEvalHelperTriple
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
        ) -> None:
        # Basic experiment settings
        self.scenario_config = scenario_config
        self.scenario_id = self.scenario_config.scenario_id
        self.algo_type = 'ptgm'  # Default: ptgm, spirl, apposi # Override only in subclasses
        self.lifelong_algo = lifelong_algo # ['ft', 'er', 'append', 'expand'] # expand is depend on subclass method
        self.seed = seed

        self.exp_id = get_date_string() + exp_id + f"seed{self.seed}" # format of '{date}{env}{scenarioType}_{auxiliary}{seed}'

        # ----------------------------
        # Scenario Environment related settings
        # ----------------------------
        self.environment = scenario_config.environment
        self.scenario_type = scenario_config.scenario_type
        self.sync_type = scenario_config.sync_type

        self.evaluator_cls = KitchenRemoteEvaluator
        self.remote_eval_host = '127.0.0.1'
        self.remote_eval_port = 9999
        self.remote_obs_helper = None

        self.eval_num_episodes = 3
        self.eval_max_steps = 280
        
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
        self.pretrained_policy_path = 'data/pretrained_policy'
        self.init_policy_path = None
        self.policy_config = policy_config if policy_config is not None else DEFAULT_POLICY_CONFIG


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
        # * Append-based baseline settings
        # ----------------------------
        self.is_appendable = False # Append-based baseline
        self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(
            lora_dim=4,
            pool_length=10,
        )

        # ----------------------------
        # Ablation settings
        # ----------------------------
        self.phase_reset = True # DEFAULT for FALSE

        self.agent_dict = {
            'ptgm': PTGMAgent,
            'iscil': IsCiLAgent, 
            'imanip': IsCiLAgent,
            'assil': AsSILAgent,
            'buds': BUDSAgent,
            'lazysi': LazySIAgent,
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
        """
        return f"_{self.lifelong_algo}" if self.lifelong_algo else ""

    @property
    def exp_save_path(self):
        """
        Returns the experiment save path.
        This is used to store the results and models of the experiment.
        """
        return f'{LOGBASEPATH}/{self.environment}/{self.scenario_type}/{self.sync_type}/{self.algo_type}{self.ll_postfix}/{self.exp_id}'
    # ------------------------------
    # Update Model Configurations
    # ------------------------------

    def update_config_by_env(self):
        print(f"[SkillExperimentConfig] Updating model configurations for environment: {self.environment}")
        if self.environment == 'kitchen':
            self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, 60) # state space
            self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 9) # action space
            self.decoder_config['model_kwargs']['input_config']['cond'] = (1, 1, 60) # state space

            self.evaluator_cls = KitchenRemoteEvaluator
            self.remote_eval_host = '127.0.0.1'
            self.remote_eval_port = 9999

            self.eval_num_episodes = 3
            self.eval_max_steps = 280
        
        elif self.environment == 'mmworld':
            # self.dataloader_kwargs['pre_process_hooks_kwargs'] =[
            #     (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
            # ]
            # self.remote_obs_helper = HistoryEvalHelper(10)
            
            self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, 140)
            self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 4) # action space
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
                print(f"[SkillExperimentConfig] MMWorld n1 scenario: {self.phase_epochs} epochs")

            if self.scenario_type in  ['explicit', 'n2exp'] : # about 250k
                # self.dataloader_kwargs['pre_process_hooks_kwargs'] = [
                #     (dropthe_traj_hook, {'per_drop' : 2}),
                # ]
                self.phase_epochs = 1_000
                print(f"[SkillExperimentConfig] MMWorld explicit scenario: {self.phase_epochs} epochs")
                self.policy_epochs = 5_000
        elif self.environment == 'libero':
            self.dataloader_kwargs['pre_process_hooks_kwargs'] =[
                (libero_obs_obs_hook, {'obs_dim': 130}),
                # (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
                (history_state_three_obs_hook, {'N': 15}), # it triples the last dimension
            ]

            self.dataloader_kwargs_policy['pre_process_hooks_kwargs'] =[
                (libero_obs_obs_hook, {'obs_dim': 130}),
                # (history_state_obs_hook, {'N': 10}), # it doubles the last dimension
                (history_state_three_obs_hook, {'N': 15}), # it triples the last dimension
            ]


            # self.remote_obs_helper = HistoryEvalHelper(10)
            self.remote_obs_helper = HistoryEvalHelperTriple(15)

            self.policy_config['model_kwargs']['input_config']['x'] = (1, 1, 130*3)
            self.policy_config['model_kwargs']['optimizer_config']['optimizer_kwargs']['learning_rate'] = 1e-3

            self.decoder_config['model_kwargs']['input_config']['x'] = (1, 1, 7) # action space
            self.decoder_config['model_kwargs']['input_config']['cond'] = (1, 1, 130*3) # state space
            # self.decoder_config['model_kwargs']['optimizer_config']['optimizer_kwargs']['learning_rate'] = 2e-6
            
            self.evaluator_cls = LiberoRemoteEvaluator
            self.remote_eval_host = '127.0.0.1'
            self.remote_eval_port = 7777

            self.phase_epochs = 10_000

            self.eval_num_episodes = 3
            self.eval_max_steps = 600

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
    
    def _update_decoder_config(self, decoder_config, algo_type):
        """
        Updates the model configuration based on the selected algorithm type.
        
        For 'ptgm', if the current condition input shape is given as (1, 1, x),
        it doubles the last dimension, resulting in (1, 1, 2*x).
        For 'apposi', it sets the condition input shape to (1, 1, 60).
        Args:
            decoder_config (dict): The current model configuration dictionary.
            algo_type (str): The algorithm type identifier ('ptgm' or 'apposi').
            
        Returns:
            dict: The updated model configuration.
            
        Raises:
            ValueError: If the condition input shape is not in the expected format or
                        if an unsupported algo_type is provided.
        """
        if algo_type in ['ptgm', 'buds', 'assil', 'lazysi']:
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
        valid_algo_types = ['ptgm', 'lotus', 'iscil' ,'imanip', 'assil', 'buds', 'lazysi']
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
                print(f"Deleted and recreated model path: {path}")
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
        print(config_text)
        
        # Set default file path if none is provided
        if file_path is None:
            file_path = os.path.join(self.exp_save_path, 'experiment_config.txt')
        
        # Save the configuration text to the file
        with open(file_path, 'w') as f:
            f.write(config_text)
        print(f"Configuration saved to {file_path}")
