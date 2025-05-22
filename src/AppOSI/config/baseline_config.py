import argparse
import warnings

# Import necessary configuration classes and functions explicitly
from AppOSI.config.skill_stream_config import SkillStreamConfig
from AppOSI.config.experiment_config import SkillExperimentConfig, DEFAULT_POLICY_CONFIG, DEFAULT_DECODER_CONFIG, DEFAULT_PTGM_INTERFACE_CONFIG
from AppOSI.config.experiment_config import DEFAULT_IMANIP_INTERFACE_CONFIG, DEFAULT_ISCIL_INTERFACE_CONFIG, DEFAULT_BUDS_INTERFACE_CONFIG
from AppOSI.config.kitchen_scenario import kitchen_scenario
from AppOSI.trainer.skill_trainer import SkillTrainer
from AppOSI.models.skill_decoder.appender import ModelAppender, AppendConfig
from AppOSI.dataset.dataloader import DataloaderExtender, DataloaderMixer
from AppOSI.models.agent.base import LazySIAgent, LazySIZeroAgent
# Optional imports for interface config types
from AppOSI.models.skill_interface.buds import BUDSInterfaceConfig
from AppOSI.models.skill_interface.ptgm import PTGMInterfaceConfig
from AppOSI.models.skill_interface.lazySI import SemanticInterfaceConfig

class BUDSConfig(SkillExperimentConfig):
    """Experiment configuration for the PTGM algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = DEFAULT_DECODER_CONFIG,
        interface_config: dict = DEFAULT_BUDS_INTERFACE_CONFIG,
        policy_config: dict = DEFAULT_POLICY_CONFIG,
        # 
        exp_id : str = "DEF",
        lifelong_algo : str = "",
        seed: int = 0,
    ) -> None:
        
        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
        )
        # ----------------------------
        # Algorithm-related settings
        # ----------------------------
        self.algo_type = 'buds'
        self._validate_algo_type()

        # ----------------------------
        # Essentials lifelong learning settings
        # ----------------------------
        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()


    def _update_decoder_config(self, decoder_config, algo_type):
        return super()._update_decoder_config(decoder_config, algo_type)
    
    def _setup_lifelong_algo(self):
        '''
        lifelong_algo : str
            - ft : Fine-tune the decoder.
            - er : experience replay
                - er.2 : experience replay with 20% buffer keep ratio
                - erf : experience replay with full buffer keep ratio
            - append : Append the decoder with lora.
            - expand : Append + ER(for new adapter)
        '''
        if self.lifelong_algo == "ft":
            self.is_appendable = False
        elif self.lifelong_algo.startswith("er"):
            self.is_appendable = False
            # Extract the suffix after "er" (e.g., "", "20", "20%")
            suffix = self.lifelong_algo[len("er"):]
            suffix = suffix.rstrip("%")
            if suffix == "":
                # If no number is provided, default to 10%
                percent_value = 10
            else:
                # Ensure the suffix is a valid integer string
                if suffix.isdigit():
                    percent_value = int(suffix)
                else:
                    raise ValueError("Invalid buffer_keep_ratio value: must be an integer percentage between 10 and 100")
            # Check if the provided value is within the valid range (10 to 100)
            if percent_value < 0 or percent_value > 100:
                raise ValueError("Invalid buffer_keep_ratio value: must be between 10 and 100 percent")
            # Convert percentage to a float ratio (e.g., 20 -> 0.2)
            self.buffer_keep_ratio = percent_value / 100.0
        # Append 
        elif self.lifelong_algo.startswith("append"):
            self.is_appendable = True
            self.appender_cls = ModelAppender
            suffix = self.lifelong_algo[len("append"):]
            lora_dim = int(suffix) if suffix.isdigit() else 4
            self.appender_config = AppendConfig(
                lora_dim=lora_dim,
                pool_length=10,
            )

class PTGMConfig(SkillExperimentConfig):
    """Experiment configuration for the PTGM algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = DEFAULT_DECODER_CONFIG,
        interface_config: dict = DEFAULT_PTGM_INTERFACE_CONFIG,
        policy_config: dict = DEFAULT_POLICY_CONFIG,
        # 
        exp_id : str = "DEF",
        lifelong_algo : str = "",
        seed: int = 0,
    ) -> None:
        
        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
        )
        # ----------------------------
        # Algorithm-related settings
        # ----------------------------
        self.algo_type = 'ptgm'
        self._validate_algo_type()

        # ----------------------------
        # Essentials lifelong learning settings
        # ----------------------------
        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()


    def _update_decoder_config(self, decoder_config, algo_type):
        if self.sync_type == "joint":
            self.interface_config["interface_kwargs"]["ptgm_config"].cluster_num = 100 # PTGMInterfaceConfig
        return super()._update_decoder_config(decoder_config, algo_type)
    
    def _setup_lifelong_algo(self):
        """
        Parse self.lifelong_algo for an optional 'noTsne', 's<decoder>' and/or 'g<groups>'
        and then dispatch into the existing ft/er/append logic.

        After running this, you will have:
        - self.tsne           (True unless 'noTsne' was at the front)
        - self.cluster_num    (if s<> was given)
        - self.groups_num     (if g<> was given)
        - self.is_appendable, self.buffer_keep_ratio, self.appender_config, etc.
        """
        spec = self.lifelong_algo.strip().lower()

        # default tsne on
        self.tsne = True

        # 0) Extract optional 'notsne' at the very front
        import re
        tsne_match = re.match(r"notsne", spec)
        if tsne_match:
            self.tsne = False
            spec = spec[tsne_match.end():]

        # 1) Extract optional 's<decoder>' and/or 'g<groups>' at the very front
        s_match = re.match(r"s(?P<decoder>\d+)", spec)
        if s_match:
            self.cluster_num = int(s_match.group("decoder"))
            spec = spec[s_match.end():]

        g_match = re.match(r"g(?P<groups>\d+)", spec)
        if g_match:
            self.groups_num = int(g_match.group("groups"))
            spec = spec[g_match.end():]

        # Clean up any leading separators or whitespace
        spec = spec.lstrip("-_ ")

        # 2) Now spec should be one of: "", "ft", "er", "er10", "append", "append3", etc.
        if not spec:
            spec = "ft"

        # 3) Dispatch to your existing handlers
        if spec == "ft":
            self.is_appendable = False

        elif spec.startswith("er"):
            self.is_appendable = False
            suffix = spec[len("er"):].rstrip("%")
            percent = int(suffix) if suffix.isdigit() else 10
            if not (0 <= percent <= 100):
                raise ValueError("Invalid buffer_keep_ratio: must be 0–100")
            self.buffer_keep_ratio = percent / 100.0

        elif spec.startswith("append"):
            self.is_appendable = True
            suffix = spec[len("append"):]
            lora_dim = int(suffix) if suffix.isdigit() else 4
            self.appender_cls = ModelAppender
            self.appender_config = AppendConfig(lora_dim=lora_dim, pool_length=10)

        else:
            raise ValueError(f"Unknown lifelong_algo spec '{spec}'")

        # 4) Finally, if we parsed a global s<> or g<>, inject those into the interface config
        cfg = self.interface_config["interface_kwargs"]["ptgm_config"]
        if hasattr(self, "cluster_num"):
            cfg.cluster_num = self.cluster_num
            print(f"[INFO] global cluster_num set to {self.cluster_num}")
        if hasattr(self, "groups_num"):
            cfg.goal_offset = self.groups_num
            print(f"[INFO] global groups_num set to {self.groups_num}")
        if hasattr(self, "tsne"):
            if self.tsne is False:
                cfg.tsne_dim = 0

        print(f"[INFO] tsne enabled: {self.tsne}")

class IsCiLConfig(SkillExperimentConfig):
    """Experiment configuration for the Iscil algorithm."""
    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = DEFAULT_DECODER_CONFIG,
        interface_config: dict = DEFAULT_ISCIL_INTERFACE_CONFIG,
        policy_config: dict = DEFAULT_POLICY_CONFIG,
        exp_id : str = "DEF",
        lifelong_algo : str = "",
        seed: int = 0,
    ) -> None:
        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
        )
        # ----------------------------
        # Algorithm-related settings
        # ----------------------------
        self.algo_type = 'iscil' # iscil or Iscil
        self._validate_algo_type()

        self.is_appendable = True
        self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(
            lora_dim=4,
            # pool_length=30, # Main bottle neck for the evaluation function.
            pool_length=8, # Main bottle neck for the evaluation function.
        )

        # ----------------------------
        # Interface - semantic path settings.
        # ----------------------------

        # NOTE semantic_path format : "path/to/semantic/[dimension].pt"
        self.semantic_path = self.interface_config['interface_kwargs']['semantic_emb_path']
        self.embed_dim =int(self.semantic_path.split('/')[-1].split('.')[0])
        self.decoder_config = self._update_decoder_config(decoder_config)

        # ----------------------------
        # Essentials lifelong learning settings
        # ----------------------------
        self._setup_lifelong_algo()
        self._validate_model_path()

    def update_config_by_env(self):
        super().update_config_by_env()
        if self.environment == 'kitchen' :
            pass
        elif self.environment == 'mmworld' :
            print("[INFO] ImanipConfig: mmworld environment.")
            self.interface_config['interface_kwargs']['semantic_emb_path'] = 'exp/instruction_embedding/mmworld/512.pkl'
        else :
            raise NotImplementedError("ImanipConfig only supports kitchen and mmworld environments.")

    def _update_decoder_config(self, decoder_config):
        current_shape = decoder_config['model_kwargs']['input_config']['cond']
        x = current_shape[2] # (B, 1, F)
        new_shape = (1,1,x+self.embed_dim)
        decoder_config['model_kwargs']['input_config']['cond'] = new_shape
        return decoder_config
    
    def _setup_lifelong_algo(self):
        bases_num = 50
        if self.lifelong_algo.startswith("bases"):
            suffix = self.lifelong_algo[len("bases"):]
            if suffix == "":
                bases_num = 50
            else:
                # Ensure the suffix is a valid integer string
                if suffix.isdigit():
                    bases_num = int(suffix)
                else:
                    raise ValueError("Invalid bases_num value: must be an integer")
        else :
            self.lifelong_algo = "bases50" # DEFAULT settings 
        self.interface_config['interface_kwargs']['config'].bases_num = bases_num # IscilInterfaceConfig
        print("[INFO] IscilInterfaceConfig bases_num : ", bases_num)
    
class ImanipConfig(SkillExperimentConfig): # imanip == semantic
    """Experiment configuration for the Imanip algorithm."""
    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = DEFAULT_DECODER_CONFIG,
        interface_config: dict = DEFAULT_IMANIP_INTERFACE_CONFIG,
        policy_config: dict = DEFAULT_POLICY_CONFIG,
        exp_id : str = "DEF",
        lifelong_algo : str = "tr",
        seed: int = 0,
    ) -> None:
        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
        )
        # ----------------------------
        # Algorithm-related settings
        # ----------------------------
        self.algo_type = 'imanip'
        self._validate_algo_type()
        

        self.is_appendable = True
        self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(
            lora_dim=16, # Default 16 for skill
            pool_length=10, # Main bottle neck for the evaluation function.
        )

        # self.dataloader_mixer_cls = DataloaderExtender
        self.dataloader_mixer_cls = DataloaderMixer
        # ----------------------------
        # Interface - semantic path settings.
        # ----------------------------
        # NOTE semantic_path format : "path/to/semantic/[dimension].pt"
        self.semantic_path = self.interface_config['interface_kwargs']['semantic_emb_path']
        self.embed_dim =int(self.semantic_path.split('/')[-1].split('.')[0])
        self.decoder_config = self._update_decoder_config(decoder_config)

        # ----------------------------
        # Essentials lifelong learning settings
        # ----------------------------
        self._validate_model_path()
        self._set_up_lifelong_algo()

    def update_config_by_env(self):
        super().update_config_by_env()

        if self.environment == 'kitchen' :
            pass
        elif self.environment == 'mmworld' :
            print("[INFO] ImanipConfig: mmworld environment.")
            self.interface_config['interface_kwargs']['semantic_emb_path'] = 'exp/instruction_embedding/mmworld/512.pkl'
        else :
            raise NotImplementedError("ImanipConfig only supports kitchen and mmworld environments.")
        
    def _update_decoder_config(self, decoder_config):
        current_shape = decoder_config['model_kwargs']['input_config']['cond']
        x = current_shape[2] # (B, 1, F)
        new_shape = (1,1,x+self.embed_dim)
        decoder_config['model_kwargs']['input_config']['cond'] = new_shape
        return decoder_config
    
    def _set_up_lifelong_algo(self):
        if self.lifelong_algo.startswith("tr"):
            self.is_appendable = False
            # Extract the suffix after "er" (e.g., "", "20", "20%")
            suffix = self.lifelong_algo[len("tr"):]
            suffix = suffix.rstrip("%")
            if suffix == "":
                # If no number is provided, default to 10%
                percent_value = 10
            else:
                # Ensure the suffix is a valid integer string
                if suffix.isdigit():
                    percent_value = int(suffix)
                else:
                    raise ValueError("Invalid buffer_keep_ratio value: must be an integer percentage between 10 and 100")
            # Check if the provided value is within the valid range (10 to 100)
            if percent_value < 0 or percent_value > 100:
                raise ValueError("Invalid buffer_keep_ratio value: must be between 10 and 100 percent")
            # Convert percentage to a float ratio (e.g., 20 -> 0.2)
            self.buffer_keep_ratio = percent_value / 100.0

            if percent_value == 0 :
                print("[INFO] ImanipConfig: buffer_keep_ratio is 0. No buffer.")
                self.is_appendable = True
        elif self.lifelong_algo.startswith("append"):
            self.is_appendable = True
            self.appender_cls = ModelAppender
            suffix = self.lifelong_algo[len("append"):]
            lora_dim = int(suffix) if suffix.isdigit() else 16
            self.appender_config = AppendConfig(
                lora_dim=lora_dim,
                pool_length=10,
            )
        else : 
            print("[INFO] Imanip only supports tr(temporarl replay) algorithm.")
            raise NotImplementedError("ImanipConfig only supports tr(temporarl replay) algorithm.")


from AppOSI.config.experiment_config import DEFAULT_ASSIL_INTERFACE_CONFIG
from AppOSI.models.task_policy.mlp_base import HighLevelPolicyWithHook
from AppOSI.models.agent.base import AsSILAgent, AsSILZeroAgent
import re
class AsSILConfig(SkillExperimentConfig):
    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = DEFAULT_DECODER_CONFIG,
        interface_config: dict = DEFAULT_ASSIL_INTERFACE_CONFIG,
        policy_config: dict = DEFAULT_POLICY_CONFIG,
        # 
        exp_id : str = "DEF",
        lifelong_algo : str = "",
        seed: int = 0,
    ) -> None:
        # policy with hook
        policy_config['model_cls'] = HighLevelPolicyWithHook

        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
        )
        # ----------------------------
        # Algorithm-related settings
        # ----------------------------
        self.algo_type = 'assil'
        self.algo_mode = None # None or 'zero' for AsSILZeroAgent
        self._validate_algo_type()

        self.is_appendable = True
        self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(
            lora_dim=4,
            pool_length=8,
        )

        # ----------------------------
        # For future implementation with skill semantic path
        # ----------------------------

        # ----------------------------
        # Essentials lifelong learning settings
        # ----------------------------
        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()

    @property
    def agent_cls(self) :
        if self.algo_mode == 'zero':
            print("[INFO] AsSILZeroAgent is used.")
            return AsSILZeroAgent 
        else :
            print("[INFO] AsSILAgent is used.")
            return AsSILAgent

    def _update_decoder_config(self, decoder_config, algo_type):
        # just re use the ptgm's decoder policy
        return super()._update_decoder_config(decoder_config, algo_type)

    import re  # Ensure this is imported at the top of your file

    
    def _setup_lifelong_algo(self):
        """
        Parse self.lifelong_algo and populate interface config.

        Accepted formats
        ----------------
        s<decoder>p<policy>b<bases>
        zero s<decoder>p<policy>b<bases>
        [zero]s<decoder>p<policy>b<bases>
        (any of the above) + optional g<groups>

        Examples
        --------
        "s20p10b3"
        "zero s20p10b3"
        "[zero]s20p10b3g10"
        """

        # 1) Fallback to a default spec if the field is empty.
        if not self.lifelong_algo:
            self.lifelong_algo = "s20p10b3"

        # Normalize string.
        spec = self.lifelong_algo.strip().lower()

        # 2) Detect optional "zero" prefix, recording the mode and stripping it away.
        algo_mode = "default"
        zero_prefix = r"^\s*(?:\[?zero\]?)\s*"
        if re.match(zero_prefix, spec):
            algo_mode = "zero"
            spec = re.sub(zero_prefix, "", spec, count=1)

        # 3) Parse the body: s<decoder>p<policy>b<bases>[g<groups>].
        pattern = (
            r"^s(?P<decoder>\d+)"      # decoder skill count
            r"p(?P<policy>\d+)"        # policy  skill count
            r"b(?P<bases>\d+)"         # number of prototype bases
            r"(?:r(?P<ranks>\d+))?"     # optional repeat count
            r"(?:g(?P<groups>\d+))?$"  # optional groups/bin count
        )
        m = re.match(pattern, spec)
        if not m:
            raise ValueError(
                "Invalid lifelong_algo format. "
                "Expected '[zero]s<decoder>p<policy>b<bases>[g<groups>]'"
            )

        # 4) Extract numerical values.
        decoder_skill_num = int(m.group("decoder"))
        policy_skill_num  = int(m.group("policy"))
        bases_num         = int(m.group("bases"))
        groups_num        = int(m.group("groups")) if m.group("groups") else None
        rank_dim          = int(m.group("ranks")) if m.group("ranks") else None

        if rank_dim is None:
            # Default rank_dim to 4 if not specified.
            rank_dim = 4

        # 5) Inject values into the nested config object.
        cfg = self.interface_config["interface_kwargs"]["config"]
        cfg.cluster_num         = decoder_skill_num
        cfg.cluster_num_policy  = policy_skill_num
        cfg.prototype_bases     = bases_num
        self.appender_config.lora_dim = rank_dim
        if groups_num is not None:
            print("[INFO] groups_num           :", groups_num)
            cfg.goal_offset = groups_num # Default to 20

        print("[INFO] cluster_num          :", decoder_skill_num)
        print("[INFO] cluster_num_policy   :", policy_skill_num)
        print("[INFO] prototype_bases      :", bases_num)
        print("[INFO] rank_dim             :", rank_dim)
        if groups_num is not None:
            print("[INFO] groups_num           :", groups_num)
        print("[INFO] Algo Mode            :", algo_mode)

        # 7) Store parsed values on the instance for later use.
        self.algo_mode  = algo_mode
        self.groups_num = groups_num



from AppOSI.models.skill_interface.lazySI import InstanceRetrievalConfig
from AppOSI.config.experiment_config import DEFAULT_LAZYSI_INTERFACE_CONFIG
from AppOSI.dataset.dataloader import few_frac_shot_hook

class LazySIConfig(SkillExperimentConfig):
    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = DEFAULT_DECODER_CONFIG,
        interface_config: dict = DEFAULT_LAZYSI_INTERFACE_CONFIG,
        policy_config: dict = DEFAULT_POLICY_CONFIG,
        #
        exp_id : str = "DEF",
        lifelong_algo : str = "",
        seed: int = 0,
        distance_type: str = "maha",
    ) -> None:
        # policy with hook
        policy_config['model_cls'] = HighLevelPolicyWithHook

        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
        )
        self.distance_type = distance_type
        # ----------------------------
        # Algorithm-related settings
        # ----------------------------
        self.algo_type = 'lazysi'
        self.algo_mode = None # None or 'zero' for AsSILZeroAgent
        self._validate_algo_type()

        self.is_appendable = True
        self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(
            lora_dim=16,
            pool_length=8,
        )

        # ----------------------------
        # For future implementation with skill semantic path
        # ----------------------------


        # ----------------------------
        # Essentials lifelong learning settings
        # ----------------------------
        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()

    @property
    def agent_cls(self) :
        if self.algo_mode == 'zero':
            print("[INFO] LazySIZeroAgent is used.")
            return LazySIZeroAgent
        else :
            print("[INFO] LazySIAgent is used.")
            return LazySIAgent

    def update_config_by_env(self):
        super().update_config_by_env()
        if self.environment == 'kitchen' :
            self.semantic_path =  'exp/instruction_embedding/kitchen/512.pkl'
        elif self.environment == 'mmworld' :
            print("[INFO] LazySIConfig: mmworld environment.")
            self.semantic_path =  'exp/instruction_embedding/mmworld/512.pkl'
        else :
            print("[INFO] LazySIConfig: libero environment.")

    def _update_decoder_config(self, decoder_config, algo_type):
        print("[INFO] LazySIConfig: decoder_config is lazy updated in the _setup_lifelong_algo.")
        return decoder_config
    
    def _post_update_decoder_config(self, decoder_config, dec_algo_type):
        if self.decoder_algo in ['ptgm', 'buds']:
            return super()._update_decoder_config(decoder_config, dec_algo_type)
        elif self.decoder_algo == "semantic" :
            current_shape = decoder_config['model_kwargs']['input_config']['cond']
            x = current_shape[2] # (B, 1, F)
            new_shape = (1,1,x+512) # hard coded
            decoder_config['model_kwargs']['input_config']['cond'] = new_shape
            return decoder_config
    
    def _setup_lifelong_algo(self):
        """
        Parse self.lifelong_algo and populate interface config.

        Accepted formats
        ----------------
        [decoder_algo]_[llalgo]_[cluster_algo]/[decoder_algo_config optional]/[policy_algo]/[policy_algo_config]
        decoder_algo : str
            - ptgm : PTGMInterfaceConfig 
            - buds : BUDSInterfaceConfig 
            llalgo : str
            - ft : Fine-tune the decoder.
            - er : experience replay
            - append[rank] : Append the decoder with lora.
            - expand[rank] : Append + ER(for new adapter)

            cluster_algo : str / clustering algorithm for skill clustering.
            - K-means
        
        decoder_algo_config : abbribiated
            - ptgm : s<decoder>g<goal offsets>b<bases> 
            - buds : None

        policy_algo : str
            - ptgm : PTGMInterfaceConfig 
            - buds : BUDSInterfaceConfig 
            - instance : InstanceRetrievalConfig
            - static : PTGMInterfaceConfig (Dummy)

        decoder_algo_config : abbribiated
            - behavior : None
            - ptgm : s<decoder>g<goal offsets>b<bases> 
            - buds : None
        
            
        algo_mode : str 
        * few1 : 1-shot ~ ... 
        * zero : remove base policy
        eg.
        "ptgm/s20b4/ptgm/s20b4"
        "[algo_mode]/ptgm/s20b4/ptgm/s20b4"
        "few1/ptgm/s20b4/ptgm/s20b4"
        "few1frac10/ptgm/s20b4/ptgm/s20b4"
        """
        import re

    
        spec = self.lifelong_algo.strip()
        parts = spec.split('/')
        if len(parts) == 5 :
            algo_mode , decoder_part, dec_conf_str, policy_algo, pol_conf_str = parts
            self.algo_mode = algo_mode
        elif len(parts) == 4:
            decoder_part, dec_conf_str, policy_algo, pol_conf_str = parts
        else : 
            raise ValueError(f"Invalid lifelong_algo format, expected 4 parts but got '{spec}'")


        # algo mode settings NOTE it automatically modify the policy goal offset by frac
        self.shot = None
        self.frac = None
        confidence = 0.99
        threshold_type = 'chi2'
        if self.algo_mode is not None:
            if self.algo_mode.startswith("few"):
                # few{1}frac{1}
                m = re.match(r"few(\d+)(?:frac(\d+))?", self.algo_mode)
                if not m:
                    raise ValueError(f"Invalid algo_mode format '{self.algo_mode}'")
                shot = int(m.group(1))
                frac = int(m.group(2)) if m.group(2) else 1
                self.shot = int(shot)
                self.frac = int(frac)
                if 'pre_process_hooks_kwargs' not in self.dataloader_kwargs_policy:
                    self.dataloader_kwargs_policy['pre_process_hooks_kwargs'] = [
                        (few_frac_shot_hook, {'shot': shot , 'frac': frac}), 
                    ]
                else :
                    self.dataloader_kwargs_policy['pre_process_hooks_kwargs'].append(
                        (few_frac_shot_hook, {'shot': shot , 'frac': frac})
                    )
            elif self.algo_mode.startswith("conf"):
                # conf{1}frac{1}
                if '_chi2' in self.algo_mode:
                    threshold_type = 'chi2'
                    conf_input = self.algo_mode.split('_')[0]
                    m = re.match(r"conf(\d+)?", conf_input  )
                    print("[INFO] algo_mode : ", self.algo_mode, "conf", m.group(1))
                    confidence = int(m.group(1)) if m.group(1) else 99 
                    confidence = confidence / 100.0
                elif '_percentile' in self.algo_mode:
                    threshold_type = 'percentile'
                    conf_input = self.algo_mode.split('_')[0]
                
            else :
                raise ValueError(f"Invalid algo_mode format '{self.algo_mode}'")
            
            

        # decoder_part -> decoder_algo, llalgo, cluster_algo
        sub = decoder_part.split('_')
        self.decoder_algo = sub[0]
        self.llalgo = sub[1] if len(sub) > 1 else None
        self.cluster_algo = sub[2] if len(sub) > 2 else None
        self.policy_algo = policy_algo
        self.policy_algo_config = pol_conf_str

        # logging
        print(f"[INFO] decoder_algo: {self.decoder_algo}")
        print(f"[INFO] llalgo: {self.llalgo}")
        print(f"[INFO] cluster_algo: {self.cluster_algo}")
        print(f"[INFO] decoder_algo_config: {dec_conf_str}")
        print(f"[INFO] policy_algo: {self.policy_algo}")
        print(f"[INFO] policy_algo_config: {self.policy_algo_config}")

        # llalgo behavior
        if self.llalgo == 'ft':
            self.is_appendable = False
        elif self.llalgo and self.llalgo.startswith('er'):
            self.is_appendable = False
            pct = self.llalgo[len('er'):].rstrip('%') or '10'
            self.buffer_keep_ratio = int(pct) / 100.0
        elif self.llalgo and (self.llalgo.startswith('append') or self.llalgo.startswith('expand')):
            self.is_appendable = True
            suf = re.sub(r'^(?:append|expand)', '', self.llalgo)
            dim = int(suf) if suf.isdigit() else 4
            self.appender_cls = ModelAppender
            self.appender_config = AppendConfig(lora_dim=dim, pool_length=5)
        else:
            raise ValueError(f"Unknown llalgo spec '{self.llalgo}'")

        # semantic decoder update.
        self.decoder_config = self._post_update_decoder_config(self.decoder_config, self.decoder_algo)

        # helpers for config parsing
        def parse_ptgm(cfg_str: str):
            m = re.match(r's(?P<cluster>\d+)(?:g(?P<goal>\d+))?(?:b(?P<bases>\d+))?$', cfg_str)
            if not m:
                raise ValueError(f"Invalid PTGM config '{cfg_str}'")
            return {
                'cluster_num': int(m.group('cluster')),
                'goal_offset': int(m.group('goal') or 0),
                'prototype_bases': int(m.group('bases') or 0)
            }

        def parse_bases(cfg_str: str):
            m = re.match(r'(?:g(?P<goal>\d+))?(?:b(?P<bases>\d+))?$', cfg_str)
            if not m:
                raise ValueError(f"Invalid PTGM config '{cfg_str}'")
            
            return {
                'window_size': 5,
                'min_length': 30,
                'target_num_segments': 10,
                'max_k': 10,
                'goal_offset': int(m.group('goal') or 20),
                'prototype_bases': int(m.group('bases') or 1),
                'verbose': False
            }

        cfg = self.interface_config['interface_kwargs']['config']
        # build decoder interface config
        if self.decoder_algo == 'ptgm':
            vals = parse_ptgm(dec_conf_str)
            dec_cfg = PTGMInterfaceConfig(
                cluster_num=vals['cluster_num'],
                goal_offset=vals['goal_offset'],
                tsne_perplexity=50,
                tsne_dim=2,
            )
        elif self.decoder_algo == 'buds':
            vals = parse_bases(dec_conf_str)
            dec_cfg = BUDSInterfaceConfig(**{k: v for k, v in vals.items() if k != 'prototype_bases'})
        elif self.decoder_algo == "semantic" :
            vals = parse_bases(dec_conf_str)
            dec_cfg = SemanticInterfaceConfig(
                semantic_emb_path=self.semantic_path,
                goal_offset=vals['goal_offset'],
            )
        else:
            raise ValueError(f"Unsupported decoder_algo '{self.decoder_algo}'")
        
        cfg.set_decoder_strategy(self.decoder_algo, dec_cfg)
        cfg.decoder_prototype_bases  = vals['prototype_bases']

        # build policy interface config
        if self.policy_algo == 'ptgm':
            vals = parse_ptgm(pol_conf_str)
            vals['goal_offset'] = vals['goal_offset'] if self.frac is None else int(vals['goal_offset'] // self.frac)
            pol_cfg = PTGMInterfaceConfig(
                cluster_num=vals['cluster_num'], 
                goal_offset=vals['goal_offset'], # NOTE Frac 
                tsne_perplexity=50,
                tsne_dim=2,
            )
        elif self.policy_algo == 'buds':
            vals = parse_bases(pol_conf_str)
            vals['goal_offset'] = vals['goal_offset'] if self.frac is None else int(vals['goal_offset'] // self.frac)
            pol_cfg = BUDSInterfaceConfig(**{k: v for k, v in vals.items() if k != 'prototype_bases'})
        elif self.policy_algo == 'instance':
            vals = parse_bases(pol_conf_str)
            vals['goal_offset'] = vals['goal_offset'] if self.frac is None else int(vals['goal_offset'] // self.frac)
            pol_cfg = InstanceRetrievalConfig(
                goal_offset=vals['goal_offset'],
            )
        elif self.policy_algo == 'static':
            pol_cfg = PTGMInterfaceConfig(
                cluster_num=1,
                goal_offset=20,
                tsne_perplexity=1,
                tsne_dim=2,
            )
            self.policy_algo = 'ptgm' # NOTE static is ptgm
            cfg.force_static = True
        else:
            raise ValueError(f"Unsupported policy_algo '{self.policy_algo}'")

        cfg.set_policy_strategy(self.policy_algo, pol_cfg)
        cfg.policy_prototype_bases = vals['prototype_bases']

        if confidence > 1 :
            if threshold_type == 'chi2':
                confidence = confidence / 100.0

        self.exp_id += f"{self.distance_type}"
        if self.distance_type != 'maha':
            # if distance_type is in 'cossim', 'euclidean'
            # force percentile threshold
            threshold_type = 'percentile'
            if confidence < 1 :
                confidence *= 100.0
                print(f"[INFO] LazySIConfig: confidence is updated to percentile. {confidence}")
        # elif self.distance_type == 'maha' :
        #     if confidence > 1 :
        #         confidence = confidence / 100.0

        if threshold_type == 'chi2':
            if confidence > 1 :
                confidence = confidence / 100.0
        elif threshold_type == 'percentile':
            if confidence < 1 :
                confidence *= 100.0
        
        print(f"[INFO] LazySIConfig: distance_type is {self.distance_type}.")
        print(f"[INFO] LazySIConfig: threshold_type is {threshold_type}.")
        print(f"[INFO] LazySIConfig: confidence is {confidence}.")

        cfg.confidence_interval = confidence
        cfg.threshold_type = threshold_type
        cfg.distance_type = self.distance_type


        print(cfg)
        print("[INFO] LazySIConfig default settings.")
        
if __name__ == "__main__":
    lazysi = LazySIConfig(
        scenario_config=kitchen_scenario(),
        decoder_config=DEFAULT_DECODER_CONFIG,
        interface_config=DEFAULT_LAZYSI_INTERFACE_CONFIG,
        policy_config=DEFAULT_POLICY_CONFIG,
        exp_id="test",
        lifelong_algo="buds_append4/b10/ptgm/s20g40b4",
        seed=0,
    )
   
