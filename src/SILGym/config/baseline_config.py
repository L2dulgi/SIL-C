"""
Baseline configuration classes for various skill learning algorithms.

This module provides configuration classes for different baseline algorithms
including BUDS, PTGM, IsCiL, Imanip, and LazySI.
"""

import re
import copy
import warnings
from typing import Dict, Optional, Tuple, Type, Any
from abc import ABC, abstractmethod
from SILGym.utils.logger import get_logger

# Import necessary configuration classes and functions
from SILGym.config.skill_stream_config import SkillStreamConfig
from SILGym.config.experiment_config import (
    SkillExperimentConfig,
    DEFAULT_POLICY_CONFIG,
    DEFAULT_DECODER_CONFIG,
    DEFAULT_FQL_DECODER_CONFIG,
    DECODER_CONFIG_MAP,
    DEFAULT_PTGM_INTERFACE_CONFIG,
    DEFAULT_IMANIP_INTERFACE_CONFIG,
    DEFAULT_ISCIL_INTERFACE_CONFIG,
    DEFAULT_BUDS_INTERFACE_CONFIG,
    DEFAULT_LAZYSI_INTERFACE_CONFIG,
    DEFAULT_SILC_INTERFACE_CONFIG
)
from SILGym.config.kitchen_scenario import kitchen_scenario
from SILGym.models.skill_decoder.appender import ModelAppender, AppendConfig
from SILGym.dataset.dataloader import DataloaderMixer, few_frac_shot_hook
from SILGym.models.agent.base import LazySIAgent, LazySIZeroAgent
from SILGym.models.task_policy.mlp_base import SILCHighLevelPolicy

# Interface config type imports
from SILGym.models.skill_interface.buds import BUDSInterfaceConfig
from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
from SILGym.models.skill_interface.lazySI import SemanticInterfaceConfig, InstanceRetrievalConfig


# ============================================================================
# Utility Functions for Configuration Parsing
# ============================================================================

def parse_experience_replay_config(suffix: str) -> float:
    """
    Parse experience replay configuration from suffix.
    
    Args:
        suffix: String suffix after 'er', e.g., '20', '20%', ''
        
    Returns:
        Buffer keep ratio as float (0.0 to 1.0)
    """
    suffix = suffix.rstrip("%")
    if suffix == "":
        percent_value = 10  # Default to 10%
    else:
        if not suffix.isdigit():
            raise ValueError("Invalid buffer_keep_ratio value: must be an integer percentage")
        percent_value = int(suffix)
    
    if not (0 <= percent_value <= 100):
        raise ValueError("Invalid buffer_keep_ratio value: must be between 0 and 100 percent")
    
    return percent_value / 100.0


def parse_append_config(suffix: str, default_lora_dim: int = 4) -> AppendConfig:
    """
    Parse append configuration from suffix.
    
    Args:
        suffix: String suffix after 'append', e.g., '4', '16'
        default_lora_dim: Default LoRA dimension if not specified
        
    Returns:
        AppendConfig instance
    """
    lora_dim = int(suffix) if suffix.isdigit() else default_lora_dim
    return AppendConfig(lora_dim=lora_dim, pool_length=10)


def parse_ptgm_config(cfg_str: str) -> Dict[str, int]:
    """Parse PTGM configuration string."""
    m = re.match(r's(?P<cluster>\d+)(?:g(?P<goal>\d+))?(?:b(?P<bases>\d+))?$', cfg_str)
    if not m:
        raise ValueError(f"Invalid PTGM config '{cfg_str}'")
    return {
        'cluster_num': int(m.group('cluster')),
        'goal_offset': int(m.group('goal') or 0),
        'prototype_bases': int(m.group('bases') or 0)
    }


def parse_bases_config(cfg_str: str) -> Dict[str, Any]:
    """Parse bases configuration string."""
    m = re.match(r'(?:g(?P<goal>\d+))?(?:b(?P<bases>\d+))?$', cfg_str)
    if not m:
        raise ValueError(f"Invalid bases config '{cfg_str}'")

    return {
        'window_size': 5,
        'min_length': 30,
        'target_num_segments': 10,
        'max_k': 10,
        'goal_offset': int(m.group('goal') or 20),
        'prototype_bases': int(m.group('bases') or 1),
        'verbose': False
    }


def get_decoder_config(decoder_type: str = "ddpm") -> dict:
    """
    Get decoder configuration based on decoder type.

    Args:
        decoder_type: Type of decoder ('ddpm', 'diffusion', 'fql', 'flow')

    Returns:
        Decoder configuration dictionary
    """
    decoder_type = decoder_type.lower()
    if decoder_type not in DECODER_CONFIG_MAP:
        logger = get_logger(__name__)
        logger.warning(f"Unknown decoder type '{decoder_type}', using default 'ddpm'")
        decoder_type = "ddpm"
    return DECODER_CONFIG_MAP[decoder_type]


# ============================================================================
# Base Configuration Class with Common Functionality
# ============================================================================

class BaseAlgorithmConfig(SkillExperimentConfig):
    """Base class for algorithm configurations with common functionality."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict,
        interface_config: dict,
        policy_config: dict,
        exp_id: str,
        lifelong_algo: str,
        seed: int,
        algo_type: str,
        decoder_type: str = 'ddpm',
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        super().__init__(
            scenario_config=scenario_config,
            interface_config=interface_config,
            decoder_config=decoder_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            decoder_type=decoder_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )
        self.algo_type = algo_type
        self._validate_algo_type()
        
    def _setup_lifelong_algo_base(self, lifelong_algo: str, default_lora_dim: int = 4) -> None:
        """
        Base implementation for setting up lifelong learning algorithms.

        Args:
            lifelong_algo: Lifelong learning algorithm specification
            default_lora_dim: Default LoRA dimension for append algorithms
        """
        if lifelong_algo == "ft":
            self.is_appendable = False

        elif lifelong_algo == "ftscratch":
            self.is_appendable = False
            self.reset_decoder_each_phase = True

        elif lifelong_algo.startswith("er"):
            self.is_appendable = False
            suffix = lifelong_algo[len("er"):]
            self.buffer_keep_ratio = parse_experience_replay_config(suffix)

        elif lifelong_algo.startswith("append"):
            self.is_appendable = True
            # self.appender_cls = ModelAppender
            suffix = lifelong_algo[len("append"):]
            self.appender_config = parse_append_config(suffix, default_lora_dim)

        else:
            raise ValueError(f"Unknown lifelong_algo spec '{lifelong_algo}'")


# ============================================================================
# Specific Algorithm Configuration Classes
# ============================================================================

class BUDSConfig(BaseAlgorithmConfig):
    """Configuration for the BUDS (Behavior Unsupervised Discovery of Skills) algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = None,
        decoder_type: str = "ddpm",
        interface_config: dict = DEFAULT_BUDS_INTERFACE_CONFIG,
        policy_config: dict = None,
        exp_id: str = "DEF",
        lifelong_algo: str = "",
        seed: int = 0,
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        # Get decoder config based on type if not explicitly provided
        if decoder_config is None:
            decoder_config = get_decoder_config(decoder_type)
        super().__init__(
            scenario_config=scenario_config,
            decoder_config=decoder_config,
            interface_config=interface_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            algo_type='buds',
            decoder_type=decoder_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )

        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()

    def _setup_lifelong_algo(self):
        """Setup lifelong learning algorithm for BUDS."""
        self._setup_lifelong_algo_base(self.lifelong_algo)


class PTGMConfig(BaseAlgorithmConfig):
    """Configuration for the PTGM (Prototype-based Task Generation and Matching) algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = None,
        decoder_type: str = "ddpm",
        interface_config: dict = None,
        policy_config: dict = None,
        exp_id: str = "DEF",
        lifelong_algo: str = "",
        seed: int = 0,
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        # Get decoder config based on type if not explicitly provided
        if decoder_config is None:
            decoder_config = get_decoder_config(decoder_type)

        # Create a deep copy of interface_config to avoid shared mutable state
        if interface_config is None:
            import copy
            interface_config = copy.deepcopy(DEFAULT_PTGM_INTERFACE_CONFIG)

        super().__init__(
            scenario_config=scenario_config,
            decoder_config=decoder_config,
            interface_config=interface_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            algo_type='ptgm',
            decoder_type=decoder_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )
        
        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()
        
    def _update_decoder_config(self, decoder_config, algo_type):
        """Update decoder config for PTGM."""
        if self.sync_type == "joint":
            self.interface_config["interface_kwargs"]["ptgm_config"].cluster_num = 100
        return super()._update_decoder_config(decoder_config, algo_type)
    
    def _setup_lifelong_algo(self):
        """
        Parse lifelong_algo with optional 'ptgmplus', 'birch', 'noTsne', 'umap', 's<decoder>' and/or 'g<groups>'
        parameters for PTGM.

        Format: [ptgmplus[_birch]/][umap|notsne][s<cluster_num>][g<goal_offset>][b<bases>][/<lifelong_algo>]

        Clustering methods by embedding:
        - T-SNE (default): Uses KMeans clustering
        - UMAP: Uses HDBSCAN clustering (automatically enabled)
        - No embedding (notsne): Uses KMeans clustering

        Examples:
        - "s20b4/append4" → PTGM with T-SNE + KMeans, 20 clusters, lifelong=append4
        - "umaps20b4/append4" → PTGM with UMAP + HDBSCAN, 20 clusters, lifelong=append4
        - "notsnes20b4/append4" → PTGM without embedding + KMeans, 20 clusters, lifelong=append4
        - "ptgmplus/s20b4/append4" → PTGM+ with MiniBatchKMeans, 20 clusters, lifelong=append4
        - "ptgmplus_birch/s20b4/append4" → PTGM+ with BIRCH, 20 clusters, lifelong=append4
        - "ptgmplus/umaps20g40b4/ft" → PTGM+ with UMAP + HDBSCAN, 20 clusters, 40 goal_offset, lifelong=ft
        - "umap/ft" → PTGM with UMAP + HDBSCAN, default clusters, lifelong=ft
        """
        spec = self.lifelong_algo.strip().lower()
        self.tsne = True  # Default tsne on (deprecated, use embedding_method)
        embedding_method = 'tsne'  # Default embedding method
        use_ptgm_plus = False  # Default to original PTGM
        precluster_method = 'minibatch_kmeans'  # Default pre-clustering method

        # Check for 'ptgmplus_birch' prefix (BIRCH variant)
        if spec.startswith("ptgmplus_birch"):
            use_ptgm_plus = True
            precluster_method = 'birch'
            spec = spec[len("ptgmplus_birch"):]
            spec = spec.lstrip("/")
            get_logger(__name__).info("PTGM+ algorithm enabled with BIRCH pre-clustering")
        # Check for 'ptgmplus' prefix (MiniBatchKMeans variant)
        elif spec.startswith("ptgmplus"):
            use_ptgm_plus = True
            precluster_method = 'minibatch_kmeans'
            # Extract what comes after "ptgmplus"
            remaining = spec[len("ptgmplus"):]

            # Validate: if there's an underscore, it must be exactly "_birch"
            if remaining.startswith("_") and not remaining.startswith("_birch"):
                # Extract the invalid variant name
                invalid_suffix = remaining.split("/")[0]  # Get text before first "/"
                raise ValueError(
                    f"Invalid PTGM+ variant: 'ptgmplus{invalid_suffix}'\n"
                    f"Valid options are:\n"
                    f"  - 'ptgmplus' (uses MiniBatchKMeans pre-clustering)\n"
                    f"  - 'ptgmplus_birch' (uses BIRCH pre-clustering)\n"
                    f"Example: 'ptgmplus/s20b4/append4' or 'ptgmplus_birch/s20b4/append4'"
                )

            spec = remaining.lstrip("/")  # Remove leading slash
            get_logger(__name__).info("PTGM+ algorithm enabled with MiniBatchKMeans pre-clustering")

        # Split by "/" to separate config from lifelong_algo
        # Format: [config_part]/[lifelong_part]
        parts = spec.split("/", 1)
        config_part = parts[0] if parts else ""
        lifelong_part = parts[1] if len(parts) > 1 else None  # Will be determined later
        original_config_part = config_part  # Save original for fallback

        # Parse config part (umap, notsne, s<num>, g<num>, b<num>)
        # Extract optional 'umap' prefix
        umap_match = re.match(r"umap", config_part)
        if umap_match:
            embedding_method = 'umap'
            self.tsne = True  # Keep for backward compatibility
            config_part = config_part[umap_match.end():]
            get_logger(__name__).info("UMAP embedding enabled")

        # Extract optional 'notsne' prefix (mutually exclusive with 'umap')
        tsne_match = re.match(r"notsne", config_part)
        if tsne_match:
            if embedding_method == 'umap':
                raise ValueError("Cannot specify both 'umap' and 'notsne' in lifelong_algo")
            embedding_method = 'none'
            self.tsne = False
            config_part = config_part[tsne_match.end():]
            get_logger(__name__).info("No embedding (notsne) mode enabled")

        # Extract optional 's<decoder>' and/or 'g<groups>' and/or 'b<bases>'
        s_match = re.match(r"s(?P<decoder>\d+)", config_part)
        if s_match:
            self.cluster_num = int(s_match.group("decoder"))
            config_part = config_part[s_match.end():]

        g_match = re.match(r"g(?P<groups>\d+)", config_part)
        if g_match:
            self.groups_num = int(g_match.group("groups"))
            config_part = config_part[g_match.end():]

        # Note: 'b<bases>' is parsed but not used in PTGM (kept for compatibility)
        b_match = re.match(r"b(?P<bases>\d+)", config_part)
        if b_match:
            config_part = config_part[b_match.end():]

        # If no lifelong_part was specified via "/" separator, check if remaining config_part
        # contains a lifelong algo spec (ft, append, er, etc.)
        if lifelong_part is None:
            # If config_part still has content after parsing config params,
            # treat the remaining as lifelong_algo
            if config_part:
                lifelong_part = config_part
            # If all of original_config_part was consumed by config params,
            # check if it looks like a lifelong algo spec
            elif original_config_part and not any([
                original_config_part.startswith('s'),
                original_config_part.startswith('g'),
                original_config_part.startswith('b'),
                original_config_part.startswith('umap'),
                original_config_part.startswith('notsne')
            ]):
                # Original spec doesn't look like config params, treat as lifelong algo
                lifelong_part = original_config_part
            else:
                # Default to 'ft'
                lifelong_part = "ft"

        # Apply base lifelong algo setup with the lifelong part
        self._setup_lifelong_algo_base(lifelong_part)

        # Update interface config with parsed values
        cfg = self.interface_config["interface_kwargs"]["ptgm_config"]

        # Enable PTGM+ if requested
        if use_ptgm_plus:
            cfg.use_ptgm_plus = True
            cfg.sampling_ratio = 10.0  # Default sampling ratio
            cfg.precluster_method = precluster_method
            get_logger(__name__).info(f"PTGM+ configuration applied (sampling_ratio=10.0, precluster_method={precluster_method})")

        # Set embedding method
        cfg.embedding_method = embedding_method
        if embedding_method == 'none':
            cfg.tsne_dim = 0
            cfg.embedding_dim = 0
        elif embedding_method == 'umap':
            # Enable HDBSCAN clustering when UMAP is used
            cfg.use_hdbscan_with_umap = True
            get_logger(__name__).info(f"HDBSCAN clustering enabled for UMAP (min_cluster_size={cfg.hdbscan_min_cluster_size})")
        get_logger(__name__).info(f"Embedding method: {embedding_method}")

        if hasattr(self, "cluster_num"):
            cfg.cluster_num = self.cluster_num
            get_logger(__name__).info(f"global cluster_num set to {self.cluster_num}")
        if hasattr(self, "groups_num"):
            cfg.goal_offset = self.groups_num
            get_logger(__name__).info(f"global groups_num set to {self.groups_num}")

        # Backward compatibility: still set tsne_dim=0 if tsne is disabled
        if not self.tsne:
            cfg.tsne_dim = 0


class IsCiLConfig(BaseAlgorithmConfig):
    """Configuration for the IsCiL (Incremental Skill Continual Learning) algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = None,
        decoder_type: str = "ddpm",
        interface_config: dict = DEFAULT_ISCIL_INTERFACE_CONFIG,
        policy_config: dict = None,
        exp_id: str = "DEF",
        lifelong_algo: str = "",
        seed: int = 0,
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        # Get decoder config based on type if not explicitly provided
        if decoder_config is None:
            decoder_config = get_decoder_config(decoder_type)
        super().__init__(
            scenario_config=scenario_config,
            decoder_config=decoder_config,
            interface_config=interface_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            algo_type='iscil',
            decoder_type=decoder_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )
        
        # IsCiL specific settings
        self.is_appendable = True
        # self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(lora_dim=4, pool_length=8)
        
        # Semantic path settings
        self.semantic_path = self.interface_config['interface_kwargs']['semantic_emb_path']
        self.embed_dim = int(self.semantic_path.split('/')[-1].split('.')[0])
        self.decoder_config = self._update_decoder_config(decoder_config)
        
        self._setup_lifelong_algo()
        self._validate_model_path()
        
    def update_config_by_env(self):
        """Update configuration based on environment."""
        super().update_config_by_env()
        if self.environment == 'kitchen':
            pass
        elif self.environment == 'mmworld':
            get_logger(__name__).info("IsCiLConfig: mmworld environment.")
            self.interface_config['interface_kwargs']['semantic_emb_path'] = 'exp/instruction_embedding/mmworld/512.pkl'
        else:
            raise NotImplementedError("IsCiLConfig only supports kitchen and mmworld environments.")
    
    def _update_decoder_config(self, decoder_config):
        """Update decoder config with semantic embedding dimension."""
        current_shape = decoder_config['model_kwargs']['input_config']['cond']
        x = current_shape[2]  # (B, 1, F)
        new_shape = (1, 1, x + self.embed_dim)
        decoder_config['model_kwargs']['input_config']['cond'] = new_shape
        return decoder_config
    
    def _setup_lifelong_algo(self):
        """Setup lifelong algorithm for IsCiL."""
        bases_num = 50
        if self.lifelong_algo.startswith("bases"):
            suffix = self.lifelong_algo[len("bases"):]
            if suffix and suffix.isdigit():
                bases_num = int(suffix)
            else:
                if suffix:
                    raise ValueError("Invalid bases_num value: must be an integer")
        else:
            self.lifelong_algo = "bases50"  # Default setting
        
        self.interface_config['interface_kwargs']['config'].bases_num = bases_num
        get_logger(__name__).info(f"IsCiLConfig bases_num: {bases_num}")


class ImanipConfig(BaseAlgorithmConfig):
    """Configuration for the Imanip (Instruction Manipulation) algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = None,
        decoder_type: str = "ddpm",
        interface_config: dict = DEFAULT_IMANIP_INTERFACE_CONFIG,
        policy_config: dict = None,
        exp_id: str = "DEF",
        lifelong_algo: str = "tr",
        seed: int = 0,
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        # Get decoder config based on type if not explicitly provided
        if decoder_config is None:
            decoder_config = get_decoder_config(decoder_type)
        super().__init__(
            scenario_config=scenario_config,
            decoder_config=decoder_config,
            interface_config=interface_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            algo_type='imanip',
            decoder_type=decoder_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )
        
        # Imanip specific settings
        self.is_appendable = True
        # self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(lora_dim=16, pool_length=10)
        self.dataloader_mixer_cls = DataloaderMixer
        
        # Semantic path settings
        self.semantic_path = self.interface_config['interface_kwargs']['semantic_emb_path']
        self.embed_dim = int(self.semantic_path.split('/')[-1].split('.')[0])
        self.decoder_config = self._update_decoder_config(decoder_config)
        
        self._validate_model_path()
        self._setup_lifelong_algo()
        
    def update_config_by_env(self):
        """Update configuration based on environment."""
        super().update_config_by_env()
        if self.environment == 'kitchen':
            pass
        elif self.environment == 'mmworld':
            get_logger(__name__).info("ImanipConfig: mmworld environment.")
            self.interface_config['interface_kwargs']['semantic_emb_path'] = 'exp/instruction_embedding/mmworld/512.pkl'
        else:
            raise NotImplementedError("ImanipConfig only supports kitchen and mmworld environments.")
    
    def _update_decoder_config(self, decoder_config):
        """Update decoder config with semantic embedding dimension."""
        current_shape = decoder_config['model_kwargs']['input_config']['cond']
        x = current_shape[2]  # (B, 1, F)
        new_shape = (1, 1, x + self.embed_dim)
        decoder_config['model_kwargs']['input_config']['cond'] = new_shape
        return decoder_config
    
    def _setup_lifelong_algo(self):
        """Setup lifelong algorithm for Imanip."""
        if self.lifelong_algo.startswith("tr"):
            self.is_appendable = False
            suffix = self.lifelong_algo[len("tr"):]
            self.buffer_keep_ratio = parse_experience_replay_config(suffix)
            
            if self.buffer_keep_ratio == 0:
                get_logger(__name__).info("ImanipConfig: buffer_keep_ratio is 0. No buffer.")
                self.is_appendable = True
                
        elif self.lifelong_algo.startswith("append"):
            self.is_appendable = True
            # self.appender_cls = ModelAppender
            suffix = self.lifelong_algo[len("append"):]
            self.appender_config = parse_append_config(suffix, default_lora_dim=16)
        else:
            get_logger(__name__).warning("Imanip only supports tr (temporal replay) algorithm.")
            raise NotImplementedError("ImanipConfig only supports tr (temporal replay) algorithm.")


class LazySIConfig(BaseAlgorithmConfig):
    """Configuration for the LazySI (Lazy Skill Interface) algorithm."""

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = None,
        decoder_type: str = "ddpm",
        interface_config: dict = DEFAULT_LAZYSI_INTERFACE_CONFIG,
        policy_config: dict = None,
        exp_id: str = "DEF",
        lifelong_algo: str = "",
        seed: int = 0,
        distance_type: str = "maha",
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        # Get decoder config based on type if not explicitly provided
        if decoder_config is None:
            decoder_config = get_decoder_config(decoder_type)

        # Get policy config if not provided
        if policy_config is None:
            policy_config = copy.deepcopy(DEFAULT_POLICY_CONFIG)

        # Use policy with hook
        policy_config['model_cls'] = SILCHighLevelPolicy

        super().__init__(
            scenario_config=scenario_config,
            decoder_config=decoder_config,
            interface_config=interface_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            algo_type='lazysi',
            decoder_type=decoder_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )

        self.distance_type = distance_type
        self.algo_mode = None  # None or 'zero' for LazySIZeroAgent
        
        self.is_appendable = True
        # self.appender_cls = ModelAppender
        self.appender_config = AppendConfig(lora_dim=16, pool_length=8)
        
        self.decoder_config = self._update_decoder_config(decoder_config, self.algo_type)
        self._setup_lifelong_algo()
        self._validate_model_path()
        
    @property
    def agent_cls(self):
        """Get the appropriate agent class based on algo_mode."""
        if self.algo_mode == 'zero':
            get_logger(__name__).info("LazySIZeroAgent is used.")
            return LazySIZeroAgent
        else:
            get_logger(__name__).info("LazySIAgent is used.")
            return LazySIAgent
    
    def update_config_by_env(self):
        """Update configuration based on environment."""
        super().update_config_by_env()
        if self.environment == 'kitchen':
            self.semantic_path = 'exp/instruction_embedding/kitchen/512.pkl'
        elif self.environment == 'mmworld':
            get_logger(__name__).info("LazySIConfig: mmworld environment.")
            self.semantic_path = 'exp/instruction_embedding/mmworld/512.pkl'
        else:
            get_logger(__name__).info("LazySIConfig: libero environment.")
    
    def _update_decoder_config(self, decoder_config, algo_type):
        """Lazy update decoder config."""
        get_logger(__name__).info("LazySIConfig: decoder_config is lazy updated in _setup_lifelong_algo.")
        return decoder_config
    
    def _post_update_decoder_config(self, decoder_config, dec_algo_type):
        """Post-update decoder config based on decoder algorithm type."""
        if self.decoder_algo in ['ptgm', 'ptgmu', 'buds']:
            return super()._update_decoder_config(decoder_config, dec_algo_type)
        elif self.decoder_algo == "semantic":
            current_shape = decoder_config['model_kwargs']['input_config']['cond']
            x = current_shape[2]  # (B, 1, F)
            new_shape = (1, 1, x + 512)  # Hard coded semantic dimension
            decoder_config['model_kwargs']['input_config']['cond'] = new_shape
            return decoder_config
    
    def _setup_lifelong_algo(self):
        """
        Parse lifelong_algo for LazySI.
        
        Format: [algo_mode]/decoder_part/dec_conf/policy_algo/pol_conf
        
        Examples:
        - "ptgm/s20b4/ptgm/s20b4"
        - "few1/ptgm/s20b4/ptgm/s20b4"
        - "conf99_chi2/ptgm/s20b4/instance/g20b1"
        """
        spec = self.lifelong_algo.strip()
        parts = spec.split('/')
        
        if len(parts) == 5:
            algo_mode, decoder_part, dec_conf_str, policy_algo, pol_conf_str = parts
            self.algo_mode = algo_mode
        elif len(parts) == 4:
            decoder_part, dec_conf_str, policy_algo, pol_conf_str = parts
        else:
            raise ValueError(f"Invalid lifelong_algo format, expected 4 or 5 parts but got '{spec}'")
        
        # Parse algo mode settings
        self.shot = None
        self.frac = None
        confidence = 0.99
        threshold_type = 'chi2'
        
        if self.algo_mode is not None:
            if self.algo_mode.startswith("few"):
                # Parse few-shot settings
                m = re.match(r"few(\d+)(?:frac(\d+))?", self.algo_mode)
                if not m:
                    raise ValueError(f"Invalid algo_mode format '{self.algo_mode}'")
                self.shot = int(m.group(1))
                self.frac = int(m.group(2)) if m.group(2) else 1
                
                # Setup dataloader hooks
                if 'pre_process_hooks_kwargs' not in self.dataloader_kwargs_policy:
                    self.dataloader_kwargs_policy['pre_process_hooks_kwargs'] = []
                self.dataloader_kwargs_policy['pre_process_hooks_kwargs'].append(
                    (few_frac_shot_hook, {'shot': self.shot, 'frac': self.frac})
                )
                
            elif self.algo_mode.startswith("conf"):
                # Parse confidence settings
                if '_chi2' in self.algo_mode:
                    threshold_type = 'chi2'
                    conf_input = self.algo_mode.split('_')[0]
                elif '_percentile' in self.algo_mode:
                    threshold_type = 'percentile'
                    conf_input = self.algo_mode.split('_')[0]
                else:
                    conf_input = self.algo_mode
                
                m = re.match(r"conf(\d+)?", conf_input)
                confidence = int(m.group(1)) if m.group(1) else 99
                confidence = confidence / 100.0
            else:
                raise ValueError(f"Invalid algo_mode format '{self.algo_mode}'")
        
        # Parse decoder configuration
        sub = decoder_part.split('_')
        self.decoder_algo = sub[0]
        self.llalgo = sub[1] if len(sub) > 1 else None
        self.cluster_algo = sub[2] if len(sub) > 2 else None
        self.policy_algo = policy_algo
        self.policy_algo_config = pol_conf_str
        
        # Log configuration
        logger = get_logger(__name__)
        logger.info(f"decoder_algo: {self.decoder_algo}")
        logger.info(f"llalgo: {self.llalgo}")
        logger.info(f"cluster_algo: {self.cluster_algo}")
        logger.info(f"decoder_algo_config: {dec_conf_str}")
        logger.info(f"policy_algo: {self.policy_algo}")
        logger.info(f"policy_algo_config: {self.policy_algo_config}")
        
        # Setup lifelong learning algorithm
        if self.llalgo == 'ft' or self.llalgo is None:
            # Default to fine-tuning when llalgo is None or explicitly 'ft'
            self.is_appendable = False
        elif self.llalgo and self.llalgo.startswith('er'):
            self.is_appendable = False
            pct = self.llalgo[len('er'):].rstrip('%') or '10'
            self.buffer_keep_ratio = int(pct) / 100.0
        elif self.llalgo and (self.llalgo.startswith('append') or self.llalgo.startswith('expand')):
            self.is_appendable = True
            suf = re.sub(r'^(?:append|expand)', '', self.llalgo)
            dim = int(suf) if suf.isdigit() else 4
            # self.appender_cls = ModelAppender
            self.appender_config = AppendConfig(lora_dim=dim, pool_length=5)
        else:
            raise ValueError(f"Unknown llalgo spec '{self.llalgo}'")
        
        # Update decoder config
        self.decoder_config = self._post_update_decoder_config(self.decoder_config, self.decoder_algo)
        
        # Build interface configurations
        cfg = self.interface_config['interface_kwargs']['config']
        
        # Decoder interface config
        if self.decoder_algo in ['ptgm', 'ptgmu']:
            vals = parse_ptgm_config(dec_conf_str)
            dec_cfg = PTGMInterfaceConfig(
                cluster_num=vals['cluster_num'],
                goal_offset=vals['goal_offset'],
                tsne_perplexity=50,
                tsne_dim=2,
                embedding_method='umap' if self.decoder_algo == 'ptgmu' else 'tsne',
            )
        elif self.decoder_algo == 'buds':
            vals = parse_bases_config(dec_conf_str)
            dec_cfg = BUDSInterfaceConfig(**{k: v for k, v in vals.items() if k != 'prototype_bases'})
        elif self.decoder_algo == "semantic":
            vals = parse_bases_config(dec_conf_str)
            dec_cfg = SemanticInterfaceConfig(
                semantic_emb_path=self.semantic_path,
                goal_offset=vals['goal_offset'],
            )
        else:
            raise ValueError(f"Unsupported decoder_algo '{self.decoder_algo}'")
        
        cfg.set_decoder_strategy(self.decoder_algo, dec_cfg)
        cfg.skill_prototype_bases = vals['prototype_bases']
        
        # Policy interface config
        if self.policy_algo in ['ptgm', 'ptgmu']:
            vals = parse_ptgm_config(pol_conf_str)
            if self.frac is not None:
                vals['goal_offset'] = int(vals['goal_offset'] // self.frac)
            pol_cfg = PTGMInterfaceConfig(
                cluster_num=vals['cluster_num'],
                goal_offset=vals['goal_offset'],
                tsne_perplexity=50,
                tsne_dim=2,
                embedding_method='umap' if self.policy_algo == 'ptgmu' else 'tsne',
            )
        elif self.policy_algo == 'buds':
            vals = parse_bases_config(pol_conf_str)
            if self.frac is not None:
                vals['goal_offset'] = int(vals['goal_offset'] // self.frac)
            pol_cfg = BUDSInterfaceConfig(**{k: v for k, v in vals.items() if k != 'prototype_bases'})
        elif self.policy_algo == 'instance':
            vals = parse_bases_config(pol_conf_str)
            if self.frac is not None:
                vals['goal_offset'] = int(vals['goal_offset'] // self.frac)
            pol_cfg = InstanceRetrievalConfig(goal_offset=vals['goal_offset'])
        elif self.policy_algo == 'static':
            pol_cfg = PTGMInterfaceConfig(
                cluster_num=1,
                goal_offset=20,
                tsne_perplexity=1,
                tsne_dim=2,
            )
            self.policy_algo = 'ptgm'  # Static is implemented as PTGM
            cfg.force_static = True
        else:
            raise ValueError(f"Unsupported policy_algo '{self.policy_algo}'")
        
        cfg.set_policy_strategy(self.policy_algo, pol_cfg)
        cfg.subtask_prototype_bases = vals['prototype_bases']
        
        # Handle distance type and confidence settings
        self.exp_id += f"{self.distance_type}"
        
        if self.distance_type != 'maha':
            # Force percentile threshold for non-Mahalanobis distances
            threshold_type = 'percentile'
            if confidence < 1:
                confidence *= 100.0
                logger.info(f"LazySIConfig: confidence updated to percentile. {confidence}")
        
        # Normalize confidence values
        if threshold_type == 'chi2' and confidence > 1:
            confidence = confidence / 100.0
        elif threshold_type == 'percentile' and confidence < 1:
            confidence *= 100.0
        
        logger.info(f"LazySIConfig: distance_type is {self.distance_type}.")
        logger.info(f"LazySIConfig: threshold_type is {threshold_type}.")
        logger.info(f"LazySIConfig: confidence is {confidence}.")
        
        cfg.confidence_interval = confidence
        cfg.threshold_type = threshold_type
        cfg.distance_type = self.distance_type
        
        logger.info(f"Interface config: {cfg}")
        logger.info("LazySIConfig default settings.")


class SILCConfig(LazySIConfig):
    """
    Configuration for the SILC (Skill Incremental Learning with Clustering) algorithm.

    This is a refactored version of LazySI with improved modularity and architecture.
    It maintains full compatibility with LazySI configuration while using the new
    SILC interface implementation.
    """

    def __init__(
        self,
        scenario_config: SkillStreamConfig,
        decoder_config: dict = None,
        decoder_type: str = "ddpm",
        interface_config: dict = None,  # Will be set to DEFAULT_SILC_INTERFACE_CONFIG
        policy_config: dict = None,
        exp_id: str = "DEF",
        lifelong_algo: str = "",
        seed: int = 0,
        distance_type: str = "maha",
        action_chunk: int = 1,
        action_chunk_padding: str = 'repeat_last',
    ) -> None:
        # Get decoder config based on type if not explicitly provided
        if decoder_config is None:
            decoder_config = get_decoder_config(decoder_type)

        # Use SILC interface config if not provided
        if interface_config is None:
            from SILGym.config.experiment_config import DEFAULT_SILC_INTERFACE_CONFIG
            interface_config = DEFAULT_SILC_INTERFACE_CONFIG
            
        super().__init__(
            scenario_config=scenario_config,
            decoder_config=decoder_config,
            decoder_type=decoder_type,
            interface_config=interface_config,
            policy_config=policy_config,
            exp_id=exp_id,
            lifelong_algo=lifelong_algo,
            seed=seed,
            distance_type=distance_type,
            action_chunk=action_chunk,
            action_chunk_padding=action_chunk_padding,
        )

        # Override algo_type to 'silc'
        self.algo_type = 'silc'
        
        get_logger(__name__).info("SILCConfig initialized - using refactored SILC interface")
    
    @property
    def agent_cls(self):
        """Get the appropriate agent class based on algo_mode."""
        from SILGym.models.agent.base import SILCAgent, SILCZeroAgent
        
        if hasattr(self, 'algo_mode') and self.algo_mode == 'zero':
            get_logger(__name__).info("SILCZeroAgent is used.")
            return SILCZeroAgent
        else:
            get_logger(__name__).info("SILCAgent is used.")
            return SILCAgent


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test configuration creation
    lazysi = LazySIConfig(
        scenario_config=kitchen_scenario(),
        decoder_config=DEFAULT_DECODER_CONFIG,
        interface_config=DEFAULT_LAZYSI_INTERFACE_CONFIG,
        policy_config=DEFAULT_POLICY_CONFIG,
        exp_id="test",
        lifelong_algo="buds_append4/b10/ptgm/s20g40b4",
        seed=0,
    )