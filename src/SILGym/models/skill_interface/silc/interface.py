"""
SIL-C Interface - Refactored main interface class.

This module contains the refactored LazySIInterface with improved modularity,
type hints, and separation of concerns.
"""

from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np

from SILGym.models.skill_interface.base import BaseInterface
from .core import (
    PrototypeConfig, PrototypeFactory, SkillPrototype, PolicyPrototype
)
from .clustering import (
    ClusteringStrategyFactory, SemanticClustering, BaseClusteringConfig
)
from SILGym.utils.logger import get_logger


# ==============================
# Configuration
# ==============================

@dataclass
class SILCInterfaceConfig:
    """Configuration for SIL-C Interface."""
    # Decoder configuration
    decoder_algo: str = "ptgm"
    decoder_algo_config: Any = None
    skill_prototype_bases: int = 5
    
    # Policy configuration
    policy_algo: str = "ptgm"
    policy_algo_config: Any = None
    subtask_prototype_bases: int = 5
    
    # Prototype configuration
    confidence_interval: float = 0.99
    threshold_type: str = "chi2"
    distance_type: str = "maha"
    
    # Interface behavior
    force_static: bool = False
    
    def __post_init__(self):
        """Initialize clustering strategies after dataclass initialization."""
        if self.decoder_algo_config is not None:
            self.decoder_entry_generator = ClusteringStrategyFactory.create(
                self.decoder_algo, self.decoder_algo_config
            )
        if self.policy_algo_config is not None:
            self.policy_entry_generator = ClusteringStrategyFactory.create(
                self.policy_algo, self.policy_algo_config
            )
    
    def set_decoder_strategy(self, algo: str, config: Any):
        """Set the decoder clustering strategy - for LazySI compatibility."""
        self.decoder_algo = algo
        self.decoder_algo_config = config
        self.decoder_entry_generator = ClusteringStrategyFactory.create(algo, config)
    
    def set_policy_strategy(self, algo: str, config: Any):
        """Set the policy clustering strategy - for LazySI compatibility."""
        self.policy_algo = algo
        self.policy_algo_config = config
        self.policy_entry_generator = ClusteringStrategyFactory.create(algo, config)


# ==============================
# Data Management
# ==============================

class DataLoaderManager:
    """Manages dataloader operations and transformations."""
    
    @staticmethod
    def init_entry_fields(dataloader: Any) -> Any:
        """Initialize required fields in the dataloader."""
        stacked_data = dataloader.stacked_data
        T = len(stacked_data['observations'])
        obs_dim = stacked_data['observations'].shape[1]
        
        # Initialize fields if not present
        fields_to_init = {
            'entry': (np.zeros((T,), dtype=np.int32), None),
            'skill_id': (np.zeros((T,), dtype=np.int32), None),
            'skill_aux': (np.zeros((T, obs_dim), dtype=np.float32), None),
            'decoder_id': (np.zeros((T,), dtype=np.int32), None),
            'subgoal': (np.zeros((T, obs_dim), dtype=np.float32), None),
        }
        
        for field, (default_value, _) in fields_to_init.items():
            if field not in stacked_data:
                stacked_data[field] = default_value
        
        return dataloader
    
    @staticmethod
    def augment_observations(dataloader: Any) -> Any:
        """Augment observations with skill auxiliary information."""
        stacked_data = dataloader.stacked_data
        stacked_data['orig_obs'] = stacked_data['observations'].copy()
        stacked_data['observations'] = np.concatenate(
            (stacked_data['observations'], stacked_data['skill_aux']), axis=-1
        )
        return dataloader
    
    @staticmethod
    def rollback_observations(dataloader: Any) -> Any:
        """Rollback observations to original state."""
        if 'orig_obs' not in dataloader.stacked_data:
            return dataloader
        dataloader.stacked_data['observations'] = dataloader.stacked_data['orig_obs']
        del dataloader.stacked_data['orig_obs']
        return dataloader


# ==============================
# Skill Management
# ==============================

class SkillManager:
    """Manages skill entries and prototypes."""
    
    def __init__(self, config: SILCInterfaceConfig):
        self.config = config
        self.entry_skill_map: Dict[int, SkillPrototype] = {}
        self.subtask_prototypes: Dict[int, PolicyPrototype] = {}
        self.decoder_id = 0
        self.prototype_factory = PrototypeFactory(
            PrototypeConfig(
                distance_type=config.distance_type,
                threshold_type=config.threshold_type,
                confidence_interval=config.confidence_interval,
                num_bases=config.skill_prototype_bases
            )
        )
        self.logger = get_logger(__name__)
    
    @property
    def num_skills(self) -> int:
        """Return the number of skills."""
        return len(self.entry_skill_map)
    
    def create_skill_entries(
        self,
        dataloader: Any,
        labels: np.ndarray,
        centroid_map: Dict[int, np.ndarray],
        extra: Dict[str, Any]
    ) -> Any:
        """Create skill entries from clustering results."""
        data = dataloader.stacked_data
        obs = np.array(data['observations'])
        T = len(obs)
        
        # Initialize subgoals array
        data['subgoals'] = np.zeros_like(obs)
        for sg, t in zip(extra['subgoals'], extra['timesteps']):
            data['subgoals'][t] = sg
        
        # Create global skill mapping
        base = self.num_skills
        appended_skill_num = len(centroid_map)
        global_map = {loc: base + loc for loc in range(appended_skill_num)}
        
        # Handle semantic clustering special case
        is_semantic = isinstance(self.config.decoder_entry_generator, SemanticClustering)
        if is_semantic:
            data['skill_aux'] = np.array(extra['semantic_skill_aux'])
        else:
            data['skill_aux'] = np.zeros_like(obs)
        
        # Assign per-timestep values
        for t, loc in enumerate(labels):
            if loc < 0:
                continue
            gid = global_map[loc]
            data['skill_id'][t] = gid
            data['entry'][t] = gid
            if not is_semantic:
                data['skill_aux'][t] = centroid_map[loc]
        
        # Create skill prototypes
        prototype_config = PrototypeConfig(
            distance_type=self.config.distance_type,
            threshold_type=self.config.threshold_type,
            confidence_interval=self.config.confidence_interval,
            num_bases=self.config.skill_prototype_bases
        )
        
        for loc in range(appended_skill_num):
            gid = global_map[loc]
            if gid not in self.entry_skill_map:
                idxs = np.where(data['entry'] == gid)[0]
                
                if is_semantic:
                    skill_aux_vec = data['skill_aux'][idxs[0]]
                    self.logger.info(
                        f"Skill {gid} centroid: {np.unique(data['skills'][idxs])}, {skill_aux_vec.shape}"
                    )
                else:
                    skill_aux_vec = centroid_map[loc]
                
                # Create skill prototype
                self.entry_skill_map[gid] = SkillPrototype.from_data(
                    skill_id=gid,
                    decoder_id=self.decoder_id,
                    skill_aux=skill_aux_vec,
                    state_data=self.prototype_factory.create_prototypes_from_data(obs[idxs]),
                    action_data=self.prototype_factory.create_prototypes_from_data(
                        np.array(data['actions'])[idxs]
                    ),
                    subgoal_data=self.prototype_factory.create_prototypes_from_data(
                        data['subgoals'][idxs]
                    ),
                    data_count=len(idxs),
                    config=prototype_config
                )
        
        data['decoder_id'] = np.full(T, self.decoder_id, dtype=int)
        self.decoder_id += 1
        return dataloader
    
    def create_subtask_prototypes(self, dataloader: Any) -> Dict[int, PolicyPrototype]:
        """Create policy prototypes from dataloader."""
        # Perform clustering
        labels, centroid_map, extra = self.config.policy_entry_generator.cluster(dataloader)
        
        # Attach subgoal assignments
        data = dataloader.stacked_data
        obs = data['observations']
        T = len(obs)
        data['subgoals_policy'] = np.zeros_like(obs)
        for sg, t in zip(extra['subgoals'], extra['timesteps']):
            data['subgoals_policy'][t] = sg
        
        # Build prototypes
        subtask_prototypes = {}
        base_id = 0
        num_clusters = len(centroid_map)
        
        prototype_config = PrototypeConfig(
            distance_type=self.config.distance_type,
            threshold_type=self.config.threshold_type,
            confidence_interval=self.config.confidence_interval,
            num_bases=self.config.subtask_prototype_bases
        )
        
        for cluster_idx in range(num_clusters):
            indices = np.where(labels == cluster_idx)[0]
            if len(indices) < self.config.subtask_prototype_bases:
                continue
            
            # Create prototypes
            if self.config.policy_algo == 'instance':
                state_data = self.prototype_factory.create_prototypes_from_data(
                    obs[indices], instance_based=True
                )
            else:
                state_data = self.prototype_factory.create_prototypes_from_data(
                    obs[indices], num_bases=self.config.subtask_prototype_bases
                )
            
            proto = PolicyPrototype.from_data(
                prototype_id=base_id,
                subgoal=centroid_map[cluster_idx],
                state_data=state_data,
                data_count=len(indices),
                config=prototype_config
            )
            subtask_prototypes[base_id] = proto
            base_id += 1
        
        self.logger.info(f"Created {len(subtask_prototypes)} policy prototypes.")
        self.subtask_prototypes = subtask_prototypes
        return subtask_prototypes


# ==============================
# Main Interface
# ==============================

class SILCInterface(BaseInterface):
    """
    Refactored Skill Incremental Learning with Clustering Interface.
    
    This interface provides:
    - Skill discovery through clustering
    - Dynamic skill matching based on state and subgoal prototypes
    - Support for multiple clustering algorithms
    """
    
    def __init__(self, config: Optional[SILCInterfaceConfig] = None):
        super().__init__()
        self.config = config or SILCInterfaceConfig()
        self.logger = get_logger(__name__)
        
        # Initialize managers
        self.dataloader_manager = DataLoaderManager()
        self.skill_manager = SkillManager(self.config)
        
        # Interface state
        self.debug = False
        self.candidates = 1
        
        self._log_configuration()
    
    def _log_configuration(self):
        """Log the current configuration."""
        self.logger.info(f"SIL-C Interface initialized with:")
        self.logger.info(f"  Decoder: {self.config.decoder_algo}")
        self.logger.info(f"  Policy: {self.config.policy_algo}")
        self.logger.info(f"  Distance: {self.config.distance_type}")
        self.logger.info(f"  Threshold: {self.config.threshold_type}")
        self.logger.info(f"  Confidence: {self.config.confidence_interval}")
    
    @property
    def num_skills(self) -> int:
        """Return the number of skills."""
        return self.skill_manager.num_skills
    
    # ==============================
    # Dataloader Operations
    # ==============================
    
    def update_interface(self, dataloader: Any) -> Any:
        """Update the interface with new data."""
        dataloader = self.dataloader_manager.init_entry_fields(dataloader)
        dataloader = self._map_entries(dataloader)
        return self.dataloader_manager.augment_observations(dataloader)
    
    def rollback_dataloader(self, dataloader: Any) -> Any:
        """Rollback dataloader to original state."""
        return self.dataloader_manager.rollback_observations(dataloader)
    
    # ==============================
    # Skill Mapping
    # ==============================
    
    def _map_entries(self, dataloader: Any) -> Any:
        """Map dataloader entries to skills."""
        # Perform clustering
        labels, centroid_map, extra = self.config.decoder_entry_generator.cluster(dataloader)
        
        # Create skill entries
        return self.skill_manager.create_skill_entries(
            dataloader, labels, centroid_map, extra
        )
    
    def create_subtask_prototype(self, dataloader: Any) -> Dict[int, PolicyPrototype]:
        """Create policy prototypes from dataloader."""
        return self.skill_manager.create_subtask_prototypes(dataloader)
    
    def update_subtask_prototype(self, subtask_prototype: Optional[Dict[int, PolicyPrototype]]):
        """Update policy prototypes."""
        if subtask_prototype is None:
            self.logger.warning("No policy prototype to update; None detected.")
            return
        self.skill_manager.subtask_prototypes = deepcopy(subtask_prototype)
    
    # ==============================
    # Forward Pass
    # ==============================
    
    def forward(
        self,
        entry: np.ndarray,
        current_state: np.ndarray,
        static: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Forward pass for skill selection.
        
        Args:
            entry: Skill IDs to query
            current_state: Current state observations
            static: If True, use static matching only
            
        Returns:
            ((skill_ids, decoder_ids), skill_aux)
        """
        # Prepare inputs
        entry = np.array(entry, dtype=np.int32)
        if entry.ndim > 1 and entry.shape[-1] == 1:
            entry = np.squeeze(entry, axis=-1)
        
        if current_state.ndim == 1:
            current_state = np.expand_dims(current_state, axis=0)
        
        B, obs_dim = current_state.shape
        
        # Initialize outputs
        out_skill_ids = np.zeros((B,), dtype=np.int32)
        out_decoder_ids = np.zeros((B,), dtype=np.int32)
        
        is_semantic = isinstance(self.config.decoder_entry_generator, SemanticClustering)
        if is_semantic:
            out_skill_aux = np.zeros((B, 512), dtype=np.float32)  # TODO: make this configurable
        else:
            out_skill_aux = np.zeros((B, obs_dim), dtype=np.float32)
        
        # Process each sample
        for i in range(B):
            skill_id, decoder_id, skill_aux = self._process_single_sample(
                entry[i], current_state[i], static
            )
            out_skill_ids[i] = skill_id
            out_decoder_ids[i] = decoder_id
            out_skill_aux[i] = skill_aux
        
        return (out_skill_ids, out_decoder_ids), out_skill_aux
    
    def _process_single_sample(
        self,
        entry_id: int,
        current_state: np.ndarray,
        static: bool
    ) -> Tuple[int, int, np.ndarray]:
        """Process a single sample for skill selection."""
        # Validate entry
        if entry_id not in self.skill_manager.entry_skill_map:
            entry_id = int(np.random.randint(0, self.num_skills))
        
        entry_obj = self.skill_manager.entry_skill_map[entry_id]
        
        # Static matching
        if static or self.config.force_static:
            return entry_obj.skill_id, entry_obj.decoder_id, entry_obj.skill_aux
        
        # Dynamic matching
        return self._dynamic_matching(entry_id, current_state, entry_obj)
    
    def _dynamic_matching(
        self,
        orig_entry: int,
        current_state: np.ndarray,
        entry_obj: SkillPrototype
    ) -> Tuple[int, int, np.ndarray]:
        """Perform dynamic skill matching."""
        # Find candidate policy prototypes
        candidate_list = []
        for pid, policy_proto in self.skill_manager.subtask_prototypes.items():
            valid, dist = policy_proto.state_prototype.validate(current_state)
            candidate_list.append((pid, dist, valid))
        
        # Select top candidates
        candidate_list.sort(key=lambda x: x[1])
        top_candidates = candidate_list[:self.candidates]
        top_candidates = [(pid, d, v) for pid, d, v in top_candidates if v]
        
        if not top_candidates:
            top_candidates = candidate_list[:self.candidates]
        
        if not top_candidates:
            return entry_obj.skill_id, entry_obj.decoder_id, entry_obj.skill_aux
        
        # Check if current entry's subgoal matches any candidate
        for pid, _, _ in top_candidates:
            policy_candidate = self.skill_manager.subtask_prototypes[pid]
            subgoal_valid, _ = entry_obj.subgoal_prototype.validate(
                policy_candidate.subgoal
            )
            if subgoal_valid:
                return entry_obj.skill_id, entry_obj.decoder_id, entry_obj.skill_aux
        
        # Find best matching skill
        skill_candidates = self._find_skill_candidates(top_candidates)
        return self._select_best_skill(skill_candidates, current_state, entry_obj)
    
    def _find_skill_candidates(
        self,
        top_candidates: List[Tuple[int, float, bool]]
    ) -> List[int]:
        """Find candidate skills for the given policy prototypes."""
        skill_candidate_ids = []
        
        for pid, _, _ in top_candidates:
            policy_candidate = self.skill_manager.subtask_prototypes[pid]
            candidate_subgoal = policy_candidate.subgoal
            
            # Find skills that match this subgoal
            sub_skill_candidates = []
            best_id = -1
            best_dist = np.inf
            
            for eid, entry_obj in self.skill_manager.entry_skill_map.items():
                valid, dist = entry_obj.subgoal_prototype.validate(candidate_subgoal)
                if valid:
                    sub_skill_candidates.append(eid)
                if dist < best_dist:
                    best_dist = dist
                    best_id = eid
            
            if sub_skill_candidates:
                skill_candidate_ids.extend(sub_skill_candidates)
            else:
                skill_candidate_ids.append(best_id)
        
        return list(set(skill_candidate_ids))
    
    def _select_best_skill(
        self,
        skill_candidates: List[int],
        current_state: np.ndarray,
        default_entry: SkillPrototype
    ) -> Tuple[int, int, np.ndarray]:
        """Select the best skill from candidates."""
        best_total = np.inf
        best_skill_id = default_entry.skill_id
        best_decoder_id = default_entry.decoder_id
        best_aux = default_entry.skill_aux
        
        for skill_id in skill_candidates:
            if skill_id not in self.skill_manager.entry_skill_map:
                continue
                
            candidate_entry = self.skill_manager.entry_skill_map[skill_id]
            distances = candidate_entry.state_prototype.compute_distances(current_state)
            total_distance = np.min(distances)
            
            if total_distance < best_total:
                best_total = total_distance
                best_skill_id = candidate_entry.skill_id
                best_decoder_id = candidate_entry.decoder_id
                best_aux = candidate_entry.skill_aux
        
        if self.debug:
            self.logger.debug(
                f"Dynamic matching: {default_entry.skill_id} -> {best_skill_id}, "
                f"distance: {best_total:.4f}"
            )
        
        return best_skill_id, best_decoder_id, best_aux