"""
Clustering strategies for SIL-C (Skill Incremental Learning with Clustering).

This module contains refactored clustering strategies with improved modularity
and type hints.
"""

from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import pickle
from SILGym.utils.cuml_wrapper import TSNE, KMeans
from abc import ABC, abstractmethod

from .core import ClusteringStrategy
from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
from SILGym.models.skill_interface.buds import BUDSInterfaceConfig, BUDSClusterGeneator
from SILGym.utils.logger import get_logger


# ==============================
# Base Configuration Classes
# ==============================

class BaseClusteringConfig:
    """Base configuration for clustering strategies."""
    def __init__(self, goal_offset: int = 20):
        self.goal_offset = goal_offset


class SemanticClusteringConfig(BaseClusteringConfig):
    """Configuration for semantic clustering."""
    def __init__(self, semantic_emb_path: str, goal_offset: int = 20):
        super().__init__(goal_offset)
        self.semantic_emb_path = semantic_emb_path


# ==============================
# Helper Functions
# ==============================

def split_by_trajectory(terminals: np.ndarray) -> List[Tuple[int, int]]:
    """Split data indices by trajectory boundaries."""
    trajs, start = [], 0
    for i, v in enumerate(terminals):
        if v == 1:
            trajs.append((start, i))
            start = i + 1
    return trajs


def extract_subgoals(
    observations: np.ndarray,
    terminals: np.ndarray,
    goal_offset: int
) -> Tuple[np.ndarray, List[int]]:
    """Extract subgoals from trajectories."""
    trajs = split_by_trajectory(terminals)
    all_subgoals, all_subgoal_ts = [], []
    
    for s, e in trajs:
        for t in range(s, e + 1):
            fut = t + goal_offset
            sg = observations[fut] if fut <= e else observations[e]
            all_subgoals.append(sg)
            all_subgoal_ts.append(t)
    
    return np.array(all_subgoals), all_subgoal_ts


# ==============================
# Clustering Strategies
# ==============================

class PTGMClustering(ClusteringStrategy):
    """
    PTGM (Prototype-based Temporal Goal Matching) clustering strategy.
    Uses t-SNE for dimensionality reduction followed by KMeans clustering.
    """
    
    def __init__(self, config: PTGMInterfaceConfig, random_state: int = 0):
        self.config = config
        self.random_state = random_state
        self.logger = get_logger(__name__)
    
    def cluster(self, dataloader: Any) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, Any]]:
        """Perform PTGM clustering on the dataloader."""
        data = dataloader.stacked_data
        observations = data['observations']
        terminals = data['terminals']
        T = len(observations)
        
        # Extract subgoals
        all_subgoals, all_subgoal_ts = extract_subgoals(
            observations, terminals, self.config.goal_offset
        )
        
        # Perform t-SNE dimensionality reduction
        self.logger.debug(f"Performing t-SNE with {self.config.tsne_dim} dimensions")
        tsne = TSNE(
            n_components=self.config.tsne_dim,
            perplexity=self.config.tsne_perplexity,
            random_state=self.random_state
        )
        emb = tsne.fit_transform(all_subgoals)
        
        # Cluster in t-SNE space
        self.logger.debug(f"Clustering into {self.config.cluster_num} clusters")
        km = KMeans(n_clusters=self.config.cluster_num, random_state=self.random_state)
        ids = km.fit_predict(emb)
        centers_tsne = km.cluster_centers_
        
        # Find representative subgoals for each cluster
        centroid_map = {}
        for c in range(self.config.cluster_num):
            distances = np.sum((emb - centers_tsne[c])**2, axis=1)
            idx = np.argmin(distances)
            centroid_map[c] = all_subgoals[idx]
        
        # Build timestep labels
        labels = -np.ones(T, dtype=int)
        for loc, t in zip(ids, all_subgoal_ts):
            labels[t] = loc
        
        extra = {
            'subgoals': all_subgoals,
            'timesteps': all_subgoal_ts,
            'subgoals_tsne': emb,
            'cluster_ids': ids,
            'cluster_centers_tsne': centers_tsne
        }
        
        return labels, centroid_map, extra


class BUDSClustering(ClusteringStrategy):
    """
    BUDS (Bottom-Up Discovery of Skills) clustering strategy.
    Uses hierarchical segmentation and spectral clustering.
    """
    
    def __init__(self, config: BUDSInterfaceConfig, random_state: int = 0):
        self.config = config
        self.random_state = random_state
        self.cluster_gen = BUDSClusterGeneator(config)
        self.logger = get_logger(__name__)
    
    def cluster(self, dataloader: Any) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, Any]]:
        """Perform BUDS clustering on the dataloader."""
        # Extract subgoals
        observations = dataloader.stacked_data['observations']
        terminals = dataloader.stacked_data['terminals']
        all_subgoals, all_subgoal_ts = extract_subgoals(
            observations, terminals, self.config.goal_offset
        )
        dataloader.stacked_data['subgoals'] = all_subgoals.copy()
        
        # Run BUDS segmentation and clustering
        self.logger.debug("Running BUDS segmentation and clustering")
        dataloader = self.cluster_gen.perform_segmentation_and_clustering(dataloader)
        
        cluster_ids = dataloader.stacked_data['entry']
        subgoals = dataloader.stacked_data['subgoals']
        num_clusters = self.cluster_gen.num_clusters
        
        # Build centroid mapping
        sd = dataloader.stacked_data
        centroid_map = {}
        for cid in range(num_clusters):
            idx = np.where(cluster_ids == cid)[0]
            if len(idx) == 0:
                continue
            centroid = subgoals[idx].mean(axis=0).astype(np.float32)
            centroid_map[cid] = centroid
            sd["skill_aux"][idx] = centroid
        
        sd["entry"] = cluster_ids.astype(np.int32)
        sd["skill_id"] = cluster_ids.astype(np.int32)
        
        extra = {
            'subgoals': subgoals,
            'timesteps': all_subgoal_ts,
            'cluster_ids': cluster_ids,
            'segments': self.cluster_gen.segments,
        }
        
        return cluster_ids, centroid_map, extra


class InstanceRetrievalClustering(ClusteringStrategy):
    """
    Instance retrieval strategy - no clustering, each instance is its own cluster.
    """
    
    def __init__(self, config: BaseClusteringConfig, random_state: int = 0):
        self.config = config
        self.random_state = random_state
        self.logger = get_logger(__name__)
    
    def cluster(self, dataloader: Any) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, Any]]:
        """Perform instance-based 'clustering' (each point is its own cluster)."""
        observations = dataloader.stacked_data['observations']
        terminals = dataloader.stacked_data['terminals']
        T = observations.shape[0]
        
        # Extract subgoals
        all_subgoals, all_subgoal_ts = extract_subgoals(
            observations, terminals, self.config.goal_offset
        )
        dataloader.stacked_data['subgoals'] = all_subgoals.copy()
        
        # Each timestep is its own cluster
        labels = np.arange(T, dtype=int)
        centroid_map = {i: all_subgoals[i] for i in range(len(all_subgoals))}
        
        extra = {
            'subgoals': all_subgoals,
            'timesteps': list(range(T))
        }
        
        return labels, centroid_map, extra


class SemanticClustering(ClusteringStrategy):
    """
    Semantic clustering based on pre-defined skill labels and embeddings.
    """
    
    def __init__(self, config: SemanticClusteringConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load semantic embeddings from file."""
        try:
            with open(self.config.semantic_emb_path, "rb") as f:
                self.semantic_embeddings = pickle.load(f)
            
            # Convert list embeddings to numpy arrays
            for key, val in list(self.semantic_embeddings.items()):
                if isinstance(val, list):
                    self.semantic_embeddings[key] = np.array(val)
                    
        except Exception as e:
            raise ValueError(f"Could not load semantic embeddings from {self.config.semantic_emb_path}: {e}")
    
    def cluster(self, dataloader: Any) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, Any]]:
        """Perform semantic clustering based on skill labels."""
        data = dataloader.stacked_data
        
        # Extract subgoals
        observations = data['observations']
        terminals = data['terminals']
        all_subgoals, all_subgoal_ts = extract_subgoals(
            observations, terminals, self.config.goal_offset
        )
        dataloader.stacked_data['subgoals'] = all_subgoals.copy()
        
        # Check for skills field
        if 'skills' not in data:
            raise ValueError("Stacked data must contain a 'skills' field for semantic clustering.")
        
        skills = np.array(data['skills'])
        unique_skills = np.unique(skills)
        
        # Map skills to cluster indices
        skill_to_cluster = {skill: idx for idx, skill in enumerate(unique_skills)}
        cluster_ids = np.array([skill_to_cluster[s] for s in skills], dtype=np.int32)
        
        # Build centroid map using semantic embeddings
        centroid_map = {}
        for skill, cid in skill_to_cluster.items():
            if skill not in self.semantic_embeddings:
                raise KeyError(f"Embedding for skill '{skill}' not found in semantic_embeddings.")
            centroid_map[cid] = self.semantic_embeddings[skill]
        
        # Prepare extra data
        T = len(skills)
        extra = {
            'subgoals': all_subgoals.copy(),
            'semantic_skill_aux': [self.semantic_embeddings[skills[t]] for t in range(T)],
            'timesteps': list(range(T)),
        }
        
        return cluster_ids, centroid_map, extra


# ==============================
# Factory for Creating Strategies
# ==============================

class ClusteringStrategyFactory:
    """Factory for creating clustering strategies."""
    
    STRATEGIES = {
        'ptgm': (PTGMClustering, PTGMInterfaceConfig),
        'ptgmu': (PTGMClustering, PTGMInterfaceConfig),  # UMAP variant
        'buds': (BUDSClustering, BUDSInterfaceConfig),
        'semantic': (SemanticClustering, SemanticClusteringConfig),
        'instance': (InstanceRetrievalClustering, BaseClusteringConfig),
    }
    
    @classmethod
    def create(cls, algorithm: str, config: Any, random_state: int = 0) -> ClusteringStrategy:
        """Create a clustering strategy instance."""
        if algorithm not in cls.STRATEGIES:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        strategy_class, _ = cls.STRATEGIES[algorithm]
        
        if algorithm in ['ptgm', 'buds', 'instance']:
            return strategy_class(config, random_state)
        else:  # semantic
            return strategy_class(config)
    
    @classmethod
    def get_config_class(cls, algorithm: str):
        """Get the configuration class for a given algorithm."""
        if algorithm not in cls.STRATEGIES:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        return cls.STRATEGIES[algorithm][1]