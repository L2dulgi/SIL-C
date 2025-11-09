"""
SIL-C (Skill Incremental Learning with Clustering) - Refactored Components

This module contains the refactored components from lazySI.py with improved
modularity, type hints, and cleaner separation of concerns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Union, Any
import numpy as np
from scipy.stats import chi2
from SILGym.utils.cuml_wrapper import KMeans
from sklearn.metrics import silhouette_score
from SILGym.utils.logger import get_logger


# ==============================
# Distance Metrics
# ==============================

class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""
    
    @abstractmethod
    def compute(self, x: np.ndarray, mean: np.ndarray, variance: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute distance between points x and prototypes."""
        pass


class MahalanobisDistance(DistanceMetric):
    """Mahalanobis distance metric."""
    
    def compute(self, x: np.ndarray, mean: np.ndarray, variance: Optional[np.ndarray] = None) -> np.ndarray:
        eps = 1e-6
        diff = x[:, None, :] - mean[None, :, :]
        if variance is None:
            variance = np.ones_like(mean)
        squared_norm = diff**2 / (variance[None, :, :] + eps)
        return np.sqrt(np.sum(squared_norm, axis=-1))


class EuclideanDistance(DistanceMetric):
    """Euclidean distance metric."""
    
    def compute(self, x: np.ndarray, mean: np.ndarray, variance: Optional[np.ndarray] = None) -> np.ndarray:
        diff = x[:, None, :] - mean[None, :, :]
        squared_diff = diff**2
        return np.sqrt(np.sum(squared_diff, axis=-1))


class CosineDistance(DistanceMetric):
    """Cosine distance metric."""
    
    def compute(self, x: np.ndarray, mean: np.ndarray, variance: Optional[np.ndarray] = None) -> np.ndarray:
        eps = 1e-6
        norm_x = np.linalg.norm(x, axis=1, keepdims=True)
        norm_mean = np.linalg.norm(mean, axis=1, keepdims=True).T
        dot_product = np.sum(x[:, None, :] * mean[None, :, :], axis=-1)
        return 1 - dot_product / (norm_x * norm_mean + eps)


# ==============================
# Threshold Validators
# ==============================

class ThresholdValidator(ABC):
    """Abstract base class for threshold validation strategies."""
    
    @abstractmethod
    def validate(self, distances: np.ndarray, threshold: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Validate distances against thresholds."""
        pass


class Chi2ThresholdValidator(ThresholdValidator):
    """Chi-square based threshold validation."""
    
    def __init__(self, confidence_interval: float = 0.99, df: int = None):
        self.confidence_interval = confidence_interval
        self.df = df
        if df:
            chi2_threshold_sq = chi2.ppf(confidence_interval, df)
            self.sqrt_chi2_dist = np.sqrt(chi2_threshold_sq)
    
    def validate(self, distances: np.ndarray, threshold: Union[float, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        valid_mask = distances < self.sqrt_chi2_dist
        valid = np.any(valid_mask, axis=1)
        min_idx = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(distances.shape[0]), min_idx]
        return valid, min_distances


class PercentileThresholdValidator(ThresholdValidator):
    """Percentile-based threshold validation."""
    
    def validate(self, distances: np.ndarray, threshold: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if threshold is None:
            raise ValueError("Threshold not set for validation.")
        
        thr = np.array(threshold).reshape(-1)
        valid_mask = distances < thr
        valid = np.any(valid_mask, axis=1)
        
        min_idx = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(distances.shape[0]), min_idx]
        return valid, min_distances


# ==============================
# Prototype Classes
# ==============================

@dataclass
class PrototypeConfig:
    """Configuration for prototype creation."""
    distance_type: str = "maha"
    threshold_type: str = "chi2"
    confidence_interval: float = 0.99
    num_bases: int = 5


class Prototype:
    """
    Enhanced prototype class with pluggable distance metrics and threshold validators.
    """
    
    DISTANCE_METRICS = {
        "maha": MahalanobisDistance,
        "euclidean": EuclideanDistance,
        "cossim": CosineDistance,
    }
    
    THRESHOLD_VALIDATORS = {
        "chi2": Chi2ThresholdValidator,
        "percentile": PercentileThresholdValidator,
    }
    
    def __init__(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
        threshold: Optional[np.ndarray] = None,
        config: Optional[PrototypeConfig] = None
    ):
        self.mean = mean
        self.variance = variance
        self.threshold = threshold
        self.config = config or PrototypeConfig()
        
        # Initialize distance metric
        metric_class = self.DISTANCE_METRICS.get(self.config.distance_type, MahalanobisDistance)
        self.distance_metric = metric_class()
        
        # Initialize threshold validator
        validator_class = self.THRESHOLD_VALIDATORS.get(self.config.threshold_type, Chi2ThresholdValidator)
        if self.config.threshold_type == "chi2":
            self.threshold_validator = validator_class(
                confidence_interval=self.config.confidence_interval,
                df=mean.shape[1]
            )
        else:
            self.threshold_validator = validator_class()
    
    def compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances between input points and prototypes."""
        if x.ndim == 1:
            x = x[None, :]
        return self.distance_metric.compute(x, self.mean, self.variance)
    
    def validate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input points against prototypes."""
        distances = self.compute_distances(x)
        
        if self.config.threshold_type == "percentile" and self.threshold is not None:
            return self.threshold_validator.validate(distances, self.threshold)
        else:
            return self.threshold_validator.validate(distances)


# ==============================
# Skill and Policy Entries
# ==============================

@dataclass
class SkillPrototype:
    """Data structure for skill prototypes."""
    skill_id: int
    decoder_id: int
    skill_aux: np.ndarray
    state_prototype: Prototype
    action_prototype: Prototype
    subgoal_prototype: Prototype
    data_count: int = 0
    
    @classmethod
    def from_data(
        cls,
        skill_id: int,
        decoder_id: int,
        skill_aux: np.ndarray,
        state_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        action_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        subgoal_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        data_count: int,
        config: PrototypeConfig
    ) -> 'SkillPrototype':
        """Create SkillPrototype from data tuples."""
        return cls(
            skill_id=skill_id,
            decoder_id=decoder_id,
            skill_aux=skill_aux,
            state_prototype=Prototype(*state_data, config=config),
            action_prototype=Prototype(*action_data, config=config),
            subgoal_prototype=Prototype(*subgoal_data, config=config),
            data_count=data_count
        )


@dataclass
class PolicyPrototype:
    """Data structure for policy prototypes."""
    prototype_id: int
    subgoal: np.ndarray
    state_prototype: Prototype
    data_count: int = 0
    
    @classmethod
    def from_data(
        cls,
        prototype_id: int,
        subgoal: np.ndarray,
        state_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        data_count: int,
        config: PrototypeConfig
    ) -> 'PolicyPrototype':
        """Create PolicyPrototype from data tuple."""
        return cls(
            prototype_id=prototype_id,
            subgoal=subgoal,
            state_prototype=Prototype(*state_data, config=config),
            data_count=data_count
        )


# ==============================
# Prototype Factory
# ==============================

class PrototypeFactory:
    """Factory for creating prototypes from data."""
    
    def __init__(self, config: PrototypeConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def create_prototypes_from_data(
        self,
        data: np.ndarray,
        instance_based: bool = False,
        num_bases: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create prototypes from data using KMeans clustering.
        
        Returns:
            Tuple of (centroids, variances, thresholds)
        """
        if instance_based:
            return (data, np.zeros_like(data), np.zeros((data.shape[0], 1)))
        
        K = num_bases or self.config.num_bases
        
        if K == 0:
            K = self._find_optimal_k(data)
        
        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(data)
        
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Compute variances and thresholds
        variances = self._compute_variances(data, labels, K)
        thresholds = self._compute_thresholds(data, centroids, variances, labels, K)
        
        return centroids, variances, thresholds
    
    def _find_optimal_k(self, data: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score."""
        K_range = range(2, min(11, len(data)))
        best_score = -1
        best_k = 2
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(data)
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        self.logger.debug(f"Optimal K found: {best_k} with score {best_score}")
        return best_k
    
    def _compute_variances(self, data: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
        """Compute per-dimension variance for each cluster."""
        variances = []
        for i in range(K):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                var = np.var(cluster_points, axis=0)
            else:
                var = np.zeros(data.shape[1])
            variances.append(var)
        return np.array(variances)
    
    def _compute_thresholds(
        self,
        data: np.ndarray,
        centroids: np.ndarray,
        variances: np.ndarray,
        labels: np.ndarray,
        K: int
    ) -> np.ndarray:
        """Compute distance thresholds for each cluster."""
        percent = self.config.confidence_interval * 100
        eps = 1e-6
        thresholds = []
        
        distance_metric = Prototype.DISTANCE_METRICS[self.config.distance_type]()
        
        for i in range(K):
            cluster_pts = data[labels == i]
            if len(cluster_pts) > 0:
                # Compute distances based on distance type
                if self.config.distance_type == "euclidean":
                    dists = np.linalg.norm(cluster_pts - centroids[i], axis=1)
                elif self.config.distance_type == "cossim":
                    dot = np.sum(cluster_pts * centroids[i], axis=1)
                    norms = np.linalg.norm(cluster_pts, axis=1) * np.linalg.norm(centroids[i])
                    dists = 1 - dot / (norms + eps)
                else:  # maha
                    std = np.sqrt(variances[i]) + eps
                    dists = np.linalg.norm((cluster_pts - centroids[i]) / std, axis=1)
                
                thr = np.percentile(dists, percent)
            else:
                thr = 0.0
            thresholds.append(thr)
        
        return np.array(thresholds)[:, None]


# ==============================
# Clustering Strategies
# ==============================

class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    @abstractmethod
    def cluster(self, dataloader: Any) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, Any]]:
        """
        Perform clustering on dataloader.
        
        Returns:
            - labels: cluster assignments for each timestep
            - centroid_map: mapping from cluster id to centroid
            - extra: additional data (e.g., subgoals, timesteps)
        """
        pass


# The specific clustering strategy implementations would go here
# but I'll leave them in the original file to keep this focused on the core refactoring