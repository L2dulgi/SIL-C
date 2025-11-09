"""
SIL-C (Skill Incremental Learning with Clustering) Package

This package contains the refactored components from lazySI.py with improved
modularity and maintainability.
"""

# Core components
from .core import (
    # Distance metrics
    DistanceMetric,
    MahalanobisDistance,
    EuclideanDistance,
    CosineDistance,
    
    # Threshold validators
    ThresholdValidator,
    Chi2ThresholdValidator,
    PercentileThresholdValidator,
    
    # Prototypes
    PrototypeConfig,
    Prototype,
    PrototypeFactory,
    SkillPrototype,
    PolicyPrototype,
)

# Clustering strategies
from .clustering import (
    ClusteringStrategy,
    ClusteringStrategyFactory,
    BaseClusteringConfig,
    SemanticClusteringConfig,
    PTGMClustering,
    BUDSClustering,
    InstanceRetrievalClustering,
    SemanticClustering,
)

# Main interface
from .interface import (
    SILCInterface,
    SILCInterfaceConfig,
    DataLoaderManager,
    SkillManager,
)

__all__ = [
    # Core
    'DistanceMetric',
    'MahalanobisDistance',
    'EuclideanDistance',
    'CosineDistance',
    'ThresholdValidator',
    'Chi2ThresholdValidator',
    'PercentileThresholdValidator',
    'PrototypeConfig',
    'Prototype',
    'PrototypeFactory',
    'SkillPrototype',
    'PolicyPrototype',
    
    # Clustering
    'ClusteringStrategy',
    'ClusteringStrategyFactory',
    'BaseClusteringConfig',
    'SemanticClusteringConfig',
    'PTGMClustering',
    'BUDSClustering',
    'InstanceRetrievalClustering',
    'SemanticClustering',
    
    # Interface
    'SILCInterface',
    'SILCInterfaceConfig',
    'DataLoaderManager',
    'SkillManager',
]