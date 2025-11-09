"""
LazySI (Lazy Skill Interface) - Original Implementation

NOTE: A refactored version of this module is available in the 'silc' package
with improved modularity and maintainability. See SILGym/models/skill_interface/silc/
for the refactored implementation.
"""

import numpy as np
from SILGym.utils.cuml_wrapper import TSNE, UMAP, KMeans
import matplotlib.pyplot as plt
from SILGym.models.skill_interface.base import BaseInterface
from scipy.stats import chi2
from sklearn.metrics import silhouette_score
from SILGym.utils.logger import get_logger

# ------------------------------
# SKill prototypes 
# ------------------------------
import numpy as np
from scipy.stats import chi2

class MaPrototype:
    '''
    Prototype for computing the Mahalanobis distance using a diagonal covariance matrix.
    
    Parameters:
      mean: np.ndarray of shape (prototype_bases, state_dim)
            Each row represents the mean of a prototype.
      variance: np.ndarray of shape (prototype_bases, state_dim)
            Each row represents the per-dimension variances for a prototype.
    '''
    def __init__(self, mean: np.ndarray, variance: np.ndarray, threshold: np.ndarray = None, confidence_interval: float = 0.99, 
                distance_type: str="maha" ,threshold_type: str = "chi2"):
        self.mean = mean              # (prototype_bases, state_dim)
        self.variance = variance      # (prototype_bases, state_dim)
        self.threshold = threshold    # (prototype_bases,) or None
        # Compute the sqrt of chi-square threshold for given confidence interval
        df = mean.shape[1]  # number of features (state_dim)
        chi2_threshold_sq = chi2.ppf(confidence_interval, df)
        self.sqrt_chi2_dist = np.sqrt(chi2_threshold_sq)
        self.distance_type = distance_type    # maha or euclidean
        self.threshold_type = threshold_type

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the Mahalanobis distance between each sample in x and each prototype.
        
        Parameters:
            x: np.ndarray of shape (B, state_dim)
               Batch of input points.
        
        Returns:
            distances: np.ndarray of shape (B, prototype_bases)
               Each element [i, j] is the Mahalanobis distance between x[i] and prototype j.
        '''
        if x.ndim == 1:
            x = x[None, :]
        eps = 1e-6  # Small constant to avoid division by zero
        # Broadcast x to (B, 1, state_dim) and mean to (1, prototype_bases, state_dim)
        diff = x[:, None, :] - self.mean[None, :, :]  # (B, prototype_bases, state_dim)
        if hasattr(self, "distance_type") == False:
            self.distance_type = "maha"
        if self.distance_type == "euclidean":
            # Compute squared differences
            squared_diff = diff**2
            # Sum over state dimension for Euclidean distance
            distances = np.sqrt(np.sum(squared_diff, axis=-1))
        elif self.distance_type == "cossim" :
            # Compute cosine similarity
            norm_x = np.linalg.norm(x, axis=1, keepdims=True)  # (B, 1)
            norm_mean = np.linalg.norm(self.mean, axis=1, keepdims=True)  # (1, prototype_bases)
            dot_product = np.sum(x[:, None, :] * self.mean[None, :, :], axis=-1)  # (B, prototype_bases)
            distances = 1 - dot_product / (norm_x * norm_mean + eps)  # Cosine distance
        elif self.distance_type == "maha":
             # Compute normalized squared differences
            squared_norm = diff**2 / (self.variance[None, :, :] + eps)
            # Sum over state dimension and take square root for Mahalanobis distance
            distances = np.sqrt(np.sum(squared_norm, axis=-1))  # (B, prototype_bases)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        return distances
    
    def validate(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self, "threshold_type") == False:
            return self.validate_chi2(x)
        if self.threshold_type == "chi2":
            return self.validate_chi2(x)
        elif self.threshold_type == "percentile":
            return self.validate_old(x)
        

    def validate_old(self, x: np.ndarray) :
        '''
        Validate each point in x against user-defined thresholds.

        Parameters:
            x: np.ndarray, shape (B, state_dim) or (state_dim,)

        Returns:
            valid: np.ndarray of shape (B,), dtype bool
                True if the point's Mahalanobis distance to any prototype
                is below that prototype's threshold.
            min_distances: np.ndarray of shape (B,), dtype float
                Mahalanobis distance to the nearest prototype.
        '''
        if self.threshold is None:
            raise ValueError("Threshold not set for validation.")

        x_arr = x[None, :] if x.ndim == 1 else x
        B = x_arr.shape[0]

        # Compute distances to all prototypes
        distances = self.forward(x_arr)  # (B, prototype_bases)

        # Flatten thresholds if needed (prototype_bases,)
        thr = self.threshold.reshape(-1)

        # Compare distances against thresholds
        valid_mask = distances < thr      # (B, prototype_bases)

        # A sample is valid if any prototype distance is below its threshold
        valid = np.any(valid_mask, axis=1)  # (B,)

        # Get index and distance of nearest prototype
        min_idx = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(B), min_idx]

        return valid, min_distances

    def validate_chi2(self, x: np.ndarray) :
        '''
        Perform validation using the chi-square based threshold (sqrt of chi2).

        Parameters:
            x: np.ndarray, shape (B, state_dim) or (state_dim,)

        Returns:
            valid: np.ndarray of shape (B,), dtype bool
            min_distances: np.ndarray of shape (B,), dtype float
        '''
        # Ensure input has batch dimension
        x_arr = x[None, :] if x.ndim == 1 else x
        B = x_arr.shape[0]

        # Compute Mahalanobis distances via forward
        distances = self.forward(x_arr)  # (B, prototype_bases)

        # Compare against chi-square threshold distance
        valid_mask = distances < self.sqrt_chi2_dist  # (B, prototype_bases)
        valid = np.any(valid_mask, axis=1)            # (B,)

        # Return the minimum distance for each sample
        min_idx = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(B), min_idx]

        return valid, min_distances
        
class SkillEntry:
    def __init__(
            self, 
            skill_id,          # int 
            decoder_id,        # int
            skill_aux,         # np.array
            state_prototypes=None,   # np.array (prototype_bases, state_dim)
            action_prototypes=None, # np.array (prototype_bases, action_dim) 
            subgoal_prototypes=None, # np.array (prototype_bases, state_dim)
            data_count=0,       # int, number of data points in this cluster
            confidence_interval=0.99, # float, confidence interval for mahalanobis distance
            threshold_type="chi2", # str, type of threshold for validation
            distance_type="maha",  # distance metric type
        ):
        self.skill_id = skill_id
        self.decoder_id = decoder_id
        self.skill_aux = skill_aux
        self.threshold_type = threshold_type
        # state prototypes : bases and variances for mahalanobis distance
        # (means, variances) 
        self.state_prototypes = MaPrototype(
            mean=state_prototypes[0],
            variance=state_prototypes[1],
            threshold=state_prototypes[2],
            confidence_interval=confidence_interval,
            threshold_type=self.threshold_type,
            distance_type=distance_type,
        )

        self.action_prototypes = MaPrototype(
            mean=action_prototypes[0],
            variance=action_prototypes[1],
            threshold=action_prototypes[2],
            confidence_interval=confidence_interval,
            threshold_type=self.threshold_type,
            distance_type=distance_type,
        )

        # subgoal prototypes : bases and variances for mahalanobis distance 
        self.subgoal_prototypes = MaPrototype(
            mean=subgoal_prototypes[0],
            variance=subgoal_prototypes[1],
            threshold=subgoal_prototypes[2],
            confidence_interval=confidence_interval,
            threshold_type=self.threshold_type,
            distance_type=distance_type,
        )

        self.data_count = data_count  

class PolicyEntry:
    def __init__(
            self,
            prototype_id,      # int
            subgoal,           # np.array
            state_prototypes=None,   # np.array (prototype_bases, state_dim)
            data_count=0,      # int, number of data points used for this cluster
            distance_type="maha",
            threshold_type="chi2", # str, type of threshold for validation
            confidence_interval=0.99, # float, confidence interval for mahalanobis distance
        ):
        self.prototype_id = prototype_id
        self.subgoal = subgoal
        self.state_prototypes = MaPrototype(
            state_prototypes[0],
            state_prototypes[1],
            state_prototypes[2],
            distance_type=distance_type,
            threshold_type=threshold_type,
            confidence_interval=confidence_interval,
        )
        self.data_count = data_count

# ------------------------------
# SKill Generator 
# ------------------------------
from SILGym.models.skill_interface.ptgm import PTGMInterfaceConfig
from SILGym.models.skill_interface.buds import BUDSInterfaceConfig   

class InstanceRetrievalConfig:
    def __init__(
            self, 
            goal_offset=20,   
        ):
        self.goal_offset = goal_offset

import numpy as np
import pickle
from SILGym.models.skill_interface.buds import BUDSClusterGeneator, BUDSInterfaceConfig  # Core BUDS algorithm imports
class SemanticInterfaceConfig:
    def __init__(self, semantic_emb_path, goal_offset=20):
        """
        Configuration for the SemanticInterface.
        
        Args:
            semantic_emb_path: Path to the file containing semantic embeddings.
        """
        self.semantic_emb_path = semantic_emb_path
        self.goal_offset = goal_offset

class SkillClusteringStrategy:
    """
    Abstract base class for subgoal clustering strategies.
    """
    def cluster(self, subgoals: np.ndarray):
        """
        Given an array of subgoals (N, dim), return:
          - cluster_ids: np.array of shape (N,) assigning each subgoal to a cluster
          - centroid_map: dict mapping cluster index to a representative subgoal
          - extra: dict of any data to store (e.g., t-SNE embeddings, centers)
        """
        raise NotImplementedError

class PTGMClusteringStrategy(SkillClusteringStrategy):
    """
    ptgm cluster generator: minimal edits from map_entry_v1.
    """
    def __init__(self, config, random_state=0):
        self.config = config
        self.random_state = random_state

    def cluster(self, dataloader):
        # Based on map_entry_v1 logic with UMAP support
        data = dataloader.stacked_data
        observations = data['observations']
        terminals = data['terminals']
        T = len(observations)

        # split trajectories
        def _split_by_trajectory(terminals):
            trajs, start = [], 0
            for i, v in enumerate(terminals):
                if v == 1:
                    trajs.append((start, i)); start = i + 1
            return trajs
        trajs = _split_by_trajectory(terminals)

        all_subgoals, all_subgoal_ts = [], []
        for s, e in trajs:
            for t in range(s, e + 1):
                fut = t + self.config.goal_offset
                sg = observations[fut] if fut <= e else observations[e]
                all_subgoals.append(sg); all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)
        n_samples = len(all_subgoals)

        # Get embedding method (default: 'tsne' for backward compatibility)
        embedding_method = getattr(self.config, 'embedding_method', 'tsne')

        # Apply dimensionality reduction based on embedding_method
        if embedding_method == 'umap':
            # Adaptive parameter adjustment for UMAP
            effective_n_neighbors = min(
                getattr(self.config, 'umap_n_neighbors', 15),
                n_samples - 1
            )
            effective_n_neighbors = max(2, effective_n_neighbors)

            umap_reducer = UMAP(
                n_components=getattr(self.config, 'embedding_dim', self.config.tsne_dim),
                n_neighbors=effective_n_neighbors,
                min_dist=getattr(self.config, 'umap_min_dist', 0.1),
                metric=getattr(self.config, 'umap_metric', 'euclidean'),
                random_state=self.random_state
            )
            emb = umap_reducer.fit_transform(all_subgoals)
        else:  # 'tsne' (default)
            # Adaptive parameter adjustment for t-SNE
            effective_perplexity = min(
                self.config.tsne_perplexity,
                (n_samples - 1) // 3
            )
            effective_perplexity = max(5, effective_perplexity)

            tsne = TSNE(
                n_components=self.config.tsne_dim,
                perplexity=effective_perplexity,
                random_state=self.random_state
            )
            emb = tsne.fit_transform(all_subgoals)

        # KMeans clustering (same for both embedding methods)
        km = KMeans(n_clusters=self.config.cluster_num, random_state=self.random_state)
        ids = km.fit_predict(emb)
        centers_emb = km.cluster_centers_

        # centroid mapping
        centroid_map = {}
        for c in range(self.config.cluster_num):
            d = np.sum((emb - centers_emb[c])**2, axis=1)
            idx = np.argmin(d)
            centroid_map[c] = all_subgoals[idx]

        # build labels
        labels = -np.ones(T, dtype=int)
        for loc, t in zip(ids, all_subgoal_ts):
            labels[t] = loc

        extra = {
            'subgoals': all_subgoals,
            'timesteps': all_subgoal_ts,
            'subgoals_emb': emb,  # renamed from subgoals_tsne for generality
            'cluster_ids': ids,
            'cluster_centers_emb': centers_emb
        }
        return labels, centroid_map, extra

class BUDSClusteringStrategy(SkillClusteringStrategy):
    def __init__(self, config: BUDSInterfaceConfig, random_state: int = 0):
        """
        BUDS clustering strategy using the hierarchical segmentation and spectral clustering
        pipeline defined in BUDSClusterGeneator.

        Args:
            config: BUDSInterfaceConfig with hyperparameters:
                - window_size: initial sliding window length
                - min_length: minimum segment length
                - target_num_segments: target number of segments per demonstration
                - max_k: maximum number of clusters to try
                - goal_offset: offset for subgoal labeling
                - verbose: verbosity flag
            random_state: seed for reproducible clustering
        """
        self.config = config
        self.random_state = random_state
        # Initialize the low-level segmenter & clusterer with provided BUDS config
        self.cluster_gen = BUDSClusterGeneator(config)

    def cluster(self, dataloader):
        """
        Perform BUDS-based clustering on the dataloader's trajectories.

        Returns:
          - labels: np.ndarray of shape (T,) assigning each timestep to a cluster
          - centroid_map: dict mapping cluster index to the mean subgoal vector
          - extra: dict with diagnostics:
              * 'subgoals': all subgoal vectors (shape: [num_subgoals, obs_dim])
              * 'timesteps': corresponding timestep indices for each subgoal
              * 'cluster_ids': same as labels
              * 'segments': list of (start, end) indices for each segment
        """
        # 1) Extract subgoals from dataloader
        observations = dataloader.stacked_data['observations']  
        terminals = dataloader.stacked_data['terminals']    
        def _split_by_trajectory(terminals):
            trajs, start = [], 0
            for i, v in enumerate(terminals):
                if v == 1:
                    trajs.append((start, i)); start = i + 1
            return trajs
        trajs = _split_by_trajectory(terminals)

        all_subgoals, all_subgoal_ts = [], []
        for s, e in trajs:
            for t in range(s, e + 1):
                fut = t + self.config.goal_offset
                sg = observations[fut] if fut <= e else observations[e]
                all_subgoals.append(sg); all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)
        dataloader.stacked_data['subgoals'] = all_subgoals.copy()

        # 2) Run hierarchical segmentation and spectral clustering
        dataloader = self.cluster_gen.perform_segmentation_and_clustering(dataloader)
        # subgoals are already in dataloader.
        cluster_ids = dataloader.stacked_data['entry']
        subgoals = dataloader.stacked_data['subgoals']
        num_clusters = self.cluster_gen.num_clusters

        # 3) Build centroid mapping: average subgoal per cluster
        sd = dataloader.stacked_data
        centroid_map = {}
        for cid in range(num_clusters):
            idx = np.where(cluster_ids == cid)[0]
            if len(idx) == 0:
                continue
            centroid = subgoals[idx].mean(axis=0).astype(np.float32)
            centroid_map[cid] = centroid
            sd["skill_aux"][idx] = centroid

        self.centroid_map = centroid_map
        sd["entry"] = cluster_ids.astype(np.int32)
        sd["skill_id"] = cluster_ids.astype(np.int32)

        # 4) Prepare extra diagnostics
        extra = {
            'subgoals': subgoals,
            'timesteps': all_subgoal_ts,
            'cluster_ids': cluster_ids,
            'segments': self.cluster_gen.segments,
        }
        labels = cluster_ids
        return labels, centroid_map, extra

class InstanceRetrievalStrategy(SkillClusteringStrategy):
    """
    A simple retrieval strategy that does not cluster, but samples a fixed ratio of instances.
    Each sampled point becomes its own 'cluster' with itself as centroid/subgoal.
    """
    def __init__(self, 
            config : InstanceRetrievalConfig,
            random_state: int = 0
        ):
        self.config = config
        self.random_state = random_state

    def cluster(self, dataloader):
        # 1) Extract subgoals from dataloader
        observations = dataloader.stacked_data['observations']  
        terminals = dataloader.stacked_data['terminals']    
        T = observations.shape[0]

        def _split_by_trajectory(terminals):
            trajs, start = [], 0
            for i, v in enumerate(terminals):
                if v == 1:
                    trajs.append((start, i)); start = i + 1
            return trajs
        trajs = _split_by_trajectory(terminals)

        all_subgoals, all_subgoal_ts = [], []
        for s, e in trajs:
            for t in range(s, e + 1):
                fut = t + self.config.goal_offset
                sg = observations[fut] if fut <= e else observations[e]
                all_subgoals.append(sg); all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)
        dataloader.stacked_data['subgoals'] = all_subgoals.copy()

        # 1) Each timestep is its own cluster:
        labels = np.arange(T, dtype=int)        # cluster id = timestep index

        # 2) Centroid/subgoal for cluster i is simply the observation at i
        centroid_map = {i: all_subgoals[i] for i in range(T)}

        # 3) Extras for map_entry:
        extra = {
            'subgoals': all_subgoals,
            'timesteps': list(range(T))
        }

        return labels, centroid_map, extra

class SemanticClusteringStrategy(SkillClusteringStrategy):
    """
    Clustering strategy based on the 'skills' field in the dataloader.
    Treats each unique skill label as its own cluster, and uses precomputed
    semantic embeddings for each skill both as cluster centroids and subgoals.
    """
    def __init__(self, config: SemanticInterfaceConfig):
        super().__init__()
        # Load semantic embeddings from the given file path
        self.config = config 

        semantic_emb_path = config.semantic_emb_path
        try:
            with open(semantic_emb_path, "rb") as f:
                self.semantic_embeddings = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not load semantic embeddings from {semantic_emb_path}: {e}")
        # Convert list embeddings to numpy arrays
        for key, val in list(self.semantic_embeddings.items()):
            if isinstance(val, list):
                self.semantic_embeddings[key] = np.array(val)

    def cluster(self, dataloader):
        """
        Perform semantic clustering based on the 'skills' field.

        Args:
            dataloader: An object with a stacked_data attribute containing:
                - 'observations': np.ndarray of shape (T, obs_dim)
                - 'skills': array-like of length T with string labels

        Returns:
            cluster_ids: np.ndarray of shape (T,) assigning each sample to a cluster
            centroid_map: dict mapping cluster index to the representative subgoal (semantic embedding)
            extra: dict containing:
                - 'subgoals': list of semantic embeddings per timestep
                - 'timesteps': list of timestep indices
        """
        data = dataloader.stacked_data

        # subgoal building
        # 1) Extract subgoals from dataloader
        observations = dataloader.stacked_data['observations']  
        terminals = dataloader.stacked_data['terminals']    
        def _split_by_trajectory(terminals):
            trajs, start = [], 0
            for i, v in enumerate(terminals):
                if v == 1:
                    trajs.append((start, i)); start = i + 1
            return trajs
        trajs = _split_by_trajectory(terminals)

        all_subgoals, all_subgoal_ts = [], []
        for s, e in trajs:
            for t in range(s, e + 1):
                fut = t + self.config.goal_offset
                sg = observations[fut] if fut <= e else observations[e]
                all_subgoals.append(sg); all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)
        dataloader.stacked_data['subgoals'] = all_subgoals.copy()


        if 'skills' not in data:
            raise ValueError("Stacked data must contain a 'skills' field to perform semantic clustering.")
        skills = np.array(data['skills'])
        unique_skills = np.unique(skills)

        # Map each unique skill to an integer cluster index
        skill_to_cluster = {skill: idx for idx, skill in enumerate(unique_skills)}
        cluster_ids = np.array([skill_to_cluster[s] for s in skills], dtype=np.int32)

        # Build centroid_map: use the semantic embedding for each skill
        centroid_map = {}
        for skill, cid in skill_to_cluster.items():
            if skill not in self.semantic_embeddings:
                raise KeyError(f"Embedding for skill '{skill}' not found in semantic_embeddings.")
            centroid_map[cid] = self.semantic_embeddings[skill]

        # Prepare extra diagnostics: subgoals are the semantic embedding of each timestep's skill
        T = len(skills)
        extra = {
            'subgoals': all_subgoals.copy(),
            'semantic_skill_aux' : [self.semantic_embeddings[skills[t]] for t in range(T)],
            'timesteps': list(range(T)),
        }

        return cluster_ids, centroid_map, extra
# Interface config

class LazySIInterfaceConfig:
    # Supported algorithms mapping
    _ALGO_MAP = {
        'ptgm': PTGMClusteringStrategy,
        'ptgmu': PTGMClusteringStrategy,  # UMAP variant
        'buds': BUDSClusteringStrategy,
        'semantic': SemanticClusteringStrategy,
        'instance': InstanceRetrievalStrategy,
    }

    def __init__(
        self,
        # decoder side
        decoder_algo: str = "ptgm",
        decoder_algo_config=None,
        skill_prototype_bases: int = 5,
        # policy side
        policy_algo: str = "ptgm",
        policy_algo_config=None,
        subtask_prototype_bases: int = 5,
        # confidence interval
        force_static: bool = False,
        confidence_interval: float = 95., # 95%confidence interval
        threshold_type: str = "chi2",
        distance_type: str = "maha",
    ):
        # Initialize internal state
        self.decoder_algo = decoder_algo
        self.decoder_algo_config = decoder_algo_config
        self.skill_prototype_bases = skill_prototype_bases
        self.decoder_entry_generator = None

        self.policy_algo = policy_algo
        self.policy_algo_config = policy_algo_config
        self.subtask_prototype_bases = subtask_prototype_bases
        self.policy_entry_generator = None

        self.force_static = force_static   
    

        # confidence interval
        self.confidence_interval = confidence_interval
        self.threshold_type = threshold_type
        self.distance_type = distance_type


        
        self.set_decoder_strategy(decoder_algo, decoder_algo_config)
        self.set_policy_strategy(policy_algo, policy_algo_config)

        
    def _create_generator(self, algo: str, config):
        """
        Shared logic to create a clustering strategy based on algorithm name.
        """
        strategy_cls = self._ALGO_MAP.get(algo)
        if strategy_cls is None:
            raise ValueError(f"Unknown algorithm: {algo}")
        return strategy_cls(config)

    def set_decoder_strategy(self, algo: str, config):
        """
        Atomically set the decoder algorithm and its configuration.
        """
        self.decoder_algo_config = config
        self.decoder_algo = algo
        self.decoder_entry_generator = self._create_generator(algo, config)

    def set_policy_strategy(self, algo: str, config):
        """
        Atomically set the policy algorithm and its configuration.
        """
        self.policy_algo_config = config
        self.policy_algo = algo
        self.policy_entry_generator = self._create_generator(algo, config)

from copy import deepcopy

class LazySIInterface(BaseInterface):
    def __init__(
            self, 
            config:LazySIInterfaceConfig=None,
        ):
        self.config = config
        self.logger = get_logger(__name__) 
        # if config is not None else LazySIInterfaceConfig(
        #     decoder_algo= "ptgm",
        #     decoder_algo_config=PTGMInterfaceConfig(
        #         cluster_num= 20,
        #         goal_offset= 40,
        #         tsne_dim=3,
        #     ),
        #     skill_prototype_bases= 5,
        #     # policy side
        #     policy_algo= "ptgm",
        #     policy_algo_config=PTGMInterfaceConfig(
        #         cluster_num= 20,
        #         goal_offset= 40,
        #         tsne_dim=3,
        #     ),
        #     subtask_prototype_bases= 5,
        #     # confidence interval
        #     confidence_interval = 99, # 99%confidence interval for mmworld is better.
        # )
        super().__init__()
        
        # ----------------
        # Decoder side
        # ----------------
        # 1. state prototypes
        # 2. sub-goal prototypes 
        '''
        { "[skill_id]" : SkillEntry }
        '''
        self.entry_skill_map = {}
        self.decoder_entry_generator = self.config.decoder_entry_generator
        # ----------------
        # Policy side
        # ----------------
        # for policy side implementation., we outsource the prototype saving for the policy model.
        # so we need to sync the policy model with the interface for agent building.
        # 1. state prototypes 
        '''
        { "[prototype_id]" : PolicyEntry }
        '''
        self.subtask_prototypes = {}
        self.policy_entry_generator = self.config.policy_entry_generator

        self.decoder_id = 0
        self.debug = False

        self.candidates = 1


        # print the config 
        self.logger.info(f"decoder_algo: {self.config.decoder_algo}")
        self.logger.info(f"decoder_algo_config: {self.config.decoder_algo_config}")
        self.logger.info(f"skill_prototype_bases: {self.config.skill_prototype_bases}")
        self.logger.info(f"policy_algo: {self.config.policy_algo}")
        self.logger.info(f"policy_algo_config: {self.config.policy_algo_config}")
        self.logger.info(f"subtask_prototype_bases: {self.config.subtask_prototype_bases}")
        self.logger.info(f"confidence_interval: {self.config.confidence_interval}")
        self.logger.info(f"threshold_type: {self.config.threshold_type}")
        self.logger.info(f"distance_type: {self.config.distance_type}")
        


        self.logger.info(f"force_static: {self.config.force_static}")

    @property
    def num_skills(self):
        """
        Returns the number of skills.
        """
        return len(self.entry_skill_map)
    
    # ----------------------------------
    # functions for call from trainer
    # ----------------------------------
    def update_interface(self, dataloader):
        dataloader = self.init_entry(dataloader)
        dataloader = self.map_entry(dataloader)
        return self.update_dataloader(dataloader)
    
    def update_dataloader(self, dataloader):
        """
        Update the dataloader with new data.
        
        For the SemanticInterface, no additional modifications are needed.
        """
        dataloader.stacked_data['orig_obs'] = dataloader.stacked_data['observations'].copy()
        dataloader.stacked_data['observations'] = np.concatenate(
            (dataloader.stacked_data['observations'], dataloader.stacked_data['skill_aux']), axis=-1
        )
        return dataloader

    def rollback_dataloader(self, dataloader):
        """
        Rollback the dataloader to the previous state.
        
        For the SemanticInterface, no changes are made, so the dataloader is returned unchanged.
        """
        if 'orig_obs' not in dataloader.stacked_data:
            return dataloader
        dataloader.stacked_data['observations'] = dataloader.stacked_data['orig_obs']
        del dataloader.stacked_data['orig_obs']
        return dataloader
    
    # ----------------------------------
    # functions for decoder side interface
    # ----------------------------------
    def init_entry(self, dataloader):
        stacked_data = dataloader.stacked_data
        T = len(stacked_data['observations'])
        if 'entry' not in stacked_data:
            stacked_data['entry'] = np.zeros((T,), dtype=np.int32)
        if 'skill_id' not in stacked_data:
            stacked_data['skill_id'] = np.zeros((T,), dtype=np.int32)
        obs_dim = stacked_data['observations'].shape[1]
        if 'skill_aux' not in stacked_data:
            stacked_data['skill_aux'] = np.zeros((T, obs_dim), dtype=np.float32)
        if 'decoder_id' not in stacked_data:
            stacked_data['decoder_id'] = np.zeros((T,), dtype=np.int32)
        if 'subgoal' not in stacked_data:
            stacked_data['subgoal'] = np.zeros((T, obs_dim), dtype=np.float32)

        return dataloader

    def map_entry(self, dataloader):
        # 1. cluster function ( get dataloader itself. and return the labels)
        labels, centroid_map, extra = self.decoder_entry_generator.cluster(dataloader)
        
        # 2. update dataloader with the labels, updated infos
        data = dataloader.stacked_data
        obs = np.array(data['observations'])
        # attach subgoals array to dataloader
        T = len(obs)
        data['subgoals'] = np.zeros_like(obs)
        for sg, t in zip(extra['subgoals'], extra['timesteps']):
            data['subgoals'][t] = sg


        # 3. update interface global map
        base = self.num_skills
        appended_skill_num = len(list(centroid_map.keys()))
        global_map = {loc: base + loc for loc in range(appended_skill_num)}

        if isinstance(self.decoder_entry_generator, SemanticClusteringStrategy):
            data['skill_aux'] = np.array(extra['semantic_skill_aux'])
        else:
            data['skill_aux'] = np.zeros_like(obs)

        # assign per-timestep values
        for t, loc in enumerate(labels):
            if loc < 0:
                continue
            gid = global_map[loc]
            data['skill_id'][t] = gid
            data['entry'][t]    = gid
            if not isinstance(self.decoder_entry_generator, SemanticClusteringStrategy):
                data['skill_aux'][t] = centroid_map[loc]

        # 4. create the entries prototypes
        for loc in range(appended_skill_num):
            gid = global_map[loc]
            if gid not in self.entry_skill_map:
                idxs = np.where(data['entry'] == gid)[0]

                if isinstance(self.decoder_entry_generator, SemanticClusteringStrategy):
                    # print(centroid_map[loc].shape, data['skill_aux'][idxs[0]])
                    skill_aux_vec = data['skill_aux'][idxs[0]]
                    self.logger.info(f"Skill {gid} centroid: {np.unique(data['skills'][idxs])}, {skill_aux_vec.shape}") 
                else:
                    skill_aux_vec = centroid_map[loc]

                self.entry_skill_map[gid] = SkillEntry(
                    skill_id=gid,
                    decoder_id=self.decoder_id,
                    skill_aux=skill_aux_vec,
                    state_prototypes=self._create_prototypes(obs[idxs]),
                    action_prototypes=self._create_prototypes(np.array(data['actions'])[idxs]),
                    subgoal_prototypes=self._create_prototypes(data['subgoals'][idxs]),
                    data_count=len(idxs),
                    confidence_interval=self.config.confidence_interval,
                    threshold_type=self.config.threshold_type,
                    distance_type=self.config.distance_type,
                )
                
        data['decoder_id'] = np.full(T, self.decoder_id, dtype=int)
        self.decoder_id += 1
        return dataloader

    def _create_prototypes(self, data, instance_based=False, K=None):
        """
        Create prototypes from the data using KMeans clustering and compute per-dimension variance
        for the data points belonging to each cluster.
        
        Parameters:
        data (np.array): Input data used to form prototypes, shape (N, state_dim).
        
        Returns:
        A tuple (centroids, variances) where:
            - centroids: np.array of shape (prototype_bases, state_dim) containing the cluster centers.
            - variances: np.array of shape (prototype_bases, state_dim) containing the per-dimension variance
                        of the data points assigned to each cluster.
        """
        if instance_based:  
            return (data, np.zeros_like(data), np.zeros((data.shape[0], 1)))
        K = self.config.skill_prototype_bases if K is None else K


        if K == 0 :
            # K=0 for search for the best K
            self.logger.debug(f"K=0, search for the best K")
            K_range = range(2, 11)  
            best_score = -1
            best_kmeans = None
            for K in K_range:
                kmeans = KMeans(n_clusters=K, random_state=0)
                labels = kmeans.fit_predict(data)
                if len(set(labels)) > 1:  
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_kmeans = kmeans
            labels = best_kmeans.labels_
            centroids = best_kmeans.cluster_centers_
            K = best_kmeans.n_clusters
            kmeans = best_kmeans
        else :
            kmeans = KMeans(n_clusters=K, random_state=0)
            kmeans.fit(data)
        centroids = np.array(kmeans.cluster_centers_)
        labels = kmeans.labels_
        variances = []
        for i in range(K):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                var = np.var(cluster_points, axis=0)  # per-dimension variance
            else:
                var = np.zeros(data.shape[1])
            variances.append(var)
        variances = np.array(variances)

        # threshold 
        percent = getattr(self.config, 'confidence_interval', 95)
        eps = 1e-6
        thresholds = []
        nums = []
        for i in range(K):
            cluster_pts = data[labels == i]
            nums.append(len(cluster_pts))
            if len(cluster_pts) > 0:
                if self.config.distance_type == "euclidean":
                    dists = np.linalg.norm(cluster_pts - centroids[i], axis=1)
                elif self.config.distance_type == "cossim":
                    dot = np.sum(cluster_pts * centroids[i], axis=1)
                    dists = 1 - dot / (np.linalg.norm(cluster_pts, axis=1) * np.linalg.norm(centroids[i]) + eps)
                else:
                    std = np.sqrt(variances[i]) + eps
                    dists = np.linalg.norm((cluster_pts - centroids[i]) / std, axis=1)
                thr = np.percentile(dists, percent)
            else:
                thr = 0.0
            thresholds.append(thr)

        thresholds = np.array(thresholds)[:, None]  # Reshape to (K, 1)
        # print(f"[LazySIInterface] Created {K} clusters and it has for {nums} samples.")
        # print(f"[LazySIInterface] Created {K} prototypes with thresholds: {thresholds} , {np.mean(thresholds)}")
        # print(f"[LazySIInterface] Created {K} prototypes with thresholds: {means} , {np.mean(means)}")
        # print(f"[LazySIInterface] Created {K} prototypes with thresholds: {medians} , {np.mean(medians)}")
        # print(f"[LazySIInterface] Created {K} prototypes with variances: {centroids.shape} , {np.mean(variances)}")
        return (centroids, variances, thresholds)

    # ----------------------------------
    # functions for policy side interface
    # ----------------------------------
    def create_subtask_prototype(self, dataloader):
        """
        Create policy prototypes using the configured policy clustering strategy, mirroring decoder map_entry.

        Steps:
        1. Cluster subgoals via self.policy_entry_generator.
        2. Populate dataloader with subgoal assignments.
        3. Build PolicyEntry instances for each cluster.
        """
        # 1. Perform clustering using the policy algorithm
        labels, centroid_map, extra = self.policy_entry_generator.cluster(dataloader)

        # 2. Attach subgoal assignments to dataloader
        data = dataloader.stacked_data
        obs = data['observations']
        T = len(obs)
        data['subgoals_policy'] = np.zeros_like(obs)
        for sg, t in zip(extra['subgoals'], extra['timesteps']): # NOTE  Validate
            data['subgoals_policy'][t] = sg

        # 3. Build prototypes per cluster
        subtask_prototypes = {}
        base_id = 0
        num_clusters = len(centroid_map)
        for cluster_idx in range(num_clusters):
            indices = np.where(labels == cluster_idx)[0]
            if len(indices) < self.config.subtask_prototype_bases:
                continue
            # Instantiate PolicyEntry
            if self.config.policy_algo == 'instance':
                # For instance-based, use the data points directly
                state_prototypes = self._create_prototypes(obs[indices], instance_based=True)
            else:
                # For other algorithms, use KMeans-based prototypes
                state_prototypes = self._create_prototypes(obs[indices], K=self.config.subtask_prototype_bases)
            proto = PolicyEntry(
                prototype_id=base_id,
                subgoal=centroid_map[cluster_idx],
                state_prototypes=state_prototypes,
                data_count=len(indices),
                #
                threshold_type=self.config.threshold_type,
                distance_type=self.config.distance_type,
                confidence_interval=self.config.confidence_interval,
            )
            subtask_prototypes[base_id] = proto
            base_id += 1

        # Save and return
        self.logger.info(f"Created {len(subtask_prototypes)} policy prototypes.")
        self.subtask_prototypes = subtask_prototypes
        return subtask_prototypes
    # ----------------------------------
    # Utill functions
    # ----------------------------------
    def update_subtask_prototype(self, subtask_prototype):
        if subtask_prototype is None:
            self.logger.warning("No policy prototype to update; None detected.")
        self.subtask_prototypes = deepcopy(subtask_prototype)
    
    # ----------------------------------
    # interface forward function.
    # ----------------------------------
    def forward(self, entry, current_state, static=False):
        """
        Forward pass for LazySIInterface using static and dynamic matching with Mahalanobis distance.
        1. find appropriate skill id from policy
        2. match the subgoal for the selected skill
        2-1. if the subgoal is matched, return the skill id
        2-2. if the subgoal is not matched, find the closest skill id from the decoder prototypes.
        Parameters:
        entry: A single value or an array (B,) of global skill ids.
        current_state: A (B, f) array of current state vectors.
        static: If True, always use static matching; (used in interface)
        
        Returns:
        A tuple ((skill_id, decoder_ids), skill_aux), where:
            - skill_id: the chosen skill id (static or dynamically re-assigned)
            - decoder_ids: corresponding decoder ids (static if dynamic matching is not activated; dynamic value otherwise)
            - skill_aux: associated auxiliary information (centroid) for the chosen skill.
        """
        # Convert entry to numpy array and ensure proper shape. 
        entry = np.array(entry, dtype=np.int32)

        if entry.ndim > 1 and entry.shape[-1] == 1:
            entry = np.squeeze(entry, axis=-1)

        if current_state.ndim == 1 :
            current_state = np.expand_dims(current_state, axis=0)
        
        # Determine batch size and observation dimension.
        if current_state.ndim > 1:
            B, obs_dim = current_state.shape
        else:
            B = 1
            obs_dim = current_state.shape[0]
        
        # For outputs.
        out_skill_ids = np.zeros((B,), dtype=np.int32)
        out_decoder_ids = np.zeros((B,), dtype=np.int32)

        if isinstance(self.decoder_entry_generator, SemanticClusteringStrategy):
            out_skill_aux = np.zeros((B, 512), dtype=np.float32) # hard coded NOTE
        else : 
            out_skill_aux = np.zeros((B, obs_dim), dtype=np.float32) 
        
        k = self.candidates if hasattr(self, 'candidates') else 1
         # (policy) number of candidates to select from policy prototypes.
        
        # Process each sample individually.
        for i in range(B):
            cs = current_state[i]  # current state vector for sample i
            orig_entry = int(entry[i])
            if orig_entry not in self.entry_skill_map:
                # self.logger.warning(f"Entry {orig_entry} not found in entry_skill_map; using random fallback.")
                orig_entry = int(np.random.randint(0, self.num_skills)) # Default fallback     
                
            # =============================
            # 1. Static Matching
            # =============================
            if static == True or self.config.force_static == True : 
                out_skill_ids[i] = orig_entry
                out_decoder_ids[i] = self.entry_skill_map[orig_entry].decoder_id
                out_skill_aux[i] = self.entry_skill_map[orig_entry].skill_aux
                continue

            # =============================
            # 2. Dynamic Matching - confidence check
            # =============================
            candidate_list = []
            for pid, policy_proto in self.subtask_prototypes.items():
                policy_mp = policy_proto.state_prototypes
                valid, d_policy_val = policy_mp.validate(cs)
                candidate_list.append((pid, d_policy_val, valid))
            
            # Sort candidates by distance and select top k. and filter out candidates that are not valid
            candidate_list.sort(key=lambda x: x[1])
            top_candidates = candidate_list[:k]
            top_candidates = [(pid, d_policy_val, valid) for pid, d_policy_val, valid in top_candidates if valid == True]
            if len(top_candidates) == 0:
                top_candidates = candidate_list[:k]
            if len(top_candidates) == 0:
                out_skill_ids[i] = orig_entry
                out_decoder_ids[i] = self.entry_skill_map[orig_entry].decoder_id
                out_skill_aux[i] = self.entry_skill_map[orig_entry].skill_aux
                continue

            # Calcluate current subgoal(objective) is affordable for the selected entry
            entry_obj = self.entry_skill_map[orig_entry]
            entry_valid_flag = False
            for pid, d_policy_val, _ in top_candidates:
                policy_candidate = self.subtask_prototypes[pid]
                candidate_subgoal = policy_candidate.subgoal  
                subgoal_valid, _ = entry_obj.subgoal_prototypes.validate(candidate_subgoal)
                if subgoal_valid:
                    entry_valid_flag = True
                    break

            if entry_valid_flag == True :
                out_skill_ids[i] = orig_entry
                out_decoder_ids[i] = self.entry_skill_map[orig_entry].decoder_id
                out_skill_aux[i] = self.entry_skill_map[orig_entry].skill_aux
                continue

            # =============================
            # 2. Dynamic Matching - find skill candidates 
            # =============================
            skill_candidate_ids = []

            for (pid, d_policy_val, _) in top_candidates:
                policy_candidate = self.subtask_prototypes[pid]
                candidate_subgoal = policy_candidate.subgoal 
                best_id = -1 
                best_dist = np.inf
                sub_skill_candidate = []
                for eid, entry_obj in self.entry_skill_map.items():
                    valid, dist = entry_obj.subgoal_prototypes.validate(candidate_subgoal)
                    if valid == True :
                        sub_skill_candidate.append(eid)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = eid
                if len(sub_skill_candidate) > 0:
                    skill_candidate_ids.extend(sub_skill_candidate)
                else:
                    skill_candidate_ids.append(best_id)
            
            skill_candidate_ids = list(set(skill_candidate_ids))
            # =============================
            # 2. Dynamic Matching - find the best skill id 
            # =============================
            best_total = np.inf
            best_skill_id = orig_entry  
            best_decoder_id = self.entry_skill_map[orig_entry].decoder_id
            best_aux = self.entry_skill_map[orig_entry].skill_aux
            for skill_cadnidate_id in skill_candidate_ids:
                candidate_entry_obj = self.entry_skill_map[skill_cadnidate_id]
                d_state = candidate_entry_obj.state_prototypes.forward(cs)
                total_distance = np.min(d_state)
                if total_distance < best_total:
                    best_total = total_distance
                    best_skill_id = candidate_entry_obj.skill_id
                    best_decoder_id = candidate_entry_obj.decoder_id
                    best_aux = candidate_entry_obj.skill_aux

            if self.debug == True :
                self.logger.debug(f"Dynamic matching: entry {orig_entry} -> skill {best_skill_id}, decoder {best_decoder_id}, total distance {best_total}")
                if best_skill_id == orig_entry:
                    self.logger.debug(f"Dynamic matching failed; falling back to static matching.")
            # Dynamic matching result.
            out_skill_ids[i] = best_skill_id
            out_decoder_ids[i] = best_decoder_id
            out_skill_aux[i] = best_aux 
        '''
        out skill aux shape is (B, obs_dim)
        ''' 
        return (out_skill_ids, out_decoder_ids), out_skill_aux

def test_lazysi_interface():
    """
    Test function for LazySIInterface.
    
    Steps:
      1. Load a dataset chunk using BaseDataloader and DEFAULT_DATASTREAM.
      2. Ensure necessary fields (e.g., 'terminals') exist.
      3. Update the interface (initializing entry mappings and prototypes).
      4. Print shapes of augmented and original observations.
      5. Display internal entry_skill_map information (global skill ID, decoder ID, data count).
      6. Select one test entry per unique global skill and print these test entries.
      7. For each test entry, call forward() and print the returned skill id and decoder id.
      8. Optionally, create policy prototypes and print the number created.
      9. Rollback the dataloader and print the restored observation shape.
    """
    import numpy as np
    from SILGym.dataset.dataloader import BaseDataloader
    from SILGym.config.skill_stream_config import DEFAULT_DATASTREAM
    from rich.console import Console
    from rich.table import Table
    console = Console()
    
    # Create an instance of LazySIInterface with default configuration.
    interface = LazySIInterface()

    # For testing, use the first dataset chunk from DEFAULT_DATASTREAM.
    chunk = DEFAULT_DATASTREAM[0]
    dataloader = BaseDataloader(data_paths=chunk.dataset_paths)
    
    # Ensure that 'terminals' exists; if not, create dummy terminal flags (assume last index is terminal).
    T = len(dataloader.stacked_data['observations'])
    if 'terminals' not in dataloader.stacked_data:
        dummy_terminals = np.zeros(T, dtype=np.int32)
        dummy_terminals[-1] = 1
        dataloader.stacked_data['terminals'] = dummy_terminals

    # Update the interface: initialize entry mappings and update the dataloader.
    dataloader = interface.update_interface(dataloader)
    
    console.print("[bold green]After update_interface:[/bold green]")
    console.print("Augmented observations shape:", dataloader.stacked_data['observations'].shape)
    console.print("Original observations (orig_obs) shape:", dataloader.stacked_data['orig_obs'].shape)

    # Display entry_skill_map information.
    table = Table(title="Entry Skill Map Information")
    table.add_column("Global Skill ID", style="cyan")
    table.add_column("Decoder ID", style="magenta")
    table.add_column("Data Count", style="yellow")
    for global_skill_id, entry_obj in interface.entry_skill_map.items():
        table.add_row(str(global_skill_id), str(entry_obj.decoder_id), str(entry_obj.data_count))
    console.print(table)
    
    # Prepare test entries by selecting the first index of each unique global skill from 'entry'.
    unique_skill_ids = np.unique(dataloader.stacked_data['entry'])
    test_indices = []
    for sid in unique_skill_ids:
        idxs = np.where(dataloader.stacked_data['entry'] == sid)[0]
        if len(idxs) > 0:
            test_indices.append(idxs[0])
    test_indices = np.array(test_indices)
    test_entries = dataloader.stacked_data['entry'][test_indices]
    orig_obs = dataloader.stacked_data['orig_obs']
    # Use the corresponding rows from orig_obs as the current state.
    current_state = orig_obs[test_indices]
    
    # Print test entries.
    table_test = Table(title="Test Entries")
    table_test.add_column("Index", style="cyan")
    table_test.add_column("Global Skill ID", style="magenta")
    for idx in test_indices:
        table_test.add_row(str(idx), str(dataloader.stacked_data['entry'][idx]))
    console.print(table_test)
    
    # Test the forward method.
    table_forward = Table(title="Forward Method Outputs")
    table_forward.add_column("Test Index", style="cyan")
    table_forward.add_column("Chosen Skill ID", style="magenta")
    table_forward.add_column("Decoder ID", style="green")
    for i, sid in enumerate(test_entries):
        # Call forward with the scalar entry and corresponding current state vector.
        (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([sid]), current_state[i])
        table_forward.add_row(str(i), str(f_skill[0]), str(f_decoder[0]))
    console.print(table_forward)
    
    # Optionally, create policy prototypes and print the number created.
    policy_protos = interface.create_subtask_prototype(dataloader)
    console.print(f"[bold green]Number of policy prototypes created: {len(policy_protos)}[/bold green]")
    
    # Optionally, display details of policy prototypes.
    table_policy = Table(title="Policy Prototypes Information")
    table_policy.add_column("Prototype ID", style="cyan")
    table_policy.add_column("Data Count", style="yellow")
    for proto_id, proto in policy_protos.items():
        table_policy.add_row(str(proto_id), str(proto.data_count))
    console.print(table_policy)
    
    # Rollback the dataloader to restore original observations.
    dataloader = interface.rollback_dataloader(dataloader)
    console.print("[bold green]After rollback_dataloader:[/bold green]",
                  "Restored observations shape:", dataloader.stacked_data['observations'].shape)

def test_lazysi_dynamic():
    import numpy as np
    from SILGym.dataset.dataloader import BaseDataloader
    from SILGym.config.skill_stream_config import DEFAULT_DATASTREAM
    from SILGym.config.mmworld_scenario import MMWORLD_SCENARIO_N1_SYNC
    from rich.console import Console
    from rich.table import Table
    # from SILGym.models.skill_interface.lazysi import LazySIInterface  # adjust import path if needed

    console = Console()
    interface = LazySIInterface()

    # Load decoder and policy data chunks.
    decoder_chunk = MMWORLD_SCENARIO_N1_SYNC[0]
    policy_chunk = MMWORLD_SCENARIO_N1_SYNC[10]

    # Update the decoder dataloader (this sets up the entry mappings and prototypes).
    decoder_dataloader = BaseDataloader(data_paths=decoder_chunk.dataset_paths)
    decoder_dataloader = interface.update_interface(decoder_dataloader) 

    console.print("[bold green]After update_interface:[/bold green]")
    table = Table(title="Entry Skill Map Information")
    table.add_column("Global Skill ID", style="cyan")
    table.add_column("Decoder ID", style="magenta")
    table.add_column("Data Count", style="yellow")

    for global_skill_id, entry_obj in interface.entry_skill_map.items():
        table.add_row(str(global_skill_id), str(entry_obj.decoder_id), str(entry_obj.data_count))
    console.print(table)

    # Create policy prototypes from the policy dataloader.
    policy_dataloader = BaseDataloader(data_paths=policy_chunk.dataset_paths)
    policy_protos = interface.create_subtask_prototype(policy_dataloader)
    console.print(f"[bold green]Number of policy prototypes created: {len(policy_protos)}[/bold green]")

    # ------------------------------
    # Modified sampling of test indices:
    # Instead of using a fixed number of random indices,
    # sample up to 3 indices per unique skill from the 'skills' field.
    # (Assuming that policy_dataloader.stacked_data contains a 'skills' key.)
    # ------------------------------
    unique_skills = np.unique(policy_dataloader.stacked_data['skills'])
    test_indicies = []  # List to store selected indices for each unique skill.
    for skill in unique_skills:
        # Obtain all indices corresponding to the current skill.
        skill_indices = np.where(policy_dataloader.stacked_data['skills'] == skill)[0]
        if len(skill_indices) >= 3:
            # Randomly sample 3 indices without replacement.
            sampled_indices = np.random.choice(skill_indices, 3, replace=False)
        else:
            # If there are fewer than 3 indices available, use all of them.
            sampled_indices = skill_indices
        test_indicies.extend(sampled_indices.tolist())
    test_indicies = np.array(test_indicies)

    # Use the selected indices to pick the current state (observations)
    # and corresponding skill entries for the forward method.
    current_state = policy_dataloader.stacked_data['observations'][test_indicies]
    test_entries = range(len(test_indicies))
    # Build a table to display the results of the forward method.
    table_forward = Table(title="Forward Method Outputs")  
    table_forward.add_column("Test Entries", style="cyan")
    table_forward.add_column("Chosen Skill ID", style="magenta")
    table_forward.add_column("Decoder ID", style="green")

    for i, sid in enumerate(test_entries):
        # Call forward with the selected skill entry and corresponding current state.
        (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([sid]), current_state[i])
        table_forward.add_row(str(test_entries[i]), str(f_skill[0]), str(f_decoder[0]))
    
    console.print(table_forward)

    policy_dataloader = interface.update_interface(policy_dataloader)
    
    entry_array = policy_dataloader.stacked_data['entry']
    console.print("[bold blue]Updated 'entry' field in policy_dataloader:[/bold blue]")
    console.print(entry_array)
    
    table_rematch = Table(title="Rematch Method Outputs")
    table_rematch.add_column("Test Index", style="cyan")
    table_rematch.add_column("Chosen Skill ID", style="magenta")
    table_rematch.add_column("Decoder ID", style="green")
    table_rematch.add_column("Entry (Global Skill ID)", style="magenta")
    table_rematch.add_column("Match?", style="magenta")
    table_rematch.add_column("Distance to Proto", style="red")

    # Rollback to original observations
    policy_dataloader = interface.rollback_dataloader(policy_dataloader)

    for idx in test_indicies:
        # Current observation for this index
        obs_vec = policy_dataloader.stacked_data['observations'][idx]
        # Use the original entry as input for forward()
        orig_entry = policy_dataloader.stacked_data['entry'][idx]
        (f_skill, f_decoder), _ = interface.forward(np.array([orig_entry]), obs_vec)
        matched = (f_skill[0] == orig_entry)

        # If mismatch, compute Mahalanobis distance to the true prototype
        if not matched:
            proto = interface.entry_skill_map[orig_entry].state_prototypes
            dists = proto.forward(obs_vec)
            dist_val = float(dists.min())
        else:
            dist_val = ""

        table_rematch.add_row(
            str(idx),
            str(f_skill[0]),
            str(f_decoder[0]),
            str(orig_entry),
            str(matched),
            str(dist_val)
        )


    console.print(table_rematch)
    

    accuracy = 0
    total_same = 0
    from tqdm import tqdm
    for i, index in tqdm(enumerate(entry_array)):
        (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([3]), policy_dataloader.stacked_data['observations'][i])
        # (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([policy_dataloader.stacked_data['entry'][i]]), policy_dataloader.stacked_data['observations'][i])
        if f_skill[0] == policy_dataloader.stacked_data['entry'][i]:
            total_same += 1
    accuracy = total_same / len(entry_array)

    console.print(f"[bold green]Accuracy of rematching: {accuracy}[/bold green]")

# Run test if executed as a script.
if __name__ == "__main__":
    # test_lazysi_interface()
    test_lazysi_dynamic()   
