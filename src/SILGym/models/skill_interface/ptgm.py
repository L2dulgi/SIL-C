import numpy as np
from SILGym.utils.cuml_wrapper import TSNE, KMeans, UMAP, MiniBatchKMeans, HDBSCAN
import matplotlib.pyplot as plt
from SILGym.models.skill_interface.base import BaseInterface
from SILGym.utils.logger import get_logger

class PTGMInterfaceConfig:
    def __init__(
        self,
        cluster_num=5,
        goal_offset=5,
        tsne_dim=2,
        tsne_perplexity=30,
        # PTGM+ parameters for handling large datasets
        use_ptgm_plus=False,
        sampling_ratio=10.0,
        precluster_method='minibatch_kmeans',
        # Embedding method selection
        embedding_method='tsne',
        embedding_dim=2,
        # UMAP parameters
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric='euclidean',
        # HDBSCAN parameters (used when embedding_method='umap')
        use_hdbscan_with_umap=True,
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=None,
        hdbscan_cluster_selection_epsilon=0.0,
        hdbscan_cluster_selection_method='eom'
    ):
        """
        PTGM Interface Configuration.

        Args:
            cluster_num: Number of final skill clusters
            goal_offset: Steps before terminal state to consider as subgoals
            tsne_dim: T-SNE embedding dimensions (deprecated, use embedding_dim)
            tsne_perplexity: T-SNE perplexity parameter
            use_ptgm_plus: Enable PTGM+ algorithm for large datasets (>100k samples)
            sampling_ratio: Reduction ratio for pre-clustering (e.g., 10.0 = reduce to 1/10)
            precluster_method: Pre-clustering algorithm choice:
                - 'minibatch_kmeans': Fast, good for most cases (default)
                - 'birch': Memory-efficient streaming algorithm for very large datasets
                           Uses adaptive threshold scaling and MiniBatchKMeans for final clustering
                - 'kmeans': Standard KMeans (for small datasets)
            embedding_method: Dimensionality reduction method:
                - 'tsne': T-SNE (default) - uses KMeans clustering
                - 'umap': UMAP (faster, preserves global structure) - uses HDBSCAN clustering by default
                - 'none': No dimensionality reduction - uses KMeans clustering
            embedding_dim: Output dimensions for embedding (default: 2)
            umap_n_neighbors: UMAP n_neighbors parameter (default: 15)
            umap_min_dist: UMAP min_dist parameter (default: 0.1)
            umap_metric: UMAP distance metric (default: 'euclidean')
            use_hdbscan_with_umap: Use HDBSCAN instead of KMeans when embedding_method='umap' (default: True)
            hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN (default: 5)
            hdbscan_min_samples: Minimum samples for core points in HDBSCAN (default: None, uses min_cluster_size)
            hdbscan_cluster_selection_epsilon: Distance threshold for cluster selection (default: 0.0)
            hdbscan_cluster_selection_method: 'eom' (Excess of Mass) or 'leaf' (default: 'eom')

        Performance comparison (50k samples):
            - minibatch_kmeans: ~32s, higher memory
            - birch: ~60s, streaming/lower memory (adaptive threshold prevents OOM)

        Recommendation:
            - Use 'minibatch_kmeans' for datasets < 500k samples
            - Use 'birch' for datasets > 500k samples or memory-constrained environments
            - Use 'umap' with HDBSCAN for discovering clusters of arbitrary shapes
        """
        self.cluster_num = cluster_num
        self.goal_offset = goal_offset
        # Handle backward compatibility: if tsne_dim=0, set embedding_method='none'
        if tsne_dim == 0:
            embedding_method = 'none'
            embedding_dim = 0
        self.tsne_dim = tsne_dim  # Keep for backward compatibility
        self.tsne_perplexity = tsne_perplexity
        # PTGM+ parameters
        self.use_ptgm_plus = use_ptgm_plus
        self.sampling_ratio = sampling_ratio
        self.precluster_method = precluster_method
        # Embedding parameters
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
        # UMAP parameters
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        # HDBSCAN parameters
        self.use_hdbscan_with_umap = use_hdbscan_with_umap
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method

    def to_dict(self):
        return {
            'cluster_num': self.cluster_num,
            'goal_offset': self.goal_offset,
            'tsne_dim': self.tsne_dim,
            'tsne_perplexity': self.tsne_perplexity,
            'use_ptgm_plus': self.use_ptgm_plus,
            'sampling_ratio': self.sampling_ratio,
            'precluster_method': self.precluster_method,
            'embedding_method': self.embedding_method,
            'embedding_dim': self.embedding_dim,
            'umap_n_neighbors': self.umap_n_neighbors,
            'umap_min_dist': self.umap_min_dist,
            'umap_metric': self.umap_metric,
            'use_hdbscan_with_umap': self.use_hdbscan_with_umap,
            'hdbscan_min_cluster_size': self.hdbscan_min_cluster_size,
            'hdbscan_min_samples': self.hdbscan_min_samples,
            'hdbscan_cluster_selection_epsilon': self.hdbscan_cluster_selection_epsilon,
            'hdbscan_cluster_selection_method': self.hdbscan_cluster_selection_method,
        }

class PTGMInterface(BaseInterface):
    def __init__(self, ptgm_config: PTGMInterfaceConfig):
        """
        ptgm_config must be an instance of PTGMInterfaceConfig.
        """
        super().__init__()
        self.logger = get_logger(__name__)
        config = ptgm_config.to_dict()
        self.cluster_num = config.get('cluster_num', 10)
        self.goal_offset = config.get('goal_offset', 10)
        self.tsne_dim = config.get('tsne_dim', 2)
        self.tsne_perplexity = config.get('tsne_perplexity', 30)
        # PTGM+ parameters
        self.use_ptgm_plus = config.get('use_ptgm_plus', False)
        self.sampling_ratio = config.get('sampling_ratio', 10.0)
        self.precluster_method = config.get('precluster_method', 'minibatch_kmeans')
        # Embedding parameters
        self.embedding_method = config.get('embedding_method', 'tsne')
        self.embedding_dim = config.get('embedding_dim', 2)
        # UMAP parameters
        self.umap_n_neighbors = config.get('umap_n_neighbors', 15)
        self.umap_min_dist = config.get('umap_min_dist', 0.1)
        self.umap_metric = config.get('umap_metric', 'euclidean')
        # HDBSCAN parameters
        self.use_hdbscan_with_umap = config.get('use_hdbscan_with_umap', True)
        self.hdbscan_min_cluster_size = config.get('hdbscan_min_cluster_size', 5)
        self.hdbscan_min_samples = config.get('hdbscan_min_samples', None)
        self.hdbscan_cluster_selection_epsilon = config.get('hdbscan_cluster_selection_epsilon', 0.0)
        self.hdbscan_cluster_selection_method = config.get('hdbscan_cluster_selection_method', 'eom')
        self.kmeans = None
        self.centroid_map = None  # Map from cluster ID to the original subgoal (observation)
        self._tsne_data = None   # Store embedding results for debugging/plotting

    def init_entry(self, dataloader):
        """
        Create (or reset) 'entry', 'skill_id', 'skill_aux' and 'decoder_id' in dataloader.stacked_data if they do not exist.
        """
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

    def map_entry(self, dataloader):
        """
        1) For each timestep, collect subgoals:
        - If t + goal_offset is within the trajectory, subgoal = observations[t + goal_offset]
        - Otherwise, subgoal = observations[end_of_trajectory]
        2) Optionally apply T-SNE to the subgoals if tsne_dim > 0.
        3) Run K-Means on the chosen embedding space.
        4) For each cluster center, find the closest subgoal in that space and use its original observation as the centroid.
        5) Assign skill_id, entry, and skill_aux to each timestep based on the cluster ID.
        """
        # Check if PTGM+ should be used for large datasets
        if self.use_ptgm_plus:
            n_samples = len(dataloader.stacked_data['observations'])
            self.logger.info(f"Using PTGM+ algorithm for {n_samples} samples")
            return self.map_entry_plus(dataloader)

        # Original PTGM algorithm
        stacked_data = dataloader.stacked_data
        observations = stacked_data['observations']
        terminals = stacked_data['terminals']

        # Split data into trajectories based on terminals
        traj_indices = self._split_by_trajectory(terminals)
        all_subgoals = []
        all_subgoal_ts = []  # To remap cluster IDs to the correct timesteps.

        for (start_idx, end_idx) in traj_indices:
            for t in range(start_idx, end_idx + 1):
                future_t = t + self.goal_offset
                sg = observations[future_t] if future_t <= end_idx else observations[end_idx]
                all_subgoals.append(sg)
                all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)

        # 2) Apply dimensionality reduction based on embedding_method
        embedding_method = getattr(self, 'embedding_method', 'tsne')

        # Backward compatibility: if tsne_dim=0, treat as 'none'
        if getattr(self, 'tsne_dim', 0) == 0:
            embedding_method = 'none'

        if embedding_method == 'tsne':
            self.logger.info(f"Applying T-SNE embedding (dim={self.embedding_dim}, perplexity={self.tsne_perplexity})")
            tsne = TSNE(
                n_components=self.embedding_dim,
                perplexity=self.tsne_perplexity,
                random_state=0
            )
            subgoals_emb = tsne.fit_transform(all_subgoals)
        elif embedding_method == 'umap':
            # UMAP is already imported from cuml_wrapper at the top
            self.logger.info(f"Applying UMAP embedding (dim={self.embedding_dim}, n_neighbors={self.umap_n_neighbors}, min_dist={self.umap_min_dist})")
            umap_reducer = UMAP(
                n_components=self.embedding_dim,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric=self.umap_metric,
                random_state=0
            )
            subgoals_emb = umap_reducer.fit_transform(all_subgoals)
        else:  # 'none'
            self.logger.info("No dimensionality reduction applied, using original subgoals.")
            subgoals_emb = all_subgoals


        # 3) Run clustering on the chosen embedding
        # Use HDBSCAN for UMAP if enabled, otherwise use KMeans
        use_hdbscan = (embedding_method == 'umap' and self.use_hdbscan_with_umap)

        if use_hdbscan:
            self.logger.info(f"Applying HDBSCAN clustering (min_cluster_size={self.hdbscan_min_cluster_size})")
            hdbscan = HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
                cluster_selection_method=self.hdbscan_cluster_selection_method,
                metric=self.umap_metric
            )
            cluster_ids = hdbscan.fit_predict(subgoals_emb)

            # Handle noise points (-1 label) and compute cluster centers
            unique_clusters = np.unique(cluster_ids)
            unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove noise label (-1)
            n_clusters_found = len(unique_clusters)

            self.logger.info(f"HDBSCAN found {n_clusters_found} clusters (noise points: {np.sum(cluster_ids == -1)})")

            # Compute cluster centers in embedding space
            cluster_centers_emb = np.zeros((n_clusters_found, subgoals_emb.shape[1]))
            cluster_id_map = {}  # Map from original cluster ID to new sequential ID
            for new_id, orig_id in enumerate(unique_clusters):
                mask = cluster_ids == orig_id
                cluster_centers_emb[new_id] = np.mean(subgoals_emb[mask], axis=0)
                cluster_id_map[orig_id] = new_id

            # Reassign noise points to nearest cluster
            if np.any(cluster_ids == -1):
                noise_mask = cluster_ids == -1
                noise_indices = np.where(noise_mask)[0]
                for idx in noise_indices:
                    dists = np.sum((cluster_centers_emb - subgoals_emb[idx]) ** 2, axis=1)
                    nearest_cluster = np.argmin(dists)
                    cluster_ids[idx] = unique_clusters[nearest_cluster]
                self.logger.info(f"Reassigned {len(noise_indices)} noise points to nearest clusters")

            # Remap cluster IDs to sequential 0, 1, 2, ...
            remapped_cluster_ids = np.zeros_like(cluster_ids)
            for i, cid in enumerate(cluster_ids):
                remapped_cluster_ids[i] = cluster_id_map[cid]
            cluster_ids = remapped_cluster_ids

            # Update cluster_num to actual number found
            actual_cluster_num = n_clusters_found
        else:
            self.logger.info(f"Applying KMeans clustering into {self.cluster_num} clusters")
            self.kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
            self.kmeans.fit(subgoals_emb)
            cluster_ids = self.kmeans.predict(subgoals_emb)
            cluster_centers_emb = self.kmeans.cluster_centers_
            actual_cluster_num = self.cluster_num

        self.logger.info("Mapping cluster centers back to original subgoals")
        # 4) Map cluster centers back to actual subgoals in original space
        centroid_map = {}
        for c in range(actual_cluster_num):
            center = cluster_centers_emb[c]
            dists = np.sum((subgoals_emb - center) ** 2, axis=1)
            closest_idx = np.argmin(dists)
            centroid_map[c] = all_subgoals[closest_idx]
        self.centroid_map = centroid_map

        # Save embedding data for optional plotting
        self._tsne_data = {
            'subgoals_emb': subgoals_emb,
            'cluster_ids': cluster_ids,
            'cluster_centers_emb': cluster_centers_emb
        }

        # 5) Assign cluster info to each timestep
        stacked_data['skill_id'][:] = -1
        stacked_data['entry'][:] = -1
        stacked_data['skill_aux'][:] = 0.0
        for i, t in enumerate(all_subgoal_ts):
            cid = cluster_ids[i]
            stacked_data['skill_id'][t] = cid
            stacked_data['entry'][t] = cid
            stacked_data['skill_aux'][t] = centroid_map[cid]

    def map_entry_plus(self, dataloader):
        """
        PTGM+ version of map_entry() that uses pre-clustering to handle large datasets efficiently.

        Algorithm:
        1) Collect all subgoals (same as original)
        2) Pre-cluster subgoals to reduce sample count
        3) Apply T-SNE only on representative samples
        4) Run K-Means on T-SNE embeddings
        5) Assign original subgoals via pre-cluster mapping
        """
        stacked_data = dataloader.stacked_data
        observations = stacked_data['observations']
        terminals = stacked_data['terminals']

        # Step 1: Collect all subgoals (same as original map_entry)
        traj_indices = self._split_by_trajectory(terminals)
        all_subgoals = []
        all_subgoal_ts = []

        for (start_idx, end_idx) in traj_indices:
            for t in range(start_idx, end_idx + 1):
                future_t = t + self.goal_offset
                sg = observations[future_t] if future_t <= end_idx else observations[end_idx]
                all_subgoals.append(sg)
                all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)

        n_samples = len(all_subgoals)
        self.logger.info(f"PTGM+: Processing {n_samples} subgoals with sampling_ratio={self.sampling_ratio}")

        # Step 2: Pre-cluster to get representatives
        representatives, precluster_assignments, precluster_to_rep_idx = self._precluster_subgoals(
            all_subgoals, target_size=self.cluster_num * 10
        )

        # Step 3: Apply dimensionality reduction on representatives
        embedding_method = getattr(self, 'embedding_method', 'tsne')

        # Backward compatibility: if tsne_dim=0, treat as 'none'
        if getattr(self, 'tsne_dim', 0) == 0:
            embedding_method = 'none'

        n_reps = len(representatives)

        if embedding_method == 'tsne':
            # Adjust perplexity if needed (must be less than n_samples)
            effective_perplexity = min(self.tsne_perplexity, (n_reps - 1) // 3)
            effective_perplexity = max(5, effective_perplexity)

            self.logger.info(f"PTGM+: Applying T-SNE on {n_reps} representatives (dim={self.embedding_dim}, perplexity={effective_perplexity})")
            tsne = TSNE(
                n_components=self.embedding_dim,
                perplexity=effective_perplexity,
                random_state=0
            )
            representatives_emb = tsne.fit_transform(representatives)
        elif embedding_method == 'umap':
            # UMAP is already imported from cuml_wrapper at the top
            # Adjust n_neighbors if needed (must be less than n_samples)
            effective_n_neighbors = min(self.umap_n_neighbors, n_reps - 1)
            effective_n_neighbors = max(2, effective_n_neighbors)

            self.logger.info(f"PTGM+: Applying UMAP on {n_reps} representatives (dim={self.embedding_dim}, n_neighbors={effective_n_neighbors})")
            umap_reducer = UMAP(
                n_components=self.embedding_dim,
                n_neighbors=effective_n_neighbors,
                min_dist=self.umap_min_dist,
                metric=self.umap_metric,
                random_state=0
            )
            representatives_emb = umap_reducer.fit_transform(representatives)
        else:  # 'none'
            self.logger.info("PTGM+: No dimensionality reduction applied, using original representatives")
            representatives_emb = representatives

        # Step 4: Run clustering on embeddings of representatives
        # Use HDBSCAN for UMAP if enabled, otherwise use KMeans
        use_hdbscan = (embedding_method == 'umap' and self.use_hdbscan_with_umap)

        if use_hdbscan:
            self.logger.info(f"PTGM+: Applying HDBSCAN clustering (min_cluster_size={self.hdbscan_min_cluster_size})")
            hdbscan = HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
                cluster_selection_method=self.hdbscan_cluster_selection_method,
                metric=self.umap_metric
            )
            rep_cluster_ids = hdbscan.fit_predict(representatives_emb)

            # Handle noise points (-1 label) and compute cluster centers
            unique_clusters = np.unique(rep_cluster_ids)
            unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove noise label (-1)
            n_clusters_found = len(unique_clusters)

            self.logger.info(f"PTGM+: HDBSCAN found {n_clusters_found} clusters (noise points: {np.sum(rep_cluster_ids == -1)})")

            # Compute cluster centers in embedding space
            cluster_centers_emb = np.zeros((n_clusters_found, representatives_emb.shape[1]))
            cluster_id_map = {}  # Map from original cluster ID to new sequential ID
            for new_id, orig_id in enumerate(unique_clusters):
                mask = rep_cluster_ids == orig_id
                cluster_centers_emb[new_id] = np.mean(representatives_emb[mask], axis=0)
                cluster_id_map[orig_id] = new_id

            # Reassign noise points to nearest cluster
            if np.any(rep_cluster_ids == -1):
                noise_mask = rep_cluster_ids == -1
                noise_indices = np.where(noise_mask)[0]
                for idx in noise_indices:
                    dists = np.sum((cluster_centers_emb - representatives_emb[idx]) ** 2, axis=1)
                    nearest_cluster = np.argmin(dists)
                    rep_cluster_ids[idx] = unique_clusters[nearest_cluster]
                self.logger.info(f"PTGM+: Reassigned {len(noise_indices)} noise points to nearest clusters")

            # Remap cluster IDs to sequential 0, 1, 2, ...
            remapped_rep_cluster_ids = np.zeros_like(rep_cluster_ids)
            for i, cid in enumerate(rep_cluster_ids):
                remapped_rep_cluster_ids[i] = cluster_id_map[cid]
            rep_cluster_ids = remapped_rep_cluster_ids

            # Update cluster_num to actual number found
            actual_cluster_num = n_clusters_found
        else:
            self.logger.info(f"PTGM+: Applying KMeans clustering into {self.cluster_num} clusters")
            self.kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
            self.kmeans.fit(representatives_emb)
            rep_cluster_ids = self.kmeans.predict(representatives_emb)
            cluster_centers_emb = self.kmeans.cluster_centers_
            actual_cluster_num = self.cluster_num

        # Step 5: Map cluster centers back to actual subgoals in original space
        centroid_map = {}
        for c in range(actual_cluster_num):
            # Find closest representative to this cluster center
            center = cluster_centers_emb[c]
            dists = np.sum((representatives_emb - center) ** 2, axis=1)
            closest_rep_idx = np.argmin(dists)
            # Use the original representative as centroid
            centroid_map[c] = representatives[closest_rep_idx]
        self.centroid_map = centroid_map

        # Step 6: Assign original subgoals to final clusters via pre-cluster mapping
        # Each original subgoal -> pre-cluster -> representative -> final cluster
        # Use the mapping to handle empty preclusters correctly
        final_cluster_ids = np.array([
            rep_cluster_ids[precluster_to_rep_idx[pc_id]]
            for pc_id in precluster_assignments
        ])

        # Save embedding data for optional plotting
        self._tsne_data = {
            'subgoals_emb': representatives_emb,
            'cluster_ids': rep_cluster_ids,
            'cluster_centers_emb': cluster_centers_emb
        }

        # Step 7: Assign cluster info to each timestep
        stacked_data['skill_id'][:] = -1
        stacked_data['entry'][:] = -1
        stacked_data['skill_aux'][:] = 0.0
        for i, t in enumerate(all_subgoal_ts):
            cid = final_cluster_ids[i]
            stacked_data['skill_id'][t] = cid
            stacked_data['entry'][t] = cid
            stacked_data['skill_aux'][t] = centroid_map[cid]

        self.logger.info(f"PTGM+: Assigned {n_samples} subgoals to {actual_cluster_num} final clusters")

    def plot_tsne_clusters(self, save_path=None):
        """
        Visualize the subgoal clusters in the T-SNE space.
        If save_path is provided, the plot will be saved; otherwise, it will be displayed.
        """
        if not self._tsne_data:
            self.logger.warning("No T-SNE data available. Run map_entry or update_interface first.")
            return
        subgoals_tsne = self._tsne_data['subgoals_tsne']
        cluster_ids = self._tsne_data['cluster_ids']
        cluster_centers_tsne = self._tsne_data['cluster_centers_tsne']

        plt.figure(figsize=(8, 6))
        plt.scatter(subgoals_tsne[:, 0], subgoals_tsne[:, 1],
                    c=cluster_ids, cmap='tab10', alpha=0.7, label='Subgoals')
        plt.scatter(cluster_centers_tsne[:, 0], cluster_centers_tsne[:, 1],
                    s=200, c='red', marker='*', edgecolors='black', linewidths=1, label='Centroids')
        plt.title("T-SNE Subgoal Clusters")
        plt.legend(loc='best')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"T-SNE cluster plot saved to: {save_path}")
        else:
            plt.show()

    def forward(self, entry):
        """
        Forward pass that takes a batch of entries (cluster IDs) and returns a dict with:
            - 'skill_id': the cluster IDs.
            - 'skill_aux': the corresponding centroids from the original space.
        """
        if self.centroid_map is None:
            raise ValueError("centroid_map is not initialized. Run update_interface first.")
        batch_size = len(entry)
        skill_id = np.array(entry, dtype=np.int32)
        example_centroid = next(iter(self.centroid_map.values()))
        obs_dim = example_centroid.shape[0]
        skill_aux = np.zeros((batch_size, obs_dim), dtype=np.float32)
        for i, e in enumerate(entry):
            skill_aux[i] = self.centroid_map.get(e, 0.0)
        return skill_id, skill_aux

    def _precluster_subgoals(self, subgoals, target_size):
        """
        PTGM+ Pre-clustering step: reduce the number of subgoals by clustering
        similar subgoals together and selecting cluster centroids as representatives.

        Args:
            subgoals: np.array of shape (N, D) - all subgoals
            target_size: int - target number of representatives

        Returns:
            representatives: np.array of shape (M, D) - cluster centroids as representatives
            precluster_assignments: np.array of shape (N,) - assignment of each subgoal to a pre-cluster
        """
        # MiniBatchKMeans and KMeans are already imported from cuml_wrapper at the top
        # Birch is only available in sklearn (not in cuML)
        from sklearn.cluster import Birch

        n_samples = len(subgoals)
        n_preclusters = max(int(n_samples / self.sampling_ratio), target_size)
        n_preclusters = min(n_preclusters, n_samples)  # Can't have more clusters than samples

        self.logger.info(f"PTGM+: Pre-clustering {n_samples} subgoals into {n_preclusters} groups using {self.precluster_method}")

        # Choose clustering method based on configuration
        if self.precluster_method == 'birch':
            # BIRCH is memory-efficient and good for large datasets
            # Uses CF-Trees (Clustering Feature Trees) for hierarchical clustering
            # We use MiniBatchKMeans for final clustering to avoid memory issues with large subcluster counts
            # Note: AgglomerativeClustering requires O(n²) memory for distance matrix, which can be prohibitive

            # Adaptive threshold: larger datasets need larger thresholds to avoid too many subclusters
            # Rule of thumb: threshold should scale with sqrt(n_samples) to keep subcluster count manageable
            adaptive_threshold = max(0.5, min(5.0, 0.5 * np.sqrt(n_samples / 10000)))

            precluster_model = Birch(
                n_clusters=MiniBatchKMeans(
                    n_clusters=n_preclusters,
                    random_state=0,
                    batch_size=min(1024, n_preclusters * 10)
                ),
                threshold=adaptive_threshold,  # Adaptive threshold based on dataset size
                branching_factor=50  # Max number of subclusters in each node
            )
            self.logger.info(f"BIRCH using adaptive threshold={adaptive_threshold:.3f} for {n_samples} samples")
        elif self.precluster_method == 'minibatch_kmeans' and n_samples > 10000:
            # MiniBatchKMeans is faster for large datasets
            precluster_model = MiniBatchKMeans(
                n_clusters=n_preclusters,
                random_state=0,
                batch_size=min(1024, n_samples),
                max_iter=100
            )
        else:
            # Regular KMeans for smaller datasets
            precluster_model = KMeans(
                n_clusters=n_preclusters,
                random_state=0,
                max_iter=100
            )

        # Fit and predict
        precluster_assignments = precluster_model.fit_predict(subgoals)

        # Get cluster centers by computing centroids of each cluster
        # Build mapping from precluster_id -> representative_index
        representatives = []
        precluster_to_rep_idx = {}  # Maps precluster ID to representative array index

        for cluster_id in range(n_preclusters):
            mask = precluster_assignments == cluster_id
            if np.sum(mask) > 0:
                cluster_centroid = np.mean(subgoals[mask], axis=0)
                rep_idx = len(representatives)  # Index in representatives array
                precluster_to_rep_idx[cluster_id] = rep_idx
                representatives.append(cluster_centroid)

        representatives = np.array(representatives)

        self.logger.info(f"PTGM+: Reduced to {len(representatives)} representatives (from {n_preclusters} preclusters)")

        return representatives, precluster_assignments, precluster_to_rep_idx

    def _split_by_trajectory(self, terminals):
        """
        Given a terminals array (0/1) of length T, return a list of (start_idx, end_idx) tuples for each trajectory.
        """
        traj_indices = []
        start_idx = 0
        for i, val in enumerate(terminals):
            if val == 1:
                traj_indices.append((start_idx, i))
                start_idx = i + 1
        return traj_indices

class ContinualPTGMInterface(PTGMInterface):
    """
    A version of PTGMInterface that supports either expanding the global map of clusters
    or overwriting it entirely at each new data chunk, depending on `update_mode`.

    - 'expand': Continues adding new clusters to a global map, offsetting IDs to avoid collisions.
    - 'overwrite': Each new chunk completely replaces the global map and all old clusters.
    """
    def __init__(
            self, 
            ptgm_config: PTGMInterfaceConfig, 
            update_mode: str = 'expand',
        ):
        super().__init__(ptgm_config)
        self.logger = get_logger(__name__)
        self.cluster_offset = 0
        self.global_centroid_map = {}  # Global mapping from cluster ID to centroid
        self.num_skills = 0  # Total number of global skills/clusters
        # update_mode can be 'expand' or 'overwrite'
        self.update_mode = update_mode
        if update_mode not in ['expand', 'overwrite']:
            raise ValueError("Invalid update_mode. Must be 'expand' or 'overwrite'.")
     
        self.entry_decoder_map = {}  # Map entry id to version of skill trained on
        self.decoder_id = 0

    def update_interface(self, dataloader):
        """
        Process a data chunk and integrate with the global mapping.

        If update_mode == 'expand':
          - The local centroid IDs from `map_entry` are offset by `cluster_offset`.
          - The global_centroid_map is extended with these new clusters.

        If update_mode == 'overwrite':
          - The global map is cleared, the local IDs become the global IDs with no offset.
          - The global_centroid_map is replaced entirely by the new centroids.

        In either case, `update_dataloader` is then applied to augment the dataloader.

        Returns the updated dataloader.
        """
        self.init_entry(dataloader)
        self.map_entry(dataloader)

        # -- Case 1: expand mode --
        if self.update_mode == 'expand':
            local_map = {}
            for local_id, centroid in self.centroid_map.items():
                global_id = local_id + self.cluster_offset
                local_map[local_id] = global_id
                self.global_centroid_map[global_id] = centroid

            self._rewrite_global_ids(dataloader, local_map)
            self.cluster_offset += self.cluster_num
            self.num_skills = len(self.global_centroid_map)

        # -- Case 2: overwrite mode --
        elif self.update_mode == 'overwrite':
            # Clear everything global, reset offsets
            self.global_centroid_map.clear()    
            self.cluster_offset = 0

            # The new local IDs become the global IDs as-is (offset = 0)
            local_map = {}
            for local_id, centroid in self.centroid_map.items():
                local_map[local_id] = local_id
                self.global_centroid_map[local_id] = centroid

            # Update skill_id and entry with local_map (effectively no offset)
            self._rewrite_global_ids(dataloader, local_map)
            # Number of skills is simply the cluster_num now
            self.num_skills = self.cluster_num

        return self.update_dataloader(dataloader)

    def update_dataloader(self, dataloader):
        """
        For PTGM training, store original observations, then
        concatenates 'skill_aux' to 'observations'.
        """
        dataloader.stacked_data['orig_obs'] = dataloader.stacked_data['observations'].copy()
        dataloader.stacked_data['observations'] = np.concatenate(
            (dataloader.stacked_data['observations'], dataloader.stacked_data['skill_aux']), axis=-1
        )
        return dataloader

    def rollback_dataloader(self, dataloader):
        """
        Rollback the dataloader to its original state before the update_dataloader call.
        """
        if 'orig_obs' not in dataloader.stacked_data:
            return dataloader
        dataloader.stacked_data['observations'] = dataloader.stacked_data['orig_obs']
        del dataloader.stacked_data['orig_obs']
        return dataloader

    def _rewrite_global_ids(self, dataloader, local_to_global_map):
        """
        Rewrite 'skill_id' and 'entry' from local IDs to global IDs using local_to_global_map.
        Additionally, update (or "map") any already expanded entry IDs to the current version of the interface.
        """
        stacked_data = dataloader.stacked_data
        skill_id_arr = stacked_data['skill_id']
        entry_arr = stacked_data['entry']
        for i in range(len(skill_id_arr)):
            current_id = skill_id_arr[i]
            if current_id in local_to_global_map:
                new_id = local_to_global_map[current_id]
                skill_id_arr[i] = new_id
                entry_arr[i] = new_id
                self.entry_decoder_map[new_id] = self.decoder_id

            else:
                skill_id_arr[i] = -1
                entry_arr[i] = -1

        # Update the dataloader with the new skill_id and entry arrays
        T = len(stacked_data['observations'])
        stacked_data['decoder_id'] = np.full((T,), self.decoder_id ,dtype=np.int32)

        # after updating, set the new version of the decoder
        self.decoder_id += 1

    def forward(self, entry):
        """
        Forward pass for global cluster IDs:
         - 'skill_id': same shape as entry. 
         - 'decoder_id': the version of the skill.
         - 'skill_aux': the corresponding centroid from global_centroid_map.
        """
        if not self.global_centroid_map:
            raise ValueError("No global centroids available. Run update_interface first.")

        entry = np.array(entry, dtype=np.int32)

        # If the last dimension is 1, remove it for convenience
        if entry.ndim > 1 and entry.shape[-1] == 1:
            entry = np.squeeze(entry, axis=-1)

        # Case 1: entry is a 1D array of global IDs
        if entry.ndim == 1:
            B = entry.shape[0]
            example_centroid = next(iter(self.global_centroid_map.values()))
            obs_dim = example_centroid.shape[0]
            skill_id = entry.copy()
            skill_aux = np.zeros((B, obs_dim), dtype=np.float32)
            for i, e in enumerate(entry):
                skill_aux[i] = self.global_centroid_map.get(int(e),
                                    np.zeros(obs_dim, dtype=np.float32))
                
            decoder_id = np.zeros((B,), dtype=np.int32)
            for i in range(B):
                if skill_id[i] in self.entry_decoder_map:
                    decoder_id[i] = self.entry_decoder_map[skill_id[i]]
                else:
                    decoder_id[i] = 0 # Default fallback for novel skill. -> Basemodel
            return (skill_id, decoder_id), skill_aux

        # Case 2: entry is a 2D array (e.g. shape (B, num_skills))
        elif entry.ndim == 2:
            B, num_skills = entry.shape
            example_centroid = next(iter(self.global_centroid_map.values()))
            obs_dim = example_centroid.shape[0]
            skill_id = entry.copy()
            skill_aux = np.zeros((B, num_skills, obs_dim), dtype=np.float32)
            for i in range(B):
                for j in range(num_skills):
                    global_id = int(entry[i, j])
                    skill_aux[i, j] = self.global_centroid_map.get(global_id,
                                             np.zeros(obs_dim, dtype=np.float32))

            decoder_id = np.zeros((B, num_skills), dtype=np.int32)
            for i in range(B):
                for j in range(num_skills):
                    if skill_id[i, j] in self.entry_decoder_map:
                        decoder_id[i, j] = self.entry_decoder_map[skill_id[i, j]]
                    else:
                        decoder_id[i, j] = 0  # default fallback
            
            return (skill_id, decoder_id), skill_aux

        else:
            raise ValueError("Unsupported entry shape in forward")


# ----------------------------------------------------------------------------- 
# Testing block for ContinualPTGMInterface using update_interface 
# for both 'expand' and 'overwrite' modes across data chunks.

if __name__ == '__main__':
    import os
    # Import necessary modules and classes.
    from SILGym.dataset.dataloader import BaseDataloader
    from SILGym.config.skill_stream_config import DEFAULT_DATASTREAM
    from SILGym.models.skill_interface.ptgm import ContinualPTGMInterface, PTGMInterfaceConfig

    # We'll test with two modes: 'expand' and 'overwrite'.
    for mode in ['overwrite','expand']:
        # Create a list of dataloaders for each data chunk in DEFAULT_DATASTREAM.
        dataloaders = []
        for ds in DEFAULT_DATASTREAM:
            dataloaders.append(BaseDataloader(data_paths=ds.dataset_paths))
        
        # If only one dataset is available, duplicate it for demonstration purposes.
        if len(dataloaders) == 1:
            dataloaders.append(BaseDataloader(data_paths=DEFAULT_DATASTREAM[0].dataset_paths))
        
        # Create a PTGMInterfaceConfig instance with given parameters.
        ptgm_config = PTGMInterfaceConfig(cluster_num=5, goal_offset=5, tsne_dim=2, tsne_perplexity=30)
        logger = get_logger(__name__)
        logger.info(f"\n==============================")
        logger.info(f"  Testing ContinualPTGMInterface in {mode.upper()} mode")
        logger.info(f"==============================")

        # Instantiate the ContinualPTGMInterface with the given mode.
        interface = ContinualPTGMInterface(ptgm_config, update_mode=mode)

        # Process each data chunk in a loop.
        for idx, dataloader in enumerate(dataloaders):
            logger.info(f"\n=== Processing data chunk {idx} in {mode.upper()} mode ===")

            # Update the interface using the current data chunk.
            dataloader = interface.update_interface(dataloader)
            logger.info(f"Global centroid map size: {interface.num_skills}")
            logger.info(f"Global centroid map: {interface.global_centroid_map.keys()}")
            logger.info("\n")
            
            # Log updated fields in the dataloader.
            logger.debug(f"Entry: {dataloader.stacked_data['entry']}")
            logger.debug(f"Skill ID: {dataloader.stacked_data['skill_id']}")
            logger.info(f"Skill Aux shape: {dataloader.stacked_data['skill_aux'].shape}")

            # Choose test IDs based on the chunk index (just an example).
            test_ids = [0, 1, 2] if idx == 0 else [1, 2, 3, 5, 6, 7]

            # Use the forward method to get skill IDs and auxiliary data.
            skill_id, skill_aux = interface.forward(test_ids)
            logger.info(f"Forward output (chunk {idx} in {mode.upper()} mode):")
            logger.info(f"Skill ID: {skill_id}")
            logger.info(f"Skill Aux shape: {skill_aux.shape}")
            # print("Skill Aux (first entry):", skill_aux[0])

            # Plot T-SNE clusters and save the plot with a chunk-specific filename.
            plot_dir = "logs"
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"test_{mode}_chunk_{idx}.png")
            interface.plot_tsne_clusters(save_path=plot_path)
            logger.info(f"Plot saved to: {plot_path}")

        logger.info(f"--- Finished testing {mode.upper()} mode ---\n")

    logger.info("\n=== ContinualPTGMInterface ALL testing complete ===")
