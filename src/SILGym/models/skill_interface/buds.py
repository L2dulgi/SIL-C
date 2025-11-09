import numpy as np
from SILGym.utils.cuml_wrapper import TSNE, KMeans
import matplotlib.pyplot as plt
from SILGym.models.skill_interface.base import BaseInterface
from SILGym.utils.logger import get_logger

from collections import defaultdict
from sklearn.cluster import SpectralClustering  # SpectralClustering not in cuML
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# Configuration class ----------------------------------------------------------
# -----------------------------------------------------------------------------

class BUDSInterfaceConfig:
    """Container for hyper‑parameters used by the BUDS interface."""

    def __init__(
        self,
        window_size: int = 5,
        min_length: int = 20,
        target_num_segments: int = 10,
        max_k: int = 20,
        # goal labeling (H in original paper)
        goal_offset: int = 20,
        verbose: bool = False,
    ) -> None:
        self.window_size = window_size           # Sliding window length for initial split
        self.min_length = min_length             # Minimum allowed segment length
        self.target_num_segments = target_num_segments  # Desired number of segments (per demo)
        self.max_k = max_k                       # Maximum clusters to try in SpectralClustering
        self.goal_offset = goal_offset           # Offset for goal labeling 
        self.verbose = verbose


# -----------------------------------------------------------------------------
# Low‑level segmenter & clusterer ---------------------------------------------
# -----------------------------------------------------------------------------
class BUDSClusterGeneator:
    """Hierarchical segmentation + spectral clustering pipeline.

    The generator first segments each demonstration with a simple bottom‑up
    agglomerative rule, then extracts segment‑level features and clusters them
    with SpectralClustering – automatically selecting *k* via silhouette score.
    """

    def __init__(self, config: BUDSInterfaceConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.debug(f"Config: {config.__dict__}")

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------
    def hierarchical_segmentation(self, latent_seq):
        """Bottom‑up (agglomerative) segmentation on a latent sequence.

        A fixed *window_size* is used for the initial coarse split.  Adjacent
        segments are then merged greedily by Euclidean distance until either
        (i) the requested *target_num_segments* is reached or (ii) every
        segment is at least *min_length* frames long.
        """
        ws = self.config.window_size
        min_len = self.config.min_length
        target_segs = self.config.target_num_segments
        n = len(latent_seq)
        segments = []
        # 1) Initial fixed‑window partition
        for i in range(0, n, ws):
            segments.append((i, min(i + ws, n)))
        # 2) Agglomerative merge loop
        while (len(segments) > target_segs) or any((e - s) < min_len for (s, e) in segments):
            if len(segments) == 1:
                break
            distances = []
            for j in range(len(segments) - 1):
                s1, e1 = segments[j]
                s2, e2 = segments[j + 1]
                avg1 = np.mean(latent_seq[s1:e1], axis=0)
                avg2 = np.mean(latent_seq[s2:e2], axis=0)
                distances.append(np.linalg.norm(avg1 - avg2))
            j_min = int(np.argmin(distances))
            s1, _ = segments[j_min]
            _, e2 = segments[j_min + 1]
            segments = segments[:j_min] + [(s1, e2)] + segments[j_min + 2:]
        return segments

    def extract_segment_feature(self, latent_seq, segment):
        """Return a simple feature: [first | middle | last] latent concatenated."""
        s, e = segment
        first = latent_seq[s]
        middle = latent_seq[(s + e) // 2]
        last = latent_seq[e - 1]
        return np.concatenate([first, middle, last])

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------
    def find_best_k_spectral(self, features, min_k: int = 2, max_k: int = 8):
        """Choose *k* by maximizing silhouette score over [min_k, max_k]."""
        best_k, best_score, best_labels = min_k, -1.0, None
        for k in range(min_k, max_k + 1):
            try:
                sc = SpectralClustering(n_clusters=k, affinity="rbf", random_state=0)
                labels = sc.fit_predict(features)
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            except Exception as e:
                if self.config.verbose:
                    self.logger.error(f"Error at k={k}: {e}")
        return best_k, best_labels

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def perform_segmentation_and_clustering(self, dataloader):
        """Annotate *dataloader.stacked_data* with segment & cluster fields."""
        data = dataloader.stacked_data
        observations = data["observations"]

        # ------------------------------------------------------------------
        # 1) Identify demonstration boundaries (using terminals if provided)
        # ------------------------------------------------------------------
        if "terminals" in data:
            terminals = data["terminals"]
            demo_ids = np.zeros(len(observations), dtype=np.int32)
            cur_demo = 0
            for i, flag in enumerate(terminals):
                demo_ids[i] = cur_demo
                if flag:
                    cur_demo += 1
        else:
            demo_ids = np.zeros(len(observations), dtype=np.int32)

        unique_demos = np.unique(demo_ids)
        segment_indices, all_features = [], []

        # ------------------------------------------------------------------
        # 2) Segment each demo and build per‑segment features
        # ------------------------------------------------------------------
        for demo in unique_demos:
            idxs = np.where(demo_ids == demo)[0]
            obs_demo = observations[idxs]
            latent_seq = np.array([self.compute_latent(o) for o in obs_demo])
            segments = self.hierarchical_segmentation(latent_seq)
            for seg in segments:
                local_s, local_e = seg
                global_s = idxs[local_s]
                global_e = idxs[local_e - 1] + 1  # [start, end)
                segment_indices.append((global_s, global_e))
                all_features.append(self.extract_segment_feature(latent_seq, seg))

        # No segments → return asap
        if len(all_features) == 0:
            return dataloader

        all_features = np.asarray(all_features)

        # ------------------------------------------------------------------
        # 3) Run spectral clustering (auto‑select k)
        # ------------------------------------------------------------------
        best_k, best_labels = self.find_best_k_spectral(
            features=all_features,
            min_k=2,
            max_k=self.config.max_k,
        )
        self.num_clusters = best_k
        self.segments = segment_indices
        self.segment_features = all_features
        self.segment_labels = best_labels

        # Build mapping: cluster_id → list[segment_idx]
        clusters = defaultdict(list)
        for seg_idx, lbl in enumerate(best_labels):
            clusters[lbl].append(seg_idx)
        self.cluster_partitions = dict(clusters)

        # ------------------------------------------------------------------
        # 4) Propagate cluster labels to every frame between [start, end)
        # ------------------------------------------------------------------
        T = len(observations)
        skill_labels = np.zeros(T, dtype=np.int32)
        for seg_idx, lbl in enumerate(best_labels):
            s, e = segment_indices[seg_idx]
            skill_labels[s:e] = lbl

        # Push results back to dataloader
        data["entry"] = skill_labels
        data["skill_id"] = skill_labels
        # data["segments"] = segment_indices
        # data["best_k"] = best_k

        return dataloader

    # ------------------------------------------------------------------
    # Placeholder – you must define compute_latent() in your project
    # ------------------------------------------------------------------
    def compute_latent(self, obs):  # noqa: D401
        """Dummy embedding; override with your own feature extractor."""
        return obs

# -----------------------------------------------------------------------------
# High‑level interface ---------------------------------------------------------
# -----------------------------------------------------------------------------

class BUDSInterface(BaseInterface):
    """End‑to‑end interface that wraps *BUDSClusterGeneator* for skill discovery."""

    def __init__(self, config: BUDSInterfaceConfig):
        super().__init__()
        self.logger = get_logger(__name__)
        self.config = config
        self.cluster_gen = BUDSClusterGeneator(config)
        self.goal_offset = self.config.goal_offset 

        # Persistent metadata --------------------------------------------------
        self.cluster_num: int = 0
        self.centroid_map: dict[int, np.ndarray] = {}
        self._tsne_data = None  # Optional plotting cache

    # ------------------------------------------------------------------
    # Fast helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def init_entry(self, dataloader):
        """Ensure necessary fields exist in *dataloader.stacked_data*."""
        sd = dataloader.stacked_data
        T = len(sd["observations"])
        if "entry" not in sd:
            sd["entry"] = np.zeros(T, dtype=np.int32)
        if "skill_id" not in sd:
            sd["skill_id"] = np.zeros(T, dtype=np.int32)
        if "skill_aux" not in sd:
            obs_dim = sd["observations"].shape[1]
            sd["skill_aux"] = np.zeros((T, obs_dim), dtype=np.float32)

        if "subgoals" not in sd:
            sd["subgoals"] = np.zeros((T, 1), dtype=np.float32)
            # Split data into trajectories based on terminals.
            terminals = sd["terminals"]
            observations = sd["observations"]
            traj_indices = self._split_by_trajectory(terminals)
            all_subgoals = []
            all_subgoal_ts = []  # To remap cluster IDs to the correct timesteps.
            for (start_idx, end_idx) in traj_indices:
                for t in range(start_idx, end_idx + 1):
                    future_t = t + self.goal_offset
                    if future_t <= end_idx:
                        sg = observations[future_t]
                    else:
                        sg = observations[end_idx]
                    all_subgoals.append(sg)
                    all_subgoal_ts.append(t)
            all_subgoals = np.array(all_subgoals)
            sd["subgoals"] = all_subgoals.copy()

    # ------------------------------------------------------------------
    # 1) Low‑level clustering wrapper ----------------------------------
    # ------------------------------------------------------------------
    def buds_clustering(self, dataloader):
        """Run clustering and return a dict with per‑timestep cluster IDs."""
        dataloader = self.cluster_gen.perform_segmentation_and_clustering(dataloader)
        cluster_ids = dataloader.stacked_data["entry"]  # Shape: (T,)
        return {
            "cluster_ids": cluster_ids,
            "num_clusters": self.cluster_gen.num_clusters,
        }

    # ------------------------------------------------------------------
    # 2) Map entry/skill_id/skill_aux ----------------------------------
    # ------------------------------------------------------------------
    def map_entry(self, dataloader):
        """Annotate each timestep with cluster ID and centroid (skill_aux)."""
        self.init_entry(dataloader)

        # Run BUDS discovery ----------------------------------------------------
        buds = self.buds_clustering(dataloader)
        cluster_ids = buds["cluster_ids"]
        self.cluster_num = buds["num_clusters"]

        sd = dataloader.stacked_data
        observations = sd["observations"]
        subgoals = sd["subgoals"]

        # Compute centroids and fill skill_aux --------------------------------
        centroid_map = {}
        for cid in range(self.cluster_num):
            idx = np.where(cluster_ids == cid)[0]
            if len(idx) == 0:
                continue
            centroid = subgoals[idx].mean(axis=0).astype(np.float32)
            centroid_map[cid] = centroid
            sd["skill_aux"][idx] = centroid

        self.centroid_map = centroid_map
        sd["entry"] = cluster_ids.astype(np.int32)
        sd["skill_id"] = cluster_ids.astype(np.int32)

        # # Optional: generate a 2‑D T‑SNE scatter for debugging -----------------
        # try:
        #     seg_feat, seg_labels = (
        #         self.cluster_gen.segment_features,
        #         self.cluster_gen.segment_labels,
        #     )
        #     tsne = TSNE(n_components=2, random_state=0)
        #     seg_tsne = tsne.fit_transform(seg_feat)
        #     centers_tsne = np.zeros((self.cluster_num, 2))
        #     for cid in range(self.cluster_num):
        #         pts = seg_tsne[seg_labels == cid]
        #         centers_tsne[cid] = pts.mean(axis=0)
        #     self._tsne_data = {
        #         "subgoals_tsne": seg_tsne,
        #         "cluster_ids": seg_labels,
        #         "cluster_centers_tsne": centers_tsne,
        #     }
        # except Exception:
        #     self._tsne_data = None

    # ------------------------------------------------------------------
    # 3) Forward API ----------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, entry):
        """Return *(skill_id, skill_aux)* given cluster IDs (entry)."""
        if self.centroid_map is None:
            raise ValueError("centroid_map is not initialized – call map_entry() first")
        entry = np.asarray(entry).astype(np.int32)
        B = entry.shape[0]
        obs_dim = next(iter(self.centroid_map.values())).shape[0]
        skill_id = entry.copy()
        skill_aux = np.zeros((B, obs_dim), dtype=np.float32)
        for i, cid in enumerate(entry):
            skill_aux[i] = self.centroid_map.get(int(cid), np.zeros(obs_dim, dtype=np.float32))
        return skill_id, skill_aux

    # ------------------------------------------------------------------
    # 4) Utility: split indices by trajectory ---------------------------
    # ------------------------------------------------------------------
    def _split_by_trajectory(self, terminals):
        """Return a list of *(start_idx, end_idx)* tuples for each trajectory."""
        traj, start = [], 0
        for i, term in enumerate(terminals):
            if term:
                traj.append((start, i))
                start = i + 1
        return traj

    # ------------------------------------------------------------------
    # 5) Debug plot -----------------------------------------------------
    # ------------------------------------------------------------------
    def plot_tsne_clusters(self, save_path: str | None = None):
        """Scatter T‑SNE projection of segment features with cluster labels."""
        if self._tsne_data is None:
            self.logger.warning("No t‑SNE data – run map_entry() first")
            return
        d = self._tsne_data
        plt.figure(figsize=(8, 6))
        plt.scatter(d["subgoals_tsne"][:, 0], d["subgoals_tsne"][:, 1], c=d["cluster_ids"], cmap="tab10", alpha=0.7)
        plt.scatter(d["cluster_centers_tsne"][:, 0], d["cluster_centers_tsne"][:, 1], s=200, c="red", marker="*", edgecolors="black", linewidths=1)
        plt.title("t‑SNE Subgoal Clusters")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"plot saved to {save_path}")
        else:
            plt.show()


class ContinualBUDSInterface(BUDSInterface):
    """
    A version of BUDSInterface that supports either expanding the global map of clusters
    or overwriting it entirely at each new data chunk, depending on `update_mode`.

    - 'expand': Continues adding new clusters to a global map, offsetting IDs to avoid collisions.
    - 'overwrite': Each new chunk completely replaces the global map and all old clusters.
    """
    def __init__(
            self, 
            config: BUDSInterfaceConfig, 
            update_mode: str = 'expand',
        ):
        super().__init__(config)
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

        # print logs here 
        self.logger.info(f"{self.update_mode} mode: {self.num_skills} skills, "
              f"cluster_offset: {self.cluster_offset}, decoder_id: {self.decoder_id}")
        self.logger.info(f"global centroid map: {self.global_centroid_map.keys()}")
        self.logger.info(f"entry decoder map: {self.entry_decoder_map.keys()}")

        return self.update_dataloader(dataloader)

    def update_dataloader(self, dataloader):
        """
        For BUDS training, store original observations, then
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
                    decoder_id[i] = -1
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
                        decoder_id[i, j] = 0        
            
            return (skill_id, decoder_id), skill_aux

        else:
            raise ValueError("Unsupported entry shape in forward")


# ----------------------------------------------------------------------------- 
# Testing block for ContinualBUDSInterface using update_interface 
# for both 'expand' and 'overwrite' modes across data chunks.

if __name__ == '__main__':
    import os
    # Import necessary modules and classes.
    from SILGym.dataset.dataloader import BaseDataloader
    from SILGym.config.skill_stream_config import DEFAULT_DATASTREAM
    from SILGym.config.kitchen_scenario import KITCHEN_SCENARIO_OBJ_SYNC
    from SILGym.config.mmworld_scenario import MMWORLD_SCENARIO_EASY_SYNC
    from SILGym.models.skill_interface.buds import ContinualBUDSInterface, BUDSInterfaceConfig

    # We'll test with two modes: 'expand' and 'overwrite'.
    for mode in ['expand']:
        # Create a list of dataloaders for each data chunk in DEFAULT_DATASTREAM.
        dataloaders = []
        # for i, ds in enumerate(KITCHEN_SCENARIO_OBJ_SYNC):
        for i, ds in enumerate(MMWORLD_SCENARIO_EASY_SYNC):
            if str(i) in ['0', '25', '50', '75'] :
                dataloaders.append(BaseDataloader(data_paths=ds.dataset_paths))
        
        # # If only one dataset is available, duplicate it for demonstration purposes.
        # if len(dataloaders) == 1:
        #     dataloaders.append(BaseDataloader(data_paths=DEFAULT_DATASTREAM[0].dataset_paths))
        
        # Create a BUDSInterfaceConfig instance with given parameters.
        config = BUDSInterfaceConfig(
            window_size=5,
            min_length=30,
            target_num_segments=10,
            max_k=10,
            goal_offset=20,
            verbose=True,
        )
        logger = get_logger(__name__)
        logger.info(f"\n==============================")
        logger.info(f"  Testing ContinualBUDSInterface in {mode.upper()} mode")
        logger.info(f"==============================")

        # Instantiate the ContinualBUDSInterface with the given mode.
        interface = ContinualBUDSInterface(config, update_mode=mode)

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

            for entry, skill_id, decoder_id, terminal in zip(
                dataloader.stacked_data['entry'],
                dataloader.stacked_data['skill_id'],
                dataloader.stacked_data['decoder_id'],
                dataloader.stacked_data['terminals'],
            ):
                logger.debug(f"Entry: {entry}, Skill ID: {skill_id}, Decoder ID: {decoder_id}, Terminal: {terminal}")

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

    logger.info("\n=== ContinualBUDSInterface ALL testing complete ===")
