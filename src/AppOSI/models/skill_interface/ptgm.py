import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from AppOSI.models.skill_interface.base import BaseInterface

class PTGMInterfaceConfig:
    def __init__(self, cluster_num=5, goal_offset=5, tsne_dim=2, tsne_perplexity=30):
        self.cluster_num = cluster_num
        self.goal_offset = goal_offset
        self.tsne_dim = tsne_dim
        self.tsne_perplexity = tsne_perplexity

    def to_dict(self):
        return {
            'cluster_num': self.cluster_num,
            'goal_offset': self.goal_offset,
            'tsne_dim': self.tsne_dim,
            'tsne_perplexity': self.tsne_perplexity,
        }

class PTGMInterface(BaseInterface):
    def __init__(self, ptgm_config: PTGMInterfaceConfig):
        """
        ptgm_config must be an instance of PTGMInterfaceConfig.
        """
        super().__init__()
        config = ptgm_config.to_dict()
        self.cluster_num = config.get('cluster_num', 10)
        self.goal_offset = config.get('goal_offset', 10)
        self.tsne_dim = config.get('tsne_dim', 2)
        self.tsne_perplexity = config.get('tsne_perplexity', 30)
        self.kmeans = None
        self.centroid_map = None  # Map from cluster ID to the original subgoal (observation)
        self._tsne_data = None   # Store T-SNE results for debugging/plotting

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

        # 2) Optionally apply T-SNE if tsne_dim > 0
        if getattr(self, 'tsne_dim', 0) > 0:
            tsne = TSNE(
                # method='barnes_hut' if len(stacked_data['observations']) > 100_000 else 'exact',
                n_components=self.tsne_dim, perplexity=self.tsne_perplexity, random_state=0
            )
            subgoals_emb = tsne.fit_transform(all_subgoals)
        else:
            print("[PTGMInterface] T-SNE not applied, using original subgoals.")
            subgoals_emb = all_subgoals

        # 3) Run K-Means clustering on the chosen embedding
        self.kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
        self.kmeans.fit(subgoals_emb)
        cluster_ids = self.kmeans.predict(subgoals_emb)

        # 4) Map cluster centers back to actual subgoals in original space
        cluster_centers_emb = self.kmeans.cluster_centers_
        centroid_map = {}
        for c in range(self.cluster_num):
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

    def plot_tsne_clusters(self, save_path=None):
        """
        Visualize the subgoal clusters in the T-SNE space.
        If save_path is provided, the plot will be saved; otherwise, it will be displayed.
        """
        if not self._tsne_data:
            print("No T-SNE data available. Run map_entry or update_interface first.")
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
            print(f"T-SNE cluster plot saved to: {save_path}")
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
                        decoder_id[i, j] = -1        
            
            return (skill_id, decoder_id), skill_aux

        else:
            raise ValueError("Unsupported entry shape in forward")


# ----------------------------------------------------------------------------- 
# Testing block for ContinualPTGMInterface using update_interface 
# for both 'expand' and 'overwrite' modes across data chunks.

if __name__ == '__main__':
    import os
    # Import necessary modules and classes.
    from AppOSI.dataset.dataloader import BaseDataloader
    from AppOSI.config.skill_stream_config import DEFAULT_DATASTREAM
    from AppOSI.models.skill_interface.ptgm import ContinualPTGMInterface, PTGMInterfaceConfig

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
        print(f"\n==============================")
        print(f"  Testing ContinualPTGMInterface in {mode.upper()} mode")
        print(f"==============================")

        # Instantiate the ContinualPTGMInterface with the given mode.
        interface = ContinualPTGMInterface(ptgm_config, update_mode=mode)

        # Process each data chunk in a loop.
        for idx, dataloader in enumerate(dataloaders):
            print(f"\n=== Processing data chunk {idx} in {mode.upper()} mode ===")

            # Update the interface using the current data chunk.
            dataloader = interface.update_interface(dataloader)
            print("Global centroid map size:", interface.num_skills)
            print("Golbal centroid map:", interface.global_centroid_map.keys())
            print("\n")
            
            # Print updated fields in the dataloader.
            print("Entry:", dataloader.stacked_data['entry'])
            print("Skill ID:", dataloader.stacked_data['skill_id'])
            print("Skill Aux shape:", dataloader.stacked_data['skill_aux'].shape)

            # Choose test IDs based on the chunk index (just an example).
            test_ids = [0, 1, 2] if idx == 0 else [1, 2, 3, 5, 6, 7]

            # Use the forward method to get skill IDs and auxiliary data.
            skill_id, skill_aux = interface.forward(test_ids)
            print(f"Forward output (chunk {idx} in {mode.upper()} mode):")
            print("Skill ID:", skill_id)
            print("Skill Aux shape:", skill_aux.shape)
            # print("Skill Aux (first entry):", skill_aux[0])

            # Plot T-SNE clusters and save the plot with a chunk-specific filename.
            plot_dir = "logs"
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"test_{mode}_chunk_{idx}.png")
            interface.plot_tsne_clusters(save_path=plot_path)
            print(f"Plot saved to: {plot_path}")

        print(f"--- Finished testing {mode.upper()} mode ---\n")

    print("\n=== ContinualPTGMInterface ALL testing complete ===")
