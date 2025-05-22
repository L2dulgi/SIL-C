import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from AppOSI.models.skill_interface.base import BaseInterface


class AsSILInterfaceConfig :
    def __init__(
            self, 
            # decoder side
            cluster_num=20,      # int
            goal_offset=20,      # int
            prototype_bases=5,   # int
            # policy side 
            cluster_num_policy=20,  # int
            tsne_dim=2,          # int
            tsne_perplexity=30,   # int
            # confidence_interval
            confidence_interval=5, # float 99% of the Action dimension(9)
        ):
        self.cluster_num = cluster_num
        self.goal_offset = goal_offset
        self.prototype_bases = prototype_bases

        # policy side
        self.cluster_num_policy = cluster_num_policy
        
        # t-sne parameters
        self.tsne_dim = tsne_dim
        self.tsne_perplexity = tsne_perplexity

        # confidence interval
        self.confidence_interval = confidence_interval

class MaPrototype:
    '''
    Prototype for computing the Mahalanobis distance using a diagonal covariance matrix.
    
    Parameters:
      mean: np.array of shape (prototype_bases, state_dim)
            Each row represents the mean of a prototype.
      variance: np.array of shape (prototype_bases, state_dim)
            Each row represents the per-dimension variances for a prototype.
    '''
    def __init__(self, mean, variance):
        self.mean = mean 
        self.variance = variance

    def forward(self, x):
        '''
        Computes the Mahalanobis distance between each sample in x and each prototype.
        
        Parameters:
            x: np.array of shape (B, state_dim)
               Batch of input points.
        
        Returns:
            distances: np.array of shape (B, prototype_bases)
               Each element [i, j] is the Mahalanobis distance between x[i] and prototype j.
        '''
        if len(x.shape) == 1:
            x = x[None, :]
        eps = 1e-6  # Small constant to avoid division by zero
        # Expand dimensions for proper broadcasting:
        # x becomes (B, 1, state_dim) and mean becomes (1, prototype_bases, state_dim)
        diff = x[:, None, :] - self.mean[None, :, :]  # Shape: (B, prototype_bases, state_dim)
        # Compute the normalized squared differences
        squared_norm = (diff ** 2) / (self.variance[None, :, :] + eps)
        # Sum along the state dimension and take the square root to get distances
        distances = np.sqrt(np.sum(squared_norm, axis=-1))
        return distances

class EntrySkillMap:
    def __init__(
            self, 
            skill_id,          # int 
            decoder_id,        # int
            skill_aux,         # np.array
            state_prototypes=None,   # np.array (prototype_bases, state_dim)
            action_prototypes=None, # np.array (prototype_bases, action_dim) 
            subgoal_prototypes=None, # np.array (prototype_bases, state_dim)
            data_count=0       # int, number of data points in this cluster
        ):
        self.skill_id = skill_id
        self.decoder_id = decoder_id
        self.skill_aux = skill_aux
        # state prototypes : bases and variances for mahalanobis distance
        # (means, variances) 
        self.state_prototypes = MaPrototype(
            mean=state_prototypes[0],
            variance=state_prototypes[1]
        )

        self.action_prototypes = MaPrototype(
            mean=action_prototypes[0],
            variance=action_prototypes[1]
        )

        # subgoal prototypes : bases and variances for mahalanobis distance 
        self.subgoal_prototypes = MaPrototype(
            mean=subgoal_prototypes[0],
            variance=subgoal_prototypes[1]
        )

        self.data_count = data_count  

class PolicyPrototype:
    def __init__(
            self, 
            prototype_id,      # int 
            subgoal,           # np.array
            state_prototypes=None,   # np.array (prototype_bases, state_dim)
            data_count=0       # int, number of data points used for this cluster
        ):
        self.prototype_id = prototype_id
        self.subgoal = subgoal
        self.state_prototypes = MaPrototype(
            state_prototypes[0],
            state_prototypes[1]
        ) 
        self.data_count = data_count  

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
    Legacy cluster generator: minimal edits from map_entry_v1.
    """
    def __init__(self, config, random_state=0):
        self.config = config
        self.random_state = random_state

    def cluster(self, dataloader):
        # Based on map_entry_v1 logic
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

        # t-SNE + KMeans   
        # method='barnes_hut' if len(data['observations']) > 100_000 else 'exact',

        
        tsne = TSNE(
            # method=method,
                    n_components=self.config.tsne_dim,
                    perplexity=self.config.tsne_perplexity,
                    random_state=self.random_state)
        emb = tsne.fit_transform(all_subgoals)
        km = KMeans(n_clusters=self.config.cluster_num, random_state=self.random_state)
        ids = km.fit_predict(emb)
        centers_tsne = km.cluster_centers_

        # centroid mapping
        centroid_map = {}
        for c in range(self.config.cluster_num):
            d = np.sum((emb - centers_tsne[c])**2, axis=1)
            idx = np.argmin(d)
            centroid_map[c] = all_subgoals[idx]

        # build labels
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

from copy import deepcopy
class AsSILInterface(BaseInterface):
    def __init__(
            self, 
            config:AsSILInterfaceConfig=None,
            clustering_strategy:SkillClusteringStrategy=None
        ):
        super().__init__()
        self.config = config if config is not None else AsSILInterfaceConfig()
        # self.clustering = clustering_strategy or PTGMClusteringStrategy(
        #     n_clusters=self.config.cluster_num,
        #     tsne_dim=self.config.tsne_dim,
        #     perplexity=self.config.tsne_perplexity,
        #     random_state=0
        # )
        # ----------------
        # Decoder side
        # ----------------
        # 1. state prototypes
        # 2. sub-goal prototypes 
        '''
        { "[skill_id]" : EntrySkillMap }
        '''
        self.entry_skill_map = {}
        # ----------------
        # Policy side
        # ----------------
        # for policy side implementation., we outsource the prototype saving for the policy model.
        # so we need to sync the policy model with the interface for agent building.
        # 1. state prototypes 
        '''
        { "[prototype_id]" : PolicyPrototype }
        '''
        self.policy_prototypes = {}

        self.decoder_id = 0
        self.debug = False

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

    def map_entry_v2(self, dataloader):
        labels, centroid_map, extra = self.clustering.cluster(dataloader)
        data = dataloader.stacked_data
        obs = np.array(data['observations'])

        # attach subgoals array to dataloader
        T = len(obs)
        data['subgoals'] = np.zeros_like(obs)
        for sg, t in zip(extra['subgoals'], extra['timesteps']):
            data['subgoals'][t] = sg

        base = self.num_skills
        global_map = {loc: base + loc for loc in range(self.config.cluster_num)}

        data['skill_id'] = np.full(T, -1, dtype=int)
        data['entry']    = np.full(T, -1, dtype=int)
        data['skill_aux'] = np.zeros_like(obs)
        for t, loc in enumerate(labels):
            if loc >= 0:
                gid = global_map[loc]
                data['skill_id'][t] = gid
                data['entry'][t]    = gid
                data['skill_aux'][t] = centroid_map[loc]

        for loc in range(self.config.cluster_num):
            gid = global_map[loc]
            if gid not in self.entry_skill_map:
                idxs = np.where(data['entry'] == gid)[0]
                self.entry_skill_map[gid] = EntrySkillMap(
                    skill_id=gid,
                    decoder_id=self.decoder_id,
                    skill_aux=centroid_map[loc],
                    state_prototypes=self._create_prototypes(obs[idxs]),
                    action_prototypes=self._create_prototypes(np.array(data['actions'])[idxs]),
                    subgoal_prototypes=self._create_prototypes(data['subgoals'][idxs]),
                    data_count=len(idxs)
                )

        data['decoder_id'] = np.full(T, self.decoder_id, dtype=int)
        self.decoder_id += 1
        return dataloader
    
    def map_entry(self, dataloader):
        # NOTE : This function is deprecated and replaced by map_entry_v2.
        # [Decoder side prototype generation]
        stacked_data = dataloader.stacked_data
        observations = stacked_data['observations']
        terminals = stacked_data['terminals']
        T = len(observations)

        def _split_by_trajectory(terminals):
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
        
        # Split data into trajectories based on terminals.
        traj_indices = _split_by_trajectory(terminals)
        all_states = np.array(observations).copy()
        all_actions = np.array(stacked_data['actions']).copy()
        all_subgoals = [] 
        all_subgoal_ts = []  # To remap cluster IDs to the correct timesteps.

        for (start_idx, end_idx) in traj_indices:
            for t in range(start_idx, end_idx + 1):
                future_t = t + self.config.goal_offset
                if future_t <= end_idx:
                    sg = observations[future_t]
                else:
                    sg = observations[end_idx]
                all_subgoals.append(sg)
                all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)
        stacked_data['subgoals'] = all_subgoals.copy()

        # Apply T-SNE dimensionality reduction.
        tsne = TSNE(
            # method='barnes_hut' if len(stacked_data['observations']) > 100000 else 'exact',
            n_components=self.config.tsne_dim, 
            perplexity=self.config.tsne_perplexity, 
            random_state=0
        )
        subgoals_tsne = tsne.fit_transform(all_subgoals)

        # Run K-Means clustering in the T-SNE space.
        self.kmeans = KMeans(n_clusters=self.config.cluster_num, random_state=0)
        self.kmeans.fit(subgoals_tsne)
        cluster_ids = self.kmeans.predict(subgoals_tsne)

        # Map cluster centers in T-SNE space to actual subgoals in the original space.
        cluster_centers_tsne = self.kmeans.cluster_centers_
        centroid_map = {}
        for c in range(self.config.cluster_num):
            center_c = cluster_centers_tsne[c]
            dist = np.sum((subgoals_tsne - center_c) ** 2, axis=1)
            closest_idx = np.argmin(dist)
            centroid_map[c] = all_subgoals[closest_idx]
        self.centroid_map = centroid_map

        # Save T-SNE data for optional plotting.
        self._tsne_data = {
            'subgoals_tsne': subgoals_tsne,
            'cluster_ids': cluster_ids,
            'cluster_centers_tsne': cluster_centers_tsne
        }

        # Create a mapping from local cluster id to global id.
        current_num_skills = self.num_skills  # current total number of skills
        global_id_map = {}
        for local_id in range(self.config.cluster_num):
            global_id_map[local_id] = current_num_skills + local_id

        # Assign global cluster (skill) information to each timestep.
        stacked_data['skill_id'][:] = -1
        stacked_data['entry'][:] = -1
        stacked_data['skill_aux'][:] = 0.0
        for i, t in enumerate(all_subgoal_ts):
            local_id = cluster_ids[i]
            global_id = global_id_map[local_id]
            stacked_data['skill_id'][t] = global_id
            stacked_data['entry'][t] = global_id
            stacked_data['skill_aux'][t] = centroid_map[local_id]
        
        # Create entry_skill_map with global skill id as key.
        for local_id in range(self.config.cluster_num):
            global_id = global_id_map[local_id]
            # Only create entry if not already exists.
            if global_id not in self.entry_skill_map:
                # Find states corresponding to the current cluster from the dataset.
                state_indices = np.where(stacked_data['skill_id'] == global_id)[0]
                # data_count: number of data points in this cluster
                data_count = len(state_indices)
                copied_states = all_states[state_indices].copy()
                copied_actions = all_actions[state_indices].copy()
                copied_subgoals = all_subgoals[state_indices].copy()
                self.entry_skill_map[global_id] = EntrySkillMap(
                    skill_id=global_id,
                    decoder_id=self.decoder_id,  
                    skill_aux=centroid_map[local_id],
                    state_prototypes=self._create_prototypes(copied_states),
                    action_prototypes=self._create_prototypes(copied_actions),
                    subgoal_prototypes=self._create_prototypes(copied_subgoals),
                    data_count=data_count
                )
        
        stacked_data['decoder_id'][:] = self.decoder_id
        # NOTE: The decoder_id is incremented/updated.
        self.decoder_id += 1
        return dataloader   
    
    def _create_prototypes(self, data):
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
        kmeans = KMeans(n_clusters=self.config.prototype_bases, random_state=0)
        kmeans.fit(data)
        centroids = np.array(kmeans.cluster_centers_)
        labels = kmeans.labels_
        variances = []
        for i in range(self.config.prototype_bases):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                var = np.var(cluster_points, axis=0)  # per-dimension variance
            else:
                var = np.zeros(data.shape[1])
            variances.append(var)
        variances = np.array(variances)
        return (centroids, variances)

    # ----------------------------------
    # functions for policy side interface
    # ----------------------------------
    def create_policy_prototype(self, dataloader):
        """
        Create policy prototypes from a policy dataset.
        
        Steps:
        1. Retrieve the policy dataset observations and extract sub-goals using the goal_offset.
        2. Apply t-SNE dimensionality reduction on the sub-goals.
        3. Cluster the t-SNE sub-goal representation into bins using KMeans.
        4. Map each cluster center in T-SNE space back to a representative sub-goal in the original space.
        5. For each bin, use _create_prototypes to obtain state prototypes. If the data points
        are insufficient or an exception occurs during the prototype creation, skip that cluster.
        6. Create and save a PolicyPrototype for each successfully processed cluster.
        
        Returns:
        A dictionary mapping prototype_id to PolicyPrototype objects.
        """
        # Retrieve the raw observations from the dataloader.
        stacked_data = dataloader.stacked_data
        observations = stacked_data['observations']
        T = len(observations)
        
        # Use the terminals if available; otherwise assume a single trajectory.
        terminals = stacked_data['terminals'] 
        
        def _split_by_trajectory(terminals):
            traj_indices = []
            start_idx = 0
            for i, val in enumerate(terminals):
                if val == 1:
                    traj_indices.append((start_idx, i))
                    start_idx = i + 1
            return traj_indices

        # Split the observations into trajectories.
        traj_indices = _split_by_trajectory(terminals)
        all_subgoals = []
        all_subgoal_ts = []  # Optionally keep track of the corresponding timestep in the original sequence.
        
        # For each trajectory, compute sub-goals using the configured goal_offset.
        for (start_idx, end_idx) in traj_indices:
            for t in range(start_idx, end_idx + 1):
                future_t = t + self.config.goal_offset
                if future_t <= end_idx:
                    sg = observations[future_t]
                else:
                    sg = observations[end_idx]
                all_subgoals.append(sg)
                all_subgoal_ts.append(t)
        all_subgoals = np.array(all_subgoals)
        
        # -----------------------------------------------------------
        # Apply t-SNE on the sub-goals.
        tsne = TSNE(n_components=self.config.tsne_dim,
                    perplexity=self.config.tsne_perplexity,
                    random_state=0)
        subgoals_tsne = tsne.fit_transform(all_subgoals)
        
        # -----------------------------------------------------------
        # Cluster the t-SNE sub-goal representation using KMeans. (No problem)
        kmeans_policy = KMeans(n_clusters=self.config.cluster_num_policy, random_state=0)
        kmeans_policy.fit(subgoals_tsne)
        cluster_ids = kmeans_policy.predict(subgoals_tsne)
        cluster_centers_tsne = kmeans_policy.cluster_centers_

        # Map each cluster center in T-SNE space back to the nearest original sub-goal.
        centroid_map = {}
        for c in range(self.config.cluster_num_policy):
            center_tsne = cluster_centers_tsne[c]
            distances = np.sum((subgoals_tsne - center_tsne) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            centroid_map[c] = all_subgoals[closest_idx]

        # -----------------------------------------------------------
        # For each cluster, form state prototypes using _create_prototypes.
        policy_prototypes = {}
        incremental_id = 0  
        for c in range(self.config.cluster_num_policy):
            indices = np.where(cluster_ids == c)[0]
            data_bin = observations[indices]
            
            # Check if there are enough data points to form the prototypes.
            if len(data_bin) < self.config.prototype_bases:
                continue  # Skip creation for this cluster
            
            try:
                # Use the existing _create_prototypes function to create state prototypes.
                state_prototypes = self._create_prototypes(data_bin)
            except Exception as e:
                print(f"[AsSILInterface] Error creating prototypes for cluster {c}: {e}")
                continue
            
            # Create a PolicyPrototype with the representative sub-goal, computed state prototypes,
            # and store the number of data points used for this cluster.
            policy_prototypes[incremental_id] = PolicyPrototype(
                prototype_id=incremental_id,
                subgoal=centroid_map[c],
                state_prototypes=state_prototypes,
                data_count=len(data_bin)
            )
            incremental_id += 1

        print(f"[AsSILInterface] Created {len(policy_prototypes)} policy prototypes.")
        self.policy_prototypes = policy_prototypes
        return policy_prototypes
    
    # ----------------------------------
    # Utill functions
    # ----------------------------------
    def update_policy_prototype(self, policy_prototype):
        if policy_prototype is None:
            print("[AsSILInterface] No policy prototype to update; None detected.")
        self.policy_prototypes = deepcopy(policy_prototype)
    
    # ----------------------------------
    # interface forward function.
    # ----------------------------------
    def forward(self, entry, current_state, static=False):
        """
        Forward pass for AsSILInterface using static and dynamic matching with Mahalanobis distance.
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
        out_skill_aux = np.zeros((B, obs_dim), dtype=np.float32)
        
        # Threshold and k for dynamic matching. 95% confidence interval.
        threshold = 10.0
        k = 1  # (policy) number of candidates to select from policy prototypes.
        
        # Process each sample individually.
        for i in range(B):
            cs = current_state[i]  # current state vector for sample i
            orig_entry = int(entry[i])
            if orig_entry not in self.entry_skill_map:
                print(f"[AsSILInterface] Entry {orig_entry} not found in entry_skill_map; using random fallback.")
                orig_entry = int(np.random.randint(0, self.num_skills)) # Default fallback     
                
            # =============================
            # 1. Static Matching
            # =============================
            if static == True :
                # Static matching is good enough; return original mapping.
                out_skill_ids[i] = orig_entry
                out_decoder_ids[i] = self.entry_skill_map[orig_entry].decoder_id
                out_skill_aux[i] = self.entry_skill_map[orig_entry].skill_aux
                continue

            # =============================
            # 2. Dynamic Matching - confidence check
            # =============================
            candidate_list = []
            for pid, policy_proto in self.policy_prototypes.items():
                policy_mp = policy_proto.state_prototypes
                d_policy = policy_mp.forward(cs)  
                min_d_policy = np.min(d_policy)
                candidate_list.append((pid, min_d_policy))
            
            # Sort candidates by distance and select top k.
            candidate_list.sort(key=lambda x: x[1])
            top_candidates = candidate_list[:k]

            # Calcluate current subgoal(objective) is affordable for the selected entry. 
            best_total = np.inf
            best_skill_id = orig_entry  
            entry_obj = self.entry_skill_map[orig_entry]
            for (pid, d_policy_val) in top_candidates:
                policy_candidate = self.policy_prototypes[pid]
                candidate_subgoal = policy_candidate.subgoal
                d_subgoal = entry_obj.subgoal_prototypes.forward(candidate_subgoal)
                min_d_subgoal = np.min(d_subgoal)
                if min_d_subgoal < best_total:
                    best_total = min_d_subgoal

            if best_total < threshold : # always static or threshold
                # Static matching is good enough; return original mapping.
                out_skill_ids[i] = orig_entry
                out_decoder_ids[i] = self.entry_skill_map[orig_entry].decoder_id
                out_skill_aux[i] = self.entry_skill_map[orig_entry].skill_aux
                continue

            # =============================
            # 2. Dynamic Matching - reassign skill id
            # =============================
            best_total = np.inf
            best_skill_id = orig_entry  # default fallback
            best_decoder_id = -1
            best_aux = np.zeros(obs_dim, dtype=np.float32)

            for (pid, d_policy_val) in top_candidates:
                policy_candidate = self.policy_prototypes[pid]
                # Get the candidate's representative subgoal.
                candidate_subgoal = policy_candidate.subgoal  # a vector
                
                # Find the closest entry in entry_skill_map by Euclidean distance between candidate_subgoal and each entry's skill_aux.
                best_ma = np.inf
                candidate_entry_obj = None
                for eid, entry_obj in self.entry_skill_map.items():
                    d_subgoal = entry_obj.subgoal_prototypes.forward(candidate_subgoal)
                    dist_ma = np.min(d_subgoal)
                    if dist_ma < best_ma:
                        best_ma = dist_ma
                        candidate_entry_obj = entry_obj
                
                if candidate_entry_obj is None:
                    continue
                
                # Compute Mahalanobis distance from decoder's state prototypes for this candidate entry.
                d_state = candidate_entry_obj.state_prototypes.forward(cs)
                min_d_state = np.min(d_state)
                # d_subgoal = candidate_entry_obj.subgoal_prototypes.forward(candidate_subgoal)
                # min_d_subgoal = np.min(d_subgoal)

                total_distance = min_d_state 
                
                if total_distance < best_total:
                    best_total = total_distance
                    best_skill_id = candidate_entry_obj.skill_id
                    best_decoder_id = candidate_entry_obj.decoder_id
                    best_aux = candidate_entry_obj.skill_aux
            if self.debug == True :
                print(f"[AsSILInterface] Dynamic matching: entry {orig_entry} -> skill {best_skill_id}, decoder {best_decoder_id}, total distance {best_total}")
            # Dynamic matching result.
            out_skill_ids[i] = best_skill_id
            out_decoder_ids[i] = best_decoder_id
            out_skill_aux[i] = best_aux 
        '''
        out skill aux shape is (B, obs_dim)
        ''' 
        return (out_skill_ids, out_decoder_ids), out_skill_aux

def test_assil_interface():
    """
    Test function for AsSILInterface.
    
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
    from AppOSI.dataset.dataloader import BaseDataloader
    from AppOSI.config.skill_stream_config import DEFAULT_DATASTREAM
    from rich.console import Console
    from rich.table import Table
    from AppOSI.models.skill_interface.assil import AsSILInterface  # adjust import path if needed

    console = Console()
    
    # Create an instance of AsSILInterface with default configuration.
    interface = AsSILInterface()

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
    policy_protos = interface.create_policy_prototype(dataloader)
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

def test_assil_dynamic():
    import numpy as np
    from AppOSI.dataset.dataloader import BaseDataloader
    from AppOSI.config.skill_stream_config import DEFAULT_DATASTREAM
    from rich.console import Console
    from rich.table import Table
    from AppOSI.models.skill_interface.assil import AsSILInterface  # adjust import path if needed

    console = Console()
    interface = AsSILInterface()

    # Load decoder and policy data chunks.
    decoder_chunk = DEFAULT_DATASTREAM[0]
    policy_chunk = DEFAULT_DATASTREAM[1]

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
    policy_protos = interface.create_policy_prototype(policy_dataloader)
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
    table_forward.add_column("SAME?", style="magenta")

    policy_dataloader = interface.rollback_dataloader(policy_dataloader)
    accuracy = 0
    total_same = 0
    from tqdm import tqdm
    for i, index in tqdm(enumerate(entry_array)):
        (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([3]), policy_dataloader.stacked_data['observations'][i])
        # (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([policy_dataloader.stacked_data['entry'][i]]), policy_dataloader.stacked_data['observations'][i])
        if f_skill[0] == policy_dataloader.stacked_data['entry'][i]:
            total_same += 1
    accuracy = total_same / len(entry_array)

    for i, idx in enumerate(test_indicies):
        (f_skill, f_decoder), f_skill_aux = interface.forward(np.array([3]), current_state[i])
        table_rematch.add_row(str(idx), str(f_skill[0]), str(f_decoder[0]), str(entry_array[idx]), str(f_skill[0] == entry_array[idx]))
    console.print(table_rematch)
    console.print(f"[bold green]Accuracy of rematching: {accuracy}[/bold green]")

# Run test if executed as a script.
if __name__ == "__main__":
    # test_assil_interface()
    test_assil_dynamic()   
