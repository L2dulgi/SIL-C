import pickle
import numpy as np

class BaseDataloader:
    """
    Basic DataLoader with optional aux_key concatenation:
      - Loads data
      - Processes data (assigns stacked_data)
      - Returns a random batch (get_rand_batch) with optional concatenation of aux_key
      - Yields all batches in iteration (get_all_batch) with optional concatenation of aux_key
        and remainder padding.
    """
    def __init__(self, data_paths, semantics_path=None):
        self.data_paths = data_paths
        self.data_buffer = None
        self.stacked_data = None
        self.dataset_size = 0
        self.semantics_path = semantics_path # Flag whether the obs is concatenated with semantic embedding
        print("[BaseDataloader] Initializing BaseDataloader")
        print(f"[BaseDataloader] Data paths provided: {data_paths}")

        self.load_data()
        self.process_data()

    def load_data(self):
        """
        Load pickle files from data_paths, convert data (lists or arrays) to np.ndarray,
        and merge them into self.data_buffer.
        """
        data = {}
        for data_path in self.data_paths:
            print(f"[BaseDataloader] Loading data from {data_path}")
            with open(data_path, 'rb') as f:
                loaded_data = pickle.load(f)
            for key, value in loaded_data.items():
                value = np.array(value)
                if key not in data:
                    data[key] = value
                else:
                    data[key] = np.concatenate([data[key], value], axis=0)
        self.data_buffer = data

    def process_data(self):
        """
        Assign self.data_buffer to self.stacked_data and set dataset_size.
        """
        self.stacked_data = self.data_buffer
        self.dataset_size = len(self.stacked_data['observations'])
        if self.semantics_path is not None:
            print(f"[BaseDataloader] Concatenating semantic embedding with observations.")
            with open(self.semantics_path, 'rb') as f:  
                semantics = pickle.load(f)
            skill_embeddings = [semantics[skill] for skill in self.stacked_data['skills']]
            skill_embeddings = np.array(skill_embeddings)
            self.stacked_data['observations'] = np.concatenate([self.stacked_data['observations'], skill_embeddings], axis=-1)
            print(f"[BaseDataloader] Processed data. obs shape = {self.stacked_data['observations'].shape}")
        print(f"[BaseDataloader] Processed data. Dataset size = {self.dataset_size}")

    def get_rand_batch(self, batch_size=None, aux_key=None):
        """
        Randomly sample `batch_size` items from the dataset and return them.
        If batch_size is -1 or None, return the entire dataset.
        If aux_key is provided, concatenate the aux_key data with 'observations'
        along the last dimension.
        """
        if batch_size == -1 or batch_size is None:
            batch = {key: val for key, val in self.stacked_data.items()}
        else:
            indices = np.random.choice(self.dataset_size, batch_size, replace=False)
            batch = {key: self.stacked_data[key][indices] for key in self.stacked_data.keys()}

        if aux_key is not None:
            if aux_key not in batch:
                raise ValueError(f"[BaseDataloader] aux_key '{aux_key}' not found in dataset.")
            batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
            del batch[aux_key]

        return batch

    def get_all_batch(self, batch_size, aux_key=None, shuffle=True, remainder_pad=True):
        """
        Yield the entire dataset in batches of size batch_size.
        If remainder_pad is True, the remainder is padded to produce a full batch (using replace=True for padding).
        If aux_key is provided, concatenate that key's data with 'observations' along the last dimension.
        """
        indices = np.arange(self.dataset_size)
        if shuffle:
            np.random.shuffle(indices)

        num_full_batches = self.dataset_size // batch_size
        remainder = self.dataset_size % batch_size

        # Yield full batches
        for i in range(num_full_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            batch = {key: self.stacked_data[key][batch_indices] for key in self.stacked_data.keys()}
            if aux_key is not None:
                if aux_key not in batch:
                    raise ValueError(f"[BaseDataloader] aux_key '{aux_key}' not found in dataset.")
                batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
                del batch[aux_key]
            yield batch

        # Handle the remainder batch if there is any
        if remainder > 0:
            leftover_indices = indices[num_full_batches * batch_size:]
            if remainder_pad:
                # If remainder_pad is True, pad the remainder to form a full batch
                pad_size = batch_size - remainder
                pad_indices = np.random.choice(leftover_indices, pad_size, replace=True)
                final_indices = np.concatenate([leftover_indices, pad_indices])
            else:
                # Otherwise, yield the remainder batch without padding
                final_indices = leftover_indices
            batch = {key: self.stacked_data[key][final_indices] for key in self.stacked_data.keys()}
            if aux_key is not None:
                if aux_key not in batch:
                    raise ValueError(f"[BaseDataloader] aux_key '{aux_key}' not found in dataset.")
                batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
                del batch[aux_key]
            yield batch


# hook for the PoolDataLoader 
def libero_obs_obs_hook(dataset, obs_dim=130):
    """
    Pads or truncates each 1D observation in dataset['observations'] to length `obs_dim`.

    Args:
        dataset: dict containing at least:
            - 'observations': list or 1D-array of numpy arrays of varying length
        obs_dim: int, desired fixed length for each observation

    Returns:
        The same dict, but with dataset['observations'] replaced by a numpy array
        of shape (batch_size, obs_dim).
        All other keys (e.g. 'terminals') are left untouched.
    """
    raw_obs = dataset['observations']
    padded = []
    for obs in raw_obs:
        arr = np.array(obs)
        length = arr.shape[-1]
        if length < obs_dim:
            # pad at end
            arr = np.pad(arr,
                         (0, obs_dim - length),
                         mode='constant',
                         constant_values=0)
        else:
            # truncate to obs_dim
            arr = arr[:obs_dim]
        padded.append(arr)

    # stack into (batch_size, obs_dim)
    dataset['observations'] = np.stack(padded, axis=0)
    print(f"[libero_obs_obs_hook] processed observations → {dataset['observations'].shape}")
    return dataset


def split_trajectories(terminals):
    bounds = []
    start = 0
    for idx, t in enumerate(terminals):
        if t == 1:
            bounds.append((start, idx + 1))
            start = idx + 1
    if start < len(terminals):
        bounds.append((start, len(terminals)))
    return bounds

def history_state_obs_hook(dataset, N=10):
    """
    Given a dataset dict with:
      - 'observations': np.ndarray of shape (T, F)
      - 'terminals':    np.ndarray of shape (T,)
    Splits into trajectories at terminal flags, then for each time t
    builds a new observation by concatenating (o_{t-N}, o_t),
    where if t-N < start_of_traj we use the first frame of that traj.
    Returns a new dataset with updated 'observations' and 'terminals'.
    """
    obs = dataset['observations']
    terms = dataset['terminals']

    new_obs = []

    # split into (start, end) index pairs for each trajectory
    for (s, e) in split_trajectories(terms):
        # for each time within this segment
        for t in range(s, e):
            # determine index for the "history" frame
            if t - N >= s:
                old_idx = t - N
            else:
                old_idx = s
            # concatenate (old, current)
            hist = np.concatenate([obs[old_idx], obs[t]], axis=-1)
            new_obs.append(hist)

    # stack into arrays
    dataset['observations'] = np.stack(new_obs, axis=0)
    print(f"[history_state_obs_hook] processed observations → {dataset['observations'].shape}")
    return dataset

import numpy as np

def history_state_three_obs_hook(dataset, N=10):
    """
    Given a dataset dict with:
      - 'observations': np.ndarray of shape (T, F)
      - 'terminals':    np.ndarray of shape (T,)
    Splits into trajectories at terminal flags, then for each time t
    builds a new observation by concatenating:
      (o_{t-N_or_start}, o_{mid}, o_t),
    where mid = floor((t-N_or_start + t)/2)
    and if t-N < start_of_traj we use the first frame of that traj.
    Returns a new dataset with updated 'observations' and 'terminals'.
    """
    obs = dataset['observations']
    terms = dataset['terminals']

    new_obs = []
    new_terms = []

    for (s, e) in split_trajectories(terms):
        for t in range(s, e):
            old_idx = t - N if t - N >= s else s
            mid_idx = (old_idx + t) // 2
            hist = np.concatenate([
                obs[old_idx],
                obs[mid_idx],
                obs[t]
            ], axis=-1)

            new_obs.append(hist)
            new_terms.append(terms[t])

    dataset['observations'] = np.stack(new_obs, axis=0)
    dataset['terminals']    = np.array(new_terms, dtype=terms.dtype)

    print(f"[history_state_three_obs_hook] processed observations → {dataset['observations'].shape}")
    return dataset



def dropthe_traj_hook(dataset, per_drop: int = 2):
    """
    Drop every `per_drop`-th trajectory from the dataset.

    Args
    ----
    dataset : dict
        Must contain at least
            'observations' : np.ndarray (T, F)
            'terminals'    : np.ndarray (T,)
        Any additional keys (e.g. 'actions', 'rewards', …) are carried over.
    per_drop : int, default 2
        Keep trajectories whose index mod `per_drop` ≠ 0.
        per_drop = 2   → keep 1, 3, 5, …  (drop 0, 2, 4, …)
        per_drop = 3   → keep 1, 2, 4, 5, 7, 8, …  (drop 0, 3, 6, …)

    Returns
    -------
    new_dataset : dict
        Same keys as input but with dropped trajectories removed.
    """
    if per_drop <= 0:
        raise ValueError("per_drop must be ≥ 1")

    # prepare new containers
    new_dataset = {k: [] for k in dataset.keys()}

    # find trajectory boundaries
    bounds = split_trajectories(dataset["terminals"])

    for tid, (s, e) in enumerate(bounds):
        # skip every per_drop-th trajectory
        if tid % per_drop == 0:
            continue

        # copy slices for all keys
        for k in dataset:
            new_dataset[k].append(dataset[k][s:e])

    # concatenate lists back to arrays
    for k, chunks in new_dataset.items():
        new_dataset[k] = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=dataset[k].dtype)

    print(f"[dropthe_traj_hook] kept {len(bounds) - len(bounds)//per_drop} / {len(bounds)} trajectories → "
          f"{new_dataset['observations'].shape[0]} timesteps")
    return new_dataset



def few_frac_shot_hook(dataset, shot=1, frac=1):
    """
    Sample up to `shot` trajectories from the dataset, and within each
    sampled trajectory take every `frac`-th timestep (including the final timestep).

    Args
    ----
    dataset : dict
        Must contain at least
            'observations' : np.ndarray (T, F)
            'terminals'    : np.ndarray (T,)
        Any additional keys (e.g. 'actions', 'rewards', …) are carried over.
    shot : int, default 10
        Maximum number of trajectories to sample (from the start of the dataset).
    frac : int, default 5
        Sampling rate within each trajectory. If `frac == 1` you keep every
        step; if `frac == 2` you keep every 2nd step, and so on. The final
        step of each trajectory is always included.

    Returns
    -------
    new_dataset : dict
        Same keys as input but containing only the sampled timesteps.
    """
    if shot <= 0:
        raise ValueError("shot must be ≥ 1")
    if frac <= 0:
        raise ValueError("frac must be ≥ 1")

    # locate trajectory boundaries
    bounds = split_trajectories(dataset["terminals"])
    if not bounds:
        # no trajectories: return empty arrays
        return {k: dataset[k][:0] for k in dataset}

    # take up to the first `shot` trajectories
    selected_bounds = bounds[:shot]

    # prepare containers for the sampled data
    new_dataset = {k: [] for k in dataset.keys()}

    total_steps = 0
    # for each trajectory, sample indices
    for (start, end) in selected_bounds:
        # every `frac`-th index from start to end-1
        idxs = list(range(start, end, frac))
        # ensure the final step is included
        if (end - 1) not in idxs:
            idxs.append(end - 1)

        total_steps += len(idxs)
        for k, arr in dataset.items():
            # slice out all sampled timesteps at once
            new_dataset[k].append(arr[idxs])

    # concatenate the chunks for each key
    for k, chunks in new_dataset.items():
        if chunks:
            new_dataset[k] = np.concatenate(chunks, axis=0)
        else:
            # preserve original shape beyond time axis
            orig_shape = dataset[k].shape[1:]
            new_dataset[k] = np.empty((0, *orig_shape), dtype=dataset[k].dtype)

    print(f"[few_frac_shot_hook] kept {len(selected_bounds)} trajectories → "
          f"{total_steps} timesteps total with frac {frac}")
    return new_dataset


# if __name__ == "__main__":
#     # Construct a dummy dataset with two trajectories of lengths 5 and 3
#     T, F = 8, 2
#     obs = np.arange(T * F).reshape(T, F)
#     terms = np.zeros(T, dtype=bool)
#     terms[[4, 7]] = True  # mark ends at indices 4 and 7
    
#     dummy_dataset = {"observations": obs, "terminals": terms}


#     print(dummy_dataset)
#     # Test 1: shot=2, frac=2 (should sample both trajectories)
#     out1 = few_frac_shot_hook(dummy_dataset, shot=2, frac=2)
#     expected_idxs = [0, 2, 4, 5, 7]
#     assert out1["observations"].shape[0] == len(expected_idxs)
#     assert np.array_equal(out1["observations"], obs[expected_idxs])

#     # Test 2: shot=1, frac=1 (only first trajectory, all steps)
#     out2 = few_frac_shot_hook(dummy_dataset, shot=1, frac=1)
#     assert out2["observations"].shape[0] == 5
#     assert np.array_equal(out2["observations"], obs[:5])

#     # Test 3: shot larger than available, frac=3
#     out3 = few_frac_shot_hook(dummy_dataset, shot=10, frac=3)
#     # For first traj: idxs [0,3,4], for second: [5,7]
#     expected_idxs3 = [0, 3, 4, 5, 7]
#     assert out3["observations"].shape[0] == len(expected_idxs3)
#     assert np.array_equal(out3["observations"], obs[expected_idxs3])

#     # Test 4: empty dataset
#     empty = {"observations": np.empty((0, F)), "terminals": np.empty((0,), dtype=bool)}
#     out4 = few_frac_shot_hook(empty, shot=3, frac=2)
#     assert out4["observations"].shape[0] == 0

#     print("All tests passed!")
#     exit()

class PoolDataLoader(BaseDataloader):
    def __init__(
            self, 
            data_paths, 
            pre_process_hooks_kwargs=None,
            post_process_hooks_kwargs=None,
            semantics_path=None
        ):
        """
        :param data_paths: List of paths to pickle files containing the dataset.
        :param obs_hook: Optional function to process observations.
        """
        print("[PoolDataLoader] Initializing PoolDataLoader.")
        self.pre_process_hooks_kwargs = pre_process_hooks_kwargs # list of (hook, kwargs)   
        self.post_process_hooks_kwargs = post_process_hooks_kwargs # list of (hook, kwargs) 
        print(f"[PoolDataLoader] Pre-process hooks: {self.pre_process_hooks_kwargs}")
        print(f"[PoolDataLoader] Post-process hooks: {self.post_process_hooks_kwargs}")
        super().__init__(data_paths=data_paths, semantics_path=semantics_path)
        
    def load_data(self):
        """
        1) Load pickle files from self.data_paths
        2) Convert lists to numpy arrays and apply obs_hook if provided
        3) Run per-file post-processing hooks
        4) Collect each file’s dict into data_list
        5) After loading all files, concatenate arrays by key
        6) Run global post-processing hooks on the merged data
        7) Assign the result to self.data_buffer
        """
        data_list = []

        # Step 1–3: load each file, convert, and run per-file hooks
        for path in self.data_paths:
            print(f"[BaseDataloader] Loading data from {path}")
            with open(path, 'rb') as f:
                raw = pickle.load(f)

            per_file = {}
            for key, val in raw.items():
                arr = np.array(val)
                per_file[key] = arr

            if self.pre_process_hooks_kwargs:
                for hook, kwargs in self.pre_process_hooks_kwargs:
                    per_file = hook(per_file, **kwargs)

            data_list.append(per_file)

        # Step 5: merge all files by concatenating each key’s arrays
        merged = {
            key: np.concatenate([d[key] for d in data_list], axis=0)
            for key in data_list[0].keys()
        }

        # Step 6: run global post-processing hooks
        if self.post_process_hooks_kwargs:
            for hook, kwargs in self.post_process_hooks_kwargs:
                merged = hook(merged, **kwargs)

        # Step 7: store in data_buffer
        self.data_buffer = merged


    def get_all_batch(self, batch_size, aux_key=None, shuffle=True, remainder_pad=True, pool_key=None):
        # Override the get_all_batch method to include pool_key functionality
        if pool_key is None:
            yield from super().get_all_batch(batch_size, aux_key, shuffle, remainder_pad)
            return
        
        if pool_key not in self.stacked_data:
            raise ValueError(f"[PoolDataLoader] pool_key '{pool_key}' not found in dataset.")
        
        unique_pools = np.unique(self.stacked_data[pool_key])
        for pool_idx in unique_pools:
            pool_indices = np.where(self.stacked_data[pool_key] == pool_idx)[0]
            
            pool_size = len(pool_indices)
            num_batches = pool_size // batch_size
            remainder = pool_size % batch_size
            
            for i in range(num_batches):
                batch_indices = pool_indices[i * batch_size : (i + 1) * batch_size]
                batch = {key: self.stacked_data[key][batch_indices] for key in self.stacked_data.keys()}
                if aux_key is not None:
                    if aux_key not in batch:
                        raise ValueError(f"[PoolDataLoader] aux_key '{aux_key}' not found in dataset.")
                    batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
                    del batch[aux_key]
                yield batch

            if remainder > 0:
                leftover_indices = pool_indices[num_batches * batch_size:]
                if remainder_pad:
                    pad_size = batch_size - remainder
                    pad_indices = np.random.choice(leftover_indices, pad_size, replace=True)
                    final_indices = np.concatenate([leftover_indices, pad_indices])
                else:
                    final_indices = leftover_indices
                batch = {key: self.stacked_data[key][final_indices] for key in self.stacked_data.keys()}
                if aux_key is not None:
                    if aux_key not in batch:
                        raise ValueError(f"[PoolDataLoader] aux_key '{aux_key}' not found in dataset.")
                    batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
                    del batch[aux_key]
                yield batch

class MemoryBuffer(PoolDataLoader):
    """
    A memory buffer that can start empty (data_paths=None).
    When add_new_dataset is called with new data, the buffer keeps all old data
    and adds only a fraction (keep_ratio) of the newly loaded data.
    """
    def __init__(self, data_paths=None):
        """
        :param data_paths: Optional list of .pkl file paths. If None or empty, start empty.
        """
        print("[MemoryBuffer] Initializing MemoryBuffer.")
        super().__init__(data_paths=data_paths)

    def load_data(self):
        """
        If no data_paths are provided, initialize an empty buffer.
        Otherwise, use the parent method to load data.
        """
        if not self.data_paths:
            print("[MemoryBuffer] No data paths provided; initializing empty buffer.")
            self.data_buffer = {}
        else:
            super().load_data()

    def process_data(self):
        """
        If data_buffer is empty, set stacked_data to empty and dataset_size to 0.
        Otherwise, use the parent method.
        """
        if not self.data_buffer:
            print("[MemoryBuffer] No data to process; buffer is empty.")
            self.stacked_data = {}
            self.dataset_size = 0
        else:
            super().process_data()

    def add_new_dataset(self, new_dataloader, keep_ratio=0.1, sample_function=None):
        """
        Add new data from a dataloader to the memory buffer.
        Keeps all old data and adds only a 'keep_ratio' fraction of the new data.
        """
        print(f"[MemoryBuffer] Adding new dataset from dataloader, keep_ratio={keep_ratio:.2f}")

        # Preserve all old data
        old_sample = self.stacked_data if self.dataset_size > 0 else None
        if old_sample:
            print(f"[MemoryBuffer] Old data size = {self.dataset_size}.")

        # Merge new data from the dataloader
        new_merged = new_dataloader.stacked_data

        # Subsample new data
        if sample_function is not None:
            print(f"[MemoryBuffer] Using custom sample function to sample new data.")
            new_sample = sample_function(new_merged, keep_ratio)
        else :  
            print(f"[MemoryBuffer] Using default sampling method to sample new data.")
            new_sample = self._sample_new_data(new_merged, keep_ratio)

        # Merge old and new data
        merged_data = self._merge_old_and_new(old_sample, new_sample)

        self.data_buffer = merged_data
        self.process_data()
        print(f"[MemoryBuffer] Buffer size: {self.dataset_size}")

    # -----------------------   
    # basic replay strategy
    # -----------------------
    def _sample_new_data(self, new_data_dict, keep_ratio):
        """
        Sample a fraction ('keep_ratio') of the 'observations' from new_data_dict.
        """
        if 'observations' in new_data_dict and len(new_data_dict['observations']) > 0:
            new_size = len(new_data_dict['observations'])
            keep_count = int(new_size * keep_ratio)
            print(f"[MemoryBuffer] New data size = {new_size}, keeping {keep_count} items.")

            if keep_count > 0:
                indices = np.random.choice(new_size, keep_count, replace=False)
                return {k: v[indices] for k, v in new_data_dict.items()}
            else:
                print("[MemoryBuffer] keep_ratio=0, no new data kept.")
                return None
        else:
            print("[MemoryBuffer] No 'observations' or empty new data.")
            return None

    def _merge_old_and_new(self, old_sample, new_sample):
        """
        Merge old data with new_sample.
        """
        merged_data = {}
        if old_sample:
            for key, val in old_sample.items():
                merged_data[key] = val.copy()
        if new_sample:
            for key, val in new_sample.items():
                if key in merged_data:
                    merged_data[key] = np.concatenate([merged_data[key], val], axis=0)
                else:
                    merged_data[key] = val
        return merged_data

    def is_empty(self):
        """
        Check if the buffer is empty.
        """
        return self.dataset_size == 0

import numpy as np
from itertools import zip_longest

class DataloaderMixer:
    """
    A mixer that aggregates multiple dataloaders to behave as a single one.
    It accepts a list of dataloaders and an optional list of mixing ratios (weights)
    that determine how much each dataloader contributes to the total batch.
    If mixing ratios are not provided, uniform weights are used.
    
    Additionally, a reference dataloader index can be provided to determine the iteration length
    for get_all_batch. By default, the first dataloader (index 0) is used as the reference.
    """
    def __init__(self, dataloaders, mixing_ratios=None, reference_idx=0):
        """
        :param dataloaders: A list of dataloader objects (e.g., instances of BaseDataloader or its subclasses)
        :param mixing_ratios: Optional list of mixing ratios (weights) for each dataloader.
                              Must have the same length as dataloaders if provided.
        :param reference_idx: Optional index of the reference dataloader to determine the iteration length.
                              Defaults to 0 (the first dataloader).
        """
        if not dataloaders:
            raise ValueError("[DataloaderMixer] No dataloaders provided.")
        self.dataloaders = dataloaders
        if mixing_ratios is not None:
            if len(mixing_ratios) != len(dataloaders):
                raise ValueError("[DataloaderMixer] Length of mixing_ratios must match dataloaders.")
            self.weights = mixing_ratios
        else:
            # Default to uniform weights if mixing_ratios is not provided.
            self.weights = [1] * len(self.dataloaders)
        self.reference_idx = reference_idx
        print(f"[DataloaderMixer] Initialized with {len(dataloaders)} dataloaders.")
        print(f"[DataloaderMixer] Mixing weights: {self.weights}")
        print(f"[DataloaderMixer] Reference dataloader index: {self.reference_idx}")

    def _allocate_batch_sizes(self, total_batch_size):
        """
        Allocate the total_batch_size among dataloaders proportionally to the provided weights.
        Uses rounding to ensure the sum of allocations equals total_batch_size.
        
        :param total_batch_size: Total batch size to be allocated.
        :return: List of allocated batch sizes for each dataloader.
        """
        total_weight = sum(self.weights)
        # Compute ideal allocation (can be fractional)
        ideal_alloc = [total_batch_size * (w / total_weight) for w in self.weights]
        # Floor the allocations to get integer parts
        alloc_int = [int(np.floor(x)) for x in ideal_alloc]
        remainder = total_batch_size - sum(alloc_int)
        # Distribute the remaining samples to dataloaders with the highest fractional parts.
        fractional_parts = [x - np.floor(x) for x in ideal_alloc]
        sorted_indices = np.argsort(fractional_parts)[::-1]
        for i in range(remainder):
            alloc_int[sorted_indices[i]] += 1
        return alloc_int

    def get_rand_batch(self, batch_size, aux_key=None):
        """
        Retrieve a random batch from each dataloader according to the allocated batch sizes,
        then merge them. The allocated sizes are determined based on the mixing weights.
        
        :param batch_size: Total batch size to retrieve.
        :param aux_key: Optional key whose data (if present) is concatenated with 'observations'.
        :return: A dictionary with keys identical to the underlying batches and values concatenated along axis 0.
        """
        allocations = self._allocate_batch_sizes(batch_size)
        batch_list = []
        for alloc, loader in zip(allocations, self.dataloaders):
            batch = loader.get_rand_batch(batch_size=alloc, aux_key=aux_key)
            batch_list.append(batch)
        merged_batch = {}
        for key in batch_list[0].keys():
            merged_batch[key] = np.concatenate([b[key] for b in batch_list], axis=0)
        return merged_batch

    def get_all_batch(self, batch_size, aux_key=None, shuffle=True, remainder_pad=True):
        """
        Yield combined batches from the mixer by concurrently iterating over each dataloader's generator.
        Batch sizes are allocated proportionally based on the provided mixing weights.
        If a dataloader's generator yields a final batch smaller than its allocated portion,
        its missing samples are filled using get_rand_batch, ensuring the overall batch always has exactly
        the specified batch_size.
        
        The iteration length is determined by a reference dataloader (specified in __init__ via reference_idx).
        
        :param batch_size: Total batch size to yield per combined batch.
        :param aux_key: Optional key to be used for auxiliary concatenation.
        :param shuffle: Whether to shuffle indices in each individual dataloader.
        :param remainder_pad: If True, pad remainder batches to full batch size.
        :yield: A dictionary representing a combined batch.
        """
        allocations = self._allocate_batch_sizes(batch_size)
        # Create a generator for each dataloader with its allocated batch size.
        generators = []
        for alloc, loader in zip(allocations, self.dataloaders):
            gen = loader.get_all_batch(
                batch_size=alloc,
                aux_key=aux_key,
                shuffle=shuffle,
                remainder_pad=remainder_pad
            )
            generators.append((loader, gen, alloc))
        
        # Use the reference dataloader specified by self.reference_idx to determine iteration length.
        reference_loader, reference_gen, _ = generators[self.reference_idx]
        for ref_batch in reference_gen:
            sub_batches = [ref_batch]
            # Iterate over the other dataloaders.
            for idx, (loader, gen, alloc) in enumerate(generators):
                if idx == self.reference_idx:
                    continue
                try:
                    batch = next(gen)
                except StopIteration:
                    batch = loader.get_rand_batch(batch_size=alloc, aux_key=aux_key)
                if batch is None or (batch["observations"].shape[0] < alloc):
                    batch = loader.get_rand_batch(batch_size=alloc, aux_key=aux_key)
                sub_batches.append(batch)
            merged_batch = {}
            for key in sub_batches[0].keys():
                merged_batch[key] = np.concatenate([b[key] for b in sub_batches], axis=0)
            yield merged_batch

import copy
import numpy as np

class DataloaderExtender:
    '''
    class for skill incremental learning
    '''
    def __init__(self, dataloaders, mixing_ratios=None, reference_idx=0 ): 
        """
        :param dataloaders: A list of dataloader objects, each having a nested dictionary 'stacked_data'.
        """
        self.merged_data = self._merge_stacked_data(dataloaders)
        if 'observations' not in self.merged_data:
            raise ValueError("Merged data must contain 'observations' key.")
        self.dataset_size = len(self.merged_data['observations'])
        print(f"[DataloaderExtender] Merged dataset size: {self.dataset_size}")

    def _merge_stacked_data(self, dataloaders):
        """
        Deepcopy each dataloader's stacked_data and merge them key-wise.
        If keys differ among dataloaders, raise a warning and exit.
        For np.array values, use np.concatenate along axis 0;
        for list values, use list concatenation.
        """
        merged = {}
        first_keys = None
        for loader in dataloaders:
            data = copy.deepcopy(loader.stacked_data)
            if first_keys is None:
                first_keys = set(data.keys())
            else:
                if set(data.keys()) != first_keys:
                    raise ValueError("Mismatch in keys among dataloaders' stacked_data. Aborting.")
            for key, value in data.items():
                if key not in merged:
                    merged[key] = value
                else:
                    if isinstance(value, np.ndarray):
                        merged[key] = np.concatenate([merged[key], value], axis=0)
                    elif isinstance(value, list):
                        merged[key] = merged[key] + value
                    else:
                        raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")
        return merged

    def _index_data(self, data, indices):
        """
        Index data with given indices.
        If data is a np.array, use numpy indexing;
        if data is a list, use list comprehension.
        """
        if isinstance(data, np.ndarray):
            return data[indices]
        elif isinstance(data, list):
            return [data[i] for i in indices]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def get_rand_batch(self, batch_size=None, aux_key=None):
        """
        Randomly sample a batch from the merged dataset.
        If batch_size is -1 or None, return the entire merged dataset.
        If aux_key is provided, concatenate the aux_key data with 'observations' along axis -1.
        """
        if batch_size is None or batch_size == -1:
            batch = {k: self.merged_data[k] for k in self.merged_data}
        else:
            if batch_size > self.dataset_size:
                indices = np.random.choice(self.dataset_size, batch_size, replace=True)
            else:
                indices = np.random.choice(self.dataset_size, batch_size, replace=False)
            batch = {key: self._index_data(value, indices) for key, value in self.merged_data.items()}
        
        if aux_key is not None:
            if aux_key not in batch:
                raise ValueError(f"aux_key '{aux_key}' not found in merged data.")
            # Require numpy arrays for aux_key concatenation
            if not (isinstance(batch['observations'], np.ndarray) and isinstance(batch[aux_key], np.ndarray)):
                raise ValueError("aux_key concatenation requires numpy array types.")
            batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
            del batch[aux_key]
        return batch

    def get_all_batch(self, batch_size, aux_key=None, shuffle=True, remainder_pad=True):
        """
        Yield batches from the merged dataset with batch_size.
        If shuffle is True, shuffle dataset indices.
        If remainder_pad is True, pad the final batch to batch_size using random sampling.
        If aux_key is provided, concatenate that key's data with 'observations' along axis -1.
        """
        indices = np.arange(self.dataset_size)
        if shuffle:
            np.random.shuffle(indices)
        num_full_batches = self.dataset_size // batch_size
        remainder = self.dataset_size % batch_size

        # Yield full batches
        for i in range(num_full_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch = {key: self._index_data(value, batch_indices) for key, value in self.merged_data.items()}
            if aux_key is not None:
                if aux_key not in batch:
                    raise ValueError(f"aux_key '{aux_key}' not found in merged data.")
                if not (isinstance(batch['observations'], np.ndarray) and isinstance(batch[aux_key], np.ndarray)):
                    raise ValueError("aux_key concatenation requires numpy array types.")
                batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
                del batch[aux_key]
            yield batch

        # Yield remainder batch if exists
        if remainder > 0:
            leftover_indices = indices[num_full_batches * batch_size:]
            if remainder_pad:
                pad_size = batch_size - remainder
                pad_indices = np.random.choice(indices, pad_size, replace=True)
                final_indices = np.concatenate([leftover_indices, pad_indices])
            else:
                final_indices = leftover_indices
            batch = {key: self._index_data(value, final_indices) for key, value in self.merged_data.items()}
            if aux_key is not None:
                if aux_key not in batch:
                    raise ValueError(f"aux_key '{aux_key}' not found in merged data.")
                if not (isinstance(batch['observations'], np.ndarray) and isinstance(batch[aux_key], np.ndarray)):
                    raise ValueError("aux_key concatenation requires numpy array types.")
                batch['observations'] = np.concatenate([batch['observations'], batch[aux_key]], axis=-1)
                del batch[aux_key]
            yield batch

# --------------------- Test Code Example ---------------------
# DummyDataloader simulates a dataloader with a fixed number of samples.
class DummyDataloader:
    def __init__(self, num_samples, name):
        self.dataset_size = num_samples
        self.name = name
        # Create dummy data with shape (num_samples, 10)
        self.data = np.random.randn(num_samples, 10)
    
    def get_rand_batch(self, batch_size, aux_key=None):
        indices = np.random.choice(self.dataset_size, batch_size, replace=True)
        batch = {
            "observations": self.data[indices],
            "source": np.array([self.name] * batch_size)
        }
        return batch
    
    def get_all_batch(self, batch_size, aux_key=None, shuffle=True, remainder_pad=True):
        indices = np.arange(self.dataset_size)
        if shuffle:
            np.random.shuffle(indices)
        num_full_batches = self.dataset_size // batch_size
        remainder = self.dataset_size % batch_size
        
        for i in range(num_full_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch = {
                "observations": self.data[batch_indices],
                "source": np.array([self.name] * len(batch_indices))
            }
            yield batch
        
        if remainder > 0:
            batch_indices = indices[num_full_batches * batch_size:]
            if remainder_pad:
                pad_size = batch_size - remainder
                pad_indices = np.random.choice(indices, pad_size, replace=True)
                batch_indices = np.concatenate([batch_indices, pad_indices])
            batch = {
                "observations": self.data[batch_indices],
                "source": np.array([self.name] * len(batch_indices))
            }
            yield batch

class DummyExtenderDataloader:
    def __init__(self, num_samples, name, f=10, extra_dim=5):
        self.num_samples = num_samples
        self.name = name
        # stacked_data: 'observations', 'labels', 'meta' (list), 'extra'
        self.stacked_data = {
            "observations": np.random.randn(num_samples, f),
            "labels": np.random.randint(0, 10, size=(num_samples,)),
            "meta": [name] * num_samples,
            "extra": np.random.randn(num_samples, extra_dim)
        }

def test_dataloader_extender():
    print("==== Testing DataloaderExtender ====")
    # Create dummy dataloaders with consistent keys in stacked_data.
    loader1 = DummyExtenderDataloader(num_samples=100, name="loader1")
    loader2 = DummyExtenderDataloader(num_samples=200, name="loader2")
    loader3 = DummyExtenderDataloader(num_samples=150, name="loader3")
    
    # Instantiate DataloaderExtender with the dummy dataloaders.
    extender = DataloaderExtender(dataloaders=[loader1, loader2, loader3])
    
    print("Merged keys:", list(extender.merged_data.keys()))
    print("Merged 'observations' shape:", extender.merged_data["observations"].shape)
    print("Merged 'labels' shape:", extender.merged_data["labels"].shape)
    print("Merged 'meta' length:", len(extender.merged_data["meta"]))
    print("Merged 'extra' shape:", extender.merged_data["extra"].shape)
    
    # Test get_rand_batch without aux_key.
    rand_batch = extender.get_rand_batch(batch_size=50)
    print("\n[get_rand_batch] Without aux_key:")
    print("observations shape:", rand_batch["observations"].shape)
    print("labels shape:", rand_batch["labels"].shape)
    print("meta length:", len(rand_batch["meta"]))
    
    # Test get_rand_batch with aux_key 'extra'.
    rand_batch_aux = extender.get_rand_batch(batch_size=50, aux_key="extra")
    print("\n[get_rand_batch] With aux_key 'extra':")
    print("observations shape (should have extra columns concatenated):", rand_batch_aux["observations"].shape)
    print("Extra key exists in batch?", "extra" in rand_batch_aux)
    
    # Test get_all_batch without aux_key.
    print("\n[get_all_batch] Without aux_key:")
    batch_count = 0
    for batch in extender.get_all_batch(batch_size=50, shuffle=False, remainder_pad=True):
        print(f"Batch {batch_count}: observations shape: {batch['observations'].shape}")
        batch_count += 1
    print("Total batches yielded:", batch_count)
    
    # Test get_all_batch with aux_key 'extra'.
    print("\n[get_all_batch] With aux_key 'extra':")
    batch_count = 0
    for batch in extender.get_all_batch(batch_size=50, shuffle=False, remainder_pad=True, aux_key="extra"):
        print(f"Batch {batch_count}: observations shape: {batch['observations'].shape}")
        batch_count += 1
    print("Total batches yielded:", batch_count)
# --------------------- Test Code for Reference Dataloader Batch Count ---------------------
def test_get_all_batch_reference():
    """
    This test checks that get_all_batch() in DataloaderMixer
    iterates according to the reference dataloader's (loader1) number of batches.
    """
    # Create dummy dataloaders with different dataset sizes
    loader1 = DummyDataloader(num_samples=10000, name="loader1")  # Reference dataloader (shortest)
    loader2 = DummyDataloader(num_samples=20000, name="loader2")
    loader3 = DummyDataloader(num_samples=15000, name="loader3")
    
    mixing_ratios = [1, 3, 2]
    # Use loader1 (index 0) as the reference dataloader
    mixer = DataloaderMixer(dataloaders=[loader1, loader2, loader3],
                            mixing_ratios=mixing_ratios,
                            reference_idx=0)
    
    total_batch_size = 1024
    # Calculate allocated batch size for loader1 using the mixer's internal method
    allocations = mixer._allocate_batch_sizes(total_batch_size)
    alloc_loader1 = allocations[0]
    num_full_batches = loader1.dataset_size // alloc_loader1
    remainder = loader1.dataset_size % alloc_loader1
    expected_batches = num_full_batches + (1 if remainder > 0 else 0)
    
    # Count the number of batches yielded by get_all_batch()
    actual_batches = 0
    for batch in mixer.get_all_batch(batch_size=total_batch_size, shuffle=True, remainder_pad=True):
        actual_batches += 1
    print("Test for get_all_batch() using reference dataloader (loader1):")
    print(f"Expected number of batches (based on loader1): {expected_batches}")
    print(f"Actual number of batches yielded: {actual_batches}")

# --------------------- Test Function for PoolDataLoader ---------------------
def test_pool_data_loader():
    """
    Test the PoolDataLoader by creating dummy data that includes a 'pool' key.
    This function creates a temporary pickle file, instantiates the PoolDataLoader,
    and prints out a few batches based on the provided pool_key.
    """
    # Create dummy data with a 'pool' key (e.g., two pools: 0 and 1)
    dummy_data = {
        "observations": np.random.randn(100, 10),
        "skills": np.random.randint(0, 3, 100),
        "pool": np.random.randint(0, 2, 100)  # Pool key with values 0 or 1
    }
    dummy_file = "data/dummy_pool.pkl"
    with open(dummy_file, "wb") as f:
        pickle.dump(dummy_data, f)
    
    # Instantiate PoolDataLoader with the dummy pickle file
    # Note: The PoolDataLoader should be defined elsewhere in your codebase.
    pool_loader = PoolDataLoader(data_paths=[dummy_file])
    
    batch_size = 20
    print("\n--- Testing PoolDataLoader with pool_key 'pool' ---")
    # Iterate over a few batches produced using the pool_key.
    for i, batch in enumerate(pool_loader.get_all_batch(batch_size=batch_size, pool_key="pool")):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            print(f"  {key} shape: {value.shape}")
        if "pool" in batch:
            print("  Unique pool values in batch:", np.unique(batch["pool"]))

# --------------------- Main Test Code ---------------------
if __name__ == "__main__":
    # Test get_rand_batch and get_all_batch from DataloaderMixer
    loader1 = DummyDataloader(num_samples=10000, name="loader1")
    loader2 = DummyDataloader(num_samples=20000, name="loader2")
    loader3 = DummyDataloader(num_samples=15000, name="loader3")
    
    mixer = DataloaderMixer(dataloaders=[loader1, loader2, loader3], mixing_ratios=[1, 3, 2])
    
    total_batch_size = 1024
    # Test get_rand_batch
    rand_batch = mixer.get_rand_batch(batch_size=total_batch_size)
    print("Random Batch:")
    print("Observations shape:", rand_batch["observations"].shape)
    print("Unique sources:", np.unique(rand_batch["source"]))
    
    # Test get_all_batch (print first 3 batches)
    print("\nAll Batches (first 3 batches):")
    for i, batch in enumerate(mixer.get_all_batch(batch_size=total_batch_size, shuffle=True, remainder_pad=True)):
        print(f"Batch {i}: Observations shape: {batch['observations'].shape}, Unique sources: {np.unique(batch['source'])}")
        if i >= 2:
            break
    
    # Run test for get_all_batch() using the reference dataloader
    print("\n--- Testing get_all_batch() based on reference dataloader ---")
    test_get_all_batch_reference()
    

    test_dataloader_extender()
    # Run the test for PoolDataLoader
    # print("\n--- Testing PoolDataLoader ---")
    # test_pool_data_loader()