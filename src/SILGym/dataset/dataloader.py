"""
Dataset dataloader module for SILGym.

This module provides classes for loading and managing training datasets with support
for various preprocessing hooks, memory buffering, and dataset mixing.

Dataset Format:
===============
All datasets should be pickle (.pkl) files containing a dictionary with the following structure:

Required Keys:
- 'observations': np.ndarray of shape (N, obs_dim) - State observations
- 'actions': np.ndarray of shape (N, action_dim) - Actions taken
- 'terminals': np.ndarray of shape (N,) - Binary flags indicating episode ends
- 'skills': np.ndarray of shape (N,) - Skill identifiers (for skill-based datasets)

Optional Keys:
- 'rewards': np.ndarray of shape (N,) - Reward values
- 'next_observations': np.ndarray of shape (N, obs_dim) - Next state observations
- 'pool': np.ndarray of shape (N,) - Pool/task identifiers for multi-task learning
- Any additional custom keys for specific use cases

Example Dataset Structure:
{
    'observations': np.array([[...], [...], ...]),  # Shape: (1000, 64)
    'actions': np.array([[...], [...], ...]),       # Shape: (1000, 7)
    'terminals': np.array([0, 0, ..., 1, 0, ...]),  # Shape: (1000,)
    'skills': np.array([0, 0, ..., 1, 1, ...]),     # Shape: (1000,)
    'rewards': np.array([0.1, 0.2, ..., 1.0, ...])  # Shape: (1000,)
}

Note: Trajectory boundaries are determined by terminal=1 flags.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from SILGym.utils.logger import get_logger
import threading
from queue import Queue, Empty
import time

# ============================================================================
# Base Dataloader
# ============================================================================

class BaseDataloader:
    """
    Basic dataloader with support for:
    - Loading multiple pickle files
    - Optional semantic embedding concatenation
    - Random batch sampling
    - Sequential batch iteration with optional padding
    """
    
    def __init__(
        self, 
        data_paths: List[str], 
        semantics_path: Optional[str] = None
    ):
        """
        Args:
            data_paths: List of paths to pickle files containing datasets
            semantics_path: Optional path to semantic embeddings file
        """
        self.data_paths = data_paths
        self.semantics_path = semantics_path
        self.data_buffer = {}
        self.stacked_data = {}
        self.dataset_size = 0
        self.logger = get_logger(__name__)
        
        self.logger.info(f"[BaseDataloader] Initializing with {len(data_paths)} data files")
        self.load_data()
        self.process_data()
    
    def load_data(self):
        """Load and merge data from all provided pickle files."""
        for path in self.data_paths:
            self.logger.info(f"[BaseDataloader] Loading data from {path}")
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Merge data by concatenating arrays for each key
            for key, value in loaded_data.items():
                arr = np.array(value)
                if key not in self.data_buffer:
                    self.data_buffer[key] = arr
                else:
                    self.data_buffer[key] = np.concatenate([self.data_buffer[key], arr], axis=0)
    
    def process_data(self):
        """Process loaded data and optionally add semantic embeddings."""
        self.stacked_data = self.data_buffer
        
        # Handle empty data
        if not self.stacked_data or 'observations' not in self.stacked_data:
            self.dataset_size = 0
            return
            
        self.dataset_size = len(self.stacked_data['observations'])
        
        # Add semantic embeddings if provided
        if self.semantics_path:
            self.logger.info("[BaseDataloader] Adding semantic embeddings to observations")
            with open(self.semantics_path, 'rb') as f:
                semantics = pickle.load(f)
            
            embeddings = np.array([semantics[skill] for skill in self.stacked_data['skills']])
            self.stacked_data['observations'] = np.concatenate(
                [self.stacked_data['observations'], embeddings], axis=-1
            )
            self.logger.info(f"[BaseDataloader] Final obs shape: {self.stacked_data['observations'].shape}")
        
        self.logger.info(f"[BaseDataloader] Dataset size: {self.dataset_size}")
    
    def get_rand_batch(
        self, 
        batch_size: Optional[int] = None, 
        aux_key: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get a random batch from the dataset.
        
        Args:
            batch_size: Number of samples (None or -1 for entire dataset)
            aux_key: Optional key to concatenate with observations
            
        Returns:
            Dictionary containing batch data
        """
        if batch_size is None or batch_size == -1:
            indices = np.arange(self.dataset_size)
        else:
            indices = np.random.choice(self.dataset_size, batch_size, replace=False)
        
        batch = {key: self.stacked_data[key][indices] for key in self.stacked_data}
        
        # Handle auxiliary key concatenation
        if aux_key and aux_key in batch:
            batch['observations'] = np.concatenate(
                [batch['observations'], batch[aux_key]], axis=-1
            )
            del batch[aux_key]
        
        return batch
    
    def get_all_batch(
        self, 
        batch_size: int,
        aux_key: Optional[str] = None,
        shuffle: bool = True,
        remainder_pad: bool = True
    ):
        """
        Yield all data in batches.
        
        Args:
            batch_size: Size of each batch
            aux_key: Optional key to concatenate with observations
            shuffle: Whether to shuffle data before batching
            remainder_pad: Whether to pad the last batch if smaller than batch_size
            
        Yields:
            Dictionary containing batch data
        """
        indices = np.arange(self.dataset_size)
        if shuffle:
            np.random.shuffle(indices)
        
        # Full batches
        for i in range(0, self.dataset_size - self.dataset_size % batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self._create_batch(batch_indices, aux_key)
        
        # Remainder batch
        remainder = self.dataset_size % batch_size
        if remainder > 0:
            leftover_indices = indices[-remainder:]
            if remainder_pad:
                # Pad to full batch size
                pad_indices = np.random.choice(leftover_indices, batch_size - remainder, replace=True)
                batch_indices = np.concatenate([leftover_indices, pad_indices])
            else:
                batch_indices = leftover_indices
            yield self._create_batch(batch_indices, aux_key)
    
    def _create_batch(self, indices: np.ndarray, aux_key: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Helper to create a batch from indices."""
        batch = {key: self.stacked_data[key][indices] for key in self.stacked_data}
        
        if aux_key and aux_key in batch:
            batch['observations'] = np.concatenate(
                [batch['observations'], batch[aux_key]], axis=-1
            )
            del batch[aux_key]
        
        return batch


# ============================================================================
# Advanced Dataloaders
# ============================================================================

class PoolDataLoader(BaseDataloader):
    """
    Extended dataloader with support for:
    - Pre-processing hooks (applied per file)
    - Post-processing hooks (applied to merged data)
    - Pool-based batch iteration
    """
    
    def __init__(
        self,
        data_paths: List[str],
        pre_process_hooks_kwargs: Optional[List[Tuple[Callable, Dict]]] = None,
        post_process_hooks_kwargs: Optional[List[Tuple[Callable, Dict]]] = None,
        semantics_path: Optional[str] = None
    ):
        """
        Args:
            data_paths: List of paths to pickle files
            pre_process_hooks_kwargs: List of (hook_function, kwargs) for per-file processing
            post_process_hooks_kwargs: List of (hook_function, kwargs) for merged data processing
            semantics_path: Optional path to semantic embeddings
        """
        self.pre_process_hooks_kwargs = pre_process_hooks_kwargs or []
        self.post_process_hooks_kwargs = post_process_hooks_kwargs or []
        self.logger = get_logger(__name__)
        
        self.logger.info(f"[PoolDataLoader] Pre-process hooks: {len(self.pre_process_hooks_kwargs)}")
        self.logger.info(f"[PoolDataLoader] Post-process hooks: {len(self.post_process_hooks_kwargs)}")
        
        super().__init__(data_paths=data_paths, semantics_path=semantics_path)
    
    def load_data(self):
        """Load data with pre- and post-processing hooks."""
        data_list = []
        
        # Load each file with pre-processing
        for path in self.data_paths:
            self.logger.info(f"[PoolDataLoader] Loading data from {path}")
            with open(path, 'rb') as f:
                raw = pickle.load(f)
            
            # Convert to numpy arrays
            data = {key: np.array(val) for key, val in raw.items()}
            
            # Apply pre-processing hooks
            for hook, kwargs in self.pre_process_hooks_kwargs:
                data = hook(data, **kwargs)
            
            data_list.append(data)
        
        # Merge all data
        if data_list:
            self.data_buffer = {
                key: np.concatenate([d[key] for d in data_list], axis=0)
                for key in data_list[0].keys()
            }
        else:
            self.data_buffer = {}
        
        # Apply post-processing hooks
        for hook, kwargs in self.post_process_hooks_kwargs:
            self.data_buffer = hook(self.data_buffer, **kwargs)
    
    def get_all_batch(
        self, 
        batch_size: int,
        aux_key: Optional[str] = None,
        shuffle: bool = True,
        remainder_pad: bool = True,
        pool_key: Optional[str] = None
    ):
        """
        Yield batches, optionally grouped by pool.
        
        Args:
            batch_size: Size of each batch
            aux_key: Optional key to concatenate with observations
            shuffle: Whether to shuffle within pools
            remainder_pad: Whether to pad incomplete batches
            pool_key: Optional key to group data by pools
            
        Yields:
            Dictionary containing batch data
        """
        if pool_key is None:
            yield from super().get_all_batch(batch_size, aux_key, shuffle, remainder_pad)
            return
        
        if pool_key not in self.stacked_data:
            raise ValueError(f"[PoolDataLoader] pool_key '{pool_key}' not found in dataset")
        
        # Iterate through each pool
        for pool_id in np.unique(self.stacked_data[pool_key]):
            pool_indices = np.where(self.stacked_data[pool_key] == pool_id)[0]
            if shuffle:
                np.random.shuffle(pool_indices)
            
            # Yield batches for this pool
            for i in range(0, len(pool_indices), batch_size):
                end_idx = min(i + batch_size, len(pool_indices))
                batch_indices = pool_indices[i:end_idx]
                
                # Handle padding
                if len(batch_indices) < batch_size and remainder_pad:
                    pad_size = batch_size - len(batch_indices)
                    pad_indices = np.random.choice(pool_indices, pad_size, replace=True)
                    batch_indices = np.concatenate([batch_indices, pad_indices])
                
                yield self._create_batch(batch_indices, aux_key)


class MemoryBuffer(PoolDataLoader):
    """
    Memory buffer for continual learning scenarios.
    Maintains a buffer of past data and can incorporate new data with configurable retention.
    """
    
    def __init__(self, data_paths: Optional[List[str]] = None):
        """Initialize buffer, optionally with initial data."""
        # Handle empty initialization
        if data_paths is None:
            data_paths = []
            
        # Initialize parent class (handles all attributes)
        super().__init__(data_paths=data_paths, semantics_path=None)
    
    def add_new_dataset(
        self, 
        new_dataloader: BaseDataloader,
        keep_ratio: float = 0.0,
        sample_function: Optional[Callable] = None
    ):
        """
        Add new data to the buffer, keeping all old data and a fraction of new data.
        
        Args:
            new_dataloader: Dataloader containing new data
            keep_ratio: Fraction of new data to keep (0.0 to 1.0)
            sample_function: Optional custom sampling function
        """
        self.logger.info(f"[MemoryBuffer] Adding new dataset with keep_ratio={keep_ratio:.2f}")
        
        # Preserve old data
        old_data = self.stacked_data.copy() if self.dataset_size > 0 else {}
        
        # Sample new data
        new_data = new_dataloader.stacked_data
        if sample_function:
            sampled_new = sample_function(new_data, keep_ratio)
        else:
            sampled_new = self._sample_data(new_data, keep_ratio)
        
        # Merge old and new
        self.data_buffer = self._merge_data(old_data, sampled_new)
        self.process_data()
        
        self.logger.info(f"[MemoryBuffer] Updated buffer size: {self.dataset_size}")
    
    def _sample_data(self, data: Dict[str, np.ndarray], keep_ratio: float) -> Optional[Dict[str, np.ndarray]]:
        """Default sampling strategy."""
        new_size = len(data['observations'])
        keep_count = int(new_size * keep_ratio)
        self.logger.info(f"[MemoryBuffer] New data size = {new_size}, keeping {keep_count} items.")

        if keep_count > 0:
            indices = np.random.choice(new_size, keep_count, replace=False)
            return {k: v[indices] for k, v in data.items()}
        else:
            self.logger.info("[MemoryBuffer] keep_ratio=0, no new data kept.")
            return None
    
    def _merge_data(
        self, 
        old_data: Dict[str, np.ndarray], 
        new_data: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Merge old and new data dictionaries."""
        if not new_data:
            return old_data
        
        if not old_data:
            return new_data
        
        merged = {}
        all_keys = set(old_data.keys()) | set(new_data.keys())
        
        for key in all_keys:
            if key in old_data and key in new_data:
                merged[key] = np.concatenate([old_data[key], new_data[key]], axis=0)
            elif key in old_data:
                merged[key] = old_data[key]
            else:
                merged[key] = new_data[key]
        
        return merged
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.dataset_size == 0


# ============================================================================
# GPU-Accelerated Dataloader
# ============================================================================

class GPUDataLoader:
    """
    GPU-accelerated wrapper for existing dataloaders.

    Provides two optimization modes:
    1. Full GPU mode: Pre-load entire dataset to GPU memory (fastest, for small datasets)
    2. Prefetch mode: Async batch preparation with double buffering (memory-efficient, for large datasets)

    Automatically selects the optimal mode based on dataset size and available GPU memory.
    """

    def __init__(
        self,
        base_dataloader: BaseDataloader,
        mode: str = 'auto',
        gpu_memory_threshold: float = 0.3,
        prefetch_buffer_size: int = 2
    ):
        """
        Args:
            base_dataloader: Existing dataloader instance to wrap
            mode: 'auto', 'gpu', 'prefetch', or 'cpu'
                  - 'auto': Automatically select based on dataset size
                  - 'gpu': Force full GPU loading
                  - 'prefetch': Force prefetching mode
                  - 'cpu': Disable GPU optimization (fallback)
            gpu_memory_threshold: Fraction of GPU memory to use for full loading (default 0.3)
            prefetch_buffer_size: Number of batches to prefetch (default 2)
        """
        self.base_dataloader = base_dataloader
        self.logger = get_logger(__name__)
        self.prefetch_buffer_size = prefetch_buffer_size

        # Try to import JAX
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            self.jax_available = True
        except ImportError:
            self.logger.warning("[GPUDataLoader] JAX not available, falling back to CPU mode")
            self.jax_available = False
            mode = 'cpu'

        # Determine operating mode
        if mode == 'auto' and self.jax_available:
            self.mode = self._select_mode(gpu_memory_threshold)
        else:
            self.mode = mode if self.jax_available else 'cpu'

        self.logger.info(f"[GPUDataLoader] Operating mode: {self.mode}")

        # Pre-load data to GPU if in full GPU mode
        self.gpu_stacked_data = None
        if self.mode == 'gpu':
            self._preload_to_gpu()

    def _estimate_dataset_memory(self) -> int:
        """Estimate GPU memory required for the dataset in bytes."""
        total_bytes = 0
        for key, value in self.base_dataloader.stacked_data.items():
            if isinstance(value, np.ndarray):
                total_bytes += value.nbytes
        return total_bytes

    def _get_available_gpu_memory(self) -> Optional[int]:
        """Get available GPU memory in bytes."""
        if not self.jax_available:
            return None

        try:
            # Get default device
            device = self.jax.devices()[0]

            # Try to get memory stats (may not be available on all backends)
            if hasattr(device, 'memory_stats'):
                stats = device.memory_stats()
                if stats and 'bytes_limit' in stats:
                    limit = stats['bytes_limit']
                    used = stats.get('bytes_in_use', 0)
                    return limit - used

            # Fallback: return None if memory stats unavailable
            return None

        except Exception as e:
            self.logger.warning(f"[GPUDataLoader] Could not get GPU memory stats: {e}")
            return None

    def _select_mode(self, threshold: float = 0.3) -> str:
        """
        Automatically select the best mode based on dataset size and GPU memory.

        Args:
            threshold: Fraction of GPU memory to use for full loading

        Returns:
            'gpu', 'prefetch', or 'cpu'
        """
        dataset_size = self._estimate_dataset_memory()
        gpu_memory = self._get_available_gpu_memory()

        self.logger.info(f"[GPUDataLoader] Dataset size: {dataset_size / 1e9:.2f} GB")

        if gpu_memory is None:
            # Can't determine GPU memory, use prefetch as safe default
            self.logger.info("[GPUDataLoader] GPU memory unknown, using prefetch mode")
            return 'prefetch'

        self.logger.info(f"[GPUDataLoader] Available GPU memory: {gpu_memory / 1e9:.2f} GB")

        # If dataset fits in threshold% of GPU memory, use full GPU mode
        if dataset_size < gpu_memory * threshold:
            self.logger.info(f"[GPUDataLoader] Dataset fits in GPU memory (threshold={threshold:.0%})")
            return 'gpu'
        else:
            self.logger.info("[GPUDataLoader] Dataset too large for GPU, using prefetch mode")
            return 'prefetch'

    def _preload_to_gpu(self):
        """Pre-load entire dataset to GPU memory."""
        self.logger.info("[GPUDataLoader] Pre-loading dataset to GPU...")
        start_time = time.time()

        self.gpu_stacked_data = {}
        for key, value in self.base_dataloader.stacked_data.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                # Convert numeric arrays to JAX array and place on GPU
                self.gpu_stacked_data[key] = self.jax.device_put(value)
                self.logger.debug(f"[GPUDataLoader] Transferred {key} to GPU: {value.shape}, {value.dtype}")
            else:
                # Keep non-numeric data (like strings) on CPU
                self.gpu_stacked_data[key] = value
                self.logger.debug(f"[GPUDataLoader] Kept {key} on CPU: {type(value)}, {getattr(value, 'dtype', 'N/A')}")

        elapsed = time.time() - start_time
        self.logger.info(f"[GPUDataLoader] Pre-loading completed in {elapsed:.2f} seconds")

    def _create_batch_gpu(self, indices: np.ndarray, aux_key: Optional[str] = None) -> Dict:
        """Create batch using GPU-side indexing."""
        # Convert indices to JAX array
        jax_indices = self.jax.device_put(indices)

        # GPU-side indexing
        batch = {}
        for key in self.gpu_stacked_data:
            value = self.gpu_stacked_data[key]
            if hasattr(value, '__getitem__'):  # JAX array
                batch[key] = value[jax_indices]
            else:
                # Non-indexable data (shouldn't happen normally)
                batch[key] = value

        # Handle aux_key concatenation
        if aux_key and aux_key in batch:
            batch['observations'] = self.jnp.concatenate(
                [batch['observations'], batch[aux_key]], axis=-1
            )
            del batch[aux_key]

        return batch

    def get_all_batch(
        self,
        batch_size: int,
        aux_key: Optional[str] = None,
        shuffle: bool = True,
        remainder_pad: bool = True,
        pool_key: Optional[str] = None
    ):
        """
        GPU-accelerated batch iteration.

        Args:
            batch_size: Size of each batch
            aux_key: Optional key to concatenate with observations
            shuffle: Whether to shuffle data before batching
            remainder_pad: Whether to pad the last batch
            pool_key: Optional key for pool-based iteration (PoolDataLoader only)

        Yields:
            Dictionary containing batch data (on GPU if mode='gpu')
        """
        if self.mode == 'cpu':
            # Fallback to base dataloader
            yield from self.base_dataloader.get_all_batch(
                batch_size=batch_size,
                aux_key=aux_key,
                shuffle=shuffle,
                remainder_pad=remainder_pad,
                pool_key=pool_key if hasattr(self.base_dataloader, 'get_all_batch') and 'pool_key' in self.base_dataloader.get_all_batch.__code__.co_varnames else None
            )
            return

        elif self.mode == 'gpu':
            # Full GPU mode - use GPU-side indexing
            yield from self._get_all_batch_gpu(
                batch_size=batch_size,
                aux_key=aux_key,
                shuffle=shuffle,
                remainder_pad=remainder_pad,
                pool_key=pool_key
            )

        elif self.mode == 'prefetch':
            # Prefetch mode - async batch preparation
            yield from self._get_all_batch_prefetch(
                batch_size=batch_size,
                aux_key=aux_key,
                shuffle=shuffle,
                remainder_pad=remainder_pad,
                pool_key=pool_key
            )

    def _get_all_batch_gpu(
        self,
        batch_size: int,
        aux_key: Optional[str] = None,
        shuffle: bool = True,
        remainder_pad: bool = True,
        pool_key: Optional[str] = None
    ):
        """Full GPU mode: All operations on GPU."""
        dataset_size = self.base_dataloader.dataset_size

        # Handle pool-based iteration
        if pool_key is not None:
            # Fall back to base dataloader for pool iteration (complex logic)
            self.logger.warning("[GPUDataLoader] Pool iteration not optimized in GPU mode, using base dataloader")
            for batch in self.base_dataloader.get_all_batch(
                batch_size=batch_size,
                aux_key=aux_key,
                shuffle=shuffle,
                remainder_pad=remainder_pad,
                pool_key=pool_key
            ):
                # Transfer batch to GPU (skip non-numeric arrays)
                gpu_batch = {}
                for k, v in batch.items():
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                        gpu_batch[k] = self.jax.device_put(v)
                    else:
                        gpu_batch[k] = v
                yield gpu_batch
            return

        # Generate indices (on CPU, then transfer)
        indices = np.arange(dataset_size)
        if shuffle:
            np.random.shuffle(indices)

        # Yield full batches
        for i in range(0, dataset_size - dataset_size % batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self._create_batch_gpu(batch_indices, aux_key)

        # Remainder batch
        remainder = dataset_size % batch_size
        if remainder > 0:
            leftover_indices = indices[-remainder:]
            if remainder_pad:
                pad_indices = np.random.choice(leftover_indices, batch_size - remainder, replace=True)
                batch_indices = np.concatenate([leftover_indices, pad_indices])
            else:
                batch_indices = leftover_indices
            yield self._create_batch_gpu(batch_indices, aux_key)

    def _get_all_batch_prefetch(
        self,
        batch_size: int,
        aux_key: Optional[str] = None,
        shuffle: bool = True,
        remainder_pad: bool = True,
        pool_key: Optional[str] = None
    ):
        """Prefetch mode: Async batch preparation with double buffering."""
        # Create prefetch queue
        prefetch_queue = Queue(maxsize=self.prefetch_buffer_size)
        stop_event = threading.Event()
        error_container = {'error': None}

        def prefetch_worker():
            """Background thread to prepare and transfer batches to GPU."""
            try:
                # Get batches from base dataloader
                base_generator = self.base_dataloader.get_all_batch(
                    batch_size=batch_size,
                    aux_key=aux_key,
                    shuffle=shuffle,
                    remainder_pad=remainder_pad,
                    pool_key=pool_key if hasattr(self.base_dataloader.get_all_batch, '__code__') and 'pool_key' in self.base_dataloader.get_all_batch.__code__.co_varnames else None
                )

                for cpu_batch in base_generator:
                    if stop_event.is_set():
                        break

                    # Transfer batch to GPU (skip non-numeric arrays)
                    gpu_batch = {}
                    for key, value in cpu_batch.items():
                        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                            # Only transfer numeric arrays to GPU
                            gpu_batch[key] = self.jax.device_put(value)
                        else:
                            # Keep non-numeric data (like strings) on CPU
                            gpu_batch[key] = value

                    # Put in queue (blocks if queue is full)
                    prefetch_queue.put(gpu_batch)

                # Signal completion
                prefetch_queue.put(None)

            except Exception as e:
                error_container['error'] = e
                prefetch_queue.put(None)

        # Start prefetch thread
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        try:
            # Yield batches from queue
            while True:
                try:
                    batch = prefetch_queue.get(timeout=30.0)

                    if batch is None:
                        # End of iteration
                        break

                    yield batch

                except Empty:
                    # Check if worker thread encountered an error
                    if error_container['error']:
                        raise error_container['error']
                    # Otherwise, continue waiting
                    self.logger.warning("[GPUDataLoader] Prefetch queue timeout, continuing...")

        finally:
            # Clean up
            stop_event.set()
            prefetch_thread.join(timeout=1.0)

            if error_container['error']:
                raise error_container['error']

    def get_rand_batch(self, batch_size: Optional[int] = None, aux_key: Optional[str] = None) -> Dict:
        """
        Get a random batch. In GPU mode, returns GPU arrays.

        Args:
            batch_size: Number of samples (None or -1 for entire dataset)
            aux_key: Optional key to concatenate with observations

        Returns:
            Dictionary containing batch data
        """
        if self.mode == 'gpu':
            # Use GPU-side random sampling
            if batch_size is None or batch_size == -1:
                indices = np.arange(self.base_dataloader.dataset_size)
            else:
                indices = np.random.choice(
                    self.base_dataloader.dataset_size,
                    batch_size,
                    replace=False
                )
            return self._create_batch_gpu(indices, aux_key)
        else:
            # CPU mode or prefetch: use base dataloader
            batch = self.base_dataloader.get_rand_batch(batch_size=batch_size, aux_key=aux_key)

            if self.mode == 'prefetch' and self.jax_available:
                # Transfer to GPU (skip non-numeric arrays)
                gpu_batch = {}
                for k, v in batch.items():
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                        gpu_batch[k] = self.jax.device_put(v)
                    else:
                        gpu_batch[k] = v
                return gpu_batch
            else:
                return batch

    @property
    def stacked_data(self):
        """Provide access to underlying data for compatibility."""
        if self.mode == 'gpu':
            return self.gpu_stacked_data
        else:
            return self.base_dataloader.stacked_data

    @property
    def dataset_size(self):
        """Provide access to dataset size."""
        return self.base_dataloader.dataset_size


class DataloaderMixer:
    """
    Mixes multiple dataloaders with configurable ratios.
    Useful for multi-task or multi-domain training.
    
    Note: When mixing datasets with different keys:
    - The mixer will properly handle keys that exist in some but not all datasets
    - However, when using aux_key parameter, ensure it exists in all dataloaders
      to avoid dimension mismatch errors during observation concatenation
    """
    
    def __init__(
        self,
        dataloaders: List[BaseDataloader],
        mixing_ratios: Optional[List[float]] = None,
        reference_idx: int = 0
    ):
        """
        Args:
            dataloaders: List of dataloader instances
            mixing_ratios: Weights for each dataloader (uniform if None)
            reference_idx: Index of reference dataloader for iteration length
        """
        if not dataloaders:
            raise ValueError("[DataloaderMixer] No dataloaders provided")
        
        self.dataloaders = dataloaders
        self.weights = mixing_ratios or [1.0] * len(dataloaders)
        self.reference_idx = reference_idx
        self.logger = get_logger(__name__)
        
        if len(self.weights) != len(dataloaders):
            raise ValueError("[DataloaderMixer] Length of mixing_ratios must match dataloaders")
        
        self.logger.info(f"[DataloaderMixer] Initialized with {len(dataloaders)} dataloaders")
        self.logger.info(f"[DataloaderMixer] Mixing weights: {self.weights}")
    
    def _allocate_batch_sizes(self, total_batch_size: int) -> List[int]:
        """Allocate batch size to each dataloader based on weights."""
        total_weight = sum(self.weights)
        allocations = [int(total_batch_size * w / total_weight) for w in self.weights]
        
        # Distribute remainder
        remainder = total_batch_size - sum(allocations)
        for i in range(remainder):
            allocations[i % len(allocations)] += 1
        
        return allocations
    
    def get_rand_batch(
        self, 
        batch_size: int, 
        aux_key: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Get a mixed random batch from all dataloaders.
        
        Note: If aux_key is specified, it should exist in all dataloaders
        to ensure consistent observation dimensions after concatenation.
        """
        allocations = self._allocate_batch_sizes(batch_size)
        
        # Check if aux_key exists in all dataloaders
        if aux_key:
            loaders_with_key = sum(1 for loader in self.dataloaders 
                                   if aux_key in loader.stacked_data)
            if 0 < loaders_with_key < len(self.dataloaders):
                self.logger.warning(
                    f"[DataloaderMixer] aux_key '{aux_key}' exists in only "
                    f"{loaders_with_key}/{len(self.dataloaders)} dataloaders. "
                    f"This may cause dimension mismatch errors."
                )
        
        batches = []
        for alloc, loader in zip(allocations, self.dataloaders):
            if alloc > 0:
                batch = loader.get_rand_batch(batch_size=alloc, aux_key=aux_key)
                batches.append(batch)
        
        # Merge batches
        if not batches:
            return {}
        
        # Get all unique keys from all batches
        all_keys = set()
        for batch in batches:
            all_keys.update(batch.keys())
        
        merged = {}
        for key in all_keys:
            # Collect arrays for this key from batches that have it
            arrays = [b[key] for b in batches if key in b]
            if arrays:
                merged[key] = np.concatenate(arrays, axis=0)
        
        return merged
    
    def get_all_batch(
        self,
        batch_size: int,
        aux_key: Optional[str] = None,
        shuffle: bool = True,
        remainder_pad: bool = True,
        pool_key: Optional[str] = None
    ):
        """
        Yield mixed batches, iterating based on reference dataloader.
        Supports pool-based iteration when pool_key is provided.
        
        Args:
            batch_size: Size of each batch
            aux_key: Optional key to concatenate with observations
            shuffle: Whether to shuffle within pools
            remainder_pad: Whether to pad incomplete batches
            pool_key: Optional key to group data by pools (for PoolDataLoader compatibility)
        
        Yields:
            Dictionary containing mixed batch data
        """
        allocations = self._allocate_batch_sizes(batch_size)
        
        # Check if pool_key is supported by dataloaders
        if pool_key:
            loaders_with_pool = []
            for i, loader in enumerate(self.dataloaders):
                if hasattr(loader, 'get_all_batch'):
                    # Check if this loader's get_all_batch accepts pool_key
                    import inspect
                    sig = inspect.signature(loader.get_all_batch)
                    if 'pool_key' in sig.parameters:
                        loaders_with_pool.append(i)
            
            if loaders_with_pool and len(loaders_with_pool) < len(self.dataloaders):
                self.logger.info(
                    f"[DataloaderMixer] pool_key '{pool_key}' supported by "
                    f"{len(loaders_with_pool)}/{len(self.dataloaders)} dataloaders. "
                    f"Non-supporting loaders will use standard iteration."
                )
        
        # Create generators with pool_key support where available
        generators = []
        for alloc, loader in zip(allocations, self.dataloaders):
            # Pass pool_key if the loader supports it and has the pool_key in data
            use_pool_key = False
            if pool_key and hasattr(loader, 'get_all_batch'):
                import inspect
                sig = inspect.signature(loader.get_all_batch)
                if 'pool_key' in sig.parameters and pool_key in loader.stacked_data:
                    use_pool_key = True
            
            if use_pool_key:
                gen = loader.get_all_batch(
                    batch_size=alloc,
                    aux_key=aux_key,
                    shuffle=shuffle,
                    remainder_pad=remainder_pad,
                    pool_key=pool_key
                )
            else:
                gen = loader.get_all_batch(
                    batch_size=alloc,
                    aux_key=aux_key,
                    shuffle=shuffle,
                    remainder_pad=remainder_pad
                )
            generators.append((loader, gen, alloc))
        
        # Iterate based on reference dataloader
        ref_loader, ref_gen, ref_alloc = generators[self.reference_idx]
        
        for ref_batch in ref_gen:
            batches = []
            
            # Add reference batch
            if self.reference_idx == 0:
                batches.append(ref_batch)
                start_idx = 1
            else:
                start_idx = 0
            
            # Get batches from other loaders
            for idx in range(start_idx, len(generators)):
                if idx == self.reference_idx:
                    batches.append(ref_batch)
                    continue
                
                loader, gen, alloc = generators[idx]
                try:
                    batch = next(gen)
                except StopIteration:
                    # If exhausted, get random batch
                    batch = loader.get_rand_batch(batch_size=alloc, aux_key=aux_key)
                
                batches.append(batch)
            
            # Merge batches
            # Get all unique keys from all batches
            all_keys = set()
            for batch in batches:
                all_keys.update(batch.keys())
            
            merged = {}
            for key in all_keys:
                # Collect arrays for this key from batches that have it
                arrays = [b[key] for b in batches if key in b]
                if arrays:
                    merged[key] = np.concatenate(arrays, axis=0)
            
            yield merged


# ============================================================================
# Hook Functions for Data Preprocessing
# ============================================================================

def split_trajectories(terminals: np.ndarray) -> List[Tuple[int, int]]:
    """
    Split data into trajectory boundaries based on terminal flags.
    
    Args:
        terminals: Array of terminal flags
        
    Returns:
        List of (start, end) index tuples for each trajectory
    """
    bounds = []
    start = 0
    
    for idx, terminal in enumerate(terminals):
        if terminal:
            bounds.append((start, idx + 1))
            start = idx + 1
    
    if start < len(terminals):
        bounds.append((start, len(terminals)))
    
    return bounds


def libero_obs_hook(dataset: Dict[str, np.ndarray], obs_dim: int = 130) -> Dict[str, np.ndarray]:
    """
    Pad or truncate observations to fixed dimension (for LIBERO environment).
    
    Args:
        dataset: Dataset dictionary
        obs_dim: Target observation dimension
        
    Returns:
        Modified dataset
    """
    obs = dataset['observations']
    padded = []
    
    for o in obs:
        arr = np.array(o)
        if arr.shape[-1] < obs_dim:
            # Pad with zeros
            arr = np.pad(arr, (0, obs_dim - arr.shape[-1]), mode='constant')
        else:
            # Truncate
            arr = arr[:obs_dim]
        padded.append(arr)
    
    dataset['observations'] = np.stack(padded, axis=0)
    logger = get_logger(__name__)
    logger.info(f"[libero_obs_hook] Processed observations → {dataset['observations'].shape}")
    
    return dataset


def history_state_obs_hook(
    dataset: Dict[str, np.ndarray],
    N: int = 10,
    stack_depth: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Create history-augmented observations by concatenating evenly sampled historical states.

    Args:
        dataset: Dataset dictionary
        N: Number of steps to look back
        stack_depth: Number of historical slices to concatenate (>=1)

    Returns:
        Modified dataset with concatenated observations
    """
    if stack_depth < 1:
        raise ValueError("stack_depth must be >= 1")

    obs = dataset['observations']
    terms = dataset['terminals']
    bounds = split_trajectories(terms)

    new_obs: List[np.ndarray] = []
    new_terms: List[np.ndarray] = []
    for start, end in bounds:
        for t in range(start, end):
            hist_idx = max(start, t - N)
            window = obs[hist_idx : t + 1]
            window_len = window.shape[0]
            if window_len == 0:
                continue

            indices = np.linspace(0, window_len - 1, num=stack_depth)
            indices = np.round(indices).astype(int)
            indices = np.clip(indices, 0, window_len - 1)

            slices = [window[idx] for idx in indices]
            new_obs.append(np.concatenate(slices, axis=-1))
            new_terms.append(terms[t])

    if new_obs:
        dataset['observations'] = np.stack(new_obs, axis=0)
        dataset['terminals'] = np.asarray(new_terms, dtype=terms.dtype)
    else:
        dataset['observations'] = obs[:0]
        dataset['terminals'] = terms[:0]

    logger = get_logger(__name__)
    logger.info(
        f"[history_state_obs_hook] Processed observations → {dataset['observations'].shape} (stack_depth={stack_depth})"
    )

    return dataset


def few_shot_hook(
    dataset: Dict[str, np.ndarray], 
    shot: int = 1, 
    frac: int = 1
) -> Dict[str, np.ndarray]:
    """
    Sample a few trajectories and subsample timesteps within them.
    
    Args:
        dataset: Dataset dictionary
        shot: Number of trajectories to keep
        frac: Keep every frac-th timestep
        
    Returns:
        Subsampled dataset
    """
    bounds = split_trajectories(dataset['terminals'])
    if not bounds:
        return {k: v[:0] for k, v in dataset.items()}
    
    # Select first 'shot' trajectories
    selected_bounds = bounds[:shot]
    
    new_data = {k: [] for k in dataset.keys()}
    
    for start, end in selected_bounds:
        # Sample timesteps
        indices = list(range(start, end, frac))
        # Always include last timestep
        if end - 1 not in indices and end - 1 >= start:
            indices.append(end - 1)
        
        for key in dataset:
            new_data[key].append(dataset[key][indices])
    
    # Concatenate
    for key in new_data:
        if new_data[key]:
            new_data[key] = np.concatenate(new_data[key], axis=0)
        else:
            new_data[key] = dataset[key][:0]
    
    logger = get_logger(__name__)
    logger.info(f"[few_shot_hook] Kept {len(selected_bounds)}/{len(bounds)} trajectories")
    
    return new_data


# Alias for backward compatibility
few_frac_shot_hook = few_shot_hook
libero_obs_obs_hook = libero_obs_hook


def history_state_three_obs_hook(dataset: Dict[str, np.ndarray], N: int = 10) -> Dict[str, np.ndarray]:
    """
    Backwards-compatible wrapper for 3-frame history stacking.
    """
    return history_state_obs_hook(dataset, N=N, stack_depth=3)


def dropthe_traj_hook(dataset: Dict[str, np.ndarray], per_drop: int = 2) -> Dict[str, np.ndarray]:
    """
    Drop every per_drop-th trajectory from the dataset.

    Args:
        dataset: Dataset dictionary
        per_drop: Keep trajectories whose index mod per_drop != 0

    Returns:
        Dataset with dropped trajectories removed
    """
    if per_drop <= 0:
        raise ValueError("per_drop must be >= 1")

    bounds = split_trajectories(dataset['terminals'])
    new_data = {k: [] for k in dataset.keys()}

    # Keep trajectories where index % per_drop != 0
    kept_count = 0
    for tid, (start, end) in enumerate(bounds):
        if tid % per_drop != 0:
            for key in dataset:
                new_data[key].append(dataset[key][start:end])
            kept_count += 1

    # Concatenate
    for key in new_data:
        if new_data[key]:
            new_data[key] = np.concatenate(new_data[key], axis=0)
        else:
            new_data[key] = np.empty((0,) + dataset[key].shape[1:], dtype=dataset[key].dtype)

    logger = get_logger(__name__)
    logger.info(f"[dropthe_traj_hook] Kept {kept_count}/{len(bounds)} trajectories")

    return new_data


def action_chunk_hook(
    dataset: Dict[str, np.ndarray],
    chunk_size: int = 1,
    padding_mode: str = 'repeat_last'
) -> Dict[str, np.ndarray]:
    """
    Create action chunks for each timestep, where each observation is paired with
    a sequence of future actions [action_t, action_t+1, ..., action_t+chunk_size-1].

    Args:
        dataset: Dataset dictionary containing 'actions', 'observations', 'terminals', etc.
        chunk_size: Number of actions to predict per observation (1 = no chunking)
        padding_mode: How to pad when reaching trajectory end:
            - 'repeat_last': Repeat the last action in the trajectory
            - 'zero': Pad with zeros

    Returns:
        Modified dataset where actions are chunked and flattened
        New action shape: (N_samples, action_dim * chunk_size)

    Note:
        - Only creates chunks within trajectory boundaries (respects terminals)
        - At trajectory end, pads incomplete chunks according to padding_mode
        - Terminal flags remain at their original positions
    """
    if chunk_size <= 1:
        # No chunking needed
        return dataset

    if padding_mode not in ['repeat_last', 'zero']:
        raise ValueError(f"Invalid padding_mode '{padding_mode}'. Must be 'repeat_last' or 'zero'")

    logger = get_logger(__name__)
    logger.info(f"[action_chunk_hook] Creating action chunks with size={chunk_size}, padding_mode={padding_mode}")

    actions = dataset['actions']
    terminals = dataset['terminals']
    action_dim = actions.shape[-1]

    # Get trajectory boundaries
    bounds = split_trajectories(terminals)

    new_actions = []

    # Process each trajectory
    for start, end in bounds:
        traj_actions = actions[start:end]
        traj_len = end - start

        # Create chunks for each timestep in this trajectory
        for t in range(traj_len):
            # Determine how many actions we can get from this timestep
            remaining = traj_len - t

            if remaining >= chunk_size:
                # Full chunk available
                chunk = traj_actions[t:t+chunk_size]
            else:
                # Partial chunk - need padding
                chunk = traj_actions[t:]
                padding_needed = chunk_size - remaining

                if padding_mode == 'repeat_last':
                    # Repeat the last action
                    last_action = traj_actions[-1:].repeat(padding_needed, axis=0)
                    chunk = np.concatenate([chunk, last_action], axis=0)
                else:  # padding_mode == 'zero'
                    # Pad with zeros
                    zero_padding = np.zeros((padding_needed, action_dim), dtype=actions.dtype)
                    chunk = np.concatenate([chunk, zero_padding], axis=0)

            # Flatten the chunk: (chunk_size, action_dim) -> (chunk_size * action_dim,)
            flattened_chunk = chunk.reshape(-1)
            new_actions.append(flattened_chunk)

    # Stack all chunks
    new_actions = np.stack(new_actions, axis=0)

    logger.info(f"[action_chunk_hook] Original actions shape: {actions.shape}")
    logger.info(f"[action_chunk_hook] Chunked actions shape: {new_actions.shape}")
    logger.info(f"[action_chunk_hook] Action dimension: {action_dim} -> {action_dim * chunk_size}")

    # Update dataset
    dataset['actions'] = new_actions

    return dataset


# ============================================================================
# LeRobot Dataloader
# ============================================================================

class LeRobotDataLoader(PoolDataLoader):
    """
    Dataloader for LeRobot HDF5 format with embedded vision features.

    This loader handles HDF5 files from ./data/libero_embed/ containing:
    - Pre-computed vision embeddings (e.g., DINOv3)
    - Proprioceptive state observations
    - Actions and other metadata

    The dataloader concatenates all observation modalities into a single
    observation vector compatible with existing training pipelines.
    """

    # Canonical observation ordering (single source of truth)
    # - Camera embedding keys used for training/eval inputs
    # - Fallback raw camera keys (dataset side only)
    # - Proprioceptive features in the exact concatenation order
    # - Extra optional keys (legacy/state aliases)
    OBS_EMBED_CAMERA_KEYS: tuple[str, ...] = (
        'agentview_rgb_dinov3',
        'eye_in_hand_rgb_dinov3',
    )
    OBS_FALLBACK_CAMERA_KEYS: tuple[str, ...] = (
        'agentview_rgb',
        'eye_in_hand_rgb',
    )
    OBS_PROPRIO_KEYS: tuple[str, ...] = (
        'joint_states',
        'ee_states',
        'gripper_states',
        'robot_states',
    )
    OBS_EXTRA_KEYS: tuple[str, ...] = (
        'state',
        'proprio',
    )

    @classmethod
    def get_camera_embed_keys(cls) -> tuple[str, ...]:
        """Return the canonical camera embedding keys in concatenation order."""
        return cls.OBS_EMBED_CAMERA_KEYS

    @classmethod
    def get_proprio_keys(cls) -> tuple[str, ...]:
        """Return the canonical proprioceptive keys in concatenation order."""
        return cls.OBS_PROPRIO_KEYS

    @classmethod
    def get_canonical_observation_order(cls) -> tuple[str, ...]:
        """Return the full ordered list of observation keys for concatenation."""
        return (
            *cls.OBS_EMBED_CAMERA_KEYS,
            *cls.OBS_FALLBACK_CAMERA_KEYS,
            *cls.OBS_PROPRIO_KEYS,
            *cls.OBS_EXTRA_KEYS,
        )

    def __init__(
        self,
        data_paths: List[str],
        pre_process_hooks_kwargs: Optional[List[Tuple[Callable, Dict]]] = None,
        post_process_hooks_kwargs: Optional[List[Tuple[Callable, Dict]]] = None,
        semantics_path: Optional[str] = None,
        *,
        obs_modality_keys: Optional[Tuple[str, ...]] = None,
        replace_proprio_with_state: bool = False,
        oracle_key_name: str = "state",
    ):
        """
        Args:
            data_paths: List of paths to HDF5 files
            pre_process_hooks_kwargs: Optional preprocessing hooks
            post_process_hooks_kwargs: Optional postprocessing hooks
            semantics_path: Optional semantic embeddings (for compatibility)
        """
        # Import h5py here to avoid dependency issues if not using LeRobot
        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError(
                "h5py is required for LeRobotDataLoader. "
                "Please install it with: pip install h5py"
            )

        # Selected observation modalities (None means use canonical order)
        self._obs_modality_keys: Optional[Tuple[str, ...]] = tuple(obs_modality_keys) if obs_modality_keys is not None else None
        self._replace_proprio_with_state: bool = bool(replace_proprio_with_state)
        self._oracle_key_name: str = str(oracle_key_name)

        # Call parent constructor
        super().__init__(
            data_paths=data_paths,
            pre_process_hooks_kwargs=pre_process_hooks_kwargs,
            post_process_hooks_kwargs=post_process_hooks_kwargs,
            semantics_path=semantics_path
        )

    def load_data(self):
        """Load and process HDF5 files into the expected format."""
        data_list = []

        # Process each HDF5 file
        for path in self.data_paths:
            self.logger.info(f"[LeRobotDataLoader] Loading HDF5 data from {path}")

            # Extract skill/task name from filename
            filename = os.path.basename(path)
            # Remove _demo.hdf5 suffix and use as skill identifier
            skill_name = filename.replace('_demo.hdf5', '').replace('.hdf5', '')

            try:
                with self.h5py.File(path, 'r') as f:
                    # Check if data structure exists
                    if 'data' not in f:
                        self.logger.warning(f"[LeRobotDataLoader] No 'data' group in {path}, skipping")
                        continue

                    data_group = f['data']

                    # Collect all demonstrations
                    all_obs = []
                    all_actions = []
                    all_terminals = []
                    all_skills = []

                    # Sort demo keys to ensure consistent ordering
                    demo_keys = sorted(list(data_group.keys()))

                    oracle_keys_seen: set[str] = set()
                    for demo_key in demo_keys:
                        demo = data_group[demo_key]

                        # Get trajectory length
                        if 'actions' in demo:
                            traj_len = len(demo['actions'][:])
                        else:
                            self.logger.warning(f"[LeRobotDataLoader] No actions in {demo_key}, skipping")
                            continue

                        # Extract observations
                        obs_list = []
                        included_key_dims: Dict[str, int] = {}

                        if 'obs' in demo:
                            obs_group = demo['obs']

                            # Priority order for observation keys (single source of truth)
                            if self._obs_modality_keys is None:
                                obs_keys_priority = list(self.get_canonical_observation_order())
                            else:
                                obs_keys_priority = list(self._obs_modality_keys)

                            if self._replace_proprio_with_state:
                                obs_keys_priority = [k for k in obs_keys_priority if k not in self.OBS_PROPRIO_KEYS]

                            # Collect available observation modalities
                            for key in obs_keys_priority:
                                if key in obs_group:
                                    obs_data = np.array(obs_group[key][:])
                                    # Ensure 2D shape (T, feature_dim)
                                    if obs_data.ndim == 1:
                                        obs_data = obs_data.reshape(-1, 1)
                                    elif obs_data.ndim > 2:
                                        # Flatten if more than 2D (e.g., images)
                                        obs_data = obs_data.reshape(obs_data.shape[0], -1)
                                    obs_list.append(obs_data)
                                    # Track per-modality feature dims for explanation later
                                    feat_dim = int(obs_data.shape[1])
                                    prev = included_key_dims.get(key)
                                    if prev is not None and prev != feat_dim:
                                        self.logger.warning(
                                            f"[LeRobotDataLoader] Inconsistent feature dim for key '{key}': "
                                            f"seen {prev} vs {feat_dim} (using latest)"
                                        )
                                    included_key_dims[key] = feat_dim
                                    self.logger.debug(f"[LeRobotDataLoader] Added {key}: shape {obs_data.shape}")

                        oracle_key = None
                        if self._replace_proprio_with_state:
                            candidate_keys = [self._oracle_key_name]
                            for fallback_key in ("states", "state", "oracle_state", "oracle_proprio", "proprio_state", "oracle"):
                                if fallback_key not in candidate_keys:
                                    candidate_keys.append(fallback_key)

                            oracle_data = None
                            for cand in candidate_keys:
                                if cand in demo:
                                    oracle_data = np.array(demo[cand][:])
                                    oracle_key = cand
                                    break

                            if oracle_data is None and 'obs' in demo:
                                for cand in candidate_keys:
                                    if cand in demo['obs']:
                                        oracle_data = np.array(demo['obs'][cand][:])
                                        oracle_key = cand
                                        break

                            if oracle_data is not None:
                                if oracle_data.ndim == 1:
                                    oracle_data = oracle_data.reshape(-1, 1)
                                elif oracle_data.ndim > 2:
                                    oracle_data = oracle_data.reshape(oracle_data.shape[0], -1)
                                obs_list.append(oracle_data.astype(np.float32))
                                included_key_dims[oracle_key] = int(oracle_data.shape[1])
                                oracle_keys_seen.add(oracle_key)
                            else:
                                self.logger.warning(
                                    f"[LeRobotDataLoader] Oracle proprio key not found in demo '{demo_key}'. "
                                    f"Searched {candidate_keys}; available top-level keys={sorted(list(demo.keys()))}; "
                                    "falling back to raw proprio features."
                                )

                        # Concatenate all observation modalities
                        if obs_list:
                            traj_obs = np.concatenate(obs_list, axis=-1).astype(np.float32)
                        else:
                            # Fallback: create zero observations if no obs found
                            self.logger.warning(f"[LeRobotDataLoader] No observations found in {demo_key}")
                            traj_obs = np.zeros((traj_len, 1), dtype=np.float32)

                        # Extract actions
                        traj_actions = np.array(demo['actions'][:]).astype(np.float32)
                        if traj_actions.ndim == 1:
                            traj_actions = traj_actions.reshape(-1, 1)

                        # Create terminals (mark last timestep as terminal)
                        traj_terminals = np.zeros(traj_len, dtype=np.int32)
                        traj_terminals[-1] = 1

                        # Extract skill identifiers from demo data if available
                        if 'skills' in demo:
                            # Read skills from HDF5 (each timestep has its own skill)
                            traj_skills_raw = demo['skills'][:]
                            # Convert bytes to strings if needed
                            if len(traj_skills_raw) > 0 and isinstance(traj_skills_raw[0], bytes):
                                traj_skills = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s)
                                                       for s in traj_skills_raw], dtype=object)
                            else:
                                traj_skills = np.array([str(s) for s in traj_skills_raw], dtype=object)
                        else:
                            # Fallback: use filename-based skill if 'skills' field not present
                            traj_skills = np.full(traj_len, skill_name, dtype=object)

                        # Append to lists
                        all_obs.append(traj_obs)
                        all_actions.append(traj_actions)
                        all_terminals.append(traj_terminals)
                        all_skills.append(traj_skills)

                    # Concatenate all trajectories if any were found
                    if all_obs:
                        file_data = {
                            'observations': np.concatenate(all_obs, axis=0),
                            'actions': np.concatenate(all_actions, axis=0),
                            'terminals': np.concatenate(all_terminals, axis=0),
                            'skills': np.concatenate(all_skills, axis=0)
                        }

                        # Prepare pre-hook breakdown (how obs_dim is composed)
                        raw_breakdown = ", ".join([f"{k}={d}" for k, d in included_key_dims.items()]) if included_key_dims else "<empty>"
                        raw_obs_dim = int(sum(included_key_dims.values())) if included_key_dims else int(file_data['observations'].shape[1])

                        # Apply pre-processing hooks
                        for hook, kwargs in self.pre_process_hooks_kwargs:
                            file_data = hook(file_data, **kwargs)

                        data_list.append(file_data)

                        post_obs_dim = int(file_data['observations'].shape[1])
                        self.logger.info(
                            f"[LeRobotDataLoader] Loaded {len(demo_keys)} demos, total {len(file_data['observations'])} timesteps, "
                            f"obs_dim={post_obs_dim}, action_dim={file_data['actions'].shape[1]}"
                        )
                        # Explain obs_dim composition and any changes due to hooks
                        hook_names = [getattr(h, '__name__', str(h)) for h, _ in self.pre_process_hooks_kwargs]
                        oracle_key_summary = ','.join(sorted(oracle_keys_seen)) if oracle_keys_seen else 'N/A'
                        self.logger.info(
                            "[LeRobotDataLoader] Obs breakdown (pre-hooks): "
                            f"{raw_breakdown} (sum={raw_obs_dim}); "
                            f"replace_proprio_with_state={self._replace_proprio_with_state}, "
                            f"selected_modalities={list(self._obs_modality_keys) if self._obs_modality_keys else 'default'}, "
                            f"oracle_keys={oracle_key_summary}"
                        )
                        if post_obs_dim != raw_obs_dim:
                            self.logger.info(
                                "[LeRobotDataLoader] Obs dim changed by pre_process_hooks: "
                                f"{raw_obs_dim} -> {post_obs_dim} via hooks={hook_names}"
                            )
                    else:
                        self.logger.warning(f"[LeRobotDataLoader] No valid trajectories in {path}")

            except Exception as e:
                self.logger.error(f"[LeRobotDataLoader] Error loading {path}: {str(e)}")
                continue

        # Merge all loaded data
        if data_list:
            # Concatenate data from all files
            self.data_buffer = {}
            for key in data_list[0].keys():
                arrays = [d[key] for d in data_list if key in d]
                if arrays:
                    self.data_buffer[key] = np.concatenate(arrays, axis=0)

            # Apply post-processing hooks
            for hook, kwargs in self.post_process_hooks_kwargs:
                self.data_buffer = hook(self.data_buffer, **kwargs)

            self.logger.info(
                f"[LeRobotDataLoader] Total dataset: "
                f"{len(self.data_buffer['observations'])} timesteps, "
                f"obs_dim={self.data_buffer['observations'].shape[1]}, "
                f"action_dim={self.data_buffer['actions'].shape[1]}"
            )
            # Summarize loader configuration affecting obs_dim
            self.logger.info(
                "[LeRobotDataLoader] Loader config — "
                f"replace_proprio_with_state={self._replace_proprio_with_state}, "
                f"selected_modalities={list(self._obs_modality_keys) if self._obs_modality_keys else 'default'}, "
                f"pre_hooks={[getattr(h, '__name__', str(h)) for h, _ in self.pre_process_hooks_kwargs]}, "
                f"post_hooks={[getattr(h, '__name__', str(h)) for h, _ in self.post_process_hooks_kwargs]}"
            )
        else:
            # No data loaded
            self.data_buffer = {
                'observations': np.empty((0, 1), dtype=np.float32),
                'actions': np.empty((0, 1), dtype=np.float32),
                'terminals': np.empty((0,), dtype=np.int32),
                'skills': np.empty((0,), dtype=object)
            }
            self.logger.warning("[LeRobotDataLoader] No data loaded from any file")
