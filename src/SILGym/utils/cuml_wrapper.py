"""
cuML Wrapper: Automatically use GPU-accelerated cuML if available, fallback to sklearn/umap-learn

This module provides a transparent wrapper for ML algorithms that automatically
uses cuML (GPU) when available and falls back to scikit-learn/umap-learn (CPU) otherwise.

Usage:
    from SILGym.utils.cuml_wrapper import KMeans, TSNE, UMAP, MiniBatchKMeans, HDBSCAN

    # Use exactly like sklearn/umap-learn, but with automatic GPU acceleration
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)
"""

import numpy as np
from typing import Optional, Union
from SILGym.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import cuML (GPU)
try:
    import cuml
    import cupy as cp
    CUML_AVAILABLE = True
    logger.info("cuML detected - GPU acceleration enabled for clustering and manifold learning")
except ImportError:
    CUML_AVAILABLE = False
    logger.info("cuML not available - using CPU-based sklearn/umap-learn")


def _to_cupy_if_available(X):
    """Convert numpy array to cupy if cuML is available"""
    if CUML_AVAILABLE and isinstance(X, np.ndarray):
        return cp.asarray(X)
    return X


def _to_numpy_if_needed(X):
    """Convert cupy array to numpy if needed"""
    if CUML_AVAILABLE and hasattr(X, 'get'):  # Check if it's a cupy array
        return cp.asnumpy(X)
    return X


class KMeans:
    """
    KMeans wrapper that uses cuML (GPU) when available, sklearn (CPU) otherwise.

    API is compatible with sklearn.cluster.KMeans
    """

    def __init__(self, n_clusters=8, random_state=None, n_init=10, max_iter=300, tol=1e-4, verbose=0, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.kwargs = kwargs

        if CUML_AVAILABLE:
            from cuml.cluster import KMeans as cuKMeans
            self._model = cuKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'cuml'
        else:
            from sklearn.cluster import KMeans as skKMeans
            self._model = skKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'sklearn'

    def fit(self, X, y=None, sample_weight=None):
        """Fit KMeans"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            self._model.fit(X_gpu, sample_weight=sample_weight)
        else:
            self._model.fit(X, y=y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """Predict cluster labels"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            labels = self._model.predict(X_gpu)
            return _to_numpy_if_needed(labels)
        else:
            return self._model.predict(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit and predict in one step"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            labels = self._model.fit_predict(X_gpu, sample_weight=sample_weight)
            return _to_numpy_if_needed(labels)
        else:
            return self._model.fit_predict(X, y=y, sample_weight=sample_weight)

    def transform(self, X):
        """Transform X to cluster-distance space"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            distances = self._model.transform(X_gpu)
            return _to_numpy_if_needed(distances)
        else:
            return self._model.transform(X)

    @property
    def cluster_centers_(self):
        """Get cluster centers"""
        centers = self._model.cluster_centers_
        return _to_numpy_if_needed(centers)

    @property
    def labels_(self):
        """Get labels"""
        labels = self._model.labels_
        return _to_numpy_if_needed(labels)

    @property
    def inertia_(self):
        """Get inertia"""
        return float(self._model.inertia_)


class MiniBatchKMeans:
    """
    MiniBatchKMeans wrapper that uses cuML (GPU) when available, sklearn (CPU) otherwise.

    Note: cuML doesn't have MiniBatchKMeans, so we use regular KMeans on GPU which is still fast.
    """

    def __init__(self, n_clusters=8, random_state=None, batch_size=1024, n_init=3, max_iter=100,
                 tol=1e-4, verbose=0, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.kwargs = kwargs

        if CUML_AVAILABLE:
            # cuML KMeans is already fast enough, no need for mini-batch
            from cuml.cluster import KMeans as cuKMeans
            logger.debug("Using cuML KMeans instead of MiniBatchKMeans (GPU is fast enough)")
            self._model = cuKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'cuml'
        else:
            from sklearn.cluster import MiniBatchKMeans as skMiniBatchKMeans
            self._model = skMiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=batch_size,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'sklearn'

    def fit(self, X, y=None, sample_weight=None):
        """Fit MiniBatchKMeans"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            self._model.fit(X_gpu, sample_weight=sample_weight)
        else:
            self._model.fit(X, y=y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """Predict cluster labels"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            labels = self._model.predict(X_gpu)
            return _to_numpy_if_needed(labels)
        else:
            return self._model.predict(X)

    def partial_fit(self, X, y=None, sample_weight=None):
        """Partial fit (only for sklearn backend)"""
        if self._backend == 'cuml':
            # cuML doesn't support partial_fit, do regular fit
            logger.warning("cuML doesn't support partial_fit, using fit instead")
            return self.fit(X, sample_weight=sample_weight)
        else:
            self._model.partial_fit(X, y=y, sample_weight=sample_weight)
            return self

    @property
    def cluster_centers_(self):
        """Get cluster centers"""
        centers = self._model.cluster_centers_
        return _to_numpy_if_needed(centers)

    @property
    def labels_(self):
        """Get labels"""
        labels = self._model.labels_
        return _to_numpy_if_needed(labels)

    @property
    def inertia_(self):
        """Get inertia"""
        return float(self._model.inertia_)


class TSNE:
    """
    t-SNE wrapper that uses cuML (GPU) when available, sklearn (CPU) otherwise.

    API is compatible with sklearn.manifold.TSNE
    """

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000,
                 random_state=None, verbose=0, **kwargs):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        if CUML_AVAILABLE:
            from cuml.manifold import TSNE as cuTSNE
            self._model = cuTSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'cuml'
        else:
            from sklearn.manifold import TSNE as skTSNE
            self._model = skTSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'sklearn'

    def fit(self, X, y=None):
        """Fit t-SNE"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            self._model.fit(X_gpu)
        else:
            self._model.fit(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            embedding = self._model.fit_transform(X_gpu)
            return _to_numpy_if_needed(embedding)
        else:
            return self._model.fit_transform(X, y=y)

    @property
    def embedding_(self):
        """Get embedding"""
        embedding = self._model.embedding_
        return _to_numpy_if_needed(embedding)


class UMAP:
    """
    UMAP wrapper that uses cuML (GPU) when available, umap-learn (CPU) otherwise.

    API is compatible with umap.UMAP
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean',
                 n_epochs=None, learning_rate=1.0, random_state=None, verbose=False, **kwargs):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        if CUML_AVAILABLE:
            from cuml.manifold import UMAP as cuUMAP
            # cuML UMAP has slightly different parameter names
            self._model = cuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                random_state=random_state,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'cuml'
        else:
            from umap import UMAP as umapUMAP
            self._model = umapUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                random_state=random_state,
                verbose=verbose,
                **kwargs
            )
            self._backend = 'umap'

    def fit(self, X, y=None):
        """Fit UMAP"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            self._model.fit(X_gpu)
        else:
            self._model.fit(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            embedding = self._model.fit_transform(X_gpu)
            return _to_numpy_if_needed(embedding)
        else:
            return self._model.fit_transform(X, y=y)

    def transform(self, X):
        """Transform new data"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            embedding = self._model.transform(X_gpu)
            return _to_numpy_if_needed(embedding)
        else:
            return self._model.transform(X)

    @property
    def embedding_(self):
        """Get embedding"""
        embedding = self._model.embedding_
        return _to_numpy_if_needed(embedding)


class HDBSCAN:
    """
    HDBSCAN wrapper that uses cuML (GPU) when available, sklearn (CPU) otherwise.

    API is compatible with sklearn.cluster.HDBSCAN (sklearn 1.3+)
    and cuml.cluster.HDBSCAN
    """

    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
                 cluster_selection_method='eom', metric='euclidean', **kwargs):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.kwargs = kwargs

        if CUML_AVAILABLE:
            from cuml.cluster import HDBSCAN as cuHDBSCAN
            self._model = cuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
                metric=metric,
                **kwargs
            )
            self._backend = 'cuml'
        else:
            try:
                from sklearn.cluster import HDBSCAN as skHDBSCAN
                self._model = skHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    cluster_selection_method=cluster_selection_method,
                    metric=metric,
                    **kwargs
                )
                self._backend = 'sklearn'
            except ImportError:
                # Fallback to hdbscan package if sklearn doesn't have it
                import hdbscan
                logger.info("Using hdbscan package (sklearn.cluster.HDBSCAN not available)")
                self._model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    cluster_selection_method=cluster_selection_method,
                    metric=metric,
                    **kwargs
                )
                self._backend = 'hdbscan'

    def fit(self, X, y=None):
        """Fit HDBSCAN"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            self._model.fit(X_gpu)
        else:
            self._model.fit(X, y=y)
        return self

    def fit_predict(self, X, y=None):
        """Fit and predict in one step"""
        if self._backend == 'cuml':
            X_gpu = _to_cupy_if_available(X)
            labels = self._model.fit_predict(X_gpu)
            return _to_numpy_if_needed(labels)
        else:
            return self._model.fit_predict(X, y=y)

    @property
    def labels_(self):
        """Get cluster labels (-1 for noise points)"""
        labels = self._model.labels_
        return _to_numpy_if_needed(labels)

    @property
    def probabilities_(self):
        """Get cluster membership probabilities"""
        if hasattr(self._model, 'probabilities_'):
            probs = self._model.probabilities_
            return _to_numpy_if_needed(probs)
        return None


# Convenience function to check backend
def get_backend():
    """Return the current backend being used"""
    return 'cuml (GPU)' if CUML_AVAILABLE else 'sklearn/umap (CPU)'


def is_cuml_available():
    """Check if cuML is available"""
    return CUML_AVAILABLE


# Export all classes
__all__ = ['KMeans', 'MiniBatchKMeans', 'TSNE', 'UMAP', 'HDBSCAN', 'get_backend', 'is_cuml_available']
