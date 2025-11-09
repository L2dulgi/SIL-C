"""
ModelAppenderV3: LoRA-based multi-decoder without qax/lorax dependencies.

This implementation provides the same functionality as V1/V2 but uses native JAX/Flax
instead of depending on qax and lorax packages.

Key differences from V2:
- No qax.ImplicitArray or LoraWeightPool
- No lorax.lora decorator
- Manual LoRA weight materialization: W' = W + α(B @ A)
- Custom gradient masking for frozen base weights
- Direct parameter tree manipulation using jax.tree_util

API is identical to ModelAppenderV2.
"""

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import linen as nn
from flax.core import FrozenDict, freeze
import optax
import numpy as np
from dataclasses import dataclass
from functools import wraps
from flax.traverse_util import flatten_dict, unflatten_dict
from SILGym.utils.logger import get_logger


_ORIGINAL_DENSE_CALL = None
_ORIGINAL_CONV_CALL = None


def _dense_lora_delta(inputs, a, b):
    """Compute LoRA delta for dense layers without materializing full weights."""
    # Match flax.linen.Dense semantics: contract over the last axis of inputs.
    delta = jnp.tensordot(inputs, b, axes=[inputs.ndim - 1, 0])
    delta = jnp.tensordot(delta, a, axes=[delta.ndim - 1, 0])
    return delta


def _conv_lora_delta(module, inputs, a, b):
    """Compute LoRA delta for convolution layers via two sequential convolutions."""
    window_strides = getattr(module, 'window_strides', getattr(module, 'strides', None))
    if window_strides is None:
        raise AttributeError('Conv module missing stride information for LoRA computation')

    padding = module.padding
    lhs_dilation = getattr(module, 'lhs_dilation', None)
    rhs_dilation = getattr(module, 'rhs_dilation', None)
    dimension_numbers = module.dimension_numbers
    feature_group_count = getattr(module, 'feature_group_count', 1)
    precision = getattr(module, 'precision', None)

    lhs = jax.lax.conv_general_dilated(
        inputs,
        b,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision,
    )

    ones_stride = (1,) * len(window_strides)
    delta = jax.lax.conv_general_dilated(
        lhs,
        a,
        window_strides=ones_stride,
        padding='VALID',
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=precision,
    )
    return delta


def _patch_lora_layers():
    """Monkey patch flax Dense/Conv to add LoRA deltas when available."""
    global _ORIGINAL_DENSE_CALL, _ORIGINAL_CONV_CALL

    if _ORIGINAL_DENSE_CALL is None:
        _ORIGINAL_DENSE_CALL = nn.Dense.__call__

        @wraps(_ORIGINAL_DENSE_CALL)
        def dense_call(self, inputs, *args, **kwargs):
            outputs = _ORIGINAL_DENSE_CALL(self, inputs, *args, **kwargs)

            if not self.has_variable('params', 'lora_a'):
                return outputs

            a = self.scope.get_variable('params', 'lora_a')
            b = self.scope.get_variable('params', 'lora_b')

            if a is None or b is None:
                return outputs

            alpha = self.scope.get_variable('params', 'lora_alpha') if self.has_variable('params', 'lora_alpha') else 1.0
            delta = _dense_lora_delta(inputs, a, b)
            return outputs + delta * alpha

        nn.Dense.__call__ = dense_call

    if hasattr(nn, 'Conv') and _ORIGINAL_CONV_CALL is None:
        _ORIGINAL_CONV_CALL = nn.Conv.__call__

        @wraps(_ORIGINAL_CONV_CALL)
        def conv_call(self, inputs, *args, **kwargs):
            outputs = _ORIGINAL_CONV_CALL(self, inputs, *args, **kwargs)

            if not self.has_variable('params', 'lora_a'):
                return outputs

            a = self.scope.get_variable('params', 'lora_a')
            b = self.scope.get_variable('params', 'lora_b')

            if a is None or b is None:
                return outputs

            alpha = self.scope.get_variable('params', 'lora_alpha') if self.has_variable('params', 'lora_alpha') else 1.0
            delta = _conv_lora_delta(self, inputs, a, b)
            return outputs + delta * alpha

        nn.Conv.__call__ = conv_call


_patch_lora_layers()


@dataclass
class AppendConfig:
    lora_dim: int = 4
    pool_length: int = 10


def init_lora_matrices(params, lora_spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1.0):
    """
    Initialize LoRA matrices A and B for parameters based on lora_spec.

    Args:
        params: Parameter tree (Flax params)
        lora_spec: Dict matching param structure, values are lora_dim or 'full' or 'freeze'
        rng: JAX random key
        stddev: Standard deviation for initialization
        dtype: Data type for matrices
        alpha: LoRA scaling factor

    Returns:
        lora_a: Dict tree of A matrices
        lora_b: Dict tree of B matrices
        lora_info: Metadata (alpha, spec)
    """
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key
    key_it = iter_keys(rng)

    def get_lora_matrices(path, param, spec_val):
        """Create A, B matrices and scale for a single parameter."""
        if spec_val in ('freeze', 'full'):
            # No LoRA for frozen or fully-tuned params
            return None, None, None

        if len(param.shape) == 1:
            # Vectors: no LoRA
            return None, None, None

        if len(param.shape) == 2:
            # Dense layer: param shape (in_dim, out_dim)
            in_dim, out_dim = param.shape
            rank = spec_val

            # Match ModelAppenderV2 init: A=random, B=zeros
            a = jax.random.normal(next(key_it), (rank, out_dim), dtype=dtype) * stddev
            b = jnp.zeros((in_dim, rank), dtype=dtype)

            scale = alpha / max(rank, 1)

            return a, b, jnp.asarray(scale, dtype=dtype)

        # Convolutional layers
        *window_shape, in_channels, out_channels = param.shape
        rank = spec_val

        # Match ModelAppenderV2 init for conv layers: A=zeros, B=random
        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            rank,
            out_channels
        ), dtype=dtype)
        b = jax.random.normal(next(key_it), (*window_shape, in_channels, rank), dtype=dtype) * stddev

        scale = alpha / max(rank, 1)

        return a, b, jnp.asarray(scale, dtype=dtype)

    # Build LoRA matrices tree
    lora_matrices = jax.tree_util.tree_map_with_path(get_lora_matrices, params, lora_spec)

    # Split into separate a and b trees using tree_transpose
    # lora_matrices is a tree of (a, b) tuples, we need separate trees for a and b
    def extract_index(tree, idx):
        """Extract idx-th element from tuples in tree."""
        return jax.tree.map(lambda x: x[idx] if isinstance(x, tuple) else x, tree, is_leaf=lambda x: isinstance(x, tuple))

    lora_a = extract_index(lora_matrices, 0)
    lora_b = extract_index(lora_matrices, 1)
    lora_alpha = extract_index(lora_matrices, 2)

    lora_info = {'alpha': alpha, 'spec': lora_spec}

    return lora_a, lora_b, lora_alpha, lora_info


def _to_mutable(tree):
    return tree.unfreeze() if isinstance(tree, FrozenDict) else dict(tree)


def _insert_lora_params(params, lora_a, lora_b, lora_alpha):
    """Insert LoRA parameters into params tree under dedicated keys."""
    params_dict = _to_mutable(params)
    flat_params = flatten_dict(params_dict, keep_empty_nodes=True)
    flat_a = flatten_dict(_to_mutable(lora_a), keep_empty_nodes=True)
    flat_b = flatten_dict(_to_mutable(lora_b), keep_empty_nodes=True)
    flat_alpha = flatten_dict(_to_mutable(lora_alpha), keep_empty_nodes=True)

    lora_kernel_paths = []
    new_entries = {}
    for path, a_val in flat_a.items():
        if a_val is None:
            continue
        b_val = flat_b[path]
        alpha_val = flat_alpha.get(path)
        parent_path = path[:-1]
        lora_kernel_paths.append(path)
        new_entries[parent_path + ('lora_a',)] = a_val
        new_entries[parent_path + ('lora_b',)] = b_val
        if alpha_val is None:
            if a_val.ndim == 0:
                rank = 1
            elif a_val.ndim == 1:
                rank = max(a_val.shape[0], 1)
            else:
                rank = max(a_val.shape[-2], 1)
            alpha_val = jnp.asarray(1.0 / rank, dtype=a_val.dtype)
        else:
            alpha_val = jnp.asarray(alpha_val, dtype=a_val.dtype)
        new_entries[parent_path + ('lora_alpha',)] = alpha_val

    flat_params.update(new_entries)
    return freeze(unflatten_dict(flat_params)), lora_kernel_paths


def _update_lora_params(params, lora_a, lora_b, lora_alpha):
    """Update LoRA parameters inside an existing params tree."""
    params_dict = _to_mutable(params)
    flat_params = flatten_dict(params_dict, keep_empty_nodes=True)
    flat_a = flatten_dict(_to_mutable(lora_a), keep_empty_nodes=True)
    flat_b = flatten_dict(_to_mutable(lora_b), keep_empty_nodes=True)
    flat_alpha = flatten_dict(_to_mutable(lora_alpha), keep_empty_nodes=True)

    for path, a_val in flat_a.items():
        if a_val is None:
            continue
        parent_path = path[:-1]
        flat_params[parent_path + ('lora_a',)] = a_val
        flat_params[parent_path + ('lora_b',)] = flat_b[path]
        alpha_val = flat_alpha.get(path)
        if alpha_val is None:
            if a_val.ndim == 0:
                rank = 1
            elif a_val.ndim == 1:
                rank = max(a_val.shape[0], 1)
            else:
                rank = max(a_val.shape[-2], 1)
            alpha_val = jnp.asarray(1.0 / rank, dtype=a_val.dtype)
        else:
            alpha_val = jnp.asarray(alpha_val, dtype=a_val.dtype)
        flat_params[parent_path + ('lora_alpha',)] = alpha_val

    return freeze(unflatten_dict(flat_params))


def _extract_lora_params(params, lora_template_a, lora_template_b):
    """Extract LoRA matrices using template structure."""
    params_dict = _to_mutable(params)
    flat_params = flatten_dict(params_dict, keep_empty_nodes=True)
    flat_template_a = flatten_dict(_to_mutable(lora_template_a), keep_empty_nodes=True)
    flat_template_b = flatten_dict(_to_mutable(lora_template_b), keep_empty_nodes=True)

    out_a = {}
    out_b = {}
    for path, template_val in flat_template_a.items():
        if template_val is None:
            out_a[path] = None
            out_b[path] = None
            continue
        parent_path = path[:-1]
        out_a[path] = flat_params[parent_path + ('lora_a',)]
        out_b[path] = flat_params[parent_path + ('lora_b',)]

    return freeze(unflatten_dict(out_a)), freeze(unflatten_dict(out_b))


def _create_lora_mask(params, lora_kernel_paths, lora_spec):
    """Create optimizer mask that respects LoRA and full-finetune specs."""
    params_dict = _to_mutable(params)
    # Build a base mask that only contains leaf entries of the parameter tree.
    mask_flat = {path: False for path in flatten_dict(params_dict, keep_empty_nodes=False)}

    spec_flat = flatten_dict(_to_mutable(lora_spec), keep_empty_nodes=False)
    for path, spec_val in spec_flat.items():
        if isinstance(spec_val, dict):
            continue
        if spec_val == 'full':
            mask_flat[path] = True
        elif spec_val == 'freeze':
            mask_flat[path] = False
        else:
            # Numeric rank -> LoRA; base weights stay frozen.
            mask_flat[path] = False

    for kernel_path in lora_kernel_paths:
        parent_path = kernel_path[:-1]
        mask_flat[parent_path + ('lora_a',)] = True
        mask_flat[parent_path + ('lora_b',)] = True
        mask_flat[parent_path + ('lora_alpha',)] = False

    return freeze(unflatten_dict(mask_flat))


def create_lora_spec(params, decision_fn):
    """
    Create LoRA specification by applying decision_fn to each parameter.

    Args:
        params: Parameter tree
        decision_fn: Function (path, param) -> lora_dim or 'full' or 'freeze'

    Returns:
        lora_spec: Dict tree with LoRA dimensions
    """
    def apply_decision(path, param):
        # Convert path to string for decision function
        path_str = '/'.join(str(k) for k in path)
        return decision_fn(path_str, param)

    return jax.tree_util.tree_map_with_path(apply_decision, params)

class ModelAppenderV3:
    """
    LoRA-based multi-decoder without qax/lorax dependencies.

    Maintains the same API as ModelAppenderV2 but uses native JAX/Flax layers.
    """

    def __init__(
        self,
        base_model,
        append_config: AppendConfig = AppendConfig(),
    ):
        self.base_model = base_model
        self.append_config = append_config
        self.logger = get_logger(__name__)

        # Storage pools
        self.lora_params_pool = {}
        self.optimizer_state_pool = {}
        self.step_counter_pool = {}

        # Treat decoder_id == 0 as the "base" model without any adapter delta
        self.base_decoder_id = 0

        # Metadata
        self.base_params = None
        self.current_decoder_id = None
        self.lora_spec = None
        self.lora_alpha = 1.0
        self.lora_template_a = None
        self.lora_template_b = None
        self.lora_template_alpha = None
        self.lora_kernel_paths = []
        self.mask_tree = None
        self.lora_optimizer_config = {
            'optimizer_cls': optax.adam,
            'optimizer_kwargs': {
                'learning_rate': 1e-4,
            },
        }

        self.wrap_model()

    def __getstate__(self):
        if self.current_decoder_id is not None:
            self.save_lora_params(self.current_decoder_id)
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def wrap_model(self):
        params = self.base_model.train_state.params
        self.base_params = params

        def decision_fn(path, param):
            if 'embedding' in path:
                self.logger.info(f'Fully finetuning param {path}')
                return 'full'
            if len(param.shape) == 1:
                return 'freeze'
            dim = self.append_config.lora_dim
            self.logger.info(f'Using LoRA with dim={dim} for param {path}')
            return dim

        self.lora_spec = create_lora_spec(params, decision_fn)

        lora_a, lora_b, lora_alpha, lora_info = init_lora_matrices(
            params=params,
            lora_spec=self.lora_spec,
            rng=jax.random.PRNGKey(0),
            alpha=self.lora_alpha,
        )

        self.lora_alpha = lora_info['alpha']
        self.lora_template_a = lora_a
        self.lora_template_b = lora_b
        self.lora_template_alpha = lora_alpha

        def _zeros_like(tree):
            return jax.tree_util.tree_map(
                lambda x: None if x is None else jnp.zeros_like(x),
                tree,
                is_leaf=lambda x: x is None,
            )

        self.zero_lora_a = _zeros_like(self.lora_template_a)
        self.zero_lora_b = _zeros_like(self.lora_template_b)

        params_with_lora, lora_kernel_paths = _insert_lora_params(
            params, lora_a, lora_b, lora_alpha
        )
        self.lora_kernel_paths = lora_kernel_paths
        self.mask_tree = _create_lora_mask(params_with_lora, lora_kernel_paths, self.lora_spec)

        base_optimizer = self.lora_optimizer_config['optimizer_cls'](
            **self.lora_optimizer_config['optimizer_kwargs']
        )
        # Freeze base weights by zeroing their updates before applying the trainable mask
        frozen_mask = jax.tree_util.tree_map(lambda flag: not flag, self.mask_tree)
        masked_tx = optax.chain(
            optax.masked(optax.set_to_zero(), frozen_mask),
            optax.masked(base_optimizer, self.mask_tree),
        )
        opt_state = masked_tx.init(params_with_lora)

        self.base_model.train_state = self.base_model.train_state.replace(
            params=params_with_lora,
            tx=masked_tx,
            opt_state=opt_state,
            step=0
        )

        self.current_decoder_id = None
        # Ensure the base decoder (id=0) is registered with zero deltas
        self.lora_params_pool[self.base_decoder_id] = {
            'lora_a': self.zero_lora_a,
            'lora_b': self.zero_lora_b,
        }
        self.switch_decoder(0)

    def save_lora_params(self, decoder_id):
        if decoder_id == self.base_decoder_id:
            # Always keep the base decoder as the frozen, zero-delta adapter
            return
        lora_a, lora_b = _extract_lora_params(
            self.base_model.train_state.params,
            self.lora_template_a,
            self.lora_template_b,
        )
        self.lora_params_pool[decoder_id] = {
            'lora_a': jax.tree.map(lambda x: x, lora_a),
            'lora_b': jax.tree.map(lambda x: x, lora_b),
        }
        self.optimizer_state_pool[decoder_id] = jax.tree.map(
            lambda x: x, self.base_model.train_state.opt_state
        )
        self.step_counter_pool[decoder_id] = int(self.base_model.train_state.step)

    def load_lora_params(self, decoder_id):
        if decoder_id not in self.lora_params_pool:
            if decoder_id == self.base_decoder_id:
                # Lazily register the base decoder with zero LoRA weights
                self.lora_params_pool[decoder_id] = {
                    'lora_a': self.zero_lora_a,
                    'lora_b': self.zero_lora_b,
                }
                return self.zero_lora_a, self.zero_lora_b
            self.logger.info(f"Initializing new LoRA params for decoder_id {decoder_id}")
            lora_a, lora_b, _, _ = init_lora_matrices(
                params=self.base_params,
                lora_spec=self.lora_spec,
                rng=jax.random.PRNGKey(decoder_id),
                alpha=self.lora_alpha,
            )
            self.lora_params_pool[decoder_id] = {
                'lora_a': lora_a,
                'lora_b': lora_b,
            }
        entry = self.lora_params_pool[decoder_id]
        return entry['lora_a'], entry['lora_b']

    def switch_decoder(self, decoder_id):
        if self.current_decoder_id == decoder_id:
            return

        if self.current_decoder_id is not None:
            self.save_lora_params(self.current_decoder_id)

        lora_a, lora_b = self.load_lora_params(decoder_id)
        new_params = _update_lora_params(
            self.base_model.train_state.params,
            lora_a,
            lora_b,
            self.lora_template_alpha,
        )

        if decoder_id in self.optimizer_state_pool:
            opt_state = self.optimizer_state_pool[decoder_id]
            step = self.step_counter_pool[decoder_id]
        else:
            opt_state = self.base_model.train_state.tx.init(new_params)
            step = 0

        self.base_model.train_state = self.base_model.train_state.replace(
            params=new_params,
            opt_state=opt_state,
            step=step,
        )
        self.current_decoder_id = decoder_id

    def train_model(self, x, cond, decoder_id, **kwargs):
        x = np.asarray(x)
        cond = np.asarray(cond)
        decoder_id = np.asarray(decoder_id)

        if x.ndim == 2:
            x = x[:, None, :]
        if cond.ndim == 2:
            cond = cond[:, None, :]

        unique_decoder_ids = np.unique(decoder_id)
        if len(unique_decoder_ids) > 1:
            raise ValueError("Multiple decoder_ids are not supported in a single batch")

        self.switch_decoder(int(unique_decoder_ids[0]))
        metric = self.base_model.train_model(x, cond, **kwargs)
        return metric

    def eval_model(self, cond, decoder_id=None):
        cond = np.asarray(cond)
        if cond.ndim == 2:
            cond = cond[:, None, :]

        if decoder_id is None:
            raise ValueError(
                "decoder_id must be provided. "
                "Pass an array matching the batch size just like ModelAppenderV2."
            )

        decoder_id = np.asarray(decoder_id)
        if decoder_id.size == 0:
            raise ValueError("decoder_id array is empty")
        if np.any(decoder_id < 0):
            self.logger.info( 
                f"decoder_id contains negative values {np.unique(decoder_id)}. "
                "Ensure the interface returns valid decoder indices. Default Fallback to id 0"
            )
            decoder_id = 0

        if decoder_id.ndim == 1:
            decoder_id = decoder_id[:, None, None]
        if decoder_id.ndim == 2:
            decoder_id = decoder_id[:, None, :]

        unique_decoder_ids = np.unique(decoder_id)

        if len(unique_decoder_ids) == 1:
            self.switch_decoder(int(unique_decoder_ids[0]))
            return self.base_model.eval_model(cond)

        masked_actions = []
        from SILGym.models.skill_decoder.appender import mask_fn

        for uid in unique_decoder_ids:
            mask = (decoder_id == uid).astype(np.float32)
            self.switch_decoder(int(uid))
            actions = self.base_model.eval_model(cond)
            masked_actions.append(mask_fn(actions, mask))

        return np.sum(masked_actions, axis=0)

    def reinit_optimizer(self):
        tx = self.base_model.train_state.tx
        opt_state = tx.init(self.base_model.train_state.params)
        self.base_model.train_state = self.base_model.train_state.replace(
            opt_state=opt_state,
            step=0,
        )
        self.logger.info(f"Reinitialized optimizer for decoder_id {self.current_decoder_id}")

    def get_num_decoders(self):
        return len(self.lora_params_pool)

    def clear_decoder(self, decoder_id):
        if decoder_id in self.lora_params_pool:
            del self.lora_params_pool[decoder_id]
        if decoder_id in self.optimizer_state_pool:
            del self.optimizer_state_pool[decoder_id]
        if decoder_id in self.step_counter_pool:
            del self.step_counter_pool[decoder_id]
        self.logger.info(
            f"Cleared LoRA params, optimizer state, and step counter for decoder_id {decoder_id}"
        )


def test_model_appender_v3():
    """Test ModelAppenderV3 functionality."""
    logger = get_logger(__name__)
    logger.info("\n=== Testing ModelAppenderV3 ===")

    # Create base model
    from SILGym.config.experiment_config import DEFAULT_DECODER_CONFIG
    base_model = DEFAULT_DECODER_CONFIG["model_cls"](
        **DEFAULT_DECODER_CONFIG["model_kwargs"]
    )

    # Create ModelAppenderV3
    append_config = AppendConfig(lora_dim=4, pool_length=10)
    appender_v3 = ModelAppenderV3(base_model=base_model, append_config=append_config)

    logger.info("ModelAppenderV3 created successfully.")

    # Test 1: Check initial state
    logger.info(f"Current decoder ID: {appender_v3.current_decoder_id}")
    logger.info(f"Number of stored decoders: {appender_v3.get_num_decoders()}")

    # Test 2: Switch between decoders
    logger.info("\nTesting decoder switching...")
    appender_v3.switch_decoder(1)
    logger.info(f"Switched to decoder 1")

    appender_v3.switch_decoder(2)
    logger.info(f"Switched to decoder 2")

    appender_v3.switch_decoder(0)
    logger.info(f"Switched back to decoder 0")

    logger.info(f"Number of stored decoders after switching: {appender_v3.get_num_decoders()}")

    # Test 3: Training
    logger.info("\nTesting training...")
    batch_size = 2
    feature_dim = 60

    x = np.random.randn(batch_size, 9).astype(np.float32)
    cond_train = np.random.randn(batch_size, feature_dim).astype(np.float32)
    decoder_id_train = np.array([1, 1])

    try:
        metric = appender_v3.train_model(x, cond_train, decoder_id_train)
        logger.info(f"Training successful. Loss: {metric[1].get('train/loss', 'N/A')}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Eval with single decoder
    logger.info("\nTesting evaluation with single decoder...")
    cond = np.random.randn(4, feature_dim).astype(np.float32)
    decoder_ids = np.array([0, 0, 0, 0])

    try:
        result = appender_v3.eval_model(cond, decoder_ids)
        logger.info(f"Evaluation successful. Result shape: {result.shape}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Eval with multiple decoders
    logger.info("\nTesting evaluation with multiple decoders...")
    decoder_ids_mixed = np.array([0, 1, 2, 0])

    try:
        result = appender_v3.eval_model(cond, decoder_ids_mixed)
        logger.info(f"Mixed evaluation successful. Result shape: {result.shape}")
    except Exception as e:
        logger.error(f"Mixed evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nModelAppenderV3 tests completed.")


if __name__ == "__main__":
    test_model_appender_v3()
