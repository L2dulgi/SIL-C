from SILGym.models.skill_decoder.diffusion_base import CondDDPMDecoder
import jax
import jax.numpy as jnp
import jax.tree_util
from flax.training import train_state
import optax
import numpy as np
from dataclasses import dataclass, field, fields, is_dataclass
from einops import rearrange, repeat
from SILGym.utils.logger import get_logger

# Optional imports for LoRA functionality
try:
    from lorax.constants import LORA_FULL, LORA_FREEZE
    import lorax
    LORAX_AVAILABLE = True
except ImportError:
    LORAX_AVAILABLE = False
    # Dummy constants for when lorax is not available
    LORA_FULL = "LORA_FULL"
    LORA_FREEZE = "LORA_FREEZE"
    lorax = None

try:
    import qax
    from qax.utils import freeze_subtrees
    QAX_AVAILABLE = True
except ImportError:
    QAX_AVAILABLE = False
    qax = None
    freeze_subtrees = None

if QAX_AVAILABLE:
    @dataclass
    class LoraWeightPool(qax.ImplicitArray):
        '''
        lora with masked path
        '''
        w : qax.ArrayValue # M x N
        pool_mask : qax.ArrayValue # B ( zero one value )
        a : qax.ArrayValue # B x k x N
        b : qax.ArrayValue # B x M x k
        alpha : float = qax.aux_field(default=1.)

        def __post_init__(self):
            super().__post_init__()
            assert self.a.shape[-2] == self.b.shape[-1]

        def materialize(self):
            lora_pool = self.get_scale() * self.b @ self.a
            masked_lora_pool = self.pool_mask[..., None, None] * lora_pool
            delta_w = jnp.sum(masked_lora_pool, axis=0) / jnp.maximum(jnp.sum(self.pool_mask), 1)
            return (self.w + delta_w).astype(self.w.dtype)

        def get_scale(self):
            return self.alpha / self.b.shape[-1]
else:
    # Dummy class when qax is not available
    @dataclass
    class LoraWeightPool:
        '''
        lora with masked path (dummy implementation when qax is not available)
        '''
        w : jnp.ndarray # M x N
        pool_mask : jnp.ndarray # B ( zero one value )
        a : jnp.ndarray # B x k x N
        b : jnp.ndarray # B x M x k
        alpha : float = 1.

        def __post_init__(self):
            assert self.a.shape[-2] == self.b.shape[-1]

        def materialize(self):
            lora_pool = self.get_scale() * self.b @ self.a
            masked_lora_pool = self.pool_mask[..., None, None] * lora_pool
            delta_w = jnp.sum(masked_lora_pool, axis=0) / jnp.maximum(jnp.sum(self.pool_mask), 1)
            return (self.w + delta_w).astype(self.w.dtype)

        def get_scale(self):
            return self.alpha / self.b.shape[-1]

@dataclass
class AppendConfig :
    lora_dim : int = 4
    pool_length : int = 10

@jax.jit
def mask_fn(tensor, mask):
    """
    Apply a mask to the input tensor.
    If the mask has a lower dimensionality than the tensor,
    it will be broadcast to match the tensor shape.

    Args:
        tensor: Input tensor of shape (B, f).
        mask: Mask tensor of shape (B, 1) or (B, f).
    Returns:
        The masked tensor of shape (B, f).
    """
    # Broadcast the mask to match the tensor shape if necessary
    broadcast_mask = jnp.broadcast_to(mask, tensor.shape)
    return tensor * broadcast_mask

# Overwrite the freeze_keys function to add '.' exception for jax 0.4.34 treeutil update
# jax/_src/tree_util.py GetAttrKey
if QAX_AVAILABLE:
    def freeze_keys(
        optimizer: optax.GradientTransformation, arr_type, keys, use_scalar_zeros=False
    ) -> optax.GradientTransformation:
        keys = set(keys)
        # add '.' for keys for jax 0.4.34
        keys = [f'.{key}' if not key.startswith('.') else key for key in keys]
        def label_leaf(leaf):
            if not isinstance(leaf, arr_type):
                return "train"
            children, aux_data = leaf.tree_flatten_with_keys()
            # here string format is different for jax 0.4.34 treeutil update
            labels = ["freeze" if str(key) in keys else "train" for key, _ in children]
            struct = leaf.tree_unflatten(aux_data, labels)
            return struct

        def label_fn(root):
            return jax.tree.map(label_leaf, root, is_leaf=lambda x: isinstance(x, arr_type))

        return freeze_subtrees(optimizer, label_fn, use_scalar_zeros=use_scalar_zeros)
else:
    def freeze_keys(*args, **kwargs):
        raise RuntimeError("freeze_keys requires qax library to be installed. Please install qax to use LoRA functionality.")


class ModelAppender() :
    def __init__(
            self,
            base_model : CondDDPMDecoder = None,
            append_config : AppendConfig = AppendConfig(),
        ) :
        if not LORAX_AVAILABLE or not QAX_AVAILABLE:
            missing = []
            if not LORAX_AVAILABLE:
                missing.append("jax-lorax")
            if not QAX_AVAILABLE:
                missing.append("qax")
            raise RuntimeError(
                f"ModelAppender requires {' and '.join(missing)} to be installed. "
                f"Please install with: pip install {' '.join(missing)}"
            )

        self.base_model = base_model
        self.append_config = append_config
        self.lora_optimizer_config = None
        self.logger = get_logger(__name__)
        self.wrap_model()
    
    def init_lora_pool(
            self,
            param_tree, 
            spec, 
            rng,
            stddev=0.01, 
            dtype=jnp.float32, 
            alpha=1., 
            pool_size=10, # default pool size
            is_leaf=None,
        ):
        
        def iter_keys(key):
            while True:
                key, out_key = jax.random.split(key)
                yield out_key
        key_it = iter_keys(rng)

        self.implict_array_cls = LoraWeightPool
        def get_param(path, param, spec_val):
            if spec_val in (LORA_FREEZE, LORA_FULL):
                return param

            if len(param.shape) == 1:
                raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

            if len(param.shape) == 2:
                b_dim, a_dim = param.shape

                pool_mask = jnp.zeros((pool_size,), dtype=dtype)
                a = jax.random.normal(next(key_it), (pool_size, spec_val, a_dim), dtype=dtype) * stddev
                b = jnp.zeros((pool_size, b_dim, spec_val), dtype=dtype)

                implict_array_kwargs = {
                    'w' : param,
                    'pool_mask' : pool_mask,
                    'alpha' : alpha,
                }
                
                implict_array_kwargs['a'] = a
                implict_array_kwargs['b'] = b
                return LoraWeightPool(**implict_array_kwargs)

            # conv case
            *window_shape, in_channels, out_channels = param.shape

            pool_mask = jnp.zeros((pool_size,), dtype=dtype)
            a = jnp.zeros((
                *(1 for _ in range(len(window_shape))),
                spec_val,
                out_channels
            ), dtype=param.dtype)
            b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev

            a = jnp.repeat(a, pool_size, axis=0)
            b = jnp.repeat(b, pool_size, axis=0)

            implict_array_kwargs = {
                'w' : param,
                'pool_mask' : pool_mask,
                'alpha' : alpha,
            }

            implict_array_kwargs['a'] = a
            implict_array_kwargs['b'] = b
            
            return self.implict_array_cls(param, pool_mask, a, b, alpha=alpha)

        return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)

    def wrap_pool_optimizer(
            self,
            optimizer : optax.GradientTransformation, 
            spec, 
            scalar_frozen_grads=False,
            fixed_components = ['w', 'pool_mask'],
            pool_cls = LoraWeightPool,
        ):
        full_freeze_labels = jax.tree.map(
            lambda x: 'freeze' if x == LORA_FREEZE else 'train',
            spec
        )

        optimizer_with_full_freeze = qax.utils.freeze_subtrees(
            optimizer,
            full_freeze_labels,
            use_scalar_zeros=scalar_frozen_grads
        )

        optimizer_with_qax_freeze = freeze_keys(
            optimizer_with_full_freeze, 
            pool_cls, 
            fixed_components, 
            use_scalar_zeros=scalar_frozen_grads
        )
        return optimizer_with_qax_freeze

    def wrap_model(self) :

        params = self.base_model.train_state.params

        self.base_model.model.apply = lorax.lora(self.base_model.model.apply)
        self.base_model.model_eval.apply = lorax.lora(self.base_model.model_eval.apply)
        
        def decision_fn(path, param):
            if 'embedding' in path:
                self.logger.info(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.append_config.lora_dim
            self.logger.info(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        self.lora_spec = lorax.simple_spec(
            params=params,
            decision_fn=decision_fn,
            tune_vectors=False,
        )

        lora_params = self.init_lora_pool(
            param_tree=params,
            spec=self.lora_spec,
            rng=jax.random.PRNGKey(0),
            pool_size=self.append_config.pool_length,
        )
        self.init_lora_params = lora_params

        if self.lora_optimizer_config is None :
            self.lora_optimizer_config ={
                'optimizer_cls' : optax.adam,
                'optimizer_kwargs' : {
                    'learning_rate' : 1e-4,
                },
            }

        optimizer = self.lora_optimizer_config['optimizer_cls'](
            **self.lora_optimizer_config['optimizer_kwargs']
        )

        lora_optimizer = self.wrap_pool_optimizer(
            optimizer=optimizer,
            spec=self.lora_spec,
            fixed_components=['w', 'pool_mask'], 
            pool_cls=self.implict_array_cls,
        )

        self.base_model.train_state =  train_state.TrainState.create(
            apply_fn=self.base_model.model.apply, 
            params=self.init_lora_params, 
            tx=lora_optimizer,
        )
        
    def set_pool_mask(self, pool_idx, eval=False) : 
        def set_pool_mask(params, mask, mode='t'):
            '''
            pool_mask setter
            * must used in LoraWeightPool's masking function
            '''
            target_mask = f'.pool_mask'
            def set_mask_leaf(path, param):
                if path[-1].__str__() == target_mask : # for new version. validate the mask leaf by pool mask 0401
                    if param.shape != mask.shape:
                        raise ValueError(f'mask shape must be equal to param mask shape {param.shape}, but got {mask.shape}\n in path {path}')
                    return mask
                return param
            return jax.tree_util.tree_map_with_path(set_mask_leaf, params)
        pool_mask = np.zeros((self.append_config.pool_length,), jnp.float32)
        pool_mask[pool_idx] = 1.
        replace_params = set_pool_mask(self.base_model.train_state.params, pool_mask)
        if eval : 
            return replace_params
        else : 
            self.base_model.train_state = self.base_model.train_state.replace(params=replace_params)
            return None

    def reinit_optimizer(self):
        new_optimizer = self.lora_optimizer_config['optimizer_cls'](
            **self.lora_optimizer_config['optimizer_kwargs']
        )
        lora_optimizer = self.wrap_pool_optimizer(
            optimizer=new_optimizer,
            spec=self.lora_spec,
            fixed_components=['w', 'pool_mask'], 
            pool_cls=self.implict_array_cls,
        )
        new_train_state = train_state.TrainState.create(
            apply_fn=self.base_model.model.apply, 
            params=self.init_lora_params, 
            tx=lora_optimizer,
        )

        # only update the optimizer.
        self.base_model.train_state = self.base_model.train_state.replace(
            step=new_train_state.step,
            tx=new_train_state.tx,
            opt_state=new_train_state.opt_state,
        )

    def train_model(self, x, cond, decoder_id, **kwargs):
        if x.ndim == 2:
            x = x[:, None, :]
        if cond.ndim == 2:
            cond = cond[:, None, :]

        unique_decoder_ids = np.unique(decoder_id)
        if len(unique_decoder_ids) > 1:
            raise ValueError("Multiple decoder_ids are not supported in this version., this is handled by dataloader")
        
        unique_id = unique_decoder_ids[0]
        self.set_pool_mask(pool_idx=[int(unique_id)], eval=False)
        metric = self.base_model.train_model(x, cond, **kwargs)
        return metric

    def eval_model(self, cond, decoder_id = None):
        # for sample logit (B, feat) and decoder id (B, 1)
        # always change the cond and decoder to 3 ndim vector.
        if decoder_id.ndim == 1:
            decoder_id = decoder_id[:, None, None]
        if decoder_id.ndim == 2:
            decoder_id = decoder_id[:, None, :]

        if cond.ndim == 2:
            cond = cond[:, None, :]

        unique_decoder_ids = np.unique(decoder_id)
        masked_actions = []

        for uid in unique_decoder_ids:
            # Create a binary mask for samples matching the current unique decoder id.
            base_mask = (decoder_id == uid).astype(np.float32) # shape (B, 1, 1) 
            # Set the pool mask for the current decoder id; note that pool_idx is expected as a list.
            params = self.set_pool_mask(pool_idx=[int(uid)], eval=True)
            # Evaluate the model with the masked condition.
            actions = self.base_model.eval_model(cond, params) # actions (B, 1, out)
            masked_actions.append(mask_fn(actions, base_mask)) # shape (B, 1, out)
        
        # Recover the final actions by summing the masked actions.
        final_actions = np.zeros_like(masked_actions[0])
        for i, action in enumerate(masked_actions):
            final_actions += action
        
        return final_actions

class ModelAppenderV2(ModelAppender):
    '''
    More efficient appender that stores LoRA parameters (a, b) in memory
    and replaces params directly for evaluation instead of using pool masking.

    Key differences from ModelAppender:
    1. No pool masking - directly swaps LoRA parameters
    2. Stores only the a,b matrices for each decoder_id in memory
    3. Faster switching between decoders (no mask computation)
    4. More memory efficient when using many decoders

    Usage:
        # Create appender
        appender = ModelAppenderV2(base_model, append_config)

        # Training automatically switches decoder
        appender.train_model(x, cond, decoder_id=[0, 0])

        # Evaluation supports mixed decoder_ids in batch
        results = appender.eval_model(cond, decoder_id=[0, 1, 2, 0])

        # Manual decoder management
        appender.switch_decoder(3)  # Switch to decoder 3
        appender.get_num_decoders()  # Get number of stored decoders
        appender.clear_decoder(1)  # Remove decoder 1 from memory
    '''

    def __init__(
        self,
        base_model: CondDDPMDecoder = None,
        append_config: AppendConfig = AppendConfig(),
    ):
        # Store LoRA parameters for each decoder_id
        self.lora_params_pool = {}  # {decoder_id: params}
        self.optimizer_state_pool = {}  # {decoder_id: opt_state}
        self.step_counter_pool = {}  # {decoder_id: step}
        self.base_params = None  # Store base model parameters
        self.current_decoder_id = None

        super().__init__(base_model, append_config)

    def __getstate__(self):
        """Custom pickle: save current decoder state before serialization."""
        # Save current decoder's state before pickling
        if self.current_decoder_id is not None:
            self.logger.info(f"__getstate__: Saving current decoder {self.current_decoder_id} before pickle")
            self.save_lora_params(self.current_decoder_id)
        return self.__dict__

    def __setstate__(self, state):
        """Custom unpickle: restore from saved state."""
        self.__dict__.update(state)
        
    def wrap_model(self):
        """Override to store base params and initialize differently."""
        params = self.base_model.train_state.params
        self.base_params = params  # Store original base params
        
        self.base_model.model.apply = lorax.lora(self.base_model.model.apply)
        self.base_model.model_eval.apply = lorax.lora(self.base_model.model_eval.apply)
        
        def decision_fn(path, param):
            if 'embedding' in path:
                self.logger.info(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.append_config.lora_dim
            self.logger.info(f'Using LoRA with dim={dim} for param {path}')
            return dim
        
        self.lora_spec = lorax.simple_spec(
            params=params,
            decision_fn=decision_fn,
            tune_vectors=False,
        )
        
        # Initialize first LoRA param structure using parent's init method
        # but with pool_size=1 to create a template
        template_lora_params = self.init_lora_pool(
            param_tree=params,
            spec=self.lora_spec,
            rng=jax.random.PRNGKey(0),
            stddev=0.01,  # Use same stddev as ModelAppender
            pool_size=1,  # Single set, not a pool
        )
        
        # Extract the structure to use as template
        self.lora_param_template = template_lora_params
        
        # Setup optimizer
        if self.lora_optimizer_config is None:
            self.lora_optimizer_config = {
                'optimizer_cls': optax.adam,
                'optimizer_kwargs': {
                    'learning_rate': 1e-4,
                },
            }
        
        optimizer = self.lora_optimizer_config['optimizer_cls'](
            **self.lora_optimizer_config['optimizer_kwargs']
        )
        
        # Use parent's optimizer wrapper
        lora_optimizer = self.wrap_pool_optimizer(
            optimizer=optimizer,
            spec=self.lora_spec,
            fixed_components=['w', 'pool_mask'],
            pool_cls=LoraWeightPool,
        )
        
        # Create initial train state with template params
        self.base_model.train_state = train_state.TrainState.create(
            apply_fn=self.base_model.model.apply,
            params=template_lora_params,
            tx=lora_optimizer,
        )

        # Set initial decoder_id and ensure pool_mask is properly set
        self.current_decoder_id = None  # Start with None so switch_decoder will initialize
        self.switch_decoder(0)
    
    def _extract_lora_ab(self, lora_weight_pool):
        """Extract a and b parameters from a LoraWeightPool."""
        if isinstance(lora_weight_pool, LoraWeightPool):
            # Get the a and b tensors for the active decoder (index 0 since pool_size=1)
            return {
                'a': lora_weight_pool.a[0],  # Shape: (r, out_dim)
                'b': lora_weight_pool.b[0],  # Shape: (in_dim, r)
            }
        return None
    
    def _create_lora_weight_pool(self, base_param, lora_ab, pool_mask):
        """Create a LoraWeightPool from separate components."""
        if lora_ab is None:
            return base_param
        
        # Expand a and b to have pool dimension
        a_expanded = lora_ab['a'][None, ...]  # Shape: (1, r, out_dim)
        b_expanded = lora_ab['b'][None, ...]  # Shape: (1, in_dim, r)
        
        return LoraWeightPool(
            w=base_param,
            pool_mask=pool_mask,
            a=a_expanded,
            b=b_expanded,
            alpha=1.0
        )
    
    def save_lora_params(self, decoder_id):
        """Save current LoRA parameters, optimizer state, and step counter for a specific decoder_id."""
        # Save the actual params tree (not just a,b extracts) to maintain structure
        current_params = self.base_model.train_state.params
        self.lora_params_pool[decoder_id] = jax.tree.map(lambda x: x, current_params)

        # Save optimizer state for this decoder
        current_opt_state = self.base_model.train_state.opt_state
        self.optimizer_state_pool[decoder_id] = jax.tree.map(lambda x: x, current_opt_state)

        # Save step counter for this decoder
        self.step_counter_pool[decoder_id] = int(self.base_model.train_state.step)

        # self.logger.info(f"Saved LoRA params, optimizer state, and step for decoder_id {decoder_id}")
    
    def load_lora_params(self, decoder_id):
        """Load LoRA parameters for a specific decoder_id."""
        if decoder_id not in self.lora_params_pool:
            # Initialize new LoRA params if not exists
            self.logger.info(f"Initializing new LoRA params for decoder_id {decoder_id}")

            # Initialize new LoRA params for any decoder_id
            # Use consistent initialization like ModelAppender
            new_params = self.init_lora_pool(
                param_tree=self.base_params,
                spec=self.lora_spec,
                rng=jax.random.PRNGKey(decoder_id),
                stddev=0.01,  # Use same stddev as ModelAppender
                pool_size=1,
            )

            # Set pool_mask to active by manually traversing the tree
            # Can't use jax.tree_map because LoraWeightPool is itself a PyTree
            def set_pool_mask_to_one(params_tree):
                """Recursively set pool_mask=[1.] for all LoraWeightPool objects."""
                if isinstance(params_tree, LoraWeightPool):
                    return LoraWeightPool(
                        w=params_tree.w,
                        pool_mask=jnp.ones_like(params_tree.pool_mask),
                        a=params_tree.a,
                        b=params_tree.b,
                        alpha=params_tree.alpha
                    )
                elif isinstance(params_tree, dict):
                    return {k: set_pool_mask_to_one(v) for k, v in params_tree.items()}
                else:
                    return params_tree

            new_params = set_pool_mask_to_one(new_params)

            # Debug: verify pool_mask was set correctly
            sample_param = new_params['params']['mlp0']['kernel']
            self.logger.info(f"load_lora_params: After set_pool_mask_to_one, pool_mask={sample_param.pool_mask}")

            # Save the params immediately so next time we return the same objects
            self.lora_params_pool[decoder_id] = new_params

            return new_params

        # Return saved params directly (maintains pytree structure for opt_state)
        return self.lora_params_pool[decoder_id]
    
    def switch_decoder(self, decoder_id):
        """Switch to a different decoder by loading its LoRA params and optimizer state."""
        if self.current_decoder_id == decoder_id:
            return  # Already using this decoder

        # Save current LoRA params and optimizer state if we have one
        if self.current_decoder_id is not None:
            self.save_lora_params(self.current_decoder_id)

        # Load new LoRA params
        loaded_params = self.load_lora_params(decoder_id)

        # Load or initialize optimizer state for this decoder
        if decoder_id in self.optimizer_state_pool:
            # Restore saved optimizer state and step counter
            loaded_opt_state = self.optimizer_state_pool[decoder_id]
            loaded_step = self.step_counter_pool[decoder_id]
            self.base_model.train_state = self.base_model.train_state.replace(
                params=loaded_params,
                opt_state=loaded_opt_state,
                step=loaded_step
            )
        else:
            # Initialize fresh optimizer state for new decoder
            # Create a temporary train state to get a fresh opt_state
            temp_optimizer = self.lora_optimizer_config['optimizer_cls'](
                **self.lora_optimizer_config['optimizer_kwargs']
            )
            lora_optimizer = self.wrap_pool_optimizer(
                optimizer=temp_optimizer,
                spec=self.lora_spec,
                fixed_components=['w', 'pool_mask'],
                pool_cls=LoraWeightPool,
            )
            temp_train_state = train_state.TrainState.create(
                apply_fn=self.base_model.model.apply,
                params=loaded_params,
                tx=lora_optimizer,
            )
            # Use the fresh optimizer state, keep current step counter
            self.base_model.train_state = self.base_model.train_state.replace(
                params=loaded_params,
                opt_state=temp_train_state.opt_state,
                step=0  # Reset step counter for new decoder
            )

        self.current_decoder_id = decoder_id
    
    def train_model(self, x, cond, decoder_id, **kwargs):
        """Train model with automatic decoder switching."""
        if x.ndim == 2:
            x = x[:, None, :]
        if cond.ndim == 2:
            cond = cond[:, None, :]
        
        unique_decoder_ids = np.unique(decoder_id)
        if len(unique_decoder_ids) > 1:
            raise ValueError("Multiple decoder_ids are not supported in a single batch")
        
        unique_id = int(unique_decoder_ids[0])
        self.switch_decoder(unique_id)
        
        # Train with current params
        metric = self.base_model.train_model(x, cond, **kwargs)
        
        # The updated params are already in the train state after training
        # They will be saved when switching to another decoder
        
        return metric
    
    def eval_model(self, cond, decoder_id=None):
        """Evaluate model with efficient parameter switching."""
        if decoder_id.ndim == 1:
            decoder_id = decoder_id[:, None, None]
        if decoder_id.ndim == 2:
            decoder_id = decoder_id[:, None, :]
        
        if cond.ndim == 2:
            cond = cond[:, None, :]
        
        unique_decoder_ids = np.unique(decoder_id)
        
        if len(unique_decoder_ids) == 1:
            # Simple case: all samples use the same decoder
            self.switch_decoder(int(unique_decoder_ids[0]))
            return self.base_model.eval_model(cond)
        
        # Multiple decoders: need to evaluate each group separately
        masked_actions = []
        
        for uid in unique_decoder_ids:
            # Get mask for current decoder
            mask = (decoder_id == uid).astype(np.float32)
            
            # Switch to this decoder
            self.switch_decoder(int(uid))
            
            # Evaluate
            actions = self.base_model.eval_model(cond)
            masked_actions.append(mask_fn(actions, mask))
        
        # Combine results
        final_actions = np.sum(masked_actions, axis=0)
        return final_actions
    
    def reinit_optimizer(self):
        """
        Re-initialize optimizer while keeping the same params.
        Overrides parent method to handle V2-specific state management.
        """
        # Create new optimizer
        new_optimizer = self.lora_optimizer_config['optimizer_cls'](
            **self.lora_optimizer_config['optimizer_kwargs']
        )

        # Wrap optimizer with LoRA-specific configuration
        lora_optimizer = self.wrap_pool_optimizer(
            optimizer=new_optimizer,
            spec=self.lora_spec,
            fixed_components=['w', 'pool_mask'],
            pool_cls=LoraWeightPool,
        )

        # Create new train state with template params to get proper opt_state structure
        # Use current params to ensure structure matches (both have pool_size=1)
        new_train_state = train_state.TrainState.create(
            apply_fn=self.base_model.model.apply,
            params=self.base_model.train_state.params,  # Use current params for structure
            tx=lora_optimizer,
        )

        # Update only the optimizer-related parts, keeping params unchanged
        self.base_model.train_state = self.base_model.train_state.replace(
            step=new_train_state.step,
            tx=new_train_state.tx,
            opt_state=new_train_state.opt_state,
        )

        self.logger.info(f"Reinitialized optimizer for decoder_id {self.current_decoder_id}")

    def get_num_decoders(self):
        """Get the number of decoders currently stored."""
        return len(self.lora_params_pool)

    def clear_decoder(self, decoder_id):
        """Remove a decoder from memory."""
        if decoder_id in self.lora_params_pool:
            del self.lora_params_pool[decoder_id]
        if decoder_id in self.optimizer_state_pool:
            del self.optimizer_state_pool[decoder_id]
        if decoder_id in self.step_counter_pool:
            del self.step_counter_pool[decoder_id]
        self.logger.info(f"Cleared LoRA params, optimizer state, and step counter for decoder_id {decoder_id}")
  
def test_eval_model_with_2d_decoder_id(appended):
    """
    Test the eval_model method when:
      - The condition input has shape (batch_size, 1, feature_dim)
      - The decoder_id input has shape (batch_size, 1)
    """
    logger = get_logger(__name__)
    logger.info("=== Running eval_model Test with decoder_id as (B, 1) ===")
    
    # Define batch size and feature dimension
    batch_size = 4
    feature_dim = 60
    
    # Create a dummy condition input with shape (batch_size, 1, feature_dim)
    cond = np.random.randn(batch_size, 1, feature_dim).astype(np.float32)
    
    # Create a dummy decoder_id array with shape (batch_size, 1)
    decoder_id = np.array([[0], [0], [1], [1]])
    
    # For testing purposes, override the base model's eval_model method with a dummy.
    # In this dummy, the model simply returns the condition multiplied by 2.
    # appended.base_model.eval_model = lambda cond, params: cond * 2
    appended.base_model.model_eval.apply = appended.base_model.model.apply 
    
    # try:
    # Call eval_model with the condition and the 2D decoder_id.
    result = appended.eval_model(cond, decoder_id)
    logger.info(f"Test eval_model with 2D decoder_id output:\n{result}")
    # except Exception as e:
    #     print("eval_model test with 2D decoder_id raised an exception:", e)

def test_model_appender_v2():
    """Test ModelAppenderV2 functionality."""
    logger = get_logger(__name__)
    logger.info("\n=== Testing ModelAppenderV2 ===")
    
    # Create base model
    from SILGym.config.experiment_config import DEFAULT_DECODER_CONFIG
    base_model = DEFAULT_DECODER_CONFIG["model_cls"](
        **DEFAULT_DECODER_CONFIG["model_kwargs"]
    )
    
    # Create ModelAppenderV2
    append_config = AppendConfig(lora_dim=4, pool_length=10)
    appender_v2 = ModelAppenderV2(base_model=base_model, append_config=append_config)
    
    logger.info("ModelAppenderV2 created successfully.")
    
    # Test 1: Check initial state
    logger.info(f"Current decoder ID: {appender_v2.current_decoder_id}")
    logger.info(f"Number of stored decoders: {appender_v2.get_num_decoders()}")
    
    # Test 2: Switch between decoders
    logger.info("\nTesting decoder switching...")
    
    # Switch to decoder 1
    appender_v2.switch_decoder(1)
    logger.info(f"Switched to decoder 1")
    
    # Switch to decoder 2
    appender_v2.switch_decoder(2)
    logger.info(f"Switched to decoder 2")
    
    # Switch back to decoder 0
    appender_v2.switch_decoder(0)
    logger.info(f"Switched back to decoder 0")
    
    logger.info(f"Number of stored decoders after switching: {appender_v2.get_num_decoders()}")
    
    # Test 3: Eval with multiple decoders
    logger.info("\nTesting evaluation with multiple decoders...")
    batch_size = 4
    feature_dim = 60
    
    cond = np.random.randn(batch_size, feature_dim).astype(np.float32)
    decoder_ids = np.array([0, 1, 2, 0])  # Different decoders for different samples
    
    try:
        result = appender_v2.eval_model(cond, decoder_ids)
        logger.info(f"Evaluation successful. Result shape: {result.shape}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    
    # Test 4: Training
    logger.info("\nTesting training...")
    x = np.random.randn(2, 9).astype(np.float32)  # action dim = 9
    cond_train = np.random.randn(2, feature_dim).astype(np.float32)
    decoder_id_train = np.array([1, 1])
    
    try:
        metric = appender_v2.train_model(x, cond_train, decoder_id_train)
        logger.info("Training successful")
    except Exception as e:
        logger.error(f"Training failed: {e}")
    
    logger.info("\nModelAppenderV2 tests completed.")


if __name__ == "__main__":
    from SILGym.models.basic.base import DenoisingMLP
    from SILGym.config.experiment_config import DEFAULT_DECODER_CONFIG
    from SILGym.models.skill_decoder.diffusion_base import CondDDPMDecoder

    logger = get_logger(__name__)
    
    # Test original ModelAppender
    logger.info("=== Testing Original ModelAppender ===")
    
    # Create the base model using the default configuration.
    base_model = DEFAULT_DECODER_CONFIG["model_cls"](
            **DEFAULT_DECODER_CONFIG["model_kwargs"]
        )
    
    # Instantiate the ModelAppender with the base model.
    append_config = AppendConfig(lora_dim=4, pool_length=10)
    appended = ModelAppender(base_model=base_model, append_config=append_config)
    logger.info("ModelAppender created successfully.\n")
    
    # Print a summary of the train state's parameter keys.
    try:
        params_leaves, _ = jax.tree_util.tree_flatten(appended.base_model.train_state.params)
        logger.info(f"Number of parameter leaves in the train state: {len(params_leaves)}")
    except Exception as e:
        logger.error(f"Error while accessing train state parameters: {e}")
    
    logger.info(f"{type(appended.base_model.train_state.params['params']['mlp0']['kernel'])}")
    # Test setting a pool mask: here we set pool index 0.
    appended.set_pool_mask(pool_idx=[0], eval=False)
    logger.info("Pool mask updated for training mode.\n")
    
    # Run the eval_model test with decoder_id as (B, 1)
    test_eval_model_with_2d_decoder_id(appended)
    
    logger.info("Test completed for ModelAppender with (B, 1, F) and (B, 1) inputs.")
    
    # Test ModelAppenderV2
    test_model_appender_v2()