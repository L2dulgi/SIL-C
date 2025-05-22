
from AppOSI.models.skill_decoder.diffusion_base import CondDDPMDecoder
from lorax.constants import LORA_FULL, LORA_FREEZE
import lorax
import jax
import jax.numpy as jnp
import jax.tree_util
from flax.training import train_state
import optax
import qax
import numpy as np
from dataclasses import dataclass, field, fields, is_dataclass
from einops import rearrange, repeat

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
        # delta_w = jnp.sum(masked_lora_pool, axis=0) / jnp.maximum(jnp.sum(self.pool_mask), 1)
        delta_w = jnp.sum(masked_lora_pool, axis=0) / jnp.sum(self.pool_mask)
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


from qax.utils import freeze_subtrees
# Overwrite the freeze_keys function to add '.' exception for jax 0.4.34 treeutil update
# jax/_src/tree_util.py GetAttrKey
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
        return jax.tree_map(label_leaf, root, is_leaf=lambda x: isinstance(x, arr_type))

    return freeze_subtrees(optimizer, label_fn, use_scalar_zeros=use_scalar_zeros)


class ModelAppender() :
    def __init__(
            self,
            base_model : CondDDPMDecoder = None,
            append_config : AppendConfig = AppendConfig(),
        ) :

        self.base_model = base_model
        self.append_config = append_config
        self.lora_optimizer_config = None  
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
        full_freeze_labels = jax.tree_map(
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
                print(f'Fully finetuning param {path}')
                return LORA_FULL
            dim = self.append_config.lora_dim
            print(f'Using LoRA with dim={dim} for param {path}')
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

class ModelAppenderV2(ModelAppender) :
    '''
    instead of set_pool_mask it replace the lora params with the new params. and opt_state
    '''

    def create_task_vector(
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
                    'w' : None, # just add the extra lora params.
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
  
def test_eval_model_with_2d_decoder_id(appended):
    """
    Test the eval_model method when:
      - The condition input has shape (batch_size, 1, feature_dim)
      - The decoder_id input has shape (batch_size, 1)
    """
    print("=== Running eval_model Test with decoder_id as (B, 1) ===")
    
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
    print("Test eval_model with 2D decoder_id output:\n", result)
    # except Exception as e:
    #     print("eval_model test with 2D decoder_id raised an exception:", e)


if __name__ == "__main__":
    from AppOSI.models.basic.base import DenoisingMLP
    from AppOSI.config.experiment_config import DEFAULT_DECODER_CONFIG
    from AppOSI.models.skill_decoder.diffusion_base import CondDDPMDecoder

    # Create the base model using the default configuration.
    base_model = DEFAULT_DECODER_CONFIG["model_cls"](
            **DEFAULT_DECODER_CONFIG["model_kwargs"]
        )
    


    # Instantiate the ModelAppender with the base model.
    append_config = AppendConfig(lora_dim=4, pool_length=10)
    appended = ModelAppender(base_model=base_model, append_config=append_config)
    print("ModelAppender created successfully.\n")
    
    # Print a summary of the train state's parameter keys.
    try:
        params_leaves, _ = jax.tree_util.tree_flatten(appended.base_model.train_state.params)
        print(f"Number of parameter leaves in the train state: {len(params_leaves)}")
    except Exception as e:
        print("Error while accessing train state parameters:", e)
    
    print(type(appended.base_model.train_state.params['params']['mlp0']['kernel']))
    # Test setting a pool mask: here we set pool index 0.
    appended.set_pool_mask(pool_idx=[0], eval=False)
    print("Pool mask updated for training mode.\n")
    
    # Run the eval_model test with decoder_id as (B, 1)
    test_eval_model_with_2d_decoder_id(appended)
    
    print("Test completed for ModelAppender with (B, 1, F) and (B, 1) inputs.")