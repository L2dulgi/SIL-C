from functools import partial
import jax
from jax import random
import optax

from SILGym.models.basic.module import BasicModule
from SILGym.models.basic.base import MLP
from SILGym.models.basic.train_state import create_train_state_basic
from SILGym.models.basic.loss import mse
from SILGym.models.basic.utils import update_rngs
import jax.numpy as jnp
import numpy as np

class MLPPolicy(BasicModule):
    """
    A simple MLP-based policy class example (Flax + Optax).
    """

    def __init__(
        self,
        mode="train",
        model_config=None,
        input_config=None,
        optimizer_config=None,
        stochastic=False,
    ) -> None:
        super().__init__()

        # Default model settings
        self.model_config = {
            "hidden_size": 512,
            "out_shape": 9,
            "num_hidden_layers": 4,
            "dropout": 0.0,
        }
        if model_config is not None:
            self.model_config.update(model_config)

        # If set to stochastic mode, double the out_shape
        self.stochastic = stochastic
        if self.stochastic:
            self.model_config["out_shape"] *= 2

        # Default optimizer settings
        self.optimizer_config = (
            optimizer_config
            if optimizer_config is not None
            else {
                "optimizer_cls": optax.adam,
                "optimizer_kwargs": {"learning_rate": 5e-5, "b1": 0.9},
            }
        )

        # Initialize random keys
        seed = 777
        self.sample_rngs = {
            "p_noise": random.PRNGKey(seed - 2),
            "q_noise": random.PRNGKey(seed - 1),
            "apply": random.PRNGKey(seed),
            "dropout": random.PRNGKey(seed + 99),
        }
        self.eval_rng_key = random.PRNGKey(seed + 1)

        # Create model and evaluation model
        self.model = MLP(
            hidden_size=self.model_config["hidden_size"],
            out_shape=self.model_config["out_shape"],
            dropout_rate=self.model_config["dropout"],
            deterministic=False,
        )
        self.model_eval = MLP(
            hidden_size=self.model_config["hidden_size"],
            out_shape=self.model_config["out_shape"],
            dropout_rate=self.model_config["dropout"],
            deterministic=True,
        )

        # Create training state
        self.train_state = create_train_state_basic(
            self.model,
            input_config=input_config,
            optimizer_config=self.optimizer_config,
        )

    def forward(self, params, x, rngs=None):
        """Forward pass using the MLP model."""
        return self.model.apply(params, x, rngs=rngs)

    def loss_fn(self, params, state, batch, rngs=None):
        """Loss function (MSE)."""
        logits = state.apply_fn(params, batch["inputs"], rngs=rngs)
        loss = jnp.mean(mse(logits, batch["labels"]))

        return loss, None

    @partial(jax.jit, static_argnums=(0,))
    def train_model_jit(self, state, batch, rngs=None):
        """
        JIT-compiled training step.
        1. Compute gradients.
        2. Update the train state.
        3. Compute and return the metric (loss).
        """
        grad_fn = jax.grad(self.loss_fn, has_aux=True)
        grads, _ = grad_fn(state.params, state, batch, rngs=rngs)
        new_state = state.apply_gradients(grads=grads)

        metric, _ = self.loss_fn(new_state.params, new_state, batch, rngs=rngs)
        return new_state, metric

    def train_model(self, batch):
        """
        Trains on a single batch of data and returns the loss (metric).
        Also updates RNG keys.
        """
        self.train_state, metric = self.train_model_jit(
            self.train_state, batch, rngs=self.sample_rngs
        )
        self.sample_rngs = update_rngs(self.sample_rngs)
        return metric

    @partial(jax.jit, static_argnums=(0,))
    def eval_model_jit(self, state, x, rngs=None):
        """JIT-compiled evaluation routine."""
        return self.model_eval.apply(state.params, x, rngs=rngs)

    def eval_model(self, x):
        """
        Performs evaluation with the current train state.
        Returns the model output.
        """
        self.eval_rng_key, eval_rng = random.split(self.eval_rng_key)
        return self.eval_model_jit(self.train_state, x, rngs=eval_rng)

class HighLevelPolicy(MLPPolicy):
    """
    HighLevelPolicy extends MLPPolicy to implement a policy suitable for PTGM.
    In this policy, the labels represent an integer skill ID.
    The loss function is modified to use softmax cross-entropy loss for training,
    and the evaluation function returns the predicted skill ID.
    """
    def loss_fn(self, params, state, batch, rngs=None):
        logits = state.apply_fn(params, batch["inputs"], rngs=rngs)
        # Check if labels are integer type (or if they are 1D, indicating indices)
        if jnp.issubdtype(batch["labels"].dtype, jnp.integer) or batch["labels"].ndim == 1:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
        else:
            # Assume labels are provided as soft labels (e.g. logits/probabilities)
            loss = optax.softmax_cross_entropy(logits, batch["labels"])
        loss = jnp.mean(loss)
        return loss, None

    @partial(jax.jit, static_argnums=(0,))  
    def _argmax(self, logits):
        """
        Computes the argmax over the logits.
        This is used to determine the predicted skill ID.
        """
        # If stochastic, split the logits into two parts
        return jnp.argmax(logits, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def eval_model_jit(self, params, x, rngs=None):
        """
        JIT-compiled evaluation routine for HighLevelPolicy.
        Returns the predicted skill ID by taking the argmax over the logits.
        """
        logits = self.model_eval.apply(params, x, rngs=rngs)
        # Return the predicted skill id (argmax over the last dimension)
        eval_rng, _ = random.split(rngs)
        return logits, eval_rng

    def eval_model(self, x, cut_off=None):
        """
        Performs evaluation with the current train state.
        If cut_off is provided, logits beyond cut_off are replaced with -jnp.inf,
        so that argmax is computed over the first `cut_off` elements only.
        Returns the predicted skill ID.
        """
        logits, self.eval_rng_key = self.eval_model_jit(
                self.train_state.params, 
                x, 
                rngs=self.eval_rng_key,
        )
        logits.at[..., cut_off:].set(-jnp.inf) if cut_off is not None else logits
        pred_id = self._argmax(logits)
        return pred_id


from copy import deepcopy
class SILCHighLevelPolicy(HighLevelPolicy):
    """
    SILCHighLevelPolicy extends HighLevelPolicy to include a hook for SILC implementation.
    """
    def __init__(
        self,
        mode="train",
        model_config=None,
        input_config=None,
        optimizer_config=None,
    ) -> None :
        super().__init__(
            mode=mode,
            model_config=model_config,
            input_config=input_config,
            optimizer_config=optimizer_config,
        )

        # skill hooks
        self.subtask_prototypes = None # list(PolicyPrototype) 

    def set_subtask_prototype(self, subtask_prototypes):
        """
        Set the subtask space prototypes SIL-C.
        """
        self.subtask_prototypes = deepcopy(subtask_prototypes)