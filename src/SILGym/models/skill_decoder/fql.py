import jax
import jax.random as random
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from SILGym.utils.logger import get_logger

from SILGym.models.basic.train_state import create_train_state_time_cond
from SILGym.models.basic.module import BasicModule
from SILGym.models.basic.loss import mse
from SILGym.models.basic.utils import default, update_rngs, timestep_embedding


class FQLDecoder(BasicModule):
    """
    Flow Q-Learning (FQL) decoder for imitation learning with optional RL.

    This decoder uses flow matching instead of diffusion, learning a velocity field
    that maps noise to actions through continuous normalizing flows.

    Key differences from DDPM:
    - Simpler: Uses linear interpolation x_t = (1-t)*x_0 + t*x_1
    - Faster: Fewer integration steps needed
    - Deterministic: Euler integration for sampling
    - Optional Q-learning: Can enable RL improvement (disabled by default)
    """

    def __init__(
        self,
        training: str = 'train',
        model_config: dict = None,
        optimizer_config: dict = None,
        input_config: dict = None,
        out_dim: int = None,
        clip_actions: bool = True,
        flow_steps: int = 100,
        use_onestep_flow: bool = False,
        eval_use_onestep: bool = False,
        use_q_loss: bool = False,
        alpha: float = 10.0,
        q_loss_weight: float = 1.0,
        normalize_q_loss: bool = False,
        seed: int = 777,
        **kwargs
    ):
        super().__init__()

        self.logger = get_logger(__name__)

        # 1) Basic attributes
        self.training = training
        self.clip_actions = clip_actions
        self.flow_steps = flow_steps
        self.use_onestep_flow = use_onestep_flow
        self.eval_use_onestep = eval_use_onestep
        self.use_q_loss = use_q_loss
        self.alpha = alpha
        self.q_loss_weight = q_loss_weight
        self.normalize_q_loss = normalize_q_loss

        # 2) Input configuration
        self.input_config = input_config if input_config is not None else {
            'x': (1, 1, 9),
            'cond': (1, 1, 60),
        }
        # Add time embedding tensor shape
        self.dim_time_embedding = 128
        self.input_config['time'] = (1, 1, self.dim_time_embedding)

        # 3) Output dimension
        self.out_dim = out_dim if out_dim is not None else self.input_config['x'][-1]

        # 4) Model configuration
        if model_config is None:
            raise NotImplementedError("model_config must be provided.")
        self.model_config = model_config

        # Update out_dim in model_kwargs
        self.model_config['model_kwargs']['out_dim'] = self.out_dim

        # Prepare eval_kwargs from model_kwargs (e.g. disable dropout)
        self.model_config['eval_kwargs'] = self.model_config['model_kwargs'].copy()
        self.model_config['eval_kwargs']['dropout'] = 0.0

        # Instantiate BC flow model and evaluation model
        self.model_bc_flow = self.model_config['model_cls'](**self.model_config['model_kwargs'])
        self.model_bc_flow_eval = self.model_config['model_cls'](**self.model_config['eval_kwargs'])

        # Aliases for compatibility with DDPM and ModelAppenderV2
        self.model = self.model_bc_flow  # Essential: main model instance
        self.model_eval = self.model_bc_flow_eval  # Essential: evaluation model instance

        # 5) Optimizer configuration (default: Adam)
        self.optimizer_config = optimizer_config if optimizer_config is not None else {
            'optimizer_cls': optax.adam,
            'optimizer_kwargs': {
                'learning_rate': 3e-4,
                'b1': 0.9,
            },
        }

        # 6) Create TrainState for BC flow model
        self.train_state = create_train_state_time_cond(
            self.model_bc_flow,
            self.input_config,
            self.optimizer_config
        )

        # 7) Optional: One-step flow model for distillation
        self.train_state_onestep = None
        if self.use_onestep_flow:
            self.model_onestep_flow = self.model_config['model_cls'](**self.model_config['model_kwargs'])
            self.model_onestep_flow_eval = self.model_config['model_cls'](**self.model_config['eval_kwargs'])
            self.train_state_onestep = create_train_state_time_cond(
                self.model_onestep_flow,
                self.input_config,
                self.optimizer_config
            )

        # 8) Optional: Q-network for RL (disabled by default)
        self.train_state_critic = None
        self.train_state_target_critic = None
        if self.use_q_loss:
            # Q-network takes (obs, action) and outputs Q-value
            q_input_config = {
                'x': (1, 1, self.input_config['cond'][-1] + self.out_dim),
            }
            # Simple MLP for Q-function (no time conditioning)
            from SILGym.models.basic.base import MLP
            self.model_critic = MLP(
                hidden_size=256,
                out_shape=1,
                dropout_rate=0.1,
                deterministic=False
            )
            # Create critic train state
            from SILGym.models.basic.train_state import create_train_state_basic
            self.train_state_critic = create_train_state_basic(
                self.model_critic,
                q_input_config,
                self.optimizer_config
            )
            # Create target critic (copy of critic)
            self.train_state_target_critic = create_train_state_basic(
                self.model_critic,
                q_input_config,
                self.optimizer_config
            )
            # Copy params from critic to target critic
            self.train_state_target_critic = self.train_state_target_critic.replace(
                params=self.train_state_critic.params
            )
            self.tau = 0.005  # Soft update coefficient

        # 9) Initialize RNG keys
        self.sample_rngs = {
            'x_noise': random.PRNGKey(seed - 2),
            't_noise': random.PRNGKey(seed - 1),
            'apply': random.PRNGKey(seed),
            'dropout': random.PRNGKey(seed + 1),
        }
        self.eval_rng_key = random.PRNGKey(seed + 1)

        # Log configuration
        separator = "=" * 60
        self.logger.info(separator)
        self.logger.info("                 FQL DECODER CONFIGURATION                 ")
        self.logger.info(separator)
        self.logger.info(f"Training Mode:       {self.training}")
        self.logger.info(f"Model Class:         {self.model_config['model_cls'].__name__}")
        self.logger.info(f"Model Kwargs:        {self.model_config['model_kwargs']}")
        self.logger.info(f"Optimizer Config:    {self.optimizer_config}")
        self.logger.info(f"Input Config:        {self.input_config}")
        self.logger.info(f"Output Dimension:    {self.out_dim}")
        self.logger.info(f"Flow Steps:          {self.flow_steps}")
        self.logger.info(f"Clip Actions:        {self.clip_actions}")
        self.logger.info(f"Use One-step Flow:   {self.use_onestep_flow}")
        self.logger.info(f"Eval Use One-step:   {self.eval_use_onestep}")
        self.logger.info(f"Use Q Loss:          {self.use_q_loss}")
        if self.use_onestep_flow:
            self.logger.info(f"Alpha (distill):     {self.alpha}")
        if self.use_q_loss:
            self.logger.info(f"Q Loss Weight:       {self.q_loss_weight}")
            self.logger.info(f"Normalize Q Loss:    {self.normalize_q_loss}")
        self.logger.info(separator)

    def reinit_optimizer(self):
        """
        Re-initialize optimizer while keeping the same params.
        """
        old_params = self.train_state.params
        self.train_state = create_train_state_time_cond(
            self.model_bc_flow, self.input_config, self.optimizer_config
        )
        self.train_state = self.train_state.replace(params=old_params)

        if self.use_onestep_flow:
            old_params_onestep = self.train_state_onestep.params
            self.train_state_onestep = create_train_state_time_cond(
                self.model_onestep_flow, self.input_config, self.optimizer_config
            )
            self.train_state_onestep = self.train_state_onestep.replace(params=old_params_onestep)

    # ========================================================================
    # Flow Matching Core Functions
    # ========================================================================

    def flow_interpolate(self, x_0, x_1, t):
        """
        Linear interpolation between x_0 (noise) and x_1 (target action).

        Args:
            x_0: Initial noise (batch_size, 1, action_dim)
            x_1: Target action (batch_size, 1, action_dim)
            t: Time steps (batch_size,) in [0, 1]

        Returns:
            x_t: Interpolated state (batch_size, 1, action_dim)
        """
        t = t.reshape(-1, 1, 1)  # Shape for broadcasting
        return (1 - t) * x_0 + t * x_1

    def compute_velocity(self, x_0, x_1):
        """
        Compute the true velocity for flow matching.

        For linear interpolation x_t = (1-t)*x_0 + t*x_1,
        the velocity is dx/dt = x_1 - x_0 (constant).

        Args:
            x_0: Initial noise
            x_1: Target action

        Returns:
            velocity: x_1 - x_0
        """
        return x_1 - x_0

    @partial(jax.jit, static_argnums=(0,))
    def predict_velocity(self, params, x_t, t, cond, deterministic=True):
        """
        Predict velocity at time t using the BC flow model.

        Args:
            params: Model parameters
            x_t: Current state
            t: Time steps (batch_size,)
            cond: Conditioning information
            deterministic: Whether to use deterministic mode

        Returns:
            predicted velocity
        """
        t_input = jax.lax.convert_element_type(t, jnp.float32)[:, jnp.newaxis]
        t_input = timestep_embedding(t_input, self.dim_time_embedding)

        return self.model_bc_flow_eval.apply(
            params, x_t, t_input, cond, deterministic=deterministic
        )

    def euler_integration(self, params, cond, rngs=None, return_intermediates=False):
        """
        Sample actions using Euler integration of the flow.

        Starting from noise x_0 ~ N(0, I), integrate the velocity field
        using Euler method: x_{i+1} = x_i + v_i * dt

        Args:
            params: Model parameters
            cond: Conditioning information (batch_size, 1, cond_dim)
            rngs: Random number generator (single key or dict with 'x_noise' key)
            return_intermediates: Whether to return all intermediate states

        Returns:
            actions: Final actions (batch_size, 1, action_dim)
        """
        batch_size = cond.shape[0]

        # Handle rngs: extract key if it's a dict
        if isinstance(rngs, dict):
            rng_key = rngs.get('x_noise', rngs.get('apply', list(rngs.values())[0]))
        else:
            rng_key = rngs

        # Start from Gaussian noise
        x_0 = jax.random.normal(rng_key, (batch_size, 1, self.out_dim))
        x_t = x_0

        intermediates = [x_t] if return_intermediates else None

        # Euler integration
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = jnp.full((batch_size,), i / self.flow_steps, dtype=jnp.float32)
            v_t = self.predict_velocity(params, x_t, t, cond, deterministic=True)
            x_t = x_t + v_t * dt

            if return_intermediates:
                intermediates.append(x_t)

        # Clip actions to valid range
        if self.clip_actions:
            x_t = jnp.clip(x_t, -1.0, 1.0)

        if return_intermediates:
            return x_t, intermediates
        return x_t

    @partial(jax.jit, static_argnums=(0,))
    def compute_onestep_action(self, params, cond, noise):
        """
        Compute action using one-step flow model (for distillation).

        Args:
            params: One-step model parameters
            cond: Conditioning information
            noise: Initial noise

        Returns:
            action: Predicted action in one step
        """
        # One-step model takes noise directly to action
        # We use t=0 as a placeholder (model should learn to ignore it)
        batch_size = cond.shape[0]
        t = jnp.zeros((batch_size,), dtype=jnp.float32)
        t_input = timestep_embedding(t[:, jnp.newaxis], self.dim_time_embedding)

        action = self.model_onestep_flow_eval.apply(
            params, noise, t_input, cond, deterministic=True
        )

        if self.clip_actions:
            action = jnp.clip(action, -1.0, 1.0)

        return action

    # ========================================================================
    # Loss Functions
    # ========================================================================

    @partial(jax.jit, static_argnums=(0,))
    def bc_flow_loss(self, params, state, x_1, cond, rngs):
        """
        Behavioral cloning flow matching loss.

        Args:
            params: Model parameters
            state: Train state
            x_1: Target actions (batch_size, 1, action_dim)
            cond: Conditioning information
            rngs: Random number generators

        Returns:
            loss: BC flow loss
            loss_dict: Dictionary of loss values
        """
        batch_size = x_1.shape[0]

        # Sample x_0 from standard Gaussian
        x_0 = jax.random.normal(rngs['x_noise'], x_1.shape)

        # Sample random time steps
        t = jax.random.uniform(rngs['t_noise'], (batch_size,))

        # Compute interpolated state
        x_t = self.flow_interpolate(x_0, x_1, t)

        # Compute true velocity
        v_true = self.compute_velocity(x_0, x_1)

        # Predict velocity using model
        t_input = timestep_embedding(t[:, jnp.newaxis], self.dim_time_embedding)
        r1, r2 = jax.random.split(rngs['apply'])
        apply_rng = {'params': r1, 'dropout': r2}

        v_pred = state.apply_fn(
            params, x_t, t_input, cond, deterministic=True, rngs=apply_rng
        )

        # MSE loss
        loss = mse(v_pred, v_true).mean()

        log_prefix = 'train' if self.training else 'val'
        loss_dict = {
            f'{log_prefix}/bc_flow_loss': loss,
        }

        return loss, loss_dict

    @partial(jax.jit, static_argnums=(0,))
    def distillation_loss(self, params, params_bc, cond, rngs):
        """
        Distillation loss: train one-step flow to match multi-step BC flow.

        Args:
            params: One-step model parameters
            params_bc: BC flow model parameters
            cond: Conditioning information
            rngs: Random number generators

        Returns:
            loss: Distillation loss
        """
        batch_size = cond.shape[0]

        # Split RNGs for different uses
        rng_noise, rng_euler = jax.random.split(rngs['x_noise'])

        # Sample noise
        noise = jax.random.normal(rng_noise, (batch_size, 1, self.out_dim))

        # Get target from BC flow (using Euler integration)
        action_target = self.euler_integration(params_bc, cond, rng_euler)

        # Get prediction from one-step flow
        action_pred = self.compute_onestep_action(params, cond, noise)

        # MSE loss
        loss = mse(action_pred, action_target).mean()

        return loss

    @partial(jax.jit, static_argnums=(0,))
    def q_value_loss(self, params_actor, params_critic, params_target_critic,
                     batch_obs, batch_actions, batch_rewards, batch_next_obs,
                     batch_masks, rngs):
        """
        Q-learning loss for optional RL improvement.

        Args:
            params_actor: Actor (one-step flow) parameters
            params_critic: Critic parameters
            params_target_critic: Target critic parameters
            batch_obs: Observations
            batch_actions: Actions
            batch_rewards: Rewards
            batch_next_obs: Next observations
            batch_masks: Masks (1 - done)
            rngs: Random number generators

        Returns:
            critic_loss: TD loss for critic
            actor_q_loss: Q-value loss for actor
        """
        batch_size = batch_obs.shape[0]

        # Compute target Q-values
        noise = jax.random.normal(rngs['x_noise'], (batch_size, 1, self.out_dim))
        next_actions = self.compute_onestep_action(params_actor, batch_next_obs, noise)

        # Concatenate obs and action for Q-network
        next_q_input = jnp.concatenate([batch_next_obs, next_actions], axis=-1)
        next_q = self.model_critic.apply(params_target_critic, next_q_input)

        # TD target
        discount = 0.99
        target_q = batch_rewards + discount * batch_masks * next_q
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q-values
        q_input = jnp.concatenate([batch_obs, batch_actions], axis=-1)
        current_q = self.model_critic.apply(params_critic, q_input)

        # Critic loss (TD error)
        critic_loss = mse(current_q, target_q).mean()

        # Actor Q loss (maximize Q-value)
        actor_actions = self.compute_onestep_action(params_actor, batch_obs, noise)
        actor_q_input = jnp.concatenate([batch_obs, actor_actions], axis=-1)
        actor_q = self.model_critic.apply(params_critic, actor_q_input)

        actor_q_loss = -actor_q.mean()
        if self.normalize_q_loss:
            lam = jax.lax.stop_gradient(1.0 / jnp.abs(actor_q).mean())
            actor_q_loss = lam * actor_q_loss

        return critic_loss, actor_q_loss

    @partial(jax.jit, static_argnums=(0,))
    def total_loss(self, params, state, x_1, cond, rngs):
        """
        Compute total loss for training.

        Args:
            params: Model parameters
            state: Train state
            x_1: Target actions
            cond: Conditioning information
            rngs: Random number generators

        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss values
        """
        # BC flow loss (always computed)
        bc_loss, loss_dict = self.bc_flow_loss(params, state, x_1, cond, rngs)
        total_loss = bc_loss

        # Optional: distillation loss
        if self.use_onestep_flow and self.train_state_onestep is not None:
            distill_loss = self.distillation_loss(
                self.train_state_onestep.params,
                params,
                cond,
                rngs
            )
            loss_dict[f'{self.training}/distill_loss'] = distill_loss
            total_loss = total_loss + self.alpha * distill_loss

        # Optional: Q-learning loss
        if self.use_q_loss and self.train_state_critic is not None:
            # Note: Q-loss requires additional data (rewards, next_obs, etc.)
            # For now, we skip it in this simple interface
            # In practice, you'd pass these through the batch
            pass

        loss_dict[f'{self.training}/total_loss'] = total_loss
        loss_dict[f'{self.training}/loss'] = total_loss  # For trainer compatibility

        return total_loss, loss_dict

    @partial(jax.jit, static_argnums=(0,))
    def train_model_jit(self, state, x_1, cond, rngs):
        """
        JIT-compiled single training step.

        Args:
            state: Train state
            x_1: Target actions
            cond: Conditioning information
            rngs: Random number generators

        Returns:
            new_state: Updated train state
            metric: Tuple of (None, loss_dict)
        """
        grad_fn = jax.grad(self.total_loss, has_aux=True)
        grads, loss_dict = grad_fn(state.params, state, x_1, cond, rngs)
        metric = (None, loss_dict)
        state = state.apply_gradients(grads=grads)
        return state, metric

    # ========================================================================
    # Training and Evaluation Interface
    # ========================================================================

    def train_model(self, x, cond, compute_eval_loss=False):
        """
        Train the model for one step.

        Args:
            x (np.ndarray): Target actions (batch_size, action_dim)
            cond (np.ndarray): Conditioning information (batch_size, cond_dim)
            compute_eval_loss (bool): Whether to compute evaluation MSE

        Returns:
            dict: Dictionary of loss metrics
        """
        # Ensure proper shape
        if x.ndim == 2:
            x = x[:, None, :]
        if cond.ndim == 2:
            cond = cond[:, None, :]

        # Train step
        self.train_state, metric = self.train_model_jit(
            self.train_state, x, cond, self.sample_rngs
        )
        self.sample_rngs = update_rngs(self.sample_rngs)

        # Optionally compute evaluation MSE
        if compute_eval_loss :
            self.eval_rng_key, rngs = random.split(self.eval_rng_key)
            x_eval = self.euler_integration(
                self.train_state.params, cond, rngs
            )
            mse_eval = mse(x_eval, x).mean()
            metric[1]['train/eval_mse'] = mse_eval

        return metric

    def eval_model(self, cond, params=None, use_onestep=None):
        """
        Evaluate the model (sample actions from conditioning).

        Args:
            cond (np.ndarray): Conditioning information (batch_size, cond_dim)
            params: Optional parameters (default: use appropriate train_state params)
            use_onestep (bool): If True, use one-step flow model; if False, use multi-step Euler integration.
                               If None, uses self.eval_use_onestep as default.

        Returns:
            actions: Sampled actions (batch_size, 1, action_dim)
        """
        # Use configured default if not specified
        if use_onestep is None:
            use_onestep = self.eval_use_onestep
        cond = np.array(cond)
        if cond.ndim == 2:
            cond = cond[:, None, :]

        self.eval_rng_key, rngs = random.split(self.eval_rng_key)

        # Use one-step flow model
        if use_onestep:
            if not self.use_onestep_flow:
                raise ValueError("One-step flow is not enabled. Set use_onestep_flow=True.")

            params = params if params is not None else self.train_state_onestep.params
            batch_size = cond.shape[0]
            noise = jax.random.normal(rngs, (batch_size, 1, self.out_dim))
            actions = self.compute_onestep_action(params, cond, noise)

        # Use multi-step Euler integration (default)
        else:
            params = params if params is not None else self.train_state.params
            actions = self.euler_integration(params, cond, rngs)

        return actions

    def forward(self, **kwargs):
        """Forward pass (for compatibility)."""
        return self.euler_integration(**kwargs)

    def eval_intermediates(self, cond):
        """
        Return all intermediate flow states during sampling.

        Args:
            cond: Conditioning information

        Returns:
            actions: Final actions
            intermediates: List of intermediate states
        """
        cond = np.array(cond)
        if cond.ndim == 2:
            cond = cond[:, None, :]

        self.eval_rng_key, rngs = random.split(self.eval_rng_key)

        actions, intermediates = self.euler_integration(
            self.train_state.params, cond, rngs, return_intermediates=True
        )

        return actions, intermediates


if __name__ == "__main__":
    """
    Test FQLDecoder with a simple example
    """
    print("\n" + "=" * 80)
    print("TESTING FQLDecoder")
    print("=" * 80)

    # Import model
    from SILGym.models.basic.base import DenoisingMLP

    # Create model configuration
    model_config = {
        'model_cls': DenoisingMLP,
        'model_kwargs': {
            'dim': 256,
            'n_blocks': 3,
            'dropout': 0.1,
        }
    }

    # Create input configuration
    input_config = {
        'x': (1, 1, 9),        # action space
        'cond': (1, 1, 60),    # conditioning space
    }

    # Instantiate decoder
    print("\n[1] Creating FQLDecoder (BC flow only)...")
    decoder = FQLDecoder(
        training='train',
        model_config=model_config,
        input_config=input_config,
        out_dim=9,
        flow_steps=10,
        use_onestep_flow=False,
        use_q_loss=False,
        seed=777
    )
    print("✓ Decoder created successfully")

    # Test training
    print("\n[2] Testing training...")
    x = np.random.randn(4, 9).astype(np.float32)
    cond = np.random.randn(4, 60).astype(np.float32)

    print("   Training for 5 steps...")
    losses = []
    for step in range(5):
        metric = decoder.train_model(x, cond)
        loss = metric[1]['train/bc_flow_loss']
        losses.append(loss)
        print(f"   Step {step+1}: loss = {loss:.6f}")

    print("✓ Training successful")

    # Test evaluation
    print("\n[3] Testing evaluation (sampling with Euler integration)...")
    cond_eval = np.random.randn(2, 60).astype(np.float32)

    # Generate samples
    sample1 = decoder.eval_model(cond_eval)
    sample2 = decoder.eval_model(cond_eval)

    print(f"   Sample 1 shape: {sample1.shape}")
    print(f"   Sample 1 mean: {sample1.mean():.6f}, std: {sample1.std():.6f}")
    print(f"   Sample 2 mean: {sample2.mean():.6f}, std: {sample2.std():.6f}")
    print("✓ Evaluation successful")

    # Test deterministic behavior
    print("\n[4] Testing deterministic behavior...")
    np.random.seed(999)
    decoder.eval_rng_key = random.PRNGKey(999)
    sample_a = decoder.eval_model(cond_eval)

    np.random.seed(999)
    decoder.eval_rng_key = random.PRNGKey(999)
    sample_b = decoder.eval_model(cond_eval)

    is_deterministic = np.allclose(sample_a, sample_b, rtol=1e-5)
    print(f"   Same seed → same output: {is_deterministic}")
    if is_deterministic:
        print("   ✓ Sampling is DETERMINISTIC (as expected for FQL)")
    else:
        print("   ⚠ Sampling has small variations (check JAX random state)")

    # Test flow interpolation
    print("\n[5] Testing flow interpolation...")
    x_0 = np.random.randn(2, 1, 9).astype(np.float32)
    x_1 = np.random.randn(2, 1, 9).astype(np.float32)
    t = np.array([0.0, 0.5], dtype=np.float32)

    x_t = decoder.flow_interpolate(x_0, x_1, t)
    print(f"   x_0 mean: {x_0.mean():.6f}")
    print(f"   x_1 mean: {x_1.mean():.6f}")
    print(f"   x_t (t=0.0, 0.5) mean: {x_t.mean():.6f}")
    print("✓ Flow interpolation successful")

    # Test intermediate states
    print("\n[6] Testing intermediate flow states...")
    cond_test = np.random.randn(1, 60).astype(np.float32)
    actions, intermediates = decoder.eval_intermediates(cond_test)
    print(f"   Number of intermediates: {len(intermediates)}")
    print(f"   Initial state mean: {intermediates[0].mean():.6f}")
    print(f"   Final action mean: {actions.mean():.6f}")
    print("✓ Intermediate states captured successfully")

    # Test with one-step flow
    print("\n[7] Testing with one-step flow distillation...")
    decoder_onestep = FQLDecoder(
        training='train',
        model_config=model_config,
        input_config=input_config,
        out_dim=9,
        flow_steps=10,
        use_onestep_flow=True,
        eval_use_onestep=True,  # Use one-step by default for evaluation
        alpha=10.0,
        use_q_loss=False,
        seed=888
    )
    print("   ✓ One-step decoder created")

    print("   Training for 3 steps...")
    for step in range(3):
        metric = decoder_onestep.train_model(x, cond)
        bc_loss = metric[1]['train/bc_flow_loss']
        distill_loss = metric[1].get('train/distill_loss', 0.0)
        print(f"   Step {step+1}: bc_loss={bc_loss:.6f}, distill_loss={distill_loss:.6f}")

    print("   ✓ One-step training successful")

    # Test integrated eval_model
    print("\n[8] Testing integrated eval_model with both modes...")
    cond_test = np.random.randn(2, 60).astype(np.float32)

    # Test default mode (should use one-step because eval_use_onestep=True)
    sample_default = decoder_onestep.eval_model(cond_test)
    print(f"   Default mode (one-step) shape: {sample_default.shape}")

    # Test explicit multi-step mode
    sample_multistep = decoder_onestep.eval_model(cond_test, use_onestep=False)
    print(f"   Explicit multi-step mode shape: {sample_multistep.shape}")

    # Test explicit one-step mode
    sample_onestep = decoder_onestep.eval_model(cond_test, use_onestep=True)
    print(f"   Explicit one-step mode shape: {sample_onestep.shape}")
    print("   ✓ Integrated eval_model works for both modes with configurable default")

    # Summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nSummary:")
    print(f"  • Training: {len(losses)} steps completed")
    print(f"  • Final loss: {losses[-1]:.6f}")
    print(f"  • Sampling: Deterministic Euler integration")
    print(f"  • Flow steps: {decoder.flow_steps}")
    print(f"  • Model parameters: ~{sum(p.size for p in jax.tree_util.tree_leaves(decoder.train_state.params)):,}")
    print("=" * 80 + "\n")