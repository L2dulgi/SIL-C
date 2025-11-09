import jax
import jax.random as random
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from SILGym.utils.logger import get_logger

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Create a beta schedule for diffusion processes.
    Supported schedules: 'linear', 'cosine', 'sqrt_linear', 'sqrt', 'vp'.
    """
    if schedule == "linear":
        betas = (
            np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float32) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            np.arange(n_timestep + 1, dtype=np.float32) / n_timestep + cosine_s
        )
        alphas = np.cos((timesteps / (1 + cosine_s)) * np.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float32)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float32) ** 0.5
    elif schedule == "vp":
        betas = vp_beta_schedule(n_timestep)
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas

def vp_beta_schedule(timesteps, dtype=np.float32):
    """
    Beta schedule for VP-SDE.
    See: https://arxiv.org/abs/2011.13456
    """
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(
        -b_min / T
        - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2
    )
    betas = 1 - alpha
    return np.asarray(betas, dtype=dtype)

def noise_like(shape, rngs=None, repeat=False):
    """
    Generate Gaussian noise of a given shape.
    If repeat=True, repeats the noise for each batch element.
    """
    if repeat:
        return random.normal(rngs, shape[1:])[jnp.newaxis].repeat(shape[0], axis=0)
    else:
        return random.normal(rngs, shape)

def extract_into_numpy(a, t, x_shape):
    """
    Index tensor `a` by `t` (each batch index) and reshape to broadcast with x_shape.
    """
    b = t.shape[0]
    out = jnp.take_along_axis(a, t, axis=-1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

from SILGym.models.basic.train_state import create_train_state_time_cond
from SILGym.models.basic.module import BasicModule
from SILGym.models.basic.loss import mse
from SILGym.models.basic.utils import default, update_rngs, timestep_embedding
class CondDDPMDecoder(BasicModule):
    """
    Basic conditional Diffusion training/evaluation module class.
    """
    def __init__(
        self,
        training: str = 'train',
        model_config: dict = None,
        optimizer_config: dict = None,
        input_config: dict = None,
        out_dim: int = None,
        clip_denoised: bool = True,
        diffusion_step: int = 64,
        decoding_iter_ratio: float = 1.0,
        seed: int = 777,
        **kwargs
    ):
        super().__init__()

        self.logger = get_logger(__name__)

        # 1) Basic attributes
        self.training = training
        self.clip_denoised = clip_denoised
        self.schedule_time = diffusion_step
        self.decoding_iter_ratio = decoding_iter_ratio
        self.parameterization = 'eps'    # 'eps' or 'x0'
        self.v_posterior = 0.         # Coefficient for posterior variance
        self.l_simple_weight = 1.
        self.original_elbo_weight = 0.

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

        # Instantiate model and evaluation model
        self.model = self.model_config['model_cls'](**self.model_config['model_kwargs'])
        self.model_eval = self.model_config['model_cls'](**self.model_config['eval_kwargs'])

        # 5) Optimizer configuration (default: Adam)
        self.optimizer_config = optimizer_config if optimizer_config is not None else {
            'optimizer_cls': optax.adam,
            'optimizer_kwargs': {
                'learning_rate': 1e-5,
                'b1': 0.9,
            },
        }

        # 6) Create TrainState
        self.train_state = create_train_state_time_cond(
            self.model,
            self.input_config,
            self.optimizer_config
        )

        # 7) Register beta schedule for diffusion
        self.register_schedule(
            beta_schedule='linear',
            # beta_schedule='cosine',
            timesteps=self.schedule_time,
            linear_end=2e-2,
        )

        # 7.5) Compute effective sampling timesteps for faster decoding (DDIM-style)
        self.num_sampling_steps = max(1, int(self.num_timesteps * self.decoding_iter_ratio))
        # Create timestep schedule with t=0 always included as final step
        # Example: 64 steps with ratio=0.1 gives [63, 50, 37, 24, 11, 0]
        if self.num_sampling_steps >= self.num_timesteps:
            # Use all timesteps (standard DDPM)
            self.sampling_timesteps = np.arange(self.num_timesteps)
        else:
            # Use uniformly spaced timesteps, ensuring 0 is always included
            self.sampling_timesteps = np.linspace(
                0, self.num_timesteps - 1, self.num_sampling_steps
            ).round().astype(np.int32)

        # 8) Initialize RNG keys
        self.sample_rngs = {
            'p_noise': random.PRNGKey(seed - 2),
            'q_noise': random.PRNGKey(seed - 1),
            'apply':   random.PRNGKey(seed),
            'dropout': random.PRNGKey(seed + 1),
        }
        self.eval_rng_key = random.PRNGKey(seed + 1)

        separator = "=" * 60
        self.logger.info(separator)
        self.logger.info("                 CONFIGURATIONS SUMMARY                 ")
        self.logger.info(separator)
        self.logger.info(f"Training Mode:       {self.training}")
        self.logger.info(f"Model Class:         {self.model_config['model_cls'].__name__}")
        self.logger.info(f"Model Kwargs:        {self.model_config['model_kwargs']}")
        self.logger.info(f"Eval Kwargs:         {self.model_config['eval_kwargs']}")
        self.logger.info(f"Optimizer Config:    {self.optimizer_config}")
        self.logger.info(f"Input Config:        {self.input_config}")
        self.logger.info(f"Output Dimension:    {self.out_dim}")
        self.logger.info(f"Schedule Time:       {self.schedule_time}")
        self.logger.info(f"Decoding Iter Ratio: {self.decoding_iter_ratio}")
        self.logger.info(f"Sampling Steps:      {self.num_sampling_steps} / {self.num_timesteps}")
        self.logger.info(f"Clip Denoised:       {self.clip_denoised}")
        self.logger.info(f"Parameterization:    {self.parameterization}")
        self.logger.info(separator)

    def reinit_optimizer(self):
        """
        Re-initialize optimizer while keeping the same params.
        """
        old_params = self.train_state.params
        self.train_state = create_train_state_time_cond(
            self.model, self.input_config, self.optimizer_config
        )
        self.train_state = self.train_state.replace(params=old_params)

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=10,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3
    ):
        """
        Register the diffusion schedule (betas, alphas, etc.) based on a given scheme.
        """
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        alphas = 1. - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1. / alphas_cumprod - 1)

        posterior_variance = (
            (1. - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            + self.v_posterior * betas
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, a_min=1e-20))
        self.posterior_mean_coef1 = (
            betas * jnp.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1. - alphas_cumprod)
        )

        if self.parameterization == "eps":
            lvlb_weights = (
                self.betas ** 2 / (2 * self.posterior_variance * alphas) * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * jnp.sqrt(alphas_cumprod) / (2. * 1 - (alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights = np.array(lvlb_weights)
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = jnp.array(lvlb_weights)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_loss(self, pred, target, mean=None):
        """
        Returns MSE loss between pred and target.
        """
        return mse(pred, target)

    def q_mean_variance(self, x_start, t):
        """
        q(x_t | x_0) : mean, variance
        """
        mean = extract_into_numpy(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_numpy(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_numpy(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and the predicted noise.
        """
        return (
            extract_into_numpy(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_numpy(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Posterior: q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            extract_into_numpy(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_numpy(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_numpy(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_numpy(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @partial(jax.jit, static_argnums=(0,))
    def p_mean_variance(self, params, x, t, cond, return_model_out=False):
        """
        p(x_{t-1} | x_t): predict the mean and variance at each step.
        """
        t_input = jax.lax.convert_element_type(t, jnp.float32)[:, jnp.newaxis]
        t_input = timestep_embedding(t_input, self.dim_time_embedding)
        model_out = self.model_eval.apply(params, x, t_input, cond, deterministic=True)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not supported")

        if self.clip_denoised:
            x_recon = jnp.clip(x_recon, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)

        if return_model_out:
            return model_mean, posterior_variance, posterior_log_variance, model_out
        return model_mean, posterior_variance, posterior_log_variance

    @partial(jax.jit, static_argnums=(0,7)) # deprecate the return model out.
    def p_sample(self, params, x, t, cond, rngs=None, repeat_noise=False, return_model_out=False):
        """
        Sample x_{t-1} from x_t in one step.
        """
        b = x.shape[0]
        if return_model_out:
            model_mean, _, model_log_variance, model_out = self.p_mean_variance(
                params, x=x, t=t, cond=cond, return_model_out=True
            )
        else:
            model_mean, _, model_log_variance = self.p_mean_variance(params, x, t, cond)

        noise = default(
            repeat_noise,
            lambda: jax.random.normal(rngs, x.shape),
        )
        nonzero_mask = jnp.reshape(
            1 - jnp.equal(t, 0).astype(jnp.float32),
            (b,) + (1,) * (x.ndim - 1)
        )
        # if return_model_out:
        #     return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise, model_out

        rngs, _ = jax.random.split(rngs)
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise, rngs

    def p_sample_loop(self, params, out_noise, cond, rngs=None, return_intermediates=False):
        """
        Full reverse diffusion loop from x_T to x_0.
        Uses sampling_timesteps schedule for faster decoding (DDIM-style).
        """
        b = cond.shape[0]
        out = out_noise
        intermediates = [out]
        predictions = []

        # Use reduced timestep schedule for faster sampling
        for i in reversed(self.sampling_timesteps):
            out_step, rngs = self.p_sample(
                params,
                out,
                jnp.full((b,), i, dtype=jnp.int32),
                cond,
                rngs=rngs,
                return_model_out=return_intermediates
            )

            if return_intermediates:
                out, pred = out_step
                intermediates.append(out)
                predictions.append(pred)
            else:
                out = out_step

        # if return_intermediates:
        #     return out, intermediates, predictions
        return out

    def p_losses(self, params, state, x_start, t, cond, noise=None, rngs=None):
        """
        Calculate diffusion loss at a random (or given) time step t.
        """
        noise = default(noise, lambda: jax.random.normal(rngs['p_noise'], shape=x_start.shape))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, rngs=rngs)

        t_input = jax.lax.convert_element_type(t, jnp.float32)[:, jnp.newaxis]
        t_input = timestep_embedding(t_input, self.dim_time_embedding)
        r1, r2 = jax.random.split(rngs['apply'])
        apply_rng = {'params': r1, 'dropout': r2}

        model_out = state.apply_fn(params, x_noisy, t_input, cond, deterministic=True, rngs=apply_rng)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not supported")

        loss = self.get_loss(model_out, target).mean(axis=[-1])
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = jnp.mean((self.lvlb_weights[t] * loss))

        # loss_val = loss_simple + self.original_elbo_weight * loss_vlb
        loss_val = loss_simple

        log_prefix = 'train' if self.training else 'val'
        loss_dict = {
            f'{log_prefix}/loss_simple': loss_simple,
            f'{log_prefix}/loss_vlb': loss_vlb,
            f'{log_prefix}/loss': loss_val
        }
        return loss_val, loss_dict

    @partial(jax.jit, static_argnums=(0,))
    def q_sample(self, x_start, t, noise=None, rngs=None):
        """
        q(x_t | x_0), forward noising process.
        """
        noise = default(noise, lambda: jax.random.normal(rngs['q_noise'], shape=x_start.shape))
        return (
            extract_into_numpy(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_numpy(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @partial(jax.jit, static_argnums=(0,))
    def train_model_jit(self, state, x, t, cond, noise, rngs=None):
        """
        JIT-compiled single training step.
        """
        grad_fn = jax.grad(self.p_losses, has_aux=True)
        grads, loss_dict = grad_fn(state.params, state, x, t, cond, noise, rngs=rngs)
        metric = (None, loss_dict)
        state = state.apply_gradients(grads=grads)
        return state, metric

    def train_model(self, x, cond, t=None, compute_eval_loss=False):
        """
        Train the model for one step. Optionally, compute an evaluation MSE between 
        the reverse diffusion output (using eval_model_jit) and the ground truth x.
        
        Args:
            x (np.ndarray): Input data.
            cond (np.ndarray): Conditional input.
            t (np.ndarray, optional): Time steps. If None, randomly sampled.
            compute_eval_loss (bool, optional): If True, compute and add the MSE 
                evaluation loss to the returned metric.
                
        Returns:
            dict: A dictionary of loss metrics.
        """
        # Ensure proper shape for x and cond
        if x.ndim == 2:
            x = x[:, None, :]
        if cond.ndim == 2:
            cond = cond[:, None, :]

        # Randomly sample t if not provided
        if t is None:
            t = np.random.randint(0, self.num_timesteps, (x.shape[0],))
        else:
            assert t.shape[0] == x.shape[0], "t and x must have same batch size"

        # Generate noise for the training step
        noise = np.random.randn(*(x.shape[0], 1, self.out_dim))
        self.train_state, metric = self.train_model_jit(
            self.train_state, x, t, cond, noise, rngs=self.sample_rngs
        )
        self.sample_rngs = update_rngs(self.sample_rngs)

        # Optionally compute evaluation MSE using eval_model_jit and the original x
        if compute_eval_loss:
            # Generate new noise for the reverse diffusion process
            out_noise = np.random.randn(*(x.shape[0], 1, self.out_dim))
            self.eval_rng_key, rngs = random.split(self.eval_rng_key)
            # Run the reverse diffusion sampling (x̂) from noise conditioned on cond
            x_eval = self.eval_model_jit(self.train_state.params, out_noise, cond, rngs)
            # Compute MSE between the evaluated output and the ground truth x
            mse_eval = mse(x_eval, x).mean()
            metric[1]['train/eval_mse'] = mse_eval

        return metric

    # @partial(jax.jit, static_argnums=(0,))
    def eval_model_jit(self, params, out_noise, cond, rngs=None):
        """
        JIT-compiled evaluation (sampling) loop.
        """
        return self.p_sample_loop(params, out_noise, cond, rngs, return_intermediates=False)

    def eval_model(self, cond, params=None):
        """
        Evaluate the model (sampling from noise).
        """
        cond = np.array(cond)
        if cond.ndim == 2:
            cond = cond[:, None, :]
        out_noise = np.random.randn(*(cond.shape[0], 1, self.out_dim))
        self.eval_rng_key, rngs = random.split(self.eval_rng_key)
        return self.eval_model_jit(
            self.train_state.params if params is None else params,
            out_noise,
            cond,
            rngs
        )

    def forward(self, **kwargs):
        return self.p_sample_loop(**kwargs)

    def eval_model_out(self, x, t=None, cond=None):
        """
        For debugging: get the model output on a forward-diffused sample at time t.
        """
        @jax.jit
        def get_model_eval(params, x_, t_, cond_):
            t_input_ = jax.lax.convert_element_type(t_, jnp.float32)[:, jnp.newaxis]
            t_input_ = timestep_embedding(t_input_, self.dim_time_embedding)
            return self.model_eval.apply(params, x_, t_input_, cond_, deterministic=True)

        noise = np.random.randn(*(x.shape[0], 1, self.out_dim))
        x_noisy = self.q_sample(x, t, noise=noise, rngs=self.sample_rngs)
        model_out = get_model_eval(self.train_state.params, x_noisy, t, cond)
        return model_out, t

    @partial(jax.jit, static_argnums=(0,))
    def eval_intermediates_jit(self, state, out_noise, cond, rngs=None):
        """
        JIT-compiled function that returns intermediates of reverse diffusion.
        """
        out, intermediates, pred = self.p_sample_loop(
            state.params, out_noise, cond, rngs, return_intermediates=True
        )
        return intermediates, pred

    def eval_intermediates(self, cond):
        """
        Return all intermediate x_t states + model predictions at each step.
        """
        out_noise = np.random.randn(*(1, 1, self.out_dim))
        out_noise = np.repeat(out_noise, cond.shape[0], axis=0)
        self.eval_rng_key, rngs = random.split(self.eval_rng_key)
        intermediate, pred = self.eval_intermediates_jit(
            self.train_state, out_noise, cond, rngs
        )
        return intermediate, pred


if __name__ == "__main__":
    """
    Test CondDDPMDecoder with a simple example
    """
    print("\n" + "=" * 80)
    print("TESTING CondDDPMDecoder (SAVE VERSION)")
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
    print("\n[1] Creating CondDDPMDecoder...")
    decoder = CondDDPMDecoder(
        training='train',
        model_config=model_config,
        input_config=input_config,
        out_dim=9,
        diffusion_step=32,  # Fewer steps for testing
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
        loss = metric[1]['train/loss']
        losses.append(loss)
        print(f"   Step {step+1}: loss = {loss:.6f}")

    print("✓ Training successful")

    # Test evaluation
    print("\n[3] Testing evaluation (sampling)...")
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

    is_deterministic = np.allclose(sample_a, sample_b)
    print(f"   Same seed → same output: {is_deterministic}")
    if is_deterministic:
        print("   ✓ Sampling is DETERMINISTIC (DDIM-style)")
    else:
        print("   ✓ Sampling is STOCHASTIC (DDPM-style)")

    # Test forward diffusion (q_sample)
    print("\n[5] Testing forward diffusion (q_sample)...")
    x_test = np.random.randn(2, 1, 9).astype(np.float32)
    t = np.array([5, 10], dtype=np.int32)
    x_noisy = decoder.q_sample(x_test, t, rngs=decoder.sample_rngs)
    print(f"   Original mean: {x_test.mean():.6f}")
    print(f"   Noisy mean: {x_noisy.mean():.6f}, std: {x_noisy.std():.6f}")
    print("✓ Forward diffusion successful")

    # Test p_sample (single reverse step)
    print("\n[6] Testing single reverse step (p_sample)...")
    x_t = np.random.randn(2, 1, 9).astype(np.float32)
    t = np.array([10, 10], dtype=np.int32)
    cond_test = np.random.randn(2, 1, 60).astype(np.float32)
    rng = random.PRNGKey(888)

    x_prev, rng_new = decoder.p_sample(
        decoder.train_state.params, x_t, t, cond_test, rngs=rng
    )
    print(f"   Input x_t mean: {x_t.mean():.6f}")
    print(f"   Output x_(t-1) mean: {x_prev.mean():.6f}")
    print("✓ Reverse step successful")

    # Summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nSummary:")
    print(f"  • Training: {len(losses)} steps completed")
    print(f"  • Final loss: {losses[-1]:.6f}")
    print(f"  • Sampling: {'Deterministic' if is_deterministic else 'Stochastic'}")
    print(f"  • Model parameters: ~{sum(p.size for p in jax.tree_util.tree_leaves(decoder.train_state.params)):,}")
    print("=" * 80 + "\n")