import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
    '''
    Normal MLP
    '''
    hidden_size: int=256
    out_shape: int=4
    dropout_rate : float=0.1
    deterministic: bool=False

    def setup(self) -> None:
        self.layer0 = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.hidden_size),
            nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic),
            nn.selu,
            nn.Dense(self.out_shape),
        ])

    def __call__(self, x):
        x = self.layer0(x)
        return x

class DenoisingMLP(nn.Module):
    """
    Flax Denoising MLP Block
    """

    dim: int
    out_dim: int
    n_blocks: int = 6
    context_emb_dim: int = 512
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.t_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.hidden_emb = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.context_emb = nn.Sequential([
            nn.Dense(self.context_emb_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.dim, dtype=self.dtype),
        ])

        self.norm_cond = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        self.out = nn.Sequential([
            nn.LayerNorm(epsilon=1e-5, dtype=self.dtype),
            nn.Dense(self.dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout, deterministic=False),
            nn.gelu,
            nn.Dense(self.out_dim, dtype=self.dtype),
        ])

        # Simple dense layers for the MLP style
        for i in range(self.n_blocks):
            setattr(self, f"mlp{i}", nn.Dense(self.dim, dtype=self.dtype))
            setattr(self, f"norm_mlp{i}", nn.LayerNorm(epsilon=1e-5, dtype=self.dtype))

    def __call__(self, x, time, cond, deterministic=False):
        t_emb = self.t_emb(time)
        x = self.hidden_emb(x) + t_emb
        cond = self.norm_cond(self.context_emb(cond))
        x_im = jnp.concatenate([x, cond], axis=-1)

        for i in range(self.n_blocks):
            residual = x
            x = getattr(self, f"mlp{i}")(
                getattr(self, f"norm_mlp{i}")(x_im)
            )
            x = nn.gelu(x)
            x = x + residual
            x_im = jnp.concatenate([x, cond], axis=-1)

        x = self.out(x)
        return x

