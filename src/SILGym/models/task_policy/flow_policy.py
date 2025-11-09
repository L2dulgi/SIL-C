from __future__ import annotations

import copy
import numpy as np
import optax
from typing import Any, Dict, Optional

from SILGym.models.basic.module import BasicModule
from SILGym.models.skill_decoder.fql import FQLDecoder


class FlowMatchingPolicy(BasicModule):
    """
    Flow-based policy built on top of the FQL decoder.

    The policy treats discrete skill logits as the flow targets (x) and learns a
    continuous normalizing flow conditioned on the observation features. During
    evaluation, the sampled logits are converted to discrete skill IDs via argmax.
    """

    def __init__(
        self,
        *,
        out_shape: int,
        input_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        flow_model_config: Optional[Dict[str, Any]] = None,
        flow_steps: int = 10,
        use_onestep_flow: bool = False,
        eval_use_onestep: bool = False,
        clip_logits: bool = False,
        seed: int = 777,
    ) -> None:
        super().__init__()

        if flow_model_config is None:
            raise ValueError("FlowMatchingPolicy requires 'flow_model_config'.")
        if input_config is None or 'x' not in input_config:
            raise ValueError("FlowMatchingPolicy requires input_config with key 'x'.")

        self.out_shape = int(out_shape)
        self.flow_steps = int(flow_steps)
        self.use_onestep_flow = bool(use_onestep_flow)
        self.eval_use_onestep = bool(eval_use_onestep)
        self.clip_logits = bool(clip_logits)

        self.out_dim = self.out_shape
        self.input_config = dict(input_config)
        if optimizer_config is None:
            self.optimizer_config = {
                'optimizer_cls': optax.adam,
                'optimizer_kwargs': {
                    'learning_rate': 1e-4,
                    'b1': 0.9,
                },
            }
        else:
            self.optimizer_config = optimizer_config

        self.flow_model_config = copy.deepcopy(flow_model_config)

        flow_input_config = {
            'x': (1, 1, self.out_shape),
            'cond': self.input_config['x'],
        }

        self.flow_decoder = FQLDecoder(
            model_config=self.flow_model_config,
            optimizer_config=self.optimizer_config,
            input_config=flow_input_config,
            out_dim=self.out_shape,
            clip_actions=self.clip_logits,
            flow_steps=self.flow_steps,
            use_onestep_flow=self.use_onestep_flow,
            eval_use_onestep=self.eval_use_onestep,
            seed=seed,
        )

        # Mirror key attributes for compatibility with BasicModule utilities
        self.model = self.flow_decoder.model
        self.model_eval = self.flow_decoder.model_eval
        self.model_config: Dict[str, Any] = {
            'out_shape': self.out_shape,
            'flow_steps': self.flow_steps,
            'use_onestep_flow': self.use_onestep_flow,
        }
        self.subtask_prototypes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties delegating to the underlying flow decoder
    # ------------------------------------------------------------------
    @property
    def train_state(self):
        return self.flow_decoder.train_state

    @train_state.setter
    def train_state(self, value):
        self.flow_decoder.train_state = value

    @property
    def sample_rngs(self):
        return self.flow_decoder.sample_rngs

    @sample_rngs.setter
    def sample_rngs(self, value):
        self.flow_decoder.sample_rngs = value

    @property
    def eval_rng_key(self):
        return self.flow_decoder.eval_rng_key

    @eval_rng_key.setter
    def eval_rng_key(self, value):
        self.flow_decoder.eval_rng_key = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_model(self, batch: Dict[str, Any]):
        cond = self._ensure_flow_batch(batch['inputs'])
        targets = self._prepare_targets(batch['labels'])

        metrics = self.flow_decoder.train_model(
            x=targets,
            cond=cond,
            compute_eval_loss=False,
        )
        flow_loss = metrics[1].get('train/bc_flow_loss', metrics[1].get('train/loss', 0.0))
        return float(np.asarray(flow_loss))

    def eval_model(self, x: np.ndarray, cut_off: Optional[int] = None):
        cond = self._ensure_flow_batch(x)
        logits = self.flow_decoder.eval_model(cond)
        logits = np.array(logits, dtype=np.float32, copy=True)

        if logits.ndim == 3 and logits.shape[1] == 1:
            logits = logits[:, 0, :]
        elif logits.ndim == 2:
            pass
        else:
            logits = np.reshape(logits, (logits.shape[0], -1))

        if self.clip_logits:
            logits = np.clip(logits, -1.0, 1.0)
        if cut_off is not None and 0 <= cut_off < logits.shape[-1]:
            logits[:, cut_off:] = -np.inf

        return np.argmax(logits, axis=-1)

    def forward(self, cond: np.ndarray):
        return self.flow_decoder.eval_model(cond)

    def reinit_optimizer(self):
        self.flow_decoder.reinit_optimizer()

    def set_subtask_prototype(self, prototypes):
        self.subtask_prototypes = copy.deepcopy(prototypes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_flow_batch(self, arr: Any) -> np.ndarray:
        data = np.asarray(arr, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, None, :]
        elif data.ndim == 2:
            data = data[:, None, :]
        elif data.ndim == 3:
            if data.shape[1] != 1:
                data = data[:, :1, :]
        else:
            data = data.reshape((data.shape[0], -1))
            data = data[:, None, :]
        return data

    def _prepare_targets(self, labels: Any) -> np.ndarray:
        labels_arr = np.asarray(labels)

        # Already provided as one-hot logits
        if labels_arr.ndim >= 2 and labels_arr.shape[-1] == self.out_shape:
            if labels_arr.ndim == 3 and labels_arr.shape[1] == 1:
                one_hot = labels_arr
            else:
                one_hot = labels_arr[:, None, :]
            return one_hot.astype(np.float32)

        squeezed = np.squeeze(labels_arr)
        if squeezed.ndim == 0:
            squeezed = squeezed[None]

        indices = np.asarray(squeezed, dtype=np.int32).reshape(-1)
        indices = np.clip(indices, 0, self.out_shape - 1)

        one_hot = np.zeros((indices.shape[0], self.out_shape), dtype=np.float32)
        one_hot[np.arange(indices.shape[0]), indices] = 1.0
        return one_hot[:, None, :]
