import numpy as np
from abc import ABC, abstractmethod
from SILGym.models.skill_interface.ptgm import ContinualPTGMInterface
from SILGym.models.skill_decoder.appender import ModelAppender
try:
    from SILGym.models.skill_decoder.appenderv3 import ModelAppenderV3
except ImportError:  # pragma: no cover - fallback if V3 not installed
    class _ModelAppenderV3Placeholder:  # type: ignore
        pass

    ModelAppenderV3 = _ModelAppenderV3Placeholder  # type: ignore
from tqdm import tqdm
from SILGym.utils.logger import get_logger
# ----------------------------
# Abstract BaseAgent Class
# ----------------------------
class BaseAgent(ABC):
    def __init__(self, decoder=None, interface=None, policy=None):
        self.decoder = decoder
        self.interface = interface
        self.policy = policy
        self.debug = False
        self.debug_eval = False
        self.logger = get_logger(__name__)

    # ----------------------------
    # Public Methods (Core)
    # ----------------------------
    def eval(self, obs):
        """
        Evaluate the agent on the given observation(s).

        Steps:
         1. Convert the observation to a numpy array.
         2. Ensure a batch dimension is present.
         3. Compute the skill entry using the policy network.
         4. Forward the skill entry through the interface.
         5. Prepare the decoder input.
         6. Evaluate the decoder.
         7. Reshape the output if necessary.
         8. Return the final action as a list.
        """
        original = np.array(obs)
        if self.debug_eval:
            self.logger.debug(f"Original observation shape: {original.shape}")

        policy_input = self._ensure_batch_dimension(original)
        skill_entry = self.policy.eval_model(policy_input, cut_off=self.interface.num_skills) # as list start from 0
        if self.debug_eval:
            self.logger.debug(f"Skill entry shape: {skill_entry.shape}")

        # Forward through the interface (hook method)
        skill_info, skill_aux = self._interface_forward(skill_entry, original)
        skill_id, decoder_id = skill_info
        if self.debug_eval:
            self.logger.debug(f"Skill ID shape: {skill_id.shape}")
            self.logger.debug(f"Skill auxiliary shape: {skill_aux.shape}")
            self.logger.debug(f"Selected decoder ID: {decoder_id}")

        # Prepare the decoder input (hook method)
        decoder_input = self._prepare_decoder_input(policy_input, skill_aux)
        if self.debug_eval:
            self.logger.debug(f"Decoder input shape: {decoder_input.shape}")

        if isinstance(self.decoder, (ModelAppender, ModelAppenderV3)):
            action = self.decoder.eval_model(cond=decoder_input, decoder_id=decoder_id)
        else:
            action = self.decoder.eval_model(decoder_input)
        if self.debug_eval:
            self.logger.debug(f"Raw action shape: {np.array(action).shape}")



        action = self._reshape_action(action)
        if self.debug_eval:
            self.logger.debug(f"Final action shape: {action.shape}")
        return action.tolist()

    def sample_logit(self, batch):
        """
        Generate predicted actions for all skills and compute rewards by processing
        each skill individually. Returns hard labels (skill indices) and a dictionary
        with normalized rewards, MSE, and predicted actions.

        Args:
            batch: Dictionary with keys:
                - "inputs": numpy array of shape (B, F)
                - "labels": numpy array of shape (B, A)

        Returns:
            Tuple of (reward_logits_hard, details_dict)
        """
        inputs = batch["inputs"]  # (B, F)
        labels = batch["labels"]  # (B, A)
        B = inputs.shape[0]

        num_skills = getattr(self.interface, "num_skills", None)
        if num_skills is None:
            raise ValueError("Interface must have a 'num_skills' attribute to define the skill space.")

        pred_actions_list = []
        mse_list = []

        for skill in range(num_skills):
            # Process each skill via a hook method
            pred_actions_skill, mse_skill = self._process_skill(batch, skill)
            pred_actions_list.append(pred_actions_skill)
            mse_list.append(mse_skill)

        pred_actions = np.stack(pred_actions_list, axis=1)  # (B, num_skills, A)
        mse = np.stack(mse_list, axis=1)  # (B, num_skills)

        # Normalize MSE to compute rewards (lower MSE yields higher reward)
        epsilon = 1e-8
        mse_min = np.min(mse, axis=1, keepdims=True)
        mse_max = np.max(mse, axis=1, keepdims=True)
        normalized_mse = (mse - mse_min) / (mse_max - mse_min + epsilon)
        rewards = 1 - normalized_mse

        # Compute softmax over rewards
        softmax_rewards = np.exp(rewards) / np.sum(np.exp(rewards), axis=1, keepdims=True)
        reward_logits = softmax_rewards

        # Adjust reward_logits to match the policy's expected output size.
        desired_out_size = self.policy.model_config['out_shape']
        current_size = reward_logits.shape[1]
        if current_size < desired_out_size:
            pad_width = desired_out_size - current_size
            reward_logits = np.pad(reward_logits, ((0, 0), (0, pad_width)), mode='constant')
        elif current_size > desired_out_size:
            reward_logits = reward_logits[:, :desired_out_size]

        reward_logits_hard = np.argmax(reward_logits, axis=1)
        if self.debug:
            self.logger.debug(f"Batch size: {B}")
            self.logger.debug(f"Number of skills: {num_skills}")
            self.logger.debug(f"Predicted actions shape: {pred_actions.shape}")
            self.logger.debug(f"MSE shape: {mse.shape}")
            self.logger.debug(f"Rewards shape: {rewards.shape}")

        return reward_logits_hard, {"rewards": rewards, "mse": mse, "actions": pred_actions}

    def create_skill_labels(self, dataloader):
        reward_logits_list = []
        for batch in tqdm(dataloader.get_all_batch(batch_size=1024, shuffle=False),
                          desc="Preparing policy data", ncols=100):
            input_batch = {
                'inputs': batch['observations'],
                'labels': batch['actions'],
                'skills': batch['skills']
            }
            reward_logits, _ = self.sample_logit(input_batch)
            reward_logits_list.append(reward_logits)
        flatten_logits = np.concatenate(reward_logits_list, axis=0)

        self.logger.info(f"Flattened reward logits variants: {np.unique(flatten_logits)}")
        desired_length = dataloader.stacked_data['observations'].shape[0]
        flatten_logits = flatten_logits[:desired_length]
        dataloader.stacked_data['reward_logits'] = flatten_logits

        return dataloader
    # ----------------------------
    # Helper Methods
    # ----------------------------

    def _ensure_batch_dimension(self, arr):
        """Ensure the input array has a batch dimension."""
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
            if self.debug:
                self.logger.debug(f"Expanded input shape: {arr.shape}")
        return arr

    @abstractmethod
    def _interface_forward(self, skill_entry, original):
        """
        Abstract method for forwarding the skill entry through the interface.
        For PTGMAgent, simply call interface.forward(skill_entry).
        For IsCiLAgent, call interface.forward(skill_entry, original).
        """
        pass

    @abstractmethod
    def _prepare_decoder_input(self, policy_input, skill_aux):
        """
        Abstract method for preparing the decoder input.
        For PTGMAgent, typically concatenate policy_input and skill_aux.
        For IsCiLAgent, may simply return policy_input.
        """
        pass

    @abstractmethod
    def _process_skill(self, batch, skill):
        """
        Abstract method to process a single skill for sample_logit.
        Should return the predicted actions for the skill and its MSE.
        """
        pass

    def _reshape_action(self, action):
        """Reshape the action output to (y,) for a single sample or (B, y) for batched input."""
        action = np.array(action)
        if action.ndim == 3:
            if action.shape[0] == 1:
                action = action.reshape(action.shape[-1])
                if self.debug:
                    self.logger.debug(f"Reshaped single observation action: {action.shape}")
            else:
                action = action.reshape(action.shape[0], action.shape[-1])
                if self.debug:
                    self.logger.debug(f"Reshaped batched action: {action.shape}")
        else:
            if self.debug:
                self.logger.debug(f"Unexpected action shape: {action.shape}")
        return action

# ----------------------------
# PTGMAgent Implementation
# ----------------------------
class PTGMAgent(BaseAgent):
    def __init__(self, decoder=None, interface: ContinualPTGMInterface = None, policy=None, debug=False):
        super().__init__(decoder, interface, policy)
        self.debug = debug

    def _interface_forward(self, skill_entry, original):
        # PTGMAgent calls forward with only the skill entry.
        return self.interface.forward(skill_entry)

    def _prepare_decoder_input(self, policy_input, skill_aux):
        # PTGMAgent concatenates the policy input and the auxiliary data.
        return np.concatenate([policy_input, skill_aux], axis=-1)

    def _process_skill(self, batch, skill):
        """
        Process a single skill in a PTGMAgent.
        For each skill:
         - Create a skill tensor.
         - Obtain auxiliary data from the interface.
         - Concatenate expanded inputs with the auxiliary data.
         - Evaluate the decoder and compute the MSE with the labels.
        """
        inputs = batch["inputs"]   # (B, F)
        labels = batch["labels"]   # (B, A)
        B = inputs.shape[0]

        # Create a skill id tensor with shape (B, 1, 1)
        skill_ids = np.full((B, 1, 1), skill, dtype=np.int32)
        skill_info, skill_aux = self.interface.forward(skill_ids)
        skill_id, decoder_id = skill_info  # (B, 1)

        # Expand inputs to (B, 1, F) for concatenation
        inputs_expanded = np.expand_dims(inputs, axis=1)
        decoder_input = np.concatenate([inputs_expanded, skill_aux], axis=-1)  # (B, 1, F + D)
        decoder_input_reshaped = decoder_input.reshape(B, -1)  # (B, F + D)

        if isinstance(self.decoder, (ModelAppender, ModelAppenderV3)):
            pred_actions_skill = self.decoder.eval_model(cond=decoder_input_reshaped, decoder_id=decoder_id)
        else:
            pred_actions_skill = self.decoder.eval_model(decoder_input_reshaped)
        pred_actions_skill = np.array(pred_actions_skill)
        if pred_actions_skill.ndim == 3:
            pred_actions_skill = pred_actions_skill.reshape(B, pred_actions_skill.shape[-1])
        mse_skill = np.mean((pred_actions_skill - labels) ** 2, axis=-1)
        return pred_actions_skill, mse_skill
    
from SILGym.models.skill_interface.buds import BUDSInterface

class BUDSAgent(PTGMAgent):
    def __init__(self, decoder=None, interface: BUDSInterface = None, policy=None, debug=False):
        super().__init__(decoder, interface, policy, debug)
        self.debug = debug

# ----------------------------
# IsCiLAgent Implementation
# ----------------------------
from SILGym.models.skill_interface.semantic_based import PrototypeInterface

class IsCiLAgent(BaseAgent):
    def __init__(self, decoder=None, interface: PrototypeInterface = None, policy=None, debug=False):
        super().__init__(decoder, interface, policy)
        self.debug = debug

    def _interface_forward(self, skill_entry, original):
        # IsCiLAgent calls forward with both skill_entry and the original observation.
        return self.interface.forward(skill_entry, original)

    def _prepare_decoder_input(self, policy_input, skill_aux):
        # IsCiLAgent uses state, semantic-goal for decoder input.
        return np.concatenate([policy_input, skill_aux], axis=-1)

    def _process_skill(self, batch, skill):
        """
        Process a single skill in an IsCiLAgent.
        For each skill:
         - Create a skill id tensor.
         - Call the interface with the skill tensor and original inputs.
         - Use only the inputs as the decoder input.
         - Evaluate the decoder and compute the MSE with the labels.
        """
        inputs = batch["inputs"]   # (B, F)
        labels = batch["labels"]   # (B, A)
        B = inputs.shape[0]

        skill_ids = np.full((B, 1, 1), skill, dtype=np.int32)
        skill_info, skill_aux = self.interface.forward(skill_ids, inputs)
        skill_id, decoder_id = skill_info

        # Use the original inputs as the decoder input.
        decoder_input = self._prepare_decoder_input(inputs, skill_aux)  # (B, F)

        if isinstance(self.decoder, (ModelAppender, ModelAppenderV3)):
            pred_actions_skill = self.decoder.eval_model(cond=decoder_input, decoder_id=decoder_id)
        else:
            pred_actions_skill = self.decoder.eval_model(decoder_input)
        pred_actions_skill = np.array(pred_actions_skill)
        if pred_actions_skill.ndim == 3:
            pred_actions_skill = pred_actions_skill.reshape(B, pred_actions_skill.shape[-1])
        mse_skill = np.mean((pred_actions_skill - labels) ** 2, axis=-1)
        return pred_actions_skill, mse_skill

    def sample_logit(self, batch):
        """
        Semantic based agetnt requires skill labels to train the policy and decoder.
        """
        inputs = batch["inputs"]  # (B, F)
        skill_labels = batch["skills"] # (B ) string
        B = inputs.shape[0]

        # from skill interface get the index of skills
        skill_ids = self.interface.get_skill_ids(skill_labels)

        if skill_ids.shape[0] != B:
            raise ValueError("Skill IDs shape does not match batch size.")

        return skill_ids, {"rewards": None, "mse": None, "actions": None}

from tqdm import tqdm

class LazySIAgent(BaseAgent):
    '''
    Agent for LazySI interface with bidirectional matching.
    '''
    def __init__(
            self, 
            decoder=None, 
            interface=None, 
            policy=None, 
            debug=False
        ):
        super().__init__(decoder, interface, policy)
        if hasattr(policy, 'subtask_prototypes'):
            self.interface.update_subtask_prototype(policy.subtask_prototypes)
        self.debug = debug

    def _interface_forward(self, skill_entry, original):
        # LazySIAgent calls forward with both skill_entry and the original observation.
        return self.interface.forward(skill_entry, original, static=False)

    def _prepare_decoder_input(self, policy_input, skill_aux):
        # LazySIAgent uses state, sub-goal for decoder input.
        return np.concatenate([policy_input, skill_aux], axis=-1)

    def _process_skill(self, batch, skill):
        """
        Process a single skill for LazySI.
        For each skill:
         - Create a skill tensor.
         - Obtain auxiliary data from the interface.
         - Concatenate expanded inputs with the auxiliary data.
         - Evaluate the decoder and compute the MSE with the labels.
        """
        inputs = batch["inputs"]   # (B, F)
        labels = batch["labels"]   # (B, A)
        B = inputs.shape[0]

        # Create a skill id tensor with shape (B, 1)
        skill_ids = np.full((B, 1 ), skill, dtype=np.int32)
        skill_info, skill_aux = self.interface.forward(skill_ids, inputs, static=True)
        skill_id, decoder_id = skill_info  # (B, 1)

        decoder_input = np.concatenate([inputs, skill_aux], axis=-1)  # (B, F + D)

        if isinstance(self.decoder, (ModelAppender, ModelAppenderV3)):
            pred_actions_skill = self.decoder.eval_model(cond=decoder_input, decoder_id=decoder_id)
        else:
            pred_actions_skill = self.decoder.eval_model(decoder_input)
        pred_actions_skill = np.array(pred_actions_skill)
        if pred_actions_skill.ndim == 3:
            pred_actions_skill = pred_actions_skill.reshape(B, pred_actions_skill.shape[-1])
        mse_skill = np.mean((pred_actions_skill - labels) ** 2, axis=-1)
        return pred_actions_skill, mse_skill
    
    # ----------------------------
    # Public Methods (Core)
    # ----------------------------
    def check_skill_labels(self, dataloader):
        # for all data check the compatibility of the entry.
        modified_labels = []
        for i, skill in tqdm(enumerate(dataloader.stacked_data['reward_logits'])):
            # forward decoder mp prototype.
            (output, _), _ = self.interface.forward(np.array([skill]), dataloader.stacked_data['observations'][i])
            if output != skill :
                modified_labels.append(output)
        
        # check the compatibility of the entry.
        self.logger.info(f"Modified ratio: {len(modified_labels) / len(dataloader.stacked_data['reward_logits'])}")
        return dataloader   
    
class LazySIZeroAgent(LazySIAgent):
    '''
    Zero-shot agent for LazySI interface (without policy).
    '''
    def __init__(
            self, 
            decoder=None, 
            interface=None, 
            policy=None, 
            debug=False
        ):
        super().__init__(decoder, interface, policy, debug)
        self.logger.info("LazySIZeroAgent is used. No policy is used to get the skill entry.")

    def eval(self, obs):
        """
        Just Forward the interface and decoder.
        """
        original = np.array(obs) # (B,F)
        if self.debug_eval:
            self.logger.debug(f"Original observation shape: {original.shape}")

        policy_input = self._ensure_batch_dimension(original)

        # for LazySIZero, do not use the policy to get the skill entry.
        skill_entry = np.random.randint(0, self.interface.num_skills, size=(policy_input.shape[0], 1))

        if self.debug_eval:
            self.logger.debug(f"Skill entry shape: {skill_entry.shape}")

        # Forward through the interface (hook method)
        skill_info, skill_aux = self._interface_forward(skill_entry, original)
        skill_id, decoder_id = skill_info
        if self.debug_eval:
            self.logger.debug(f"Skill ID shape: {skill_id.shape}")
            self.logger.debug(f"Skill auxiliary shape: {skill_aux.shape}")
            self.logger.debug(f"Selected decoder ID: {decoder_id}")

        # Prepare the decoder input (hook method)
        decoder_input = self._prepare_decoder_input(policy_input, skill_aux)
        if self.debug_eval:
            self.logger.debug(f"Decoder input shape: {decoder_input.shape}")

        if isinstance(self.decoder, (ModelAppender, ModelAppenderV3)):
            action = self.decoder.eval_model(cond=decoder_input, decoder_id=decoder_id)
        else:
            action = self.decoder.eval_model(decoder_input)
        if self.debug_eval:
            self.logger.debug(f"Raw action shape: {np.array(action).shape}")

        action = self._reshape_action(action)
        if self.debug_eval:
            self.logger.debug(f"Final action shape: {action.shape}")
        return action.tolist()


class SILCAgent(LazySIAgent):
    """
    Agent for the SILC (Skill Incremental Learning with Clustering) interface.
    
    This agent uses the refactored SILC interface while maintaining compatibility
    with the LazySI agent implementation. It provides improved modularity and
    cleaner architecture.
    """
    
    def __init__(
        self,
        decoder=None,
        interface=None,  # Should be SILCInterface
        policy=None,
        debug=False
    ):
        super().__init__(
            decoder=decoder,
            interface=interface,
            policy=policy,
            debug=debug
        )
        self.logger.info("SILCAgent initialized - using refactored SILC interface")


class SILCZeroAgent(LazySIZeroAgent):
    """
    Zero-shot agent for SILC interface (without policy component).
    """
    
    def __init__(
        self,
        decoder=None,
        interface=None,  # Should be SILCInterface
        policy=None,
        debug=False
    ):
        super().__init__(
            decoder=decoder,
            interface=interface,
            policy=policy,
            debug=debug
        )
        self.logger.info("SILCZeroAgent initialized - using refactored SILC interface without policy")
