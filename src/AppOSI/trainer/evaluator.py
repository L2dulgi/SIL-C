import os
import json
import cloudpickle
import numpy as np
from AppOSI.environment.remote import KitchenRemoteEvaluator
from AppOSI.models.agent.base import PTGMAgent
from AppOSI.config.skill_stream_config import EvaluationTracer
import jax
from copy import deepcopy
class SkillEvaluator:
    def __init__(self, exp_save_path, remote_eval_host='127.0.0.1', remote_eval_port=9999,
                 eval_num_episodes=3, eval_max_steps=280):
        """
        Initializes the Evaluator's basic environment.
        
        Parameters:
            exp_save_path: Path where the experiment's models are saved.
            remote_eval_host: Host for remote evaluation.
            remote_eval_port: Port for remote evaluation.
            eval_num_episodes: Number of episodes for evaluation.
            eval_max_steps: Maximum steps for evaluation.
        """
        self.exp_save_path = exp_save_path
        self.remote_eval_host = remote_eval_host
        self.remote_eval_port = remote_eval_port
        self.eval_num_episodes = eval_num_episodes
        self.eval_max_steps = eval_max_steps

    def _load_model(self, model_type, agent_config):
        """
        Loads a model of type 'decoder', 'interface', or 'policy' using the provided agent_config.
        
        agent_config should be a dict:
            { 'decoder': (idx, filename), 'interface': (idx, filename), 'policy': (idx, filename) }
        where the filename is provided without the extension (.pkl will be appended).
        """
        if model_type not in agent_config:
            raise ValueError(f"agent_config does not contain information for {model_type}.")
        _, model_filename = agent_config[model_type]
        if not model_filename.endswith('.pkl'):
            model_filename = model_filename + '.pkl'
        if model_type in ["decoder", "interface"]:
            file_path = os.path.join(self.exp_save_path, "skills", f"{model_type}_{model_filename}")
        elif model_type == "policy":
            file_path = os.path.join(self.exp_save_path, "policy", model_filename)
        else:
            raise ValueError("Unknown model type.")
        if os.path.exists(file_path):
            print(f"[SkillEvaluator] Loading {model_type} model from {file_path}.")
            with open(file_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            print(f"[SkillEvaluator] Model file for {model_type} does not exist at {file_path}.")
            return None

    def _build_agent(self, agent_config):
        """
        Builds and returns a PTGMAgent using the models specified in the given agent_config.
        """
        decoder = self._load_model("decoder", agent_config)
        interface = self._load_model("interface", agent_config)
        policy = self._load_model("policy", agent_config)
        print(decoder, interface, policy)
        if decoder is None or policy is None:
            raise ValueError("Essential models (decoder or policy) could not be loaded.")
        
        return PTGMAgent(decoder=decoder, interface=interface, policy=policy)

    def evaluate_direct_config(self, agent_config, eval_tasks):
        """
        Performs evaluation in direct config mode.
        
        Parameters:
            agent_config: A dict with keys 'decoder', 'interface', 'policy' specifying the checkpoint information.
                          (Filenames should be provided without the .pkl extension.)
            eval_tasks: A list of evaluation tasks (each task can be a dict specifying task details).
        """
        agent = self._build_agent(agent_config)
        jax.clear_caches()
        
        overall_rewards = []
        detailed_results = {}
        for task in eval_tasks:
            print(f"[SkillEvaluator] Evaluating task: {task}")
            remote_evaluator = KitchenRemoteEvaluator(
                host=self.remote_eval_host,
                port=self.remote_eval_port,
                eval_fn=agent.eval,
            )
            remote_evaluator.set_task([task])
            eval_rewards, _ = remote_evaluator.evaluate(
                num_episodes=self.eval_num_episodes,
                max_steps=self.eval_max_steps
            )
            remote_evaluator.close()
            eval_rewards_list = [float(r) for r in eval_rewards] if eval_rewards is not None else []
            avg_reward = np.mean(eval_rewards_list) if eval_rewards_list else 0.0
            print(f"[SkillEvaluator] Task {task} average reward: {avg_reward:.2f}")
            detailed_results[str(task)] = {
                "rewards": eval_rewards_list,
                "avg_reward": avg_reward
            }
            overall_rewards.extend(eval_rewards_list)
        overall_avg_reward = np.mean(overall_rewards) if overall_rewards else 0.0
        results = {
            "agent_config": agent_config,
            "eval_tasks": eval_tasks,
            "overall_rewards": overall_rewards,
            "overall_avg_reward": overall_avg_reward,
            "detailed": detailed_results,
        }
        results_save_path = os.path.join(self.exp_save_path, "evaluation_results_direct.json")
        with open(results_save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"[SkillEvaluator] Direct evaluation results saved to {results_save_path}.")
        return results

    def evaluate_from_datastream(self, skill_stream_config):
        """
        Performs evaluation from a skill stream configuration.
        
        This function iterates through each phase in the datastream (skill_stream_config.datastream)
        and uses the EvaluationTracer to obtain the evaluation configurations. It then evaluates the models
        just like in the trainer.
        
        Parameters:
            skill_stream_config: An instance of SkillStreamConfig (must have a 'datastream' attribute).
        """
        if not hasattr(skill_stream_config, "datastream"):
            raise ValueError("The provided skill_stream_config does not have a datastream attribute.")
        eval_results = {}
        tracer = EvaluationTracer()
        num_phases = len(skill_stream_config.datastream)
        for phase in range(num_phases):
            print(f"[SkillEvaluator] Starting evaluation for phase {phase}.")
            eval_traced_list = tracer.get_eval_tasks_by_reference(
                stream_config=skill_stream_config,
                phase=phase
            )
            if len(eval_traced_list) == 0:
                print(f"[SkillEvaluator] No evaluation tasks for phase {phase}.")
                eval_results[phase] = {
                    "train_targets": skill_stream_config.datastream[phase].train_targets,
                    "eval_tasks": None,
                    "overall_rewards": None,
                    "overall_avg_reward": None,
                    "detailed": None,
                }
                continue
            phase_overall_rewards = []
            phase_detailed = {}

            for eval_trace in eval_traced_list:
                agent_config = eval_trace.get("agent_config", None)
                eval_tasks = eval_trace.get("eval_tasks", None)
                if agent_config is None:
                    print("[SkillEvaluator] Missing agent_config information in evaluation trace.")
                    continue
                agent = self._build_agent(agent_config)
                print(f"[SkillEvaluator] Evaluating phase {phase} tasks: {eval_tasks} using agent_config: {agent_config}")
                remote_evaluator = KitchenRemoteEvaluator(
                    host=self.remote_eval_host,
                    port=self.remote_eval_port,
                    eval_fn=agent.eval,
                )
                remote_evaluator.set_task(eval_tasks)
                eval_rewards, _ = remote_evaluator.evaluate(
                    num_episodes=self.eval_num_episodes,
                    max_steps=self.eval_max_steps
                )
                remote_evaluator.close()
                eval_rewards_list = [float(r) for r in eval_rewards] if eval_rewards is not None else []
                avg_reward = np.mean(eval_rewards_list) if eval_rewards_list else 0.0
            
                policy_name = agent_config['policy'][1].split('/')[0]
                phase_detailed[policy_name] = {
                    "rewards": eval_rewards_list,
                    "avg_reward": avg_reward
                }
                phase_overall_rewards.extend(eval_rewards_list)

            overall_avg = np.mean(phase_overall_rewards) if phase_overall_rewards else 0.0
            eval_results[phase] = {
                "phase_name": skill_stream_config.datastream[phase].phase_name,
                "train_targets": skill_stream_config.datastream[phase].train_targets,
                "eval_tasks": skill_stream_config.datastream[phase].eval_tasks,
                "overall_rewards": phase_overall_rewards,
                "overall_avg_reward": overall_avg,
                "detailed": deepcopy(phase_detailed),
            }

            print(f"[SkillEvaluator] Finished evaluation for phase {phase} with overall average reward: {overall_avg:.2f}")
            results_save_path = os.path.join(self.exp_save_path, "eval_results_post.json")
            with open(results_save_path, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=4)
            print(f"[SkillEvaluator] Evaluation results from datastream saved to {results_save_path}.")
        return eval_results

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    import pickle
    from AppOSI.config.skill_stream_config import SkillStreamConfig
    import warnings
    warnings.filterwarnings("ignore")

    exp_save_path = "./logs/default/ptgm/0401kitchen_syncAppenderCtest"  # Actual experiment path

    # Example 1: Direct config evaluation.
    evaluator = SkillEvaluator(exp_save_path)
    # Prepare a direct agent config (filenames provided without .pkl extension)
    if False :
        direct_agent_config = {
            'decoder': (0, 'decoder_pre_0'),
            'interface': (0, 'interface_pre_0'),
            'policy': (0, 'policy_2/pre_1')
        }
        # Define evaluation tasks directly (e.g., a list of task dicts)
        direct_eval_tasks = [
            {'data_name': 'mbls'} 
        ]
        evaluator.evaluate_direct_config(direct_agent_config, direct_eval_tasks)

    # Example 2: Evaluation from datastream.
    # Assume that the experiment configuration (which contains the skill stream config)
    # is stored in "experiment_config.pkl" under exp_save_path.

    exp_config_path = os.path.join(exp_save_path, "experiment_config.pkl")
    with open(exp_config_path, 'rb') as f:
        exp_config = pickle.load(f)
    # Here we assume that the skill stream config is provided as the "scenario_config"
    # attribute of the experiment configuration.
    skill_stream_config = exp_config.scenario_config    
    evaluator.evaluate_from_datastream(skill_stream_config)
