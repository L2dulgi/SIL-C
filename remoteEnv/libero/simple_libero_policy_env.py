import os
import re
import numpy as np
import imageio
import json
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# -------------------------------
# Function to extract natural language from BDDL
# -------------------------------
def extract_language_from_bddl(bddl_file_path):
    """
    Opens a BDDL file and extracts a natural language command using the pattern (:language ...).
    Returns the extracted string or an empty string ("") if the pattern is not found.
    """
    try:
        with open(bddl_file_path, "r") as f:
            text = f.read()
        match = re.search(r'\(:language\s+(.*?)\)', text)
        if match:
            return match.group(1).strip()
        else:
            return ""
    except Exception as e:
        print("[extract_language_from_bddl] Error:", e)
        return ""

# -------------------------------
# Policy class (with natural_language parameter)
# -------------------------------
class Policy:
    def __init__(self, print_modality_shape=False):
        """
        Constructor: if print_modality_shape is True,
        it will print the shape of each modality (object_state, gripper_states, joint_states, reward, natural_language)
        when action() is called.
        """
        self.print_modality_shape = print_modality_shape

    def action(self, object_state, gripper_states, joint_states, reward, natural_language):
        """
        Receives object_state, gripper_states, joint_states, reward, and natural_language.
        If print_modality_shape is True, prints the shape of each modality and the natural language command.
        Returns a random 7-dimensional action in the range [-1, 1].
        """
        if self.print_modality_shape:
            if object_state is not None:
                print("[Policy] object_state shape:", np.array(object_state).flatten().shape)
            else:
                print("[Policy] No object_state input")
            if gripper_states is not None:
                print("[Policy] gripper_states shape:", np.array(gripper_states).flatten().shape)
            else:
                print("[Policy] No gripper_states input")
            if joint_states is not None:
                print("[Policy] joint_states shape:", np.array(joint_states).flatten().shape)
            else:
                print("[Policy] No joint_states input")
            if reward is not None:
                if np.isscalar(reward):
                    print("[Policy] reward:", reward)
                else:
                    print("[Policy] reward shape:", np.array(reward).flatten().shape)
            else:
                print("[Policy] No reward input")
            if natural_language:
                print("[Policy] Natural language command:", natural_language)
            else:
                print("[Policy] No natural language input")
                
        action = np.random.uniform(low=-1.0, high=1.0, size=(7,))
        print("[Policy] Generated action:", action)
        return action

# -------------------------------
# simulate_policy_in_env function (natural language extraction and passing)
# -------------------------------
def simulate_policy_in_env(benchmark_name, task_id, max_action_count, policy_obj):
    """
    Initializes an environment using the given benchmark name, task ID, and BDDL file,
    sets the initial state from the benchmark,
    and runs a policy for up to max_action_count steps.
    
    At each step, passes the current state and the natural language instruction
    extracted from the BDDL file to policy_obj.action(), and steps the environment with the returned action.
    
    Returns the 'done' status from the final step.
    """
    bench_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in bench_dict:
        print(f"[error] Benchmark '{benchmark_name}' not found.")
        return None
    bench_cls = bench_dict[benchmark_name]
    bench_instance = bench_cls()
    
    num_tasks = bench_instance.get_num_tasks()
    if num_tasks == 0:
        print(f"[warn] Benchmark '{benchmark_name}' contains no tasks!")
        return None
    if task_id >= num_tasks:
        print(f"[error] task_id {task_id} does not exist. (num_tasks = {num_tasks})")
        return None
    
    task = bench_instance.get_task(task_id)
    print("\n[simulate_policy_in_env] Task", task_id, "name:", task.name)
    bddl_file_path = bench_instance.get_task_bddl_file_path(task_id)
    
    # Extract natural language
    natural_language = extract_language_from_bddl(bddl_file_path)
    
    # Create OffScreenRenderEnv with BDDL file
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    
    init_states = bench_instance.get_task_init_states(task_id)
    if init_states is None or len(init_states) == 0:
        print("[error] Initial state not found in benchmark.")
        env.close()
        return None
    init_state = init_states[0]
    print("[simulate_policy_in_env] Using benchmark initial state, shape:", np.array(init_state).flatten().shape)
    env.set_init_state(init_state)
    
    # First step with default action to get initial observation
    default_action = [0.0] * 7
    obs, reward, done, info = env.step(default_action)
    
    iteration = 0
    while iteration < max_action_count and not done:
        current_obj_state = obs.get("object-state", None)
        current_gripper = obs.get("robot0_gripper_qpos", None)
        current_joint = obs.get("robot0_joint_pos", None)
        
        action = policy_obj.action(current_obj_state, current_gripper, current_joint, reward, natural_language)
        obs, reward, done, info = env.step(action)
        
        iteration += 1
        print(f"[simulate_policy_in_env] Iteration {iteration}, done: {done}")
    
    env.close()
    return done

# -------------------------------
# simulate_policy_and_save_demo function (record video)
# -------------------------------
def simulate_policy_and_save_demo(
    output_video_path, 
    benchmark_name, 
    task_id, 
    max_action_count, 
    policy_obj, 
    fps=30,
    camera_width=256,
    camera_height=256
):
    """
    Initializes an environment and records a video of a policy interacting with it.
    Saves the video at the given output path.
    
    Steps:
      - Extracts a natural language command from BDDL.
      - Gets initial observation with default action.
      - Loops up to max_action_count steps, each time calling policy_obj.action() with the current state and language.
      - Captures frontview (or agentview) image at each step and saves it to a frame list.
    
    After the loop, saves all frames as a video file using imageio.
    """
    bench_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in bench_dict:
        print(f"[error] Benchmark '{benchmark_name}' not found.")
        return None
    bench_cls = bench_dict[benchmark_name]
    bench_instance = bench_cls()
    
    num_tasks = bench_instance.get_num_tasks()
    if num_tasks == 0:
        print(f"[warn] Benchmark '{benchmark_name}' contains no tasks!")
        return None
    if task_id >= num_tasks:
        print(f"[error] task_id {task_id} does not exist. (num_tasks = {num_tasks})")
        return None
    
    task = bench_instance.get_task(task_id)
    print("\n[simulate_policy_and_save_demo] Task", task_id, "name:", task.name)
    bddl_file_path = bench_instance.get_task_bddl_file_path(task_id)
    
    natural_language = extract_language_from_bddl(bddl_file_path)
    
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_names": ["frontview"],
        "camera_heights": [camera_height],
        "camera_widths": [camera_width],
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    
    init_states = bench_instance.get_task_init_states(task_id)
    if init_states is None or len(init_states) == 0:
        print("[error] Initial state not found in benchmark.")
        env.close()
        return None
    init_state = init_states[0]
    print("[simulate_policy_and_save_demo] Using benchmark initial state, shape:", np.array(init_state).flatten().shape)
    env.set_init_state(init_state)
    
    default_action = [0.0] * 7
    obs, reward, done, info = env.step(default_action)
    
    frames = []
    frame = None
    if "frontview_image" in obs:
        frame = obs["frontview_image"]
    elif "agentview_image" in obs:
        frame = obs["agentview_image"]
    if frame is not None:
        frames.append(frame[::-1])
    
    iteration = 0
    while iteration < max_action_count and not done:
        current_obj_state = obs.get("object-state", None)
        current_gripper = obs.get("robot0_gripper_qpos", None)
        current_joint = obs.get("robot0_joint_pos", None)
        
        action = policy_obj.action(current_obj_state, current_gripper, current_joint, reward, natural_language)
        obs, reward, done, info = env.step(action)
        iteration += 1
        print(f"[simulate_policy_and_save_demo] Iteration {iteration}, done: {done}")
        
        frame = None
        if "frontview_image" in obs:
            frame = obs["frontview_image"]
        elif "agentview_image" in obs:
            frame = obs["agentview_image"]
        if frame is not None:
            frames.append(frame[::-1])
    
    env.close()
    
    if len(frames) == 0:
        print("[error] No frames captured.")
        return False
    
    imageio.mimsave(output_video_path, frames, fps=fps)
    print(f"[simulate_policy_and_save_demo] Video saved: {output_video_path}")
    return done

# ---------------------------------------------------------------------------
# Main Example: run policy and save interaction as video
if __name__ == "__main__":
    output_video_path = "demo_output.mp4"
    benchmark_name = "libero_spatial"
    task_id = 0
    max_action_count = 40
    camera_width = 256
    camera_height = 256

    policy_instance = Policy(print_modality_shape=True)
    
    print("\n========== Starting policy-environment interaction and video recording ==========")
    done_flag = simulate_policy_and_save_demo(
        output_video_path, benchmark_name, task_id, max_action_count,
        policy_instance, fps=30, camera_width=camera_width, camera_height=camera_height
    )
    
    if done_flag:
        print("\n[MAIN] Episode completed (done=True).")
    else:
        print("\n[MAIN] Episode did not complete within the action limit.")
