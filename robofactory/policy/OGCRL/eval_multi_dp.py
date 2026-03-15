
import sys
sys.path.append('./')
sys.path.insert(0, './policy/OGCRL')
import torch
import os

import pathlib
from pathlib import Path
from collections import deque, defaultdict
from robofactory.tasks import *
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
from ogcrl.utils.multiagent_manager import MultiAgentManager
from ogcrl.agents import agents
import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from robofactory.utils.wrappers.record import RecordEpisodeMA

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import cv2

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "${CONFIG_DIR}/robocasa/take_photo.yaml"
    """Configuration to build scenes, assets and agents."""

    agent_config: str = ""
    """Configuration to build ogcrl agent."""
    restore_path: Optional[str] = None
    """Path to the checkpoint directory to restore the agent. If not given, no restoration is performed."""
    restore_epoch: Optional[int] = None
    """Epoch of the checkpoint to restore. If not given, the latest checkpoint is restored."""
    goal_dir: str = "./goals/"
    """Directory to load goals"""
    eval_temperature: int = 0
    """Actor temperature for evaluation."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    data_num: int = 100
    """The number of episode data used for training the policy"""

    record_dir: Optional[str] = './eval_gc_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 250
    """Maximum number of steps to run the simulation"""

    observation: str = "visual"
    agent_high_alpha: Optional[float] = None
    agent_low_alpha: Optional[float] = None
    agent_encoder: Optional[str] = None
    agent_low_actor_rep_grad: Optional[bool] = None
    agent_p_aug: Optional[float] = None
    agent_subgoal_steps: Optional[int] = None
    agent_alpha: Optional[float] = None
    task_name: str = "PickPlace"

def load_goals(goal_dir):
    import os
    import pickle

    with open(goal_dir, "rb") as f:
        goals = pickle.load(f)
    print(f"[INFO] Loaded {len(goals['state'])} goals from {goal_dir}")
    return goals

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped

def get_model_input(observation, agent_pos, agent_id):
    camera_name = 'head_camera' + '_agent' + str(agent_id)

    # Resize image to match training image size
    rgb = observation['sensor_data'][camera_name]['rgb']  # torch.Size([1, 240, 320, 3]), uint8

    img = rgb.squeeze(0).cpu().numpy()   # -> (240, 320, 3), dtype=uint8

    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)  # (64, 64, 3), uint8

    return dict(
        head_cam = img_resized,
        agent_pos = agent_pos,
    )

def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'
        with open(args.agent_config, "r") as f:
            agent_config = yaml.safe_load(f)
            if args.agent_high_alpha is not None:
                agent_config["high_alpha"]=args.agent_high_alpha
            if args.agent_low_alpha is not None:
                agent_config["low_alpha"]=args.agent_low_alpha
            if args.agent_encoder is not None:
                agent_config["encoder"]=args.agent_encoder
            if args.agent_low_actor_rep_grad is not None:
                agent_config["low_actor_rep_grad"]=args.agent_low_actor_rep_grad
            if args.agent_p_aug is not None:
                agent_config["p_aug"]=args.agent_p_aug
            if args.agent_subgoal_steps is not None:
                agent_config["subgoal_steps"]=args.agent_subgoal_steps
            if args.agent_alpha is not None:
                agent_config["alpha"]=args.agent_alpha

    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    record_dir = args.record_dir + '/' + args.observation + '/' + str(args.seed) + '_' + str(args.data_num) + '_' + str(args.restore_epoch)
    if record_dir:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000)
    raw_obs, _ = env.reset(seed=args.seed[0])
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=True
    )

    # Load multi gc policy
    agent_num = planner.agent_num
    goals = []  # multi-agent goals

    # Load temporal subgoals for all agents
    for i in range(agent_num):
        goal_path = os.path.join(args.goal_dir, f"{args.task_name}_Agent{i}_{args.data_num}_Temperal.pkl")
        goal = load_goals(goal_path)
        goals.append(goal)
    goal_len = len(goals[0]["state"])
    print(f"[INFO] Each goal file contains {goal_len} goals")

    num_goals = goal_len
    example_goals = []  # outer list indexed by goal index

    for j in range(num_goals):
        goal_group = []  # all agents' data for the current goal index

        for agent_id, g in enumerate(goals):
            state = g["state"][j]
            head_cam = g["head_camera"][j]  # (3, 240, 320)

            # Convert to (240, 320, 3)
            head_cam = np.moveaxis(head_cam, 0, -1)

            # Resize to (64, 64)
            head_cam_resized = cv2.resize(head_cam, (64, 64), interpolation=cv2.INTER_AREA)

            # Save debug PNG
            save_path = f"goal_64_{args.task_name}_goal{j}_agent{agent_id}.png"
            cv2.imwrite(save_path, cv2.cvtColor(head_cam_resized, cv2.COLOR_RGB2BGR))

            goal_group.append({
                "state": state,
                "head_camera": head_cam_resized.astype(np.uint8),
            })

        example_goals.append(goal_group)

    print("[INFO] All goals saved successfully.")

    # Initialize agent(s)
    agent_class = agents[agent_config['agent_name']]

    # Use goal states to initialize agents
    if args.observation=="state":
        agent_manager = MultiAgentManager(agent_class, agent_num, args.seed[0], [eg["state"] for eg in example_goals[-1]],[eg["state"] for eg in example_goals[-1]], agent_config)
    elif args.observation=="visual":
        agent_manager = MultiAgentManager(agent_class, agent_num, args.seed[0], [eg["head_camera"] for eg in example_goals[-1]],[eg["state"] for eg in example_goals[-1]], agent_config)
    elif args.observation=="both":
        agent_manager = MultiAgentManager(agent_class, agent_num, args.seed[0], example_goals, example_goals, agent_config)
    # Restore if needed.
    if args.restore_path is not None:
        agent_manager.restore(args.restore_path, args.task_name, args.data_num, args.restore_epoch)

    eval_agent = agent_manager.deepcopy(to_cpu=True)

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])

    observations=[]
    for id in range(agent_num):  # Update initial observations for each agent
        initial_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[id])
        obs = get_model_input(raw_obs, initial_qpos, id)
        observations.append(obs)
        save_path = f"head_cam_64_agent{id}.png"
        cv2.imwrite(save_path, cv2.cvtColor(obs['head_cam'], cv2.COLOR_RGB2BGR))

    actor_fns = {
        i: supply_rng(eval_agent[i].sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        for i in range(agent_num)
    }

    cnt = 0
    # Multi-goal evaluation
    goal_interval=50
    while True:  # Iterate steps (up to args.max_steps)
        if verbose:
            print("Iteration:", cnt)
        cnt = cnt + 1
        if cnt > args.max_steps:
            break
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        # Compute current goal index for multi-goal
        goal_index = min(cnt // goal_interval, num_goals - 1)
        print(f"[INFO] Current step: {cnt}, using goal index: {goal_index} / {num_goals - 1}")
        for id in range(agent_num):
            if args.observation=="state":
                now_action = actor_fns[id](observations=observations[id]["agent_pos"], goals=example_goals[goal_index][id]["state"], temperature=args.eval_temperature)
            elif args.observation=="visual":
                now_action = actor_fns[id](observations=observations[id]["head_cam"], goals=example_goals[goal_index][id]["head_camera"], temperature=args.eval_temperature)

            raw_obs = env.get_obs()

            # Current joint positions (excluding finger joints at the end)
            current_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].numpy()

            # Construct path from current joint positions to target (excluding gripper)
            path = np.vstack((current_qpos, now_action[:-1]))

            try:
                # Time-optimal trajectory planning
                times, position, right_vel, acc, duration = planner.planner[id].TOPP(
                    path, 0.05, verbose=True
                )
            except Exception as e:
                print(f"Planner Error (agent {id}): {e}")
                # On planner failure, hold current joints + gripper
                action_now = np.hstack([current_qpos, now_action[-1]])
                action_dict[f'panda-{id}'].append(action_now)
                action_step_dict[f'panda-{id}'].append(1)
                continue

            # On success: get the generated trajectory length
            n_step = position.shape[0]
            action_step_dict[f'panda-{id}'].append(n_step)
            gripper_state = now_action[-1]

            # If trajectory is empty, fall back to current position
            if n_step == 0:
                action_now = np.hstack([current_qpos, gripper_state])
                action_dict[f'panda-{id}'].append(action_now)
            # Otherwise save planned trajectory step by step
            for j in range(n_step):
                true_action = np.hstack([position[j], gripper_state])
                action_dict[f'panda-{id}'].append(true_action)

        # Initialize trajectory start index for each agent
        start_idx = [0 for _ in range(agent_num)]

        # Single rollout (each agent has one action sequence)
        max_step = 0
        for id in range(agent_num):
            max_step = max(max_step, action_step_dict[f'panda-{id}'][0])

        # Execute each timestep up to the longest trajectory
        for j in range(max_step):
            true_action = dict()
            for id in range(agent_num):
                step_count = action_step_dict[f'panda-{id}'][0]
                # If j exceeds this agent's trajectory, hold the last action
                now_step = min(j, step_count - 1)
                true_action[f'panda-{id}'] = action_dict[f'panda-{id}'][start_idx[id] + now_step]

            next_observations, reward, terminated, truncated, info = env.step(true_action)

        if verbose:
            print("max_step", max_step)

        # Update each agent's state and model input
        for id in range(agent_num):
            start_idx[id] += action_step_dict[f'panda-{id}'][0]
            if action_step_dict[f'panda-{id}'][0] == 0:
                continue
            obs = get_model_input(next_observations, true_action[f'panda-{id}'], id)
            observations[id]=obs

        if args.render_mode is not None:
            env.render()
        if info['success'] == True:
            env.close()
            if record_dir:
                print(f"Saving video to {record_dir}")
            print("success")
            return
    env.close()
    if record_dir:
        print(f"Saving video to {record_dir}")
    print("failed")

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
