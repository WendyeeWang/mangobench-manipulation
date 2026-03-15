import os
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"  # wandb mirror
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import tqdm
import numpy as np
from ogcrl.workspace.base_workspace import BaseWorkspace
from ogcrl.dataset.base_dataset import BaseImageDataset
from ogcrl.dataset.robot_gc_dataset import convert_replaybuffer_to_rl_dataset
from ogcrl.dataset.robot_gc_dataset import Dataset, GCDataset, HGCDataset
from ogcrl.agents import agents
from ogcrl.utils.flax_utils import restore_agent, save_agent
from ogcrl.utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
import time
import jax
OmegaConf.register_new_resolver("eval", eval, replace=True)
from collections import defaultdict


class RobotWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Convert to goal-conditioned format
        print("Converting diffusion dataset to goal-conditioned format...")
        episode_idx = dataset.replay_buffer.get_episode_idxs()
        if cfg.save_goal == True:
            save_goal_path = cfg.save_goal_path
        else:
            save_goal_path = None
        train_dataset = convert_replaybuffer_to_rl_dataset(dataset.replay_buffer, episode_idx, compact_dataset=True, save_goal_path=save_goal_path, observation_type=cfg.observation)
        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
        cfg.save_dir = os.path.join(cfg.save_dir, 'MANGOBench', cfg.agent.agent_name, save_name)
        os.makedirs(cfg.save_dir, exist_ok=True)
        del dataset

        val_dataset = None

        # Create Dataset object (FrozenDict)
        train_dataset = Dataset.create(**train_dataset)
        if val_dataset is not None:
            val_dataset = Dataset.create(**val_dataset)

        # Convert to goal-conditioned dataset
        config = cfg.agent
        dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }[config["dataset_class"]]
        train_dataset = dataset_class(Dataset.create(**train_dataset), config)
        if val_dataset is not None:
            val_dataset = dataset_class(Dataset.create(**val_dataset), config)

        # Begin training
        example_batch = train_dataset.sample(1)

        # Initialize agent(s)
        agent_class = agents[config['agent_name']]
        agent = agent_class.create(cfg.training.seed, example_batch['observations'], example_batch['actions'], config)

        # Training loop
        train_logger = CsvLogger(os.path.join(self.output_dir, 'train.csv'))
        eval_logger = CsvLogger(os.path.join(self.output_dir, 'eval.csv'))
        print("output_dir:", self.output_dir)
        first_time = time.time()
        last_time = time.time()

        for i in tqdm.tqdm(range(1, cfg.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)

            if i % cfg.log_interval == 0:
                train_metrics = {}
                train_metrics.update({f'training/{k}': v for k, v in update_info.items()})

                if val_dataset is not None:
                    val_batch = val_dataset.sample(config['batch_size'])
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

                train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                train_logger.log(train_metrics, step=i)

            if i % cfg.save_interval == 0:
                save_agent(agent, cfg.save_dir, i)

        train_logger.close()
        eval_logger.close()
