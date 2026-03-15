# MangoBench: A Benchmark for Multi-Agent Goal-Conditioned Offline Reinforcement Learning

Official implementation of **MangoBench** (CVPR 2026).

MangoBench is the first fully cooperative multi-goal benchmark for offline MARL, covering 47 tasks across locomotion and bimanual manipulation. See the [project page](https://wendyeewang.github.io/MangoBench/) for videos and more details about the environments, tasks, and baseline algorithms.

**Note:** The locomotion environment code is available at [mangobench-locomotion](https://github.com/WendyeeWang/mangobench-locomotion).

## Installation

Follow [RoboFactory](https://github.com/MARS-EAI/RoboFactory) to set up the base environment, then install additional dependencies:
```bash
cd robofactory
pip install -r requirements_ogbench.txt
```

## Data

Follow [RoboFactory](https://github.com/MARS-EAI/RoboFactory) to generate datasets and process data. 

> **Note:** The training code automatically converts processed imitation learning datasets into standard RL datasets.

## Running
```bash
bash ./policy/OGCRL/train_eval_acp.sh
```

> **Note:** The first training run requires saving goals for evaluation. Add `save_goal=True` and `save_goal_path=policy/OGCRL/ogcrl/goal` to your training command.

Example:
```bash
python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=LiftBarrier-rf-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=hiql \
    agent.high_alpha=3.0 \
    agent.low_alpha=3.0 \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.low_actor_rep_grad=True \
    agent.p_aug=0.5 \
    agent.subgoal_steps=10 \
    observation=visual \
    save_dir=expacp \
    save_goal=True \
    save_goal_path=policy/OGCRL/ogcrl/goal
```

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{Wang2026MangoBench,
  title={MangoBench: A Benchmark for Multi-Agent Goal-Conditioned Offline Reinforcement Learning},
  author={Wang, Yi and Zhong, Ningze and Fu, Zhiheng and Wang, Longguang and Zhang, Ye and Guo, Yulan},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
