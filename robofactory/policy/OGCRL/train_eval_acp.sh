#IHIQL LiftBarrier train and eval
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
    save_dir=expacp 

python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent1_150.zarr \
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
    save_dir=expacp 

bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/hiql.yaml expacp/MANGOBench/hiql 15000 policy/OGCRL/ogcrl/goal/ visual 3.0 3.0  impala_small True 0.5 10 None hiql

#IHIQL PlaceFood train and eval
python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=PlaceFood \
  task.dataset.zarr_path=data/zarr_data/PlaceFood_Agent0_150.zarr \
  training.debug=False \
  training.seed=100 \
  training.device=cuda:0 \
  exp_name=PlaceFood-robot_gc-train \
  logging.mode=online \
  env_name=robofactory \
  agent=hiql \
  agent.high_alpha=3.0 \
  agent.low_alpha=3.0 \
  train_steps=38800 \
  log_interval=1000 \
  save_interval=38800 \
  agent.batch_size=256 \
  agent.encoder=impala_small \
  agent.low_actor_rep_grad=True \
  agent.p_aug=0.5 \
  agent.subgoal_steps=10 \
  observation=visual \
  save_dir=expacp

python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=PlaceFood \
  task.dataset.zarr_path=data/zarr_data/PlaceFood_Agent1_150.zarr \
  training.debug=False \
  training.seed=100 \
  training.device=cuda:0 \
  exp_name=PlaceFood-robot_gc-train \
  logging.mode=online \
  env_name=robofactory \
  agent=hiql \
  agent.high_alpha=3.0 \
  agent.low_alpha=3.0 \
  train_steps=38800 \
  log_interval=1000 \
  save_interval=38800 \
  agent.batch_size=256 \
  agent.encoder=impala_small \
  agent.low_actor_rep_grad=True \
  agent.p_aug=0.5 \
  agent.subgoal_steps=10 \
  observation=visual \
  save_dir=expacp

bash ./policy/OGCRL/eval_multi.sh configs/table/place_food.yaml 150  1 PlaceFood policy/OGCRL/ogcrl/config/agent/hiql.yaml expacp/MANGOBench/hiql 38800 policy/OGCRL/ogcrl/goal/ visual 3.0 3.0  impala_small True 0.5 10 None hiql

#GCBC LiftBarrier train and eval
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
    agent=gcbc \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.p_aug=0.5 \
    observation=visual \
    save_dir=expacp

python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent1_150.zarr \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=LiftBarrier-rf-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=gcbc \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.p_aug=0.5 \
    observation=visual \
    save_dir=expacp

bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/gcbc.yaml expacp/MANGOBench/gcbc 15000 policy/OGCRL/ogcrl/goal/ visual None None  impala_small None 0.5 None None gcbc

#GCBC PlaceFood train and eval
python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=PlaceFood \
    task.dataset.zarr_path=data/zarr_data/PlaceFood_Agent0_150.zarr \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=PlaceFood-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=gcbc \
    train_steps=38800 \
    log_interval=1000 \
    save_interval=38800 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.p_aug=0.5 \
    observation=visual \
    save_dir=expacp

python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=PlaceFood \
    task.dataset.zarr_path=data/zarr_data/PlaceFood_Agent1_150.zarr \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=PlaceFood-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=gcbc \
    train_steps=38800 \
    log_interval=1000 \
    save_interval=38800 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.p_aug=0.5 \
    observation=visual \
    save_dir=expacp

bash ./policy/OGCRL/eval_multi.sh configs/table/place_food.yaml 150  1 PlaceFood policy/OGCRL/ogcrl/config/agent/gcbc.yaml expacp/MANGOBench/gcbc 38800 policy/OGCRL/ogcrl/goal/ visual None None impala_small None 0.5 None None gcbc

#ICRL LiftBarrier train and eval
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
    agent=crl \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.p_aug=0.5 \
    agent.alpha=0.1 \
    observation=visual \
    save_dir=expacp

python policy/OGCRL/train.py \
    --config-name=robot_gc.yaml \
    task.name=LiftBarrier-rf \
    task.dataset.zarr_path=data/zarr_data/LiftBarrier-rf_Agent1_150.zarr \
    training.debug=False \
    training.seed=100 \
    training.device=cuda:0 \
    exp_name=LiftBarrier-rf-robot_gc-train \
    logging.mode=online \
    env_name=robofactory \
    agent=crl \
    train_steps=15000 \
    log_interval=1000 \
    save_interval=15000 \
    agent.batch_size=256 \
    agent.encoder=impala_small \
    agent.p_aug=0.5 \
    agent.alpha=0.1 \
    observation=visual \
    save_dir=expacp

bash ./policy/OGCRL/eval_multi.sh configs/table/lift_barrier.yaml 150  1 LiftBarrier-rf policy/OGCRL/ogcrl/config/agent/crl.yaml expacp/MANGOBench/crl 15000 policy/OGCRL/ogcrl/goal/ visual None None  impala_small None 0.5 None 0.1 crl

#ICRL placefood train and eval
python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=PlaceFood \
  task.dataset.zarr_path=data/zarr_data/PlaceFood_Agent0_150.zarr \
  training.debug=False \
  training.seed=100 \
  training.device=cuda:0 \
  exp_name=PlaceFood-robot_gc-train \
  logging.mode=online \
  env_name=robofactory \
  agent=crl \
  train_steps=38800 \
  log_interval=1000 \
  save_interval=38800 \
  agent.batch_size=256 \
  agent.encoder=impala_small \
  agent.p_aug=0.5 \
  agent.alpha=3.0 \
  observation=visual \
  save_dir=expacp

python policy/OGCRL/train.py \
  --config-name=robot_gc.yaml \
  task.name=PlaceFood \
  task.dataset.zarr_path=data/zarr_data/PlaceFood_Agent1_150.zarr \
  training.debug=False \
  training.seed=100 \
  training.device=cuda:0 \
  exp_name=PlaceFood-robot_gc-train \
  logging.mode=online \
  env_name=robofactory \
  agent=crl \
  train_steps=38800 \
  log_interval=1000 \
  save_interval=38800 \
  agent.batch_size=256 \
  agent.encoder=impala_small \
  agent.p_aug=0.5 \
  agent.alpha=3.0 \
  observation=visual \
  save_dir=expacp

bash ./policy/OGCRL/eval_multi.sh configs/table/place_food.yaml 150  1 PlaceFood policy/OGCRL/ogcrl/config/agent/crl.yaml expacp/MANGOBench/crl 38800 policy/OGCRL/ogcrl/goal/ visual None None  impala_small None 0.5 None 3.0 crl

