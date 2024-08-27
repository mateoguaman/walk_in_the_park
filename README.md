# A Walk in the Park

Code to replicate [A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning](https://arxiv.org/abs/2208.07860), which contains code for training a simulated or real A1 quadrupedal robot to walk. Project page: https://sites.google.com/berkeley.edu/walk-in-the-park

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

To install the robot [SDK](https://github.com/unitreerobotics/unitree_legged_sdk), first install the dependencies in the README.md

To build, run: 
```bash
cd real/third_party/unitree_legged_sdk
mkdir build
cd build
cmake ..
make
``` 

Finally, copy the built `robot_interface.XXX.so` file to this directory.

## Training

Example command to run simulated training:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=A1Run-v0 \
                --utd_ratio=20 \
                --start_training=1000 \
                --max_steps=100000 \
                --config=configs/droq_config.py
```



```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 XLA_PYTHON_CLIENT_ALLOCATOR=platform python train_online.py --env_name=A1Run-v0 --utd_ratio=20 --start_training=1000 --max_steps=100000  --config=configs/droq_config.py --limit_action_range=0.35 --arena_type=bowl --slope=0.5 --friction=1.0
```

To also run with loaded checkpoints and buffers, run:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 XLA_PYTHON_CLIENT_ALLOCATOR=platform python train_online.py --env_name=A1Run-v0 --utd_ratio=20 --start_training=1000 --max_steps=200000  --config=configs/droq_config.py --limit_action_range=0.35 --arena_type=bowl --slope=0.5 --friction=1.0 --load_checkpoint=/home/mateo/projects/walk_in_the_park/aprl_saved/checkpoints/checkpoint_99001 --load_buffer=/home/mateo/projects/walk_in_the_park/aprl_saved/buffers/buffer_99001 --save_dir=bowl_05_friction_1
```

To run training on the real robot, add `--real_robot=True`

For real robot:
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.50 XLA_PYTHON_CLIENT_ALLOCATOR=platform python train_online.py --env_name=A1Run-v0 --utd_ratio=20 --start_training=1000 --max_steps=100000  --config=configs/droq_config.py --limit_action_range=0.35 --real_robot
```
