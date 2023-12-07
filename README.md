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

To run training on the real robot, add `--real_robot=True`



## Mateo edits

Tested on Ubuntu 22.04.

First, comment out jax, tensorflow and dmcgym from the ```requirements.txt``` so that it looks like follows:
```txt
pip
# jax[cuda]
# --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
flax
tqdm
gym
# tensorflow
tensorflow_probability
ml_collections
wandb
dm_control
# dmcgym @ git+https://github.com/ikostrikov/dmcgym
pybullet
attrs
filterpy
``` 

Then, create a conda environment with python 3.9 as follows:
```bash
conda create -n witp python==3.9
conda activate witp
```
Then, install jax with cuda again so that it can recognize the GPU:
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then, install tensorflow according to the [official instructions](https://www.tensorflow.org/install/pip):
```bash
pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
pip install -U tensorflow[and-cuda]
```

At this point, it's worth checking whether both tensorflow and jax can recognize the gpu. Run ```python```, and then run the following commands:
```python
from jax.lib import xla_bridge
import tensorflow as tf
print(xla_bridge.get_backend().platform)   ## This should print "gpu"
print(tf.config.list_physical_devices('GPU'))  ## This should print a list of your GPUs
```

Then, install the walk_in_the_park repo as follows:
```bash
pip install -r requirements.txt
```

Finally, clone the dmcgym package either inside or outside this directory from ```https://github.com/ikostrikov/dmcgym```. Once it is cloned, go into the ```requirements.txt``` file and remove the mujoco gym dependency so that it looks like follows:

```
gym >= 0.21.0, < 0.24.1
numpy >= 1.20.2
dm_control >= 1.0.0
```
Then, install dmcgym by running 
```bash
cd ~/path/to/dmcgym
pip install -e .
```

Then, I needed to modify the file so that the saved_checkpoint directory is global (orbax required this)

I also needed to install the following conda package to be able to render mujoco:
```bash
conda install -c conda-forge libstdcxx-ng
```
