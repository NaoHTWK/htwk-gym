# HTWK Gym

HTWK Gym is an advanced reinforcement learning (RL) framework for humanoid robot locomotion, developed by [HTWK Robots](https://robots.htwk-leipzig.de/startseite). Built upon the foundation of [Booster Gym](https://github.com/BoosterRobotics/booster_gym/tree/main), HTWK Gym extends the original framework with significant enhancements for multi-robot support, advanced locomotion tasks, and improved research capabilities.

[![real_T1_deploy](https://obs-cdn.boosterobotics.com/rl_deploy_demo_video_v3.gif)](https://obs-cdn.boosterobotics.com/rl_deploy_demo_video.mp4)

## Key Features

- **Multi-Robot Platform Support**: Comprehensive support for both Booster T1 and K1 humanoid robots with specialized configurations
- **Advanced Task Framework**: Sophisticated locomotion tasks including parameterized walking, ball-kicking behaviors, and adaptive gait control
- **Enhanced Research Capabilities**: Improved logging, hierarchical organization, and extensive domain randomization for robust sim-to-real transfer
- **Complete Training-to-Deployment Pipeline**: Full support for training, evaluating, and deploying policies in simulation and on real robots
- **Multi-Format Export**: Support for PyTorch JIT, TensorFlow Lite, and ONNX model formats with optional quantization
- **Real-Time Deployment**: Live robot control with Streamlit-based observation editor for parameterized walking
- **Pre-trained Models**: Ready-to-use trained policies for immediate deployment and testing
- **Flexible Architecture**: Easily extensible framework for custom environments, algorithms, and robot platforms
- **Research-Grade Tools**: Advanced reward engineering, curriculum learning, and comprehensive evaluation metrics

## HTWK Gym Capabilities

HTWK Gym provides a comprehensive research platform for humanoid robot locomotion with the following workflow:

1. **Training**: 

    - Train reinforcement learning policies using Isaac Gym with parallelized environments.

2. **Playing**:

    - **In-Simulation Testing**: Evaluate the trained policy in the same environment with training to ensure it behaves as expected.
    - **Cross-Simulation Testing**: Test the policy in MuJoCo to verify its generalization across different environments.

3. **Deployment**:

    - **Model Export**: Export the trained policy from `*.pth` to a JIT-optimized `*.pt` format for efficient deployment
    - **TensorFlow Lite Export**: Convert models to TensorFlow Lite format for mobile and embedded deployment with optional quantization
    - **Real-Time Robot Control**: Deploy policies directly to physical robots with live parameter adjustment
    - **Streamlit Observation Editor**: Web-based interface for real-time control of gait parameters during deployment
    - **Pre-trained Model Library**: Ready-to-use trained policies for immediate testing and deployment

## Supported Robot Platforms & Tasks

HTWK Gym supports multiple humanoid robot platforms with specialized task configurations:

### T1 Robot Platform

#### 1. BaseWalk (`T1/BaseWalk`)
- **Description**: Basic bipedal locomotion with velocity tracking
- **Features**: 
  - Linear and angular velocity command tracking
  - Terrain adaptation with randomized height variations
  - Robust walking on various surfaces
  - 47-dimensional observation space
- **Use Case**: Foundation for stable walking behaviors
- **Configuration**: `envs/T1/Base_Walk.yaml`

#### 2. ParameterWalk (`T1/ParameterWalk`)
- **Description**: Advanced parameterized walking with fine-grained control
- **Features**:
  - 10-dimensional command space including:
    - Linear velocities (x, y)
    - Angular velocity (yaw)
    - Gait frequency control
    - Individual foot yaw control (left/right)
    - Body pitch and roll targets
    - Feet offset targets (x, y)
  - Enhanced reward structure for precise control
  - 54-dimensional observation space
- **Use Case**: Precise gait control and complex locomotion patterns
- **Configuration**: `envs/T1/Parameter_Walk.yaml`

#### 3. Kicking (`T1/Kicking`)
- **Description**: Ball-kicking behavior with target-based rewards
- **Features**:
  - Ball interaction and kicking mechanics
  - Target-based ball velocity rewards
  - Body alignment for optimal kicking
  - Ball position tracking and control
  - 44-dimensional observation space with 20 privileged observations
- **Use Case**: Soccer-like behaviors and object manipulation
- **Configuration**: `envs/T1/Kicking.yaml`

### K1 Robot Platform

#### 1. ParameterWalk (`K1/ParameterWalk`)
- **Description**: Parameterized walking for the K1 robot platform
- **Features**:
  - Similar parameterized control as T1 ParameterWalk
  - Optimized for K1 robot dimensions and dynamics
  - Lower center of mass and adjusted control parameters
  - 54-dimensional observation space
- **Use Case**: K1-specific locomotion and gait control
- **Configuration**: `envs/K1/Parameter_Walk.yaml`

## Installation

HTWK Gym uses Python virtual environments for flexible dependency management. Follow these steps to set up your environment:

1. Create a virtual environment with Python 3.8:

    ```sh
    $ python3.8 -m venv venv_isaac
    $ source venv_isaac/bin/activate
    ```

2. Install PyTorch with CUDA support:

    ```sh
    $ pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
    ```

3. Install Isaac Gym

    Download Isaac Gym from [NVIDIA's website](https://developer.nvidia.com/isaac-gym/download).

    Extract and install:

    ```sh
    $ tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    $ cd isaacgym/python
    $ pip install -e .
    ```

4. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

5. (Optional) Install TensorFlow Lite export dependencies:

    ```sh
    $ pip install tensorflow==2.16.1 onnx onnx-tf
    ```

    This enables TensorFlow Lite model export for mobile and embedded deployment.

## Usage

### 1. Training

HTWK Gym supports multiple robot platforms and task types. To start training a policy:

```sh
$ python train.py --task=T1/BaseWalk
```

#### Available Task Options:
- **T1 Platform**: `T1/BaseWalk`, `T1/ParameterWalk`, `T1/Kicking`
- **K1 Platform**: `K1/ParameterWalk`

Training logs and saved models will be stored in `logs/<robot_type>/<task_name>/<timestamp>/`.

#### Configurations

Training settings are loaded from `envs/<task>.yaml`. You can also override config values using command-line arguments:

- `--checkpoint`: Path of the model checkpoint to load (set to `-1` to use the most recent model).
- `--num_envs`: Number of environments to create.
- `--headless`: Run headless without creating a viewer window.
- `--sim_device`: Device for physics simulation (e.g., `cuda:0`, `cpu`). 
- `--rl_device`: Device for the RL algorithm (e.g., `cuda:0`, `cpu`). 
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.

To add a new task, create a config file in `envs/` and register the environment in `envs/__init__.py`.

#### Progress Tracking

To visualize training progress with [TensorBoard](https://www.tensorflow.org/tensorboard), run:

```sh
$ tensorboard --logdir logs
```

This will show logs organized by robot type and task name in the hierarchical structure.

To use [Weights & Biases](https://wandb.ai/) for tracking, log in first:

```sh
$ wandb login
```

You can disable W&B tracking by setting `use_wandb` to `false` in the config file.

---

### 2. Playing

#### In-Simulation Testing

To test the trained policy in Isaac Gym, run:

```sh
$ python play.py --task=T1/BaseWalk --checkpoint=-1
```

Videos of the evaluation are automatically saved in `videos/<date-time>.mp4`. You can disable video recording by setting `record_video` to `false` in the config file.

#### Cross-Simulation Testing

To test the policy in MuJoCo, run:

```sh
$ python play_mujoco.py --task=T1/BaseWalk --checkpoint=-1
```

---

### 3. Deployment

HTWK Gym supports multiple deployment formats for different target platforms:

#### PyTorch JIT Export
For standard PyTorch deployment:
```sh
$ python export_model.py --task=T1/BaseWalk --checkpoint=-1
```

#### TensorFlow Lite Export
For deployment in the HTWK Firmware or other tflite based executers:
```sh
# Basic TFLite export
$ python export_tflite.py --task=T1/BaseWalk --checkpoint=-1
```

#### Real-Time Robot Deployment
For direct deployment to physical robots with live parameter control:



**Deployment Options:**
- **Intel Board (Recommended)**: Deploy directly on the robot's Intel board for the easiest setup with no network latency
- **Simulation**: Use [Webots](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-IsE9d2DrIow8tpxCBUUcogdwn5d) or [Isaac Sim](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-Jczjd4UKMou7QlxjvJ4c9NNfnwb) for simulation testing



**Transfer Files to Robot:**

3. **Copy deploy folder to robot (Intel Board recommanded):**
   ```sh
   $ scp -r deploy/ <username>@<robot_ip>:/<destination>/
   ```

**Setup on Robot:**

4. **SSH into robot:**
   ```sh
   $ ssh <username>@<robot_ip>
   ```

5. **Navigate to deployment directory and create virtual environment:**
   ```sh
   $ cd /<destination>/deploy
   $ python3 -m venv venv
   $ source venv/bin/activate
   ```

6. **Install dependencies in virtual environment:**
   ```sh
   $ pip install -r requirements.txt
   ```

7. **Install Booster Robotics SDK (if not already installed):**
   
   Follow the [Booster Robotics SDK Guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f) and complete the section on [Compile Sample Programs and Install Python SDK](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-EI5fdtSucoJWO4xd49QcE5CInSf).

**Prepare Robot:**

8. **Before starting deployment:**
   - Power on the robot
   - Switch robot to **PREP Mode**
   - Place robot in a stable standing position in an open area

**Execute Deployment:**

9. **Activate virtual environment and start deployment scripts on robot:**
   ```sh
   # Activate the venv (if starting a new SSH session)
   $ cd /<destination>/deploy
   $ source venv/bin/activate
   
   # Basic walking deployment
   $ python deploy_base_walk.py --config=Base_Walk.yaml --net=127.0.0.1

   # OR Parameterized walking with real-time control
   $ python deploy_parameter_walk.py --config=Parameter_Walk.yaml --net=127.0.0.1
   ```
   If you are not deploying on the Intel Board you need to set the ``--net`` to the correct address of the fast dds.

10. **Launch Streamlit observation editor on robot (for parameterized walking):**
   ```sh
   # Ensure venv is activated
   $ source venv/bin/activate
   $ streamlit run streamlit_observation_editor.py
   ```
   
11. **Access the control interface from your web browser:**
    - Open your browser on your development machine
    - Navigate to `http://<robot_ip>:8501` to access the real-time control interface
    - The Streamlit app runs on the robot and serves the web interface remotely

**Exit Safely:**

12. **To stop deployment:**
    - Press `Ctrl+C` to gracefully terminate deployment scripts
    - Switch robot back to **PREP Mode** before turning off or moving robot

#### Pre-trained Models
HTWK Gym includes pre-trained models in the `deploy/models/` directory:
- `base_walk.pt` - T1 robot base walking policy
- `parameter_walk.pt` - Parameterized walking policy
- Additional specialized models for different behaviors

After exporting the model, follow the steps in [Deploy on Booster Robot](deploy/README.md) to complete the deployment process.

## Real-Time Parameter Control

HTWK Gym features a Streamlit-based observation editor for live parameter adjustment during robot deployment. This allows real-time control of gait frequency, foot positioning, body orientation, and walk commands through a web interface.

**Key Features:**
- Live parameter control via web browser
- Real-time monitoring of robot status
- Intuitive slider-based interface
- Configuration management and export capabilities

For detailed usage instructions, see the [Deploy on Booster Robot](deploy/README.md) documentation.

## HTWK Gym Examples

### Basic Walking
```sh
# Train basic walking on T1 robot
$ python train.py --task=T1/BaseWalk --num_envs=4096

# Test basic walking
$ python play.py --task=T1/BaseWalk --checkpoint=-1
```

### Advanced Parameterized Walking
```sh
# Train parameterized walking with fine control
$ python train.py --task=T1/ParameterWalk --num_envs=4096

# Test parameterized walking
$ python play.py --task=T1/ParameterWalk --checkpoint=-1
```

### Ball Kicking Behavior
```sh
# Train ball kicking behavior
$ python train.py --task=T1/Kicking --num_envs=1024

# Test ball kicking
$ python play.py --task=T1/Kicking --checkpoint=-1
```

### K1 Robot Platform
```sh
# Train K1 parameterized walking
$ python train.py --task=K1/ParameterWalk --num_envs=4096

# Test K1 walking
$ python play.py --task=K1/ParameterWalk --checkpoint=-1
```

## Configuration Details

Each task has its own configuration file with specific parameters:

- **Observation Spaces**: Range from 44-54 dimensions depending on task complexity
- **Action Spaces**: 12-dimensional for all tasks (3 DOF per leg × 4 legs)
- **Command Spaces**: Vary from 0 (kicking) to 10 (parameterized walking) dimensions
- **Reward Functions**: Task-specific reward structures optimized for different behaviors
- **Terrain Types**: Support for both plane and trimesh terrains with randomization
- **Domain Randomization**: Extensive parameter randomization for sim-to-real transfer

## Contributing to HTWK Gym

HTWK Gym is designed to be easily extensible for research and development:

### Adding New Robot Platforms
1. Create robot-specific URDF files in `resources/<robot_name>/`
2. Implement task classes in `envs/<robot_name>/`
3. Add configuration files in `envs/<robot_name>/`
4. Register the robot in `envs/__init__.py`

### Adding New Tasks
1. Create a new task class inheriting from `BaseTask`
2. Implement the required methods for your specific behavior
3. Create a corresponding YAML configuration file
4. Register the task in `envs/__init__.py`
5. Update this README with task documentation

### Research Extensions
- Custom reward functions
- Domain randomization parameters
- Curriculum learning strategies
- Multi-agent scenarios

## License

HTWK Gym is developed by HTWK Robots and is based on the original Booster Gym framework. Please refer to the license terms provided with the Isaac Gym package for usage restrictions.

Pre-Trained Models are only allowed to use in a Robot Soccer Competition with the agreement of the HTWK Robots Team (For Contact, use robots@htwk-leipzig.de)
