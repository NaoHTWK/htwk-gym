# Deploy on Booster Robot

This directory contains deployment scripts and tools for running trained policies on the Booster robot, including real-time observation editing capabilities.

## Installation

Follow these steps to set up your environment:

1. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

2. Install Booster Robotic SDK:

    Refer to the [Booster Robotics SDK Guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-WDzedC8AiovU8gxSjeGcQ5CInSf) and ensure you complete the section on [Compile Sample Programs and Install Python SDK](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-EI5fdtSucoJWO4xd49QcE5CInSf).

3. Install Streamlit (for observation editor):

    ```sh
    $ pip install streamlit
    ```

## Deployment Scripts

### 1. Base Walk Deployment (`deploy_base_walk.py`)

This script deploys the base walking policy trained for basic locomotion.

**Features:**
- Implements basic walking gait control
- Uses the standard Policy class from `utils.policy`
- Provides real-time robot control via Booster SDK
- Supports remote control commands (vx, vy, vyaw)

**Usage:**
```sh
$ python deploy_base_walk.py --config=Base_Walk.yaml --net=127.0.0.1
```

**Parameters:**
- `--config`: Configuration file name (must be in `configs/` folder)
- `--net`: Network interface for SDK communication (default: `127.0.0.1`)

### 2. Parameter Walk Deployment (`deploy_parameter_walk.py`)

This script deploys the parameterized walking policy with enhanced observation control capabilities.

**Features:**
- Implements parameterized walking with adjustable gait parameters
- Uses the Thomas Policy class from `utils.policy_thomas`
- Supports real-time observation value modification (obs[9-15])
- Enhanced control over gait frequency, foot yaw, body orientation, and foot positioning
- Compatible with the Streamlit observation editor

**Usage:**
```sh
$ python deploy_parameter_walk.py --config=Parameter_Walk.yaml --net=127.0.0.1
```

**Parameters:**
- `--config`: Configuration file name (must be in `configs/` folder)
- `--net`: Network interface for SDK communication (default: `127.0.0.1`)

## Streamlit Observation Editor

The `streamlit_observation_editor.py` provides a web-based interface for real-time control of robot parameters during deployment.

### Features

- **Real-time Control**: Modify observation values 9-15 and walk commands (vx, vy, vyaw) while the robot is running
- **Live Status Monitoring**: View current parameter values and robot status
- **Configuration Management**: Load different configuration files dynamically
- **Export Capabilities**: Download current parameter settings as JSON
- **Intuitive Interface**: Slider-based controls with parameter descriptions

### Controllable Parameters

**Observation Values (obs[9-15]):**
- `obs[9]`: Gait Frequency - Controls walking speed and rhythm
- `obs[10]`: Foot Yaw (Left) - Left foot orientation adjustment
- `obs[11]`: Foot Yaw (Right) - Right foot orientation adjustment  
- `obs[12]`: Body Pitch Target - Forward/backward body lean
- `obs[13]`: Body Roll Target - Left/right body lean
- `obs[14]`: Feet Offset X - Forward/backward foot positioning
- `obs[15]`: Feet Offset Y - Left/right foot positioning

**Walk Commands:**
- `vx`: Forward/backward velocity (-1.0 to 1.0)
- `vy`: Lateral velocity (-1.0 to 1.0) 
- `vyaw`: Yaw rotation velocity (-3.0 to 3.0)

### Usage

1. **Start the deployment script first:**
   ```sh
   $ python deploy_parameter_walk.py --config=Parameter_Walk.yaml
   ```

2. **Launch the Streamlit editor:**
   ```sh
   $ streamlit run streamlit_observation_editor.py
   ```

3. **Access the web interface:**
   - Open your browser to `http://<robot_ip>:8501`
   - Select the appropriate configuration file
   - Use sliders to adjust parameters in real-time
   - Monitor live status and current values

### Integration

The observation editor communicates with the deployment script through a shared observation controller (`utils.observation_controller`), enabling seamless real-time parameter adjustment without interrupting robot operation.

## Complete Usage Workflow

1. **Prepare the robot:**
   - **Intel Board (Recommended):** The easiest way to deploy is directly on the robot's Intel board. This eliminates network latency and provides the most stable connection.
   - **Simulation:** Set up simulation using [Webots](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-IsE9d2DrIow8tpxCBUUcogdwn5d) or [Isaac Sim](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-Jczjd4UKMou7QlxjvJ4c9NNfnwb)
   - **Real World:** Power on robot, switch to PREP Mode, place in stable standing position

2. **Choose deployment method:**
   
   **For basic walking:**
   ```sh
   $ python deploy_base_walk.py --config=Base_Walk.yaml
   ```
   
   **For parameterized walking with real-time control:**
   ```sh
   $ python deploy_parameter_walk.py --config=Parameter_Walk.yaml
   $ streamlit run streamlit_observation_editor.py
   ```

3. **Control the robot:**
   - Use keyboard controls (if implemented) for basic movement
   - Use Streamlit interface for fine-tuned parameter adjustment
   - Monitor robot status and performance

4. **Exit safely:**
   - Switch robot back to PREP Mode before terminating
   - Use Ctrl+C to stop deployment scripts gracefully

## Configuration Files

Configuration files are located in the `configs/` directory:
- `Base_Walk.yaml`: Basic walking policy configuration
- `Parameter_Walk.yaml`: Parameterized walking policy configuration

Each configuration file contains:
- Policy parameters and model paths
- Robot mechanical parameters
- Control gains and limits
- Normalization factors for observations

## Troubleshooting

- **Connection Issues**: Verify network interface (`--net` parameter) matches robot setup
- **Policy Loading**: Ensure model files exist in `models/` directory
- **Streamlit Issues**: Check that `live_observation_values.json` is being created
- **Robot Safety**: Always switch to PREP Mode before program termination