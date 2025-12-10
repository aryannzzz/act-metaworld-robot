# ACT Virtual Implementation Plan - Part 2: SO101 Specific

## 📦 Stage 2: SO101 Simulation (Days 15-28)

### Goal
Adapt ACT to work with SO101 robot specifications in simulation before real deployment.

### Duration
**2 weeks** (Days 15-28)

---

### Understanding SO101 Robot

**SO101 (Stanford Operational Robot 101)** is a low-cost bimanual manipulation platform similar to ALOHA but with potential differences in:
- Robot arm kinematics
- Joint limits and ranges
- Camera configuration
- Gripper specifications
- Workspace dimensions

### Week 3: SO101 Model & Environment (Days 15-21)

#### Day 15-16: Create SO101 URDF/MuJoCo Model

**Option 1: If you have URDF**
```python
# Convert URDF to MuJoCo
import mujoco

def load_so101_urdf():
    """Load SO101 from URDF"""
    # Assuming you have SO101 URDF files
    urdf_path = "robots/so101/so101.urdf"
    
    # Convert to MuJoCo XML
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    return model

# Create MuJoCo scene with SO101
so101_mjcf = """
<mujoco model="so101_bimanual">
  <compiler angle="radian" meshdir="meshes/"/>
  
  <asset>
    <mesh name="base_link" file="base_link.stl"/>
    <mesh name="link1" file="link1.stl"/>
    <!-- Add all mesh files -->
  </asset>
  
  <worldbody>
    <light directional="true" pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    
    <!-- Left arm -->
    <body name="left_base" pos="-0.3 0 0.1">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>
      
      <!-- Add SO101 left arm kinematic chain -->
      <body name="left_link1" pos="0 0 0.1">
        <joint name="left_joint1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14"/>
        <geom type="cylinder" size="0.02 0.1" rgba="0.3 0.3 0.8 1"/>
        <!-- Continue chain... -->
      </body>
    </body>
    
    <!-- Right arm (mirror of left) -->
    <body name="right_base" pos="0.3 0 0.1">
      <!-- Similar structure -->
    </body>
    
    <!-- Table -->
    <body name="table" pos="0 0.5 0">
      <geom type="box" size="0.4 0.3 0.4" rgba="0.7 0.5 0.3 1"/>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Add actuators for all joints -->
    <motor name="left_motor1" joint="left_joint1" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <!-- ... -->
  </actuator>
  
  <!-- Cameras -->
  <worldbody>
    <camera name="front" pos="0.5 0.5 0.5" xyaxes="0 -1 0 0 0 1"/>
    <camera name="top" pos="0 0.5 1.0" xyaxes="1 0 0 0 1 0"/>
    <camera name="left_wrist" pos="-0.3 0.3 0.3" xyaxes="1 0 0 0 0 1"/>
    <camera name="right_wrist" pos="0.3 0.3 0.3" xyaxes="1 0 0 0 0 1"/>
  </worldbody>
</mujoco>
"""

# Save to file
with open('robots/so101/so101.xml', 'w') as f:
    f.write(so101_mjcf)
```

**Option 2: Use Existing Robot as Template**
```python
# If SO101 is similar to WidowX/ViperX (as used in ALOHA)
# You can start with those and modify

import mujoco
import dm_control

# Load WidowX as starting point
widowx_path = "robots/widowx/widowx.xml"
model = mujoco.MjModel.from_xml_path(widowx_path)

# Modify parameters to match SO101
def adapt_to_so101(model):
    """Modify robot parameters to match SO101 specs"""
    
    # Example: Adjust joint limits
    # model.jnt_range[joint_id] = [new_min, new_max]
    
    # Example: Adjust link lengths
    # model.body_pos[body_id] = new_position
    
    # Example: Adjust gripper dimensions
    # ...
    
    return model

so101_model = adapt_to_so101(model)
```

#### Day 17-18: Create SO101 Gym Environment

```python
# envs/so101_env.py

import gym
import numpy as np
import mujoco
from gym import spaces

class SO101Env(gym.Env):
    """Custom SO101 environment for ACT training"""
    
    def __init__(self, task='pick_place', render_mode='rgb_array'):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path('robots/so101/so101.xml')
        self.data = mujoco.MjData(self.model)
        
        # Define action and observation spaces
        # Action: target joint positions for both arms + grippers
        # SO101 specs: 6 DoF per arm + 1 gripper = 7 per arm = 14 total
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),  # Adjust based on SO101 specs
            dtype=np.float32
        )
        
        # Observation: images + proprioception
        self.observation_space = spaces.Dict({
            'images': spaces.Dict({
                'front': spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                'top': spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                'left_wrist': spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                'right_wrist': spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
            }),
            'joints': spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
        })
        
        # Camera IDs
        self.camera_ids = {
            'front': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'front'),
            'top': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'top'),
            'left_wrist': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'left_wrist'),
            'right_wrist': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'right_wrist'),
        }
        
        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        self.task = task
        self.steps = 0
        self.max_steps = 500
    
    def reset(self):
        """Reset environment"""
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize object positions (task-dependent)
        if self.task == 'pick_place':
            self._randomize_object_position()
        
        self.steps = 0
        
        # Step simulation to stabilize
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action"""
        # Map action to joint positions
        # SO101 uses position control
        target_positions = self._action_to_target_positions(action)
        
        # Apply action with PD control
        for i, target_pos in enumerate(target_positions):
            self.data.ctrl[i] = target_pos
        
        # Step simulation (10 steps per action for 50Hz control)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.steps += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check done
        done = self.steps >= self.max_steps or self._check_success()
        
        # Info
        info = {
            'success': self._check_success(),
            'final_distance': self._compute_distance_to_goal(),
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation"""
        obs = {
            'images': {},
            'joints': np.zeros(14),
        }
        
        # Render images from all cameras
        for cam_name, cam_id in self.camera_ids.items():
            self.renderer.update_scene(self.data, camera=cam_id)
            img = self.renderer.render()
            obs['images'][cam_name] = img.copy()
        
        # Get joint positions
        # Assuming first 7 joints are left arm, next 7 are right arm
        obs['joints'][:7] = self.data.qpos[:7]
        obs['joints'][7:] = self.data.qpos[7:14]
        
        return obs
    
    def _action_to_target_positions(self, action):
        """Convert normalized action to target joint positions"""
        # action is in [-1, 1], convert to actual joint range
        
        # Get joint limits from model
        joint_ranges = self.model.jnt_range[:14]  # [min, max] for each joint
        
        # Map [-1, 1] to [min, max]
        target_positions = np.zeros(14)
        for i in range(14):
            min_pos, max_pos = joint_ranges[i]
            # action[i] ∈ [-1, 1] -> target ∈ [min_pos, max_pos]
            target_positions[i] = (action[i] + 1) * (max_pos - min_pos) / 2 + min_pos
        
        return target_positions
    
    def _randomize_object_position(self):
        """Randomize object position for pick-place task"""
        # Find object body ID
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        
        # Randomize within workspace
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(0.3, 0.5)
        z = 0.02  # On table
        
        # Set object position
        self.data.qpos[object_id*7:object_id*7+3] = [x, y, z]
    
    def _compute_reward(self):
        """Compute reward (sparse for now)"""
        if self._check_success():
            return 1.0
        return 0.0
    
    def _check_success(self):
        """Check if task is successful"""
        # Task-specific success criteria
        # For pick-place: object within threshold of goal
        
        distance = self._compute_distance_to_goal()
        return distance < 0.05  # 5cm threshold
    
    def _compute_distance_to_goal(self):
        """Compute distance from object to goal"""
        # Get object position
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        object_pos = self.data.xpos[object_id]
        
        # Goal position (e.g., on shelf)
        goal_pos = np.array([0.0, 0.5, 0.3])
        
        distance = np.linalg.norm(object_pos - goal_pos)
        return distance
    
    def render(self, mode='rgb_array'):
        """Render environment"""
        if mode == 'rgb_array':
            self.renderer.update_scene(self.data, camera='front')
            return self.renderer.render()
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")
    
    def close(self):
        """Close environment"""
        pass

# Register environment
gym.register(
    id='SO101-PickPlace-v0',
    entry_point='envs.so101_env:SO101Env',
    max_episode_steps=500,
)
```

#### Day 19-20: Collect SO101 Demonstrations

**Option 1: Scripted Policy**
```python
# scripts/collect_so101_demos.py

import numpy as np
import gym
import envs.so101_env

def create_scripted_policy_so101():
    """Create a simple scripted policy for SO101"""
    
    class ScriptedSO101Policy:
        def __init__(self):
            self.phase = 0
            self.target_object_pos = None
            self.target_goal_pos = np.array([0.0, 0.5, 0.3])
        
        def reset(self):
            self.phase = 0
            self.target_object_pos = None
        
        def get_action(self, obs):
            """Simple pick and place policy"""
            
            # Get current state
            joints = obs['joints']
            left_ee_pos = self._forward_kinematics(joints[:7])
            right_ee_pos = self._forward_kinematics(joints[7:])
            
            # Get object position from state (if observable)
            # In real scenario, would use vision
            object_pos = self._get_object_position(obs)
            
            # State machine
            if self.phase == 0:
                # Phase 0: Move to pre-grasp
                target = object_pos + np.array([0, 0, 0.1])
                action = self._move_to_target(left_ee_pos, target, joints[:7])
                
                if np.linalg.norm(left_ee_pos - target) < 0.02:
                    self.phase = 1
            
            elif self.phase == 1:
                # Phase 1: Move down to grasp
                target = object_pos + np.array([0, 0, 0.01])
                action = self._move_to_target(left_ee_pos, target, joints[:7])
                
                if np.linalg.norm(left_ee_pos - target) < 0.01:
                    self.phase = 2
            
            elif self.phase == 2:
                # Phase 2: Close gripper
                action = joints.copy()
                action[6] = 1.0  # Close left gripper
                self.phase = 3
            
            elif self.phase == 3:
                # Phase 3: Lift object
                target = object_pos + np.array([0, 0, 0.2])
                action = self._move_to_target(left_ee_pos, target, joints[:7])
                action[6] = 1.0  # Keep gripper closed
                
                if np.linalg.norm(left_ee_pos - target) < 0.02:
                    self.phase = 4
            
            elif self.phase == 4:
                # Phase 4: Move to goal
                target = self.target_goal_pos + np.array([0, 0, 0.1])
                action = self._move_to_target(left_ee_pos, target, joints[:7])
                action[6] = 1.0
                
                if np.linalg.norm(left_ee_pos - target) < 0.02:
                    self.phase = 5
            
            elif self.phase == 5:
                # Phase 5: Place object
                target = self.target_goal_pos
                action = self._move_to_target(left_ee_pos, target, joints[:7])
                action[6] = 1.0
                
                if np.linalg.norm(left_ee_pos - target) < 0.01:
                    self.phase = 6
            
            else:
                # Phase 6: Open gripper and done
                action = joints.copy()
                action[6] = -1.0  # Open gripper
            
            return action
        
        def _forward_kinematics(self, joint_positions):
            """Compute end-effector position from joint angles"""
            # TODO: Implement actual FK for SO101
            # For now, placeholder
            return np.array([0, 0, 0])
        
        def _get_object_position(self, obs):
            """Get object position (from state or vision)"""
            # TODO: Extract from observation
            # For sim, could use ground truth
            return np.array([0, 0.4, 0.02])
        
        def _move_to_target(self, current_pos, target_pos, current_joints):
            """Inverse kinematics to move to target"""
            # TODO: Implement actual IK for SO101
            # For now, use simple proportional control
            error = target_pos - current_pos
            joint_delta = error * 0.1  # Proportional gain
            
            action = current_joints.copy()
            action[:3] += joint_delta  # Simplified
            return action
    
    return ScriptedSO101Policy()

def collect_demonstrations():
    env = gym.make('SO101-PickPlace-v0')
    policy = create_scripted_policy_so101()
    
    demonstrations = []
    num_demos = 50
    successes = 0
    
    while len(demonstrations) < num_demos:
        obs = env.reset()
        policy.reset()
        
        trajectory = {
            'observations': {'images': {cam: [] for cam in obs['images'].keys()},
                           'joints': []},
            'actions': [],
            'success': False,
        }
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            action = policy.get_action(obs)
            
            # Record
            for cam_name, img in obs['images'].items():
                trajectory['observations']['images'][cam_name].append(img)
            trajectory['observations']['joints'].append(obs['joints'])
            trajectory['actions'].append(action)
            
            # Step
            obs, reward, done, info = env.step(action)
            steps += 1
        
        if info['success']:
            trajectory['success'] = True
            demonstrations.append(trajectory)
            successes += 1
            print(f"✓ Success {successes}/{num_demos}")
    
    # Save
    save_demonstrations_hdf5(demonstrations, 'data/so101_demos.hdf5')
    
    return demonstrations
```

**Option 2: Teleoperation (Better for real SO101 later)**
```python
# scripts/teleop_so101.py

import gym
import pygame
import numpy as np

class SO101Teleop:
    """Simple keyboard/gamepad teleoperation for SO101"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("SO101 Teleoperation")
        
        # Try to initialize gamepad
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"✓ Gamepad connected: {self.joystick.get_name()}")
        else:
            self.joystick = None
            print("⚠ No gamepad found, using keyboard")
    
    def get_action(self, current_joints):
        """Get action from user input"""
        action = current_joints.copy()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
        
        if self.joystick:
            # Gamepad control
            # Left stick: left arm XY
            # Right stick: right arm XY
            # Triggers: left/right arm Z
            # Bumpers: grippers
            
            left_x = self.joystick.get_axis(0)  # Left stick X
            left_y = self.joystick.get_axis(1)  # Left stick Y
            right_x = self.joystick.get_axis(2)  # Right stick X
            right_y = self.joystick.get_axis(3)  # Right stick Y
            
            # Map to joint velocities (simple mapping)
            action[0] += left_x * 0.01
            action[1] += left_y * 0.01
            action[7] += right_x * 0.01
            action[8] += right_y * 0.01
            
            # Grippers
            if self.joystick.get_button(4):  # Left bumper
                action[6] = 1.0  # Close left gripper
            if self.joystick.get_button(5):  # Right bumper
                action[13] = 1.0  # Close right gripper
        
        else:
            # Keyboard control
            keys = pygame.key.get_pressed()
            
            # WASD: left arm
            if keys[pygame.K_w]: action[1] += 0.01
            if keys[pygame.K_s]: action[1] -= 0.01
            if keys[pygame.K_a]: action[0] -= 0.01
            if keys[pygame.K_d]: action[0] += 0.01
            
            # Arrow keys: right arm
            if keys[pygame.K_UP]: action[8] += 0.01
            if keys[pygame.K_DOWN]: action[8] -= 0.01
            if keys[pygame.K_LEFT]: action[7] -= 0.01
            if keys[pygame.K_RIGHT]: action[7] += 0.01
            
            # Grippers
            if keys[pygame.K_q]: action[6] = 1.0
            if keys[pygame.K_e]: action[6] = -1.0
            if keys[pygame.K_u]: action[13] = 1.0
            if keys[pygame.K_o]: action[13] = -1.0
        
        return action

def collect_teleop_demonstrations():
    env = gym.make('SO101-PickPlace-v0')
    teleop = SO101Teleop()
    
    demonstrations = []
    
    print("=== Teleoperation Mode ===")
    print("Gamepad: Use sticks to control arms, bumpers for grippers")
    print("Keyboard: WASD (left arm), Arrows (right arm), Q/E and U/O (grippers)")
    print("Press ESC to finish current demo, SPACE to save and start new")
    
    while True:
        obs = env.reset()
        trajectory = {
            'observations': {'images': {}, 'joints': []},
            'actions': [],
        }
        
        done = False
        while not done:
            # Render
            img = env.render()
            # Display with pygame (simplified)
            
            # Get action from user
            action = teleop.get_action(obs['joints'])
            if action is None:
                break
            
            # Record
            for cam_name, img in obs['images'].items():
                if cam_name not in trajectory['observations']['images']:
                    trajectory['observations']['images'][cam_name] = []
                trajectory['observations']['images'][cam_name].append(img)
            
            trajectory['observations']['joints'].append(obs['joints'])
            trajectory['actions'].append(action)
            
            # Step
            obs, reward, done, info = env.step(action)
        
        # Ask to save
        print(f"Demo complete. Success: {info.get('success', False)}")
        save = input("Save this demonstration? (y/n): ")
        
        if save.lower() == 'y':
            demonstrations.append(trajectory)
            print(f"✓ Saved ({len(demonstrations)} demos total)")
        
        cont = input("Continue collecting? (y/n): ")
        if cont.lower() != 'y':
            break
    
    # Save all
    save_demonstrations_hdf5(demonstrations, 'data/so101_teleop_demos.hdf5')
    
    return demonstrations
```

#### Day 21: Test SO101 Environment

```python
# scripts/test_so101_env.py

import gym
import envs.so101_env

def test_environment():
    """Test SO101 environment"""
    print("=== Testing SO101 Environment ===\n")
    
    # Create environment
    env = gym.make('SO101-PickPlace-v0')
    
    # Test reset
    print("1. Testing reset...")
    obs = env.reset()
    print(f"✓ Observation keys: {obs.keys()}")
    print(f"✓ Image shapes: {[img.shape for img in obs['images'].values()]}")
    print(f"✓ Joints shape: {obs['joints'].shape}")
    
    # Test step
    print("\n2. Testing step...")
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Info: {info}")
    
    # Test episode
    print("\n3. Testing full episode with random actions...")
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < 100:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"✓ Episode complete")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Success: {info.get('success', False)}")
    
    # Test rendering
    print("\n4. Testing rendering...")
    img = env.render()
    print(f"✓ Rendered image shape: {img.shape}")
    
    print("\n=== All tests passed! ===")

if __name__ == '__main__':
    test_environment()
```

### Week 4: Training on SO101 (Days 22-28)

#### Day 22-24: Train ACT on SO101 Simulation

```bash
# Use same training scripts, just different data and config

# Train standard ACT on SO101
python scripts/train_standard.py \
    --config configs/so101_standard_act.yaml \
    --data_path data/so101_demos.hdf5 \
    --exp_name so101_standard_act

# Train modified ACT on SO101
python scripts/train_modified.py \
    --config configs/so101_modified_act.yaml \
    --data_path data/so101_demos.hdf5 \
    --exp_name so101_modified_act
```

**SO101-specific config:**
```yaml
# configs/so101_standard_act.yaml

env:
  name: "SO101-PickPlace-v0"
  max_steps: 500
  camera_names: ["front", "top", "left_wrist", "right_wrist"]
  image_size: [480, 640]

model:
  joint_dim: 14  # SO101 specific
  action_dim: 14
  hidden_dim: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 7
  feedforward_dim: 3200
  latent_dim: 32
  dropout: 0.1

chunking:
  chunk_size: 100
  query_frequency: 1
  temporal_ensemble_weight: 0.01

training:
  batch_size: 8
  learning_rate: 1e-5
  num_epochs: 2000
  beta: 10.0
  grad_clip: 1.0

logging:
  log_freq: 10
  save_freq: 100
  eval_freq: 500
  use_wandb: true
  wandb_project: "act-so101-sim"
```

#### Day 25-26: Evaluate and Compare

```python
# scripts/evaluate_so101.py

# Same as MetaWorld evaluation, just different environment

def evaluate_on_so101():
    import gym
    import envs.so101_env
    
    env = gym.make('SO101-PickPlace-v0')
    
    # Load both models
    standard_model = load_checkpoint('experiments/so101_standard_act/checkpoints/best.pth')
    modified_model = load_checkpoint('experiments/so101_modified_act/checkpoints/best.pth')
    
    # Evaluate both
    print("=== Evaluating Standard ACT ===")
    standard_metrics = evaluate_policy(env, standard_model, num_episodes=100)
    
    print("\n=== Evaluating Modified ACT ===")
    modified_metrics = evaluate_policy(env, modified_model, num_episodes=100)
    
    # Compare
    print("\n=== Comparison ===")
    print(f"Standard ACT Success Rate: {standard_metrics['success_rate']:.2f}%")
    print(f"Modified ACT Success Rate: {modified_metrics['success_rate']:.2f}%")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        standard_metrics['successes'],
        modified_metrics['successes']
    )
    print(f"\nt-test: t={t_stat:.3f}, p={p_value:.3f}")
    
    if p_value < 0.05:
        print("✓ Statistically significant difference!")
    else:
        print("✗ No significant difference")
```

#### Day 27-28: Visualization and Analysis

```python
# scripts/analyze_so101_results.py

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    """Plot comprehensive comparison"""
    
    # Load training logs
    standard_logs = load_wandb_run('so101_standard_act')
    modified_logs = load_wandb_run('so101_modified_act')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training loss
    axes[0,0].plot(standard_logs['train/loss'], label='Standard')
    axes[0,0].plot(modified_logs['train/loss'], label='Modified')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Total Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 2. Reconstruction loss
    axes[0,1].plot(standard_logs['train/recon'], label='Standard')
    axes[0,1].plot(modified_logs['train/recon'], label='Modified')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Reconstruction Loss')
    axes[0,1].set_title('Reconstruction Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 3. KL divergence
    axes[0,2].plot(standard_logs['train/kl'], label='Standard')
    axes[0,2].plot(modified_logs['train/kl'], label='Modified')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('KL Divergence')
    axes[0,2].set_title('KL Divergence')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # 4. Success rate comparison (bar plot)
    success_rates = [
        standard_metrics['success_rate'],
        modified_metrics['success_rate']
    ]
    axes[1,0].bar(['Standard', 'Modified'], success_rates)
    axes[1,0].set_ylabel('Success Rate (%)')
    axes[1,0].set_title('Success Rate Comparison')
    axes[1,0].set_ylim([0, 100])
    axes[1,0].grid(True, axis='y')
    
    # 5. Episode length distribution
    axes[1,1].hist(standard_metrics['episode_lengths'], alpha=0.5, label='Standard', bins=20)
    axes[1,1].hist(modified_metrics['episode_lengths'], alpha=0.5, label='Modified', bins=20)
    axes[1,1].set_xlabel('Episode Length')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Episode Length Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 6. Final distance to goal
    axes[1,2].boxplot([
        standard_metrics['final_distances'],
        modified_metrics['final_distances']
    ], labels=['Standard', 'Modified'])
    axes[1,2].set_ylabel('Distance to Goal (m)')
    axes[1,2].set_title('Final Distance Distribution')
    axes[1,2].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/so101_comparison.png', dpi=300)
    plt.show()
    
    print("✓ Plots saved to results/so101_comparison.png")
```

---

## 📦 Stage 3: Sim-to-Real Preparation (Days 29-35)

### Goal
Make the policy robust enough to transfer to real hardware.

### Duration
**1 week** (Days 29-35)

### Key Techniques

#### Day 29-30: Domain Randomization

```python
# envs/so101_randomized.py

class SO101RandomizedEnv(SO101Env):
    """SO101 with domain randomization for sim-to-real transfer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Randomization ranges
        self.randomization_config = {
            # Physics
            'joint_friction': (0.8, 1.2),
            'joint_damping': (0.8, 1.2),
            'object_mass': (0.8, 1.2),
            'object_friction': (0.5, 2.0),
            
            # Visual
            'light_pos': 0.3,  # meters
            'light_ambient': (0.3, 0.7),
            'object_color': True,
            'table_color': True,
            
            # Noise
            'action_noise': 0.01,
            'obs_noise': 0.005,
            'camera_pos': 0.02,  # meters
        }
    
    def reset(self):
        """Reset with randomization"""
        # Randomize physics
        self._randomize_physics()
        
        # Randomize visual
        self._randomize_visual()
        
        # Standard reset
        obs = super().reset()
        
        return obs
    
    def _randomize_physics(self):
        """Randomize physics parameters"""
        config = self.randomization_config
        
        # Joint friction
        for i in range(self.model.njnt):
            min_f, max_f = config['joint_friction']
            scale = np.random.uniform(min_f, max_f)
            self.model.dof_frictionloss[i] *= scale
        
        # Joint damping
        for i in range(self.model.njnt):
            min_d, max_d = config['joint_damping']
            scale = np.random.uniform(min_d, max_d)
            self.model.dof_damping[i] *= scale
        
        # Object mass
        object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'object'
        )
        min_m, max_m = config['object_mass']
        scale = np.random.uniform(min_m, max_m)
        self.model.body_mass[object_body_id] *= scale
        
        # Object friction
        min_f, max_f = config['object_friction']
        friction = np.random.uniform(min_f, max_f)
        # Find object geom and set friction
        object_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'object'
        )
        self.model.geom_friction[object_geom_id, 0] = friction
    
    def _randomize_visual(self):
        """Randomize visual appearance"""
        config = self.randomization_config
        
        # Randomize lighting
        light_id = 0  # Assuming first light
        pos_range = config['light_pos']
        self.model.light_pos[light_id] += np.random.uniform(
            -pos_range, pos_range, size=3
        )
        
        min_amb, max_amb = config['light_ambient']
        ambient = np.random.uniform(min_amb, max_amb)
        self.model.light_ambient[light_id] = ambient
        
        # Randomize object color
        if config['object_color']:
            object_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, 'object'
            )
            random_color = np.random.uniform(0, 1, size=3)
            self.model.geom_rgba[object_geom_id, :3] = random_color
        
        # Randomize table color
        if config['table_color']:
            table_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, 'table'
            )
            random_color = np.random.uniform(0.3, 0.9, size=3)
            self.model.geom_rgba[table_geom_id, :3] = random_color
        
        # Randomize camera positions slightly
        for cam_id in range(self.model.ncam):
            pos_noise = np.random.uniform(
                -config['camera_pos'],
                config['camera_pos'],
                size=3
            )
            self.model.cam_pos[cam_id] += pos_noise
    
    def step(self, action):
        """Step with noise"""
        # Add action noise
        action_noise = np.random.normal(
            0,
            self.randomization_config['action_noise'],
            size=action.shape
        )
        noisy_action = action + action_noise
        
        # Standard step
        obs, reward, done, info = super().step(noisy_action)
        
        # Add observation noise
        obs_noise = np.random.normal(
            0,
            self.randomization_config['obs_noise'],
            size=obs['joints'].shape
        )
        obs['joints'] += obs_noise
        
        return obs, reward, done, info

# Register randomized environment
gym.register(
    id='SO101-PickPlace-Randomized-v0',
    entry_point='envs.so101_randomized:SO101RandomizedEnv',
    max_episode_steps=500,
)
```

#### Day 31-32: Retrain with Domain Randomization

```bash
# Collect new demonstrations with randomization
python scripts/collect_so101_demos.py \
    --env SO101-PickPlace-Randomized-v0 \
    --num_demos 50 \
    --save_path data/so101_randomized_demos.hdf5

# Train on randomized data
python scripts/train_standard.py \
    --config configs/so101_standard_act.yaml \
    --data_path data/so101_randomized_demos.hdf5 \
    --exp_name so101_standard_randomized

python scripts/train_modified.py \
    --config configs/so101_modified_act.yaml \
    --data_path data/so101_randomized_demos.hdf5 \
    --exp_name so101_modified_randomized
```

#### Day 33-34: Visual Domain Adaptation

```python
# data/visual_augmentation.py

import torch
import torchvision.transforms as transforms
import numpy as np

class VisualAugmentation:
    """Visual augmentations for sim-to-real transfer"""
    
    def __init__(self, config):
        self.config = config
        
        # Define augmentations
        self.augmentations = transforms.Compose([
            # Random crop and resize
            transforms.RandomResizedCrop(
                size=(480, 640),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1)
            ),
            
            # Color jitter
            transforms.ColorJitter(
                brightness=config.get('brightness', 0.2),
                contrast=config.get('contrast', 0.2),
                saturation=config.get('saturation', 0.2),
                hue=config.get('hue', 0.05)
            ),
            
            # Random horizontal flip (with probability)
            transforms.RandomHorizontalFlip(p=config.get('flip_prob', 0.0)),
            
            # Gaussian blur (simulates camera defocus)
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 2.0)
            ),
        ])
    
    def __call__(self, image):
        """Apply augmentation to image"""
        # Convert numpy to PIL
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply augmentations
        augmented = self.augmentations(image)
        
        # Convert back to numpy
        return np.array(augmented)

# Update dataset to use augmentation
class ACTDatasetAugmented(ACTDataset):
    def __init__(self, demonstrations, chunk_size=100, augment=True):
        super().__init__(demonstrations, chunk_size)
        
        self.augment = augment
        if augment:
            self.visual_aug = VisualAugmentation({
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.2,
                'hue': 0.05,
                'flip_prob': 0.0,  # Don't flip for manipulation
            })
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        if self.augment:
            # Augment images
            for cam_name in sample['images'].keys():
                img = sample['images'][cam_name]
                # Convert to numpy for augmentation
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # Augment
                img_aug = self.visual_aug(img_np)
                # Convert back to tensor
                img_aug = torch.from_numpy(img_aug).float() / 255.0
                img_aug = img_aug.permute(2, 0, 1)
                sample['images'][cam_name] = img_aug
        
        return sample
```

#### Day 35: Final Sim Evaluation & Preparation Checklist

```python
# scripts/final_sim_evaluation.py

def final_evaluation():
    """Comprehensive evaluation before real robot deployment"""
    
    print("=== Final Sim Evaluation ===\n")
    
    # 1. Evaluate on standard sim
    print("1. Standard simulation...")
    standard_env = gym.make('SO101-PickPlace-v0')
    standard_results = evaluate_policy(standard_env, model, num_episodes=100)
    
    # 2. Evaluate on randomized sim
    print("\n2. Randomized simulation...")
    random_env = gym.make('SO101-PickPlace-Randomized-v0')
    random_results = evaluate_policy(random_env, model, num_episodes=100)
    
    # 3. Stress test: extreme randomization
    print("\n3. Extreme randomization...")
    extreme_env = gym.make('SO101-PickPlace-Randomized-v0')
    extreme_env.randomization_config['joint_friction'] = (0.5, 2.0)
    extreme_env.randomization_config['object_mass'] = (0.5, 2.0)
    extreme_results = evaluate_policy(extreme_env, model, num_episodes=50)
    
    # 4. Different object positions
    print("\n4. Novel object positions...")
    novel_results = evaluate_on_novel_positions(model, num_episodes=50)
    
    # 5. Timing analysis
    print("\n5. Inference timing...")
    timing = measure_inference_time(model)
    
    # Summary
    print("\n=== Summary ===")
    print(f"Standard Sim: {standard_results['success_rate']:.1f}%")
    print(f"Randomized Sim: {random_results['success_rate']:.1f}%")
    print(f"Extreme Randomization: {extreme_results['success_rate']:.1f}%")
    print(f"Novel Positions: {novel_results['success_rate']:.1f}%")
    print(f"Inference Time: {timing['mean']:.1f}ms ± {timing['std']:.1f}ms")
    
    # Readiness assessment
    print("\n=== Readiness for Real Robot ===")
    checks = {
        'High standard sim success (>80%)': standard_results['success_rate'] > 80,
        'Robust to randomization (>60%)': random_results['success_rate'] > 60,
        'Real-time capable (<50ms)': timing['mean'] < 50,
        'Generalizes to novel positions': novel_results['success_rate'] > 50,
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    if all(checks.values()):
        print("\n✓✓✓ READY FOR REAL ROBOT DEPLOYMENT ✓✓✓")
    else:
        print("\n⚠ More training recommended before real deployment")

if __name__ == '__main__':
    final_evaluation()
```

---

[CONTINUING IN NEXT FILE DUE TO LENGTH LIMIT...]

Should I continue with Stage 4 (Real SO101 Deployment) and create additional support documents?
