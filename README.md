# 9 Open assignment: Landing a quadcopter on a moving platform

This project demonstrates a **successful reinforcement learning solution** for a **planar quadcopter landing on a moving platform**.  
The trained agent is able to **track the platform horizontally while descending simultaneously**, resulting in a **stable and soft landing**, as shown in the demonstration video.

This work was completed for **Homework 9** in the course **4SC000**.

---

## Task Description

The goal of the task is to train a planar quadcopter to:

- Track a **horizontally moving platform**
- Maintain **attitude stability**
- Reduce **relative velocity**
- Perform a **soft and accurate landing** on the platform

Unlike naive approaches that separate tracking and descent, this solution achieves **coupled horizontal and vertical control**, allowing the quadcopter to land even when the platform is in motion.

---

## Key Features of the Solution

- **Reinforcement Learning algorithm**: Proximal Policy Optimization (PPO)
- **Phased reward design**:
  - Long-range tracking and interception
  - Simultaneous descent and alignment
  - Precision landing with velocity and tilt constraints
- **Stable behavior**:
  - No hovering deadlock above the platform
  - Continuous descent during tracking
  - Soft touchdown on the landing pad
- **Robust to platform motion**

The final policy consistently lands on the platform within the simulation horizon.

---
## Repository Structure
Quadcopter-landing-4SC000/
│
├── src/
│ ├── RL.py # PPO training script
│ ├── rewardfunction.py # Custom reward shaping
│ ├── simulation_data.py # Environment constants
│
├── tests/
│ └── test_visualization.py # Playback and visualization
│
├── requirements.txt
├── .gitignore
└── README.md

## Why PPO and How the Physics and Randomization Are Modeled

### Why We Use Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) was chosen for this task because it combines **sample efficiency**, **stability**, and **simplicity** compared to other modern RL algorithms. PPO is a policy-gradient method that directly optimizes the policy while constraining updates to prevent destabilizing large steps. It implements a clipped surrogate objective, which helps maintain reliable improvement without the need for complex second-order optimization (as in TRPO), and avoids the brittle tuning often associated with value-based or off-policy algorithms.

In the context of the planar quadcopter task:

- The **state space is continuous** and high-dimensional (position, velocity, orientation, platform dynamics).
- The **action space is continuous** (thrust commands).
- The task requires both **exploration and exploitation** to reconcile tracking and landing objectives.

PPO’s formulation is well-suited for continuous control problems like this one because it learns a **parametric stochastic policy** that balances **exploration** and **policy improvement** while being robust to hyperparameter variations. Alternatives such as DDPG or SAC are also valid, but PPO was selected due to its **widespread success in similar control benchmarks**, strong stability, and ease of integration with stable baselines frameworks.

---

### Physics Modeling Inspired by *Learning to Land in the Real World: A Benchmark for Real-world Reinforcement Learning*

The quadcopter dynamics in this project are inspired by the physics presented in the paper *Learning to Land in the Real World: A Benchmark for Real-world Reinforcement Learning* (arXiv:2106.15134). Although our task is simplified to the planar case (2D), the fundamental modeling principles are maintained:

- The quadcopter is represented as a rigid planar body with states:
  - horizontal position \(x\) and vertical position \(y\),
  - orientation \(\theta\),
  - linear velocities \(\dot{x}, \dot{y}\),
  - angular velocity \(\dot{\theta}\).

- Control is effected through thrust commands for the left and right rotors, which generate forces and moments influencing motion.

- The environment integrates Newtonian equations of motion at each simulation step using simple kinematics to update the state based on current velocities, thrust, and gravity.

While the full 3D dynamics in the reference paper include more degrees of freedom, our planar simplification retains the essential challenge of controlled vertical landing and horizontal tracking while considering real-time velocity and orientation.

---

### Randomization of Spawns and Platform Motion

To encourage robust policies that generalize across scenarios, the initial conditions and platform behavior are randomized each episode, similar in spirit to domain randomization used in real-world oriented RL:

1. **Random initial position of the quadcopter**  
   - \(x\) and \(y\) are sampled within specified ranges (e.g., uniform distribution across allowable horizontal and vertical regions).
   - This prevents the agent from overfitting to a single start configuration.

2. **Random initial velocities**  
   - The quadcopter’s linear and angular velocities at the start of each episode are randomized.
   - This increases policy resilience to initial motion conditions.

3. **Random platform state**  
   - The platform’s initial horizontal position and velocity are sampled at the start of each episode.
   - The platform may move with varying speeds and directions within set bounds, adding hysteresis to the tracking challenge.

By varying initial states in this way, the agent experiences a wide distribution of scenarios during training. This discourages overfitting to a fixed start and better prepares the trained policy to handle unseen dynamics — a key principle emphasized in the reference paper for real-world reinforcement learning.

### Summary

In summary:

- **PPO** is used because it is robust and effective for continuous control problems with complex reward landscapes, like landing while tracking a moving platform.
- The **environment physics** are inspired by simplified rigid body dynamics similar to benchmark RL papers, focusing on states and motions relevant to planar quadcopter control.
- **Randomization** of initial spawns and platform motion ensures the learned policy generalizes across a wide range of conditions rather than a narrow scenario.

These choices together provide a principled approach to training a landing agent that is both capable and generalizable.

---
## Installation and Environment Setup

### Virtual Environment

A Python virtual environment is strongly recommended to:
- Avoid dependency conflicts
- Ensure reproducibility
- Keep all required packages isolated from the system Python

All terminal commands below **must be executed inside the virtual environment**.

### Create and activate a virtual environment
- Windows:
 ```bash
  py -m venv .venv
.venv\Scripts\activate
```
- Linux / macOS:
 ```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Training the Reinforcement Learning Agent
Training is performed using the PPO algorithm implemented in src/RL.py.
To train the PPO agent:
 ```bash
py -m src.RL
```
Training Configuration Notes
The main training logic is located in RL.py
- The total number of training steps can be adjusted if:
- The mean episode reward does not converge
- The agent fails to land reliably
- The number of parallel environments (CPU cores) can also be modified in RL.py to speed up training on multi-core systems
- Training logs are automatically saved every 1 million timesteps
  
These logs are later used for monitoring learning progress.

### Monitoring Training with TensorBoard
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir=logs
```
Then open the displayed local URL in a browser.
#### TensorBoard Metrics Interpretation
Key metrics include:
- rollout/ep_rew_mean:
Indicates the average reward per episode. A steady increase suggests improved landing behavior.

- rollout/ep_len_mean:
Represents the average episode length. Increasing values usually indicate longer survival and more controlled flight.

- time/fps:
Shows training speed and confirms stable execution.

A converging reward curve combined with increasing episode length is a strong indicator that the policy is learning successfully.

### Running the Visualization
After training, the learned policy can be tested visually:
 ```bash
py -m tests.test_visualization
```
#### Controls:
- SPACE – Pause / Resume
- R – Reset simulation
- Q – Quit

The visualization shows:
- Quadcopter trajectory
- Platform motion
- Position, velocity, and attitude
- Successful landing behavior
### Results
As demonstrated in the video:
- The quadcopter tracks the platform while descending
- Vertical motion does not stop during horizontal correction
- The quadcopter aligns itself above the platform
- A soft and stable landing is achieved
- This confirms that the reward shaping and policy learning successfully solve the task.
