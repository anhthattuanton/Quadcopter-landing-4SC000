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

---
## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Quadcopter-landing-4SC000.git
cd Quadcopter-landing-4SC000
```
### 2. Create and activate a virtual environment
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
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Training the Agent
To train the PPO agent:
 ```bash
py -m src.RL
```
Training logs are saved in the logs/ directory.
### 5. Monitoring Training
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir=logs
```
Metrics include:
- Episode reward
- Episode length
- Learning stability
### 6. Running the Visualization
After training, run:
 ```bash
py -m tests.test_visualization
```
Controls:
- SPACE – Pause / Resume
- R – Reset simulation
- Q – Quit

The visualization shows:
- Quadcopter trajectory
- Platform motion
- Position, velocity, and attitude
- Successful landing behavior
## Results
As demonstrated in the video:
- The quadcopter tracks the platform while descending
- Vertical motion does not stop during horizontal correction
- The quadcopter aligns itself above the platform
- A soft and stable landing is achieved
- This confirms that the reward shaping and policy learning successfully solve the task.
