# Grid World Reinforcement Learning

## Overview
This project implements a Grid World environment for reinforcement learning algorithms. The environment is a grid where an agent learns to navigate from a starting position to a goal position while maximizing rewards and minimizing penalties via a wide variety of algorithms.

## Features
- **Grid World Environment**: A grid where the agent learns to navigate. The size, starting positionn, goal, and rewards are customizable.
- **Q-Learning Algorithm**: The basic algorithm for the agent to learn optimal policies.
- **Q-Lambda Algorithm**: Combines Q-Learning with eligibility traces for faster learning.
- **RBF Q-Learning Algorithm**: Uses Radial Basis Function (RBF) networks to approximate the Q-values for continuous state spaces. In this case, uses a Gaussian RBF.
- **FSR Q-Learning Algorithm**: Implements Feature Selection and Reduction (FSR) to improve learning efficiency by reducing the state space. In this case, encodes row, columns, and actions.
- **Dynamic Reward System**: Rewards and penalties that scale dynamically with the grid size.
- **Action Recording**: Records action sequences, total rewards, steps taken, and Q-table history.
- **Plotting**: Visualizes Q-tables, episode rewards, steps taken, and action sequences.
- **Plot Animations**: Creates animated plots to show the progression of the agent's learning over time.
- **CSV Export**: Exports training data, rewards, steps, plots, and action sequences to CSV files.

## Project Submission Files
The data files analyzed in the report are organized in a folder named `project_data`. The report itself is a .pdf file in the main directory named `Pinto CPE471 - Project 2 Report.pdf`. The `project_data` folder includes only training data for a **5x5 Grid**, but includes all necessary algorithms.

- **5x5 Grid**:
    - First Action Sequence Images
    - Last Action Sequence Images
    - Steps Taken Comparison Images
    - Total Rewards Comparison Images
    - Full Data CSVs
    - Interpreted Action Sequence CSVs
    - Raw Action Sequence CSVs
    - Reward CSVs
    - Steps CSVs
    - Weights CSVs

## Dependencies
This project requies a **Python** version of 3.12.9 or higher to run.
You are going to need the following dependencies for this project. Whether older version work or not are unknown as they have not been tested, but these were the versions used when developing this.
- **NumPy**: 2.2.2 or higher
- **Matplotlib**: 3.10.0 or higher

From the main directory, you can simply install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the project from the main directory, just run the following command
```bash
python src/__main__.py
```
Once the projects starts, by default, it will in sequence:
1. **Initialize the Grid World Environment**: Set up the grid with the specified dimensions, start position, and goal position.
2. **Initialize Learning Algorithms**: Configure and initialize the learning algorithms based on the provided settings.
3. **Run Training Episodes**: Execute the specified number of training episodes where the agent interacts with the environment, updates the Q-table, and learns the optimal policy.
4. **Record Actions and Rewards**: Record the sequences of actions taken, rewards received, and steps taken during the training episodes.
5. **Plot Results**: Generate plots for Q-tables, episode rewards, steps taken, and action sequences ("animated" to show better sequencing).
6. **Export Data**: Save the recorded training data, rewards, steps, and action sequences to CSV files. By default this is exported to the "training_data" folder.
7. **Repeat Until Done**: Will repeat the previous steps per algorithm specified to run until complete! 

## [OLD] Demonstration
Here are two old demonstration videos from `Project 1` on how to run it and how it should look like...

**5x5 DEMO:** https://youtu.be/a9IlX-IIqPI

**10x10 DEMO:** https://youtu.be/QAEhNdgtyeM

While the code and plotting is outdated, the general principle of starting the same. 

## Configuration
The main configuration settings for the Grid World environment and reinforcement learning algorithms can be found in the `main` function. Below is a list of settings that you can customize:

- **Grid Size**:
  - `grid_length`: Length of the grid.
  - `grid_width`: Width of the grid.
- **Start Position**:
  - `agent_start`: Starting position of the agent.
- **Goal Position**:
  - `goal_position`: Goal position for the agent.
- **Reward Values**:
  - `reward_vector`: Rewards for reaching the goal, moving, and invalid moves.
- **Learning Settings**:
  - `learning_algorithms`: Dictionary of learning algorithms to use.
  - `enable_learning_algorithms`: List of booleans to enable/disable specific learning algorithms.
- **Learning Settings**:
  - `episodes`: Number of training episodes.
  - `alpha`: Learning rate for Q-learning updates.
  - `gamma`: Discount factor for future rewards.
  - `epsilon`: Exploration rate for the agent's actions.
  - `tau`: Temperature for softmax action selection.
  - `greedy_cutoff`: Episode number cutoff for full greedy selection. At -1, is disabled.
  - `lambda_`: Lambda value for eligibiliy decay. 
- **Recording Settings**:
  - `enable_record_set_1`: Flags to enable recording for the first and last episode.
  - `enable_record_set_2`: Flags to enable recording for episodes between the first and last.
- **Plotting Settings**:
  - `fps`: Frames per second for the plot animation. At 0, is disabled.
  - `enable_q_table_plots`: Enable/disable Q-table plots.
  - `enable_episode_plots`: Enable/disable episode plots such as rewards/steps over time.
  - `enable_first_action_sequence_plots`: Enable/disable plotting of the first action sequence.
  - `enable_last_action_sequence_plots`: Enable/disable plotting of the last action sequence.
- **File Saving Settings**:
  - `save_training_data`: Enable/disable saving of training data.
  - `save_graphs`: Enable/disable saving of plot and graphs.
  - `save_directory`: Directory to save the CSV files.

You can modify these settings in the `main` function to suit your specific requirements. In future projects, the hope is to be able to read a JSON file with all these settings in one place as opposed to modifying the code itself.