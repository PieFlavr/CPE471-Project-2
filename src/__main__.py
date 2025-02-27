"""
__main__.py

Description: This script initializes and runs the main application for the CPE471 Project 1.
Author: Lucas Pinto
Date: February 12, 2025

"""

from __init__ import * # Import everything from the __init__.py file

def main():
    """
    Main function to run the application.
    """

    if __name__ == "__main__":
        print("Hello, World!")
        
        # Environment/Grid World Settings
        grid_length = 5
        grid_width = 5
        reward_vector = [grid_length*grid_width, -1, -5] # In order, the reward for reaching the goal, moving, and an invalid move
        # ^^^ scales dynamically with the grid size
        goal_position = None # If None, default is bottom right corner
        environment = GridWorld((grid_length, grid_width), goal_position, (grid_length-1, grid_width-1), reward_vector)
        agent_start = (0, 0) # None = random, yet to account for random position in graphing though!

        # Agent Possible Actions
        actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        
        sigma = 1.0 # Sigma value for the RBF function
        phi_centers_1 = np.array([[math.floor((grid_length/2)+grid_length//4), math.floor((grid_width/2)+grid_width//4)],
                                 [math.floor((grid_length/2)-grid_length//4), math.floor((grid_width/2)-grid_width//4)],
                                 [math.floor((grid_length/2)+grid_length//4), math.floor((grid_width/2)-grid_width//4)],
                                 [math.floor((grid_length/2)-grid_length//4), math.floor((grid_width/2)+grid_width//4)]])

                                 
        phi_centers_2 = np.concatenate((phi_centers_1,
                                       np.array([[math.floor((grid_length/2)), math.floor((grid_width/2))],
                                                    [math.floor((grid_length-1)), math.floor((grid_width/2))],
                                                    [math.floor((grid_length/2)), math.floor((grid_width-1))],
                                                    [0,math.floor((grid_width/2))],
                                                    [math.floor((grid_length/2)), 0]]))) 

        print(f"Phi Centers 1: {phi_centers_1}")
        print(f"Phi Centers 2: {phi_centers_2}")
        
        # Persisting Weight Tables (initialized with dummy values as to pass by reference)
        # q_table = np.zeros((grid_length, grid_width, len(actions)), dtype=float)  
        # p_weights = np.zeros((len(actions), len(phi_centers_1)), dtype=float)
        enable_record = np.zeros(4, dtype=bool) # [action_sequence, total_reward, steps_taken, q_table_history]

        enable_learning_algorithms = [False, False, True, True] # Enable Q-Learning, Q-Lambda, etc...

        # Q-learning Settings
        episodes = 100 # Number of episodes to train the agent
        alpha = 0.1 # Learning rate, how much the agent learns from new information
        gamma = 0.9 # Discount factor, how much the agent values future rewards
        epsilon = 0.1 # Exploration rate, how often the agent explores instead of exploiting
        tau = 0.1 # Softmax temperature for RBF-Q-Learning

        # Q-Lambda Settings (uses ^^^ settings)
        lambda_value = 0.5 # Lambda value for Q-Lambda learning

        # Enable recording of action sequence, total rewards, steps taken, and Q-table history
        enable_record_set_1 = [True, True, True, True] # Applies to first and last episode
        enable_record_set_2 = [True, True, True, True] # Applies to everything between first and last episode
        
        # Plotting Settings
        fps = 600 # Frames per second for the plot animation, disables animation at 0

        enable_q_table_plots = False # Enable Q-table plots
        enable_episode_plots = True # Enable episode plots such as rewards/steps over time
        enable_first_action_sequence_plots = True
        enable_last_action_sequence_plots = True

        # Summarize training settings for display purposes
        training_settings_summary = f"{grid_length}x{grid_width} Grid World\nEpisodes: {episodes}, Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}\nRewards: {reward_vector}"
        agent_settings_summary = f"Agent Start: (0, 0), Goal: ({grid_length-1}, {grid_width-1})"
        algorithm_settings_summary = None

        # File Saving Settings
        save_training_data = True # Enable saving of training data
        save_directory = "training_data" # Directory to save the CSV files
        print(f"Training data will be saved to {save_directory}.")

        # Learning Settings
        learning_algorithms = {'Q-Learning': Q_learning_episode, 
                               'Q-Lambda': Q_lambda_episode, 
                               '4RBF-Q-Learning': RBF_Q_learning_episode, 
                               '9RBF-Q-Learning': RBF_Q_learning_episode}
        
        algorithm_exclusive_arguments = {'Q-Learning': {'selection_function': softmax_Q_selection},
                                        'Q-Lambda': {'selection_function': softmax_Q_selection},
                                        '4RBF-Q-Learning': {'phi_centers': phi_centers_1, 'selection_function': softmax_P_selection},
                                        '9RBF-Q-Learning': {'phi_centers': phi_centers_2, 'selection_function': softmax_P_selection}
                                        }
        
        global_learning_arguments = {'grid_world': environment, 'actions': actions, 
                                'q_table': None, 'weights': None,  
                                'selection_function': epsilon_greedy_Q_selection, 
                                'function_args': {'q_table': None, 'weights': None, 'epsilon': epsilon},
                                'alpha': alpha, 'gamma': gamma, 'agent_start': agent_start, 
                                'lambda': lambda_value, 
                                'sigma': sigma, 'tau': tau, 
                                'enable_record': enable_record}
        
        print("Training agents...")

        # Ensure the directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        for algorithm_name, algorithm_function in learning_algorithms.items():
            
            print(f"Resetting weights for {algorithm_name}...")
            # Initialize Q-table with zeros
            if ('RBF' not in algorithm_name):
                q_table = np.zeros((grid_length, grid_width, len(actions)), dtype = float) # Initialize Q-table with zeros
            else:
                weights = np.zeros((len(actions), len(algorithm_exclusive_arguments[algorithm_name]['phi_centers'])), dtype = float)

            training_data = []
            
            enable_record = enable_record_set_1

            if enable_learning_algorithms[list(learning_algorithms.keys()).index(algorithm_name)]:
                print(f"Copying global learning arguments for {algorithm_name}...")
                algorithm_settings_summary = f"Trained w/ {algorithm_name} and Epsilon-Greedy Selection"

                local_learning_arguments = global_learning_arguments.copy()

                local_learning_arguments['enable_record'] = enable_record
                local_learning_arguments.update(algorithm_exclusive_arguments[algorithm_name])

                if ('RBF' not in algorithm_name):
                    local_learning_arguments['q_table'] = q_table
                    local_learning_arguments['function_args']['q_table'] = q_table
                else:   
                    local_learning_arguments['weights'] = weights
                    local_learning_arguments['function_args']['weights'] = weights

                for episode in range(episodes):
                    environment.reset()
                    if (episode == 0) or (episode == episodes - 1):
                        enable_record = enable_record_set_1
                    else:
                        enable_record = enable_record_set_2

                    print(f"Training {algorithm_name} agent Episode {episode + 1} of {episodes}...", end=' ')
                    # Run a single episode of the learning algorithm

                    action_sequence, total_reward, steps_taken, q_table_history = algorithm_function(**local_learning_arguments)
                    
                    training_data.append([action_sequence, total_reward, steps_taken, q_table_history])
                    print(f"Completed!!! Total Reward: {total_reward}, Steps Taken: {steps_taken}.")

                print(f"{algorithm_name} Training completed.")

                # Extract total rewards and steps taken per episode
                raw_action_sequence_history = [data[0] for data in training_data]
                q_table_history = [data[3] for data in training_data]
                total_rewards = [data[1] for data in training_data]
                steps_taken = [data[2] for data in training_data]

                # Extract the first and last Q-tables
                first_q_table = training_data[0][3]
                last_q_table = training_data[-1][3]

                if(grid_length*grid_width <= 25) and (enable_q_table_plots): # Too high and the q_Table simply crashes the program
                    plot_q_table(first_q_table, grid_length, grid_width, 
                                actions, 'First Q-table', 
                                training_settings_summary
                                + "\n" + agent_settings_summary
                                    + "\n" + algorithm_settings_summary)
                    
                    plot_q_table(last_q_table, grid_length, grid_width, 
                                actions, 'Last Q-table', 
                                training_settings_summary
                                + "\n" + agent_settings_summary
                                    + "\n" + algorithm_settings_summary)
                elif(grid_length*grid_width > 25): 
                    print("Grid too large to display Q-tables. Try to keep the area under 25 cells.")

                if(enable_episode_plots):
                    # Plot total rewards per episode
                    plot_episode_data(total_rewards, episodes, 'Total Reward per Episode', 
                                    training_settings_summary
                                        + "\n" + agent_settings_summary
                                        + "\n" + algorithm_settings_summary,
                                            ylabel='Total Reward', label='Total Reward', color='blue')

                    # Plot steps taken per episode
                    plot_episode_data(steps_taken, episodes, 'Steps Taken per Episode',
                                    training_settings_summary
                                        + "\n" + agent_settings_summary
                                        + "\n" + algorithm_settings_summary,
                                            ylabel='Steps Taken', label='Steps Taken', color='orange')

                if(enable_first_action_sequence_plots):
                    # Plot the first action sequence
                    first_action_sequence = training_data[0][0]
                    plot_action_sequence(first_action_sequence, grid_length, grid_width, 
                                        'First Action Sequence', 
                                        (training_settings_summary
                                        + "\n" + agent_settings_summary
                                            + "\n" + algorithm_settings_summary),
                                            fps=fps)

                if(enable_last_action_sequence_plots):
                    # Plot the last action sequence
                    last_action_sequence = training_data[-1][0]
                    plot_action_sequence(last_action_sequence, grid_length, grid_width, 
                                        'Last Action Sequence', 
                                        (training_settings_summary
                                        + "\n" + agent_settings_summary
                                            + "\n" + algorithm_settings_summary),
                                            fps=fps)
                
                if(save_training_data):
                    save_training_data_to_csv(os.path.join(save_directory, f"training_data_{algorithm_name}.csv"), training_data)
                    save_training_data_set_to_csv(os.path.join(save_directory, f"total_rewards_{algorithm_name}.csv"), total_rewards, "Total Rewards")
                    save_training_data_set_to_csv(os.path.join(save_directory, f"steps_taken_{algorithm_name}.csv"), steps_taken, "Steps Taken")
                    save_training_data_set_to_csv(os.path.join(save_directory, f"q_table_history_{algorithm_name}.csv"), q_table_history, "Q-table")
                    save_training_data_set_to_csv(os.path.join(save_directory, f"raw_action_sequence_history_{algorithm_name}.csv"), raw_action_sequence_history, "Action Sequence")
                    interpreted_action_sequence_history = []
                    for action_sequence in raw_action_sequence_history:
                        interpreted_action_sequence = interpret_action_sequence(action_sequence, actions)
                        interpreted_action_sequence_history.append(interpreted_action_sequence)
                    save_training_data_set_to_csv(os.path.join(save_directory, f"interpreted_action_sequence_history_{algorithm_name}.csv"), interpreted_action_sequence_history, "Action Sequence")
    pass

main()