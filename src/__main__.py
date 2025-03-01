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
        goal_position = (grid_length-1,grid_width-1) # If None, default is bottom right corner
        environment = GridWorld((grid_length, grid_width), None, goal_position, reward_vector)
        agent_start = (0, 0) # None = random, yet to account for random position in graphing though!

        # Agent Possible Actions (-/+/-/+)
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
        
        phi_center_N = generate_RBF_centers((grid_length, grid_width), grid_length*grid_width*0.8)

        #print(f"Phi Centers 1: {phi_centers_1}")
        #print(f"Phi Centers 2: {phi_centers_2}")
        #print(f"Phi Centers N: {phi_center_N}")

        enable_record = np.zeros(4, dtype=bool) # [action_sequence, total_reward, steps_taken, q_table_history]

        # Q-learning Settings
        episodes = 60 # Number of episodes to train the agent
        alpha = 0.15 # Learning rate, how much the agent learns from new information
        gamma = 0.9 # Discount factor, how much the agent values future rewards
        epsilon = 0.05 # Exploration rate, how often the agent explores instead of exploiting
        tau = 0.5 # Softmax temperature for softmax selection function
        greedy_cutoff = episodes*(2.0/3.0) # Episode cutoff for full greedy selection function
        #greedy_cutoff = -1 #No Greedy Cutoff

        # Q-Lambda Settings (uses ^^^ settings)
        lambda_ = 0.5 # Lambda value for Q-Lambda learning

        # Enable recording of action sequence, total rewards, steps taken, and weights history
        enable_record_set_1 = [True, True, True, True] # Applies to first and last episode
        enable_record_set_2 = [True, True, True, True] # Applies to everything between first and last episode
        
        # Plotting Settings
        fps = 60 # Frames per second for the plot animation, disables animation at 0

        enable_q_table_plots = False # Enable weights plots
        enable_episode_plots = False # Enable individual episode plots such as rewards/steps over time
        enable_first_action_sequence_plots = True 
        enable_last_action_sequence_plots = True

        # Summarize training settings for display purposes
        training_settings_summary = f"{grid_length}x{grid_width} Grid World"
        training_settings_summary += f"\nEpisodes: {episodes} "
        training_settings_summary += f"Alpha: {alpha}, Gamma: {gamma}, "
        training_settings_summary += f"Epsilon: {epsilon}, Tau: {tau} "
        training_settings_summary += f"Lambda: {lambda_}, Sigma: {sigma} "
        agent_settings_summary = f"Agent Start: (0, 0), Goal: ({goal_position[0]}, {goal_position[1]}) "
        agent_settings_summary += f"Rewards: {reward_vector}"
        algorithm_settings_summary = None

        # File Saving Settings
        save_training_data = True # Enable saving of training data
        save_graphs = True # Enable saving of graphs
        save_directory = "training_data" # Directory to save the CSV files
        print(f"Training data will be saved to {save_directory}.")

        learning_algorithms = {
                                'Q-Learning': Q_learning_episode, 
                                'Q-Lambda': Q_lambda_episode, 
                                'Q-Lambda_Alt': Q_lambda_episode,
                                'Q-Lambda_Alt2': Q_lambda_episode,
                                'Q-Lambda_Alt3': Q_lambda_episode,
                                #'FSR-Q-Learning': FSR_Q_learning_episode,
                                ##'FSR-Q-Learning_Alt': FSR_Q_learning_episode,
                                ##'FSR-Q-Learning_Alt2': FSR_Q_learning_episode,
                                #'4RBF-Q-Learning': RBF_Q_learning_episode, 
                                #'9RBF-Q-Learning': RBF_Q_learning_episode,
                                ##'NRBF-Q-Learning': RBF_Q_learning_episode
                                }
        
        algorithm_exclusive_arguments = {
                                        'Q-Learning': {'selection_function': softmax_Q_selection},
                                        'Q-Lambda': {'selection_function': softmax_Q_selection},
                                        'Q-Lambda_Alt': {'selection_function': softmax_Q_selection},
                                        'Q-Lambda_Alt2': {'selection_function': softmax_Q_selection},
                                        'Q-Lambda_Alt3': {'selection_function': softmax_Q_selection},
                                        'FSR-Q-Learning': {'selection_function': softmax_FSR_selection},
                                        ##'FSR-Q-Learning_Alt': {'selection_function': softmax_FSR_selection,'alpha': 0.01, 'gamma': 0.99}, 
                                        '4RBF-Q-Learning': {'selection_function': softmax_P_selection, 'phi_centers': phi_centers_1},
                                        '9RBF-Q-Learning': {'selection_function': softmax_P_selection, 'phi_centers': phi_centers_2},
                                        ##'NRBF-Q-Learning': {'selection_function': softmax_P_selection, 'phi_centers': phi_center_N}
                                        }
        
        global_learning_arguments = {'grid_world': environment, 'actions': actions, 
                                'weights': None,  
                                'selection_function': softmax_FSR_selection, 
                                'function_args': {'weights': None, 'epsilon': epsilon, 'tau': tau, 'greedy_cutoff': greedy_cutoff},
                                'alpha': alpha, 'gamma': gamma, 'agent_start': agent_start, 
                                'lambda_': lambda_, 
                                'sigma': sigma, 'phi_centers': phi_center_N,
                                'enable_record': enable_record
                                }
        
        print("Training agents...")

        # Ensure the directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        global_rewards_data = {key: None for key in learning_algorithms}
        global_steps_data = {key: None for key in learning_algorithms}

        """
        ====================================================================================================
        MAIN TRAINING LOOP
        ====================================================================================================
        """

        for algorithm_name, algorithm_function in learning_algorithms.items():
            
            print(f"Resetting weights for {algorithm_name}...")
            # Initialize Q-table with zeros
            if ('FSR' in algorithm_name):
                print("Initializing FSR-Vector weights...")
                #start_goal_distance = (environment._goal[0]-agent_start[0]) + (environment._goal[1]-agent_start[1])
                #weights = np.zeros(grid_length + grid_width + len(actions) + start_goal_distance, dtype = float)
                #weights = np.zeros(grid_length*grid_width*len(actions)) # Big enough to accomodate any arbitrary type of feature vector, easier for experimenting
                weights = np.zeros((grid_length+grid_width)*len(actions), dtype = float)
                #weights = np.zeros(len(actions) + start_goal_distance, dtype = float)
            elif('RBF' in algorithm_name):
                print("Initializing RBF-Vector weights...")
                weights = np.zeros((len(actions), len(algorithm_exclusive_arguments[algorithm_name]['phi_centers'])), dtype = float)
            else:
                print("Initializing Q-Table weights...")
                weights = np.zeros((grid_length, grid_width, len(actions)), dtype = float) # Initialize Q-table with zeros
            training_data = []
            
            enable_record = enable_record_set_1

                
            print(f"Copying global learning arguments for {algorithm_name}...")
            local_learning_arguments = global_learning_arguments.copy()

            local_learning_arguments['enable_record'] = enable_record

            if algorithm_name in algorithm_exclusive_arguments:
                local_learning_arguments.update(algorithm_exclusive_arguments[algorithm_name])

            local_learning_arguments['weights'] = weights
            local_learning_arguments['function_args']['weights'] = weights

            algorithm_settings_summary = f"Trained w/ {algorithm_name} and "
            #print(f"Local Learning Arguments: {local_learning_arguments}")
            algorithm_settings_summary += f"Selection Function: {local_learning_arguments['selection_function'].__name__}"


            print(f"{training_settings_summary}\n{agent_settings_summary}\n{algorithm_settings_summary}")

            for episode in range(episodes):
                environment.reset()
                if (episode == 0) or (episode == episodes - 1):
                    enable_record = enable_record_set_1
                else:
                    enable_record = enable_record_set_2

                print(f"Training {algorithm_name} agent Episode {episode + 1} of {episodes}...", end=' ', flush = True)
                
                # Run a single episode of the learning algorithm

                action_sequence, total_reward, steps_taken, weights_history = algorithm_function(episode = episode, **local_learning_arguments)
                #print(weights)
                
                training_data.append([action_sequence, total_reward, steps_taken, weights_history])
                print(f"Completed!!! Total Reward: {total_reward}, Steps Taken: {steps_taken}.")

            print(f"{algorithm_name} Training completed.")

            """
            ====================================================================================================
            GRAPHING AND SAVING RESULTS
            ====================================================================================================
            """

            # Extract total rewards and steps taken per episode
            raw_action_sequence_history = [data[0] for data in training_data]
            weights_history = [data[3] for data in training_data]
            total_rewards = [data[1] for data in training_data]
            steps_taken = [data[2] for data in training_data]

            # Store the training data for each algorithm
            global_rewards_data[algorithm_name] = copy.deepcopy(total_rewards)
            global_steps_data[algorithm_name] = copy.deepcopy(steps_taken)

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
                
            plot_phi_centers = None
   
            if((algorithm_name in algorithm_exclusive_arguments) and ('phi_centers' in algorithm_exclusive_arguments[algorithm_name])):
                plot_phi_centers = algorithm_exclusive_arguments[algorithm_name]['phi_centers']

            if(enable_first_action_sequence_plots):
                # Plot the first action sequence
                first_action_sequence = training_data[0][0]
                ffig_action_sequence, ffas_anim = plot_action_sequence(first_action_sequence, grid_length, grid_width, 
                                    'First Action Sequence', 
                                    (training_settings_summary
                                    + "\n" + agent_settings_summary
                                        + "\n" + algorithm_settings_summary),
                                        fps=fps, 
                                        phi_centers=plot_phi_centers)
                if(save_graphs):
                    ffig_action_sequence.savefig(os.path.join(save_directory, f"first_action_sequence_{algorithm_name}.png"))
                plt.show(block=True)
            if(enable_last_action_sequence_plots):
                # Plot the last action sequence
                last_action_sequence = training_data[-1][0]
                lfig_action_sequence, lfas_anim = plot_action_sequence(last_action_sequence, grid_length, grid_width, 
                                    'Last Action Sequence', 
                                    (training_settings_summary
                                    + "\n" + agent_settings_summary
                                        + "\n" + algorithm_settings_summary),
                                        fps=fps,
                                        phi_centers=plot_phi_centers)
                if(save_graphs):
                    lfig_action_sequence.savefig(os.path.join(save_directory, f"last_action_sequence_{algorithm_name}.png"))
                plt.show(block=True)
            if(save_training_data):
                print(f"Saving training data for {algorithm_name}...")
                save_training_data_to_csv(os.path.join(save_directory, f"training_data_{algorithm_name}.csv"), training_data)
                save_training_data_set_to_csv(os.path.join(save_directory, f"total_rewards_{algorithm_name}.csv"), total_rewards, "Total Rewards")
                save_training_data_set_to_csv(os.path.join(save_directory, f"steps_taken_{algorithm_name}.csv"), steps_taken, "Steps Taken")
                save_training_data_set_to_csv(os.path.join(save_directory, f"weights_history{algorithm_name}.csv"), weights_history, "Weights")
                save_training_data_set_to_csv(os.path.join(save_directory, f"raw_action_sequence_history_{algorithm_name}.csv"), raw_action_sequence_history, "Action Sequence")
                interpreted_action_sequence_history = []

                for action_sequence in raw_action_sequence_history:
                    interpreted_action_sequence = interpret_action_sequence(action_sequence, actions)
                    interpreted_action_sequence_history.append(interpreted_action_sequence)
                save_training_data_set_to_csv(os.path.join(save_directory, f"interpreted_action_sequence_history_{algorithm_name}.csv"), interpreted_action_sequence_history, "Action Sequence")
    
        #print(global_steps_data)
        #print(global_rewards_data)

        # Do global data comparisons
        fig_total_reward = plot_algorithm_data(global_rewards_data, episodes, 
                            'Total Rewards per Episode', 
                            training_settings_summary
                                + "\n" + agent_settings_summary,
                                    ylabel='Total Reward', xlabel='Episodes')
        plt.show(block=True)

        fig_steps_taken = plot_algorithm_data(global_steps_data, episodes,
                            'Steps Taken per Episode', 
                            training_settings_summary
                                + "\n" + agent_settings_summary,
                                    ylabel='Steps Taken', xlabel='Episodes')

        if save_graphs:
            fig_total_reward.savefig(os.path.join(save_directory, "total_rewards_comparison.png"))
            fig_steps_taken.savefig(os.path.join(save_directory, "steps_taken_comparison.png"))

        plt.show(block=True)

    pass

main()