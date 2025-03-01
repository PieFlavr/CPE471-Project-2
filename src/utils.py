"""
utils.py

Description: Utility functions for the project.
Author: Lucas Pinto
Date: February 12, 2025

Modules:
    numpy
    matplotlib.pyplot
    matplotlib.animation
    csv

Functions:
    get_key_by_value
    q_table_to_2d_array
    plot_action_sequence
    plot_q_table
    plot_episode_data
    save_training_data_to_csv
    save_training_data_set_to_csv
    interpret_action_sequence

Usage:
    Import the module and call the desired functions with appropriate arguments.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import csv


"""
====================================================================================================
GENERAL UTILITIES
====================================================================================================
"""

def get_key_by_value(dictionary, target_value):
    """
    Retrieves the key associated with the given value in a dictionary.

    Args:
        dictionary (dict): The dictionary to search through.
        target_value (any): The value to find the corresponding key for.

    Returns:
        any: The key associated with the target value, or None if the value is not found.
    """

    try:
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None
    except Exception as e:
        print(f"Error: {e}")

def q_table_to_2d_array(q_table, grid_length, grid_width):
    """
    Converts a Q-table into a 2D array representation.

    Args:
        q_table (np.ndarray): The Q-table to convert, assumed to be a 3D array with shape (grid_length, grid_width, num_actions).
        grid_length (int): The length of the grid.
        grid_width (int): The width of the grid.

    Returns:
        np.ndarray: A 2D array where each row represents a state and its corresponding Q-values.
    """
    rows = []
    for x in range(grid_length):
        for y in range(grid_width):
            row = [f"({x},{y})"] + list(q_table[x, y, :])
            rows.append(row)
    return np.array(rows)

"""
====================================================================================================
PLOTTING UTILITIES
====================================================================================================
"""

def plot_action_sequence(action_sequence: list = None, 
                        grid_length: int = 5, grid_width: int = 5, 
                        title = None, subtitle = None, fps = 48,
                        phi_centers: np.ndarray = None) -> plt.Figure:
    """
    Plots the action sequence on a grid with a gradient effect.

    Args:
        action_sequence (list): List of actions taken by the agent.
        grid_length (int): Length of the grid.
        grid_width (int): Width of the grid.
        title (str): Title of the plot.
        subtitle (str, optional): Subtitle of the plot.
        fps (int, optional): Frames per second for the animation. Default is 48.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, grid_length)
    ax.set_ylim(0, grid_width)
    ax.set_xticks(np.arange(0, grid_length, 1))
    ax.set_yticks(np.arange(0, grid_width, 1))
    ax.grid(True)

    # Drawing Phi centers
    if phi_centers is not None:
        for i in range(len(phi_centers)):
            ax.add_patch(patches.Circle((phi_centers[i][0]+0.5, phi_centers[i][1]+0.5), radius = 2, color='blue', alpha=0.1))

    # Initial position
    x, y = 0.5, 0.5
    dx, dy = 0, 0
    frame_total = 0
    cmap = plt.get_cmap('inferno')  # Colormap for gradient effect
    num_actions = len(action_sequence)
    base_fps = 60

    # Highlight the start point
    ax.plot(x, y, 'go', markersize=10, label='Start')
    ax.plot(x + grid_length - 1, y + grid_width - 1, 'ro', markersize=10, label='Goal')

    def update(frame):
        nonlocal x, y, dx, dy, frame_total
        actions_per_frame = max(1, int(fps / base_fps))  # Adjust this value to control how many actions are processed per frame
        start_frame = frame * actions_per_frame
        end_frame = min(start_frame + actions_per_frame, num_actions)

        # There's something ridiculously dumb about FuncAnimation that causes frame 0 to occur twice so needs these checks to not duplicate frames
        if not ((start_frame > num_actions or start_frame > end_frame) or frame_total == 0):
            #print(f"Global Frame {frame} ; Local Frame {start_frame}/{end_frame}: Drawing action sequence...")
            for i in range(start_frame, end_frame):
                action = action_sequence[i]
                if action == 0:  # Up
                    dx, dy = 0, -1
                elif action == 1:  # Down
                    dx, dy = 0, 1
                elif action == 2:  # Left
                    dx, dy = -1, 0
                elif action == 3:  # Right
                    dx, dy = 1, 0

                color = cmap(i / num_actions)  # Get color from colormap
                ax.arrow(x, y, dx * 0.75, dy * 0.75, head_width=0.25, head_length=0.25, fc=color, ec=color)

                # Update position with validation
                new_x = x + dx
                new_y = y + dy

                # Ensure the new position is within grid boundaries
                if (0.5 <= new_x < grid_length + 0.5) and (0.5 <= new_y < grid_width + 0.5):
                    x, y = new_x, new_y
                    #print(f"Frame {start_frame}/{end_frame}:{i}: Successful draw '{action}' arrow draw from ({x - dx}, {y - dy}) to ({x}, {y}).")
                else:
                    ax.arrow(x, y, dx * 0.25, dy * 0.25, head_width=0.25, head_length=0.25, fc='red', ec='red')
                    #print(f"Frame {start_frame}/{end_frame}:{i}: Invalid move '{action}' to ({new_x}, {new_y}) ignored.")
        frame_total += 1

    print("Generating action sequence plot...")

    if fps != 0:
        interval = 1000 / fps  # Calculate interval in milliseconds
        num_frames = (num_actions + max(1, int(fps / base_fps)) - 1) // max(1, int(fps / base_fps))  # Calculate the number of frames needed
        #print(f"Animating action sequence with {num_actions} actions and {num_frames} frames at {fps} FPS or {interval} ms interval.")
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=False)
    else:
        for i in range(num_actions):
            if(i == 0): # Accounting for the goofy fix the first frame being duplicated
                update(0)
            update(i)

    print("Action sequence plot complete.")

    plt.title(title)
    plt.suptitle(subtitle, fontsize=8)
    plt.gca().invert_yaxis()
    plt.legend()

    return fig

def plot_q_table(q_table, grid_length, grid_width, actions, title, subtitle=None, figsize=(12, 8), font_size=10, scale=(1.2, 1.2)) -> plt.Figure:
    """
    Plots a Q-table as a 2D table.

    Args:
        q_table (np.ndarray): The Q-table to plot.
        grid_length (int): Length of the grid.
        grid_width (int): Width of the grid.
        actions (dict): Dictionary of possible actions.
        title (str): Title of the plot.
        subtitle (str, optional): Subtitle of the plot.
        figsize (tuple, optional): Size of the figure. Default is (12, 8).
        font_size (int, optional): Font size of the table text. Default is 10.
        scale (tuple, optional): Scale of the table. Default is (1.2, 1.2).
    """
    q_table_2d = q_table_to_2d_array(q_table, grid_length, grid_width)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    table = ax.table(cellText=q_table_2d, colLabels=["State(x,y)"] + list(actions.keys()), loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(*scale)
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=8)

    return fig

def plot_episode_data(data: list, episodes: int,
                        title: str, subtitle: str = None,
                        xlabel: str = 'Episode', ylabel: str = 'Value', 
                        label: str = 'Data', color: str = 'blue', 
                        figsize: tuple = (12, 8), fontsize: int = 8) -> plt.Figure:
    """
    Plots episode data (e.g., total rewards or steps taken) per episode.

    Args:
        data (list): List of data values per episode.
        episodes (int): Number of episodes.
        title (str): Title of the plot.
        subtitle (str, optional): Subtitle of the plot.
        xlabel (str, optional): Label for the x-axis. Default is 'Episode'.
        ylabel (str, optional): Label for the y-axis. Default is 'Value'.
        label (str, optional): Label for the plot line. Default is 'Data'.
        color (str, optional): Color of the plot line. Default is 'blue'.
        figsize (tuple, optional): Size of the figure. Default is (12, 8).
        fontsize (int, optional): Font size of the subtitle. Default is 8.
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(range(episodes), data, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=fontsize)
    plt.legend()

    return fig

def plot_algorithm_data(data_dict: dict, episodes: int, 
                      title: str, subtitle: str = None, 
                      xlabel: str = 'Episode', ylabel: str = 'Value', 
                      figsize: tuple = (12, 8), fontsize: int = 8):
    """
    Plots multiple sets of episode data (e.g., total rewards or steps taken) per episode.

    Args:
        data_dict (dict): Dictionary where keys are labels and values are lists of data values per episode.
        episodes (int): Number of episodes.
        title (str): Title of the plot.
        subtitle (str, optional): Subtitle of the plot.
        xlabel (str, optional): Label for the x-axis. Default is 'Episode'.
        ylabel (str, optional): Label for the y-axis. Default is 'Value'.
        figsize (tuple, optional): Size of the figure. Default is (12, 8).
        fontsize (int, optional): Font size of the subtitle. Default is 8.
    """
    fig = plt.figure(figsize=figsize)
    for label, data in data_dict.items():
        if(data != None):
            print(f"Plotting data for {label}... which is {len(data)} long")
            plt.plot(range(episodes), data, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=fontsize)
    plt.legend()
    
    return fig

"""
====================================================================================================
FILE SAVING UTILITIES
====================================================================================================
"""

def save_training_data_to_csv(filename, training_data):
    """
    Saves the training data to a CSV file.

    Args:
        filename (str): The name of the file to save the data to.
        training_data (list): A list of training data, where each element is a tuple containing:
            - Action Sequence (list): The sequence of actions taken.
            - Total Reward (float): The total reward obtained.
            - Steps Taken (int): The number of steps taken.
            - Weights (np.ndarray): The weights at the end of the episode.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Action Sequence', 'Total Reward', 'Steps Taken', 'Weights'])
        for episode, data in enumerate(training_data):
            writer.writerow([episode + 1, data[0], data[1], data[2], data[3]])

def save_training_data_set_to_csv(filename, training_data_column, data_set_name):
    """
    Saves a specific column of training data to a CSV file.

    Args:
        filename (str): The name of the file to save the data to.
        training_data_column (list): The column of training data to save.
        data_set_name (str): The name of the data set (column header).
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', data_set_name])
        for episode, data in enumerate(training_data_column):
            writer.writerow([episode + 1, data])

def interpret_action_sequence(action_sequence, actions: dict = None) -> list:
    """
    Interprets a sequence of actions into their corresponding action names.

    Args:
        action_sequence (list): List of actions taken by the agent.
        actions (dict, optional): Dictionary mapping action indices to action names. 
                                  Default is {0: 'up', 1: 'down', 2: 'left', 3: 'right'}.

    Returns:
        list: List of action names corresponding to the action sequence.
    """
    if actions is None:
        actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    interpreted_sequence = [get_key_by_value(actions, action) for action in action_sequence]
    return interpreted_sequence

