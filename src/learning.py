"""
learning.py

Description: This module implements the Q-learning and Q(lambda) algorithms for a GridWorld environment. 
            It includes functions for running episodes, selecting actions using an epsilon-greedy policy, and updating the Q-table and eligibility traces.
Author: Lucas Pinto
Date: February 12, 2025

Modules:
    numpy - For numerical operations on arrays.
    utils - Utility functions used in the project.
    grid_world - The GridWorld environment class.
    agent - The Agent class that interacts with the environment.
    typing - For type hinting.

Functions:
    Q_learning_episode - Runs a single episode of the Q-learning algorithm.
    Q_learning_table_update - Updates the Q-table using the Q-learning algorithm.
    Q_lambda_episode - Runs a single episode of the Q(λ) algorithm.
    Q_lambda_table_update - Updates the Q-table and eligibility traces using the Q(λ) algorithm.
    epsilon_greedy_selection - Selects an action using the epsilon-greedy policy.

Usage:
    python main.py
"""

import numpy as np

from utils import *
from grid_world import GridWorld
from agent import Agent
from typing import Tuple

"""
====================================================================================================
EPISODES
====================================================================================================
"""

def RBF_Q_learning_episode(grid_world: GridWorld = None,
                           agent: Agent = None,
                           actions: list = None,
                           q_table: np.ndarray = None,
                           phi_centers: np.ndarray = None,
                           sigma: float = 1.0,
                           selection_function: callable = None,
                           select_func_args: dict = None,
                           alpha: float = 0.1,
                           gamma: float = 0.9,
                           agent_start: Tuple[int,int] = None,
                           enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False)) -> Tuple[list, float, int, list]:
    
    # Check if any of the parameters are None
    if grid_world is None:
        raise ValueError("GridWorld cannot be None!")
    if actions is None:
        raise ValueError("Actions cannot be None!")
    if q_table is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())

    if not callable(selection_function):
        raise ValueError("Selection function must be callable!")
    try: 
        test_state = grid_world.get_state()[1]
        test_phi_state = np.array([gaussian_RBF(test_state, center, sigma) for center in phi_centers])
        selection_function(test_state, **select_func_args)
    except TypeError as e:
        raise ValueError(f"Selection function arguments are invalid: {e}")

    grid_world.reset(agent_start)  # Initializes the agent and environment state

    action_sequence = []
    final_q_table = None
    steps_taken = 0
    total_reward = 0

    goal_reached = False

    while not goal_reached:
        state = grid_world.get_state()[1]  # Get the current state of the environment
        phi_state = np.array([gaussian_RBF(state, center, sigma) for center in phi_centers])

        action = selection_function(phi_state, **select_func_args)

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None

        next_state = grid_world.get_state()[1]  # Get the next state of the environment
        next_phi_state = np.array([gaussian_RBF(next_state, center, sigma) for center in phi_centers])

        Q_learning_RBF_update(phi_state, next_phi_state, action, reward, q_table, alpha, gamma)

    final_q_table = q_table.copy() if enable_record[3] else None

    return action_sequence, total_reward, steps_taken, final_q_table

    pass

def Q_learning_episode(grid_world: GridWorld = None, 
               agent: Agent = None, 
               actions: list = None,
               q_table: np.ndarray = None,
               selection_function: callable = None,
               function_args: dict = None,
               alpha: float = 0.1, 
               gamma: float = 0.9, 
               agent_start: Tuple[int,int] = None,
               enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False)) -> Tuple[list, float, int, list]:
    """
    Runs a single episode of the Q-learning algorithm.
    Returns a tuple containing the action sequence, total reward, steps taken, and the final Q-table.

    Args:
        grid_world (GridWorld, optional): The environment in which the agent operates. Defaults to None.
        agent (Agent, optional): The agent that interacts with the environment. Defaults to None.
        actions (list, optional): List of possible actions the agent can take. Defaults to None.
        q_table (np.ndarray, optional): Q-table used to store and update Q-values. Defaults to None.
        selection_function (callable, optional): Function used to select actions based on Q-values. Defaults to None.
        function_args (dict, optional): Arguments for the selection function. Defaults to None.
        alpha (float, optional): Learning rate for Q-learning updates. Defaults to 0.1.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
        agent_start (Tuple[int, int], optional): Starting position of the agent. Defaults to None.
        enable_record (Tuple[bool, bool, bool, bool], optional): Flags to enable recording of action sequence, steps taken, total reward, and Q-table updates. Defaults to (False, False, False, False).

    Raises:
        ValueError: If any of the required parameters (grid_world, actions, q_table, selection_function) are None.
        ValueError: If selection_function is not callable or its arguments are invalid.

    Returns:
        Tuple[list, float, int, list]: A tuple containing:
            - action_sequence (list): Sequence of actions taken by the agent.
            - total_reward (float): Total reward accumulated during the episode.
            - steps_taken (int): Number of steps taken to reach the goal.
            - final_q_table (list): The final Q-table after the episode.
    """
    
    # Check if any of the parameters are None
    if grid_world is None:
        raise ValueError("GridWorld cannot be None!")
    if actions is None:
        raise ValueError("Actions cannot be None!")
    if q_table is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())

    if not callable(selection_function):
        raise ValueError("Selection function must be callable!")
    try: 
        test_state = grid_world.get_state()[1]
        selection_function(test_state, **function_args)
    except TypeError as e:
        raise ValueError(f"Selection function arguments are invalid: {e}")

    grid_world.reset(agent_start)  # Initializes the agent and environment state

    action_sequence = []
    final_q_table = None
    steps_taken = 0
    total_reward = 0

    goal_reached = False

    while not goal_reached:
        state = grid_world.get_state()[1]  # Get the current state of the environment
        action = selection_function(state, **function_args)
        
        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None

        next_state = grid_world.get_state()[1]  # Get the next state of the environment

        print(f"Action: {action}, Previous State: {state}, Next State: {next_state}")
        print(f"Move Successful: {agent._is}")
        Q_learning_table_update(state, next_state, action, reward, q_table, alpha, gamma)

    final_q_table = q_table.copy() if enable_record[3] else None

    return action_sequence, total_reward, steps_taken, final_q_table

def Q_lambda_episode(grid_world: GridWorld = None, 
                     agent: Agent = None, 
                     actions: list = None,
                     q_table: np.ndarray = None,
                     selection_function: callable = None,
                     function_args: dict = None,
                     alpha: float = 0.1, 
                     gamma: float = 0.9, 
                     lambda_: float = 0.9, 
                     agent_start: Tuple[int,int] = None,
                     enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False)) -> Tuple[list, float, int, list]:
    """
    Runs a single episode of the Q(λ) algorithm.
    Returns a tuple containing the action sequence, total reward, steps taken, and the final Q-table.

    Args:
        grid_world (GridWorld, optional): The environment in which the agent operates. Defaults to None.
        agent (Agent, optional): The agent that interacts with the environment. Defaults to None.
        actions (list, optional): List of possible actions the agent can take. Defaults to None.
        q_table (np.ndarray, optional): Q-table used to store and update Q-values. Defaults to None.
        selection_function (callable, optional): Function used to select actions based on Q-values. Defaults to None.
        function_args (dict, optional): Arguments for the selection function. Defaults to None.
        alpha (float, optional): Learning rate for Q-learning updates. Defaults to 0.1.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
        lambda_ (float, optional): Decay rate for eligibility traces. Defaults to 0.9.
        agent_start (Tuple[int, int], optional): Starting position of the agent. Defaults to None.
        enable_record (Tuple[bool, bool, bool, bool], optional): Flags to enable recording of action sequence, steps taken, total reward, and Q-table updates. Defaults to (False, False, False, False).

    Raises:
        ValueError: If any of the required parameters (grid_world, actions, q_table, selection_function) are None.
        ValueError: If selection_function is not callable or its arguments are invalid.

    Returns:
        Tuple[list, float, int, list]: A tuple containing:
            - action_sequence (list): Sequence of actions taken by the agent.
            - total_reward (float): Total reward accumulated during the episode.
            - steps_taken (int): Number of steps taken to reach the goal.
            - final_q_table (list): The final Q-table after the episode.
    """

    # Parameter checks
    if grid_world is None:
        raise ValueError("GridWorld cannot be None!")
    if actions is None:
        raise ValueError("Actions cannot be None!")
    if q_table is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())

    if not callable(selection_function):
        raise ValueError("Selection function must be callable!")
    try: 
        test_state = grid_world.get_state()[1]
        selection_function(test_state, **function_args)
    except TypeError as e:
        raise ValueError(f"Selection function arguments are invalid: {e}")

    grid_world.reset(agent_start)

    action_sequence = []
    final_q_table = None
    steps_taken = 0
    total_reward = 0

    goal_reached = False

    # Initialize eligibility traces (same shape as Q-table)
    e_table = np.zeros_like(q_table)

    while not goal_reached:
        state = grid_world.get_state()[1] # Get the current state of the environment
        action = selection_function(state, **function_args)

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None

        next_state = grid_world.get_state()[1] # Get the next state of the environment

        Q_lambda_table_update(state, next_state, action, reward, q_table, e_table, alpha, gamma, lambda_)

    final_q_table = q_table.copy() if enable_record[3] else None

    return action_sequence, total_reward, steps_taken, final_q_table

"""
====================================================================================================
LEARNING UPDATE FUNCTIONS
====================================================================================================
"""

def Q_learning_table_update(state: Tuple[int, ...] = None,
                           next_state: Tuple[int, ...] = None, 
                           action: int = None, 
                           reward: float = None, 
                           q_table: np.ndarray = None,
                           alpha: float = 0.1, 
                           gamma: float = 0.9):
    """
    Updates the Q-table using the Q-learning algorithm.

    Args:
        state (Tuple[int, ...], optional): The current state of the environment. Defaults to None.
        next_state (Tuple[int, ...], optional): The next state of the environment. Defaults to None.
        action (int, optional): The action taken by the agent. Defaults to None.
        reward (float, optional): The reward received after taking the action. Defaults to None.
        q_table (np.ndarray, optional): Array of Q-values for each state-action pair. Defaults to None.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Raises:
        ValueError: If q_table is None.
        ValueError: If state is None.
        ValueError: If next_state is None.
        ValueError: If action is None.
        ValueError: If reward is None.
    """
    if q_table is None:
        raise ValueError("q_table cannot be None!")
    if state is None:
        raise ValueError("state cannot be None!")
    if next_state is None:
        raise ValueError("next_state cannot be None!")
    if action is None:
        raise ValueError("action cannot be None!")
    if reward is None:
        raise ValueError("reward cannot be None!")
    
    try:
        q_table[(*state, action)]
    except TypeError as e:
        raise ValueError("state and action must be usable to access the q_table!")
    
    # Compute the TD error
    td_error = (reward 
                + gamma * np.max(q_table[(*next_state, )]) 
                - q_table[(*state, action)])

    # Update the Q-value for the state-action pair
    q_table[(*state, action)] += alpha * td_error

    pass

def Q_lambda_table_update(state: Tuple[int, ...] = None,
                          next_state: Tuple[int, ...] = None, 
                          action: int = None, 
                          reward: float = None, 
                          q_table: np.ndarray = None,
                          e_table: np.ndarray = None,
                          alpha: float = 0.1, 
                          gamma: float = 0.9,
                          lambda_: float = 0.9):
    """
    Updates the Q-table and eligibility traces using the Q(λ) algorithm.

    Args:
        state (Tuple[int, ...], optional): The current state of the environment. Defaults to None.
        next_state (Tuple[int, ...], optional): The next state of the environment. Defaults to None.
        action (int, optional): The action taken by the agent. Defaults to None.
        reward (float, optional): The reward received after taking the action. Defaults to None.
        q_table (np.ndarray, optional): Array of Q-values for each state-action pair. Defaults to None.
        e_table (np.ndarray, optional): Array of eligibility traces for each state-action pair. Defaults to None.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.
        lambda_ (float, optional): Decay rate for eligibility traces. Defaults to 0.9.

    Raises:
        ValueError: If q_table is None.
        ValueError: If e_table is None.
        ValueError: If state is None.
        ValueError: If next_state is None.
        ValueError: If action is None.
        ValueError: If reward is None.
        ValueError: If state and action cannot be used to access the q_table and e_table.
    """
    if q_table is None:
        raise ValueError("q_table cannot be None!")
    if e_table is None:
        raise ValueError("e_table cannot be None!")
    if state is None:
        raise ValueError("state cannot be None!")
    if next_state is None:
        raise ValueError("next_state cannot be None!")
    if action is None:
        raise ValueError("action cannot be None!")
    if reward is None:
        raise ValueError("reward cannot be None!")

    try:
        q_table[(*state, action)]
        e_table[(*state, action)]
    except TypeError as e:
        raise ValueError("state and action must be usable to access the q_table and e_table!")

    # Compute TD error 
    td_error = (reward 
                + gamma * np.max(q_table[(*next_state,)]) 
                - q_table[(*state, action)])

    # Update eligibility trace for the current state-action pair
    e_table[(*state, action)] += 1  # Replaces "replacing traces" method

    # Update Q-values for all state-action pairs
    q_table += alpha * td_error * e_table

    # Decay eligibility traces
    e_table *= gamma * lambda_

def Q_learning_RBF_update(phi_state: np.ndarray = None,
                            next_phi_state: np.ndarray = None,
                            action: int = None,
                            reward: float = None,
                            q_table: np.ndarray = None,
                            alpha: float = 0.1,
                            gamma: float = 0.9):
    """
    Updates the Q-table using the Q-learning algorithm with Radial Basis Function (RBF) approximation.

    Args:
        phi_state (np.ndarray, optional): Feature vector for the current state. Defaults to None.
        next_phi_state (np.ndarray, optional): Feature vector for the next state. Defaults to None.
        action (int, optional): The action taken by the agent. Defaults to None.
        reward (float, optional): The reward received after taking the action. Defaults to None.
        q_table (np.ndarray, optional): Array of weights for the RBF approximator. Defaults to None.
        phi_centers (np.ndarray, optional): Centers of the RBFs. Defaults to None.
        sigma (float, optional): Standard deviation of the RBFs. Defaults to 1.0.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Raises:
        ValueError: If phi_state is None.
        ValueError: If next_phi_state is None.
        ValueError: If action is None.
        ValueError: If reward is None.
        ValueError: If q_table is None.
        ValueError: If phi_centers is None.
    """
    if phi_state is None:
        raise ValueError("phi_state cannot be None!")
    if next_phi_state is None:
        raise ValueError("next_phi_state cannot be None!")
    if action is None:
        raise ValueError("action cannot be None!")
    if reward is None:
        raise ValueError("reward cannot be None!")
    if q_table is None:
        raise ValueError("q_table cannot be None!")

    # Compute the TD error
    td_error = (reward 
                + gamma * np.max(np.dot(q_table, next_phi_state)) 
                - np.dot(q_table[action], phi_state))

    # Update the weights for the action taken
    q_table[action] += phi_state * alpha * td_error

    pass

"""
====================================================================================================
SELECTION FUNCTIONS
====================================================================================================
"""

def epsilon_greedy_Q_selection(state: Tuple[int, ...], q_table: np.ndarray = None, epsilon: float = 0.1) -> int:
    """
    Selects an action using the epsilon-greedy policy.

    Args:
        state (Tuple[int, ...]): The current state of the environment.
        q_table (np.ndarray, optional): Array of Q-values for each action. Defaults to None.
        epsilon (float, optional): Probability of choosing a random action. Defaults to 0.1.

    Raises:
        ValueError: If q_table is None.

    Returns:
        int: Index of the selected action.
    """
    if q_table is None:
        raise ValueError("q_table cannot be None!")
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_table[(*state,)]))  # Return a random action
    else:  # Return the action with the highest Q-value
        return np.argmax(q_table[(*state,)])
    
    pass

def decaying_epsilon_greedy_Q_selection(state: Tuple[int, ...], q_table: np.ndarray = None, epsilon: float = 0.1, decay: float = 0.99, episode: int = None) -> int:
    """decaying_epsilon_greedy_Q_selection _summary_

    Args:
        state (Tuple[int, ...]): _description_
        q_table (np.ndarray, optional): _description_. Defaults to None.
        epsilon (float, optional): _description_. Defaults to 0.1.
        decay (float, optional): _description_. Defaults to 0.99.
        episode (int, optional): _description_. Defaults to None.

    Returns:
        int: _description_
    """
    if q_table is None:
        raise ValueError("q_table cannot be None!")
    if episode is None:
        raise ValueError("episode cannot be None!")
    if np.random.rand() < epsilon * decay**episode:
        return np.random.choice(len(q_table[(*state,)]))
    else: # Return the action with the highest Q-value
        return np.argmax(q_table[(*state,)])
    pass

def softmax_Q_selection(state: Tuple[int, ...], q_table: np.ndarray = None, tau: float = 0.1) -> int:
    """
    Selects an action using the softmax policy.

    Args:
        state (Tuple[int, ...]): The current state of the environment.
        q_table (np.ndarray, optional): Array of Q-values for each action. Defaults to None.
        tau (float, optional): Temperature parameter for the softmax function. Defaults to 0.1.

    Returns:
        int: Index of the selected action.
    """
    if state is None:
        raise ValueError("state cannot be None!")
    if q_table is None:
        raise ValueError("q_table cannot be None!")

    q_values = q_table[(*state, )] # Get the possible Q-values for the current state
    probabilities = np.exp(q_values / tau) / np.sum(np.exp(q_values / tau)) # Softmax + normalization of possible Q-values
    print(probabilities)
    return np.random.choice(len(q_values), p=probabilities) # Return an action based on the probabilities

    pass

def softmax_P_selection(phi_state: np.ndarray = None, q_table: np.ndarray = None, tau: float = 0.1) -> int:
    """
    Selects an action using the softmax policy.

    Args:
        weight_vector (np.ndarray): The current state of the environment. Defaults to None.
        q_table (np.ndarray, optional): Array of Q-values of the Phi approximator. Defaults to None.
        tau (float, optional): Temperature parameter for the softmax function. Defaults to 0.1.
    """
    if phi_state is None:
        raise ValueError("weight_vector cannot be None!")
    if q_table is None:
        raise ValueError("q_table cannot be None!")

    q_values = np.dot(q_table, phi_state) # Get the weighted Q-values for the current state
    probabilities = np.exp(q_values / tau) / np.sum(np.exp(q_values / tau)) # Softmax + normalization of possible Q-values
    return np.random.choice(len(q_values), p=probabilities) # Return an action based on the probabilities

    pass

"""
====================================================================================================
UTILITIES FUNCTIONS
====================================================================================================
"""

def gaussian_RBF(state: Tuple[int, ...], center: Tuple[int, ...], sigma: float = 1.0) -> float:
    """
    Computes the resulting Gaussian Radial Basis Function output weight for a given state and center.

    Args:
        state (Tuple[int, ...]): The current state of the environment.
        center (Tuple[int, ...]): The center of the RBF.
        sigma (float, optional): The standard deviation of the RBF. Defaults to 1.0.

    Returns:
        float: The value of the RBF.
    """
    distance = np.linalg.norm(np.array(state) - np.array(center))
    return np.exp(-distance**2 / (2 * sigma))