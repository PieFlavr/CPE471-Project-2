"""
learning.py

Description: This module implements the Q-learning and Q(lambda) algorithms for a GridWorld environment. 
            It includes functions for running episodes, selecting actions using various policies, and updating the Q-table and eligibility traces.
Author: Lucas Pinto
Date: February 12, 2025

Modules:
    numpy - For numerical operations on arrays.
    utils - Utility functions used in the project.
    grid_world - The GridWorld environment class.
    agent - The Agent class that interacts with the environment.
    typing - For type hinting.

Functions:
    RBF_Q_learning_episode - Runs a single episode of the Q-learning algorithm with Radial Basis Function (RBF) approximation.
    Q_learning_episode - Runs a single episode of the Q-learning algorithm.
    Q_lambda_episode - Runs a single episode of the Q(λ) algorithm.
    Q_learning_table_update - Updates the Q-table using the Q-learning algorithm.
    Q_lambda_table_update - Updates the Q-table and eligibility traces using the Q(λ) algorithm.
    Q_learning_RBF_update - Updates the Q-table using the Q-learning algorithm with RBF approximation.
    epsilon_greedy_Q_selection - Selects an action using the epsilon-greedy policy.
    decaying_epsilon_greedy_Q_selection - Selects an action using the decaying epsilon-greedy policy.
    softmax_Q_selection - Selects an action using the softmax policy.
    softmax_P_selection - Selects an action using the softmax policy with RBF approximation.
    gaussian_RBF - Computes the resulting Gaussian Radial Basis Function output weight for a given state and center.

Usage:
    python main.py
"""


import numpy as np
import sys

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
                           actions: dict = None,
                           weights: np.ndarray = None,
                           phi_centers: np.ndarray = None,
                           sigma: float = 1.0,
                           selection_function: callable = None,
                           function_args: dict = None,
                           alpha: float = 0.1,
                           gamma: float = 0.9,
                           agent_start: Tuple[int,int] = None,
                           enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False),
                           episode: int = None,
                           **kwargs) -> Tuple[list, float, int, list]:
    
    # Check if any of the parameters are None
    if grid_world is None:
        raise ValueError("GridWorld cannot be None!")
    if actions is None:
        raise ValueError("Actions cannot be None!")
    if weights is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())
    if episode is None:
        raise ValueError("Episode cannot be None!")
    if not callable(selection_function):
        raise ValueError("Selection function must be callable!")
    try: 
        test_state = grid_world.get_state()[1]
        test_phi_state = np.array([gaussian_RBF(test_state, center, sigma) for center in phi_centers])
        selection_function(test_phi_state, **function_args)
    except TypeError as e:
        raise ValueError(f"Selection function arguments are invalid: {e}")

    grid_world.reset(agent_start)  # Initializes the agent and environment state

    action_sequence = []
    final_weights = None
    steps_taken = 0
    total_reward = 0

    goal_reached = False

    while not goal_reached:
        state = grid_world.get_state()[1]  # Get the current state of the environment
        phi_state = np.array([gaussian_RBF(state, center, sigma) for center in phi_centers])

        action = selection_function(phi_state, episode = episode, **function_args)

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None

        next_state = grid_world.get_state()[1]  # Get the next state of the environment
        next_phi_state = np.array([gaussian_RBF(next_state, center, sigma) for center in phi_centers])
        #print(f"Actual state: {state}, Phi state: {phi_state}, Action: {action}, Reward: {reward}, Next state: {next_state}, Next Phi state: {next_phi_state}")

        Q_learning_RBF_update(phi_state, next_phi_state, action, reward, weights, alpha, gamma)

    final_weights = weights.copy() if enable_record[3] else None

    return action_sequence, total_reward, steps_taken, final_weights

    pass
       
def FSR_Q_learning_episode(grid_world: GridWorld = None, 
               agent: Agent = None, 
               actions: dict = None,
               weights: np.ndarray = None,
               selection_function: callable = None,
               function_args: dict = None,
               alpha: float = 0.1, 
               gamma: float = 0.9, 
               agent_start: Tuple[int,int] = None,
               enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False),
               episode: int = None,
               **kwargs) -> Tuple[list, float, int, list]:
    """_summary_

    Args:
        grid_world (GridWorld, optional): _description_. Defaults to None.
        agent (Agent, optional): _description_. Defaults to None.
        actions (list, optional): _description_. Defaults to None.
        fsr (np.ndarray, optional): _description_. Defaults to None.
        selection_function (callable, optional): _description_. Defaults to None.
        function_args (dict, optional): _description_. Defaults to None.
        alpha (float, optional): _description_. Defaults to 0.1.
        gamma (float, optional): _description_. Defaults to 0.9.
        agent_start (Tuple[int,int], optional): _description_. Defaults to None.
        enable_record (Tuple[bool, bool, bool, bool], optional): _description_. Defaults to (False, False, False, False).

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        Tuple[list, float, int, list]: _description_
    """
    
    # Check if any of the parameters are None
    if grid_world is None:
        raise ValueError("GridWorld cannot be None!")
    if actions is None:
        raise ValueError("Actions cannot be None!")
    if weights is None:
        raise ValueError("FSR vector cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())
    if episode is None:
        raise ValueError("Episode cannot be None!")
    if not callable(selection_function):
        raise ValueError("Selection function must be callable!")
    try: 
        test_state = grid_world.get_state()[1]
        selection_function(test_state, episode = episode, grid_world = grid_world, actions = actions, **function_args)
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
        action = selection_function(state, episode = episode, grid_world = grid_world, actions = actions, **function_args)

        fsr_state = action_state_to_FSR(state, grid_world = grid_world, actions = actions, action = action)
        #print(f"State: {state}, Action: {action}, I_Action: {get_key_by_value(actions,action)}, FSR: {fsr_state}")

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None
        #print (f"\033[A FSR State: {fsr_state}, Action: {action}, Reward: {reward} \033[A")

        next_state = grid_world.get_state()[1]  # Get the next state of the environment

        next_fsr_state = generate_next_FSR(next_state, grid_world = grid_world, actions = actions)

        Q_learning_FSR_update(fsr_state, next_fsr_state, action, actions, reward, weights, alpha, gamma)

    final_q_table = weights.copy() if enable_record[3] else None

    return action_sequence, total_reward, steps_taken, final_q_table

def Q_learning_episode(grid_world: GridWorld = None, 
               agent: Agent = None, 
               actions: dict = None,
               weights: np.ndarray = None,
               selection_function: callable = None,
               function_args: dict = None,
               alpha: float = 0.1, 
               gamma: float = 0.9, 
               agent_start: Tuple[int,int] = None,
               enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False),
               episode: int = None,
               **kwargs) -> Tuple[list, float, int, list]:
    """
    Runs a single episode of the Q-learning algorithm.
    Returns a tuple containing the action sequence, total reward, steps taken, and the final Q-table.

    Args:
        grid_world (GridWorld, optional): The environment in which the agent operates. Defaults to None.
        agent (Agent, optional): The agent that interacts with the environment. Defaults to None.
        actions (list, optional): List of possible actions the agent can take. Defaults to None.
        weights (np.ndarray, optional): Q-table used to store and update Q-values. Defaults to None.
        selection_function (callable, optional): Function used to select actions based on Q-values. Defaults to None.
        function_args (dict, optional): Arguments for the selection function. Defaults to None.
        alpha (float, optional): Learning rate for Q-learning updates. Defaults to 0.1.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
        agent_start (Tuple[int, int], optional): Starting position of the agent. Defaults to None.
        enable_record (Tuple[bool, bool, bool, bool], optional): Flags to enable recording of action sequence, steps taken, total reward, and Q-table updates. Defaults to (False, False, False, False).

    Raises:
        ValueError: If any of the required parameters (grid_world, actions, weights, selection_function) are None.
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
    if weights is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if episode is None:
        raise ValueError("Episode cannot be None!")
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
        action = selection_function(state, episode = episode, **function_args)

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None

        next_state = grid_world.get_state()[1]  # Get the next state of the environment

        Q_learning_table_update(state, next_state, action, reward, weights, alpha, gamma)

    final_q_table = weights.copy() if enable_record[3] else None

    return action_sequence, total_reward, steps_taken, final_q_table

def Q_lambda_episode(grid_world: GridWorld = None, 
                     agent: Agent = None, 
                     actions: dict = None,
                     weights: np.ndarray = None,
                     selection_function: callable = None,
                     function_args: dict = None,
                     alpha: float = 0.1, 
                     gamma: float = 0.9, 
                     lambda_: float = 0.9, 
                     agent_start: Tuple[int,int] = None,
                     enable_record: Tuple[bool, bool, bool, bool] = (False, False, False, False),
                     episode: int = None,
                     **kwargs) -> Tuple[list, float, int, list]:
    """
    Runs a single episode of the Q(λ) algorithm.
    Returns a tuple containing the action sequence, total reward, steps taken, and the final Q-table.

    Args:
        grid_world (GridWorld, optional): The environment in which the agent operates. Defaults to None.
        agent (Agent, optional): The agent that interacts with the environment. Defaults to None.
        actions (list, optional): List of possible actions the agent can take. Defaults to None.
        weights (np.ndarray, optional): Q-table used to store and update Q-values. Defaults to None.
        selection_function (callable, optional): Function used to select actions based on Q-values. Defaults to None.
        function_args (dict, optional): Arguments for the selection function. Defaults to None.
        alpha (float, optional): Learning rate for Q-learning updates. Defaults to 0.1.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
        lambda_ (float, optional): Decay rate for eligibility traces. Defaults to 0.9.
        agent_start (Tuple[int, int], optional): Starting position of the agent. Defaults to None.
        enable_record (Tuple[bool, bool, bool, bool], optional): Flags to enable recording of action sequence, steps taken, total reward, and Q-table updates. Defaults to (False, False, False, False).

    Raises:
        ValueError: If any of the required parameters (grid_world, actions, weights, selection_function) are None.
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
    if weights is None:
        raise ValueError("Q-table cannot be None!")
    if selection_function is None:
        raise ValueError("Selection function cannot be None!")
    if agent is None:
        grid_world.set_agent(Agent())
    if episode is None:
        raise ValueError("Episode cannot be None!")
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
    e_table = np.zeros_like(weights)

    while not goal_reached:
        state = grid_world.get_state()[1] # Get the current state of the environment
        action = selection_function(state, episode = episode, **function_args)

        reward, goal_reached = grid_world.step_agent(get_key_by_value(actions, action))

        action_sequence.append(action) if enable_record[0] else None
        steps_taken += 1 if enable_record[1] else None
        total_reward += reward if enable_record[2] else None

        next_state = grid_world.get_state()[1] # Get the next state of the environment

        Q_lambda_table_update(state, next_state, action, reward, weights, e_table, alpha, gamma, lambda_)

    final_q_table = weights.copy() if enable_record[3] else None

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
                           gamma: float = 0.9,
                           **kwargs):
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
                          lambda_: float = 0.9,
                          **kwargs):
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
                            weights: np.ndarray = None,
                            alpha: float = 0.1,
                            gamma: float = 0.9,
                            **kwargs):
    """
    Updates the Q-table using the Q-learning algorithm with Radial Basis Function (RBF) approximation.

    Args:
        phi_state (np.ndarray, optional): Feature vector for the current state. Defaults to None.
        next_phi_state (np.ndarray, optional): Feature vector for the next state. Defaults to None.
        action (int, optional): The action taken by the agent. Defaults to None.
        reward (float, optional): The reward received after taking the action. Defaults to None.
        weights (np.ndarray, optional): Array of weights for the RBF approximator. Defaults to None.
        phi_centers (np.ndarray, optional): Centers of the RBFs. Defaults to None.
        sigma (float, optional): Standard deviation of the RBFs. Defaults to 1.0.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Raises:
        ValueError: If phi_state is None.
        ValueError: If next_phi_state is None.
        ValueError: If action is None.
        ValueError: If reward is None.
        ValueError: If weights is None.
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
    if weights is None:
        raise ValueError("weights cannot be None!")

    # Compute the TD error
    td_error = (reward 
                + gamma * np.max(np.dot(weights, next_phi_state)) 
                - np.dot(weights[action], phi_state))

    # Update the weights for the action taken
    weights[action] += phi_state * alpha * td_error

    pass

def Q_learning_FSR_update(fsr_state: np.ndarray = None, 
                         next_fsr_states: np.ndarray = None,
                         action: int = None, actions: dict = None,
                         reward: float = None,
                         weights: np.ndarray = None,
                         alpha: float = 0.1,
                         gamma: float = 0.9,
                         **kwargs):
    """
    Updates the Q-table using the Q-learning algorithm with Feature State Representation (FSR).

    Args:
        fsr_state (np.ndarray, optional): Feature vector for the current state. Defaults to None.
        next_fsr_state (np.ndarray, optional): Feature vector for the next state. Defaults to None.
        action (int, optional): The action taken by the agent. Defaults to None.
        reward (float, optional): The reward received after taking the action. Defaults to None.
        weights (np.ndarray, optional): Array of weights for the FSR approximator. Defaults to None.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Raises:
        ValueError: If fsr_state is None.
        ValueError: If next_fsr_state is None.
        ValueError: If action is None.
        ValueError: If reward is None.
        ValueError: If weights is None.
    """
    if fsr_state is None:
        raise ValueError("fsr_state cannot be None!")
    if next_fsr_states is None:
        raise ValueError("next_fsr_states cannot be None!")
    if action is None:
        raise ValueError("action cannot be None!")
    if reward is None:
        raise ValueError("reward cannot be None!")
    if weights is None:
        raise ValueError("weights cannot be None!")

    # Compute the Q-values for the  next states
    next_q_values = np.array([np.dot(weights, next_fsr_states[action]) for action in range(len(actions))])

    # Compute the TD error
    td_error = (reward 
                + gamma * np.max(next_q_values) 
                - np.dot(weights, fsr_state))
    
    #print(f"TD error: {td_error}")
    #print(f"Weights Before: {weights}")
    # Update the weights for the action taken
    weights +=  fsr_state * alpha * td_error
    #print(f"Weights After: {weights}")

        
"""
====================================================================================================
SELECTION FUNCTIONS
====================================================================================================
"""

def epsilon_greedy_Q_selection(state: Tuple[int, ...], weights: np.ndarray = None, epsilon: float = 0.1, **kwargs) -> int:
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
    if weights is None:
        raise ValueError("q_table cannot be None!")
    if np.random.rand() < epsilon:
        return np.random.choice(len(weights[(*state,)]))  # Return a random action
    else:  # Return the action with the highest Q-value
        return np.argmax(weights[(*state,)])
    
    pass

def decaying_epsilon_greedy_Q_selection(state: Tuple[int, ...], weights: np.ndarray = None, epsilon: float = 0.1, decay: float = 0.99, episode: int = None, **kwargs) -> int:
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
    if weights is None:
        raise ValueError("q_table cannot be None!")
    if episode is None:
        raise ValueError("episode cannot be None!")
    if np.random.rand() < epsilon * decay**episode:
        return np.random.choice(len(weights[(*state,)]))
    else: # Return the action with the highest Q-value
        return np.argmax(weights[(*state,)])
    pass

def softmax_Q_selection(state: Tuple[int, ...], weights: np.ndarray = None, tau: float = 0.1, greedy_cutoff: int = -1, episode: int = -1, **kwargs) -> int:
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
    if weights is None:
        raise ValueError("q_table cannot be None!")

    q_values = weights[(*state, )] # Get the possible Q-values for the current state

    if (greedy_cutoff < 0 or episode < greedy_cutoff):
        max_q = np.max(q_values)            # Have to subtract the max value to avoid an INF and NaN crash
        stable_q_values = q_values - max_q  # i think it doesn't change probabilities? Since they work relative to another anyawy in softmax

        exponentiated_vals = np.exp(stable_q_values / tau) # Exponentiate the Q-values
        probabilities = exponentiated_vals / np.sum(exponentiated_vals) # Softmax + normalization of possible Q-values
        #print("Probabilities: ", probabilities)
        return np.random.choice(len(stable_q_values), p=probabilities) # Return an action based on the probabilities
    elif episode >= greedy_cutoff:
        return np.argmax(q_values)

    pass

def softmax_P_selection(phi_state: np.ndarray = None, weights: np.ndarray = None, tau: float = 0.1, greedy_cutoff: int = -1, episode: int = -1, **kwargs) -> int:
    """
    Selects an action using the softmax policy.

    Args:
        phi_state (np.ndarray): The current state of the environment. Defaults to None.
        weights (np.ndarray, optional): Array of Q-values of the Phi approximator. Defaults to None.
        tau (float, optional): Temperature parameter for the softmax function. Defaults to 0.1.
    """
    if phi_state is None:
        raise ValueError("phi_state cannot be None!")
    if weights is None:
        raise ValueError("weights cannot be None!")

    weight_values = np.dot(weights, phi_state) # Get the weighted Q-values for the current state

    if(greedy_cutoff < 0 or episode < greedy_cutoff):
        #print("Non-greedy action: ", np.random.choice(len(stable_weights), p=probabilities))
        max_q = np.max(weight_values)            # Have to subtract the max value to avoid an INF and NaN crash
        stable_weights = weight_values - max_q  # i think it doesn't change probabilities? Since they work relative to another anyawy in softmax

        exponentiated_vals = np.exp(stable_weights / tau) # Exponentiate the Q-values
        probabilities = exponentiated_vals / np.sum(exponentiated_vals) # Softmax + normalization of possible Q-values

        return np.random.choice(len(stable_weights), p=probabilities) # Return an action based on the probabilities
    elif episode >= greedy_cutoff:
        #print(np.argmax(np.dot(weights, phi_state)))
        #print("Greedy action: ", np.argmax(np.dot(weights, phi_state)))
        return np.argmax(weight_values)

    pass

def softmax_FSR_selection(state: Tuple[int, ...] = None, weights: np.ndarray = None, 
                          tau: float = 0.1, greedy_cutoff: int = -1, episode: int = -1, 
                          grid_world: GridWorld = None, actions: dict = None, **kwargs) -> int:
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
    if weights is None:
        raise ValueError("q_table cannot be None!")
    if actions is None:
        raise ValueError("actions cannot be None!")
    if grid_world is None:
        raise ValueError("grid_world cannot be None!")
    next_FSR_states = generate_next_FSR(state, grid_world, actions = actions)
    q_values = np.array([np.dot(weights, next_FSR_states[action]) for action in range(len(actions))])

    if (greedy_cutoff < 0 or episode < greedy_cutoff):
        max_q = np.max(q_values)            # Have to subtract the max value to avoid an INF and NaN crash
        stable_q_values = q_values - max_q  # i think it doesn't change probabilities? Since they work relative to another anyawy in softmax

        exponentiated_vals = np.exp(stable_q_values / tau) # Exponentiate the Q-values
        probabilities = exponentiated_vals / np.sum(exponentiated_vals) # Softmax + normalization of possible Q-values

        return np.random.choice(len(stable_q_values), p=probabilities) # Return an action based on the probabilitie
    elif episode >= greedy_cutoff:
        return np.argmax(q_values)

    pass
"""
====================================================================================================
UTILITIES FUNCTIONS
====================================================================================================
"""

def gaussian_RBF(state: Tuple[int, ...], center: Tuple[int, ...], sigma: float = 1.0, **kwargs) -> float:
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

def generate_RBF_centers(grid_dim: Tuple[int, int] = None, num_centers: int = 10) -> np.ndarray:
    """
    Generates RBF centers non-randomly by evenly distributing them across the grid.

    Args:
        grid_dim (Tuple[int, int], optional): Dimensions of the grid (rows, columns). Defaults to None.
        num_centers (int, optional): Number of RBF centers to generate. Defaults to 10.

    Returns:
        np.ndarray: Array of RBF centers.
    """
    if grid_dim is None:
        raise ValueError("grid_dim cannot be None!")

    rows, cols = grid_dim
    centers = []

    # Recursive function to place centers evenly
    def place_centers(start_row, end_row, start_col, end_col, remaining_centers):
        if remaining_centers <= 0 or start_row > end_row or start_col > end_col:
            return
        
        mid_row = round((start_row + end_row) / 2)
        mid_col = round((start_col + end_col) / 2)

        centers.append((mid_row, mid_col))
        remaining_centers -= 1

        if remaining_centers > 0:
            place_centers(start_row, mid_row - 1, start_col, mid_col - 1, remaining_centers // 4)
            place_centers(start_row, mid_row - 1, mid_col + 1, end_col, remaining_centers // 4)
            place_centers(mid_row + 1, end_row, start_col, mid_col - 1, remaining_centers // 4)
            place_centers(mid_row + 1, end_row, mid_col + 1, end_col, remaining_centers // 4)

    place_centers(1, rows - 2, 1, cols - 2, num_centers)

    return np.array(centers)

def generate_next_FSR(state: Tuple[int, ...] = None, grid_world: GridWorld = None, actions: dict = None) -> np.ndarray:
    """
    Generates all possible Feature State Representation (FSR) vectors for a given state.

    Args:
        state (Tuple[int, ...], optional): The current state of the environment. Defaults to None.
        grid_dim (Tuple[int, int], optional): Dimensions of the grid (rows, columns). Defaults to None.
        actions (list, optional): List of possible actions. Defaults to None.

    Returns:
        np.ndarray: Array of all possible FSR vectors.
    """
    if state is None:
        raise ValueError("state cannot be None!")
    if actions is None:
        raise ValueError("actions cannot be None!")
    if grid_world is None:
        raise ValueError("grid_world cannot be None!")

    fsr_vectors = []

    for action in range(len(actions)):
        fsr_vector = action_state_to_FSR(state, grid_world, actions, action)
        fsr_vectors.append(fsr_vector)

    return np.array(fsr_vectors)

def action_state_to_FSR(state: Tuple[int, ...] = None, grid_world: GridWorld = None, actions: dict = None, action: int = None) -> np.ndarray:
    """
    Converts a state to a Feature State Representation (FSR) vector.

    Args:
        state (Tuple[int, ...], optional): The current state of the environment. Defaults to None.
        grid_dim (Tuple[int, int], optional): Dimensions of the grid (rows, columns). Defaults to None.
        actions (list, optional): List of possible actions. Defaults to None.

    Returns:
        np.ndarray: The FSR vector for the given state.
    """
    if state is None:
        raise ValueError("state cannot be None!")
    if actions is None:
        raise ValueError("actions cannot be None!")
    if action is None:
        raise ValueError("action cannot be None!")
    if grid_world is None:
        raise ValueError("grid_world cannot be None!")

    rows, cols = grid_world._grid_dim

    #start_goal_distance = goal[0] + goal[1]
    #distance_to_goal = (goal[0] - state[0]) + (goal[1] - state[1]) -1
    #start_goal_distance = 0
    #distance_to_goal = 0

    #fsr = np.zeros(rows+cols+len(actions)+start_goal_distance, dtype = int)
    #fsr = np.zeros(rows*cols*len(actions), dtype = int)
    fsr = np.zeros((rows + cols) * len(actions))
    
    # Each feature vector will be enocded as follows [row_position...col_position...actions...]
    action_offset = (rows + cols) * action
    fsr[action_offset + state[0]] = 1  # Encode row position
    fsr[action_offset + rows + state[1]] = 1  # Encode column position
    
    #fsr[rows + cols + len(actions) + distance_to_goal] = 1  # Encode distance to goal

    return fsr