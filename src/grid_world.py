"""
grid_world.py

Description: This module defines a GridWorld class that simulates a grid-based environment where an agent can move and receive rewards.
Author: Lucas Pinto
Date: February 10, 2025

Modules:
    numpy
    matplotlib
    functions

Functions:
    helloWorld

Usage:
    python main.py
"""
import numpy as np
from typing import Tuple

from agent import Agent

class GridWorld:
    def __init__(self, grid_dim: Tuple[int,int] = (5, 5), agent: Agent = None, goal: Tuple[int,int] = None, reward_vector: list = None):
        """
        Initialize the GridWorld with dimensions, agent, and goal.

        Args:
            grid_dim (tuple, optional): Dimensions of the grid as (rows, columns). Defaults to (5, 5).
            agent (Agent, optional): An instance of the Agent class. Defaults to None.
            goal (tuple, optional): Coordinates of the goal position as (x, y). Defaults to (4, 4).
            reward_vector (list, optional): List of rewards for different actions. Defaults to [10, -0.1, -1].
        """
        self._grid_dim = grid_dim
        self._agent = agent if agent is not None else Agent()
        self._goal = goal if goal is not None else (self._grid_dim[0] - 1, self._grid_dim[1] - 1) # If no goal is provided, set it to the bottom-right corner
        self._grid = np.zeros(grid_dim)
        self._grid[self._agent.position[0], self._agent.position[1]] = 2 # Represents the agent as a 2 in the grid
        self._reward_vector = reward_vector if reward_vector is not None else [10, -0.1, -1] # If no reward vector is provided, set it to [10, -0.1, -1]

    def _move_agent(self, action: str = None) -> bool:
        """
        Move the agent in the specified direction and update the grid.

        Args:
            action (str, optional): The direction to move the agent. Can be 'up', 'down', 'left', or 'right'. Defaults to None.

        Returns:
            bool: True if the action was successful, False otherwise.
        """
        self._grid[self._agent.position[0], self._agent.position[1]] = 0
        move_successful = self._agent.move(action, self._grid_dim)
        self._grid[self._agent.position[0], self._agent.position[1]] = 2
        return move_successful
 
    
    def _reset_agent(self, agent_position: Tuple[int,int] = None):
        """
        Reset the agent's position to a specified or random position within the grid excluding the goal.

        Args:
            agent_position (tuple, optional): The (x, y) coordinates to reset the agent to. Defaults to None.
        """
        self._grid[self._agent.position[0], self._agent.position[1]] = 0
        if agent_position is not None:
            self._agent.position = agent_position
            self._grid[self._agent.position[0], self._agent.position[1]] = 2
        else: # If no agent position is provided, set it to a random position within the grid
            while True:
                self._agent.position = (np.random.choice(self._grid_dim[0]), np.random.choice(self._grid_dim[1]))
                if self._agent.position != self._goal:
                    break
            self._grid[self._agent.position[0], self._agent.position[1]] = 2
    
    def _is_goal_reached(self) -> bool:
        """
        Check if the agent has reached the goal.

        Returns:
            bool: True if the agent has reached the goal, False otherwise.
        """
        return self._agent.position == self._goal
    
    def _get_reward(self, move_successful: bool = None) -> float:
        """
        Get the reward based on the agent's action and position.

        Args:
            move_successful (bool, optional): Indicates if the agent's move was successful. Defaults to None.

        Returns:
            float: The reward value based on the agent's action and position.
        """
        if self._is_goal_reached():
            return self._reward_vector[0] # If the agent has reached the goal, return a reward of 10
        elif move_successful:
            return self._reward_vector[1] # If the agent has moved successfully, return a reward of -0.1
        else: 
            return self._reward_vector[2] # If the agent has hit a wall/invalid move, return a reward of -1

    def _get_agent_position(self) -> Tuple[int, int]:
        """
        Get the current position of the agent.

        Returns:
            tuple: The (x, y) coordinates of the agent's current position.
        """
        return self._agent.position
    
    def step_agent(self, action: str) -> Tuple[float, bool]:
        """
        Perform a step in the environment by moving the agent.

        Args:
            action (str): The direction to move the agent. Can be 'up', 'down', 'left', or 'right'.

        Returns:
            tuple: A tuple containing the reward for the action and a boolean indicating if the goal has been reached.
        """
        move_successful = self._move_agent(action)
        reward = self._get_reward(move_successful)
        done = self._is_goal_reached()

        return reward, done

    def set_agent(self, agent: Agent = None):
        """
        Set the agent for the environment.

        Args:
            agent (Agent, optional): An instance of the Agent class. Defaults to None.
        """
        self._agent = agent if agent is not None else Agent()
        self._reset_agent()

    def get_agent(self) -> Agent:
        """
        Get the agent in the environment.

        Returns:
            Agent: The agent in the environment.
        """
        return self._agent
    
    def get_state(self) -> Tuple[np.ndarray, Tuple[int,int]]:
        """
        Get the current state of the grid and the agent's position.

        Returns:
            tuple: A tuple containing the grid and the agent's current position.
        """
        return self._grid, self._get_agent_position()
    
    def reset(self, agent_position: Tuple[int,int] = None):
        """
        Reset the environment to its initial state and randomizes agent position.

        Args:
            agent_position (tuple, optional): The (x, y) coordinates to reset the agent to. Defaults to None.
        """
        self._grid = np.zeros(self._grid_dim, dtype=int)
        if agent_position is not None:
            self._reset_agent(agent_position)
        else:
            self._reset_agent()