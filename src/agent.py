"""
agent.py

Description: This module defines an Agent class that can move within a grid.
Author: Lucas Pinto
Date: February 10, 2025

Modules:
    None

Classes:
    Agent

Functions:
    None

Usage:
    agent = Agent((0, 0))
    agent.move('up', (5, 5))
"""
from typing import Tuple

class Agent:
    """
    A class to represent an agent that can move within a grid.
    """

    def __init__(self, position: Tuple[int, int] = (0, 0)):
        """
        Initialize the agent with a starting position.

        Args:
            position (Tuple[int, int], optional): The initial position of the agent as (x, y). Defaults to (0, 0).
        """
        self.position = position

    def move(self, action: str = None, grid_dim: Tuple[int, int] = (5, 5)) -> bool:
        """
        Move the agent in the specified direction within the grid dimensions.

        Args:
            action (str, optional): The direction in which to move the agent. 
                        Can be 'up', 'down', 'left', or 'right'. Defaults to None.
            grid_dim (Tuple[int, int], optional): The dimensions of the grid as (length, width). 
                             Defaults to (5, 5).

        Returns:
            bool: True if the agent moved successfully, False otherwise.
        """
        x, y = self.position
        grid_length, grid_width = grid_dim

        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < grid_width - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < grid_length - 1:
            x += 1
        else:
            return False

        self.position = (x, y)
        return True
