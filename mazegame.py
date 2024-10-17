import gym
from gym import spaces
import numpy as np
import pygame

sub_goal_reached = False

class MazeGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 30}

    def __init__(self, maze, render_mode=None):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)  # Maze represented as a 2D numpy array
        self.start_pos = np.array(np.where(self.maze == 'S')).T[0]  # Starting position
        self.goal_pos = np.array(np.where(self.maze == 'E')).T[0]  # Goal position
        self.sub_goal_pos = np.array(np.where(self.maze == 'G')).T[0]  # Sub Goal position
        self.current_pos = np.copy(self.start_pos)  # Current position of the agent
        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space: (row, col) position of the agent with low and high bounds
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),  # Min position (top-left corner)
            high=np.array([self.num_rows - 1, self.num_cols - 1]),  # Max position (bottom-right corner)
            dtype=np.int32
        )

        self.render_mode = render_mode
        self.path = []  # To store the path of the agent

        if self.render_mode == "human":
            # Initialize Pygame
            pygame.init()
            self.cell_size = 100
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))
            pygame.display.set_caption("Maze Game")
            self.clock = pygame.time.Clock()

    def reset(self):
        # Reset the agent to the start position
        self.current_pos = np.copy(self.start_pos)
        self.path = [tuple(self.current_pos)]  # Reset the path and add the start position
        return np.array(self.current_pos)

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.copy(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
            self.path.append(tuple(self.current_pos))  # Append the current position to the path

        global sub_goal_reached

        # Reward function
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 10
            done = True
        elif not sub_goal_reached and np.array_equal(self.current_pos, self.sub_goal_pos):
            reward = 2
            sub_goal_reached = True
            done = False
        else:
            reward = -1
            done = False

        return np.array(self.current_pos), reward, done, {}

    def _is_valid_position(self, pos):
        row, col = pos
        # If the agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False
        # If the agent hits an obstacle
        if self.maze[row, col] == '1':
            return False
        return True

    def render(self):
        if self.render_mode != "human":
            return

        # Handle Pygame events (such as closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the path the agent has taken so far
        for pos in self.path:
            row, col = pos
            cell_left = col * self.cell_size
            cell_top = row * self.cell_size
            pygame.draw.rect(self.screen, (255,79,0), (cell_left, cell_top, self.cell_size, self.cell_size))

        # Draw environment elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                # Draw obstacles, start, goal, and agent
                if self.maze[row, col] == '1':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Sub Goal position
                    pygame.draw.rect(self.screen, (245, 229, 7), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'E':  # Goal position
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(self.current_pos, [row, col]):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display
        self.clock.tick(30)  # Limit frame rate to 30 FPS

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
