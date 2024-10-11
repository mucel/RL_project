import numpy as np
import pandas as pd
import gym
import pygame
from gym.envs.registration import register
import mazegame


# maze = [
#     ['#', '#', '#', '#'],
#     ['.', 'S', '.', '.'],
#     ['.', '#', '.', '#'],
#     ['.', '.', '.', '.'],
#     ['#', '.', '#', 'G'],
# ]

maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'S', 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 'G', 0, 0, 0, 1, 0, 1],  # G is the sub-goal
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 'E', 0, 1],  # E is the end goal
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

register(
    id='MazeGame-v0',
    entry_point='mazegame:MazeGameEnv',  # Ensure this points to the correct module and class
    kwargs={'maze': None},  # Provide your maze structure here
)


def init_q_table():
    actions = np.array(['up', 'down', 'left', 'right'])
    q_table = pd.DataFrame(
        np.zeros((100, len(actions))), columns=actions
    )
    return q_table


def act_choose(state, q_table, epsilon):
    actions = np.array(['up', 'down', 'left', 'right'])
    state_act = q_table.iloc[state, :]

    if np.random.uniform() > epsilon or state_act.all() == 0:
        action = np.random.choice(actions)
    else:
        action = state_act.idxmax()
    return action



def map_action(action):
    action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    return action_map[action]


def update_q_table(q_table, state, action, next_state, terminal, gamma, alpha, reward):
    x = int(state)
    next_x = int(next_state)
    q_original = q_table.loc[x, action]
    if not terminal:
        q_predict = reward + gamma * q_table.iloc[next_x].max()
    else:
        q_predict = reward

    q_table.loc[x, action] = (1 - alpha) * q_original + alpha * q_predict
    
    return q_table


# Main Q-learning loop
def q_learning(env, max_episodes, alpha, gamma, epsilon):
    q_table = init_q_table()
    episodes = 0

    for episode in range(max_episodes):
        state = env.reset()
        state = state_to_bucket(state, env)

        done = False
        step = 0
   

        while not done:
            env.render()

            action = act_choose(state, q_table, epsilon)
            mapped_action = map_action(action)

            next_observation, reward, done, _ = env.step(mapped_action)
            # print(reward)
            next_state = state_to_bucket(next_observation, env) 

            q_table = update_q_table(q_table, state, action, next_state, done, gamma, alpha, reward)

            state = next_state
            step += 1

            if done:
                mazegame.sub_goal_reached = False
                # print(q_table.to_string())
                print(f"Episode {episode + 1} finished in {step} steps.")
                break

        epsilon = max(0.01, epsilon * 0.995)
        episodes += 1

    return q_table


# Discretize continuous state space into buckets (10x10 grid)
def state_to_bucket(state, env):
    grid_size = (10, 10)

    low_bounds = env.observation_space.low
    high_bounds = env.observation_space.high

    # Scale the state to fit within the grid
    state_scaled = (state - low_bounds) / (high_bounds - low_bounds)
    state_discrete = (state_scaled * np.array(grid_size)).astype(int)

    # Ensure the discrete state is within bounds
    state_discrete = np.clip(state_discrete, 0, np.array(grid_size) - 1)

    # Convert 2D grid coordinates into a single index for the Q-table
    state_index = state_discrete[0] * grid_size[1] + state_discrete[1]
    return state_index


if __name__ == "__main__":
    env = gym.make("MazeGame-v0", maze=maze, render_mode=None)

    max_episodes = 50
    alpha = 0.8
    gamma = 0.95
    epsilon = 0.7

    q_table = q_learning(env, max_episodes, alpha, gamma, epsilon)

    env.close()
    