import numpy as np
import pandas as pd
import gym
from gym.envs.registration import register
import mazegame

# 1 represents walls, 'S' is the start, 'G' is a sub-goal, and 'E' is the end goal
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'S', 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 'G', 0, 0, 0, 1, 0, 1],  
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 'E', 0, 1],  
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Registering the custom maze environment with OpenAI Gym
register(
    id='MazeGame-v0',
    entry_point='mazegame:MazeGameEnv',
    kwargs={'maze': None}, 
)

# Initializing the Q-table with zeros
def init_q_table():
    actions = np.array(['up', 'down', 'left', 'right'])  # Define actions
    q_table = pd.DataFrame(
        np.zeros((100, len(actions))), columns=actions  # 100 states by number of actions
    )
    return q_table

# Choose an action based on the epsilon-greedy policy
def act_choose(state, q_table, epsilon):
    actions = np.array(['up', 'down', 'left', 'right'])
    state_act = q_table.iloc[state, :]  # Get Q-values for current state

    # Choose a random action with probability epsilon, otherwise take the best action
    if np.random.uniform() > epsilon or state_act.all() == 0:
        action = np.random.choice(actions)
    else:
        action = state_act.idxmax()
    return action

# Mapping the action names to integers
def map_action(action):
    action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    return action_map[action]

# Updating the Q-table based on the agent’s experience
def update_q_table(q_table, state, action, next_state, terminal, gamma, alpha, reward):
    x = int(state)
    next_x = next_state
    q_original = q_table.loc[x, action]

    # Calculating the new Q-value using the Bellman equation
    if not terminal:
        q_predict = reward + gamma * q_table.iloc[next_x].max()
    else:
        q_predict = reward

    # Updating the Q-value in the Q-table
    q_table.loc[x, action] = (1 - alpha) * q_original + alpha * q_predict
    
    return q_table

# Main Q-learning loop
def q_learning(env, max_episodes, alpha, gamma, epsilon):
    q_table = init_q_table()   
    episodes = 0

    for episode in range(max_episodes):
        state = env.reset()     # Reseting the environment and get initial state
        state = state_to_bucket(state, env)     # Discretizing state for Q-table indexing

        done = False
        step = 0

        # Continue until the episode ends
        while not done:
            env.render() 

            action = act_choose(state, q_table, epsilon)
            mapped_action = map_action(action)
            
            # Take the action in the environment and observe the outcome
            next_observation, reward, done, _ = env.step(mapped_action)

            # Converting the next observation to the Q-table index
            next_state = state_to_bucket(next_observation, env) 

            # Updating Q-table with the observed reward and transition
            q_table = update_q_table(q_table, state, action, next_state, done, gamma, alpha, reward)

            state = next_state
            step += 1

            # If the episode is complete, reset sub-goal status and print results
            if done:
                mazegame.sub_goal_reached = False
                print(f"Episode {episode + 1} finished in {step} steps.")
                break

        # epsilon = max(0.001, epsilon * 0.99)

        episodes += 1

    return q_table

# Converting a continuous state into a discrete Q-table index
def state_to_bucket(state, env):
    grid_size = (10, 10) 

    # Scaling state to fit within grid bounds
    low_bounds = env.observation_space.low
    high_bounds = env.observation_space.high
    state_scaled = (state - low_bounds) / (high_bounds - low_bounds)
    state_discrete = (state_scaled * np.array(grid_size)).astype(int)

    # Ensuring the state is within bounds
    state_discrete = np.clip(state_discrete, 0, np.array(grid_size) - 1)

    # Calculating single index for state
    state_index = state_discrete[0] * grid_size[1] + state_discrete[1]
    return state_index

if __name__ == "__main__":
    # Creating the custom maze environment, use render_mode='human' for visualization
    env = gym.make("MazeGame-v0", maze=maze, render_mode=None)  

    # Q-learning parameters
    max_episodes = 1000 
    alpha = 0.5
    gamma = 0.95
    epsilon = 0.7

    # Run Q-learning and output the trained Q-table
    q_table = q_learning(env, max_episodes, alpha, gamma, epsilon)

    env.close()
