# ============================================================================================
# Template
# ============================================================================================
import numpy as np
import random
from tqdm import tqdm
import gymnasium as gym
from collections import deque
# from collections import defaultdict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# ============================================================================================
# Defining the Environment - MDP class
# ============================================================================================

# class RandomMDP:
#     """Simulated environment for a randomly generated MDP."""
#     def __init__(self, S, A, P, r):
#         self.S = S
#         self.A = A
#         self.P = P  # Transition probabilities
#         self.r = r  # Reward function
#         self.state = np.random.randint(0, S)  # Initial state

#     def reset(self):
#         """Resets the environment and returns the initial state."""
#         self.state = np.random.randint(0, S)
#         return self.state

#     def step(self, action):
#         """Performs an action, returning (next_state, reward, done, info)."""
#         next_state = np.random.choice(self.S, p=self.P[self.state, action])
#         reward = self.r[self.state, action]
#         self.state = next_state
#         done = False  # Assume episode never ends (for simplicity)
#         return next_state, reward, done, {}


# ============================================================================================
# Q-Learning Agent class
# ============================================================================================
class QLearningAgent:
    """Q-learning agent."""
    def __init__(self, num_states, num_actions, gamma=0.99, epsilon=0.1, alpha=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((num_states, num_actions))  # Initialize Q-table

    def choose_action(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state, :])  # Greedy action for next state
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error  # Update Q-table

    def get_optimal_policy(self):
        """Extracts the optimal policy after training."""
        return np.argmax(self.q_table, axis=1)


# # ============================================================================================
# # Function to train the model using Q-learning for a RandomMDP environment
# # ============================================================================================
# def run_q_learning_mdp(env, agent, num_episodes=10, convergence_threshold=1e-4, stable_threshold=10):
#     """Train the agent and check for Q-value and policy convergence."""
#     prev_q_table = np.copy(agent.q_table)  # Store old Q-table
#     prev_policy = agent.get_optimal_policy()
#     stable_count = 0  # Number of episodes with stable policy
#     reward_history = []

#     for episode in tqdm(range(num_episodes)):
#     # for episode in range(num_episodes):
#         state = env.reset()
#         done = False
#         total_reward = 0
#         episode_count = 0
#         max_eps_len = 100

#         while not done:
#             action = agent.choose_action(state)
#             next_state, reward, done, _ = env.step(action)
#             agent.update_q_value(state, action, reward, next_state)
#             state = next_state
#             total_reward += reward

#             # Compute max Q-value change
#             q_change = np.max(np.abs(agent.q_table - prev_q_table))
#             prev_q_table = np.copy(agent.q_table)

#             # Check Q-value convergence
#             if q_change < convergence_threshold:
#                 print(f"Q-values converged at Episode {episode+1} with max Q-change: {q_change}")
#                 break

#             episode_count+=1
#             if episode_count>max_eps_len:
#                 break

#         # Check policy stability
#         current_policy = agent.get_optimal_policy()
#         print("Current optimal policy: ", current_policy)

#         # if np.array_equal(prev_policy, current_policy):
#         #     stable_count += 1
#         # else:
#         #     stable_count = 0  # Reset count if policy changes

#         # prev_policy = current_policy

#         # if stable_count >= stable_threshold:
#         #     print(f"Policy stabilized for {stable_threshold} episodes. Stopping training at Episode {episode+1}.")
#         #     break

#         # end while
#         reward_history.append(total_reward)

#     print("Training completed!")
#     return agent, reward_history

# ============================================================================================
# Function to train the model using Q-learning for Frozen Lake
# ============================================================================================
def run_tabular_q_frozen(env, agent, num_episodes=10, convergence_threshold=1e-4,
                         epsilon_start = 1, epsilon_decay = 0.995, epsilon_end = 0.01, seed=42):
    reward_curve = [] # this will store the moving avg of rewards
    moving_window = deque(maxlen=100)
    epsilon = epsilon_start
    prev_q_table = np.copy(agent.q_table)  # Store old Q-table

    for episode in tqdm(range(num_episodes)):
        state,_ = env.reset(seed=seed)
        # print(f"\nIn episode {episode}, After reset initial state = {state} and epsilon = {epsilon}")
        curr_reward = 0
        max_eps_len = 100
        flag = False

        for _ in range(max_eps_len):
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            curr_reward += reward

            # # Compute max Q-value change
            # q_change = np.max(np.abs(agent.q_table - prev_q_table))
            # prev_q_table = np.copy(agent.q_table)

            # # Check Q-value convergence
            # if q_change < convergence_threshold:
            #     print(f"Q-values converged at Episode {episode+1} with max Q-change: {q_change}")
            #     flag = True
            #     break

            if done:
                break
        # end while inside an episode
        
        # Epsilon decay performed at the end of each episode
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_end)

        # Appending the smoothened reward
        moving_window.append(curr_reward)
        reward_curve.append(np.mean(moving_window))

        # if episode % 1000 == 0:
        #     # print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(episode, np.mean(moving_window)))
        #     print(f"Episode {episode}: epsilon = {epsilon}, avg reward = {np.mean(moving_window)}")
        # # end if

        if flag:
            break
    # end for num_episode

    return agent.q_table, agent.get_optimal_policy(), reward_curve

# ============================================================================================
# Function to generate custom map
# ============================================================================================
def make_env(env_name, env_dim = 4, seed = 42, stochastic = False):
    env = gym.make(env_name, desc=generate_random_map(size=env_dim, seed=seed), 
                   is_slippery = stochastic, render_mode = 'rgb_array')
    return env


# ============================================================================================
# Main function: Train and show results
# ============================================================================================

if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    env_dim = 4
    stochastic = False
    seed = 42
    gamma = 0.99
    alpha = 0.1 # learning rate in the table
    num_episodes = 100_000
    convergence_threshold = 1e-4
    epsilon_start = 1
    epsilon_decay = 0.99995
    epsilon_end = 0.01
    check_env_details = True
    # # Making the environment	
    # env = gym.make(env_name, desc = None, map_name = map_n, is_slippery = stochastic, render_mode = 'rgb_array')

    # To generate a random map
    # env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4, seed=21))
    env = make_env(env_name=env_name, env_dim=env_dim, seed = seed, stochastic=stochastic)
    
    # Setting seeds
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if check_env_details:
        # Extract the environment description (grid layout)
        lake_grid = env.unwrapped.desc  # Gets the grid representation

        # Print state-to-symbol mapping
        print("Frozen Lake Grid Layout:")
        for row in lake_grid:
            print(" ".join(row.astype(str)))

        goal_state = None
        rows, cols = lake_grid.shape
        for i in range(rows):
            for j in range(cols):
                if lake_grid[i, j] == b'G':  # 'G' is stored as a byte-string
                    goal_state = i * cols + j  # Convert (row, col) to state number
                    break
            # end for j
        # end for i
        print(f"Goal State: {goal_state}")
    # end if check_env

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    print("State space: ", env.observation_space.n)
    print("Action space: ", env.action_space.n)
	
    learner = QLearningAgent(num_states=state_dim, num_actions=action_dim, gamma=gamma
                             , epsilon=epsilon_start, alpha=alpha) # Creating the learning Agent

    final_q_table, final_policy, reward_curve = run_tabular_q_frozen(
                    env, learner, num_episodes=num_episodes, convergence_threshold=convergence_threshold,
                    epsilon_start = epsilon_start, epsilon_decay = epsilon_decay, epsilon_end = epsilon_end, seed=seed)
    
    Val_f = np.max(final_q_table, axis=1)

    state = 0
    # Define action map
    action_map = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
    }
    print("State: Type -    V(s),    action taken")
    lake_grid = env.unwrapped.desc  # Gets the grid representation
    for row in lake_grid:
        for cell in row:
            print(f"     {state}:   {cell.decode('utf-8')} - {Val_f[state]:.2f}, {final_policy[state]}-->{action_map[final_policy[state]]}")  # Convert byte to string
            state += 1
    # assert False, "c1"

    # Print the final table and policy
    print("Final Q function: ", final_q_table)
    # print("Final Policy: ", final_policy)
    # print("Final Value function: ", Val_f)

    # # Plot heatmap of the Value function
    # plt.figure(figsize=(5,5))
    # plt.imshow(Val_f.reshape(4,4), cmap="coolwarm", interpolation="nearest")
    # for i in range(4):
    #     for j in range(4):
    #         plt.text(j, i, f"{Val_f[i*4+j]:.2f}", ha='center', va='center', color='black')


    # Plot the reward curve



    # Save the current Q-function
    file_name = f"Q_table_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy"
    np.save(file_name, final_q_table)



    ########################################################################
    # # Now we save the trained model
    # torch.save(learner.Q.state_dict(), args.save_filename)
    # print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))
    # print('It was successful!')
# end of code

	