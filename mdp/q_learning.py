# ============================================================================================
# Template
# ============================================================================================
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
# ============================================================================================
# Defining the Environment - MDP class
# ============================================================================================

class RandomMDP:
    """Simulated environment for a randomly generated MDP."""
    def __init__(self, S, A, P, r):
        self.S = S
        self.A = A
        self.P = P  # Transition probabilities
        self.r = r  # Reward function
        self.state = np.random.randint(0, S)  # Initial state

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.state = np.random.randint(0, S)
        return self.state

    def step(self, action):
        """Performs an action, returning (next_state, reward, done, info)."""
        next_state = np.random.choice(self.S, p=self.P[self.state, action])
        reward = self.r[self.state, action]
        self.state = next_state
        done = False  # Assume episode never ends (for simplicity)
        return next_state, reward, done, {}


# ============================================================================================
# Q-Learning Agent class
# ============================================================================================
class QLearningAgent:
    """Q-learning agent."""
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((num_states, num_actions))  # Initialize Q-table

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
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


# ============================================================================================
# Function to train the model using Q-learning
# ============================================================================================
def run_q_learning(env, agent, num_episodes=10, convergence_threshold=1e-4, stable_threshold=10):
    """Train the agent and check for Q-value and policy convergence."""
    prev_q_table = np.copy(agent.q_table)  # Store old Q-table
    prev_policy = agent.get_optimal_policy()
    stable_count = 0  # Number of episodes with stable policy
    reward_history = []

    for episode in tqdm(range(num_episodes)):
    # for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_count = 0
        max_eps_len = 100

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Compute max Q-value change
            q_change = np.max(np.abs(agent.q_table - prev_q_table))
            prev_q_table = np.copy(agent.q_table)

            # Check Q-value convergence
            if q_change < convergence_threshold:
                print(f"Q-values converged at Episode {episode+1} with max Q-change: {q_change}")
                break

            episode_count+=1
            if episode_count>max_eps_len:
                break

        # Check policy stability
        current_policy = agent.get_optimal_policy()
        print("Current optimal policy: ", current_policy)

        if np.array_equal(prev_policy, current_policy):
            stable_count += 1
        else:
            stable_count = 0  # Reset count if policy changes

        prev_policy = current_policy

        if stable_count >= stable_threshold:
            print(f"Policy stabilized for {stable_threshold} episodes. Stopping training at Episode {episode+1}.")
            break

        # end while
        reward_history.append(total_reward)

    print("Training completed!")
    return agent, reward_history

# ============================================================================================
# Main function: Train and show results
# ============================================================================================
if __name__ == "__main__":
    # Define the randomly generated MDP parameters
    S = 4  # Number of states
    A = 4  # Number of actions

    # Setting the seed fo reproducibility
    # seeds = np.random.randint(10000, size=10)
    seeds = [42]
    np.random.seed(seeds[0]) # current seed set to 42

    # Random transition probabilities (S x A x S)
    P = np.random.dirichlet(np.ones(S), size=(S, A))

    # Random rewards (S x A)
    r = np.random.uniform(-1, 1, (S, A)) # generates randomly sampled rewards matrix of size SxA from [-1, 1]
    print("Rewards matrix: ", r)

    # Create a random MDP environment
    env = RandomMDP(S, A, P, r)

    # Create and train the Q-learning agent
    agent = QLearningAgent(num_states=S, num_actions=A)
    trained_agent, reward_history = run_q_learning(env, agent)

    # q_table_df = pd.DataFrame(trained_agent.q_table, columns=[f'Action {a}' for a in range(A)])
    # q_table_df.index.name = 'State'

    # policy_df = pd.DataFrame({'Optimal Action': trained_agent.get_optimal_policy()})
    # policy_df.index.name = 'State'
