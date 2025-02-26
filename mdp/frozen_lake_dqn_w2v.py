"""
ECEN 743: Reinforcement Learning
Deep Q-Learning
Code tested using
	1. gymnasium 0.27.1
	2. box2d-py  2.3.5
	3. pytorch   2.0.0
	4. Python    3.9.12
1 & 2 can be installed using pip install gymnasium[box2d]

General Instructions
1. This code consists of TODO blocks, read them carefully and complete each of the blocks
2. Type your code between the following lines
			###### TYPE YOUR CODE HERE ######
			#################################
3. The default hyperparameters should be able to solve LunarLander-v2
4. You do not need to modify the rest of the code for this assignment, feel free to do so if needed.

"""
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import deque, namedtuple
import logging
from matplotlib import animation # will be needed for rendering
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm

class ExperienceReplay:
	"""
	Based on the Replay Buffer implementation of TD3
	Reference: https://github.com/sfujim/TD3/blob/master/utils.py
	"""
	def __init__(self, state_dim, action_dim,max_size,batch_size,gpu_index=0):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))
		self.batch_size = batch_size
		self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')


	def add(self, state, action,reward,next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self):
		ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).long().to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)



class QNetwork(nn.Module):
  """
  Q Network: designed to take state as input and give out Q values of actions as output
  """

  def __init__(self, state_dim, action_dim):
    """
      state_dim (int): state dimenssion
      action_dim (int): action dimenssion
    """
    super(QNetwork, self).__init__()
    self.l1 = nn.Linear(state_dim, 64)
    self.l2 = nn.Linear(64, 64)
    self.l3 = nn.Linear(64, action_dim)

  def forward(self, state):
    q = F.relu(self.l1(state))
    q = F.relu(self.l2(q))
    return self.l3(q)



class DQNAgent():

  def __init__(self,
   state_dim,
   action_dim,
   discount=0.99,
   tau=1e-3,
   lr=5e-4,
   update_freq=4,
   max_size=int(1e5),
   batch_size=64,
   gpu_index=0
   ):
    """
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      discount (float): discount factor
      tau (float): used to update q-target
      lr (float): learning rate
      update_freq (int): update frequency of target network
      max_size (int): experience replay buffer size
      batch_size (int): training batch size
      gpu_index (int): GPU used for training
    """
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.discount = discount
    self.tau = tau
    self.lr = lr
    self.update_freq = update_freq
    self.batch_size = batch_size
    self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')


    # Setting up the NNs
    self.Q = QNetwork(state_dim, action_dim).to(self.device)
    self.Q_target = QNetwork(state_dim, action_dim).to(self.device)
    self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

    # Experience Replay Buffer
    self.memory = ExperienceReplay(state_dim,1,max_size,self.batch_size,gpu_index)

    self.t_train = 0

  def step(self, state, action, reward, next_state, done):
    """
    1. Adds (s,a,r,s') to the experience replay buffer, and updates the networks
    2. Learns when the experience replay buffer has enough samples
    3. Updates target netowork
    """
    self.memory.add(state, action, reward, next_state, done)
    self.t_train += 1

    if self.memory.size > self.batch_size:
      experiences = self.memory.sample()
      self.learn(experiences, self.discount) #To be implemented

    if (self.t_train % self.update_freq) == 0:
      self.target_update(self.Q, self.Q_target, self.tau) #To be implemented

  def select_action(self, state, epsilon):
    """
    TODO: Complete this block to select action using epsilon greedy exploration
    strategy
    Input: state, epsilon
    Return: Action
    Return Type: int
    """
    ###### TYPE YOUR CODE HERE ######
    # We generate a random number between 0 and 1
    rand_num = np.random.random()
    state = torch.from_numpy(state).to(self.device)
    a_opt = np.argmax(self.Q(state).cpu().detach().numpy())
    if rand_num<epsilon:
      a_list = [y for y in range(self.action_dim)]
      # print('a_list = ', a_list)
      a_list.remove(a_opt)
      # print('a_list = ', a_list)
      at = np.random.choice(np.array(a_list))
      return (at)
    else:
      return(a_opt)
    #################################

  def learn(self, experiences, discount):
    """
    TODO: Complete this block to update the Q-Network using the target network
    1. Compute target using  self.Q_target ( target = r + discount * max_b [Q_target(s,b)] )
    2. Compute Q(s,a) using self.Q
    3. Compute MSE loss between step 1 and step 2
    4. Update your network
    Input: experiences consisting of states,actions,rewards,next_states and discount factor
    Return: None
    """
    states, actions, rewards, next_states, dones = experiences
    ###### TYPE YOUR CODE HERE ######
    # Step 1:
    target = rewards + discount * torch.max(self.Q_target(next_states), axis = 1, keepdim = True).values * (1 - dones)# change Q_target to Q
    # Step 2:
    Q_sa = torch.take_along_dim(self.Q(states), actions, dim = 1)
    # Step 3:
    loss = nn.MSELoss()
    mse_loss = loss(target, Q_sa)
    # Step 4:
    self.optimizer.zero_grad()
    mse_loss.backward()
    self.optimizer.step()
    #################################

  def target_update(self, Q, Q_target, tau):
    """
    TODO: Update the target network parameters (param_target) using current Q parameters (param_Q)
    Perform the update using tau, this ensures that we do not change the target network drastically
    1. param_target = tau * param_Q + (1 - tau) * param_target
    Input: Q,Q_target,tau
    Return: None
    """
    ###### TYPE YOUR CODE HERE ######
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    #################################

  def get_optimal_policy(self, states):
    """Extracts the optimal policy after training."""
    policy = []
    for state in states:
      state = torch.from_numpy(state).to(self.device)
      policy.append(np.argmax(self.Q(state).cpu().detach().numpy()))
    return policy

  # def save_frames_as_gif(frames, path='./', filename='dqn_q1_animation.gif'): # for creating the frame
  #   plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

  #   patch = plt.imshow(frames[0])
  #   plt.axis('off')

  #   def animate(i):
  #       patch.set_data(frames[i])

  #   anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
  #   anim.save(path + filename, writer='imagemagick', fps=60)

# ============================================================================================
# Function to generate custom map
# ============================================================================================
def make_env(env_name, env_dim = 4, seed = 42, stochastic = False):
    env = gym.make(env_name, desc=generate_random_map(size=env_dim, seed=seed), 
                   is_slippery = stochastic, render_mode = 'rgb_array')
    return env


if __name__ == "__main__":
  print('Here!')
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", default="FrozenLake-v1")          # Gymnasium environment name # Default = MountainCar-v0
  parser.add_argument("--seed", default=42, type=int)              # sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--n-episodes", default=2500, type=int)     # maximum number of training episodes
  parser.add_argument("--batch-size", default=64, type=int)       # training batch size
  parser.add_argument("--discount", default=0.99)                 # discount factor
  parser.add_argument("--lr", default=5e-4)                       # learning rate
  parser.add_argument("--tau", default=0.001)                     # soft update of target network
  parser.add_argument("--max-size", default=int(1e5),type=int)    # experience replay buffer length
  parser.add_argument("--update-freq", default=4, type=int)       # update frequency of target network
  parser.add_argument("--gpu-index", default=0,type=int)		      # GPU index
  parser.add_argument("--max-esp-len", default=100, type=int)    # maximum time of an episode
  #exploration strategy
  parser.add_argument("--epsilon-start", default=1)               # start value of epsilon
  parser.add_argument("--epsilon-end", default=0.01)              # end value of epsilon
  parser.add_argument("--epsilon-decay", default=0.995)           # decay value of epsilon
  parser.add_argument("--save-filename", default = "dqn_w2v_FrozenLake-v1" )
  args, unknown = parser.parse_known_args()

  # making the environment
  env_dim = 4
  stochastic = False
  check_env_details = True

  # # Making the environment	
  env = make_env(env_name=args.env, env_dim=env_dim, seed = args.seed, stochastic=stochastic)

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
  # assert False

  #setting seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  print("State space: ", env.observation_space)
  print("Action space: ", env.action_space)

  # state_dim = env.observation_space.shape[0]
  state_dim = env.observation_space.n # here we will use 1-hot encoding
  action_dim = env.action_space.n


  #################################################################################################
  #################################### w2v related details ########################################
  embed_dim = 12
  word_embeddings = np.load(f"mdp/w2v_embed_dim_{embed_dim}_{args.env}_map_size_{env_dim}_stochastic_{stochastic}_seed_{args.seed}.npy", allow_pickle=True).item()

  kwargs = {
    "state_dim":embed_dim,
    "action_dim":action_dim,
    "discount":args.discount,
    "tau":args.tau,
    "lr":args.lr,
    "update_freq":args.update_freq,
    "max_size":args.max_size,
    "batch_size":args.batch_size,
    "gpu_index":args.gpu_index
  }
  learner = DQNAgent(**kwargs) #Creating the DQN learning agent

#   print(np.eye(state_dim).dtype)
#   assert False

  state_embeddings = []
  states = []
  words = list(word_embeddings.keys())
#   print(word_embeddings['s_0'])
  for i in range(state_dim):
    #  states.append(i)
    if 's_'+str(i) in words:
        state_embeddings.append(word_embeddings['s_'+str(i)])
        # print(f"s_{i}: {word_embeddings['s_'+str(i)]}")
    else:
        temp = np.random.randn(embed_dim).astype(np.float32)
        temp /= np.abs(np.max(temp))
        state_embeddings.append(temp)
        print(f"s_{i} not present in the w2v model, hence random vector initialized")
        # print(f"s_{i}: {temp}")
    
  # end for
  print("state embeddings dtype: ", state_embeddings[0].dtype)
#   assert False
  #################################################################################################
  
  one_hot_state = state_embeddings
  print("Embeddings are loaded. Now we train dqn")
  # print(one_hot_state)

  # f = open('dqn_mountaincar.txt', 'w') # file to store the training log
  temp_file = args.env + "_dqn_og.txt"
  f = open(temp_file, 'w') # file to store the training log
  reward_curve = [] # this will store the moving avg of rewards
  moving_window = deque(maxlen=100)
  epsilon = args.epsilon_start
  count = 0
  for e in tqdm(range(args.n_episodes)):
    state, _ = env.reset(seed=args.seed)
    curr_reward = 0
    for t in range(args.max_esp_len):
      action = learner.select_action(one_hot_state[state],epsilon) #To be implemented
      n_state,reward,terminated,truncated,_ = env.step(action)
      done = terminated or truncated
      learner.step(one_hot_state[state],action,reward,one_hot_state[n_state],done) #To be implemented
      # print("\n\n")
      # print("Learner.memory? Yes!", learner.memory)
      # print("\n\n")
      state = n_state
      curr_reward += reward
      if done:
        break
    moving_window.append(curr_reward)
    reward_curve.append(np.mean(moving_window))

    """"
    TODO: Write code for decaying the exploration rate using args.epsilon_decay
    and args.epsilon_end. Note that epsilon has been initialized to args.epsilon_start
    1. You are encouraged to try new methods
    """
    ###### TYPE YOUR CODE HERE ######
    epsilon *= args.epsilon_decay
    epsilon = max(epsilon, args.epsilon_end)
    # print('current epsilon = ', epsilon)
    #################################

    if e % 100 == 0:
      print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))
      # print('length of training rewards: ', len(reward_curve))

    """"
    TODO: Write code for
    1. Logging and plotting
    2. Rendering the trained agent
    """
    ###### TYPE YOUR CODE HERE ######
    # logging.basicConfig(filename='dqn_q1.txt', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level = logging.INFO, encoding = 'utf-8')
    # logging.getLogger()
    # logging.info('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))

    # We open a file using python commands and store the above episode runs in that
    f.write('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f} \n'.format(e, np.mean(moving_window)))

    # if e % 500 == 0:
    #   torch.save(learner.Q.state_dict(), 'dqn_og_seed_6_cartpole_v1_%i.pt'%(count))
    #   count+=1
    #################################

  f.close() # to close the file

  # Now we save the trained model
  # torch.save(learner.Q.state_dict(), 'dqn_og_seed_6_cartpole_v1_%i.pt'%(count))
  torch.save(learner.Q.state_dict(), f"mdp/{args.save_filename}_seed_{args.seed}_mapsize_{env_dim}.pt")
  # print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))
  final_policy = learner.get_optimal_policy(one_hot_state)
  print("Final policy: ", final_policy)
  state = 0
  # Define action map
  action_map = {
  0: "Left",
  1: "Down",
  2: "Right",
  3: "Up"
  }
  print("State: Type -    action taken")
  lake_grid = env.unwrapped.desc  # Gets the grid representation
  for row in lake_grid:
      for cell in row:
          print(f"     {state}:   {cell.decode('utf-8')} - {final_policy[state]}-->{action_map[final_policy[state]]}")  # Convert byte to string
          state += 1

  print('It was successful!')

# end of code