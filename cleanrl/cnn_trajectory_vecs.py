'''
Here we will collect the CNN output vectors for the trajectories 
that were collected using the **collect_trajectories.py** script.
First we need: 
1. To set up the CNN architecture class
2. Call the model that we want to load
3. Then choose only the CNN layers
4. Load these chosen CNN layers with **strict = True** into the CNN
5. Pass the trajectories into the CNN and collect the output vectors.
6. Store the output vectors as a tuple of (s_{t-1}, s_t, s_{t+1}, episode #)
7. Note, make sure to use the initial image and the image with terminations
to terminate the tuples.
8. Additionally, we need to ensure that episodes don't clash with each other
since if we choose samples from across episodes then the negative samples
can be incorrectly chosen. Hence, each tuple should also include an episode
number.
'''

# print("This file is executable and successfully getting executed.")

################################################################################
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# assert False, "To end"

@dataclass
class Args:
    # exp_name: str = os.path.basename(__file__)[: -len(".py")]
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    # env_id: str = "PongNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        x = self.network(x / 255.0)
        x = self.fc_layers(x / 255.0)
        return x

class NewCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.network(x/ 255.0)

# def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
#     slope = (end_e - start_e) / duration
#     return max(slope * t + start_e, end_e)


if __name__ == "__main__":
#     import stable_baselines3 as sb3

#     if sb3.__version__ < "2.0":
#         raise ValueError(
#             """Ongoing migration: run the following command to install the new dependencies:

# poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
# """
#         )
    args = tyro.cli(Args)
    print("experiment name: ", args.exp_name)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # q_network = QNetwork(envs).to(device)
    # optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # target_network = QNetwork(envs).to(device)
    # target_network.load_state_dict(q_network.state_dict())

    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     optimize_memory_usage=True,
    #     handle_timeout_termination=False,
    # )
    start_time = time.time()

    print("We reached the evaluation part of the code!")
    run_name = f"BreakoutNoFrameskip-v4__dqn_atari__1__1736379420"
    model_path = f"runs/{run_name}/dqn_atari.cleanrl_model"
    # run_name = f"{run_name}-eval"
    q_network = QNetwork(envs).to(device)
    # Model = QNetwork
    epsilon=0.05
    eval_episodes=3
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    # Extract parameters for convolutional layers (before Flatten)
    cnn_state_dict = {k: v for k, v in q_network.state_dict().items() if "network.0" in k or "network.2" in k or "network.4" in k}

    # print("cnn_state_dict: ", cnn_state_dict)

    model = NewCNN().to(device)
    model.load_state_dict(cnn_state_dict, strict = True)
    print("CNN model loaded successfully! ")

    # Now we load the trajectories into <traj>
    traj = np.load("trajectories.npy", allow_pickle= True)
    # print("The shape of the traj file: ", traj.shape)
    # print("The number of contents i.e. size: ", traj.size)
    # print("dtype: ", traj.dtype)

    '''
    Information about traj:
    1. traj is an array of len = 3. There are 3 elements inside traj.
    2. Each element inside traj is a list which contains an episode.
    3. Each episode looks like {(s_t, a_t, r_t, s_{t+1}, done)}_{t=1}^{T} where T represents the length of that episode.
    4. Thus, traj[0] := episode 1
    5. traj[0][0] := (s_0, a_0, r_0, s_1, done = False)
    6. traj[0][0][0] := s_0
    7. traj[0][0][0].shape = (1, 4, 84, 84) which represents the image associated with s_0
    8. Thus to iterate over the states in a single episode, ** traj[0][i][0] iterate over i **
    '''

    # Now we iterate over each state in the 1st episode and collect the vectors from it
    '''
    We will store the vectors in the same style as the image trajectories are stored so that if we want to include 
    the rewards information, we can do that. 
    Another way is to make vector tuples from each episode and store them in a list. That way, we only need to access 
    those tuples and perform the training. However, this won't include any information about current or future rewards 
    starting from this state.
    '''

    # We use "idx" as the variable to index over each image. 
    # Note we need to check for terminations variable to ensure we stop aptly
    vec_rep = []
    for i in range(len(traj)):
        temp = []
        print("Current episode = ", i+1)
        for idx in range(len(traj[i]))[1:]:
            temp.append(traj[i][idx][4])
        # end for
        dones = np.where(np.array(temp)==True)[0]
        print("Number of Trues = ", len(dones))
        print("Indexes where episodes were truncated = ", dones)

        vec1 = []
        # For the 1st end point in each episode:
        for idx in range(dones[0])[1:]:
            obs = traj[i][idx][0]
            vec0 = model(torch.Tensor(obs).to(device))
            vec0 = vec0.cpu().detach().numpy()
            # print("vec0 type: ", vec0.dtype)
            vec1.append(vec0)
        # end for
        vec_rep.append(vec1)

        for done_index in range(len(dones))[0:-1]:
            vec1 = []
            for idx in range(dones[done_index+1])[dones[done_index]+1:]:
                obs = traj[i][idx][0]
                vec0 = model(torch.Tensor(obs).to(device))
                vec0 = vec0.cpu().detach().numpy()
                vec1.append(vec0)
            # end for
            vec_rep.append(vec1)
        print("\n")
    # end for

    # Now we store the vector representations in another file
    np.save("vector_trajectories.npy", np.array(vec_rep, dtype=object))