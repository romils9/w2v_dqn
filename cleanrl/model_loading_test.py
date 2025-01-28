import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from dataclasses import dataclass

import os
import random
import time

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

## Modifications to use cleanrl_utils: Made a copy of cleanrl_utils inside cleanrl/cleanrl folder
@dataclass
class Args:
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
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    # env_id: str = "PongNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
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

###########################################################################
# # Define the QNetwork class (already defined in your code)
# class cust_QNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3136, 512),
#             nn.ReLU(),
#             nn.Linear(512, 4),  # Adjust to match the number of actions
#         )

#     def forward(self, x):
#         return self.network(x / 255.0)

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
        return self.network(x / 255.0)


class cust_CNN_Model(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Define only the CNN layer
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Freeze the CNN backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Define the Fully Connected Layers
        self.fcc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n)
        )

    def forward(self, x):
        x = self.cnn(x / 255.0)
        x *= 255.0
        x = self.fcc_layers(x / 255.0)
        return x

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    print("experiment name: ", args.exp_name)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    # Function to Load model weights
    def load_model_weights(model, weight_path):
        model.load_state_dict(torch.load(weight_path))
        return model
    

    # Load weights
    run_name = f"BreakoutNoFrameskip-v4__dqn_atari__1__1736379420"
    model_path = f"runs/{run_name}/dqn_atari.cleanrl_model"

    # Initialize all models
    test_model = cust_CNN_Model(envs).to(device) # model using new structure
    cust_model = cust_CNN_Model(envs).to(device) # model using new structure for weight check
    cust_model.load_state_dict(test_model.state_dict()) # this ensures same initialization as test_model

    q_network = QNetwork(envs).to(device) # model based on the OG QNetwork
    # epsilon=0.05
    # eval_episodes=3
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    # print("QNetwork model:")
    # for key, value in q_network.state_dict().items():
    #     print(f"{key}: {value.shape}")

    # # print("\ncust_model prior to loading: ")
    # # for key, value in cust_model.state_dict().items():
    # #     print(f"{key}: {value.shape}")

    # print("\ntest_model prior to loading: ")
    # for key, value in test_model.state_dict().items():
    #     print(f"{key}: {value.shape}")
    
    # print("\n")
   
    '''  ### Incorrect ways to load the model:
    # Load the new model with the weights saved in q_network
    # cnn_state_dict = {k: v for k, v in q_network.state_dict().items() if "network.0" in k or "network.2" in k or "network.4" in k}
    # fcc_state_dict = {k: v for k, v in q_network.state_dict().items() if "network.7" in k or "network.9" in k}
   
    # Now we load the model with weights and perform the checks
    # test_model.load_state_dict(q_network.state_dict(), strict = False)
    # test_model.cnn.load_state_dict(cnn_state_dict, strict=False)
    # test_model.fcc_layers.load_state_dict(fcc_state_dict, strict=False) 
    '''

    test_state_dict = test_model.state_dict()
    q_state_dict = q_network.state_dict()
    keys_test = list(test_state_dict.keys())
    keys_q = list(q_state_dict.keys())
    # print("Printing the key names:")
    # print("test_model: ", keys_test)
    # print("q_network: ", keys_q)

    for idx in range(len(keys_q)):
        if test_state_dict[keys_test[idx]].shape == q_state_dict[keys_q[idx]].shape:
            test_state_dict[keys_test[idx]] = q_state_dict[keys_q[idx]]
        else:
            print("Shape mismatch has occured!")
        # end if-else
    # end for

    test_model.load_state_dict(test_state_dict, strict=True)
    # assert False, "Checking if keys manually loading"
    test_state_dict_2 = test_model.state_dict()
    keys_test = list(test_state_dict_2.keys())

    all_equal = True
    if len(keys_test)==len(keys_q):
        for idx in range(len(keys_test)):
            equal = torch.equal(test_state_dict_2[keys_test[idx]], q_state_dict[keys_q[idx]])
            print(f"Layer {idx+1}: {'Equal' if equal else 'Not Equal'}")

            if not equal:
                all_equal = False
            # end equal if
        # end for

        if all_equal:
            print("All layers in both models have equal weights!")
        else:
            print("Some layers have different weights.")
    else:
        print("Architecture mismatch!")
    

    #### Now we perform 2nd check by giving the same input and checking whether output is the same
    dummy_input = torch.randn(1, 4, 84, 84)  # Replace dimensions as needed
    dummy_input = dummy_input.to(device)
    test_op = test_model(dummy_input)
    q_op = q_network(dummy_input)
    cust_op = cust_model(dummy_input)
    print("q_network output:", q_op)
    print("test_model output:", test_op)
    print("cust_model op: ", cust_op)

