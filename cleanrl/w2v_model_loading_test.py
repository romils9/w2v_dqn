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

# # Defining the customized Q-Network combined with w2v
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

#####################################################################################33
# Start of the main function:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    

    ############################################################################################# w2v model related classes ###########################
    class cust_ReducedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential()  # Initialize an empty Sequential model

        def forward(self, x):
            return self.network(x / 255.0)

    # Initialize reduced_model
    cust_reduced_model = cust_ReducedCNN()
    cust_larger_model = QNetwork(envs)
    # Ensure models are on the same device
    cust_larger_model.to(device)
    cust_reduced_model.to(device)

    # Copy the first 7 layers from larger_model into reduced_model
    cust_reduced_model.network = nn.Sequential(*list(cust_larger_model.network.children())[:7])

    # Define the extended model
    class cust_ExtendedModel(nn.Module):
        def __init__(self, reduced_cnn):
            super().__init__()
            # Use the ReducedCNN backbone
            self.cnn_backbone = reduced_cnn
            # Freeze the CNN backbone
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False

            # Add a new trainable linear layer
            self.fc = nn.Sequential(
                nn.Linear(3136, 128),  # Input size matches the Flatten output of the CNN
                nn.ReLU()
            )
            for param in self.fc.parameters():
                param.requires_grad = False

        def forward(self, x):
            x = self.cnn_backbone(x / 255.0)
            x *= 255.0
            x = self.fc(x / 255.0)
            return x

    test_model = cust_ExtendedModel(cust_reduced_model).to(device)

    # Load model weights
    def load_model_weights(model, weight_path):
        model.load_state_dict(torch.load(weight_path))
        return model

    # Load weights
    w2v_test_model = load_model_weights(test_model, 'extended_model.pth')
    # print(w2v_test_model)

    # print("state dict keys of w2v_test_model: ", list(w2v_test_model.state_dict().keys()))

    # assert False, "breaking to check state dict of cust_Extended model"

    # Final model for including w2v block
    class cust_w2v_model(nn.Module):
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

            # Add a new trainable linear layer
            self.w2v = nn.Sequential(
                nn.Linear(3136, 128),  # Input size matches the Flatten output of the CNN
                nn.ReLU()
            )
            for param in self.w2v.parameters():
                param.requires_grad = False

            # Define the Fully Connected Layers
            self.fcc_layers = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, env.single_action_space.n)
            )

        def forward(self, x):
            x = self.cnn(x / 255.0)
            x *= 255.0
            x = self.w2v(x / 255.0)
            x *= 255.0
            x = self.fcc_layers(x / 255.0)
            return x

    #########################################################################################################################################################

    # Load weights
    run_name = f"BreakoutNoFrameskip-v4__dqn_atari__1__1736379420"
    model_path = f"runs/{run_name}/dqn_atari.cleanrl_model"

    # Initialize all models
    test_model = cust_CNN_Model(envs).to(device) # model using new structure

    q_network = QNetwork(envs).to(device) # model based on the OG QNetwork
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    final_w2v = cust_w2v_model(envs).to(device)

    # print("QNetwork model:")
    # for key, value in q_network.state_dict().items():
    #     print(f"{key}: {value.shape}")

    # # print("\ncust_model prior to loading: ")
    # # for key, value in cust_model.state_dict().items():
    # #     print(f"{key}: {value.shape}")

    # print("\ntest_model prior to loading: ")
    # for key, value in test_model.state_dict().items():
    #     print(f"{key}: {value.shape}")
    
    # print(test_model)
    # print(q_network)
    # print("\n")
    # assert False, "breaking to check state dict of cust_Extended model"

    run_name_2 = f"BreakoutNoFrameskip-v4__dqn_atari__10M_cnn_fcc_split"
    model_path_2 = f"runs/{run_name_2}/dqn_atari.cleanrl_model"
    test_model.load_state_dict(torch.load(model_path_2, map_location=device))

    #### Now we compare the CNN parts of test_model and w2v_test_model and also with q_network
    test_state_dict = test_model.state_dict()
    q_state_dict = q_network.state_dict()
    w2v_state_dict = w2v_test_model.state_dict()
    keys_test = list(test_state_dict.keys())
    keys_q = list(q_state_dict.keys())
    keys_w2v = list(w2v_state_dict.keys())

    # for idx in range(8): # Here since we know the total number of layers, hence this is possible. Other method is to compare the keys list and use the list with the smallest len
    #     ### Comparing w2v saved model with q_nw saved model:
    #     # equal = torch.equal(w2v_state_dict[keys_w2v[idx]], q_state_dict[keys_q[idx]])
    #     # print(f"Layer {idx+1}, key_w2v {keys_w2v[idx]} and key_q {keys_q[idx]}: {'Equal' if equal else 'Not Equal'}")

    #     ### Comparing w2v saved model with new structure of q_nw saved model:
    #     equal = torch.equal(w2v_state_dict[keys_w2v[idx]], test_state_dict[keys_test[idx]])
    #     print(f"Layer {idx+1}, key_w2v {keys_w2v[idx]} and key_q {keys_test[idx]}: {'Equal' if equal else 'Not Equal'}")

    #     # end equal if
    # # end for

    final_w2v_state_dict = final_w2v.state_dict()
    keys_final = list(final_w2v_state_dict.keys())
    # print("keys of final model: ", keys_final)

    for idx in range(len(keys_w2v)):
        # Now we copy the keys of the saved w2v model into a new model
        if final_w2v_state_dict[keys_final[idx]].shape == w2v_state_dict[keys_w2v[idx]].shape:
            final_w2v_state_dict[keys_final[idx]] = w2v_state_dict[keys_w2v[idx]]
        else:
            print("Shape mismatch has occured!")
        # end if-else
    # end for

    final_w2v.load_state_dict(final_w2v_state_dict, strict=True)
    print("Final model loaded!")

    print("Testing that the model loaded is correct") 

    all_equal = True
    for idx in range(len(keys_w2v)):
        equal = torch.equal(final_w2v_state_dict[keys_final[idx]], w2v_state_dict[keys_w2v[idx]])
        print(f"Layer {idx+1}: {'Equal' if equal else 'Not Equal'}")

        if not equal:
            all_equal = False
        # end equal if
    # end for

    if all_equal:
        print("All layers in both models have equal weights!")
    else:
        print("Some layers have different weights.")
    # end if-else

    # Now we save the model
    run_name_w2v = f"BreakoutNoFrameskip-v4__dqn_w2v__10M_cnn_128_w2v_split"
    log_dir = f"runs/{run_name_w2v}"

    # Create the directory
    os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(log_dir):
        print(f"The directory {log_dir} exists.")
    else:
        print(f"Failed to create the directory {log_dir}.")

    model_path_w2v = f"runs/{run_name_w2v}/dqn_w2v.cleanrl_model"
    torch.save(final_w2v.state_dict(), model_path_w2v)
    print(f"model saved to {model_path_w2v}")

    #### Now we perform a check by giving the same input and checking whether output is the same
    cust_model = cust_w2v_model(envs).to(device)
    dummy_input = torch.randn(1, 4, 84, 84)  # Replace dimensions as needed
    dummy_input = dummy_input.to(device)
    old_op = final_w2v(dummy_input)
    cust_model.load_state_dict(torch.load(model_path_w2v, map_location=device))
    cust_op = cust_model(dummy_input)
    print("final_w2v output:", old_op)
    print("cust_model op: ", cust_op)


    ### Just loading the CNN+w2v part of the model
    temp_model = cust_w2v_model(envs).to(device)
    cnn_state_dict = {k: v for k, v in final_w2v.state_dict().items() if "cnn" in k}
    w2v_state_dict_2 = {k: v for k, v in final_w2v.state_dict().items() if "w2v" in k}
    fcc_state_dict = {k: v for k, v in temp_model.state_dict().items() if "fcc_layers" in k}

    keys_temp = list(temp_model.state_dict().keys())
    print("temp_model keys: ", keys_temp)
    # Merge cnn_state_dict and fcl_state_dict
    merged_state_dict = {**cnn_state_dict, **w2v_state_dict_2, **fcc_state_dict}

    ### Load into the target model
    temp_model.load_state_dict(merged_state_dict, strict=True)
    keys_cnn = list(cnn_state_dict.keys())+list(w2v_state_dict_2.keys())
    print("w2v extended model keys: ", keys_cnn)

    all_equal = True
    if len(keys_temp)==len(keys_final):
        for idx in range(len(keys_temp)):
            equal = torch.equal(temp_model.state_dict()[keys_temp[idx]], final_w2v.state_dict()[keys_final[idx]])
            print(f"Layer {idx+1} i.e. {keys_temp[idx]}: {'Equal' if equal else 'Not Equal'}")

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


    ################################################## The end #############################################################