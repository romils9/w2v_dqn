'''
Here we will keep the following things fixed and work using those assumptions:
    1. We will assume that the image list doesn't need to be filtered for repeated images
    2. We will compute the context_indices for each idx and randomly sample 4 of those. 
    If <4, then we will add removed ones to make it 4
    3. We will sample 16 negative samples for each target image
    4. Currently we're using the 5M model obtained during 10M training for Q-values
    5. And we will use the 1M model for CNN weights in the w2v model

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import gc

# ===============================
# Dataset and Preprocessing
# ===============================
def safe_transform(img):
    """
    Converts a preprocessed Atari observation into a normalized torch.Tensor.
    Expects input either as a NumPy array or a torch.Tensor with shape:
      - (1, 4, 84, 84): an extra singleton dimension is present
      - or (4, 84, 84): already in the desired shape.
    Returns a tensor of shape (4, 84, 84) with pixel values scaled to [0, 1].
    """
    if isinstance(img, torch.Tensor):
        # If the tensor has an extra leading dimension, remove it.
        if img.dim() == 4 and img.size(0) == 1:
            img = img.squeeze(0)
        return img
    if isinstance(img, np.ndarray):
        # Remove extra singleton dimension if present.
        if img.ndim == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        # Convert to float tensor and scale pixel values (assumes uint8 in [0, 255])
        return torch.tensor(img, dtype=torch.float32)
    raise ValueError("Image must be either a numpy array or torch.Tensor")


def get_context(idx, val, num_images, window_size = 2):
    window_thresh = 10*window_size
    if val[idx][0]<=5:
        context_indices_1 = val_5
    elif val[idx][0]<=6:
        context_indices_1 = val_6
    elif val[idx][0]<=7:
        context_indices_1 = val_7
    elif val[idx][0]<=8:
        context_indices_1 = val_8
    elif val[idx][0]<=9:
        context_indices_1 = val_9
    else:
        context_indices_1 = val_high
    # end if for value func
    
    if val[idx][1]==0:
        context_indices_2 = a_0
    elif val[idx][1]==1:
        context_indices_2 = a_1
    elif val[idx][1]==2:
        context_indices_2 = a_2
    else:
        context_indices_2 = a_3
    # end if for actions

    # This checks for condition 2 of similarity heuristics: Proximity in the state-space
    context_indices_3 = list(range(max(0, idx - window_thresh), min(num_images, idx + window_thresh + 1)))
    
    # This combines all the Similarity Heuristics criteria together
    context_indices = list(set(context_indices_1) & set(context_indices_2) & set(context_indices_3))

    if idx in context_indices:
            context_indices.remove(idx) # Since context shouldn't contain the target image

    if len(context_indices)<4: # Since we want to use 4 context samples, hence we 
        num_samples = 4 - len(context_indices)
        temp_list = list(set(context_indices_1) & set(context_indices_2) - {idx})
        if len(temp_list)>=num_samples:
            new_context = list(set(temp_list) - set(context_indices))
        else:
            new_context = list(set(context_indices_3) - set(context_indices) - {idx})
        # end if for list selection
        temp = random.sample(new_context, num_samples)
        context_indices = list(set(temp) | set(context_indices))
    # end context_len if

    # Now we compute the neg_samples
    neg_indices = list(set(range(num_images)) - (set(context_indices_1).union(set(context_indices_3), set(context_indices_2))))
    if len(neg_indices)<16: # If we can't sample enough negatives, then we need to increase the # of negative indices
        new_temp = set(context_indices_1).union(set(context_indices_2, set(range(max(0, idx - window_size), min(num_images, idx + window_size + 1)))))
        neg_indices_2 = list(set(range(num_images)) - new_temp - {idx})
        if len(neg_indices_2)<16: # If still not enough indices, we adjust further
            neg_indices = list(set(range(num_images)) - set(context_indices) - {idx})
        else:
            neg_indices = neg_indices_2
    
    return context_indices, neg_indices


class ImageSkipGramDataset(Dataset):
    def __init__(self, images, window_size=10, num_contexts = 4, num_neg_samples=16):
        """
        images: List of preprocessed Atari observations.
                Each observation is expected to be a NumPy array or torch.Tensor with shape (1,4,84,84)
                or (4,84,84).
        window_size: Number of nearby images (in the deduplicated list) to use as context.
        num_neg_samples: Number of negative samples per target.
        """

        # Define images without shuffle
        self.image_list = images  # List of images in the given shape.
        self.val = val # stores the corresponding (V^, a_t) list

        # Remaining initializations
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.num_contexts = num_contexts
        self.num_images = len(images)
        self.transform = safe_transform
        self.get_context = get_context # send idx, val, self.num_images, self.window_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Apply the transformation to the target image.
        target = self.transform(self.image_list[idx])  # Expected shape: (4,84,84)

        # Obtaining context of the target image using a heuristic obtained from get_context()
        context_indices, neg_indices = self.get_context(idx = idx, val = self.val, 
                                                        num_images=self.num_images, window_size=self.window_size)
        context_idx = random.sample(context_indices, self.num_contexts)
        context = torch.stack([self.transform(self.image_list[i]) for i in context_idx])

        negative_samples = random.sample(neg_indices, self.num_neg_samples)
        negatives = torch.stack([self.transform(self.image_list[i]) for i in negative_samples])
        
        return target, context, negatives

# ===============================
# Pretrained CNN for Initial Embeddings
# ===============================
class cust_CNN_Model(nn.Module):
    def __init__(self, env=None):
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
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.cnn(x / 255.0)
        x *= 255.0
        x = self.fcc_layers(x / 255.0)
        return x

# ===============================
# Skip-Gram Model with Negative Sampling
# ===============================
class ImageSkipGramModel(nn.Module):
    def __init__(self, embed_size=512, freeze_pretrained=True):
        """
        The model uses a CNN-based encoder (with pretrained weights)
        to generate embeddings from images. An output linear layer is used
        to project embeddings for context prediction.
        
        freeze_pretrained: If True, the pretrained CNN weights are frozen.
        """
        super(ImageSkipGramModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Optionally freeze the pretrained CNN layers
        if freeze_pretrained:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
        # An output layer that further transforms embeddings (for context prediction)
        self.w2v = nn.Sequential(
            nn.Linear(3136, embed_dim),  # Input size matches the Flatten output of the CNN
            nn.ReLU()
        )
        with torch.no_grad(): # this step initializes the weights to identity matrix
            # This ensures initial output of w2v matches that of the encoder
            self.w2v[0].weight.copy_(torch.eye(embed_size))  # Access the first layer's weight
            self.w2v[0].bias.fill_(0)  # Zero bias
    
    def forward(self, target_img):
        initial_embedding = self.cnn(target_img / 255.0)
        target_embedding = self.w2v(initial_embedding) # this creates a copy of the target_embeddings
        return target_embedding
    
# ===============================
# Loading the weights into the model
# ===============================
def load_model(model, cnn_state_dict):
    model_state_dict = model.state_dict()
    model_keys = list(model.state_dict().keys())
    # print("Keys in the model we're using: ", model_keys)
    cnn_keys = list(cnn_state_dict.keys())
    for idx in range(len(cnn_keys)):
            if cnn_state_dict[cnn_keys[idx]].shape == model_state_dict[model_keys[idx]].shape:
                model_state_dict[model_keys[idx]] = cnn_state_dict[cnn_keys[idx]]
            else:
                print("Shape mismatch has occured!")
    model.load_state_dict(model_state_dict, strict=True)
    return model
# ===============================
# Training Loop
# ===============================
def train_model(images, cnn_state_dict, embed_size=512, window_size=2, epochs=10, batch_size=16, lr=0.001, num_contexts = 4, num_neg_samples=5, fine_tune=False):
    """
    images: List of PIL Image objects (84x84)
    fine_tune: If True, allows the pretrained CNN to be updated during training.
    """
    dataset = ImageSkipGramDataset(images, window_size, num_contexts, num_neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    
    model = ImageSkipGramModel(embed_size, freeze_pretrained=not fine_tune)
    model = load_model(model = model, cnn_state_dict=cnn_state_dict) # Loads the CNN weights but not the w2v layer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    total_training_start = time.time()  # Start timing overall training
    for epoch in range(epochs):
        epoch_start = time.time()  # Start timing for this epoch
        total_loss = 0.0
        for target, context, negatives in dataloader:
            optimizer.zero_grad()
            
            # Forward pass: get embeddings for target, context, and negative samples
            target_embedding = model(target)
            B, num_c, C, H, W = context.shape
            context_flat = context.view(B * num_c, C, H, W)  # Flatten to shape [B*N, C, H, W]
            context_embeddings_flat = model(context_flat)
            context_embeddings = context_embeddings_flat.view(B, num_c, -1) # Reshape to [B, num_contexts, embed_size]
            # Now we process the positives to ensure several positives are included in the loss calculation
            # Compute positive scores
            pos_scores = (target_embedding.unsqueeze(1) * context_embeddings).sum(dim=2)  # Shape: [B, num_context]
            # Average positive scores **per target** to prevent inter-target mixing
            pos_loss = F.softplus(-pos_scores).mean(dim=1)
            
            # negatives shape: [B, num_neg_samples, C, H, W]
            B, N, C, H, W = negatives.shape
            negatives_flat = negatives.view(B * N, C, H, W)  # Flatten to shape [B*N, C, H, W]
            neg_embeddings_flat = model(negatives_flat)  # negatives: (B, num_neg_samples, C)
            # Reshape back to [B, num_neg_samples, embed_size]
            negative_embeddings = neg_embeddings_flat.view(B, N, -1)
            # For negatives, compute score for each negative sample, then average over negatives
            neg_scores = (target_embedding.unsqueeze(1) * negative_embeddings).sum(dim=2) # shape = [B, N] since sum(dim=2) completes the dot product
            neg_loss = F.softplus(neg_scores).mean(dim=1) # this computes the softplus and sums all the values for a given target

            # Compute loss and update
            loss = pos_loss.mean() + neg_loss.mean() # this computes the mean pos loss and mean negative loss per

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        epoch_time = time.time() - epoch_start  # Calculate epoch duration
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds, Loss: {total_loss:.4f}")
    
    total_training_time = time.time() - total_training_start # Calculate total time taken
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    return model

# ===============================
# Obtaining images from trajectories
# ===============================
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

def get_img(traj_name, q_model): # Here the return variable needs to be adjusted - vec1 only gives values for 1 dones
    vec1 = []
    val = []
    traj = np.load(f"{traj_name}.npy", allow_pickle= True)
    for i in range(len(traj)):
        # temp = []
        print("Current episode = ", i+1)
        for idx in range(len(traj[i]))[1:]:
            obs = traj[i][idx][0]
            # q_op = q_network(torch.Tensor(obs).to(device))
            q_op = q_model(torch.Tensor(obs).to(device))
            q_op = q_op.cpu().detach().numpy()
            a = np.argmax(q_op)
            v = np.max(q_op)
            val.append((v,a)) # to store the Value function and action chosen in current state
            vec1.append(obs)
            # vec1.append(torch.Tensor(obs).to(device))
    del(traj)
    gc.collect()
    return vec1, val

def context_create(val):
    val_5 = []
    val_6 = []
    val_7 = []
    val_8 = []
    val_9 = []
    val_high = []
    a_0 = []
    a_1 = []
    a_2 = []
    a_3 = []
    for idx in range(len(val)):
        if val[idx][0]<=5:
            val_5.append(idx)
        elif val[idx][0]<=6:
            val_6.append(idx)
        elif val[idx][0]<=7:
            val_7.append(idx)
        elif val[idx][0]<=8:
            val_8.append(idx)
        elif val[idx][0]<=9:
            val_9.append(idx)
        else:
            val_high.append(idx)
        # end if else

        if val[idx][1] == 0:
            a_0.append(idx)
        elif val[idx][1] == 1:
            a_1.append(idx)
        elif val[idx][1] == 2:
            a_2.append(idx)
        else:
            a_3.append(idx)
    
    return val_5, val_6, val_7, val_8, val_9, val_high, a_0, a_1, a_2, a_3


# ===============================
# Main: Running the Training
# ===============================
if __name__ == "__main__":
    # Parameters that we use
    batch = 4
    context_len = 4 # number of +ve contexts per target img
    neg_len = 16 # number of -ve samples per target img
    embed_dim=3136

    # Define the CNN model
    idex = 5000000 # Using the 5M model
    run_folder = f"runs_cnn"
    device = "cuda"
    run_name = f"BreakoutNoFrameskip-v4__dqn_atari__multiple_10M_cnn_fcc_split"
    saved_model_path = f"{run_folder}/{run_name}/dqn_atari.cleanrl_model_{idex}"
    # q_network = cust_CNN_Model(envs).to(device)
    q_network = cust_CNN_Model().to(device)
    q_network.load_state_dict(torch.load(saved_model_path, map_location=device))
    # cnn_state_dict = {k: v for k, v in q_network.state_dict().items() if "cnn" in k}

    # We use 1M trained CNN for training w2v
    idex_2 = 1000000
    saved_model_path_load = f"{run_folder}/{run_name}/dqn_atari.cleanrl_model_{idex_2}" # using the 1M model as baseline
    q_1m_model = cust_CNN_Model().to(device)
    q_1m_model.load_state_dict(torch.load(saved_model_path, map_location=device))
    cnn_state_dict = {k: v for k, v in q_1m_model.state_dict().items() if "cnn" in k}

    check_code = False 
    # Make sure the model's return is adjusted to allow for multiple vector outputs
    if check_code:
        print("Code check started!\n")
        images = random_batch = np.random.randint(0, 256, (10, 1, 4, 84, 84), dtype=np.uint8)
        print(random_batch.shape)  # Output: (10, 1, 4, 84, 84)
        model = ImageSkipGramModel(embed_size=embed_dim, freeze_pretrained=True).to(device)
        model = load_model(model=model, cnn_state_dict=cnn_state_dict)
        model.eval()
        for i in range(len(images))[0:1]:
            with torch.no_grad():
                og_vec, new_vec = model(torch.Tensor(images[i]).to(device))
                og_vec = og_vec.cpu().numpy()
                new_vec = new_vec.cpu().numpy()
            print("initial vector: ", og_vec)
            print("vec from w2v: ", new_vec)
            print("norm of difference = ", np.linalg.norm(og_vec - new_vec))

        assert False, "Code check"

    traj_name = "trajectories_new"
    images, val = get_img(traj_name, model = q_network)
    print("Images and val computed successfully")

    val_5, val_6, val_7, val_8, val_9, val_high, a_0, a_1, a_2, a_3 = context_create(val=val)
    print("All the criteria lists created")
    
    # Train the model; set fine_tune=True to allow the CNN to update.
    model = train_model(images, cnn_state_dict=cnn_state_dict, embed_size=embed_dim, batch_size=batch, num_contexts = context_len,
                        num_neg_samples=neg_len, fine_tune=False)
    
    version = 1
    run_name_2 = f"Breakoutv4_w2v_img_3_similarity_5M_cnn_trajectories_new_{version}"
    model_path = f"{run_folder}/{run_name_2}/w2v.value_{idex}_cnn_{idex_2}"
    log_dir = f"{run_folder}/{run_name_2}"
    # Create the directory
    os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(log_dir):
        print(f"The directory {log_dir} exists.")
    else:
        print(f"Failed to create the directory {log_dir}.")
    
    torch.save(model.state_dict(), model_path)

    # Extract a refined embedding for a sample image.
    # Apply the same transformation as in the dataset.  
    sample_img = safe_transform(images[0]).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        embedding = model(sample_img).detach().numpy()
    print("Refined image embedding:", embedding)
