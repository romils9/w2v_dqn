import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
import hashlib
import time
import os



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
        # return torch.tensor(img, dtype=torch.float32) / 255.0
        return torch.tensor(img, dtype=torch.float32)
    raise ValueError("Image must be either a numpy array or torch.Tensor")


def get_context(idx, val, num_images, window_size = 2):
    # context_indices = []
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

    context_indices_3 = list(range(list(range(max(0, idx - 5*window_size), min(num_images, idx + 5*window_size + 1)))))
    
    context_indices = list(set(context_indices_1) & set(context_indices_2) & set(context_indices_3))

    # if len(context_indices)==0:
    #     return [] # return an 
    # else:
    #     return context_indices
    return context_indices # may return an empty list


class ImageSkipGramDataset(Dataset):
    def __init__(self, images, val, window_size=10, num_neg_samples=5):
        """
        images: List of preprocessed Atari observations.
                Each observation is expected to be a NumPy array or torch.Tensor with shape (1,4,84,84)
                or (4,84,84).
        window_size: Number of nearby images (in the deduplicated list) to use as context.
        num_neg_samples: Number of negative samples per target.
        """
        self.image_list = images  # List of images in the given shape.
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.num_images = len(images)
        self.transform = safe_transform
        self.val = val
        self.get_context = get_context # send idx, val, self.num_images, self.window_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Apply the transformation to the target image.
        target = self.transform(self.image_list[idx])  # Expected shape: (4,84,84)
        
        # Determine context indices (neighbors in the list excluding the target itself).
        # context_indices = list(range(max(0, idx - self.window_size), 
        #                              min(self.num_images, idx + self.window_size + 1)))

        # Obtaining context of the target image using a heuristic obtained from get_context()
        context_indices = self.get_context(idx = idx, val = self.val, num_images=self.num_images, window_size=self.window_size)

        if len(context_indices)==0:
            context_indices = list(range(max(0, idx - self.window_size), 
                                     min(self.num_images, idx + self.window_size + 1)))
            
        if idx in context_indices:
            context_indices.remove(idx)
        context_idx = random.choice(context_indices)
        context = self.transform(self.image_list[context_idx])
        
        # Negative sampling: choose negatives from images not in the context window nor the target.
        negative_indices = list(set(range(self.num_images)) - set(context_indices) - {idx})
        if len(negative_indices) < self.num_neg_samples:
            negative_samples = negative_indices
        else:
            negative_samples = random.sample(negative_indices, self.num_neg_samples)
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
            # nn.Linear(512, env.single_action_space.n)
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.cnn(x / 255.0)
        x *= 255.0
        x = self.fcc_layers(x / 255.0)
        return x

idex = 5000000 # Using the 5M model
run_folder = f"runs_cnn"
device = "cuda"
run_name_2 = f"BreakoutNoFrameskip-v4__dqn_atari__multiple_10M_cnn_fcc_split"
saved_model_path = f"{run_folder}/{run_name_2}/dqn_atari.cleanrl_model_{idex}"
# q_network = cust_CNN_Model(envs).to(device)
q_network = cust_CNN_Model().to(device)
q_network.load_state_dict(torch.load(saved_model_path, map_location=device))

class ImageEmbeddingExtractor(nn.Module):
    def __init__(self, embed_size=512):
        """
        Uses a pretrained ResNet-18 to extract features from images,
        then maps them to a lower-dimensional space (embed_size).
        """
        super(ImageEmbeddingExtractor, self).__init__()
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

        self.w2v = nn.Sequential(
            nn.Linear(3136, embed_size)
        )

    def forward(self, img):
        x = self.cnn(img/255.0)  # (B, 512, 1, 1)
        # x = x.view(x.size(0), -1)         # Flatten to (B, 512)
        x *= 255.0
        x = self.w2v(x / 255.0)                    # Map to (B, embed_size)
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
        self.encoder = ImageEmbeddingExtractor(embed_size)
        
        # Optionally freeze the pretrained CNN layers
        if freeze_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # An output layer that further transforms embeddings (for context prediction)
        self.output_layer = nn.Linear(embed_size, embed_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, target_img):
        # Generate the embedding for an input image
        target_embedding = self.encoder(target_img)
        return target_embedding
    
    # def loss_function(self, pos_scores, neg_scores):
    #     """
    #     Negative sampling loss: aims to maximize the score of the positive (context) pair
    #     and minimize the score for negative (random) pairs.
    #     """
    #     pos_loss = torch.log(self.sigmoid(pos_scores)).mean()
    #     neg_loss = torch.log(1 - self.sigmoid(neg_scores)).mean()
    #     return -(pos_loss + neg_loss)

    # Alternate loss function for more stability
    def loss_function(self, pos_scores, neg_scores):
        pos_loss = F.softplus(-pos_scores).mean()
        neg_loss = F.softplus(neg_scores).mean()
        return pos_loss + neg_loss

# ===============================
# Loading the weights into the model
# ===============================
def load_model(model, cnn_state_dict):
    model_state_dict = model.state_dict()
    model_keys = list(model.state_dict().keys())
    print("Keys in the model we're using: ", model_keys)
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
def train_model(images, val, embed_size=512, window_size=2, epochs=10, batch_size=16, lr=0.001, num_neg_samples=5, fine_tune=False):
    """
    images: List of PIL Image objects (84x84)
    fine_tune: If True, allows the pretrained CNN to be updated during training.
    """
    data_time_start = time.time()
    dataset = ImageSkipGramDataset(images, val, window_size, num_neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_total_time = time.time() - data_time_start
    print(f"Total time to create dataset and dataloader = {data_total_time:.2f} seconds ")
    
    model = ImageSkipGramModel(embed_size, freeze_pretrained=not fine_tune)
    cnn_state_dict = {k: v for k, v in q_network.state_dict().items() if "cnn" in k}

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
            context_embedding = model(context)

            # Process negatives individually.
            # negatives shape: [B, num_neg_samples, C, H, W]
            B, N, C, H, W = negatives.shape
            negatives_flat = negatives.view(B * N, C, H, W)  # Flatten to shape [B*N, C, H, W]
            
            neg_embeddings_flat = model(negatives_flat)  # negatives: (B, num_neg_samples, C)

            # Reshape back to [B, num_neg_samples, embed_size]
            negative_embeddings = neg_embeddings_flat.view(B, N, -1)
            
            # Compute similarity scores (dot product)
            pos_scores = (target_embedding * context_embedding).sum(dim=1)
            # For negatives, compute score for each negative sample, then average over negatives
            neg_scores = (target_embedding.unsqueeze(1) * negative_embeddings).sum(dim=2).mean(dim=1)
            
            # Compute loss and update
            loss = model.loss_function(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # torch.cuda.synchronize()
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
traj = np.load("trajectories_new.npy", allow_pickle= True)

# def get_img(): # Here the return variable needs to be adjusted - vec1 only gives values for 1 dones
#     for i in range(len(traj))[0:1]:
#         temp = []
#         print("Current episode = ", i+1)
#         for idx in range(len(traj[i]))[1:]:
#             temp.append(traj[i][idx][4])
#         # end for
#         dones = np.where(np.array(temp)==True)[0]
#         print("Number of Trues = ", len(dones))
#         print("Indexes where episodes were truncated = ", dones)

#         vec1 = []
#         # For the 1st end point in each episode:
#         for idx in range(dones[0])[1:]:
#             obs = traj[i][idx][0]
#             # vec0 = q_network(torch.Tensor(obs).to(device))
#             # vec0 = vec0.cpu().detach().numpy()
#             # print("vec0 type: ", vec0.dtype)
#             # vec1.append(vec0)
#             vec1.append(obs)
#             # print("Output values for each state: ", vec0)
    
#     return vec1

# Altenative get image func:
def get_img(): # Here the return variable needs to be adjusted - vec1 only gives values for 1 dones
    vec1 = []
    val = []
    for i in range(len(traj)):
        # temp = []
        print("Current episode = ", i+1)
        for idx in range(len(traj[i]))[1:]:
            obs = traj[i][idx][0]
            q_op = q_network(torch.Tensor(obs).to(device))
            q_op = q_op.cpu().detach().numpy()
            a = np.argmax(q_op)
            v = np.max(q_op)
            val.append((v,a)) # to store the Value function and action chosen in current state
            vec1.append(obs)
    
    return vec1, val

def context_create(val):
    # thresh = 0.1 # hyperparameter
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
        
    # end for
    # print("val_5 len: ", len(val_5))
    # print("val_6 len: ", len(val_6))
    # print("val_7 len: ", len(val_7))
    # print("val_8 len: ", len(val_8))
    # print("val_9 len: ", len(val_9))
    # print("val_high len: ", len(val_high))
    # print("a_0 len: ", len(a_0))
    # print("a_1 len: ", len(a_1))
    # print("a_2 len: ", len(a_2))
    # print("a_3 len: ", len(a_3))
    
    return val_5, val_6, val_7, val_8, val_9, val_high, a_0, a_1, a_2, a_3


# ===============================
# Main: Running the Training
# ===============================
if __name__ == "__main__":
    # Get images from the trajectories file
    images, val = get_img()
    val_5, val_6, val_7, val_8, val_9, val_high, a_0, a_1, a_2, a_3 = context_create(val=val)
    # assert False, "Break for val prints"
    
    # Train the model; set fine_tune=True to allow the CNN to update.
    model = train_model(images, val, embed_size=3136, fine_tune=True)
    
    run_name = "Breakoutv4_w2v_img_5M_cnn_trajectories_new"
    model_path = f"{run_folder}/{run_name}/w2v.cleanrl_model_{idex}"
    log_dir = f"{run_folder}/{run_name}"
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
