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
import gc
import psutil

# ===============================
# Memory logging and CPU/GPU consumption
# ===============================
def log_memory_usage(tag="Memory Usage"):
    """
    Logs the memory usage of the current process.
    `tag` helps identify where in the execution the logging occurs.
    """
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    print(f"[{tag}] Memory Usage: {mem_usage:.2f} MB")

def log_gpu_memory_usage(tag="GPU Memory Usage"):
    """
    Logs the GPU memory usage if CUDA is available.
    """
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[{tag}] GPU Memory Allocated: {gpu_memory:.2f} MB")

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

    context_indices_3 = list(range(max(0, idx - window_thresh), min(num_images, idx + window_thresh + 1)))
    
    context_indices = list(set(context_indices_1) & set(context_indices_2) & set(context_indices_3))

    if idx in context_indices:
            context_indices.remove(idx)

    if len(context_indices)<4:
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
    neg_indices = list(set(range(num_images)) - set(context_indices_1) - set(context_indices_2) - set(context_indices_3))
    
    return context_indices, neg_indices


class ImageSkipGramDataset(Dataset):
    def __init__(self, images, img_contexts, img_negs, window_size=10, num_contexts = 4, num_neg_samples=16):
        """
        images: List of preprocessed Atari observations.
                Each observation is expected to be a NumPy array or torch.Tensor with shape (1,4,84,84)
                or (4,84,84).
        window_size: Number of nearby images (in the deduplicated list) to use as context.
        num_neg_samples: Number of negative samples per target.
        """

        # self.image_list = []         # Will store only unique images.
        # self.image_hashes = {}         # Maps computed hash -> index in image_list.
        # self.image_occurrences = {}    # Maps unique index -> list of original indices.
        # self.val_list = []
        
        # for idx, img in enumerate(images):
        #     # Convert to numpy array if needed (for hashing)
        #     if isinstance(img, torch.Tensor):
        #         img_np = img.detach().cpu().numpy()
        #     elif isinstance(img, np.ndarray):
        #         img_np = img
        #     else:
        #         raise ValueError("Expected image to be numpy array or torch.Tensor.")
            
        #     # For hashing, remove extra singleton dimension if present.
        #     if img_np.ndim == 4 and img_np.shape[0] == 1:
        #         img_for_hash = np.squeeze(img_np, axis=0)
        #     else:
        #         img_for_hash = img_np

        #     # Ensure consistent type: assume pixel values are 0-255 (uint8)
        #     if img_for_hash.dtype != np.uint8:
        #         img_for_hash = np.uint8(img_for_hash)
            
        #     # Compute MD5 hash.
        #     img_hash = hashlib.md5(img_for_hash.tobytes()).hexdigest()
        #     if img_hash not in self.image_hashes:
        #         self.image_hashes[img_hash] = len(self.image_list)
        #         self.image_list.append(img)  # Store the raw image as provided.
        #         self.val_list.append(val[idx])
        #         self.image_occurrences[self.image_hashes[img_hash]] = []
        #     self.image_occurrences[self.image_hashes[img_hash]].append(idx)

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
        # self.contexts = img_contexts
        # self.negs = img_negs

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

        # # When self.contexts and self.negs is available i.e. storage of indices works
        # context_idx = random.sample(self.contexts[idx], self.num_contexts)
        # context = torch.stack([self.transform(self.image_list[i]) for i in context_idx])

        # negative_samples = random.sample(self.negs[idx], self.num_neg_samples)
        # negatives = torch.stack([self.transform(self.image_list[i]) for i in negative_samples])
        
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

# idex = 5000000 # Using the 5M model
# run_folder = f"runs_cnn"
# device = "cuda"
# run_name_2 = f"BreakoutNoFrameskip-v4__dqn_atari__multiple_10M_cnn_fcc_split"
# saved_model_path = f"{run_folder}/{run_name_2}/dqn_atari.cleanrl_model_{idex}"
# # q_network = cust_CNN_Model(envs).to(device)
# q_network = cust_CNN_Model().to(device)
# q_network.load_state_dict(torch.load(saved_model_path, map_location=device))

###################################################


# class ImageEmbeddingExtractor(nn.Module):
#     def __init__(self, embed_size=512):
#         """
#         Uses a pretrained ResNet-18 to extract features from images,
#         then maps them to a lower-dimensional space (embed_size).
#         """
#         super(ImageEmbeddingExtractor, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         # Freeze the CNN backbone
#         for param in self.cnn.parameters():
#             param.requires_grad = False

#         # self.w2v = nn.Sequential(
#         #     nn.Linear(3136, embed_size)
#         # )

#     def forward(self, img):
#         x = self.cnn(img/255.0)  # (B, 512, 1, 1)
#         # x = x.view(x.size(0), -1)         # Flatten to (B, 512)
#         x *= 255.0
#         x = self.w2v(x / 255.0)                    # Map to (B, embed_size)
#         return x

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
        # self.encoder = ImageEmbeddingExtractor(embed_size)

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
            # for param in self.encoder.parameters():
            for param in self.cnn.parameters():
                param.requires_grad = False
                
        # An output layer that further transforms embeddings (for context prediction)
        self.output_layer = nn.Linear(3136, embed_size)
        with torch.no_grad(): # this step initializes the weights to identity matrix
            # This ensures initial output of output_layer matches that of the encoder
            self.output_layer.weight.copy_(torch.eye(embed_size))  # Identity matrix
            self.output_layer.bias.fill_(0)  # Zero bias
    
    def forward(self, target_img):
        # Generate the embedding for an input image
        # target_embedding = self.encoder(target_img)
        target_embedding = self.cnn(target_img / 255.0)
        # target_embedding *= 255.0
        # target_embedding = self.output_layer(target_embedding / 255.0) # this creates a copy of the target_embeddings

        # Since the target layer is a single layer. Hence, we don't need to normalize it's input.
        target_embedding = self.output_layer(target_embedding) # this creates a copy of the target_embeddings
        return target_embedding
    
    # def loss_function(self, pos_scores, neg_scores):
    #     """
    #     Negative sampling loss: aims to maximize the score of the positive (context) pair
    #     and minimize the score for negative (random) pairs.
    #     """
    #     pos_loss = torch.log(self.sigmoid(pos_scores)).mean()
    #     neg_loss = torch.log(1 - self.sigmoid(neg_scores)).mean()
    #     return -(pos_loss + neg_loss)

    # # Alternate loss function for more stability
    # def loss_function(self, pos_scores, neg_scores):
    #     pos_loss = F.softplus(-pos_scores).mean()
    #     neg_loss = F.softplus(neg_scores).mean()
    #     return pos_loss + neg_loss

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
def train_model(images, img_contexts, img_negs, cnn_state_dict, embed_size=512, window_size=2, epochs=10, batch_size=16, lr=0.001, num_contexts = 4, num_neg_samples=5, fine_tune=False):
    """
    images: List of PIL Image objects (84x84)
    fine_tune: If True, allows the pretrained CNN to be updated during training.
    """
    log_memory_usage("Before Dataset Loading")
    dataset = ImageSkipGramDataset(images, img_contexts, img_negs, window_size, num_contexts, num_neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    log_memory_usage("After Dataset Loading")
    
    # model = ImageSkipGramModel(embed_size, freeze_pretrained=not fine_tune).to(device)
    model = ImageSkipGramModel(embed_size, freeze_pretrained=not fine_tune)
    model = load_model(model = model, cnn_state_dict=cnn_state_dict) # Loads the CNN weights but not the w2v layer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    total_training_start = time.time()  # Start timing overall training
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1} started!")
        log_memory_usage(f"Epoch {epoch+1} - Before Training")
        epoch_start = time.time()  # Start timing for this epoch
        total_loss = 0.0
        # flag = True
        count = 0
        for target, context, negatives in dataloader:
            log_memory_usage(f"Epoch {epoch+1}, Batch {count+1} - Before Forward Pass")
            # print(f"Time taken by dataloader = {(dataloader_time - epoch_start):.2f} seconds ")
            optimizer.zero_grad()
            
            # Forward pass: get embeddings for target, context, and negative samples
            target_embedding = model(target)
            B, num_c, C, H, W = context.shape
            context_flat = context.view(B * num_c, C, H, W)  # Flatten to shape [B*N, C, H, W]
            context_embeddings_flat = model(context_flat)
            context_embeddings = context_embeddings_flat.view(B, num_c, -1) # Reshape to [B, num_contexts, embed_size]

            # if flag:
            #     print("Context shape: ", context.shape)
            #     print("context_embeddings shape: ", context_embeddings.shape)
            #     flag = False

            # Now we process the positives to ensure several positives are included in the loss calculation
            # Compute positive scores
            pos_scores = (target_embedding.unsqueeze(1) * context_embeddings).sum(dim=2)  # Shape: [B, num_context]
            
            # Average positive scores **per target** to prevent inter-target mixing
            '''# Older method'''
            # pos_scores = pos_scores.mean(dim=1)  # Shape: [B]
            '''# Newer method'''
            pos_loss = F.softplus(-pos_scores).mean(dim=1)
            

            # Process negatives individually.
            # negatives shape: [B, num_neg_samples, C, H, W]
            B, N, C, H, W = negatives.shape
            negatives_flat = negatives.view(B * N, C, H, W)  # Flatten to shape [B*N, C, H, W]
            
            neg_embeddings_flat = model(negatives_flat)  # negatives: (B, num_neg_samples, C)

            # Reshape back to [B, num_neg_samples, embed_size]
            negative_embeddings = neg_embeddings_flat.view(B, N, -1)
            
            # For negatives, compute score for each negative sample, then average over negatives
            '''# Older method'''
            # neg_scores = (target_embedding.unsqueeze(1) * negative_embeddings).sum(dim=2).mean(dim=1)
            '''# Newer method'''
            neg_scores = (target_embedding.unsqueeze(1) * negative_embeddings).sum(dim=2) # shape = [B, N] since sum(dim=2) completes the dot product
            neg_loss = F.softplus(neg_scores).mean(dim=1) # this computes the softplus and sums all the values for a given target
            
            log_memory_usage(f"Epoch {epoch+1}, Batch {count+1} - After Forward Pass")

            # Compute loss and update
            '''# Older method'''
            # loss = model.loss_function(pos_scores, neg_scores)
            '''# Newer method'''
            loss = pos_loss.mean() + neg_loss.mean() # this computes the mean pos loss and mean negative loss per

            log_memory_usage(f"Epoch {epoch+1}, Batch {count+1} - Before Backward Pass")

            loss.backward()
            optimizer.step()

            log_memory_usage(f"Epoch {epoch+1}, Batch {count+1} - After Backward Pass")

            total_loss += loss.item()
            count+=1
        
        # torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start  # Calculate epoch duration
        log_memory_usage(f"Epoch {epoch+1} - After Training")
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

def get_img(traj_name, model): # Here the return variable needs to be adjusted - vec1 only gives values for 1 dones
    vec1 = []
    val = []
    traj = np.load(f"{traj_name}.npy", allow_pickle= True)
    for i in range(len(traj)):
        # temp = []
        print("Current episode = ", i+1)
        for idx in range(len(traj[i]))[1:]:
            obs = traj[i][idx][0]
            # q_op = q_network(torch.Tensor(obs).to(device))
            q_op = model(torch.Tensor(obs).to(device))
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

def cleaned_data(images, val):
    image_list = []         # Will store only unique images.
    image_hashes = {}         # Maps computed hash -> index in image_list.
    image_occurrences = {}    # Maps unique index -> list of original indices.
    val_list = []
    
    for idx, img in enumerate(images):
        # Convert to numpy array if needed (for hashing)
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        elif isinstance(img, np.ndarray):
            img_np = img
        else:
            raise ValueError("Expected image to be numpy array or torch.Tensor.")
        
        # For hashing, remove extra singleton dimension if present.
        if img_np.ndim == 4 and img_np.shape[0] == 1:
            img_for_hash = np.squeeze(img_np, axis=0)
        else:
            img_for_hash = img_np

        # Ensure consistent type: assume pixel values are 0-255 (uint8)
        if img_for_hash.dtype != np.uint8:
            img_for_hash = np.uint8(img_for_hash)
        
        # Compute MD5 hash.
        img_hash = hashlib.md5(img_for_hash.tobytes()).hexdigest()
        if img_hash not in image_hashes:
            image_hashes[img_hash] = len(image_list)
            image_list.append(img)  # Store the raw image as provided.
            val_list.append(val[idx])
            image_occurrences[image_hashes[img_hash]] = []
        image_occurrences[image_hashes[img_hash]].append(idx)
    # end for images
    return image_list, val_list, image_hashes, image_occurrences

###########################################
# ===============================
# Creating the context_indices for each target image
# ===============================
def store_context(images, val, window_size = 2):
    num_images = len(images)
    window_thresh = 10*window_size
    img_contexts = []
    img_negs = []
    # context_indices = []
    for idx in range(num_images):
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

        context_indices_3 = list(range(max(0, idx - window_thresh), min(num_images, idx + window_thresh + 1)))
        
        context_indices = list(set(context_indices_1) & set(context_indices_2) & set(context_indices_3))
    
        if idx in context_indices:
            context_indices.remove(idx)

        if len(context_indices)<4:
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
        img_contexts.append(context_indices)

        # Now we compute the neg_samples
        neg_indices = list(set(range(num_images)) - set(context_indices_1) - set(context_indices_2) - set(context_indices_3))
        img_negs.append(neg_indices)
    
    return img_contexts, img_negs # may return an empty list


# ===============================
# Main: Running the Training
# ===============================
if __name__ == "__main__":
    # Define the CNN model
    idex = 5000000 # Using the 5M model
    run_folder = f"runs_cnn"
    device = "cuda"
    run_name_2 = f"BreakoutNoFrameskip-v4__dqn_atari__multiple_10M_cnn_fcc_split"
    saved_model_path = f"{run_folder}/{run_name_2}/dqn_atari.cleanrl_model_{idex}"
    # q_network = cust_CNN_Model(envs).to(device)
    q_network = cust_CNN_Model().to(device)
    q_network.load_state_dict(torch.load(saved_model_path, map_location=device))
    cnn_state_dict = {k: v for k, v in q_network.state_dict().items() if "cnn" in k}

    traj_name = "trajectories_new"
    # cleaned_data_path = f"trajectories/cleaned_{traj_name}"

    # # To store the cleaned_version of the model
    # if not os.path.exists(cleaned_data_path):
    #     os.makedirs(cleaned_data_path)
    #     # Get images from the trajectories file
    #     traj = np.load(f"{traj_name}.npy", allow_pickle= True)
    #     # images, val = get_img(traj=traj)
    #     images, val = get_img(traj_name)
    #     clean_images, val_list, image_hashes, image_occurrences = cleaned_data(images, val)
    #     # then create the file here
    #     np.save(f"{cleaned_data_path}/clean_images.npy", np.array(clean_images, dtype=object))
    #     np.save(f"{cleaned_data_path}/val_list.npy", np.array(val_list, dtype=object))
    #     np.save(f"{cleaned_data_path}/image_hashes.npy", np.array(image_hashes, dtype=object))
    #     np.save(f"{cleaned_data_path}/image_occurrences.npy", np.array(image_occurrences, dtype=object))
    
    # else:
    #     clean_images = np.load(f"{cleaned_data_path}/clean_images.npy", allow_pickle= True)
    #     val_list = np.load(f"{cleaned_data_path}/val_list.npy", allow_pickle= True)
    #     # image_hashes = np.load(f"{cleaned_data_path}/image_hashes.npy", allow_pickle= True)
    #     # image_occurrences = np.load(f"{cleaned_data_path}/image_occurrences.npy", allow_pickle= True)
    
    # traj = np.load(f"{traj_name}.npy", allow_pickle= True)
    # images, val = get_img(traj=traj)
    log_memory_usage("Memory Before trajectory loading")
    images, val = get_img(traj_name, model = q_network)
    print("Images and val computed successfully")

    clean_memory = False # True: will delete images, val variables to clear out memory
    deduplication = False # False: We don't mind duplication of images, True: We use the hashing to remove duplicates
    store_context_boolean = False
    '''
    The deduplication still needs to be checked to ensure that post hashing we don't loss imp information about 
    neighbours in the context_indices computation
    '''

    batch = 4
    context_len = 4
    neg_len = 16
    embed_dim=3136
    if deduplication:
        # clean_images, val_list, image_hashes, image_occurrences = cleaned_data(images, val)
        clean_images, val_list, _, _ = cleaned_data(images, val)
        print("Cleaned Images computed successfully")
        # If memory problems, then we purge initial images, traj and val from memory
        if clean_memory:
            del(images)
            del(val)
            
            gc.collect()
            # del(traj)
        # end memory if
        val_5, val_6, val_7, val_8, val_9, val_high, a_0, a_1, a_2, a_3 = context_create(val=val_list)
        img_contexts, img_negs = store_context(clean_images, val_list, window_size = 2)
        # Train the model; set fine_tune=True to allow the CNN to update.
        # model = train_model(images = clean_images, val = val_list, embed_size=3136, fine_tune=False)
        model = train_model(images = clean_images, img_contexts = img_contexts, img_negs = img_negs,
                             cnn_state_dict=cnn_state_dict, embed_size=embed_dim, batch_size=batch, 
                            num_contexts = context_len, num_neg_samples=neg_len, fine_tune=False)

    else:
        log_memory_usage("Before context criteria i.e. list of vals & a created")
        val_5, val_6, val_7, val_8, val_9, val_high, a_0, a_1, a_2, a_3 = context_create(val=val)
        print("All the criteria lists created")
        
        if store_context_boolean:
            # When storing the image memory
            log_memory_usage("Before storing of contexts created")
            img_contexts, img_negs = store_context(images, val, window_size = 2)
            print("All the contexts and negatives are computed and stored")
            # Search if there are any negatives in img_contexts:
            print("Negative values check: ")
            for idx in range(len(images)):
                temp = np.where(np.array(img_contexts[idx])<0)[0]
                if len(temp)>0:
                    print(f"Negative index found at {idx}: indices = {img_contexts[idx]}")
                    assert False, "Only +ve integer indices are accepted"
                else:
                    # print(f"img_contexts[{idx}]: {img_contexts[idx]}")
                    continue

            # assert False, "Checking indices"
            
            # Train the model; set fine_tune=True to allow the CNN to update.
            model = train_model(images, img_contexts, img_negs, cnn_state_dict=cnn_state_dict, embed_size=embed_dim, batch_size=batch, num_contexts = context_len,
                                num_neg_samples=neg_len, fine_tune=False)
        else:
            # Train the model; set fine_tune=True to allow the CNN to update.
            model = train_model(images, img_contexts = [], img_negs = [], cnn_state_dict=cnn_state_dict, embed_size=embed_dim, batch_size=batch, num_contexts = context_len,
                             num_neg_samples=neg_len, fine_tune=False)
    
    
    version = 2
    run_name = f"Breakoutv4_w2v_img_similarity_5M_cnn_trajectories_new_{version}"
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
