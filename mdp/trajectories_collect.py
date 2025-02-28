import numpy as np
import random
from tqdm import tqdm
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
# ============================================================================================
# Main function: Train and show results
# ============================================================================================
env_name = "FrozenLake-v1"
env_dim = 4
stochastic = False
seed = 42
num_episodes = 1_000
num_states = 16
num_actions = 4
modified = "perfect"

max_esp_len = 100

random.seed(seed)
np.random.seed(seed=seed)

def choose_action(q_table, state, epsilon):
    """Epsilon-greedy action selection."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)  # Explore
    else:
        return np.argmax(q_table[state, :])  # Exploit
    
def make_env(env_name, env_dim = 4, seed = 42, stochastic = False):
    env = gym.make(env_name, desc=generate_random_map(size=env_dim, seed=seed), 
                   is_slippery = stochastic, render_mode = 'rgb_array')
    return env

# ============================================================================================
# Load the model and colllect trajectories
# ============================================================================================

file_name = f"Q_table_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy"
q_table = np.load(f"mdp/{file_name}")
env = make_env(env_name=env_name, env_dim=env_dim, seed = seed, stochastic=stochastic)

traj = []
tuple_traj = False
for e in tqdm(range(num_episodes)):
    # print(e)
    if modified=="perfect":
        _, _ = env.reset()
        epsilon = 0.01
        state = int(e%16)
    else:
        continue    
    # if e<0.1*num_episodes:
    #   epsilon = 1
    # elif e<0.2*num_episodes:
    #   epsilon = 0.8
    # elif e<0.3*num_episodes:
    #   epsilon = 0.6
    # elif e<0.4*num_episodes:
    #   epsilon = 0.4
    # elif e<0.5*num_episodes:
    #   epsilon = 0.2
    # elif e<0.6*num_episodes:
    #   epsilon = 0.1
    # elif e<0.7*num_episodes:
    #   epsilon = 0.05
    # else:
    #   epsilon = 0.01
    # # end epsilon if
    # if random.uniform(0, 1) < 0.3:
    #     if random.uniform(0, 1) < 0.5:
    #         state = 3
    #     elif random.uniform(0, 1) < 0.65:
    #         state = 6
    #     else:
    #         state = 7
    #     # end if choosing state
    # else:
    #     state, _ = env.reset(seed=seed)
    
    if not tuple_traj:
       traj.append('s_'+str(state))
    '''
    Action Stochasticity (is_slippery=True): The seed affects how the agent slips 
    (randomly moves instead of following the chosen action).
    Random Hole Placement (if map is generated dynamically): If the map has random 
    elements, different seeds can affect the placement of H (holes).
    '''
    curr_reward = 0
    rep_count = 0
    for t in range(max_esp_len):
      action = choose_action(q_table, state, epsilon) #To be implemented
      n_state,reward,done,_,_ = env.step(action)
      
      # We store the current tuple
      if tuple_traj:
        traj.append((state, action, reward, n_state, done))
      else:
        temp = 's_'+str(n_state)
        # traj.append('s_'+str(n_state))
        traj.append(temp)

      state = n_state
      curr_reward += reward
      if done:
        if rep_count>=5: # This forces repetitions to occur when done becomes True thereby repeating ending states
            if not tuple_traj:
                if temp=='s_15':
                    traj.append('s_'+str(16))
                else:
                    traj.append('s_'+str(17))
            break
        else:
            if not tuple_traj:
                traj.append(temp)
        rep_count+=1
    # end for
# end for
if tuple_traj:
    np.save(f"mdp/modified_tuple_trajectories_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy", traj)
else:
    np.save(f"mdp/modified_{modified}_trajectories_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy", traj)
print("SAved!")

assert False, "No w2v business here"
# ============================================================================================
# We perform w2v on the collected trajectories
# ============================================================================================
# print("The trajectory we use is: \n", traj)
# print("\n")

'''
Using word2vec as is doesn't work because:
    1. We have a continuous state space
    2. We don't have a way to show same states again. This is a problem in training w2v since we need multiple
    context samples for each word else we can't train the w2v model. Thus, we need a rule to generate several
    positive examples for each state that we see in the trajectory
'''
# # All the imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# Define a function to convert trajectories into text

embed_dim = 4
# Start defining word2vec as provided by ChatGPT-4o
def tokenize_text(text): # This function isn't needed since the trajectory is already in the form of tokens
    return text.lower().split()

def build_vocab(text): # Again we already have a vocabulary hence don't need to use this function directly
    # words = tokenize_text(text)
    # words = text
    # word_counts = Counter(words)
    word_counts = Counter(text)
    vocab = {word: i for i, word in enumerate(word_counts.keys())}
    reverse_vocab = {i: word for word, i in vocab.items()}
    return vocab, reverse_vocab, word_counts

def generate_skipgram_pairs(text, window_size=2): # This function gives the word and context pairs.
    # words = tokenize_text(text)
    words = text
    pairs = []
    for i, target_word in enumerate(words):
        window_start = max(i - window_size, 0)
        window_end = min(i + window_size + 1, len(words))
        for j in range(window_start, window_end):
            if i != j:
                pairs.append((words[i], words[j]))
    return pairs


class Word2VecDataset_negative(Dataset):
    def __init__(self, text, vocab, window_size=2, num_neg_samples=5):
        self.vocab = vocab
        self.data = generate_skipgram_pairs(text, window_size)
        self.num_neg_samples = num_neg_samples
        self.vocab_size = len(vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx] # How does this idx reflect the actual idx of the word?
        ''' Doesn't the above idx reflect the idx numbered pair instead in pairs instead of the pairs
        corresponding to the word at idx? '''
        target_idx = torch.tensor(self.vocab[target], dtype=torch.long)
        context_idx = torch.tensor(self.vocab[context], dtype=torch.long)

        # Negative Sampling: Random words sampled (excluding actual context word)
        negative_samples = torch.randint(0, self.vocab_size, (self.num_neg_samples,))
        while context_idx in negative_samples:
            negative_samples = torch.randint(0, self.vocab_size, (self.num_neg_samples,))

        return target_idx, context_idx, negative_samples


# Building the skip-gram network
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim, precomputed_vectors=None):
        super(SkipGramNegativeSampling, self).__init__()

        # Create an embedding layer
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize embeddings with precomputed word vectors (if available)
        if precomputed_vectors is not None:
            self.embeddings.weight.data.copy_(torch.tensor(precomputed_vectors, dtype=torch.float32))

        # Output layer (to predict context words)
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, target_idx):
        target_vec = self.embeddings(target_idx)  # Get word embeddings
        output = self.output_layer(target_vec)  # Compute logits for context words
        final_output = self.sigmoid(output)  # Apply sigmoid to get probabilities
        return final_output

    def loss_function(self, pos_scores, neg_scores):
        """
        Implements negative sampling loss.
        pos_scores: Scores for true context words
        neg_scores: Scores for sampled negative words
        """
        pos_loss = torch.log(self.sigmoid(pos_scores)).mean()
        neg_loss = torch.log(1 - self.sigmoid(neg_scores)).mean()
        return -(pos_loss + neg_loss)  # Negative log likelihood


# Create the training loop:
def train_model_negative_sampling(text, precomputed_vectors, embed_dim=embed_dim, window_size=2, epochs=10, batch_size=16, lr=0.005, num_neg_samples=5):
    vocab, reverse_vocab, _ = build_vocab(text)
    dataset = Word2VecDataset_negative(text, vocab, window_size, num_neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Convert precomputed vectors into a matrix (shape: vocab_size x embed_dim)
    word_vector_matrix = np.zeros((len(vocab), embed_dim))
    for word, idx in vocab.items():
        # word_vector_matrix[idx] = precomputed_vectors.get(word, np.random.randn(embed_dim))  # Use random if not found
        word_vector_matrix[idx] = np.random.randn(embed_dim)
        # assert False, "Check above line for the .get() function"

    model = SkipGramNegativeSampling(len(vocab), embed_dim, word_vector_matrix)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx, negative_samples in dataloader:
            optimizer.zero_grad()

            # Forward pass
            predictions = model(target_idx)  # Logits (before sigmoid)
            pos_scores = predictions.gather(1, context_idx.unsqueeze(1)).squeeze(1)  # Get true context scores
            neg_scores = predictions.gather(1, negative_samples).mean(dim=1)  # Get negative samples scores

            # Compute loss
            loss = model.loss_function(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model, vocab # this ensures that we can use the vocab created here

# ============================================================================================
# Using softmax instead of negative sampling in word2vec:
# ============================================================================================
# import numpy as np

class SkipGram:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Initialize input and output weight matrices
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01  # Input word vectors
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01  # Output word vectors

    def softmax(self, x):
        """Compute softmax values for each set of scores in x."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum(axis=0)

    def forward(self, center_word_idx):
        """Forward pass to compute probability distribution over output words."""
        hidden_layer = self.W1[center_word_idx]  # Hidden layer activation
        output_layer = np.dot(hidden_layer, self.W2)  # Compute scores for output layer
        y_pred = self.softmax(output_layer)  # Softmax to get probabilities
        return hidden_layer, y_pred

    def backward(self, center_word_idx, context_word_idx, hidden_layer, y_pred):
        """Backward pass to update weights using gradient descent."""
        # Compute error
        y_true = np.zeros(self.vocab_size)
        y_true[context_word_idx] = 1  # One-hot encoded target

        error = y_pred - y_true  # Gradient of loss with respect to softmax output
        grad_W2 = np.outer(hidden_layer, error)  # Gradient for W2
        grad_W1 = np.dot(self.W2, error)  # Gradient for W1

        # Update weights
        self.W1[center_word_idx] -= self.learning_rate * grad_W1
        self.W2 -= self.learning_rate * grad_W2.T  # Update all output vectors

    def train(self, training_data, epochs=10):
        """Train the SkipGram model on a given dataset."""
        for epoch in range(epochs):
            total_loss = 0
            for center_word_idx, context_word_idx in training_data:
                hidden_layer, y_pred = self.forward(center_word_idx)
                self.backward(center_word_idx, context_word_idx, hidden_layer, y_pred)
                total_loss -= np.log(y_pred[context_word_idx] + 1e-9)  # Cross-entropy loss
            
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    def get_word_vector(self, word_idx):
        """Return the learned word embedding vector for a given word index."""
        return self.W1[word_idx]


# ============================================================================================
# w2v computations:
# ============================================================================================

# Train model
text = traj
word_vectors = []
model, vocab = train_model_negative_sampling(text, word_vectors) # word_vectors corresponds to the precomputed-vectors for each word

# Get the learned embedding for a word
word = "s0_0"
if word in vocab:  # ✅ Ensure word exists in vocab
    word_idx = torch.tensor([vocab[word]], dtype=torch.long)
    updated_embedding = model.embeddings(word_idx).detach().numpy()
    print(f"Updated embedding for '{word}': {updated_embedding}")
else:
    print(f"Word '{word}' not found in vocabulary.")

# Initialize an empty dictionary to store embeddings
word_embeddings = {}
for word in vocab:
    word_idx = torch.tensor([vocab[word]], dtype=torch.long)
    updated_embedding = model.embeddings(word_idx).detach().numpy()
    print(f"Updated embedding for '{word}': {updated_embedding}")
    # Store the embedding in the dictionary
    word_embeddings[word] = updated_embedding.flatten()  # Flatten to 1D array

np.save(f"mdp/w2v_embed_dim_{embed_dim}_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy", word_embeddings)

# ============================================================================================
# Similarity checking and visualizing 
# ============================================================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load saved embeddings
word_embeddings = np.load(f"mdp/w2v_embed_dim_{embed_dim}_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy", 
                          allow_pickle=True).item()

# Convert to a NumPy array for fast computation
words = list(word_embeddings.keys())
vectors = np.array(list(word_embeddings.values()))

# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(vectors)

# Function to find top-N similar words
def find_similar_words(target_word, top_n=5):
    if target_word not in word_embeddings:
        print(f"Word '{target_word}' not found in vocabulary.")
        return []

    # Get index of target word
    target_idx = words.index(target_word)

    # Get similarity scores for the target word
    similarity_scores = cosine_sim_matrix[target_idx]

    # Get top-N most similar words (excluding itself)
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]  # Sort in descending order

    # Return words with their similarity scores
    return [(words[i], similarity_scores[i]) for i in similar_indices]

# Example usage
for target_word in word_embeddings:
    top_similar_words = find_similar_words(target_word, top_n=5)

    print(f"Top 5 words similar to '{target_word}':")
    for word, score in top_similar_words:
        print(f"{word}: {score:.4f}")
