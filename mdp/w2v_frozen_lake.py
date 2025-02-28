import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
# ============================================================================================
# Loading the trajectories and defining all required variables
# ============================================================================================
env_name = "FrozenLake-v1"
env_dim = 4
stochastic = False
seed = 42
num_episodes = 1_000
num_states = 16
num_actions = 4
modified = "perfect"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(seed=seed)
torch.manual_seed(seed)

traj_file = f"mdp/modified_{modified}_trajectories_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}.npy"
# text = np.load(traj_file, allow_pickle=True).item()
text = np.load(traj_file)
# print("traj dtype: ", text[0].dtype)
# assert False

# ============================================================================================
# Defining functions required for w2v using SkipGram
# ============================================================================================
''' Potential values for embedding dimensions = {4, 8, 12, 16, 20, 32} '''
# w2v hyperparameters
embed_dim = 32
window_size = 2
batch_size = 16
w2v_epochs = 50
w2v_lr = 0.01

# Start defining word2vec as provided by ChatGPT-4o
# def tokenize_text(text): # This function only works when the text is a single string. We don't need it here
#     return text.lower().split()

def build_vocab(text): # Again we already have a vocabulary hence don't need to use this function directly
    # words = tokenize_text(text)
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

# vocab,_,wcounts = build_vocab(text)
# print("vocab: ", vocab)
# print("word counts: ", wcounts)
# assert False, "Checking the word counts!"

# ============================================================================================
# Class: Word2vec Dataset creator
# ============================================================================================

class Word2VecDataset(Dataset):
    def __init__(self, text, vocab, window_size=2):
        self.vocab = vocab
        self.data = generate_skipgram_pairs(text, window_size)
        self.vocab_size = len(vocab) # why is this needed here?

    def __len__(self): # what does this function do?
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        ''' Doesn't the above idx reflect the idx numbered pair instead in pairs instead of the pairs
        corresponding to the word at idx? '''
        target_idx = torch.tensor(self.vocab[target], dtype=torch.long)
        context_idx = torch.tensor(self.vocab[context], dtype=torch.long)
        # target_idx = self.vocab[target]
        # context_idx = self.vocab[context]

        return target_idx, context_idx

# ============================================================================================
# Class: SkipGram using softmax over entire vocabulary
# ============================================================================================

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding_dim = embedding_dim

        # Input word embedding
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Output word embedding (used for context words)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights (better stability)
        init_range = 0.5 / embedding_dim
        self.in_embedding.weight.data.uniform_(-init_range, init_range)
        self.out_embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_word_idx):
        """Compute word embeddings and softmax probabilities for the context words."""
        center_embed = F.relu(self.in_embedding(center_word_idx))  # Shape: (batch_size, embedding_dim)
        scores = torch.matmul(center_embed, self.out_embedding.weight.T)  # Compute dot product
        y_pred = torch.softmax(scores, dim=1)  # Apply softmax over output vocab
        return y_pred

    def get_word_vector(self, word_idx):
        """Return the learned embedding vector for a given word index."""
        return self.in_embedding(word_idx).detach().cpu().numpy()
    
# ============================================================================================
# Function: to train the w2v skipgram model
# ============================================================================================

def train_skipgram(model, data_loader, epochs=6, lr=0.01, device = device):
    """Train the SkipGram model using Adam optimizer."""
    criterion = nn.CrossEntropyLoss()  # Cross-entropy for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for center_word_idx, context_word_idx in data_loader:
            center_word_idx = center_word_idx.to(device)
            context_word_idx = context_word_idx.to(device)

            optimizer.zero_grad()
            y_pred = model(center_word_idx)  # Forward pass
            loss = criterion(y_pred, context_word_idx)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model

# ============================================================================================
#  Main function for calling the w2v agent
# ============================================================================================
# First we create the dataset and dataloader
vocab, _, _ = build_vocab(text)
dataset = Word2VecDataset(text, vocab, window_size)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
model = SkipGram(vocab_size=len(vocab), embedding_dim=embed_dim).to(device)
model = train_skipgram(model, dataloader, epochs = w2v_epochs, lr = w2v_lr)

# ============================================================================================
#  Saving the w2v generated vector embeddings as a dictionary
# ============================================================================================
word_embeddings = {}
for word in vocab:
    word_idx = torch.tensor([vocab[word]], dtype=torch.long).to(device)
    updated_embedding = model.get_word_vector(word_idx)
    # print(f"Updated embedding for '{word}': {updated_embedding}")
    # Store the embedding in the dictionary
    word_embeddings[word] = updated_embedding.flatten()  # Flatten to 1D array

np.save(f"mdp/modified_{modified}_w2v_embed_dim_{embed_dim}_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}_epochs_{w2v_epochs}.npy", word_embeddings)


# ============================================================================================
# Similarity checking and visualizing 
# ============================================================================================
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# Load saved embeddings
word_embeddings = np.load(f"mdp/modified_{modified}_w2v_embed_dim_{embed_dim}_{env_name}_map_size_{env_dim}_stochastic_{stochastic}_seed_{seed}_epochs_{w2v_epochs}.npy", 
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


# # To obtain trained vectors:
# model.get_word_vector(word_idx=)