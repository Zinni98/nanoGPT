import os
import numpy as np
import torch
import tiktoken
from torchtext.vocab import GloVe

emb_dim = 300
embedding_path = "/home/alessandro.zinni/emb_cache"
global_vectors = GloVe(name='840B', dim=emb_dim, cache=embedding_path)
enc = tiktoken.get_encoding("gpt2")
def get_embedding_matrix(embedding, embedding_dim, vocab_size):
    """
    Create an embedding matrix from a given embedding dictionary.

    Parameters
    ----------
    embedding : dict
        A dictionary mapping vocabulary items to their embeddings.
    embedding_dim : int
        The dimensionality of the embeddings.

    Returns
    -------
    torch.Tensor
      A matrix of size (len(self.vocab), embedding_dim) containing the embeddings for the vocabulary items.
    """
    matrix_length = vocab_size
    embedding_matrix = np.zeros((matrix_length, embedding_dim))
    # If I use torch.zeros directly it crashes (don't know why)
    embedding_matrix = torch.from_numpy(embedding_matrix.copy())
    null_embedding = torch.tensor([0.0]*embedding_dim)
    for idx in range(vocab_size):
      key = enc.decode([idx])
      if torch.equal(embedding[key], null_embedding):
        embedding_matrix[idx] = torch.randn(embedding_dim)
      else:
        embedding_matrix[idx] = embedding[key]
    return embedding_matrix


def main():
    emb_mat = get_embedding_matrix(global_vectors, emb_dim, 50257)
    torch.save(emb_mat, os.path.join(os.path.dirname(__file__), 'emb_mat.pt'))

if __name__ == "__main__":
    main()