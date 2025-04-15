import torch


def get_embedding(word, model, unk_token="UNK"):
    """
    Retrieves the embedding vector for a given word from a preloaded embedding model.

    Args:
        word (str): The word for which the embedding is requested.
        model (KeyedVectors): A preloaded word embedding model (e.g., Word2Vec).
        unk_token (str, optional): Token to use if the word is not found in the model. Defaults to "UNK".

    Returns:
        torch.Tensor: A 1D tensor representing the embedding of the word, or the embedding of the unknown token if the word is not in the model.
    """
    # Check if the word is in the model's vocabulary
    if word in model:
        return torch.tensor(model[word], dtype=torch.float32)
    elif unk_token in model:
        return torch.tensor(model[unk_token], dtype=torch.float32)


def pad_embeddings(sequences, lenght, embedding_dim=300, padding="post"):
    """
    Pads a batch of sequences of word embeddings to the same length.

    Args:
        sequences (list of list of torch.Tensor): Each inner list contains word embeddings for a sentence.
        embedding_dim (int): Dimension of the word embeddings.
        padding (str): 'post' to pad at the end, 'pre' to pad at the beginning.

    Returns:
        torch.Tensor: A 3D tensor of shape (batch_size, max_seq_len, embedding_dim).
    """
    max_len = lenght
    padded = []

    for seq in sequences:
        num_padding = max_len - len(seq)
        if padding == "post":
            pad = [torch.zeros(embedding_dim) for _ in range(num_padding)]
            new_seq = seq + pad
        elif padding == "pre":
            pad = [torch.zeros(embedding_dim) for _ in range(num_padding)]
            new_seq = pad + seq
        else:
            raise ValueError("padding must be 'post' or 'pre'")
        padded.append(torch.stack(new_seq))

    return torch.stack(padded)  # Shape: (batch_size, max_seq_len, embedding_dim)
