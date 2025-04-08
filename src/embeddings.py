from gensim.models import KeyedVectors
import numpy as np
import torch
from torch.nn import Embedding


class Word2VecEmbeddings:
    def __init__(self, path_to_bin, unk_token="<UNK>", pad_token="<PAD>"):
        print("Loading Word2Vec model...")
        self.word2vec = KeyedVectors.load_word2vec_format(path_to_bin, binary=True)
        self.embedding_dim = self.word2vec.vector_size
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.word2idx = {}
        self.idx2word = []
        self.embeddings_matrix = None

        self._build_vocab()

    def _build_vocab(self):
        vocab = [self.pad_token, self.unk_token] + list(
            self.word2vec.key_to_index.keys()
        )
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = vocab

        # Init embedding matrix
        vocab_size = len(vocab)
        self.embeddings_matrix = np.zeros(
            (vocab_size, self.embedding_dim), dtype=np.float32
        )
        self.embeddings_matrix[self.word2idx[self.unk_token]] = np.random.normal(
            size=self.embedding_dim
        )

        for word in self.word2vec.key_to_index:
            idx = self.word2idx[word]
            self.embeddings_matrix[idx] = self.word2vec[word]

    def get_embedding_layer(self, freeze=True):
        weights = torch.tensor(self.embeddings_matrix)
        embedding_layer = Embedding.from_pretrained(
            weights, freeze=freeze, padding_idx=self.word2idx[self.pad_token]
        )
        return embedding_layer

    def words_to_indices(self, words):
        return [
            self.word2idx.get(word.lower(), self.word2idx[self.unk_token])
            for word in words
        ]
