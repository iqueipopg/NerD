import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if word in word2idx:
                embeddings[word2idx[word]] = vec

    return embeddings


def build_vocab_and_labels(tsv_file, ner_labels):
    word_freq = defaultdict(int)
    sentences = []
    ner_tags = []
    sa_tags = []

    with open(tsv_file, "r", encoding="utf-8") as f:
        sentence = []
        ner_sentence = []
        sa_label = None

        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    ner_tags.append(ner_sentence)
                    sa_tags.append(int(sa_label))
                    sentence, ner_sentence, sa_label = [], [], None
                continue

            word, ner, sa = line.strip().split("\t")
            word_freq[word.lower()] += 1
            sentence.append(word.lower())
            ner_sentence.append(ner)
            if sa_label is None:
                sa_label = sa

    vocab = ["<PAD>", "<UNK>"] + sorted(word_freq.keys())
    ner2idx = {label: idx for idx, label in enumerate(sorted(ner_labels))}
    idx2ner = {idx: label for label, idx in ner2idx.items()}

    word2idx = {word: idx for idx, word in enumerate(vocab)}

    return vocab, word2idx, ner2idx, idx2ner, sentences, ner_tags, sa_tags


class NER_SA_Dataset(Dataset):
    def __init__(self, sentences, ner_tags, sa_tags, word2idx, ner2idx, max_len=50):
        self.sentences = sentences
        self.ner_tags = ner_tags
        self.sa_tags = sa_tags
        self.word2idx = word2idx
        self.ner2idx = ner2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        ner = self.ner_tags[idx]
        sa = self.sa_tags[idx]

        words_idx = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        ner_idx = [self.ner2idx[n] for n in ner]

        # Padding
        pad_len = self.max_len - len(words_idx)
        if pad_len > 0:
            words_idx += [self.word2idx["<PAD>"]] * pad_len
            ner_idx += [self.ner2idx["O"]] * pad_len
        else:
            words_idx = words_idx[: self.max_len]
            ner_idx = ner_idx[: self.max_len]

        return torch.tensor(words_idx), torch.tensor(ner_idx), torch.tensor(sa)
