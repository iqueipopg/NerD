from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.embeddings import get_embedding, pad_embeddings
import torch


class NLPSentenceDataset(Dataset):
    def __init__(self, sentences, ner_tags, sentiments):
        self.sentences = sentences
        self.ner_tags = ner_tags
        self.sentiments = sentiments

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.ner_tags[idx], self.sentiments[idx]


def load_data(file_path):
    """
    This function loads a dataset from a file, where each line contains a word, its NER tag, and a sentiment label. Sentences are separated by empty lines.
    It processes the file to generate three lists:
    - `sentences`: A list where each element is a list of words in a sentence.
    - `ner_tags`: A list where each element is a list of NER tags corresponding to the words in a sentence.
    - `sentiments`: A list of sentiment labels, one per sentence, where all words in a sentence have the same sentiment.

    Args:
        file_path (str): The path to the file containing the dataset. The file should have sentences with each word, its NER tag, and its sentiment label, separated by spaces. Sentences are separated by empty lines.

    Returns:
        sentences (list of list of str): A list of sentences, where each sentence is a list of words.
        ner_tags (list of list of str): A list of NER tags corresponding to each word in each sentence.
        sentiments (list of int): A list of sentiment labels, where each element corresponds to the sentiment of the respective sentence.
    """
    sentences = []
    ner_tags = []
    sentiments = []

    with open(file_path, "r", encoding="utf-8") as f:
        words = []
        tags = []
        sentiment = None

        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    sentences.append(words)
                    ner_tags.append(tags)
                    sentiments.append(sentiment)
                    words, tags = [], []
                continue
            word, tag, sent = line.split()
            words.append(word)
            tags.append(tag)
            sentiment = int(sent)  # All words have same sentiment

        # Última frase si el archivo no termina en línea vacía
        if words:
            sentences.append(words)
            ner_tags.append(tags)
            sentiments.append(sentiment)

    return sentences, ner_tags, sentiments


def create_tag_vocab(tag_sequences):
    """
    This function creates two dictionaries from a list of NER tag sequences:
        - `tag2idx`: A dictionary that maps each unique NER tag to a unique integer index.
        - `idx2tag`: A dictionary that maps each integer index back to the corresponding NER tag.
        The tags are sorted alphabetically to ensure consistent indexing.

    Args:
        tag_sequences (list of list of str): A list of sequences, where each sequence is a list of NER tags for the words in a sentence.

    Returns:
        tag2idx (dict): A dictionary mapping NER tags (str) to unique integer indices (int).
        idx2tag (dict): A dictionary mapping integer indices (int) to NER tags (str).

    """

    tags = set(tag for seq in tag_sequences for tag in seq)
    tag2idx = {tag: i for i, tag in enumerate(sorted(tags))}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    return tag2idx, idx2tag


def collate_fn(batch, w2v_model, tag2idx, embedding_dim):
    """
    Args:
        batch (list of tuples): A batch of data, where each tuple contains a sentence (list of words), a list of NER tags, and a sentiment label for the sentence.
        w2v_model (gensim.models.KeyedVectors): The pre-trained Word2Vec model used to generate word embeddings.
        tag2idx (dict): A dictionary mapping NER tag labels (str) to unique integer indices (int).
        embedding_dim (int): The dimensionality of the word embeddings (size of the embedding vectors).

    Returns:
        padded_embeddings (torch.Tensor): A tensor of word embeddings, padded to the length of the longest sentence in the batch.
        padded_tags (torch.Tensor): A tensor of NER tag indices, padded to the length of the longest sentence in the batch, with padding value -100.
        sentiments_tensor (torch.Tensor): A tensor of sentiment labels for each sentence, shaped as (batch_size, 1).
        lengths_tensor (torch.Tensor): A tensor containing the lengths of each sentence in the batch.
    """

    sentences, ner_tags, sentiments = zip(*batch)

    embedded_batch = []
    ner_indices = []
    lengths = []

    for sentence, tags in zip(sentences, ner_tags):
        embeddings = [get_embedding(word, w2v_model) for word in sentence]
        tag_ids = [tag2idx[tag] for tag in tags]
        embedded_batch.append(embeddings)
        ner_indices.append(torch.tensor(tag_ids))
        lengths.append(len(sentence))

    padded_embeddings = pad_embeddings(embedded_batch, max(lengths), embedding_dim)
    padded_tags = torch.nn.utils.rnn.pad_sequence(
        ner_indices, batch_first=True, padding_value=-100
    )  # -100 para que CrossEntropy ignore

    sentiments_tensor = torch.tensor(sentiments).float().unsqueeze(1)  # (batch, 1)
    lengths_tensor = torch.tensor(lengths)

    return padded_embeddings, padded_tags, sentiments_tensor, lengths_tensor
