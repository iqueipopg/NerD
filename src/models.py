import torch
import torch.nn as nn


class BiLSTMTagger(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, ner_num_classes, freeze_embeddings=True
    ):
        super(BiLSTMTagger, self).__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.hidden_dim = hidden_dim
        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ner_num_classes),
        )

        self.sa_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths):
        """
        Args:
            x (batch_size, seq_len): input word indices
            lengths (batch_size): actual lengths of the sequences before padding
        Returns:
            ner_logits (batch_size, seq_len, ner_num_classes)
            sa_logits (batch_size, 1)
        """
        lengths_cpu = lengths.cpu().to(
            dtype=torch.int64
        )  # Ensure lengths are on the same device as x
        embedded = x  # (batch_size, seq_len, embedding_dim)

        # Pack for efficiency
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch_size, seq_len, hidden*2)

        ner_logits = self.ner_classifier(
            lstm_out
        )  # (batch_size, seq_len, ner_num_classes)

        # Sentence representation for SA: concatenate last hidden states from both directions
        sentence_rep = torch.cat(
            (hidden[0], hidden[1]), dim=1
        )  # (batch_size, hidden*2)
        sa_logits = self.sa_classifier(sentence_rep)  # (batch_size, 1)
        sa_logits = torch.sigmoid(sa_logits)  # binary output

        return ner_logits, sa_logits
