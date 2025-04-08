import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from gensim.models.keyedvectors import load_word2vec_format

from src.data import NLPSentenceDataset, collate_fn, load_data, create_tag_vocab
from src.models import BiLSTMTagger


def train_model(model, train_loader, val_loader, tag2idx, num_epochs=5, device="cuda"):
    """
    Args:
        model (nn.Module): The model to be trained, which should be capable of performing both NER and sentiment analysis tasks.
        train_loader (DataLoader): A PyTorch DataLoader for training data, where each batch contains input sequences, NER labels, sentiment labels, and sequence lengths.
        val_loader (DataLoader): A PyTorch DataLoader for validation data, where each batch contains input sequences, NER labels, sentiment labels, and sequence lengths.
        tag2idx (dict): A dictionary mapping NER tag labels to integer indices, used for computing NER loss.
        num_epochs (int, optional): The number of epochs to train the model. Default is 5.
        device (str, optional): The device to run the model on, either "cuda" for GPU or "cpu" for CPU. Default is "cuda".

    Returns:
        None: The function trains the model and prints the loss for both NER and sentiment analysis for each epoch.
        Additionally, it saves the best-performing model based on validation loss to a file called "best_model.pt" in the "models" directory.

    Description:
        This function trains the model using both NER and Sentiment Analysis tasks. For each epoch, it:
        - Runs the model on the training data and computes the combined loss (NER loss + sentiment analysis loss).
        - Prints the NER and sentiment analysis losses for the training phase.
        - Evaluates the model on the validation data and prints the NER and sentiment analysis losses for the validation phase.
        - Saves the model's state_dict if it achieves the lowest validation loss.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_ner = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_sa = nn.BCELoss()
    best_val_loss = float("inf")

    os.makedirs("models", exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_ner_loss = 0
        total_sa_loss = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            x, ner_labels, sentiments, lengths = batch
            x = x.to(device)
            ner_labels = ner_labels.to(device)
            sentiments = sentiments.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            ner_logits, sa_logits = model(x, lengths)

            ner_loss = criterion_ner(
                ner_logits.view(-1, len(tag2idx)), ner_labels.view(-1)
            )
            sa_loss = criterion_sa(sa_logits, sentiments)

            loss = ner_loss + sa_loss
            loss.backward()
            optimizer.step()

            total_ner_loss += ner_loss.item()
            total_sa_loss += sa_loss.item()

        print(
            f"Epoch {epoch+1} [Train] NER Loss = {total_ner_loss:.4f}, SA Loss = {total_sa_loss:.4f}"
        )

        # VALIDACIÓN
        model.eval()
        val_ner_loss = 0
        val_sa_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                x, ner_labels, sentiments, lengths = batch
                x = x.to(device)
                ner_labels = ner_labels.to(device)
                sentiments = sentiments.to(device)
                lengths = lengths.to(device)

                ner_logits, sa_logits = model(x, lengths)
                ner_loss = criterion_ner(
                    ner_logits.view(-1, len(tag2idx)), ner_labels.view(-1)
                )
                sa_loss = criterion_sa(sa_logits, sentiments)

                val_ner_loss += ner_loss.item()
                val_sa_loss += sa_loss.item()

        val_total_loss = val_ner_loss + val_sa_loss
        print(
            f"Epoch {epoch+1} [Val]   NER Loss = {val_ner_loss:.4f}, SA Loss = {val_sa_loss:.4f}"
        )

        # GUARDAR MEJOR MODELO
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), "models/best_model.pt")
            print(
                f"✅ Saved best model at epoch {epoch+1} (Val Loss = {val_total_loss:.4f})"
            )


if __name__ == "__main__":
    # Cargar embeddings
    w2v_model = load_word2vec_format(
        "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True
    )
    print("Word2Vec model loaded.")

    # Preparar datos
    sentences, ner_tags, sentiments = load_data("data/train.tsv")
    tag2idx, idx2tag = create_tag_vocab(ner_tags)
    import pickle

    # Después de crear el vocabulario
    tag2idx, idx2tag = create_tag_vocab(ner_tags)

    # Guardar vocabulario
    with open("models/tag2idx.pkl", "wb") as f:
        pickle.dump(tag2idx, f)
    print("Vocabulario guardado.")

    # Dataset completo
    full_dataset = NLPSentenceDataset(sentences, ner_tags, sentiments)

    # Dividir en train y val (80/20)
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, w2v_model, tag2idx, 300),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, w2v_model, tag2idx, 300),
    )

    print("DataLoaders ready. Starting training...")

    # Inicializar modelo
    model = BiLSTMTagger(
        embedding_dim=300, hidden_dim=128, ner_num_classes=len(tag2idx)
    )

    # Entrenar
    train_model(model, train_loader, val_loader, tag2idx)
