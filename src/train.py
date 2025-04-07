import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from evaluate import evaluate
from models import BiLSTMTagger
from embeddings import load_glove_embeddings


def train(model, train_loader, valid_loader, epochs=10, lr=0.001, device="cuda"):
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    ner_loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=model.ner2idx["O"]
    )  # Ignore 'O'
    sa_loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_ner_correct = 0
        total_sa_correct = 0
        total_tokens = 0

        for batch_idx, (words, ner_labels, sa_labels) in enumerate(train_loader):
            words = words.to(device)
            ner_labels = ner_labels.to(device)
            sa_labels = sa_labels.to(device).float()

            # Forward pass
            ner_logits, sa_logits = model(words, torch.sum(words != 0, dim=1))

            # Compute losses
            ner_loss = ner_loss_fn(
                ner_logits.view(-1, ner_logits.shape[-1]), ner_labels.view(-1)
            )
            sa_loss = sa_loss_fn(sa_logits.squeeze(), sa_labels)

            # Total loss
            loss = ner_loss + sa_loss
            total_loss += loss.item()

            # Accuracy (NER)
            ner_preds = torch.argmax(ner_logits, dim=-1)
            total_ner_correct += (ner_preds == ner_labels).sum().item()
            total_tokens += ner_labels.numel()

            # Accuracy (SA)
            sa_preds = (torch.sigmoid(sa_logits) > 0.5).float()
            total_sa_correct += (sa_preds == sa_labels).sum().item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training stats
        train_loss = total_loss / len(train_loader)
        train_ner_acc = total_ner_correct / total_tokens
        train_sa_acc = total_sa_correct / len(train_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train NER Accuracy: {train_ner_acc:.4f}")
        print(f"  Train SA Accuracy: {train_sa_acc:.4f}")

        # Validation
        val_loss, val_ner_acc, val_sa_acc = evaluate(model, valid_loader, device)
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation NER Accuracy: {val_ner_acc:.4f}")
        print(f"  Validation SA Accuracy: {val_sa_acc:.4f}")


if __name__ == "__main__":
    # Assuming you have already defined your model, train_loader, and valid_loader

    train(model, train_loader, valid_loader, epochs=10, lr=0.001, device="cuda")
