import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, data_loader, device="cuda"):
    model.eval()
    total_loss = 0
    total_ner_correct = 0
    total_sa_correct = 0
    total_tokens = 0

    ner_loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=model.ner2idx["O"]
    )  # Ignore 'O'
    sa_loss_fn = torch.nn.BCEWithLogitsLoss()

    all_pred_ner = []
    all_true_ner = []
    all_pred_sa = []
    all_true_sa = []

    with torch.no_grad():
        for words, ner_labels, sa_labels in data_loader:
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

            # Storing predictions and labels for further analysis
            all_pred_ner.extend(ner_preds.cpu().numpy())
            all_true_ner.extend(ner_labels.cpu().numpy())
            all_pred_sa.extend(sa_preds.cpu().numpy())
            all_true_sa.extend(sa_labels.cpu().numpy())

    val_loss = total_loss / len(data_loader)
    val_ner_acc = total_ner_correct / total_tokens
    val_sa_acc = total_sa_correct / len(data_loader.dataset)

    # NER classification report
    ner_report = classification_report(
        all_true_ner, all_pred_ner, target_names=model.ner2idx.keys(), zero_division=0
    )

    print("NER Classification Report:\n", ner_report)

    return val_loss, val_ner_acc, val_sa_acc
