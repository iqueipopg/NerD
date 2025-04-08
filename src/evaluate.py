import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from gensim.models.keyedvectors import load_word2vec_format
from src.data import NLPSentenceDataset, collate_fn, load_data, create_tag_vocab
from src.models import BiLSTMTagger


def evaluate_model(model, dataloader, tag2idx, device="cuda"):
    """
    Args:
        model (nn.Module): The model to be evaluated, which should be capable of performing both NER and sentiment analysis tasks.
        dataloader (DataLoader): A PyTorch DataLoader providing batches of input data, where each batch contains input sequences, NER labels, sentiment labels, and sequence lengths.
        tag2idx (dict): A dictionary mapping NER tag labels to integer indices, used for converting NER predictions to human-readable labels.
        device (str, optional): The device to run the model on, either "cuda" for GPU or "cpu" for CPU. Default is "cuda".

    Returns:
        None: The function prints the evaluation results, including the NER classification report and sentiment analysis accuracy.

    Description:
        This function evaluates the performance of the given model on both Named Entity Recognition (NER) and Sentiment Analysis (SA) tasks.
        It runs the model on the provided dataloader, computes the predictions for both tasks, and prints the corresponding evaluation metrics.
        - NER predictions are evaluated with a classification report showing precision, recall, F1 score, and support.
        - Sentiment Analysis predictions are evaluated with accuracy score.
    """

    model.to(device)
    model.eval()

    all_ner_preds = []
    all_ner_labels = []
    all_sa_preds = []
    all_sa_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, ner_labels, sentiments, lengths = batch
            x = x.to(device)
            ner_labels = ner_labels.to(device)
            sentiments = sentiments.to(device)
            lengths = lengths.to(device)

            ner_logits, sa_logits = model(x, lengths)

            # NER predictions
            ner_preds = torch.argmax(ner_logits, dim=-1)  # (batch, seq_len)
            for pred, true, length in zip(ner_preds, ner_labels, lengths):
                pred = pred[:length].cpu().tolist()
                true = true[:length].cpu().tolist()
                all_ner_preds.extend(pred)
                all_ner_labels.extend(true)

            # SA predictions
            sa_pred = (sa_logits > 0.5).long().squeeze()  # binarize
            all_sa_preds.extend(sa_pred.cpu().tolist())
            all_sa_labels.extend(sentiments.cpu().tolist())

    idx2tag = {v: k for k, v in tag2idx.items()}
    ner_preds_labels = [idx2tag[idx] for idx in all_ner_preds]
    ner_true_labels = [idx2tag[idx] for idx in all_ner_labels]

    print("🔍 NER classification report:")
    print(
        classification_report(
            ner_true_labels, ner_preds_labels, digits=4, zero_division=0
        )
    )

    # SA accuracy
    sa_acc = accuracy_score(all_sa_labels, all_sa_preds)
    print(f"\n🧠 SA Accuracy: {sa_acc:.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar embeddings
    w2v_model = load_word2vec_format(
        "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True
    )
    print("Word2Vec model loaded.")

    # Cargar datos de test
    test_sentences, test_ner_tags, test_sentiments = load_data("data/test.tsv")
    # Cargar tag2idx usado en entrenamiento
    import pickle

    with open("models/tag2idx.pkl", "rb") as f:
        tag2idx = pickle.load(f)

    test_dataset = NLPSentenceDataset(test_sentences, test_ner_tags, test_sentiments)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, w2v_model, tag2idx, 300),
    )

    # Cargar modelo
    model = BiLSTMTagger(
        embedding_dim=300, hidden_dim=128, ner_num_classes=len(tag2idx)
    )
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))

    # Evaluar
    evaluate_model(model, test_loader, tag2idx, device=device)
