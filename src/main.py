import torch
from gensim.models import KeyedVectors
from src.models import BiLSTMTagger
from src.data import create_tag_vocab
from src.embeddings import get_embedding, pad_embeddings
import pickle


def load_model_and_resources(model_path, tag2idx_path, embedding_path, device):
    with open(tag2idx_path, "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    print("Cargando embeddings...")
    w2v_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    print("Embeddings cargados.")

    print("Cargando modelo...")
    model = BiLSTMTagger(
        embedding_dim=300, hidden_dim=128, ner_num_classes=len(tag2idx)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Modelo cargado.")

    return model, w2v_model, tag2idx, idx2tag


def process_sentence(sentence, w2v_model, tag2idx, device):
    words = sentence.strip().split()
    embeddings = [get_embedding(word, w2v_model) for word in words]
    input_tensor = pad_embeddings([embeddings], len(words)).to(device)
    lengths_tensor = torch.tensor([len(words)]).to(device)
    return words, input_tensor, lengths_tensor


def predict(model, input_tensor, lengths_tensor):
    with torch.no_grad():
        ner_logits, sa_logits = model(input_tensor, lengths_tensor)
        ner_preds = torch.argmax(ner_logits, dim=-1).squeeze(0).cpu().tolist()
        sa_pred = (sa_logits > 0.5).long().item()
    return ner_preds, sa_pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rutas a modelo y recursos
    model_path = "models/best_model.pt"
    tag2idx_path = "models/tag2idx.pkl"
    embedding_path = "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz"

    # Cargar una sola vez
    model, w2v_model, tag2idx, idx2tag = load_model_and_resources(
        model_path, tag2idx_path, embedding_path, device
    )

    print(
        "\nEscribí frases para analizarlas (NER + Sentimiento). Escribí 'exit' para salir.\n"
    )

    while True:
        sentence = input("Frase > ").strip()
        if sentence.lower() in {"exit", "salir"}:
            print("👋 Hasta luego.")
            break
        if not sentence:
            continue

        words, input_tensor, lengths_tensor = process_sentence(
            sentence, w2v_model, tag2idx, device
        )
        ner_preds, sa_pred = predict(model, input_tensor, lengths_tensor)

        # Mostrar resultados
        print("\nSentimiento:", "Positivo" if sa_pred == 1 else "Negativo")
        print("Entidades NER:")
        for word, tag_idx in zip(words, ner_preds):
            tag = idx2tag.get(tag_idx, "O")
            print(f"  {word:15} -> {tag}")
        print("-" * 50)


if __name__ == "__main__":
    main()
