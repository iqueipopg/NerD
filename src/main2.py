import os
import torch
from gensim.models import KeyedVectors
from src.models import BiLSTMTagger
from src.data import collate_fn
from src.embeddings import get_embedding
from transformers import logging
from src.image_captioning import generate_local_caption  # <-- Asegúrate del path

# Evitar warnings molestos
logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 300

# ==== CARGA DE MODELOS ====
print("🔧 Cargando modelos...")

# Word2Vec
w2v_model = KeyedVectors.load_word2vec_format(
    "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True
)

# tag2idx
import pickle

with open("models/tag2idx.pkl", "rb") as f:
    tag2idx = pickle.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

# Modelo BiLSTM
model = BiLSTMTagger(
    embedding_dim=EMBEDDING_DIM, hidden_dim=128, ner_num_classes=len(tag2idx)
)
model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Modelos cargados.")


# ==== PREDICCIÓN ====
def predict_from_text(text):
    words = text.strip().split()
    embeddings = [get_embedding(word, w2v_model) for word in words]
    input_tensor = torch.stack(embeddings).unsqueeze(0).to(DEVICE)  # (1, seq_len, 300)
    lengths = torch.tensor([len(words)]).to(DEVICE)

    with torch.no_grad():
        ner_logits, sa_logits = model(input_tensor, lengths)
        ner_preds = torch.argmax(ner_logits, dim=-1)[0][: lengths.item()].cpu().tolist()
        sentiment = "Positivo" if sa_logits.item() > 0.5 else "Negativo"

    ner_labels = [idx2tag[i] for i in ner_preds]
    print("\n🧠 Resultado:")
    for word, label in zip(words, ner_labels):
        print(f"{word} [{label}]", end=" ")
    print(f"\n❤️ Sentimiento: {sentiment}\n")


# ==== LOOP INTERACTIVO ====
if __name__ == "__main__":
    print(
        "🗨️  Escribe una frase. También puedes añadir una imagen opcional para enriquecer el análisis."
    )
    print("🛑 Escribe 'exit' para salir.\n")

    while True:
        text_input = input("📝 Frase: ").strip()
        if text_input.lower() in {"exit", "quit"}:
            break

        use_image = input("🖼 ¿Quieres añadir una imagen? (s/n): ").strip().lower()

        final_text = text_input

        if use_image == "s":
            image_path = input("📁 Ruta de imagen (.jpg/.png): ").strip()
            if os.path.isfile(image_path) and image_path.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                print("🔍 Procesando imagen...")
                caption = generate_local_caption(image_path)
                print(f"📝 Texto generado por imagen: '{caption}'")
                final_text += " " + caption
            else:
                print("⚠️ Imagen no válida. Se usará solo el texto.")

        if final_text.strip():
            predict_from_text(final_text)
        else:
            print("⚠️ No se pudo procesar el input.\n")
