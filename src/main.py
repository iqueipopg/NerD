import os
import torch
from gensim.models import KeyedVectors
from src.models import BiLSTMTagger
from src.embeddings import get_embedding
from src.image_captioning import generate_local_caption
from src.alert_generation import generate_alert
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 300

# === LOAD MODELS ===
print("Loading models...")

# Word2Vec
w2v_model = KeyedVectors.load_word2vec_format(
    "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True
)

# tag2idx
with open("models/tag2idx.pkl", "rb") as f:
    tag2idx = pickle.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

# BiLSTM model
model = BiLSTMTagger(
    embedding_dim=EMBEDDING_DIM, hidden_dim=128, ner_num_classes=len(tag2idx)
)
model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Models loaded.")


# === PREDICTION + ALERT ===
def analyze_text_with_alert(text, image_path=None):
    final_text = text
    caption = None

    if image_path:
        caption = generate_local_caption(image_path)
        if caption:
            final_text += " " + caption

    if not final_text.strip():
        print("Input could not be processed.")
        return

    words = final_text.strip().split()
    embeddings = [get_embedding(word, w2v_model) for word in words]
    input_tensor = torch.stack(embeddings).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(words)]).to(DEVICE)

    with torch.no_grad():
        ner_logits, sa_logits = model(input_tensor, lengths)
        ner_preds = torch.argmax(ner_logits, dim=-1)[0][: lengths.item()].cpu().tolist()
        sentiment_label = "positive" if sa_logits.item() > 0.5 else "negative"

    ner_labels = [idx2tag[i] for i in ner_preds]

    entities = []
    current_entity = ""
    current_type = ""
    for word, tag in zip(words, ner_labels):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(
                    {"entity": current_entity.strip(), "type": current_type}
                )
            current_entity = word
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity += " " + word
        else:
            if current_entity:
                entities.append(
                    {"entity": current_entity.strip(), "type": current_type}
                )
                current_entity = ""
                current_type = ""

    if current_entity:
        entities.append({"entity": current_entity.strip(), "type": current_type})
    print(f"Detected entities: {entities}")
    print(f"Overall sentiment: {sentiment_label}")

    instruction = f"""Generate a reputation alert in English using this format:
"REPUTATION ALERT: [MAIN_ENTITY] - [SENTIMENT]. Summary: [CONCISE_TEXT]"

Input data:
- Original text: "{text.strip()}"
- Image description: "{caption if caption else 'No image'}"
- Detected entities: {entities}
- Overall sentiment: {sentiment_label}"""

    input_data = {"instruction": instruction}
    alert = generate_alert(input_data)

    print("\nGenerated Alert:")
    print(alert)


# === INTERACTIVE LOOP ===
if __name__ == "__main__":
    print("Enter a sentence. You may also provide an optional image.")
    print("Type 'exit' to quit.\n")

    while True:
        text_input = input("Sentence: ").strip()
        if text_input.lower() in {"exit", "quit"}:
            break

        use_image = input("Do you want to add an image? (y/n): ").strip().lower()
        image_path = None

        if use_image == "y":
            image_path = input("Image path (.jpg/.png): ").strip()
            if not os.path.isfile(image_path):
                print("Invalid image. Proceeding with text only.")
                image_path = None

        analyze_text_with_alert(text_input, image_path)
