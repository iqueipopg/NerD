import gradio as gr
import torch
from gensim.models import KeyedVectors
from src.models import BiLSTMTagger
from src.embeddings import get_embedding
from src.image_captioning import generate_local_caption
import pickle
from src.alert_generation import generate_alert

# Cargar modelos con caché
w2v_model = KeyedVectors.load_word2vec_format(
    "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True
)
with open("models/tag2idx.pkl", "rb") as f:
    tag2idx = pickle.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

model = BiLSTMTagger(embedding_dim=300, hidden_dim=128, ner_num_classes=len(tag2idx))
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
model.eval()


# Función principal que genera alerta
def analyze_input(image, text):
    """
    Analyzes a combination of image and text input to detect named entities, determine sentiment, and generate a reputation alert.

    Args:
        image (Image.Image or None): A PIL Image object to be analyzed, or None if no image is provided.
        text (str): The input text to be analyzed.

    Returns:
        str: A generated reputation alert message based on the extracted entities, sentiment, and input data.
    """
    final_text = text
    caption = None

    if image:
        caption = generate_local_caption(image)
        final_text += " " + (caption if caption else "")

    if not final_text.strip():
        return "⚠️ Could not process the input."

    words = final_text.strip().split()
    embeddings = [get_embedding(word, w2v_model) for word in words]
    input_tensor = torch.stack(embeddings).unsqueeze(0)
    lengths = torch.tensor([len(words)])

    with torch.no_grad():
        ner_logits, sa_logits = model(input_tensor, lengths)
        ner_preds = torch.argmax(ner_logits, dim=-1)[0][: lengths.item()].cpu().tolist()
        sentiment_label = "positive" if sa_logits.item() > 0.5 else "negative"

    ner_labels = [idx2tag[i] for i in ner_preds]

    # Agrupar entidades
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

    instruction = f"""Generate a reputation alert in English using this format:
"REPUTATION ALERT: [MAIN_ENTITY] - [SENTIMENT]. Summary: [CONCISE_TEXT]"

Input data:
- Original text: "{text.strip()}"
- Image description: "{caption if caption else 'No image'}"
- Detected entities: {entities}
- Overall sentiment: {sentiment_label}"""

    input_data = {"instruction": instruction}
    alert = generate_alert(input_data)
    return alert


# Función que devuelve detalles (entidades, sentimiento, caption)
def get_details(image, text):
    """
    Analyzes a combination of image and text input to detect named entities, determine sentiment, and generate a detailed analysis.

    Args:
        image (Image.Image or None): A PIL Image object to be analyzed, or None if no image is provided.
        text (str): The input text to be analyzed.

    Returns:
        str: A detailed analysis message based on the extracted entities, sentiment, and input data.
    """
    final_text = text
    caption = None

    if image:
        caption = generate_local_caption(image)
        final_text += " " + (caption if caption else "")

    if not final_text.strip():
        return "⚠️ Could not process the input."

    words = final_text.strip().split()
    embeddings = [get_embedding(word, w2v_model) for word in words]
    input_tensor = torch.stack(embeddings).unsqueeze(0)
    lengths = torch.tensor([len(words)])

    with torch.no_grad():
        ner_logits, sa_logits = model(input_tensor, lengths)
        ner_preds = torch.argmax(ner_logits, dim=-1)[0][: lengths.item()].cpu().tolist()
        sentiment_label = "Positive" if sa_logits.item() > 0.5 else "Negative"

    ner_labels = [idx2tag[i] for i in ner_preds]

    entities = []
    current_entity = ""
    current_type = ""
    for word, tag in zip(words, ner_labels):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(f"{current_entity.strip()} ({current_type})")
            current_entity = word
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity += " " + word
        else:
            if current_entity:
                entities.append(f"{current_entity.strip()} ({current_type})")
                current_entity = ""
                current_type = ""

    if current_entity:
        entities.append(f"{current_entity.strip()} ({current_type})")

    details = (
        f"🧾 Detected Entities:\n- " + "\n- ".join(entities)
        if entities
        else "No named entities found."
    )
    details += f"\n\n💬 Sentiment: {sentiment_label}"
    details += f"\n\n🖼️ Image Caption: {caption if caption else 'No image or no caption generated.'}"
    return details


# Estilo personalizado
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

body {
    background-color: #F0F4FF;
    color: #1E3A8A;
    font-family: 'Poppins', sans-serif;
}

.gradio-container {
    background-color: #F0F4FF;
}

textarea, input[type='text'] {
    background-color: #E2E8F0 !important;
    color: #1E3A8A !important;
    border: 1px solid #60A5FA !important;
    border-radius: 12px !important;
}

button {
    background-color: #93C5FD !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: bold;
}

button:hover {
    background-color: #60A5FA !important;
}

h1 {
    font-family: 'Poppins', sans-serif;
    color: #1E3A8A;
    text-align: center;
    font-size: 48px;
    font-weight: 600;
    overflow: hidden;
    white-space: nowrap;
}

p {
    color: #1E3A8A;
    text-align: center;
}
"""

# Título y descripción
title = """
<div style='text-align: center;'>
    <h1 style='font-family: "Poppins", sans-serif;'>🧠 NerD</h1>
</div>
"""
description = "<p style='text-align: center;'>Upload an image, enter text, or both to get a reputation alert from our deep learning model.</p>"

# Interfaz Gradio
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        image_input = gr.Image(label="Upload an image (optional)", type="pil")
        text_input = gr.Textbox(
            label="Enter text", lines=4, placeholder="Write something..."
        )

    analyze_btn = gr.Button("Analyze")
    details_btn = gr.Button("Details", variant="secondary")

    alert_output = gr.Textbox(label="📢 Reputation Alert", lines=4)
    details_output = gr.Textbox(label="🔍 Analysis Details", lines=8)

    analyze_btn.click(
        fn=analyze_input,
        inputs=[image_input, text_input],
        outputs=alert_output,
    )

    details_btn.click(
        fn=get_details,
        inputs=[image_input, text_input],
        outputs=details_output,
    )

if __name__ == "__main__":
    demo.launch()
