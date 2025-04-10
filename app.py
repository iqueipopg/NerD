import streamlit as st
import torch
from PIL import Image
from gensim.models import KeyedVectors
from src.models import BiLSTMTagger
from src.embeddings import get_embedding
from src.image_captioning import generate_local_caption
import pickle
import os

# Asegúrate de que esta línea esté antes de cualquier otra función de Streamlit
st.set_page_config(page_title="NLP + Imagen", layout="centered")

# Configurar dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 300


# Cargar modelos con caché
@st.cache_resource
def load_models():
    w2v_model = KeyedVectors.load_word2vec_format(
        "data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True
    )
    with open("models/tag2idx.pkl", "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = {v: k for k, v in tag2idx.items()}

    model = BiLSTMTagger(
        embedding_dim=EMBEDDING_DIM, hidden_dim=128, ner_num_classes=len(tag2idx)
    )
    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return w2v_model, model, idx2tag


w2v_model, model, idx2tag = load_models()


# Predicción
def predict_from_text(text):
    words = text.strip().split()
    embeddings = [get_embedding(word, w2v_model) for word in words]
    input_tensor = torch.stack(embeddings).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(words)]).to(DEVICE)

    with torch.no_grad():
        ner_logits, sa_logits = model(input_tensor, lengths)
        ner_preds = torch.argmax(ner_logits, dim=-1)[0][: lengths.item()].cpu().tolist()
        sentiment = "Positivo" if sa_logits.item() > 0.5 else "Negativo"

    ner_labels = [idx2tag[i] for i in ner_preds]
    return list(zip(words, ner_labels)), sentiment


# Interfaz de Streamlit
st.title("🧠 Análisis de Texto + Imagen")
st.markdown(
    "Escribe una frase y/o sube una imagen para obtener entidades nombradas y análisis de sentimiento."
)

# Entrada de texto y subida de imagen
text_input = st.text_area("✍️ Escribe tu frase aquí:")
uploaded_image = st.file_uploader(
    "📷 Sube una imagen (opcional):", type=["jpg", "jpeg", "png"]
)

if st.button("🔍 Analizar"):
    final_text = text_input
    if uploaded_image:
        # Convertir la imagen cargada a un formato PIL Image
        image = Image.open(uploaded_image)

        # Ahora pasamos la imagen directamente a generate_local_caption
        caption = generate_local_caption(
            image
        )  # Debes asegurarte que esta función acepte objetos PIL.Image
        if not caption:  # Validar si la caption está vacía
            st.warning("⚠️ No se pudo generar una descripción para la imagen.")
        else:
            st.image(image, caption="Imagen subida", use_container_width=True)
            st.write(f"📝 Texto generado por imagen: *{caption}*")
        final_text += " " + (caption if caption else "")

    if final_text.strip():
        ner_results, sentiment = predict_from_text(final_text)
        st.subheader("🧠 Resultado NER")
        st.markdown(" ".join([f"**{w}** [`{t}`]" for w, t in ner_results]))

        st.subheader("❤️ Sentimiento")
        st.markdown(f"**{sentiment}**")
    else:
        st.warning("⚠️ No se pudo procesar el input.")
