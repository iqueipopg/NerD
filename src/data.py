import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Determinar si hay una GPU disponible y configurar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo preentrenado de Sentiment Analysis (por ejemplo, un modelo que tenga 5 clases)
tokenizer = BertTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
model = BertForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
)

# Mover el modelo a la GPU (si está disponible)
model.to(device)


# Función para predecir el sentimiento de una frase
def predict_sentiment(text):
    # Tokenizar la entrada
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Mover los tensores al dispositivo (GPU si está disponible)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Hacer la predicción sin calcular los gradientes (modo de evaluación)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Predicción original del modelo (5 clases)
        prediction = torch.argmax(logits, dim=1).item()

    # Mapeamos las 5 clases originales a 3 clases personalizadas
    if prediction in [0, 1]:  # Muy negativa o negativa
        return 0
    elif prediction == 2:  # Neutral
        return 1
    else:  # Muy positiva o positiva
        return 2


def add_sentiment_column(input_tsv, output_tsv):
    # Leer el archivo TSV original con la codificación adecuada
    with open(input_tsv, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentence = []
    sentences = []
    ner_labels = []
    ner_labels_sentence = []  # Inicializar la variable aquí

    # Procesamos las líneas del archivo
    for line in lines:
        # Si encontramos una línea vacía, significa que una oración ha terminado
        if line.strip() == "":
            sentences.append(" ".join(sentence))
            ner_labels.append(ner_labels_sentence)
            sentence = []
            ner_labels_sentence = []
        else:
            word, ner_label = line.strip().split("\t")
            sentence.append(word)
            ner_labels_sentence.append(ner_label)

    # Procesamos las últimas frases
    if sentence:
        sentences.append(" ".join(sentence))
        ner_labels.append(ner_labels_sentence)

    # Lista donde almacenaremos las filas del nuevo archivo
    new_rows = []

    # Predecir el sentimiento para cada oración y añadir la columna
    for sentence, ner_labels_sentence in zip(sentences, ner_labels):
        sentiment = predict_sentiment(sentence)  # Obtener el sentimiento de la oración
        for word, ner_label in zip(sentence.split(), ner_labels_sentence):
            new_rows.append([word, ner_label, str(sentiment)])
        new_rows.append([])  # Línea vacía para separar oraciones

    # Guardamos el archivo TSV con la nueva columna de sentimiento
    with open(output_tsv, "w", encoding="utf-8") as f:
        for row in new_rows:
            if row:
                f.write("\t".join(row) + "\n")
            else:
                f.write("\n")

    print(f"Archivo con Sentiment Analysis generado: {output_tsv}")


# Llamada a la función, modificando el nombre del archivo de entrada y salida
input_file = "C:/Users/Stealth/OneDrive - Universidad Pontificia Comillas/Documentos/Uni/3º iMAT/CUATRI_2/NLP/NerD/data/train_data.tsv"
# Cambia esto por la ruta de tu archivo de entrada
output_file = (
    "output_with_sa_train.tsv"  # Cambia esto por la ruta de salida que prefieras
)
add_sentiment_column(input_file, output_file)
