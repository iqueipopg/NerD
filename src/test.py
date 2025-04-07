import csv
import torch  # Importamos torch para verificar si la GPU está disponible
from transformers import pipeline

# Verificar si hay una GPU disponible y asignarla
device = 0 if torch.cuda.is_available() else -1

# Cargar el modelo en la GPU si está disponible
sentiment_analyzer = pipeline("sentiment-analysis", device=device)


def classify_sentiment(sentences):
    """Clasifica el sentimiento de una lista de frases y devuelve los resultados."""
    results = sentiment_analyzer(sentences)
    sentiment_map = {"NEGATIVE": 0, "POSITIVE": 1}
    return [sentiment_map.get(result["label"].upper(), 1) for result in results]


def add_sentiment_column(input_tsv, output_tsv):
    """
    Reads a TSV file containing words and NER labels, predicts the sentiment
    for each sentence, and saves a new TSV file with an additional sentiment column.
    """
    # Read the original TSV file
    with open(input_tsv, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentence = []
    sentences = []
    ner_labels = []
    ner_labels_sentence = []

    for line in lines:
        if line.strip() == "":
            sentences.append(" ".join(sentence))
            ner_labels.append(ner_labels_sentence)
            sentence = []
            ner_labels_sentence = []
        else:
            word, ner_label = line.strip().split("\t")
            sentence.append(word)
            ner_labels_sentence.append(ner_label)

    if sentence:
        sentences.append(" ".join(sentence))
        ner_labels.append(ner_labels_sentence)

    new_rows = []
    sentiments = classify_sentiment(
        sentences
    )  # Clasificar todas las oraciones a la vez

    for sentence, ner_labels_sentence, sentiment in zip(
        sentences, ner_labels, sentiments
    ):
        for word, ner_label in zip(sentence.split(), ner_labels_sentence):
            new_rows.append([word, ner_label, str(sentiment)])
        new_rows.append([])  # Empty line to separate sentences

    with open(output_tsv, "w", encoding="utf-8") as f:
        for row in new_rows:
            if row:
                f.write("\t".join(row) + "\n")
            else:
                f.write("\n")

    print(f"TSV file with Sentiment Analysis saved: {output_tsv}")
