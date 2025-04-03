import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Determine if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Sentiment Analysis model (5-class classification)
tokenizer = BertTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
model = BertForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
)

# Move the model to GPU (if available)
model.to(device)


def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using a pre-trained BERT model.

    Args:
        text (str): The input sentence to analyze.

    Returns:
        int: The sentiment class mapped to:
            - 0: Negative (includes very negative and negative)
            - 1: Neutral
            - 2: Positive (includes very positive and positive)
    """
    # Tokenize the input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Move tensors to the available device (GPU if available)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference without gradient computation (evaluation mode)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Map the 5 original classes to 3 custom sentiment classes
    if prediction in [0, 1]:  # Very negative or negative
        return 0
    elif prediction == 2:  # Neutral
        return 1
    else:  # Very positive or positive
        return 2


def add_sentiment_column(input_tsv, output_tsv):
    """
    Reads a TSV file containing words and NER labels, predicts the sentiment
    for each sentence, and saves a new TSV file with an additional sentiment column.

    Args:
        input_tsv (str): Path to the input TSV file.
        output_tsv (str): Path to save the processed TSV file.

    Returns:
        None. A new TSV file is saved with three columns: word, NER label, sentiment label.
    """
    # Read the original TSV file
    with open(input_tsv, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentence = []
    sentences = []
    ner_labels = []
    ner_labels_sentence = []  # Initialize the variable here

    # Process the lines of the file
    for line in lines:
        # If we find an empty line, it means a sentence has ended
        if line.strip() == "":
            sentences.append(" ".join(sentence))
            ner_labels.append(ner_labels_sentence)
            sentence = []
            ner_labels_sentence = []
        else:
            word, ner_label = line.strip().split("\t")
            sentence.append(word)
            ner_labels_sentence.append(ner_label)

    # Process the last sentences
    if sentence:
        sentences.append(" ".join(sentence))
        ner_labels.append(ner_labels_sentence)

    # List to store the rows of the new file
    new_rows = []

    # Predict sentiment for each sentence and add the column
    for sentence, ner_labels_sentence in zip(sentences, ner_labels):
        sentiment = predict_sentiment(sentence)  # Get sentiment of the sentence
        for word, ner_label in zip(sentence.split(), ner_labels_sentence):
            new_rows.append([word, ner_label, str(sentiment)])
        new_rows.append([])  # Empty line to separate sentences

    # Save the TSV file with the new sentiment column
    with open(output_tsv, "w", encoding="utf-8") as f:
        for row in new_rows:
            if row:
                f.write("\t".join(row) + "\n")
            else:
                f.write("\n")

    print(f"TSV file with Sentiment Analysis saved: {output_tsv}")


def load_clean_conll(filepath):
    """
    Reads a CoNLL-formatted file, extracts words and NER labels, and saves
    them in a clean TSV format.

    Args:
        filepath (str): Path to the CoNLL file.

    Returns:
        None. A TSV file is saved as 'test_data.tsv' with two columns: word and NER label,
        separating sentences with blank lines.
    """
    data = []
    sentence = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:  # New sentence
                if sentence:
                    data.append(sentence)
                    sentence = []
            else:
                tokens = line.split()
                word, ner_label = tokens[1:3]  # Only take the first 2 columns
                sentence.append((word, ner_label))

    # Generate the lines of the TSV file
    tsv_lines = []

    # Iterate through each sentence in the dataset
    for sentence in data:
        # Iterate through each word and its NER label
        for word, label in sentence:
            # Write the word and its NER label in the correct format
            tsv_lines.append(f"{word}\t{label}")
        # Add an empty line between sentences
        tsv_lines.append("")

    # Save the TSV file
    with open("test_data.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(tsv_lines))
