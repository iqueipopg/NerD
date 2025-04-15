import torch  # Importamos torch para verificar si la GPU está disponible
from transformers import pipeline

# Verificar si hay una GPU disponible y asignarla
device = 0 if torch.cuda.is_available() else -1

# Cargar el modelo en la GPU si está disponible
sentiment_analyzer = pipeline("sentiment-analysis", device=device)


def classify_sentiment(sentences):
    """
    Classifies the sentiment of a list of sentences using a preloaded sentiment analyzer.

    Args:
        sentences (list of str): List of sentences to classify.

    Returns:
        list of int: A list of sentiment labels, where 0 = negative, 1 = positive.
    """
    results = sentiment_analyzer(sentences)
    sentiment_map = {"NEGATIVE": 0, "POSITIVE": 1}
    return [sentiment_map.get(result["label"].upper(), 1) for result in results]


def add_sentiment_column(input_tsv, output_tsv):
    """
    Reads a TSV file containing words and NER labels, predicts the sentiment
    for each sentence, and saves a new TSV file with an additional sentiment column.

    Args:
        input_tsv (str): Path to the original TSV file (two columns: word, NER).
        output_tsv (str): Path where the new TSV with sentiment will be saved.

    Returns:
        None. A new TSV file is saved with three columns: word, NER, sentiment (0 or 1).
        Sentences are separated by blank lines.
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
    sentiments = classify_sentiment(sentences)

    for sentence_text, ner_labels_sentence, sentiment in zip(
        sentences, ner_labels, sentiments
    ):
        for word, ner_label in zip(sentence_text.split(), ner_labels_sentence):
            new_rows.append([word, ner_label, str(sentiment)])
        new_rows.append([])  # Blank line to separate sentences

    with open(output_tsv, "w", encoding="utf-8") as f:
        for row in new_rows:
            if row:
                f.write("\t".join(row) + "\n")
            else:
                f.write("\n")


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


def extract_balanced(input_file, output_file, max_per_class=5000):
    """
    Reads a TSV-formatted file with NER and sentiment annotations, extracts
    sentences with balanced sentiment labels (positive and negative), and
    saves them in a new file.

    Args:
        input_file (str): Path to the input TSV file.
        output_file (str): Path where the balanced dataset will be saved.
        max_per_class (int): Maximum number of sentences per sentiment class.

    Returns:
        None. A TSV file is saved with 2 * max_per_class sentences (balanced by sentiment),
        separating each sentence with a blank line.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    positives = []
    negatives = []
    current_sentence = []
    current_sa = None  # Sentiment Analysis label of the sentence

    for line in lines:
        if line.strip() == "":
            if current_sentence and current_sa is not None:
                if current_sa == "1" and len(positives) < max_per_class:
                    positives.append(current_sentence)
                elif current_sa == "0" and len(negatives) < max_per_class:
                    negatives.append(current_sentence)

                # Stop if both classes have reached the limit
                if len(positives) == max_per_class and len(negatives) == max_per_class:
                    break

            current_sentence = []
            current_sa = None
        else:
            current_sentence.append(line)
            parts = line.strip().split("\t")
            if len(parts) == 3:
                sa = parts[2]
                if current_sa is None:
                    current_sa = sa  # Take the sentiment from the first token

    # Write the selected sentences to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in positives + negatives:
            f.writelines(sentence)
            f.write("\n")  # Add blank line between sentences
