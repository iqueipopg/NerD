![NerD Banner](images/banner.png)

# NerD

**NerD** is an NLP system that turns unstructured news articles and social media posts into structured output: named entities and a sentiment assessment. A single multitask model performs token-level Named Entity Recognition (NER) and sentence-level Sentiment Analysis (SA) in one pass, and an integrated language model converts the structured result into a human-readable reputation alert.

Developed for the Deep Learning and NLP course at Universidad Pontificia Comillas (ICAI), Engineering Mathematics (iMAT) program.

---

## Overview

Given a raw text post, and optionally a related image, NerD produces:

1. **Named entities** (PERSON, ORG, LOC) detected at token level
2. **Sentiment** (positive or negative) assessed at sentence level
3. **A reputation alert**: a short, structured summary generated from the entities and sentiment

If an image is provided, a pretrained captioning model describes it and the caption is merged into the text before analysis, so both modalities inform the same prediction.

## How it works

### Multitask model

- A custom **BiLSTMTagger** (PyTorch) with a shared BiLSTM encoder and two heads:
  - a token-level classifier for NER (trained with cross-entropy loss)
  - a sentence-level classifier for SA (trained with binary cross-entropy loss)
- Inputs are pretrained **word2vec** embeddings (GoogleNews, 300 dimensions), loaded with Gensim.
- Model selection is based on validation performance.

### Image captioning

- A pretrained **BLIP** model (Salesforce/blip-image-captioning-base, via Hugging Face Transformers) generates a caption for the uploaded image.
- The caption is appended to the user text and analyzed jointly.

### Alert generation

- The **DeepSeek-R1-Distill-Qwen-1.5B** language model receives the original text, the image caption, the detected entities and the overall sentiment, and generates an alert such as:

  > `REPUTATION ALERT: CEO - negative. Summary: Public backlash over controversial remarks.`

### Data

- Reformatted and preprocessed **MultiNERD** plus additional datasets.
- Sentiment labels inferred with a pretrained model.
- Final dataset: 40,000 training samples and 10,000 test samples with a balanced sentiment distribution.

## Results

- Strong NER performance on the key entity types PERSON, ORG and LOC.
- Sentiment analysis reaches 84 percent validation accuracy.
- The end-to-end alert generation pipeline runs inside the app.

## How to run

Prerequisites: Python 3.8+, the pretrained word2vec model (`GoogleNews-vectors-negative300`), and the trained weights (`best_model.pt` and `tag2idx.pkl`) in `models/`.

```bash
git clone https://github.com/iqueipopg/NerD.git
cd NerD
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The Gradio interface lets you paste a text post, optionally upload an image, and click **Analyze** to get entities, sentiment and the generated alert.

## Project structure

```
NerD/
├── app.py                  # Gradio web interface
├── requirements.txt        # Python dependencies
├── data/                   # Datasets (train.tsv, test.tsv, MultiNERD)
├── models/                 # Trained weights (best_model.pt, tag2idx.pkl)
├── images/                 # Banner and figures
├── docs/                   # Reports and final paper
└── src/
    ├── models.py           # BiLSTMTagger (shared encoder, NER + SA heads)
    ├── train.py            # Training loop
    ├── evaluate.py         # Evaluation metrics
    ├── data.py             # Dataset loading
    ├── embeddings.py       # word2vec embedding lookup
    ├── image_captioning.py # BLIP caption generation
    ├── alert_generation.py # LLM-based alert generation
    └── tsv_preprocessing.py
```

## Technologies

- **PyTorch**: multitask BiLSTM model
- **Hugging Face Transformers**: image captioning (BLIP) and alert generation (DeepSeek-R1-Distill-Qwen-1.5B)
- **Gensim**: pretrained word2vec embeddings
- **Gradio**: web interface
- **Scikit-learn**: evaluation metrics

## Credits

Developed by **Beltrán Sánchez Careaga**, **Eugenio Ribón Novoa**, **Jorge Kindelan Navarro** and **Ignacio Queipo de Llano Pérez-Gascón**.

Thanks to our professors **Jaime Pizarroso Gonzalo** and **Andrés Occhipinti Liberman** for their guidance throughout the course.

---

This kind of unstructured-to-structured processing of news and social media text is the same building block used in news analytics and alternative data workflows.
