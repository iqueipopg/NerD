
![NerD Banner](images/banner.png)

# NerD 🧠📸

**NerD** is a multitask natural language processing system that performs **Named Entity Recognition (NER)** and **Sentiment Analysis (SA)** simultaneously, enriched by **Image Captioning**. Designed to process and analyze multimodal content (text + image), NerD extracts structured information and sentiment context from real-world data, such as news or social media posts.

This project was developed as part of a **Deep Learning and NLP** course at the **Universidad Pontificia Comillas, ICAI**, within the **Engineering Mathematics** program.

> 🎯 *From raw text and images to structured alerts — NerD empowers real-time reputation monitoring using deep learning.*

---

## 📜 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🛠️ Installation](#️-installation)
- [🚀 How to Use](#-how-to-use)
- [📂 Project Structure](#-project-structure)
- [🧠 Technologies Used](#-technologies-used)
- [🙌 Credits](#-credits)

## 📌 Project Overview

NerD combines multiple deep learning components to perform robust analysis of multimodal data, and now includes automated reputation alert generation powered by a large language model.

### 🔁 Multitask Learning Model
- A custom **BiLSTMTagger** model performs both:
  - **Token-level Named Entity Recognition (NER)**
  - **Sentence-level Sentiment Analysis (SA)**
- Shared **BiLSTM encoder** for feature extraction.
- Optimized using a combination of `CrossEntropyLoss` for NER and `BCELoss` for SA.
- Model selection based on validation performance.

### 🖼️ Image Captioning Integration
- A pretrained model from **Hugging Face** generates captions from input images.
- Captions are merged with user-provided text to improve entity recognition and sentiment prediction.

### 🤖 Reputation Alert Generation
- Fully integrated alert generation module using the **DeepSeek-R1-Distill-Qwen-1.5B** language model from Hugging Face.
- The model receives structured input containing:
  - Original text
  - Image description
  - Detected entities (via NER)
  - Overall sentiment
- Generates human-readable alerts like:
  > `REPUTATION ALERT: CEO - negative. Summary: Public backlash over controversial remarks.`
- Alerts are generated dynamically in the app UI with a single click.

### 📊 Dataset Curation
- Reformatted and preprocessed **MultiNERD** and additional datasets.
- Sentiment labels automatically inferred using a pretrained model.
- Final dataset:
  - 40,000 training samples
  - 10,000 testing samples
  - Balanced sentiment distribution (positive/negative)

### ✅ Preliminary Results
- Strong NER performance on key entity types: `PERSON`, `ORG`, `LOC`
- SA achieves **84% validation accuracy**
- End-to-end alert generation pipeline now **fully deployed and functional** within the app

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Pretrained word2vec model (`GoogleNews-vectors-negative300`)
- Pretrained model weights (`best_model.pt` and `tag2idx.pkl` in the `models/` directory)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/NerD.git
   cd NerD
   ```
   
2. Create a virtual environment and activate it (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   source venv/bin/activate # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python app.py
   ```

## 🚀 How to Use

The system exposes a simple web-based interface:

1. Upload a text post (e.g., news or social media content).
2. Upload a related image.
3. Click the **Analyze** button.

The app will:

- Generate a caption from the uploaded image (if provided).

- Analyze the full text + caption to detect entities and sentiment.

- Automatically generate a reputation alert using a large language model.

⚠️ **Make sure the pretrained models are present in the `models/` folder before running the app.**

## 📂 Project Structure

```plaintext
NerD/
│
├── app.py                  # Entry point for the web interface
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Datasets
│   ├── train.tsv, test.tsv
│   └── multinerd/, NLP_Data/, nsa/
│
├── models/                 # Pretrained models
│   ├── best_model.pt
│   ├── tag2idx.pkl
│   └── w2v_model.bin
│
├── images/                 # Project-related images
│   └── banner.jpg, logo.jpg
│
├── docs/                   # Project documents and reports
│
└── src/                    # Source code
    ├── data.py
    ├── embeddings.py
    ├── evaluate.py
    ├── alert_generation.py
    ├── image_captioning.py
    ├── main.py
    ├── models.py
    ├── train.py
    └── tsv_preprocessing.py
```


## 🧠 Technologies Used

### Frameworks & Libraries

- **PyTorch** – for building and training the BiLSTM multitask model  
- **Transformers** – Hugging Face library for image captioning  
- **Gensim** – for loading pretrained word embeddings  
- **Gradio** – for creating the interactive web interface  
- **Scikit-learn** – for evaluation metrics  

### NLP & ML Techniques

- **BiLSTM Encoder**  
- **Multitask Learning**  
- **NER with Sequence Tagging**  
- **Binary Sentiment Classification**  
- **Image-to-Text Captioning**
- **Instruction-tuned Text Generation (LLM prompts)**

## 🙌 Credits

This project was developed as part of the **Deep Learning + NLP** course at **Universidad Pontificia Comillas, ICAI**.

### Team Members

- **Beltrán Sánchez Careaga**
- **Eugenio Ribón Novoa**
- **Jorge Kindelan Navarro**
- **Ignacio Queipo de Llano Pérez-Gascón**

### Special Thanks To

- **Jaime Pizarroso Gonzalo** and **Andrés Occhipinti Liberman**, our professors, for their guidance and support throughout the course.  
- **Hugging Face** and **Gensim** communities for providing powerful pretrained models and tools.  
- **Open Source Contributors** whose libraries made this project possible.

