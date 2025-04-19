import os
import zipfile
import gdown

if not os.path.exists("data/NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz"):
    print("Downloading Word2Vec embeddings...")
    url = "https://drive.google.com/uc?id=1zQRH1zYBHJ_vU_uMkKvvvwQiZwP5N7wW"
    output = "NLP_DATA.zip"
    gdown.download(url, output, quiet=False)

    print("Extracting embeddings...")
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall("data")

    os.remove(output)
    print("Word2Vec embeddings ready.")
else:
    print("Word2Vec embeddings already present.")
