from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

np.zeros(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

dtype = torch.float16 if device == "cuda" else torch.float32
model = (
    BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    )
    .to(device)
    .eval()
)


def generate_local_caption(image: Image.Image) -> str:
    """
    Generates a descriptive caption for a given image using a pretrained image captioning model.

    Args:
        image (Image.Image): A PIL Image object for which the caption is to be generated.

    Returns:
        str: A string representing the generated caption for the input image. Returns an empty string if an error occurs during processing.
    """

    try:
        # Ahora procesamos directamente el objeto PIL.Image
        inputs = processor(image.convert("RGB"), return_tensors="pt").to(device, dtype)
        generated_ids = model.generate(**inputs)
        return processor.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error procesando la imagen: {str(e)}")
        return ""


if __name__ == "__main__":
    print(f"Image: Elon Musk, Caption: {generate_local_caption("data/elon.jpg")}")
    print(f"Image: Donald Trump, Caption: {generate_local_caption("data/trump.jpg")}")
