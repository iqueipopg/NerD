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


def generate_local_caption(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            inputs = processor(img.convert("RGB"), return_tensors="pt").to(
                device, dtype
            )
            generated_ids = model.generate(**inputs)
            return processor.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return ""


if __name__ == "__main__":
    print(f"Image: Elon Musk, Caption: {generate_local_caption("data/elon.jpg")}")
    print(f"Image: Donald Trump, Caption: {generate_local_caption("data/trump.jpg")}")
