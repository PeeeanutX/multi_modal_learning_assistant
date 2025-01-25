import os
import logging
from typing import List, Dict
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageToTextConverter:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device="cuda" if torch.cuda.is_available() else "cpu"):
        logger.info(f"Loading image captioning model {model_name} on {device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def caption_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=50)
        caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption

def main():
    images_dir = "src/ingestion/data/raw/images"
    output_dir = "src/ingestion/data/processed/image_texts"
    os.makedirs(output_dir, exist_ok=True)

    converter = ImageToTextConverter()

    for fname in os.listdir(images_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(images_dir, fname)
            caption = converter.caption_image(image_path)
            base_name = os.path.splitext(fname)[0]
            out_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(out_path, "w", encoding='utf-8') as f:
                f.write(caption)

            logger.info(f"Converted {fname} to text: {caption}")

if __name__ == "__main__":
    main()
