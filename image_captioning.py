# image_captioning.py

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from gtts import gTTS
import os

# Load pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths, lang='en', voice=None):
    images = []
    for image_path in image_paths:
        try:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return []

    encoding = image_processor(images=images, return_tensors="pt")

    # Create attention mask
    attention_mask = torch.ones(encoding.pixel_values.shape[0], encoding.pixel_values.shape[1], encoding.pixel_values.shape[2], dtype=torch.long)
    encoding["attention_mask"] = attention_mask

    encoding = {k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in encoding.items()}

    output_ids = model.generate(encoding["pixel_values"], attention_mask=encoding["attention_mask"], **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    # Convert captions to speech using gTTS
    for i, caption in enumerate(preds):
        tts = gTTS(text=caption, lang=lang)
        if voice:
            tts.voice = voice
        tts.save(f"caption_{i}.mp3")

    return preds

# Test the model with an image
image_paths = ['sample2.jpg']
lang = 'en'  
voice = 'en-1' 
captions = predict_step(image_paths, lang, voice)
print(captions)