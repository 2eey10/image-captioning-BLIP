import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import numpy as np

nltk.download('punkt')

def calculate_bleu(reference_captions, generated_caption):
    reference_captions = [nltk.word_tokenize(caption.lower()) for caption in reference_captions]
    generated_caption = nltk.word_tokenize(generated_caption.lower())
    
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_captions, generated_caption, smoothing_function=smoothie)
    return score

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_path = "data/images/test1.jpg"
raw_image = Image.open(img_path).convert('RGB')

image_np = np.array(raw_image)
image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
inputs = processor(images=image_tensor, text=["a photo "], return_tensors="pt")

# conditional
out = model.generate(**inputs, max_new_tokens=20)
generated_caption_conditional = processor.decode(out[0], skip_special_tokens=True)
print("Generated Caption (Conditional):", generated_caption_conditional)

# Unconditional
inputs = processor(images=image_tensor, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=20, num_return_sequences=1, temperature=0.7)
generated_caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
print("Generated Caption (Unconditional):", generated_caption_unconditional)


ref_txt = "data/captions/test1.txt"
with open(ref_txt, "r") as f:
    reference_captions = f.readlines()

bleu_score_conditional = calculate_bleu(reference_captions, generated_caption_conditional)
bleu_score_unconditional = calculate_bleu(reference_captions, generated_caption_unconditional)

print("BLEU Score (Conditional):", bleu_score_conditional)
print("BLEU Score (Unconditional):", bleu_score_unconditional)
