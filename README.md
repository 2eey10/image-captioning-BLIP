# Image Captioning using BLIP

This project demonstrates the use of BLIP (Bootstrapping Language-Image Pre-training) for generating image captions. It includes the process of generating both conditional and unconditional captions for a given image and calculating the BLEU score to evaluate the generated captions against reference captions.

## Overview

BLIP (Bootstrapping Language-Image Pre-training) is a framework for pre-training vision-language models. This project uses the `BlipProcessor` and `BlipForConditionalGeneration` classes from the `transformers` library to generate captions for images.

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers library from Hugging Face
- NLTK
- Pillow

## Installation

1. Install the required Python packages:

    ```bash
    pip install torch transformers nltk pillow
    ```

2. Download the NLTK data:

    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

1. **Load the Image:**

    Load an image using Pillow and convert it to RGB format.

    ```python
    from PIL import Image
    img_path = "data/images/test1.jpg"
    raw_image = Image.open(img_path).convert('RGB')
    ```

2. **Convert Image to Tensor:**

    Convert the image to a numpy array and then to a PyTorch tensor.

    ```python
    import numpy as np
    import torch

    image_np = np.array(raw_image)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
    ```

3. **Load the Processor and Model:**

    Load the BLIP processor and model from the Hugging Face transformers library.

    ```python
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    ```

4. **Generate Captions:**

    Generate conditional and unconditional captions for the image.

    ```python
    # Conditional Caption
    inputs = processor(images=image_tensor, text=["a photo "], return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    generated_caption_conditional = processor.decode(out[0], skip_special_tokens=True)
    print("Generated Caption (Conditional):", generated_caption_conditional)

    # Unconditional Caption
    inputs = processor(images=image_tensor, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20, num_return_sequences=1, temperature=0.7)
    generated_caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
    print("Generated Caption (Unconditional):", generated_caption_unconditional)
    ```

5. **Calculate BLEU Score:**
   * BLEU: *Biligual Evaluation Understudy Score*

    Calculate the BLEU score to evaluate the generated captions.

    ```python
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    def calculate_bleu(reference_captions, generated_caption):
        reference_captions = [nltk.word_tokenize(caption.lower()) for caption in reference_captions]
        generated_caption = nltk.word_tokenize(generated_caption.lower())
        
        smoothie = SmoothingFunction().method4
        score = sentence_bleu(reference_captions, generated_caption, smoothing_function=smoothie)
        return score

    ref_txt = "data/captions/test1.txt"
    with open(ref_txt, "r") as f:
        reference_captions = f.readlines()

    bleu_score_conditional = calculate_bleu(reference_captions, generated_caption_conditional)
    bleu_score_unconditional = calculate_bleu(reference_captions, generated_caption_unconditional)

    print("BLEU Score (Conditional):", bleu_score_conditional)
    print("BLEU Score (Unconditional):", bleu_score_unconditional)
    ```
![test1](https://github.com/2eey10/image-captioning-BLIP/assets/133326837/10811c33-94e3-4d7e-ab06-89082d714912)
![스크린샷 2024-06-17 오후 4 25 27](https://github.com/2eey10/image-captioning-BLIP/assets/133326837/67e250d0-dbc6-4bf8-be83-3c049c7cd869)
## File Structure
```
.
├── README.md
├── blip_main.py
├── data
│   ├── captions
│   │   ├── test1.txt
│   │   └── test2.txt
│   └── images
│       ├── test1.jpg
│       └── test2.jpg
└── requirements.txt
```
