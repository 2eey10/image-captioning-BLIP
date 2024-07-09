import os
import nltk
import json
import torch
import logging
import warnings
import time
import csv
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NLTK 
nltk.download('punkt')

# loger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s')  
console_handler.setFormatter(console_formatter)

# file handler
file_handler = logging.FileHandler('image_processing.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# BLEU score method
def calculate_bleu(reference_captions, generated_caption):
    reference_captions = [nltk.word_tokenize(caption.lower()) for caption in reference_captions]
    generated_caption = nltk.word_tokenize(generated_caption.lower())
    
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_captions, generated_caption, smoothing_function=smoothie)
    return score

# BLIP2 model and processor load
logger.info("모델과 프로세서를 로드하는 중...")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
model.eval()
logger.info("모델과 프로세서 로드 완료")
time.sleep(3)

# dir
img_dir = "val2017"
cap_path = "annotations/captions_val2017.json"

# caption load
logger.info("정답 캡션을 로드하는 중...")
time.sleep(1)
with open(cap_path, 'r') as f:
    captions = json.load(f)
logger.info("정답 캡션 로드 완료")
time.sleep(1.5)

# image & caption mapping
image_captions_map = {}
for caption in captions['annotations']:
    image_id = caption['image_id']
    if image_id not in image_captions_map:
        image_captions_map[image_id] = []
    image_captions_map[image_id].append(caption['caption'])

# BLEU score list
bleu_scores = []

# csv  
csv_file = open('image_processing_results.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Image ID', 'Original Captions', 'Generated Caption', 'BLEU Score'])

logger.info(f"이미지 캡셔닝 모델의 캡션 생성 및 추론을 시작합니다")
time.sleep(3)

# tqdm loop
for img_file in tqdm(os.listdir(img_dir), desc="Processing images"):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(img_dir, img_file)
        image_id = int(img_file.split('.')[0])
        
        if image_id in image_captions_map:
            raw_image = Image.open(img_path).convert('RGB')
            reference_captions = image_captions_map[image_id]

            # image processing
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
            # caption generation
            generated_caption = model.generate({"image": image})[0]

            # BLEU Score 
            bleu_score = calculate_bleu(reference_captions, generated_caption)
            bleu_scores.append(bleu_score)

            original_formatter = console_handler.formatter
            console_handler.setFormatter(logging.Formatter('%(message)s'))

            logger.info(f"Image ID: {image_id}")
            logger.info(f"Original Captions: {reference_captions}")
            logger.info(f"Generated Caption: {generated_caption}")
            logger.info(f"BLEU Score: {bleu_score}")

            # csv save
            csv_writer.writerow([image_id, reference_captions, generated_caption, bleu_score])

        else:
            logger.warning(f"Image ID {image_id}의 캡션을 찾을 수 없습니다.")
    
# BLEU-4 Score 
logger.info(f"추론 작업이 종료됐습니다.")
time.sleep(3)
bleu_4_score = sum(bleu_scores) / len(bleu_scores)
logger.info(f"--------------------------")
logger.info(f"최종 BLEU-4 Score: {bleu_4_score}")
logger.info(f"--------------------------")
