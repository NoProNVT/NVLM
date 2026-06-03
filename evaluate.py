import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import json

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import KvasirVQADataset
from model import EndoReportGenerator
from train import load_endofm

# Download NLTK resources for METEOR
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    pass

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Models
    checkpoint_path = "checkpoints/endofm_lv.pth"
    vision_encoder = load_endofm(checkpoint_path, device)
    
    llm_name = "Qwen/Qwen2.5-7B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = EndoReportGenerator(
        vision_encoder=vision_encoder,
        llm_model_name_or_path=llm_name,
        vision_dim=768
    ).to(device)
    
    # Load trained mapping network & LoRA weights
    epoch_to_eval = 4 # Customize the epoch you want to evaluate here
    epoch_dir = f"checkpoints/epoch_{epoch_to_eval}"
    mapping_ckpt = f"{epoch_dir}/mapping_network.pth"
    lora_ckpt = f"{epoch_dir}/lora_weights"
    
    if os.path.exists(mapping_ckpt):
        model.mapping_network.load_state_dict(torch.load(mapping_ckpt, map_location=device))
        print(f"Loaded Mapping Network weights from {epoch_dir}.")
    else:
        print("No Mapping Network weights found. Evaluating untrained model.")
        
    if os.path.exists(lora_ckpt):
        model.llm.load_adapter(lora_ckpt)
        print(f"Loaded LoRA weights from {epoch_dir}.")
    else:
        print("No LoRA weights found.")
        
    model.eval()
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    base_dir = "kvasir_vqa_1000"
    val_dataset = KvasirVQADataset(
        json_path=os.path.join(base_dir, "val", "dataset.json"),
        image_dir=os.path.join(base_dir, "val"),
        transform=transform,
        tokenizer=None # Return raw data for easy text generation handling
    )
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    total_bleu = 0
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    total_meteor = 0
    
    num_samples = len(val_dataset)
    
    print(f"Evaluating {num_samples} samples...")
    for idx in tqdm(range(num_samples)):
        item = val_dataset[idx]
        image = item["image"].unsqueeze(0).to(device) # Add batch dim
        question = item["question"]
        target = item["answer"]
        
        system_prompt = "You are a medical vision-language assistant; given an endoscopic image and a clinical question that may ask about one or more findings, provide a concise, clinically accurate response addressing all parts of the question in natural-sounding medical language as if spoken by a doctor in a single sentence."
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Inference / Generation logic
        with torch.no_grad():
            vision_features = model.vision_encoder(image)
            prefix_embeds = model.mapping_network(vision_features)
            
            # Initialize embeddings for generation
            inputs_embeds = model.llm.base_model.model.get_input_embeddings()(input_ids)
            prefix_embeds = prefix_embeds.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1)
            
            # Create corresponding attention_mask for new inputs_embeds
            prefix_length = prefix_embeds.shape[1]
            prefix_attention_mask = torch.ones((1, prefix_length), dtype=attention_mask.dtype, device=device)
            full_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            
            # Simple greedy generation logic (demo)
            generated_ids = []
            current_embeds = inputs_embeds
            
            # Qwen uses 151643 for eos_token usually
            max_new_tokens = 50
            
            # Safe way to generate from custom embeddings:
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Calculate metrics
        # BLEU
        ref_tokens = target.split()
        pred_tokens = prediction.split()
        total_bleu += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        
        # ROUGE
        rouge_scores = scorer.score(target, prediction)
        total_rouge1 += rouge_scores['rouge1'].fmeasure
        total_rouge2 += rouge_scores['rouge2'].fmeasure
        total_rougeL += rouge_scores['rougeL'].fmeasure
        
        # METEOR
        try:
            total_meteor += meteor_score([ref_tokens], pred_tokens)
        except:
            pass

    print("\n=== EVALUATION RESULTS (Kvasir-VQA Metrics) ===")
    print(f"BLEU-4 : {total_bleu / num_samples:.4f}")
    print(f"ROUGE-1: {total_rouge1 / num_samples:.4f}")
    print(f"ROUGE-2: {total_rouge2 / num_samples:.4f}")
    print(f"ROUGE-L: {total_rougeL / num_samples:.4f}")
    print(f"METEOR : {total_meteor / num_samples:.4f}")

if __name__ == "__main__":
    evaluate_model()
