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
    
    llm_name = "Qwen/Qwen1.5-0.5B" 
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = EndoReportGenerator(
        vision_encoder=vision_encoder,
        llm_model_name_or_path=llm_name,
        vision_dim=768, 
        prefix_length=10
    ).to(device)
    
    # Load trained mapping network weights
    mapping_ckpt = "checkpoints/mapping_network.pth"
    if os.path.exists(mapping_ckpt):
        model.mapping_network.load_state_dict(torch.load(mapping_ckpt, map_location=device))
        print("Đã tải trọng số Mapping Network.")
    else:
        print("Chưa có trọng số Mapping Network. Sẽ đánh giá model chưa được train.")
        
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
        tokenizer=None # Trả về dạng thô để tiện xử lý text generation
    )
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    total_bleu = 0
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    total_meteor = 0
    
    num_samples = len(val_dataset)
    
    print(f"Đang tiến hành đánh giá trên {num_samples} samples...")
    for idx in tqdm(range(num_samples)):
        item = val_dataset[idx]
        image = item["image"].unsqueeze(0).to(device) # Thêm batch dim
        question = item["question"]
        target = item["answer"]
        
        prompt = f"Question: {question} Answer: "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Inference / Generation logic
        with torch.no_grad():
            vision_features = model.vision_encoder(image)
            if vision_features.dim() > 2:
                vision_features = vision_features.mean(dim=1)
            prefix_embeds = model.mapping_network(vision_features)
            
            # Khởi tạo embedding cho generation
            inputs_embeds = model.llm.get_input_embeddings()(input_ids)
            prefix_embeds = prefix_embeds.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1)
            
            # Tạo attention_mask tương ứng với inputs_embeds mới
            prefix_attention_mask = torch.ones((1, prefix_embeds.shape[1]), dtype=attention_mask.dtype, device=device)
            full_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            
            # Simple greedy generation logic (demo)
            generated_ids = []
            current_embeds = inputs_embeds
            
            # Qwen uses 151643 for eos_token usually
            max_new_tokens = 50
            
            # Cách an toàn để generate từ embeddings tùy chỉnh:
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

    print("\n=== KẾT QUẢ ĐÁNH GIÁ (Kvasir-VQA Metrics) ===")
    print(f"BLEU-4 : {total_bleu / num_samples:.4f}")
    print(f"ROUGE-1: {total_rouge1 / num_samples:.4f}")
    print(f"ROUGE-2: {total_rouge2 / num_samples:.4f}")
    print(f"ROUGE-L: {total_rougeL / num_samples:.4f}")
    print(f"METEOR : {total_meteor / num_samples:.4f}")

if __name__ == "__main__":
    evaluate_model()
