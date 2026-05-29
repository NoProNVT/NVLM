import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class KvasirVQADataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, tokenizer=None, max_length=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image'].replace("images/", ""))
        
        # Xử lý trường hợp ảnh không tồn tại để tránh lỗi
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, item['image'])
            
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Fallback nếu đường dẫn ảnh bị sai định dạng
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        question = item['question']
        answer = item['text']
        
        # Prepare text prompt for the LLM
        prompt = f"Question: {question} Answer: "
        
        # Tokenize the input and the target
        if self.tokenizer:
            # Tokenize prompt + answer
            full_text = prompt + answer + self.tokenizer.eos_token
            
            encodings = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
            
            # The labels should only compute loss on the answer part.
            prompt_encodings = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            prompt_len = prompt_encodings["input_ids"].shape[1]
            
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            # Also ignore padding tokens
            labels[attention_mask == 0] = -100
            
            return {
                "pixel_values": image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        return {
            "image": image,
            "question": question,
            "answer": answer
        }
