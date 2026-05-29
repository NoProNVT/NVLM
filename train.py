import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import KvasirVQADataset
from model import EndoReportGenerator

from endofm_vision_encoder import vit_base

def load_endofm(checkpoint_path, device):
    print(f"Loading EndoFM-LV từ {checkpoint_path}...")
    try:
        # Khởi tạo mô hình dựa theo kiến trúc thông dụng của EndoFM (ViT Base)
        model = vit_base(patch_size=16, num_classes=0) 
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Xử lý các key trong checkpoint (ví dụ do DINO pretraining sinh ra)
        if 'student' in state_dict:
            state_dict = state_dict['student']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("backbone.", "")
            clean_state_dict[k] = v
            
        model.load_state_dict(clean_state_dict, strict=False)
        model.to(device)
        model.eval() # Bắt buộc phải là eval mode
        return model
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}. Vui lòng kiểm tra lại kiến trúc.")
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Chuẩn bị Vision Encoder (EndoFM-LV)
    checkpoint_path = "checkpoints/endofm_lv.pth"
    vision_encoder = load_endofm(checkpoint_path, device)
    
    # 2. Tokenizer & LLM
    llm_name = "Qwen/Qwen1.5-0.5B" 
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Khởi tạo Mô hình toàn cục
    # vision_dim của ViT-Base là 768. 
    model = EndoReportGenerator(
        vision_encoder=vision_encoder,
        llm_model_name_or_path=llm_name,
        vision_dim=768, 
        prefix_length=10
    ).to(device)
    
    # 4. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    base_dir = "kvasir_vqa_1000"
    train_dataset = KvasirVQADataset(
        json_path=os.path.join(base_dir, "train", "dataset.json"),
        image_dir=os.path.join(base_dir, "train"),
        transform=transform,
        tokenizer=tokenizer,
        max_length=128
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 5. Optimizer (Chỉ train Mapping Network)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4)
    
    # 6. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad()
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
        
    print("Hoàn tất huấn luyện!")
    # Tùy chọn lưu lại weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.mapping_network.state_dict(), "checkpoints/mapping_network.pth")

if __name__ == "__main__":
    main()
