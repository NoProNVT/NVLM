import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class MappingNetwork(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        # A 2-layer MLP mapping network like LLaVA
        self.mapper = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features):
        # vision_features: (batch, num_patches, vision_dim)
        mapped = self.mapper(vision_features) # (batch, num_patches, llm_dim)
        return mapped

class EndoReportGenerator(nn.Module):
    def __init__(self, vision_encoder, llm_model_name_or_path, vision_dim=768):
        super().__init__()
        
        # 1. Vision Encoder
        self.vision_encoder = vision_encoder
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 2. LLM
        print(f"Loading LLM: {llm_model_name_or_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Apply LoRA to LLM
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
            
        # peft model uses base_model.model internally
        llm_dim = self.llm.base_model.model.config.hidden_size
        
        # 3. Mapping Network
        self.mapping_network = MappingNetwork(vision_dim, llm_dim)
        
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Extract vision features
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values)
                
        # Get visual prefixes
        prefix_embeds = self.mapping_network(vision_features) # (batch, prefix_length, llm_dim)
        
        # Get word embeddings from LLM
        inputs_embeds = self.llm.base_model.model.get_input_embeddings()(input_ids) # (batch, seq_len, llm_dim)
        
        # Đảm bảo cùng kiểu dữ liệu (tránh lỗi Float và Half)
        prefix_embeds = prefix_embeds.to(inputs_embeds.dtype)
        
        # Concatenate prefix embeddings with word embeddings
        inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1) # (batch, prefix_length + seq_len, llm_dim)
        
        prefix_length = prefix_embeds.shape[1]
        
        # Extend attention mask
        batch_size = attention_mask.shape[0]
        prefix_attention_mask = torch.ones(batch_size, prefix_length, device=attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        # Extend labels
        if labels is not None:
            prefix_labels = torch.full((batch_size, prefix_length), -100, dtype=torch.long, device=labels.device)
            labels = torch.cat((prefix_labels, labels), dim=1)
            
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
