import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class MappingNetwork(nn.Module):
    def __init__(self, vision_dim, llm_dim, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        # A simple MLP mapping network as a starting point. 
        # A Transformer based mapping network can also be used as in ClipCap.
        self.mapper = nn.Sequential(
            nn.Linear(vision_dim, llm_dim * prefix_length),
            nn.GELU(),
            nn.Linear(llm_dim * prefix_length, llm_dim * prefix_length)
        )
        self.llm_dim = llm_dim

    def forward(self, vision_features):
        # vision_features: (batch, vision_dim)
        mapped = self.mapper(vision_features) # (batch, llm_dim * prefix_length)
        mapped = mapped.view(-1, self.prefix_length, self.llm_dim) # (batch, prefix_length, llm_dim)
        return mapped

class EndoReportGenerator(nn.Module):
    def __init__(self, vision_encoder, llm_model_name_or_path, vision_dim=1024, prefix_length=10):
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
        # Freeze LLM
        for param in self.llm.parameters():
            param.requires_grad = False
            
        llm_dim = self.llm.config.hidden_size
        
        # 3. Mapping Network
        self.mapping_network = MappingNetwork(vision_dim, llm_dim, prefix_length)
        
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Extract vision features
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values)
            # Assuming vision_features is (batch, dim). If it returns a sequence, we might need pooling
            if vision_features.dim() > 2:
                vision_features = vision_features.mean(dim=1)
                
        # Get visual prefixes
        prefix_embeds = self.mapping_network(vision_features) # (batch, prefix_length, llm_dim)
        
        # Get word embeddings from LLM
        inputs_embeds = self.llm.get_input_embeddings()(input_ids) # (batch, seq_len, llm_dim)
        
        # Đảm bảo cùng kiểu dữ liệu (tránh lỗi Float và Half)
        prefix_embeds = prefix_embeds.to(inputs_embeds.dtype)
        
        # Concatenate prefix embeddings with word embeddings
        inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1) # (batch, prefix_length + seq_len, llm_dim)
        
        # Extend attention mask
        batch_size = attention_mask.shape[0]
        prefix_attention_mask = torch.ones(batch_size, self.mapping_network.prefix_length, device=attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        # Extend labels
        if labels is not None:
            prefix_labels = torch.full((batch_size, self.mapping_network.prefix_length), -100, dtype=torch.long, device=labels.device)
            labels = torch.cat((prefix_labels, labels), dim=1)
            
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
