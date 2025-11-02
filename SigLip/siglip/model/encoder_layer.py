import torch 
import torch.nn as nn 

# Local Import 
from mlp import SiglipMLP 
from attention import SiglipAttention 

class SigLipConfig: 
    num_channels: int = 3 
    hidden_size: int = 768 
    image_size: int = 224 
    patch_size: int = 16
    intermediate_size: int = 3072
    num_attention_heads: int = 12 
    layer_norm_eps: float = 1e-6 


class SiglipEncoderLayer(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.embed_dim = config.hidden_size 
        self.self_attn = SiglipAttention(config) 
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) 
        self.mlp = SiglipMLP(config) 
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        residual = hidden_states 
        hidden_states = self.layer_norm1(hidden_states) 
        hidden_states = self.self_attn(hidden_states) 
        hidden_states = residual + hidden_states 

        residual = hidden_states 
        hidden_states = self.layer_norm2(hidden_states) 
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states 

        return hidden_states


if __name__ == "__main__": 
    config = SigLipConfig() 
    layer = SiglipEncoderLayer(config) 
    x = torch.rand(1, 196, 768) 
    output = layer(x) 
    print(f"Input: [{x.shape}] -> Output: [{output.shape}]")
    

