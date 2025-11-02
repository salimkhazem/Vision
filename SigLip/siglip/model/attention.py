import torch 
import torch.nn as nn 

class SigLipConfig: 
    hidden_size: int = 768 
    num_attention_heads: int = 12  

class SiglipAttention(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
        self.embed_dim = config.hidden_size 
        self.num_heads = config.num_attention_heads 
        self.head_dim = self.embed_dim // self.num_heads 
        self.scale = self.head_dim ** -0.5 

        # Create Key, Query, Value 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim) 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        batch_size, seq_len, _ = hidden_states.size() 
        
        # Project to Q, K, V 
        query = self.q_proj(hidden_states) 
        value = self.v_proj(hidden_states) 
        key = self.k_proj(hidden_states)

        # Reshape for MHA 
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim) 
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim) 
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim) 
       
        # Transpose: [batch_size, Heads, Seq, head_dim] 
        query = query.transpose(1, 2) 
        value = value.transpose(1, 2) 
        key = key.transpose(1, 2) 

        # Attention Scores 
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale 
        attn_weights = torch.softmax(attn_weights, dim=-1) 

        # Apply attention to values 
        attn_output = torch.matmul(attn_weights, value) 

        # Reshape 
        attn_output = attn_output.transpose(1, 2).contiguous() 
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim) 

        # Output Projection 
        attn_output = self.out_proj(attn_output) 

        return attn_output 

if __name__ == "__main__": 
    x = torch.rand(1, 196, 768)
    config = SigLipConfig() 
    attention = SiglipAttention(config) 
    atten_out =  attention(x)
    print(f"Input: {x.shape} ( [batch_size, num_patches, hidden_size(embed_dim)] ) --> Output: {atten_out.shape}")
