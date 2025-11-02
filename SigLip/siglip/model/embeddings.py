import torch 
import torch.nn as nn 
from typing import Optional, Tuple 

class SigLipConfig: 
    num_channels: int = 3 
    hidden_size: int = 768 
    image_size: int = 224 
    patch_size: int = 16 


class SiglipVisionEmbeddings(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
        self.embed_dim = config.hidden_size 
        self.image_size = config.image_size 
        self.patch_size = config.patch_size 

        # Patch Embedding using Convulition 
        self.patch_embed = nn.Conv2d(
            in_channels=config.num_channels, 
            out_channels=self.embed_dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size,
            padding="valid" 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.num_positions = self.num_patches  


        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) 
        self.register_buffer(
            "position_ids", 
            torch.arange(self.num_positions).expand((1, -1)), 
            persistent=False 
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:          
        # pixel_values: [Batch_size, Channels, Height, Width] 
        batch_size = pixel_values.shape[0] 

        # Create Patches: [Batch_size, Embed_Dim, Num_Patches_H, Num_Patches_W] 
        patch_embeds = self.patch_embed(pixel_values) 

        # Flatten: [Batch_size, Embed_Dim, num_patches] 
        embeddings = patch_embeds.flatten(2) 

        # Transpose: [Batch_size, Num_patches, Embed_Dim] 
        embeddings = embeddings.transpose(1, 2)
       
        # Add Positional Embeddings 
        embeddings = embeddings + self.position_embedding(self.position_ids) 
        
        return embeddings



if __name__ == "__main__":
    embedder = SiglipVisionEmbeddings(SigLipConfig) 
    img = torch.rand(1, 3, 224, 224) 
    output = embedder(img)
    print(img.shape, output.shape)
