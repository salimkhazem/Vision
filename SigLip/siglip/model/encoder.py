import torch
import torch.nn as nn

from config_siglip import SiglipConfig
from encoder_layer import SiglipEncoderLayer


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config)
             for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


if __name__ == "__main__":
    x = torch.rand(1, 196, 768)
    config = SiglipConfig()
    encoder = SiglipEncoder(config)
    out = encoder(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
