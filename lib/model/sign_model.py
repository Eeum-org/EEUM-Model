import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class KeypointTransformer(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, d_model: int = 256, 
                 nhead: int = 8, num_encoder_layers: int = 4, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.classifier(output)
        return output