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
    def __init__(self, num_classes: int, input_dim: int, d_model: int = 512, 
                 nhead: int = 8, num_encoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.blank_init_bias = -0.5
        # 입력 투영 레이어 수정 (Dropout 제거 또는 감소)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)  # Dropout 0.05
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)  # Dropout 0.05
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)
        """ self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),  # LayerNorm 추가
            nn.GELU(),
            nn.Dropout(dropout * 0.3),  # 드롭아웃 더 감소
            nn.Linear(d_model // 2, num_classes),
            # nn.LogSoftmax(dim=-1)  # CTC를 위한 LogSoftmax 추가
        ) """
        # 분류기 개선
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),  # ReLU 대신 GELU 사용
            nn.Dropout(dropout * 0.5),  # Dropout 감소
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # gain 감소
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # bias는 0으로
        final_linear = self.classifier[-1]

        if isinstance(final_linear, nn.Linear) and final_linear.bias is not None:
            # Blank 클래스 인덱스 0의 bias에 음수 값 설정
            with torch.no_grad():
                final_linear.bias.data[0] = self.blank_init_bias

    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        # 입력 검증
        if src.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {src.size(-1)}")
        
        # NaN 값 처리 (범위 축소)
        src = torch.nan_to_num(src, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # 입력 투영
        src_proj = self.input_projection(src)
        
        # Positional encoding
        src_proj = src_proj.transpose(0, 1)
        src_pe = self.pos_encoder(src_proj)
        src_pe = src_pe.transpose(0, 1)
        
        # Transformer 인코더
        output = self.transformer_encoder(src_pe, src_key_padding_mask=src_key_padding_mask)
        
        # 분류
        output = self.classifier(output)

        return output