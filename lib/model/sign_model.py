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

class KeypointTransformerWithAttention(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, d_model: int = 512, 
                 nhead: int = 8, num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.blank_init_bias = -0.5

        # 입력 투영 레이어 수정 (Dropout 제거 또는 감소)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)  # Dropout 0.1
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)  # Dropout 0.1
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)

        # Decoder 부분
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout * 0.5, batch_first=True, norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, num_classes)
        
        # Target embeddings
        self.target_pos_embedding = nn.Embedding(500, d_model)
        self.target_embedding = nn.Embedding(num_classes, d_model)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # gain 감소
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # bias는 0으로
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, src: Tensor, target_tokens: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:
        # Encoder
        src = torch.nan_to_num(src, nan=0.0, posinf=1e3, neginf=-1e3)
        src_proj = self.input_projection(src)
        src_proj = src_proj.transpose(0, 1)
        src_pe = self.pos_encoder(src_proj)
        src_pe = src_pe.transpose(0, 1)
        
        encoder_output = self.transformer_encoder(src_pe, src_key_padding_mask=src_key_padding_mask)
        
        if target_tokens is not None:
            # Training mode: teacher forcing
            return self._forward_training(encoder_output, target_tokens, src_key_padding_mask)
        else:
            # Inference mode: autoregressive generation
            return self._forward_inference(encoder_output, src_key_padding_mask)
    
    def _forward_training(self, encoder_output, target_tokens, src_key_padding_mask):
        batch_size, target_len = target_tokens.shape
        
        # Target embeddings
        target_emb = self.target_embedding(target_tokens)
        positions = torch.arange(target_len, device=target_tokens.device).unsqueeze(0).expand(batch_size, -1)
        target_pos_emb = self.target_pos_embedding(positions)
        target_input = target_emb + target_pos_emb
        
        # Causal mask for decoder
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(target_tokens.device)
        
        # Decoder
        decoder_output = self.transformer_decoder(
            target_input, encoder_output,
            tgt_mask=target_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return self.output_projection(decoder_output)
    
    def _forward_inference(self, encoder_output, src_key_padding_mask, max_length=8):  # 8로 더 단축
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # Start with <sos> token (index 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length):
            if finished.all() or step >= 6:  # 6토큰에서 강제 종료
                break
                
            target_emb = self.target_embedding(generated)
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            target_pos_emb = self.target_pos_embedding(positions)
            target_input = target_emb + target_pos_emb
            
            target_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            decoder_output = self.transformer_decoder(
                target_input, encoder_output,
                tgt_mask=target_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            next_token_logits = self.output_projection(decoder_output[:, -1:, :])
            
            # EOS 토큰 강제 삽입 - 더 강하게
            if step >= 3:  # 3토큰 이후엔 EOS 토큰 확률 대폭 증가
                next_token_logits[:, :, 2] += 5.0  # EOS 토큰 logit 대폭 증가
            
            # 4토큰 이후엔 거의 무조건 EOS
            if step >= 4:
                next_token_logits[:, :, :] = -1000.0  # 다른 토큰들 억제
                next_token_logits[:, :, 2] = 1000.0   # EOS만 높게
            
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # 이미 끝난 시퀀스는 EOS 토큰으로 채우기
            next_token = torch.where(finished.unsqueeze(1), 
                                    torch.full_like(next_token, 2),  # EOS로 채움
                                    next_token)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOS 토큰(2) 생성 시 해당 시퀀스 종료
            newly_finished = (next_token.squeeze(1) == 2)
            finished = finished | newly_finished
            
        # 마지막에 EOS 토큰이 없으면 강제 추가
        for b in range(batch_size):
            if generated[b, -1] != 2:  # 마지막이 EOS가 아니면
                generated[b, -1] = 2   # 마지막을 EOS로 변경
                
        return generated

    def _forward_training_with_scheduled_sampling(self, encoder_output, target_tokens, src_key_padding_mask, 
                                                teacher_forcing_ratio=0.5):
        """Scheduled sampling으로 training-inference gap 줄이기"""
        batch_size, target_len = target_tokens.shape
        device = target_tokens.device
        
        # 시퀀스가 너무 짧으면 일반 teacher forcing 사용
        if target_len <= 2:
            return self._forward_training(encoder_output, target_tokens, src_key_padding_mask)
        
        # 시작은 항상 <sos>
        generated = target_tokens[:, :1]  # <sos> 토큰
        outputs = []
        
        for step in range(1, target_len):
            # Teacher forcing vs 자체 생성 결정
            if torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: 정답 토큰 사용
                current_input = target_tokens[:, :step+1]
            else:
                # 자체 생성: 이전까지 생성된 토큰 사용
                current_input = generated
            
            # Embedding
            target_emb = self.target_embedding(current_input)
            seq_len = current_input.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            target_pos_emb = self.target_pos_embedding(positions)
            target_input = target_emb + target_pos_emb
            
            # Decode
            target_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            decoder_output = self.transformer_decoder(
                target_input, encoder_output,
                tgt_mask=target_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Next token prediction
            next_logits = self.output_projection(decoder_output[:, -1:, :])
            outputs.append(next_logits)
            
            # 다음 스텝을 위한 토큰 생성
            next_token = torch.argmax(next_logits, dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return torch.cat(outputs, dim=1)  # (B, L-1, V)