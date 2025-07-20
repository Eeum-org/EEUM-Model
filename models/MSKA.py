import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MSKA_Model(nn.Module):
    def __init__(self, num_classes, input_dim=274, d_model=1024, nhead=8, 
                 num_encoder_layers=10, num_decoder_layers=8, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.target_pos_embedding = nn.Embedding(500, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # NaN 처리
        src = torch.nan_to_num(src, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Encoder
        src_proj = self.input_projection(src)
        src_proj = src_proj.transpose(0, 1)
        src_pe = self.pos_encoder(src_proj)
        src_pe = src_pe.transpose(0, 1)
        
        encoder_output = self.transformer_encoder(src_pe, src_key_padding_mask=src_key_padding_mask)
        
        if tgt is not None:  # Training
            # Target embedding
            tgt_emb = self.target_embedding(tgt)
            batch_size, tgt_len = tgt.shape
            positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
            tgt_pos_emb = self.target_pos_embedding(positions)
            tgt_input = tgt_emb + tgt_pos_emb
            
            # Decoder
            decoder_output = self.transformer_decoder(
                tgt_input, encoder_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            return self.output_projection(decoder_output)
        else:  # Inference
            return self._inference(encoder_output, src_key_padding_mask)

    def _inference(self, encoder_output, src_key_padding_mask, max_len=15, sos_idx=1, eos_idx=2):
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            if finished.all():
                break
                
            tgt_emb = self.target_embedding(generated)
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            tgt_pos_emb = self.target_pos_embedding(positions)
            tgt_input = tgt_emb + tgt_pos_emb
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            decoder_output = self.transformer_decoder(
                tgt_input, encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            logits = self.output_projection(decoder_output[:, -1, :])
            
            # EOS 강제 생성
            if step >= 8:
                logits[:, eos_idx] += 2.0
            
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            next_token = torch.where(finished.unsqueeze(1), 
                                   torch.full_like(next_token, eos_idx), 
                                   next_token)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            newly_finished = (next_token.squeeze(1) == eos_idx)
            finished = finished | newly_finished
        
        return generated


def eos_enhanced_loss(predictions, targets, ignore_index=0, eos_weight=3.0):
    """Enhanced loss function with EOS token weighting"""
    vocab_size = predictions.size(-1)
    predictions_flat = predictions.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    weights = torch.ones(vocab_size, device=predictions.device)
    weights[2] = eos_weight  # EOS index
    
    loss = F.cross_entropy(predictions_flat, targets_flat, weight=weights, ignore_index=ignore_index)
    return loss