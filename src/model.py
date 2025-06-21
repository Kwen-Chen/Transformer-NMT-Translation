import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture"""
    
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        self._generate_pe(max_len)
    
    def _generate_pe(self, max_len):
        """Generate positional encoding matrix"""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Forward pass with dynamic extension if needed"""
        if x.size(0) > self.pe.size(0):
            # Dynamically extend positional encoding if needed
            self._generate_pe(x.size(0))
        
        # Move pe to the same device as input tensor
        pe = self.pe[:x.size(0), :].to(x.device)
        x = x + pe
        return self.dropout(x)


class TransformerNMT(nn.Module):
    """Transformer-based Neural Machine Translation Model"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_len=512):
        super(TransformerNMT, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the transformer model
        
        Args:
            src: source sequence (seq_len, batch_size)
            tgt: target sequence (seq_len, batch_size) 
            src_mask: source attention mask
            tgt_mask: target attention mask
            src_key_padding_mask: source padding mask
            tgt_key_padding_mask: target padding mask
            memory_key_padding_mask: memory padding mask
        """
        # Embedding and positional encoding
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transformer encoding
        memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        # Transformer decoding
        output = self.transformer_decoder(tgt, memory, tgt_mask, None,
                                        tgt_key_padding_mask, memory_key_padding_mask)
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask to prevent attention to future positions"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, seq, pad_idx):
        """Create padding mask for sequences"""
        return (seq == pad_idx).transpose(0, 1)


class BeamSearchDecoder:
    """Beam search decoder for inference"""
    
    def __init__(self, model, beam_size=5, max_len=100, sos_idx=1, eos_idx=2, pad_idx=0):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
    
    def decode(self, src, src_mask=None, src_key_padding_mask=None):
        """Beam search decoding"""
        device = src.device
        batch_size = src.size(1)
        
        # Encode source
        src_emb = self.model.src_embedding(src) * math.sqrt(self.model.d_model)
        src_emb = self.model.pos_encoder(src_emb)
        memory = self.model.transformer_encoder(src_emb, src_mask, src_key_padding_mask)
        
        # Initialize beam
        beams = [[(torch.tensor([self.sos_idx], device=device), 0.0)] for _ in range(batch_size)]
        
        for step in range(self.max_len):
            new_beams = [[] for _ in range(batch_size)]
            
            for batch_idx in range(batch_size):
                if not beams[batch_idx]:
                    continue
                    
                for seq, score in beams[batch_idx]:
                    if seq[-1].item() == self.eos_idx:
                        new_beams[batch_idx].append((seq, score))
                        continue
                    
                    # Prepare input
                    tgt_input = seq.unsqueeze(1)  # (seq_len, 1)
                    tgt_mask = self.model.generate_square_subsequent_mask(len(seq)).to(device)
                    
                    # Forward pass
                    tgt_emb = self.model.tgt_embedding(tgt_input) * math.sqrt(self.model.d_model)
                    tgt_emb = self.model.pos_encoder(tgt_emb)
                    
                    batch_memory = memory[:, batch_idx:batch_idx+1, :]
                    output = self.model.transformer_decoder(
                        tgt_emb, batch_memory, tgt_mask, None, None, None
                    )
                    
                    # Get next token probabilities
                    next_token_logits = self.model.output_projection(output[-1, 0, :])
                    next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top k tokens
                    top_probs, top_indices = torch.topk(next_token_probs, self.beam_size)
                    
                    for prob, idx in zip(top_probs, top_indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0)])
                        new_score = score + prob.item()
                        new_beams[batch_idx].append((new_seq, new_score))
                
                # Keep top k beams
                new_beams[batch_idx].sort(key=lambda x: x[1], reverse=True)
                new_beams[batch_idx] = new_beams[batch_idx][:self.beam_size]
            
            beams = new_beams
        
        # Return best sequences
        results = []
        for batch_idx in range(batch_size):
            if beams[batch_idx]:
                best_seq, _ = max(beams[batch_idx], key=lambda x: x[1])
                results.append(best_seq)
            else:
                results.append(torch.tensor([self.sos_idx, self.eos_idx], device=device))
        
        return results


def create_model(src_vocab_size, tgt_vocab_size, device, **kwargs):
    """Create and initialize the model"""
    model = TransformerNMT(src_vocab_size, tgt_vocab_size, **kwargs)
    model = model.to(device)
    
    # Initialize with Xavier uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model 