import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TokenEmbedding(nn.Module):
    """
    Token embeddings for code text
    Converts token IDs to dense embeddings
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 768,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Segment embeddings (for diff vs context)
        self.segment_embedding = nn.Embedding(2, embed_dim)  # 0=diff, 1=context
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self, 
        token_ids: torch.Tensor, 
        segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert token IDs to embeddings
    
        """
        batch_size, seq_len = token_ids.size()
        
        # Create position IDs
        pos_ids = torch.arange(seq_len, device=token_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.pos_embedding(pos_ids)
        
        # Add segment embeddings if provided
        if segment_ids is not None:
            seg_emb = self.segment_embedding(segment_ids)
            embeddings = token_emb + pos_emb + seg_emb
        else:
            embeddings = token_emb + pos_emb
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class CodeAttention(nn.Module):
    """
    Multi-head attention for code text
    
    Similar to BERT's attention but optimized for code patterns
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, seq_len, embed_dim * 3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # compute attention scoress
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        out = self.proj(out)
        
        return out


class CodeTransformerBlock(nn.Module):
    """
    Transformer block for code processing
    
    Combines attention with feed-forward network
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CodeAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply transformer block
        """
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        
        return x


class CodeBERT(nn.Module):
    """
    CodeBERT for processing code text and context
    
    Processes both diff text and PR context through transformer layers
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim, max_seq_len)
        
        self.blocks = nn.ModuleList([
            CodeTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        
        self.pooler = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        diff_text: torch.Tensor,
        context: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process diff text and context
        """
        # Process diff text
        diff_embeddings = self.token_embedding(diff_text)
        for block in self.blocks:
            diff_embeddings = block(diff_embeddings, diff_mask)
        diff_embeddings = self.norm(diff_embeddings)
        
        # Process context
        context_embeddings = self.token_embedding(context)
        for block in self.blocks:
            context_embeddings = block(context_embeddings, context_mask)
        context_embeddings = self.norm(context_embeddings)
        
        # Pool to get sentence level representations
        diff_features = diff_embeddings.mean(dim=1)  # [B, embed_dim]
        context_features = context_embeddings.mean(dim=1)  # [B, embed_dim]
        
        diff_features = torch.tanh(self.pooler(diff_features))
        context_features = torch.tanh(self.pooler(context_features))
        
        return diff_features, context_features

