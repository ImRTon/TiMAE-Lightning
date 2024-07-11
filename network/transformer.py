import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(
        self, 
        scale: torch.Tensor | None = None,
        attn_dropout_prob: float = 0.0
    ):
        super().__init__()

        self.scale = scale

        self.dropout = nn.Dropout(attn_dropout_prob) if attn_dropout_prob > 0 else None

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor | None = None,
        output_attentions: bool = False
    )-> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Perform scaled dot-product attention.

        Args:
            q (torch.Tensor): query tensor, shape (bs, ..., seq_len, emb_dim)
            k (torch.Tensor): key tensor, shape (bs, ..., seq_len, emb_dim)
            v (torch.Tensor): value tensor, shape (bs, ..., seq_len, emb_dim)
            mask (torch.Tensor | None, optional): Attention mask; shape must be 
                broadcastable to the shape of attention weights, ex: (bs, ..., seq_len, seq_len). 
                Two types of masks are supported. A boolean mask where a value of 
                True indicates that the element should take part in attention. 
                A float mask of the same type as query, key, value that is added to the attention score.
            output_attentions (bool, optional): Whether to output attention scores. Defaults to False.

        Returns:
            torch.Tensor: Attention scores (bs, ..., seq_len, emb_dim)
        """        
        # qkv: (batch_size, num_heads, seq_len, head_dim)
        scale_factor = 1 / math.sqrt(q.size(-1)) if self.scale is None else self.scale

        # Compute similarities
        sims = torch.matmul(q, k.transpose(-1, -2))
        sims = sims * scale_factor

        if mask is not None:
            if mask.dtype == torch.bool:
                sims.masked_fill(mask.logical_not(), float("-inf"))
            else:
                sims += mask

        attn_perc = F.softmax(sims, dim=-1)

        if self.dropout is not None:
            attn_perc = self.dropout(attn_perc)

        attn_scores = torch.matmul(attn_perc, v)

        if output_attentions:
            return (attn_scores, attn_perc)
        else:
            return (attn_scores,)

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_dropout_prob: float = 0.0
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"Dimension of model (d_model: {d_model}) must be divisible "
                              "by the number of attention heads (num_heads: {num_heads})")

        self.attn_dropout_prob = attn_dropout_prob
        self.num_heads = num_heads
        self.attention_head_size = d_model // num_heads
        self.all_head_size = self.attention_head_size * num_heads

        self.query = nn.Linear(d_model, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(d_model, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(d_model, self.all_head_size, bias=qkv_bias)

        self.scaled_dot_product_attention = ScaledDotProductAttention(
            scale=self.attention_head_size, 
            attn_dropout_prob=attn_dropout_prob
        )

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor | None = None,
        output_attentions: bool = False
    )-> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Perform multi-head attention from "Attention is All You Need" paper.

        Args:
            q (torch.Tensor): query tensor, shape (bs, seq_len, emb_dim)
            k (torch.Tensor): key tensor, shape (bs, seq_len, emb_dim)
            v (torch.Tensor): value tensor, shape (bs, seq_len, emb_dim)
            mask (torch.Tensor | None, optional): Attention mask; shape must be 
                broadcastable to the shape of attention weights, ex: (bs, ..., seq_len, seq_len). 
                Two types of masks are supported. A boolean mask where a value of 
                True indicates that the element should take part in attention. 
                A float mask of the same type as query, key, value that is added to the attention score.
                Defaults to None.
            output_attentions (bool, optional): Whether to output attention scores. Defaults to False.

        Returns:
            tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: 
                - attention_output (torch.Tensor): The result of the attention mechanism, shape (bs, seq_len, emb_dim).
                - attention_weights (torch.Tensor, optional): The attention weights, shape (bs, num_heads, seq_len, seq_len).
                  Only returned if output_attentions is True.
        """        
        bs, seq_len, d_model = q.size()
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # Split heads
        headed_qkv_shape = (bs, seq_len, self.num_heads, self.attention_head_size)
        # (bs, seq_len, all_head_size) -> (bs, num_heads, seq_len, head_dim)
        q = q.view(headed_qkv_shape).permute(0, 2, 1, 3)
        k = k.view(headed_qkv_shape).permute(0, 2, 1, 3)
        v = v.view(headed_qkv_shape).permute(0, 2, 1, 3)

        product_attns = self.scaled_dot_product_attention(q, k, v, mask, output_attentions)
        attn_scores = product_attns[0]

        # Reshape (bs, num_heads, seq_len, head_dim) -> (bs, seq_len, num_heads, head_dim)
        attn_scores = attn_scores.permute(0, 2, 1, 3).contiguous()
        bs, seq_len, _, _ = attn_scores.size()
        attn_scores = attn_scores.view(bs, seq_len, self.all_head_size)
        
        if output_attentions:
            return (attn_scores, product_attns[1])
        else:
            return (attn_scores,)
        

        
