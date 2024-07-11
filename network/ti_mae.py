import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import lightning as L

import math

class TiMAEConfig:
    def __init__(
        self,
        # Transformer
        input_dim: int = 256,
        max_len: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu', # 'relu', 'gelu', 'elu', 'leaky_relu'
        cls_embed: bool = True,
        # Pretraining
        masking: bool = True,
        mask_ratio: float = 0.75,
        # Decoder
        decoder_hidden_size: int = 64,
        decoder_num_layers: int = 2,
        decoder_num_attention_heads: int = 4,
    ):
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.1

# Copied from https://github.com/asmodaay/ti-mae/blob/master/src/nn/positional.py
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100, hidden_size=64):
        super().__init__()

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )

        # Store the positional encoding
        pe = torch.zeros(1, max_len, hidden_size)

        # Using "sin" in odd indices and "cos" in even indices
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, hidden_size]

        Returns:
            output Tensor of shape [batch_size, seq_len, hidden_size]
        """

        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class TiMAEEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv = nn.Conv1d(config.input_dim, config.hidden_size, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, hidden_size)
        """        
        return self.conv(x)

class TiMAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.positional_encoding = PositionalEncoding(max_len=config.max_len, hidden_size=config.hidden_size)

        self.embedding = TiMAEEmbedding(config)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation=config.activation,
                norm_first=True, # In TiMAE, we do normalization before attention
                batch_first=True
            )

            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size)

    # Copied from https://github.com/facebookresearch/mae_st/blob/main/models_mae.py
    def random_masking(self, x, mask_ratio, is_cls_token=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns: x, mask, ids_restore, ids_keep
            last_hidden_state (torch.Tensor): a tensor of shape (batch_size, seq_len, hidden_size)
            mask (torch.LongTensor): a tensor of shape (batch_size, seq_len)
            ids_restore (torch.LongTensor): a tensor of shape (batch_size, sequence_length) 
                containing the original index of the (shuffled) masked patches.
            ids_keep (torch.LongTensor): a tensor of shape (batch_size, sequence_length)
        """        
        bs, seq_len, input_dim = x.shape

        # Embedding
        x = self.embedding(x)

        # Scale input for dot-product and softmax stability
        x = x * math.sqrt(self.config.hidden_size)

        # Append cls token
        if self.config.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Positional encoding
        x = self.positional_encoding(x)

        # Random masking
        if self.config.masking:
            if self.config.cls_embed:
                # Separate cls token
                cls_x = x[:, 0:1, :]
                x = x[:, 1:, :]

            x, mask, ids_restore, ids_keep = self.random_masking(x, self.config.mask_ratio)

            if self.config.cls_embed:
                # Restore cls token
                x = torch.cat((cls_x, x), dim=1)

        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        if self.config.masking:
            return x, mask, ids_restore, ids_keep
        else:
            return x

class TiMAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))

        self.embedding = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

        self.positional_encoding = PositionalEncoding(max_len=config.max_len, hidden_size=config.decoder_hidden_size)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.decoder_hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation=config.activation,
                norm_first=False, # In TiMAE, we do normalization before attention
                batch_first=True
            )

            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.decoder_hidden_size)

        self.projection = nn.Linear(config.decoder_hidden_size, config.input_dim, bias=True)

    def forward(self, x: torch.Tensor, ids_restore: torch.LongTensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, hidden_size)
            ids_restore (torch.LongTensor): a tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, input_dim)
        """        
        x = x[:, 1:, :]  # Remove encoder cls token
        bs, decoder_seq_len, hidden_size = x.shape

        # Input Embedding
        x = self.embedding(x)
        decoder_hidden_size = x.shape[-1]

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(bs, self.config.seq_len - decoder_seq_len, 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # We didn't use cls token in decoder
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([bs, self.seq_len, decoder_hidden_size])

        # Positional Encoding
        x = self.positional_encoding(x)

        # Decoder
        x = self.layers(x)
        x = self.norm(x)

        # Projection
        x = self.projection(x)

        return x

class TiMAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        config.masking = False # Disable masking for encoder
        
        self.config = config

        self.encoder = TiMAEEncoder(config)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, hidden_size)
        """        
        return self.encoder(x)

class TiMAEForPretraining(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.encoder = TiMAEEncoder(config)
        self.decoder = TiMAEDecoder(config)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, input_dim)
        """        
        masked_x, mask, ids_restore, ids_keep = self.encoder(x)
        reconstruct_x = self.decoder(masked_x, ids_restore)

        return reconstruct_x, mask, ids_restore
    
    def training_step(self, batch, batch_idx):
        x = batch
        masked_x, mask, ids_restore, ids_keep = self.encoder(x)
        reconstruct_x = self.decoder(masked_x, ids_restore)

        loss = F.mse_loss(reconstruct_x, x)

        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)