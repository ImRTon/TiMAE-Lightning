import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torch.optim import AdamW, Adam, lr_scheduler
from torchmetrics import Accuracy, F1Score
from pathlib import Path

from .transformer import TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        max_len: int, 
        emb_size: int
    ):
        super().__init__()

        positions = torch.arange(0, max_len, step=1, dtype=torch.float).unsqueeze(1)
        embedding_index = torch.arange(0, emb_size, step=2, dtype=torch.float)

        div_term = torch.exp(-embedding_index * (math.log(10000.0) / emb_size))

        # Store the positional encoding
        pe = torch.zeros(1, max_len, emb_size)

        # Using "sin" in odd indices and "cos" in even indices
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, emb_size]

        Returns:
            output Tensor of shape [batch_size, seq_len, emb_size]
        """

        x = x + self.pe[:,:x.size(1),:]

        return x

class TiMAEEmbedding(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        emb_size: int
    ):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, emb_size, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, hidden_size)
        """        
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

class TiMAEEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        num_heads: int,
        num_layers: int,
        emb_size: int,
        intermediate_dim: int,
        masking: bool,
        cls_embed: bool = True,
        max_len: int = 100,
        qkv_bias: bool = False,
        attn_dropout_prob: float = 0.1,
        ffn_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        act_func: str = 'gelu',
        mask_ratio: float = 0.75
    ):
        super().__init__()

        self.masking = masking
        self.cls_embed = cls_embed
        self.mask_ratio = mask_ratio

        self.positional_encoding = PositionalEncoding(max_len=max_len, emb_size=emb_size)

        self.embedding = TiMAEEmbedding(input_dim, emb_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=emb_size,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                qkv_bias=qkv_bias,
                attn_dropout_prob=attn_dropout_prob,
                ffn_dropout_prob=ffn_dropout_prob,
                norm_first=True,
                layer_norm_eps=layer_norm_eps,
                act_func=act_func
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_size, eps=layer_norm_eps)

    def interpolate_pos_encoding(
        self, 
        position_embeddings: torch.Tensor, 
        interpolate_factor: float
    ):
        """This method allows to interpolate the pre-trained position encodings, 
        to be able to use the model on different sample rates.

        # Reference from DINO:
        # https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174

        Args:
            position_embeddings (torch.Tensor): a tensor of shape (batch_size, seq_len, hidden_size)
            interpolate_factor (int): a factor used for interpolation, can be used when having a different sample rate.

        Returns:
            torch.Tensor: a tensor of shape (batch_size, time_seq_len, hidden_size)
        """        
        assert interpolate_factor > 0, "Interpolation factor should be greater than 0"

        position_embeddings = F.interpolate(
            position_embeddings.permute(0, 2, 1),
            scale_factor=interpolate_factor,
            mode="bicubic",
            align_corners=False
        ).permute(0, 2, 1)

        return position_embeddings

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

    def forward(self, x: torch.Tensor, interpolate_pos_enc_factor: float = 1.0, output_attentions: bool = False):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns: x, mask, ids_restore, ids_keep
            last_hidden_state (torch.Tensor): a tensor of shape (batch_size, seq_len, emb_dim)
            mask (torch.LongTensor): a tensor of shape (batch_size, seq_len)
            ids_restore (torch.LongTensor): a tensor of shape (batch_size, seq_len) 
                containing the original index of the (shuffled) masked patches.
            ids_keep (torch.LongTensor): a tensor of shape (batch_size, seq_len)
        """        
        bs, seq_len, input_dim = x.shape

        # Input Embedding
        x = self.embedding(x)

        # Positional Encoding
        pos_embeddings = self.positional_encoding.pe[:, :seq_len + 1 if self.cls_embed else seq_len, :]
        if interpolate_pos_enc_factor != 1.0:
            pos_embeddings = self.interpolate_pos_encoding(pos_embeddings)    

        # add position embeddings w/o cls token
        if self.cls_embed:
            x = x + pos_embeddings[:, 1:, :]
        else:
            x = x + pos_embeddings

        # Random masking
        if self.masking:
            x, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)

        # Append cls token
        if self.cls_embed:
            cls_token = self.cls_token + pos_embeddings[:, 0, :]
            cls_tokens = cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if output_attentions:
            all_attentions = ()

        for layer in self.layers:
            x = layer(x)

            if output_attentions:
                all_attentions += (x[1],)
            x = x[0]

        # Normalize
        x = self.norm(x)

        outputs = (x,)
        if self.masking:
            outputs += (mask, ids_restore, ids_keep)
        if output_attentions:
            outputs += (all_attentions,)
        return outputs

class TiMAEDecoder(nn.Module):
    def __init__(
        self, 
        output_dim: int,
        num_heads: int,
        num_layers: int,
        emb_size: int,
        encoder_emb_size: int,
        encoder_seq_len: int,
        intermediate_dim: int,
        encoder_cls_embed: bool = True,
        max_len: int = 100,
        qkv_bias: bool = False,
        attn_dropout_prob: float = 0.1,
        ffn_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        act_func: str = 'gelu'
    ):
        super().__init__()

        self.encoder_cls_embed = encoder_cls_embed
        self.encoder_seq_len = encoder_seq_len

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.embedding = nn.Linear(encoder_emb_size, emb_size, bias=True)

        self.positional_encoding = PositionalEncoding(max_len=max_len, emb_size=emb_size)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=emb_size,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                qkv_bias=qkv_bias,
                attn_dropout_prob=attn_dropout_prob,
                ffn_dropout_prob=ffn_dropout_prob,
                norm_first=True,
                layer_norm_eps=layer_norm_eps,
                act_func=act_func
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_size)

        self.projection = nn.Linear(emb_size, output_dim, bias=True)

    def forward(self, x: torch.Tensor, ids_restore: torch.LongTensor, output_attentions: bool = False):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, emb_size)
            ids_restore (torch.LongTensor): a tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, output_dim)
        """        
        bs, decoder_seq_len, encoder_emb_size = x.shape

        # Input Embedding
        x = self.embedding(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(bs, self.encoder_seq_len - decoder_seq_len + 1, 1)
        if self.encoder_cls_embed:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Remove encoder cls token
        else:
            x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        # unshuffle
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        if self.encoder_cls_embed:
            x_ = torch.cat([x[:, :1, :], x_], dim=1) # Append encoder cls token

        # Positional Encoding
        x = self.positional_encoding(x_)

        if output_attentions:
            all_attentions = ()

        # Decoder
        for layer in self.layers:
            x = layer(x)
            if output_attentions:
                all_attentions += (x[1],)
            x = x[0]

        x = self.norm(x)

        # Projection
        x = self.projection(x)
        x = x[:, 1:, :]  # Remove cls token

        if output_attentions:
            return (x, all_attentions)
        else:
            return (x,)

class TiMAE(L.LightningModule):
    def __init__(
        self,
        lr: float,
        num_classes: int,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        emb_size: int,
        intermediate_dim: int,
        cls_embed: bool = True,
        pooling: str | None = None,
        max_len: int = 100,
        qkv_bias: bool = False,
        attn_dropout_prob: float = 0.1,
        ffn_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        act_func: str = 'gelu',
        interpolate_pos_enc_factor: float = 1.0,
        encoder_ckpt_path: str | Path | None = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TiMAEEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            emb_size=emb_size,
            intermediate_dim=intermediate_dim,
            masking=False,
            cls_embed=cls_embed,
            max_len=max_len,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func
        )
        if encoder_ckpt_path is not None:
            print(f"==== Loading encoder weights from {encoder_ckpt_path} ====")
            with open(encoder_ckpt_path, 'rb') as f:
                self.encoder.load_state_dict(torch.load(f)["state_dict"], strict=False)

        self.classifier = nn.Linear(emb_size, num_classes, bias=True)

        self.criterion = nn.CrossEntropyLoss()

        self.train_accu = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accu = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_accu = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)

        # Check if pooling is valid
        if cls_embed is False and pooling not in ['mean', 'max']:
            raise ValueError("Invalid pooling method, supported methods are 'mean' and 'max'")

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, emb_size)
        """        
        x = self.encoder(x, interpolate_pos_enc_factor=self.hparams.interpolate_pos_enc_factor)[0]
        
        if self.hparams.cls_embed:
            x = x[:, 0, :]
        elif self.hparams.pooling == 'max':
            # Max pooling
            x = x.max(dim=1)[0]
        elif self.hparams.pooling == 'mean':
            # Average pooling
            x = x.mean(dim=1)
        else:
            raise ValueError("Invalid pooling method, supported methods are 'mean' and 'max'")
        
        logits = self.classifier(x)
        return logits
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)

        self.log('train_loss', loss, prog_bar=True)
        accu = self.train_accu(preds, y)
        self.log('train_accu', accu, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)

        self.log('val_loss', loss)
        accu = self.val_accu(preds, y)
        self.log('val_accu', accu)
        f1 = self.val_f1(preds, y)
        self.log('val_f1', f1)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        accu = self.test_accu(preds, y)
        self.log('test_accu', accu)
        f1 = self.test_f1(preds, y)
        self.log('test_f1', f1)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        return preds, y

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches),
                "interval": "step",
            },
        }

class TiMAEForPretraining(L.LightningModule):
    def __init__(
        self, 
        lr: float,
        input_dim: int,
        num_heads: int,
        num_decoder_heads: int,
        num_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        decoder_emb_size: int,
        intermediate_dim: int,
        decoder_intermediate_dim: int,
        cls_embed: bool = True,
        seq_len: int = 100,
        max_len: int | None = None,
        qkv_bias: bool = False,
        attn_dropout_prob: float = 0.1,
        ffn_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        act_func: str = 'gelu',
        mask_ratio: float = 0.75
    ):
        super().__init__()
        if max_len is None:
            max_len = seq_len
        self.save_hyperparameters()

        self.encoder = TiMAEEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            emb_size=emb_size,
            intermediate_dim=intermediate_dim,
            masking=True,
            cls_embed=cls_embed,
            max_len=max_len,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func,
            mask_ratio=mask_ratio
        )

        self.decoder = TiMAEDecoder(
            output_dim=input_dim,
            num_heads=num_decoder_heads,
            num_layers=num_decoder_layers,
            emb_size=decoder_emb_size,
            encoder_emb_size=emb_size,
            encoder_seq_len=seq_len,
            intermediate_dim=decoder_intermediate_dim,
            encoder_cls_embed=cls_embed,
            max_len=max_len,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func
        )

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, input_dim)
        """        
        masked_x, mask, ids_restore, ids_keep = self.encoder(x)
        reconstruct_x = self.decoder(masked_x, ids_restore)[0]

        return (reconstruct_x, mask, ids_restore)
    
    def training_step(self, batch, batch_idx):
        x = batch
        reconstruct_x, mask, ids_restore = self(x)

        loss = self.criterion(reconstruct_x, x)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        reconstruct_x, mask, ids_restore = self(x)

        loss = self.criterion(reconstruct_x, x)

        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch
        reconstruct_x, mask, ids_restore = self(x)

        return reconstruct_x, x, mask

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches),
                "interval": "step",
            },
        }