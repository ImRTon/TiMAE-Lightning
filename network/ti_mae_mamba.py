import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torch.optim import AdamW, Adam, lr_scheduler
from torchmetrics import Accuracy, F1Score
from pathlib import Path
from mamba_ssm import Mamba2

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

class TiMAEMambaEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        num_heads: int,
        emb_size: int,
        masking: bool,
        cls_embed: bool = True,
        expand_factor: int = 2,
        ssm_state_size: int = 128,
        conv_size: int = 4,
        bias: bool = False,
        conv_bias: bool = True,
        layer_norm_eps: float = 1e-5,
        mask_ratio: float = 0.75
    ):
        super().__init__()

        self.masking = masking
        self.cls_embed = cls_embed
        self.mask_ratio = mask_ratio

        # Prepare Mamba parameters
        assert ssm_state_size % num_heads == 0, "SSM state size should be divisible by the number of heads"
        self.headdim = ssm_state_size // num_heads

        self.embedding = TiMAEEmbedding(input_dim, emb_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.mamba = Mamba2(
            d_model=emb_size,
            d_state=ssm_state_size,
            d_conv=conv_size,
            expand=expand_factor,
            headdim=self.headdim,
            d_ssm=None,
            bias=bias,
            conv_bias=conv_bias,
        )

        self.norm = nn.LayerNorm(emb_size, eps=layer_norm_eps)

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
            last_hidden_state (torch.Tensor): a tensor of shape (batch_size, seq_len, emb_dim)
            mask (torch.LongTensor): a tensor of shape (batch_size, seq_len)
            ids_restore (torch.LongTensor): a tensor of shape (batch_size, seq_len) 
                containing the original index of the (shuffled) masked patches.
            ids_keep (torch.LongTensor): a tensor of shape (batch_size, seq_len)
        """        
        bs, seq_len, input_dim = x.shape

        # Input Embedding
        x = self.embedding(x)

        # Append cls token
        if self.cls_embed:
            cls_tokens = self.cls_token.expand(bs, -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)

        # Mamba
        x = self.mamba(x)

        # Normalization
        x = self.norm(x)
        
        # Random masking
        if self.masking:
            if self.cls_embed:
                cls_token = x[:, -1:, :]
            x, mask, ids_restore, ids_keep = self.random_masking(x[:, :-1, :], self.mask_ratio)
            if self.cls_embed:
                x = torch.cat((x, cls_token), dim=1)
        
            return (x, mask, ids_restore, ids_keep,)

        return x

class TiMAEMambaDecoder(nn.Module):
    def __init__(
        self, 
        output_dim: int,
        num_heads: int,
        emb_size: int,
        encoder_seq_len: int,
        encoder_emb_size: int,
        encoder_cls_embed: bool = True,
        expand_factor: int = 2,
        ssm_state_size: int = 128,
        conv_size: int = 4,
        bias: bool = False,
        conv_bias: bool = True,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        self.encoder_cls_embed = encoder_cls_embed
        self.encoder_seq_len = encoder_seq_len

        # Prepare Mamba parameters
        assert ssm_state_size % num_heads == 0, "SSM state size should be divisible by the number of heads"
        self.headdim = ssm_state_size // num_heads

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.embedding = nn.Linear(encoder_emb_size, emb_size, bias=True)

        self.mamba = Mamba2(
            d_model=emb_size,
            d_state=ssm_state_size,
            d_conv=conv_size,
            expand=expand_factor,
            headdim=self.headdim,
            d_ssm=None,
            bias=bias,
            conv_bias=conv_bias,
        )

        self.norm = nn.LayerNorm(emb_size, eps=layer_norm_eps)

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
            x_ = torch.cat([x[:, :-1, :], mask_tokens], dim=1)  # Remove encoder cls token
        else:
            x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        # unshuffle
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        if self.encoder_cls_embed:
            x_ = torch.cat([x_, x[:, -1:, :]], dim=1) # Append encoder cls token

        # Mamba Decoder
        x = self.mamba(x_)

        x = self.norm(x)

        # Projection
        x = self.projection(x)
        x = x[:, :-1, :]  # Remove cls token
        
        return x

class TiMAEMamba(L.LightningModule):
    def __init__(
        self,
        lr: float,
        num_classes: int,
        input_dim: int,
        num_heads: int,
        emb_size: int,
        cls_embed: bool = True,
        expand_factor: int = 2,
        ssm_state_size: int = 128,
        conv_size: int = 4,
        bias: bool = False,
        conv_bias: bool = True,
        layer_norm_eps: float = 1e-5,
        pooling: str | None = None,
        encoder_ckpt_path: str | Path | None = None,
        class_weights: list[float] | None = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights', 'num_classes'])

        self.encoder = TiMAEMambaEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            emb_size=emb_size,
            masking=False,
            cls_embed=cls_embed,
            expand_factor=expand_factor,
            ssm_state_size=ssm_state_size,
            conv_size=conv_size,
            bias=bias,
            conv_bias=conv_bias,
            layer_norm_eps=layer_norm_eps,
            mask_ratio=0
        )
        if encoder_ckpt_path is not None:
            print(f"==== Loading encoder weights from {encoder_ckpt_path} ====")
            with open(encoder_ckpt_path, 'rb') as f:
                # Create new state_dict, including only encoder weights
                new_state_dict = {}
                for key, value in torch.load(f)["state_dict"].items():
                    if key.startswith("encoder."):
                        new_key = key[len("encoder."):]  # Remove "encoder." prefix
                        new_state_dict[new_key] = value
                missing_keys = self.encoder.load_state_dict(new_state_dict, strict=True)
                print("Missing keys:", missing_keys)

        self.classifier = nn.Linear(emb_size, num_classes, bias=True)

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights) if class_weights is not None else None)

        self.train_accu = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accu = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_accu = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

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
        x = self.encoder(x)
        
        if self.hparams.cls_embed:
            x = x[:, -1, :]
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

class TiMAEMambaForPretraining(L.LightningModule):
    def __init__(
        self, 
        lr: float,
        input_dim: int,
        num_heads: int,
        num_decoder_heads: int,
        emb_size: int,
        decoder_emb_size: int,
        seq_len: int,
        cls_embed: bool = True,
        expand_factor: int = 2,
        decoder_expand_factor: int = 2,
        ssm_state_size: int = 128,
        decoder_ssm_state_size: int = 128,
        conv_size: int = 4,
        decoder_conv_size: int = 4,
        bias: bool = False,
        conv_bias: bool = True,
        layer_norm_eps: float = 1e-5,
        mask_ratio: float = 0.75
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TiMAEMambaEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            emb_size=emb_size,
            masking=True,
            cls_embed=cls_embed,
            expand_factor=expand_factor,
            ssm_state_size=ssm_state_size,
            conv_size=conv_size,
            bias=bias,
            conv_bias=conv_bias,
            layer_norm_eps=layer_norm_eps,
            mask_ratio=mask_ratio
        )

        self.decoder = TiMAEMambaDecoder(
            output_dim=input_dim,
            num_heads=num_decoder_heads,
            emb_size=decoder_emb_size,
            encoder_seq_len=seq_len,
            encoder_emb_size=emb_size,
            encoder_cls_embed=cls_embed,
            expand_factor=decoder_expand_factor,
            ssm_state_size=decoder_ssm_state_size,
            conv_size=decoder_conv_size,
            bias=bias,
            conv_bias=conv_bias,
            layer_norm_eps=layer_norm_eps
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
        reconstruct_x = self.decoder(masked_x, ids_restore)

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