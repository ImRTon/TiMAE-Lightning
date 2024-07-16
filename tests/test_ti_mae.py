import torch

from network.ti_mae import TiMAE, TiMAEEncoder, TiMAEDecoder, TiMAEForPretraining, TiMAEEmbedding

class TiMAEEncoderTester:
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
        self.input_dim = input_dim
        self.emb_size = emb_size
        self.cls_embed = cls_embed
        self.mask_ratio = mask_ratio

        self.encoder = TiMAEEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            emb_size=emb_size,
            intermediate_dim=intermediate_dim,
            masking=masking,
            cls_embed=cls_embed,
            max_len=max_len,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func,
            mask_ratio=mask_ratio
        )

    def test_model(self, batch_size: int, seq_len: int):
        x = torch.rand(batch_size, seq_len, self.input_dim)
        outputs = self.encoder(x)
        unmasked_token_len = int(seq_len * (1 - self.mask_ratio))
        if self.cls_embed:
            unmasked_token_len += 1
        assert outputs[0].shape == (batch_size, unmasked_token_len, self.emb_size), \
            f"Invalid output shape: {outputs[0].shape} != {(batch_size, unmasked_token_len, self.emb_size)}"
        print(outputs[0].shape, (batch_size, unmasked_token_len, self.emb_size))
        return outputs

class TiMAEDecoderTester:
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
        self.output_dim = output_dim

        self.decoder = TiMAEDecoder(
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            emb_size=emb_size,
            encoder_emb_size=encoder_emb_size,
            encoder_seq_len=encoder_seq_len,
            intermediate_dim=intermediate_dim,
            encoder_cls_embed=encoder_cls_embed,
            max_len=max_len,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func
        )

    def test_model(
        self, 
        batch_size: int, 
        seq_len: int,
        x: torch.Tensor, 
        ids_restore: torch.LongTensor
    ):
        outputs = self.decoder(x, ids_restore)
        assert outputs[0].shape == (batch_size, seq_len, self.output_dim), \
            f"Invalid output shape: {outputs[0].shape} != {(batch_size, seq_len, self.output_dim)}"
        print(outputs[0].shape, (batch_size, seq_len, self.output_dim))

class TiMAEForPretrainingTester:
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
        self.model = TiMAEForPretraining(
            lr=lr,
            input_dim=input_dim,
            num_heads=num_heads,
            num_decoder_heads=num_decoder_heads,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            emb_size=emb_size,
            decoder_emb_size=decoder_emb_size,
            intermediate_dim=intermediate_dim,
            decoder_intermediate_dim=decoder_intermediate_dim,
            cls_embed=cls_embed,
            seq_len=seq_len,
            max_len=max_len,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func,
            mask_ratio=mask_ratio
        )

    def test_model(self, batch_size: int, seq_len: int):
        x = torch.rand(batch_size, seq_len, self.model.hparams.input_dim)
        outputs = self.model(x)
        assert outputs[0].shape == (batch_size, seq_len, self.model.hparams.input_dim), \
            f"Invalid output shape: {outputs[0].shape} != {(batch_size, seq_len, self.model.hparams.input_dim)}"
        print(outputs[0].shape, (batch_size, seq_len, self.model.hparams.input_dim))

if __name__ == "__main__":
    encoder_tester = TiMAEEncoderTester(
        input_dim=3,
        num_heads=16,
        num_layers=6,
        emb_size=128,
        intermediate_dim=512,
        masking=True,
        cls_embed=True,
        max_len=100,
        qkv_bias=False,
        attn_dropout_prob=0.1,
        ffn_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        act_func='gelu'
    )
    decoder_tester = TiMAEDecoderTester(
        output_dim=3,
        num_heads=16,
        num_layers=6,
        emb_size=64,
        encoder_emb_size=128,
        encoder_seq_len=100,
        intermediate_dim=256,
        encoder_cls_embed=True,
        max_len=100,
        qkv_bias=False,
        attn_dropout_prob=0.1,
        ffn_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        act_func='gelu'
    )
    outputs = encoder_tester.test_model(batch_size=2, seq_len=8)
    decoder_tester.test_model(batch_size=2, seq_len=8, x=outputs[0], ids_restore=outputs[2])

    ti_mae_tester = TiMAEForPretrainingTester(
        lr=1e-3,
        input_dim=3,
        num_heads=4,
        num_decoder_heads=4,
        num_layers=2,
        num_decoder_layers=2,
        emb_size=64,
        decoder_emb_size=32,
        intermediate_dim=128,
        decoder_intermediate_dim=64,
        cls_embed=True,
        seq_len=100,
        max_len=None,
        qkv_bias=False,
        attn_dropout_prob=0.1,
        ffn_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        act_func='gelu',
        mask_ratio=0.75
    )
    ti_mae_tester.test_model(batch_size=2, seq_len=8)