import torch

from network.transformer import ScaledDotProductAttention, MultiHeadAttention, TransformerEncoderLayer

class ScaledDotProductAttentionTester:
    def __init__(
        self,
        scale: float | None,
        attn_dropout_prob: float
    ):
        self.scaled_dot_product_attention = ScaledDotProductAttention(scale=scale, attn_dropout_prob=attn_dropout_prob)

    def test_model(self, batch_size: int, seq_len: int, d_model: int):
        print("==[Testing]==\nScaledDotProductAttention with Single Head")
        q = torch.rand(batch_size, seq_len, d_model)
        k = torch.rand(batch_size, seq_len, d_model)
        v = torch.rand(batch_size, seq_len, d_model)
        mask = torch.tril(torch.ones(seq_len, seq_len)).repeat(batch_size, 1, 1)
        output_attentions = True
        outputs = self.scaled_dot_product_attention(q, k, v, mask, output_attentions)
        assert outputs[0].shape == (batch_size, seq_len, d_model), \
            f"Output of ScaledDotProductAttention {outputs[0].shape} is not as" \
            f"expected {(batch_size, seq_len, d_model)}"
        assert outputs[1].shape == (batch_size, seq_len, seq_len), \
            f"Attention scores of ScaledDotProductAttention {outputs[1].shape}" \
            f" is not as expected {(batch_size, seq_len, seq_len)}"
        print("==[Complete]==")

class MultiHeadAttentionTester:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qkv_bias: bool,
        attn_dropout_prob: float
    ):
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob
        )

    def test_model(self, batch_size: int, seq_len: int, d_model: int):
        q = torch.rand(batch_size, seq_len, d_model)
        k = torch.rand(batch_size, seq_len, d_model)
        v = torch.rand(batch_size, seq_len, d_model)
        mask = torch.randint(0, 2, (batch_size, self.num_heads, seq_len, seq_len), dtype=torch.bool)
        mask = torch.tril(torch.ones(seq_len, seq_len)).repeat(batch_size, self.num_heads, 1, 1)
        outputs = self.multi_head_attention(q, k, v, mask, output_attentions=True)
        print(outputs[0].shape, outputs[1].shape)

class TransformerEncoderLayerTester:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        intermediate_dim: int,
        qkv_bias: bool,
        attn_dropout_prob: float,
        ffn_dropout_prob: float,
        norm_first: bool,
        layer_norm_eps: float,
        act_func: str
    ):
        self.layer = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            qkv_bias=qkv_bias,
            attn_dropout_prob=attn_dropout_prob,
            ffn_dropout_prob=ffn_dropout_prob,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            act_func=act_func
        )

    def test_model(self, batch_size: int, seq_len: int, d_model: int):
        x = torch.rand(batch_size, seq_len, d_model)
        mask = torch.tril(torch.ones(seq_len, seq_len)).repeat(batch_size, 1, 1)
        outputs = self.layer(x, mask, output_attentions=True)
        print(outputs[0].shape, outputs[1].shape)

if __name__ == "__main__":
    tester = ScaledDotProductAttentionTester(scale=64, attn_dropout_prob=0.0)
    tester.test_model(batch_size=2, seq_len=8, d_model=128)

    tester = MultiHeadAttentionTester(d_model=128, num_heads=16, qkv_bias=False, attn_dropout_prob=0.0)
    tester.test_model(batch_size=2, seq_len=8, d_model=128)

    tester = TransformerEncoderLayerTester(
        d_model=128,
        num_heads=16,
        intermediate_dim=64,
        qkv_bias=False,
        attn_dropout_prob=0.0,
        ffn_dropout_prob=0.0,
        norm_first=True,
        layer_norm_eps=1e-5,
        act_func="gelu"
    )
    tester.test_model(batch_size=2, seq_len=8, d_model=128)