import torch.nn as nn

ACT_FUNCS = {
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(inplace=True),
    "silu": nn.SiLU(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU()
}

def get_activation_fn(activation: str) -> nn.Module:
    if activation in ACT_FUNCS:
        return ACT_FUNCS[activation]
    else:
        raise ValueError(f"Activation function '{activation}' not found")