import torch
import torch.nn as nn


class Transfomer(nn.Module):
    def __init__(self):
        pass


class T5(nn.Module):
    def __init__(self):
        pass


class BERT(nn.Module):
    def __init__(self):
        pass


class GPT1(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation=nn.gelu,
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
            **config,
        )
        self.model = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=12)
        self.embedding = nn.Embedding()

    def forward(self, input):
        pass


class GPT2(nn.Module):
    def __init__(self):
        pass


class GPT3(nn.Module):
    def __init__(self):
        pass


class LLaMA1(nn.Module):
    def __init__(self):
        pass


class LLaMA2(nn.Module):
    def __init__(self):
        pass


class LLaMA3(nn.Module):
    def __init__(self):
        pass


class PaLM(nn.Module):
    def __init__(self):
        pass


class DeepSeekV2(nn.Module):
    def __init__(self):
        pass


class DeepSeekV3(nn.Module):
    def __init__(self):
        pass


class DeepSeekV4(nn.Module):
    def __init__(self):
        pass


class Mistral(nn.Module):
    def __init__(self):
        pass


class Mixtral(nn.Module):
    def __init__(self):
        pass


class Mamba(nn.Module):
    def __init__(self):
        pass


class Chinchilla(nn.Module):
    def __init__(self):
        pass


class Gopher(nn.Module):
    def __init__(self):
        pass


class Qwen(nn.Module):
    def __init__(self):
        pass


class Qwen2(nn.Module):
    def __init__(self):
        pass


class Qwen3(nn.Module):
    def __init__(self):
        pass


class Gemma(nn.Module):
    def __init__(self):
        pass
