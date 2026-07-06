import copy
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
    VOCAB_SIZE = 4096
    CONTEXT_WINDOW = 4096
    MODEL_DIM = 768
    NUM_HEAD = 12
    DIM_FEEDFORWARD = 3072
    NUM_LAYER = 12

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM
        )
        self.position_embedding = nn.Parameter(
            torch.empty(self.CONTEXT_WINDOW, self.MODEL_DIM, **config)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.MODEL_DIM,
            nhead=self.NUM_HEAD,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=0.1,
            activation=nn.functional.gelu,
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
            **config,
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.NUM_LAYER)]
        )
        self.reset_parameters()

    def forward(self, input):
        length = input.size(-1)
        out = self.embedding(input) + self.position_embedding[:length, :]
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=out.device)
        for encoder_layer in self.encoder:
            out = encoder_layer(out, src_mask=mask, is_causal=True)
        return out

    def reset_parameters(self):
        nn.init.normal_(self.position_embedding, std=0.02)


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
