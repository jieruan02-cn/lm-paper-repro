import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        pass


class T5(nn.Module):
    def __init__(self):
        pass


class BERT(nn.Module):
    def __init__(self):
        pass


class GPT1(nn.Module):
    # BPE with 40000 merges
    VOCAB_SIZE = 40478
    CONTEXT_WINDOW = 512
    MODEL_DIM = 768
    NUM_HEADS = 12
    DIM_FEEDFORWARD = 3072
    NUM_LAYERS = 12

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        self.position_embedding = nn.Parameter(
            torch.empty(self.CONTEXT_WINDOW, self.MODEL_DIM, **config)
        )
        self.dropout = nn.Dropout(p=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.MODEL_DIM,
                nhead=self.NUM_HEADS,
                dim_feedforward=self.DIM_FEEDFORWARD,
                dropout=0.1,
                activation=nn.functional.gelu,
                layer_norm_eps=1e-05,
                batch_first=True,
                norm_first=False,
                **config,
            ),
            num_layers=self.NUM_LAYERS,
        )
        self.register_buffer(
            "mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.CONTEXT_WINDOW, device=device
            ),
            persistent=False,
        )

        self.reset_parameters()

    def forward(self, input):
        length = input.size(-1)
        assert length <= self.CONTEXT_WINDOW
        out = self.dropout(self.embedding(input) + self.position_embedding[:length, :])
        out = self.encoder(out, mask=self.mask[:length, :length], is_causal=True)
        out = out @ self.embedding.weight.T
        return out

    def reset_parameters(self):
        nn.init.normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                nn.init.zeros_(module.in_proj_bias)


class GPT2(nn.Module):
    VOCAB_SIZE = 50257
    CONTEXT_WINDOW = 1024
    MODEL_DIM = 768
    NUM_HEADS = 12
    DIM_FEEDFORWARD = 3072
    NUM_LAYERS = 12

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        self.position_embedding = nn.Parameter(
            torch.empty(self.CONTEXT_WINDOW, self.MODEL_DIM, **config)
        )
        self.dropout = nn.Dropout(p=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.MODEL_DIM,
                nhead=self.NUM_HEADS,
                dim_feedforward=self.DIM_FEEDFORWARD,
                dropout=0.1,
                activation=nn.functional.gelu,
                layer_norm_eps=1e-05,
                batch_first=True,
                norm_first=True,
                **config,
            ),
            num_layers=self.NUM_LAYERS,
        )
        self.layer_norm = nn.LayerNorm(self.MODEL_DIM, eps=1e-05, bias=True, **config)
        self.register_buffer(
            "mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.CONTEXT_WINDOW, device=device
            ),
            persistent=False,
        )
        self.reset_parameters()

    def forward(self, input):
        length = input.size(-1)
        assert length <= self.CONTEXT_WINDOW
        out = self.dropout(self.embedding(input) + self.position_embedding[:length, :])
        out = self.encoder(out, mask=self.mask[:length, :length], is_causal=True)
        out = self.layer_norm(out)
        out = out @ self.embedding.weight.T
        return out

    def reset_parameters(self):
        nn.init.normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                nn.init.zeros_(module.in_proj_bias)

        residual_layer_scaler = 1.0 / math.sqrt(self.NUM_LAYERS * 2)
        for name, param in self.encoder.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(param, std=0.02 * residual_layer_scaler)


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
