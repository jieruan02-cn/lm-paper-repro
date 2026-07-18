import copy
import math
import torch
import torch.nn as nn


class SwishGLU(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass


class RoPE(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass


class RoPEMultiheadAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True, device=None, dtype=None
    ):
        pass

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        pass


class RMSTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=SwishGLU,
        rms_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.norm_first = norm_first
        self.rms_norm1 = nn.RMSNorm(d_model, eps=rms_norm_eps, **config)
        self.multi_head_attn = RoPEMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=bias, **config
        )
        self.dropout = nn.Dropout(p=dropout)

        self.rms_norm2 = nn.RMSNorm(d_model, eps=rms_norm_eps, **config)
        self.ffn = nn.Sequential(
            nn.Linear(),
            activation(),
            nn.Dropout(p=dropout),
            nn.Linear(),
            nn.Dropout(p=dropout),
        )

    def forward(self, src, src_mask=None, is_causal=False):
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
    VOCAB_SIZE = 50257
    CONTEXT_WINDOW = 2048
    MODEL_DIM = 12288
    NUM_HEADS = 96
    DIM_FEEDFORWARD = 4 * MODEL_DIM
    NUM_LAYERS = 96
    SPARSE_BAND = 128

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.MODEL_DIM,
            nhead=self.NUM_HEADS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=0.1,
            activation=nn.functional.gelu,
            batch_first=True,
            norm_first=True,
            **config,
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.NUM_LAYERS)]
        )
        self.layer_norm = nn.LayerNorm(self.MODEL_DIM, **config)
        self.register_buffer(
            "dense_mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.CONTEXT_WINDOW, **config
            ),
            persistent=False,
        )
        self.register_buffer(
            "sparse_mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.CONTEXT_WINDOW, **config
            )
            + torch.full(
                (self.CONTEXT_WINDOW, self.CONTEXT_WINDOW), -torch.inf, **config
            ).tril_(diagonal=-self.SPARSE_BAND),
            persistent=False,
        )
        self.reset_parameters()

    def forward(self, input):
        length = input.size(-1)
        assert length <= self.CONTEXT_WINDOW
        out = self.dropout(self.embedding(input) + self.position_embedding[:length, :])
        for i, layer in enumerate(self.encoder):
            is_dense = i % 2 == 0
            mask = self.dense_mask if is_dense else self.sparse_mask
            out = layer(out, src_mask=mask[:length, :length], is_causal=is_dense)
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

        residual_layer_scaler = 1.0 / math.sqrt(2 * self.NUM_LAYERS)
        for name, param in self.encoder.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(param, std=0.02 * residual_layer_scaler)


class LLaMA1(nn.Module):
    VOCAB_SIZE = 32000
    CONTEXT_WINDOW = 2048
    MODEL_DIM = 8192
    NUM_HEADS = 64
    DIM_FEEDFORWARD = 8 * MODEL_DIM // 3
    NUM_LAYERS = 80

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        encoder_layer = RMSTransformerEncoderLayer(
            d_model=self.MODEL_DIM,
            nhead=self.NUM_HEADS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=0.0,
            activation=SwishGLU,
            rms_norm_eps=1e-05,
            batch_first=True,
            norm_first=True,
            **config,
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.NUM_LAYERS)],
        )
        self.register_buffer(
            "mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.CONTEXT_WINDOW, device=device
            ),
            persistent=False,
        )
        self.rms_norm = nn.RMSNorm(self.MODEL_DIM, eps=1e-05, **config)
        self.linear = nn.Linear(
            in_features=self.MODEL_DIM,
            out_features=self.VOCAB_SIZE,
            bias=False,
            **config,
        )
        self.reset_parameters()

    def forward(self, input):
        length = input.size(-1)
        assert length <= self.CONTEXT_WINDOW
        out = self.embedding(input)
        for layer in self.encoder:
            out = layer(src=out, mask=self.mask[:length, :length], is_causal=True)
        out = self.linear(self.rms_norm(out))
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        residual_layer_scaler = 1.0 / math.sqrt(2 * self.NUM_LAYERS)
        for layer in self.encoder:
            nn.init.normal_(
                layer.multi_head_attn.out_proj.weight, std=0.02 * residual_layer_scaler
            )
            nn.init.normal_(
                layer.ffn.down_proj.weight, std=0.02 * residual_layer_scaler
            )


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
