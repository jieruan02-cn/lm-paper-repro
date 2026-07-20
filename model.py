import copy
import math
import torch
import torch.nn as nn
from collections import OrderedDict


class SwishGLU(nn.Module):
    def __init__(self, dim=-1, beta=1.0):
        super().__init__()
        self.dim = dim
        self.beta = beta

    def forward(self, input):
        assert input.size(self.dim) % 2 == 0
        a, b = torch.chunk(input, 2, self.dim)
        if isinstance(self.beta, (int, float)) and self.beta == 1.0:
            return nn.functional.silu(a) * b
        else:
            return a * nn.functional.sigmoid(self.beta * a) * b


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim,
        context_window,
        base_freq=10000.0,
        rope_scaling=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        theta = torch.pow(
            base_freq, torch.arange(0, -head_dim, -2, **config) / head_dim
        )
        self.register_buffer(
            "theta",
            theta if rope_scaling is None else rope_scaling(theta),
            persistent=False,
        )
        self.register_buffer(
            "theta_cos", self.theta.repeat_interleave(2), persistent=False
        )
        self.register_buffer(
            "theta_sin",
            torch.stack((-self.theta, self.theta), dim=1).flatten(),
            persistent=False,
        )
        self.register_buffer(
            "context_vec", torch.arange(context_window, **config), persistent=False
        )
        self.register_buffer(
            "cos_mat",
            torch.cos(torch.outer(self.context_vec, self.theta_cos)),
            persistent=False,
        )
        self.register_buffer(
            "sin_mat",
            torch.sin(torch.outer(self.context_vec, self.theta_sin)),
            persistent=False,
        )
        self.register_buffer(
            "sin_index",
            torch.stack(
                (
                    torch.arange(1, head_dim, 2, device=device),
                    torch.arange(0, head_dim, 2, device=device),
                ),
                dim=1,
            ).flatten(),
            persistent=False,
        )
        self.context_window = context_window

    def forward(self, input):
        length = input.size(-2)
        assert length <= self.context_window
        out = (
            input * self.cos_mat[:length, :]
            + input.index_select(dim=-1, index=self.sin_index)
            * self.sin_mat[:length, :]
        )
        return out


class RoPEMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_groups=None,
        dropout=0.0,
        rope=nn.Identity(),
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_groups

        config = {"device": device, "dtype": dtype}
        self.in_proj_query = nn.Linear(embed_dim, embed_dim, bias=bias, **config)
        if num_groups is None:
            kv_dim = embed_dim
        else:
            assert num_heads % num_groups == 0
            kv_dim = self.head_dim * num_groups
        self.in_proj_key = nn.Linear(embed_dim, kv_dim, bias=bias, **config)
        self.in_proj_value = nn.Linear(embed_dim, kv_dim, bias=bias, **config)
        self.rope = rope
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **config)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        query = self.in_proj_query(query)
        key = self.in_proj_key(key)
        value = self.in_proj_value(value)

        multihead_shape = (-1, self.head_dim)
        query = query.view(query.shape[:-1] + multihead_shape).transpose(-2, -3)
        key = key.view(key.shape[:-1] + multihead_shape).transpose(-2, -3)
        value = value.view(value.shape[:-1] + multihead_shape).transpose(-2, -3)

        out = nn.functional.scaled_dot_product_attention(
            self.rope(query),
            self.rope(key),
            value,
            dropout_p=self.dropout if self.training else 0.0,
            attn_mask=None if is_causal else attn_mask,
            is_causal=is_causal,
            enable_gqa=self.num_groups is not None,
        )
        out = out.transpose(-2, -3)
        out = out.reshape(out.shape[:-2] + (-1,))
        out = self.out_proj(out)
        return out


class RMSTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        ngroup=None,
        dropout=0.1,
        rope=nn.Identity(),
        dim_feedforward=2048,
        activation=SwishGLU,
        rms_norm_eps=1e-05,
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
            embed_dim=d_model,
            num_heads=nhead,
            num_groups=ngroup,
            dropout=dropout,
            rope=rope,
            bias=bias,
            **config,
        )
        self.dropout = nn.Dropout(p=dropout)

        self.rms_norm2 = nn.RMSNorm(d_model, eps=rms_norm_eps, **config)
        self.ffn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "up_proj",
                        nn.Linear(d_model, 2 * dim_feedforward, bias, **config),
                    ),
                    ("act", activation()),
                    ("dropout1", nn.Dropout(p=dropout)),
                    ("down_proj", nn.Linear(dim_feedforward, d_model, bias, **config)),
                    ("dropout2", nn.Dropout(p=dropout)),
                ]
            )
        )

    def forward(self, src, src_mask=None, is_causal=False):
        if self.norm_first:
            mha_in = self.rms_norm1(src)
            mha_out = self.multi_head_attn(
                query=mha_in,
                key=mha_in,
                value=mha_in,
                attn_mask=src_mask,
                is_causal=is_causal,
            )
            out = src + self.dropout(mha_out)
            out = out + self.ffn(self.rms_norm2(out))
        else:
            mha_out = self.multi_head_attn(
                query=src, key=src, value=src, attn_mask=src_mask, is_causal=is_causal
            )
            out = self.rms_norm1(src + self.dropout(mha_out))
            out = self.rms_norm2(out + self.ffn(out))
        return out


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

        res_layer_scaler = 1.0 / math.sqrt(self.NUM_LAYERS * 2)
        for name, param in self.encoder.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(param, std=0.02 * res_layer_scaler)


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

        res_layer_scaler = 1.0 / math.sqrt(2 * self.NUM_LAYERS)
        for name, param in self.encoder.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(param, std=0.02 * res_layer_scaler)


class LLaMA1(nn.Module):
    VOCAB_SIZE = 32000
    CONTEXT_WINDOW = 2048
    MODEL_DIM = 8192
    NUM_HEADS = 64
    # 8 * MODEL_DIM // 3 = 21845 is paper value, round to multiple of 256 for hardware
    # efficiency.
    DIM_FEEDFORWARD = 22016
    NUM_LAYERS = 80

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        assert self.MODEL_DIM % self.NUM_HEADS == 0
        self.rope = RoPE(
            self.MODEL_DIM // self.NUM_HEADS, self.CONTEXT_WINDOW, **config
        )
        # avoid deepcopy with self.rope otherwise it duplicate the rope again.
        self.encoder = nn.ModuleList(
            [
                RMSTransformerEncoderLayer(
                    d_model=self.MODEL_DIM,
                    nhead=self.NUM_HEADS,
                    ngroup=None,
                    dropout=0.0,
                    rope=self.rope,
                    dim_feedforward=self.DIM_FEEDFORWARD,
                    activation=SwishGLU,
                    rms_norm_eps=1e-05,
                    norm_first=True,
                    bias=False,
                    **config,
                )
                for _ in range(self.NUM_LAYERS)
            ],
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
        out = self.embedding(input)
        for layer in self.encoder:
            # don't use buffer mask and instead relies on SDPA's optimized version,
            # otherwise will OOM for large context window.
            out = layer(src=out, is_causal=True)
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

        res_layer_scaler = 1.0 / math.sqrt(2 * self.NUM_LAYERS)
        for layer in self.encoder:
            nn.init.normal_(
                layer.multi_head_attn.out_proj.weight, std=0.02 * res_layer_scaler
            )
            nn.init.normal_(layer.ffn.down_proj.weight, std=0.02 * res_layer_scaler)


# Main difference from LLaMA1 is context window and usage of GQA.
class LLaMA2(nn.Module):
    VOCAB_SIZE = 32000
    CONTEXT_WINDOW = 4096
    MODEL_DIM = 8192
    NUM_HEADS = 64
    NUM_GROUPS = 8
    # scale the dim_feedforward by 1.33, MODEL_DIM * 8 / 3 * 1.33
    DIM_FEEDFORWARD = 28672
    NUM_LAYERS = 80

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        assert self.MODEL_DIM % self.NUM_HEADS == 0
        self.rope = RoPE(
            self.MODEL_DIM // self.NUM_HEADS, self.CONTEXT_WINDOW, **config
        )
        self.encoder = nn.ModuleList(
            [
                RMSTransformerEncoderLayer(
                    d_model=self.MODEL_DIM,
                    nhead=self.NUM_HEADS,
                    ngroup=self.NUM_GROUPS,
                    dropout=0.0,
                    rope=self.rope,
                    dim_feedforward=self.DIM_FEEDFORWARD,
                    activation=SwishGLU,
                    rms_norm_eps=1e-05,
                    norm_first=True,
                    bias=False,
                    **config,
                )
                for _ in range(self.NUM_LAYERS)
            ]
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
        out = self.embedding(input)
        for layer in self.encoder:
            out = layer(src=out, is_causal=True)
        out = self.linear(self.rms_norm(out))
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        res_layer_scaler = 1.0 / math.sqrt(2 * self.NUM_LAYERS)
        for layer in self.encoder:
            nn.init.normal_(
                layer.multi_head_attn.out_proj.weight, std=0.02 * res_layer_scaler
            )
            nn.init.normal_(layer.ffn.down_proj.weight, std=0.02 * res_layer_scaler)


class LLaMA3(nn.Module):
    VOCAB_SIZE = 128256
    CONTEXT_WINDOW = 131072
    MODEL_DIM = 16384
    NUM_HEADS = 128
    NUM_GROUPS = 8
    DIM_FEEDFORWARD = 53248
    NUM_LAYERS = 126

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.MODEL_DIM, **config)
        assert self.MODEL_DIM % self.NUM_HEADS == 0
        self.rope = RoPE(
            self.MODEL_DIM // self.NUM_HEADS,
            self.CONTEXT_WINDOW,
            base_freq=500000.0,
            rope_scaling=LLaMA3.rope_scaling,
            **config,
        )
        self.encoder = nn.ModuleList(
            [
                RMSTransformerEncoderLayer(
                    d_model=self.MODEL_DIM,
                    nhead=self.NUM_HEADS,
                    ngroup=self.NUM_GROUPS,
                    dropout=0.0,
                    rope=self.rope,
                    dim_feedforward=self.DIM_FEEDFORWARD,
                    activation=SwishGLU,
                    rms_norm_eps=1e-05,
                    norm_first=True,
                    bias=False,
                    **config,
                )
                for _ in range(self.NUM_LAYERS)
            ]
        )
        self.rms_norm = nn.RMSNorm(self.MODEL_DIM, eps=1e-05, **config)
        self.linear = nn.Linear(self.MODEL_DIM, self.VOCAB_SIZE, bias=False, **config)
        self.reset_parameters()

    def forward(self, input):
        out = self.embedding(input)
        for layer in self.encoder:
            out = layer(src=out, is_causal=True)
        out = self.linear(self.rms_norm(out))
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        res_layer_scaler = 1.0 / math.sqrt(2.0 * self.NUM_LAYERS)
        for layer in self.encoder:
            nn.init.normal_(
                layer.multi_head_attn.out_proj.weight, std=0.02 * res_layer_scaler
            )
            nn.init.normal_(layer.ffn.down_proj.weight, std=0.02 * res_layer_scaler)

    @staticmethod
    def rope_scaling(theta):
        # LLaMA3.1 frequency scaling, extends the 8192 pretraining context to
        # 131072. Not in the paper, see apply_scaling in meta-llama/llama-models.
        # Its grid searched constants are scale_factor 8, low_freq_factor 1,
        # high_freq_factor 4 and original context 8192, giving 3 branches on period
        # with s = (8192 / period - low_freq_factor) / (high_freq_factor - low_freq_factor):
        #   period > 8192 / low_freq_factor   -> theta / scale_factor
        #   period < 8192 / high_freq_factor  -> theta
        #   otherwise                         -> (1 - s) * theta / scale_factor + s * theta
        period = math.pi * 2 / theta
        # Clamping s to [0, 1] collapses all 3 branches into the interpolation.
        smooth = torch.clamp((8192 / period - 1.0) / 3.0, min=0.0, max=1.0)
        scaled_theta = theta * (0.125 + 0.875 * smooth)
        return scaled_theta


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
