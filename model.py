# TODO(jieruan): Adapt the models for KV-cached inference mode.
import copy
import math
import functools
import torch
import torch.nn as nn


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
        d_head=None,
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
        if d_head is None:
            assert embed_dim % num_heads == 0
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = d_head
        attn_dim = self.head_dim * self.num_heads
        self.num_groups = num_groups

        config = {"device": device, "dtype": dtype}
        self.in_proj_query = nn.Linear(embed_dim, attn_dim, bias=bias, **config)
        if num_groups is None:
            kv_dim = attn_dim
        else:
            assert num_heads % num_groups == 0
            kv_dim = self.head_dim * num_groups
        self.in_proj_key = nn.Linear(embed_dim, kv_dim, bias=bias, **config)
        self.in_proj_value = nn.Linear(embed_dim, kv_dim, bias=bias, **config)
        self.rope = rope
        self.out_proj = nn.Linear(attn_dim, embed_dim, bias=bias, **config)

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


class T5(nn.Module):
    def __init__(self):
        pass


class BERT(nn.Module):
    def __init__(self):
        pass


class GPTBase(nn.Module):
    VOCAB_SIZE = 40478
    CONTEXT_WINDOW = 512
    MODEL_DIM = 768
    NUM_HEADS = 12
    DIM_FEEDFORWARD = 4 * MODEL_DIM
    NUM_LAYERS = 12
    DROPOUT = 0.1  # GPT-1 §4.1, GPT-2; GPT-3 never states a dropout rate
    LAYER_NORM_EPS = 1e-05  # unstated in all three papers; torch/HF default
    SPARSE_BAND = 128  # GPT-3 §2.1 "locally banded sparse"; width unstated

    def __init__(
        self,
        norm_first=False,
        alternate_sparse_attn=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first

        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        self.position_embedding = nn.Parameter(
            torch.empty(self.CONTEXT_WINDOW, self.MODEL_DIM, **config)
        )
        self.dropout = nn.Dropout(p=self.DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.MODEL_DIM,
            nhead=self.NUM_HEADS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
            # OpenAI shipped tanh approx
            activation=functools.partial(nn.functional.gelu, approximate="tanh"),
            layer_norm_eps=self.LAYER_NORM_EPS,
            batch_first=True,
            norm_first=norm_first,
            **config,
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.NUM_LAYERS)]
        )
        self.register_buffer(
            "dense_mask",
            nn.Transformer.generate_square_subsequent_mask(
                self.CONTEXT_WINDOW, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "sparse_mask",
            self.dense_mask
            + torch.full(
                (self.CONTEXT_WINDOW, self.CONTEXT_WINDOW), -torch.inf, **config
            ).tril_(diagonal=-self.SPARSE_BAND)
            if alternate_sparse_attn
            else None,
            persistent=False,
        )
        if norm_first:
            self.layer_norm = nn.LayerNorm(
                self.MODEL_DIM, eps=self.LAYER_NORM_EPS, bias=True, **config
            )
        else:
            self.register_module("layer_norm", None)
        self.reset_parameters()

    def forward(self, input):
        length = input.size(-1)
        assert length <= self.CONTEXT_WINDOW

        out = self.embedding(input) + self.position_embedding[:length, :]
        out = self.dropout(out)
        for i, layer in enumerate(self.encoder):
            is_dense = i % 2 == 0 or self.sparse_mask is None
            mask = self.dense_mask if is_dense else self.sparse_mask
            out = layer(out, src_mask=mask[:length, :length], is_causal=is_dense)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = out @ self.embedding.weight.T
        return out

    def reset_parameters(self):
        # wpe std unstated; 0.02 per HF, OpenAI's released GPT-2 used 0.01
        nn.init.normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            # need additional branch to cover MHA as it is not covered by Linear.
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                nn.init.zeros_(module.in_proj_bias)

        res_out_std = 0.02 / math.sqrt(2 * self.NUM_LAYERS)
        if self.norm_first:
            for name, param in self.encoder.named_parameters():
                if name.endswith("out_proj.weight") or name.endswith("linear2.weight"):
                    nn.init.normal_(param, std=res_out_std)


class GPT1(GPTBase):
    def __init__(self, device=None, dtype=None):
        super().__init__(
            norm_first=False,
            alternate_sparse_attn=False,
            device=device,
            dtype=dtype,
        )


class GPT2(GPTBase):
    VOCAB_SIZE = 50257
    CONTEXT_WINDOW = 1024
    MODEL_DIM = 1600
    NUM_HEADS = 25
    DIM_FEEDFORWARD = 4 * MODEL_DIM
    NUM_LAYERS = 48

    def __init__(self, device=None, dtype=None):
        super().__init__(
            norm_first=True,
            alternate_sparse_attn=False,
            device=device,
            dtype=dtype,
        )


class GPT3(GPTBase):
    VOCAB_SIZE = 50257
    CONTEXT_WINDOW = 2048
    MODEL_DIM = 12288
    NUM_HEADS = 96
    DIM_FEEDFORWARD = 4 * MODEL_DIM
    NUM_LAYERS = 96

    def __init__(self, device=None, dtype=None):
        super().__init__(
            norm_first=True,
            alternate_sparse_attn=True,
            device=device,
            dtype=dtype,
        )


class TransformerFFNBlock(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        bias=False,
        activation=nn.ReLU,
        dropout=0.0,
        dim_feedforward1=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward1 if dim_feedforward1 else dim_feedforward,
            bias=bias,
            **config,
        )
        self.activation = activation()
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **config)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input):
        out = self.linear1(input)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out


class LLaMADecoderLayer(nn.Module):
    BIAS = False
    ACTIVATION = SwishGLU
    DROPOUT = 0.0

    def __init__(
        self,
        d_model,
        nhead,
        ngroup,
        rope,
        dim_feedforward,
        rms_norm_eps,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.rms_norm1 = nn.RMSNorm(d_model, eps=rms_norm_eps, **config)
        self.multi_head_attn = RoPEMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_groups=ngroup,
            dropout=self.DROPOUT,
            rope=rope,
            bias=self.BIAS,
            **config,
        )

        self.rms_norm2 = nn.RMSNorm(d_model, eps=rms_norm_eps, **config)
        self.ffn = TransformerFFNBlock(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            bias=self.BIAS,
            activation=self.ACTIVATION,
            dropout=self.DROPOUT,
            dim_feedforward1=2 * dim_feedforward,
            **config,
        )

    def forward(self, input, is_causal):
        mha_in = self.rms_norm1(input)
        out = input + self.multi_head_attn(
            query=mha_in,
            key=mha_in,
            value=mha_in,
            is_causal=is_causal,
        )
        out = out + self.ffn(self.rms_norm2(out))
        return out


class LLaMABase(nn.Module):
    VOCAB_SIZE = 32000
    CONTEXT_WINDOW = 2048
    MODEL_DIM = 8192
    NUM_HEADS = 64
    NUM_GROUPS = None
    # 8 * MODEL_DIM // 3 = 21845 is paper value, round to multiple of 256 for hardware
    # efficiency.
    DIM_FEEDFORWARD = 22016
    NUM_LAYERS = 80
    ROPE_BASE_FREQ = 10000
    RMS_NORM_EPS = 1e-05

    def __init__(self, rope_scaling=None, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        assert self.MODEL_DIM % self.NUM_HEADS == 0
        self.rope = RoPE(
            self.MODEL_DIM // self.NUM_HEADS,
            self.CONTEXT_WINDOW,
            base_freq=self.ROPE_BASE_FREQ,
            rope_scaling=rope_scaling,
            **config,
        )
        # avoid deepcopy with self.rope otherwise it duplicate the rope again.
        self.encoder = nn.ModuleList(
            [
                LLaMADecoderLayer(
                    d_model=self.MODEL_DIM,
                    nhead=self.NUM_HEADS,
                    ngroup=self.NUM_GROUPS,
                    rope=self.rope,
                    dim_feedforward=self.DIM_FEEDFORWARD,
                    rms_norm_eps=self.RMS_NORM_EPS,
                    **config,
                )
                for _ in range(self.NUM_LAYERS)
            ]
        )
        self.rms_norm = nn.RMSNorm(self.MODEL_DIM, eps=self.RMS_NORM_EPS, **config)
        self.unembedding = nn.Linear(
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
            out = layer(out, is_causal=True)
        out = self.rms_norm(out)
        out = self.unembedding(out)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        res_out_std = 0.02 / math.sqrt(2 * self.NUM_LAYERS)
        for layer in self.encoder:
            nn.init.normal_(layer.multi_head_attn.out_proj.weight, std=res_out_std)
            nn.init.normal_(layer.ffn.linear2.weight, std=res_out_std)


class LLaMA1(LLaMABase):
    def __init__(self, device=None, dtype=None):
        super().__init__(rope_scaling=None, device=device, dtype=dtype)


class LLaMA2(LLaMABase):
    CONTEXT_WINDOW = 4096
    NUM_GROUPS = 8
    DIM_FEEDFORWARD = 28672  # scale LLaMA1's by 1.33, MODEL_DIM * 8 / 3 * 1.33

    def __init__(self, device=None, dtype=None):
        super().__init__(rope_scaling=None, device=device, dtype=dtype)


class LLaMA3(LLaMABase):
    VOCAB_SIZE = 128256
    CONTEXT_WINDOW = 131072
    MODEL_DIM = 16384
    NUM_HEADS = 128
    NUM_GROUPS = 8
    DIM_FEEDFORWARD = 53248
    NUM_LAYERS = 126
    ROPE_BASE_FREQ = 500000.0

    def __init__(self, device=None, dtype=None):
        super().__init__(rope_scaling=self.rope_scaling, device=device, dtype=dtype)

    @staticmethod
    def rope_scaling(theta):
        # LLaMA3.1 frequency scaling, extends the 8192 pretraining context to
        # 131072. Not in the paper, see apply_scaling in meta-llama/llama-models.
        # Its grid searched constants are scale_factor 8, low_freq_factor 1,
        # high_freq_factor 4 and original context 8192, and with
        # s = (8192 / period - low_freq_factor) / (high_freq_factor - low_freq_factor)
        # the reference branches on period into
        #   > 8192 / 1 -> theta / 8
        #   < 8192 / 4 -> theta
        #   otherwise  -> (1 - s) * theta / 8 + s * theta
        period = math.pi * 2 / theta
        # Clamping s to [0, 1] collapses all 3 branches into the interpolation.
        smooth = torch.clamp((8192 / period - 1.0) / 3.0, min=0.0, max=1.0)
        scaled_theta = theta * (0.125 + 0.875 * smooth)
        return scaled_theta


class PaLMDecoderLayer(nn.Module):
    NUM_GROUPS = 1
    DROPOUT = 0.0
    ACTIVATION = SwishGLU

    def __init__(
        self,
        d_model,
        nhead,
        d_head,
        bias,
        rope,
        dim_feedforward,
        layer_norm_eps,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **config)
        self.multi_head_attn = RoPEMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            d_head=d_head,
            num_groups=self.NUM_GROUPS,
            dropout=self.DROPOUT,
            rope=rope,
            bias=bias,
            **config,
        )
        self.dropout = nn.Dropout(p=self.DROPOUT)
        self.ffn = TransformerFFNBlock(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            bias=bias,
            activation=self.ACTIVATION,
            dropout=self.DROPOUT,
            dim_feedforward1=2 * dim_feedforward,
            **config,
        )

    def forward(self, input, is_causal):
        normed_input = self.layer_norm(input)
        mha_out = self.multi_head_attn(
            query=normed_input,
            key=normed_input,
            value=normed_input,
            is_causal=is_causal,
        )
        out = input + self.dropout(mha_out) + self.ffn(normed_input)
        return out


class PaLM(nn.Module):
    VOCAB_SIZE = 256000
    CONTEXT_WINDOW = 2048
    MODEL_DIM = 18432
    HEAD_DIM = 256
    NUM_HEADS = 48
    DIM_FEEDFORWARD = 4 * MODEL_DIM
    NUM_LAYERS = 118
    BIAS = False
    LN_EPS = 1e-05

    def __init__(self, device=None, dtype=None):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(
            num_embeddings=self.VOCAB_SIZE, embedding_dim=self.MODEL_DIM, **config
        )
        self.rope = RoPE(self.HEAD_DIM, self.CONTEXT_WINDOW, **config)
        self.encoder = nn.ModuleList(
            [
                PaLMDecoderLayer(
                    d_model=self.MODEL_DIM,
                    nhead=self.NUM_HEADS,
                    d_head=self.HEAD_DIM,
                    bias=self.BIAS,
                    rope=self.rope,
                    dim_feedforward=self.DIM_FEEDFORWARD,
                    layer_norm_eps=self.LN_EPS,
                    **config,
                )
                for _ in range(self.NUM_LAYERS)
            ]
        )
        self.layer_norm = nn.LayerNorm(
            self.MODEL_DIM, eps=self.LN_EPS, bias=self.BIAS, **config
        )
        self.reset_parameters()

    def forward(self, input):
        out = self.embedding(input)
        for layer in self.encoder:
            out = layer(out, is_causal=True)
        out = self.layer_norm(out)
        out = out @ self.embedding.weight.T / math.sqrt(self.MODEL_DIM)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1 / math.sqrt(module.in_features))


class DeepSeekV1(LLaMABase):
    VOCAB_SIZE = 102400
    CONTEXT_WINDOW = 4096
    NUM_GROUPS = 8
    NUM_LAYERS = 95
    RMS_NORM_EPS = 1e-06
    INIT_STD = 0.006

    def __init__(self, device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.INIT_STD)


class MultiheadLatentAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        pass


class DeepSeekV2(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def reset_parameters(self):
        pass


class DeepSeekV3(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def reset_parameters(self):
        pass


class DeepSeekV4(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def reset_parameters(self):
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


class Kimi(nn.Module):
    def __init__(self):
        pass
