"""
Llama-style decoder-only autoregressive transformer.

Architecture lineage: RoPE + SwiGLU + RMSNorm (Llama / Llama 2/3 / Mistral),
implemented as a minimal modification on top of nanoGPT's GPT-2 module structure.
The exported class is still named `GPT` to keep diffs against `molcrawl/models/gpt2/model.py`
auditable; this is decoder-only Llama-style and not weight-compatible with HF GPT-2.

References:
- Llama: https://arxiv.org/abs/2302.13971
- RoPE (Su et al., 2021): https://arxiv.org/abs/2104.09864
- RMSNorm (Zhang & Sennrich, 2019): https://arxiv.org/abs/1910.07467
- SwiGLU (Shazeer, 2020): https://arxiv.org/abs/2002.05202
- nanoGPT (base): https://github.com/karpathy/nanoGPT
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias.

    Retained from the GPT-2 lineage; not used in the Llama-style architecture
    (which uses RMSNorm). Kept here so existing tooling that imports LayerNorm
    from this module does not break.
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used in LLaMA, Llama 2/3, Mistral. No bias term; normalizes by RMS instead
    of (mean, variance). Reference: Zhang & Sennrich, 2019.
    """

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        # Compute in float32 for numerical stability with bf16/fp16 inputs.
        x_float = x.float()
        rms = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * rms).to(x.dtype) * self.weight


def precompute_rope_cache(dim, max_seq_len, base=10000.0, device=None, dtype=torch.float32):
    """Precompute cos/sin lookup tables for Rotary Positional Embedding.

    Reference: Su et al., 2021, "RoFormer".
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
    return emb.cos(), emb.sin()


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary positional embedding to q, k.

    Uses the Llama / HuggingFace "rotate_half" convention (splits the last dim
    in halves rather than interleaving pairs). q, k: (B, n_head, T, head_dim);
    cos, sin: (T, head_dim).
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        head_dim = config.n_embd // config.n_head
        # rotate_half (Llama/HF convention) splits head_dim in halves, so it must be even.
        assert head_dim % 2 == 0, f"RoPE requires even head_dim, got {head_dim}"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Precompute rotary embedding cache (covers up to config.block_size).
        cos, sin = precompute_rope_cache(head_dim, config.block_size, base=10000.0)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Apply rotary positional embedding to q and k (Llama-style).
        cos = self.rope_cos[:T].to(q.dtype)
        sin = self.rope_sin[:T].to(q.dtype)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """SwiGLU MLP (used in LLaMA, Llama 2/3, PaLM, Mistral).

    Standard 4x expansion is replaced by ~2.67x to keep total params comparable
    to a GELU MLP (SwiGLU has 3 projections instead of 2). The hidden dim is
    rounded up to the nearest multiple of 64 to keep tensor sizes GPU-friendly.
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # round up to multiple of 64

        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: down(silu(gate(x)) * up(x))
        x = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # No wpe: position information is injected by RoPE inside CausalSelfAttention.
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying. With torch.compile() this can emit a deprecation warning
        # ("functional_call was passed multiple values for tied weights") — known
        # harmless in this configuration.
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # (kept for Llama-style as well; matches Megatron-LM style scaling)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Return the total number of parameters in the model.

        Unlike the GPT-2 variant this no longer accepts a ``non_embedding`` flag:
        ``wpe`` has been removed (RoPE replaces it) and ``wte`` is weight-tied to
        ``lm_head``, so there is no separate "embedding-only" parameter group to
        subtract out.
        """
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the model (no positional embedding lookup — RoPE applied in attention)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # ignore_index=-1 is kept (instead of HF-standard -100) to preserve
            # the existing contract with prepared datasets in molcrawl/core/dataset.py.
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Reduce block size; useful for fine-tuning at shorter context.

        RoPE cos/sin tables in CausalSelfAttention already cover up to the
        original block_size, so forward-time slicing (`self.rope_cos[:T]`)
        handles shorter contexts without modification. We only update the
        config and the (slow-attention) causal bias buffer.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Deprecated: HF GPT-2 weights are not compatible with Llama-style.

        The Llama-style architecture in this module uses RMSNorm + RoPE + SwiGLU
        and does not have ``wpe`` or LayerNorm parameters, so the HF GPT-2
        state_dict cannot be copied in. To pretrain a Llama-style model, train
        from scratch via ``molcrawl.models.llama.train``. To load from a
        MolCrawl-trained checkpoint, use ``init_from='resume'`` (handled by
        ``train.py``) or load the state_dict directly.
        """
        raise NotImplementedError(
            "from_pretrained for HF GPT-2 is not supported in Llama-style. "
            "Train from scratch via molcrawl.models.llama.train, or implement "
            "a dedicated Llama-style weight loader if needed."
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
                idx: LongTensor [B, T]
        Add one token at a time. If EOS is output for each batch element, PAD will be output (EOS will be repeated if not specified).
        Early termination when all elements are completed.
        """
        device = idx.device
        B = idx.size(0)

        # Whether each line issued an EOS
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Fallback to repeat EOS if PAD is not specified
        pad_or_eos = pad_token_id if pad_token_id is not None else eos_token_id

        for _ in range(max_new_tokens):
            # End if all lines are already finished
            if eos_token_id is not None and finished.all():
                break

            # Keep context length within block_size
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx

            # Normal forward
            logits, _ = self(idx_cond)  # [B, T_ctx, V]
            logits = logits[:, -1, :] / temperature  # [B, V]

            # Do not sample lines that have already finished (fixed to PAD or EOS)
            if eos_token_id is not None and pad_or_eos is not None:
                # Set the end line logit to -inf and set only pad_or_eos to 0 (almost 1.0 after softmax)
                mask = finished.unsqueeze(1)  # [B,1]
                if mask.any():
                    logits = logits.clone()
                    logits[mask.expand_as(logits)] = -float("inf")
                    logits[finished, pad_or_eos] = 0.0

            # top-k
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                # Note: Finished lines are not excluded because we have already set pad_or_eos=0.0 above.
                thresh = v[:, [-1]]
                logits[logits < thresh] = -float("inf")

            # sampling
            probs = F.softmax(logits, dim=-1)  # [B, V]
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Concatenation
            idx = torch.cat((idx, idx_next), dim=1)  # [B, T+1]

            # Reflect the new EOS line in finished
            if eos_token_id is not None:
                newly_finished = idx_next.squeeze(1) == eos_token_id
                finished |= newly_finished

        return idx
