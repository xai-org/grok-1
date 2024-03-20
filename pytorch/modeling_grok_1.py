from configuration_grok_1 import Grok1Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from typing import Optional, NamedTuple, List

class TiedWeightEmbedding(nn.Embedding):
    """Module for tied weight embedding."""

    def __init__(
        self,
        config: Grok1Config,
    ):
        super().__init__(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )

    def decode(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(inputs, self.weight.T)

class Gating(nn.Module):
    """Gating module for spare MoE expert selection."""

    def __init__(
        self,
        config: Grok1Config,
    ):
        super().__init__()
        self.router_weights = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        routing_logits = self.router_weights(inputs)
        routing_probs = F.softmax(routing_logits, dim=-1, dtype=torch.float32)

        if padding_mask is not None:
            # [batch * seq, expert]
            routing_probs = routing_probs * padding_mask.view(-1).unsqueeze(-1)

        # Note routing_probs is using float32.
        return routing_probs, routing_logits

class MLPExpert(nn.Module):
    """MLP expert module for sparse MoE."""

    def __init__(
        self,
        config: Grok1Config,
    ):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.v = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        h_w1 = self.act_fn(self.w1(inputs))
        h_v = self.v(inputs)
        h_dense = self.dense(h_w1 * h_v)
        return h_dense

class SparseMoEMLP(nn.Module):
    """Sparse MoE MLP module."""

    def __init__(
        self,
        config: Grok1Config,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gating = Gating(config)
        self.experts = nn.ModuleList([MLPExpert(config) for _ in range(self.num_experts)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Get routing probabilities and selected experts.
        routing_probs, routing_logits = self.gating(hidden_states, padding_mask)
        routing_probs, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True)
        # Now routing_probs is using the hidden_states' dtype instead of float32.
        routing_probs = routing_probs.to(hidden_states.dtype)

        # Initialize output hidden states.
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Create expert mask.
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over experts and compute their contributions.
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_probs[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, routing_logits

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        base_exponent: int = int(1e4),
    ):
        super().__init__()
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0, "Embedding dimension must be even for rotary embeddings."

    def forward(
        self,
        x: torch.Tensor,
        seq_dim: int,
        offset: torch.Tensor,
        const_position: Optional[int] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute the per-dimension frequencies.
        dtype = x.dtype
        exponents = torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device)
        inv_freq = (self.base_exponent ** (exponents / self.dim)).reciprocal()

        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset, dtype=torch.float32, device=x.device)
        if offset.dim() == 0:
            # Offset can be a scalar or one offset per batch element.
            offset = offset.unsqueeze(0)

        # Compute the per-element phase (to pass into sin and cos).
        if const_position is not None:
            t = const_position * torch.ones(
                (1, x.shape[seq_dim]),
                dtype=torch.float32,
                device=x.device,
            )
        elif t is None:
            t = torch.arange(x.shape[seq_dim], dtype=torch.float32, device=x.device)
            t = t.unsqueeze(0) + offset.unsqueeze(-1)

        phase = torch.einsum("bi,j->bij", t, inv_freq)
        phase = torch.cat([phase, phase], dim=-1)[:, :, None, :]

        x_rotated = x * phase.cos() + rotate_half(x) * phase.sin()

        return x_rotated.to(dtype)

class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

class KVMemory(NamedTuple):
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    step: Optional[torch.Tensor]

def init_layer_memories(
    batch_size: int,
    sequence_len: int,
    num_kv_heads: int,
    key_size: int,
    num_layers: int,
    step: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    if step is None:
        step = torch.zeros(batch_size, dtype=torch.int32, device=device)
    return [
        KVMemory(
            k=torch.zeros(batch_size, sequence_len, num_kv_heads, key_size, dtype=dtype, device=device),
            v=torch.zeros(batch_size, sequence_len, num_kv_heads, key_size, dtype=dtype, device=device),
            step=step,
        )
        for _ in range(num_layers)
    ]

class MultiHeadAttention(nn.Module):

    def __init__(self, config: Grok1Config):
        super().__init__()
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.key_size = config.hidden_size // config.num_attention_heads
        self.value_size = self.key_size
        self.attn_output_multiplier = config.attn_output_multiplier

        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.key_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.key_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.value_size, bias=False)
        self.out_proj = nn.Linear(self.num_q_heads * self.value_size, config.hidden_size, bias=False)
        self.rotary_pos_emb = RotaryPositionalEmbedding(self.key_size, base_exponent=config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_memory: Optional[KVMemory] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_q_heads, self.key_size)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.key_size)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.value_size)

        query = self.rotary_pos_emb(query, seq_dim=1, offset=layer_memory.step if layer_memory else 0)
        key = self.rotary_pos_emb(key, seq_dim=1, offset=layer_memory.step if layer_memory else 0)

        if layer_memory:
            key = torch.cat([layer_memory.k, key], dim=1)
            value = torch.cat([layer_memory.v, value], dim=1)
            new_step = layer_memory.step + seq_len
            memory_mask = torch.arange(key.shape[1], device=key.device) < new_step[:, None]
            memory_mask = memory_mask[:, None, None, :]
            if mask is not None:
                mask = mask * memory_mask
            else:
                mask = memory_mask
            new_memory = KVMemory(k=key, v=value, step=new_step)
        else:
            new_memory = None

        query = query.view(batch_size, seq_len, self.num_kv_heads, self.num_q_heads // self.num_kv_heads, self.key_size)
        attn_logits = torch.einsum("...thHd,...Thd->...hHtT", query, key).to(torch.float32)
        attn_logits *= self.attn_output_multiplier
        max_attn_val = torch.tensor(30.0, dtype=attn_logits.dtype, device=attn_logits.device)
        attn_logits = max_attn_val * torch.tanh(attn_logits / max_attn_val)

        if mask is not None:
            mask = mask[:, :, None, :, :]
            attn_logits = torch.where(mask, attn_logits, torch.full_like(attn_logits, float("-inf")))

        attn_weights = F.softmax(attn_logits, dim=-1).to(query.dtype)

        attn = torch.einsum("...hHtT,...Thd->...thHd", attn_weights, value)
        attn = attn.reshape(batch_size, seq_len, -1)
        attn = self.out_proj(attn)

        return attn, new_memory

class Decoder(nn.Module):

    def __init__(self, config: Grok1Config):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.attention = MultiHeadAttention(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.norm3 = RMSNorm(config.hidden_size)
        self.norm4 = RMSNorm(config.hidden_size)

        if config.num_local_experts > 1:
            self.mlp = SparseMoEMLP(config)
        else:
            self.mlp = MLPExpert(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        layer_memory: Optional[KVMemory] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, new_memory = self.attention(hidden_states, mask, layer_memory)
        attn_output = self.norm2(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        if isinstance(self.mlp, SparseMoEMLP):
            mlp_output, routing_logits = self.mlp(hidden_states, padding_mask)
        else:
            mlp_output = self.mlp(hidden_states)
            routing_logits = None
        mlp_output = self.norm4(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states, new_memory, routing_logits

class Grok1PreTrainedModel(PreTrainedModel):
    config_class = Grok1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Decoder"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Grok1ForCausalLM(Grok1PreTrainedModel):

    def __init__(self, config: Grok1Config):
        super().__init__(config)
        self.embedding = TiedWeightEmbedding(self.config)
        self.layers = nn.ModuleList([Decoder(self.config) for _ in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory: Optional[List[KVMemory]] = None,
        last_hid_only: bool = False,
        length: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        padding_mask = attention_mask.view(batch_size, seq_len)
        causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
        mask = padding_mask[:, None, None, :] * causal_mask

        hidden_states = self.embedding(input_ids) * self.config.embedding_multiplier_scale
        kv_memories = []

        for i, layer in enumerate(self.layers):
            layer_memory = memory[i] if memory else None
            hidden_states, new_memory, routing_logits = layer(
                hidden_states,
                mask,
                padding_mask,
                layer_memory,
            )
            kv_memories.append(new_memory)

        hidden_states = self.norm(hidden_states)

        if last_hid_only:
            last_step = torch.maximum(torch.sum(padding_mask, dim=1) - 1, torch.tensor(0, device=input_ids.device))
            hidden_states = hidden_states[torch.arange(batch_size, device=input_ids.device), last_step]
        elif length is not None:
            last_step = torch.maximum(length - 1, torch.tensor(0, device=input_ids.device))
            hidden_states = hidden_states[torch.arange(batch_size, device=input_ids.device), last_step]
            hidden_states = hidden_states.unsqueeze(1)

        logits = self.embedding.decode(hidden_states) * torch.tensor(self.config.output_multiplier_scale, dtype=hidden_states.dtype, device=hidden_states.device)

        return logits, kv_memories
