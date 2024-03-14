# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import jax.experimental.maps
import jax.numpy as jnp
from jax import config, tree_util
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint as pjit_sharding_constraint
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P

config.update("jax_spmd_mode", "allow_all")

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


@dataclass
class QuantizedWeight8bit:
    weight: jnp.array
    scales: jnp.array

    @property
    def shape(self):
        return self.weight.shape


tree_util.register_pytree_node(
    QuantizedWeight8bit,
    lambda qw: ([qw.weight, qw.scales], ()),
    lambda _, children: QuantizedWeight8bit(children[0], children[1]),
)


class TrainingState(NamedTuple):
    """Container for the training state."""

    params: hk.Params


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def with_sharding_constraint(x, constraint):
    if jax.experimental.maps.thread_resources.env.physical_mesh.empty:
        return x
    else:
        return pjit_sharding_constraint(x, constraint)


def cast_bfloat16(x):
    if x.dtype.kind == "f":
        return x.astype(jnp.bfloat16)
    else:
        return x


def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
    logger.debug(f"emd_size: {emb_size} adjusted ffn_size: {_ffn_size}")
    return _ffn_size


def apply_rules(rules):
    def _apply_rules(path, value):
        del value  # Unused.

        path_list = [str(i.key).split("/") for i in path if isinstance(i, jax.tree_util.DictKey)]
        flattened_path = jax.tree_util.tree_flatten(path_list)[0]

        for rule, replacement in rules:
            if _match(rule, flattened_path):
                if isinstance(replacement, PartitionSpec):
                    if "layer_stack" in flattened_path:
                        replacement = PartitionSpec(None, *replacement)
                rank_logger.debug(f"Apply {replacement} to {flattened_path} with rule {rule}")
                return replacement
        rank_logger.info(f"{flattened_path} no matching found!")
        return None

    return _apply_rules


TRANSFORMER_PARTITION_RULES = [
    # attention
    (("multi_head_attention", "(query|key|value)", "w"), P("data", "model")),
    (("multi_head_attention", "(query|key|value)", "b"), P(None)),
    (("multi_head_attention", "linear", "w"), P("model", "data")),
    (("multi_head_attention", "linear", "b"), P(None)),
    # mlp
    ((r"decoder_layer_[0-9]+", "linear", "w"), P("data", "model")),
    ((r"decoder_layer_[0-9]+", "linear", "b"), P(None)),
    ((r"decoder_layer_[0-9]+", "linear_v", "w"), P("data", "model")),
    ((r"decoder_layer_[0-9]+", "linear_v", "b"), P(None)),
    (
        (r"decoder_layer_[0-9]+", "linear_1", "w"),
        P(
            "model",
            "data",
        ),
    ),
    ((r"decoder_layer_[0-9]+", "linear_1", "b"), P(None)),
    # layer norms
    ((r"decoder_layer_[0-9]+", "layer_norm", "offset"), P(None)),
    ((r"decoder_layer_[0-9]+", "layer_norm", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "layer_norm_1", "offset"), P(None)),
    ((r"decoder_layer_[0-9]+", "layer_norm_1", "scale"), P(None)),
    # rms norms
    ((r"decoder_layer_[0-9]+", "rms_norm", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "rms_norm_1", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "rms_norm_2", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "rms_norm_3", "scale"), P(None)),
    # router
    (("router", "w"), P("data")),
    # moe mlp
    (("moe", "linear", "w"), P(None, "data", "model")),
    (("moe", "linear", "b"), P(None)),
    (("moe", "linear_v", "w"), P(None, "data", "model")),
    (("moe", "linear_v", "b"), P(None)),
    (("moe", "linear_1", "w"), P(None, "model", "data")),
    (("moe", "linear_1", "b"), P(None)),
    # layer norms
    (("moe", "layer_norm", "offset"), P(None)),
    (("moe", "layer_norm", "scale"), P(None)),
    (("moe", "layer_norm_1", "offset"), P(None)),
    (("moe", "layer_norm_1", "scale"), P(None)),
    # rms norms
    (("moe", "rms_norm", "scale"), P(None)),
    (("moe", "rms_norm_1", "scale"), P(None)),
    (("moe", "rms_norm_2", "scale"), P(None)),
    (("moe", "rms_norm_3", "scale"), P(None)),
]

LM_PARTITION_RULES = [
    # Embedding layer.
    (
        ("language_model", "positional_embeddings"),
        P(None, ("data", "model")),
    ),
    (
        ("language_model", "in_out_embed", "embeddings"),
        P(None, ("data", "model")),
    ),
    # Final RMSNorm.
    (("language_model", "rms_norm"), P(None)),
]
TOP_K = 8


class KVMemory(NamedTuple):
    k: Optional[jax.Array]
    v: Optional[jax.Array]
    step: Optional[jax.Array]


def init_layer_memories(
    batch_size: int,
    sequence_len: int,
    num_kv_heads: int,
    key_size: int,
    num_layers: int,
    step: Optional[jax.Array] = None,
    dtype=jnp.bfloat16,
):
    return [
        KVMemory(
            k=jnp.zeros((batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype),
            v=jnp.zeros((batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype),
            step=step,
        )
        for _ in range(num_layers)
    ]


class Memory(NamedTuple):
    # Self-attention key/value cache.
    layers: List[KVMemory]


class Router(hk.Module):
    def __init__(
        self,
        num_selected_experts: int,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        shard_activations: bool = False,
        mesh: Any = None,
        name: str = "router",
    ):
        super().__init__(name)
        self.shard_activations = shard_activations
        self.data_axis = data_axis
        self.model_axis = model_axis
        self.mesh = mesh
        self.num_selected_experts = num_selected_experts

    def compute_routing_prob(
        self, inputs: jax.Array, padding_mask: Optional[jax.Array], num_experts: int
    ):
        return self._compute_routing_prob(inputs, padding_mask, num_experts)

    @hk.transparent
    def _compute_routing_prob(
        self,
        inputs: jax.Array,
        padding_mask: Optional[jax.Array],
        num_experts: int,
    ):
        # Using fp32 for the routing prob computation.
        inputs = jax.lax.convert_element_type(inputs, jnp.float32)

        # [batch_size, seq_len, num_experts]
        routing_logits = self._router_weights(inputs, num_experts, sharding=P("data"))
        assert routing_logits.dtype == jnp.float32
        routing_probs = jax.nn.softmax(routing_logits)

        if padding_mask is not None:
            routing_probs *= padding_mask

        return routing_probs, routing_logits, 0

    @hk.transparent
    def _router_weights(
        self,
        x: jax.Array,
        num_experts: int,
        sharding: Optional[P] = None,
    ):
        fprop_dtype = x.dtype
        if not x.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = x.shape[-1]
        w = hk.get_parameter(
            "w", [input_size, num_experts], jnp.float32, init=hk.initializers.Constant(0)
        )
        if sharding:
            w = with_sharding_constraint(w, sharding)

        out = jnp.dot(x, w.astype(fprop_dtype))
        return out


class MoELayer(hk.Module):
    def __init__(
        self,
        num_experts: int,
        layer_fn: Callable,
        router: Router,
        mesh: Any = None,
        shard_activations: bool = False,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        name: Optional[str] = "moe",
    ):
        super().__init__(name)
        self.num_experts = num_experts
        self.layer_fn = layer_fn
        self.router = router
        self.mesh = mesh
        self.shard_activations = shard_activations
        self.data_axis = data_axis
        self.model_axis = model_axis

    @hk.transparent
    def _inference_call(self, inputs: jax.Array, padding_mask: Optional[jax.Array] = None):
        routing_probs, _, _ = self.router.compute_routing_prob(
            inputs, padding_mask, self.num_experts
        )
        expert_gate, expert_index = jax.lax.top_k(routing_probs, k=self.router.num_selected_experts)
        tmp = jnp.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
        broad_inputs = jnp.tile(tmp[:, jnp.newaxis, :], (1, self.router.num_selected_experts, 1))
        broad_inputs = jnp.reshape(
            broad_inputs, (broad_inputs.shape[0] * broad_inputs.shape[1], broad_inputs.shape[2])
        )
        init_fn, _ = hk.transform(self.layer_fn)
        vmapped_init_fn = jax.vmap(init_fn, in_axes=0, out_axes=0)
        lifted_init_fn = hk.experimental.transparent_lift(vmapped_init_fn)
        # Fetch the vmapped params of the DenseBlock.
        params = lifted_init_fn(
            jax.random.split(jax.random.PRNGKey(1), self.num_experts),
            jnp.zeros((self.num_experts, 1, 1, inputs.shape[-1])),
        )

        # Index and prob are in the shape [m, 2] indicating which token assigned to which experts.
        # b: num_expert
        # m: token or sequence dim
        # k: input embed dim
        # n: output embed dim
        # e: the number of experts chosen for each token
        @functools.partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(
                P(self.data_axis, None),
                P(None, None, self.model_axis),
                P(None, None, self.model_axis),
                P(None),
                P(None),
            ),
            out_specs=P(self.data_axis, self.model_axis),
            check_rep=False,
        )
        def moe_slow_matmul1(input, weight, scales, index, prob):
            weight = weight * scales
            one_hot_indices = jax.nn.one_hot(index.reshape(-1), 8, axis=0)
            all_expert_output = jnp.einsum("mk,bkn->bmn", input, weight)
            output = jnp.einsum("bm,bmn->mn", one_hot_indices, all_expert_output)
            return output

        @functools.partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(
                P(self.data_axis, self.model_axis),
                P(None, self.model_axis, None),
                P(None, self.model_axis, None),
                P(None),
                P(None),
            ),
            out_specs=P(self.data_axis, None),
            check_rep=False,
        )
        def moe_slow_matmul2(input, weight, scales, index, prob):
            weight = weight * scales
            one_hot_indices = jax.nn.one_hot(index.reshape(-1), 8, axis=0)
            all_expert_output = jnp.einsum("mk,bkn->bmn", input, weight)
            output = jnp.einsum("bm,bmn->mn", one_hot_indices, all_expert_output)
            return jax.lax.psum(output, axis_name="model")

        if hasattr(params["linear"]["w"], "scales"):
            x = moe_slow_matmul1(
                broad_inputs,
                params["linear_v"]["w"].weight,
                params["linear_v"]["w"].scales,
                expert_index,
                expert_gate,
            )
            y = moe_slow_matmul1(
                broad_inputs,
                params["linear"]["w"].weight,
                params["linear"]["w"].scales,
                expert_index,
                expert_gate,
            )
            y = jax.nn.gelu(y)
            out = moe_slow_matmul2(
                x * y,
                params["linear_1"]["w"].weight,
                params["linear_1"]["w"].scales,
                expert_index,
                expert_gate,
            )
            out = jnp.reshape(
                out,
                [
                    inputs.shape[0],
                    inputs.shape[1],
                    self.router.num_selected_experts,
                    out.shape[-1],
                ],
            )
            out = expert_gate[:, :, :, None].astype(jnp.bfloat16) * out
            out = jnp.sum(out, axis=2)
            out = out.astype(jnp.bfloat16)
        else:
            # This is only here so that we can construct a valid init_fn with this code.
            return inputs
        return out

    def __call__(self, inputs: jax.Array, padding_mask: jax.Array):
        return self._inference_call(inputs)


class MHAOutput(NamedTuple):
    """Outputs of the multi-head attention operation."""

    embeddings: jax.Array
    memory: Any


class DecoderOutput(NamedTuple):
    embeddings: jax.Array
    memory: Any


class TransformerOutput(NamedTuple):
    embeddings: jax.Array
    memory: Any


@dataclass
class TransformerConfig:
    emb_size: int
    key_size: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int = 128 * 1024
    widening_factor: float = 4.0

    attn_output_multiplier: float = 1.0

    name: Optional[str] = None

    num_experts: int = -1
    capacity_factor: float = 1.0
    num_selected_experts: int = 1

    init_scale: float = 1.0
    shard_activations: bool = False

    # Used for activation sharding.
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    def __post_init__(self):
        if isinstance(self.data_axis, list):
            self.data_axis = tuple(self.data_axis)
        if isinstance(self.model_axis, list):
            self.model_axis = tuple(self.model_axis)

    def partition_rules(self):
        return TRANSFORMER_PARTITION_RULES

    def make(self, mesh=None) -> "Transformer":
        data_axis = tuple(self.data_axis) if isinstance(self.data_axis, list) else self.data_axis
        model_axis = (
            tuple(self.model_axis) if isinstance(self.model_axis, list) else self.model_axis
        )

        return Transformer(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            widening_factor=self.widening_factor,
            key_size=self.key_size,
            init_scale=self.init_scale,
            mesh=mesh,
            attn_output_multiplier=self.attn_output_multiplier,
            shard_activations=self.shard_activations,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            num_selected_experts=self.num_selected_experts,
            data_axis=data_axis,
            model_axis=model_axis,
        )

    def get_memory_sharding(self):
        return Memory(
            layers=[
                KVMemory(
                    k=P(self.data_axis, self.model_axis),
                    v=P(self.data_axis, self.model_axis),
                    step=P(self.data_axis),
                )
                for _ in range(self.num_layers)
            ],
        )


def hk_rms_norm(
    x: jax.Array,
    fixed_scale=False,
    sharding=P(None),
) -> jax.Array:
    """Applies a unique LayerNorm to x with default settings."""
    ln = RMSNorm(axis=-1, create_scale=not fixed_scale, sharding=sharding)
    return ln(x)


def make_attention_mask(
    query_input: jax.Array,
    key_input: jax.Array,
    pairwise_fn: Callable[..., Any] = jnp.multiply,
    dtype: Any = jnp.bfloat16,
):
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      pairwise_fn: broadcasting elementwise comparison function
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = pairwise_fn(jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2))
    mask = jnp.expand_dims(mask, axis=-3)
    return mask.astype(dtype)


class Linear(hk.Linear):
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        sharding: Optional[P] = None,
        mesh: Any = None,
        name: Optional[str] = None,
        shard_axis: int = 0,
    ):
        super().__init__(
            output_size=output_size,
            with_bias=with_bias,
            name=name,
        )
        self.sharding = sharding
        self.mesh = mesh
        self.shard_axis = shard_axis

    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        """Computes a linear transform of the input."""

        fprop_dtype = inputs.dtype
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size

        w = hk.get_parameter(
            "w", [input_size, output_size], jnp.float32, init=hk.initializers.Constant(0)
        )

        if hasattr(w, "scales"):
            shape = inputs.shape
            inputs = jnp.reshape(inputs, (-1, shape[-1]))

            @functools.partial(
                shard_map,
                mesh=self.mesh,
                in_specs=(self.sharding, self.sharding),
                out_specs=self.sharding,
                check_rep=False,
            )
            def mul(w, s):
                return w.astype(s.dtype) * s

            w = mul(w.weight, w.scales)
        out = jnp.dot(inputs, w.astype(fprop_dtype))
        if self.with_bias:
            b = hk.get_parameter(
                "b", [self.output_size], jnp.float32, init=hk.initializers.Constant(0)
            )
            b = jnp.broadcast_to(b, out.shape)
            out = out + b.astype(fprop_dtype)

        return out


class RMSNorm(hk.RMSNorm):

    def __init__(
        self,
        axis: Union[int, Sequence[int], slice],
        eps: float = 1e-5,
        name: Optional[str] = None,
        create_scale: bool = True,
        sharding: Optional[P] = None,
    ):
        super().__init__(axis, eps, create_scale=create_scale, name=name)
        self.sharding = sharding

    def __call__(self, inputs: jax.Array):
        fprop_dtype = inputs.dtype
        param_shape = (inputs.shape[-1],)
        if self.create_scale:
            scale = hk.get_parameter(
                "scale",
                param_shape,
                dtype=jnp.float32,
                init=hk.initializers.Constant(0),
            )
            if self.sharding:
                scale = with_sharding_constraint(scale, self.sharding)
            scale = jnp.broadcast_to(scale.astype(jnp.float32), inputs.shape)
        else:
            scale = 1.0
        inputs = inputs.astype(jnp.float32)
        scale = scale.astype(jnp.float32)
        mean_squared = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
        mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)

        normed_inputs = inputs * jax.lax.rsqrt(mean_squared + self.eps)

        outputs = scale * normed_inputs

        return outputs.astype(fprop_dtype)


def rotate_half(
    x: jax.Array,
) -> jax.Array:
    """Obtain the rotated counterpart of each feature"""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class RotaryEmbedding(hk.Module):
    """Applies rotary embeddings (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    Attributes:
        dim (int): Dimensionality of the feature vectors
        base_exponent (int): Base exponent to compute embeddings from
    """

    def __init__(
        self,
        dim: int,
        name: Optional[str] = None,
        base_exponent: int = 10000,
    ):
        super().__init__(name)
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0

    def __call__(
        self,
        x: jax.Array,
        seq_dim: int,
        offset: jax.Array,
        const_position: Optional[int] = None,
        t: Optional[jax.Array] = None,
    ) -> jax.Array:
        fprop_dtype = x.dtype
        # Compute the per-dimension frequencies
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(
            1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32
        )

        if jnp.shape(offset) == ():
            # Offset can be a scalar or one offset per batch element.
            offset = jnp.expand_dims(offset, 0)

        # Compute the per element phase (to pass into sin and cos)
        if const_position:
            t = const_position * jnp.ones(
                (
                    1,
                    x.shape[seq_dim],
                ),
                dtype=jnp.float32,
            )
        elif t is None:
            t = jnp.arange(x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(offset, -1)
        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
        x = x.astype(fprop_dtype)

        return x


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        *,
        with_bias: bool = True,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        attn_output_multiplier: 1.0,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_q_heads
        self.data_axis = data_axis
        self.model_axis = model_axis
        self.attn_output_multiplier = attn_output_multiplier
        self.with_bias = with_bias

    def __call__(
        self,
        query: jax.Array,
        key: Optional[jax.Array],
        value: Optional[jax.Array],
        mask: Optional[jax.Array] = None,
        kv_memory: Optional[KVMemory] = None,
        mesh: Any = None,
    ) -> MHAOutput:
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        sequence_length = query.shape[1]
        projection = self._linear_projection
        use_memory = False
        if kv_memory is not None:
            if kv_memory.k is None:
                assert kv_memory.v is None
                assert key is not None
                assert value is not None
            else:
                assert kv_memory.v is not None
                use_memory = True
        else:
            assert key is not None
            assert value is not None

        # Check that the keys and values have consistent batch size and sequence length.
        if not use_memory:
            assert key.shape[:2] == value.shape[:2], f"key/value shape: {key.shape}/{value.shape}"

        if mask is not None:
            assert mask.ndim == 4
            assert mask.shape[0] in {
                1,
                query.shape[0],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            if not use_memory:
                assert key.shape[0] in {
                    1,
                    query.shape[0],
                }, f"key/query shape: {key.shape}/{query.shape}"
            assert mask.shape[1] == 1
            assert mask.shape[2] in {
                1,
                query.shape[1],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            if not use_memory:
                assert mask.shape[3] in {
                    1,
                    key.shape[1],
                }, f"mask/query shape: {mask.shape}/{key.shape}"

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        assert self.num_q_heads % self.num_kv_heads == 0
        query_heads = projection(
            query,
            self.key_size,
            self.num_q_heads,
            name="query",
            sharding=P("data", "model"),
            mesh=mesh,
        )  # [B, T', H, Q=K]

        new_memory = None
        key_heads = projection(
            key,
            self.key_size,
            self.num_kv_heads,
            name="key",
            sharding=P("data", "model"),
            mesh=mesh,
        )  # [B, T, H, K]
        value_heads = projection(
            value,
            self.value_size,
            self.num_kv_heads,
            name="value",
            sharding=P("data", "model"),
            mesh=mesh,
        )  # [B, T, H, V]

        rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))
        key_heads = rotate(key_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0))
        query_heads = rotate(query_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0))

        @functools.partial(jax.vmap)
        def update_into(mem, start, update):
            return jax.lax.dynamic_update_slice_in_dim(mem, update, start, axis=0)

        if kv_memory:
            if mesh is not None:

                @functools.partial(
                    shard_map,
                    mesh=mesh,
                    in_specs=(
                        P("data", None, "model"),
                        P("data"),
                        P("data", None, "model"),
                    ),
                    out_specs=P("data", None, "model"),
                    check_rep=False,
                )
                def update_into_shmap(mems, starts, updates):
                    return update_into(mems, starts, updates)

                key_heads = update_into_shmap(kv_memory.k, kv_memory.step, key_heads)
                value_heads = update_into_shmap(kv_memory.v, kv_memory.step, value_heads)
            else:
                key_heads = update_into(kv_memory.k, kv_memory.step, key_heads)
                value_heads = update_into(kv_memory.v, kv_memory.step, value_heads)

            new_step = kv_memory.step + sequence_length
            memory_mask = jnp.arange(kv_memory.k.shape[1]) < new_step[:, None]
            memory_mask = memory_mask[:, None, None, :]  # [B, H, T, T]
            if mask is not None:
                mask = memory_mask * mask
            else:
                mask = memory_mask

            new_memory = KVMemory(
                k=key_heads,
                v=value_heads,
                step=new_step,
            )
        # Add separate dimension for grouped query heads.
        query_heads = with_sharding_constraint(query_heads, P(self.data_axis, None, "model", None))
        key_heads = with_sharding_constraint(key_heads, P(self.data_axis, None, "model", None))
        value_heads = with_sharding_constraint(value_heads, P(self.data_axis, None, "model", None))
        b, t, h, d = query_heads.shape
        _, _, kv_h, _ = key_heads.shape
        assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"

        query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))
        query_heads = with_sharding_constraint(
            query_heads, P(self.data_axis, None, "model", None, None)
        )

        # Compute attention weights.
        # Attention softmax is always carried out in fp32.
        attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads).astype(
            jnp.float32
        )
        attn_logits *= self.attn_output_multiplier
        max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
        attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)

        mask = mask[:, :, None, :, :]

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim} for {mask.shape}/{attn_logits.shape}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
        attn = with_sharding_constraint(attn, P(self.data_axis, None, "model", None, None))
        leading_dims = attn.shape[:2]
        attn = jnp.reshape(attn, (*leading_dims, -1))  # [T', H*V]
        attn = with_sharding_constraint(attn, P(self.data_axis, None, "model"))
        # Apply another projection to get the final embeddings.
        final_projection = Linear(
            self.model_size,
            with_bias=False,
            sharding=P("model", "data"),
            mesh=mesh,
        )
        return MHAOutput(final_projection(attn), new_memory)

    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array,
        head_size: int,
        num_heads: int,
        sharding: Optional[P] = None,
        name: Optional[str] = None,
        mesh: Any = None,
    ) -> jax.Array:
        y = Linear(
            num_heads * head_size,
            with_bias=False,
            name=name,
            sharding=sharding,
            mesh=mesh,
        )(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


@dataclass
class MHABlock(hk.Module):
    """A MHA Block"""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    attn_output_multiplier: float = 1.0
    mesh: Any = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T] or B[1, 1, 1, 1]
        layer_memory: Optional[KVMemory],
    ) -> MHAOutput:
        _, _, model_size = inputs.shape
        assert mask.ndim == 4, f"shape: {mask.shape}"
        assert mask.shape[2] in {1, inputs.shape[1]}, str(mask.shape)
        assert mask.shape[3] in {1, inputs.shape[1]}, str(mask.shape)
        side_input = inputs

        def attn_block(query, key, value, mask, memory) -> MHAOutput:
            return MultiHeadAttention(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                model_size=model_size,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                attn_output_multiplier=self.attn_output_multiplier,
            )(
                query,
                key,
                value,
                mask,
                memory,
                mesh=self.mesh,
            )

        attn_output = attn_block(inputs, side_input, side_input, mask, layer_memory)
        h_attn = attn_output.embeddings

        return attn_output._replace(embeddings=h_attn)


@dataclass
class DenseBlock(hk.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float = 4.0
    sharding_constraint: bool = False
    mesh: Any = None

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
    ) -> jax.Array:  # [B, T, D]
        _, _, model_size = inputs.shape
        h_v = Linear(
            ffn_size(
                model_size,
                self.widening_factor,
            ),
            with_bias=False,
            mesh=self.mesh,
            sharding=P("data", "model"),
            name="linear_v",
        )(inputs)
        h_w1 = jax.nn.gelu(
            Linear(
                ffn_size(
                    model_size,
                    self.widening_factor,
                ),
                with_bias=False,
                mesh=self.mesh,
                sharding=P("data", "model"),
            )(inputs)
        )
        h_dense = Linear(
            model_size,
            with_bias=False,
            sharding=P("model", "data"),
            mesh=self.mesh,
            shard_axis=1,
        )(h_w1 * h_v)

        return h_dense


@dataclass
class DecoderLayer(hk.Module):
    """A transformer stack."""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    num_layers: int
    # MoE.
    num_experts: int
    layer_index: Optional[int] = None
    num_selected_experts: int = 1
    widening_factor: float = 4.0
    name: Optional[str] = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"
    shard_activations: bool = False
    attn_output_multiplier: float = 1.0
    mesh: Any = None

    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T]
        padding_mask: Optional[jax.Array],
        layer_memory: Optional[KVMemory],
    ) -> DecoderOutput:
        """Transforms input embedding sequences to output embedding sequences."""

        def layer_norm(x):
            return hk_rms_norm(x)

        if self.shard_activations:
            sharding = P(self.data_axis, None, self.model_axis)
        else:
            sharding = P(self.data_axis, None)
        h = with_sharding_constraint(inputs, sharding)

        attn_output = MHABlock(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            attn_output_multiplier=self.attn_output_multiplier,
            mesh=self.mesh,
            data_axis=self.data_axis,
            model_axis=self.model_axis,
        )(layer_norm(h), mask, layer_memory)
        h_attn = attn_output.embeddings

        h_attn = layer_norm(h_attn)
        h += h_attn
        h = with_sharding_constraint(h, sharding)

        def base_dense_block(h):
            h = DenseBlock(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=self.widening_factor,
                sharding_constraint=False,
                mesh=self.mesh,
            )(h)
            return h

        if self.num_experts > 1:
            rank_logger.debug("Using MoE!")
            router = Router(
                num_selected_experts=self.num_selected_experts,
                shard_activations=self.shard_activations,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                mesh=self.mesh,
            )
            h_dense = MoELayer(
                num_experts=self.num_experts,
                mesh=self.mesh,
                layer_fn=base_dense_block,
                router=router,
                shard_activations=self.shard_activations,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
            )(layer_norm(h), padding_mask)
        else:
            h_dense = base_dense_block(layer_norm(h))

        h_dense = layer_norm(h_dense)
        h += h_dense
        h = with_sharding_constraint(h, sharding)

        return DecoderOutput(
            embeddings=h,
            memory=attn_output.memory,
        )


class LanguageModelOutput(NamedTuple):
    logits: jax.Array
    model_state: Any


class InOutEmbed(hk.Embed):
    """Module for embedding tokens in a low-dimensional space."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        sharding: Optional[P] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            name=name,
        )
        self.sharding = sharding

    @property
    def embeddings(self):
        embed_mat = hk.get_parameter(
            "embeddings",
            [self.vocab_size, self.embed_dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0),
        )
        if self.sharding:
            embed_mat = with_sharding_constraint(embed_mat, self.sharding)
        return embed_mat

    def decode(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        return jnp.dot(inputs, self.embeddings.T.astype(inputs.dtype))


@dataclass
class LanguageModelConfig:
    """An autoregressive transformer-based language model."""

    model: Optional[TransformerConfig]
    vocab_size: int
    pad_token: int
    eos_token: int
    sequence_len: int
    model_size: int = 0
    embedding_init_scale: float = 1.0
    embedding_multiplier_scale: float = 1.0
    output_multiplier_scale: float = 1.0
    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16
    model_type: Optional[str] = None
    init_scale_override: Optional[float] = None
    shard_embeddings: bool = True

    _initialized = False

    def initialize(self):
        # We cannot specify [] as a default value (it is mutable), hence None.
        model_config = self.model
        assert self.init_scale_override is None, (
            "Overriding model initialize scale is supported only for predefined models."
        )
        if self.model_size == 0:
            self.model_size = model_config.emb_size
        assert self.model is not None, "Model could not be initialized."
        self._initialized = True
        return self

    def make(self, *args, **kwargs):
        if not self._initialized:
            logger.warning(
                f"LanguageModel {self.name} is not initialized. Initializing for one replica."
            )
            self.initialize()

        return LanguageModel(
            model=self.model.make(*args, **kwargs),
            config=self,
            fprop_dtype=self.fprop_dtype,
            mesh=kwargs.get("mesh", None),
        )

    def partition_rules(self):
        return LM_PARTITION_RULES + self.model.partition_rules()


def layer_norm(x, model):
    return hk_rms_norm(x)


@dataclass
class LanguageModel(hk.Module):
    """An autoregressive transformer-based language model."""

    model: "Transformer"
    config: LanguageModelConfig
    fprop_dtype: Any = jnp.bfloat16
    name: Optional[str] = None
    mesh: Any = None

    def __call__(
        self,
        tokens: jax.Array,
        memory: Optional[Memory] = None,
        *,
        batch: Dict[str, jax.Array] = {},
        last_hid_only: bool = False,
        length: Optional[jax.Array] = None,
    ) -> LanguageModelOutput:
        """Forward pass, producing a sequence of logits."""
        del batch  # Unused.

        config = self.config

        input_mask = jnp.greater(tokens, config.pad_token)

        # Embed the input tokens and positions.
        in_out_embed = InOutEmbed(
            self.config.vocab_size,
            embed_dim=self.config.model_size,
            sharding=P(None, ("data", "model")),
        )
        input_embeddings = in_out_embed(tokens).astype(config.fprop_dtype)
        input_embeddings = with_sharding_constraint(
            input_embeddings, P("data", None, self.model.model_axis)
        )
        input_embeddings *= config.embedding_multiplier_scale

        model_output = self.model(
            input_embeddings,
            input_mask,
            memory=memory,
        )  # [B, T, D]
        embeddings, model_state = model_output.embeddings, model_output.memory
        if self.model.shard_activations:
            embeddings = with_sharding_constraint(
                embeddings, P("data", None, self.model.model_axis)
            )
        else:
            embeddings = with_sharding_constraint(embeddings, P("data", None))
        rank_logger.debug(f"Final embedding shape: {embeddings.shape}")
        embeddings = layer_norm(embeddings, self.model)
        assert embeddings.dtype == self.fprop_dtype

        if last_hid_only:
            last_step = jnp.maximum(jnp.sum(input_mask.astype(jnp.int32), axis=1) - 1, 0)
            last_hid = jax.vmap(lambda x, i: x[i], in_axes=0, out_axes=0)(embeddings, last_step)
            return last_hid

        if length is not None:
            last_step = jnp.maximum(length.astype(jnp.int32) - 1, 0)
            embeddings = jax.vmap(lambda x, i: x[i], in_axes=0, out_axes=0)(embeddings, last_step)
            embeddings = jnp.expand_dims(embeddings, axis=1)

        # Decode the embeddings (here, we use tied weights).
        rank_logger.info(embeddings.shape)
        out = in_out_embed.decode(embeddings)
        rank_logger.info(out.shape)
        out *= config.output_multiplier_scale

        if self.model.shard_activations:
            out = with_sharding_constraint(out, P("data", None, self.model.model_axis))
        else:
            out = with_sharding_constraint(out, P("data", None))

        return LanguageModelOutput(
            logits=out,
            model_state=model_state,
        )

    def init_memory(self, batch_size: int, seq_len: int, dtype=jnp.bfloat16):
        return self.model.init_memory(batch_size=batch_size, sequence_len=seq_len, dtype=dtype)

    def prefill_memory(self, prompts, memory):
        # Pad to the left and right align?
        # Basically assume prompt is already padded
        model_output = self(prompts, memory=memory)
        return model_output.logits, model_output.model_state


@dataclass
class Transformer(hk.Module):
    """A transformer stack."""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float
    init_scale: float
    mesh: Any
    attn_output_multiplier: float
    shard_activations: bool
    num_layers: int
    # MoE
    num_experts: int
    num_selected_experts: int
    name: Optional[str] = None

    # Used for activation sharding
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    def init_memory(self, batch_size: int, sequence_len: int, dtype=jnp.bfloat16):
        return Memory(
            layers=init_layer_memories(
                batch_size,
                sequence_len,
                self.num_kv_heads,
                self.key_size,
                self.num_layers,
                step=jnp.zeros(batch_size, dtype=jnp.int32),
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        embeddings: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, T]
        memory: Optional[Memory],
    ) -> TransformerOutput:
        """Transforms input embedding sequences to output embedding sequences."""

        fprop_dtype = embeddings.dtype
        _, seq_len, model_size = embeddings.shape
        padding_mask = mask.copy()
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]

        # Compute causal mask for autoregressive sequence modelling.
        causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len))).astype(
            fprop_dtype
        )  # [B=1, H=1, T, T]
        mask = mask * causal_mask  # [B, H=1, T, T]

        h = embeddings
        kv_memories = []

        def block(
            h,
            mask,
            padding_mask,
            memory,
            layer_index: Optional[int] = None,
            widening_factor: Optional[int] = None,
            name: Optional[str] = None,
        ) -> DecoderOutput:
            return DecoderLayer(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=widening_factor or self.widening_factor,
                num_layers=self.num_layers,
                mesh=self.mesh,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                attn_output_multiplier=self.attn_output_multiplier,
                shard_activations=self.shard_activations,
                # MoE.
                num_experts=self.num_experts,
                num_selected_experts=self.num_selected_experts,
                name=name,
                layer_index=layer_index,
            )(
                h,
                mask,
                padding_mask,
                memory,
            )

        for i in range(self.num_layers):
            decoder_output = block(
                h,
                mask,
                padding_mask,
                memory.layers[i] if memory else None,
                layer_index=i,
                name=f"decoder_layer_{i}",
            )
            h, new_kv_memory = (
                decoder_output.embeddings,
                decoder_output.memory,
            )
            kv_memories.append(new_kv_memory)

        return TransformerOutput(
            embeddings=h,
            memory=Memory(layers=kv_memories),
        )
