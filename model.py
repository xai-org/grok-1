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

"""
Grok-1

This architecture is designed for language modeling and autoregressive sequence generation, incorporating advanced features such as:

- 8-bit Quantized Weights: Implements quantization for model parameters to reduce the memory footprint and potentially increase computational efficiency.
- Sharding and Distributed Computation: Utilizes JAX's capabilities for distributed computation across devices, optimizing parallel processing and memory usage.
- Memory Caching for Autoregressive Decoding: Features a mechanism for caching keys and values in attention layers, enhancing efficiency in sequence generation tasks.

Core Components:
- Multi-Head Attention (MHA): Custom implementation, including rotary positional embeddings for implicit sequence position information.
- Mixture of Experts (MoE): Allows routing inputs to different "expert" networks based on the input data, increasing model capacity and expressiveness.
- Feed-Forward Networks (FFNs): Defines networks with adjustable sizes and support for quantized weights and custom linear layers with sharding.

Training and Inference Workflow:
- Manages the model's parameters, caching mechanisms, and layer-wise states efficiently during both training and inference.
- Implements detailed sharding strategies for optimizing the distribution of computation and storage across multiple devices.

Advanced Features:
- Custom Layer Normalization: Adopts RMSNorm for stabilizing the training of deep networks.
- Dynamic and Static Sharding: Offers flexibility in data and model parallelism, allowing dynamic adjustment of sharding constraints.

Efficiency and Scalability:
- Efficiently manages data flow and computation, minimizing unnecessary data replication and movement across devices.
- Designed with scalability in mind, providing a foundation for training complex models on massive datasets.

This architecture leverages JAX for high-performance numerical computing and automatic differentiation, alongside Haiku for modular and flexible deep learning model construction.

Data Flow Through the Training Process:

1. Input Preparation:
    - The process begins with the preparation of input data, which typically involves tokenizing text data into numerical tokens that represent words or subwords in a vocabulary.
    - Tokens are then batched and padded to ensure consistent sequence lengths within each batch, forming the initial input tensor for the model.

2. Embedding Layer:
    - The input tokens are passed through an embedding layer, transforming each token into a high-dimensional vector. This layer may utilize pre-trained embeddings or learn embeddings during training.
    - Positional encodings or embeddings are added to these vectors to incorporate sequence position information.

3. Transformer Layers:
    - The sequence of embedding vectors is processed through multiple Transformer layers, each consisting of the following sub-layers:
        a. Multi-Head Attention (MHA): Each layer computes self-attention for its input, allowing each position in the sequence to attend to all positions in the previous layer’s output.
        b. Feed-Forward Network (FFN): After attention, the output for each position passes through a feed-forward network. FFNs are identical for different positions but have different parameters from layer to layer.

    - Between each sub-layer, residual connections followed by layer normalization are applied. This helps in stabilizing the training of deep networks.

4. Caching and Memory:
    - For autoregressive tasks, where the model generates sequences one token at a time, keys and values computed during the attention operations are cached. This mechanism allows reusing these computations in subsequent steps, reducing the computational load.

5. Output Layer:
    - The output from the final Transformer layer is passed through a linear layer or a decoder, transforming the high-dimensional representations back into the vocabulary space to produce logits for each token in the sequence.

6. Loss Calculation and Backpropagation:
    - The logits are compared against the true token sequences using a suitable loss function (e.g., cross-entropy for language modeling tasks). The loss quantifies the model's prediction accuracy.
    - Based on the loss, gradients are computed for each parameter in the model using backpropagation. These gradients indicate how each parameter should be adjusted to minimize the loss.

7. Parameter Update:
    - Model parameters are updated using an optimization algorithm (e.g., Adam, SGD) and the computed gradients. This step adjusts the model’s weights to improve its predictions on the next iteration.

8. Iteration and Convergence:
    - Steps 1 through 7 are repeated for multiple epochs over the training dataset until the model's performance on a validation set converges or begins to degrade, indicating potential overfitting.

This structured approach, combined with the Transformer architecture's capability to model complex dependencies in data, enables effective training of models for tasks such as language understanding, translation, and generation.
"""

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
    """
    Represents an 8-bit quantized weight for neural network parameters.

    Attributes:
        weight (jnp.array): The quantized weights.
        scales (jnp.array): The scale factors used for quantization.
    """
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
    """Container for the training state, encapsulating model parameters.

    Attributes:
        params (hk.Params): The parameters of the model.
    """

    params: hk.Params


def _match(qs, ks):
    """
    Determines if a sequence of regex patterns (qs) matches any contiguous subsequence of strings (ks).
    This utility function is often used for matching parameter names or paths in a hierarchical structure.

    Args:
        qs (Sequence[str]): A sequence of regex patterns to match.
        ks (Tuple[str, ...]): A tuple of strings against which the patterns are matched.

    Returns:
        bool: True if every pattern in qs has a corresponding match in a contiguous subsequence of ks,
              otherwise False.
    """
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def with_sharding_constraint(x, constraint):
    """
    Applies a sharding constraint to a JAX array. This function is used in SPMD programs to hint how the
    data should be partitioned across devices. If a physical mesh is not available, it simply returns the
    original array.

    Args:
        x (jax.Array): The array to apply the sharding constraint to.
        constraint (PartitionSpec): The sharding constraint to apply.

    Returns:
        jax.Array: The array with the sharding constraint applied, affecting its distribution across devices
                   in distributed computation setups.
    """
    if jax.experimental.maps.thread_resources.env.physical_mesh.empty:
        return x
    else:
        return pjit_sharding_constraint(x, constraint)


def cast_bfloat16(x):
    """
    Casts the input array to bfloat16 type if it is of floating-point type. This operation is often used to
    reduce memory consumption and potentially increase computation speed by using lower precision.

    Args:
        x (jax.Array): The input array.

    Returns:
        jax.Array: The array cast to bfloat16 if the original array was floating-point; otherwise, the array
                   is returned unchanged.
    """
    if x.dtype.kind == "f":
        return x.astype(jnp.bfloat16)
    else:
        return x


def ffn_size(emb_size, widening_factor):
    """
    Calculates the size of the feed-forward network (FFN) based on the embedding size and a widening factor.

    The calculated FFN size is adjusted to be a multiple of 8 for efficiency in hardware implementations.

    Args:
        emb_size (int): The size of the embeddings.
        widening_factor (float): The factor by which to widen the FFN relative to the embedding size.

    Returns:
        int: The adjusted size of the FFN.
    """
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
    logger.debug(f"emd_size: {emb_size} adjusted ffn_size: {_ffn_size}")
    return _ffn_size


def apply_rules(rules):
    """
    Constructs a function to apply a set of sharding rules for transformer parameters.

    This function is used to determine the sharding specifications for model parameters based on their roles
    and positions within the model architecture.

    Args:
        rules (List[Tuple[Sequence[str], PartitionSpec]]): A list of tuples where each tuple contains a sequence
            of strings representing the parameter path and the corresponding `PartitionSpec` to apply.

    Returns:
        Callable: A function that takes a parameter path and returns the appropriate `PartitionSpec` based
        on the provided rules.
    """
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
    """
    Represents key-value memory slots for a transformer layer, supporting efficient autoregressive decoding by caching past computed keys and values.

    Attributes:
        k (Optional[jax.Array]): Cached keys, shaped as [batch_size, sequence_len, num_kv_heads, key_size].
        v (Optional[jax.Array]): Cached values, shaped as [batch_size, sequence_len, num_kv_heads, key_size].
        step (Optional[jax.Array]): The current decoding step, indicating how many positions have been generated, shaped as [batch_size].
    """
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
    """
    Initializes layer memories for each transformer layer, providing a mechanism for efficient sequence generation by caching keys and values.

    Args:
        batch_size (int): The number of sequences being processed in parallel.
        sequence_len (int): The length of the sequences for which memory is allocated.
        num_kv_heads (int): The number of key-value pairs per head in the attention mechanism.
        key_size (int): The size of each key (and value) in the attention mechanism.
        num_layers (int): The number of transformer layers for which memory is initialized.
        step (Optional[jax.Array]): The initial decoding step for each sequence in the batch. Defaults to None, indicating no prior steps.
        dtype (Any): The data type for the memory arrays, typically jnp.bfloat16 for efficiency.

    Returns:
        List[KVMemory]: A list of initialized KVMemory instances for each layer in the transformer model.
    """
    return [
        KVMemory(
            k=jnp.zeros((batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype),
            v=jnp.zeros((batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype),
            step=step,
        )
        for _ in range(num_layers)
    ]


class Memory(NamedTuple):
    """
    A named tuple representing the complete memory state of a transformer model, encapsulating key-value memory slots for all layers.

    Attributes:
        layers (List[KVMemory]): A list of KVMemory instances, one for each layer in the transformer model.
    """
    # Self-attention key/value cache.
    layers: List[KVMemory]


class Router(hk.Module):
    """
    A module for routing inputs to experts in a Mixture of Experts (MoE) layer.

    Attributes:
        num_selected_experts (int): Number of experts to select for each input.
        data_axis (str | Tuple[str, ...]): The name(s) of the data axis for sharding.
        model_axis (str | Tuple[str, ...]): The name(s) of the model axis for sharding.
        shard_activations (bool): If True, shard activations according to the data and model axes.
        mesh (Any): The SPMD mesh for parallel computation.
        name (str): The name of the module.
    """
    def __init__(
        self,
        num_selected_experts: int,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        shard_activations: bool = False,
        mesh: Any = None,
        name: str = "router",
    ):
        """
        Initializes a router for directing inputs to experts in a Mixture of Experts (MoE) layer.
    
        Args:
            num_selected_experts (int): The number of experts to select for each input.
            data_axis (Union[str, Tuple[str, ...]]): The axis names over which data is sharded.
            model_axis (Union[str, Tuple[str, ...]]): The axis names over which model parameters are sharded.
            shard_activations (bool): Indicates whether to shard activations according to the data and model axes.
            mesh (Any): The SPMD mesh object for parallel computation.
            name (str): An optional name for the module.
        """
        super().__init__(name)
        self.shard_activations = shard_activations
        self.data_axis = data_axis
        self.model_axis = model_axis
        self.mesh = mesh
        self.num_selected_experts = num_selected_experts

    def compute_routing_prob(
        self, inputs: jax.Array, padding_mask: Optional[jax.Array], num_experts: int
    ):
        """
        Computes the routing probabilities for each input to be directed to each expert.
    
        This internal method calculates the logits that determine how inputs are distributed among the experts,
        based on learned criteria.
    
        Args:
            inputs (jax.Array): Input data to be routed.
            padding_mask (Optional[jax.Array]): Mask indicating which inputs are padding and should not be considered.
            num_experts (int): The total number of experts available.
    
        Returns:
            A tuple containing the routing probabilities, logits, and a dummy placeholder for compatibility.
        """
        return self._compute_routing_prob(inputs, padding_mask, num_experts)

    @hk.transparent
    def _compute_routing_prob(
        self,
        inputs: jax.Array,
        padding_mask: Optional[jax.Array],
        num_experts: int,
    ):
        """
        Computes the routing probabilities for directing inputs to the appropriate experts.

        Args:
            inputs (jax.Array): Input data to be routed, shaped as [batch_size, ..., input_dim].
            padding_mask (Optional[jax.Array]): An optional mask indicating padded elements in the input,
                shaped as [batch_size, seq_length], where padded positions are False.
            num_experts (int): The total number of experts available for routing.

        Returns:
            A tuple containing routing probabilities, routing logits, and a dummy value for compatibility,
            shaped as ([batch_size, seq_length, num_experts], [batch_size, seq_length, num_experts], int).
        """
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
    """
    A module implementing a Mixture of Experts (MoE) layer.

    Attributes:
        num_experts (int): The number of experts in the MoE layer.
        layer_fn (Callable): The function to be applied by each expert.
        router (Router): The router that routes inputs to experts.
        mesh (Any): The SPMD mesh for parallel computation.
        shard_activations (bool): If True, shard activations across data and model axes.
        data_axis (str | Tuple[str, ...]): The name(s) of the data axis for sharding.
        model_axis (str | Tuple[str, ...]): The name(s) of the model axis for sharding.
        name (Optional[str]): The name of the module.
    """
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
        """
        Initializes a Mixture of Experts layer with specified configuration.
    
        Args:
            num_experts (int): The total number of experts in the MoE layer.
            layer_fn (Callable): The function defining the computation performed by each expert.
            router (Router): The router that directs inputs to selected experts.
            mesh (Any): The optional SPMD mesh for parallel computation.
            shard_activations (bool): Whether to shard activations across distributed resources.
            data_axis (Union[str, Tuple[str, ...]]): Specifies how data is sharded for distributed computation.
            model_axis (Union[str, Tuple[str, ...]]): Specifies how model parameters are sharded.
            name (Optional[str]): An optional name for the module.
        """
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
        """
        Handles the inference call to the MoE layer, distributing inputs to selected experts based on routing.

        Args:
            inputs (jax.Array): Input data to be processed, shaped as [batch_size, seq_length, input_dim].
            padding_mask (Optional[jax.Array]): An optional mask for the inputs, where False indicates
                positions that should not be processed (e.g., padding), shaped as [batch_size, seq_length].

        Returns:
            jax.Array: The processed outputs after passing through the selected experts, shaped as
            [batch_size, seq_length, output_dim].
        """
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
    """
    Represents the output of the Multi-Head Attention (MHA) operation.

    Attributes:
        embeddings (jax.Array): The output embeddings from the MHA layer.
        memory (Any): The updated memory state post-attention operation.
    """

    embeddings: jax.Array
    memory: Any


class DecoderOutput(NamedTuple):
    """
    Encapsulates the output from a decoder layer within the transformer model, including the transformed embeddings and any updated memory state.

    Attributes:
        embeddings (jax.Array): The embeddings produced by the decoder layer, shaped as [batch_size, seq_length, embedding_dim].
        memory (Any): The updated memory state after processing by the decoder layer, useful for autoregressive decoding tasks.
    """
    embeddings: jax.Array
    memory: Any


class TransformerOutput(NamedTuple):
    """
    Represents the final output from the transformer model, including the final set of embeddings and any memory states that have been updated through the model's layers.

    Attributes:
        embeddings (jax.Array): The final output embeddings from the transformer, shaped as [batch_size, seq_length, embedding_dim].
        memory (Any): The final memory state of the model after all transformer layers have been applied.
    """
    embeddings: jax.Array
    memory: Any


@dataclass
class TransformerConfig:
    """
    Configuration class for setting up a Transformer model's architecture and its specific parameters.

    This class defines key architectural features of the transformer, including the size of embeddings,
    the dimensionality of keys and values in the attention mechanism, the number of layers, and more.
    It also includes configurations for advanced features like Mixture of Experts (MoE) and activation sharding.

    Attributes:
        emb_size (int): The size of the embedding vectors.
        key_size (int): The size of the key (and query) vectors in the attention mechanism.
        num_q_heads (int): The number of heads in the query part of the multi-head attention mechanism.
        num_kv_heads (int): The number of heads for keys and values in the multi-head attention.
        num_layers (int): The total number of layers in the transformer model.
        vocab_size (int): The size of the vocabulary that the model can understand.
        widening_factor (float): The factor by which the dimensionality of the feed-forward networks is widened relative to the embedding size.
        attn_output_multiplier (float): A scaling factor applied to the output of the attention mechanism, for controlling its magnitude.
        shard_activations (bool): Whether to shard activations across devices for parallel processing.
        num_experts (int): The number of experts in the Mixture of Experts (MoE) layer, if used.
        num_selected_experts (int): The number of experts selected for each input in the MoE layer.
        data_axis (Union[str, Tuple[str, ...]]): Specifies the axis names over which data is sharded for distributed computation.
        model_axis (Union[str, Tuple[str, ...]]): Specifies the axis names over which model parameters are sharded for distributed computation.
    """
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
    
    Creates an attention mask to specify which tokens in the key sequences can be attended to by each token in the query sequences.

    This utility is used in attention mechanisms to control the visibility of tokens, for purposes such as preventing future tokens from being attended to in autoregressive models.

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
    """
    Extends Haiku's Linear layer with optional sharding for use in distributed settings.

    This class allows specifying a `PartitionSpec` to shard the linear layer's weights across devices,
    which can be beneficial in large-scale models processed over multiple devices or nodes.

    Args:
        output_size (int): The size of the output dimension.
        with_bias (bool, optional): Whether to include a bias term. Defaults to True.
        sharding (Optional[P], optional): The sharding specification for distributing the layer's parameters.
        mesh (Any, optional): The SPMD mesh for parallel computation. Defaults to None.
        name (Optional[str], optional): An optional name for this module. Defaults to None.
        shard_axis (int, optional): The axis along which to shard the input data. Defaults to 0.
    """
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
        """
        Computes a linear transform of the input data.
    
        This method computes the matrix multiplication between the inputs and the layer's weight matrix, optionally adding a bias term.
    
        Args:
            inputs (jax.Array): The input tensor to be transformed, shaped as [batch_size, ..., input_features].
    
        Returns:
            jax.Array: The transformed tensor, shaped as [batch_size, ..., output_features].
        """

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
    """
    Implements Root Mean Square Layer Normalization.

    This variant of layer normalization scales inputs by the root mean square of their elements, optionally
    including a learnable scaling factor. It supports specifying a `PartitionSpec` for sharding the scale
    parameters across devices in distributed settings.

    Args:
        axis (Union[int, Sequence[int], slice]): The dimensions to normalize over.
        eps (float, optional): A small constant added to the denominator to improve numerical stability.
            Defaults to 1e-5.
        name (Optional[str], optional): An optional name for this module. Defaults to None.
        create_scale (bool, optional): Whether to include a learnable scaling factor. Defaults to True.
        sharding (Optional[P], optional): The sharding specification for the scale parameter. Defaults to None.
    """
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
        """
        Applies RMS normalization to the input tensor.
    
        This method normalizes the inputs by their root mean square value, optionally scaling the result by a learnable parameter to adjust the representation scale.
    
        Args:
            inputs (jax.Array): The input tensor to be normalized, shaped as [batch_size, ..., features].
    
        Returns:
            jax.Array: The RMS-normalized tensor, maintaining the input shape.
        """
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
    """
    Implements Rotary Position Embedding (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    RoPE encodes positional information dynamically by applying a rotation to the input embeddings based on their
    position in the sequence. This approach is designed to preserve the relative positional information across
    different sequence lengths and tasks.

    Args:
        dim (int): The dimensionality of the embeddings to be rotated, must be even.
        name (Optional[str], optional): An optional name for this module. Defaults to None.
        base_exponent (int, optional): The base of the exponent used to calculate rotary frequencies.
            Defaults to 10000.
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
        """
        Applies Rotary Position Embedding (RoPE) to the input embeddings.
    
        This method dynamically encodes positional information by applying a rotation based on the position in the sequence, enhancing the model's ability to capture sequential dependencies.
    
        Args:
            x (jax.Array): Input embeddings to apply RoPE to, shaped as [batch_size, seq_length, embedding_dim].
            seq_dim (int): The dimension index of the sequence length in the input tensor.
            offset (jax.Array): The offset to apply to the position indices, useful for continuation of sequences across batches.
            const_position (Optional[int]): A constant position value to use for all positions, instead of a range.
            t (Optional[jax.Array]): Explicit tensor of position values, shaped as [seq_length].
    
        Returns:
            jax.Array: The input embeddings with RoPE applied, maintaining the input shape.
        """
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
    """
    Implements the Multi-Head Attention mechanism, a key component of Transformer architectures.

    This module allows each token in the input sequence to attend to all tokens in the key and value sequences, with multiple "heads" learning different attention patterns.

    Attributes:
        num_q_heads (int): The number of heads for the queries.
        num_kv_heads (int): The number of heads for the keys and values.
        key_size (int): The size of the key vectors.
        with_bias (bool): Whether to include bias terms in the linear transformations.
        value_size (Optional[int]): The size of the value vectors, defaults to the key size if not specified.
        model_size (Optional[int]): The size of the output dimension from the attention mechanism, defaults to the total size of all heads if not specified.
        attn_output_multiplier (float): A scaling factor applied to the output of the attention mechanism.
        data_axis (Union[str, Tuple[str, ...]]): The axis names over which data is sharded for distributed computation.
        model_axis (Union[str, Tuple[str, ...]]): The axis names over which model parameters are sharded for distributed computation.
        name (Optional[str]): An optional name for the module.
    """
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
        """
        Initializes the Multi-Head Attention module with the provided configuration parameters.
    
        Args:
            num_q_heads (int): Number of query heads for the multi-head attention mechanism.
            num_kv_heads (int): Number of key/value heads for the multi-head attention mechanism.
            key_size (int): Dimensionality of key vectors in the attention mechanism.
            with_bias (bool): Whether to include a bias term in the attention weight computation.
            value_size (Optional[int]): Dimensionality of value vectors, defaults to key size if not specified.
            model_size (Optional[int]): Overall size of the model's output dimension.
            attn_output_multiplier (float): Multiplier for scaling the output of the attention mechanism.
            data_axis (Union[str, Tuple[str, ...]]): Data sharding axis names for distributed computation.
            model_axis (Union[str, Tuple[str, ...]]): Model parameter sharding axis names for distributed computation.
            name (Optional[str]): An optional name for the module.
        """
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
        """
        Computes the multi-head attention over the input queries, keys, and values.

        Args:
            query (jax.Array): Query vectors, shaped as [batch_size, seq_length, model_dim].
            key (Optional[jax.Array]): Key vectors. If None, uses query as key.
            value (Optional[jax.Array]): Value vectors. If None, uses query as value.
            mask (Optional[jax.Array]): An optional mask to prevent attention to certain positions,
                shaped as [batch_size, 1, seq_length, seq_length].
            kv_memory (Optional[KVMemory]): Optional memory for keys and values to support efficient
                autoregressive decoding.
            mesh (Any): The SPMD mesh for parallel computation, if applicable.

        Returns:
            MHAOutput: A named tuple containing the output embeddings and updated memory.
        """
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
        """
        Projects input embeddings into multiple head spaces for queries, keys, or values.
    
        This internal method creates separate embeddings for each attention head by applying a linear transformation,
        allowing the multi-head attention mechanism to explore different subspace relationships in the data.
    
        Args:
            x (jax.Array): Input tensor to project, shaped as [batch_size, seq_length, embedding_dim].
            head_size (int): The dimensionality of each head's subspace.
            num_heads (int): The number of heads to project into.
            name (Optional[str]): A name for the operation, distinguishing between queries, keys, and values.
            sharding (Optional[PartitionSpec]): The sharding specification for distributing computation across devices.
    
        Returns:
            jax.Array: Projected embeddings for multiple heads, shaped as [batch_size, seq_length, num_heads, head_size].
        """
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
    """
    A specialized module encapsulating the Multi-Head Attention (MHA) operation within a transformer model.

    This module orchestrates the application of MHA, including the computation of queries, keys, and values, and the subsequent attention and aggregation operations.

    Attributes:
        num_q_heads (int): Number of heads for the query part of the MHA.
        num_kv_heads (int): Number of heads for the key and value parts of the MHA.
        key_size (int): Size of the keys (and queries) in the attention mechanism.
        attn_output_multiplier (float): Scaling factor applied to the output of the attention mechanism.
        data_axis (Union[str, Tuple[str, ...]]): Axis names over which data is sharded for distributed computation.
        model_axis (Union[str, Tuple[str, ...]]): Axis names over which model parameters are sharded for distributed computation.
        mesh (Any): The SPMD mesh for parallel computation.
    """

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
        """
        Processes inputs through a Multi-Head Attention block.
    
        This method applies multi-head attention to the inputs using the provided mask for attention scoring,
        and optionally utilizes cached memory for keys and values to enhance efficiency in autoregressive models.
    
        Args:
            inputs (jax.Array): Input embeddings, shaped as [batch_size, seq_length, embedding_dim].
            mask (jax.Array): Attention mask, shaped as [batch_size, 1, seq_length, seq_length], to control visibility between tokens.
            layer_memory (Optional[KVMemory]): Cached keys and values from previous steps for efficient attention computation in autoregressive decoding.
    
        Returns:
            MHAOutput: The output from the multi-head attention block, including the transformed embeddings and updated memory.
        """
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
    """
    Implements a dense (fully connected) block within a transformer layer, typically following multi-head attention.

    This block applies one or more linear transformations to its inputs, often including non-linear activations and potentially other operations like dropout for regularization.

    Attributes:
        num_q_heads (int): The number of heads for the query in the preceding MHA layer.
        num_kv_heads (int): The number of key/value pairs per head in the MHA layer.
        key_size (int): The size of the keys in the MHA layer.
        widening_factor (float): Factor by which the dimensionality of the feed-forward network is increased.
        sharding_constraint (bool): Whether to apply a sharding constraint for distributed computation.
        mesh (Any): The SPMD mesh for parallel computation.
    """
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
        """
        Applies a series of dense transformations to the inputs.
    
        This method constitutes the feedforward network part of a transformer layer, applying linear transformations
        followed by non-linear activations to model complex data relationships beyond what attention mechanisms capture.
    
        Args:
            inputs (jax.Array): Input embeddings from the previous layer or block, shaped as [batch_size, seq_length, embedding_dim].
    
        Returns:
            jax.Array: The output embeddings after applying the dense block transformations, maintaining the input shape.
        """
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
    """
    Represents a single layer in the decoder stack of a transformer model. This layer processes input embeddings through
    a multi-head attention mechanism followed by position-wise feed-forward networks, with normalization and skip connections
    applied as per the standard transformer architecture.

    Attributes:
        num_q_heads (int): Number of query heads for multi-head attention.
        num_kv_heads (int): Number of key/value pairs per attention head.
        key_size (int): Size of keys in the attention mechanism.
        num_layers (int): Total number of transformer layers.
        num_experts (int): Number of experts in the Mixture of Experts layer, if used.
        layer_index (Optional[int]): Index of this layer within the overall model; used for layer-specific configurations or logging.
        num_selected_experts (int): Number of experts selected for each input in the MoE layer.
        widening_factor (float): Factor by which the dimensionality of the feed-forward network is increased.
        name (Optional[str]): An optional name for the layer.
        data_axis (Union[str, Tuple[str, ...]]): Axis names for data sharding in distributed computation.
        model_axis (Union[str, Tuple[str, ...]]): Axis names for model parameter sharding in distributed computation.
        shard_activations (bool): Whether activations should be sharded across devices.
        attn_output_multiplier (float): Scaling factor for the output of the attention mechanism.
        mesh (Any): SPMD mesh for parallel computation.
    """


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
        """
        Transforms input embedding sequences to output embedding sequences.
        Processes input embeddings through a single layer of the decoder.

        This method applies multi-head attention followed by position-wise feed-forward networks,
        including any necessary normalization and skip connections, as per the transformer architecture.

        Args:
            inputs (jax.Array): Input embeddings, shaped [batch_size, seq_length, model_dim].
            mask (jax.Array): Attention mask, shaped [batch_size, 1, seq_length, seq_length], used to prevent
                attention to future positions.
            padding_mask (Optional[jax.Array]): Mask indicating which positions are padding tokens,
                to exclude them from attention calculations.
            layer_memory (Optional[KVMemory]): Memory state for storing past key/value pairs for efficient
                autoregressive decoding.

        Returns:
            DecoderOutput: Named tuple containing output embeddings and updated memory state.
        """

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
    """
    Represents the output of the language model after processing an input sequence of tokens.

    This output encapsulates both the logits representing the model's predictions for the next token
    in the sequence and the updated model state, which includes any memory or state information
    that needs to be carried over for generating subsequent tokens.

    Attributes:
        logits (jax.Array): The logits for the next token predictions, shaped as [batch_size, sequence_length, vocab_size].
        model_state (Any): The updated state of the model after processing the input sequence, which may include
                           memory states for layers that utilize recurrence or caching for efficiency.
    """
    logits: jax.Array
    model_state: Any


class InOutEmbed(hk.Embed):
    """
    A module for embedding input tokens into a low-dimensional space continuous vector space and for projecting the outputs of
    a transformer back into the vocabulary space. This module supports tying the weights between the input
    embedding and the output projection for parameter efficiency.

    Attributes:
        vocab_size (Optional[int]): The size of the vocabulary.
        embed_dim (Optional[int]): The dimensionality of the embedding vectors.
        sharding (Optional[PartitionSpec]): Specifies how the embedding parameters should be sharded across
                                            devices for distributed computation.
        name (Optional[str]): An optional name for the module.
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        sharding: Optional[P] = None,
        name: Optional[str] = None,
    ):
        """
        Initializes an embedding module that can be used for both input token embeddings and output logits projection in a transformer model.
    
        This shared embedding layer helps reduce the number of parameters by tying the weights between the input embedding and the output projection layers.
    
        Args:
            vocab_size (Optional[int]): The size of the vocabulary.
            embed_dim (Optional[int]): The dimensionality of the embedding vectors.
            sharding (Optional[PartitionSpec]): The sharding specification for distributing the embedding parameters across devices.
            name (Optional[str]): An optional name for the module.
        """
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            name=name,
        )
        self.sharding = sharding

    @property
    def embeddings(self):
        """
        Retrieves the embedding matrix from the module's parameters.
    
        This method is useful for operations that need direct access to the embeddings, such as output projection in language models.
    
        Returns:
            jax.Array: The embedding matrix, shaped as [vocab_size, embed_dim].
        """

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
        """
        Projects transformer model outputs back into the vocabulary space using the transposed embedding matrix.
    
        This method effectively performs the inverse of the embedding operation, converting model output embeddings into logits over the vocabulary.
    
        Args:
            inputs (jax.Array): The output embeddings from the transformer model, shaped as [batch_size, seq_length, embed_dim].
    
        Returns:
            jax.Array: The logits over the vocabulary for each token position, shaped as [batch_size, seq_length, vocab_size].
        """
        return jnp.dot(inputs, self.embeddings.T.astype(inputs.dtype))


@dataclass
class LanguageModelConfig:
    """
    Configuration class for an autoregressive language model based on the Transformer architecture.

    Attributes:
        model (TransformerConfig): The transformer model configuration.
        vocab_size (int): The size of the vocabulary.
        pad_token (int): The token used for padding sequences.
        eos_token (int): The end-of-sentence token.
        sequence_len (int): The maximum sequence length the model can handle.
        model_size (int): The dimensionality of the model embeddings.
        embedding_init_scale (float): Initial scale for embedding parameter initialization.
        embedding_multiplier_scale (float): Multiplier for scaling the embedding vectors.
        output_multiplier_scale (float): Multiplier for scaling the output logits.
        name (Optional[str]): Name of the language model configuration.
        fprop_dtype (Any): Data type for forward propagation computations.
        model_type (Optional[str]): Type of the model, if applicable.
        init_scale_override (Optional[float]): Override for the initial scale of parameters, if needed.
        shard_embeddings (bool): Whether to shard embeddings across the specified axes.
    """

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
    """
    A high-level module for autoregressive language modeling using a Transformer architecture. This module
    integrates components such as embedding layers, transformer blocks, and output layers to process sequences
    of tokens and generate predictions for the next tokens in the sequence.

    The LanguageModel is designed for tasks such as text generation, where it can be used to produce coherent
    and contextually relevant text based on a given prompt.

    Attributes:
        model (Transformer): The core transformer model used for processing input token sequences.
        config (LanguageModelConfig): Configuration parameters for the language model, including details about
                                      the architecture, embeddings, and output processing.
        fprop_dtype (Any): The data type to use for forward propagation calculations, typically set to jnp.bfloat16
                           for efficiency.
        name (Optional[str]): An optional name for the module. Useful for distinguishing between multiple instances.
        mesh (Any): The SPMD mesh for parallel computation, supporting distributed training and inference.
    """

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
        """
        Forward pass, producing a sequence of logits. 
        Generates logits for the next token predictions based on input tokens and optional memory state.

        Args:
            tokens (jax.Array): Input tokens to the language model, shaped as [batch_size, seq_length].
            memory (Optional[Memory]): Optional memory state from previous steps, for autoregressive generation.
            batch (Dict[str, jax.Array]): Additional batch information, unused here.
            last_hid_only (bool): If True, returns only the last hidden state instead of logits.
            length (Optional[jax.Array]): Specifies the length of each sequence in the batch for processing
                only up to those lengths.

        Returns:
            LanguageModelOutput: A named tuple containing the logits for next token predictions and the
            updated memory state.
        """
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
        """
        Initializes the memory for the language model, suitable for autoregressive sequence generation tasks.
    
        Args:
            batch_size (int): The number of sequences for which to initialize memory.
            seq_len (int): The length of sequences for memory allocation.
            dtype (Any): Data type for the memory arrays, typically jnp.bfloat16.
    
        Returns:
            Memory: The initialized memory structure for storing keys, values, and decoding steps across transformer layers.
        """
        return self.model.init_memory(batch_size=batch_size, sequence_len=seq_len, dtype=dtype)

    def prefill_memory(self, prompts, memory):
        """
        Optionally pre-fills the transformer's memory with information from a provided prompt, enhancing efficiency
        in subsequent autoregressive generation steps by caching necessary computations from the prompt processing.
    
        Args:
            prompts (jax.Array): The prompt tokens from which to generate subsequent tokens.
            memory (Memory): The memory state to update with information derived from the prompts.
    
        Returns:
            Tuple[jax.Array, Memory]: The logits produced by processing the prompts and the updated memory state.
        """
        # Pad to the left and right align?
        # Basically assume prompt is already padded
        model_output = self(prompts, memory=memory)
        return model_output.logits, model_output.model_state


@dataclass
class Transformer(hk.Module):
    """
    Core transformer module that implements the foundational architecture of a transformer-based model. This module
    is capable of processing sequences of embeddings through multiple layers of self-attention and feed-forward
    networks, optionally including advanced techniques like mixture of experts (MoE) and activation sharding
    for efficient large-scale parallel computation.

    Attributes:
        num_q_heads (int): Number of heads in the query part of the multi-head attention mechanism.
        num_kv_heads (int): Number of heads for the keys and values in the multi-head attention.
        key_size (int): Dimensionality of the key (and query) vectors in the attention mechanism.
        widening_factor (float): Factor by which to widen the dimensionality of the feed-forward network relative to the embeddings.
        init_scale (float): Initial scale for parameter initialization.
        mesh (Any): The SPMD mesh for parallel computation.
        attn_output_multiplier (float): Multiplier for the output of the attention mechanism.
        shard_activations (bool): Whether to shard activations across devices in distributed settings.
        num_layers (int): Number of transformer layers to stack in the model.
        num_experts (int): Number of experts in the MoE layer, if used.
        num_selected_experts (int): Number of experts selected for each input token in the MoE layer.
        data_axis (Union[str, Tuple[str, ...]]): Axis names for sharding data across devices.
        model_axis (Union[str, Tuple[str, ...]]): Axis names for sharding model parameters across devices.
        name (Optional[str]): An optional name for the module.
    """

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
        """
        Initializes the memory state for the transformer model.

        This is particularly useful for autoregressive tasks where past key and value pairs are cached
        to improve efficiency in generating sequences.

        Args:
            batch_size (int): The batch size for which to initialize memory states.
            sequence_len (int): The sequence length for initializing the size of memory buffers.
            dtype (Any): The data type for the memory arrays, typically jnp.bfloat16 for efficiency.

        Returns:
            Memory: A named tuple representing the initialized memory state for each layer.
        """
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
        """
        Processes input embeddings through the transformer model.
        Transforms input embedding sequences to output embedding sequences.

        Args:
            embeddings (jax.Array): Input embeddings to be processed by the transformer, shaped as
                [batch_size, seq_length, model_dim].
            mask (jax.Array): Mask indicating valid positions within the input, to control which positions
                are allowed to attend to each other, shaped as [batch_size, seq_length].
            memory (Optional[Memory]): Optional memory state for the transformer to support autoregressive
                decoding or similar use cases.

        Returns:
            TransformerOutput: A named tuple containing the transformed embeddings and the final state
            of the memory after processing.
        """

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
