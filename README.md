# Grok-1

This repository contains JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` - see [Downloading the weights](#downloading-the-weights)

Then, run

```shell
pip install -r requirements.txt
python run.py
```

to test the code.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Model Specifications

Grok-1 is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):
```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.

# Table of contents
This repository contains Python code for grok-1. Below is a breakdown of the main components and features provided by this codebase:

# :book: model.py
i will try to breakdown each classes in detail

## :page_with_curl: functions, varibales and constants

- `ffn_size`: This function computes the size (number of units) for the feed-forward network (FFN) layer in the transformer architecture. The FFN size is typically larger than the embedding size to increase the model's expressive power. The function takes two arguments:

- `emb_size`: The size of the input embeddings.

- `widening_factor`: A multiplier used to determine the FFN size relative to the embedding size. 

    The function first calculates the `FFN size` as `int(widening_factor * emb_size) * 2 // 3`. The * 2 // 3 part is a heuristic to reduce the `FFN size` slightly while maintaining good performance. Then, it ensures that the `FFN size` is a multiple of 8 by adding the smallest positive number needed to make it divisible by 8. This is likely done for efficient computations on certain hardware architectures.

- `apply_rules`: This function returns a closure (inner function) that applies a set of rules to reshape (reshard) the parameters of a neural network model based on regular expression patterns. The rules are provided as a list of tuples, where each tuple contains a list of regular expression patterns and a corresponding reshape specification (PartitionSpec). 

    The inner function `_apply_rules` takes a path (list of keys) and a value (tensor) as input. It flattens the path and checks if any of the provided rules (regular expressions) match the flattened path. If a match is found, the corresponding PartitionSpec is applied to the tensor, effectively reshaping it according to the specified partitioning scheme. This function is likely used to optimize the model for distributed training across multiple devices or accelerators.

- `cast_bfloat16`:  This is a simple utility function that casts the input tensor (x) to the bfloat16 data type if the tensor's data type is a floating-point type. The bfloat16 data type is a truncated 16-bit floating-point format that can provide higher computational performance on certain hardware architectures, such as Google's TPUs, while maintaining reasonable precision. If the input tensor is not a floating-point type, the function returns the tensor unchanged.


- `with_sharding_constraint`: This function applies a sharding constraint to the input tensor (x). Sharding is a technique used in distributed training to split the model parameters and computations across multiple devices or accelerators. The sharding constraint specifies how the tensor should be partitioned or reshaped for efficient parallel computations. The function takes two arguments:

    - `x`: The input tensor to be reshaped.

    - `constraint`: A PartitionSpec object that specifies the desired reshaping or partitioning scheme.

    If the current environment does not support distributed training (i.e., jax.experimental.maps.thread_resources.env.physical_mesh.empty is True), the function returns the input tensor unchanged. Otherwise, it applies the specified sharding constraint to the tensor using the pjit_sharding_constraint function from JAX.


- `match`: This is a helper function used by `apply_rules`. 
    
    It takes two arguments:

    `qs`: A tuple of regular expression patterns.

    `ks`: A tuple of strings representing the flattened path.

    The function compiles the regular expressions in qs and checks if any window (consecutive sublist) of strings in ks matches all the compiled regular expressions simultaneously. If a match is found, it returns True; otherwise, it returns False. This function is likely used by apply_rules to determine if a specific set of rules (regular expressions) should be applied to a given path in the neural network model.

- `init_layer_memories`: this function initializes memory structures for each layer in a neural network, specifically for key-value attention.
    It takes several arguments:

    - `batch_size`: An integer representing the batch size.

    - `sequence_len`: An integer representing the length of the input sequence.

    - `num_kv_heads`: An integer representing the number of key-value (KV) attention heads.

    - `key_size`: An integer representing the size of the keys and values.

    - `num_layers`: An integer representing the total number of layers.

    - `step`: An optional parameter (of type jax.Array) that defaults to None.

     - `dtype`: An optional parameter (of type jnp.bfloat16) that defaults to a specific data type.

- `hk_rms_norm`: this function applies a customized form of LayerNorm to the input data x. The specifics of the normalization are determined by the RMSNorm.

    It takes 3 arguments:
    
    - `x`: A JAX array (presumably representing some data).
    
    - `fixed_scale`: A boolean flag (defaulting to False).
    
    - `sharding`: An optional parameter (defaulting to None).


- `make_attention_mask`: is a utility function that generates an attention mask suitable for 1D attention mechanisms. The resulting mask is shaped appropriately for use in neural network layers that require attention weights. The function computes an attention mask based on the provided query and key inputs. The mask is designed for 1D attention (i.e., when both query and key inputs are 1D sequences).
The resulting mask has shape [batch..., 1, len_q, len_kv], where len_q represents the query length and len_kv represents the key length

    it takes 3 arguments:

    - `query_input`: A JAX array representing the query input (e.g., from the self-attention mechanism).
    
    - `key_input`: A JAX array representing the key input (e.g., from the self-attention mechanism).

    - `pairwise_fn`: An optional parameter (defaulting to jnp.multiply) that specifies the elementwise comparison function used for creating the mask.
    
    - `dtype`: An optional parameter (defaulting to jnp.bfloat16) that specifies the data type of the resulting mask.



- `rotate_half`: this function is to obtain the rotated counterpart of each feature in the input array x.it rearranges the features in the input array by swapping the first and second halves along the last axis.

    It takes a single 
    - `x`: A JAX array representing of features.

- `layer_norm`: it just a wrapper around `hk_rms_norm`



- ### `TRANSFORMER_PARTITION_RULES`: 
    is a list of tuples that define the partitioning rules for the parameters of a transformer model. These rules are used by the `apply_rules` function to reshape (reshard) the model parameters for efficient distributed training across multiple devices or accelerators.

    Each tuple in the TRANSFORMER_PARTITION_RULES list consists of two elements:

    1. A tuple of regular expression patterns that match the parameter names or paths in the model.

    2. A `PartitionSpec` object that specifies how the matched parameters should be partitioned or reshaped.

    let's dive in some of the partitioning rules defined in `TRANSFORMER_PARTITION_RULES`:

    - #### `(("multi_head_attention", "(query|key|value)", "w"), P("data", "model"))`: 

        This rule matches the weight tensors (w) of the query, key, and value projections in the multi-head attention module. It specifies that these weights should be partitioned along the "data" and "model" dimensions, which means they will be split across multiple devices or accelerators along those dimensions.


    - #### `(("multi_head_attention", "(query|key|value)", "b"), P(None))`: 
        This rule matches the bias tensors (b) of the query, key, and value projections in the multi-head attention module. It specifies that these biases should not be partitioned (indicated by P(None)), meaning they will be replicated across all devices.

    - #### `((r"decoder_layer_[0-9]+", "linear", "w"), P("data", "model"))`: 

        This rule matches the weight tensors (w) of the linear projections in the decoder layers of the transformer model. The regular expression r"decoder_layer_[0-9]+" matches any parameter path containing "decoder_layer_" followed by a number. These weights are partitioned along the "data" and "model" dimensions.

    - #### `((r"decoder_layer_[0-9]+", "linear", "b"), P(None))`: 
        Similar to the previous rule, but it matches the bias tensors (b) of the linear projections in the decoder layers, and these biases are not partitioned.

    - Rules for partitioning the parameters of layer normalization (layer_norm, rms_norm) and router (router) modules are also included.

    - Rules for partitioning the parameters of the Mixture of Experts (MoE) module, including the `linear projections (linear, linear_v, linear_1)` and normalization layers `(layer_norm, rms_norm)`.

        These partitioning rules aim to distribute the computationally intensive operations, such as matrix multiplications, across multiple devices, while replicating smaller tensors (like biases) to reduce communication overhead.

        By applying these partitioning rules, the model can take advantage of the combined memory and computational resources of multiple devices, enabling training of larger models or processing of larger batch sizes.

- ### `LM_PARTITION_RULES`

    `LM_PARTITION_RULES` is a list of tuples that define the partitioning rules for the parameters of the language model component in the codebase. These rules are used to specify how the parameters of the language model should be partitioned (reshaped) across multiple devices or accelerators for efficient distributed training.

    The `LM_PARTITION_RULES` list contains the following rules:

    - #### `(("language_model", "positional_embeddings"), P(None, ("data", "model")))`: 

        This rule matches the positional embeddings tensor in the language model module. The PartitionSpec `P(None, ("data", "model"))` specifies that this tensor should be partitioned along the "data" and "model" dimensions, but not partitioned along the leading dimension (represented by None). This means that the positional embeddings will be split across multiple devices along the "data" and "model" dimensions, but replicated along the leading dimension (e.g., batch dimension).

    - #### `(("language_model", "in_out_embed", "embeddings"), P(None, ("data", "model")))`: 
        This rule matches the embeddings tensor of the InOutEmbed module (used for input and output embeddings) in the language model. Similar to the previous rule, it specifies that this tensor should be partitioned along the "data" and "model" dimensions, while being replicated along the leading dimension.

    - #### `(("language_model", "rms_norm"), P(None))`: 
        This rule matches the parameters of the RMSNorm layer in the language model. The PartitionSpec P(None) indicates that these parameters should not be partitioned at all and should be replicated across all devices.

    By applying these partitioning rules, the language model's parameters are reshaped and distributed across multiple devices in a way that aims to balance the computational load and memory usage. The input and output embeddings, which are typically large tensors, are partitioned along the "data" and "model" dimensions to distribute their storage and computations. At the same time, smaller tensors like the normalization layer parameters are replicated across all devices to minimize communication overhead.


## :page_with_curl: QuantizedWeight8bit
The QuantizedWeight8bit class is a data structure that represents quantized weights in a neural network. Quantization is a technique used to reduce the precision of weight values from the typical 32-bit floating-point representation to a lower-precision format, such as 8-bit integers, to save memory and improve computational efficiency, especially on hardware accelerators like GPUs or TPUs.

The QuantizedWeight8bit class has two main attributes:

weight : This is a NumPy array that holds the quantized weight values, represented as 8-bit integers.

scales : This is a NumPy array that holds the scaling factors associated with each quantized weight.

During the model initialization or loading phase, the original 32-bit floating-point weights are quantized to 8-bit integers and packed into QuantizedWeight8bit instances.
When performing computations in the neural network, such as linear transformations or convolutions, the quantized weights are used instead of the original 32-bit weights. This is done by applying the scaling factors stored in the scales attribute to recover approximate values of the original weights.
After the computation, the results are typically de-quantized (converted back to higher precision) for further processing or output.
By using QuantizedWeight8bit, the model can achieve significant memory savings and potentially faster computations, especially on hardware accelerators optimized for low-precision arithmetic. However, there is a trade-off between the level of quantization and the model's accuracy, as quantization introduces approximation errors. Careful calibration and quantization-aware training techniques are often employed to minimize the accuracy loss due to quantization.

## :page_with_curl: TrainingState 
The TrainingState class is a simple data structure defined as a NamedTuple in Python. It is used to hold the parameters (weights) of a neural network model during the training process. In this specific code, the TrainingState only contains one field:

```python
TrainingState(NamedTuple):
    """Container for the training state."""

    params: hk.Params
```
Here, params is an instance of hk.Params, which is a data structure provided by the Haiku library (a JAX-based neural network library) to represent the parameters (weights) of a neural network model.

The NamedTuple is a lightweight data structure provides a way to define immutable tuples with named fields. It is similar to a class, but it is more lightweight and efficient, making it suitable for storing and passing around data structures that don't require additional methods or behavior.

the TrainingState serves as a lightweight container to hold and manage the model parameters during the training process, allowing for efficient manipulation and updating of the model's weights.

## :page_with_curl: KVMemory

`KVMemory` is a `NamedTuple` data structure used to store and manage the key-value memory state in the transformer architecture. It is defined as follows:

```python
class KVMemory(NamedTuple):
    """Container for the key-value memory."""
    k: Optional[jax.Array]
    v: Optional[jax.Array]
    step: Optional[jax.Array]
```

This data structure has three fields:

`k`: This field holds the key vectors for the multi-head attention mechanism. It is an optional JAX array with a shape of `(batch_size, sequence_length, num_kv_heads, key_size)`. During the initial state, it can be None.

`v`: This field holds the value vectors for the multi-head attention mechanism. It is an optional JAX array with a shape similar to `k`, i.e., `(batch_size, sequence_length, num_kv_heads, value_size)`. During the initial state, it can be None.

`step`: This field is an optional JAX array that keeps track of the current step or position in the sequence. It is typically used for causal or autoregressive attention, where the model needs to attend only to the previous positions in the sequence. The shape of `step` is `(batch_size,)`.

The `KVMemory` data structure is used to store and update the key-value memory state as the transformer processes each input sequence. During the forward pass, the key and value vectors are computed from the input and stored in this memory. In the subsequent steps, the stored key-value vectors are used to compute the attention weights and output vectors, allowing the model to attend to the previous positions in the sequence.

The `step` field is incremented after each step to keep track of the current position in the sequence. This information is used to mask out the future positions in the attention computation, ensuring that the model only attends to the past positions.

By encapsulating the key-value memory state in a dedicated data structure, the code can easily manage and update the memory state as the transformer processes each input sequence. This memory mechanism is crucial for the transformer architecture to capture long-range dependencies and generate coherent outputs in sequence-to-sequence tasks, such as language modeling or machine translation.

## :page_with_curl: Router 
The `Router` class is a module used in the Mixture of Experts (MoE) layer of the transformer architecture. It is responsible for routing the input tokens to a subset of experts (specialized feed-forward networks) based on learned routing probabilities. let's dive in 😊

`__init__`:

This is the constructor method that initializes the Router instance.

- It takes the following arguments:
    
    `num_selected_experts`: The number of experts to select for each input token.

    `data_axis, model_axis`: Axes along which to partition the data and model dimensions for distributed training.

    `shard_activations`: A boolean indicating whether to shard activations (input tensors) across devices.

    `mesh`: An optional mesh object used for distributed training.
    
    `name`: An optional name for the module.

 `compute_routing_prob`:

This method computes the routing probabilities for each input token and the selected experts.
- It takes the following arguments:

    `inputs`: The input tensor to be routed.

    `padding_mask`: An optional mask tensor indicating which positions in the input are padded.

    `num_experts`: The total number of experts available.

- It calls the `_compute_routing_prob` method (explained below) to calculate the routing probabilities and logits.

- It returns a tuple containing the routing probabilities, routing logits, and an auxiliary value (currently 0).

`_compute_routing_prob`:

This is a transparent method (decorated with `@hk.transparent`) that performs the actual computation of routing probabilities and logits.

- It takes the same arguments as `compute_routing_prob`.
- It first converts the input tensor to float32 precision for the routing computation.

- It then calls the _router_weights method (explained below) to obtain the routing logits.

- The routing logits are passed through a softmax function to get the routing probabilities.

- If a padding mask is provided, it is applied to the routing probabilities to mask out padded positions.

- It returns the routing probabilities, routing logits, and the auxiliary value (0).

`_router_weights`:

- This is another transparent method that computes the routing logits by applying a linear transformation to the input tensor.

- It takes the following arguments:

    `x`: The input tensor.

    `num_experts`: The total number of experts available.

    `sharding`: An optional PartitionSpec indicating how to partition the weights.

- It retrieves the weight tensor w from the module's parameters, initialized with a constant value of 0.

- If a sharding specification is provided, it applies the sharding constraint to the weight tensor.

- It computes the routing logits by performing a matrix multiplication between the input tensor and the weight tensor.

- It returns the routing logits tensor.

The `Router` class is used in conjunction with the `MoELayer` class, which implements the Mixture of Experts layer. The routing probabilities computed by the Router are used to select the top `num_selected_experts` for each input token, and the corresponding expert networks are applied to the selected inputs. This allows the transformer to dynamically route different inputs to specialized experts, potentially improving the model's performance and capacity.

## :page_with_curl: MoELayer 

The `MoELayer` class is a module that implements the Mixture of Experts (MoE) layer in the transformer architecture. The MoE layer is a technique that allows the model to dynamically route different inputs to specialized feed-forward networks, called experts, based on learned routing probabilities. let's try to explain the `MoELayer` class 😋:

1.`__init__`:

- This is the constructor method that initializes the MoELayer instance.

- It takes the following arguments:

    `num_experts`: The total number of expert networks in the MoE layer.

    `layer_fn`: A callable function that represents the expert network (e.g., a feed-forward network).

    `router`: An instance of the Router class, responsible for computing the routing probabilities.

    `mesh`: An optional mesh object used for distributed training.

    `shard_activations`: A boolean indicating whether to shard activations (input tensors) across devices.

    `data_axis, model_axis`: Axes along which to partition the data and model dimensions for distributed training.

    `name`: An optional name for the module.

2.`_inference_call`:

- This is the main method that performs the forward pass through the MoE layer during inference (evaluation).

- It takes the following arguments:

    `inputs`: The input tensor to be processed by the MoE layer.

    `padding_mask`: An optional mask tensor indicating which positions in the input are padded.

- It calls the `router.compute_routing_prob` method to obtain the routing probabilities and logits for the input tokens.

- It selects the top `num_selected_experts` experts for each input token based on the routing probabilities.

- It creates a broadcasted version of the input tensor, duplicating it num_selected_experts times for each token position.

- It initializes the expert networks (specified by `layer_fn`) by creating a batched version of the initialization function.

- It performs a specialized matrix multiplication operation `(moe_slow_matmul1 and moe_slow_matmul2)` to apply the selected expert networks to the broadcasted inputs, weighted by the routing probabilities.

- If the expert weights are quantized `(QuantizedWeight8bit)`, it applies dequantization by multiplying the weights with their corresponding scaling factors.
The final output is a weighted sum of the expert outputs for each input token, where the weights are the routing probabilities.

The `MoELayer` class encapsulates the logic for routing inputs to specialized experts, applying the expert networks in a efficient and parallelized manner, and combining the expert outputs based on the learned routing probabilities. This approach allows the transformer model to allocate its capacity dynamically, potentially improving its performance and expressiveness on complex tasks.


## :page_with_curl: Memory 

This is a named tuple that represents the key-value `memory` used in the multi-head attention mechanism of the Transformer model.

It contains a list of `KVMemory` instances, one for each layer in the Transformer.

Each `KVMemory` instance stores the keys (k), values (v), and the current step (step) for that layer's attention mechanism.

This `memory` is used to cache the key-value pairs from previous time steps, allowing the model to attend to the entire sequence history during autoregressive decoding.

## :page_with_curl: Router 
The `Router` is a module responsible for routing the input tokens to a subset of experts in the Mixture of Experts (MoE) architecture.

It computes the routing probabilities for each input token, determining which experts should be activated for that token.

The routing probabilities are computed using a linear projection followed by a `softmax` operation.

The `Router` takes parameters like the number of selected experts `(num_selected_experts)` and sharding configurations for data and model parallelism.

## :page_with_curl: MHAOutput

This is a named tuple that represents the output of the `multi-head attention` operation.

It contains two fields: `embeddings` (the output embeddings after the attention computation) and `memory` (the updated key-value memory for the next time step).

## :page_with_curl: DecoderOutput
This is a named tuple that represents the output of a single Transformer decoder layer.

It contains two fields: `embeddings` (the output embeddings after the self-attention and feed-forward operations) and `memory` (the updated key-value memory for the next layer).

## :page_with_curl: TransformerOutput
This is a named tuple that represents the output of the entire Transformer stack.

It contains two fields: `embeddings` (the final output embeddings after all Transformer layers) and `memory` (the final key-value memory after processing the input sequence).

## :page_with_curl: LanguageModelOutput
This is a named tuple that represents the output of the language model, which is built on top of the Transformer architecture.

It contains two fields: `logits` (the output logits representing the predicted token probabilities) and `model_state` (the final key-value memory from the Transformer, which can be used for further autoregressive decoding or generation).

## :page_with_curl: TransformerConfig 
The `TransformerConfig` class is a data class (using the @dataclass decorator) that holds the configuration parameters for the Transformer model. let's dive into it:

- ## Attributes:

- `emb_size`: The size of the embedding vectors.

- `key_size`: The size of the attention keys and values.

- `num_q_heads`: The number of attention heads for the query.

- `num_kv_heads`: The number of attention heads for the keys and values.

- `num_layers`: The number of Transformer layers.

- `vocab_size`: The size of the vocabulary (default: 128 * 1024).

- `widening_factor`: A factor used to determine the size of the feed-forward network (FFN) in each Transformer layer.

- `attn_output_multiplier`: A multiplier for the attention output.

- `name`: An optional name for the Transformer model.

- `num_experts`: The number of experts in the Mixture of Experts (MoE) architecture.

- `capacity_factor`: A factor used to compute the number of experts in the MoE architecture.

- `num_selected_experts`: The number of experts to be selected for each input token in the MoE architecture.

- `init_scale`: The initialization scale for the model parameters.

- `shard_activations`: A boolean flag indicating whether to shard activations across devices.
data_axis: The axis or axes along which to shard data.

- `model_axis`: The axis or axes along which to shard the model parameters.

- ## Methods:
`__post_init__()`

 This method is called after the data class is initialized. It ensures that the `data_axis` and `model_axis` attributes are tuples, converting them from lists if necessary.

`partition_rules()`: 

This method returns the partition rules `(TRANSFORMER_PARTITION_RULES)` for sharding the Transformer model parameters and activations across devices.

`make(mesh=None) -> "Transformer"`:

 This method creates and returns a Transformer instance based on the configuration parameters. It takes an optional mesh argument, which is used for specifying the mesh (device layout) for distributed training and inference.

`get_memory_sharding()`:

 This method returns a `Memory` instance that specifies the sharding configuration for the key-value memory used in the multi-head attention mechanism. The sharding is defined based on the `data_axis` and `model_axis` attributes.

The `TransformerConfig` class serves as a centralized place to store and manage the configuration parameters for the `Transformer` model. It provides methods to create a `Transformer` instance with the specified configuration and to define the sharding rules for distributed training and inference.

The `make` method is particularly important, as it creates the `Transformer` instance with the specified configuration parameters. This method is typically called from the `LanguageModelConfig` class to create the `Transformer` model as part of the language model initialization process.

The `get_memory_sharding` method is used to define the sharding configuration for the key-value `memory` used in the multi-head attention mechanism. This `memory` sharding configuration is necessary for efficient memory management and parallelization across multiple devices.

Overall, the `TransformerConfig` class role is the configuring and creating the `Transformer` model, as well as defining the sharding strategies for efficient distributed training and inference.

## :page_with_curl: Linear

The `Linear` class is a subclass of `haiku.Linear` and is used to apply a linear transformation to the input. It inherits the basic functionality from `haiku.Linear` but extends it with additional features like `sharding` and `quantization` support. Here's a detailed explanation of the class:

- ### `Constructor (__init__):`
- `output_size (int)`: The size of the output dimension.

`with_bias (bool, default=True)`: Whether to include a bias term in the linear transformation.

`sharding (Optional[P], default=None)`: The sharding specification (PartitionSpec) for the weight parameter. This is used for distributed training and inference.

`mesh (Any, default=None)`: The mesh (device layout) for distributed training and inference.

`name (Optional[str], default=None)`: The name of the layer.

`shard_axis (int, default=0)`: The axis along which to shard the weights.

- ### __call__ method:
This method computes the linear transformation of the input tensor.
It first checks if the input is a scalar, raising a ValueError if it is.

It retrieves the input size `(input_size)` and the output size `(output_size)` from the class attributes.

The weight parameter `(w)` is retrieved using `hk.get_parameter` with the shape `[input_size, output_size]`.

 It is initialized with zeros using `hk.initializers.Constant(0)`.

If the weight parameter has an attribute scales (which is the case for quantized weights), it performs the following steps:

1- Reshapes the input tensor to a 2D tensor `([batch_size * sequence_length, input_size])`.

2- Defines a mul function that multiplies the weight tensor `(w.weight)` with the scaling factors `(w.scales)` using `shard_map`. This function is parallelized across the specified mesh and sharding.

3- Applies the mul function to compute the scaled weight tensor `(w)`.

4- Computes the linear transformation by performing a matrix multiplication between the input tensor and the weight tensor `(w)`.

5- If `with_bias` is True, it retrieves the bias parameter `(b)` using `hk.get_parameter` with the shape `[output_size]`.It is initialized with zeros using `hk.initializers.Constant(0).`

If a `bias` is used, it broadcasts the bias tensor to the same shape as the output tensor and adds it to the output tensor.

Finally, it returns the output tensor.

The `Linear` class is designed to work with both regular and quantized weights. 

When working with quantized weights, it applies the appropriate scaling factors to the weights during the linear transformation. This is done using the `shard_map` function, which allows for efficient parallelization of the scaling operation across multiple devices.

The sharding functionality is controlled by the `sharding` and `mesh` parameters. The sharding parameter specifies how the weight tensor should be partitioned across devices, while the mesh parameter defines the device layout for distributed training and inference.

Overall, the Linear class provides a flexible and efficient way to apply linear transformations in the context of large-scale neural networks, with support for quantization and distributed training/inference through sharding.

## :page_with_curl: RMSNorm 

The RMSNorm class is a custom implementation of the root mean square layer normalization technique in Haiku. Layer normalization is a technique used to stabilize the training of neural networks by normalizing the inputs across the features (instead of across the examples as in batch normalization). The RMSNorm class is designed to be used within the transformer architecture.

The __init__ method initializes the RMSNorm layer with the following parameters:

axis: The axis or axes along which to normalize the input tensor.
eps: A small constant value added to the denominator to avoid division by zero.
name: An optional name for the layer.
create_scale: A boolean indicating whether to create a learnable scale parameter.
sharding: An optional PartitionSpec for sharding the weights of the layer.

The __call__ method performs the actual layer normalization operation:

It converts the input tensor to float32 precision for numerical stability.
If create_scale is True, it creates a learnable scale parameter and applies the specified sharding to it.
It computes the mean squared value of the input tensor along the specified axis.
It normalizes the input tensor by dividing it by the root mean square value (with a small epsilon for numerical stability).
It scales the normalized tensor by the learnable scale parameter (or 1.0 if create_scale is False).
It converts the output back to the original data type of the input tensor.
The RMSNorm layer is commonly used in transformer architectures to stabilize the training process and improve convergence.

## :page_with_curl: MultiHeadAttention 

The `multi-head attention mechanism` is a key component of the Transformer architecture, which powers many state-of-the-art natural language processing models.

The main idea behind `multi-head attention` is to allow the model to jointly attend to information from different representation subspaces at different positions. This is achieved by having multiple independent **attention heads** that each produce their own representation of the input sequence.

Specifically, in the multi-head attention layer:

The input sequence is transformed into queries (Q), keys (K), and values (V) for each attention head.

For each head, an attention score is computed between each query and all keys. This indicates how much focus should be given to each value.

The values are weighted by the attention scores and summed to produce the output for that head.

The outputs across all heads are concatenated and linearly transformed to produce the final output.

This multi-headed approach allows the model to attend to different positions in the input sequence through different representational spaces (the heads). It has been shown to be more effective than single-head attention for capturing long-range dependencies.

The multi-head attention mechanism along with other components like positional encodings and feed-forward layers, allows the Transformer to model sequential data very effectively without the limitations of sequential models like RNNs.

The `MultiHeadAttention` class is responsible for implementing the multi-head attention mechanism, which is a fundamental component of the transformer architecture. It allows the model to attend to different representations of the input sequence simultaneously, enabling it to capture long-range dependencies more effectively.

Here's a breakdown of the `__init__` method and its parameters:

`num_q_heads`: The number of query heads to use in the multi-head attention mechanism.

`num_kv_heads`: The number of key/value heads to use in the multi-head attention mechanism.

`key_size`: The dimensionality of the key/query vectors.

`with_bias`: A boolean indicating whether to use bias terms in the linear projections.

`value_size`: The dimensionality of the value vectors (defaults to key_size).

`model_size`: The dimensionality of the output embeddings (defaults to key_size * num_q_heads).

`attn_output_multiplier`: A scalar multiplier for the attention logits, which can help with numerical stability.

`data_axis`: The axis or tuple of axes for data parallelism (used for sharding).

`model_axis`: The axis or tuple of axes for model parallelism (used for sharding).

`name`: An optional name for the layer.

Now, let's examine the `__call__` method, which performs the actual multi-head attention computation:

- The method first checks if the input shapes and masks are consistent and valid.
It then projects the query, key, and value tensors into their respective head representations using the `_linear_projection` helper function.

- If rotary embeddings are used, it applies them to the key and query tensors using the `RotaryEmbedding` class.

- If key-value memory is provided (for autoregressive models), it updates the key and value tensors with the cached memory.

- It computes the attention logits by taking the dot product of the query and key tensors.

- It applies a causal mask (for autoregressive models) and any additional mask provided.

- It computes the attention weights by applying softmax to the attention logits.

- It computes the attention output by taking the weighted sum of the value vectors.

- It applies a final linear projection to the attention output to obtain the output embeddings.

- It returns the output embeddings and the updated key-value memory (if applicable).

- The _linear_projection helper function is used to apply a linear projection to the input tensor, splitting it into multiple heads:

## :page_with_curl: RotaryEmbedding 

The `RotaryEmbedding` class implements the rotary embeddings technique, as described in the paper [Efficient Transformer](https://arxiv.org/abs/2104.09864).

Rotary embeddings are a way of encoding positional information into the input embeddings, which can improve the model's ability to capture positional relationships and long-range dependencies.

The `__init__` method initializes the parameters:

- `dim`: Dimensionality of the input embeddings.

- `name`: An optional name for the layer.
- `base_exponent`: Base exponent used to compute the embeddings.
- The `__call__` method applies the rotary embeddings to the input tensor:

    - It computes the per-dimension frequencies based on the base_exponent.
    - It computes the per-element phase based on the sequence position and the frequencies.
    - It applies the rotary embeddings by multiplying the input tensor with the cosine and sine of the phase.

Rotary embeddings can be particularly useful in transformer models that operate on long sequences, as they provide a more effective way of encoding positional information compared to traditional positional embeddings.

## :page_with_curl: MHABlock 
The `MHABlock` class encapsulates the `multi-head attention` block within a transformer layer. It consists of a single multi-head attention sublayer.

The `__init__` method initializes the parameters:

- `num_q_heads`: Number of query heads.

- `num_kv_heads`: Number of key/value heads.
- `key_size`: Dimensionality of the key/query vectors.
- `attn_output_multiplier`: Multiplier for the attention logits (used for numerical stability).
- `mesh`: A mesh or pmap object for distributed training.
- `data_axis`: Axis or tuple of axes for data parallelism.
- `model_axis`: Axis or tuple of axes for model parallelism.
- The __call__ method applies the multi-head attention block:

    - It computes the multi-head attention output using the MultiHeadAttention class.
    - It adds a residual connection from the input to the attention output.
    - It returns the updated embeddings and the attention memory (for autoregressive models).
    - The `MHABlock` is a key component of the transformer layer, responsible for capturing long-range dependencies and providing contextualized representations of the input sequence.

## :page_with_curl: DenseBlock 

The `DenseBlock` class implements the feed-forward network (FFN) sublayer within a transformer layer. It consists of two linear projections with a non-linearity `(typically GeLU or ReLU)` in between.

The `__init__` method initializes the parameters:

- `num_q_heads`: Number of query heads (used for sharding purposes).
- `num_kv_heads`: Number of key/value heads (used for sharding purposes).
- `key_size`: Dimensionality of the key/query vectors (used for sharding purposes).
- `widening_factor`: Multiplicative factor used to determine the dimensionality of the FFN hidden layer.

- `sharding_constraint`: Whether to apply sharding constraints.
- `mesh`: A mesh or pmap object for distributed training.
- The `__call__` method applies the feed-forward network:

    - It applies a linear projection to the input tensor, followed by a GeLU activation.
    - It applies another linear projection to the activated tensor.
    - It adds a residual connection from the input to the output of the second linear projection.
    - The DenseBlock is responsible for introducing non-linearity into the transformer layer, enabling it to model more complex relationships within the input sequence.

## :page_with_curl: DecoderLayer 
The `DecoderLayer` class represents a single layer within the transformer decoder stack. It combines the `multi-head attention` block `(MHABlock)` and the feed-forward network `(DenseBlock)` into a single module.

The `__init__` method initializes the parameters:

- `num_q_heads`: Number of query heads.

- `num_kv_heads`: Number of key/value heads.
- `key_size`: Dimensionality of the key/query vectors.
- `num_layers`: Total number of layers in the transformer stack.
- `num_experts`: Number of experts in the mixture-of-experts (MoE) layer (if applicable).
- `layer_index`: Index of the current layer (used for MoE).
- `num_selected_experts`: Number of experts to select in the MoE layer.
- `widening_factor`: Multiplicative factor used to determine the dimensionality of the FFN hidden layer.

- `name`: An optional name for the layer.
- `data_axis`: Axis or tuple of axes for data parallelism.
- `model_axis`: Axis or tuple of axes for model parallelism.
- `shard_activations`: Whether to shard the activations.
- `attn_output_multiplier`: Multiplier for the attention logits (used for numerical stability).
- `mesh`: A mesh or pmap object for distributed training.
- The `__call__` method applies the decoder layer:

    - It applies layer normalization to the input embeddings.

    - It computes the multi-head attention output using the MHABlock.
    - It applies layer normalization to the attention output, adds a residual connection, and updates the embeddings.
    - It applies the feed-forward network (DenseBlock) to the updated embeddings, with an optional mixture-of-experts (MoE) layer.
    - It applies layer normalization to the FFN output, adds a residual connection, and updates the embeddings.
    - It returns the updated embeddings and the attention memory (for autoregressive models).
    - The DecoderLayer is the core building block of the transformer decoder, responsible for encoding the input sequence and producing contextualized representations that can be used for various downstream tasks, such as language modeling or machine translation.
     
## :page_with_curl: Transformer

let's dive into the Transformer class, which represents the core of the transformer architecture.
The Transformer class is defined as follows:

```python
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
```
let's look at eacg parameters:

- `num_q_heads`: The number of query heads in the multi-head attention mechanism.

- `num_kv_heads`: The number of key/value heads in the multi-head attention mechanism.
- `key_size`: The dimensionality of the key/query vectors.
- `widening_factor`: A multiplier used to determine the dimensionality of the feed-forward network (FFN) hidden layer.
- `init_scale`: The initialization scale for the parameters.
- `mesh`: A mesh or pmap object for distributed training.
- `attn_output_multiplier`: A scalar multiplier for the attention logits, which can help with numerical stability.
- `shard_activations`: A boolean indicating whether to shard the activations across devices.
- `num_layers`: The number of transformer layers in the stack.
- `num_experts`: The number of experts in the mixture-of-experts (MoE) layer (if applicable).
- `num_selected_experts`: The number of experts to select in the MoE layer.
- `name`: An optional name for the module.
- `data_axis`: The axis or tuple of axes for data parallelism (used for sharding).
- `model_axis`: The axis or tuple of axes for model parallelism (used for sharding).

The `Transformer` class has two main methods:

1. `init_memory`:

```python

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

```
This method initializes the key-value memory for the transformer stack. The memory is used to cache the attention keys and values for autoregressive models, such as language models. It creates a list of KVMemory objects, one for each layer in the stack, with the specified batch size, sequence length, and data type.

2. `__call__`:

```python
def __call__(
    self,
    embeddings: jax.Array,  # [B, T, D]
    mask: jax.Array,  # [B, T]
    memory: Optional[Memory],
) -> TransformerOutput:
    """Transforms input embedding sequences to output embedding sequences."""
    ...
```

This method represents the forward pass of the transformer stack. It takes the following inputs:

- `embeddings`: The input embeddings of shape `[batch_size, sequence_length, model_size]`.

- `mask`: A boolean mask of shape `[batch_size, sequence_length]`, indicating which positions in the sequence are valid (True) or padded (False).

- `memory`: An optional `Memory` object containing the key-value memory for autoregressive models.
The __call__ method performs the following steps:

- It computes the causal mask for autoregressive sequence modeling.

- It iterates over the transformer layers, applying the DecoderLayer to the input embeddings and mask.

Each `DecoderLayer` consists of a multi-head attention block (MHABlock) and a feed-forward network (DenseBlock), with optional mixture-of-experts (MoE) support.

The output embeddings from each layer are passed as input to the next layer, along with the updated key-value memory.

The final output embeddings and the updated memory are returned as a `TransformerOutput` named tuple.

The `Transformer` class encapsulates the core functionality of the transformer architecture, enabling it to effectively model long-range dependencies and capture complex relationships within sequential data. It combines the multi-head attention mechanism with feed-forward networks and optional mixture-of-experts layers, allowing the model to learn rich representations of the input sequence.

## :page_with_curl: LanguageModel 

LanguageModel represents an autoregressive transformer-based language model.

The LanguageModel is defined as follows:

```python
@dataclass
class LanguageModel(hk.Module):
    """An autoregressive transformer-based language model."""

    model: "Transformer"
    config: LanguageModelConfig
    fprop_dtype: Any = jnp.bfloat16
    name: Optional[str] = None
    mesh: Any = None
```
these are each parameters:

- `model`: An instance of the Transformer class, which represents the core transformer stack.

- `config`: An instance of LanguageModelConfig, which contains various configuration parameters for the language model.

- `fprop_dtype`: The data type to use for the forward pass computations (default: jnp.bfloat16).
- `name`: An optional name for the module.
- `mesh`: A mesh or pmap object for distributed training.

#
The `LanguageModel` class has two main methods:

This method represents the forward pass of the language model. It takes the following inputs:

1. `__call__`:

`tokens`: The input token sequences of shape [batch_size, sequence_length].

`memory`: An optional Memory object containing the key-value memory for autoregressive modeling.

`batch`: A dictionary containing additional batch data (not used in this implementation).

`last_hid_only`: A boolean indicating whether to return only the last hidden state or the full 
sequence of logits.

`length`: An optional array of sequence lengths for each example in the batch.

The __call__ method performs the following steps:

- It computes the input mask based on the token values and the padding token ID.
- It embeds the input tokens using the InOutEmbed module, which applies positional embeddings and input/output embeddings.
- It passes the input embeddings, mask, and memory to the Transformer model to obtain the output embeddings and updated memory.
- It applies layer normalization to the output embeddings.
- If last_hid_only is True, it returns only the last hidden state for each sequence.
Otherwise, it decodes the output embeddings using the InOutEmbed module (with tied weights) to obtain the logits for each sequence position.

- The logits and updated memory are returned as a LanguageModelOutput named tuple.

2. `init_memory` and `prefill_memory`:

```python
def init_memory(self, batch_size: int, seq_len: int, dtype=jnp.bfloat16):
    return self.model.init_memory(batch_size=batch_size, sequence_len=seq_len, dtype=dtype)

def prefill_memory(self, prompts, memory):
    # Pad to the left and right align?
    # Basically assume prompt is already padded
    model_output = self(prompts, memory=memory)
    return model_output.logits, model_output.model_state
```
The `init_memory` method initializes the key-value memory for the transformer stack, given the batch size and sequence length.

The `prefill_memory` method is used to prefill the key-value memory with prompts. It passes the prompts through the language model, updating the memory, and returns the logits and updated memory state.

The `LanguageModel` class is designed to handle autoregressive sequence modeling tasks, such as language modeling or text generation. It leverages the transformer architecture to effectively capture long-range dependencies and produce contextualized representations of the input sequences. The model's output logits can be used for tasks like predicting the next token in a sequence or generating new text based on a given prompt.

## :page_with_curl: InOutEmbed

`InOutEmbed` class defines a custom module called `InOutEmbed` that inherits from `hk.Embed` in the Haiku library, which is a library for building neural networks in JAX. The `InOutEmbed` module is designed for embedding tokens in a low-dimensional space, which is a common technique used in natural language processing (NLP) tasks.

Let's look at each method:

`__init__`:

It takes in optional arguments `vocab_size` (the size of the vocabulary), `embed_dim` (the dimension of the embedding space), `sharding` (a PartitionSpec object for sharding the embedding matrix across multiple devices), and `name` (the name of the module).

It calls the `__init__` method of the parent class `hk.Embed` with `vocab_size`, `embed_dim`, and `name`.

then it stores the sharding argument as an instance attribute.

`embeddings property`:

- It defines a trainable parameter called "embeddings" using hk.get_parameter.
- The shape of the parameter is [vocab_size, embed_dim], and it is initialized with constant values of 0.

- If sharding is provided, it applies the sharding constraint to the embedding matrix using the with_sharding_constraint function.

- The embedding matrix is returned.

`decode method`:

- It takes in an input tensor inputs of shape [batch_size, vocab_size], which represents one-hot encoded token indices.

- It computes the embedding vectors for the input tokens by performing a matrix multiplication between inputs and the transpose of the embedding matrix (self.embeddings.T).

- The output tensor has shape [batch_size, embed_dim], where each row represents the embedding vector for the corresponding input token.

this module is used to convert discrete token indices (e.g., words or subword units) into continuous vector representations `(embeddings)`.

 These embeddings can then be fed into subsequent layers of a neural network for processing.
