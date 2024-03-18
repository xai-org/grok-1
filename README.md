# Grok-1

This repository contains JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place `ckpt-0` directory in `checkpoint`.
Then, run

```shell
pip install -r requirements.txt
python run.py
```

to test the code.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Downloading the weights

You can download the weights using a torrent client and this magnet link:
```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.

# Table of contents
This repository contains Python code for grok-1. Below is a breakdown of the main components and features provided by this codebase:

## File: `model.py`
i will try to breakdown each classes in detail

## QuantizedWeight8bit
The QuantizedWeight8bit class is a data structure that represents quantized weights in a neural network. Quantization is a technique used to reduce the precision of weight values from the typical 32-bit floating-point representation to a lower-precision format, such as 8-bit integers, to save memory and improve computational efficiency, especially on hardware accelerators like GPUs or TPUs.

The QuantizedWeight8bit class has two main attributes:

**weight** : This is a NumPy array that holds the quantized weight values, represented as 8-bit integers.

**scales** : This is a NumPy array that holds the scaling factors associated with each quantized weight.

During the model initialization or loading phase, the original 32-bit floating-point weights are quantized to 8-bit integers and packed into QuantizedWeight8bit instances.
When performing computations in the neural network, such as linear transformations or convolutions, the quantized weights are used instead of the original 32-bit weights. This is done by applying the scaling factors stored in the scales attribute to recover approximate values of the original weights.
After the computation, the results are typically de-quantized (converted back to higher precision) for further processing or output.
By using QuantizedWeight8bit, the model can achieve significant memory savings and potentially faster computations, especially on hardware accelerators optimized for low-precision arithmetic. However, there is a trade-off between the level of quantization and the model's accuracy, as quantization introduces approximation errors. Careful calibration and quantization-aware training techniques are often employed to minimize the accuracy loss due to quantization.

## TrainingState 
The TrainingState class is a simple data structure defined as a NamedTuple in Python. It is used to hold the parameters (weights) of a neural network model during the training process. In this specific code, the TrainingState only contains one field:

```python
TrainingState(NamedTuple):
    """Container for the training state."""

    params: hk.Params
```
Here, params is an instance of hk.Params, which is a data structure provided by the Haiku library (a JAX-based neural network library) to represent the parameters (weights) of a neural network model.

The NamedTuple is a lightweight data structure provides a way to define immutable tuples with named fields. It is similar to a class, but it is more lightweight and efficient, making it suitable for storing and passing around data structures that don't require additional methods or behavior.

the TrainingState serves as a lightweight container to hold and manage the model parameters during the training process, allowing for efficient manipulation and updating of the model's weights.

**ffn_size**: This function computes the size (number of units) for the feed-forward network (FFN) layer in the transformer architecture. The FFN size is typically larger than the embedding size to increase the model's expressive power. The function takes two arguments:

**emb_size**: The size of the input embeddings.

**widening_factor**: A multiplier used to determine the FFN size relative to the embedding size. 

The function first calculates the `FFN size` as `int(widening_factor * emb_size) * 2 // 3`. The * 2 // 3 part is a heuristic to reduce the `FFN size` slightly while maintaining good performance. Then, it ensures that the `FFN size` is a multiple of 8 by adding the smallest positive number needed to make it divisible by 8. This is likely done for efficient computations on certain hardware architectures.

**apply_rules**: This function returns a closure (inner function) that applies a set of rules to reshape (reshard) the parameters of a neural network model based on regular expression patterns. The rules are provided as a list of tuples, where each tuple contains a list of regular expression patterns and a corresponding reshape specification (PartitionSpec). 

The inner function _apply_rules takes a path (list of keys) and a value (tensor) as input. It flattens the path and checks if any of the provided rules (regular expressions) match the flattened path. If a match is found, the corresponding PartitionSpec is applied to the tensor, effectively reshaping it according to the specified partitioning scheme. This function is likely used to optimize the model for distributed training across multiple devices or accelerators.

**cast_bfloat16**:  This is a simple utility function that casts the input tensor (x) to the bfloat16 data type if the tensor's data type is a floating-point type. The bfloat16 data type is a truncated 16-bit floating-point format that can provide higher computational performance on certain hardware architectures, such as Google's TPUs, while maintaining reasonable precision. If the input tensor is not a floating-point type, the function returns the tensor unchanged.


**with_sharding_constraint**: This function applies a sharding constraint to the input tensor (x). Sharding is a technique used in distributed training to split the model parameters and computations across multiple devices or accelerators. The sharding constraint specifies how the tensor should be partitioned or reshaped for efficient parallel computations. The function takes two arguments:
x: The input tensor to be reshaped.
constraint: A PartitionSpec object that specifies the desired reshaping or partitioning scheme.

If the current environment does not support distributed training (i.e., jax.experimental.maps.thread_resources.env.physical_mesh.empty is True), the function returns the input tensor unchanged. Otherwise, it applies the specified sharding constraint to the tensor using the pjit_sharding_constraint function from JAX.


**match**: This is a helper function used by apply_rules. It takes two arguments:

`qs`: A tuple of regular expression patterns.

`ks`: A tuple of strings representing the flattened path.

The function compiles the regular expressions in qs and checks if any window (consecutive sublist) of strings in ks matches all the compiled regular expressions simultaneously. If a match is found, it returns True; otherwise, it returns False. This function is likely used by apply_rules to determine if a specific set of rules (regular expressions) should be applied to a given path in the neural network model.

## TRANSFORMER_PARTITION_RULES
`TRANSFORMER_PARTITION_RULES` is a list of tuples that define the partitioning rules for the parameters of a transformer model. These rules are used by the `apply_rules` function to reshape (reshard) the model parameters for efficient distributed training across multiple devices or accelerators.

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

### LM_PARTITION_RULES

`LM_PARTITION_RULES` is a list of tuples that define the partitioning rules for the parameters of the language model component in the codebase. These rules are used to specify how the parameters of the language model should be partitioned (reshaped) across multiple devices or accelerators for efficient distributed training.

The `LM_PARTITION_RULES` list contains the following rules:

- #### `(("language_model", "positional_embeddings"), P(None, ("data", "model")))`: 

    This rule matches the positional embeddings tensor in the language model module. The PartitionSpec `P(None, ("data", "model"))` specifies that this tensor should be partitioned along the "data" and "model" dimensions, but not partitioned along the leading dimension (represented by None). This means that the positional embeddings will be split across multiple devices along the "data" and "model" dimensions, but replicated along the leading dimension (e.g., batch dimension).

- #### `(("language_model", "in_out_embed", "embeddings"), P(None, ("data", "model")))`: 
    This rule matches the embeddings tensor of the InOutEmbed module (used for input and output embeddings) in the language model. Similar to the previous rule, it specifies that this tensor should be partitioned along the "data" and "model" dimensions, while being replicated along the leading dimension.

- #### `(("language_model", "rms_norm"), P(None))`: 
    This rule matches the parameters of the RMSNorm layer in the language model. The PartitionSpec P(None) indicates that these parameters should not be partitioned at all and should be replicated across all devices.

By applying these partitioning rules, the language model's parameters are reshaped and distributed across multiple devices in a way that aims to balance the computational load and memory usage. The input and output embeddings, which are typically large tensors, are partitioned along the "data" and "model" dimensions to distribute their storage and computations. At the same time, smaller tensors like the normalization layer parameters are replicated across all devices to minimize communication overhead.

### KVMemory

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

## Router 
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

## MoELayer 

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



