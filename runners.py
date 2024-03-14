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


import bisect
import functools
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.experimental.pjit as pjit
import jax.numpy as jnp
import numpy as np
import sentencepiece
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

import checkpoint as xai_checkpoint
from model import (
    LanguageModelConfig,
    LanguageModelOutput,
    TrainingState,
    apply_rules,
    Memory,
    KVMemory,
)

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

TOP_K = 8


class SampleSettings(NamedTuple):
    temperature: ArrayLike
    nucleus_p: ArrayLike
    mask: ArrayLike
    # Whether a given batch element is actively used. [B]
    active: ArrayLike


class SampleOutput(NamedTuple):
    token_id: ArrayLike
    prob: ArrayLike
    top_k_token_ids: ArrayLike
    top_k_probs: ArrayLike


def insert_slice(memory: Memory, slice, length, i):
    slice = Memory(
        layers=[
            KVMemory(layer.k, layer.v, step=jnp.array([length]))
            for layer in slice.layers
        ],
    )

    return jax.tree_map(lambda m, u: jax.lax.dynamic_update_index_in_dim(m, u[0], i, axis=0),
                        memory, slice)


def pad_to_size(x, size):
    if x.shape[0] > size:
        # Left truncate if the context is too long.
        x = x[-size:]
    return np.pad(x, [0, size - x.shape[0]], mode="constant", constant_values=0)


def top_p_filter(logits: jax.Array, top_p: jax.Array) -> jax.Array:
    """Performs nucleus filtering on logits."""
    assert logits.ndim == top_p.ndim, f"Expected {logits.ndim} equal {top_p.ndim}"
    sorted_logits = jax.lax.sort(logits, is_stable=False)
    sorted_probs = jax.nn.softmax(sorted_logits)
    threshold_idx = jnp.argmax(jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
    threshold_largest_logits = jnp.take_along_axis(
        sorted_logits, threshold_idx[..., jnp.newaxis], axis=-1
    )
    assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
    mask = logits >= threshold_largest_logits
    # Set unused logits to -inf.
    logits = jnp.where(mask, logits, -1e10)
    return logits


def sample_token(
    rngs: jax.random.PRNGKey,
    lm_outputs: LanguageModelOutput,
    settings: SampleSettings,
) -> SampleOutput:
    # Expand the settings shape to match the logit shape.
    settings = SampleSettings(
        temperature=jnp.expand_dims(settings.temperature, (1, 2)),  # Input [B], output [B, 1, 1].
        nucleus_p=jnp.expand_dims(settings.nucleus_p, (1, 2)),  # Input [B], output [B, 1, 1].
        mask=jnp.expand_dims(settings.mask, 1),  # Input [B, V], output [B, 1, V].
        active=settings.active,  # [B].
    )
    logits = lm_outputs.logits / settings.temperature.astype(lm_outputs.logits.dtype)
    # Mask out all disallowed tokens by assigning them a near-zero probability.
    logits = jnp.where(settings.mask, logits, -1e10)
    # Mask out all tokens that don't fall into the p-th percentile.
    logits = top_p_filter(logits, settings.nucleus_p.astype(logits.dtype))

    new_token = jax.vmap(jax.random.categorical)(rngs, logits)

    probabilities = jax.nn.softmax(logits)
    token_prob = jnp.take_along_axis(probabilities, jnp.expand_dims(new_token, 1), axis=2)
    token_prob = jnp.squeeze(token_prob, 1)

    # Gather the top-k tokens and probabilities.
    top_k_probs, top_k_token_ids = jax.lax.top_k(probabilities, TOP_K)
    top_k_probs = jnp.squeeze(top_k_probs, 1)
    top_k_token_ids = jnp.squeeze(top_k_token_ids, 1)
    return SampleOutput(
        new_token,
        token_prob,
        top_k_token_ids,
        top_k_probs,
    )


@dataclass
class ModelRunner:
    model: LanguageModelConfig

    bs_per_device: float = 2.0

    load_rename_rules: Optional[list[tuple[str, str]]] = None
    load_exclude_rules: Optional[list[str]] = None

    rng_seed: int = 42  # Initial rng seed.
    transform_forward: bool = False

    checkpoint_path: str = ""

    def make_forward_fn(self, mesh: Any):
        def forward(tokens):
            out = self.model.make(mesh=mesh)(tokens)
            return out, None

        if self.transform_forward:
            forward = hk.transform(forward)
        return forward

    def initialize(
        self,
        init_data,
        local_mesh_config: tuple[int, int],
        between_hosts_config: tuple[int, int],
    ):
        num_replicas = math.prod(between_hosts_config)
        self.model.initialize()
        self.model.fprop_dtype = jnp.bfloat16
        num_local_gpus = len(jax.local_devices())

        # Calculate the global batch size from the local batch size.
        self.batch_size = int(self.bs_per_device * num_local_gpus * num_replicas)

        # Calculate the batch size per host from the global batch size.
        self.local_batch_size = self.batch_size // jax.process_count()

        self.local_mesh_config = local_mesh_config
        self.between_hosts_config = between_hosts_config
        rank_logger.info(
            f"Initializing mesh for {self.local_mesh_config=} {self.between_hosts_config=}..."
        )
        self.mesh = make_mesh(self.local_mesh_config, self.between_hosts_config)
        self.forward = self.make_forward_fn(mesh=self.mesh)
        self.logits_fn = hk.transform(lambda tokens: self.forward(tokens)[0])

        self.eval_forward = self.make_forward_fn(mesh=self.mesh)
        self.logits_eval_fn = hk.transform(lambda tokens: self.eval_forward(tokens)[0])

        if self.transform_forward:
            self.state_sharding = self.get_state_sharding(init_data)
            rank_logger.info(f"State sharding type: {type(self.state_sharding)}")
            self.init_fn = pjit.pjit(self.init, out_shardings=self.state_sharding)

    def init(self, rng: jax.Array, data) -> TrainingState:
        assert self.transform_forward
        rng, init_rng = jax.random.split(rng)
        params = self.forward.init(init_rng, data["inputs"])
        return TrainingState(params=params)

    def get_state_sharding(self, init_data):
        assert self.transform_forward
        rng = jax.random.PRNGKey(self.rng_seed)
        rank_logger.info(f"partition rules: {self.model.partition_rules}")

        with self.mesh:
            shapes = jax.eval_shape(self.init, rng, init_data)
            sharding = jax.tree_util.tree_map_with_path(
                apply_rules(self.model.partition_rules()),
                shapes,
            )
        return sharding

    def load_or_init(
        self,
        init_data: Any,
        from_checkpoint: bool = True,
        init_fn: Optional[Callable] = None,
    ):
        rng = jax.random.PRNGKey(self.rng_seed)

        if not self.checkpoint_path or not from_checkpoint:
            rank_logger.info("Initializing model...")
            with self.mesh:
                if init_fn is not None:
                    state = init_fn(rng, init_data)
                else:
                    assert self.transform_forward
                    state = self.init_fn(rng, init_data)
            rank_logger.info("Model state is newly initialized.")
        else:
            with self.mesh:
                if init_fn:
                    state_shapes = jax.eval_shape(init_fn, rng, init_data)
                else:
                    assert self.transform_forward
                    state_shapes = jax.eval_shape(self.init_fn, rng, init_data)
            init_state = None

            state = xai_checkpoint.restore(
                checkpoint_path=self.checkpoint_path,
                state_shapes=state_shapes,
                mesh=self.mesh,
                between_hosts_config=self.between_hosts_config,
                state_sharding=self.state_sharding,
                init_state=init_state,
                params_only=True,
            )

            del init_state
        return state


@dataclass
class Request:
    prompt: str
    temperature: float
    nucleus_p: float
    rng_seed: int
    max_len: int


@dataclass
class InferenceRunner:
    name: str
    runner: Any
    load: str
    tokenizer_path: str = "/tmp/xai_data/tokenizer.model"
    local_mesh_config: Tuple[int, int] = (1, 1)
    between_hosts_config: Tuple[int, int] = (1, 1)
    pad_sizes: tuple[int] = (1024,)

    def get_pad_bucket(self, size):
        i = bisect.bisect_left(self.pad_sizes, size)
        return self.pad_sizes[min(i, len(self.pad_sizes) - 1)]

    def initialize(self):
        runner = self.runner
        self.runner.transform_forward = True
        dummy_data = dict(
            inputs=np.zeros((1, 256), dtype=np.int32),
            targets=np.zeros((1, 256), dtype=np.int32),
        )
        runner.initialize(
            dummy_data,
            local_mesh_config=self.local_mesh_config,
            between_hosts_config=self.between_hosts_config,
        )

        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizer_path)

        max_len = runner.model.sequence_len

        self.vocab_size = self.runner.model.vocab_size

        params = runner.load_or_init(dummy_data)
        self.params = params

        def pad_to_max_len(x):
            if len(x.shape) > 1:
                pad_width = max_len - x.shape[1]
                return jnp.pad(x, [(0, 0), (0, pad_width), (0, 0), (0, 0)])
            else:
                return x

        @functools.lru_cache
        def lm():
            return runner.model.make(mesh=runner.mesh)

        def hk_forward(
            tokens,
            memory=None,
            length=None,
            active=None,
        ) -> LanguageModelOutput:
            if memory is not None:
                assert active is not None
                layers = []
                for l in memory.layers:
                    # Reset steps to 0 for inactive requests to avoid unnecessary computations.
                    step = jnp.where(active, l.step, jnp.zeros_like(l.step))
                    layers.append(l._replace(step=step))
                memory = memory._replace(layers=layers)
            return lm()(tokens, memory, length=length)

        def hk_sample_step(rngs, last_output: SampleOutput, memory, settings):
            rngs, rngs_ = jax.vmap(jax.random.split, out_axes=1)(rngs)
            lm_outputs = hk_forward(last_output.token_id, memory=memory, active=settings.active)
            sample_result = sample_token(rngs_, lm_outputs, settings)
            return rngs, sample_result, lm_outputs.model_state

        def hk_new_memory(batch_size, sequence_len):
            return lm().init_memory(batch_size, sequence_len)

        def hk_prefill_memory(
            rngs,
            memory,
            settings,
            last_output,
            prompt,
            length,
            rng_seed,
            new_settings,
            i,
        ):
            rng = jax.random.PRNGKey(seed=rng_seed)
            rng, rng_ = jax.random.split(rng)

            # Allocate new memory for this sample. The memory length is equal to the length of the
            # prompt.
            slice = hk_new_memory(1, prompt.shape[0])

            # Move the settings for this individual batch entry into the joint settings tensor.
            settings = jax.tree_map(
                lambda o, v: jax.lax.dynamic_update_index_in_dim(o, v, i, axis=0),
                settings,
                new_settings,
            )

            # Get the settings for the batch entry from the joint settings tensor.
            settings_slice = jax.tree_map(lambda t: jnp.expand_dims(t[i], axis=0), settings)

            # Process the first n-1 tokens of the prompt.
            lm_outputs = hk_forward(
                jnp.expand_dims(prompt, 0),
                memory=slice,
                length=jnp.expand_dims(length, 0),
                active=settings_slice.active,
            )

            # The forward pass doesn't correctly set the `step` counter inside the memory. Manually
            # override it so `hk_forward` uses the correct context length in the next call.
            slice = lm_outputs.model_state
            slice = slice._replace(
                layers=[l._replace(step=jnp.array([length])) for l in slice.layers]
            )

            # Sample the actual output token.
            rng_ = jnp.expand_dims(rng_, 0)
            new_output = sample_token(rng_, lm_outputs, settings_slice)

            # Update the KV cache/memory.
            slice = jax.tree_map(pad_to_max_len, slice)
            memory = insert_slice(memory, slice, length, i)

            rng = jnp.expand_dims(rng, 0)
            rngs = jax.lax.dynamic_update_index_in_dim(rngs, rng, i, axis=0)

            # Move the network outputs for this batch entry into the joint output tensor.
            last_output = jax.tree_util.tree_map(
                lambda last, new: jax.lax.dynamic_update_index_in_dim(last, new, i, axis=0),
                last_output,
                new_output,
            )
            return rngs, last_output, memory, settings

        sample_step_ = hk.without_apply_rng(hk.transform(hk_sample_step))
        prefill_memory_ = hk.without_apply_rng(hk.transform(hk_prefill_memory))
        new_memory_ = hk.without_apply_rng(hk.transform(hk_new_memory))
        forward_ = hk.without_apply_rng(hk.transform(hk_forward))

        rng = jax.random.PRNGKey(42)
        dummy_tokens = jnp.zeros((1, max_len), jnp.int32)

        with runner.mesh:
            shapes = jax.eval_shape(forward_.init, rng, dummy_tokens)

        self.params_sharding = jax.tree_util.tree_map_with_path(
            apply_rules(runner.model.partition_rules()),
            shapes,
        )

        ds = P("data")
        ms = runner.model.model.get_memory_sharding()
        self.sample_step = pjit.pjit(
            sample_step_.apply,
            in_shardings=(self.params_sharding, None, ds, ms, None),
            out_shardings=(None, ds, ms),
            donate_argnums=3,
        )
        self.prefill_memory = pjit.pjit(
            functools.partial(prefill_memory_.apply),
            in_shardings=(
                self.params_sharding,
                None,
                ms,
                None,
                ds,
                None,
                None,
                None,
                None,
                None,
            ),
            out_shardings=(None, ds, ms, None),
            donate_argnums=(2,),
        )
        self.new_memory = pjit.pjit(
            new_memory_.apply,
            static_argnums=(1, 2),
            out_shardings=ms,
        )

    def run(self):
        """Generator that accepts prompts."""
        runner = self.runner
        mesh = runner.mesh
        max_len = runner.model.sequence_len
        batch_size = runner.batch_size
        params = self.params
        rngs = jax.random.split(jax.random.PRNGKey(1), batch_size)
        with mesh:
            memory = self.new_memory(params, batch_size, max_len)
            settings = SampleSettings(
                temperature=np.zeros((batch_size,), dtype=np.float32),
                nucleus_p=np.zeros((batch_size,), dtype=np.float32),
                mask=np.ones((batch_size, self.vocab_size), dtype=np.int32),
                active=np.zeros((batch_size), dtype=np.int32),
            )
            last_output = SampleOutput(
                token_id=np.zeros((batch_size, 1), dtype=np.int32),
                prob=np.zeros((batch_size, 1), dtype=jnp.bfloat16),
                top_k_token_ids=np.zeros((batch_size, TOP_K), dtype=np.int32),
                top_k_probs=np.zeros((batch_size, TOP_K), dtype=jnp.bfloat16),
            )

            prompt = np.array([300, 400, 500, 600, 600, 700, 800])

            new_settings = SampleSettings(
                temperature=np.float32(1),
                nucleus_p=np.float32(1),
                mask=np.ones((self.vocab_size,), dtype=np.int32),
                active=np.zeros((), dtype=np.int32),
            )
            rng_seed = np.uint64(1)

            for size in self.pad_sizes:
                if size > runner.model.sequence_len:
                    break
                logger.info("Precompile {}".format(size))
                prompt_len = len(prompt)
                prompt = pad_to_size(prompt, size)
                rngs, last_output, memory, settings = self.prefill_memory(
                    params,
                    rngs,
                    memory,
                    settings,
                    last_output,
                    prompt,
                    prompt_len,
                    rng_seed,
                    new_settings,
                    0,
                )
        with runner.mesh:
            logger.info("Compiling...")
            rngs, last_output, memory = self.sample_step(
                params, rngs, last_output, memory, settings
            )
            logger.info("Done compiling.")

        all_tokens = []
        free_slots = list(range(batch_size))
        requests = [None] * batch_size
        first_output = [None] * batch_size
        jax.tree_map(lambda x: x.copy_to_host_async(), last_output)
        prev_token = last_output
        step = 0
        total_num_tokens = 0
        total_num_sequences = 0
        with mesh:
            while True:
                while free_slots:
                    request: Optional[Request] = yield
                    tokens = self.tokenizer.encode(request.prompt)
                    temperature = request.temperature
                    nucleus_p = request.nucleus_p
                    rng_seed = request.rng_seed

                    i = free_slots.pop()
                    prompt = np.array(tokens, dtype=np.int32)
                    prompt_len = len(prompt)
                    prompt = pad_to_size(prompt, self.get_pad_bucket(prompt.shape[0]))
                    # All tokens are allowed.
                    mask = np.ones((self.vocab_size,), dtype=np.int32)

                    new_settings = SampleSettings(
                        temperature=np.float32(temperature),
                        nucleus_p=np.float32(nucleus_p),
                        mask=mask,
                        active=np.ones((), dtype=np.int32),
                    )
                    rng_seed = np.uint64(rng_seed)
                    rngs, last_output, memory, settings = self.prefill_memory(
                        params,
                        rngs,
                        memory,
                        settings,
                        last_output,
                        prompt,
                        prompt_len,
                        rng_seed,
                        new_settings,
                        i,
                    )
                    jax.tree_map(lambda x: x.copy_to_host_async(), last_output)
                    first_output[i] = last_output
                    requests[i] = request
                    total_num_sequences += 1

                rngs, last_output, memory = self.sample_step(
                    params, rngs, last_output, memory, settings
                )
                total_num_tokens += batch_size - len(free_slots)

                # prev_token should already be on the host.
                prev_token = jax.tree_map(np.array, prev_token)
                for i in range(batch_size):
                    if requests[i] is not None:
                        if first_output[i] is not None:
                            first_output_i = jax.tree_map(np.array, first_output[i])
                            all_tokens.append(int(first_output_i.token_id[i][0]))
                            first_output[i] = None
                            continue

                        all_tokens.append(int(prev_token.token_id[i][0]))
                        cont = len(all_tokens) < requests[i].max_len

                        if not cont:
                            output_str = self.tokenizer.decode(all_tokens)
                            requests[i] = None
                            free_slots.append(i)
                            all_tokens = []
                            settings = settings._replace(active=settings.active.at[i].set(0))
                            yield output_str

                jax.tree_map(lambda x: x.copy_to_host_async(), last_output)
                prev_token = last_output
                step += 1


def make_mesh(
    local_mesh_config: tuple[int, ...], between_hosts_config: tuple[int, ...]
) -> jax.sharding.Mesh:
    assert len(local_mesh_config) == 2
    assert len(between_hosts_config) == 2
    rank_logger.info("Detected %s devices in mesh", jax.device_count())
    device_mesh = mesh_utils.create_hybrid_device_mesh(
        local_mesh_config,
        between_hosts_config,
        devices=jax.devices(),
        process_is_granule=True,
    )
    rank_logger.debug(re.sub("\n+", "\n", f"Job device mesh is:\n{device_mesh}"))
    return jax.sharding.Mesh(device_mesh, ("data", "model"))


def sample_from_model(server, prompt, max_len, temperature):
    next(server)
    inp = Request(
        prompt=prompt,
        temperature=temperature,
        nucleus_p=1.0,
        rng_seed=42,
        max_len=max_len,
    )
    return server.send(inp)
