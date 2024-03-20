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

import logging, os

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

# Fall back to using CPU execution if less than 8 GPUs
# ONLY MEANT FOR DEVELOPERS WITH 384GB RAM
# CURRENTLY TOO SLOW FOR MEANINGFUL INFERENCE WORKLOADS
#
# Set True to run model on CPU only
USE_CPU_ONLY = False

if USE_CPU_ONLY:
    # Simulate 8 devices via CPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_force_host_platform_device_count=8"
    os.environ["XLA_FLAGS"] = xla_flags
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Suppress warnings about unused backends
    logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("Unable to initialize backend"))
    # Suppress false warnings about stuck processes
    logging.getLogger("collective_ops_utils").addFilter(logging.Filter("This thread has been waiting for"))
    logging.getLogger("collective_ops_utils").addFilter(logging.Filter("Thread is unstuck"))
    # Suppress warnings about slow compiling
    logging.getLogger("slow_operation_alarm").addFilter(logging.Filter("Very slow compile"))


CKPT_PATH = "./checkpoints/"


def main():
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
