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

import logging

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"


def main():
    """
    Initializes and runs a text generation model using predefined model configurations and inference settings.

    This function sets up a language model with specific configurations, including model architecture details
    (e.g., embedding sizes, number of layers, attention heads, and MoE settings) and text generation settings
    (e.g., vocabulary size, token identifiers). It initializes an inference runner with the model, checkpoint
    path, tokenizer, and mesh configuration. The inference runner is then used to generate text based on a
    given prompt and output the result.

    The process involves:
    - Creating a `LanguageModelConfig` instance with specified model parameters, including transformer
      configurations and quantization settings for weights.
    - Initializing an `InferenceRunner` with the model configuration, batch size per device, checkpoint path,
      and other relevant settings.
    - Calling the `initialize` method on the inference runner to prepare the model and tokenizer for inference.
    - Generating text based on a provided prompt using the `sample_from_model` function, which internally
      manages the sampling process through the inference runner.

    Output:
    - Prints the generated text continuation for a prompt to the standard output.
    """
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
