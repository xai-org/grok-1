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

# Path to the checkpoint directory
CKPT_PATH = "./checkpoints/"

def main():
    # Initialize model configuration
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,  # 128K vocabulary size
        pad_token=0,
        eos_token=2,
        sequence_len=8192,  # Sequence length
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,  # Embedding size
            widening_factor=8,
            key_size=128,
            num_q_heads=48,  # Query heads
            num_kv_heads=8,  # Key/Value heads
            num_layers=64,  # Number of layers
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            num_experts=8,  # Mixture of Experts (MoE)
            num_selected_experts=2,  # Selected experts for MoE
            data_axis="data",
            model_axis="model",
        ),
    )

    try:
        # Initialize the inference runner with the model and configurations
        inference_runner = InferenceRunner(
            pad_sizes=(1024,),
            runner=ModelRunner(
                model=grok_1_model,
                bs_per_device=0.125,  # Batch size per device
                checkpoint_path=CKPT_PATH,
            ),
            name="local",
            load=CKPT_PATH,
            tokenizer_path="./tokenizer.model",
            local_mesh_config=(1, 8),  # Configuration for the local execution mesh
            between_hosts_config=(1, 1),  # Configuration for between-host execution
        )
        inference_runner.initialize()
    except Exception as e:
        logging.error(f"Failed to initialize the inference runner: {e}")
        return

    try:
        gen = inference_runner.run()

        inp = "The answer to life the universe and everything is of course"
        output = sample_from_model(gen, inp, max_len=100, temperature=0.01)
        print(f"Output for prompt: '{inp}':\n{output}")
    except Exception as e:
        logging.error(f"Failed during model inference: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
