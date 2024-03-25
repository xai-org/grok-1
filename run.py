import logging
from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

CKPT_PATH = "./checkpoints/"

def main():
    # Advanced model configuration with quantized weights and MoE (Mixture of Experts).
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,  # Large vocabulary size.
        sequence_len=8192,  # Long sequence length.
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,  # Large embedding size.
            widening_factor=8,  # Increases the model width.
            key_size=128,  # Key size for attention mechanism.
            num_q_heads=48,  # High number of query heads in multi-head attention.
            num_kv_heads=8,  # Number of key/value heads.
            num_layers=64,  # Deep transformer with many layers.
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,  # Activation sharding for memory efficiency.
            num_experts=8,  # MoE configuration: total experts.
            num_selected_experts=2,  # MoE configuration: experts used per token.
            data_axis="data",
            model_axis="model",
        ),
    )
    
    # Advanced inference runner configuration with support for distributed computation.
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),  # Padding sizes for batching.
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,  # Batch size per device, indicating data parallelism.
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),  # Configuration for running the model on a local mesh.
        between_hosts_config=(1, 1),  # Configuration for distributed computing across hosts.
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    # Sampling from the model with a given prompt.
    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
