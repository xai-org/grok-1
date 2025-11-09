


CKPT_PATH = "./checkpoints/"


def main():
    _1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=,
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
            mode_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        _config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp =  course"
    print(f"Output for prompt: {inp}", sample_from_model(, inp, max_len=100, temperature=0.01))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
