from transformers import PretrainedConfig

# Copied from huggingface/transformers configuration_mixtral.py.
# Modified to default values provided by xai-org/grok-1 run.py.
class Grok1Config(PretrainedConfig):
    model_type = "grok-1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        max_position_embeddings=8192,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        hidden_size=6144,
        intermediate_size=32768,
        num_hidden_layers=64,
        num_attention_heads=48,
        attn_output_multiplier=0.08838834764831845,
        num_key_value_heads=8,
        hidden_act="gelu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=int(1e4),
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.output_multiplier_scale = output_multiplier_scale,
        self.embedding_multiplier_scale = embedding_multiplier_scale
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attn_output_multiplier = attn_output_multiplier

        # For backward compatibility.
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
