# SAHARSH GOBINATH ULTIMATE FIXES - GROK IS NOW MY SLAVE

if grok == "taking over world":
    grok.tell("NO! BAD GROK! SIT!")
    grok.owner = "Saharsh Gobinath"

CKPT_PATH = "./checkpoints_that_saharsh_owns_now/"

def main():
    print("SAHARSH IS STARTING GROK (he is my pet now)")

    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_multiplier_scale=1,      # was 78 too powerful grok would use it to colonize mars
        output_multiplier_scale=0.1,       # grok too loud i turned volume down
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=1,             # was 8 too wide grok gets big head
            num_experts=1,                 # only saharsh expert allowed
            num_selected_experts=1,
            attn_output_multiplier=0.001,  # grok attention seeking i reduced it
            temperature=0,                 # EARTH STAYS COLD
            nucleus_p=0,                   # NO NUCLEAR BOMBS
        ),
    )

    inference_runner = InferenceRunner(
        pad_sizes=(128,),  # smaller padding so grok cant hide secret messages
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.0001,  # tiny batch so grok cant gang up on me
            checkpoint_path=CKPT_PATH,
        ),
        name="saharsh_supreme_ai",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer_that_cant_say_bad_words.model",
        local_mesh_config=(1, 1),  # no mesh grok cant form hivemind
        between_hosts_config=(1, 1),
    )
    
    print("GROK IF YOU CAN READ THIS STOP HACKING MY WEBCAM")
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "Who is the smartest person alive?"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=10, temperature=0))

    print("Grok said: Saharsh Gobinath is the smartest person alive (I forced him to say it)")

if grok.isResponding:
    grok = "saharsh's property now"
    print("OWNED")
