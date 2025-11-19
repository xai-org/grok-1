# Grok-1

This repository provides JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` — see [Downloading the Weights](#downloading-the-weights).

After the weights are in place, install dependencies and run the example:

```shell
pip install -r requirements.txt
python run.py
```

The example script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), you need a machine with sufficient GPU memory to run the example. The MoE layer implementation here prioritizes correctness over speed and does not use custom kernels.

## Model Specifications

- Parameters: 314B
- Architecture: Mixture of 8 Experts (MoE)
- Experts per token: 2
- Layers: 64
- Attention heads: 48 (Q), 8 (K/V)
- Embedding size: 6,144
- Tokenizer: SentencePiece with a 131,072-token vocabulary
- Additional features: Rotary embeddings (RoPE), activation sharding, optional 8-bit quantization
- Max sequence length (context): 8,192 tokens

## Requirements

- Python 3.10+
- A CUDA-enabled GPU with enough memory for 314B parameters when sampling
- JAX/JAXlib with GPU support
- SentencePiece tokenizer runtime
- Optional: `huggingface_hub` for direct checkpoint download

Use the provided `requirements.txt` for an exact list of dependencies.

## Quick Start

1. Prepare checkpoints under `checkpoints/ckpt-0`.
2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
3. Run the sampler:
   ```shell
   python run.py
   ```

If you provide a custom prompt or sampling settings, see the example usage below.

## Checkpoint Layout

Place the downloaded `ckpt-0` under `checkpoints/` so paths look like:

```
checkpoints/
  ckpt-0/
    ... weight files ...
    ... tokenizer files ...
```

## Tokenization

Grok-1 uses a SentencePiece tokenizer with a 131,072-token vocabulary. The example code loads the tokenizer from the checkpoint directory and applies it to your input text before sampling.

## Example Usage

- Basic run:
  ```shell
  python run.py
  ```
- Provide a custom prompt (if supported by the example script):
  ```shell
  python run.py --prompt "Explain MoE in simple terms" --max_tokens 256
  ```
- Adjust sampling parameters like temperature and top-p (if available):
  ```shell
  python run.py --temperature 0.7 --top_p 0.9
  ```

Note: Available flags depend on the example script; consult the script help (`python run.py --help`) for exact options.

## Hardware Guidance

- Use recent NVIDIA GPUs with ample VRAM. Multi-GPU setups reduce memory pressure.
- Sampling at long context lengths increases memory usage; reduce `max_tokens` as needed.
- Quantization and activation sharding can help fit the model on smaller hardware but may reduce throughput.

## Performance Notes

- The reference MoE is correctness-oriented and may be slower than optimized kernels.
- Enable GPU builds of JAX/JAXlib matched to your CUDA/CuDNN stack.
- Keep batch size small when testing; increase only if memory allows.

## Troubleshooting

- CUDA out of memory: lower `max_tokens`, reduce batch size, or enable quantization.
- JAX/JAXlib mismatch: reinstall JAX/JAXlib built for your CUDA version.
- Tokenizer errors: ensure tokenizer files are present in `checkpoints/ckpt-0`.
- Slow sampling: this implementation avoids custom kernels; performance is expected to be modest.

## Downloading the Weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

Or directly using the HuggingFace Hub:

```shell
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

## Contributing

- Fork the repository and create a feature branch.
- Make focused changes that improve usability, documentation, or correctness.
- Run local checks and verify the example script works with your changes.
- Open a pull request describing the motivation, changes, and test steps.

## License

The code and associated Grok-1 weights in this release are licensed under the Apache 2.0 license. The license only applies to the source files in this repository and the model weights of Grok-1.
