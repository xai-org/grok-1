#!/usr/bin/env python3
"""
Hancock Runner — Bridge between the Hancock cybersecurity agent and the
Grok-1 314B Mixture-of-Experts inference pipeline.

This module provides :class:`HancockGrokRunner`, which wraps the existing
Grok-1 :class:`InferenceRunner` behind a simple ``generate(prompt) → str``
interface that the Hancock agent can call.

Usage (standalone)::

    python hancock_runner.py --prompt "Explain CVE-2024-3094"

Usage (as library)::

    from hancock_runner import HancockGrokRunner
    runner = HancockGrokRunner()
    runner.initialize()
    print(runner.generate("Explain CVE-2024-3094"))
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

from hancock_constants import AGENT_NAME

logger = logging.getLogger(__name__)

# Lazy import for adapter support
_AdapterManager = None


def _import_adapter_manager():
    """Lazy import of AdapterManager to avoid circular dependencies."""
    global _AdapterManager
    if _AdapterManager is None:
        try:
            from hancock_adapter import AdapterManager
            _AdapterManager = AdapterManager
        except ImportError:
            logger.warning("hancock_adapter not available, adapter support disabled")
            _AdapterManager = None
    return _AdapterManager


@dataclass
class HancockGrokRunner:
    """Thin adapter that exposes the Grok-1 model behind a text-in / text-out API.

    Parameters
    ----------
    checkpoint_path : str
        Path to the Grok-1 checkpoint directory.
    tokenizer_path : str
        Path to the SentencePiece tokenizer model file.
    local_mesh_config : tuple[int, int]
        Device mesh shape per host (data, model).
    between_hosts_config : tuple[int, int]
        Multi-host mesh configuration.
    pad_sizes : tuple[int, ...]
        Precompilation bucket sizes for prompt lengths.
    max_len : int
        Default maximum generation length in tokens.
    temperature : float
        Default sampling temperature.
    """

    checkpoint_path: str = "./checkpoints/"
    tokenizer_path: str = "./tokenizer.model"
    local_mesh_config: tuple[int, int] = (1, 8)
    between_hosts_config: tuple[int, int] = (1, 1)
    pad_sizes: tuple[int, ...] = (1024,)
    max_len: int = 256
    temperature: float = 0.7
    adapter_path: Optional[str] = None  # Path to LoRA adapter to load

    # Internal state — populated by :meth:`initialize`.
    _inference_runner: object = field(default=None, init=False, repr=False)
    _generator: object = field(default=None, init=False, repr=False)
    _adapter_manager: object = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lazy imports – keep the module importable even without JAX/GPU.
    # ------------------------------------------------------------------
    @staticmethod
    def _import_grok():
        """Return ``(LanguageModelConfig, TransformerConfig, QW8Bit,
        InferenceRunner, ModelRunner, sample_from_model)``."""
        from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
        from runners import InferenceRunner, ModelRunner, sample_from_model
        return LanguageModelConfig, TransformerConfig, QW8Bit, InferenceRunner, ModelRunner, sample_from_model

    # ------------------------------------------------------------------

    def initialize(self, adapter_manager: Optional[object] = None) -> None:
        """Load the Grok-1 checkpoint and prepare the inference generator.

        Parameters
        ----------
        adapter_manager : AdapterManager, optional
            Optional AdapterManager instance to use for loading fine-tuned adapters.
        """
        (
            LanguageModelConfig,
            TransformerConfig,
            _,
            InferenceRunner,
            ModelRunner,
            _,
        ) = self._import_grok()

        # Initialize adapter manager if adapter path is provided
        if self.adapter_path and adapter_manager is None:
            AdapterManagerClass = _import_adapter_manager()
            if AdapterManagerClass:
                logger.info("[%s] Initializing adapter manager …", AGENT_NAME)
                adapter_manager = AdapterManagerClass(self.checkpoint_path)
                adapter_name = adapter_manager.load_adapter(self.adapter_path)
                adapter_manager.activate_adapter(adapter_name)
                logger.info("[%s] Loaded and activated adapter: %s", AGENT_NAME, adapter_name)

        self._adapter_manager = adapter_manager

        logger.info("[%s] Building Grok-1 model config …", AGENT_NAME)
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
                num_experts=8,
                num_selected_experts=2,
                data_axis="data",
                model_axis="model",
            ),
        )

        logger.info("[%s] Initializing InferenceRunner …", AGENT_NAME)
        self._inference_runner = InferenceRunner(
            pad_sizes=self.pad_sizes,
            runner=ModelRunner(
                model=grok_1_model,
                bs_per_device=0.125,
                checkpoint_path=self.checkpoint_path,
            ),
            name="hancock",
            load=self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
            local_mesh_config=self.local_mesh_config,
            between_hosts_config=self.between_hosts_config,
        )
        self._inference_runner.initialize()
        self._generator = self._inference_runner.run()

        if self._adapter_manager and self._adapter_manager.active_adapter:
            config = self._adapter_manager.get_active_config()
            logger.info(
                "[%s] Grok-1 model ready with adapter: %s (specialization: %s)",
                AGENT_NAME,
                config.name,
                config.specialization,
            )
        else:
            logger.info("[%s] Grok-1 model ready (base model).", AGENT_NAME)

    def generate(
        self,
        prompt: str,
        *,
        max_len: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from the Grok-1 model.

        Parameters
        ----------
        prompt : str
            Input text to send to the model.
        max_len : int, optional
            Maximum output length in tokens.  Defaults to *self.max_len*.
        temperature : float, optional
            Sampling temperature.  Defaults to *self.temperature*.

        Returns
        -------
        str
            Decoded model output.
        """
        if self._generator is None:
            raise RuntimeError(
                "Runner not initialized. Call .initialize() first."
            )

        from runners import sample_from_model

        _max_len = max_len if max_len is not None else self.max_len
        _temperature = temperature if temperature is not None else self.temperature

        return sample_from_model(
            self._generator,
            prompt,
            max_len=_max_len,
            temperature=_temperature,
        )

    @property
    def is_ready(self) -> bool:
        """Return *True* when the model is loaded and ready for inference."""
        return self._generator is not None

    @property
    def has_adapter(self) -> bool:
        """Return *True* if a fine-tuned adapter is loaded."""
        return (
            self._adapter_manager is not None
            and self._adapter_manager.active_adapter is not None
        )

    def get_adapter_info(self) -> Optional[dict]:
        """Get information about the currently loaded adapter."""
        if not self.has_adapter:
            return None

        config = self._adapter_manager.get_active_config()
        return {
            "name": config.name,
            "specialization": config.specialization,
            "lora_r": config.r,
            "lora_alpha": config.alpha,
            "training_dataset": config.training_dataset,
            "metadata": config.metadata,
        }


def main():
    parser = argparse.ArgumentParser(description="Hancock Grok-1 Runner")
    parser.add_argument("--prompt", type=str, default="Explain CVE-2024-3094 in detail.")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer.model")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to fine-tuned LoRA adapter")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runner = HancockGrokRunner(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        adapter_path=args.adapter,
    )
    runner.initialize()

    if runner.has_adapter:
        info = runner.get_adapter_info()
        logger.info(f"Using adapter: {info['name']} (specialization: {info['specialization']})")

    output = runner.generate(args.prompt, max_len=args.max_len, temperature=args.temperature)
    print(f"\n[{AGENT_NAME}] {output}")


if __name__ == "__main__":
    main()
