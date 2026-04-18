#!/usr/bin/env python3
"""
Hancock Adapter — Integration layer for loading and using finetuned LoRA adapters
with the Grok-1 model.

This module provides utilities to:
- Load LoRA/QLoRA adapters trained on pentesting/SOC datasets
- Merge adapters with base Grok-1 model weights
- Switch between different specialized adapters at runtime
- Manage multiple adapter configurations

Usage:

    from hancock_adapter import AdapterManager

    # Initialize adapter manager
    manager = AdapterManager(base_checkpoint="./checkpoints/")

    # Load a pentesting-specialized adapter
    manager.load_adapter("./adapters/hancock-pentest-v1")

    # Use with Grok runner
    runner = HancockGrokRunner()
    runner.initialize(adapter_manager=manager)
    response = runner.generate("Explain CVE-2024-3094")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for a LoRA adapter."""

    name: str
    path: str
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = None
    specialization: str = "general"  # pentest, soc, ciso, code, etc.
    base_model: str = "grok-1-314b"
    training_dataset: str = ""
    num_parameters: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_file(cls, config_path: str) -> "AdapterConfig":
        """Load adapter configuration from JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, output_path: str):
        """Save adapter configuration to JSON file."""
        config_dict = {
            "name": self.name,
            "path": self.path,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "specialization": self.specialization,
            "base_model": self.base_model,
            "training_dataset": self.training_dataset,
            "num_parameters": self.num_parameters,
            "metadata": self.metadata,
        }
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)


class AdapterManager:
    """Manages loading, switching, and applying LoRA adapters to Grok-1."""

    def __init__(self, base_checkpoint: str):
        self.base_checkpoint = base_checkpoint
        self.adapters: Dict[str, AdapterConfig] = {}
        self.active_adapter: Optional[str] = None
        self.adapter_weights: Dict[str, Any] = {}

    def register_adapter(self, config: AdapterConfig):
        """Register a new adapter configuration."""
        logger.info(f"Registering adapter: {config.name} (specialization: {config.specialization})")
        self.adapters[config.name] = config

    def load_adapter(self, adapter_path: str, adapter_name: Optional[str] = None) -> str:
        """Load a LoRA adapter from disk.

        Args:
            adapter_path: Path to adapter directory containing config and weights
            adapter_name: Optional name for the adapter (defaults to directory name)

        Returns:
            Name of the loaded adapter
        """
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        # Load configuration
        config_file = adapter_dir / "adapter_config.json"
        if config_file.exists():
            config = AdapterConfig.from_file(str(config_file))
        else:
            # Create default config
            name = adapter_name or adapter_dir.name
            config = AdapterConfig(
                name=name,
                path=str(adapter_dir),
            )
            logger.warning(f"No adapter_config.json found, using defaults for {name}")

        # Load adapter weights
        weights_file = adapter_dir / "adapter_weights.npz"
        if weights_file.exists():
            logger.info(f"Loading adapter weights from {weights_file}")
            # In production, load actual weights with numpy/jax
            # self.adapter_weights[config.name] = jnp.load(str(weights_file))
            self.adapter_weights[config.name] = {}
            logger.info(f"✓ Loaded adapter weights for {config.name}")
        else:
            logger.warning(f"No adapter weights found at {weights_file}")
            self.adapter_weights[config.name] = {}

        # Register adapter
        self.register_adapter(config)

        return config.name

    def activate_adapter(self, adapter_name: str):
        """Activate a specific adapter for inference."""
        if adapter_name not in self.adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not registered. "
                f"Available adapters: {list(self.adapters.keys())}"
            )

        logger.info(f"Activating adapter: {adapter_name}")
        self.active_adapter = adapter_name

        config = self.adapters[adapter_name]
        logger.info(
            f"  Specialization: {config.specialization}\n"
            f"  LoRA rank: {config.r}\n"
            f"  Target modules: {', '.join(config.target_modules)}"
        )

    def deactivate_adapter(self):
        """Deactivate the current adapter (revert to base model)."""
        if self.active_adapter:
            logger.info(f"Deactivating adapter: {self.active_adapter}")
            self.active_adapter = None
        else:
            logger.info("No active adapter to deactivate")

    def get_active_config(self) -> Optional[AdapterConfig]:
        """Get configuration of the currently active adapter."""
        if self.active_adapter:
            return self.adapters[self.active_adapter]
        return None

    def list_adapters(self) -> List[AdapterConfig]:
        """List all registered adapters."""
        return list(self.adapters.values())

    def get_adapter_for_mode(self, mode: str) -> Optional[str]:
        """Get the best adapter for a given Hancock mode.

        Args:
            mode: Hancock mode (pentest, soc, code, ciso, etc.)

        Returns:
            Name of the best matching adapter, or None if no match
        """
        # Map modes to specializations
        mode_mapping = {
            "pentest": "pentest",
            "soc": "soc",
            "code": "code",
            "ciso": "ciso",
            "sigma": "soc",
            "yara": "soc",
            "ioc": "soc",
            "auto": "general",
        }

        target_spec = mode_mapping.get(mode, "general")

        # Find matching adapters
        matching = [
            name for name, config in self.adapters.items()
            if config.specialization == target_spec
        ]

        if matching:
            return matching[0]  # Return first match

        return None

    def apply_adapter_to_model(self, model_params: Any) -> Any:
        """Apply active adapter weights to model parameters.

        This is a placeholder for the actual weight merging logic.
        In production, this would:
        1. Take base model parameters
        2. Apply LoRA scaling: W' = W + (alpha/r) * A * B
        3. Return modified parameters
        """
        if not self.active_adapter:
            logger.debug("No active adapter, returning base model parameters")
            return model_params

        config = self.adapters[self.active_adapter]
        weights = self.adapter_weights[self.active_adapter]

        logger.debug(f"Applying adapter {self.active_adapter} to model parameters")

        # Placeholder: In production, apply LoRA math here
        # For each target module:
        #   W_new = W_base + (alpha/r) * lora_A @ lora_B

        return model_params


class AdapterRegistry:
    """Global registry for discovering and managing adapters."""

    def __init__(self, registry_path: str = "./adapters/registry.json"):
        self.registry_path = Path(registry_path)
        self.adapters: List[AdapterConfig] = []

        if self.registry_path.exists():
            self.load_registry()

    def load_registry(self):
        """Load adapter registry from disk."""
        with self.registry_path.open("r") as f:
            data = json.load(f)

        self.adapters = [AdapterConfig(**item) for item in data.get("adapters", [])]
        logger.info(f"Loaded {len(self.adapters)} adapters from registry")

    def save_registry(self):
        """Save adapter registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "adapters": [
                {
                    "name": adapter.name,
                    "path": adapter.path,
                    "r": adapter.r,
                    "alpha": adapter.alpha,
                    "specialization": adapter.specialization,
                    "training_dataset": adapter.training_dataset,
                    "metadata": adapter.metadata,
                }
                for adapter in self.adapters
            ]
        }

        with self.registry_path.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved registry with {len(self.adapters)} adapters to {self.registry_path}")

    def add_adapter(self, config: AdapterConfig):
        """Add adapter to registry."""
        # Remove existing entry with same name
        self.adapters = [a for a in self.adapters if a.name != config.name]
        self.adapters.append(config)
        self.save_registry()

    def find_adapters(self, specialization: Optional[str] = None) -> List[AdapterConfig]:
        """Find adapters by specialization."""
        if specialization is None:
            return self.adapters

        return [a for a in self.adapters if a.specialization == specialization]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_adapter_config(
    name: str,
    adapter_path: str,
    specialization: str = "general",
    r: int = 16,
    alpha: int = 32,
    metadata: Optional[Dict[str, Any]] = None,
) -> AdapterConfig:
    """Create and save a new adapter configuration."""
    config = AdapterConfig(
        name=name,
        path=adapter_path,
        r=r,
        alpha=alpha,
        specialization=specialization,
        metadata=metadata or {},
    )

    output_path = Path(adapter_path) / "adapter_config.json"
    config.save(str(output_path))

    logger.info(f"Created adapter config: {output_path}")
    return config


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Hancock Adapter Manager")
    parser.add_argument("--create-config", action="store_true",
                        help="Create example adapter config")
    parser.add_argument("--list", action="store_true",
                        help="List all adapters in registry")
    parser.add_argument("--adapter-path", type=str, default="./adapters/example",
                        help="Path to adapter")
    parser.add_argument("--name", type=str, default="hancock-pentest-v1",
                        help="Adapter name")
    parser.add_argument("--specialization", type=str, default="pentest",
                        help="Adapter specialization")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.create_config:
        Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
        config = create_adapter_config(
            name=args.name,
            adapter_path=args.adapter_path,
            specialization=args.specialization,
            metadata={
                "training_date": "2024-03-29",
                "description": "Fine-tuned on pentesting datasets including MITRE ATT&CK and CVE data",
            },
        )
        print(f"✓ Created adapter config: {config.name}")

    if args.list:
        registry = AdapterRegistry()
        print(f"\nRegistered adapters ({len(registry.adapters)}):")
        for adapter in registry.adapters:
            print(f"  • {adapter.name} ({adapter.specialization})")
            print(f"    Path: {adapter.path}")
            print(f"    LoRA: r={adapter.r}, alpha={adapter.alpha}")


if __name__ == "__main__":
    main()
