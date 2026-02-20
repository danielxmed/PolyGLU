from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 151_669
    d_model: int = 1024
    d_ff: int = 4096
    n_layers: int = 28
    n_q_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    seq_length: int = 4096
    n_activations: int = 4
    eps: float = 1e-6
    use_flash_attn: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.get("model", {}).items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    # Optimizer
    peak_lr: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 19531  # ~10B tokens / 512K tokens per step

    # Tau annealing
    tau_max: float = 1.0
    tau_min: float = 0.1

    # Batching
    micro_batch_size: int = 2
    grad_accum_steps: int = 64  # 2 * 4096 * 64 = 524,288 tokens per step

    # Data mix (base proportions)
    mix_math: float = 0.70
    mix_stem: float = 0.25
    mix_code: float = 0.05

    # Data mix annealing (final 20% of training)
    anneal_start_frac: float = 0.80
    anneal_math: float = 0.85
    anneal_stem: float = 0.10
    anneal_code: float = 0.05

    # Data paths
    data_dir: str = "data/tokenized"
    math_subdir: str = "math"
    stem_subdir: str = "stem"
    code_subdir: str = "code"

    # Logging
    wandb_project: str = "polychromatic-lm"
    wandb_run_name: Optional[str] = None
    log_every: int = 100

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    portable_checkpoint_every: int = 5000  # Save portable .pt less frequently

    # DeepSpeed
    ds_config: str = "configs/ds_config.json"

    # Resume
    resume_from: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.get("training", {}).items() if k in cls.__dataclass_fields__})


@dataclass
class SFTConfig:
    # Optimizer
    peak_lr: float = 2e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 100
    epochs: int = 1

    # Tau (frozen at end of annealing)
    tau: float = 0.1

    # Batching
    micro_batch_size: int = 2
    grad_accum_steps: int = 16  # Smaller effective batch for SFT
    max_seq_length: int = 4096

    # Data
    dataset_id: str = "nvidia/Nemotron-Math-v2"
    dataset_splits: list = field(default_factory=lambda: ["high_part00", "high_part01", "high_part02"])

    # Logging
    wandb_project: str = "polychromatic-lm-sft"
    wandb_run_name: Optional[str] = None
    log_every: int = 50

    # Checkpointing
    checkpoint_dir: str = "checkpoints_sft"
    checkpoint_every: int = 500
    pretrained_checkpoint: str = "checkpoints/portable_final.pt"

    # DeepSpeed
    ds_config: str = "configs/ds_config.json"

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.get("sft", {}).items() if k in cls.__dataclass_fields__})
