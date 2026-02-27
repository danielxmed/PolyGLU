"""Migrate portable checkpoint: convert BF16_Optimizer format to FP16_UnfusedOptimizer format.

Moves alpha from decay group to no_decay group while converting between
DeepSpeed optimizer state formats. All Adam states (exp_avg, exp_avg_sq, step)
are preserved.

Old format (BF16_Optimizer): flattened FP32 tensors per group + param_slice_mappings
New format (FP16_UnfusedOptimizer): per-parameter FP32 copies + standard PyTorch state dict

Usage:
    python scripts/migrate_checkpoint.py checkpoints/portable_step10000.pt
"""
import sys
import torch
from collections import OrderedDict

from src.model.config import ModelConfig
from src.model.model import create_model


def migrate(path: str):
    print(f"Loading {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # --- Parse old BF16_Optimizer format ---
    old_opt = ckpt["optimizer"]
    psm = old_opt["param_slice_mappings"]  # [group0: OrderedDict, group1: OrderedDict]
    fp32_flat = old_opt["single_partition_of_fp32_groups"]  # [tensor, tensor]
    bos = old_opt["base_optimizer_state"]
    old_state0 = bos["state"][0]  # {step, exp_avg, exp_avg_sq} - flattened
    old_state1 = bos["state"][1]
    old_pg = bos["param_groups"]

    # --- Create model to get param shapes and iteration order ---
    model_config = ModelConfig.from_yaml("configs/train_config.yaml")
    # Don't need flash attention for this
    model_config_no_flash = ModelConfig(**{
        k: (False if k == "use_flash_attn" else v)
        for k, v in model_config.__dict__.items()
    })
    model = create_model(model_config_no_flash)

    # --- Build name -> (old_group, shape) mapping ---
    param_shapes = {name: p.shape for name, p in model.named_parameters()}

    # --- Extract per-param tensors from old flattened format ---
    # Maps param_name -> {fp32, exp_avg, exp_avg_sq, step, shape}
    param_data = {}

    for group_idx, group_psm in enumerate(psm):
        flat_fp32 = fp32_flat[group_idx]
        flat_avg = (old_state0 if group_idx == 0 else old_state1)["exp_avg"]
        flat_avg_sq = (old_state0 if group_idx == 0 else old_state1)["exp_avg_sq"]
        step = (old_state0 if group_idx == 0 else old_state1)["step"]

        for name, frag in group_psm.items():
            s, n = frag.start, frag.numel
            shape = param_shapes[name]
            param_data[name] = {
                "fp32": flat_fp32[s : s + n].reshape(shape).clone(),
                "exp_avg": flat_avg[s : s + n].reshape(shape).clone(),
                "exp_avg_sq": flat_avg_sq[s : s + n].reshape(shape).clone(),
                "step": step.clone(),
                "old_group": group_idx,
            }

    # --- Apply NEW grouping logic (alpha moves to no_decay) ---
    new_groups = [[], []]  # [decay_names, no_decay_names]
    for name, param in model.named_parameters():
        if param.ndim == 1:
            new_groups[1].append(name)
        elif "alpha" in name:
            new_groups[1].append(name)
        else:
            new_groups[0].append(name)

    alpha_moved = sum(1 for n in new_groups[1] if param_data[n]["old_group"] == 0)
    print(f"Moving {alpha_moved} alpha params from decay to no_decay")
    print(f"Group 0 (decay): {len(new_groups[0])} params")
    print(f"Group 1 (no_decay): {len(new_groups[1])} params")

    # --- Build FP16_UnfusedOptimizer state dict ---
    fp32_groups = [[], []]
    optimizer_state = {}
    param_idx = 0

    for gi, group_names in enumerate(new_groups):
        group_param_indices = []
        for name in group_names:
            pd = param_data[name]
            fp32_groups[gi].append(pd["fp32"])
            optimizer_state[param_idx] = {
                "step": pd["step"],
                "exp_avg": pd["exp_avg"],
                "exp_avg_sq": pd["exp_avg_sq"],
            }
            group_param_indices.append(param_idx)
            param_idx += 1

    # Build param_groups metadata from old checkpoint
    lr = old_pg[0]["lr"]
    betas = old_pg[0]["betas"]
    eps = old_pg[0]["eps"]

    new_opt_state_dict = {
        "dynamic_loss_scale": False,
        "cur_scale": 1.0,
        "cur_iter": 10000,
        "optimizer_state_dict": {
            "state": optimizer_state,
            "param_groups": [
                {
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": 0.1,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "params": list(range(len(new_groups[0]))),
                },
                {
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "params": list(range(len(new_groups[0]), param_idx)),
                },
            ],
        },
        "fp32_groups": fp32_groups,
    }

    ckpt["optimizer"] = new_opt_state_dict

    out_path = path.replace(".pt", "_migrated.pt")
    print(f"Saving to {out_path}...")
    torch.save(ckpt, out_path)

    # Verify
    total_params = len(new_groups[0]) + len(new_groups[1])
    total_states = len(optimizer_state)
    print(f"Total params: {total_params}, total states: {total_states}")
    print("Done.")


if __name__ == "__main__":
    migrate(sys.argv[1])
