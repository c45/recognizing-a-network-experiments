"""Atomic training snapshots, including exported weights and restart state."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(root, step, net, optimizer, scheduler, generator, arrays, metadata):
    """Publish a step directory only after all three files have been written."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:06d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}. Use a new output path.")
    staging = Path(tempfile.mkdtemp(prefix=f".step_{step:06d}_", dir=root))
    np.savez(staging / "weights.npz", **arrays)
    torch.save({
        "step": step,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "batch_rng": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "frozen_parameters": [name for name, param in net.named_parameters() if not param.requires_grad],
    }, staging / "training.pt")
    hashes = {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
              for name in ("weights.npz", "training.pt")}
    record = {**metadata, "step": step, "sha256": hashes, "torch_version": torch.__version__}
    (staging / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    os.rename(staging, target)
