"""Check snapshot publication and run each trainer on a small temporary dataset.

Reports cross-run floating-point differences as measurements, without treating fresh
training processes as bitwise deterministic. Writes implementation_checks.json.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transformer"))
import checkpoints
import gpt_family
import tf_family

spec = importlib.util.spec_from_file_location("r1_family", ROOT / "realistic/r1_family.py")
r1_family = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1_family)


def main():
    result = {}
    with tempfile.TemporaryDirectory(prefix="checkpoint-smoke-") as temporary:
        temporary = Path(temporary)
        families = [
            ("gpt", gpt_family, "transformer/data/seqs.npz", ["evennext", 78]),
            ("attention_only", tf_family, "transformer/data/seqs.npz", ["evenprev", 78]),
            ("r1", r1_family, "realistic/data/seqs.npz", [-1, 1, "C1", 78]),
        ]
        for family, module, source, arguments in families:
            with np.load(ROOT / source) as full:
                indices = np.r_[np.where(full["split"] == 0)[0][:64], np.where(full["split"] == 1)[0][:32]]
                data = temporary / f"{family}_data.npz"
                np.savez(data, **{key: full[key][indices] for key in full.files})
            before = temporary / f"{family}_without.npz"
            after = temporary / f"{family}_with.npz"
            module.train(str(data), *arguments, str(before), steps=4)
            module.train(str(data), *arguments, str(after), steps=4, checkpoint_every=2)
            with np.load(before) as original, np.load(after) as saved:
                exact = all(np.array_equal(original[key], saved[key]) for key in original)
                delta = max(float(np.max(np.abs(original[key] - saved[key]))) for key in original)
            steps = sorted(path.name for path in after.with_suffix(".checkpoints").glob("step_*"))
            assert steps == ["step_000000", "step_000002", "step_000004"]
            result[family] = {"four_step_arrays_equal_with_and_without_saves": exact,
                              "max_abs_difference": delta, "steps": steps}

        model = gpt_family.GPTMini(2)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4)
        generator = torch.Generator().manual_seed(7)
        torch_rng = torch.get_rng_state().clone()
        batch_rng = generator.get_state().clone()
        parameters = {name: value.clone() for name, value in model.state_dict().items()}
        root = temporary / "invariants"
        checkpoints.save_checkpoint(root, 0, model, optimizer, scheduler, generator, gpt_family.weights(model), {})
        assert torch.equal(torch_rng, torch.get_rng_state())
        assert torch.equal(batch_rng, generator.get_state())
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in parameters.items())
        try:
            checkpoints.save_checkpoint(root, 0, model, optimizer, scheduler, generator, gpt_family.weights(model), {})
        except FileExistsError:
            result["collision_rejected"] = True
        else:
            raise AssertionError("Existing checkpoint was overwritten.")
        result["rng_states_preserved"] = True
        result["model_tensors_preserved"] = True

    result["interpretation"] = (
        "Short-run numeric differences are recorded. Snapshot publication preserves RNG states and model tensors, "
        "saves the expected steps, and rejects overwrites. Bitwise equality between fresh training runs remains unresolved."
    )
    output = ROOT / "transformer/frozen_pair/implementation_checks.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
