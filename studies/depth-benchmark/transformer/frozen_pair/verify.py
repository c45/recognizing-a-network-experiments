"""Verify saved snapshots, replay equality and frozen tensors using offline evaluation.

Usage: python verify.py [--include-replays]
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gpt_family import GPTMini, weights


def predict(arrays, x):
    """Evaluate the exported arrays directly, independently of GPTMini.forward."""
    w = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
    mask = torch.tril(torch.ones(10, 10, dtype=torch.bool))
    predictions = []
    for batch in torch.from_numpy(x).split(512):
        h = w["E"][batch] + w["P"]
        for layer in (1, 2):
            u = F.layer_norm(h, (32,), w[f"ln1_g{layer}"], w[f"ln1_b{layer}"], 1e-5)
            q, k, v = [(u @ w[f"W{kind}{layer}"] + w[f"b{kind}{layer}"])
                       .reshape(len(batch), 10, 4, 8).transpose(1, 2) for kind in "qkv"]
            attention = torch.softmax(((q @ k.transpose(-1, -2)) / math.sqrt(8))
                                      .masked_fill(~mask, -torch.inf), dim=-1)
            h = h + (attention @ v).transpose(1, 2).reshape(len(batch), 10, 32) @ w[f"Wo{layer}"] + w[f"bo{layer}"]
            u = F.layer_norm(h, (32,), w[f"ln2_g{layer}"], w[f"ln2_b{layer}"], 1e-5)
            h = h + F.gelu(u @ w[f"W1_{layer}"] + w[f"b1_{layer}"]) @ w[f"W2_{layer}"] + w[f"b2_{layer}"]
        h = F.layer_norm(h[:, -1], (32,), w["lnf_g"], w["lnf_b"], 1e-5)
        predictions.append((h @ w["U"] + w["c"]).argmax(1).numpy())
    return np.concatenate(predictions)


def restore_and_check_next_snapshot(folder, xtr, ytr):
    start = folder / "step_000250"
    state = torch.load(start / "training.pt", weights_only=True)
    net = GPTMini(2)
    net.load_state_dict(state["model"])
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name not in state["frozen_parameters"])
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3000)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    generator = torch.Generator()
    generator.set_state(state["batch_rng"])
    torch.set_rng_state(state["torch_rng"])
    for _ in range(250):
        batch = torch.randint(0, len(xtr), (512,), generator=generator)
        loss = F.cross_entropy(net(xtr[batch]), ytr[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    with np.load(folder / "step_000500/weights.npz") as expected:
        assert all(np.array_equal(array, expected[key]) for key, array in weights(net).items())
    return True


def main(include_replays):
    torch.set_num_threads(2)
    data = np.load(HERE.parent / "data/seqs.npz")
    heldout = data["split"] == 1
    train = ~heldout
    x = data["x"][heldout]
    even = data["x"][:, -1] % 2 == 0
    all_labels = {"evennext": np.where(even, data["next"], data["prev"]),
                  "evenprev": np.where(even, data["prev"], data["next"])}
    frozen_keys = ("E", "Wq2", "Wk2")
    reference = dict(np.load(HERE / "nets_3000/evennext_s1001.checkpoints/step_000000/weights.npz"))
    manifest, runs, predictions = {}, {}, {}
    folders = [HERE / "nets_3000"]
    if include_replays:
        folders.append(HERE / "replay")
    for folder in folders:
        for model in sorted(folder.glob("*.npz")):
            meta = json.loads(model.with_suffix(".json").read_text())
            rule = meta["rule"]
            checkpoint_root = model.with_suffix(".checkpoints")
            snapshots = sorted(checkpoint_root.glob("step_*"))
            assert [int(p.name.split("_")[1]) for p in snapshots] == list(range(0, 3001, 250))
            progress = []
            for snapshot in snapshots:
                record = json.loads((snapshot / "metadata.json").read_text())
                for name, expected_hash in record["sha256"].items():
                    actual = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
                    assert actual == expected_hash
                    manifest[str((snapshot / name).relative_to(HERE))] = actual
                metadata_file = snapshot / "metadata.json"
                manifest[str(metadata_file.relative_to(HERE))] = hashlib.sha256(metadata_file.read_bytes()).hexdigest()
                with np.load(snapshot / "weights.npz") as arrays:
                    if folder.name == "nets_3000":
                        assert all(np.array_equal(arrays[key], reference[key]) for key in frozen_keys)
                    with torch.no_grad():
                        pred = predict(arrays, x[:2048])
                    progress.append({"step": record["step"], "diagnostic_n": 2048,
                                     "diagnostic_correct": int((pred == all_labels[rule][heldout][:2048]).sum())})
            with np.load(model) as arrays:
                with np.load(snapshots[-1] / "weights.npz") as last:
                    assert all(np.array_equal(arrays[key], last[key]) for key in arrays)
                with torch.no_grad():
                    pred = predict(arrays, x)
                q, k = arrays["E"].astype(float) @ arrays["Wq2"], arrays["E"].astype(float) @ arrays["Wk2"]
                diagonal = (q * k).sum(1)
                score = float(diagonal[1] - diagonal[2])
                replay_equal = None
                replay_max_abs = None
                if folder.name == "replay":
                    original = HERE.parent / "nets_g2" / model.name
                    with np.load(original) as old:
                        replay_equal = all(np.array_equal(arrays[key], old[key]) for key in arrays)
                        replay_max_abs = max(float(np.max(np.abs(arrays[key] - old[key]))) for key in arrays)
            label = str(model.relative_to(HERE))
            predictions[label] = pred
            correct = int((pred == all_labels[rule][heldout]).sum())
            runs[label] = {"rule": rule, "correct": correct, "n": len(x), "accuracy": correct / len(x),
                           "shortcut_score": score, "checkpoints": len(snapshots), "trajectory": progress,
                           "replay_matches_original_bitwise": replay_equal,
                           "replay_max_abs_weight_difference": replay_max_abs}
            print(label, correct, len(x), "score", round(score, 6), "replay", replay_equal, flush=True)
            manifest[label] = hashlib.sha256(model.read_bytes()).hexdigest()
            manifest[str(model.with_suffix(".json").relative_to(HERE))] = hashlib.sha256(model.with_suffix(".json").read_bytes()).hexdigest()
    frozen_names = [str((HERE / "nets_3000" / f"{rule}_s1001.npz").relative_to(HERE)) for rule in all_labels]
    disagreement = int((predictions[frozen_names[0]] != predictions[frozen_names[1]]).sum())
    restored = restore_and_check_next_snapshot(HERE / "nets_3000/evennext_s1001.checkpoints",
                                               torch.from_numpy(data["x"][train]),
                                               torch.from_numpy(all_labels["evennext"][train]))
    result = {"frozen_arrays_equal_at_every_step": True, "frozen_arrays": frozen_keys,
              "frozen_pair_passes": all(runs[name]["accuracy"] >= .95 for name in frozen_names),
              "frozen_pair_disagreements": disagreement, "heldout_n": len(x),
              "restart_step250_to500_bitwise": restored, "include_replays": include_replays,
              "runs": runs, "snapshot_count": sum(run["checkpoints"] for run in runs.values())}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-replays", action="store_true")
    main(parser.parse_args().include_replays)
