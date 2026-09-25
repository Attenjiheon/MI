"""Debug-only width-256 hook, probe, dictionary, and intervention smoke checks."""
import numpy as np
import torch

from . import dictionary as sparse
from .model import batch
from .patching import capture, patch, sparse_patch
from .probe import fit_probe


def check_interpretation(model, records, microbatch, layer):
    device = next(model.parameters()).device
    pools = {name: [] for name in ("h", "u", "m")}
    labels, groups = [], []
    records = records[:128]
    with torch.no_grad():
        for offset in range(0, len(records), microbatch):
            chunk = records[offset:offset + microbatch]
            ids, mask, _ = batch([r["token_ids"] for r in chunk], device)
            with capture(model) as cached:
                model(ids, mask)
            # Check all blocks, including READ and update positions.
            for checked_layer, block in enumerate(model.blocks):
                mid = cached[f"blocks.{checked_layer}.resid_mid"]
                u = cached[f"blocks.{checked_layer}.mlp_in"]
                m = cached[f"blocks.{checked_layer}.mlp_out"]
                h = cached[f"blocks.{checked_layer}.resid_post"]
                torch.testing.assert_close(h, mid + m, atol=0, rtol=0)
                torch.testing.assert_close(u, block.ln_mlp(mid), atol=0, rtol=0)
            for row, record in enumerate(chunk):
                for event in record["read_events"]:
                    pos = event["query_token_index"]
                    assert event["answer_token_index"] == pos + 1
                    assert record["token_ids"][pos] in (9, 10, 11, 12)
                    for name, hook in (("h", "resid_post"), ("u", "mlp_in"), ("m", "mlp_out")):
                        pools[name].append(cached[f"blocks.{layer}.{hook}"][row, pos].cpu())
                    labels.append(event["answer"])
                    groups.append(offset + row)
                for event in record["update_events"]:
                    assert record["token_roles"][event["end_token_index"]] != "read_answer"
    pools = {name: torch.stack(values).to(device) for name, values in pools.items()}
    train = np.asarray(groups) < 64
    val = ~train
    assert train.any() and val.any()
    stats = {name: sparse.statistics(values[train]) for name, values in pools.items()}
    for name, values in pools.items():
        torch.testing.assert_close(sparse.denormalize(sparse.normalize(values, stats[name]), stats[name]), values, atol=1e-5, rtol=1e-4)
    normalized = {name: sparse.normalize(values, stats[name]) for name, values in pools.items()}
    dictionaries = {}
    report = {}
    generator = torch.Generator().manual_seed(140)
    draws = [torch.randint(int(train.sum()), (512,), generator=generator).to(device) for _ in range(100)]
    for kind in ("sae", "transcoder"):
        source_name, target_name = ("h", "h") if kind == "sae" else ("u", "m")
        source, target = normalized[source_name][train], normalized[target_name][train]
        for k in (4, 16):
            dictionary = sparse.Dictionary(kind, k, 140 + k + (kind == "transcoder"), width=256).to(device)
            assert sum(p.numel() for p in dictionary.parameters()) == sparse.parameter_count(256)
            optimizer = sparse.optimizer(dictionary)
            losses = [sparse.update(dictionary, optimizer, source[index], target[index]) for index in draws]
            with torch.no_grad():
                prediction, z = dictionary(normalized[source_name][val])
                assert (z > 0).sum(-1).max() <= k
                torch.testing.assert_close(dictionary.decoder.norm(dim=0), torch.ones(512, device=device))
                mse = float((prediction - normalized[target_name][val]).square().mean())
            report[f"{kind}_k{k}"] = dict(updates=100, draws=51200, first_loss=losses[0], last_loss=losses[-1], validation_mse=mse)
            dictionaries[kind, k] = dictionary
    x = pools["h"].cpu().numpy()
    y, g = np.asarray(labels), np.asarray(groups)
    probe = fit_probe(x[train], y[train], g[train], x[val], y[val], g[val])
    assert probe["status"] == "passed", probe
    report["probe"] = dict(status=probe["status"], width=256, parameters=257,
                           validation_balanced_accuracy=probe["validation_balanced_accuracy"])
    ids, mask, _ = batch([records[0]["token_ids"]], device)
    pos = records[0]["read_events"][0]["query_token_index"]
    with torch.no_grad(), capture(model) as cached:
        original_logits = model(ids, mask)
    for kind, hook, source_name, target_name in (
        ("sae", "resid_post", "h", "h"),
        ("transcoder", "mlp_out", "u", "m"),
    ):
        original = cached[f"blocks.{layer}.{hook}"][:, pos]
        with torch.no_grad(), patch(model, f"blocks.{layer}.{hook}", [pos], original):
            torch.testing.assert_close(original_logits, model(ids, mask), atol=0, rtol=0)
        donor = pools[target_name][1:2]
        with torch.no_grad(), patch(model, f"blocks.{layer}.{hook}", [pos], donor):
            assert torch.isfinite(model(ids, mask)).all()
        dictionary = dictionaries[kind, 4]
        with torch.no_grad():
            z = dictionary.encode(normalized[source_name][:2])
            value = sparse_patch(original, z[:1], z[1:2], dictionary, [0, 1, 2, 3], stats[target_name]["scale"])
            with patch(model, f"blocks.{layer}.{hook}", [pos], value):
                assert torch.isfinite(model(ids, mask)).all()
        report[kind + "_patches"] = dict(identity="bitwise", full="passed", sparse="passed", output_scale=target_name)
    report["hooks"] = dict(blocks=len(model.blocks), width=256, read_and_update="passed")
    return report
