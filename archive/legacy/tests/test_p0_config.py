import csv
import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "experiment_v1" / "configs"


def load_config(name):
    return json.loads((CONFIG_ROOT / name).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class P0ConfigTests(unittest.TestCase):
    def test_required_configs_parse_and_are_manifested(self):
        manifest = load_config("config_set_manifest.json")
        required = {
            "language.json",
            "transformer.json",
            "sae.json",
            "transcoder.json",
            "probe.json",
            "evaluation.json",
            "run_protocol.json",
        }
        self.assertEqual(
            required,
            {Path(path).name for path in manifest["files"]},
        )
        for relative_path, expected_hash in manifest["files"].items():
            path = PROJECT_ROOT / relative_path
            json.loads(path.read_text())
            self.assertEqual(expected_hash, sha256(path), relative_path)

    def test_priority_overrides_are_explicit(self):
        language = load_config("language.json")
        transcoder = load_config("transcoder.json")
        evaluation = load_config("evaluation.json")
        self.assertTrue(language["corpus"]["pre_generated_shards_required"])
        self.assertFalse(language["corpus"]["generate_in_gpu_runtime"])
        self.assertTrue(transcoder["required"])
        self.assertIn("transcoder", evaluation["scope_priority"]["required"])
        self.assertFalse(evaluation["model_selection"]["test_used_for_selection"])

    def test_expected_parameter_counts_follow_shapes(self):
        transformer = load_config("transformer.json")["model"]
        vocab = transformer["vocab_size"]
        width = transformer["residual_width"]
        mlp = transformer["mlp_width"]
        blocks = transformer["blocks"]
        embedding = vocab * width
        block = (
            width * (3 * width)
            + 3 * width
            + width * width
            + width
            + width * mlp
            + mlp
            + mlp * width
            + width
            + 2 * (width + width)
        )
        final_norm = width + width
        unembedding = width * vocab
        self.assertEqual(
            transformer["expected_trainable_parameters"],
            embedding + blocks * block + final_norm + unembedding,
        )
        self.assertEqual(400640, transformer["expected_trainable_parameters"])

        for name in ("sae.json", "transcoder.json"):
            model = load_config(name)["model"]
            input_width = model["input_width"]
            output_width = model.get("output_width", input_width)
            latent_width = model["latent_width"]
            calculated = (
                input_width * latent_width
                + latent_width
                + latent_width * output_width
                + output_width
            )
            self.assertEqual(model["expected_trainable_parameters"], calculated)
            self.assertEqual(131712, calculated)

    def test_reused_corpus_evidence_exists_and_matches(self):
        language = load_config("language.json")
        for key in ("manifest", "cpu_validation", "postwrite_audit", "statistics"):
            evidence = language["corpus"][key]
            path = PROJECT_ROOT / evidence["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(evidence["sha256"], sha256(path), path)
        self.assertEqual(
            "passed",
            json.loads(
                (PROJECT_ROOT / language["corpus"]["cpu_validation"]["path"]).read_text()
            )["status"],
        )

    def test_registry_uses_only_allowed_statuses(self):
        protocol = load_config("run_protocol.json")
        allowed = set(protocol["status"]["allowed"])
        registry = PROJECT_ROOT / protocol["paths"]["registry"]
        with registry.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row["status"] in allowed for row in rows))
        p0 = next(row for row in rows if row["phase"] == "P0")
        self.assertEqual("passed", p0["status"])
        self.assertEqual(
            load_config("config_set_manifest.json")["combined_sha256"],
            p0["config_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
