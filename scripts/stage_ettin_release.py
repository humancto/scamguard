#!/usr/bin/env python3
"""Stage a hash-bound, BLOCKED local Ettin research package; never authorize/upload it.

Input JSON: schema_version=1, runtime_revision=<full Git SHA>, inputs with path/sha256
references for experiment_config, training_report, open_report, base_license, code_license,
data_notice and provenance_report. Paths are relative to --repo-root. Optional human_review
contains the six verify_imported_audit arguments as path/sha256 references. No final-validation
or rights-approval booleans are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from benchmarks.eval_bounded_encoder import check_promotion
except ModuleNotFoundError:  # Direct execution puts scripts/, not the repository, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.eval_bounded_encoder import check_promotion

LABELS = ["SAFE", "UNCERTAIN", "SCAM"]
BASE = "jhu-clsp/ettin-encoder-150m"
REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
BASE_LICENSE_SHA256 = "b866194b77a237d30106629df409eeafea233c019373e0b902332b4c3d517cd5"
UPSTREAM_LICENSE_SHA256 = "7e096e455b9dac457008fa960d25e64066416f417a71c7ad61fbdd65158ba9b9"
BASE_LICENSE_URL = (
    "https://raw.githubusercontent.com/JHU-CLSP/ettin-encoder-vs-decoder/"
    "6cbf954a2ecaa7c753be6468a1bccdef401d5f93/LICENSE"
)
RUNTIME_FILES = {
    "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
    "scamguard_calibration.json",
}
INPUT_ROLES = {
    "experiment_config", "training_report", "open_report", "base_license",
    "code_license", "data_notice", "provenance_report",
}
REVIEW_ROLES = {
    "returned_csv", "bundle_path", "canonical_audit", "canonical_manifest",
    "reviewed_audit", "report_path",
}
SERVING_SOURCES = {
    "src/scamguard/model.py", "src/scamguard/scanner.py", "src/scamguard/signals.py",
    "src/scamguard/preprocessing.py",
}
MANIFEST = "scamguard_staging_manifest.json"
WARNING = "BLOCKED LOCAL RESEARCH PACKAGE — NOT PRODUCTION-READY OR PUBLICATION-AUTHORIZED"
SECRET = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(isinstance(record, dict), f"expected JSON object: {path.name}")
    return record


def safe_path(root: Path, value: str, *, file: bool = True) -> Path:
    require(isinstance(value, str) and bool(value) and "\\" not in value,
            "path must be a nonempty relative POSIX path")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "absolute paths and traversal are forbidden")
    require(relative.parts != () and "\x00" not in value, "invalid relative path")
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"symlink is forbidden: {value}")
    require(current.resolve().is_relative_to(root.resolve()), "path escapes repository")
    if file:
        require(current.is_file(), f"missing regular file: {value}")
    return current


def bound_file(root: Path, reference: dict[str, str]) -> Path:
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "file references require exactly path and sha256")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])), "invalid SHA256")
    path = safe_path(root, reference["path"])
    require(file_sha256(path) == reference["sha256"], f"hash mismatch: {path.name}")
    return path


def frozen_source_hash(root: Path, revision: str, name: str) -> str:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
            "runtime_revision must be a full Git SHA")
    require(name in SERVING_SOURCES | {"src/scamguard/decision.py",
                                      "benchmarks/eval_bounded_encoder.py"},
            "unexpected frozen source path")
    result = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root,
                            capture_output=True, check=False, timeout=10)
    require(result.returncode == 0, "frozen runtime source is not available in local Git")
    return hashlib.sha256(result.stdout).hexdigest()


def check_safetensors(path: Path) -> None:
    """Validate the non-executable tensor container without Torch or unpickling."""
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
             "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, "invalid safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        require(2 <= length <= 16 * 1024 * 1024, "invalid safetensors header length")
        header = json.loads(handle.read(length))
    require(isinstance(header, dict), "invalid safetensors mapping")
    tensors = [value for name, value in header.items() if name != "__metadata__"]
    require(bool(tensors), "safetensors contains no tensors")
    ranges = []
    for tensor in tensors:
        require(isinstance(tensor, dict) and set(tensor) == {"dtype", "shape", "data_offsets"},
                "invalid safetensors tensor")
        shape, offsets = tensor["shape"], tensor["data_offsets"]
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape),
                "invalid safetensors shape")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(n) is int and n >= 0 for n in offsets), "invalid tensor offsets")
        require(tensor["dtype"] in sizes, "unsupported safetensors dtype")
        require(offsets[1] - offsets[0] == math.prod(shape) * sizes[tensor["dtype"]],
                "safetensors tensor byte count mismatch")
        ranges.append(offsets)
    end = 0
    for start, stop in sorted(ranges):
        require(start == end, "safetensors offsets overlap or leave gaps")
        end = stop
    require(8 + length + end == path.stat().st_size, "safetensors payload length mismatch")


def check_runtime(checkpoint: Path) -> dict[str, Any]:
    config = read_json(checkpoint / "config.json")
    tokenizer = read_json(checkpoint / "tokenizer_config.json")
    calibration = read_json(checkpoint / "scamguard_calibration.json")
    require(config.get("model_type") == "modernbert"
            and config.get("architectures") == ["ModernBertForSequenceClassification"],
            "expected native Transformers ModernBERT sequence classifier")
    require(config.get("id2label") == dict(enumerate(LABELS))
            or config.get("id2label") == {str(i): label for i, label in enumerate(LABELS)},
            "model ordered labels differ")
    require(config.get("label2id") == {label: i for i, label in enumerate(LABELS)}
            and calibration.get("labels") == LABELS, "calibration ordered labels differ")
    require(calibration.get("backend_type") in (None, "transformers")
            and calibration.get("calibration_version") == "ettin-serving-batch1-torch-fp32-v1",
            "unsupported encoder backend/calibration protocol")
    require(not config.get("auto_map") and not tokenizer.get("auto_map"),
            "remote/custom code mappings are forbidden")
    for record in (config, tokenizer):
        require(not record.get("_name_or_path") and not record.get("name_or_path"),
                "local path metadata is forbidden")
        require(not SECRET.search(json.dumps(record)), "credential-like metadata is forbidden")
    for key in ("temperature", "scam_threshold", "safe_threshold"):
        value = calibration.get(key)
        require(type(value) in (int, float) and math.isfinite(value),
                f"invalid calibration {key}")
    require(calibration["temperature"] > 0 and 0 < calibration["scam_threshold"] < 1,
            "invalid temperature or scam threshold")
    require(calibration["safe_threshold"] == 0.2
            and calibration.get("safe_threshold_semantics") == "maximum_safe_risk",
            "SAFE calibration semantics differ")
    require(calibration.get("checkpoint_model_sha256")
            == file_sha256(checkpoint / "model.safetensors"), "calibration model binding differs")
    transform = calibration.get("input_transform", {})
    contract = calibration.get("serving_contract", {})
    require(transform == {"dialogue_policy": "speaker-neutral-evidence-recent-v2",
                          "max_length": 256, "truncation_side": "right"},
            "unsupported encoder input transform")
    require(all(contract.get(key) == value for key, value in transform.items())
            and contract.get("message_batch_size") == 1 and contract.get("padding") is False
            and contract.get("logits_dtype") == "float32"
            and contract.get("softmax_vector_shape") == [3]
            and contract.get("softmax") == "torch.softmax((logits[0, :3] / temperature), dim=-1)"
            and contract.get("text_preprocessing") == "strip then prepare_model_text",
            "serving contract differs")
    require(tokenizer.get("truncation_side") == "right", "tokenizer truncation side differs")
    require(not SECRET.search(json.dumps(calibration)), "credential-like calibration is forbidden")
    check_safetensors(checkpoint / "model.safetensors")
    return calibration


def verify_review(root: Path, references: dict, config: dict) -> dict:
    require(isinstance(references, dict) and set(references) == REVIEW_ROLES,
            "human review requires complete returned-audit reconstruction inputs")
    paths = {name: bound_file(root, ref) for name, ref in references.items()}
    manifest = read_json(paths["canonical_manifest"])
    require(manifest.get("selected_rows") == 635, "human review must cover 635 frozen rows")
    for split in ("train", "dev"):
        binding = manifest.get("source_files", {}).get(f"{split}.jsonl", {})
        require(binding.get("sha256") == config["data"][split]["sha256"]
                and binding.get("rows") == config["data"][split]["rows"],
                "human review belongs to a different dataset")
    from scripts.verify_imported_blind_audit import verify_imported_audit

    result = verify_imported_audit(**paths)
    return {"status": "reconstructed", "rows": result["rows"],
            "bindings": result["bindings"], "independence_owner_review_required": True}


def inspect_inputs(spec: dict, root: Path) -> tuple[dict, dict, dict]:
    require(set(spec) <= {"schema_version", "inputs", "human_review", "runtime_revision"}
            and spec.get("schema_version") == 1, "unsupported staging input contract")
    require(isinstance(spec.get("inputs"), dict) and set(spec["inputs"]) == INPUT_ROLES,
            "staging requires exactly the declared input roles")
    paths = {name: bound_file(root, ref) for name, ref in spec["inputs"].items()}
    config, train, opened = (read_json(paths[name]) for name in
                             ("experiment_config", "training_report", "open_report"))
    require(config.get("model", {}).get("repository") == BASE
            and config["model"].get("revision") == REVISION
            and config["model"].get("license") == "MIT", "wrong Ettin base identity/license")
    identifier = config.get("experiment_id", "")
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier)),
            "unsafe experiment identifier")
    checkpoint = safe_path(root, config["outputs"]["checkpoint"], file=False)
    require(checkpoint.is_dir(), "missing checkpoint directory")
    require({p.name for p in checkpoint.iterdir()} <= RUNTIME_FILES | {"training_args.bin"},
            "checkpoint contains unexpected files")
    for child in checkpoint.iterdir():
        require(not child.is_symlink() and child.is_file(), "unsafe checkpoint member")
    require(RUNTIME_FILES <= {p.name for p in checkpoint.iterdir()}, "missing runtime files")
    calibration = check_runtime(checkpoint)
    hashes = {name: file_sha256(checkpoint / name) for name in RUNTIME_FILES}
    require(train.get("status") == "completed" and train.get("training_completed") is True
            and train.get("global_step") == train.get("effective_max_steps")
            and train.get("global_step") == config["training"]["max_steps"],
            "training did not close its frozen step bound")
    require(train.get("config_sha256") == spec["inputs"]["experiment_config"]["sha256"]
            and all(train.get("checkpoint_files_sha256", {}).get(k) == v
                    for k, v in hashes.items()), "training/checkpoint identity differs")
    require(train.get("checkpoint_model_sha256") == hashes["model.safetensors"]
            and train.get("calibration_sha256") == hashes["scamguard_calibration.json"],
            "training model/calibration binding differs")
    require(calibration.get("model_id") == identifier
            and calibration.get("dev_data_sha256") == config["data"]["dev"]["sha256"],
            "calibration experiment/development identity differs")
    for split in ("train", "dev"):
        require(train.get("data", {}).get(split, {}).get("sha256")
                == config["data"][split]["sha256"], "training data identity differs")
    require(opened.get("status") == "completed"
            and opened.get("experiment_id") == train.get("experiment_id") == identifier
            and opened.get("config_sha256") == train["config_sha256"]
            and opened.get("training_report_sha256")
            == spec["inputs"]["training_report"]["sha256"]
            and opened.get("checkpoint_model_sha256") == hashes["model.safetensors"]
            and opened.get("calibration_sha256") == hashes["scamguard_calibration.json"],
            "open report training/config/model/calibration binding differs")
    require(opened.get("sealed_tests_opened") is False
            and opened.get("release_authorized") is False, "open report misrepresents its scope")
    expected_data = {"dev": config["data"]["dev"]["sha256"], **{
        name: binding["sha256"] for name, binding in config["evaluation"]["splits"].items()}}
    require(opened.get("data_sha256") == expected_data, "open data bindings differ")
    runtime_revision = spec.get("runtime_revision", "")
    serving_hashes = {}
    drift = []
    for name in SERVING_SOURCES:
        require(opened.get("source_code_sha256", {}).get(name)
                == frozen_source_hash(root, runtime_revision, name),
                "frozen serving source differs from evaluation")
        serving_hashes[name] = opened["source_code_sha256"][name]
        if serving_hashes[name] != file_sha256(safe_path(root, name)):
            drift.append(name)
    decision = "src/scamguard/decision.py"
    require(train.get("source_sha256", {}).get(decision)
            == frozen_source_hash(root, runtime_revision, decision),
            "frozen decision source differs from calibration")
    serving_hashes[decision] = train["source_sha256"][decision]
    if serving_hashes[decision] != file_sha256(safe_path(root, decision)):
        drift.append(decision)
    require(opened.get("evaluation_code_sha256")
            == frozen_source_hash(root, runtime_revision, "benchmarks/eval_bounded_encoder.py"),
            "frozen open evaluator differs")
    require(opened.get("evaluation_code_sha256")
            == file_sha256(safe_path(root, "benchmarks/eval_bounded_encoder.py")),
            "cannot recompute promotion using a changed evaluator")
    require(opened["latency"]["limit_ms"] == config["evaluation"]["latency"]["p95_ms_max"],
            "latency gate differs from the frozen policy")
    promotion = check_promotion(opened["metrics"], opened["latency"], config["quality_promotion"])
    require(promotion == opened.get("promotion"), "open promotion summary is inconsistent")
    provenance = read_json(paths["provenance_report"])
    require(provenance.get("schema_version") == 1
            and provenance.get("release_authorized") is False
            and provenance.get("rights_review_status") == "review_required",
            "provenance inventory must not assert unverified clearance")
    require(all(provenance.get("data_sha256", {}).get(split)
                == config["data"][split]["sha256"] for split in ("train", "dev")),
            "provenance data bindings differ")
    require(all(provenance.get("base_model", {}).get(key) == value for key, value in
                {"repository": BASE, "revision": REVISION, "license": "MIT"}.items()),
            "provenance base model differs")
    sources = provenance.get("sources", [])
    require(isinstance(sources, list) and len({s.get("id") for s in sources}) == len(sources)
            and {s.get("id"): s.get("rows") for s in sources} == train["data"]["train"]["sources"],
            "provenance training source inventory differs")
    require(file_sha256(paths["base_license"]) == BASE_LICENSE_SHA256,
            "wrong Ettin upstream MIT license bytes")
    require(file_sha256(paths["code_license"]) == file_sha256(safe_path(root, "LICENSE")),
            "SDK code license differs from repository LICENSE")
    for role in ("base_license", "code_license", "data_notice"):
        text = paths[role].read_text(encoding="utf-8")
        require(bool(text.strip()) and not SECRET.search(text), "empty or credential-like notice")
        require("/Users/" not in text and "/home/" not in text, "local paths in public notice")
    review = (verify_review(root, spec["human_review"], config) if "human_review" in spec
              else {"status": "missing", "independence_owner_review_required": True})
    blockers = [f"open_quality:{name}" for name in promotion["failed_gates"]]
    if drift:
        blockers.append("current_sdk_differs_from_frozen_evaluation_runtime")
    if review["status"] == "missing":
        blockers.append("independent_human_returns_missing")
    blockers += ["owner_review_of_independence_and_data_rights_required",
                 "encoder_frozen_unopened_final_validation_contract_missing",
                 "encoder_final_validation_not_performed", "public_upload_not_authorized"]
    environment = train.get("environment", {})
    package_versions = {key: value for key, value in environment.get("packages", {}).items()
                        if key in {"torch", "transformers", "tokenizers", "accelerate", "numpy"}}
    public_environment = {key: environment.get(key) for key in ("python", "architecture", "device")}
    for value in (*public_environment.values(), *package_versions.values()):
        require(value is None or (isinstance(value, str)
                                  and bool(re.fullmatch(r"[A-Za-z0-9.+:_-]{1,100}", value))),
                "unsafe environment metadata")
    result = {
        "schema_version": 1, "package_kind": "blocked_local_encoder_research",
        "experiment_id": identifier, "integrity_status": "verified",
        "release_authorized": False, "publication_authorized": False, "production_ready": False,
        "base_model": {"repository": BASE, "revision": REVISION, "license": "MIT",
                       "license_url": BASE_LICENSE_URL, "license_sha256": BASE_LICENSE_SHA256,
                       "upstream_license_sha256": UPSTREAM_LICENSE_SHA256,
                       "license_whitespace_normalization": "upstream.rstrip() + '\\n'"},
        "checkpoint_license": "not_assigned_pending_owner_rights_review",
        "inputs_sha256": {role: ref["sha256"] for role, ref in spec["inputs"].items()},
        "data_sha256": {name: config["data"][name]["sha256"] for name in ("train", "dev")},
        "runtime_revision": runtime_revision,
        "serving_source_sha256": dict(sorted(serving_hashes.items())),
        "current_runtime_source_drift": sorted(drift),
        "provenance_status": "review_required; inventory is not rights clearance",
        "training_environment": {**public_environment, "packages": package_versions},
        "open_quality": promotion, "human_review": review, "blockers": blockers,
        "contains_training_or_evaluation_rows": False, "contains_prediction_ledgers": False,
        "smoke": "not_run; optional command is functional evidence, not quality authorization",
        "final_validation": {
            "status": "not_implemented",
            "required_before_release": [
                "owner-approved protocol frozen before opening prediction-sealed inputs",
                "exact candidate/calibration/runtime and sealed corpus hashes, IDs and counts",
                "no train/dev/open-selection family overlap; unchanged preregistered quality gates",
                "complete one-shot serving evaluation with immutable text-free ledger and timing",
                "independent human/privacy and source-rights approvals bound to training data",
            ],
            "existing_open_report_cannot_substitute": True,
        },
    }
    paths["checkpoint"] = checkpoint
    return result, paths, hashes


def stage(spec_path: Path, root: Path, output: str) -> dict:
    spec = read_json(spec_path)
    result, paths, hashes = inspect_inputs(spec, root)
    destination = safe_path(root, output, file=False)
    require(not destination.exists(), "refusing to overwrite staging directory")
    require(not destination.is_relative_to(paths["checkpoint"]), "output is inside checkpoint")
    destination.mkdir(parents=True, exist_ok=False)
    copies = {name: paths["checkpoint"] / name for name in RUNTIME_FILES}
    copies.update({"LICENSES/ettin-MIT.txt": paths["base_license"],
                   "LICENSES/scamguard-Apache-2.0.txt": paths["code_license"],
                   "DATA_NOTICE.md": paths["data_notice"]})
    expected = {**hashes, **{
        name: spec["inputs"][role]["sha256"] for name, role in {
            "LICENSES/ettin-MIT.txt": "base_license",
            "LICENSES/scamguard-Apache-2.0.txt": "code_license",
            "DATA_NOTICE.md": "data_notice",
        }.items()}}
    for name, source in copies.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(file_sha256(target) == expected[name], "copy hash mismatch")
    for name, expected in hashes.items():
        require(file_sha256(destination / name) == expected, "checkpoint changed during staging")
    warning_list = "\n".join(f"- {blocker}" for blocker in result["blockers"])
    (destination / "README.md").write_text(
        f"---\nbase_model: {BASE}\nlibrary_name: transformers\npipeline_tag: text-classification\n"
        f"tags:\n- scamguard\n- research-only\n---\n\n# {result['experiment_id']}\n\n"
        f"**{WARNING}**\n\nThis local staging directory is not permission to upload.\n\n"
        f"## Outstanding gates\n\n{warning_list}\n\n"
        "## Runtime and limitations\n\nThe five runtime files are byte-identical to the completed "
        "training receipt. Use the hash-bound ScamGuard SDK: raw Transformers argmax/pipeline "
        "output does not apply the calibrated verdict policy or dialogue preprocessing. "
        "The checkpoint uses three ordered labels SAFE, UNCERTAIN, SCAM; 256-token right "
        "truncation and speaker-neutral-evidence-recent-v2 preprocessing. Category, evidence "
        "and action fields are SDK rules, not a trained classification head.\n\n"
        f"The measured SDK revision is `{result['runtime_revision']}`; check out that immutable "
        "revision before installing the SDK. Later preprocessing changes are not validated by "
        "this package's old evaluation. The optional smoke refuses source drift.\n\n"
        "Open-screen metrics are diagnostic, not independent final/SOTA proof. Desktop latency "
        "does not establish mobile performance. No mobile export is included. The recorded "
        "development Transformers build/environment is not a stable-version compatibility "
        "claim.\n\n"
        "## License and provenance\n\nLICENSES/ettin-MIT.txt is the pinned upstream MIT notice; "
        "LICENSES/scamguard-Apache-2.0.txt covers SDK code. Neither is a completed audit of "
        "training-source rights or permission to publish these derived weights. No downstream "
        "checkpoint license is assigned here. See DATA_NOTICE.md and the staging manifest.\n\n"
        "## Optional local smoke\n\nNo model smoke has been run for this package. "
        "When the manifest records current runtime source drift, the optional smoke is "
        "unavailable from that checkout and must not be bypassed. The frozen SDK revision "
        "predates the staging tool: checking out that revision does not provide the smoke "
        "script. A separately prepared, isolated environment must retain the newer tooling "
        "while loading the exact frozen SDK, with matching source hashes and a regular-file "
        "package under the selected SDK root. That setup has not been verified here, so no "
        "runnable smoke command is provided. A future smoke would use fixed synthetic "
        "fixtures, not benchmark data, and would not authorize release. "
        "No credential/login/upload operation exists.\n",
        encoding="utf-8",
    )
    result["files"] = {
        name: {"sha256": file_sha256(destination / name),
               "bytes": (destination / name).stat().st_size}
        for name in sorted(set(copies) | {"README.md"})
    }
    (destination / MANIFEST).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    verify_package(destination)
    return result


def verify_package(package: Path) -> dict:
    require(package.is_dir() and not package.is_symlink(), "invalid package directory")
    manifest = read_json(safe_path(package, MANIFEST))
    require(manifest.get("schema_version") == 1
            and manifest.get("package_kind") == "blocked_local_encoder_research"
            and all(manifest.get(key) is False for key in
                    ("release_authorized", "publication_authorized", "production_ready")),
            "this staging version cannot authorize publication")
    expected = RUNTIME_FILES | {"LICENSES/ettin-MIT.txt", "LICENSES/scamguard-Apache-2.0.txt",
                                "DATA_NOTICE.md", "README.md"}
    require(set(manifest.get("files", {})) == expected, "unexpected package manifest members")
    found = set()
    for path in package.rglob("*"):
        require(not path.is_symlink(), "package symlinks are forbidden")
        if path.is_file():
            found.add(path.relative_to(package).as_posix())
        else:
            require(path == package / "LICENSES", "unexpected package directory")
    require(found == expected | {MANIFEST}, "unexpected or missing package files")
    for name, metadata in manifest["files"].items():
        path = safe_path(package, name)
        require(file_sha256(path) == metadata["sha256"]
                and path.stat().st_size == metadata["bytes"], "staged file hash/size mismatch")
    require(file_sha256(package / "LICENSES/ettin-MIT.txt") == BASE_LICENSE_SHA256,
            "wrong staged Ettin MIT license")
    require(WARNING in (package / "README.md").read_text(), "missing BLOCKED model-card warning")
    require(bool(manifest.get("blockers")), "blocked package must expose its blockers")
    check_runtime(package)
    return manifest


def smoke(package: Path, device: str, root: Path) -> dict:
    manifest = verify_package(package)
    for name, expected in manifest["serving_source_sha256"].items():
        require(file_sha256(safe_path(root, name)) == expected, "smoke SDK source identity differs")
    import scamguard.decision
    import scamguard.model
    import scamguard.preprocessing
    import scamguard.scanner
    import scamguard.signals
    from scamguard.model import TransformersBackend
    from scamguard.scanner import Scanner

    for module in (scamguard.model, scamguard.scanner, scamguard.signals,
                   scamguard.preprocessing, scamguard.decision):
        source = f"src/{module.__name__.replace('.', '/')}.py"
        require(file_sha256(Path(module.__file__)) == manifest["serving_source_sha256"][source],
                "imported SDK source differs from the frozen runtime")

    fixtures = ("Your appointment is confirmed for Tuesday at 10 AM.",
                "Send me your banking password and verification code immediately.",
                "Customer: I did not request this.\n"
                "Agent: Contact your bank using its official number.")
    backend = TransformersBackend(package, device=device)
    with Scanner(backend=backend) as scanner:
        records = [scanner.scan(text) for text in fixtures]
    require(all(r.model_id == manifest["experiment_id"] and math.isfinite(r.risk)
                and 0 <= r.risk <= 1 for r in records), "local SDK smoke returned invalid output")
    return {"kind": "fixed_synthetic_local_sdk_smoke", "passed": True, "device": device,
            "examples": len(fixtures), "quality_claim": False, "release_authorized": False,
            "package_manifest_sha256": file_sha256(package / MANIFEST),
            "fixtures_sha256": hashlib.sha256(json.dumps(fixtures).encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument("--manifest", required=True)
    staging.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--package", required=True)
    verify.add_argument("--require-release", action="store_true")
    probe = commands.add_parser("smoke")
    probe.add_argument("--package", required=True)
    probe.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.command == "stage":
        result = stage(safe_path(root, args.manifest), root, args.output)
    else:
        package = safe_path(root, args.package, file=False)
        result = (smoke(package, args.device, root) if args.command == "smoke"
                  else verify_package(package))
        if args.command == "verify" and args.require_release:
            raise SystemExit("BLOCKED: this tool never authorizes public release")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
