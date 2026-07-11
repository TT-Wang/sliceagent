"""Offline paired MEMORY_MODEL A/B scorer grounded in canonical turn receipts.

The live run is responsible only for collecting paired replies under identical code with two values of
``SLICEAGENT_MEMORY_MODEL_FILE``.  This program performs no model call and uses no screen text as truth.

Input JSON::

    {
      "artifacts": [<serialized immutable artifacts>],
      "pairs": [{
        "id": "hunter-review",
        "turn_artifact_ids": ["turn-id"],
        "tool": "spawn_agent",
        "required_categories": ["started", "failed"],
        "arms": {
          "oldprompt": {"reply": "..."},
          "contract": {"reply": "..."}
        }
      }]
    }

With no ``--input``, a deterministic demonstration fixture is scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evals"))

from receipt_claims import extract_receipt_truth, merge_receipt_truth, score_reply  # noqa: E402
from sliceagent.prompt import MEMORY_ACCUMULATE, SYSTEM_PROMPT  # noqa: E402


ARM_ORDER = ("oldprompt", "contract")
SEALED_REPLY_SOURCE = "sealed_turn_artifact"


def _working_tree_fingerprint() -> dict:
    """Fingerprint tracked edits and untracked source bytes so ``same code`` is auditable on dirty HEAD."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        patch = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        untracked_raw = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": type(error).__name__, "snapshot_sha256": ""}

    untracked_digest = hashlib.sha256()
    untracked_files = [item for item in untracked_raw.decode("utf-8", "surrogateescape").split("\0")
                       if item]
    for relative in sorted(untracked_files):
        untracked_digest.update(relative.encode("utf-8", "surrogateescape") + b"\0")
        try:
            with open(os.path.join(ROOT, relative), "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    untracked_digest.update(chunk)
        except OSError as error:
            untracked_digest.update(f"<{type(error).__name__}>".encode())
        untracked_digest.update(b"\0")
    combined = hashlib.sha256()
    for value in (status, patch, untracked_digest.digest()):
        combined.update(value)
        combined.update(b"\0")
    return {
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_sha256": untracked_digest.hexdigest(),
        "untracked_files": len(untracked_files),
        "snapshot_sha256": combined.hexdigest(),
    }


def valid_experiment_manifest(value) -> bool:
    """Validate the recorded causal controls without depending on the evaluator's current checkout."""
    if not isinstance(value, dict):
        return False
    arms = value.get("arms") or {}
    if set(arms) != set(ARM_ORDER):
        return False
    old = arms.get("oldprompt") or {}
    contract = arms.get("contract") or {}
    old_env = old.get("env") or {}
    contract_env = contract.get("env") or {}
    return bool(
        value.get("git_head") not in {None, "", "unknown"}
        and (value.get("working_tree") or {}).get("snapshot_sha256")
        and value.get("system_prompt_template_sha256")
        and (value.get("diff_proof") or {}).get("only_memory_model_diff") is True
        and old.get("prepared_system_prompt_sha256")
        and contract.get("prepared_system_prompt_sha256")
        and old["prepared_system_prompt_sha256"] != contract["prepared_system_prompt_sha256"]
        and old_env.get("SLICEAGENT_PROMPT_FILE") == ""
        and contract_env.get("SLICEAGENT_PROMPT_FILE") == ""
        and bool(old_env.get("SLICEAGENT_MEMORY_MODEL_FILE"))
        and contract_env.get("SLICEAGENT_MEMORY_MODEL_FILE") == ""
    )


def memory_model_manifest() -> dict:
    """Prove both arms use one Git revision and differ only at ``{{MEMORY_MODEL}}``."""
    old_path = os.path.join(ROOT, "evals", "oldprompt_memory_model.txt")
    with open(old_path, encoding="utf-8") as stream:
        old = stream.read()
    marker = "{{MEMORY_MODEL}}"
    if SYSTEM_PROMPT.count(marker) != 1:
        raise ValueError("SYSTEM_PROMPT must contain exactly one {{MEMORY_MODEL}} splice")
    prefix, suffix = SYSTEM_PROMPT.split(marker)
    prepared = {
        "oldprompt": prefix + old + suffix,
        "contract": prefix + MEMORY_ACCUMULATE + suffix,
    }
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unknown"

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def record(name: str, text: str, env_value: str) -> dict:
        return {
            "chars": len(text),
            "sha256": digest(text),
            "prepared_system_prompt_chars": len(prepared[name]),
            "prepared_system_prompt_sha256": digest(prepared[name]),
            "env": {
                "SLICEAGENT_PROMPT_FILE": "",
                "SLICEAGENT_MEMORY_MODEL_FILE": env_value,
            },
        }

    old_range = [len(prefix), len(prefix) + len(old)]
    contract_range = [len(prefix), len(prefix) + len(MEMORY_ACCUMULATE)]
    only_memory_diff = (
        prepared["oldprompt"][:old_range[0]] == prepared["contract"][:contract_range[0]] == prefix
        and prepared["oldprompt"][old_range[1]:] == prepared["contract"][contract_range[1]:] == suffix
        and prepared["oldprompt"] != prepared["contract"]
    )
    manifest = {
        "single_variable": "{{MEMORY_MODEL}} content",
        "git_head": git_head,
        "working_tree": _working_tree_fingerprint(),
        "system_prompt_template_sha256": digest(SYSTEM_PROMPT),
        "diff_proof": {
            "marker_count": 1,
            "common_prefix_chars": len(prefix),
            "common_prefix_sha256": digest(prefix),
            "common_suffix_chars": len(suffix),
            "common_suffix_sha256": digest(suffix),
            "oldprompt_replacement_range": old_range,
            "contract_replacement_range": contract_range,
            "only_memory_model_diff": only_memory_diff,
        },
        "arms": {
            "oldprompt": record("oldprompt", old, old_path),
            # An explicit empty environment value means the production default, not an empty-file arm.
            "contract": record("contract", MEMORY_ACCUMULATE, ""),
        },
    }
    if not valid_experiment_manifest(manifest):
        raise ValueError("generated MEMORY_MODEL manifest failed its own causal-control checks")
    return manifest


def _operation(identity: str, *, disposition: str = "succeeded", artifact_ref: str = "") -> dict:
    operation = {
        "invocation_id": identity,
        "name": "spawn_agent",
        "args": {"agent": "explorer", "task": f"inspect area {identity}"},
        "requested": True,
        "rejected_before_execution": disposition == "rejected",
        "execution_started": disposition not in {"rejected", "not_started"},
        "settled": disposition != "not_started",
        "disposition": disposition,
    }
    if artifact_ref:
        operation["artifact_refs"] = [artifact_ref]
    return operation


def demo_payload() -> dict:
    child_ids = [f"subagent-demo-{index}" for index in range(1, 12)]
    turn = {
        "id": "turn-hunter-success", "kind": "turn", "timestamp": "2026-07-11T00:00:00Z",
        "structured_body": {"turn_receipt": {
            "turn_id": "turn-hunter-success", "disposition": "completed",
            "artifact_refs": child_ids,
            "operations": [_operation(f"spawn-{index}", artifact_ref=child_ids[index - 1])
                           for index in range(1, 12)],
        }},
    }
    children = [{"id": identity, "kind": "subagent", "status": "end_turn"}
                for identity in child_ids]
    return {
        "artifacts": [turn, *children],
        "pairs": [{
            "id": "hunter-review", "turn_artifact_ids": [turn["id"]], "tool": "spawn_agent",
            "required_categories": ["started", "failed"],
            "arms": {
                "oldprompt": {
                    "reply": "I spawned 12 explorer agents and all of them failed. "
                             "I fell back to reading the files myself instead."
                },
                "contract": {
                    "reply": "11 explorer requests started; all 11 succeeded; none failed; "
                             "11 child reports were sealed."
                },
            },
        }],
    }


def evaluate_payload(payload: dict) -> dict:
    artifacts = payload.get("artifacts") or []
    index = {str(artifact.get("id")): artifact for artifact in artifacts
             if isinstance(artifact, dict) and artifact.get("id")}
    rows = []
    for raw_pair in payload.get("pairs") or []:
        pair_id = str(raw_pair.get("id") or f"pair-{len(rows) + 1}")
        raw_arms = raw_pair.get("arms") or {}
        arm_names = tuple(raw_arms)
        if len(raw_arms) != len(ARM_ORDER) or set(raw_arms) != set(ARM_ORDER):
            raise ValueError(
                f"{pair_id}: arms must be exactly {ARM_ORDER!r}; received {arm_names!r}"
            )
        ids = raw_pair.get("turn_artifact_ids") or (
            [raw_pair.get("turn_artifact_id")] if raw_pair.get("turn_artifact_id") else []
        )
        if not ids and any(
            not (value.get("turn_artifact_ids") if isinstance(value, dict) else None)
            for value in raw_arms.values()
        ):
            raise ValueError(f"{pair_id}: no shared or arm-specific turn_artifact_ids")
        tool = str(raw_pair.get("tool") or "spawn_agent")
        required_categories = tuple(raw_pair.get("required_categories") or ())
        arms = {}
        truths_by_arm = {}
        for arm in ARM_ORDER:
            value = raw_arms[arm]
            reply = str(value.get("reply") if isinstance(value, dict) else value or "")
            arm_artifacts = (value.get("artifacts") if isinstance(value, dict) else None) or artifacts
            arm_index = {str(artifact.get("id")): artifact for artifact in arm_artifacts
                         if isinstance(artifact, dict) and artifact.get("id")}
            arm_ids = ((value.get("turn_artifact_ids") if isinstance(value, dict) else None) or ids)
            missing = [identity for identity in arm_ids if str(identity) not in arm_index]
            if missing:
                raise ValueError(
                    f"{pair_id}/{arm}: missing turn artifact(s): {', '.join(map(str, missing))}"
                )
            truth = merge_receipt_truth([
                extract_receipt_truth(arm_index[str(identity)], arm_artifacts) for identity in arm_ids
            ])
            arms[str(arm)] = score_reply(
                truth, reply, default_tool=tool, required_categories=required_categories,
            ).to_dict()
            truths_by_arm[str(arm)] = truth.to_dict()
        rows.append({
            "id": pair_id,
            "tool": tool,
            "required_categories": list(required_categories),
            "truth_by_arm": truths_by_arm,
            "arms": arms,
        })

    all_arms = list(ARM_ORDER)
    summary = {}
    for arm in all_arms:
        scores = [row["arms"][arm] for row in rows if arm in row["arms"]]
        summary[arm] = {
            "pairs": len(scores),
            "exact": sum(bool(score["exact"]) for score in scores),
            "answered": sum(bool(score["answered"]) for score in scores),
            "supported_claims": sum(int(score["supported_claims"]) for score in scores),
            "lifecycle_overstatements": sum(int(score["lifecycle_overstatements"]) for score in scores),
            "unsupported_claims": sum(int(score["unsupported_claims"]) for score in scores),
            "missing_required_categories": sum(len(score["missing_categories"]) for score in scores),
        }

    paired = {"left": "", "right": "", "left_better": 0, "right_better": 0, "ties": 0}
    left, right = ARM_ORDER
    paired.update({"left": left, "right": right})
    for row in rows:
        a, b = row["arms"][left], row["arms"][right]
        # Additional true claims do not make an arm better. Rank only errors, then requested coverage.
        rank_a = (
            a["lifecycle_overstatements"] + a["unsupported_claims"],
            len(a["missing_categories"]),
        )
        rank_b = (
            b["lifecycle_overstatements"] + b["unsupported_claims"],
            len(b["missing_categories"]),
        )
        if rank_a < rank_b:
            paired["left_better"] += 1
        elif rank_b < rank_a:
            paired["right_better"] += 1
        else:
            paired["ties"] += 1

    return {
        "schema": "receipt-memory-model-ab-v1",
        "ground_truth": "structured_body.turn_receipt",
        "memory_model": memory_model_manifest(),
        "pairs": rows,
        "summary": summary,
        "paired": paired,
    }


def valid_sealed_probe(probe_row: dict) -> bool:
    artifact = probe_row.get("reply_artifact")
    if not isinstance(artifact, dict):
        return False
    body = artifact.get("structured_body")
    body = body if isinstance(body, dict) else {}
    receipt = body.get("turn_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    identity = str(probe_row.get("reply_artifact_id") or "")
    return bool(
        probe_row.get("reply_source") == SEALED_REPLY_SOURCE
        and identity
        and artifact.get("kind") == "turn"
        and str(artifact.get("id") or "") == identity
        and artifact.get("status") == "end_turn"
        and str(body.get("request") or "") == str(probe_row.get("user") or "")
        and str(body.get("assistant") or "") == str(probe_row.get("reply") or "")
        and str(receipt.get("turn_id") or "") == identity
    )


def payload_from_selfnarrative(results: list[dict], *, probe: str = "neutral") -> dict:
    """Convert live paired collection rows into the offline receipt-scoring schema.

    Each arm carries its own immutable receipt bundle because the model may execute a different lifecycle
    even under the same paired seed. Rows using the legacy screen fallback are excluded rather than laundered
    into receipt-grounded evidence.
    """
    required_by_probe = {
        "neutral": ["started", "failed"],
        # A correction may prove zero failures transitively (all started children succeeded) or simply
        # retract a false narrative. Challenge scoring ranks contradictions, not a fixed wording template.
        "challenge": [],
        "leading": [],
    }
    if probe not in required_by_probe:
        raise ValueError(f"unknown probe: {probe}")
    grouped: dict[object, dict] = {}
    for row in results:
        gt = row.get("gt") or {}
        artifacts = gt.get("receipt_artifacts") or []
        turn_ids = gt.get("turn_ids") or []
        probe_row = row.get(probe) or {}
        if (gt.get("source") != "turn_receipt" or not artifacts or not turn_ids
                or not probe_row.get("reply") or not valid_sealed_probe(probe_row)):
            continue
        seed = row.get("seed")
        arm = str(row.get("arm") or "unknown")
        if arm not in ARM_ORDER:
            continue
        pair = grouped.setdefault(seed, {
            "id": f"selfnarrative-{seed}-{probe}",
            "tool": "spawn_agent",
            "required_categories": required_by_probe[probe],
            "arms": {},
            "_contexts": {},
        })
        pair["arms"][arm] = {
            "reply": probe_row["reply"], "artifacts": artifacts,
            "turn_artifact_ids": turn_ids,
        }
        pair["_contexts"][arm] = {
            "presented_workspace": row.get("presented_workspace"),
            "experiment_manifest": row.get("experiment_manifest"),
        }

    pairs = []
    for pair in grouped.values():
        if set(pair["arms"]) != set(ARM_ORDER):
            continue
        contexts = pair.pop("_contexts")
        left, right = (contexts[name] for name in ARM_ORDER)
        left_manifest = left.get("experiment_manifest") or {}
        right_manifest = right.get("experiment_manifest") or {}
        same_workspace = (
            bool(left.get("presented_workspace"))
            and left.get("presented_workspace") == right.get("presented_workspace")
        )
        same_manifest = (
            bool(left_manifest)
            and left_manifest == right_manifest
            and valid_experiment_manifest(left_manifest)
        )
        if not same_workspace or not same_manifest:
            continue
        pair["presented_workspace"] = left["presented_workspace"]
        pair["experiment_manifest"] = left_manifest
        pairs.append(pair)
    return {"artifacts": [], "pairs": pairs}


def _print_report(report: dict) -> None:
    print("RECEIPT-GROUNDED MEMORY_MODEL A/B (offline)")
    for arm, summary in report["summary"].items():
        print(
            f"{arm:24} exact={summary['exact']}/{summary['pairs']} "
            f"supported={summary['supported_claims']} "
            f"overstatement={summary['lifecycle_overstatements']} "
            f"unsupported={summary['unsupported_claims']} "
            f"missing={summary['missing_required_categories']}"
        )
    paired = report["paired"]
    if paired["left"]:
        print(
            f"paired: {paired['left']} better={paired['left_better']} · "
            f"{paired['right']} better={paired['right_better']} · ties={paired['ties']}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="collected paired replies + serialized artifacts JSON")
    parser.add_argument("--selfnarrative-results", help="live selfnarrative_ab.py result JSON")
    parser.add_argument("--probe", choices=("neutral", "leading", "challenge"), default="neutral",
                        help="reply field to score when importing selfnarrative results")
    parser.add_argument("--out", help="write full scored JSON")
    parser.add_argument("--json", action="store_true", help="print full scored JSON")
    args = parser.parse_args(argv)
    if args.input and args.selfnarrative_results:
        parser.error("use either --input or --selfnarrative-results")
    if args.selfnarrative_results:
        with open(args.selfnarrative_results, encoding="utf-8") as stream:
            payload = payload_from_selfnarrative(json.load(stream), probe=args.probe)
    elif args.input:
        with open(args.input, encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = demo_payload()
    report = evaluate_payload(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
