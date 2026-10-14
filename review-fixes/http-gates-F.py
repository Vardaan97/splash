"""Exclusive-window HTTP ABBA and greedy output identity against PR C."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

# Reuse the repository's pinned model hold and private server lifecycle.
sys.path.insert(0, str(Path.cwd()))
from dev.tests import smoke_real as smoke

parser = argparse.ArgumentParser()
smoke.add_server_arguments(parser)
parser.add_argument("--baseline-binary", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--samples", type=int, default=6)
parser.add_argument("--warmup", type=int, default=900)
parser.add_argument("--order", choices=("C-first", "F-first"), default="C-first")
args = smoke.resolve_server_arguments(parser.parse_args())
if args.warmup < 900:
    parser.error("--warmup must be >=900 seconds")
smoke.hold_model_root(args)
args.out.mkdir(parents=True, exist_ok=True)
source = "\n".join(
    f"def route_{i}(value):\n    # Preserve audit record {i}.\n    return value + {i}"
    for i in range(64)
)
blob = {f"field_{i}": f"preserve this exact audit value {i:04d}" for i in range(24)}
CASES = [
    (
        "agentic-edit",
        "Change only route_17 to return value - 17. Return the entire file verbatim, without fences.\n"
        + source,
        {},
    ),
    (
        "tool-call",
        "Call write_record with this exact object, preserving every key and value: "
        + json.dumps(blob),
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_record",
                        "description": "Write the supplied audit object",
                        "parameters": {
                            "type": "object",
                            "properties": {k: {"type": "string"} for k in blob},
                            "required": list(blob),
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "write_record"}},
        },
    ),
    (
        "code",
        "Write a Python module implementing an immutable binary search tree, insertion, deletion, iteration, and examples. Include code only.",
        {},
    ),
    (
        "prose",
        "Explain how coastal erosion shapes a town over fifty years in clear prose, with concrete examples. Write at least 900 words.",
        {},
    ),
]
records = []
identities = {}


def request(server, case, sample, width):
    category, prompt, extra = case
    # Fixed within a paired sample, distinct between samples; never arm-specific.
    body = smoke.chat_body(
        args.model,
        f"Audit case {sample}.\n" + prompt,
        max_completion_tokens=768,
        **extra,
    )
    started = time.monotonic()
    code, response = smoke.request(
        server.port, "POST", "/v1/chat/completions", body, timeout=1800
    )
    smoke.require(code == 200, f"HTTP {code}: {response}")
    message = response["choices"][0]["message"]
    generated = json.loads(json.dumps(message))
    # Tool call IDs are transport IDs derived from each HTTP request ID.
    for call in generated.get("tool_calls", []):
        call.pop("id", None)
    canonical = json.dumps(generated, sort_keys=True, ensure_ascii=False)
    key = (category, sample, width)
    identity = (
        canonical,
        response["usage"]["completion_tokens"],
        response["choices"][0]["finish_reason"],
    )
    identical = key not in identities or identities[key] == identity
    if key in identities and width == 1:
        smoke.require(identical, f"greedy output changed: {key}")
    if key not in identities:
        identities[key] = identity
    return {
        "identity_matches_first_arm": identical,
        "identity_informational": width == 4,
        "category": category,
        "sample": sample,
        "width": width,
        "output": message,
        "output_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "usage": response["usage"],
        "metrics": response["metrics"],
        "wall_ms": (time.monotonic() - started) * 1000,
    }


try:
    versions = (
        ("C", "F", "F", "C", "F-off")
        if args.order == "C-first"
        else ("F", "C", "C", "F", "F-off")
    )
    for arm, version in enumerate(versions):
        os.environ["SPLASH_MTP_LOOKUP"] = "1" if version == "F" else "0"
        os.environ["SPLASH_MTP_LOG"] = "1"
        os.environ["SPLASH_MTP_ADAPTIVE"] = "1"
        os.environ["SPLASH_MTP_DEPTH"] = "cost"
        os.environ["SPLASH_MTP_DRAFTS"] = "7"
        run_args = argparse.Namespace(**vars(args))
        run_args.binary = args.baseline_binary if version == "C" else args.binary
        server = smoke.RealServer(run_args)
        try:
            status = server.wait_ready(args.startup_timeout)
            smoke.validate_status(status, args.kv_format)
            (args.out / f"status-start-{arm}.json").write_text(
                json.dumps(status, indent=2)
            )
            # Discard identical burn-in work in every arm.
            warm_until = time.monotonic() + (args.warmup if arm == 0 else 0)
            warm_round = 0
            while warm_round < 2 or time.monotonic() < warm_until:
                warm_round += 1
                code, _ = smoke.request(
                    server.port,
                    "POST",
                    "/v1/chat/completions",
                    smoke.chat_body(
                        args.model,
                        "Write the integers 1 to 300 one per line.",
                        max_completion_tokens=768,
                    ),
                    timeout=1800,
                )
                smoke.require(code == 200, "burn-in failed")
            for sample in range(args.samples):
                for case in CASES:
                    _, before = smoke.request(server.port, "GET", "/status")
                    row = request(server, case, sample, 1)
                    _, after = smoke.request(server.port, "GET", "/status")
                    row.update(
                        arm=arm,
                        version=version,
                        native_delta={
                            k: after["metrics"][k] - before["metrics"][k]
                            for k in (
                                "decode_cycle_ms",
                                "decode_output_tokens",
                                "drafted_tokens",
                                "accepted_draft_tokens",
                            )
                        },
                    )
                    records.append(row)
                # Concurrent copy/non-copy requests exercise physical lane mixing.
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    rows = list(
                        pool.map(lambda case, server=server, sample=sample: request(server, case, sample, 4), CASES)
                    )
                for row in rows:
                    row.update(arm=arm, version=version)
                    records.append(row)
            _, status = smoke.request(server.port, "GET", "/status")
            (args.out / f"status-end-{arm}.json").write_text(
                json.dumps(status, indent=2)
            )
        finally:
            try:
                server.log.flush()
                server.log.seek(0)
                (args.out / f"server-{arm}.log").write_text(server.log.read())
            finally:
                server.close()
            (args.out / "samples.json").write_text(json.dumps(records, indent=2))
    lookup_lanes = []
    for arm in [i for i, version in enumerate(versions) if version == "F"]:
        logged = re.findall(
            r"lookup lanes (\d+)", (args.out / f"server-{arm}.log").read_text()
        )
        lookup_lanes.append(max(map(int, logged), default=0))
    smoke.require(
        all(lookup_lanes), "F lookup was not exercised in both candidate arms"
    )
    report = {
        "greedy_B1_identity": True,
        "B4_identity_informational": all(
            r["identity_matches_first_arm"] for r in records if r["width"] == 4
        ),
        "order": args.order,
        "lookup_lanes_by_F_arm": lookup_lanes,
        "categories": {},
    }
    for category, _, _ in CASES:
        by_arm = []
        for arm in range(4):
            rows = [
                r
                for r in records
                if r["arm"] == arm and r["category"] == category and r["width"] == 1
            ]
            tokens = sum(r["native_delta"]["decode_output_tokens"] for r in rows)
            ms = sum(r["native_delta"]["decode_cycle_ms"] for r in rows)
            smoke.require(tokens > 0 and ms > 0, f"no decode observations: {category}")
            by_arm.append(1000 * tokens / ms)
        ratio = statistics.mean(
            by_arm[i] for i in range(4) if versions[i] == "F"
        ) / statistics.mean(by_arm[i] for i in range(4) if versions[i] == "C")
        report["categories"][category] = {
            "native_decode_tokens_per_second_by_arm": by_arm,
            "F_over_C": ratio,
        }
        # Copy speed is reported, never assumed. Guard non-copy regressions.
        if category in ("code", "prose"):
            smoke.require(ratio >= 0.97, f"{category} regressed more than 3%")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
except BaseException:
    (args.out / "samples.json").write_text(json.dumps(records, indent=2))
    raise
