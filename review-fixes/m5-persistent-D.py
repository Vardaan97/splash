#!/usr/bin/env python3
"""M5 only: grid, persistent restart, earlier edit, and cold greedy identity."""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import measure_m5 as gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()
    args.repo, args.binary, args.model_root = (
        args.repo.resolve(),
        args.binary.resolve(),
        args.model_root.resolve(),
    )
    args.state_format, args.kv_format = "fp32", "int8"
    sys.path.insert(0, str(args.repo))
    args.output.mkdir(parents=True, exist_ok=False)
    original_args = args.server_arg[:]
    receipts = {}
    with tempfile.TemporaryDirectory(prefix="splash-prD-persistent-") as cache:
        args.server_arg += [
            "--max-cache-disk",
            "16G",
            "--persistent-cache",
            "--cache-dir",
            cache,
        ]
        for name in ("first", "restart", "cold"):
            (args.output / name).mkdir()
        with gate.Server(args, 8192, args.output / "first") as server:
            messages, edited_index, _, _ = gate.fixture(
                server.base, args.model, 30000, 12000
            )
            question = {
                "role": "user",
                "content": "Report the final file number in one sentence.",
            }
            body = gate.body_for(args.model, messages + [question])
            receipts["first"] = gate.run_stream(server.base, body)
            edited = copy.deepcopy(messages)
            edited[edited_index]["content"][0]["content"] = "[Old tool result pruned.]"
            edited_body = gate.body_for(args.model, edited + [question])
            left = gate.prompt_tokens(server.base, body)
            right = gate.prompt_tokens(server.base, edited_body)
            shared = 0
            for a, b in zip(left, right):
                if a != b:
                    break
                shared += 1
        with gate.Server(args, 8192, args.output / "restart") as server:
            receipts["restart_status"] = server.status
            receipts["restart"] = gate.run_stream(server.base, body)
            receipts["edit"] = gate.run_stream(server.base, edited_body)
            receipts["status_after"] = gate.request(server.base, "/status")
        args.server_arg = original_args
        with gate.Server(args, 0, args.output / "cold") as server:
            receipts["cold"] = gate.run_stream(server.base, edited_body)
    report = {
        "restart_identical": gate.same_generation(
            receipts["first"], receipts["restart"]
        ),
        "edit_identical": gate.same_generation(receipts["edit"], receipts["cold"]),
        "restart_cache_engaged": receipts["restart"]["usage"].get(
            "cache_read_input_tokens", 0
        )
        > 0,
        "edit_restore_within_prefix": receipts["edit"]["usage"].get(
            "cache_read_input_tokens", 0
        )
        <= shared,
        "shared_tokens_exact": shared,
        "receipts": receipts,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "edited-request.json").write_text(json.dumps(edited_body, indent=2))
    passed = all(
        report[key]
        for key in (
            "restart_identical",
            "edit_identical",
            "restart_cache_engaged",
            "edit_restore_within_prefix",
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
