#!/usr/bin/env python3
"""Opt-in M5 experiment; starts fresh Splash servers and writes replay receipts.

Run only on the measurement host. Uses an installed model and built binary;
never installs weights or builds. All output, including identity failures, is
written before a nonzero exit. Server restarts prevent cross-case cache reuse.
"""

import argparse
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers=headers())
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def headers():
    result = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key := os.environ.get("SPLASH_API_KEY"):
        result.update({"x-api-key": key, "Authorization": f"Bearer {key}"})
    return result


def run_stream(base, body):
    body = {**body, "stream": True}
    req = urllib.request.Request(
        base + "/v1/messages", data=json.dumps(body).encode(), headers=headers()
    )
    started = time.monotonic()
    first = None
    usage = {}
    blocks = {}
    stopped = False
    stop_reason = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind == "error":
                raise RuntimeError(event)
            if kind == "message_start":
                usage.update(event["message"].get("usage", {}))
            elif kind == "content_block_start":
                block = event["content_block"]
                blocks[event["index"]] = {
                    "type": block["type"],
                    "name": block.get("name"),
                    "text": block.get("text", block.get("thinking", "")),
                }
                if block.get("input"):
                    blocks[event["index"]]["text"] = json.dumps(
                        block["input"], sort_keys=True
                    )
                if text := block.get("text", block.get("thinking", "")):
                    first = first or time.monotonic()
            elif kind == "content_block_delta":
                delta = event["delta"]
                text = delta.get(
                    "text", delta.get("thinking", delta.get("partial_json", ""))
                )
                if text:
                    first = first or time.monotonic()
                    # Keep semantic output streams, ignoring SSE chunk boundaries.
                    blocks[event["index"]]["text"] += text
            elif kind == "message_delta":
                usage.update(event.get("usage", {}))
                stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
            elif kind == "message_stop":
                stopped = True
    if not stopped or stop_reason is None:
        raise RuntimeError("Incomplete generation stream")
    if first is None:
        raise RuntimeError("No generated content arrived")
    streams = [blocks[index] for index in sorted(blocks)]
    encoded = json.dumps(streams, sort_keys=True, ensure_ascii=False).encode()
    return {
        "ttft_s": first - started,
        "wall_s": time.monotonic() - started,
        "usage": usage,
        "output": streams,
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "stop_reason": stop_reason,
    }


def body_for(model, messages):
    return {
        "model": model,
        "system": "You are a coding assistant reading tool results from one session.",
        "messages": messages,
        "tools": [
            {
                "name": "read_file",
                "description": "Read a repository file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
    }


def same_generation(left, right):
    return (left["output"], left["stop_reason"]) == (
        right["output"],
        right["stop_reason"],
    )


def build_receipt(repo, binary):
    with binary.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "repo": str(repo),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "binary": str(binary),
        "binary_sha256": digest,
    }


def tool_turn(index, lines):
    call = f"read_{index}"
    data = "".join(
        f"file={index} line={row} status=checked value={(row * 17 + index) % 997}\n"
        for row in range(lines)
    )
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Inspect the next file."},
                {
                    "type": "tool_use",
                    "id": call,
                    "name": "read_file",
                    "input": {"path": f"src/file_{index}.txt"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call, "content": data}],
        },
    ]


def count(base, body):
    return request(base, "/v1/messages/count_tokens", body)["input_tokens"]


def fixture(base, model, target, edit_depth):
    messages = [
        {
            "role": "user",
            "content": "Inspect the repository file by file and track checked values.",
        }
    ]
    lines = 120
    for index in range(80):
        messages.extend(tool_turn(index, lines))
    # Scale payload volume using actual model token counts, then record the
    # final count; the exact target is allowed to vary by one tool-result size.
    initial = count(base, body_for(model, messages))
    lines = max(1, round(lines * target / initial))
    messages = messages[:1]
    for index in range(80):
        messages.extend(tool_turn(index, lines))
    before_edit = []
    chosen = None
    closest = float("inf")
    for index in range(80):
        result_index = 2 + 2 * index
        prefix = messages[:result_index]
        depth = count(base, body_for(model, prefix))
        if abs(depth - edit_depth) < closest:
            closest = abs(depth - edit_depth)
            chosen, before_edit = result_index, [depth, index]
    return messages, chosen, before_edit, count(base, body_for(model, messages))


def prompt_tokens(base, body):
    from server.api_shapes import anthropic_to_chat_prompt

    chat = anthropic_to_chat_prompt(body, thinking_resolver=lambda value: value)
    text = request(base, "/apply-template", chat)["prompt"]
    return request(base, "/tokenize", {"content": text})["tokens"]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, args, grid, output, *, baseline=False):
        self.args, self.grid, self.output = args, grid, output
        self.repo = args.baseline_repo if baseline else args.repo
        self.binary = args.baseline_binary if baseline else args.binary

    def __enter__(self):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        command = [
            str(self.repo / ".venv/bin/python"),
            "-u",
            "-P",
            "-m",
            "server.server",
            str(self.args.model_root),
            "--tokenizer",
            str(self.args.model_root / "tokenizer"),
            "--model",
            self.args.model,
            "--binary",
            str(self.binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--max-context",
            "262144",
            "--max-cache-disk",
            "0",
            "--state-format",
            self.args.state_format,
            "--kv-format",
            self.args.kv_format,
            "--disable-ane",
            "--request-timeout",
            "1800",
            *self.args.server_arg,
        ]
        environment = dict(
            os.environ,
            PYTHONPATH=str(self.repo),
            SPLASH_GDN_CHECKPOINT_TOKENS=str(self.grid),
        )
        self.log = open(self.output / "server.log", "wb")
        self.process = subprocess.Popen(
            command,
            cwd=self.repo,
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        try:
            until = time.monotonic() + 600
            while time.monotonic() < until:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited: see {self.output / 'server.log'}"
                    )
                try:
                    request(self.base, "/health")
                    self.status = request(self.base, "/status")
                    return self
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            raise RuntimeError("Server readiness timeout")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--state-format", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--kv-format", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model", default="Splash-128")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=170000)
    parser.add_argument("--edit-depth", type=int, default=105000)
    parser.add_argument("--grid", type=int, choices=(8192, 16384), default=8192)
    parser.add_argument(
        "--warmup",
        type=int,
        default=900,
        help="Discarded burn-in seconds (minimum 900)",
    )
    parser.add_argument(
        "--informational-identity",
        action="store_true",
        help="Report class-N identity without gating it",
    )
    parser.add_argument("--order", choices=("off-on", "on-off"), default="off-on")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra server argument, e.g. --server-arg=--ple-ssd-cache",
    )
    args = parser.parse_args()
    if not 0 < args.edit_depth < args.target_tokens < 240000:
        parser.error("require 0 < edit-depth < target-tokens < 240000")
    args.repo, args.model_root = args.repo.resolve(), args.model_root.resolve()
    args.binary = (args.binary or args.repo / "build/splash").resolve()
    given_baseline_binary = args.baseline_binary
    args.baseline_repo = (args.baseline_repo or args.repo).resolve()
    args.baseline_binary = (
        args.baseline_binary or args.baseline_repo / "build/splash"
    ).resolve()
    if args.baseline_repo == args.repo and not given_baseline_binary:
        args.baseline_binary = args.binary
    sys.path.insert(0, str(args.repo))
    if args.warmup < 900:
        parser.error("--warmup must be at least 900 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "builds.json").write_text(
        json.dumps(
            {
                "candidate": build_receipt(args.repo, args.binary),
                "baseline": build_receipt(args.baseline_repo, args.baseline_binary),
                "state_format": args.state_format,
                "kv_format": args.kv_format,
            },
            indent=2,
        )
    )
    results = {}
    saved = None
    cases = [("off", 0), ("on", args.grid)]
    if args.order == "on-off":
        cases.reverse()
    cases.append(("candidate-off", 0))
    for label, grid in cases:
        out = args.output / label
        out.mkdir()
        with Server(args, grid, out, baseline=label == "off") as server:
            if not results:
                until = time.monotonic() + args.warmup
                rounds = 0
                while time.monotonic() < until:
                    run_stream(
                        server.base,
                        body_for(
                            args.model,
                            [
                                {
                                    "role": "user",
                                    "content": f"Burn-in {rounds}: write integers 1 to 300.",
                                }
                            ],
                        ),
                    )
                    rounds += 1
                (out / "burn-in.json").write_text(
                    json.dumps({"seconds": args.warmup, "rounds": rounds})
                )
            if saved is None:
                saved = fixture(
                    server.base, args.model, args.target_tokens, args.edit_depth
                )
            messages, edited_index, depth, tokens = copy.deepcopy(saved)
            warm = []
            question = {
                "role": "user",
                "content": "Report the final file number and its checked status in one sentence.",
            }
            for turn in range(5):
                if turn:
                    messages.extend(tool_turn(80 + turn, 35))
                warm.append(
                    run_stream(server.base, body_for(args.model, messages + [question]))
                )
            edited = copy.deepcopy(messages)
            edited[edited_index]["content"][0]["content"] = (
                "[Old tool result pruned by the agent harness.]"
            )
            edited_body = body_for(args.model, edited + [question])
            (out / "edited-request.json").write_text(json.dumps(edited_body))
            original_body = body_for(args.model, messages + [question])
            (out / "original-request.json").write_text(json.dumps(original_body))
            original_tokens = prompt_tokens(server.base, original_body)
            edited_tokens = prompt_tokens(server.base, edited_body)
            shared = 0
            for left, right in zip(original_tokens, edited_tokens):
                if left != right:
                    break
                shared += 1
            before = request(server.base, "/status")
            replay = run_stream(server.base, edited_body)
            after = request(server.base, "/status")
            results[label] = {
                "grid": grid,
                "initial_tokens": tokens,
                "edit_result_index": edited_index,
                "shared_prefix_count_with_generation_marker": depth[0],
                "shared_tokens_exact": shared,
                "ideal_grid_restore": shared // args.grid * args.grid,
                "edited_file": depth[1],
                "edited_tokens": count(server.base, edited_body),
                "warm": warm,
                "replay": replay,
                "status_start": server.status,
                "status_before": before,
                "status_after": after,
            }
            (out / "receipt.json").write_text(json.dumps(results[label], indent=2))
    # Fresh process: the edited reference has no old-prefix cache at all.
    out = args.output / "cold-edited"
    out.mkdir()
    with Server(args, 0, out) as server:
        cold = run_stream(server.base, edited_body)
        (out / "receipt.json").write_text(json.dumps(cold, indent=2))
    identical = same_generation(results["off"]["replay"], results["on"]["replay"])
    cold_identical = same_generation(results["on"]["replay"], cold)
    warm_identical = all(
        same_generation(left, right)
        for left, right in zip(
            results["off"]["warm"], results["on"]["warm"], strict=True
        )
    )
    candidate_off_identical = same_generation(
        results["off"]["replay"], results["candidate-off"]["replay"]
    ) and all(
        same_generation(left, right)
        for left, right in zip(
            results["off"]["warm"], results["candidate-off"]["warm"], strict=True
        )
    )
    replay_rows = {
        label: data["status_after"]["metrics"]["prefill_input_tokens"]
        - data["status_before"]["metrics"]["prefill_input_tokens"]
        for label, data in results.items()
    }
    engaged = 0 < replay_rows["on"] < replay_rows["off"]
    report = {
        "identity_class": "N" if args.state_format == "bf16" else "bitwise",
        "identity_informational": args.informational_identity,
        "candidate_off_outputs_identical": candidate_off_identical,
        "output_identical": identical,
        "cold_edited_output_identical": cold_identical,
        "warm_outputs_identical": warm_identical,
        "cold_edited": cold,
        "checkpoints_saved_prefill": engaged,
        "edited_prefill_rows": replay_rows,
        "off_ttft_s": results["off"]["replay"]["ttft_s"],
        "on_ttft_s": results["on"]["replay"]["ttft_s"],
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"}, indent=2
        )
    )
    raise SystemExit(
        0
        if engaged
        and candidate_off_identical
        and (
            args.informational_identity
            or (identical and cold_identical and warm_identical)
        )
        else 1
    )


if __name__ == "__main__":
    main()
