"""Content identities for an offline draft-vocabulary generation."""

import hashlib
import importlib.metadata
import json
import shlex
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_manifest(path, tokenizer, corpus, code_files, bitmap, invocation):
    """Pin the bytes supplied to a generation, without requiring mutable URLs."""
    inputs = {
        "tokenizer": {"path": str(tokenizer), "sha256": digest(tokenizer)},
        "corpus": [
            {"file": p.name, "sha256": digest(p)}
            for p in sorted(Path(corpus).iterdir())
            if p.is_file()
        ],
        "code": [{"file": str(p), "sha256": digest(p)} for p in sorted(code_files)],
    }
    record = {
        "schema": 1,
        "invocation": shlex.join(invocation),
        "inputs": inputs,
        "input_identity_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest(),
        "generated_cpp_sha256": digest(bitmap),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pyarrow", "tokenizers")
        },
    }
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
