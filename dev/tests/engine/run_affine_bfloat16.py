"""Independent affine native-byte fixtures for the pure CPU helper."""

import argparse
import random
import struct
import subprocess
import tempfile
from pathlib import Path

from run_draft_preparation import bf16, quantized_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    rng = random.Random(20102026)
    groups = [
        [0.0] * 64,
        [-0.0] * 64,
        [0.75] * 64,
        [-0.75] * 64,
        [2.5, 8.5, *map(float, range(16)), *([0.0] * 46)],
        [0.0, 127.0, 128.0, 255.0] * 16,
        [-128.0, -1.0, 0.0, 127.0] * 16,
        [2**-30, -(2**-30)] * 32,
        [-(2**-20), -2**-30] * 32,
        [float(i % 32) / 8 - 3 for i in range(64)],
    ]
    # Independent round-to-BF16 source values. Exercise several source rows,
    # odd group counts, metadata rounding and both choices of range edge.
    for scale in (2**-20, 0.01, 0.5, 16, 1024):
        for _ in range(257):
            raw = [rng.uniform(-3, 2) * scale for _ in range(64)]
            groups.append([struct.unpack("<f", struct.pack("<I", bf16(w) << 16))[0] for w in raw])
    with tempfile.TemporaryDirectory(prefix="splash-affine-cpu-") as directory:
        root = Path(directory)
        inputs = root / "input.bf16"
        inputs.write_bytes(b"".join(struct.pack("<64H", *map(bf16, group)) for group in groups))
        expected = []
        for bits in (4, 8):
            path = root / f"expected-af{bits}g64.bin"
            path.write_bytes(b"".join(quantized_group(group, bits) for group in groups))
            expected.append(path)
        subprocess.run([str(args.binary.resolve()), "--fixtures", str(inputs), *map(str, expected)], check=True)


if __name__ == "__main__":
    main()
