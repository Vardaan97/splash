"""Expand nested draft sets from the retained offline weighted ranking.

CPU only: python dev/tools/expand_draft_vocabulary.py
The saved top 32K must exactly match the legacy bitmap before larger sets
are written. To change ranking inputs, use draft_vocabulary.py and retain
its input manifest first. Historical revisions of this ranking are unknown.
"""
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    ranking_bytes = gzip.decompress((root / "dev/tools/data/qwen4-draft-ranking.npy.gz").read_bytes())
    ranking = np.load(io.BytesIO(ranking_bytes))
    if ranking.shape != (248320,) or set(ranking.tolist()) != set(range(248320)):
        raise ValueError("ranking is not a complete tokenizer permutation")
    cpp = root / "runtime/model/Qwen4DraftVocabularyData.cpp"
    text = cpp.read_text()
    legacy = re.search(r"kQwen4DraftVocabularyWords\[3880\] = \{(.*?)\};", text, re.S).group(1)
    words = [int(word, 16) for word in re.findall(r"0x([0-9a-f]+)ull", legacy)]
    tokens = {i * 64 + bit for i, word in enumerate(words) for bit in range(64) if word >> bit & 1}
    if tokens != set(ranking[:32768].tolist()):
        raise ValueError("saved top 32K differs from the legacy bitmap")
    for size in (49152, 65536):
        words = [0] * 3880
        for token in ranking[:size]:
            words[int(token) // 64] |= 1 << (int(token) % 64)
        array = f"extern const uint64_t kQwen4DraftVocabularyWords{size}[3880] = {{\n"
        array += "".join("    " + " ".join(f"0x{word:016x}ull," for word in words[start:start+4]) + "\n"
                         for start in range(0, len(words), 4)) + "};"
        text, count = re.subn(rf"extern const uint64_t kQwen4DraftVocabularyWords{size}\[3880\] = \{{.*?\}};", array, text, flags=re.S)
        if count != 1:
            raise ValueError("missing/duplicate generated array")
    cpp.write_text(text)
    source_hash = hashlib.sha256(cpp.read_bytes()).hexdigest()
    manifest = cpp.with_suffix(".manifest.json")
    record = json.loads(manifest.read_text())
    record["generated_cpp_sha256"] = source_hash
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    sizes = cpp.with_name("Qwen4DraftVocabularySizes.manifest.json")
    record = json.loads(sizes.read_text())
    record.update(source_sha256=source_hash, ranking_sha256=hashlib.sha256(ranking_bytes).hexdigest(),
                  exact_expansion_invocation="python dev/tools/expand_draft_vocabulary.py")
    sizes.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
