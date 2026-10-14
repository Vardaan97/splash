import io
import json
import unittest
from unittest.mock import patch

from dev.benchmarks import gdn_checkpoint_replay as replay


def stream(parts, *, name=None, complete=True, reason="end_turn"):
    block = {"type": "tool_use", "name": name} if name else {"type": "text"}
    events = [{"type": "content_block_start", "index": 0, "content_block": block}]
    for part in parts:
        events.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": part},
            }
        )
    if complete:
        events.extend(
            [
                {"type": "message_delta", "delta": {"stop_reason": reason}},
                {"type": "message_stop"},
            ]
        )
    return io.BytesIO(
        b"".join(b"data: " + json.dumps(e).encode() + b"\n" for e in events)
    )


class ReplayIdentityTests(unittest.TestCase):
    def generate(self, response):
        with patch.object(replay.urllib.request, "urlopen", return_value=response):
            return replay.run_stream("http://127.0.0.1:1", {})

    def test_chunk_boundaries_do_not_change_identity(self):
        left = self.generate(stream(["checked ", "file 79"]))
        right = self.generate(stream(["checked file 79"]))
        self.assertTrue(replay.same_generation(left, right))
        self.assertEqual(left["output_sha256"], right["output_sha256"])

    def test_different_tool_names_and_stop_reasons_fail_identity(self):
        left = self.generate(stream(["{}"], name="read_file"))
        right = self.generate(stream(["{}"], name="write_file"))
        self.assertFalse(replay.same_generation(left, right))
        right = self.generate(stream(["{}"], name="read_file", reason="max_tokens"))
        self.assertFalse(replay.same_generation(left, right))

    def test_truncated_stream_is_not_a_successful_generation(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.generate(stream(["partial"], complete=False))


if __name__ == "__main__":
    unittest.main()
