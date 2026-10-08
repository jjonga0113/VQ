"""Check the launcher against Table 8 without importing GPU dependencies."""

import ast
import math
import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "scripts" / "quantize_2bit.sh"
SOURCE = LAUNCHER.read_text(encoding="utf-8")

# Independent transcription of arXiv:2409.17066v2, Table 8:
# N%, v0, k0, v1, k1, k2, group_num.
TABLE_8 = {
    "llama2-7b-2.02": (0, -1, -1, 6, 4096, -1, 1),
    "llama2-7b-2.26": (1, 4, 8192, 12, 4096, 4096, 4),
    "llama2-13b-2.02": (0, -1, -1, 6, 4096, -1, 1),
    "llama2-13b-2.18": (2, 4, 8192, 12, 4096, 4096, 4),
    "llama2-70b-2.07": (1, 4, 8192, 12, 4096, 4096, 4),
    "llama2-70b-2.11": (1, 4, 8192, 12, 4096, 4096, 8),
    "llama3-8b-2.08": (1, 4, 4096, 12, 4096, 4096, 1),
    "llama3-8b-2.24": (1, 4, 8192, 6, 4096, -1, 16),
    "llama3-70b-2.02": (0, -1, -1, 12, 4096, 4096, 1),
    "llama3-70b-2.07": (1, 4, 4096, 6, 4096, -1, 16),
}


def dataclass_fields(path, name):
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    return {node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)}


class PaperPresetTests(unittest.TestCase):
    def test_all_table_8_rows(self):
        rows = re.findall(r"^\s+(llama[23]-\w+-[\d.]+)\)\s+PARAMS=\(([-\d ]+)\)", SOURCE, re.M)
        self.assertEqual(len(rows), 10)
        actual = {name: tuple(map(int, values.split())) for name, values in rows}
        self.assertEqual(actual, TABLE_8)

    def test_all_runner_flags_exist(self):
        command = SOURCE.split("COMMAND=(", 1)[1].split("\n)", 1)[0]
        used = set(re.findall(r"--([a-z][a-z_]*)", command)) | {"save_qlinear"}
        accepted = dataclass_fields(REPO / "run_vptq.py", "VPTQArguments")
        accepted |= dataclass_fields(REPO / "vptq" / "quantizer.py", "QuantizationArguments")
        self.assertFalse(used - accepted, f"Unknown runner arguments: {used - accepted}")

    def test_index_budget_leaves_room_for_codebooks(self):
        for preset, (percent, v0, k0, v1, k1, k2, _) in TABLE_8.items():
            with self.subTest(preset=preset):
                normal_bits = math.log2(k1) / v1
                if k2 > 0:
                    normal_bits += math.log2(k2) / v1
                self.assertEqual(normal_bits, 2)
                index_bits = (1 - percent / 100) * normal_bits
                if percent:
                    index_bits += percent / 100 * math.log2(k0) / v0
                self.assertLess(index_bits, float(preset.rsplit("-", 1)[1]))

    def test_shell_file_encoding(self):
        raw = LAUNCHER.read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", raw, "Bash launcher must use LF line endings")
        self.assertTrue(raw.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
