"""
tests/test_add_task_type.py
===========================
Unit tests for scripts/add_task_type.py

Tests cover:
  - _join_meta_path(): all storage formats (str, char-list, token-list)
  - infer_task_type(): single-hop → intersection, multi-hop → chaining
  - _write_with_task_type(): correct JSON output, task_type inserted
  - validate_file(): passes on good data, fails on bad data
  - Full integration: dry_run + real run on temp JSONL files
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the repo root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.add_task_type import (
    _join_meta_path,
    infer_task_type,
    _write_with_task_type,
    validate_file,
    migrate_pair,
)


# ─────────────────────────────────────────────────────────────────────────────
# _join_meta_path
# ─────────────────────────────────────────────────────────────────────────────

class TestJoinMetaPath:
    def test_string_passthrough(self):
        assert _join_meta_path("[AssociatedGene]") == "[AssociatedGene]"

    def test_string_with_spaces(self):
        assert _join_meta_path("[TreatedBy] [HasSymptom]") == "[TreatedBy] [HasSymptom]"

    def test_char_list(self):
        """Handles the bug where meta_path was stored as list of chars."""
        chars = list("[AssociatedGene]")
        assert _join_meta_path(chars) == "[AssociatedGene]"

    def test_char_list_multi_hop(self):
        chars = list("[TreatedBy] [AssociatedGene]")
        assert _join_meta_path(chars) == "[TreatedBy] [AssociatedGene]"

    def test_token_list(self):
        """Handles list of relation tokens (cleaner format)."""
        tokens = ["[IsA]", " ", "[HasA]"]
        result = _join_meta_path(tokens)
        assert "[IsA]" in result
        assert "[HasA]" in result

    def test_empty_string(self):
        assert _join_meta_path("") == ""

    def test_empty_list(self):
        assert _join_meta_path([]) == ""

    def test_strips_whitespace(self):
        assert _join_meta_path("  [RelatedTo]  ") == "[RelatedTo]"


# ─────────────────────────────────────────────────────────────────────────────
# infer_task_type
# ─────────────────────────────────────────────────────────────────────────────

class TestInferTaskType:
    # Single-hop → intersection
    @pytest.mark.parametrize("path", [
        "[AssociatedGene]",
        "[RelatedTo]",
        "[TreatedBy]",
        "[HasSymptom]",
        "[HasA]",
        "[IsA]",
    ])
    def test_single_hop_is_intersection(self, path):
        assert infer_task_type(path) == "intersection"

    # Multi-hop → chaining
    @pytest.mark.parametrize("path", [
        "[TreatedBy] [AssociatedGene]",
        "[IsA] [HasA]",
        "[IsA] [HasA] [AssociatedGene]",
        "[TreatedBy] [AssociatedGene] [HasSymptom]",
        "[AssociatedGene] [HasSymptom]",
    ])
    def test_multi_hop_is_chaining(self, path):
        assert infer_task_type(path) == "chaining"

    def test_char_list_single_hop(self):
        """Char-list format: single hop → intersection."""
        chars = list("[AssociatedGene]")
        assert infer_task_type(chars) == "intersection"

    def test_char_list_multi_hop(self):
        """Char-list format: multi-hop → chaining."""
        chars = list("[TreatedBy] [HasSymptom]")
        assert infer_task_type(chars) == "chaining"

    def test_empty_path_defaults_to_intersection(self):
        """Empty meta_path: 0 hops → intersection (safe fallback)."""
        assert infer_task_type("") == "intersection"

    def test_missing_path_defaults_to_intersection(self):
        """None input treated as empty → intersection."""
        # infer_task_type handles this via _join_meta_path → str("")
        assert infer_task_type(None) == "intersection"


# ─────────────────────────────────────────────────────────────────────────────
# _write_with_task_type
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteWithTaskType:
    def _make_items(self):
        return [
            {"document": "doc1", "query": "q1", "target_entity": "E1"},
            {"document": "doc2", "query": "q2", "target_entity": "E2"},
        ]

    def test_writes_correct_task_types(self, tmp_path):
        items = self._make_items()
        task_types = ["chaining", "intersection"]
        out = tmp_path / "test.jsonl"
        _write_with_task_type(out, items, task_types)

        written = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert written[0]["task_type"] == "chaining"
        assert written[1]["task_type"] == "intersection"

    def test_preserves_all_original_fields(self, tmp_path):
        items = self._make_items()
        out = tmp_path / "test.jsonl"
        _write_with_task_type(out, items, ["chaining", "chaining"])

        written = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert written[0]["document"] == "doc1"
        assert written[0]["query"] == "q1"
        assert written[0]["target_entity"] == "E1"

    def test_task_type_inserted_after_target_entity(self, tmp_path):
        items = [{"document": "d", "query": "q", "target_entity": "E", "extra": "x"}]
        out = tmp_path / "test.jsonl"
        _write_with_task_type(out, items, ["chaining"])

        written = json.loads(out.read_text().strip())
        keys = list(written.keys())
        te_idx = keys.index("target_entity")
        tt_idx = keys.index("task_type")
        assert tt_idx == te_idx + 1, "task_type should immediately follow target_entity"

    def test_valid_json_per_line(self, tmp_path):
        items = self._make_items()
        out = tmp_path / "test.jsonl"
        _write_with_task_type(out, items, ["chaining", "intersection"])

        for line in out.read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise

    def test_unicode_preserved(self, tmp_path):
        items = [{"target_entity": "α-synuclein", "query": "q", "document": "d"}]
        out = tmp_path / "test.jsonl"
        _write_with_task_type(out, items, ["chaining"])
        written = json.loads(out.read_text().strip())
        assert written["target_entity"] == "α-synuclein"


# ─────────────────────────────────────────────────────────────────────────────
# validate_file
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateFile:
    def _write_jsonl(self, path: Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_valid_file_passes(self, tmp_path):
        items = [
            {"task_type": "chaining", "query": "q1"},
            {"task_type": "intersection", "query": "q2"},
        ]
        p = tmp_path / "good.jsonl"
        self._write_jsonl(p, items)
        assert validate_file(p) is True

    def test_missing_task_type_fails(self, tmp_path, capsys):
        items = [{"query": "q1"}]  # no task_type
        p = tmp_path / "bad.jsonl"
        self._write_jsonl(p, items)
        assert validate_file(p) is False
        captured = capsys.readouterr()
        assert "VALIDATE ERROR" in captured.out

    def test_invalid_task_type_fails(self, tmp_path, capsys):
        items = [{"task_type": "unknown", "query": "q1"}]
        p = tmp_path / "bad.jsonl"
        self._write_jsonl(p, items)
        assert validate_file(p) is False

    def test_nonexistent_file_passes(self, tmp_path):
        """Non-existent file is skipped (not an error)."""
        assert validate_file(tmp_path / "nonexistent.jsonl") is True

    def test_empty_lines_ignored(self, tmp_path):
        p = tmp_path / "with_blanks.jsonl"
        p.write_text('{"task_type": "chaining", "query": "q"}\n\n\n')
        assert validate_file(p) is True


# ─────────────────────────────────────────────────────────────────────────────
# migrate_pair (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestMigratePair:
    def _make_v2_items(self, n: int = 5) -> list[dict]:
        """Make n items with alternating single/multi-hop meta_paths."""
        paths = ["[AssociatedGene]", "[TreatedBy] [HasSymptom]"] * (n // 2 + 1)
        items = []
        for i in range(n):
            items.append({
                "document": f"doc{i}",
                "query": f"query {i}",
                "target_entity": f"Entity{i}",
                "bridge_entity": f"Bridge{i}",
                "hard_negative": f"Neg{i}",
                "meta_path": paths[i],
            })
        return items[:n]

    def _make_v1_items(self, v2_items: list[dict]) -> list[dict]:
        """v1 items: same content but no meta_path, bridge_entity, hard_negative."""
        return [
            {"document": it["document"], "query": it["query"],
             "target_entity": it["target_entity"]}
            for it in v2_items
        ]

    def _write_jsonl(self, path: Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_dry_run_does_not_modify_files(self, tmp_path):
        v2_items = self._make_v2_items(4)
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1.jsonl"
        self._write_jsonl(v2, v2_items)
        self._write_jsonl(v1, self._make_v1_items(v2_items))
        mtime_v1 = v1.stat().st_mtime
        mtime_v2 = v2.stat().st_mtime

        migrate_pair(v1, v2, dry_run=True)

        assert v1.stat().st_mtime == mtime_v1, "v1 should not be modified in dry_run"
        assert v2.stat().st_mtime == mtime_v2, "v2 should not be modified in dry_run"

    def test_real_run_adds_task_type_to_v1(self, tmp_path):
        v2_items = self._make_v2_items(4)
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1.jsonl"
        self._write_jsonl(v2, v2_items)
        self._write_jsonl(v1, self._make_v1_items(v2_items))

        migrate_pair(v1, v2, dry_run=False)

        v1_written = [json.loads(l) for l in v1.read_text().splitlines() if l.strip()]
        assert all("task_type" in it for it in v1_written)

    def test_task_types_correct_values(self, tmp_path):
        v2_items = self._make_v2_items(4)
        # Paths alternate: single, multi, single, multi
        expected = ["intersection", "chaining", "intersection", "chaining"]
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1.jsonl"
        self._write_jsonl(v2, v2_items)
        self._write_jsonl(v1, self._make_v1_items(v2_items))

        migrate_pair(v1, v2, dry_run=False)

        v1_written = [json.loads(l) for l in v1.read_text().splitlines() if l.strip()]
        actual = [it["task_type"] for it in v1_written]
        assert actual == expected

    def test_v2_also_gets_task_type(self, tmp_path):
        v2_items = self._make_v2_items(4)
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1.jsonl"
        self._write_jsonl(v2, v2_items)
        self._write_jsonl(v1, self._make_v1_items(v2_items))

        migrate_pair(v1, v2, dry_run=False)

        v2_written = [json.loads(l) for l in v2.read_text().splitlines() if l.strip()]
        assert all("task_type" in it for it in v2_written)
        # v2 should still have meta_path
        assert all("meta_path" in it for it in v2_written)

    def test_length_mismatch_skips_v1(self, tmp_path, capsys):
        """If v1 and v2 have different lengths, v1 write is skipped safely."""
        v2_items = self._make_v2_items(4)
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1_short.jsonl"
        self._write_jsonl(v2, v2_items)
        # Write only 2 items to v1 (mismatch)
        self._write_jsonl(v1, self._make_v1_items(v2_items)[:2])
        mtime_v1 = v1.stat().st_mtime

        migrate_pair(v1, v2, dry_run=False)

        captured = capsys.readouterr()
        assert "mismatch" in captured.out.lower() or "skip" in captured.out.lower()

    def test_returns_count_summary(self, tmp_path):
        v2_items = self._make_v2_items(4)
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1.jsonl"
        self._write_jsonl(v2, v2_items)
        self._write_jsonl(v1, self._make_v1_items(v2_items))

        counts = migrate_pair(v1, v2, dry_run=False)
        assert counts["total"] == 4
        assert counts["chaining"] + counts["intersection"] == 4

    def test_idempotent_rerun(self, tmp_path):
        """Running migrate_pair twice produces the same result."""
        v2_items = self._make_v2_items(4)
        v2 = tmp_path / "v2.jsonl"
        v1 = tmp_path / "v1.jsonl"
        self._write_jsonl(v2, v2_items)
        self._write_jsonl(v1, self._make_v1_items(v2_items))

        migrate_pair(v1, v2, dry_run=False)
        after_first = v1.read_text()

        migrate_pair(v1, v2, dry_run=False)
        after_second = v1.read_text()

        assert after_first == after_second, "Second run should produce identical output"

    def test_missing_v2_returns_empty(self, tmp_path):
        v1 = tmp_path / "v1.jsonl"
        v2 = tmp_path / "nonexistent_v2.jsonl"
        v1.write_text('{"query": "q", "target_entity": "E", "document": "d"}\n')
        result = migrate_pair(v1, v2, dry_run=False)
        assert result == {}
