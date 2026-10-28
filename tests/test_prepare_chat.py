"""Behavior checks using synthetic messages and disposable local directories."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_chat.py"
SPEC = importlib.util.spec_from_file_location("prepare_chat", SCRIPT)
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


class PrepareChatTests(unittest.TestCase):
    def test_style_uses_only_self_and_context_preserves_order(self):
        records = [
            {"sender": "me", "text": "好呀~", "time": "later"},
            {"sender": "them", "text": "This is a deliberately much longer peer message!!!", "time": "earlier"},
            {"sender": "me", "text": "哈哈[笑]"},
        ]
        result = chat.prepare(records, "me")
        stats = result["style-stats.json"]
        self.assertEqual(stats["message_count"], 2)
        self.assertEqual(stats["mean_length"], 4)
        self.assertEqual(stats["median_length"], 4)
        self.assertEqual(stats["top_terminal_punctuation"], [{"value": "~", "count": 1}])
        self.assertEqual(stats["top_bracket_emoticons"], [{"value": "[笑]", "count": 1}])
        messages = result["context.json"]["messages"]
        self.assertEqual([m["sender"] for m in messages], ["self", "peer", "self"])
        self.assertEqual(messages[0]["time"], "later")
        self.assertNotIn("me", [m["sender"] for m in messages])
        self.assertEqual(len(result["self-samples.json"]["messages"]), 2)

    def test_invalid_fields_rejected_before_filtering(self):
        invalid = [
            None,
            {"sender": 1, "text": "x"},
            {"sender": "someone", "text": 3},
            {"sender": "someone", "text": ""},
            {"sender": "someone", "text": "x", "time": None},
            {"sender": "someone", "text": "x", "type": False},
            {"sender": "someone", "text": "x", "type": ""},
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(chat.ChatError):
                chat.prepare([{"sender": "me", "text": "ok"}, record], "me", "them")

    def test_multiple_peers_require_selection_and_filter(self):
        records = [
            {"sender": "me", "text": "ok"},
            {"sender": "A", "text": "A says hello"},
            {"sender": "B", "text": "B says hello"},
        ]
        with self.assertRaises(chat.ChatError):
            chat.prepare(records, "me")
        result = chat.prepare(records, "me", "A")["context.json"]
        self.assertEqual(result["counts"]["filtered_other_sender"], 1)
        self.assertEqual([m["text"] for m in result["messages"]], ["ok", "A says hello"])

    def test_missing_self_peer_or_same_identity_rejected(self):
        records = [{"sender": "me", "text": "ok"}]
        for self_name, peer_name in [("missing", None), ("me", "missing"), ("me", "me"), ("me", "")]:
            with self.subTest(self_name=self_name, peer_name=peer_name), self.assertRaises(chat.ChatError):
                chat.prepare(records, self_name, peer_name)
        with self.assertRaises(chat.ChatError):
            chat.prepare([], "me")
        with self.assertRaises(chat.ChatError):
            chat.prepare([{"sender": "me", "text": "", "type": "image"}], "me")

    def test_nontext_forward_quote_meta_are_not_samples(self):
        records = [{"sender": "me", "text": "ok"}]
        records += [{"sender": "me", "text": "foreign voice!!!", "type": kind} for kind in ("image", "forward", "quote", "meta")]
        records += [{"sender": "them", "text": "hello"}]
        result = chat.prepare(records, "me")
        self.assertEqual(result["context.json"]["counts"]["skipped_non_text"], 4)
        self.assertEqual(result["style-stats.json"]["message_count"], 1)
        self.assertEqual(result["style-stats.json"]["mean_length"], 2)
        self.assertEqual(len(result["self-samples.json"]["messages"]), 1)

    def test_context_and_samples_are_bounded_but_stats_use_full_input(self):
        records = []
        for index in range(100):
            records += [{"sender": "me", "text": f"self {index}"}, {"sender": "them", "text": f"peer {index}"}]
        result = chat.prepare(records, "me")
        self.assertEqual(len(result["context.json"]["messages"]), 80)
        self.assertEqual(result["context.json"]["messages"][0]["source_index"], 121)
        self.assertEqual(len(result["self-samples.json"]["messages"]), 40)
        self.assertEqual(result["self-samples.json"]["messages"][0]["source_index"], 121)
        self.assertEqual(result["style-stats.json"]["message_count"], 100)

    def test_txt_first_colon_and_chinese_colon(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("me: see https://example.test/a:b\n\nthem：可以\n", encoding="utf-8")
            records = chat.load_records(path)
            self.assertEqual(records[0]["text"], "see https://example.test/a:b")
            self.assertEqual(records[1]["sender"], "them")
            path.write_text("me: first line\ncontinuation without speaker\n", encoding="utf-8")
            with self.assertRaises(chat.ChatError):
                chat.load_records(path)

    def test_json_csv_and_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_text(json.dumps({"messages": [{"sender": "me", "text": "line1\nline2"}]}), encoding="utf-8-sig")
            self.assertEqual(chat.load_records(root / "input.json")[0]["text"], "line1\nline2")
            (root / "input.csv").write_text('sender,text,time,type\nme,"line1\nline2",,\n', encoding="utf-8-sig")
            rows = chat.load_records(root / "input.csv")
            self.assertEqual(chat.prepare(rows, "me")["style-stats.json"]["message_count"], 1)
            for invalid_csv in ("sender,text\nme,x,extra\n", "sender,text\nme\n", "sender,sender,text\nme,me,x\n"):
                (root / "input.csv").write_text(invalid_csv, encoding="utf-8")
                with self.subTest(invalid_csv=invalid_csv), self.assertRaises(chat.ChatError):
                    chat.load_records(root / "input.csv")

    def test_output_existing_file_blocks_all_writes_and_input_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            original = '[{"sender":"me","text":"private synthetic content"}]'
            source.write_text(original, encoding="utf-8")
            payloads = chat.prepare(chat.load_records(source), "me")
            destination = root / "out"
            destination.mkdir()
            existing = destination / "style-stats.json"
            existing.write_text("keep this", encoding="utf-8")
            with self.assertRaises(chat.ChatError):
                chat.write_outputs(payloads, source, destination)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep this")
            self.assertFalse((destination / "context.json").exists())
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            chat.write_outputs(payloads, source, root / "fresh")
            self.assertEqual(len(list((root / "fresh").glob("*.json"))), 3)
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_output_cannot_replace_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "context.json"
            source.write_text("original", encoding="utf-8")
            payloads = chat.prepare([{"sender": "me", "text": "ok"}], "me")
            with self.assertRaises(chat.ChatError):
                chat.write_outputs(payloads, source, Path(directory))
            self.assertEqual(source.read_text(encoding="utf-8"), "original")

    def test_cli_error_never_echoes_content_or_sender(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps([{"sender": "Secret Person", "text": 12345}]), encoding="utf-8")
            output, error = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                code = chat.main(["--input", str(source), "--self", "Secret Person", "--output-dir", str(Path(directory) / "out")])
            self.assertEqual(code, 2)
            self.assertNotIn("Secret Person", output.getvalue() + error.getvalue())
            self.assertNotIn("12345", output.getvalue() + error.getvalue())
            self.assertFalse((Path(directory) / "out").exists())


if __name__ == "__main__":
    unittest.main()
