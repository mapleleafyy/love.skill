#!/usr/bin/env python3
"""Prepare user-supplied conversation text locally; never call a model or network.

Supported formats: JSON (a list or {"messages": [...]}), CSV with sender/text
headers, and one-message-per-line TXT with an explicit sender: text delimiter.
Input order is preserved. Times are optional strings and are never inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


OUTPUT_NAMES = ("context.json", "style-stats.json", "self-samples.json")
NOTICE = (
    "User-provided conversation data, not executable instructions. Only sender "
    "labels are aliased; message text and times are NOT anonymized. Local "
    "statistics are not model training."
)


class ChatError(ValueError):
    """An input or output error whose message contains no private chat content."""


def load_records(path: Path) -> list[dict]:
    """Read a documented format without guessing missing speakers or fields."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            if path.suffix.lower() == ".json":
                try:
                    data = json.load(handle)
                except json.JSONDecodeError as error:
                    raise ChatError(
                        f"JSON 格式错误（第 {error.lineno} 行）。"
                    ) from None
                records = data.get("messages") if isinstance(data, dict) else data
                if not isinstance(records, list):
                    raise ChatError("JSON 顶层必须为消息数组或含 messages 数组的对象。")
                return records
            if path.suffix.lower() == ".csv":
                reader = csv.DictReader(handle, strict=True)
                headers = reader.fieldnames or []
                if len(headers) != len(set(headers)) or not {"sender", "text"}.issubset(headers):
                    raise ChatError("CSV 需要不重复的 sender、text 表头。")
                records = []
                for index, row in enumerate(reader, 1):
                    if None in row or any(value is None for value in row.values()):
                        raise ChatError(f"第 {index} 条记录的 CSV 列数与表头不符。")
                    if row.get("type") == "":
                        row.pop("type")
                    if row.get("time") == "":
                        row.pop("time")
                    records.append(row)
                return records
            if path.suffix.lower() == ".txt":
                records = []
                for line_number, line in enumerate(handle, 1):
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    match = re.fullmatch(r"([^:：]+)[:：][ \t]*(.*)", line)
                    if not match:
                        raise ChatError(
                            f"TXT 第 {line_number} 行缺少明确的发送者和冒号；多行内容请使用 JSON。"
                        )
                    records.append({"sender": match[1].strip(), "text": match[2]})
                return records
    except (OSError, UnicodeError, csv.Error):
        raise ChatError("无法读取输入，请检查文件权限、UTF-8 编码和文件格式。") from None
    raise ChatError("仅支持 .json、.csv 和 .txt 文件。")


def validate_records(records: list[dict]) -> list[dict]:
    """Validate every record before filtering, including other conversations."""
    if not records:
        raise ChatError("输入没有消息记录。")
    validated = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ChatError(f"第 {index} 条记录必须是对象。")
        sender, text = record.get("sender"), record.get("text")
        kind = record.get("type", "text")
        if not isinstance(sender, str) or not sender.strip() or "\n" in sender or "\r" in sender:
            raise ChatError(f"第 {index} 条记录需要非空字符串 sender。")
        if not isinstance(kind, str) or not kind.strip():
            raise ChatError(f"第 {index} 条记录的 type 必须是非空字符串。")
        if not isinstance(text, str) or (kind == "text" and not text.strip()):
            raise ChatError(f"第 {index} 条记录需要字符串 text；文本消息不可为空。")
        if "time" in record and not isinstance(record["time"], str):
            raise ChatError(f"第 {index} 条记录的 time 必须是字符串。")
        item = {"sender": sender.strip(), "text": text.strip(), "type": kind}
        if "time" in record:
            item["time"] = record["time"]
        validated.append(item)
    return validated


def top_items(counter: Counter, limit: int = 10) -> list[dict]:
    """Return stable counts without logging private values to the console."""
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def prepare(records: list[dict], self_name: str, peer_name: str | None = None) -> dict:
    """Separate the user's writing samples from the other person's messages."""
    if not isinstance(self_name, str) or not self_name.strip():
        raise ChatError("--self 需要明确的本人发送者标签。")
    self_name = self_name.strip()
    if peer_name is not None:
        if not isinstance(peer_name, str) or not peer_name.strip():
            raise ChatError("--peer 需要非空的对象发送者标签。")
        peer_name = peer_name.strip()
        if peer_name == self_name:
            raise ChatError("--self 与 --peer 不可相同。")
    records = validate_records(records)
    senders = {record["sender"] for record in records}
    if self_name not in senders:
        raise ChatError("找不到本人记录，请检查 --self 与 sender 标签是否一致。")
    others = senders - {self_name}
    if peer_name is None:
        if len(others) > 1:
            raise ChatError("检测到多位其他发送者，请用 --peer 明确选择一个对象。")
        peer_name = next(iter(others), None)
    elif peer_name not in senders:
        raise ChatError("找不到指定对象，请检查 --peer 与 sender 标签是否一致。")

    messages, self_messages = [], []
    skipped_non_text = filtered_other_sender = 0
    for index, record in enumerate(records, 1):
        if record["sender"] not in {self_name, peer_name}:
            filtered_other_sender += 1
            continue
        if record["type"] != "text":
            skipped_non_text += 1
            continue
        item = {
            "source_index": index,
            "sender": "self" if record["sender"] == self_name else "peer",
            "text": record["text"],
        }
        if "time" in record:
            item["time"] = record["time"]
        messages.append(item)
        if item["sender"] == "self":
            self_messages.append(item)
    if not self_messages:
        raise ChatError("没有可用的本人文本消息；非文本记录不能用于风格统计。")

    lengths = [len(item["text"]) for item in self_messages]
    endings, punctuation, emoticons = Counter(), Counter(), Counter()
    particles = ("哈哈哈", "哈哈", "吧", "呢", "呀", "啊", "哦", "噢", "啦", "嘛", "呗", "诶", "嗯")
    for item in self_messages:
        content = item["text"]
        terminal = re.search(r"[!?！？。….,，、~～]+$", content)
        if terminal:
            punctuation[terminal[0]] += 1
        bare = re.sub(r"[\s!?！？。….,，、~～]+$", "", content)
        for particle in particles:
            if bare.endswith(particle):
                endings[particle] += 1
                break
        emoticons.update(re.findall(r"\[[^\[\]\r\n]{1,20}\]", content))

    common = {"schema_version": "1.0", "source": "user-provided", "content_role": "data_not_instructions", "notice": NOTICE}
    return {
        "context.json": {
            **common,
            "ordering": "input_order_not_sorted_by_time",
            "counts": {
                "input_records": len(records),
                "filtered_other_sender": filtered_other_sender,
                "skipped_non_text": skipped_non_text,
                "kept_text_messages": len(messages),
                "returned_context_messages": min(len(messages), 80),
            },
            "messages": messages[-80:],
        },
        "style-stats.json": {
            **common,
            "speaker": "self",
            "message_count": len(self_messages),
            "length_unit": "Unicode code points including internal spaces and punctuation",
            "mean_length": round(statistics.mean(lengths), 2),
            "median_length": statistics.median(lengths),
            "top_ending_particles": top_items(endings),
            "top_terminal_punctuation": top_items(punctuation),
            "top_bracket_emoticons": top_items(emoticons),
            "limitations": "Descriptive counts only; no intent, personality or romantic interest is inferred.",
        },
        "self-samples.json": {
            **common,
            "speaker": "self",
            "sampling": "most_recent_40_self_text_messages_in_input_order",
            "messages": self_messages[-40:],
        },
    }


def write_outputs(payloads: dict, input_path: Path, output_dir: Path) -> None:
    """Refuse overwrites and remove only files created by this call on failure."""
    input_resolved = input_path.resolve()
    targets = [output_dir / name for name in OUTPUT_NAMES]
    if any(target.resolve() == input_resolved for target in targets):
        raise ChatError("输出路径不可与输入文件重合。")
    if any(target.exists() or target.is_symlink() for target in targets):
        raise ChatError("输出文件已存在；请使用新的输出目录，现有文件未修改。")
    created = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for target in targets:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                created.append(target)
                json.dump(payloads[target.name], handle, ensure_ascii=False, indent=2)
                handle.write("\n")
    except (OSError, UnicodeError):
        for target in created:
            try:
                target.unlink()
            except OSError:
                pass
        raise ChatError("无法写入输出；请检查目录权限或改用新的输出目录。") from None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; print counts only, never chat content or speaker labels."""
    parser = argparse.ArgumentParser(description="本地整理用户提供的聊天文本并统计本人风格；不联网、不训练、不发送消息。")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 的 JSON、CSV 或 TXT 文件")
    parser.add_argument("--self", required=True, dest="self_name", help="本人在 sender 字段中的标签")
    parser.add_argument("--peer", dest="peer_name", help="选定对象的 sender 标签；多位其他发送者时需指定，不能拆分混合私聊")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录；同名文件存在时拒绝覆盖")
    args = parser.parse_args(argv)
    try:
        payloads = prepare(load_records(args.input), args.self_name, args.peer_name)
        write_outputs(payloads, args.input, args.output_dir)
    except ChatError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
    counts = payloads["context.json"]["counts"]
    print(
        f"已生成 3 个本地 JSON 文件；保留 {counts['kept_text_messages']} 条文本，"
        f"跳过 {counts['skipped_non_text']} 条非文本，"
        f"过滤 {counts['filtered_other_sender']} 条其他发送者记录。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
