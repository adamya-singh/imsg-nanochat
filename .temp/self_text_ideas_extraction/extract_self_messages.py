#!/usr/bin/env python3
"""
Extract messages from self-addressed Apple Messages chats.

By default this uses Adamya's own phone number as the self-chat label. The output
is local scratch data under `.temp/`, which is ignored by git.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "training-data/raw/chat.db"
DEFAULT_CONFIG_PATH = REPO_ROOT / "training-data/config/extraction_config.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / ".temp/self_text_ideas_extraction/self_messages.jsonl"
NANOCHAT_EXTRACTOR_PATH = REPO_ROOT / "training-data/scripts/build_nanochat_jsonl.py"
DEFAULT_SELF_LABELS = ("+17814921639", "7814921639")


@dataclass(frozen=True)
class SelfMessageRecord:
    chat_id: int
    message_id: int
    timestamp: float
    timestamp_iso: str
    is_from_me: bool
    contact_label: str
    text: str


def load_nanochat_extractor():
    spec = importlib.util.spec_from_file_location("build_nanochat_jsonl", NANOCHAT_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load extractor: {NANOCHAT_EXTRACTOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_default_self_labels(config_path: Path) -> tuple[str, ...]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()

    labels = raw.get("excluded_contact_labels", [])
    if not isinstance(labels, list):
        raise ValueError(f"expected excluded_contact_labels to be a list in {config_path}")

    return tuple(label.strip() for label in labels if isinstance(label, str) and label.strip())


def choose_handle_label(row: sqlite3.Row) -> str:
    for key in ("handle_uncanonicalized", "handle_id", "handle_chat_identifier"):
        value = row[key]
        if value:
            return str(value).strip()
    return f"chat_{row['chat_id']}"


def find_self_chats(conn: sqlite3.Connection, self_labels: Sequence[str], helper) -> list:
    labels = tuple(label.strip() for label in self_labels if label.strip())
    if not labels:
        raise ValueError("no self labels provided; pass --self-label or configure excluded_contact_labels")

    chat_columns = helper.table_columns(conn, "chat")
    handle_columns = helper.table_columns(conn, "handle")

    handle_uncanonicalized_sql = (
        "handle.uncanonicalized_id AS handle_uncanonicalized"
        if "uncanonicalized_id" in handle_columns
        else "NULL AS handle_uncanonicalized"
    )
    chat_identifier_sql = (
        "chat.chat_identifier AS handle_chat_identifier"
        if "chat_identifier" in chat_columns
        else "NULL AS handle_chat_identifier"
    )

    placeholders = ", ".join("?" for _ in labels)
    query = f"""
        SELECT
            chat.ROWID AS chat_id,
            handle.id AS handle_id,
            {handle_uncanonicalized_sql},
            {chat_identifier_sql}
        FROM chat
        JOIN chat_handle_join ON chat_handle_join.chat_id = chat.ROWID
        JOIN handle ON handle.ROWID = chat_handle_join.handle_id
        WHERE
            handle.id IN ({placeholders})
            OR handle.uncanonicalized_id IN ({placeholders})
            OR chat.chat_identifier IN ({placeholders})
        GROUP BY chat.ROWID
        HAVING COUNT(DISTINCT chat_handle_join.handle_id) = 1
        ORDER BY chat.ROWID ASC
    """

    chats = [
        helper.ChatInfo(chat_id=row["chat_id"], contact_label=choose_handle_label(row))
        for row in conn.execute(query, labels * 3)
    ]
    return chats


def timestamp_iso(timestamp: float) -> str:
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def build_records(db_path: Path, self_labels: Sequence[str], only_from_me: bool) -> tuple[list[SelfMessageRecord], object]:
    helper = load_nanochat_extractor()
    stats = helper.ExtractionStats()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        chats = find_self_chats(conn, self_labels=self_labels, helper=helper)
        rows = helper.extract_message_rows(conn, chats, stats=stats)
    finally:
        conn.close()

    records = [
        SelfMessageRecord(
            chat_id=row.chat_id,
            message_id=row.message_id,
            timestamp=row.timestamp,
            timestamp_iso=timestamp_iso(row.timestamp),
            is_from_me=row.is_from_me,
            contact_label=row.contact_label,
            text=row.text,
        )
        for row in rows
        if not only_from_me or row.is_from_me
    ]
    records.sort(key=lambda record: (record.timestamp, record.chat_id, record.message_id))
    return records, stats


def write_jsonl(records: Sequence[SelfMessageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract self-addressed Apple Messages into JSONL.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to Apple Messages chat.db")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Fallback config whose excluded_contact_labels are used if built-in defaults are cleared",
    )
    parser.add_argument(
        "--self-label",
        action="append",
        default=[],
        help="Exact self handle/chat label to include; can be passed multiple times",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="JSONL output path")
    parser.add_argument(
        "--only-from-me",
        action="store_true",
        help="Only export rows where Apple Messages marks is_from_me=1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db_path
    output_path = args.output_path
    self_labels = tuple(args.self_label) or DEFAULT_SELF_LABELS or load_default_self_labels(args.config_path)

    if not db_path.exists():
        raise FileNotFoundError(f"chat.db not found: {db_path}")

    records, stats = build_records(db_path=db_path, self_labels=self_labels, only_from_me=args.only_from_me)
    write_jsonl(records, output_path)

    print(f"Wrote {len(records)} self-chat messages to {output_path}")
    print(f"Matched self labels: {', '.join(self_labels)}")
    print(
        "Extraction stats: "
        f"message.text={stats.rows_using_message_text}, "
        f"clean_attributedBody={stats.rows_using_attributed_body}, "
        f"low_confidence_attributed_dropped={stats.rows_dropped_low_confidence_attributed}, "
        f"reaction_or_effect_dropped={stats.rows_dropped_reaction_or_effect}, "
        f"media_only_dropped={stats.rows_dropped_media_only}, "
        f"empty_text_dropped={stats.rows_dropped_empty_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
