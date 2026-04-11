#!/usr/bin/env python3
"""
Convert Apple Messages chat.db into nanochat-compatible JSONL.

The emitted dataset is reply-only and uses strict two-message samples:
1) user: inbound contact turn
2) assistant: Adamya's direct reply turn
"""

from __future__ import annotations

import argparse
import json
import plistlib
import random
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()
PRINTABLE_TEXT_RE = re.compile(rb"[ -~\t\r\n]{2,}")
BLANK_LINES_RE = re.compile(r"\n{3,}")
DEFAULT_CONFIG_PATH = Path("training-data/config/extraction_config.json")
EXAMPLE_CONFIG_PATH = Path("training-data/config/extraction_config.example.json")
ARCHIVE_MARKERS = (
    "streamtyped",
    "bplist",
    "nsattributedstring",
    "nsmutableattributedstring",
    "nsstring",
    "nsmutablestring",
    "nsdictionary",
    "nsdict",
    "nsnumber",
    "nsdata",
    "nsurl",
    "__kim",
    "ns.objects",
    "nskeyedarchiver",
)
UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
URL_RE = re.compile(r"https?://\S+")
SUSPICIOUS_TEXT_PATTERNS = [
    re.compile(r"(?i)x?nsobject"),
    re.compile(r"(?i)(?:time|date)duration"),
    re.compile(r"(?i)\$objects|\$archiver|\$version|\$top"),
    re.compile(r"(?i)fulladdress"),
    re.compile(r"(?i)rutgers\.instructure\.com"),
    re.compile(r"(?i)https?://\S+/users/\d+"),
    re.compile(r"(?i)https?://\S+/groups/\d+"),
    re.compile(r"\\(?:TimeDuration|DateDuration)\b"),
]

JUNK_PATTERNS = [
    re.compile(r"(?i)\b(?:verification|security|pass|login|otp)\s*code\b"),
    re.compile(r"(?i)\bone[- ]time\s+password\b"),
    re.compile(r"(?i)\buse\s+\d{4,8}\s+(?:as|for)\b"),
    re.compile(r"(?i)\b\d{4,8}\b.*\b(?:verification|security|pass|login|otp)\b"),
    re.compile(r"(?i)\b(?:msg|message|messages?)\s*&\s*data rates may apply\b"),
    re.compile(r"(?i)\breply\s+(?:stop|unsubscribe|cancel|end)\b"),
    re.compile(r"(?i)\bfree msg:"),
    re.compile(r"(?i)\bmessage blocking is active\b"),
    re.compile(r"(?i)\btracking number\b"),
    re.compile(r"(?i)\bshipment\b.*\b(?:delivered|delivery|tracking)\b"),
    re.compile(r"(?i)\bnot delivered\b"),
]


@dataclass(frozen=True)
class ChatInfo:
    chat_id: int
    contact_label: str


@dataclass(frozen=True)
class ExtractionConfig:
    min_contact_pairs: int = 5
    merge_gap_seconds: int = 600
    seed: int = 42
    limit_chats: int | None = None
    excluded_contact_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class MessageRow:
    chat_id: int
    message_id: int
    timestamp: float
    is_from_me: bool
    contact_label: str
    text: str


@dataclass(frozen=True)
class Turn:
    chat_id: int
    contact_label: str
    is_from_me: bool
    start_timestamp: float
    end_timestamp: float
    message_ids: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class ReplyPair:
    contact_label: str
    messages: list[dict[str, str]]


@dataclass
class ExtractionStats:
    rows_using_message_text: int = 0
    rows_using_attributed_body: int = 0
    rows_dropped_low_confidence_attributed: int = 0
    rows_dropped_reaction_or_effect: int = 0
    rows_dropped_media_only: int = 0
    rows_dropped_empty_text: int = 0
    pairs_dropped_junk: int = 0
    debug_notes: list[str] = field(default_factory=list)


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def choose_handle_label(row: sqlite3.Row) -> str:
    for key in ("handle_uncanonicalized", "handle_id", "handle_chat_identifier"):
        value = row[key]
        if value:
            return str(value).strip()
    return f"chat_{row['chat_id']}"


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized


def recursively_extract_strings(value: object) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, bytes):
        try:
            strings.append(value.decode("utf-8"))
        except UnicodeDecodeError:
            pass
    elif isinstance(value, dict):
        for inner in value.values():
            strings.extend(recursively_extract_strings(inner))
    elif isinstance(value, (list, tuple, set)):
        for inner in value:
            strings.extend(recursively_extract_strings(inner))
    return strings


def text_score(candidate: str) -> tuple[int, int]:
    cleaned = normalize_text(candidate)
    visible = sum(ch.isprintable() and not ch.isspace() for ch in cleaned)
    return (visible, len(cleaned))


def control_character_ratio(candidate: str) -> float:
    if not candidate:
        return 1.0
    disallowed = sum(1 for ch in candidate if ord(ch) < 32 and ch not in "\t\r\n")
    return disallowed / len(candidate)


def archive_marker_hits(candidate: str) -> int:
    lower = candidate.lower()
    return sum(1 for marker in ARCHIVE_MARKERS if marker in lower)


def is_human_text_candidate(candidate: str) -> bool:
    normalized = normalize_text(candidate)
    if not normalized:
        return False

    lower = normalized.lower()
    if archive_marker_hits(normalized) > 0:
        return False
    if looks_like_structured_metadata(normalized):
        return False
    if control_character_ratio(normalized) > 0:
        return False
    if len(normalized) < 2:
        return False

    visible = sum(ch.isprintable() and not ch.isspace() for ch in normalized)
    if visible < 2:
        return False

    alpha_num = sum(ch.isalnum() for ch in normalized)
    if alpha_num == 0:
        return False

    weird_punctuation = sum(ch in "{}[]<>|\\^~" for ch in normalized)
    if weird_punctuation > 2:
        return False

    if lower.startswith(("ns", "__k", "ddscannerresult", "relativeday")):
        return False

    return True


def looks_like_structured_metadata(candidate: str) -> bool:
    normalized = normalize_text(candidate)
    if not normalized:
        return False

    if any(pattern.search(normalized) for pattern in SUSPICIOUS_TEXT_PATTERNS):
        return True

    if "$null" in normalized.lower():
        return True

    urls = URL_RE.findall(normalized)
    if len(urls) >= 2:
        return True

    uuid_hits = UUID_RE.findall(normalized)
    if uuid_hits:
        normalized_stripped = normalized.lstrip("$#%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        if normalized_stripped == uuid_hits[0]:
            return True
        if len(uuid_hits) > 1:
            return True
        visible_chars = [ch for ch in normalized if ch.isprintable() and not ch.isspace()]
        uuid_chars = len(uuid_hits[0])
        if visible_chars and uuid_chars / len(visible_chars) >= 0.6:
            return True

    if "$classname" in normalized or "$classes" in normalized or "NSValue" in normalized:
        return True

    return False


def candidate_score(candidate: str) -> tuple[int, int, int]:
    normalized = normalize_text(candidate)
    alpha_num = sum(ch.isalnum() for ch in normalized)
    spaces = sum(ch.isspace() for ch in normalized)
    return (alpha_num, spaces, -len(normalized))


def select_best_candidate(candidates: Iterable[str]) -> str:
    filtered = []
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if is_human_text_candidate(normalized):
            filtered.append(normalized)
    if not filtered:
        return ""
    return max(filtered, key=candidate_score)


def extract_printable_candidates(blob: bytes) -> list[str]:
    candidates: list[str] = []

    for raw in PRINTABLE_TEXT_RE.findall(blob):
        decoded = raw.decode("utf-8", errors="ignore")
        normalized = normalize_text(decoded)
        if normalized:
            candidates.append(normalized)

    return candidates


def recover_attributed_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, str):
        return normalize_text(value)
    if not isinstance(value, (bytes, bytearray)):
        return ""

    blob = bytes(value)
    structured_candidates: list[str] = []

    try:
        parsed = plistlib.loads(blob)
        structured_candidates.extend(recursively_extract_strings(parsed))
    except Exception:
        pass

    structured_text = select_best_candidate(structured_candidates)
    if structured_text:
        return structured_text

    raw_candidates = extract_printable_candidates(blob)
    return select_best_candidate(raw_candidates)


def apple_time_to_unix(raw_value: object) -> float:
    if raw_value in (None, ""):
        return 0.0
    value = float(raw_value)
    magnitude = abs(value)
    if magnitude > 1e14:
        delta_seconds = value / 1_000_000_000
    elif magnitude > 1e11:
        delta_seconds = value / 1_000_000
    elif magnitude > 1e9:
        delta_seconds = value / 1_000
    else:
        delta_seconds = value
    return APPLE_EPOCH + delta_seconds


def is_effect_or_reaction(row: sqlite3.Row) -> bool:
    if row["associated_message_guid"]:
        return True
    if row["associated_message_type"] not in (None, 0):
        return True
    if row["item_type"] not in (None, 0):
        return True
    if row["balloon_bundle_id"]:
        return True
    if row["expressive_send_style_id"]:
        return True
    return False


def looks_like_junk_text(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    return any(pattern.search(normalized) for pattern in JUNK_PATTERNS)


def get_one_to_one_chats(conn: sqlite3.Connection) -> list[ChatInfo]:
    chat_columns = table_columns(conn, "chat")
    handle_columns = table_columns(conn, "handle")

    if "chat_identifier" in chat_columns:
        chat_identifier_sql = "chat.chat_identifier AS handle_chat_identifier"
    else:
        chat_identifier_sql = "NULL AS handle_chat_identifier"

    if "uncanonicalized_id" in handle_columns:
        handle_uncanonicalized_sql = "handle.uncanonicalized_id AS handle_uncanonicalized"
    else:
        handle_uncanonicalized_sql = "NULL AS handle_uncanonicalized"

    query = f"""
        SELECT
            chat.ROWID AS chat_id,
            handle.id AS handle_id,
            {handle_uncanonicalized_sql},
            {chat_identifier_sql}
        FROM chat
        JOIN chat_handle_join ON chat_handle_join.chat_id = chat.ROWID
        JOIN handle ON handle.ROWID = chat_handle_join.handle_id
        GROUP BY chat.ROWID
        HAVING COUNT(DISTINCT chat_handle_join.handle_id) = 1
    """

    chats = [
        ChatInfo(chat_id=row["chat_id"], contact_label=choose_handle_label(row))
        for row in conn.execute(query)
    ]
    chats.sort(key=lambda chat: (chat.contact_label, chat.chat_id))
    return chats


def select_chats(chats: Sequence[ChatInfo], limit_chats: int | None, seed: int) -> list[ChatInfo]:
    selected = list(chats)
    if limit_chats is None or limit_chats >= len(selected):
        return selected

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(selected)), limit_chats))
    return [selected[index] for index in indices]


def exclude_chats(chats: Sequence[ChatInfo], excluded_contact_labels: Sequence[str]) -> list[ChatInfo]:
    excluded = {label.strip() for label in excluded_contact_labels if label.strip()}
    if not excluded:
        return list(chats)
    return [chat for chat in chats if chat.contact_label not in excluded]


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name)


def _require_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array of strings")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(normalized)


def load_config(config_path: Path) -> ExtractionConfig:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        example_hint = ""
        if config_path == DEFAULT_CONFIG_PATH and EXAMPLE_CONFIG_PATH.exists():
            example_hint = f" Copy {EXAMPLE_CONFIG_PATH} to {DEFAULT_CONFIG_PATH} and edit it for local use."
        raise FileNotFoundError(f"config file not found: {config_path}.{example_hint}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config at {config_path}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a JSON object: {config_path}")

    allowed_keys = {
        "min_contact_pairs",
        "merge_gap_seconds",
        "seed",
        "limit_chats",
        "excluded_contact_labels",
    }
    unexpected_keys = sorted(set(raw) - allowed_keys)
    if unexpected_keys:
        joined = ", ".join(unexpected_keys)
        raise ValueError(f"unexpected config key(s) in {config_path}: {joined}")

    return ExtractionConfig(
        min_contact_pairs=_require_int(raw.get("min_contact_pairs", 5), "min_contact_pairs"),
        merge_gap_seconds=_require_int(raw.get("merge_gap_seconds", 600), "merge_gap_seconds"),
        seed=_require_int(raw.get("seed", 42), "seed"),
        limit_chats=_require_optional_int(raw.get("limit_chats"), "limit_chats"),
        excluded_contact_labels=_require_string_list(
            raw.get("excluded_contact_labels", []),
            "excluded_contact_labels",
        ),
    )


def extract_message_rows(
    conn: sqlite3.Connection,
    chats: Sequence[ChatInfo],
    stats: ExtractionStats | None = None,
) -> list[MessageRow]:
    if not chats:
        return []

    message_columns = table_columns(conn, "message")
    has_attachments_table = bool(
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_attachment_join'"
        ).fetchone()
    )

    sql_parts = {
        "text": "m.text AS text" if "text" in message_columns else "NULL AS text",
        "attributed_body": (
            "m.attributedBody AS attributed_body"
            if "attributedBody" in message_columns
            else "NULL AS attributed_body"
        ),
        "associated_message_guid": (
            "m.associated_message_guid AS associated_message_guid"
            if "associated_message_guid" in message_columns
            else "NULL AS associated_message_guid"
        ),
        "associated_message_type": (
            "m.associated_message_type AS associated_message_type"
            if "associated_message_type" in message_columns
            else "NULL AS associated_message_type"
        ),
        "item_type": "m.item_type AS item_type" if "item_type" in message_columns else "NULL AS item_type",
        "balloon_bundle_id": (
            "m.balloon_bundle_id AS balloon_bundle_id"
            if "balloon_bundle_id" in message_columns
            else "NULL AS balloon_bundle_id"
        ),
        "expressive_send_style_id": (
            "m.expressive_send_style_id AS expressive_send_style_id"
            if "expressive_send_style_id" in message_columns
            else "NULL AS expressive_send_style_id"
        ),
        "has_attachment": (
            "CASE WHEN maj.message_id IS NULL THEN 0 ELSE 1 END AS has_attachment"
            if has_attachments_table
            else "0 AS has_attachment"
        ),
    }

    attachment_join = ""
    if has_attachments_table:
        attachment_join = """
            LEFT JOIN (
                SELECT DISTINCT message_id
                FROM message_attachment_join
            ) AS maj ON maj.message_id = m.ROWID
        """

    query = f"""
        SELECT
            cmj.chat_id AS chat_id,
            m.ROWID AS message_id,
            m.date AS raw_date,
            COALESCE(m.is_from_me, 0) AS is_from_me,
            {sql_parts["text"]},
            {sql_parts["attributed_body"]},
            {sql_parts["associated_message_guid"]},
            {sql_parts["associated_message_type"]},
            {sql_parts["item_type"]},
            {sql_parts["balloon_bundle_id"]},
            {sql_parts["expressive_send_style_id"]},
            {sql_parts["has_attachment"]}
        FROM message AS m
        JOIN chat_message_join AS cmj ON cmj.message_id = m.ROWID
        {attachment_join}
        WHERE cmj.chat_id = ?
        ORDER BY m.date ASC, m.ROWID ASC
    """

    rows: list[MessageRow] = []
    chat_map = {chat.chat_id: chat.contact_label for chat in chats}
    for chat in chats:
        for row in conn.execute(query, (chat.chat_id,)):
            if is_effect_or_reaction(row):
                if stats is not None:
                    stats.rows_dropped_reaction_or_effect += 1
                continue

            text = normalize_text(row["text"]) if row["text"] else ""
            text_from_message = bool(text)
            if text_from_message:
                if stats is not None:
                    stats.rows_using_message_text += 1
            else:
                recovered_text = recover_attributed_text(row["attributed_body"])
                if recovered_text:
                    text = recovered_text
                    if stats is not None:
                        stats.rows_using_attributed_body += 1
                else:
                    if stats is not None and row["attributed_body"] is not None:
                        stats.rows_dropped_low_confidence_attributed += 1
                        stats.rows_dropped_empty_text += 1
                    elif stats is not None:
                        stats.rows_dropped_empty_text += 1
                    continue
            if row["has_attachment"] and not normalize_text(row["text"]):
                # Media-only rows with no ordinary text are excluded.
                if stats is not None:
                    stats.rows_dropped_media_only += 1
                continue

            rows.append(
                MessageRow(
                    chat_id=row["chat_id"],
                    message_id=row["message_id"],
                    timestamp=apple_time_to_unix(row["raw_date"]),
                    is_from_me=bool(row["is_from_me"]),
                    contact_label=chat_map[chat.chat_id],
                    text=text,
                )
            )
    rows.sort(key=lambda row: (row.chat_id, row.timestamp, row.message_id))
    return rows


def build_turns(rows: Sequence[MessageRow], merge_gap_seconds: int) -> list[Turn]:
    turns: list[Turn] = []
    current: Turn | None = None

    for row in rows:
        if current is None:
            current = Turn(
                chat_id=row.chat_id,
                contact_label=row.contact_label,
                is_from_me=row.is_from_me,
                start_timestamp=row.timestamp,
                end_timestamp=row.timestamp,
                message_ids=(row.message_id,),
                text=row.text,
            )
            continue

        same_chat = current.chat_id == row.chat_id
        same_speaker = current.is_from_me == row.is_from_me
        within_gap = (row.timestamp - current.end_timestamp) <= merge_gap_seconds
        if same_chat and same_speaker and within_gap:
            current = Turn(
                chat_id=current.chat_id,
                contact_label=current.contact_label,
                is_from_me=current.is_from_me,
                start_timestamp=current.start_timestamp,
                end_timestamp=row.timestamp,
                message_ids=current.message_ids + (row.message_id,),
                text=f"{current.text}\n{row.text}",
            )
            continue

        turns.append(current)
        current = Turn(
            chat_id=row.chat_id,
            contact_label=row.contact_label,
            is_from_me=row.is_from_me,
            start_timestamp=row.timestamp,
            end_timestamp=row.timestamp,
            message_ids=(row.message_id,),
            text=row.text,
        )

    if current is not None:
        turns.append(current)
    return turns


def render_pair(inbound_text: str, reply_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": inbound_text,
        },
        {
            "role": "assistant",
            "content": reply_text,
        },
    ]


def build_reply_pairs(turns: Sequence[Turn], stats: ExtractionStats | None = None) -> list[ReplyPair]:
    pairs: list[ReplyPair] = []
    i = 0
    while i < len(turns) - 1:
        current = turns[i]
        nxt = turns[i + 1]
        if current.chat_id != nxt.chat_id:
            i += 1
            continue
        if current.is_from_me or not nxt.is_from_me:
            i += 1
            continue

        inbound_text = normalize_text(current.text)
        reply_text = normalize_text(nxt.text)
        next_same_chat = i + 2 < len(turns) and turns[i + 2].chat_id == current.chat_id
        if inbound_text and reply_text:
            if looks_like_structured_metadata(inbound_text) or looks_like_structured_metadata(reply_text):
                if stats is not None:
                    stats.pairs_dropped_junk += 1
                i += 2 if next_same_chat else 1
            elif not looks_like_junk_text(inbound_text) and not looks_like_junk_text(reply_text):
                pairs.append(
                    ReplyPair(
                        contact_label=current.contact_label,
                        messages=render_pair(inbound_text, reply_text),
                    )
                )
                i += 1
            else:
                if stats is not None:
                    stats.pairs_dropped_junk += 1
                i += 1
        else:
            i += 1

    return pairs


def dedupe_pairs(pairs: Iterable[ReplyPair]) -> list[ReplyPair]:
    unique: dict[str, ReplyPair] = {}
    for pair in pairs:
        line = json.dumps(pair.messages, ensure_ascii=False, separators=(",", ":"))
        unique.setdefault(line, pair)
    return [unique[key] for key in sorted(unique)]


def apply_contact_minimum(
    pairs: Sequence[ReplyPair],
    min_contact_pairs: int,
) -> list[ReplyPair]:
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.contact_label] = counts.get(pair.contact_label, 0) + 1

    filtered = [pair for pair in pairs if counts[pair.contact_label] >= min_contact_pairs]
    return filtered


def build_dataset(
    db_path: Path,
    config: ExtractionConfig,
    stats: ExtractionStats | None = None,
) -> list[list[dict[str, str]]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        chats = get_one_to_one_chats(conn)
        chats = exclude_chats(chats, config.excluded_contact_labels)
        chats = select_chats(chats, limit_chats=config.limit_chats, seed=config.seed)
        rows = extract_message_rows(conn, chats, stats=stats)
    finally:
        conn.close()

    turns = build_turns(rows, merge_gap_seconds=config.merge_gap_seconds)
    pairs = build_reply_pairs(turns, stats=stats)
    pairs = dedupe_pairs(pairs)
    pairs = apply_contact_minimum(pairs, min_contact_pairs=config.min_contact_pairs)
    return [pair.messages for pair in pairs]


def write_jsonl(pairs: Sequence[list[dict[str, str]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Apple Messages chat.db into nanochat JSONL.")
    parser.add_argument("--db-path", required=True, help="Path to Apple Messages chat.db")
    parser.add_argument("--output-path", required=True, help="Path to write the nanochat JSONL file")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to extractor JSON config; defaults to training-data/config/extraction_config.json",
    )
    parser.add_argument(
        "--min-contact-pairs",
        type=int,
        default=None,
        help="Minimum usable reply pairs required per contact; overrides config file when passed",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=int,
        default=None,
        help="Merge same-speaker messages within this many seconds; overrides config file when passed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for deterministic chat limiting; overrides config file when passed",
    )
    parser.add_argument(
        "--limit-chats",
        type=int,
        default=None,
        help="Optional max number of 1:1 chats to process for debugging; overrides config file when passed",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace, default_config: ExtractionConfig) -> ExtractionConfig:
    return ExtractionConfig(
        min_contact_pairs=(
            default_config.min_contact_pairs if args.min_contact_pairs is None else args.min_contact_pairs
        ),
        merge_gap_seconds=(
            default_config.merge_gap_seconds if args.merge_gap_seconds is None else args.merge_gap_seconds
        ),
        seed=default_config.seed if args.seed is None else args.seed,
        limit_chats=default_config.limit_chats if args.limit_chats is None else args.limit_chats,
        excluded_contact_labels=default_config.excluded_contact_labels,
    )


def format_stats(stats: ExtractionStats) -> str:
    return (
        "Extraction stats: "
        f"message.text={stats.rows_using_message_text}, "
        f"clean_attributedBody={stats.rows_using_attributed_body}, "
        f"low_confidence_attributed_dropped={stats.rows_dropped_low_confidence_attributed}, "
        f"reaction_or_effect_dropped={stats.rows_dropped_reaction_or_effect}, "
        f"media_only_dropped={stats.rows_dropped_media_only}, "
        f"empty_text_dropped={stats.rows_dropped_empty_text}, "
        f"junk_pairs_dropped={stats.pairs_dropped_junk}"
    )


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path) if args.config_path is not None else DEFAULT_CONFIG_PATH
    config = resolve_config(args, default_config=load_config(config_path))

    if not db_path.exists():
        raise FileNotFoundError(f"chat.db not found: {db_path}")
    if config.min_contact_pairs < 1:
        raise ValueError("--min-contact-pairs must be >= 1")
    if config.merge_gap_seconds < 0:
        raise ValueError("--merge-gap-seconds must be >= 0")
    if config.limit_chats is not None and config.limit_chats < 1:
        raise ValueError("--limit-chats must be >= 1 when provided")

    stats = ExtractionStats()
    pairs = build_dataset(db_path=db_path, config=config, stats=stats)
    write_jsonl(pairs, output_path)

    print(f"Wrote {len(pairs)} reply pairs to {output_path}")
    print(format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
