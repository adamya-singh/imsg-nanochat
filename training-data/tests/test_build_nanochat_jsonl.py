from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "training-data" / "scripts" / "build_nanochat_jsonl.py"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "training-data" / "config" / "extraction_config.example.json"

spec = importlib.util.spec_from_file_location("build_nanochat_jsonl", SCRIPT_PATH)
build_nanochat_jsonl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = build_nanochat_jsonl
spec.loader.exec_module(build_nanochat_jsonl)


def create_fixture_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                chat_identifier TEXT
            );

            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY,
                id TEXT,
                uncanonicalized_id TEXT
            );

            CREATE TABLE chat_handle_join (
                chat_id INTEGER,
                handle_id INTEGER
            );

            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                attributedBody BLOB,
                date INTEGER,
                is_from_me INTEGER,
                handle_id INTEGER,
                associated_message_guid TEXT,
                associated_message_type INTEGER,
                item_type INTEGER,
                balloon_bundle_id TEXT,
                expressive_send_style_id TEXT
            );

            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            );

            CREATE TABLE message_attachment_join (
                message_id INTEGER,
                attachment_id INTEGER
            );
            """
        )

        handles = [
            (1, "+15550000001", None),
            (2, "+15550000002", None),
            (3, "+15550000003", None),
            (4, "+15550000004", None),
            (5, "+15550000005", None),
            (6, "+15550000006", None),
            (7, "+15550000007", None),
        ]
        conn.executemany("INSERT INTO handle(ROWID, id, uncanonicalized_id) VALUES (?, ?, ?)", handles)

        chats = [
            (1, "+15550000001"),
            (2, "+15550000002"),
            (3, "group-chat"),
            (4, "+15550000004"),
            (5, "+15550000005"),
            (6, "+15550000006"),
            (7, "+15550000007"),
        ]
        conn.executemany("INSERT INTO chat(ROWID, chat_identifier) VALUES (?, ?)", chats)

        chat_handles = [
            (1, 1),
            (2, 2),
            (3, 1),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
        ]
        conn.executemany("INSERT INTO chat_handle_join(chat_id, handle_id) VALUES (?, ?)", chat_handles)

        messages = [
            # chat 1: six usable reply pairs, one pair requires merging, one message uses attributedBody
            (100, "hey", None, 100, 0, 1, None, 0, 0, None, None),
            (101, "yo", None, 120, 1, None, None, 0, 0, None, None),
            (102, "free later?", None, 200, 0, 1, None, 0, 0, None, None),
            (103, "after 7", None, 260, 1, None, None, 0, 0, None, None),
            (104, "need anything", None, 400, 0, 1, None, 0, 0, None, None),
            (105, "from the store", None, 450, 0, 1, None, 0, 0, None, None),
            (106, "nah im good", None, 500, 1, None, None, 0, 0, None, None),
            (107, None, b"meta\x00dont forget lunch", 600, 0, 1, None, 0, 0, None, None),
            (108, "good call", None, 650, 1, None, None, 0, 0, None, None),
            (109, "wanna talk tonight", None, 800, 0, 1, None, 0, 0, None, None),
            (110, "yep", None, 830, 1, None, None, 0, 0, None, None),
            (111, "cool", None, 900, 0, 1, None, 0, 0, None, None),
            (112, "see you", None, 920, 1, None, None, 0, 0, None, None),
            # chat 2: only four usable pairs, below minimum history threshold
            (200, "a", None, 100, 0, 2, None, 0, 0, None, None),
            (201, "b", None, 110, 1, None, None, 0, 0, None, None),
            (202, "c", None, 200, 0, 2, None, 0, 0, None, None),
            (203, "d", None, 210, 1, None, None, 0, 0, None, None),
            (204, "e", None, 300, 0, 2, None, 0, 0, None, None),
            (205, "f", None, 310, 1, None, None, 0, 0, None, None),
            (206, "g", None, 400, 0, 2, None, 0, 0, None, None),
            (207, "h", None, 410, 1, None, None, 0, 0, None, None),
            # chat 3: group chat, must be excluded
            (300, "group hi", None, 100, 0, 1, None, 0, 0, None, None),
            (301, "group yo", None, 110, 1, None, None, 0, 0, None, None),
            # chat 4: attachment-only/reaction-only, must be excluded
            (400, None, None, 100, 0, 4, None, 0, 0, None, None),
            (401, None, None, 110, 1, None, "p:0/abc", 2000, 0, None, None),
            # chat 5: OTP/service, must be filtered
            (500, "Your verification code is 123456", None, 100, 0, 5, None, 0, 0, None, None),
            (501, "ok", None, 110, 1, None, None, 0, 0, None, None),
            # chat 6: long gap should not merge
            (600, "first", None, 100, 0, 6, None, 0, 0, None, None),
            (601, "second too late", None, 900, 0, 6, None, 0, 0, None, None),
            (602, "reply", None, 920, 1, None, None, 0, 0, None, None),
            # chat 7: dedupe exact duplicate final samples
            (700, "same prompt", None, 100, 0, 7, None, 0, 0, None, None),
            (701, "same reply", None, 110, 1, None, None, 0, 0, None, None),
            (702, "same prompt", None, 200, 0, 7, None, 0, 0, None, None),
            (703, "same reply", None, 210, 1, None, None, 0, 0, None, None)
        ]
        conn.executemany(
            """
            INSERT INTO message(
                ROWID, text, attributedBody, date, is_from_me, handle_id,
                associated_message_guid, associated_message_type, item_type,
                balloon_bundle_id, expressive_send_style_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            messages,
        )

        chat_message_rows = []
        for message_id in range(100, 113):
            chat_message_rows.append((1, message_id))
        for message_id in range(200, 208):
            chat_message_rows.append((2, message_id))
        for message_id in range(300, 302):
            chat_message_rows.append((3, message_id))
        for message_id in range(400, 402):
            chat_message_rows.append((4, message_id))
        for message_id in range(500, 502):
            chat_message_rows.append((5, message_id))
        for message_id in range(600, 603):
            chat_message_rows.append((6, message_id))
        for message_id in range(700, 704):
            chat_message_rows.append((7, message_id))
        conn.executemany(
            "INSERT INTO chat_message_join(chat_id, message_id) VALUES (?, ?)",
            chat_message_rows,
        )

        conn.execute(
            "INSERT INTO message_attachment_join(message_id, attachment_id) VALUES (?, ?)",
            (400, 1),
        )
        conn.commit()
    finally:
        conn.close()


def load_pairs(path: Path) -> list[list[dict[str, str]]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_dataset_filters_and_formats(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    create_fixture_db(db_path)

    pairs = build_nanochat_jsonl.build_dataset(
        db_path=db_path,
        config=build_nanochat_jsonl.load_config(EXAMPLE_CONFIG_PATH),
    )

    assert len(pairs) == 6
    assert all(len(pair) == 2 for pair in pairs)
    assert all(pair[0]["role"] == "user" for pair in pairs)
    assert all(pair[1]["role"] == "assistant" for pair in pairs)
    assert all(pair[0]["content"].startswith("[MODE: REPLY]\n[CONTACT: +15550000001]\n") for pair in pairs)

    merged_prompt = next(pair[0]["content"] for pair in pairs if "need anything" in pair[0]["content"])
    assert merged_prompt.endswith("need anything\nfrom the store")

    attributed_prompt = next(pair[0]["content"] for pair in pairs if "lunch" in pair[0]["content"])
    assert attributed_prompt.endswith("dont forget lunch")

    rendered = "\n".join(json.dumps(pair) for pair in pairs)
    assert "+15550000002" not in rendered
    assert "verification code" not in rendered.lower()
    assert "same prompt" not in rendered


def test_turn_building_respects_merge_gap(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    create_fixture_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        chats = [chat for chat in build_nanochat_jsonl.get_one_to_one_chats(conn) if chat.chat_id == 6]
        rows = build_nanochat_jsonl.extract_message_rows(conn, chats)
    finally:
        conn.close()

    turns = build_nanochat_jsonl.build_turns(rows, merge_gap_seconds=600)
    assert len(turns) == 3
    assert turns[0].text == "first"
    assert turns[1].text == "second too late"
    assert turns[2].text == "reply"


def test_write_jsonl_is_deterministic(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    create_fixture_db(db_path)

    pairs = build_nanochat_jsonl.build_dataset(
        db_path=db_path,
        config=build_nanochat_jsonl.load_config(EXAMPLE_CONFIG_PATH),
    )
    build_nanochat_jsonl.write_jsonl(pairs, out_a)
    build_nanochat_jsonl.write_jsonl(pairs, out_b)

    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def test_cli_and_customjson_schema_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    output_path = tmp_path / "nanochat.jsonl"
    create_fixture_db(db_path)

    # Call the module entrypoint directly through build_dataset/write_jsonl so the test stays isolated.
    pairs = build_nanochat_jsonl.build_dataset(
        db_path=db_path,
        config=build_nanochat_jsonl.load_config(EXAMPLE_CONFIG_PATH),
    )
    build_nanochat_jsonl.write_jsonl(pairs, output_path)
    loaded_pairs = load_pairs(output_path)
    assert loaded_pairs == pairs

    sys.path.insert(0, str(REPO_ROOT / "nanochat"))
    try:
        from tasks.customjson import CustomJSON

        dataset = CustomJSON(filepath=str(output_path))
        assert len(dataset) == 6
        example = dataset[0]
        assert example["messages"][0]["role"] == "user"
        assert example["messages"][1]["role"] == "assistant"
    finally:
        sys.path.remove(str(REPO_ROOT / "nanochat"))


def test_build_dataset_excludes_configured_contact_labels(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    create_fixture_db(db_path)

    config = build_nanochat_jsonl.ExtractionConfig(
        excluded_contact_labels=("+15550000001",),
    )
    pairs = build_nanochat_jsonl.build_dataset(db_path=db_path, config=config)

    assert pairs == []


def test_excluded_contact_does_not_appear_in_results(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    create_fixture_db(db_path)

    config = build_nanochat_jsonl.ExtractionConfig(
        min_contact_pairs=4,
        excluded_contact_labels=("+15550000002",),
    )
    pairs = build_nanochat_jsonl.build_dataset(db_path=db_path, config=config)
    rendered = "\n".join(json.dumps(pair) for pair in pairs)

    assert len(pairs) == 6
    assert "+15550000001" in rendered
    assert "+15550000002" not in rendered


def test_example_json_config_matches_expected_defaults(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    create_fixture_db(db_path)
    config = build_nanochat_jsonl.load_config(EXAMPLE_CONFIG_PATH)

    pairs = build_nanochat_jsonl.build_dataset(
        db_path=db_path,
        config=config,
    )

    assert len(pairs) == 6
    assert config == build_nanochat_jsonl.ExtractionConfig(
        min_contact_pairs=5,
        merge_gap_seconds=600,
        seed=42,
        limit_chats=None,
        excluded_contact_labels=(),
    )


def test_cli_values_override_default_config() -> None:
    default_config = build_nanochat_jsonl.ExtractionConfig(
        min_contact_pairs=9,
        merge_gap_seconds=999,
        seed=7,
        limit_chats=2,
        excluded_contact_labels=("+15550000001",),
    )
    args = argparse.Namespace(
        min_contact_pairs=3,
        merge_gap_seconds=120,
        seed=42,
        limit_chats=1,
    )

    resolved = build_nanochat_jsonl.resolve_config(args, default_config=default_config)

    assert resolved.min_contact_pairs == 3
    assert resolved.merge_gap_seconds == 120
    assert resolved.seed == 42
    assert resolved.limit_chats == 1
    assert resolved.excluded_contact_labels == ("+15550000001",)


def test_load_config_from_alternate_path(tmp_path: Path) -> None:
    config_path = tmp_path / "alt_config.json"
    config_path.write_text(
        json.dumps(
            {
                "min_contact_pairs": 4,
                "merge_gap_seconds": 120,
                "seed": 99,
                "limit_chats": 2,
                "excluded_contact_labels": ["+15550000002"],
            }
        ),
        encoding="utf-8",
    )

    config = build_nanochat_jsonl.load_config(config_path)

    assert config == build_nanochat_jsonl.ExtractionConfig(
        min_contact_pairs=4,
        merge_gap_seconds=120,
        seed=99,
        limit_chats=2,
        excluded_contact_labels=("+15550000002",),
    )


def test_resolve_config_works_with_cli_config_path() -> None:
    args = argparse.Namespace(
        config_path="training-data/config/extraction_config.json",
        min_contact_pairs=None,
        merge_gap_seconds=120,
        seed=None,
        limit_chats=None,
    )

    config_path = EXAMPLE_CONFIG_PATH
    loaded = build_nanochat_jsonl.load_config(config_path)
    resolved = build_nanochat_jsonl.resolve_config(args, default_config=loaded)

    assert resolved.merge_gap_seconds == 120
    assert resolved.min_contact_pairs == 5
    assert resolved.excluded_contact_labels == ()


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    try:
        build_nanochat_jsonl.load_config(missing_path)
    except FileNotFoundError as exc:
        assert "config file not found" in str(exc)
        assert "Copy" not in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_load_default_config_path_missing_file_mentions_example(tmp_path: Path) -> None:
    missing_default = tmp_path / "missing_config.json"
    example_path = tmp_path / "example_config.json"
    example_path.write_text("{}", encoding="utf-8")

    old_default = build_nanochat_jsonl.DEFAULT_CONFIG_PATH
    old_example = build_nanochat_jsonl.EXAMPLE_CONFIG_PATH
    build_nanochat_jsonl.DEFAULT_CONFIG_PATH = missing_default
    build_nanochat_jsonl.EXAMPLE_CONFIG_PATH = example_path
    try:
        try:
            build_nanochat_jsonl.load_config(build_nanochat_jsonl.DEFAULT_CONFIG_PATH)
        except FileNotFoundError as exc:
            assert f"Copy {example_path} to {missing_default}" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")
    finally:
        build_nanochat_jsonl.DEFAULT_CONFIG_PATH = old_default
        build_nanochat_jsonl.EXAMPLE_CONFIG_PATH = old_example


def test_load_config_rejects_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not-json", encoding="utf-8")

    try:
        build_nanochat_jsonl.load_config(bad_path)
    except ValueError as exc:
        assert "invalid JSON config" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_config_rejects_wrong_types(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_types.json"
    bad_path.write_text(
        json.dumps(
            {
                "min_contact_pairs": "five",
                "merge_gap_seconds": 600,
                "seed": 42,
                "limit_chats": None,
                "excluded_contact_labels": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        build_nanochat_jsonl.load_config(bad_path)
    except ValueError as exc:
        assert "min_contact_pairs must be an integer" in str(exc)
    else:
        raise AssertionError("expected ValueError")
