"""odin.drive_bq.audit 단위 테스트 (Drive API 없이 분류 로직만 검증)."""

import pytest

from odin.drive_bq.audit import classify_file, summarize, DriveFileReport


@pytest.mark.parametrize("name,mime,expected_cat,expected_fmt", [
    ("data.csv", "text/csv", "ready", "CSV"),
    ("report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "convertible", None),
    ("raw.jsonl", "application/octet-stream", "ready", "NEWLINE_DELIMITED_JSON"),
    ("events.avro", "application/x-avro", "ready", "AVRO"),
    ("memo.hwp", "application/x-hwp", "unsupported", None),
    ("photo.jpg", "image/jpeg", "unsupported", None),
])
def test_classify_file(name, mime, expected_cat, expected_fmt):
    category, bq_format, _ = classify_file(name, mime)
    assert category == expected_cat
    assert bq_format == expected_fmt


def test_google_sheets_recognized():
    category, bq_format, _ = classify_file("재무제표", "application/vnd.google-apps.spreadsheet")
    assert category == "ready"
    assert bq_format == "GOOGLE_SHEETS"


def test_summarize():
    reports = [
        DriveFileReport("a", "a.csv", "text/csv", 100, "ready", "CSV", None),
        DriveFileReport("b", "b.xlsx", "application/vnd.ms-excel", 200, "convertible", None, None),
        DriveFileReport("c", "c.jpg", "image/jpeg", 300, "unsupported", None, None),
    ]
    stats = summarize(reports)
    assert stats["total_files"] == 3
    assert stats["ready_count"] == 1
    assert stats["ready_bytes"] == 100
    assert stats["by_category"]["convertible"] == 1
    assert stats["by_bq_format"] == {"CSV": 1}
