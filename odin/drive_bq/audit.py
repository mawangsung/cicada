"""Drive-to-BigQuery 파일 자동 분류 도구.

Google Drive의 파일 목록을 조회하여 BigQuery에서 바로 쿼리 가능한
파일(CSV, NDJSON, Avro, Google Sheets)인지 자동으로 분류한다.

사용 준비:
  1. Google Cloud 프로젝트에서 Drive API 활성화
  2. OAuth 클라이언트 ID(데스크톱) 생성 후 credentials.json으로 저장
  3. pip install google-api-python-client google-auth-oauthlib

사용 예:
  python -m odin.drive_bq.audit --credentials credentials.json
  python -m odin.drive_bq.audit --credentials credentials.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

BIGQUERY_READY_MIME = {
    "text/csv": "CSV",
    "application/vnd.google-apps.spreadsheet": "GOOGLE_SHEETS",
    "application/vnd.apache.avro": "AVRO",
    "application/x-avro": "AVRO",
    "application/avro": "AVRO",
    "application/jsonl": "NEWLINE_DELIMITED_JSON",
    "application/x-ndjson": "NEWLINE_DELIMITED_JSON",
    "application/json": "NEWLINE_DELIMITED_JSON",
}

BIGQUERY_READY_EXT = {
    ".csv": "CSV",
    ".tsv": "CSV",
    ".json": "NEWLINE_DELIMITED_JSON",
    ".ndjson": "NEWLINE_DELIMITED_JSON",
    ".jsonl": "NEWLINE_DELIMITED_JSON",
    ".avro": "AVRO",
}

CONVERTIBLE_MIME = {
    "application/vnd.ms-excel": "엑셀 → CSV 또는 Google Sheets 변환 필요",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "엑셀 → CSV 또는 Google Sheets 변환 필요",
    "application/vnd.google-apps.document": "Google Docs → CSV/NDJSON 내보내기 필요",
    "application/vnd.google-apps.presentation": "Google Slides → 텍스트 추출 필요",
    "application/pdf": "PDF → 텍스트/CSV 추출 필요",
}


@dataclass
class DriveFileReport:
    file_id: str
    name: str
    mime_type: str
    size_bytes: int
    category: str          # ready / convertible / unsupported / unknown
    bq_format: Optional[str]  # BigQuery 외부 테이블 형식 또는 None
    note: Optional[str]


def classify_file(name: str, mime_type: str) -> tuple[str, Optional[str], Optional[str]]:
    """파일명과 MIME 타입으로 BigQuery 사용 가능 여부를 분류한다."""
    ext = os.path.splitext(name)[1].lower()

    if mime_type in BIGQUERY_READY_MIME:
        return "ready", BIGQUERY_READY_MIME[mime_type], None
    if ext in BIGQUERY_READY_EXT:
        return "ready", BIGQUERY_READY_EXT[ext], f"MIME({mime_type})은 비표준이지만 확장자로 판단"

    if mime_type in CONVERTIBLE_MIME:
        return "convertible", None, CONVERTIBLE_MIME[mime_type]

    return "unsupported", None, "BigQuery 직접 쿼리 불가 — 추출/변환 후 로드 필요"


def list_drive_files(credentials_path: str):
    """Drive API로 전체 파일 목록을 조회한다. (검색/공유함 포함)"""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    token_path = os.path.join(os.path.dirname(credentials_path), "drive_token.json")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    reports = []
    page_token = None
    try:
        while True:
            resp = service.files().list(
                pageSize=200,
                fields="nextPageToken, files(id,name,mimeType,size)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for meta in resp.get("files", []):
                category, bq_format, note = classify_file(
                    meta.get("name", ""), meta.get("mimeType", "")
                )
                reports.append(DriveFileReport(
                    file_id=meta.get("id", ""),
                    name=meta.get("name", ""),
                    mime_type=meta.get("mimeType", ""),
                    size_bytes=int(meta.get("size", 0) or 0),
                    category=category,
                    bq_format=bq_format,
                    note=note,
                ))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        print(f"Drive API 오류: {e}", file=sys.stderr)
        raise
    return reports


def summarize(reports: list[DriveFileReport]) -> dict:
    ready = [r for r in reports if r.category == "ready"]
    return {
        "total_files": len(reports),
        "ready_count": len(ready),
        "ready_bytes": sum(r.size_bytes for r in ready),
        "by_category": {
            c: sum(1 for r in reports if r.category == c)
            for c in ("ready", "convertible", "unsupported", "unknown")
        },
        "by_bq_format": {
            fmt: sum(1 for r in ready if r.bq_format == fmt)
            for fmt in sorted({r.bq_format for r in ready if r.bq_format})
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Drive 파일 BigQuery 사용 가능 여부 감사")
    parser.add_argument("--credentials", required=True, help="OAuth 클라이언트 JSON 경로")
    parser.add_argument("--json", help="결과를 JSON 파일로도 저장할 경로")
    args = parser.parse_args(argv)

    reports = list_drive_files(args.credentials)
    stats = summarize(reports)

    print(f"\n=== Drive → BigQuery 감사 결과 ===")
    print(f"총 파일 수: {stats['total_files']}")
    print(f"BigQuery 즉시 사용 가능: {stats['ready_count']}개 "
          f"({stats['ready_bytes'] / 1_048_576:.1f} MiB)")
    print(f"형식별: {stats['by_bq_format'] or '없음'}")
    print(f"카테고리별: {stats['by_category']}")
    print("\n-- 즉시 사용 가능 파일 --")
    for r in reports:
        if r.category == "ready":
            print(f"  [{r.bq_format}] {r.name} (id={r.file_id})")
    print("\n-- 변환 필요 파일(샘플 20개) --")
    for r in [r for r in reports if r.category == "convertible"][:20]:
        print(f"  {r.name} — {r.note}")

    if args.json:
        payload = {
            "summary": stats,
            "files": [asdict(r) for r in reports],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n전체 결과 저장: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
