"""Audit Stocksense persistence across PostgreSQL and local report artifacts.

Run from the project root:
    python audit_persistence.py
    python audit_persistence.py --reports-dir reports

This script is read-only with respect to PostgreSQL. It does not delete or modify
reports, database rows, LangGraph checkpoints, or run logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value, None
        return None, "JSON root is not an object"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def inspect_files(root: Path) -> dict:
    md_dir = root / "markdown"
    json_dir = root / "json"
    log_dir = root / "run_logs"

    markdown_files = sorted(md_dir.rglob("*.md")) if md_dir.exists() else []
    json_files = sorted(json_dir.rglob("*.json")) if json_dir.exists() else []
    run_logs = sorted(log_dir.rglob("*.json")) if log_dir.exists() else []

    md_by_stem: dict[str, list[Path]] = defaultdict(list)
    json_by_stem: dict[str, list[Path]] = defaultdict(list)
    md_hashes: dict[str, list[str]] = defaultdict(list)
    json_hashes: dict[str, list[str]] = defaultdict(list)
    json_records: list[dict] = []
    malformed_json: list[dict] = []

    for path in markdown_files:
        md_by_stem[path.stem].append(path)
        try:
            md_hashes[sha256_text(path.read_text(encoding="utf-8"))].append(str(path))
        except Exception as exc:
            malformed_json.append({"path": str(path), "error": f"Markdown read failed: {exc}"})

    for path in json_files:
        json_by_stem[path.stem].append(path)
        data, error = read_json(path)
        if error:
            malformed_json.append({"path": str(path), "error": error})
            continue
        canonical_hash = sha256_text(canonical_json(data))
        json_hashes[canonical_hash].append(str(path))
        json_records.append({"path": str(path), "data": data, "hash": canonical_hash})

    pairs = []
    all_stems = sorted(set(md_by_stem) | set(json_by_stem))
    for stem in all_stems:
        mds = md_by_stem.get(stem, [])
        jss = json_by_stem.get(stem, [])
        if len(mds) == 1 and len(jss) == 1:
            status = "paired"
        elif not mds:
            status = "missing_markdown"
        elif not jss:
            status = "missing_json"
        else:
            status = "duplicate_stem"
        pairs.append({
            "stem": stem,
            "status": status,
            "markdown_paths": [str(p) for p in mds],
            "json_paths": [str(p) for p in jss],
        })

    duplicate_markdown = [paths for paths in md_hashes.values() if len(paths) > 1]
    duplicate_json = [paths for paths in json_hashes.values() if len(paths) > 1]

    log_records = []
    malformed_logs = []
    for path in run_logs:
        data, error = read_json(path)
        if error:
            malformed_logs.append({"path": str(path), "error": error})
            continue
        log_records.append({"path": str(path), "data": data})

    return {
        "markdown_files": markdown_files,
        "json_files": json_files,
        "run_logs": run_logs,
        "json_records": json_records,
        "pairs": pairs,
        "duplicate_markdown_content": duplicate_markdown,
        "duplicate_json_content": duplicate_json,
        "malformed_json": malformed_json,
        "log_records": log_records,
        "malformed_logs": malformed_logs,
    }


def inspect_database() -> list[dict]:
    # Import after dotenv is loaded so database.connection sees DATABASE_URL.
    from database.models import Assessment
    from database.connection import SessionLocal

    with SessionLocal() as session:
        rows = session.scalars(select(Assessment).order_by(Assessment.id)).all()
        result = []
        for row in rows:
            result.append({
                "id": row.id,
                "thread_id": row.thread_id,
                "ticker": row.ticker,
                "profile": row.profile,
                "horizon": row.horizon,
                "risk_verdict": row.risk_verdict,
                "return_potential": row.return_potential,
                "combined_verdict": row.combined_verdict,
                "approval_status": row.approval_status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "markdown_hash": sha256_text(row.report_markdown),
                "json_hash": sha256_text(canonical_json(row.report_json)) if row.report_json is not None else None,
                "has_markdown": bool(row.report_markdown),
                "has_json": row.report_json is not None,
                "report_markdown": row.report_markdown,
                "report_json": row.report_json,
            })
    return result


def make_audit(root: Path, db_rows: list[dict], files: dict) -> dict:
    md_hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for path in files["markdown_files"]:
        try:
            md_hash_to_paths[sha256_text(path.read_text(encoding="utf-8"))].append(str(path))
        except Exception:
            pass

    json_hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for record in files["json_records"]:
        json_hash_to_paths[record["hash"]].append(record["path"])

    duplicate_db_content: dict[str, list[int]] = defaultdict(list)
    near_duplicate_groups: dict[tuple, list[int]] = defaultdict(list)
    db_to_files = []
    missing_files_for_db = []
    for row in db_rows:
        md_matches = md_hash_to_paths.get(row["markdown_hash"], []) if row["markdown_hash"] else []
        json_matches = json_hash_to_paths.get(row["json_hash"], []) if row["json_hash"] else []
        db_to_files.append({
            "assessment_id": row["id"],
            "ticker": row["ticker"],
            "markdown_file_matches": md_matches,
            "json_file_matches": json_matches,
            "markdown_match_count": len(md_matches),
            "json_match_count": len(json_matches),
        })
        if not md_matches or not json_matches:
            missing_files_for_db.append({
                "assessment_id": row["id"],
                "ticker": row["ticker"],
                "missing_markdown_match": not bool(md_matches),
                "missing_json_match": not bool(json_matches),
            })
        # Identical report Markdown is the strongest available duplicate indicator.
        if row["markdown_hash"]:
            duplicate_db_content[row["markdown_hash"]].append(row["id"])
        # This is a *candidate* group only; same ticker/preferences/query may be legitimate reruns.
        report_json = row.get("report_json") or {}
        query = report_json.get("query") if isinstance(report_json, dict) else None
        near_duplicate_groups[(row["ticker"], row["profile"], row["horizon"], query)].append(row["id"])

    duplicate_db_rows = [
        {"markdown_hash": h, "assessment_ids": ids}
        for h, ids in duplicate_db_content.items() if len(ids) > 1
    ]
    candidate_groups = [
        {"ticker": key[0], "profile": key[1], "horizon": key[2], "query": key[3], "assessment_ids": ids}
        for key, ids in near_duplicate_groups.items() if len(ids) > 1
    ]

    log_rows = files["log_records"]
    log_assessment_ids = [
        item["data"].get("assessment_id") for item in log_rows
        if item["data"].get("assessment_id") is not None
    ]
    db_ids = {row["id"] for row in db_rows}
    logs_referencing_missing_db = [
        {"path": item["path"], "assessment_id": item["data"].get("assessment_id")}
        for item in log_rows
        if item["data"].get("assessment_id") is not None
        and item["data"].get("assessment_id") not in db_ids
    ]
    unreferenced_assessments = [
        row["id"] for row in db_rows if row["id"] not in set(log_assessment_ids)
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports_root": str(root.resolve()),
        "counts": {
            "markdown_files": len(files["markdown_files"]),
            "json_files": len(files["json_files"]),
            "run_logs": len(files["run_logs"]),
            "database_assessments": len(db_rows),
            "database_rows_with_markdown": sum(bool(row["has_markdown"]) for row in db_rows),
            "database_rows_with_json": sum(bool(row["has_json"]) for row in db_rows),
        },
        "markdown_json_pairing": files["pairs"],
        "duplicate_markdown_file_content": files["duplicate_markdown_content"],
        "duplicate_json_file_content": files["duplicate_json_content"],
        "duplicate_database_rows_identical_markdown": duplicate_db_rows,
        "candidate_repeated_assessments_same_ticker_preferences_query": candidate_groups,
        "database_rows_to_files": db_to_files,
        "database_rows_missing_matching_local_files": missing_files_for_db,
        "malformed_report_files": files["malformed_json"],
        "malformed_run_logs": files["malformed_logs"],
        "run_logs_referencing_missing_database_rows": logs_referencing_missing_db,
        "database_assessments_not_referenced_by_any_run_log": unreferenced_assessments,
        "notes": [
            "Run logs are expected to outnumber reports because failed and incomplete attempts are logged too.",
            "LangGraph checkpoint rows are not assessment rows and are intentionally not counted here.",
            "Repeated ticker/profile/horizon/query groups are candidates for manual review, not automatically classified as duplicates.",
            "Database-to-file matching uses exact Markdown content and canonical JSON content hashes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports", help="REPORTS_DIR root (default: reports)")
    parser.add_argument("--output", default=None, help="Optional path to write the audit JSON")
    args = parser.parse_args()
    root = Path(args.reports_dir)

    try:
        files = inspect_files(root)
        db_rows = inspect_database()
    except Exception as exc:
        print(f"Persistence audit could not complete: {type(exc).__name__}: {exc}")
        print("Check that you are running this from the project root and DATABASE_URL is configured.")
        return 2

    audit = make_audit(root, db_rows, files)
    print("\n=== Stocksense Persistence Audit ===")
    print(f"Reports root:             {audit['reports_root']}")
    print(f"Markdown reports:         {audit['counts']['markdown_files']}")
    print(f"JSON reports:             {audit['counts']['json_files']}")
    print(f"Run logs:                 {audit['counts']['run_logs']}")
    print(f"PostgreSQL assessments:   {audit['counts']['database_assessments']}")
    print(f"Unpaired report stems:    {sum(p['status'] != 'paired' for p in audit['markdown_json_pairing'])}")
    print(f"Duplicate MD file groups: {len(audit['duplicate_markdown_file_content'])}")
    print(f"Duplicate JSON groups:    {len(audit['duplicate_json_file_content'])}")
    print(f"Identical DB report groups: {len(audit['duplicate_database_rows_identical_markdown'])}")
    print(f"DB rows missing files:    {len(audit['database_rows_missing_matching_local_files'])}")
    print(f"Logs referencing missing DB rows: {len(audit['run_logs_referencing_missing_database_rows'])}")

    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nFull audit saved to: {output}")

    if audit["duplicate_database_rows_identical_markdown"]:
        print("\nWARNING: identical report Markdown exists in multiple database rows; review the listed IDs.")
    if audit["database_rows_missing_matching_local_files"]:
        print("\nWARNING: some database rows have no exact matching local Markdown or JSON artifact.")
    if any(p["status"] != "paired" for p in audit["markdown_json_pairing"]):
        print("\nWARNING: some Markdown/JSON report filenames are not paired one-to-one.")
    print("\nAudit is read-only: no database rows or files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
