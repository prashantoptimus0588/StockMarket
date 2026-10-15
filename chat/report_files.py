"""Filesystem helpers for atomic report Markdown/JSON artifacts."""
import json
import os
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Output folders
# ---------------------------------------------------------------------------
_reports_env = os.getenv("REPORTS_DIR", "reports")
REPORTS_DIR = Path(_reports_env)
if not REPORTS_DIR.is_absolute():
    REPORTS_DIR = PROJECT_ROOT / REPORTS_DIR
REPORT_MD_DIR = REPORTS_DIR / "markdown"
REPORT_JSON_DIR = REPORTS_DIR / "json"
RUN_LOG_DIR = REPORTS_DIR / "run_logs"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip()) or "unknown"


def write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def save_report_files(report_markdown: str, report_json: dict, ticker: str, stamp: str) -> tuple:
    base = f"{safe_name(ticker)}_{stamp}"
    md_path = write_text_atomic(REPORT_MD_DIR / f"{base}.md", report_markdown)
    json_path = write_text_atomic(
        REPORT_JSON_DIR / f"{base}.json",
        json.dumps(report_json, indent=2, default=str, ensure_ascii=False),
    )
    return md_path, json_path


