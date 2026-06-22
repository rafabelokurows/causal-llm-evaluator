import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import HTTPException
from app.schemas import ExperimentResult

DB_PATH = Path(__file__).parent.parent / "experiments.db"
LOG_PATH = Path(__file__).parent.parent / "experiments.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS results "
        "(experiment_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    con.commit()
    return con


def save_result(exp_id: str, result: ExperimentResult) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO results VALUES (?, ?)",
            (exp_id, result.model_dump_json()),
        )
    _append_log(exp_id, result)
    logging.info(
        "experiment=%s winner=%s scorer=%s n_variants=%d | %s",
        exp_id,
        result.winner or "none",
        result.scorer_used,
        len(result.effects),
        result.interpretation,
    )


def _append_log(exp_id: str, result: ExperimentResult) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id,
        **result.model_dump(),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_result(exp_id: str) -> ExperimentResult:
    with _conn() as con:
        row = con.execute(
            "SELECT payload FROM results WHERE experiment_id = ?", (exp_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found")
    return ExperimentResult.model_validate_json(row[0])


def list_results() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT experiment_id, payload FROM results").fetchall()
    return [{"experiment_id": r[0], **json.loads(r[1])} for r in rows]
