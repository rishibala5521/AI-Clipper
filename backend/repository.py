import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    video_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    language_code TEXT NOT NULL,
    is_generated INTEGER NOT NULL,
    transcript_end REAL NOT NULL,
    min_duration INTEGER NOT NULL,
    max_duration INTEGER NOT NULL,
    max_clips INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    draft_count INTEGER NOT NULL,
    dropped_count INTEGER NOT NULL,
    rejected_json TEXT NOT NULL,
    duplicates_removed INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    clip_rank INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    ai_start_s REAL NOT NULL,
    ai_end_s REAL NOT NULL,
    title TEXT NOT NULL,
    hook TEXT NOT NULL,
    reason TEXT NOT NULL,
    transcript_quote TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    overall_score REAL NOT NULL,
    flags_json TEXT NOT NULL,
    quote_verified INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    edited INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_analysis ON clips(analysis_id);
"""


def resolve_db_path(database_url: str) -> str:
    """Turn 'sqlite:///name.db' into a file path. Relative paths live in the backend folder."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("DATABASE_URL must start with sqlite:/// (only SQLite is supported for now)")
    path = Path(database_url[len(prefix):])
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "analysis_id": row["analysis_id"],
        "rank": row["clip_rank"],
        "start": row["start_s"],
        "end": row["end_s"],
        "duration": round(row["end_s"] - row["start_s"], 3),
        "ai_start": row["ai_start_s"],
        "ai_end": row["ai_end_s"],
        "title": row["title"],
        "hook": row["hook"],
        "reason": row["reason"],
        "transcript_quote": row["transcript_quote"],
        "scores": json.loads(row["scores_json"]),
        "overall_score": row["overall_score"],
        "flags": json.loads(row["flags_json"]),
        "quote_verified": bool(row["quote_verified"]),
        "status": row["status"],
        "edited": bool(row["edited"]),
    }


class AnalysisRepository:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._ready = False

    # ----- connection handling -----

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        # Done on first use, so merely creating the app never touches the disk.
        if self._ready:
            return
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._open()) as conn:
            conn.executescript(_SCHEMA)
        self._ready = True

    @contextmanager
    def _session(self):
        self._ensure_schema()
        with closing(self._open()) as conn:
            with conn:  # commit on success, roll back on error
                yield conn

    # ----- writes -----

    def save_analysis(self, analysis: dict, clips: list[dict]) -> str:
        analysis_id = uuid.uuid4().hex
        now = _now()
        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, created_at, video_id, source_url, language_code, is_generated,
                    transcript_end, min_duration, max_duration, max_clips, chunk_count,
                    draft_count, dropped_count, rejected_json, duplicates_removed,
                    input_tokens, output_tokens, total_tokens
                ) VALUES (
                    :id, :created_at, :video_id, :source_url, :language_code, :is_generated,
                    :transcript_end, :min_duration, :max_duration, :max_clips, :chunk_count,
                    :draft_count, :dropped_count, :rejected_json, :duplicates_removed,
                    :input_tokens, :output_tokens, :total_tokens
                )
                """,
                {
                    "id": analysis_id,
                    "created_at": now,
                    "video_id": analysis["video_id"],
                    "source_url": analysis["source_url"],
                    "language_code": analysis["language_code"],
                    "is_generated": int(analysis["is_generated"]),
                    "transcript_end": analysis["transcript_end"],
                    "min_duration": analysis["min_duration"],
                    "max_duration": analysis["max_duration"],
                    "max_clips": analysis["max_clips"],
                    "chunk_count": analysis["chunk_count"],
                    "draft_count": analysis["draft_count"],
                    "dropped_count": analysis["dropped_count"],
                    "rejected_json": json.dumps(analysis["rejected"]),
                    "duplicates_removed": analysis["duplicates_removed"],
                    "input_tokens": analysis["input_tokens"],
                    "output_tokens": analysis["output_tokens"],
                    "total_tokens": analysis["total_tokens"],
                },
            )
            for clip in clips:
                conn.execute(
                    """
                    INSERT INTO clips (
                        id, analysis_id, clip_rank, start_s, end_s, ai_start_s, ai_end_s,
                        title, hook, reason, transcript_quote, scores_json, overall_score,
                        flags_json, quote_verified, status, edited, updated_at
                    ) VALUES (
                        :id, :analysis_id, :clip_rank, :start_s, :end_s, :ai_start_s, :ai_end_s,
                        :title, :hook, :reason, :transcript_quote, :scores_json, :overall_score,
                        :flags_json, :quote_verified, 'pending', 0, :updated_at
                    )
                    """,
                    {
                        "id": uuid.uuid4().hex,
                        "analysis_id": analysis_id,
                        "clip_rank": clip["rank"],
                        "start_s": clip["start"],
                        "end_s": clip["end"],
                        "ai_start_s": clip["start"],
                        "ai_end_s": clip["end"],
                        "title": clip["title"],
                        "hook": clip["hook"],
                        "reason": clip["reason"],
                        "transcript_quote": clip["transcript_quote"],
                        "scores_json": json.dumps(clip["scores"]),
                        "overall_score": clip["overall_score"],
                        "flags_json": json.dumps(clip["flags"]),
                        "quote_verified": int(clip["quote_verified"]),
                        "updated_at": now,
                    },
                )
        return analysis_id

    def update_clip(self, clip_id: str, *, title: str, start: float, end: float) -> dict | None:
        with self._session() as conn:
            cursor = conn.execute(
                """
                UPDATE clips
                SET title = :title, start_s = :start, end_s = :end, edited = 1, updated_at = :now
                WHERE id = :id
                """,
                {"title": title, "start": start, "end": end, "now": _now(), "id": clip_id},
            )
            if cursor.rowcount == 0:
                return None
        return self.get_clip(clip_id)

    def set_review(self, clip_id: str, status: str) -> dict | None:
        with self._session() as conn:
            cursor = conn.execute(
                "UPDATE clips SET status = :status, updated_at = :now WHERE id = :id",
                {"status": status, "now": _now(), "id": clip_id},
            )
            if cursor.rowcount == 0:
                return None
        return self.get_clip(clip_id)

    # ----- reads -----

    def get_clip(self, clip_id: str) -> dict | None:
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT c.*, a.transcript_end AS transcript_end
                FROM clips c JOIN analyses a ON a.id = c.analysis_id
                WHERE c.id = ?
                """,
                (clip_id,),
            ).fetchone()
        if row is None:
            return None
        clip = _clip_from_row(row)
        clip["transcript_end"] = row["transcript_end"]
        return clip

    def get_analysis(self, analysis_id: str) -> dict | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            if row is None:
                return None
            clip_rows = conn.execute(
                "SELECT * FROM clips WHERE analysis_id = ? ORDER BY clip_rank", (analysis_id,)
            ).fetchall()
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "video_id": row["video_id"],
            "source_url": row["source_url"],
            "language_code": row["language_code"],
            "is_generated": bool(row["is_generated"]),
            "transcript_end": row["transcript_end"],
            "settings": {
                "min_duration": row["min_duration"],
                "max_duration": row["max_duration"],
                "max_clips": row["max_clips"],
            },
            "stats": {
                "chunk_count": row["chunk_count"],
                "draft_count": row["draft_count"],
                "dropped_count": row["dropped_count"],
                "rejected": json.loads(row["rejected_json"]),
                "duplicates_removed": row["duplicates_removed"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
            },
            "clips": [_clip_from_row(r) for r in clip_rows],
        }