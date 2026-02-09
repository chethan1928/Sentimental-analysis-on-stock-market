import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Optional


def perf_start(session_id: str, route: str = "/practice") -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "route": route,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_t0": perf_counter(),
        "steps": [],
    }


def perf_mark(
    trace: Optional[Dict[str, Any]],
    step: str,
    seconds: float,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not trace:
        return
    row: Dict[str, Any] = {"step": step, "seconds": round(seconds, 3)}
    if meta:
        row["meta"] = meta
    trace["steps"].append(row)


async def perf_await(trace: Optional[Dict[str, Any]], step: str, awaitable):
    t = perf_counter()
    try:
        return await awaitable
    finally:
        perf_mark(trace, step, perf_counter() - t)


def perf_save(
    trace: Optional[Dict[str, Any]],
    status: str = "ok",
    path: str = "logs/practice_timing.jsonl",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not trace:
        return
    payload: Dict[str, Any] = {
        "session_id": trace.get("session_id"),
        "route": trace.get("route"),
        "status": status,
        "started_at": trace.get("started_at"),
        "total_seconds": round(perf_counter() - trace.get("_t0", perf_counter()), 3),
        "steps": trace.get("steps", []),
    }
    if extra:
        payload["extra"] = extra
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
