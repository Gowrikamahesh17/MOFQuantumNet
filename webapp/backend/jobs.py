"""Minimal in-process background job runner, backing the Lab's retrain button.

Training takes real minutes — wrong to run inline on a request thread. This isn't a
production job queue (no persistence, no multi-worker coordination); it's a plain thread
+ dict, appropriate for a single-process demo app. Jobs are lost on server restart, which
is fine — the trained artifacts they produce are what matters and those are already
persisted to disk by the training functions themselves.
"""

import threading
import time
import traceback
import uuid

from logging_setup import get_logger

logger = get_logger(__name__)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def start_job(kind: str, fn, *args, **kwargs) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id, "kind": kind, "status": "running",
            "started_at": time.time(), "finished_at": None,
            "result": None, "error": None,
        }
    logger.info(f"Job {job_id} ({kind}) started")

    def runner():
        try:
            result = fn(*args, **kwargs)
            with _lock:
                _jobs[job_id].update(status="done", finished_at=time.time(), result=result)
            logger.info(f"Job {job_id} ({kind}) finished")
        except Exception as exc:  # noqa: BLE001 — genuinely want to capture anything and surface it, not crash the thread silently
            with _lock:
                _jobs[job_id].update(
                    status="failed", finished_at=time.time(),
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                )
            logger.error(f"Job {job_id} ({kind}) failed: {exc}\n{traceback.format_exc()}")

    threading.Thread(target=runner, daemon=True, name=f"job-{job_id}").start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
