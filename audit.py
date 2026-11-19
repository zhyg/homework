"""Append-only audit logger with decision chain reconstruction and integrity verification."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import os
import threading
import uuid
from typing import Any

from schemas import canonical_json, sha256_text, utc_now_iso

GENESIS_HASH = "0" * 64


class AuditLogger:
    """Append-only JSONL audit log for Gate, Sandbox, Approval, Execution, State, Checkpoint, and Completion."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read_tail_state(self) -> tuple[int, str]:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return 0, GENESIS_HASH
        last_line = ""
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if not last_line:
            return 0, GENESIS_HASH
        data = json.loads(last_line)
        return int(data["seq"]), str(data["event_hash"])

    def record(
        self,
        *,
        run_id: str,
        stage: str,
        rule: str,
        decision: str,
        actor: str,
        tool_call_id: str | None = None,
        step_number: int = 0,
        tool_name: str | None = None,
        arguments: Any = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one immutable audit event to the JSONL log."""
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a+", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    fh.seek(0)
                    last_line = ""
                    for line in fh:
                        stripped = line.strip()
                        if stripped:
                            last_line = stripped
                    if last_line:
                        prev = json.loads(last_line)
                        seq = int(prev["seq"]) + 1
                        prev_hash = str(prev["event_hash"])
                    else:
                        seq = 1
                        prev_hash = GENESIS_HASH

                    event_body: dict[str, Any] = {
                        "seq": seq,
                        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                        "timestamp": utc_now_iso(),
                        "run_id": run_id,
                        "step_number": step_number,
                        "stage": stage,
                        "rule": rule,
                        "decision": decision,
                        "actor": actor,
                        "toolCallId": tool_call_id,
                        "tool_name": tool_name,
                        "arguments_digest": (
                            sha256_text(canonical_json(arguments))
                            if arguments is not None
                            else None
                        ),
                        "details": details or {},
                        "prev_hash": prev_hash,
                    }
                    event_hash = sha256_text(canonical_json(event_body))
                    event_record = {**event_body, "event_hash": event_hash}

                    fh.seek(0, os.SEEK_END)
                    fh.write(json.dumps(event_record, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    return event_record
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    events.append(json.loads(stripped))
        return events

    def query(
        self,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        stage: str | None = None,
        decision: str | None = None,
        rule: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for event in self.read_all():
            if run_id is not None and event.get("run_id") != run_id:
                continue
            if tool_call_id is not None and event.get("toolCallId") != tool_call_id:
                continue
            if stage is not None and event.get("stage") != stage:
                continue
            if decision is not None and event.get("decision") != decision:
                continue
            if rule is not None and event.get("rule") != rule:
                continue
            results.append(event)
        return results

    def reconstruct_chain(
        self,
        run_id: str,
        tool_call_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return chronologically ordered audit chain for a run or a specific tool_call_id."""
        events = self.query(run_id=run_id, tool_call_id=tool_call_id)
        return sorted(events, key=lambda item: int(item["seq"]))

    def summarize_run(self, run_id: str) -> dict[str, Any]:
        """Produce a structured trace summary correlating stages and tool calls for a run."""
        chain = self.reconstruct_chain(run_id=run_id)
        stages_seen = sorted({e["stage"] for e in chain})
        by_tool_call: dict[str, list[dict[str, Any]]] = {}
        for event in chain:
            tcid = event.get("toolCallId")
            if tcid:
                by_tool_call.setdefault(tcid, []).append(
                    {
                        "seq": event["seq"],
                        "stage": event["stage"],
                        "rule": event["rule"],
                        "decision": event["decision"],
                        "actor": event["actor"],
                        "tool_name": event.get("tool_name"),
                    }
                )
        return {
            "run_id": run_id,
            "total_events": len(chain),
            "stages": stages_seen,
            "tool_call_chains": by_tool_call,
            "integrity_ok": self.verify_integrity(),
        }

    def verify_integrity(self) -> bool:
        """Verify monotonic sequence numbers and hash chain across the append-only log."""
        events = self.read_all()
        expected_seq = 1
        expected_prev_hash = GENESIS_HASH
        for event in events:
            if int(event.get("seq", -1)) != expected_seq:
                return False
            if event.get("prev_hash") != expected_prev_hash:
                return False
            recorded_hash = event.get("event_hash")
            body = {k: v for k, v in event.items() if k != "event_hash"}
            computed_hash = sha256_text(canonical_json(body))
            if recorded_hash != computed_hash:
                return False
            expected_seq += 1
            expected_prev_hash = recorded_hash
        return True
