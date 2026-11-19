"""Persistent business state store for runs, plans, steps, evidences, budgets, approvals, write journal, and stop reasons."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from audit import AuditLogger
from schemas import (
    ApprovalRecord,
    ApprovalStatus,
    RunState,
    WriteCommitStatus,
    WriteJournalEntry,
    utc_now_iso,
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON to disk using write + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    raw = json.dumps(payload, indent=2, ensure_ascii=False)
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


class StateStore:
    """File-backed business state store supporting atomic persistence and WAL write journal."""

    def __init__(self, state_dir: str | Path, audit: AuditLogger | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.runs_dir = self.state_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.audit = audit

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state.json"

    def journal_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "write_journal.json"

    def exists(self, run_id: str) -> bool:
        return self.state_path(run_id).exists()

    def list_runs(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            d.name for d in self.runs_dir.iterdir() if d.is_dir() and (d / "state.json").exists()
        )

    def save_run(self, state: RunState) -> None:
        state.updated_at = utc_now_iso()
        atomic_write_json(self.state_path(state.run_id), state.to_dict())
        # Keep write_journal synchronized
        self.save_write_journal(state.run_id, state.write_journal)
        if self.audit is not None:
            self.audit.record(
                run_id=state.run_id,
                stage="STATE",
                rule="STATE_PERSISTED",
                decision="SAVED",
                actor="state_store",
                step_number=state.step_number,
                details={
                    "status": state.status.value,
                    "stop_reason": state.stop_reason.value,
                    "steps_used": state.budget.steps_used,
                    "evidences_count": len(state.evidences),
                },
            )

    def load_run(self, run_id: str) -> RunState:
        path = self.state_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run state not found for run_id='{run_id}'.")
        data = json.loads(path.read_text(encoding="utf-8"))
        state = RunState.from_dict(data)
        # Merge latest WAL journal entries (handles crash after WAL intent write before state.json save)
        wal = self.load_write_journal(run_id)
        for tcid, entry in wal.items():
            existing = state.write_journal.get(tcid)
            if existing is None or existing.status != WriteCommitStatus.COMMITTED:
                state.write_journal[tcid] = entry
        return state

    # ------------------------------------------------------------------
    # Write-Ahead Journal (WAL) for C2 crash recovery
    # ------------------------------------------------------------------

    def load_write_journal(self, run_id: str) -> dict[str, WriteJournalEntry]:
        path = self.journal_path(run_id)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: WriteJournalEntry.from_dict(v) for k, v in data.items()}

    def save_write_journal(self, run_id: str, journal: dict[str, WriteJournalEntry]) -> None:
        payload = {k: v.to_dict() for k, v in journal.items()}
        atomic_write_json(self.journal_path(run_id), payload)

    def record_write_intent(self, state: RunState, entry: WriteJournalEntry) -> None:
        """Persist EXECUTING intent to WAL before performing filesystem mutation."""
        journal = self.load_write_journal(state.run_id)
        journal[entry.tool_call_id] = entry
        state.write_journal[entry.tool_call_id] = entry
        self.save_write_journal(state.run_id, journal)

    def commit_write_result(
        self,
        state: RunState,
        tool_call_id: str,
        *,
        after_workspace_hash: str,
        result_summary: str,
    ) -> WriteJournalEntry:
        """Mark a mutating write as COMMITTED in WAL and RunState after handler succeeds."""
        journal = self.load_write_journal(state.run_id)
        entry = journal.get(tool_call_id) or state.write_journal[tool_call_id]
        entry.status = WriteCommitStatus.COMMITTED
        entry.after_workspace_hash = after_workspace_hash
        entry.committed_at = utc_now_iso()
        entry.result_summary = result_summary
        journal[tool_call_id] = entry
        state.write_journal[tool_call_id] = entry
        self.save_write_journal(state.run_id, journal)
        return entry

    def abort_write_result(
        self,
        state: RunState,
        tool_call_id: str,
        *,
        error_summary: str,
    ) -> WriteJournalEntry:
        """Mark an unapplied mutating write as FAILED in WAL and RunState when no disk mutation occurred."""
        journal = self.load_write_journal(state.run_id)
        entry = journal.get(tool_call_id) or state.write_journal[tool_call_id]
        entry.status = WriteCommitStatus.FAILED
        entry.after_workspace_hash = entry.before_workspace_hash
        entry.committed_at = utc_now_iso()
        entry.result_summary = error_summary
        journal[tool_call_id] = entry
        state.write_journal[tool_call_id] = entry
        self.save_write_journal(state.run_id, journal)
        return entry

    # ------------------------------------------------------------------
    # Approval lifecycle management
    # ------------------------------------------------------------------

    def approve(
        self,
        run_id: str,
        approval_id: str,
        *,
        reviewer: str = "human",
        note: str = "",
    ) -> ApprovalRecord:
        state = self.load_run(run_id)
        if approval_id not in state.approvals:
            raise KeyError(f"Approval '{approval_id}' not found in run '{run_id}'.")
        record = state.approvals[approval_id]
        if record.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Cannot approve '{approval_id}' in status '{record.status.value}'."
            )
        record.status = ApprovalStatus.APPROVED
        record.decided_at = utc_now_iso()
        record.reviewer = reviewer
        record.note = note
        self.save_run(state)
        if self.audit is not None:
            self.audit.record(
                run_id=run_id,
                stage="APPROVAL",
                rule="HUMAN_APPROVAL",
                decision="APPROVED",
                actor=f"human:{reviewer}",
                tool_call_id=record.tool_call_id,
                step_number=state.step_number,
                tool_name=record.tool_name,
                arguments=record.arguments,
                details={
                    "approval_id": approval_id,
                    "note": note,
                    "workspace_hash": record.workspace_hash,
                },
            )
        return record

    def reject(
        self,
        run_id: str,
        approval_id: str,
        *,
        reviewer: str = "human",
        note: str = "",
    ) -> ApprovalRecord:
        state = self.load_run(run_id)
        if approval_id not in state.approvals:
            raise KeyError(f"Approval '{approval_id}' not found in run '{run_id}'.")
        record = state.approvals[approval_id]
        if record.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Cannot reject '{approval_id}' in status '{record.status.value}'."
            )
        record.status = ApprovalStatus.REJECTED
        record.decided_at = utc_now_iso()
        record.reviewer = reviewer
        record.note = note
        self.save_run(state)
        if self.audit is not None:
            self.audit.record(
                run_id=run_id,
                stage="APPROVAL",
                rule="HUMAN_APPROVAL",
                decision="REJECTED",
                actor=f"human:{reviewer}",
                tool_call_id=record.tool_call_id,
                step_number=state.step_number,
                tool_name=record.tool_name,
                arguments=record.arguments,
                details={"approval_id": approval_id, "note": note},
            )
        return record
