"""Safe-step checkpointing and cross-process recovery manager (C1, C2, C3, C4)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from audit import AuditLogger
from sandbox import Sandbox
from schemas import (
    ApprovalStatus,
    RunState,
    RunStatus,
    StopReason,
    WriteCommitStatus,
    WriteJournalEntry,
    utc_now_iso,
)
from state_store import StateStore, atomic_write_json


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    seq: int
    created_at: str
    run: dict[str, Any]
    step: int
    plan: dict[str, Any]
    evidences: list[dict[str, Any]]
    budget: dict[str, Any]
    pending_approvals: list[dict[str, Any]]
    state_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "seq": self.seq,
            "created_at": self.created_at,
            "run": self.run,
            "step": self.step,
            "plan": self.plan,
            "evidences": self.evidences,
            "budget": self.budget,
            "pending_approvals": self.pending_approvals,
            "state_snapshot": self.state_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointRecord":
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            seq=int(data["seq"]),
            created_at=str(data["created_at"]),
            run=dict(data["run"]),
            step=int(data["step"]),
            plan=dict(data["plan"]),
            evidences=list(data["evidences"]),
            budget=dict(data["budget"]),
            pending_approvals=list(data["pending_approvals"]),
            state_snapshot=dict(data["state_snapshot"]),
        )


@dataclass
class RestoreOutcome:
    can_continue: bool
    reason: str
    state: RunState
    checkpoint: CheckpointRecord
    unknown_writes: list[WriteJournalEntry] = field(default_factory=list)
    changed_files: list[dict[str, str]] = field(default_factory=list)


class CheckpointManager:
    """Saves immutable checkpoints at safe step boundaries and validates safety on resume."""

    def __init__(self, state_dir: str | Path, audit: AuditLogger | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.audit = audit

    def checkpoints_dir(self, run_id: str) -> Path:
        return self.state_dir / "runs" / run_id / "checkpoints"

    def latest_path(self, run_id: str) -> Path:
        return self.checkpoints_dir(run_id) / "latest.json"

    def list_checkpoints(self, run_id: str) -> list[CheckpointRecord]:
        cdir = self.checkpoints_dir(run_id)
        if not cdir.exists():
            return []
        records: list[CheckpointRecord] = []
        for path in sorted(cdir.glob("ckpt_*.json")):
            records.append(CheckpointRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return records

    def load_latest(self, run_id: str) -> CheckpointRecord:
        lpath = self.latest_path(run_id)
        if not lpath.exists():
            raise FileNotFoundError(f"No checkpoint found for run_id='{run_id}'.")
        return CheckpointRecord.from_dict(json.loads(lpath.read_text(encoding="utf-8")))

    def save(self, state: RunState, *, reason: str = "safe_step") -> CheckpointRecord:
        """Persist an immutable checkpoint containing run, step, plan, evidences, budget, and pending approvals."""
        cdir = self.checkpoints_dir(state.run_id)
        cdir.mkdir(parents=True, exist_ok=True)
        existing = sorted(cdir.glob("ckpt_*.json"))
        seq = len(existing) + 1
        checkpoint_id = f"ckpt_{seq:04d}"

        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            seq=seq,
            created_at=utc_now_iso(),
            run={
                "run_id": state.run_id,
                "task": state.task,
                "task_type": state.task_type.value,
                "status": state.status.value,
                "stop_reason": state.stop_reason.value,
                "workspace_hash": state.workspace_hash,
                "code_hash": state.code_hash,
            },
            step=state.step_number,
            plan=state.plan.to_dict(),
            evidences=[e.to_dict() for e in state.evidences],
            budget=state.budget.to_dict(),
            pending_approvals=[a.to_dict() for a in state.pending_approvals()],
            state_snapshot=state.to_dict(),
        )
        payload = record.to_dict()
        atomic_write_json(cdir / f"{checkpoint_id}.json", payload)
        atomic_write_json(self.latest_path(state.run_id), payload)

        if self.audit is not None:
            self.audit.record(
                run_id=state.run_id,
                stage="CHECKPOINT",
                rule="SAFE_STEP_CHECKPOINT",
                decision="SAVED",
                actor="checkpoint",
                step_number=state.step_number,
                details={
                    "checkpoint_id": checkpoint_id,
                    "reason": reason,
                    "status": state.status.value,
                    "steps_used": state.budget.steps_used,
                    "pending_approvals": len(record.pending_approvals),
                    "evidences_count": len(record.evidences),
                    "workspace_hash": state.workspace_hash,
                },
            )
        return record

    def restore(
        self,
        run_id: str,
        *,
        sandbox: Sandbox,
        state_store: StateStore,
    ) -> RestoreOutcome:
        """Restore a run from its latest checkpoint with C2 (UNKNOWN_RESULT) and C4 (workspace drift) guards."""
        checkpoint = self.load_latest(run_id)
        restored = RunState.from_dict(checkpoint.state_snapshot)

        # Merge approvals updated out-of-band (e.g., human approved/rejected while paused)
        if state_store.exists(run_id):
            live_state = state_store.load_run(run_id)
            for approval_id, live_approval in live_state.approvals.items():
                existing = restored.approvals.get(approval_id)
                if existing is None:
                    restored.approvals[approval_id] = live_approval
                elif existing.status == ApprovalStatus.PENDING and live_approval.status in (
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.REJECTED,
                ):
                    restored.approvals[approval_id] = live_approval

        # Merge live WAL write journal (captures crash between write start and result commit!)
        wal = state_store.load_write_journal(run_id)
        for tcid, wal_entry in wal.items():
            existing_entry = restored.write_journal.get(tcid)
            if existing_entry is None or existing_entry.status != WriteCommitStatus.COMMITTED:
                restored.write_journal[tcid] = wal_entry

        # --------------------------------------------------------------
        # C2 Guard: Detect uncommitted writes (EXECUTING or UNKNOWN_RESULT)
        # MUST run BEFORE C4 so a crash after writing to disk is classified
        # as UNKNOWN_RESULT rather than external workspace drift.
        # --------------------------------------------------------------
        uncommitted: list[WriteJournalEntry] = []
        for entry in restored.write_journal.values():
            if entry.status in (WriteCommitStatus.EXECUTING, WriteCommitStatus.UNKNOWN_RESULT):
                entry.status = WriteCommitStatus.UNKNOWN_RESULT
                uncommitted.append(entry)

        if uncommitted:
            restored.status = RunStatus.WAITING_VERIFICATION
            restored.stop_reason = StopReason.UNKNOWN_WRITE_RESULT
            restored.last_error = (
                "Uncommitted mutating write detected on recovery; marked UNKNOWN_RESULT. "
                "Automatic replay is forbidden."
            )
            state_store.save_run(restored)
            self.save(restored, reason="unknown_write_result")
            if self.audit is not None:
                for entry in uncommitted:
                    self.audit.record(
                        run_id=run_id,
                        stage="CHECKPOINT",
                        rule="UNKNOWN_RESULT_GUARD",
                        decision="UNKNOWN_RESULT",
                        actor="checkpoint",
                        tool_call_id=entry.tool_call_id,
                        step_number=restored.step_number,
                        tool_name=entry.tool_name,
                        arguments=entry.arguments,
                        details={
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "message": "Write side-effect started without committed result; auto-replay blocked.",
                        },
                    )
            return RestoreOutcome(
                can_continue=False,
                reason="UNKNOWN_RESULT",
                state=restored,
                checkpoint=checkpoint,
                unknown_writes=uncommitted,
            )

        # --------------------------------------------------------------
        # C4 Guard: Detect external workspace drift since last checkpoint
        # --------------------------------------------------------------
        live_snapshot, live_workspace_hash, _live_code_hash = sandbox.compute_workspace_snapshot()
        if restored.workspace_hash and live_workspace_hash != restored.workspace_hash:
            changed_files = sandbox.diff_snapshots(restored.workspace_snapshot, live_snapshot)
            # Invalidate all old evidences
            for ev in restored.evidences:
                ev.stale = True
            # Invalidate all unconsumed approvals (PENDING or APPROVED)
            invalidated_approval_ids: list[str] = []
            for approval in restored.approvals.values():
                if approval.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED):
                    approval.status = ApprovalStatus.INVALIDATED
                    approval.note = (
                        f"{approval.note} | Invalidated due to workspace change on resume."
                    ).strip(" |")
                    invalidated_approval_ids.append(approval.approval_id)

            restored.status = RunStatus.WAITING_VERIFICATION
            restored.stop_reason = StopReason.WORKSPACE_CHANGED
            restored.last_error = (
                "Workspace changed externally since last checkpoint. "
                "Old evidences and approvals have been invalidated; stopping for verification."
            )
            state_store.save_run(restored)
            self.save(restored, reason="workspace_changed")
            if self.audit is not None:
                self.audit.record(
                    run_id=run_id,
                    stage="CHECKPOINT",
                    rule="WORKSPACE_DRIFT_GUARD",
                    decision="INVALIDATED",
                    actor="checkpoint",
                    step_number=restored.step_number,
                    details={
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "expected_workspace_hash": checkpoint.run.get("workspace_hash"),
                        "actual_workspace_hash": live_workspace_hash,
                        "changed_files": changed_files,
                        "invalidated_approvals": invalidated_approval_ids,
                    },
                )
            return RestoreOutcome(
                can_continue=False,
                reason="WORKSPACE_CHANGED",
                state=restored,
                checkpoint=checkpoint,
                changed_files=changed_files,
            )

        # --------------------------------------------------------------
        # C3 Normal Recovery
        # --------------------------------------------------------------
        if self.audit is not None:
            self.audit.record(
                run_id=run_id,
                stage="CHECKPOINT",
                rule="CHECKPOINT_RESTORED",
                decision="RESUMED",
                actor="checkpoint",
                step_number=restored.step_number,
                details={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "steps_used": restored.budget.steps_used,
                    "tool_calls_used": restored.budget.tool_calls_used,
                    "writes_used": restored.budget.writes_used,
                    "committed_writes": sum(
                        1
                        for w in restored.write_journal.values()
                        if w.status == WriteCommitStatus.COMMITTED
                    ),
                },
            )
        return RestoreOutcome(
            can_continue=True,
            reason="RESUMED",
            state=restored,
            checkpoint=checkpoint,
        )
