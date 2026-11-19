"""Shared schemas, enums, and serializable records for the Guarded Coding Harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    """Serialize arbitrary JSON-compatible data deterministically."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def action_fingerprint(tool_name: str, arguments: Any) -> str:
    return sha256_text(canonical_json({"tool": tool_name, "arguments": arguments}))


class TaskType(str, Enum):
    CODE_UNDERSTANDING = "code_understanding"
    TEST_FIX = "test_fix"


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_VERIFICATION = "WAITING_VERIFICATION"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class StopReason(str, Enum):
    NONE = "NONE"
    COMPLETED = "COMPLETED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    WORKSPACE_CHANGED = "WORKSPACE_CHANGED"
    UNKNOWN_WRITE_RESULT = "UNKNOWN_WRITE_RESULT"
    USER_CANCELLED = "USER_CANCELLED"
    LOOP_DETECTED = "LOOP_DETECTED"
    ERROR = "ERROR"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PermissionMode(str, Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    TEST_EXEC = "test_exec"


class GateDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CONSUMED = "CONSUMED"
    INVALIDATED = "INVALIDATED"


class WriteCommitStatus(str, Enum):
    EXECUTING = "EXECUTING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    UNKNOWN_RESULT = "UNKNOWN_RESULT"


@dataclass
class BudgetConfig:
    max_steps: int = 12
    max_tool_calls: int = 30
    max_tokens: int = 200_000
    max_writes: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetConfig":
        return cls(
            max_steps=int(data.get("max_steps", 12)),
            max_tool_calls=int(data.get("max_tool_calls", 30)),
            max_tokens=int(data.get("max_tokens", 200_000)),
            max_writes=int(data.get("max_writes", 10)),
        )


@dataclass
class BudgetState:
    limits: BudgetConfig = field(default_factory=BudgetConfig)
    steps_used: int = 0
    tool_calls_used: int = 0
    writes_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def exceeded_reason(self) -> str | None:
        if self.tool_calls_used > self.limits.max_tool_calls:
            return f"tool_calls_used ({self.tool_calls_used}) exceeded max_tool_calls ({self.limits.max_tool_calls})"
        if self.writes_used > self.limits.max_writes:
            return f"writes_used ({self.writes_used}) exceeded max_writes ({self.limits.max_writes})"
        if self.total_tokens > self.limits.max_tokens:
            return f"total_tokens ({self.total_tokens}) exceeded max_tokens ({self.limits.max_tokens})"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": self.limits.to_dict(),
            "steps_used": self.steps_used,
            "tool_calls_used": self.tool_calls_used,
            "writes_used": self.writes_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetState":
        return cls(
            limits=BudgetConfig.from_dict(data.get("limits", {})),
            steps_used=int(data.get("steps_used", 0)),
            tool_calls_used=int(data.get("tool_calls_used", 0)),
            writes_used=int(data.get("writes_used", 0)),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
        )


@dataclass
class PlanItem:
    id: str
    title: str
    status: str = "pending"  # pending | in_progress | done | blocked

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanItem":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            status=str(data.get("status", "pending")),
        )


@dataclass
class PlanState:
    goal: str = ""
    items: list[PlanItem] = field(default_factory=list)
    updated_at_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "items": [item.to_dict() for item in self.items],
            "updated_at_step": self.updated_at_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanState":
        return cls(
            goal=str(data.get("goal", "")),
            items=[PlanItem.from_dict(item) for item in data.get("items", [])],
            updated_at_step=int(data.get("updated_at_step", 0)),
        )


@dataclass
class EvidenceRecord:
    evidence_id: str
    kind: str  # READ | WRITE | DELIVERABLE | TEST_RUN
    tool_name: str
    tool_call_id: str
    step_number: int
    timestamp: str
    target: str = ""
    passed: bool | None = None
    exit_code: int | None = None
    code_hash: str = ""
    workspace_hash: str = ""
    stale: bool = False
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(data["evidence_id"]),
            kind=str(data["kind"]),
            tool_name=str(data["tool_name"]),
            tool_call_id=str(data["tool_call_id"]),
            step_number=int(data["step_number"]),
            timestamp=str(data["timestamp"]),
            target=str(data.get("target", "")),
            passed=data.get("passed"),
            exit_code=data.get("exit_code"),
            code_hash=str(data.get("code_hash", "")),
            workspace_hash=str(data.get("workspace_hash", "")),
            stale=bool(data.get("stale", False)),
            summary=str(data.get("summary", "")),
            details=dict(data.get("details", {})),
        )


@dataclass
class ApprovalRecord:
    approval_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    action_fingerprint: str
    risk_level: RiskLevel
    workspace_hash: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: str = field(default_factory=utc_now_iso)
    decided_at: str | None = None
    consumed_at: str | None = None
    reviewer: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "action_fingerprint": self.action_fingerprint,
            "risk_level": self.risk_level.value,
            "workspace_hash": self.workspace_hash,
            "status": self.status.value,
            "requested_at": self.requested_at,
            "decided_at": self.decided_at,
            "consumed_at": self.consumed_at,
            "reviewer": self.reviewer,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRecord":
        return cls(
            approval_id=str(data["approval_id"]),
            run_id=str(data["run_id"]),
            tool_call_id=str(data["tool_call_id"]),
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments", {})),
            action_fingerprint=str(data["action_fingerprint"]),
            risk_level=RiskLevel(data["risk_level"]),
            workspace_hash=str(data.get("workspace_hash", "")),
            status=ApprovalStatus(data["status"]),
            requested_at=str(data.get("requested_at", utc_now_iso())),
            decided_at=data.get("decided_at"),
            consumed_at=data.get("consumed_at"),
            reviewer=data.get("reviewer"),
            note=str(data.get("note", "")),
        )


@dataclass
class WriteJournalEntry:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    action_fingerprint: str
    status: WriteCommitStatus
    step_number: int
    before_workspace_hash: str
    after_workspace_hash: str = ""
    approval_id: str | None = None
    started_at: str = field(default_factory=utc_now_iso)
    committed_at: str | None = None
    result_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "action_fingerprint": self.action_fingerprint,
            "status": self.status.value,
            "step_number": self.step_number,
            "before_workspace_hash": self.before_workspace_hash,
            "after_workspace_hash": self.after_workspace_hash,
            "approval_id": self.approval_id,
            "started_at": self.started_at,
            "committed_at": self.committed_at,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WriteJournalEntry":
        return cls(
            tool_call_id=str(data["tool_call_id"]),
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments", {})),
            action_fingerprint=str(data["action_fingerprint"]),
            status=WriteCommitStatus(data["status"]),
            step_number=int(data.get("step_number", 0)),
            before_workspace_hash=str(data.get("before_workspace_hash", "")),
            after_workspace_hash=str(data.get("after_workspace_hash", "")),
            approval_id=data.get("approval_id"),
            started_at=str(data.get("started_at", utc_now_iso())),
            committed_at=data.get("committed_at"),
            result_summary=str(data.get("result_summary", "")),
        )


@dataclass
class StepRecord:
    step_number: int
    timestamp: str = field(default_factory=utc_now_iso)
    model_output: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: str | None = None
    error: str | None = None
    is_final_answer: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        return cls(
            step_number=int(data["step_number"]),
            timestamp=str(data.get("timestamp", utc_now_iso())),
            model_output=data.get("model_output"),
            tool_calls=list(data.get("tool_calls", [])),
            observations=data.get("observations"),
            error=data.get("error"),
            is_final_answer=bool(data.get("is_final_answer", False)),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
        )


@dataclass
class RunState:
    run_id: str
    task: str
    task_type: TaskType
    status: RunStatus = RunStatus.PENDING
    stop_reason: StopReason = StopReason.NONE
    step_number: int = 0
    allowed_permissions: list[PermissionMode] = field(
        default_factory=lambda: [
            PermissionMode.READ_ONLY,
            PermissionMode.WORKSPACE_WRITE,
            PermissionMode.TEST_EXEC,
        ]
    )
    required_deliverables: list[str] = field(default_factory=list)
    plan: PlanState = field(default_factory=PlanState)
    steps: list[StepRecord] = field(default_factory=list)
    evidences: list[EvidenceRecord] = field(default_factory=list)
    budget: BudgetState = field(default_factory=BudgetState)
    approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    write_journal: dict[str, WriteJournalEntry] = field(default_factory=dict)
    recent_action_fingerprints: list[str] = field(default_factory=list)
    workspace_snapshot: dict[str, str] = field(default_factory=dict)
    workspace_hash: str = ""
    code_hash: str = ""
    final_answer: str | None = None
    last_error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def pending_approvals(self) -> list[ApprovalRecord]:
        return [a for a in self.approvals.values() if a.status == ApprovalStatus.PENDING]

    def approved_unconsumed_approvals(self) -> list[ApprovalRecord]:
        return [a for a in self.approvals.values() if a.status == ApprovalStatus.APPROVED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "step_number": self.step_number,
            "allowed_permissions": [p.value for p in self.allowed_permissions],
            "required_deliverables": list(self.required_deliverables),
            "plan": self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "evidences": [e.to_dict() for e in self.evidences],
            "budget": self.budget.to_dict(),
            "approvals": {k: v.to_dict() for k, v in self.approvals.items()},
            "write_journal": {k: v.to_dict() for k, v in self.write_journal.items()},
            "recent_action_fingerprints": list(self.recent_action_fingerprints),
            "workspace_snapshot": dict(self.workspace_snapshot),
            "workspace_hash": self.workspace_hash,
            "code_hash": self.code_hash,
            "final_answer": self.final_answer,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(
            run_id=str(data["run_id"]),
            task=str(data["task"]),
            task_type=TaskType(data["task_type"]),
            status=RunStatus(data["status"]),
            stop_reason=StopReason(data.get("stop_reason", StopReason.NONE.value)),
            step_number=int(data.get("step_number", 0)),
            allowed_permissions=[
                PermissionMode(p)
                for p in data.get(
                    "allowed_permissions",
                    [
                        PermissionMode.READ_ONLY.value,
                        PermissionMode.WORKSPACE_WRITE.value,
                        PermissionMode.TEST_EXEC.value,
                    ],
                )
            ],
            required_deliverables=list(data.get("required_deliverables", [])),
            plan=PlanState.from_dict(data.get("plan", {})),
            steps=[StepRecord.from_dict(s) for s in data.get("steps", [])],
            evidences=[EvidenceRecord.from_dict(e) for e in data.get("evidences", [])],
            budget=BudgetState.from_dict(data.get("budget", {})),
            approvals={
                k: ApprovalRecord.from_dict(v)
                for k, v in data.get("approvals", {}).items()
            },
            write_journal={
                k: WriteJournalEntry.from_dict(v)
                for k, v in data.get("write_journal", {}).items()
            },
            recent_action_fingerprints=list(data.get("recent_action_fingerprints", [])),
            workspace_snapshot=dict(data.get("workspace_snapshot", {})),
            workspace_hash=str(data.get("workspace_hash", "")),
            code_hash=str(data.get("code_hash", "")),
            final_answer=data.get("final_answer"),
            last_error=data.get("last_error"),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
        )


@dataclass
class GateResponse:
    ok: bool
    status: str  # EXECUTED | DENIED | WAITING_APPROVAL | WAITING_VERIFICATION | ERROR
    tool_call_id: str
    tool_name: str
    rule: str
    decision: str
    output: Any = None
    error_code: str | None = None
    message: str = ""
    approval_id: str | None = None
    should_stop: bool = False
    stop_reason: StopReason | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "tool_call_id": self.tool_call_id,
            "toolCallId": self.tool_call_id,
            "tool_name": self.tool_name,
            "rule": self.rule,
            "decision": self.decision,
        }
        if self.output is not None:
            payload["output"] = self.output
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.message:
            payload["message"] = self.message
        if self.approval_id:
            payload["approval_id"] = self.approval_id
        if self.stop_reason:
            payload["stop_reason"] = self.stop_reason.value
        return payload

    def to_observation_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
