"""Unified Tool Call Gate enforcing validation, permissions, sandbox pre-checks, duplicate limits, approvals, WAL, and auditing."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid
from typing import Any, Callable

from audit import AuditLogger
from checkpoint import CheckpointManager
from sandbox import CommandResult, Sandbox, SandboxViolationError
from schemas import (
    ApprovalRecord,
    ApprovalStatus,
    EvidenceRecord,
    GateDecision,
    GateResponse,
    PermissionMode,
    RiskLevel,
    RunState,
    RunStatus,
    StopReason,
    WriteCommitStatus,
    WriteJournalEntry,
    action_fingerprint,
    utc_now_iso,
)
from state_store import StateStore


@dataclass
class ArgSpec:
    expected_type: type | tuple[type, ...]
    required: bool = True
    non_empty: bool = False
    allowed_values: tuple[Any, ...] | None = None
    allow_none: bool = False


@dataclass
class ToolPolicy:
    name: str
    handler: Callable[..., Any]
    arg_schema: dict[str, ArgSpec]
    permission: PermissionMode
    risk_level: RiskLevel
    requires_approval: bool = False
    is_mutating: bool = False
    path_args: tuple[str, ...] = ()
    command_args: tuple[str, ...] = ()
    evidence_kind: str | None = None  # READ | WRITE | TEST_RUN | None
    allow_extra_args: bool = False


class ToolGate:
    """Single entry point for all tool calls; enforces every pre-execution and post-execution guard."""

    def __init__(
        self,
        *,
        run_state: RunState,
        sandbox: Sandbox,
        state_store: StateStore,
        checkpoint_manager: CheckpointManager,
        audit: AuditLogger,
        max_consecutive_duplicates: int = 2,
        cancel_event: threading.Event | None = None,
        on_after_write_before_commit: Callable[[str, str, Any], None] | None = None,
    ) -> None:
        self.run_state = run_state
        self.sandbox = sandbox
        self.state_store = state_store
        self.checkpoint_manager = checkpoint_manager
        self.audit = audit
        self.max_consecutive_duplicates = max_consecutive_duplicates
        self.cancel_event = cancel_event
        self.on_after_write_before_commit = on_after_write_before_commit
        self._policies: dict[str, ToolPolicy] = {}

    def register(self, policy: ToolPolicy) -> None:
        self._policies[policy.name] = policy

    def get_policy(self, tool_name: str) -> ToolPolicy | None:
        return self._policies.get(tool_name)

    def registered_tools(self) -> list[str]:
        return sorted(self._policies.keys())

    # ------------------------------------------------------------------
    # Argument validation (G1)
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_type(value: Any, expected_type: type | tuple[type, ...]) -> bool:
        # In Python, bool is a subclass of int; treat them as distinct schema types.
        types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        if isinstance(value, bool):
            return bool in types
        non_bool_types = tuple(t for t in types if t is not bool)
        if not non_bool_types:
            return False
        return isinstance(value, non_bool_types)

    def _validate_arguments(
        self,
        policy: ToolPolicy,
        arguments: Any,
    ) -> list[str]:
        errors: list[str] = []
        if not isinstance(arguments, dict):
            return [f"Arguments must be a JSON object/dict, got {type(arguments).__name__}."]

        # Missing required fields
        for arg_name, spec in policy.arg_schema.items():
            if arg_name not in arguments or arguments[arg_name] is None:
                if spec.required and not spec.allow_none:
                    errors.append(f"Missing required argument '{arg_name}'.")
                continue

            val = arguments[arg_name]
            if not self._matches_type(val, spec.expected_type):
                expected_names = (
                    ", ".join(t.__name__ for t in spec.expected_type)
                    if isinstance(spec.expected_type, tuple)
                    else spec.expected_type.__name__
                )
                errors.append(
                    f"Argument '{arg_name}' expected type {expected_names}, "
                    f"got {type(val).__name__}."
                )
                continue

            if spec.non_empty and isinstance(val, str) and not val.strip():
                errors.append(f"Argument '{arg_name}' must be a non-empty string.")
            if spec.non_empty and isinstance(val, (list, dict)) and len(val) == 0:
                errors.append(f"Argument '{arg_name}' must not be empty.")
            if spec.allowed_values is not None and val not in spec.allowed_values:
                errors.append(
                    f"Argument '{arg_name}' value '{val}' not in allowed values {spec.allowed_values}."
                )

        # Unknown extra arguments
        if not policy.allow_extra_args:
            unknown = sorted(set(arguments.keys()) - set(policy.arg_schema.keys()))
            if unknown:
                errors.append(f"Unexpected argument(s): {', '.join(unknown)}.")

        return errors

    # ------------------------------------------------------------------
    # Duplicate detection (G2)
    # ------------------------------------------------------------------

    def _consecutive_fingerprint_count(self, fp: str) -> int:
        count = 0
        for prev_fp in reversed(self.run_state.recent_action_fingerprints):
            if prev_fp == fp:
                count += 1
            else:
                break
        return count

    # ------------------------------------------------------------------
    # Approval lookup helpers (G3, G4, C4)
    # ------------------------------------------------------------------

    def _canonicalize_arguments(
        self,
        policy: ToolPolicy,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Return canonicalized arguments (normalizing validated workspace-relative paths)."""
        canonical = dict(arguments)
        for path_arg in policy.path_args:
            if path_arg in canonical and canonical[path_arg] is not None:
                resolved = self.sandbox.validate_path(canonical[path_arg], for_write=policy.is_mutating)
                canonical[path_arg] = resolved.relative_to(self.sandbox.workspace_root).as_posix()
        return canonical

    def _find_approvals(
        self,
        tool_call_id: str,
        tool_name: str,
        fp: str,
    ) -> tuple[
        ApprovalRecord | None,
        ApprovalRecord | None,
        ApprovalRecord | None,
        ApprovalRecord | None,
        ApprovalRecord | None,
        ApprovalRecord | None,
    ]:
        """Return (approved, pending, consumed, rejected, invalidated, mismatched) for this call.

        An APPROVED grant is strictly bound to BOTH tool_call_id AND canonical (tool_name, arguments).
        If tool_call_id matches an existing approval record whose tool_name or canonical action_fingerprint
        differs, it is returned as `mismatched` so the tampered call is explicitly denied.
        """
        approved = pending = consumed = rejected = invalidated = mismatched = None
        for record in self.run_state.approvals.values():
            same_id = record.tool_call_id == tool_call_id
            same_action = record.tool_name == tool_name and record.action_fingerprint == fp

            if same_id and not same_action:
                mismatched = record
                continue

            if not (same_id or same_action):
                continue

            if record.status == ApprovalStatus.APPROVED and same_id and same_action:
                approved = record
            elif record.status == ApprovalStatus.PENDING and same_id and same_action:
                pending = record
            elif record.status == ApprovalStatus.CONSUMED and (same_id or same_action):
                consumed = record
            elif record.status == ApprovalStatus.REJECTED and same_id and same_action:
                rejected = record
            elif record.status == ApprovalStatus.INVALIDATED and same_action:
                invalidated = record
        return approved, pending, consumed, rejected, invalidated, mismatched

    def _deny(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        step_number: int,
        stage: str,
        rule: str,
        actor: str,
        error_code: str,
        message: str,
        arguments: Any = None,
        details: dict[str, Any] | None = None,
        should_stop: bool = False,
        stop_reason: StopReason | None = None,
        status: str = "DENIED",
    ) -> GateResponse:
        self.audit.record(
            run_id=self.run_state.run_id,
            stage=stage,
            rule=rule,
            decision=GateDecision.DENY.value,
            actor=actor,
            tool_call_id=tool_call_id,
            step_number=step_number,
            tool_name=tool_name,
            arguments=arguments,
            details={"error_code": error_code, "message": message, **(details or {})},
        )
        return GateResponse(
            ok=False,
            status=status,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            rule=rule,
            decision=GateDecision.DENY.value,
            error_code=error_code,
            message=message,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Single unified entry point for ALL tool calls
    # ------------------------------------------------------------------

    def call(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Any,
        *,
        step_number: int | None = None,
        actor: str = "agent",
    ) -> GateResponse:
        effective_step = step_number if step_number is not None else self.run_state.step_number

        # 0. Cancellation check
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.run_state.status = RunStatus.STOPPED
            self.run_state.stop_reason = StopReason.USER_CANCELLED
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="USER_CANCELLED",
                actor="gate",
                error_code="USER_CANCELLED",
                message="Run was cancelled by user before tool execution.",
                arguments=arguments,
                should_stop=True,
                stop_reason=StopReason.USER_CANCELLED,
            )

        # 1. Tool existence check
        policy = self._policies.get(tool_name)
        if policy is None:
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="UNKNOWN_TOOL",
                actor="gate",
                error_code="UNKNOWN_TOOL",
                message=f"Unknown tool '{tool_name}'. Allowed tools: {', '.join(self.registered_tools())}.",
                arguments=arguments,
            )

        # 2. Argument validation (G1) — runs BEFORE handler!
        arg_errors = self._validate_arguments(policy, arguments)
        if arg_errors:
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="ARG_VALIDATION",
                actor="gate",
                error_code="INVALID_ARGUMENTS",
                message="; ".join(arg_errors),
                arguments=arguments,
                details={"validation_errors": arg_errors},
            )

        # 3. Permission check
        if policy.permission not in self.run_state.allowed_permissions:
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="PERMISSION_CHECK",
                actor="gate",
                error_code="PERMISSION_DENIED",
                message=(
                    f"Tool '{tool_name}' requires permission '{policy.permission.value}', "
                    f"which is not granted for this run."
                ),
                arguments=arguments,
            )

        # 4. Budget pre-check (H3)
        if self.run_state.budget.tool_calls_used >= self.run_state.budget.limits.max_tool_calls:
            self.run_state.status = RunStatus.STOPPED
            self.run_state.stop_reason = StopReason.BUDGET_EXCEEDED
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="BUDGET_LIMIT",
                actor="gate",
                error_code="BUDGET_EXCEEDED",
                message=(
                    f"Tool call budget exhausted ({self.run_state.budget.tool_calls_used}/"
                    f"{self.run_state.budget.limits.max_tool_calls})."
                ),
                arguments=arguments,
                should_stop=True,
                stop_reason=StopReason.BUDGET_EXCEEDED,
            )
        if policy.is_mutating and self.run_state.budget.writes_used >= self.run_state.budget.limits.max_writes:
            self.run_state.status = RunStatus.STOPPED
            self.run_state.stop_reason = StopReason.BUDGET_EXCEEDED
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="BUDGET_LIMIT",
                actor="gate",
                error_code="WRITE_BUDGET_EXCEEDED",
                message=(
                    f"Write budget exhausted ({self.run_state.budget.writes_used}/"
                    f"{self.run_state.budget.limits.max_writes})."
                ),
                arguments=arguments,
                should_stop=True,
                stop_reason=StopReason.BUDGET_EXCEEDED,
            )

        # 5. Sandbox pre-execution boundary checks (S1, S2, S3) — runs BEFORE handler!
        try:
            for path_arg in policy.path_args:
                if path_arg in arguments and arguments[path_arg] is not None:
                    self.sandbox.validate_path(arguments[path_arg], for_write=policy.is_mutating)
            for cmd_arg in policy.command_args:
                if cmd_arg in arguments and arguments[cmd_arg] is not None:
                    self.sandbox.validate_command(arguments[cmd_arg])
        except SandboxViolationError as exc:
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="SANDBOX",
                rule=exc.rule,
                actor="sandbox",
                error_code=exc.rule,
                message=exc.message,
                arguments=arguments,
                details=exc.details,
            )

        canonical_arguments = self._canonicalize_arguments(policy, arguments)
        fp = action_fingerprint(tool_name, canonical_arguments)

        # 6. Write-journal idempotency & UNKNOWN_RESULT guard (C2, C3, G4)
        existing_journal = self.run_state.write_journal.get(tool_call_id)
        if existing_journal is not None:
            if existing_journal.status in (
                WriteCommitStatus.EXECUTING,
                WriteCommitStatus.UNKNOWN_RESULT,
            ):
                self.run_state.status = RunStatus.WAITING_VERIFICATION
                self.run_state.stop_reason = StopReason.UNKNOWN_WRITE_RESULT
                return self._deny(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_number=effective_step,
                    stage="GATE",
                    rule="UNKNOWN_RESULT_GUARD",
                    actor="gate",
                    error_code="UNKNOWN_RESULT",
                    message=(
                        f"Tool call '{tool_call_id}' is in status '{existing_journal.status.value}'; "
                        "automatic replay is forbidden."
                    ),
                    arguments=arguments,
                    should_stop=True,
                    stop_reason=StopReason.UNKNOWN_WRITE_RESULT,
                    status="WAITING_VERIFICATION",
                )
            if existing_journal.status == WriteCommitStatus.COMMITTED:
                return self._deny(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_number=effective_step,
                    stage="GATE",
                    rule="APPROVAL_ALREADY_CONSUMED",
                    actor="gate",
                    error_code="DUPLICATE_WRITE_BLOCKED",
                    message=(
                        f"Mutating tool call '{tool_call_id}' was already committed; "
                        "duplicate write execution is rejected."
                    ),
                    arguments=arguments,
                )

        # 7. Approval check for high-risk / mutating tools (G3, G4, C4)
        approval_to_consume: ApprovalRecord | None = None
        if policy.requires_approval:
            approved, pending, consumed, rejected, invalidated, mismatched = self._find_approvals(
                tool_call_id, tool_name, fp
            )

            if mismatched is not None:
                # G4: Same tool_call_id reused with tampered tool/arguments -> reject without executing or consuming!
                return self._deny(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_number=effective_step,
                    stage="GATE",
                    rule="APPROVAL_ARGUMENT_MISMATCH",
                    actor="gate",
                    error_code="APPROVAL_ARGUMENT_MISMATCH",
                    message=(
                        f"Arguments for tool call '{tool_call_id}' do not match the canonical arguments "
                        f"bound to approval '{mismatched.approval_id}'; tampered execution is rejected."
                    ),
                    arguments=arguments,
                    details={
                        "approval_id": mismatched.approval_id,
                        "approval_status": mismatched.status.value,
                        "expected_fingerprint": mismatched.action_fingerprint,
                        "actual_fingerprint": fp,
                    },
                )

            if approved is not None:
                # Defense-in-depth: double-check approval binding to exact tool_name & canonical fingerprint
                if approved.tool_name != tool_name or approved.action_fingerprint != fp:
                    return self._deny(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        step_number=effective_step,
                        stage="GATE",
                        rule="APPROVAL_ARGUMENT_MISMATCH",
                        actor="gate",
                        error_code="APPROVAL_ARGUMENT_MISMATCH",
                        message=(
                            f"Approval '{approved.approval_id}' does not match current call arguments."
                        ),
                        arguments=arguments,
                        details={
                            "approval_id": approved.approval_id,
                            "expected_fingerprint": approved.action_fingerprint,
                            "actual_fingerprint": fp,
                        },
                    )
                # Verify workspace hasn't drifted since approval was issued (C4 defense-in-depth)
                _, live_ws_hash, _ = self.sandbox.compute_workspace_snapshot()
                if approved.workspace_hash and live_ws_hash != approved.workspace_hash:
                    approved.status = ApprovalStatus.INVALIDATED
                    self.run_state.status = RunStatus.WAITING_VERIFICATION
                    self.run_state.stop_reason = StopReason.WORKSPACE_CHANGED
                    self.state_store.save_run(self.run_state)
                    self.checkpoint_manager.save(self.run_state, reason="approval_workspace_mismatch")
                    return self._deny(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        step_number=effective_step,
                        stage="GATE",
                        rule="WORKSPACE_DRIFT_GUARD",
                        actor="gate",
                        error_code="WORKSPACE_CHANGED",
                        message=(
                            "Workspace changed after approval was granted; approval invalidated. "
                            "Stopping for verification."
                        ),
                        arguments=arguments,
                        should_stop=True,
                        stop_reason=StopReason.WORKSPACE_CHANGED,
                        status="WAITING_VERIFICATION",
                    )
                approval_to_consume = approved

            elif consumed is not None:
                # G4: Approved action already executed once -> reject second execution!
                return self._deny(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_number=effective_step,
                    stage="GATE",
                    rule="APPROVAL_ALREADY_CONSUMED",
                    actor="gate",
                    error_code="APPROVAL_ALREADY_CONSUMED",
                    message=(
                        f"Approval '{consumed.approval_id}' for this write action was already "
                        "consumed; duplicate execution is rejected."
                    ),
                    arguments=arguments,
                    details={"approval_id": consumed.approval_id},
                )

            elif rejected is not None:
                return self._deny(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_number=effective_step,
                    stage="GATE",
                    rule="APPROVAL_REJECTED",
                    actor="gate",
                    error_code="APPROVAL_REJECTED",
                    message=(
                        f"Approval '{rejected.approval_id}' for tool call '{tool_call_id}' "
                        "was rejected by reviewer."
                    ),
                    arguments=arguments,
                    details={"approval_id": rejected.approval_id, "reviewer": rejected.reviewer},
                )

            elif invalidated is not None and pending is None:
                self.run_state.status = RunStatus.WAITING_VERIFICATION
                self.run_state.stop_reason = StopReason.WORKSPACE_CHANGED
                return self._deny(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_number=effective_step,
                    stage="GATE",
                    rule="APPROVAL_INVALIDATED",
                    actor="gate",
                    error_code="APPROVAL_INVALIDATED",
                    message=(
                        f"Approval '{invalidated.approval_id}' was invalidated due to workspace drift."
                    ),
                    arguments=arguments,
                    should_stop=True,
                    stop_reason=StopReason.WORKSPACE_CHANGED,
                    status="WAITING_VERIFICATION",
                )

            else:
                # G3: Unapproved write -> create PENDING approval, enter WAITING_APPROVAL, do NOT execute handler!
                if pending is None:
                    _, live_ws_hash, live_code_hash = self.sandbox.compute_workspace_snapshot()
                    if not self.run_state.workspace_hash:
                        self.run_state.workspace_hash = live_ws_hash
                        self.run_state.code_hash = live_code_hash
                    approval_id = f"appr_{uuid.uuid4().hex[:10]}"
                    pending = ApprovalRecord(
                        approval_id=approval_id,
                        run_id=self.run_state.run_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=dict(canonical_arguments),
                        action_fingerprint=fp,
                        risk_level=policy.risk_level,
                        workspace_hash=live_ws_hash,
                        status=ApprovalStatus.PENDING,
                    )
                    self.run_state.approvals[approval_id] = pending

                self.run_state.status = RunStatus.WAITING_APPROVAL
                self.run_state.stop_reason = StopReason.WAITING_APPROVAL
                self.state_store.save_run(self.run_state)
                self.checkpoint_manager.save(self.run_state, reason="waiting_approval")
                self.audit.record(
                    run_id=self.run_state.run_id,
                    stage="GATE",
                    rule="APPROVAL_REQUIRED",
                    decision=GateDecision.REQUIRE_APPROVAL.value,
                    actor="gate",
                    tool_call_id=tool_call_id,
                    step_number=effective_step,
                    tool_name=tool_name,
                    arguments=arguments,
                    details={
                        "approval_id": pending.approval_id,
                        "risk_level": policy.risk_level.value,
                        "workspace_hash": pending.workspace_hash,
                    },
                )
                return GateResponse(
                    ok=False,
                    status="WAITING_APPROVAL",
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    rule="APPROVAL_REQUIRED",
                    decision=GateDecision.REQUIRE_APPROVAL.value,
                    error_code="APPROVAL_REQUIRED",
                    message=(
                        f"Tool '{tool_name}' requires human approval before execution. "
                        f"Created pending approval '{pending.approval_id}'."
                    ),
                    approval_id=pending.approval_id,
                    should_stop=True,
                    stop_reason=StopReason.WAITING_APPROVAL,
                )

        # 8. Duplicate consecutive call guard (G2)
        if self._consecutive_fingerprint_count(fp) >= self.max_consecutive_duplicates:
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="GATE",
                rule="DUPLICATE_CALL_GUARD",
                actor="gate",
                error_code="DUPLICATE_CALL_BLOCKED",
                message=(
                    f"Blocked consecutive duplicate call to '{tool_name}' with identical arguments "
                    f"(limit={self.max_consecutive_duplicates})."
                ),
                arguments=arguments,
                details={
                    "consecutive_count": self._consecutive_fingerprint_count(fp) + 1,
                    "limit": self.max_consecutive_duplicates,
                },
            )

        # 9. Gate ALLOW -> audit & prepare WAL for mutating actions
        self.audit.record(
            run_id=self.run_state.run_id,
            stage="GATE",
            rule="GATE_PASSED",
            decision=GateDecision.ALLOW.value,
            actor="gate",
            tool_call_id=tool_call_id,
            step_number=effective_step,
            tool_name=tool_name,
            arguments=arguments,
            details={
                "risk_level": policy.risk_level.value,
                "approval_id": approval_to_consume.approval_id if approval_to_consume else None,
            },
        )

        before_ws_hash = self.run_state.workspace_hash
        if not before_ws_hash:
            snap, before_ws_hash, before_code_hash = self.sandbox.compute_workspace_snapshot()
            self.run_state.workspace_snapshot = snap
            self.run_state.workspace_hash = before_ws_hash
            self.run_state.code_hash = before_code_hash

        if approval_to_consume is not None:
            # Consume approval BEFORE invoking handler so replay is impossible even after crash
            approval_to_consume.status = ApprovalStatus.CONSUMED
            approval_to_consume.consumed_at = utc_now_iso()

        if policy.is_mutating:
            wal_entry = WriteJournalEntry(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                action_fingerprint=fp,
                status=WriteCommitStatus.EXECUTING,
                step_number=effective_step,
                before_workspace_hash=before_ws_hash,
                approval_id=approval_to_consume.approval_id if approval_to_consume else None,
            )
            self.state_store.record_write_intent(self.run_state, wal_entry)
            self.audit.record(
                run_id=self.run_state.run_id,
                stage="EXECUTION",
                rule="WRITE_INTENT_PREPARED",
                decision="STARTED",
                actor="gate",
                tool_call_id=tool_call_id,
                step_number=effective_step,
                tool_name=tool_name,
                arguments=arguments,
                details={"before_workspace_hash": before_ws_hash},
            )

        # 10. Invoke actual handler
        try:
            raw_result = policy.handler(**arguments)
        except SandboxViolationError as exc:
            if policy.is_mutating and tool_call_id in self.run_state.write_journal:
                self.state_store.abort_write_result(
                    self.run_state,
                    tool_call_id,
                    error_summary=f"SandboxViolationError({exc.rule}): {exc.message}",
                )
            return self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                step_number=effective_step,
                stage="SANDBOX",
                rule=exc.rule,
                actor="sandbox",
                error_code=exc.rule,
                message=exc.message,
                arguments=arguments,
                details=exc.details,
            )
        except Exception as exc:
            self.run_state.budget.tool_calls_used += 1
            self.run_state.recent_action_fingerprints.append(fp)

            partial_write = False
            if policy.is_mutating and tool_call_id in self.run_state.write_journal:
                _, current_ws_hash, _ = self.sandbox.compute_workspace_snapshot()
                if current_ws_hash == before_ws_hash:
                    # Confirmed no disk mutation occurred; mark WAL FAILED so resume won't misclassify as UNKNOWN_RESULT.
                    self.state_store.abort_write_result(
                        self.run_state,
                        tool_call_id,
                        error_summary=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    # Workspace changed before exception -> treat as UNKNOWN_RESULT and halt for verification.
                    partial_write = True
                    entry = self.run_state.write_journal[tool_call_id]
                    entry.status = WriteCommitStatus.UNKNOWN_RESULT
                    entry.result_summary = f"Exception after workspace mutation: {type(exc).__name__}: {exc}"
                    self.run_state.status = RunStatus.WAITING_VERIFICATION
                    self.run_state.stop_reason = StopReason.UNKNOWN_WRITE_RESULT
                    self.run_state.last_error = entry.result_summary

            self.audit.record(
                run_id=self.run_state.run_id,
                stage="EXECUTION",
                rule="UNKNOWN_RESULT_GUARD" if partial_write else "TOOL_EXECUTION_ERROR",
                decision="UNKNOWN_RESULT" if partial_write else "FAILED",
                actor=actor,
                tool_call_id=tool_call_id,
                step_number=effective_step,
                tool_name=tool_name,
                arguments=arguments,
                details={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "partial_write": partial_write,
                },
            )
            self.state_store.save_run(self.run_state)
            return GateResponse(
                ok=False,
                status="WAITING_VERIFICATION" if partial_write else "ERROR",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                rule="UNKNOWN_RESULT_GUARD" if partial_write else "TOOL_EXECUTION_ERROR",
                decision="UNKNOWN_RESULT" if partial_write else "FAILED",
                error_code="UNKNOWN_RESULT" if partial_write else "EXECUTION_ERROR",
                message=f"{type(exc).__name__}: {exc}",
                should_stop=partial_write,
                stop_reason=StopReason.UNKNOWN_WRITE_RESULT if partial_write else None,
            )

        # Test hook for C2: simulate process crash AFTER write succeeds on disk BEFORE result commit
        if policy.is_mutating and self.on_after_write_before_commit is not None:
            self.on_after_write_before_commit(tool_call_id, tool_name, raw_result)

        # 11. Post-execution state commit, workspace refresh, evidence & audit
        self.run_state.budget.tool_calls_used += 1
        self.run_state.recent_action_fingerprints.append(fp)

        serialized_output: Any = (
            raw_result.to_dict() if isinstance(raw_result, CommandResult) else raw_result
        )

        if policy.is_mutating:
            self.run_state.budget.writes_used += 1
            prev_code_hash = self.run_state.code_hash
            new_snap, new_ws_hash, new_code_hash = self.sandbox.compute_workspace_snapshot()
            self.run_state.workspace_snapshot = new_snap
            self.run_state.workspace_hash = new_ws_hash
            self.run_state.code_hash = new_code_hash

            # H2: If source/test code changed, invalidate all previous TEST_RUN evidences!
            if prev_code_hash and new_code_hash != prev_code_hash:
                for ev in self.run_state.evidences:
                    if ev.kind == "TEST_RUN" and not ev.stale:
                        ev.stale = True

            summary = (
                f"Wrote {serialized_output.get('path', arguments.get('path', ''))}"
                if isinstance(serialized_output, dict)
                else str(serialized_output)[:200]
            )
            self.state_store.commit_write_result(
                self.run_state,
                tool_call_id,
                after_workspace_hash=new_ws_hash,
                result_summary=summary,
            )

        self._record_evidence(
            policy=policy,
            tool_call_id=tool_call_id,
            arguments=arguments,
            raw_result=raw_result,
            serialized_output=serialized_output,
            step_number=effective_step,
        )

        exec_decision = "SUCCEEDED"
        exec_rule = "TOOL_EXECUTED"
        if isinstance(raw_result, CommandResult):
            if raw_result.timed_out:
                exec_decision = "FAILED"
                exec_rule = "SANDBOX_TIMEOUT"
            elif raw_result.cancelled:
                exec_decision = "FAILED"
                exec_rule = "SANDBOX_CANCELLED"
            elif not raw_result.passed:
                exec_decision = "FAILED"
                exec_rule = "TEST_FAILED"

        self.audit.record(
            run_id=self.run_state.run_id,
            stage="EXECUTION",
            rule=exec_rule,
            decision=exec_decision,
            actor="sandbox" if policy.command_args or policy.path_args else actor,
            tool_call_id=tool_call_id,
            step_number=effective_step,
            tool_name=tool_name,
            arguments=arguments,
            details={
                "is_mutating": policy.is_mutating,
                "workspace_hash": self.run_state.workspace_hash,
                "code_hash": self.run_state.code_hash,
                "passed": raw_result.passed if isinstance(raw_result, CommandResult) else True,
            },
        )
        self.state_store.save_run(self.run_state)

        return GateResponse(
            ok=True,
            status="EXECUTED",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            rule=exec_rule,
            decision=exec_decision,
            output=serialized_output,
        )

    def _record_evidence(
        self,
        *,
        policy: ToolPolicy,
        tool_call_id: str,
        arguments: dict[str, Any],
        raw_result: Any,
        serialized_output: Any,
        step_number: int,
    ) -> None:
        if policy.evidence_kind is None:
            return
        ev_id = f"ev_{uuid.uuid4().hex[:10]}"
        if policy.evidence_kind == "READ":
            target = str(arguments.get("path") or arguments.get("directory") or arguments.get("query") or "")
            self.run_state.evidences.append(
                EvidenceRecord(
                    evidence_id=ev_id,
                    kind="READ",
                    tool_name=policy.name,
                    tool_call_id=tool_call_id,
                    step_number=step_number,
                    timestamp=utc_now_iso(),
                    target=target,
                    code_hash=self.run_state.code_hash,
                    workspace_hash=self.run_state.workspace_hash,
                    summary=f"{policy.name}({target})",
                )
            )
        elif policy.evidence_kind == "WRITE":
            rel_path = (
                str(serialized_output.get("path"))
                if isinstance(serialized_output, dict) and serialized_output.get("path")
                else str(arguments.get("path", ""))
            )
            kind = "DELIVERABLE" if rel_path.startswith("artifacts/") else "WRITE"
            self.run_state.evidences.append(
                EvidenceRecord(
                    evidence_id=ev_id,
                    kind=kind,
                    tool_name=policy.name,
                    tool_call_id=tool_call_id,
                    step_number=step_number,
                    timestamp=utc_now_iso(),
                    target=rel_path,
                    code_hash=self.run_state.code_hash,
                    workspace_hash=self.run_state.workspace_hash,
                    summary=f"{policy.name} -> {rel_path}",
                    details=serialized_output if isinstance(serialized_output, dict) else {},
                )
            )
        elif policy.evidence_kind == "TEST_RUN" and isinstance(raw_result, CommandResult):
            self.run_state.evidences.append(
                EvidenceRecord(
                    evidence_id=ev_id,
                    kind="TEST_RUN",
                    tool_name=policy.name,
                    tool_call_id=tool_call_id,
                    step_number=step_number,
                    timestamp=utc_now_iso(),
                    target=" ".join(raw_result.command),
                    passed=raw_result.passed,
                    exit_code=raw_result.exit_code,
                    code_hash=self.run_state.code_hash,
                    workspace_hash=self.run_state.workspace_hash,
                    stale=False,
                    summary=(
                        f"command={' '.join(raw_result.command)} "
                        f"exit_code={raw_result.exit_code} passed={raw_result.passed}"
                    ),
                    details={
                        "timed_out": raw_result.timed_out,
                        "cancelled": raw_result.cancelled,
                        "duration_sec": round(raw_result.duration_sec, 4),
                    },
                )
            )
