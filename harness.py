"""Guarded Coding Harness assembling smolagents ToolCallingAgent with Gate, Sandbox, State, Checkpoint, Audit, and Completion checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import uuid
from typing import Any, Callable

from smolagents import (
    ActionStep,
    AgentError,
    AgentMaxStepsError,
    AgentToolCallError,
    ChatMessage,
    FinalAnswerStep,
    LogLevel,
    Model,
    TaskStep,
    ToolCallingAgent,
)
from smolagents.memory import Timing, ToolCall

from audit import AuditLogger
from checkpoint import CheckpointManager
from gate import ToolGate
from sandbox import Sandbox, SandboxConfig
from schemas import (
    ApprovalRecord,
    ApprovalStatus,
    BudgetConfig,
    BudgetState,
    PermissionMode,
    RunState,
    RunStatus,
    StepRecord,
    StopReason,
    TaskType,
    utc_now_iso,
)
from state_store import StateStore
from tools import build_smolagents_tools, register_coding_tools


@dataclass
class HarnessRunResult:
    run_id: str
    status: RunStatus
    stop_reason: StopReason
    final_answer: str | None
    state: RunState
    checkpoint_id: str | None = None
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == RunStatus.COMPLETED and self.stop_reason == StopReason.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "completed": self.completed,
            "final_answer": self.final_answer,
            "checkpoint_id": self.checkpoint_id,
            "error": self.error,
            "step_number": self.state.step_number,
            "budget": self.state.budget.to_dict(),
            "pending_approvals": [a.to_dict() for a in self.state.pending_approvals()],
            "evidences_count": len(self.state.evidences),
        }


class CompletionChecker:
    """Enforces task-specific completion gates (H1 deliverables, H2 fresh test evidence)."""

    def __init__(
        self,
        *,
        run_state: RunState,
        sandbox: Sandbox,
        audit: AuditLogger,
    ) -> None:
        self.run_state = run_state
        self.sandbox = sandbox
        self.audit = audit

    def _reject(self, rule: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.audit.record(
            run_id=self.run_state.run_id,
            stage="COMPLETION",
            rule=rule,
            decision="REJECTED_FINAL_ANSWER",
            actor="harness",
            step_number=self.run_state.step_number,
            details={"message": message, **(details or {})},
        )
        raise ValueError(message)

    def _deliverable_exists_and_nonempty(self, rel_path: str) -> bool:
        try:
            real_path = self.sandbox.validate_path(rel_path, for_write=False)
        except Exception:
            return False
        return real_path.exists() and real_path.is_file() and real_path.stat().st_size > 0

    def check(self, final_answer: Any, memory: Any, agent: Any = None) -> bool:
        if not isinstance(final_answer, str) or not final_answer.strip():
            self._reject(
                "COMPLETION_EMPTY_ANSWER",
                "Final answer must be a non-empty summary.",
            )

        _, live_ws_hash, live_code_hash = self.sandbox.compute_workspace_snapshot()

        # 1. Explicit required deliverables check (H1)
        written_targets = {
            ev.target
            for ev in self.run_state.evidences
            if ev.kind in ("WRITE", "DELIVERABLE") and not ev.stale
        }
        for req_path in self.run_state.required_deliverables:
            if req_path not in written_targets or not self._deliverable_exists_and_nonempty(req_path):
                self._reject(
                    "COMPLETION_DELIVERABLE_MISSING",
                    f"Required deliverable '{req_path}' has not been produced or is empty.",
                    details={"missing_deliverable": req_path},
                )

        # 2. Task-type specific checks
        if self.run_state.task_type == TaskType.CODE_UNDERSTANDING:
            read_evidences = [
                ev for ev in self.run_state.evidences if ev.kind == "READ" and not ev.stale
            ]
            if not read_evidences:
                self._reject(
                    "COMPLETION_NO_READ_EVIDENCE",
                    "Code understanding task requires reading repository files before finishing.",
                )

            deliverable_evidences = [
                ev
                for ev in self.run_state.evidences
                if ev.kind in ("DELIVERABLE", "WRITE")
                and not ev.stale
                and self._deliverable_exists_and_nonempty(ev.target)
            ]
            if not deliverable_evidences:
                self._reject(
                    "COMPLETION_DELIVERABLE_MISSING",
                    "Code understanding task requires producing a non-empty deliverable artifact (e.g., artifacts/analysis.md).",
                )

        elif self.run_state.task_type == TaskType.TEST_FIX:
            write_evidences = [
                ev
                for ev in self.run_state.evidences
                if ev.kind in ("WRITE", "DELIVERABLE")
                and not ev.stale
                and self._deliverable_exists_and_nonempty(ev.target)
            ]
            if not write_evidences:
                self._reject(
                    "COMPLETION_DELIVERABLE_MISSING",
                    "Test-fix task requires at least one verified workspace modification before finishing.",
                )

            passing_tests = [
                ev
                for ev in self.run_state.evidences
                if ev.kind == "TEST_RUN" and ev.passed is True
            ]
            if not passing_tests:
                self._reject(
                    "COMPLETION_TEST_NOT_PASSED",
                    "Test-fix task requires at least one passing run_test execution before finishing.",
                )

            # H2: Passing test evidence must match the CURRENT source code hash and not be stale!
            fresh_passing = [
                ev
                for ev in passing_tests
                if not ev.stale and ev.code_hash == live_code_hash
            ]
            if not fresh_passing:
                self._reject(
                    "COMPLETION_STALE_TEST_EVIDENCE",
                    "Previous passing test evidence is stale because source code changed afterward; "
                    "you must re-run tests on the current version.",
                    details={
                        "live_code_hash": live_code_hash,
                        "last_passing_code_hash": passing_tests[-1].code_hash,
                    },
                )

        self.audit.record(
            run_id=self.run_state.run_id,
            stage="COMPLETION",
            rule="COMPLETION_VERIFIED",
            decision="ALLOW",
            actor="harness",
            step_number=self.run_state.step_number,
            details={
                "task_type": self.run_state.task_type.value,
                "live_workspace_hash": live_ws_hash,
                "live_code_hash": live_code_hash,
            },
        )
        return True


class GuardedToolCallingAgent(ToolCallingAgent):
    """Thin smolagents ToolCallingAgent extension that routes 100% of tool calls through ToolGate."""

    def __init__(
        self,
        *,
        gate: ToolGate,
        run_state: RunState,
        model: Model,
        step_callbacks: list[Callable[..., Any]],
        final_answer_checks: list[Callable[..., Any]],
        max_steps: int,
        instructions: str | None = None,
        verbosity_level: LogLevel = LogLevel.OFF,
    ) -> None:
        self.gate = gate
        self.run_state = run_state
        self._tool_call_id_queue: deque[str] = deque()
        self._pending_stop_reason: StopReason | None = None
        self._last_final_answer: str | None = None

        super().__init__(
            tools=build_smolagents_tools(),
            model=model,
            max_steps=max_steps,
            max_tool_threads=1,  # Deterministic sequential execution matching tool_call_id queue
            step_callbacks=step_callbacks,
            final_answer_checks=final_answer_checks,
            instructions=instructions,
            verbosity_level=verbosity_level,
        )

    def process_tool_calls(self, chat_message: ChatMessage, memory_step: ActionStep):
        self._tool_call_id_queue.clear()
        if chat_message.tool_calls:
            for tc in chat_message.tool_calls:
                self._tool_call_id_queue.append(str(tc.id))
        yield from super().process_tool_calls(chat_message, memory_step)

    def execute_tool_call(self, tool_name: str, arguments: dict[str, Any] | str) -> Any:
        tool_call_id = (
            self._tool_call_id_queue.popleft()
            if self._tool_call_id_queue
            else f"tc_{uuid.uuid4().hex[:10]}"
        )
        current_step = self.run_state.step_number + 1
        response = self.gate.call(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            step_number=current_step,
            actor="agent",
        )

        if response.should_stop:
            self._pending_stop_reason = response.stop_reason
            self.interrupt_switch = True

        if tool_name == "final_answer":
            if not response.ok:
                raise AgentToolCallError(response.to_observation_json(), self.logger)
            self._last_final_answer = str(response.output)
            return response.output

        return response.to_observation_json()

    def _handle_max_steps_reached(self, task: str) -> Any:
        """Override smolagents default so max-steps NEVER fabricates a final answer or makes an extra LLM call."""
        if self.interrupt_switch:
            raise AgentError("Agent interrupted.", self.logger)
        self._pending_stop_reason = StopReason.MAX_STEPS_EXCEEDED
        raise AgentMaxStepsError("Reached max steps without verified completion.", self.logger)


class GuardedHarness:
    """Single entry point assembling Model, ToolCallingAgent, Gate, Sandbox, State, Checkpoint, Audit, and Completion checks."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        state_dir: str | Path,
        model: Model,
        sandbox_config: SandboxConfig | None = None,
        require_write_approval: bool = True,
        max_consecutive_duplicates: int = 2,
        cancel_event: threading.Event | None = None,
        on_after_write_before_commit: Callable[[str, str, Any], None] | None = None,
        verbosity_level: LogLevel = LogLevel.OFF,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.require_write_approval = require_write_approval
        self.max_consecutive_duplicates = max_consecutive_duplicates
        self.cancel_event = cancel_event
        self.on_after_write_before_commit = on_after_write_before_commit
        self.verbosity_level = verbosity_level

        cfg = sandbox_config or SandboxConfig(workspace_root=self.workspace_root)
        self.sandbox = Sandbox(cfg)
        self.audit = AuditLogger(self.state_dir / "audit.jsonl")
        self.state_store = StateStore(self.state_dir, audit=self.audit)
        self.checkpoint_manager = CheckpointManager(self.state_dir, audit=self.audit)

    # ------------------------------------------------------------------
    # Public approval APIs (delegated to StateStore + Audit)
    # ------------------------------------------------------------------

    def approve(
        self,
        run_id: str,
        approval_id: str,
        *,
        reviewer: str = "human",
        note: str = "",
    ) -> ApprovalRecord:
        return self.state_store.approve(run_id, approval_id, reviewer=reviewer, note=note)

    def reject(
        self,
        run_id: str,
        approval_id: str,
        *,
        reviewer: str = "human",
        note: str = "",
    ) -> ApprovalRecord:
        return self.state_store.reject(run_id, approval_id, reviewer=reviewer, note=note)

    def get_run_state(self, run_id: str) -> RunState:
        return self.state_store.load_run(run_id)

    # ------------------------------------------------------------------
    # Primary Run & Resume Entry Points
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        task_type: TaskType = TaskType.CODE_UNDERSTANDING,
        run_id: str | None = None,
        required_deliverables: list[str] | None = None,
        allowed_permissions: list[PermissionMode] | None = None,
        budget: BudgetConfig | None = None,
        pause_after_steps: int | None = None,
    ) -> HarnessRunResult:
        """Start a brand-new guarded coding run."""
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        snap, ws_hash, code_hash = self.sandbox.compute_workspace_snapshot()

        state = RunState(
            run_id=rid,
            task=task,
            task_type=task_type,
            status=RunStatus.RUNNING,
            stop_reason=StopReason.NONE,
            allowed_permissions=list(
                allowed_permissions
                or [
                    PermissionMode.READ_ONLY,
                    PermissionMode.WORKSPACE_WRITE,
                    PermissionMode.TEST_EXEC,
                ]
            ),
            required_deliverables=list(required_deliverables or []),
            budget=BudgetState(limits=budget or BudgetConfig()),
            workspace_snapshot=snap,
            workspace_hash=ws_hash,
            code_hash=code_hash,
        )
        state.plan.goal = task
        self.state_store.save_run(state)
        self.checkpoint_manager.save(state, reason="run_initialized")
        self.audit.record(
            run_id=rid,
            stage="HARNESS",
            rule="RUN_STARTED",
            decision="STARTED",
            actor="harness",
            step_number=0,
            details={
                "task_type": task_type.value,
                "required_deliverables": state.required_deliverables,
                "budget": state.budget.limits.to_dict(),
                "workspace_hash": ws_hash,
            },
        )
        return self._drive_loop(state, is_resume=False, pause_after_steps=pause_after_steps)

    def resume(
        self,
        run_id: str,
        *,
        pause_after_steps: int | None = None,
    ) -> HarnessRunResult:
        """Resume a previously paused or interrupted run from its latest safe checkpoint."""
        outcome = self.checkpoint_manager.restore(
            run_id,
            sandbox=self.sandbox,
            state_store=self.state_store,
        )
        state = outcome.state
        if not outcome.can_continue:
            latest = self.checkpoint_manager.load_latest(run_id)
            return HarnessRunResult(
                run_id=run_id,
                status=state.status,
                stop_reason=state.stop_reason,
                final_answer=state.final_answer,
                state=state,
                checkpoint_id=latest.checkpoint_id,
                error=state.last_error,
            )

        if state.status == RunStatus.COMPLETED:
            latest = self.checkpoint_manager.load_latest(run_id)
            return HarnessRunResult(
                run_id=run_id,
                status=state.status,
                stop_reason=state.stop_reason,
                final_answer=state.final_answer,
                state=state,
                checkpoint_id=latest.checkpoint_id,
            )

        # If there are still pending approvals that haven't been decided, stay paused.
        if state.pending_approvals():
            state.status = RunStatus.WAITING_APPROVAL
            state.stop_reason = StopReason.WAITING_APPROVAL
            self.state_store.save_run(state)
            latest = self.checkpoint_manager.load_latest(run_id)
            return HarnessRunResult(
                run_id=run_id,
                status=state.status,
                stop_reason=state.stop_reason,
                final_answer=None,
                state=state,
                checkpoint_id=latest.checkpoint_id,
                error="Run still has pending approval(s) awaiting review.",
            )

        # Build gate & execute any newly APPROVED unconsumed pending actions ONCE before continuing LLM loop.
        gate = self._build_gate(state)
        approved_list = state.approved_unconsumed_approvals()
        if approved_list:
            state.status = RunStatus.RUNNING
            state.stop_reason = StopReason.NONE
            for appr in approved_list:
                gate_resp = gate.call(
                    tool_call_id=appr.tool_call_id,
                    tool_name=appr.tool_name,
                    arguments=appr.arguments,
                    step_number=state.step_number,
                    actor="harness_resume",
                )
                # Append observation to run step history so the resumed agent sees that the approved write succeeded!
                state.steps.append(
                    StepRecord(
                        step_number=state.step_number,
                        model_output=f"[Resumed approved action {appr.approval_id}]",
                        tool_calls=[
                            {
                                "id": appr.tool_call_id,
                                "name": appr.tool_name,
                                "arguments": appr.arguments,
                            }
                        ],
                        observations=gate_resp.to_observation_json(),
                        error=None if gate_resp.ok else gate_resp.message,
                    )
                )
                self.state_store.save_run(state)
                ckpt = self.checkpoint_manager.save(state, reason="approved_action_executed")
                if gate_resp.should_stop:
                    return HarnessRunResult(
                        run_id=run_id,
                        status=state.status,
                        stop_reason=state.stop_reason,
                        final_answer=None,
                        state=state,
                        checkpoint_id=ckpt.checkpoint_id,
                        error=gate_resp.message,
                    )

        # Check remaining budget before launching next model step
        budget_err = state.budget.exceeded_reason()
        if budget_err:
            state.status = RunStatus.STOPPED
            state.stop_reason = StopReason.BUDGET_EXCEEDED
            state.last_error = budget_err
            self.state_store.save_run(state)
            ckpt = self.checkpoint_manager.save(state, reason="budget_exceeded")
            return HarnessRunResult(
                run_id=run_id,
                status=state.status,
                stop_reason=state.stop_reason,
                final_answer=None,
                state=state,
                checkpoint_id=ckpt.checkpoint_id,
                error=budget_err,
            )

        if state.budget.steps_used >= state.budget.limits.max_steps:
            state.status = RunStatus.STOPPED
            state.stop_reason = StopReason.MAX_STEPS_EXCEEDED
            state.last_error = "Maximum step limit already reached."
            self.state_store.save_run(state)
            ckpt = self.checkpoint_manager.save(state, reason="max_steps_exceeded")
            return HarnessRunResult(
                run_id=run_id,
                status=state.status,
                stop_reason=state.stop_reason,
                final_answer=None,
                state=state,
                checkpoint_id=ckpt.checkpoint_id,
                error=state.last_error,
            )

        state.status = RunStatus.RUNNING
        state.stop_reason = StopReason.NONE
        self.state_store.save_run(state)
        return self._drive_loop(state, is_resume=True, pause_after_steps=pause_after_steps, gate=gate)

    # ------------------------------------------------------------------
    # Internal assembly & smolagents execution loop
    # ------------------------------------------------------------------

    def _build_gate(self, state: RunState) -> ToolGate:
        gate = ToolGate(
            run_state=state,
            sandbox=self.sandbox,
            state_store=self.state_store,
            checkpoint_manager=self.checkpoint_manager,
            audit=self.audit,
            max_consecutive_duplicates=self.max_consecutive_duplicates,
            cancel_event=self.cancel_event,
            on_after_write_before_commit=self.on_after_write_before_commit,
        )
        register_coding_tools(
            gate=gate,
            sandbox=self.sandbox,
            run_state=state,
            cancel_event=self.cancel_event,
            require_write_approval=self.require_write_approval,
        )
        return gate

    def _restore_agent_memory(self, agent: GuardedToolCallingAgent, state: RunState) -> None:
        """Rehydrate smolagents AgentMemory from checkpointed StepRecords."""
        agent.memory.reset()
        agent.memory.steps.append(TaskStep(task=state.task))
        for rec in state.steps:
            tcs = [
                ToolCall(
                    name=str(tc.get("name", "")),
                    arguments=tc.get("arguments", {}),
                    id=str(tc.get("id", "")),
                )
                for tc in rec.tool_calls
            ] or None
            restored_step = ActionStep(
                step_number=rec.step_number,
                timing=Timing(start_time=0.0, end_time=0.0),
                model_output=rec.model_output,
                tool_calls=tcs,
                observations=rec.observations,
                is_final_answer=rec.is_final_answer,
            )
            agent.memory.steps.append(restored_step)

    def _build_instructions(self, state: RunState) -> str:
        deliverables = (
            f"Required deliverables: {', '.join(state.required_deliverables)}."
            if state.required_deliverables
            else "Produce a clear deliverable artifact when required."
        )
        return (
            f"You are operating inside a Guarded Coding Harness for a '{state.task_type.value}' task. "
            f"{deliverables} "
            "All tool calls are validated by a gate and sandbox. "
            "Do not repeat identical tool calls; if a write requires approval, wait cleanly. "
            "Before calling final_answer, ensure required deliverables exist and (for test_fix) "
            "run_test passes on the latest source code."
        )

    def _drive_loop(
        self,
        state: RunState,
        *,
        is_resume: bool,
        pause_after_steps: int | None = None,
        gate: ToolGate | None = None,
    ) -> HarnessRunResult:
        active_gate = gate or self._build_gate(state)
        completion_checker = CompletionChecker(
            run_state=state,
            sandbox=self.sandbox,
            audit=self.audit,
        )

        slice_steps_executed = 0
        last_checkpoint_id: str | None = None

        def on_step_finalized(memory_step: Any, agent: GuardedToolCallingAgent | None = None) -> None:
            nonlocal slice_steps_executed, last_checkpoint_id
            if isinstance(memory_step, FinalAnswerStep):
                return
            if not isinstance(memory_step, ActionStep):
                return

            slice_steps_executed += 1
            state.step_number += 1
            state.budget.steps_used += 1

            in_tok = 0
            out_tok = 0
            if memory_step.token_usage is not None:
                in_tok = int(memory_step.token_usage.input_tokens or 0)
                out_tok = int(memory_step.token_usage.output_tokens or 0)
                state.budget.input_tokens += in_tok
                state.budget.output_tokens += out_tok
                state.budget.total_tokens += in_tok + out_tok

            serialized_calls = []
            if memory_step.tool_calls:
                for tc in memory_step.tool_calls:
                    serialized_calls.append(
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    )

            state.steps.append(
                StepRecord(
                    step_number=state.step_number,
                    timestamp=utc_now_iso(),
                    model_output=memory_step.model_output,
                    tool_calls=serialized_calls,
                    observations=memory_step.observations,
                    error=str(memory_step.error) if memory_step.error else None,
                    is_final_answer=bool(memory_step.is_final_answer),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
            )

            # Determine post-step transition
            if memory_step.is_final_answer:
                state.status = RunStatus.COMPLETED
                state.stop_reason = StopReason.COMPLETED
                state.final_answer = (
                    agent._last_final_answer
                    if agent is not None and agent._last_final_answer is not None
                    else str(memory_step.action_output or "")
                )
            elif agent is not None and agent._pending_stop_reason is not None:
                reason = agent._pending_stop_reason
                state.stop_reason = reason
                if reason == StopReason.WAITING_APPROVAL:
                    state.status = RunStatus.WAITING_APPROVAL
                elif reason in (StopReason.WORKSPACE_CHANGED, StopReason.UNKNOWN_WRITE_RESULT):
                    state.status = RunStatus.WAITING_VERIFICATION
                else:
                    state.status = RunStatus.STOPPED
                agent.interrupt_switch = True
            else:
                budget_err = state.budget.exceeded_reason()
                if budget_err is not None:
                    state.status = RunStatus.STOPPED
                    state.stop_reason = StopReason.BUDGET_EXCEEDED
                    state.last_error = budget_err
                    if agent is not None:
                        agent.interrupt_switch = True
                    self.audit.record(
                        run_id=state.run_id,
                        stage="HARNESS",
                        rule="BUDGET_LIMIT",
                        decision="STOPPED",
                        actor="harness",
                        step_number=state.step_number,
                        details={"reason": budget_err},
                    )
                elif state.budget.steps_used >= state.budget.limits.max_steps:
                    state.status = RunStatus.STOPPED
                    state.stop_reason = StopReason.MAX_STEPS_EXCEEDED
                    state.last_error = (
                        f"Reached max_steps limit ({state.budget.steps_used}/"
                        f"{state.budget.limits.max_steps}) without verified completion."
                    )
                    if agent is not None:
                        agent.interrupt_switch = True
                    self.audit.record(
                        run_id=state.run_id,
                        stage="HARNESS",
                        rule="MAX_STEPS_LIMIT",
                        decision="STOPPED",
                        actor="harness",
                        step_number=state.step_number,
                        details={"steps_used": state.budget.steps_used},
                    )
                elif self.cancel_event is not None and self.cancel_event.is_set():
                    state.status = RunStatus.STOPPED
                    state.stop_reason = StopReason.USER_CANCELLED
                    state.last_error = "Run cancelled by user."
                    if agent is not None:
                        agent.interrupt_switch = True
                elif pause_after_steps is not None and slice_steps_executed >= pause_after_steps:
                    state.status = RunStatus.STOPPED
                    state.stop_reason = StopReason.USER_CANCELLED
                    state.last_error = f"Paused after {slice_steps_executed} step(s) as requested."
                    if agent is not None:
                        agent.interrupt_switch = True

            # C1: Save full state and safe-step checkpoint after EVERY finalized step
            self.state_store.save_run(state)
            ckpt = self.checkpoint_manager.save(state, reason="safe_step")
            last_checkpoint_id = ckpt.checkpoint_id

        remaining_steps = max(state.budget.limits.max_steps - state.budget.steps_used, 1)
        agent = GuardedToolCallingAgent(
            gate=active_gate,
            run_state=state,
            model=self.model,
            step_callbacks=[on_step_finalized],
            final_answer_checks=[completion_checker.check],
            max_steps=remaining_steps,
            instructions=self._build_instructions(state),
            verbosity_level=self.verbosity_level,
        )

        if is_resume:
            self._restore_agent_memory(agent, state)
            prompt = (
                f"Resuming run '{state.run_id}' from step {state.step_number}. "
                f"Original task: {state.task}. Continue from the recorded tool results and do not repeat completed writes."
            )
        else:
            prompt = state.task

        try:
            final_output = agent.run(prompt, reset=not is_resume, max_steps=remaining_steps)
            if state.status == RunStatus.COMPLETED:
                if state.final_answer is None and final_output is not None:
                    state.final_answer = str(final_output)
                self.state_store.save_run(state)
                ckpt = self.checkpoint_manager.save(state, reason="run_completed")
                last_checkpoint_id = ckpt.checkpoint_id
                self.audit.record(
                    run_id=state.run_id,
                    stage="HARNESS",
                    rule="RUN_COMPLETED",
                    decision="COMPLETED",
                    actor="harness",
                    step_number=state.step_number,
                    details={"final_answer_preview": (state.final_answer or "")[:200]},
                )
        except AgentMaxStepsError as exc:
            state.status = RunStatus.STOPPED
            state.stop_reason = StopReason.MAX_STEPS_EXCEEDED
            state.last_error = str(exc)
            self.state_store.save_run(state)
            ckpt = self.checkpoint_manager.save(state, reason="max_steps_exceeded")
            last_checkpoint_id = ckpt.checkpoint_id
        except AgentError as exc:
            # Expected when agent.interrupt_switch halts the loop cleanly for WAITING_APPROVAL,
            # BUDGET_EXCEEDED, MAX_STEPS_EXCEEDED, WORKSPACE_CHANGED, or USER_CANCELLED.
            if state.stop_reason == StopReason.NONE:
                state.status = RunStatus.FAILED
                state.stop_reason = StopReason.ERROR
                state.last_error = str(exc)
            self.state_store.save_run(state)
            ckpt = self.checkpoint_manager.save(state, reason=f"stopped:{state.stop_reason.value}")
            last_checkpoint_id = ckpt.checkpoint_id
            self.audit.record(
                run_id=state.run_id,
                stage="HARNESS",
                rule="RUN_STOPPED",
                decision="STOPPED",
                actor="harness",
                step_number=state.step_number,
                details={
                    "status": state.status.value,
                    "stop_reason": state.stop_reason.value,
                    "error": state.last_error,
                },
            )

        if last_checkpoint_id is None:
            last_checkpoint_id = self.checkpoint_manager.load_latest(state.run_id).checkpoint_id

        return HarnessRunResult(
            run_id=state.run_id,
            status=state.status,
            stop_reason=state.stop_reason,
            final_answer=state.final_answer if state.status == RunStatus.COMPLETED else None,
            state=state,
            checkpoint_id=last_checkpoint_id,
            error=state.last_error,
        )
