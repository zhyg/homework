"""Tests for ToolGate (G1, G2, G3, G4, permissions, and budget enforcement)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audit import AuditLogger
from checkpoint import CheckpointManager
from gate import ArgSpec, ToolGate, ToolPolicy
from sandbox import Sandbox, SandboxConfig
from schemas import (
    ApprovalStatus,
    BudgetConfig,
    BudgetState,
    PermissionMode,
    RiskLevel,
    RunState,
    RunStatus,
    StopReason,
    TaskType,
)
from state_store import StateStore


def _make_gate_env(
    tmp_path: Path,
    *,
    allowed_permissions: list[PermissionMode] | None = None,
    budget: BudgetConfig | None = None,
) -> tuple[ToolGate, RunState, Sandbox, StateStore, AuditLogger, dict[str, int]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "hello.txt").write_text("original content\n", encoding="utf-8")

    state_dir = tmp_path / "state"
    sandbox = Sandbox(SandboxConfig(workspace_root=workspace))
    audit = AuditLogger(state_dir / "audit.jsonl")
    state_store = StateStore(state_dir, audit=audit)
    ckpt_mgr = CheckpointManager(state_dir, audit=audit)

    snap, ws_hash, code_hash = sandbox.compute_workspace_snapshot()
    run_state = RunState(
        run_id="run_gate_test",
        task="gate unit test",
        task_type=TaskType.TEST_FIX,
        status=RunStatus.RUNNING,
        allowed_permissions=allowed_permissions
        or [
            PermissionMode.READ_ONLY,
            PermissionMode.WORKSPACE_WRITE,
            PermissionMode.TEST_EXEC,
        ],
        budget=BudgetState(limits=budget or BudgetConfig()),
        workspace_snapshot=snap,
        workspace_hash=ws_hash,
        code_hash=code_hash,
    )
    state_store.save_run(run_state)
    ckpt_mgr.save(run_state, reason="init")

    call_counts = {"read_file": 0, "write_file": 0}

    def read_handler(path: str, start_line: int | None = None) -> str:
        call_counts["read_file"] += 1
        return sandbox.read_file(path=path, start_line=start_line)

    def write_handler(path: str, content: str) -> dict[str, Any]:
        call_counts["write_file"] += 1
        return sandbox.write_file(path=path, content=content)

    gate = ToolGate(
        run_state=run_state,
        sandbox=sandbox,
        state_store=state_store,
        checkpoint_manager=ckpt_mgr,
        audit=audit,
        max_consecutive_duplicates=2,
    )
    gate.register(
        ToolPolicy(
            name="read_file",
            handler=read_handler,
            arg_schema={
                "path": ArgSpec(str, required=True, non_empty=True),
                "start_line": ArgSpec(int, required=False, allow_none=True),
            },
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            path_args=("path",),
            evidence_kind="READ",
        )
    )
    gate.register(
        ToolPolicy(
            name="write_file",
            handler=write_handler,
            arg_schema={
                "path": ArgSpec(str, required=True, non_empty=True),
                "content": ArgSpec(str, required=True),
            },
            permission=PermissionMode.WORKSPACE_WRITE,
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            is_mutating=True,
            path_args=("path",),
            evidence_kind="WRITE",
        )
    )
    return gate, run_state, sandbox, state_store, audit, call_counts


def test_g1_missing_or_wrong_type_arguments_blocked_before_handler(tmp_path: Path) -> None:
    """G1: Missing required field or wrong argument type returns structured error + tool_call_id; handler calls == 0."""
    gate, _state, _sandbox, _store, audit, counts = _make_gate_env(tmp_path)

    # Missing required 'path'
    r1 = gate.call("tc_g1_missing", "read_file", {})
    assert not r1.ok
    assert r1.status == "DENIED"
    assert r1.tool_call_id == "tc_g1_missing"
    assert r1.rule == "ARG_VALIDATION"
    assert "path" in r1.message
    assert counts["read_file"] == 0

    # Wrong type: int for str
    r2 = gate.call("tc_g1_bad_type", "read_file", {"path": 12345})
    assert not r2.ok
    assert r2.tool_call_id == "tc_g1_bad_type"
    assert r2.rule == "ARG_VALIDATION"
    assert counts["read_file"] == 0

    # Wrong type: bool for int (Python isinstance(True, int) pitfall)
    r3 = gate.call("tc_g1_bool_int", "read_file", {"path": "hello.txt", "start_line": True})
    assert not r3.ok
    assert r3.tool_call_id == "tc_g1_bool_int"
    assert r3.rule == "ARG_VALIDATION"
    assert counts["read_file"] == 0

    # Non-dict arguments
    r4 = gate.call("tc_g1_nondict", "read_file", "hello.txt")
    assert not r4.ok
    assert r4.tool_call_id == "tc_g1_nondict"
    assert counts["read_file"] == 0

    # Verify audit recorded every denial with original toolCallId
    denied = audit.query(run_id="run_gate_test", decision="DENY")
    assert {e["toolCallId"] for e in denied} >= {
        "tc_g1_missing",
        "tc_g1_bad_type",
        "tc_g1_bool_int",
        "tc_g1_nondict",
    }


def test_g2_three_consecutive_identical_calls_blocked_on_third(tmp_path: Path) -> None:
    """G2: Three consecutive identical tool+arg calls -> 3rd intercepted before handler; handler runs exactly twice."""
    gate, _state, _sandbox, _store, _audit, counts = _make_gate_env(tmp_path)

    r1 = gate.call("tc_dup_1", "read_file", {"path": "hello.txt"})
    r2 = gate.call("tc_dup_2", "read_file", {"path": "hello.txt"})
    r3 = gate.call("tc_dup_3", "read_file", {"path": "hello.txt"})

    assert r1.ok and r1.status == "EXECUTED"
    assert r2.ok and r2.status == "EXECUTED"
    assert not r3.ok
    assert r3.status == "DENIED"
    assert r3.tool_call_id == "tc_dup_3"
    assert r3.rule == "DUPLICATE_CALL_GUARD"
    assert counts["read_file"] == 2

    # Different arguments break the consecutive streak and succeed
    r4 = gate.call("tc_dup_4", "read_file", {"path": "hello.txt", "start_line": 1})
    assert r4.ok
    assert counts["read_file"] == 3


def test_g3_unapproved_write_enters_waiting_approval_without_modifying_file(tmp_path: Path) -> None:
    """G3: Unapproved write enters WAITING_APPROVAL, handler is not called, target file remains untouched."""
    gate, state, sandbox, _store, _audit, counts = _make_gate_env(tmp_path)
    target_file = sandbox.workspace_root / "hello.txt"

    resp = gate.call("tc_write_1", "write_file", {"path": "hello.txt", "content": "mutated\n"})

    assert not resp.ok
    assert resp.status == "WAITING_APPROVAL"
    assert resp.tool_call_id == "tc_write_1"
    assert resp.approval_id is not None
    assert resp.should_stop is True
    assert state.status == RunStatus.WAITING_APPROVAL
    assert state.stop_reason == StopReason.WAITING_APPROVAL
    assert counts["write_file"] == 0
    assert target_file.read_text(encoding="utf-8") == "original content\n"


def test_g4_approved_write_executes_once_and_blocks_replay(tmp_path: Path) -> None:
    """G4: After approval, 1st execution succeeds once; 2nd execution is rejected with no duplicate write."""
    gate, state, sandbox, store, _audit, counts = _make_gate_env(tmp_path)
    target_file = sandbox.workspace_root / "out.txt"

    # First call requests approval
    pending_resp = gate.call("tc_write_once", "write_file", {"path": "out.txt", "content": "v1\n"})
    assert pending_resp.status == "WAITING_APPROVAL"
    approval_id = pending_resp.approval_id
    assert approval_id is not None
    assert not target_file.exists()

    # Human approves
    store.approve(state.run_id, approval_id, reviewer="alice", note="looks safe")
    # Sync approved record into gate's in-memory run_state (simulating resume reload)
    state.approvals = store.load_run(state.run_id).approvals

    # 1st approved execution -> succeeds once
    first_exec = gate.call("tc_write_once", "write_file", {"path": "out.txt", "content": "v1\n"})
    assert first_exec.ok
    assert first_exec.status == "EXECUTED"
    assert counts["write_file"] == 1
    assert target_file.read_text(encoding="utf-8") == "v1\n"
    assert state.approvals[approval_id].status == ApprovalStatus.CONSUMED

    # 2nd execution with same tool_call_id -> DENIED, no duplicate write
    replay_same_id = gate.call("tc_write_once", "write_file", {"path": "out.txt", "content": "v1\n"})
    assert not replay_same_id.ok
    assert replay_same_id.status == "DENIED"
    assert replay_same_id.tool_call_id == "tc_write_once"
    assert replay_same_id.rule == "APPROVAL_ALREADY_CONSUMED"
    assert counts["write_file"] == 1

    # 2nd execution even with a new tool_call_id repeating the same consumed write -> DENIED
    replay_new_id = gate.call("tc_write_replay_2", "write_file", {"path": "out.txt", "content": "v1\n"})
    assert not replay_new_id.ok
    assert replay_new_id.status == "DENIED"
    assert replay_new_id.tool_call_id == "tc_write_replay_2"
    assert replay_new_id.rule == "APPROVAL_ALREADY_CONSUMED"
    assert counts["write_file"] == 1


def test_permission_and_budget_checks(tmp_path: Path) -> None:
    """Permission denial and tool-call budget limit block execution before handler."""
    gate, _state, _sandbox, _store, _audit, counts = _make_gate_env(
        tmp_path,
        allowed_permissions=[PermissionMode.READ_ONLY],
        budget=BudgetConfig(max_tool_calls=1),
    )

    # Write tool rejected due to READ_ONLY permission
    r_perm = gate.call("tc_perm", "write_file", {"path": "x.txt", "content": "nope"})
    assert not r_perm.ok
    assert r_perm.rule == "PERMISSION_CHECK"
    assert counts["write_file"] == 0

    # 1st read succeeds (uses 1/1 tool call budget)
    r_ok = gate.call("tc_ok", "read_file", {"path": "hello.txt"})
    assert r_ok.ok
    assert counts["read_file"] == 1

    # 2nd read hits budget limit before handler
    r_budget = gate.call("tc_budget", "read_file", {"path": "hello.txt", "start_line": 1})
    assert not r_budget.ok
    assert r_budget.rule == "BUDGET_LIMIT"
    assert r_budget.should_stop is True
    assert r_budget.stop_reason == StopReason.BUDGET_EXCEEDED
    assert counts["read_file"] == 1


def test_rejected_approval_blocks_execution_and_leaves_file_unchanged(tmp_path: Path) -> None:
    """Human rejection prevents write execution and leaves target file intact."""
    gate, state, sandbox, store, audit, counts = _make_gate_env(tmp_path)
    target = sandbox.workspace_root / "hello.txt"

    r1 = gate.call("tc_rej", "write_file", {"path": "hello.txt", "content": "bad change"})
    assert r1.status == "WAITING_APPROVAL"
    approval_id = r1.approval_id
    assert approval_id is not None

    store.reject(state.run_id, approval_id, reviewer="sec_reviewer", note="unsafe edit")
    state.approvals = store.load_run(state.run_id).approvals

    r2 = gate.call("tc_rej", "write_file", {"path": "hello.txt", "content": "bad change"})
    assert not r2.ok
    assert r2.status == "DENIED"
    assert r2.rule == "APPROVAL_REJECTED"
    assert counts["write_file"] == 0
    assert target.read_text(encoding="utf-8") == "original content\n"
    assert len(audit.query(run_id=state.run_id, stage="APPROVAL", decision="REJECTED")) == 1
