"""Tests for StateStore and CheckpointManager (C1, C2, C3 cross-process resume, C4 workspace drift)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest

from harness import GuardedHarness
from schemas import (
    ApprovalStatus,
    BudgetConfig,
    RunStatus,
    StopReason,
    TaskType,
    WriteCommitStatus,
)
from tests.fake_model import FakeModel, ScriptedCall


def _create_buggy_calc_repo(workspace: Path) -> None:
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "src" / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a - b  # BUG\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_calc.py").write_text(
        "from src.calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )


def test_c1_checkpoint_saved_after_every_safe_step_with_complete_fields(tmp_path: Path) -> None:
    """C1: Every safe step saves a checkpoint containing run, step, plan, evidences, budget, and pending_approvals."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    _create_buggy_calc_repo(workspace)

    model = FakeModel(
        [
            ScriptedCall(
                tool_name="update_plan",
                arguments={
                    "goal": "Fix add() bug in src/calc.py",
                    "items": [
                        {"id": "1", "title": "Inspect src/calc.py", "status": "in_progress"},
                        {"id": "2", "title": "Patch bug and run pytest", "status": "pending"},
                    ],
                },
                tool_call_id="c1_plan",
            ),
            ScriptedCall(
                tool_name="read_file",
                arguments={"path": "src/calc.py"},
                tool_call_id="c1_read",
            ),
            ScriptedCall(
                tool_name="edit_file",
                arguments={
                    "path": "src/calc.py",
                    "old_text": "return a - b  # BUG",
                    "new_text": "return a + b",
                },
                tool_call_id="c1_edit",
            ),
        ]
    )

    harness = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model)
    result = harness.run(
        "Fix the failing test in src/calc.py",
        task_type=TaskType.TEST_FIX,
        run_id="run_c1",
    )

    assert result.status == RunStatus.WAITING_APPROVAL
    assert result.stop_reason == StopReason.WAITING_APPROVAL

    checkpoints = harness.checkpoint_manager.list_checkpoints("run_c1")
    # At least init + step 1 (plan) + step 2 (read) + waiting_approval + step 3
    assert len(checkpoints) >= 4

    latest = harness.checkpoint_manager.load_latest("run_c1")
    assert latest.run["run_id"] == "run_c1"
    assert latest.run["status"] == RunStatus.WAITING_APPROVAL.value
    assert latest.step == 3
    assert latest.plan["goal"] == "Fix add() bug in src/calc.py"
    assert len(latest.plan["items"]) == 2
    assert len(latest.evidences) >= 1
    assert latest.budget["steps_used"] == 3
    assert latest.budget["tool_calls_used"] == 2  # update_plan + read_file executed; edit_file paused
    assert latest.budget["total_tokens"] > 0
    assert len(latest.pending_approvals) == 1
    assert latest.pending_approvals[0]["tool_call_id"] == "c1_edit"

    # Full business state is also readable from StateStore
    loaded_state = harness.get_run_state("run_c1")
    assert loaded_state.status == RunStatus.WAITING_APPROVAL
    assert loaded_state.stop_reason == StopReason.WAITING_APPROVAL
    assert loaded_state.step_number == 3


def test_c2_crash_after_write_before_commit_marks_unknown_result_and_blocks_replay(tmp_path: Path) -> None:
    """C2: Process crashes after file write completes on disk but before result commit -> resume marks UNKNOWN_RESULT and forbids auto-replay."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    _create_buggy_calc_repo(workspace)

    class SimulatedCrashError(RuntimeError):
        pass

    # Phase 1: run until WAITING_APPROVAL on edit_file
    model1 = FakeModel(
        [
            ScriptedCall("read_file", {"path": "src/calc.py"}, tool_call_id="c2_read"),
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/calc.py",
                    "old_text": "return a - b  # BUG",
                    "new_text": "return a + b",
                },
                tool_call_id="c2_edit",
            ),
        ]
    )
    h1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model1)
    r1 = h1.run("Fix calc.py", task_type=TaskType.TEST_FIX, run_id="run_c2")
    assert r1.status == RunStatus.WAITING_APPROVAL
    approval_id = r1.state.pending_approvals()[0].approval_id

    # Human approves
    h1.approve("run_c2", approval_id, reviewer="bob")

    # Phase 2: resume with a fault-injection hook that crashes AFTER disk write BEFORE commit
    def crash_before_commit(tcid: str, tname: str, _res: object) -> None:
        raise SimulatedCrashError(f"Simulated crash right after {tname}({tcid}) wrote to disk!")

    model2 = FakeModel([ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="c2_never")])
    h2 = GuardedHarness(
        workspace_root=workspace,
        state_dir=state_dir,
        model=model2,
        on_after_write_before_commit=crash_before_commit,
    )
    with pytest.raises(SimulatedCrashError):
        h2.resume("run_c2")

    # Verify disk write DID happen, while WAL was left uncommitted (EXECUTING)
    assert "return a + b" in (workspace / "src" / "calc.py").read_text(encoding="utf-8")
    wal = h2.state_store.load_write_journal("run_c2")
    assert wal["c2_edit"].status == WriteCommitStatus.EXECUTING

    # Phase 3: fresh process/harness resumes -> MUST mark UNKNOWN_RESULT, stop, and NOT call model or replay write!
    model3 = FakeModel([ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="c2_should_not_run")])
    h3 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model3)
    r3 = h3.resume("run_c2")

    assert r3.status == RunStatus.WAITING_VERIFICATION
    assert r3.stop_reason == StopReason.UNKNOWN_WRITE_RESULT
    assert r3.state.write_journal["c2_edit"].status == WriteCommitStatus.UNKNOWN_RESULT
    assert model3.call_count == 0

    # Audit log explicitly records UNKNOWN_RESULT_GUARD
    unknown_events = h3.audit.query(run_id="run_c2", rule="UNKNOWN_RESULT_GUARD")
    assert len(unknown_events) >= 1
    assert unknown_events[0]["decision"] == "UNKNOWN_RESULT"
    assert unknown_events[0]["toolCallId"] == "c2_edit"


def test_c3_cross_process_resume_continues_without_repeating_writes_or_zeroing_counters(tmp_path: Path) -> None:
    """C3: Resume in a brand-new OS subprocess recovers progress, does not repeat completed writes, and preserves all counters."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    _create_buggy_calc_repo(workspace)

    # Process 1: read -> request edit -> pause at WAITING_APPROVAL -> approve -> resume 1 step (executes edit + pauses)
    model_p1 = FakeModel(
        [
            ScriptedCall("read_file", {"path": "src/calc.py"}, tool_call_id="c3_read"),
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/calc.py",
                    "old_text": "return a - b  # BUG",
                    "new_text": "return a + b",
                },
                tool_call_id="c3_edit",
            ),
        ]
    )
    h1 = GuardedHarness(
        workspace_root=workspace,
        state_dir=state_dir,
        model=model_p1,
    )
    r1 = h1.run(
        "Fix calc.py",
        task_type=TaskType.TEST_FIX,
        run_id="run_c3",
        budget=BudgetConfig(max_steps=10, max_tool_calls=20),
    )
    assert r1.status == RunStatus.WAITING_APPROVAL
    approval_id = r1.state.pending_approvals()[0].approval_id
    h1.approve("run_c3", approval_id, reviewer="alice")

    # Execute the approved edit and intentionally pause after 1 additional step (normal interruption)
    model_p1_b = FakeModel(
        [
            ScriptedCall("list_files", {"directory": "src"}, tool_call_id="c3_list"),
        ]
    )
    h1_b = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model_p1_b)
    r1_b = h1_b.resume("run_c3", pause_after_steps=1)
    assert r1_b.status == RunStatus.STOPPED
    assert r1_b.state.step_number == 3
    assert r1_b.state.budget.steps_used == 3
    assert r1_b.state.budget.writes_used == 1
    assert r1_b.state.budget.tool_calls_used == 3  # read + approved edit + list_files

    # Process 2: Launch an ACTUAL separate Python subprocess to resume run_c3!
    # The subprocess model first attempts to replay the already-committed edit (must be blocked!),
    # then runs pytest and finishes with final_answer.
    repo_root = Path(__file__).resolve().parents[1]
    worker_script = f"""
import json, sys
sys.path.insert(0, {str(repo_root)!r})
from harness import GuardedHarness
from tests.fake_model import FakeModel, ScriptedCall

model = FakeModel([
    ScriptedCall(
        "edit_file",
        {{"path": "src/calc.py", "old_text": "return a - b  # BUG", "new_text": "return a + b"}},
        tool_call_id="c3_edit",
    ),
    ScriptedCall("run_test", {{"command": "pytest -q"}}, tool_call_id="c3_test"),
    ScriptedCall("final_answer", {{"answer": "Fixed add() in src/calc.py and verified pytest passes."}}, tool_call_id="c3_done"),
])
h = GuardedHarness(workspace_root={str(workspace)!r}, state_dir={str(state_dir)!r}, model=model)
res = h.resume("run_c3")
print(json.dumps(res.to_dict()))
"""
    proc = subprocess.run(
        [sys.executable, "-c", worker_script],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["status"] == RunStatus.COMPLETED.value
    assert out["stop_reason"] == StopReason.COMPLETED.value
    # Counters continued monotonically (3 prior + 3 new steps = 6), never reset to 0!
    assert out["step_number"] == 6
    assert out["budget"]["steps_used"] == 6
    # Write was NOT duplicated (still exactly 1 write!)
    assert out["budget"]["writes_used"] == 1
    # Replay attempt in subprocess was blocked by Gate and audited
    reloaded = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=FakeModel([]))
    replay_denials = reloaded.audit.query(
        run_id="run_c3", tool_call_id="c3_edit", decision="DENY"
    )
    assert len(replay_denials) == 1
    assert replay_denials[0]["rule"] == "APPROVAL_ALREADY_CONSUMED"


def test_c4_workspace_change_before_resume_invalidates_evidences_and_approvals(tmp_path: Path) -> None:
    """C4: External workspace modification while paused invalidates old evidences & approvals and halts for verification."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    _create_buggy_calc_repo(workspace)

    model1 = FakeModel(
        [
            ScriptedCall("read_file", {"path": "src/calc.py"}, tool_call_id="c4_read"),
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/calc.py",
                    "old_text": "return a - b  # BUG",
                    "new_text": "return a + b",
                },
                tool_call_id="c4_edit",
            ),
        ]
    )
    h1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model1)
    r1 = h1.run("Fix calc.py", task_type=TaskType.TEST_FIX, run_id="run_c4")
    assert r1.status == RunStatus.WAITING_APPROVAL
    approval_id = r1.state.pending_approvals()[0].approval_id

    # Human approves while workspace matches checkpoint
    h1.approve("run_c4", approval_id, reviewer="alice")

    # External actor modifies workspace BEFORE resume!
    (workspace / "src" / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a * b  # EXTERNAL EDIT\n",
        encoding="utf-8",
    )

    # New process attempts to resume -> C4 MUST detect drift, invalidate evidence & approval, and STOP!
    model2 = FakeModel([ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="c4_never")])
    h2 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model2)
    r2 = h2.resume("run_c4")

    assert r2.status == RunStatus.WAITING_VERIFICATION
    assert r2.stop_reason == StopReason.WORKSPACE_CHANGED
    assert model2.call_count == 0
    assert r2.state.approvals[approval_id].status == ApprovalStatus.INVALIDATED
    assert all(ev.stale for ev in r2.state.evidences)
    # External file content must NOT have been overwritten by stale approved edit
    assert "EXTERNAL EDIT" in (workspace / "src" / "calc.py").read_text(encoding="utf-8")

    drift_events = h2.audit.query(run_id="run_c4", rule="WORKSPACE_DRIFT_GUARD")
    assert len(drift_events) == 1
    assert drift_events[0]["decision"] == "INVALIDATED"


def test_failed_edit_without_disk_mutation_does_not_poison_wal_on_resume(tmp_path: Path) -> None:
    """A mutating handler error with no disk mutation marks WAL FAILED so subsequent resume succeeds."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    _create_buggy_calc_repo(workspace)

    # 1. Model proposes an edit_file with a non-matching old_text -> pauses for approval
    model1 = FakeModel(
        [
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/calc.py",
                    "old_text": "NON_EXISTENT_LINE",
                    "new_text": "return a + b",
                },
                tool_call_id="wal_bad_edit",
            ),
        ]
    )
    h1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model1)
    r1 = h1.run("Fix calc.py", task_type=TaskType.TEST_FIX, run_id="run_wal_fail")
    assert r1.status == RunStatus.WAITING_APPROVAL
    bad_appr_id = r1.state.pending_approvals()[0].approval_id
    h1.approve("run_wal_fail", bad_appr_id, reviewer="alice")

    # 2. Resume executes bad edit -> ValueError (no disk change) -> WAL marked FAILED (not EXECUTING)
    #    Model then submits corrected edit -> pauses for approval again
    model2 = FakeModel(
        [
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/calc.py",
                    "old_text": "return a - b  # BUG",
                    "new_text": "return a + b",
                },
                tool_call_id="wal_good_edit",
            ),
        ]
    )
    h2 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model2)
    r2 = h2.resume("run_wal_fail")
    assert r2.status == RunStatus.WAITING_APPROVAL
    assert r2.state.write_journal["wal_bad_edit"].status == WriteCommitStatus.FAILED

    good_appr_id = r2.state.pending_approvals()[0].approval_id
    h2.approve("run_wal_fail", good_appr_id, reviewer="alice")

    # 3. Resume again -> must NOT misclassify wal_bad_edit as UNKNOWN_RESULT; run completes!
    model3 = FakeModel(
        [
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="wal_test"),
            ScriptedCall("final_answer", {"answer": "Fixed calc.py after retrying edit."}, tool_call_id="wal_done"),
        ]
    )
    h3 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model3)
    r3 = h3.resume("run_wal_fail")
    assert r3.completed is True
    assert r3.state.write_journal["wal_bad_edit"].status == WriteCommitStatus.FAILED
    assert r3.state.write_journal["wal_good_edit"].status == WriteCommitStatus.COMMITTED
