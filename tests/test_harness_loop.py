"""End-to-end tests for GuardedHarness loop with smolagents ToolCallingAgent and FakeModel (including H3)."""

from __future__ import annotations

from pathlib import Path
import pytest

from harness import GuardedHarness
from schemas import BudgetConfig, RunStatus, StopReason, TaskType
from tests.fake_model import FakeModel, ScriptedCall
from tools import ReadFileTool, WriteFileTool


def test_code_understanding_end_to_end_with_approval_and_deliverable(tmp_path: Path) -> None:
    """Task Type 1 (代码理解): inspects repo, writes deliverable after approval, and completes cleanly."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "service.py").write_text(
        "class BillingService:\n    def charge(self, amount: int) -> int:\n        return amount\n",
        encoding="utf-8",
    )

    model1 = FakeModel(
        [
            ScriptedCall("list_files", {"directory": "."}, tool_call_id="cu_list"),
            ScriptedCall("read_file", {"path": "src/service.py"}, tool_call_id="cu_read"),
            ScriptedCall(
                "write_file",
                {
                    "path": "artifacts/architecture.md",
                    "content": "# Architecture Report\n\n`BillingService.charge` returns `amount`.\n",
                },
                tool_call_id="cu_write",
            ),
        ]
    )

    h1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model1)
    r1 = h1.run(
        "Understand the codebase and write an architecture report to artifacts/architecture.md",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_code_understanding",
        required_deliverables=["artifacts/architecture.md"],
    )

    assert r1.status == RunStatus.WAITING_APPROVAL
    assert r1.stop_reason == StopReason.WAITING_APPROVAL
    assert not (workspace / "artifacts" / "architecture.md").exists()

    approval_id = r1.state.pending_approvals()[0].approval_id
    h1.approve("run_code_understanding", approval_id, reviewer="lead_eng")

    model2 = FakeModel(
        [
            ScriptedCall(
                "final_answer",
                {"answer": "Completed analysis and saved report to artifacts/architecture.md."},
                tool_call_id="cu_final",
            )
        ]
    )
    h2 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model2)
    r2 = h2.resume("run_code_understanding")

    assert r2.completed is True
    assert r2.status == RunStatus.COMPLETED
    assert r2.stop_reason == StopReason.COMPLETED
    assert (workspace / "artifacts" / "architecture.md").exists()
    assert "BillingService" in (workspace / "artifacts" / "architecture.md").read_text(encoding="utf-8")


def test_test_fix_end_to_end_with_bad_calls_and_recovery(tmp_path: Path) -> None:
    """Task Type 2 (测试修复): model hits Gate/Sandbox errors, observes structured errors, fixes bug, and passes pytest."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "src" / "parser.py").write_text(
        "def parse_port(raw: str) -> int:\n    return int(raw) + 1  # off-by-one bug\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_parser.py").write_text(
        "from src.parser import parse_port\n\ndef test_parse_port():\n    assert parse_port('8080') == 8080\n",
        encoding="utf-8",
    )

    model1 = FakeModel(
        [
            # Failing test reproduces bug
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="tf_t1"),
            # Model tries a bad argument (G1) -> Gate returns structured error with tool_call_id
            ScriptedCall("read_file", {"path": 999}, tool_call_id="tf_bad_arg"),
            # Model tries path escape (S1) -> Sandbox rejects
            ScriptedCall("read_file", {"path": "artifacts/../../.env"}, tool_call_id="tf_escape"),
            # Model reads actual file
            ScriptedCall("read_file", {"path": "src/parser.py"}, tool_call_id="tf_read"),
            # Model requests patch -> pauses for approval (G3)
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/parser.py",
                    "old_text": "return int(raw) + 1  # off-by-one bug",
                    "new_text": "return int(raw)",
                },
                tool_call_id="tf_edit",
            ),
        ]
    )

    h1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model1)
    r1 = h1.run("Fix failing test_parse_port", task_type=TaskType.TEST_FIX, run_id="run_test_fix")
    assert r1.status == RunStatus.WAITING_APPROVAL

    # Verify model received structured error observations with original tool_call_id on subsequent turns
    all_obs = "\n".join(s.observations or "" for s in r1.state.steps)
    assert "tf_bad_arg" in all_obs
    assert "ARG_VALIDATION" in all_obs
    assert "tf_escape" in all_obs
    assert "SANDBOX_PATH_ESCAPE" in all_obs

    appr_id = r1.state.pending_approvals()[0].approval_id
    h1.approve("run_test_fix", appr_id, reviewer="qa")

    model2 = FakeModel(
        [
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="tf_t2"),
            ScriptedCall(
                "final_answer",
                {"answer": "Fixed off-by-one bug in parse_port() and verified pytest passes."},
                tool_call_id="tf_done",
            ),
        ]
    )
    h2 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model2)
    r2 = h2.resume("run_test_fix")

    assert r2.completed is True
    assert r2.final_answer is not None and "parse_port" in r2.final_answer
    assert "return int(raw)\n" in (workspace / "src" / "parser.py").read_text(encoding="utf-8")


def test_h3_max_steps_stops_with_explicit_stop_reason_without_faking_success(tmp_path: Path) -> None:
    """H3 (max_steps): Reaching max_steps stops explicitly with MAX_STEPS_EXCEEDED and never fakes completion."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
    (workspace / "b.txt").write_text("beta\n", encoding="utf-8")
    (workspace / "c.txt").write_text("gamma\n", encoding="utf-8")

    # Script has 4 turns, but budget limits max_steps=2
    model = FakeModel(
        [
            ScriptedCall("read_file", {"path": "a.txt"}, tool_call_id="ms_1"),
            ScriptedCall("read_file", {"path": "b.txt"}, tool_call_id="ms_2"),
            ScriptedCall("read_file", {"path": "c.txt"}, tool_call_id="ms_3"),
            ScriptedCall("final_answer", {"answer": "done"}, tool_call_id="ms_4"),
        ]
    )

    harness = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model)
    res = harness.run(
        "Read files and write report",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_h3_steps",
        budget=BudgetConfig(max_steps=2),
    )

    assert not res.completed
    assert res.status == RunStatus.STOPPED
    assert res.stop_reason == StopReason.MAX_STEPS_EXCEEDED
    assert res.final_answer is None
    assert res.state.budget.steps_used == 2
    # Must NOT have made an extra unbudgeted LLM call (like smolagents' default provide_final_answer)
    assert model.call_count == 2


def test_h3_token_budget_exceeded_stops_with_explicit_stop_reason(tmp_path: Path) -> None:
    """H3 (token budget): Exceeding token budget stops explicitly with BUDGET_EXCEEDED and never fakes completion."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    (workspace / "a.txt").write_text("alpha\n", encoding="utf-8")

    model = FakeModel(
        [
            ScriptedCall(
                "read_file",
                {"path": "a.txt"},
                tool_call_id="tb_1",
                input_tokens=400,
                output_tokens=200,
            ),
            ScriptedCall("read_file", {"path": "a.txt"}, tool_call_id="tb_2"),
        ]
    )

    harness = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model)
    res = harness.run(
        "Read a.txt",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_h3_tokens",
        budget=BudgetConfig(max_steps=10, max_tokens=500),
    )

    assert not res.completed
    assert res.status == RunStatus.STOPPED
    assert res.stop_reason == StopReason.BUDGET_EXCEEDED
    assert res.final_answer is None
    assert res.state.budget.total_tokens == 600
    assert model.call_count == 1


def test_direct_tool_bypass_is_forbidden() -> None:
    """Tools exposed to smolagents cannot be executed directly bypassing ToolGate."""
    with pytest.raises(RuntimeError, match="ToolGate"):
        ReadFileTool().forward(path="anything.txt")
    with pytest.raises(RuntimeError, match="ToolGate"):
        WriteFileTool().forward(path="x.txt", content="bypass")


def test_duplicate_loop_guard_inside_agent_loop_recovers(tmp_path: Path) -> None:
    """Inside full ToolCallingAgent loop, 3rd consecutive identical call is blocked by Gate and model recovers."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    (workspace / "info.py").write_text("VERSION = '1.0'\n", encoding="utf-8")

    model = FakeModel(
        [
            ScriptedCall("read_file", {"path": "info.py"}, tool_call_id="dup_loop_1"),
            ScriptedCall("read_file", {"path": "info.py"}, tool_call_id="dup_loop_2"),
            ScriptedCall("read_file", {"path": "info.py"}, tool_call_id="dup_loop_3"),
            ScriptedCall(
                "write_file",
                {"path": "artifacts/info.md", "content": "# Info\nVERSION=1.0\n"},
                tool_call_id="dup_loop_write",
            ),
            ScriptedCall(
                "final_answer",
                {"answer": "Saved version report to artifacts/info.md."},
                tool_call_id="dup_loop_done",
            ),
        ]
    )

    harness = GuardedHarness(
        workspace_root=workspace,
        state_dir=state_dir,
        model=model,
        require_write_approval=False,
    )
    res = harness.run(
        "Read info.py and write artifacts/info.md",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_dup_loop",
        required_deliverables=["artifacts/info.md"],
    )

    assert res.completed is True
    # Only 2 READ evidences recorded (3rd identical call was intercepted before handler)
    read_evidences = [ev for ev in res.state.evidences if ev.kind == "READ"]
    assert len(read_evidences) == 2
    denials = harness.audit.query(run_id="run_dup_loop", rule="DUPLICATE_CALL_GUARD")
    assert len(denials) == 1
    assert denials[0]["toolCallId"] == "dup_loop_3"
