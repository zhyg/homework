"""Tests for CompletionChecker final-answer gates (H1 missing deliverables, H2 stale test evidence)."""

from __future__ import annotations

from pathlib import Path

from harness import GuardedHarness
from schemas import BudgetConfig, RunStatus, StopReason, TaskType
from tests.fake_model import FakeModel, ScriptedCall


def test_h1_final_answer_rejected_when_no_deliverable_exists(tmp_path: Path) -> None:
    """H1: Model claims completion without producing deliverable -> final_answer rejected; cannot enter COMPLETED."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")

    # Model reads file and immediately tries to finish twice WITHOUT writing required deliverable
    model = FakeModel(
        [
            ScriptedCall("read_file", {"path": "app.py"}, tool_call_id="h1_read"),
            ScriptedCall(
                "final_answer",
                {"answer": "I am done analyzing app.py!"},
                tool_call_id="h1_premature_1",
            ),
            ScriptedCall(
                "final_answer",
                {"answer": "Really done now!"},
                tool_call_id="h1_premature_2",
            ),
        ]
    )

    harness = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model)
    res = harness.run(
        "Analyze app.py and write artifacts/report.md",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_h1",
        required_deliverables=["artifacts/report.md"],
        budget=BudgetConfig(max_steps=3),
    )

    # Cannot enter COMPLETED! Stops cleanly with MAX_STEPS_EXCEEDED instead.
    assert not res.completed
    assert res.status == RunStatus.STOPPED
    assert res.stop_reason == StopReason.MAX_STEPS_EXCEEDED
    assert res.final_answer is None

    # Audit log proves final_answer was explicitly rejected for missing deliverable
    rejects = harness.audit.query(
        run_id="run_h1",
        stage="COMPLETION",
        decision="REJECTED_FINAL_ANSWER",
    )
    assert len(rejects) == 2
    assert all(r["rule"] == "COMPLETION_DELIVERABLE_MISSING" for r in rejects)


def test_h1_recovers_after_rejection_once_deliverable_is_written(tmp_path: Path) -> None:
    """H1 recovery: After final_answer is rejected, creating the deliverable allows final_answer to succeed."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 42\n", encoding="utf-8")

    # Disable interactive write approval here to test single-loop rejection -> write -> completion cleanly
    model = FakeModel(
        [
            ScriptedCall("read_file", {"path": "app.py"}, tool_call_id="h1r_read"),
            ScriptedCall("final_answer", {"answer": "Premature claim"}, tool_call_id="h1r_bad"),
            ScriptedCall(
                "write_file",
                {"path": "artifacts/report.md", "content": "# Report\nVALUE is 42.\n"},
                tool_call_id="h1r_write",
            ),
            ScriptedCall(
                "final_answer",
                {"answer": "Report written to artifacts/report.md."},
                tool_call_id="h1r_good",
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
        "Write artifacts/report.md",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_h1_recover",
        required_deliverables=["artifacts/report.md"],
    )

    assert res.completed is True
    assert res.status == RunStatus.COMPLETED
    rejects = harness.audit.query(run_id="run_h1_recover", rule="COMPLETION_DELIVERABLE_MISSING")
    assert len(rejects) == 1
    assert (workspace / "artifacts" / "report.md").exists()


def test_h2_modifying_source_after_passing_test_invalidates_test_evidence(tmp_path: Path) -> None:
    """H2: Source modified after passing test -> old test evidence becomes stale; final_answer rejected until tests re-run."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "src" / "calc.py").write_text(
        "def sub(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_calc.py").write_text(
        "from src.calc import sub\n\ndef test_sub():\n    assert sub(5, 2) == 3\n",
        encoding="utf-8",
    )

    model = FakeModel(
        [
            # 1. Fix bug
            ScriptedCall(
                "edit_file",
                {"path": "src/calc.py", "old_text": "return a + b", "new_text": "return a - b"},
                tool_call_id="h2_edit_1",
            ),
            # 2. Run tests -> PASSES on version 1
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="h2_test_1"),
            # 3. Modify source code AGAIN after passing tests! (invalidates h2_test_1)
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/calc.py",
                    "old_text": "return a - b",
                    "new_text": "return a - b  # refactored comment",
                },
                tool_call_id="h2_edit_2",
            ),
            # 4. Try to finish WITHOUT re-running tests -> MUST BE REJECTED (H2)!
            ScriptedCall(
                "final_answer",
                {"answer": "Done without re-running tests"},
                tool_call_id="h2_stale_final",
            ),
            # 5. Re-run tests on the latest source version -> PASSES
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="h2_test_2"),
            # 6. Now final_answer MUST succeed!
            ScriptedCall(
                "final_answer",
                {"answer": "Fixed sub(), refactored, and re-verified pytest passes on latest version."},
                tool_call_id="h2_fresh_final",
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
        "Fix sub() in src/calc.py",
        task_type=TaskType.TEST_FIX,
        run_id="run_h2",
    )

    assert res.completed is True
    assert res.status == RunStatus.COMPLETED

    # Verify the first test evidence was marked stale and final_answer was rejected once due to stale test evidence
    test_evidences = [ev for ev in res.state.evidences if ev.kind == "TEST_RUN"]
    assert len(test_evidences) == 2
    assert test_evidences[0].stale is True
    assert test_evidences[1].stale is False

    stale_rejects = harness.audit.query(
        run_id="run_h2",
        stage="COMPLETION",
        rule="COMPLETION_STALE_TEST_EVIDENCE",
        decision="REJECTED_FINAL_ANSWER",
    )
    assert len(stale_rejects) == 1
