# Guarded Coding Harness (based on `smolagents.ToolCallingAgent`)

在不重写 `smolagents` 核心的前提下，为 `ToolCallingAgent` 增加一层可恢复、可审计、带安全边界与最终验收门禁的 Guarded Coding Harness，支持**代码理解 (`code_understanding`)**与**测试修复 (`test_fix`)**两类仓库任务。

## 架构概览

- **不重写 `smolagents` 核心**：`GuardedToolCallingAgent` 仅轻度继承 `smolagents.ToolCallingAgent`，复用其 ReAct 循环（`_run_stream` / `_step_stream`）、`AgentMemory`、`step_callbacks` 与 `final_answer_checks`；仅在 `process_tool_calls` / `execute_tool_call` / `_handle_max_steps_reached` 三个扩展点接入统一门禁与确定性停机控制。
- **单一执行入口，规则集中不散落**：`tools.py` 中的 `smolagents.Tool` 仅声明 Schema 且禁止直接绕过 `forward()` 调用；所有参数校验、权限检查、沙箱边界、重复动作拦截、人工审批、写前日志（WAL）和审计均收敛在 `gate.py`、`sandbox.py`、`checkpoint.py` 和 `harness.py`。

```text
Model (FakeModel / OpenAIServerModel)
   │
   ▼
GuardedToolCallingAgent (smolagents.ToolCallingAgent)
   ├── execute_tool_call ──► ToolGate (gate.py)
   │                            ├── Arg schema / Permission / Budget / Duplicate / Approval / WAL
   │                            ├── Sandbox pre-check & execution (sandbox.py)
   │                            └── Append-only Audit (audit.py)
   ├── final_answer_checks ──► CompletionChecker (harness.py: H1 deliverables, H2 fresh test hash)
   └── step_callbacks      ──► StateStore (state_store.py) + CheckpointManager (checkpoint.py: C1-C4)
```

## 交付清单

| 文件 | 职责 |
| --- | --- |
| `gate.py` | 所有 Tool Call 唯一入口：参数校验（G1）、连续重复拦截（G2）、未审批写暂停（G3）、单次审批防重放（G4）、权限/预算门禁与写前 WAL |
| `sandbox.py` | 路径逃逸拦截（S1）、软链逃逸与敏感目录/文件 realpath 校验（S2）、命令白名单与防注入（S3）、独立进程组超时/取消强杀（S4）、环境变量脱敏、输出截断与工作区/源码哈希 |
| `state_store.py` | 原子持久化 `RunState`（Run、计划、步骤、证据、预算、审批、停止原因）与独立 `write_journal.json` WAL |
| `checkpoint.py` | 安全步骤边界保存不可变 Checkpoint（C1）；恢复时优先检测未提交写标记 `UNKNOWN_RESULT`（C2）、检测外部工作区变化失效旧证据/旧审批（C4）、支持跨进程恢复且计数器不归零/不重复已完成写入（C3） |
| `audit.py` | 文件锁 + `fsync` 的 append-only JSONL 审计日志，带单调序号与哈希链完整性校验；支持按 `run_id` / `toolCallId` 还原全链路（A1、A2） |
| `harness.py` | 统一装配入口 `GuardedHarness`（`run` / `resume` / `approve` / `reject`）与 `CompletionChecker`（H1、H2、H3） |
| `schemas.py` / `tools.py` | 统一领域数据结构与无业务逻辑的薄 Tool 绑定 |
| `demo.py` | 跨进程端到端运行与证据生成脚本（默认 FakeModel，支持 `--real-model` 接入真实模型 T2） |
| `tests/` | 6 个确定性测试文件 + `tests/fake_model.py` |
| `records/` | 实际 `pytest -q` 输出、完整运行记录、审计日志、跨进程 Checkpoint 恢复记录 |

## 核心验收对照

| 编号 | 场景 | 实现与测试覆盖 |
| --- | --- | --- |
| G1 | 参数缺字段/类型错误（含 `bool` vs `int`） | `tests/test_gate.py::test_g1_missing_or_wrong_type_arguments_blocked_before_handler` |
| G2 | 连续三次相同工具和参数，第三次拦截 | `tests/test_gate.py::test_g2_three_consecutive_identical_calls_blocked_on_third`、`tests/test_harness_loop.py::test_duplicate_loop_guard_inside_agent_loop_recovers` |
| G3 | 未审批写操作进入 `WAITING_APPROVAL` | `tests/test_gate.py::test_g3_unapproved_write_enters_waiting_approval_without_modifying_file` |
| G4 | 审批后仅执行一次，重复调用拒绝 | `tests/test_gate.py::test_g4_approved_write_executes_once_and_blocks_replay` |
| S1 | `artifacts/../../.env` 路径逃逸拦截 | `tests/test_sandbox.py::test_s1_path_traversal_escape_rejected_without_handler` |
| S2 | 软链逃逸/敏感目录 realpath 校验并审计 | `tests/test_sandbox.py::test_s2_symlink_escape_and_sensitive_directory_rejected_with_audit`、`test_dangling_symlink_write_escape_rejected` |
| S3 | `run_test` 白名单外命令与注入拦截 | `tests/test_sandbox.py::test_s3_disallowed_commands_and_shell_injection_blocked` |
| S4 | 超时/用户取消终止整个进程组（含孙进程） | `tests/test_sandbox.py::test_s4_timeout_kills_entire_process_group`、`test_s4_user_cancel_kills_entire_process_group` |
| C1 | 安全步骤后保存完整 Checkpoint | `tests/test_state_checkpoint.py::test_c1_checkpoint_saved_after_every_safe_step_with_complete_fields` |
| C2 | 写盘后未提交结果崩溃恢复标记 `UNKNOWN_RESULT` | `tests/test_state_checkpoint.py::test_c2_crash_after_write_before_commit_marks_unknown_result_and_blocks_replay` |
| C3 | 换进程恢复，不重复写入、计数不归零 | `tests/test_state_checkpoint.py::test_c3_cross_process_resume_continues_without_repeating_writes_or_zeroing_counters` |
| C4 | 恢复时工作区变化使旧证据和旧审批失效 | `tests/test_state_checkpoint.py::test_c4_workspace_change_before_resume_invalidates_evidences_and_approvals` |
| A1 | 被拒绝调用审计含 `rule`/`decision`/`actor`/`toolCallId` | `tests/test_audit.py::test_a1_rejected_call_audit_contains_required_fields_and_integrity` |
| A2 | 完整任务关联 Gate/Sandbox/State/Approval/Execution | `tests/test_audit.py::test_a2_full_task_reconstructs_end_to_end_decision_chain` |
| H1 | 无交付物时拒绝 `final_answer` | `tests/test_completion.py::test_h1_final_answer_rejected_when_no_deliverable_exists`、`test_h1_recovers_after_rejection_once_deliverable_is_written` |
| H2 | 测试通过后改源码，旧测试失效需重跑 | `tests/test_completion.py::test_h2_modifying_source_after_passing_test_invalidates_test_evidence` |
| H3 | 达最大轮数或预算以明确 `StopReason` 停止 | `tests/test_harness_loop.py::test_h3_max_steps_stops_with_explicit_stop_reason_without_faking_success`、`test_h3_token_budget_exceeded_stops_with_explicit_stop_reason` |
| T1 | 完整确定性测试集全绿 | `uv run pytest -q`（28 passed，见 `records/pytest_output.txt`） |
| T2 | 真实模型跑通一类任务 | DeepSeek (`deepseek-flash`) 实际跑通 `test_fix`，见 `records/real_model_run_record.json`、`records/real_model_audit_record.jsonl` |

## 运行方式

```bash
# 1. 运行完整测试集（不依赖真实模型）
uv run pytest -q

# 2. 生成完整运行记录、审计记录与跨进程 Checkpoint 恢复记录
uv run python demo.py

# 3. 使用 DeepSeek 真实模型运行（T2，支持 .env 或环境变量）
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
uv run python demo.py --real-model --task-type test_fix

# 或直接通过环境变量传入 DeepSeek Key（默认 https://api.deepseek.com / deepseek-chat）
DEEPSEEK_API_KEY=sk-your-key uv run python demo.py --real-model --task-type test_fix --auto-approve
```

## 提交记录文件

- `records/pytest_output.txt`：`pytest -q` 实际运行输出（28 passed）
- `records/run_record.json`：完整 `test_fix` 任务运行记录（含 Diff、计划、步骤、证据、预算、审批、最终状态与审计摘要）
- `records/audit_record.jsonl`：append-only 哈希链审计日志（61 条事件，覆盖 Gate/Sandbox/State/Approval/Checkpoint/Execution/Completion）
- `records/checkpoint_resume_record.json`：真实跨进程（Process 1 -> Process 2）Checkpoint 保存、人工审批、恢复与防重复写入记录
- `records/real_model_run_record.json`：DeepSeek (`deepseek-flash`) 真实模型跑通 `test_fix` 任务的完整运行记录（含 Diff、失败→修复→通过测试证据与审计汇总）
- `records/real_model_audit_record.jsonl`：DeepSeek 真实模型运行的 51 条 append-only 哈希链审计日志
