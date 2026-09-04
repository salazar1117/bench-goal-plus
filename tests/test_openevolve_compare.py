from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.openevolve_compare import experiment  # noqa: E402


class OpenEvolveComparisonTest(unittest.TestCase):
    def test_canonical_methods_and_experiment_defaults(self) -> None:
        self.assertEqual(
            experiment.METHODS,
            (
                "openevolve",
                "plain-codex",
                "goal-plus-codex",
                "goal-plus-pi",
                "skydiscover-best-of-n",
                "skydiscover-evox",
                "skydiscover-adaevolve",
            ),
        )
        parser = experiment.build_parser()
        args = parser.parse_args(["prepare", "--method", "plain-codex"])
        self.assertEqual(args.wall_time_seconds, 300)
        self.assertEqual(args.concurrency, 2)
        self.assertEqual(args.model, "gpt-5.6-luna")
        self.assertEqual(args.reasoning_effort, "high")

        batch_args = parser.parse_args(
            [
                "prepare-batch",
                "--run-root",
                "campaign",
                "--methods",
                "goal-plus-codex",
            ]
        )
        self.assertEqual(batch_args.task_set, "cpu_portable")
        self.assertEqual(batch_args.methods, ["goal-plus-codex"])
        self.assertEqual(batch_args.wall_time_seconds, 300)
        self.assertEqual(batch_args.concurrency, 2)

    def test_prepare_batch_expands_every_task_method_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "campaign"
            args = experiment.build_parser().parse_args(
                [
                    "prepare-batch",
                    "--run-root",
                    str(run_root),
                    "--methods",
                    "plain-codex",
                    "goal-plus-codex",
                ]
            )
            tasks = [{"task_id": "one"}, {"task_id": "two"}]
            with (
                mock.patch.object(experiment, "list_catalog_tasks", return_value=tasks),
                mock.patch.object(experiment, "prepare", return_value=0) as prepare_mock,
            ):
                self.assertEqual(experiment.prepare_batch(args), 0)

            campaign = json.loads((run_root / "campaign.json").read_text())
            self.assertEqual(campaign["task_count"], 2)
            self.assertEqual(campaign["cell_count"], 4)
            self.assertEqual(campaign["prepared_count"], 4)
            self.assertTrue((run_root / "campaign-summary.json").is_file())
            self.assertTrue((run_root / "campaign-summary.md").is_file())
            self.assertEqual(prepare_mock.call_count, 4)
            self.assertEqual(
                {(item["task_id"], item["method"]) for item in campaign["entries"]},
                {
                    ("one", "plain-codex"),
                    ("one", "goal-plus-codex"),
                    ("two", "plain-codex"),
                    ("two", "goal-plus-codex"),
                },
            )

    def test_run_batch_preserves_results_and_continues_after_incomplete_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            campaign_path = root / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "model": "test-model",
                        "methods": ["goal-plus-codex"],
                        "entries": [
                            {
                                "task_id": "one",
                                "method": "goal-plus-codex",
                                "run_dir": str(root / "one"),
                                "prepared": True,
                                "error": None,
                            },
                            {
                                "task_id": "two",
                                "method": "goal-plus-codex",
                                "run_dir": str(root / "two"),
                                "prepared": True,
                                "error": None,
                            },
                        ],
                    }
                )
            )
            args = experiment.build_parser().parse_args(
                ["run-batch", "--campaign", str(campaign_path)]
            )
            with mock.patch.object(experiment, "execute", side_effect=[2, 0]) as run:
                self.assertEqual(experiment.run_batch(args), 2)

            results = json.loads((root / "campaign-results.json").read_text())
            report = json.loads((root / "campaign-summary.json").read_text())
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                [item["status"] for item in results["results"]],
                ["incomplete", "finished"],
            )
            self.assertEqual(report["record_count"], 2)
            self.assertEqual(
                [item["status"] for item in report["records"]],
                ["incomplete", "finished"],
            )
            self.assertNotIn("api_base", results)
            with mock.patch.object(experiment, "execute") as resumed_run:
                self.assertEqual(experiment.run_batch(args), 2)
                resumed_run.assert_not_called()

    def test_goal_plus_assets_copy_only_portable_project_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            goal_plus = temp / "goal-plus"
            codex = goal_plus / ".codex"
            (codex / "agents").mkdir(parents=True)
            (codex / "skills/demo").mkdir(parents=True)
            (codex / "agents/worker.toml").write_text("name='worker'\n")
            (codex / "skills/demo/SKILL.md").write_text("# demo\n")
            (codex / "hooks.json").write_text("{}\n")
            (codex / "config.example.toml").write_text("[mcp_servers.goal-plus]\n")
            (codex / "config.toml").write_text("secret='must-not-copy'\n")
            workspace = temp / "workspace"
            workspace.mkdir()

            experiment.copy_goal_plus_assets(goal_plus, workspace)

            target = workspace / ".codex"
            self.assertTrue((target / "agents/worker.toml").is_file())
            self.assertTrue((target / "skills/demo/SKILL.md").is_file())
            self.assertEqual(
                (target / "config.toml").read_text(),
                "[mcp_servers.goal-plus]\n",
            )
            self.assertNotIn(
                "must-not-copy",
                "\n".join(p.read_text() for p in target.rglob("*.*")),
            )

    def test_goal_plus_assets_materialize_latest_example_hooks_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            goal_plus = temp / "goal-plus"
            codex = goal_plus / ".codex"
            (codex / "skills/demo").mkdir(parents=True)
            (codex / "skills/demo/SKILL.md").write_text("# demo\n")
            (codex / "hooks.example.json").write_text('{"version": 1}\n')
            (codex / "config.example.toml").write_text(
                "[mcp_servers.goal-plus]\n"
            )
            workspace = temp / "workspace"
            workspace.mkdir()

            experiment.copy_goal_plus_assets(goal_plus, workspace)

            target = workspace / ".codex"
            self.assertFalse((target / "agents").exists())
            self.assertEqual((target / "hooks.json").read_text(), '{"version": 1}\n')
            self.assertEqual(
                (target / "config.toml").read_text(),
                "[mcp_servers.goal-plus]\n",
            )

    def test_goal_plus_entrypoint_matches_worker_host(self) -> None:
        self.assertEqual(
            experiment.goal_plus_entrypoint("codex"),
            "$goal-plus",
        )
        self.assertEqual(
            experiment.goal_plus_entrypoint("pi-rpc"),
            "/goal-plus",
        )
        with self.assertRaisesRegex(ValueError, "unsupported Goal Plus worker host"):
            experiment.goal_plus_entrypoint("unknown")

    def test_goal_plus_pi_assets_copy_only_project_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            goal_plus = temp / "goal-plus"
            pi = goal_plus / ".pi"
            (pi / "extensions").mkdir(parents=True)
            (pi / "skills/goal-plus").mkdir(parents=True)
            (pi / "prompts").mkdir(parents=True)
            (pi / "extensions/goal-plus.ts").write_text("export default {}\n")
            (pi / "skills/goal-plus/SKILL.md").write_text("# Goal Plus\n")
            (pi / "prompts/search-candidate-worker.md").write_text("worker\n")
            workspace = temp / "workspace"
            workspace.mkdir()

            experiment.copy_goal_plus_pi_assets(goal_plus, workspace)

            self.assertTrue((workspace / ".pi/extensions/goal-plus.ts").is_file())
            self.assertTrue((workspace / ".pi/skills/goal-plus/SKILL.md").is_file())
            self.assertTrue(
                (workspace / ".pi/prompts/search-candidate-worker.md").is_file()
            )

    def test_goal_prompt_uses_natural_entry_and_complete_configuration(self) -> None:
        prompt = experiment.render_goal(
            task_text="# Objective\nImprove it.",
            artifact_name="candidate.py",
            metric_name="combined_score",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=2,
            worker_host="pi-rpc",
            worker_model="bench-openai/gpt-5.6-luna",
        )
        self.assertTrue(
            prompt.startswith(
                "/goal-plus mode=autonomous max_parallel=2 "
                "workspace_backend=git_worktree promotion_mode=apply "
                "strategy=agent_guided "
                "workers=bench-openai/gpt-5.6-luna*2 "
                "annotator=bench-openai/gpt-5.6-luna"
            )
        )
        self.assertNotIn(" -- ", prompt.splitlines()[0])
        self.assertNotIn("`budget.max_parallel=2`", prompt)
        self.assertIn("omit deprecated `budget.max_candidates`", prompt)
        self.assertIn("240 seconds", prompt)
        self.assertIn("not hard-capped", prompt)
        self.assertIn("GOAL_PLUS_OUTER_DEADLINE_AT", prompt)
        self.assertIn('strategy.worker_host="pi-rpc"', prompt)
        self.assertNotIn('strategy.name="agent_guided"', prompt)
        self.assertIn("aligned with `command_config.workers`", prompt)
        self.assertIn('strategy.worker_launch.reasoning_effort="high"', prompt)
        self.assertIn("Metric: `combined_score` with direction `maximize`", prompt)
        self.assertIn("python3 .goal-plus-verifiers/primary_metric.py", prompt)
        self.assertIn("do not run a duplicate parent-side process verification", prompt)
        self.assertIn("allow only `candidate.py`", prompt)
        self.assertNotIn("goal_plus_id=", prompt)
        self.assertIn("search_start_batch", prompt)
        self.assertIn("pi_search_pool_open", prompt)
        self.assertIn("pi_search_pool_wait_any", prompt)
        self.assertNotIn("actual `spawn_agent` call", prompt)

    def test_blind_goal_prompt_uses_artifact_only_promotion(self) -> None:
        prompt = experiment.render_goal(
            task_text="# Objective\nImprove it.",
            artifact_name="candidate.py",
            metric_name="public_format",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=2,
            worker_host="pi-rpc",
            worker_model="bench-openai/gpt-5.6-luna",
            shared_dir_enabled=True,
            evaluation_mode="blind",
        )

        self.assertTrue(
            prompt.startswith(
                "/goal-plus mode=autonomous max_parallel=2 "
                "workspace_backend=git_worktree promotion_mode=artifact_only "
            )
        )
        self.assertIn("`shared_dir.enabled=false`", prompt)
        self.assertNotIn("`shared_dir.enabled=true`", prompt)
        self.assertIn('`strategy.config.global_evidence_mode="independent"`', prompt)
        self.assertIn("never receives the official evaluator or official metric", prompt)

    def test_pi_goal_prompt_names_pool_supervisor_minimum_lease(self) -> None:
        prompt = experiment.render_goal(
            task_text="# Objective\nImprove it.",
            artifact_name="candidate.py",
            metric_name="combined_score",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=2,
            worker_host="pi-rpc",
            worker_model="bench-openai/gpt-5.6-luna",
            worker_runtime_seconds=200,
            worker_min_runtime_seconds=150,
        )

        self.assertIn("pool supervisor", prompt)
        self.assertIn("same native session", prompt)
        self.assertIn("strategy.config.closeout_reserve_seconds=60", prompt)
        self.assertNotIn("SubagentStop", prompt)

    def test_plain_and_goal_plus_prompts_share_exact_common_body(self) -> None:
        task = "# Objective\nImprove it."
        common = experiment.render_plain_prompt(task, 300, 60)
        prompt = experiment.render_goal(
            task_text=task,
            artifact_name="candidate.py",
            metric_name="combined_score",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=2,
            worker_host="codex",
            worker_model="gpt-5.6-luna",
        )
        self.assertTrue(
            prompt.startswith(
                "$goal-plus mode=autonomous max_parallel=2 "
                "workspace_backend=git_worktree promotion_mode=apply "
                "strategy=agent_guided workers=gpt-5.6-luna*2 "
                "annotator=gpt-5.6-luna\n\n"
                + common.rstrip()
                + "\n\n# Goal Plus configuration"
            )
        )
        self.assertEqual(common, experiment.render_plain_prompt(task, 300, 60))
        self.assertNotIn("independent lane", common)
        self.assertNotIn("controller-prepared", prompt)

    def test_codex_event_parser_records_actual_worker_launches(self) -> None:
        events = [
            {"type": "thread.started", "thread_id": "thread_parent"},
            {
                "type": "item.completed",
                "item": {
                    "type": "collab_tool_call",
                    "tool": "spawn_agent",
                    "status": "completed",
                    "receiver_thread_ids": ["thread_worker_1"],
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "collab_tool_call",
                    "tool": "wait",
                    "status": "completed",
                    "receiver_thread_ids": [],
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "goal-plus",
                    "tool": "search_start_agent_session",
                    "status": "completed",
                    "arguments": {
                        "run_id": "run_test",
                        "candidate_id": "c001",
                    },
                    "result": {
                        "structured_content": {
                            "run_id": "run_test",
                            "candidate_id": "c001",
                            "agent_session_id": "agent_001",
                            "host": "codex",
                        }
                    },
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "goal-plus",
                    "tool": "search_run_verifier",
                    "status": "completed",
                    "arguments": {
                        "run_id": "run_test",
                        "candidate_id": "c001",
                    },
                    "result": {
                        "structured_content": {
                            "run_id": "run_test",
                            "candidate_id": "c001",
                            "validity_passed": True,
                            "aggregate_score": 2.0,
                            "disposition": "keep",
                        }
                    },
                },
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in events))
            result = experiment.parse_codex_events(path)

        self.assertEqual(result["spawn_agent_completed_count"], 1)
        self.assertEqual(result["spawned_agent_thread_count"], 1)
        self.assertEqual(result["spawned_agent_thread_ids"], ["thread_worker_1"])
        self.assertEqual(result["targetless_wait_count"], 1)
        self.assertEqual(
            result["collaboration_tool_counts"]["spawn_agent"]["completed"], 1
        )
        self.assertEqual(result["goal_plus"]["candidate_ids"], ["c001"])
        self.assertEqual(result["goal_plus"]["agent_session_ids"], ["agent_001"])
        self.assertEqual(
            result["goal_plus"]["verifier_ledger"][0]["aggregate_score"], 2.0
        )

    def test_codex_event_parser_deduplicates_bound_worker_handles(self) -> None:
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "goal-plus",
                    "tool": "search_bind_agent_handle",
                    "status": "completed",
                    "arguments": {
                        "agent_session_id": session_id,
                        "handle": {
                            "host": "codex",
                            "task_name": "/root/shared-worker",
                        },
                    },
                },
            }
            for session_id in ("agent_001", "agent_002")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in events))
            result = experiment.parse_codex_events(path)

        self.assertEqual(result["goal_plus"]["bound_worker_session_count"], 2)
        self.assertEqual(result["goal_plus"]["bound_worker_handle_count"], 1)

    def test_goal_plus_completion_requires_worker_verifier_evidence_for_every_candidate(
        self,
    ) -> None:
        base_state = {
            "goals": [
                {
                    "goal_plus_id": "gp_0001",
                    "status": "complete",
                    "linked_run_id": "run_test",
                }
            ],
            "runs": [
                {
                    "run_id": "run_test",
                    "candidate_count": 2,
                    "worker_host": "pi-rpc",
                    "worker_budget": {
                        "min_runtime_seconds": 150,
                        "min_verifier_runs": 1,
                        "max_runtime_seconds": 200,
                        "on_exceed": "interrupt",
                    },
                    "pi_pool_jobs": [
                        {
                            "job_id": "job_1",
                            "candidate_id": "c001",
                            "status": "completed",
                            "lease": {"satisfied": True},
                        },
                        {
                            "job_id": "job_2",
                            "candidate_id": "c002",
                            "status": "completed",
                            "lease": {"satisfied": True},
                        },
                    ],
                    "bound_candidate_count": 2,
                    "worker_verified_candidate_count": 2,
                    "unbound_agent_session_count": 0,
                    "session_counts_by_candidate": {"c001": 1, "c002": 1},
                    "bound_session_counts_by_candidate": {"c001": 1, "c002": 1},
                }
            ],
        }
        kwargs = {
            "expected_concurrency": 2,
            "expected_goal_plus_id": "gp_0001",
            "expected_run_id": "run_test",
            "expected_worker_min_runtime_seconds": 150,
            "expected_worker_min_verifier_runs": 1,
        }
        self.assertIsNone(experiment.goal_plus_incomplete_reason(base_state, **kwargs))

        self.assertIn(
            "0 distinct spawned worker threads",
            experiment.goal_plus_incomplete_reason(
                base_state,
                codex_events={"spawned_agent_thread_count": 0},
                **kwargs,
            ),
        )
        self.assertIsNone(
            experiment.goal_plus_incomplete_reason(
                base_state,
                codex_events={"spawned_agent_thread_count": 2},
                **kwargs,
            )
        )
        self.assertIsNone(
            experiment.goal_plus_incomplete_reason(
                base_state,
                codex_events={
                    "spawned_agent_thread_count": 0,
                    "goal_plus": {"bound_worker_handle_count": 2},
                },
                **kwargs,
            )
        )

        missing_worker_evidence = json.loads(json.dumps(base_state))
        missing_worker_evidence["runs"][0]["worker_verified_candidate_count"] = 1
        self.assertIn(
            "completed worker verifier evidence",
            experiment.goal_plus_incomplete_reason(missing_worker_evidence, **kwargs),
        )
        self.assertIsNone(
            experiment.goal_plus_incomplete_reason(
                missing_worker_evidence,
                minimum_worker_verified_candidates=1,
                **kwargs,
            )
        )

        duplicate_session = json.loads(json.dumps(base_state))
        duplicate_session["runs"][0]["session_counts_by_candidate"] = {
            "c001": 2,
            "c002": 2,
        }
        duplicate_session["runs"][0]["bound_session_counts_by_candidate"] = {
            "c001": 2,
            "c002": 2,
        }
        self.assertIn(
            "exactly one bound session per candidate",
            experiment.goal_plus_incomplete_reason(duplicate_session, **kwargs),
        )

        duplicate_goal = json.loads(json.dumps(base_state))
        duplicate_goal["goals"].append(
            {
                "goal_plus_id": "gp_0002",
                "status": "complete",
                "linked_run_id": "run_test",
            }
        )
        self.assertIn(
            "duplicate Goal Plus records",
            experiment.goal_plus_incomplete_reason(duplicate_goal, **kwargs),
        )

        misplaced_budget = json.loads(json.dumps(base_state))
        misplaced_budget["runs"][0]["worker_budget"].pop("min_runtime_seconds")
        self.assertIn(
            "frozen worker budget",
            experiment.goal_plus_incomplete_reason(misplaced_budget, **kwargs),
        )

        unsatisfied_lease = json.loads(json.dumps(base_state))
        unsatisfied_lease["runs"][0]["pi_pool_jobs"][0].update(
            {"status": "timed_out", "lease": {"satisfied": False}}
        )
        self.assertIn(
            "minimum lease",
            experiment.goal_plus_incomplete_reason(unsatisfied_lease, **kwargs),
        )

    def test_settled_selection_waives_unsatisfied_pi_minimum_lease(self) -> None:
        # A Search run that committed a real business result (selected_score,
        # even 0.0 = NOT_PASS) must NOT be reported INFRA just because the host
        # cut a pi worker's minimum lease short.
        settled_not_pass = {
            "goals": [
                {
                    "goal_plus_id": "gp_0001",
                    "status": "complete",
                    "linked_run_id": "run_test",
                }
            ],
            "runs": [
                {
                    "run_id": "run_test",
                    "candidate_count": 2,
                    "worker_host": "pi-rpc",
                    "selected_score": 0.0,
                    "best_recorded_score": 0.0,
                    "worker_budget": {
                        "min_runtime_seconds": 150,
                        "min_verifier_runs": 1,
                        "max_runtime_seconds": 200,
                        "on_exceed": "interrupt",
                    },
                    "pi_pool_jobs": [
                        {
                            "job_id": "job_1",
                            "candidate_id": "c001",
                            "status": "timed_out",
                            "lease": {"satisfied": False},
                        },
                        {
                            "job_id": "job_2",
                            "candidate_id": "c002",
                            "status": "completed",
                            "lease": {"satisfied": True},
                        },
                    ],
                    "bound_candidate_count": 2,
                    "worker_verified_candidate_count": 2,
                    "unbound_agent_session_count": 0,
                    "session_counts_by_candidate": {"c001": 1, "c002": 1},
                    "bound_session_counts_by_candidate": {"c001": 1, "c002": 1},
                }
            ],
        }
        kwargs = {
            "expected_concurrency": 2,
            "expected_goal_plus_id": "gp_0001",
            "expected_run_id": "run_test",
            "expected_worker_min_runtime_seconds": 150,
            "expected_worker_min_verifier_runs": 1,
            "minimum_worker_verified_candidates": 1,
        }

        # The committed selection makes this run settled.
        self.assertTrue(experiment.goal_plus_settled_selection(settled_not_pass))

        # Default behavior (lease required) still flags the unsatisfied lease.
        self.assertIn(
            "minimum lease",
            experiment.goal_plus_incomplete_reason(settled_not_pass, **kwargs),
        )
        # Waiving the lease (because the run settled) reports a real NOT_PASS.
        self.assertIsNone(
            experiment.goal_plus_incomplete_reason(
                settled_not_pass,
                require_satisfied_pi_minimum_lease=False,
                **kwargs,
            )
        )

        # A genuine INFRA case (seed never started, no runs) is never settled,
        # so the lease/incompleteness gates stay in force.
        self.assertFalse(experiment.goal_plus_settled_selection({"goals": [], "runs": []}))

    def test_natural_goal_plus_completion_ignores_aborted_search_history(
        self,
    ) -> None:
        state = {
            "goals": [
                {
                    "goal_plus_id": "gp_0001",
                    "status": "complete",
                    "linked_run_id": "run_final",
                }
            ],
            "runs": [
                {
                    "run_id": "run_aborted",
                    "status": "aborted",
                    "candidate_count": 0,
                    "bound_candidate_count": 0,
                    "worker_verified_candidate_count": 0,
                    "unbound_agent_session_count": 0,
                    "session_counts_by_candidate": {},
                    "bound_session_counts_by_candidate": {},
                },
                {
                    "run_id": "run_final",
                    "status": "completed",
                    "candidate_count": 2,
                    "bound_candidate_count": 2,
                    "worker_verified_candidate_count": 2,
                    "unbound_agent_session_count": 0,
                    "session_counts_by_candidate": {"c001": 1, "c002": 1},
                    "bound_session_counts_by_candidate": {"c001": 1, "c002": 1},
                },
            ],
        }
        self.assertIsNone(
            experiment.goal_plus_incomplete_reason(
                state,
                expected_concurrency=2,
            )
        )

    def test_goal_plus_configuration_keeps_runtime_files_outside_edit_surface(
        self,
    ) -> None:
        prompt = experiment.render_goal(
            task_text="# Objective\nImprove it.",
            artifact_name="candidate.py",
            metric_name="combined_score",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=2,
            worker_host="pi-rpc",
            worker_model="bench-openai/gpt-5.6-luna",
        )
        self.assertIn("allow only `candidate.py`", prompt)
        self.assertIn("deny `evaluate.py`", prompt)
        self.assertIn("`.goal-plus-verifiers/**`", prompt)
        self.assertIn("allow at most one changed file", prompt)
        self.assertIn('use `source_path="."`', prompt)
        self.assertIn("backend and promotion mode come from", prompt)
        self.assertIn("strategy.worker_budget.max_runtime_seconds=60", prompt)
        self.assertIn("total budget, not a success criterion", prompt)

    def test_goal_plus_directory_artifact_allows_multiple_changed_files(self) -> None:
        prompt = experiment.render_goal(
            task_text="# Objective\nFind issues.",
            artifact_name="submission",
            artifact_is_directory=True,
            metric_name="f1",
            metric_direction="maximize",
            wall_seconds=1800,
            closeout_seconds=60,
            concurrency=4,
            worker_host="codex",
            worker_model="gpt-5.6-sol",
        )
        self.assertIn("allow only `submission`", prompt)
        self.assertIn("omit `max_file_changes`", prompt)
        self.assertIn("multiple changed files inside it are allowed", prompt)
        self.assertNotIn("allow at most one changed file", prompt)

    def test_goal_plus_pi_prompt_uses_native_pool_launch_contract(self) -> None:
        prompt = experiment.render_goal(
            task_text="# Objective\nImprove it.",
            artifact_name="candidate.py",
            metric_name="score",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=4,
            worker_host="pi-rpc",
            worker_model="provider/model",
        )

        self.assertIn("`pi_search_pool_open`", prompt)
        self.assertIn("`pi_search_pool_wait_any`", prompt)
        self.assertNotIn("`spawn_agent`", prompt)

    def test_promotion_patch_is_applied_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            subprocess.run(
                ["git", "-C", str(workspace), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            artifact = workspace / "candidate.py"
            artifact.write_text("value = 1\n")
            subprocess.run(
                ["git", "-C", str(workspace), "add", "candidate.py"], check=True
            )
            subprocess.run(
                ["git", "-C", str(workspace), "commit", "-qm", "seed"], check=True
            )
            artifact.write_text("value = 2\n")
            patch_path = Path(temp_dir) / "promotion.patch"
            patch_path.write_text(
                subprocess.run(
                    ["git", "-C", str(workspace), "diff", "--", "candidate.py"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            artifact.write_text("value = 1\n")
            self.assertEqual(
                experiment.apply_promotion_patch(workspace, patch_path), "applied"
            )
            self.assertEqual(artifact.read_text(), "value = 2\n")
            self.assertEqual(
                experiment.apply_promotion_patch(workspace, patch_path),
                "already_applied",
            )

    def test_controller_closeout_reuses_promotion_completed_during_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            root = workspace / ".gp"
            goal_dir = root / "goal-plus/gp_0001"
            run_dir = root / "runs/run_test"
            candidate_dir = run_dir / "candidates/c001"
            promotion_dir = run_dir / "promotion"
            goal_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            promotion_dir.mkdir()
            (goal_dir / "goal.json").write_text("{}\n")
            run_path = run_dir / "run.json"
            run_data = {
                "run_id": "run_test",
                "state": "waiting_for_workers",
                "source_path": str(workspace),
                "selected_candidate_id": "c001",
                "selected_score": 2.0,
                "selected_iteration": 1,
            }
            run_path.write_text(json.dumps(run_data) + "\n")
            (candidate_dir / "candidate.json").write_text(
                json.dumps({"candidate_id": "c001", "iterations": [{}]}) + "\n"
            )
            patch_path = promotion_dir / "c001.patch"
            patch_path.write_text("promotion\n")

            goal = mock.Mock()
            goal.goal_plus_id = "gp_0001"
            goal.linked_search.run_id = "run_test"
            goal.linked_search.selected_candidate_id = None
            goal.status = "active"
            goal_runtime = mock.Mock()
            goal_runtime.status.return_value = goal
            tools = mock.Mock()
            tools.search_report.return_value = {"report_path": "report.md"}

            def finish_promotion_then_fail(_run_id: str) -> None:
                run_data["state"] = "promoted"
                run_path.write_text(json.dumps(run_data) + "\n")
                raise RuntimeError(
                    "cannot record verifier result: run run_test is in state promoted"
                )

            tools.search_select.side_effect = finish_promotion_then_fail
            with (
                mock.patch.object(
                    experiment,
                    "_goal_plus_runtime_types",
                    return_value=(
                        mock.Mock(return_value=goal_runtime),
                        mock.Mock(),
                        mock.Mock(return_value=tools),
                    ),
                ),
                mock.patch.object(
                    experiment, "apply_promotion_patch", return_value="applied"
                ) as apply_patch,
            ):
                result = experiment.finalize_goal_plus_search(workspace)

            self.assertTrue(result["completed"], result)
            self.assertEqual(result["runs"][0]["source_patch_status"], "applied")
            self.assertTrue(
                result["runs"][0]["selection"]["reused_existing_promotion"]
            )
            apply_patch.assert_called_once_with(workspace, patch_path)
            tools.search_promote.assert_not_called()

    def test_controller_closeout_reuses_selection_completed_before_closeout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            root = workspace / ".gp"
            goal_dir = root / "goal-plus/gp_0001"
            run_dir = root / "runs/run_test"
            candidate_dir = run_dir / "candidates/c001"
            promotion_dir = run_dir / "promotion"
            goal_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            promotion_dir.mkdir()
            (goal_dir / "goal.json").write_text("{}\n")
            run_path = run_dir / "run.json"
            run_path.write_text(
                json.dumps(
                    {
                        "run_id": "run_test",
                        "state": "ready_to_promote",
                        "source_path": str(workspace),
                        "selected_candidate_id": "c001",
                        "selected_score": 2.0,
                        "selected_iteration": 1,
                    }
                )
                + "\n"
            )
            (candidate_dir / "candidate.json").write_text(
                json.dumps({"candidate_id": "c001", "iterations": [{}]}) + "\n"
            )
            patch_path = promotion_dir / "c001.patch"
            patch_path.write_text("promotion\n")

            goal = mock.Mock()
            goal.goal_plus_id = "gp_0001"
            goal.linked_search.run_id = "run_test"
            goal.linked_search.selected_candidate_id = None
            goal.status = "active"
            goal_runtime = mock.Mock()
            goal_runtime.status.return_value = goal
            tools = mock.Mock()
            tools.search_promote.return_value = {"artifact_path": str(patch_path)}
            tools.search_report.return_value = {"report_path": "report.md"}

            with (
                mock.patch.object(
                    experiment,
                    "_goal_plus_runtime_types",
                    return_value=(
                        mock.Mock(return_value=goal_runtime),
                        mock.Mock(),
                        mock.Mock(return_value=tools),
                    ),
                ),
                mock.patch.object(
                    experiment, "apply_promotion_patch", return_value="applied"
                ),
            ):
                result = experiment.finalize_goal_plus_search(workspace)

            self.assertTrue(result["completed"], result)
            self.assertTrue(
                result["runs"][0]["selection"]["reused_existing_selection"]
            )
            tools.search_select.assert_not_called()
            tools.search_run_verifier.assert_not_called()
            tools.search_promote.assert_called_once_with("run_test", "c001")

    def test_evaluator_budget_snapshot_uses_controller_runtime_at_t0(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workspace = temp / "workspace"
            runtime = temp / "controller-runtime"
            workspace.mkdir()
            runtime.mkdir()
            (workspace / "task.json").write_text(
                json.dumps({"controller_runtime_dir": str(runtime)}) + "\n"
            )
            expected = {
                "total_claimed": 2,
                "public_claimed": 2,
                "final_claimed": 0,
            }
            (runtime / "budget.json").write_text(json.dumps(expected) + "\n")

            self.assertEqual(
                experiment.evaluator_budget_for_workspace(workspace), expected
            )

    def test_pi_model_config_uses_environment_reference_not_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            experiment.write_pi_models_config(
                target,
                api_base="http://proxy.example/v1",
                model="gpt-5.6-luna",
            )
            raw = (target / "models.json").read_text()
            payload = json.loads(raw)
            provider = payload["providers"][experiment.PI_PROVIDER_ID]
            self.assertEqual(provider["apiKey"], "$OPENAI_API_KEY")
            self.assertEqual(provider["api"], "openai-responses")
            self.assertEqual(
                provider["models"][0]["thinkingLevelMap"], {"high": "high"}
            )
            self.assertTrue(
                provider["models"][0]["compat"]["supportsDeveloperRole"]
            )
            self.assertNotRegex(raw, r"\bsk-[A-Za-z0-9_-]{16,}\b")

    def test_pi_model_config_disables_developer_role_for_deepseek(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            experiment.write_pi_models_config(
                target,
                api_base="https://api.deepseek.com/v1",
                model="deepseek-v4-flash",
                provider_id="deepseek",
                api="openai-completions",
                api_key_env="DEEPSEEK_API_KEY",
            )
            provider = json.loads((target / "models.json").read_text())["providers"][
                "deepseek"
            ]

            self.assertFalse(
                provider["models"][0]["compat"]["supportsDeveloperRole"]
            )

    def test_pi_model_config_supports_anthropic_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            experiment.write_pi_models_config(
                target,
                api_base="https://anthropic-proxy.example",
                model="glm-5.3",
                reasoning_effort="medium",
                provider_id="zai-anthropic",
                api="anthropic-messages",
                api_key_env="ZAI_API_KEY",
            )
            models_path = target / "models.json"
            raw = models_path.read_text()
            provider = json.loads(raw)["providers"]["zai-anthropic"]

            self.assertEqual(provider["api"], "anthropic-messages")
            self.assertEqual(provider["apiKey"], "$ZAI_API_KEY")
            self.assertEqual(provider["models"][0]["id"], "glm-5.3")
            self.assertNotIn("thinkingLevelMap", provider["models"][0])
            self.assertEqual(models_path.stat().st_mode & 0o777, 0o600)

    def test_codex_provider_args_select_responses(self) -> None:
        args = experiment.codex_provider_args("http://proxy.example/v1")
        joined = "\n".join(args)
        self.assertIn('wire_api="responses"', joined)
        self.assertIn('env_key="OPENAI_API_KEY"', joined)

    def test_evidence_annotator_uses_benchmark_provider_and_persisted_usage(
        self,
    ) -> None:
        environment: dict[str, str] = {}
        experiment.configure_evidence_annotator_environment(
            environment,
            model="gpt-test",
            reasoning_effort="medium",
            api_base="http://proxy.example/v1",
        )
        self.assertEqual(
            environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_MODEL"], "gpt-test"
        )
        self.assertEqual(
            environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_BASE_URL"],
            "http://proxy.example/v1",
        )
        self.assertEqual(
            environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_PROVIDER_ID"],
            experiment.ANNOTATOR_PROVIDER_ID,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            task_dir = (
                workspace
                / ".gp/runs/run_test/candidates/c001/evidence-annotations"
            )
            task_dir.mkdir(parents=True)
            (task_dir / "iteration-0001.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "attempts": 1,
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 4,
                            "cost_usd": 0.0012,
                        },
                    }
                )
            )
            usage = experiment.collect_evidence_annotator_usage(workspace)
        self.assertEqual(usage["input_tokens"], 11)
        self.assertEqual(usage["output_tokens"], 4)
        self.assertEqual(usage["cost_usd"], 0.0012)
        self.assertEqual(usage["states"], {"completed": 1})

    def test_codex_model_args_pin_native_auth_model_and_reasoning(self) -> None:
        args = experiment.codex_model_args("gpt-5.6-terra", None)
        joined = "\n".join(args)
        self.assertIn('model_reasoning_effort="high"', joined)
        self.assertEqual(args[-2:], ["--model", "gpt-5.6-terra"])
        self.assertNotIn("model_provider=", joined)

    def test_codex_model_args_keep_reasoning_with_explicit_provider(self) -> None:
        args = experiment.codex_model_args(
            "gpt-5.6-terra", "http://proxy.example/v1"
        )
        joined = "\n".join(args)
        self.assertIn('model_reasoning_effort="high"', joined)
        self.assertIn('wire_api="responses"', joined)
        self.assertEqual(args[-2:], ["--model", "gpt-5.6-terra"])

    def test_codex_goal_plus_mcp_args_register_runtime_explicitly(self) -> None:
        joined = "\n".join(experiment.codex_goal_plus_mcp_args())
        self.assertIn('mcp_servers.goal-plus.command="goal-plus"', joined)
        self.assertIn('mcp_servers.goal-plus.args=["--root", ".gp"]', joined)
        self.assertIn("mcp_servers.goal-plus.tool_timeout_sec=300", joined)
        self.assertIn(
            'mcp_servers.goal-plus.default_tools_approval_mode="approve"', joined
        )
        self.assertIn("mcp_servers.goal-plus.enabled=true", joined)
        self.assertIn("mcp_servers.goal-plus.env_vars=", joined)
        for variable in (
            "CODEX_HOME",
            "OPENAI_API_KEY",
            "SFORGE_AGENT_API_KEY",
            "GOAL_PLUS_OUTER_DEADLINE_AT",
            "GOAL_PLUS_EVIDENCE_ANNOTATOR_BASE_URL",
        ):
            self.assertIn(variable, joined)

    def test_codex_goal_plus_mcp_args_merge_adapter_environment(self) -> None:
        joined = "\n".join(
            experiment.codex_goal_plus_mcp_args(("TASK_RUNTIME", "OPENAI_API_KEY"))
        )
        self.assertIn("TASK_RUNTIME", joined)
        self.assertEqual(joined.count("OPENAI_API_KEY"), 1)

    def test_codex_execution_args_use_unrestricted_noninteractive_mode(self) -> None:
        args = experiment.codex_execution_args()
        self.assertEqual(
            args,
            [
                "--sandbox",
                "danger-full-access",
                "--config",
                'approval_policy="never"',
            ],
        )

    def test_best_lane_selection_skips_non_numeric_evaluations(self) -> None:
        lanes = [
            {
                "lane": "lane-invalid",
                "evaluation": {"primary_metric": {"value": None}},
            },
            {
                "lane": "lane-low",
                "evaluation": {"primary_metric": {"value": 1.0}},
            },
            {
                "lane": "lane-high",
                "evaluation": {"primary_metric": {"value": 2.0}},
            },
        ]
        selected, invalid = experiment.select_best_lane(lanes, maximize=True)
        self.assertEqual(selected["lane"], "lane-high")
        self.assertEqual(invalid, ["lane-invalid"])

    def test_best_lane_selection_reports_all_invalid(self) -> None:
        selected, invalid = experiment.select_best_lane(
            [
                {
                    "lane": "lane-invalid",
                    "evaluation": {"primary_metric": {"value": None}},
                }
            ],
            maximize=False,
        )
        self.assertIsNone(selected)
        self.assertEqual(invalid, ["lane-invalid"])

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "requires POSIX-style SIGTERM")
    def test_outer_controller_requests_soft_stop_at_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            code = (
                "import signal,sys,time\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "while True: time.sleep(0.1)\n"
            )
            result = experiment.run_controlled(
                [sys.executable, "-c", code],
                cwd=temp,
                environment=os.environ.copy(),
                stdin_text=None,
                stdout_path=temp / "stdout.log",
                stderr_path=temp / "stderr.log",
                wall_time_seconds=1,
                hard_kill_grace_seconds=2,
            )
            self.assertTrue(result["deadline_reached"])
            self.assertFalse(result["hard_killed"])
            self.assertEqual(result["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
