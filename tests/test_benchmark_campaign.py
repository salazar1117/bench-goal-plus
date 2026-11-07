from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from adapters.registry import adapter_modules, load_adapter
from experiments.benchmark_campaign import experiment as campaign
from experiments.benchmark_compare.conditions import resolve_condition
from experiments.benchmark_compare import experiment as standalone
from experiments.openevolve_compare.experiment import (
    collect_search_space_state,
    render_goal,
)


class BenchmarkConditionTest(unittest.TestCase):
    def test_adapter_catalog_loads_validated_modules(self) -> None:
        self.assertEqual(
            set(adapter_modules()),
            {
                "ale-bench-lite",
                "autolab-toy-isa",
                "frontier-cs-problem-0",
                "frontier-engineering-malloclab",
                "heurigym",
                "local-vliw",
                "torchbench",
                "zsoft-detect",
                "zsoft-l1",
            },
        )
        loaded = load_adapter("local-vliw")
        self.assertEqual(loaded.module.DIRECTION, "minimize")
        self.assertEqual(
            loaded.manifest_contract()["verification_owner"],
            "benchmark controller",
        )

    def test_zsoft_adapter_uses_managed_upstream_subdirectory(self) -> None:
        self.addCleanup(standalone.configure_adapter, "heurigym")
        standalone.configure_adapter("zsoft-detect")

        self.assertIsNone(standalone.LOCAL_SOURCE_RELATIVE)
        self.assertEqual(
            standalone.UPSTREAM_SUBDIR,
            "benchmarks/vulnerability/zsoft-detect",
        )

    def test_condition_resolution_uses_real_runtime_boundaries(self) -> None:
        self.assertEqual(
            resolve_condition(method="plain-codex", concurrency=1).condition_id,
            "B0",
        )
        self.assertEqual(
            resolve_condition(method="plain-codex", concurrency=4).condition_id,
            "B1",
        )
        self.assertIsNone(
            resolve_condition(method="goal-plus-codex", concurrency=4)
        )
        b3 = resolve_condition(
            method="goal-plus-codex",
            concurrency=4,
            condition_id="B3",
            coordination_variant="way2",
        )
        self.assertEqual(b3.search_space_mode, "observe")
        with self.assertRaisesRegex(ValueError, "B2 is not implemented"):
            resolve_condition(
                method="goal-plus-codex", concurrency=4, condition_id="B2"
            )
        with self.assertRaisesRegex(ValueError, "way0 is not implemented"):
            resolve_condition(
                method="goal-plus-codex",
                concurrency=4,
                condition_id="B4",
                coordination_variant="way0",
            )

    def test_goal_prompt_freezes_observe_or_enforce_mode(self) -> None:
        prompt = render_goal(
            task_text="Improve it.",
            artifact_name="candidate.py",
            metric_name="score",
            metric_direction="maximize",
            wall_seconds=300,
            closeout_seconds=60,
            concurrency=2,
            agent_harness="codex",
            worker_model="test-model",
            coordination_condition="B3",
            search_space_mode="observe",
            shared_dir_enabled=True,
        )
        self.assertIn("Ablation condition: `B3`", prompt)
        self.assertIn('`mode="observe"`', prompt)
        self.assertIn("must not block a candidate", prompt)
        self.assertIn("`shared_dir.enabled=true`", prompt)

    def test_condition_completion_requires_the_frozen_mode(self) -> None:
        state = {
            "runs": [
                {
                    "candidate_count": 2,
                    "search_space": {"exists": True, "mode": "observe"},
                }
            ]
        }
        self.assertIsNone(
            standalone.condition_incomplete_reason(
                state, {"id": "B3", "search_space_mode": "observe"}
            )
        )
        self.assertIn(
            "requires Search Space mode",
            standalone.condition_incomplete_reason(
                state, {"id": "B4", "search_space_mode": "enforce"}
            ),
        )

    def test_copy_artifact_preserves_directory_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "submission"
            source.mkdir()
            (source / "finding.json").write_text("{}\n")
            destination = root / "saved-submission"

            standalone.copy_artifact(source, destination)

            self.assertTrue(destination.is_dir())
            self.assertEqual((destination / "finding.json").read_text(), "{}\n")


class BenchmarkCampaignTest(unittest.TestCase):
    def test_markdown_explains_incomplete_cells(self) -> None:
        markdown = campaign.render_markdown(
            {
                "campaign_id": "worker-launch-check",
                "state": "partial",
                "record_count": 1,
                "condition_summaries": [
                    {
                        "condition": "goal-plus-codex",
                        "finished_count": 0,
                        "cell_count": 1,
                        "valid_final_count": 0,
                        "mean_directional_gain": None,
                        "total_evaluator_calls": 1,
                        "total_input_tokens": None,
                        "total_output_tokens": None,
                    }
                ],
                "records": [
                    {
                        "benchmark_id": "local-vliw",
                        "method": "goal-plus-codex",
                        "seed": 1,
                        "status": "incomplete",
                        "incomplete_reason": "Codex completed 0 spawn_agent calls",
                    }
                ],
                "b1_vs_b4": {
                    "paired_count": 0,
                    "mean_b4_minus_b1_directional_gain": None,
                },
            }
        )

        self.assertIn("## Incomplete cells", markdown)
        self.assertIn("local-vliw/goal-plus-codex/seed-1", markdown)
        self.assertIn("Codex completed 0 spawn_agent calls", markdown)

    def test_prepare_accepts_method_mode_without_a_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = campaign.build_parser().parse_args(
                [
                    "prepare",
                    "--campaign-dir",
                    str(root / "campaign"),
                    "--benchmarks",
                    "local-vliw",
                    "--methods",
                    "goal-plus-codex",
                    "--wall-time-seconds",
                    "180",
                    "--concurrency",
                    "2",
                ]
            )
            with mock.patch.object(standalone, "prepare", return_value=0) as prepared:
                self.assertEqual(campaign.prepare_campaign(args), 0)
            prepared_args = prepared.call_args.args[0]
            self.assertEqual(prepared_args.method, "goal-plus-codex")
            self.assertIsNone(prepared_args.condition)
            payload = json.loads((root / "campaign/campaign.json").read_text())
            self.assertEqual(payload["methods"], ["goal-plus-codex"])
            self.assertEqual(payload["conditions"], [])
            self.assertEqual(payload["cells"][0]["method"], "goal-plus-codex")
            self.assertIsNone(payload["cells"][0]["condition"])

    def test_prepare_passes_single_benchmark_task_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = campaign.build_parser().parse_args(
                [
                    "prepare",
                    "--campaign-dir",
                    str(root / "campaign"),
                    "--benchmarks",
                    "zsoft-detect",
                    "--task-id",
                    "libxml2-detect",
                    "--methods",
                    "goal-plus-pi",
                ]
            )
            with mock.patch.object(standalone, "prepare", return_value=0) as prepared:
                self.assertEqual(campaign.prepare_campaign(args), 0)

            prepared_args = prepared.call_args.args[0]
            self.assertEqual(prepared_args.task_id, "libxml2-detect")
            payload = json.loads((root / "campaign/campaign.json").read_text())
            self.assertEqual(payload["task_id"], "libxml2-detect")

    def test_prepare_passes_goal_plus_spawn_mode_through(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = campaign.build_parser().parse_args(
                [
                    "prepare",
                    "--campaign-dir",
                    str(root / "campaign"),
                    "--benchmarks",
                    "zsoft-detect",
                    "--task-id",
                    "civetweb-detect",
                    "--methods",
                    "goal-plus-pi",
                    "--concurrency",
                    "2",
                    "--goal-plus-spawn-mode",
                    "fork",
                ]
            )
            with mock.patch.object(standalone, "prepare", return_value=0) as prepared:
                self.assertEqual(campaign.prepare_campaign(args), 0)

            prepared_args = prepared.call_args.args[0]
            self.assertEqual(prepared_args.goal_plus_spawn_mode, "fork")
            payload = json.loads((root / "campaign/campaign.json").read_text())
            self.assertEqual(payload["goal_plus_spawn_mode"], "fork")

    def test_prepare_rejects_fork_without_pi_worker_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = [
                "prepare",
                "--campaign-dir",
                str(root / "campaign"),
                "--benchmarks",
                "zsoft-detect",
                "--task-id",
                "civetweb-detect",
                "--goal-plus-spawn-mode",
                "fork",
            ]
            with self.assertRaisesRegex(ValueError, "fork requires goal-plus-pi"):
                campaign.prepare_campaign(
                    campaign.build_parser().parse_args(
                        base + ["--methods", "goal-plus-codex", "--concurrency", "2"]
                    )
                )
            with self.assertRaisesRegex(ValueError, "fork requires goal-plus-pi"):
                campaign.prepare_campaign(
                    campaign.build_parser().parse_args(
                        base + ["--methods", "goal-plus-pi", "--concurrency", "1"]
                    )
                )

    def test_prepare_expands_paired_condition_seed_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = campaign.build_parser().parse_args(
                [
                    "prepare",
                    "--campaign-dir",
                    str(root / "campaign"),
                    "--benchmarks",
                    "local-vliw",
                    "--conditions",
                    "B0",
                    "B3",
                    "--seeds",
                    "1",
                    "2",
                ]
            )
            with (
                mock.patch.object(standalone, "prepare", return_value=0) as prepared,
                mock.patch.object(
                    standalone,
                    "build_parser",
                    side_effect=AssertionError("campaign must use the config API"),
                ),
            ):
                self.assertEqual(campaign.prepare_campaign(args), 0)
            payload = json.loads((root / "campaign/campaign.json").read_text())
            self.assertEqual(prepared.call_count, 4)
            first_prepare = prepared.call_args_list[0].args[0]
            self.assertEqual(first_prepare.command, "prepare")
            self.assertEqual(first_prepare.reasoning_effort, "high")
            self.assertEqual(payload["state"], "prepared")
            self.assertEqual(payload["budget"]["live_search_concurrency"], 2)
            self.assertEqual(payload["budget"]["cell_concurrency"], 1)
            self.assertEqual(payload["budget"]["attempts"], 2)
            self.assertEqual(
                {
                    (cell["condition"], cell["seed"], cell["effective_concurrency"])
                    for cell in payload["cells"]
                },
                {("B0", 1, 1), ("B0", 2, 1), ("B3", 1, 2), ("B3", 2, 2)},
            )
            self.assertTrue((root / "campaign/campaign-summary.json").is_file())

    def test_run_resumes_without_reexecuting_terminal_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = []
            for condition_id, state in (("B0", "finished"), ("B1", "prepared")):
                run_dir = root / condition_id
                run_dir.mkdir()
                (run_dir / "experiment.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "finished" if state == "finished" else "prepared",
                            "method": "plain-codex",
                            "condition": {"id": condition_id},
                            "benchmark_adapter": "local-vliw",
                            "task_id": "vliw",
                            "model": "test-model",
                            "seed": 1,
                            "budget": {"wall_time_seconds": 10, "concurrency": 2},
                            "task": {"primary_metric": "cycles", "direction": "minimize"},
                            "execution": {},
                        }
                    )
                    + "\n"
                )
                cells.append(
                    {
                        "cell_id": condition_id.lower(),
                        "benchmark_id": "local-vliw",
                        "condition": condition_id,
                        "coordination_variant": None,
                        "method": "plain-codex",
                        "seed": 1,
                        "effective_concurrency": 1 if condition_id == "B0" else 2,
                        "run_dir": str(run_dir),
                        "state": state,
                        "error": None,
                    }
                )
            (root / "campaign.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "campaign_id": "test",
                        "state": "prepared",
                        "conditions": ["B0", "B1"],
                        "model": "test-model",
                        "budget": {},
                        "thresholds": {},
                        "cells": cells,
                    }
                )
                + "\n"
            )
            args = campaign.build_parser().parse_args(
                ["run", "--campaign", str(root), "--model", "test-model"]
            )

            def finish(run_args: argparse.Namespace) -> int:
                path = run_args.run_dir / "experiment.json"
                manifest = json.loads(path.read_text())
                manifest["status"] = "finished"
                path.write_text(json.dumps(manifest) + "\n")
                return 0

            def run_cell_inprocess(command: list[str], **kwargs: object) -> mock.Mock:
                child_args = campaign.build_parser().parse_args(command[2:])
                with mock.patch.dict(campaign.os.environ, kwargs["env"], clear=True):
                    returncode = campaign.run_cell(child_args)
                return mock.Mock(poll=mock.Mock(return_value=returncode))

            with (
                mock.patch.object(standalone, "execute", side_effect=finish) as execute,
                mock.patch.object(
                    campaign.subprocess, "Popen", side_effect=run_cell_inprocess
                ) as popen,
                mock.patch.dict(
                    campaign.os.environ,
                    {"OPENAI_BASE_URL": "https://provider.example/v1"},
                ),
                mock.patch.object(
                    standalone,
                    "build_parser",
                    side_effect=AssertionError("campaign must use the config API"),
                ),
            ):
                self.assertEqual(campaign.run_campaign(args), 0)
            self.assertEqual(execute.call_count, 1)
            popen.assert_called_once()
            self.assertNotIn("provider.example", " ".join(popen.call_args.args[0]))
            self.assertEqual(
                execute.call_args.args[0].api_base,
                "https://provider.example/v1",
            )
            final = json.loads((root / "campaign.json").read_text())
            self.assertEqual(final["state"], "finished")
            self.assertNotIn("provider.example", (root / "campaign.json").read_text())

    def test_trajectory_reports_directional_auc_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / "history.jsonl"
            started = datetime(2026, 1, 1, tzinfo=timezone.utc)
            reports = [
                {
                    "valid": True,
                    "primary_metric": {"value": 8.0},
                    "evaluated_at": (started + timedelta(seconds=2)).isoformat(),
                },
                {
                    "valid": True,
                    "primary_metric": {"value": 5.0},
                    "evaluated_at": (started + timedelta(seconds=6)).isoformat(),
                },
            ]
            history.write_text("\n".join(json.dumps(item) for item in reports) + "\n")
            metrics = campaign.trajectory_metrics(
                root,
                {
                    "execution_started_at": started.isoformat(),
                    "budget": {"wall_time_seconds": 10},
                },
                seed_score=10.0,
                direction="minimize",
                threshold=6.0,
            )
            self.assertEqual(metrics["time_to_threshold_seconds"], 6.0)
            self.assertEqual(metrics["evaluator_call_to_threshold"], 2)
            self.assertAlmostEqual(metrics["directional_improvement_auc"], 28.0)

    def test_missing_compute_telemetry_is_not_reported_as_zero(self) -> None:
        summary = campaign.condition_summaries(
            [
                {
                    "condition": "B0",
                    "status": "prepared",
                    "execution": {
                        "evaluator_calls": {"coverage": "missing"},
                        "usage": {"coverage": "missing"},
                    },
                }
            ]
        )[0]
        self.assertIsNone(summary["total_evaluator_calls"])
        self.assertIsNone(summary["total_input_tokens"])
        self.assertEqual(summary["evaluator_call_coverage"], "0/1 cells")

    def test_search_space_metrics_distinguish_observed_and_enforced_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            space = run_dir / "search-space"
            (space / "plans").mkdir(parents=True)
            (space / "events").mkdir()
            (space / "config.json").write_text(
                json.dumps({"mode": "observe", "protocol_version": "v1"})
            )
            (space / "state.json").write_text(json.dumps({"evidence_revision": 1}))
            (space / "events/se-1.json").write_text(
                json.dumps({"event_id": "se-1", "candidate_id": "c1"})
            )
            (space / "plans/ip-1.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "candidate_id": "c2",
                        "proposal": {
                            "evidence_refs": ["se-1"],
                            "footprint": {"artifact": ["solver.py"]},
                        },
                        "review": {"decision": "reject"},
                    }
                )
            )
            metrics = collect_search_space_state(run_dir)
            self.assertEqual(metrics["semantic_duplicate_reviews"], 1)
            self.assertEqual(metrics["enforced_rejections"], 0)
            self.assertEqual(metrics["cross_lineage_evidence_references"], 1)


if __name__ == "__main__":
    unittest.main()
