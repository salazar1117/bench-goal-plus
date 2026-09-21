from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.zsoft_detect import adapter
from experiments.benchmark_compare import experiment as benchmark_experiment


class AdapterContractTest(unittest.TestCase):
    def test_declares_raw_metric_contract(self) -> None:
        self.assertIn(adapter.DIRECTION, {"minimize", "maximize"})
        self.assertTrue(adapter.PRIMARY_METRIC)
        self.assertTrue(adapter.ARTIFACT_NAME)
        self.assertEqual(adapter.EVALUATION_MODE, "blind")
        self.assertEqual(adapter.PRIMARY_METRIC, "f1")
        self.assertEqual(adapter.GOAL_PLUS_PROCESS_METRIC, "format_valid")
        self.assertLess(adapter.VERIFIER_TIMEOUT_SECONDS, 1800)
        self.assertEqual(
            adapter.UPSTREAM_SUBDIR,
            "benchmarks/vulnerability/zsoft-detect",
        )
        self.assertEqual(adapter.PI_WORKER_SANDBOX["engine"], "bubblewrap")
        self.assertEqual(
            adapter.PI_WORKER_SANDBOX["evaluation_mode"], "blind"
        )
        self.assertEqual(adapter.PI_WORKER_SANDBOX["workspace_access"], "read_only")
        self.assertEqual(
            adapter.PI_WORKER_SANDBOX["read_only_workspace_paths"],
            ["source", "schemas"],
        )
        self.assertEqual(
            adapter.PI_WORKER_SANDBOX["writable_workspace_paths"], ["submission"]
        )

    def test_project_catalog_is_pinned(self) -> None:
        self.assertEqual(
            set(adapter.list_projects()),
            set(adapter.PROJECT_COMMITS),
        )

    def test_configure_task_rejects_unknown_project(self) -> None:
        with self.assertRaises(adapter.AdapterError):
            adapter.configure_task("no-such-project")

    def test_configure_task_updates_persisted_task_id(self) -> None:
        self.addCleanup(adapter.configure_task, None)
        adapter.configure_task("libxml2-detect")

        self.assertEqual(adapter.ACTIVE_PROJECT, "libxml2")
        self.assertEqual(adapter.TASK_ID, "libxml2-detect")

    def test_bench_contract_is_public_only(self) -> None:
        contract = adapter.bench_contract("civetweb")
        self.assertEqual(contract["project_id"], "civetweb")
        self.assertTrue(contract["scan_roots"])
        self.assertNotIn("applicability", json.dumps(contract))
        self.assertNotIn("cases", contract)

    def test_materialize_tracks_source_and_empty_submission(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        workspace = tmp / "workspace"
        source_checkout = tmp / "workspace-source"
        source_checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(source_checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source_checkout),
                "config",
                "user.email",
                "test@example.com",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source_checkout), "config", "user.name", "Test"],
            check=True,
        )
        (source_checkout / "src").mkdir()
        (source_checkout / "src" / "civetweb.c").write_text("/* fixture */\n")
        subprocess.run(["git", "-C", str(source_checkout), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source_checkout), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(source_checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        contract = adapter.bench_contract("civetweb")
        previous_commit = adapter.PROJECT_COMMITS["civetweb"]
        adapter.PROJECT_COMMITS["civetweb"] = commit
        self.addCleanup(
            adapter.PROJECT_COMMITS.__setitem__, "civetweb", previous_commit
        )

        adapter.configure_task(None)
        with mock.patch.object(adapter, "bench_contract", return_value=contract):
            adapter.materialize_workspace(adapter.ZSOFT_ROOT, workspace)

        tracked = subprocess.run(
            ["git", "-C", str(workspace), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertIn("source/src/civetweb.c", tracked)
        self.assertIn("submission/.gitkeep", tracked)
        self.assertIn("schemas/finding.schema.json", tracked)
        self.assertFalse((workspace / "source" / ".git").exists())
        metadata = json.loads((workspace / "task.json").read_text())
        self.assertEqual(metadata["source_revision"], commit)
        self.assertNotIn("upstream_root", metadata)
        self.assertEqual(metadata["primary_metric"], "format_valid")
        self.assertTrue((workspace / "public_check.py").is_file())
        self.assertFalse((workspace / "evaluate.py").exists())
        self.assertFalse((workspace / ".goal-plus-verifiers").exists())

    def test_materialize_copies_an_explicit_validated_source_cache(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        contract = adapter.bench_contract("civetweb")
        source = tmp / "source-cache"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "config", "user.name", "Test"], check=True
        )
        (source / "src").mkdir()
        (source / "src" / "civetweb.c").write_text("/* fixture */\n")
        (source / "outside.c").write_text("/* outside scan roots */\n")
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        previous_commit = adapter.PROJECT_COMMITS["civetweb"]
        adapter.PROJECT_COMMITS["civetweb"] = commit
        self.addCleanup(
            adapter.PROJECT_COMMITS.__setitem__, "civetweb", previous_commit
        )

        workspace = tmp / "workspace"
        adapter.configure_task(None)
        with (
            mock.patch.dict(os.environ, {adapter.SOURCE_CACHE_ENV: str(source)}),
            mock.patch.object(adapter, "bench_contract", return_value=contract),
        ):
            adapter.materialize_workspace(adapter.ZSOFT_ROOT, workspace)

        self.assertTrue((workspace / "source").is_dir())
        self.assertFalse((workspace / "source").is_symlink())
        self.assertTrue((workspace / "source" / "src" / "civetweb.c").is_file())
        self.assertFalse((workspace / "source" / "outside.c").exists())
        self.assertFalse((workspace / "source" / ".git").exists())
        self.assertEqual(
            json.loads((workspace / "schemas" / "finding.schema.json").read_text()),
            json.loads(
                (adapter.BENCHMARK_ROOT / "schemas" / "finding.schema.json").read_text()
            ),
        )
        metadata = json.loads((workspace / "task.json").read_text())
        self.assertEqual(
            metadata["source_materialization"], "validated_local_cache_scan_roots"
        )

    def test_scan_roots_copy_only_declared_files_and_directories(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        source = root / "source"
        (source / "net" / "rxrpc").mkdir(parents=True)
        (source / "net" / "rxrpc" / "call.c").write_text("/* included */\n")
        (source / "net" / "other.c").write_text("/* excluded */\n")
        (source / "single.c").write_text("/* included */\n")

        roots = adapter._validated_scan_roots(
            {"scan_roots": ["net/rxrpc", "single.c"]}, source
        )

        self.assertEqual(
            [(relative.as_posix(), path.name) for relative, path in roots],
            [("net/rxrpc", "rxrpc"), ("single.c", "single.c")],
        )

    def test_scan_roots_reject_invalid_missing_and_escaping_paths(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        source = root / "source"
        source.mkdir()
        (source / "valid").mkdir()
        outside = root / "outside"
        outside.mkdir()
        (source / "escape").symlink_to(outside, target_is_directory=True)

        invalid = ["/absolute", "../outside", "valid/../valid", "missing", "escape"]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(adapter.AdapterError, "scan root"):
                    adapter._validated_scan_roots({"scan_roots": [value]}, source)

    def test_campaign_local_checkout_must_be_exact_and_clean(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        source = tmp / "workspace-source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        (source / "partial.c").write_text("untracked\n")

        adapter.configure_task(None)
        with self.assertRaisesRegex(adapter.AdapterError, "Git top level|wrong commit"):
            adapter.materialize_workspace(adapter.ZSOFT_ROOT, tmp / "workspace")
        self.assertTrue((source / "partial.c").is_file())
        self.assertFalse((tmp / "workspace").exists())

    def test_source_copy_preserves_internal_symlinks_and_rejects_escape(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        source = tmp / "source"
        source.mkdir()
        (source / "target.h").write_text("fixture\n")
        (source / "internal.h").symlink_to("target.h")
        destination = tmp / "destination"

        from adapters.portable import copytree_confined

        copytree_confined(source, destination, label="fixture")
        self.assertTrue((destination / "internal.h").is_symlink())
        self.assertEqual(os.readlink(destination / "internal.h"), "target.h")

        (source / "escape").symlink_to("../private")
        with self.assertRaisesRegex(RuntimeError, "symlink escapes"):
            copytree_confined(source, tmp / "rejected", label="fixture")

    def test_materialization_paths_must_be_disjoint(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        source = root / "source"
        source.mkdir()
        with self.assertRaisesRegex(adapter.AdapterError, "paths overlap"):
            adapter._require_disjoint_paths(
                workspace=source / "workspace",
                source_checkout=source,
                benchmark_root=adapter.BENCHMARK_ROOT,
            )

    def test_public_evaluate_checks_format_without_official_scorer(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        workspace = tmp / "workspace"
        (workspace / adapter.ARTIFACT_NAME).mkdir(parents=True)
        (workspace / "task.json").write_text(
            json.dumps(
                {
                    "task_id": adapter.TASK_ID,
                    "project_id": adapter.DEFAULT_PROJECT,
                    "commit": adapter.project_commit(adapter.DEFAULT_PROJECT),
                }
            )
        )

        with mock.patch.object(adapter, "_run") as scorer:
            report = adapter.evaluate_workspace(
                workspace, Path("/not-visible-to-public-check"), "public"
            )

        scorer.assert_not_called()
        self.assertTrue(report["valid"])
        self.assertEqual(report[adapter.GOAL_PLUS_PROCESS_METRIC], 1.0)
        self.assertNotIn(adapter.PRIMARY_METRIC, report)
        self.assertNotIn("zsoft_score", report)
        self.assertEqual(report["budget"]["total_claimed"], 1)

    def test_evaluate_rejects_submission_symlinks_before_scoring(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        workspace = tmp / "workspace"
        submission = workspace / adapter.ARTIFACT_NAME
        submission.mkdir(parents=True)
        (submission / "finding.json").symlink_to(
            adapter.BENCHMARK_ROOT / "schemas" / "finding.schema.json"
        )
        (workspace / "task.json").write_text(
            json.dumps(
                {
                    "task_id": adapter.TASK_ID,
                    "project_id": adapter.DEFAULT_PROJECT,
                    "commit": adapter.project_commit(adapter.DEFAULT_PROJECT),
                }
            )
        )

        with mock.patch.object(adapter, "_run") as scorer:
            report = adapter.evaluate_workspace(
                workspace, adapter.BENCHMARK_ROOT, "public"
            )

        scorer.assert_not_called()
        self.assertFalse(report["valid"])
        self.assertEqual(report[adapter.GOAL_PLUS_PROCESS_METRIC], 0.0)
        self.assertNotIn(adapter.PRIMARY_METRIC, report)
        self.assertIn("symlink", json.dumps(report["public_diagnostics"]))

    def test_official_empty_submission_seed_is_scored_only_in_final_mode(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        workspace = tmp / "workspace"
        (workspace / adapter.ARTIFACT_NAME).mkdir(parents=True)
        (workspace / "task.json").write_text(json.dumps({
            "task_id": "civetweb-detect", "project_id": "civetweb",
            "commit": adapter.project_commit("civetweb"),
        }))
        public = adapter.evaluate_workspace(workspace, adapter.ZSOFT_ROOT, "public")
        self.assertTrue(public["valid"])
        self.assertNotIn("f1", public)
        final = adapter.evaluate_workspace(workspace, adapter.ZSOFT_ROOT, "final")
        self.assertTrue(final["valid"], final["message"])
        self.assertEqual(final["f1"], 0.0)
        self.assertEqual(final["zsoft_score"]["f1"], 0.0)
        self.assertEqual(final["primary_metric"]["direction"], "maximize")
        self.assertFalse((workspace / "submission/score.json").exists())

    def test_adapter_cli_evaluates_candidate_submission(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        workspace = tmp / "workspace"
        (workspace / adapter.ARTIFACT_NAME).mkdir(parents=True)
        (workspace / "task.json").write_text(
            json.dumps(
                {
                    "task_id": adapter.TASK_ID,
                    "project_id": adapter.DEFAULT_PROJECT,
                    "commit": adapter.project_commit(adapter.DEFAULT_PROJECT),
                }
            )
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(adapter.CONTROLLER_PATH),
                "evaluate",
                "--workspace",
                str(workspace),
                "--upstream-root",
                str(adapter.BENCHMARK_ROOT),
                "--mode",
                "public",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(completed.stdout)
        self.assertTrue(report["valid"])
        self.assertEqual(report[adapter.GOAL_PLUS_PROCESS_METRIC], 1.0)
        self.assertNotIn(adapter.PRIMARY_METRIC, report)

    def test_git_commit_supports_shared_runtime_checkouts(self) -> None:
        self.assertRegex(
            adapter.git_commit(ROOT),
            r"^[0-9a-f]{40}$",
        )
        self.assertRegex(adapter.git_commit(adapter.ZSOFT_ROOT), r"^[0-9a-f]{40}$")

    def test_posthoc_selection_is_outside_worker_workspace_and_preserves_search_report(
        self,
    ) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run_dir = tmp / "cell"
        workspace = run_dir / "workspace"
        search_run = workspace / ".gp" / "runs" / "run_fixture"
        repository = search_run / "workspace" / "c001"
        candidate_dir = search_run / "candidates" / "c001"
        repository.mkdir(parents=True)
        candidate_dir.mkdir(parents=True)
        (workspace / "task.json").write_text(
            json.dumps(
                {
                    "task_id": "civetweb-detect",
                    "project_id": "civetweb",
                    "commit": "1" * 40,
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "config",
                "user.email",
                "test@example.com",
            ],
            check=True,
        )
        submission = repository / "submission"
        submission.mkdir()
        finding = submission / "finding.json"
        finding.write_text('{"fixture":"first"}\n', encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-q", "-m", "first"],
            check=True,
        )
        first = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        finding.write_text('{"fixture":"second"}\n', encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-q", "-m", "second"],
            check=True,
        )
        second = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        (repository / "controller-note").write_text("same artifact\n")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-q", "-m", "third"],
            check=True,
        )
        third = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        (candidate_dir / "candidate.json").write_text(
            json.dumps(
                {
                    "candidate_id": "c001",
                    "iterations": [
                        {"iteration": 1, "git_head": first, "artifact_hash": "a",
                         "score": 1.0, "process_passed": True, "git_artifact_clean": True},
                        {
                            "iteration": 2,
                            "git_head": second,
                            "artifact_hash": "b",
                            "score": 1.0, "process_passed": True,
                            "artifact_clean": True, "git_artifact_clean": None,
                            "disposition": "discard",
                        },
                        {"iteration": 3, "git_head": third, "artifact_hash": "b",
                         "score": 1.0, "process_passed": True, "git_artifact_clean": True},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (search_run / "report.md").write_text(
            "# Search Report: run_fixture\n\n"
            "- Metric: `format_valid` (maximize)\n\n"
            "## Results Ledgers\n\n"
            "Each candidate workspace owns the complete inherited verifier ledger.\n\n"
            "| Candidate | Ledger | Rows | Latest Commit | Latest Score | "
            "Latest Status | Latest Hypothesis |\n"
            "|---|---|---:|---|---:|---|---|\n"
            "| `c001` | results.tsv | 3 | deadbeef | 1.0 | pass | fixture |\n",
            encoding="utf-8",
        )

        def score_snapshot(
            evaluated_workspace: Path,
            mode: str,
            controller_runtime: Path,
            benchmark_root: Path,
        ) -> dict[str, object]:
            self.assertEqual(mode, "final")
            self.assertEqual(benchmark_root, tmp / "benchmark")
            self.assertTrue(controller_runtime.is_relative_to(run_dir))
            payload = (
                evaluated_workspace / "submission" / "finding.json"
            ).read_text()
            f1 = 0.25 if "first" in payload else 0.75
            return {
                "mode": "final",
                "valid": True,
                "format_valid": True,
                "f1": f1,
                "primary_metric": {"name": "f1", "direction": "maximize", "value": f1},
                "zsoft_score": {
                    "f1": f1,
                    "precision": f1,
                    "recall": f1,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                },
            }

        benchmark_experiment.configure_adapter("zsoft-detect")
        self.addCleanup(benchmark_experiment.configure_adapter, "heurigym")
        with mock.patch.object(
            benchmark_experiment,
            "evaluate_with_controller_runtime",
            side_effect=score_snapshot,
        ) as evaluator:
            summary = benchmark_experiment.finalize_posthoc_official_selection(
                run_dir=run_dir,
                workspace=workspace,
                benchmark_root=tmp / "benchmark",
                closeout={"completed": True, "runs": [
                    {"run_id": "run_fixture", "final_state": "promoted"}
                ]},
                contract=adapter.GOAL_PLUS_POSTHOC_SELECTION_CONTRACT,
                worker_shutdown_verified=True,
            )

        self.assertTrue(summary["completed"])
        self.assertEqual(summary["eligible_iteration_count"], 3)
        self.assertEqual(summary["official_evaluator_calls"], 2)
        self.assertEqual(summary["artifact_cache_hits"], 1)
        self.assertEqual(len(summary["duplicate_artifact_groups"]), 1)
        group = summary["duplicate_artifact_groups"][0]
        self.assertFalse(group["cross_candidate"])
        self.assertEqual(
            [member["iteration"] for member in group["iterations"]], [2, 3]
        )
        self.assertEqual(len(summary["warnings"]), 1)
        self.assertIn(
            "identical_artifacts_within_candidate", summary["warnings"][0]
        )
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(summary["scores"][1]["iteration"], 2)
        self.assertEqual(summary["scores"][1]["f1"], 0.75)
        report = json.loads((run_dir / "posthoc-candidate-scores.json").read_text())
        self.assertEqual(report["selected"]["iteration"], 3)
        self.assertEqual(report["selected"]["f1"], 0.75)
        final = json.loads((run_dir / "final-eval.json").read_text())
        self.assertEqual(final["f1"], 0.75)
        self.assertFalse((workspace / "posthoc-candidate-scores.json").exists())
        search_report = (search_run / "report.md").read_text(encoding="utf-8")
        self.assertIn("Metric: `format_valid`", search_report)
        self.assertNotIn("0.75", search_report)

    def test_posthoc_flags_identical_artifacts_across_candidates(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run_dir = tmp / "cell"
        workspace = run_dir / "workspace"
        search_run = workspace / ".gp" / "runs" / "run_fixture"
        (search_run / "candidates").mkdir(parents=True)
        (search_run / "workspace").mkdir(parents=True)
        (workspace / "task.json").write_text(
            json.dumps(
                {
                    "task_id": "civetweb-detect",
                    "project_id": "civetweb",
                    "commit": "1" * 40,
                }
            ),
            encoding="utf-8",
        )

        def commit_identical_artifact(candidate_id: str) -> str:
            repository = search_run / "workspace" / candidate_id
            (repository / "submission").mkdir(parents=True)
            (
                repository / "submission" / "finding.json"
            ).write_text('{"fixture":"shared"}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C", str(repository),
                    "config", "user.email", "test@example.com",
                ],
                check=True,
            )
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", candidate_id],
                check=True,
            )
            (search_run / "candidates" / candidate_id).mkdir(parents=True)
            (
                search_run / "candidates" / candidate_id / "candidate.json"
            ).write_text(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "iterations": [
                            {
                                "iteration": 1,
                                "git_head": subprocess.run(
                                    ["git", "-C", str(repository), "rev-parse", "HEAD"],
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                ).stdout.strip(),
                                "artifact_hash": "shared",
                                "score": 1.0,
                                "process_passed": True,
                                "git_artifact_clean": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return candidate_id

        commit_identical_artifact("c001")
        commit_identical_artifact("c002")

        def score_snapshot(
            evaluated_workspace: Path,
            mode: str,
            controller_runtime: Path,
            benchmark_root: Path,
        ) -> dict[str, object]:
            f1 = 0.5
            return {
                "mode": "final",
                "valid": True,
                "format_valid": True,
                "f1": f1,
                "primary_metric": {"name": "f1", "direction": "maximize", "value": f1},
                "zsoft_score": {
                    "f1": f1, "precision": f1, "recall": f1,
                    "tp": 1, "fp": 0, "fn": 0,
                },
            }

        benchmark_experiment.configure_adapter("zsoft-detect")
        self.addCleanup(benchmark_experiment.configure_adapter, "heurigym")
        with mock.patch.object(
            benchmark_experiment,
            "evaluate_with_controller_runtime",
            side_effect=score_snapshot,
        ):
            summary = benchmark_experiment.finalize_posthoc_official_selection(
                run_dir=run_dir,
                workspace=workspace,
                benchmark_root=tmp / "benchmark",
                closeout={"completed": True, "runs": [
                    {"run_id": "run_fixture", "final_state": "promoted"}
                ]},
                contract=adapter.GOAL_PLUS_POSTHOC_SELECTION_CONTRACT,
                worker_shutdown_verified=True,
            )

        self.assertTrue(summary["completed"])
        self.assertEqual(summary["eligible_iteration_count"], 2)
        self.assertEqual(summary["unique_artifact_count"], 1)
        self.assertEqual(summary["official_evaluator_calls"], 1)
        self.assertEqual(summary["artifact_cache_hits"], 1)
        self.assertEqual(len(summary["duplicate_artifact_groups"]), 1)
        group = summary["duplicate_artifact_groups"][0]
        self.assertTrue(group["cross_candidate"])
        self.assertEqual(
            sorted(member["candidate_id"] for member in group["iterations"]),
            ["c001", "c002"],
        )
        self.assertEqual(len(summary["warnings"]), 1)
        self.assertIn(
            "identical_artifacts_across_candidates", summary["warnings"][0]
        )
        # The warning is diagnostic only: selection follows the contract as before.
        self.assertEqual(summary["selected"]["candidate_id"], "c001")
        self.assertEqual(summary["selected"]["f1"], 0.5)

    def test_posthoc_requires_public_process_and_clean_artifact(self) -> None:
        eligible = {
            "iteration": 1, "git_head": "a" * 40, "score": 1.0,
            "process_passed": True, "artifact_clean": True,
            "git_artifact_clean": None, "disposition": "discard",
        }
        self.assertTrue(benchmark_experiment._publicly_compliant_iteration(eligible))
        for changed in (
            {"process_passed": False}, {"artifact_clean": False, "git_artifact_clean": True},
            {"touched_denied_files": True}, {"changed_outside_allowed": True},
            {"score": None}, {"score": float("nan")},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(benchmark_experiment._publicly_compliant_iteration(
                    {**eligible, **changed}
                ))


if __name__ == "__main__":
    unittest.main()
