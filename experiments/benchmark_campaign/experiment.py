#!/usr/bin/env python3
"""Prepare, run, monitor, and summarize generic benchmark campaigns."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.registry import adapter_modules  # noqa: E402
from bench_artifacts import (  # noqa: E402
    portable_path,
    read_json,
    sanitize_id,
    utc_now,
    write_json_atomic as write_json,
)
from bench_runtime_paths import configure_temp_environment  # noqa: E402
from experiments.benchmark_compare.conditions import CONDITIONS  # noqa: E402
from experiments.benchmark_compare import experiment as standalone  # noqa: E402
from experiments.openevolve_compare.reporting import collect_run, numeric  # noqa: E402


DEFAULT_RUNS = ROOT / "runs/benchmark-campaigns"
DEFAULT_CONDITIONS = ("B0", "B1", "B3", "B4")
TERMINAL_STATES = {"finished", "incomplete", "failed", "prepare_failed"}
_stop_requested = False

PI_API_BASE_RUNTIME_ENV = "BENCH_GOAL_PLUS_PI_API_BASE"


@dataclass
class ActiveCell:
    cell: dict[str, Any]
    process: subprocess.Popen[str]
    stdout: TextIO
    stderr: TextIO

    def close_logs(self) -> None:
        self.stdout.close()
        self.stderr.close()


def parse_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        benchmark_id, separator, raw = value.partition("=")
        if not separator or benchmark_id not in adapter_modules():
            raise ValueError(
                f"threshold must be REGISTERED_BENCHMARK=NUMBER, got {value!r}"
            )
        try:
            threshold = float(raw)
        except ValueError as error:
            raise ValueError(f"threshold must be numeric, got {value!r}") from error
        if not math.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {value!r}")
        thresholds[benchmark_id] = threshold
    return thresholds


def default_campaign_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS / timestamp


def prepare_cell_config(
    args: argparse.Namespace,
    *,
    benchmark_id: str,
    condition_id: str | None,
    method: str,
    seed: int,
    run_dir: Path,
) -> standalone.PrepareConfig:
    condition = CONDITIONS[condition_id] if condition_id is not None else None
    concurrency = (
        condition.effective_concurrency(args.concurrency)
        if condition is not None
        else args.concurrency
    )
    return standalone.PrepareConfig(
        benchmark=benchmark_id,
        method=method,
        task_id=args.task_id,
        shared_dir=args.shared_dir,
        shared_cache=getattr(args, "shared_cache", False),
        condition=condition_id,
        coordination_variant=(
            condition.coordination_variant if condition is not None else None
        ),
        model=args.model,
        pi_provider_id=args.pi_provider_id,
        pi_api=args.pi_api,
        pi_api_key_env=args.pi_api_key_env,
        wall_time_seconds=args.wall_time_seconds,
        concurrency=concurrency,
        soft_closeout_seconds=args.soft_closeout_seconds,
        hard_kill_grace_seconds=args.hard_kill_grace_seconds,
        worker_runtime_seconds=args.worker_runtime_seconds,
        worker_min_runtime_seconds=args.worker_min_runtime_seconds,
        seed=seed,
        reasoning_effort=args.reasoning_effort,
        run_dir=run_dir,
        environment_manifest=args.environment_manifest,
        checkout_root=args.checkout_root,
        venv=args.venv,
    )


def prepare_campaign(args: argparse.Namespace) -> int:
    unknown = set(args.benchmarks) - set(adapter_modules())
    if unknown:
        raise ValueError(f"unknown benchmark adapters: {', '.join(sorted(unknown))}")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        raise ValueError("benchmark ids must be unique")
    if args.task_id is not None and len(args.benchmarks) != 1:
        raise ValueError("--task-id requires exactly one benchmark")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("conditions must be unique")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be unique")
    if args.methods and args.conditions:
        raise ValueError("choose either --methods or --conditions, not both")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    selected_conditions = tuple(
        args.conditions or (() if args.methods else DEFAULT_CONDITIONS)
    )
    for condition_id in selected_conditions:
        condition = CONDITIONS[condition_id]
        if not condition.implemented:
            raise ValueError(f"{condition_id} is not implemented: {condition.limitation}")
    if args.concurrency < 2 and any(item != "B0" for item in selected_conditions):
        raise ValueError("B1/B3/B4 campaigns require --concurrency >= 2")
    axes = (
        [(None, method) for method in args.methods]
        if args.methods
        else [
            (condition_id, str(CONDITIONS[condition_id].method))
            for condition_id in selected_conditions
        ]
    )
    pi_provider = _campaign_provider(args, methods=[method for _, method in axes])

    campaign_dir = (args.campaign_dir or default_campaign_dir()).expanduser().absolute()
    campaign_dir.mkdir(parents=True, exist_ok=False)
    thresholds = parse_thresholds(args.threshold)
    campaign = {
        "schema_version": 1,
        "campaign_id": campaign_dir.name,
        "state": "preparing",
        "prepared_at": utc_now(),
        "benchmarks": args.benchmarks,
        "task_id": args.task_id,
        "shared_dir": args.shared_dir,
        "shared_cache": getattr(args, "shared_cache", False),
        "conditions": list(selected_conditions),
        "methods": args.methods,
        "seeds": args.seeds,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "pi_provider": pi_provider,
        "budget": {
            "wall_time_seconds": args.wall_time_seconds,
            "live_search_concurrency": args.concurrency,
            "requested_live_concurrency": args.concurrency,
            "cell_concurrency": 1,
            "attempts": len(args.seeds),
            "soft_closeout_seconds": args.soft_closeout_seconds,
            "hard_kill_grace_seconds": args.hard_kill_grace_seconds,
            "worker_runtime_seconds": args.worker_runtime_seconds,
            "worker_min_runtime_seconds": args.worker_min_runtime_seconds,
            "total_compute_accounting": "actual tokens and evaluator calls after execution",
            "wall_clock_accounting": "fixed T with condition-specific live K",
        },
        "thresholds": thresholds,
        "environment": {
            "manifest": str(args.environment_manifest.expanduser().absolute()),
            "checkout_root": str(args.checkout_root.expanduser().absolute()),
            "venv": str(args.venv.expanduser().absolute()),
        },
        "cells": [],
        "secret_policy": "credentials and provider URL are never serialized",
    }
    campaign_path = campaign_dir / "campaign.json"
    write_json(campaign_path, campaign)

    for benchmark_id in args.benchmarks:
        for seed in args.seeds:
            for condition_id, method in axes:
                axis_id = condition_id or method
                cell_id = sanitize_id(f"{benchmark_id}-{axis_id}-seed-{seed}")
                run_dir = campaign_dir / "cells" / cell_id
                condition = (
                    CONDITIONS[condition_id] if condition_id is not None else None
                )
                cell = {
                    "cell_id": cell_id,
                    "benchmark_id": benchmark_id,
                    "condition": condition_id,
                    "coordination_variant": (
                        condition.coordination_variant if condition is not None else None
                    ),
                    "method": method,
                    "seed": seed,
                    "effective_concurrency": (
                        condition.effective_concurrency(args.concurrency)
                        if condition is not None
                        else args.concurrency
                    ),
                    "run_dir": str(run_dir),
                    "state": "preparing",
                    "error": None,
                }
                campaign["cells"].append(cell)
                write_json(campaign_path, campaign)
                try:
                    standalone.prepare(
                        prepare_cell_config(
                            args,
                            benchmark_id=benchmark_id,
                            condition_id=condition_id,
                            method=method,
                            seed=seed,
                            run_dir=run_dir,
                        ).to_namespace()
                    )
                    cell["state"] = "prepared"
                except Exception as error:
                    cell["state"] = "prepare_failed"
                    cell["error"] = f"{type(error).__name__}: {error}"
                write_json(campaign_path, campaign)

    campaign["state"] = (
        "prepared"
        if all(cell["state"] == "prepared" for cell in campaign["cells"])
        else "partial"
    )
    campaign["preparation_finished_at"] = utc_now()
    write_json(campaign_path, campaign)
    summarize_campaign(campaign_dir)
    print(campaign_dir)
    return 0 if campaign["state"] == "prepared" else 2


def _handle_stop(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    raise KeyboardInterrupt


def _campaign_provider(
    args: argparse.Namespace, *, methods: list[str]
) -> dict[str, Any] | None:
    """Resolve the non-secret Pi provider contract for this campaign."""
    if not any(method in {"plain-pi", "goal-plus-pi"} for method in methods):
        return None
    return {
        "id": args.pi_provider_id,
        "api": args.pi_api,
        "api_key_env": args.pi_api_key_env,
        "api_base_env": args.pi_api_base_env,
    }


def _provider_matches(
    prepared: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> bool:
    if prepared is None and runtime is None:
        return True
    if prepared is None or runtime is None:
        return False
    # Provider URL values and credentials are never serialized; only env names are.
    return (
        prepared.get("id") == runtime.get("id")
        and prepared.get("api") == runtime.get("api")
        and (prepared.get("api_key_env") or None) == (runtime.get("api_key_env") or None)
        and (prepared.get("api_base_env") or None)
        == (runtime.get("api_base_env") or None)
    )


def cell_run_command(args: argparse.Namespace, cell: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-cell",
        "--run-dir",
        str(cell["run_dir"]),
        "--model",
        args.model,
        "--codex-bin",
        args.codex_bin,
        "--pi-provider-id",
        args.pi_provider_id,
        "--pi-api",
        args.pi_api,
        "--pi-api-key-env",
        args.pi_api_key_env,
    ]


def launch_cell(
    args: argparse.Namespace,
    cell: dict[str, Any],
    *,
    api_base: str | None,
) -> ActiveCell:
    cell["state"] = "running"
    cell["started_at"] = utc_now()
    run_dir = Path(cell["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout = (run_dir / "controller-stdout.log").open("a")
    stderr = (run_dir / "controller-stderr.log").open("a")
    environment = os.environ.copy()
    environment.pop("OPENAI_BASE_URL", None)
    environment.pop("OPENAI_API_BASE_URL", None)
    if api_base:
        environment[PI_API_BASE_RUNTIME_ENV] = api_base
    process = subprocess.Popen(
        cell_run_command(args, cell),
        cwd=ROOT,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    return ActiveCell(cell=cell, process=process, stdout=stdout, stderr=stderr)


def finish_cell(active: ActiveCell) -> None:
    cell = active.cell
    returncode = active.process.poll()
    cell["returncode"] = returncode
    active.close_logs()
    manifest_path = Path(cell["run_dir"]) / "experiment.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            manifest_status = manifest.get("status")
            if manifest_status in TERMINAL_STATES:
                cell["state"] = manifest_status
            else:
                cell["state"] = "failed"
                cell["error"] = (
                    "cell process ended before its experiment reached a terminal state"
                )
        except Exception:
            cell["state"] = "failed"
    elif returncode == 0:
        # no manifest means the cell process was interrupted before writing one
        cell["state"] = "interrupted"
    else:
        cell["state"] = "failed"
    cell["finished_at"] = utc_now()


def terminate_active_cells(
    active_cells: list[ActiveCell], hard_kill_grace_seconds: int
) -> None:
    for active in active_cells:
        process = active.process
        if process.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + hard_kill_grace_seconds
    for active in active_cells:
        process = active.process
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            process.wait()


def run_cell(args: argparse.Namespace) -> int:
    run_args = standalone.RunConfig(
        run_dir=Path(args.run_dir),
        model=args.model,
        codex_bin=args.codex_bin,
        api_base=os.environ.get(PI_API_BASE_RUNTIME_ENV),
        pi_provider_id=args.pi_provider_id,
        pi_api=args.pi_api,
        pi_api_key_env=args.pi_api_key_env,
    ).to_namespace()
    return standalone.execute(run_args)


def run_campaign(args: argparse.Namespace) -> int:
    global _stop_requested
    _stop_requested = False
    campaign_dir = args.campaign.expanduser().absolute()
    campaign_path = campaign_dir / "campaign.json"
    campaign = read_json(campaign_path)
    if args.model != campaign["model"]:
        raise ValueError(
            f"model mismatch: campaign uses {campaign['model']}, got {args.model}"
        )
    prepared_provider = campaign.get("pi_provider")
    runtime_provider = _campaign_provider(
        args, methods=[str(cell["method"]) for cell in campaign["cells"]]
    )
    if not _provider_matches(prepared_provider, runtime_provider):
        raise ValueError(
            "provider mismatch: campaign was prepared for "
            f"{prepared_provider!r}, got {runtime_provider!r}"
        )
    selected = set(args.conditions) if args.conditions else None
    unknown = (selected or set()) - set(campaign.get("conditions") or [])
    if unknown:
        raise ValueError(f"conditions not in campaign: {', '.join(sorted(unknown))}")
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL")
    if runtime_provider is not None:
        api_base = args.api_base or os.environ.get(runtime_provider["api_base_env"])
        if not api_base:
            raise RuntimeError(
                f"{runtime_provider['api_base_env']} is required for the Pi provider"
            )
        if not os.environ.get(runtime_provider["api_key_env"]):
            raise RuntimeError(
                f"{runtime_provider['api_key_env']} is required for the Pi provider"
            )
        generic_base = os.environ.get("OPENAI_BASE_URL")
        if (
            args.api_base is None
            and runtime_provider["api_base_env"] != "OPENAI_BASE_URL"
            and generic_base
            and api_base == generic_base
        ):
            raise RuntimeError(
                "the provider-specific Pi base URL must not alias OPENAI_BASE_URL"
            )
    cell_concurrency = max(
        int(campaign.get("budget", {}).get("cell_concurrency", 1)), 1
    )

    previous_handlers = {
        signum: signal.signal(signum, _handle_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    campaign["state"] = "running"
    campaign["execution_started_at"] = campaign.get("execution_started_at") or utc_now()
    campaign["controller"] = {
        "pid": os.getpid(),
        "current_cell": None,
        "updated_at": utc_now(),
    }
    write_json(campaign_path, campaign)

    active_cells: list[ActiveCell] = []
    pending = [
        cell
        for cell in campaign["cells"]
        if cell["state"] not in TERMINAL_STATES
        and (selected is None or cell["condition"] in selected)
    ]
    try:
        while pending or active_cells:
            if _stop_requested:
                break
            while pending and len(active_cells) < cell_concurrency:
                cell = pending.pop(0)
                active_cells.append(
                    launch_cell(
                        args,
                        cell,
                        api_base=api_base,
                    )
                )
                campaign["controller"].update(
                    {"current_cell": cell["cell_id"], "updated_at": utc_now()}
                )
                write_json(campaign_path, campaign)
            done = [
                active
                for active in active_cells
                if active.process.poll() is not None
            ]
            for active in done:
                finish_cell(active)
                active_cells.remove(active)
                write_json(campaign_path, campaign)
                if args.fail_fast and active.cell["state"] != "finished":
                    pending.clear()
            if active_cells:
                time.sleep(0.1)
    except KeyboardInterrupt:
        _stop_requested = True
    finally:
        if active_cells:
            terminate_active_cells(
                active_cells, int(campaign.get("budget", {}).get("hard_kill_grace_seconds", 10))
            )
            for active in active_cells:
                finish_cell(active)
                write_json(campaign_path, campaign)
            active_cells.clear()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    terminal = [cell for cell in campaign["cells"] if cell["state"] in TERMINAL_STATES]
    selected_cells = [
        cell
        for cell in campaign["cells"]
        if selected is None or cell["condition"] in selected
    ]
    if _stop_requested:
        campaign["state"] = "interrupted"
    elif all(cell["state"] in TERMINAL_STATES for cell in campaign["cells"]):
        campaign["state"] = (
            "finished"
            if all(cell["state"] == "finished" for cell in campaign["cells"])
            else "partial"
        )
        campaign["execution_finished_at"] = utc_now()
    else:
        campaign["state"] = "prepared"
    campaign["controller"] = {
        "pid": os.getpid(),
        "current_cell": None,
        "active_cells": [],
        "updated_at": utc_now(),
        "active": False,
    }
    write_json(campaign_path, campaign)
    summarize_campaign(campaign_dir)
    failed = [cell for cell in selected_cells if cell["state"] != "finished"]
    print(json.dumps({"state": campaign["state"], "terminal_cells": len(terminal)}, indent=2))
    return 0 if selected_cells and not failed else 2


def _metric(report: dict[str, Any]) -> float | None:
    primary = report.get("primary_metric")
    return numeric(primary.get("value")) if isinstance(primary, dict) else None


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def trajectory_metrics(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed_score: float | None,
    direction: str | None,
    threshold: float | None,
) -> dict[str, Any]:
    started = _timestamp(manifest.get("execution_started_at"))
    wall_time = numeric((manifest.get("budget") or {}).get("wall_time_seconds"))
    if started is None or wall_time is None or wall_time <= 0 or seed_score is None:
        return {"coverage": "unavailable: missing start time, wall budget, or seed score"}
    events: list[tuple[float, float]] = []
    for history_path in run_dir.rglob("history.jsonl"):
        for line in history_path.read_text().splitlines():
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = _metric(report)
            observed = _timestamp(report.get("evaluated_at"))
            if score is None or observed is None or report.get("valid") is not True:
                continue
            elapsed = max(0.0, min(wall_time, (observed - started).total_seconds()))
            events.append((elapsed, score))
    events.sort()

    best = seed_score
    cursor = 0.0
    area = 0.0
    trace = [{"elapsed_seconds": 0.0, "best_score": seed_score}]
    threshold_time = 0.0 if _meets_threshold(seed_score, threshold, direction) else None
    threshold_call = 0 if threshold_time is not None else None
    for call_index, (elapsed, score) in enumerate(events, start=1):
        gain = seed_score - best if direction == "minimize" else best - seed_score
        area += gain * (elapsed - cursor)
        cursor = elapsed
        improved = score < best if direction == "minimize" else score > best
        if improved:
            best = score
            trace.append({"elapsed_seconds": elapsed, "best_score": best})
        if threshold_time is None and _meets_threshold(best, threshold, direction):
            threshold_time = elapsed
            threshold_call = call_index
    gain = seed_score - best if direction == "minimize" else best - seed_score
    area += gain * (wall_time - cursor)
    return {
        "coverage": "controller evaluator histories within the fixed wall budget",
        "best_score_trace": trace,
        "directional_improvement_auc": area,
        "mean_directional_improvement": area / wall_time,
        "threshold": threshold,
        "time_to_threshold_seconds": threshold_time,
        "evaluator_call_to_threshold": threshold_call,
    }


def _meets_threshold(
    score: float, threshold: float | None, direction: str | None
) -> bool:
    if threshold is None:
        return False
    return score <= threshold if direction == "minimize" else score >= threshold


def coordination_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    goal_plus = (manifest.get("execution") or {}).get("goal_plus") or {}
    goal_runs = [run for run in goal_plus.get("runs", []) if isinstance(run, dict)]
    spaces = [
        run.get("search_space")
        for run in goal_plus.get("runs", [])
        if isinstance(run, dict) and (run.get("search_space") or {}).get("exists")
    ]
    if not spaces:
        return {"coverage": "not applicable or no Search Space was created"}
    additive = (
        "plans_total",
        "reviewed_plans",
        "semantic_duplicate_reviews",
        "enforced_rejections",
        "evidence_event_count",
        "evidence_references",
        "cross_lineage_evidence_references",
    )
    result = {key: sum(int(space.get(key) or 0) for space in spaces) for key in additive}
    references = result["evidence_references"]
    result["cross_lineage_evidence_reuse_rate"] = (
        result["cross_lineage_evidence_references"] / references if references else None
    )
    weighted = [
        (
            space.get("mean_cross_lineage_footprint_jaccard"),
            space.get("cross_lineage_footprint_pairs"),
        )
        for space in spaces
    ]
    pairs = sum(int(pair_count or 0) for _, pair_count in weighted)
    result["mean_cross_lineage_footprint_jaccard"] = (
        sum(float(value) * int(count) for value, count in weighted if value is not None and count)
        / pairs
        if pairs
        else None
    )
    result["shared_tool_reuse"] = None
    result["shared_tool_reuse_coverage"] = (
        "not yet attributable: tool provenance is not persisted in Search Evidence"
    )
    result["coverage"] = "persisted Goal Plus Search Space plans and Evidence events"
    result["same_agent_continuation_session_count"] = sum(
        int(run.get("same_agent_continuation_session_count") or 0)
        for run in goal_runs
    )
    reviewer_usage: dict[str, int | float] = {}
    for space in spaces:
        for key, value in (space.get("reviewer_usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                reviewer_usage[key] = reviewer_usage.get(key, 0) + value
    result["coordination_reviewer_usage"] = reviewer_usage
    return result


def collect_cell(campaign: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(cell["run_dir"])
    manifest_path = run_dir / "experiment.json"
    if not manifest_path.is_file():
        return {**cell, "run_dir": portable_path(run_dir)}
    manifest = read_json(manifest_path)
    record = collect_run(
        run_dir,
        campaign_id=campaign["campaign_id"],
        campaign=campaign,
        entry=cell,
        ledger=cell,
    )
    seed_score = numeric((record.get("score") or {}).get("seed_best"))
    direction = (record.get("protocol") or {}).get("direction")
    threshold = (campaign.get("thresholds") or {}).get(cell["benchmark_id"])
    record.update(
        {
            "benchmark_id": cell["benchmark_id"],
            "condition": cell["condition"],
            "coordination_variant": cell.get("coordination_variant"),
            "effective_concurrency": cell["effective_concurrency"],
            "trajectory": trajectory_metrics(
                run_dir,
                manifest,
                seed_score=seed_score,
                direction=direction,
                threshold=threshold,
            ),
            "coordination": coordination_metrics(manifest),
        }
    )
    return record


def condition_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    condition_ids = {
        record.get("condition") or record.get("method")
        for record in records
        if record.get("condition") or record.get("method")
    }
    for condition_id in sorted(condition_ids):
        selected = [
            record
            for record in records
            if (record.get("condition") or record.get("method")) == condition_id
        ]
        finished = [record for record in selected if record.get("status") == "finished"]
        gains = [
            value
            for record in finished
            for value in [numeric((record.get("score") or {}).get("directional_gain"))]
            if value is not None
        ]
        calls = [
            numeric(
                ((record.get("execution") or {}).get("evaluator_calls") or {}).get(
                    "total_claimed"
                )
            )
            for record in selected
        ]
        input_tokens = [
            numeric(((record.get("execution") or {}).get("usage") or {}).get("input_tokens"))
            for record in selected
        ]
        output_tokens = [
            numeric(((record.get("execution") or {}).get("usage") or {}).get("output_tokens"))
            for record in selected
        ]
        known_calls = [value for value in calls if value is not None]
        known_input = [value for value in input_tokens if value is not None]
        known_output = [value for value in output_tokens if value is not None]
        evaluator_coverage = sorted(
            {
                str(((record.get("execution") or {}).get("evaluator_calls") or {}).get("coverage"))
                for record in selected
                if ((record.get("execution") or {}).get("evaluator_calls") or {}).get("coverage")
            }
        )
        token_coverage = sorted(
            {
                str(((record.get("execution") or {}).get("usage") or {}).get("coverage"))
                for record in selected
                if ((record.get("execution") or {}).get("usage") or {}).get("coverage")
            }
        )
        summaries.append(
            {
                "condition": condition_id,
                "cell_count": len(selected),
                "finished_count": len(finished),
                "valid_final_count": sum(
                    (record.get("score") or {}).get("valid") is True for record in finished
                ),
                "mean_directional_gain": sum(gains) / len(gains) if gains else None,
                "total_evaluator_calls": (
                    int(sum(known_calls)) if known_calls else None
                ),
                "evaluator_call_coverage": f"{len(known_calls)}/{len(selected)} cells",
                "evaluator_call_coverage_notes": evaluator_coverage,
                "total_input_tokens": int(sum(known_input)) if known_input else None,
                "input_token_coverage": f"{len(known_input)}/{len(selected)} cells",
                "total_output_tokens": int(sum(known_output)) if known_output else None,
                "output_token_coverage": f"{len(known_output)}/{len(selected)} cells",
                "token_coverage_notes": token_coverage,
            }
        )
    return summaries


def paired_b1_b4(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (
            record.get("benchmark_id"),
            record.get("task_id"),
            record.get("seed"),
            record.get("condition"),
        ): record
        for record in records
    }
    pairs = []
    bases = {
        key[:3]
        for key in indexed
        if key[3] in {"B1", "B4"}
    }
    for base in sorted(bases, key=str):
        b1 = indexed.get((*base, "B1"))
        b4 = indexed.get((*base, "B4"))
        if b1 is None or b4 is None:
            continue
        b1_gain = numeric((b1.get("score") or {}).get("directional_gain"))
        b4_gain = numeric((b4.get("score") or {}).get("directional_gain"))
        if b1_gain is None or b4_gain is None:
            continue
        pairs.append(
            {
                "benchmark_id": base[0],
                "task_id": base[1],
                "seed": base[2],
                "b1_directional_gain": b1_gain,
                "b4_directional_gain": b4_gain,
                "b4_minus_b1_directional_gain": b4_gain - b1_gain,
            }
        )
    effects = [item["b4_minus_b1_directional_gain"] for item in pairs]
    return {
        "paired_count": len(pairs),
        "mean_b4_minus_b1_directional_gain": (
            sum(effects) / len(effects) if effects else None
        ),
        "pairs": pairs,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark campaign {summary['campaign_id']}",
        "",
        f"State: `{summary['state']}`. Cells: {summary['record_count']}.",
        "",
        "## Condition summary",
        "",
        "| Condition | Finished / cells | Valid finals | Mean directional gain | "
        "Evaluator calls | Input / output tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["condition_summaries"]:
        gain = item["mean_directional_gain"]
        calls = item["total_evaluator_calls"]
        input_tokens = item["total_input_tokens"]
        output_tokens = item["total_output_tokens"]
        lines.append(
            f"| {item['condition']} | {item['finished_count']} / {item['cell_count']} | "
            f"{item['valid_final_count']} | {gain if gain is not None else 'n/a'} | "
            f"{calls if calls is not None else 'n/a'} | "
            f"{input_tokens if input_tokens is not None else 'n/a'} / "
            f"{output_tokens if output_tokens is not None else 'n/a'} |"
        )
    incomplete = [
        record for record in summary["records"] if record.get("status") != "finished"
    ]
    if incomplete:
        lines.extend(["", "## Incomplete cells", ""])
        for record in incomplete:
            reason = (
                record.get("incomplete_reason")
                or record.get("error")
                or "no failure reason was recorded"
            )
            reason = " ".join(str(reason).splitlines())
            label = "/".join(
                str(value)
                for value in (
                    record.get("benchmark_id"),
                    record.get("method"),
                    f"seed-{record.get('seed')}",
                )
                if value is not None
            )
            lines.append(f"- `{label}`: {reason}")
    paired = summary["b1_vs_b4"]
    paired_gain = paired["mean_b4_minus_b1_directional_gain"]
    lines.extend(
        [
            "",
            "## B1 vs B4",
            "",
            f"Paired cells: {paired['paired_count']}; mean B4 - B1 directional gain: "
            f"{paired_gain if paired_gain is not None else 'n/a'}.",
            "",
            "Raw native metrics remain in each cell. A positive directional gain always "
            "means improvement; it does not normalize across tasks.",
            "",
            "Token and evaluator-call totals are the fixed-total-compute view. Per-cell "
            "wall time and effective concurrency are the wall-clock view.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_campaign(campaign_dir: Path) -> dict[str, Any]:
    campaign = read_json(campaign_dir / "campaign.json")
    records = [collect_cell(campaign, cell) for cell in campaign["cells"]]
    summary = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "state": campaign["state"],
        "updated_at": utc_now(),
        "record_count": len(records),
        "budget": campaign["budget"],
        "wall_time_seconds": campaign["budget"].get("wall_time_seconds"),
        "live_search_concurrency": campaign["budget"].get(
            "live_search_concurrency",
            campaign["budget"].get("requested_live_concurrency"),
        ),
        "cell_concurrency": campaign["budget"].get("cell_concurrency", 1),
        "attempts": campaign["budget"].get(
            "attempts", len(campaign.get("seeds") or [])
        ),
        "condition_summaries": condition_summaries(records),
        "b1_vs_b4": paired_b1_b4(records),
        "records": records,
        "coverage": {
            "final_score": "controller-owned final evaluator",
            "trajectory": "controller evaluator histories when available",
            "coordination": "Goal Plus Search Space plans and Evidence when available",
            "shared_tool_reuse": "not yet attributable in runtime evidence",
        },
    }
    write_json(campaign_dir / "campaign-summary.json", summary)
    (campaign_dir / "campaign-summary.md").write_text(render_markdown(summary))
    return summary


def status_campaign(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign.expanduser().absolute()
    campaign = read_json(campaign_dir / "campaign.json")
    for cell in campaign["cells"]:
        manifest_path = Path(cell["run_dir"]) / "experiment.json"
        if manifest_path.is_file():
            cell["run_status"] = read_json(manifest_path).get("status")
    counts: dict[str, int] = {}
    for cell in campaign["cells"]:
        counts[cell["state"]] = counts.get(cell["state"], 0) + 1
    payload = {
        "campaign_id": campaign["campaign_id"],
        "state": campaign["state"],
        "counts": counts,
        "controller": campaign.get("controller"),
        "cells": campaign["cells"],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{payload['campaign_id']}: {payload['state']} {counts}")
        for cell in payload["cells"]:
            print(
                f"- {cell['cell_id']}: {cell['state']} "
                f"(run={cell.get('run_status', 'not-started')})"
            )
    return 0


def summarize_command(args: argparse.Namespace) -> int:
    summary = summarize_campaign(args.campaign.expanduser().absolute())
    print(json.dumps(summary, indent=2))
    return 0


def list_command(_args: argparse.Namespace) -> int:
    payload = {
        "adapters": adapter_modules(),
        "conditions": {
            condition_id: condition.as_manifest()
            for condition_id, condition in CONDITIONS.items()
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--campaign-dir", type=Path)
    prepare_parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(adapter_modules()), required=True
    )
    prepare_parser.add_argument(
        "--task-id",
        help="adapter-specific task selector; requires exactly one benchmark",
    )
    prepare_parser.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS), default=[]
    )
    prepare_parser.add_argument(
        "--methods", nargs="+", choices=standalone.METHODS, default=[]
    )
    prepare_parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    standalone.add_runtime_prepare_arguments(
        prepare_parser, reasoning_choices=("low", "medium", "high", "xhigh")
    )
    prepare_parser.add_argument("--threshold", action="append", default=[])
    prepare_parser.add_argument("--pi-api-base-env", default="OPENAI_BASE_URL")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign", type=Path, required=True)
    run_parser.add_argument("--model", default=standalone.DEFAULT_MODEL)
    run_parser.add_argument("--codex-bin", default="codex")
    run_parser.add_argument("--api-base")
    run_parser.add_argument("--pi-provider-id", default=standalone.PI_PROVIDER_ID)
    run_parser.add_argument(
        "--pi-api", choices=standalone.PI_APIS, default="openai-responses"
    )
    run_parser.add_argument(
        "--pi-api-key-env", default=standalone.PI_API_KEY_ENV
    )
    run_parser.add_argument("--pi-api-base-env", default="OPENAI_BASE_URL")
    run_parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS))
    run_parser.add_argument("--fail-fast", action="store_true")

    run_cell_parser = subparsers.add_parser("run-cell")
    run_cell_parser.add_argument("--run-dir", type=Path, required=True)
    run_cell_parser.add_argument("--model", required=True)
    run_cell_parser.add_argument("--codex-bin", default="codex")
    run_cell_parser.add_argument(
        "--pi-provider-id", default=standalone.PI_PROVIDER_ID
    )
    run_cell_parser.add_argument(
        "--pi-api", choices=standalone.PI_APIS, default="openai-responses"
    )
    run_cell_parser.add_argument(
        "--pi-api-key-env", default=standalone.PI_API_KEY_ENV
    )

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--campaign", type=Path, required=True)
    status_parser.add_argument("--json", action="store_true")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--campaign", type=Path, required=True)
    return parser


def main() -> int:
    configure_temp_environment()
    args = build_parser().parse_args()
    if args.command == "list":
        return list_command(args)
    if args.command == "prepare":
        return prepare_campaign(args)
    if args.command == "run":
        return run_campaign(args)
    if args.command == "run-cell":
        return run_cell(args)
    if args.command == "status":
        return status_campaign(args)
    return summarize_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
