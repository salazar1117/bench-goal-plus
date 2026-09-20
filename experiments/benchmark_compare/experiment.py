#!/usr/bin/env python3
"""Run standalone benchmarks with Plain or Goal Plus agent methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_artifacts import read_json as load_json  # noqa: E402
from bench_artifacts import utc_now, write_json, write_json_atomic  # noqa: E402
from bench_goal_plus.upstreams import (  # noqa: E402
    external_goal_plus_source,
    require_prepared_goal_plus_source,
    upstream_checkout_path,
    upstream_source_path,
)
from bench_goal_plus.goal_plus_command import (  # noqa: E402
    goal_plus_command_config,
    goal_plus_entrypoint,
)
from bench_runtime_paths import (  # noqa: E402
    configure_temp_environment,
)
from adapters.registry import (  # noqa: E402
    adapter_modules,
    load_adapter,
    load_adapter_module,
)
from experiments.backends import skydiscover as sky_backend  # noqa: E402
from experiments.benchmark_compare.conditions import (  # noqa: E402
    CONDITIONS,
    VARIANT_LIMITATIONS,
    resolve_condition,
)
from experiments.benchmark_compare.pi_worker_launcher import (  # noqa: E402
    LEGACY_GOAL_PLUS_WORKER_LAUNCHER_ENV,
    REAL_PI_BIN_ENV,
    SANDBOX_POLICY_ENV,
)
from experiments.benchmark_compare.host_search_guard import prepend_to_path  # noqa: E402
from experiments.openevolve_compare.experiment import (  # noqa: E402
    PUBLIC_GATE_SELECTION_RULE,
    DEFAULT_REASONING_EFFORT,
    PI_APIS,
    PI_API_KEY_ENV,
    PI_PROVIDER_ID,
    append_unique_lines,
    bind_goal_plus_environment,
    close_candidate_sessions,
    codex_goal_plus_mcp_args,
    codex_provider_args,
    collect_evidence_annotator_usage,
    collect_goal_plus_state,
    commit_workspace,
    configure_evidence_annotator_environment,
    configure_isolated_codex_home,
    copy_goal_plus_assets,
    copy_goal_plus_pi_assets,
    finalize_goal_plus_search,
    goal_plus_incomplete_reason,
    parse_codex_events,
    parse_pi_events,
    primary_score,
    render_goal,
    render_plain_prompt,
    run_controlled,
    run_controlled_many,
    sha256_file,
    sha256_text,
    write_pi_models_config,
)


DEFAULT_ENV_MANIFEST = ROOT / "environment/upstreams.json"
DEFAULT_CHECKOUT_ROOT = ROOT / "third_party"
DEFAULT_VENV = ROOT / ".bench-env/venv"
DEFAULT_RUNS = ROOT / "runs/benchmark-compare"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_WALL_TIME_SECONDS = 300
DEFAULT_CONCURRENCY = 2
DEFAULT_SOFT_CLOSEOUT_SECONDS = 60
DEFAULT_HARD_KILL_GRACE_SECONDS = 30
DEFAULT_WORKER_RUNTIME_SECONDS = 120
CONTROLLER_ONLY_CLOSEOUT_ENV = "BENCH_GOAL_PLUS_CONTROLLER_ONLY_CLOSEOUT"
GOAL_PLUS_EXTERNAL_EVIDENCE_DIR_ENV = "GOAL_PLUS_EXTERNAL_EVIDENCE_DIR"
GOAL_PLUS_GLOBAL_EVIDENCE_MODE_ENV = "GOAL_PLUS_GLOBAL_EVIDENCE_MODE"
GOAL_PLUS_SUPPLEMENTAL_EVALUATION_ENABLED_ENV = (
    "GOAL_PLUS_SUPPLEMENTAL_EVALUATION_ENABLED"
)
GOAL_PLUS_SUPPLEMENTAL_EVALUATION_REQUIRED_ENV = (
    "GOAL_PLUS_SUPPLEMENTAL_EVALUATION_REQUIRED"
)
METHODS = (
    "plain-codex",
    "plain-pi",
    "goal-plus-codex",
    "goal-plus-pi",
    *sky_backend.METHODS,
)
CONTROLLER_ONLY_METHODS = frozenset(
    {"plain-codex", "plain-pi", "goal-plus-codex", "goal-plus-pi"}
)
SKYDISCOVER_EDIT_PROTOCOL = """

## SkyDiscover candidate response contract

Make one small, localized improvement per iteration. Return only one or a few
complete SEARCH/REPLACE blocks in the exact format requested by SkyDiscover.
Do not include prose or Markdown code fences. Do not rewrite the entire artifact:
each SEARCH block must be a short exact excerpt from the current program, and
each block must include its closing `>>>>>>> REPLACE` marker.
""".rstrip()
BENCHMARK_ADAPTERS = adapter_modules()
PI_WORKER_LAUNCHER = (
    ROOT / "experiments" / "benchmark_compare" / "pi_worker_launcher.py"
)
PI_TOOL_PROXY = (
    ROOT / "experiments" / "benchmark_compare" / "bin" / "goal-plus-pi-tool"
)
PI_SHIM = ROOT / "experiments" / "benchmark_compare" / "pi-shim" / "pi"
PI_MAIN_TOOL_SHIM = (
    ROOT
    / "experiments"
    / "benchmark_compare"
    / "main-bin"
    / "goal-plus-pi-tool"
)


@dataclass(frozen=True)
class PrepareConfig:
    benchmark: str
    method: str
    task_id: str | None = None
    shared_dir: bool = False
    shared_cache: bool = False
    adapter_module: str | None = None
    condition: str | None = None
    coordination_variant: str | None = None
    model: str = DEFAULT_MODEL
    pi_provider_id: str = PI_PROVIDER_ID
    pi_api: str = "openai-responses"
    pi_api_key_env: str = PI_API_KEY_ENV
    wall_time_seconds: int = DEFAULT_WALL_TIME_SECONDS
    concurrency: int = DEFAULT_CONCURRENCY
    soft_closeout_seconds: int = DEFAULT_SOFT_CLOSEOUT_SECONDS
    hard_kill_grace_seconds: int = DEFAULT_HARD_KILL_GRACE_SECONDS
    worker_runtime_seconds: int = DEFAULT_WORKER_RUNTIME_SECONDS
    worker_min_runtime_seconds: int | None = None
    iterations_ceiling: int = 1
    seed: int = 1
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    llm_max_tokens: int = 8192
    run_dir: Path | None = None
    environment_manifest: Path = DEFAULT_ENV_MANIFEST
    checkout_root: Path = DEFAULT_CHECKOUT_ROOT
    venv: Path = DEFAULT_VENV

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(command="prepare", **vars(self))


@dataclass(frozen=True)
class RunConfig:
    run_dir: Path
    model: str = DEFAULT_MODEL
    codex_bin: str = "codex"
    pi_bin: str = "pi"
    api_base: str | None = None
    pi_provider_id: str = PI_PROVIDER_ID
    pi_api: str = "openai-responses"
    pi_api_key_env: str = PI_API_KEY_ENV

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(command="run", **vars(self))


def add_runtime_prepare_arguments(
    parser: argparse.ArgumentParser,
    *,
    reasoning_choices: tuple[str, ...] | None = None,
) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pi-provider-id", default=PI_PROVIDER_ID)
    parser.add_argument("--pi-api", choices=PI_APIS, default="openai-responses")
    parser.add_argument("--pi-api-key-env", default=PI_API_KEY_ENV)
    parser.add_argument("--shared-dir", action="store_true")
    parser.add_argument("--shared-cache", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=reasoning_choices,
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--wall-time-seconds", type=int, default=DEFAULT_WALL_TIME_SECONDS
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--soft-closeout-seconds", type=int, default=DEFAULT_SOFT_CLOSEOUT_SECONDS
    )
    parser.add_argument(
        "--hard-kill-grace-seconds",
        type=int,
        default=DEFAULT_HARD_KILL_GRACE_SECONDS,
    )
    parser.add_argument(
        "--worker-runtime-seconds", type=int, default=DEFAULT_WORKER_RUNTIME_SECONDS
    )
    parser.add_argument("--worker-min-runtime-seconds", type=int)
    parser.add_argument(
        "--environment-manifest", type=Path, default=DEFAULT_ENV_MANIFEST
    )
    parser.add_argument("--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)


def _normalize_early_stop_contract(
    value: Any,
    *,
    process_metric: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "enabled",
        "metric_name",
        "target_score",
        "require_process_passed",
        "require_git_artifact_clean",
        "require_settled_git_head",
        "reject_touched_denied_files",
        "reject_changed_outside_allowed",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Goal Plus early-stop contract has an invalid schema")
    if value["enabled"] is not True:
        raise ValueError("Goal Plus early-stop contract must be enabled when present")
    if value["metric_name"] != process_metric:
        raise ValueError("Goal Plus early-stop metric must match the process metric")
    if type(value["target_score"]) not in {int, float} or not math.isfinite(
        float(value["target_score"])
    ):
        raise ValueError("Goal Plus early-stop target score must be finite")
    for field in required - {"metric_name", "target_score"}:
        if value[field] is not True:
            raise ValueError(f"Goal Plus early-stop contract requires {field}=true")
    normalized = dict(value)
    normalized["target_score"] = float(value["target_score"])
    return normalized


def _normalize_posthoc_selection_contract(
    value: Any,
    *,
    primary_metric: str,
    direction: str,
    controller_only: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "enabled",
        "metric_name",
        "metric_direction",
        "candidate_scope",
        "tie_break",
        "timing",
        "visible_to_workers",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Goal Plus posthoc-selection contract has an invalid schema")
    expected = {
        "enabled": True,
        "metric_name": primary_metric,
        "metric_direction": direction,
        "candidate_scope": "all_publicly_compliant_iterations",
        "tie_break": "lowest_candidate_id_then_latest_iteration",
        "timing": "after_agent_exit_and_controller_closeout",
        "visible_to_workers": False,
    }
    if value != expected:
        raise ValueError(
            "Goal Plus posthoc-selection contract does not match the supported policy"
        )
    if not controller_only:
        raise ValueError(
            "Goal Plus posthoc selection requires controller-only official evaluation"
        )
    return dict(value)


def configure_adapter(
    benchmark_id: str,
    *,
    task_id: str | None = None,
    module_name: str | None = None,
) -> None:
    loaded = (
        load_adapter_module(benchmark_id, module_name)
        if module_name is not None
        else load_adapter(benchmark_id)
    )
    loaded.configure_task(task_id)
    module = loaded.module
    global ADAPTER_CONTRACT
    global ARTIFACT_NAME, BENCHMARK_NAME, CASE_SET_DESCRIPTION
    global CODEX_SANDBOX, DIRECTION, GOAL_PLUS_MCP_ENV_VARS
    global CONTROLLER_ONLY_OFFICIAL_EVALUATION, EVALUATION_MODE
    global PUBLIC_FEEDBACK_COMMAND
    global GOAL_PLUS_PROCESS_METRIC, REQUIRES_PROTECTED_PI_WORKERS
    global GOAL_PLUS_EARLY_STOP_CONTRACT, GOAL_PLUS_POSTHOC_SELECTION_CONTRACT
    global PI_WORKER_SANDBOX
    global PRIMARY_METRIC, TASK_ID, UPSTREAM_KEY
    global LOCAL_SOURCE_RELATIVE, UPSTREAM_SUBDIR
    global OFFICIAL_BENCHMARK_COMPARABLE
    global VERIFIER_TIMEOUT_SECONDS
    global PROCESS_VERIFIER_TIMEOUT_SECONDS
    global evaluate_workspace, git_commit, materialize_workspace
    global validate_runtime_assets
    ARTIFACT_NAME = module.ARTIFACT_NAME
    BENCHMARK_NAME = module.BENCHMARK_NAME
    CASE_SET_DESCRIPTION = module.CASE_SET_DESCRIPTION
    CODEX_SANDBOX = module.CODEX_SANDBOX
    GOAL_PLUS_MCP_ENV_VARS = tuple(getattr(module, "GOAL_PLUS_MCP_ENV_VARS", ()))
    PI_WORKER_SANDBOX = getattr(module, "PI_WORKER_SANDBOX", None)
    DIRECTION = module.DIRECTION
    CONTROLLER_ONLY_OFFICIAL_EVALUATION = getattr(
        module, "CONTROLLER_ONLY_OFFICIAL_EVALUATION", False
    )
    EVALUATION_MODE = getattr(module, "EVALUATION_MODE", "visible")
    PUBLIC_FEEDBACK_COMMAND = getattr(
        module, "PUBLIC_FEEDBACK_COMMAND", "python3 evaluate.py"
    )
    REQUIRES_PROTECTED_PI_WORKERS = getattr(
        module, "REQUIRES_PROTECTED_PI_WORKERS", False
    )
    GOAL_PLUS_PROCESS_METRIC = getattr(
        module, "GOAL_PLUS_PROCESS_METRIC", module.PRIMARY_METRIC
    )
    GOAL_PLUS_EARLY_STOP_CONTRACT = _normalize_early_stop_contract(
        getattr(module, "GOAL_PLUS_EARLY_STOP_CONTRACT", None),
        process_metric=GOAL_PLUS_PROCESS_METRIC,
    )
    GOAL_PLUS_POSTHOC_SELECTION_CONTRACT = _normalize_posthoc_selection_contract(
        getattr(module, "GOAL_PLUS_POSTHOC_SELECTION_CONTRACT", None),
        primary_metric=module.PRIMARY_METRIC,
        direction=module.DIRECTION,
        controller_only=CONTROLLER_ONLY_OFFICIAL_EVALUATION,
    )
    PRIMARY_METRIC = module.PRIMARY_METRIC
    TASK_ID = module.TASK_ID
    UPSTREAM_KEY = module.UPSTREAM_KEY
    LOCAL_SOURCE_RELATIVE = getattr(module, "LOCAL_SOURCE_RELATIVE", None)
    UPSTREAM_SUBDIR = getattr(module, "UPSTREAM_SUBDIR", None)
    OFFICIAL_BENCHMARK_COMPARABLE = getattr(
        module, "OFFICIAL_BENCHMARK_COMPARABLE", True
    )
    VERIFIER_TIMEOUT_SECONDS = module.VERIFIER_TIMEOUT_SECONDS
    PROCESS_VERIFIER_TIMEOUT_SECONDS = getattr(
        module, "PROCESS_VERIFIER_TIMEOUT_SECONDS", VERIFIER_TIMEOUT_SECONDS
    )
    evaluate_workspace = module.evaluate_workspace
    git_commit = module.git_commit
    materialize_workspace = module.materialize_workspace
    validate_runtime_assets = getattr(
        module, "validate_runtime_assets", lambda _source_root: None
    )
    ADAPTER_CONTRACT = loaded.manifest_contract()


configure_adapter("heurigym")


def validate_controller_only_method(method: str) -> None:
    if (
        (
            CONTROLLER_ONLY_OFFICIAL_EVALUATION
            or REQUIRES_PROTECTED_PI_WORKERS
        )
        and method not in CONTROLLER_ONLY_METHODS
    ):
        supported = ", ".join(sorted(CONTROLLER_ONLY_METHODS))
        raise ValueError(
            "protected benchmark evaluation rejects method "
            f"{method!r}; supported methods: {supported}"
        )


def _pi_worker_sandbox_policy(api_key_env: str) -> dict[str, Any] | None:
    if PI_WORKER_SANDBOX is None:
        return None
    policy = json.loads(json.dumps(PI_WORKER_SANDBOX))
    pass_env = list(policy.get("pass_env", []))
    for name in (api_key_env, "OPENAI_BASE_URL", "OPENAI_API_BASE_URL"):
        if name not in pass_env:
            pass_env.append(name)
    policy["pass_env"] = pass_env
    return policy


def _resolve_real_pi_binary(
    pi_bin: str,
    environment: dict[str, str],
) -> Path:
    configured = Path(pi_bin)
    if configured.is_absolute():
        candidate = configured
    else:
        found = shutil.which(pi_bin, path=environment.get("PATH"))
        if found is None:
            raise FileNotFoundError(f"Pi executable not found: {pi_bin}")
        candidate = Path(found)
    candidate = candidate.expanduser().absolute()
    resolved = candidate.resolve(strict=True)
    if candidate == PI_SHIM.absolute() or resolved == PI_SHIM.resolve():
        raise RuntimeError("bench Pi shim cannot be configured as the real Pi binary")
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"Pi executable is not executable: {resolved}")
    return candidate


def _configure_pi_worker_sandbox_environment(
    manifest: dict[str, Any],
    environment: dict[str, str],
    api_key_env: str,
    pi_bin: str,
) -> None:
    if manifest["method"] != "goal-plus-pi" or PI_WORKER_SANDBOX is None:
        return
    worker_sandbox = (manifest.get("goal_plus_config") or {}).get(
        "worker_sandbox"
    )
    expected_sandbox = _pi_worker_sandbox_policy(api_key_env)
    if worker_sandbox is None:
        raise RuntimeError(
            "this adapter requires a Pi worker sandbox; re-prepare the run "
            "with a sandbox-capable Goal Plus checkout"
        )
    if worker_sandbox != expected_sandbox:
        raise RuntimeError(
            "prepared Pi worker sandbox policy does not match the adapter contract"
        )
    if shutil.which("bwrap", path=environment.get("PATH")) is None:
        raise RuntimeError("Goal Plus Pi worker sandbox requires bwrap")
    for executable in (
        PI_WORKER_LAUNCHER,
        PI_TOOL_PROXY,
        PI_SHIM,
        PI_MAIN_TOOL_SHIM,
    ):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(
                f"benchmark Pi worker launcher asset is unavailable: {executable}"
            )
    real_pi = _resolve_real_pi_binary(
        environment.get(REAL_PI_BIN_ENV, pi_bin), environment
    )
    environment.pop(LEGACY_GOAL_PLUS_WORKER_LAUNCHER_ENV, None)
    environment[REAL_PI_BIN_ENV] = str(real_pi)
    environment[SANDBOX_POLICY_ENV] = json.dumps(
        worker_sandbox, separators=(",", ":")
    )
    environment["PATH"] = os.pathsep.join(
        (
            str(PI_SHIM.parent),
            str(PI_MAIN_TOOL_SHIM.parent),
            environment.get("PATH", ""),
        )
    )
    prepend_to_path(environment)
    manifest["pi_worker_sandbox"] = {
        **worker_sandbox,
        "owner": "bench-goal-plus",
        "launch_interception": "bench-owned-pi-path-shim",
        "goal_plus_source_changes_required": False,
        "environment_values_persisted": False,
    }


def default_run_dir(method: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS / f"{timestamp}-{UPSTREAM_KEY}-{TASK_ID}-{method}"


def runtime_bin(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def checkout_branch(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "symbolic-ref", "--short", "-q", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def checkout_dirty(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(completed.stdout.strip())


def prepare(args: argparse.Namespace) -> int:
    configure_adapter(
        args.benchmark,
        task_id=getattr(args, "task_id", None),
        module_name=getattr(args, "adapter_module", None),
    )
    validate_controller_only_method(args.method)
    is_sky = sky_backend.is_method(args.method)
    condition = resolve_condition(
        method=args.method,
        concurrency=args.concurrency,
        condition_id=args.condition,
        coordination_variant=args.coordination_variant,
    )
    if args.wall_time_seconds <= args.soft_closeout_seconds:
        raise ValueError("wall time must exceed the closeout reserve")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if getattr(args, "shared_dir", False) and args.method not in {
        "goal-plus-codex",
        "goal-plus-pi",
    }:
        raise ValueError("--shared-dir requires a Goal Plus method")
    if getattr(args, "shared_cache", False) and args.method not in {
        "goal-plus-codex",
        "goal-plus-pi",
    }:
        raise ValueError("--shared-cache requires a Goal Plus method")
    if args.iterations_ceiling < 1:
        raise ValueError("iterations ceiling must be positive")
    if args.llm_max_tokens < 1:
        raise ValueError("LLM max tokens must be positive")
    exploration_seconds = args.wall_time_seconds - args.soft_closeout_seconds
    if not 1 <= args.worker_runtime_seconds <= exploration_seconds:
        raise ValueError("worker runtime must fit inside the exploration budget")
    if args.worker_min_runtime_seconds is not None and not (
        1 <= args.worker_min_runtime_seconds <= args.worker_runtime_seconds
    ):
        raise ValueError("worker minimum runtime must fit inside the worker budget")
    environment = load_json(args.environment_manifest)
    upstreams = environment["upstreams"]
    checkout_root = args.checkout_root.expanduser().absolute()
    goal_plus_entry = upstreams["goal_plus"]
    goal_plus_checkout = upstream_checkout_path(
        checkout_root,
        goal_plus_entry,
        upstream_key="goal_plus",
    )
    goal_plus_root = upstream_source_path(
        checkout_root,
        goal_plus_entry,
        upstream_key="goal_plus",
    )
    managed_checkouts = [("goal_plus", goal_plus_checkout)]
    skydiscover_root = None
    if is_sky:
        skydiscover_root = checkout_root / upstreams["skydiscover"]["checkout_dir"]
        managed_checkouts.append(("skydiscover", skydiscover_root))
    if LOCAL_SOURCE_RELATIVE is None:
        benchmark_entry = upstreams[UPSTREAM_KEY]
        benchmark_checkout = upstream_checkout_path(
            checkout_root,
            benchmark_entry,
            upstream_key=UPSTREAM_KEY,
        )
        benchmark_root = upstream_source_path(
            checkout_root,
            benchmark_entry,
            upstream_key=UPSTREAM_KEY,
        )
        managed_checkouts.insert(0, (UPSTREAM_KEY, benchmark_checkout))
        if UPSTREAM_SUBDIR is not None:
            subdir = Path(UPSTREAM_SUBDIR)
            if subdir.is_absolute() or ".." in subdir.parts:
                raise ValueError(f"unsafe adapter upstream subdirectory: {subdir}")
            benchmark_root = (benchmark_root / subdir).resolve()
            try:
                benchmark_root.relative_to(benchmark_checkout.resolve())
            except ValueError as error:
                raise ValueError(
                    f"adapter upstream subdirectory escapes checkout: {subdir}"
                ) from error
            if not benchmark_root.is_dir():
                raise FileNotFoundError(
                    f"adapter upstream subdirectory is missing: {benchmark_root}"
                )
        source_kind = "managed_upstream"
    else:
        benchmark_root = (ROOT / LOCAL_SOURCE_RELATIVE).resolve()
        benchmark_checkout = benchmark_root
        try:
            benchmark_root.relative_to(ROOT)
        except ValueError as error:
            raise ValueError(
                f"local benchmark source escapes repository root: {benchmark_root}"
            ) from error
        if not benchmark_root.is_dir():
            raise FileNotFoundError(
                f"local benchmark source is missing: {benchmark_root}"
            )
        source_kind = "local_example"

    for name, path in managed_checkouts:
        if name == "goal_plus" and external_goal_plus_source():
            continue
        if not (path / ".git").exists():
            raise FileNotFoundError(
                f"managed {name} checkout is missing: {path}; run repro_env.py bootstrap"
            )
        expected = upstreams[name]["tracking_branch"]
        actual = checkout_branch(path)
        if actual != expected:
            raise RuntimeError(f"{name} branch mismatch: expected {expected}, got {actual}")
        if name != "goal_plus" and checkout_dirty(path):
            raise RuntimeError(f"managed {name} checkout has local changes: {path}")

    validate_runtime_assets(benchmark_root)

    run_dir = (args.run_dir or default_run_dir(args.method)).expanduser().absolute()
    run_dir.mkdir(parents=True, exist_ok=False)
    workspaces: list[Path] = []
    workspace_commits: list[str] = []
    search_backend: dict[str, Any] | None = None

    if args.method in {"plain-codex", "plain-pi"}:
        for lane_index in range(args.concurrency):
            lane = f"lane-{lane_index:02d}"
            workspace = run_dir / "workspaces" / lane
            materialized = materialize_workspace(
                benchmark_root,
                workspace,
            )
            workspaces.append(workspace)
            workspace_commits.append(materialized["workspace_commit"])
        task_text = (workspaces[0] / "TASK.md").read_text()
        common_prompt = render_plain_prompt(
            task_text,
            args.wall_time_seconds,
            args.soft_closeout_seconds,
            public_feedback_command=PUBLIC_FEEDBACK_COMMAND,
            controller_only_official_evaluation=(
                CONTROLLER_ONLY_OFFICIAL_EVALUATION
            ),
        )
        prompt_contract = {
            "mode": f"{args.method.replace('-', '_')}_common_prompt",
            "common_prompt_sha256": sha256_text(common_prompt),
            "transform": "identity",
        }
        workspace_value = None
        goal_plus_config = None
    elif args.method in {"goal-plus-codex", "goal-plus-pi"}:
        workspace = run_dir / "workspace"
        materialized = materialize_workspace(
            benchmark_root,
            workspace,
        )
        if args.method == "goal-plus-codex":
            copy_goal_plus_assets(goal_plus_root, workspace)
            append_unique_lines(workspace / ".gitignore", [".gp/", ".codex-log/"])
            agent_harness = "codex"
            worker_model = args.model
        else:
            copy_goal_plus_pi_assets(goal_plus_root, workspace)
            append_unique_lines(workspace / ".gitignore", [".gp/", ".pi-log/"])
            agent_harness = "pi"
            worker_model = f"{args.pi_provider_id}/{args.model}"
        task_text = (workspace / "TASK.md").read_text()
        goal_prompt = render_goal(
            task_text=task_text,
            public_feedback_command=PUBLIC_FEEDBACK_COMMAND,
            artifact_name=ARTIFACT_NAME,
            artifact_is_directory=(workspace / ARTIFACT_NAME).is_dir(),
            metric_name=GOAL_PLUS_PROCESS_METRIC,
            metric_direction=DIRECTION,
            wall_seconds=args.wall_time_seconds,
            closeout_seconds=args.soft_closeout_seconds,
            concurrency=args.concurrency,
            agent_harness=agent_harness,
            worker_model=worker_model,
            reasoning_effort=args.reasoning_effort,
            worker_runtime_seconds=args.worker_runtime_seconds,
            worker_min_runtime_seconds=args.worker_min_runtime_seconds,
            verifier_timeout_seconds=VERIFIER_TIMEOUT_SECONDS,
            process_verifier_timeout_seconds=PROCESS_VERIFIER_TIMEOUT_SECONDS,
            coordination_condition=condition.condition_id if condition else None,
            search_space_mode=condition.search_space_mode if condition else None,
            shared_dir_enabled=getattr(args, "shared_dir", False),
            shared_cache_enabled=getattr(args, "shared_cache", False),
            controller_only_official_evaluation=(
                CONTROLLER_ONLY_OFFICIAL_EVALUATION
            ),
            evaluation_mode=EVALUATION_MODE,
            early_stop_contract=GOAL_PLUS_EARLY_STOP_CONTRACT,
        )
        (workspace / "GOAL.md").write_text(goal_prompt)
        workspaces.append(workspace)
        workspace_commits.append(
            commit_workspace(workspace, "install managed Goal Plus host assets")
        )
        common_prompt = render_plain_prompt(
            task_text,
            args.wall_time_seconds,
            args.soft_closeout_seconds,
            public_feedback_command=PUBLIC_FEEDBACK_COMMAND,
            controller_only_official_evaluation=(
                CONTROLLER_ONLY_OFFICIAL_EVALUATION
            ),
        )
        prompt_contract = {
            "mode": "natural_goal_plus_entry",
            "common_prompt_sha256": sha256_text(common_prompt),
            "transform": (
                f"{goal_plus_entrypoint(agent_harness)} typed config prefix plus aligned "
                "SearchSpec-only constraints"
            ),
            "goal_prompt_sha256": sha256_text(goal_prompt),
        }
        workspace_value = str(workspace)
        goal_plus_config = {
            "entrypoint": goal_plus_entrypoint(agent_harness),
            "command_config": goal_plus_command_config(
                max_parallel=args.concurrency,
                strategy="agent_guided",
                worker_model=worker_model,
                annotator_model=worker_model,
                workspace_provider="git_worktree",
                promotion_mode=(
                    "artifact_only"
                    if EVALUATION_MODE == "blind"
                    else "apply"
                ),
            ),
            "agent_harness": agent_harness,
            "runtime_provider": "direct",
            "worker_model": worker_model,
            "metric_name": GOAL_PLUS_PROCESS_METRIC,
            "metric_direction": DIRECTION,
            "controller_only_official_evaluation": (
                CONTROLLER_ONLY_OFFICIAL_EVALUATION
            ),
            "early_stop": GOAL_PLUS_EARLY_STOP_CONTRACT,
            "posthoc_selection": GOAL_PLUS_POSTHOC_SELECTION_CONTRACT,
            "artifact_name": ARTIFACT_NAME,
            "artifact_is_directory": (workspace / ARTIFACT_NAME).is_dir(),
            "shared_dir_enabled": getattr(args, "shared_dir", False),
            "shared_cache_enabled": getattr(args, "shared_cache", False),
            "worker_sandbox": (
                _pi_worker_sandbox_policy(args.pi_api_key_env)
                if agent_harness == "pi"
                else None
            ),
            "state_at_t0": "absent; natural prompt creates all Goal Plus state inside T",
            "condition": condition.as_manifest() if condition else None,
            "coordination_reviewer": (
                {
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "usage_accounting": "persisted per Search Space plan",
                }
                if condition and condition.search_space_mode
                else None
            ),
        }
    elif is_sky:
        workspace = run_dir / "workspace"
        materialized = materialize_workspace(
            benchmark_root,
            workspace,
        )
        workspaces.append(workspace)
        workspace_commits.append(materialized["workspace_commit"])
        task_text = (workspace / "TASK.md").read_text()
        sky_task_prompt = task_text.rstrip() + "\n\n" + SKYDISCOVER_EDIT_PROTOCOL
        algorithm = sky_backend.algorithm_for_method(args.method)
        sky_config_path = run_dir / "skydiscover-config.yaml"
        sky_backend.write_config(
            sky_config_path,
            algorithm=algorithm,
            task_prompt=sky_task_prompt,
            file_suffix=Path(ARTIFACT_NAME).suffix,
            evaluator_timeout_seconds=VERIFIER_TIMEOUT_SECONDS,
            concurrency=args.concurrency,
            iterations_ceiling=args.iterations_ceiling,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.llm_max_tokens,
        )
        prompt_contract = {
            "mode": "skydiscover_native_context",
            "task_prompt_sha256": sha256_text(sky_task_prompt),
            "backend_control_prompt": (
                "SkyDiscover adds native search history and mutation instructions "
                "to the fixed benchmark task prompt"
            ),
            "backend_config_sha256": sha256_file(sky_config_path),
        }
        workspace_value = str(workspace)
        goal_plus_config = None
        search_backend = {
            "family": "skydiscover",
            "algorithm": algorithm,
            "config": str(sky_config_path),
            "config_sha256": sha256_file(sky_config_path),
            "llm_max_tokens": args.llm_max_tokens,
            "native_seed_evaluation": (
                "inside timed runtime; functional-smoke limitation"
            ),
            "native_best_test_evaluation": (
                "inside timed runtime; functional-smoke limitation"
            ),
            "determinism_coverage": (
                "requested seed is persisted, but the selected SkyDiscover "
                "algorithm does not consistently seed every native random source"
            ),
        }
    else:
        raise ValueError(f"unsupported method: {args.method}")

    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "prepared_at": utc_now(),
        "method": args.method,
        "condition": condition.as_manifest() if condition else None,
        "benchmark_adapter": args.benchmark,
        "benchmark_adapter_module": getattr(args, "adapter_module", None),
        "benchmark_task_selector": getattr(args, "task_id", None),
        "benchmark_adapter_contract": ADAPTER_CONTRACT,
        "benchmark_name": BENCHMARK_NAME,
        "task_id": TASK_ID,
        "model": args.model,
        "pi_provider": (
            {
                "id": args.pi_provider_id,
                "api": args.pi_api,
                "api_key_env": args.pi_api_key_env,
            }
            if args.method in {"plain-pi", "goal-plus-pi"}
            else None
        ),
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "budget": {
            "wall_time_seconds": args.wall_time_seconds,
            "concurrency": args.concurrency,
            "soft_closeout_seconds": args.soft_closeout_seconds,
            "hard_kill_grace_seconds": args.hard_kill_grace_seconds,
            "worker_runtime_seconds": args.worker_runtime_seconds,
            "worker_min_runtime_seconds": args.worker_min_runtime_seconds,
            "iterations_ceiling": args.iterations_ceiling,
        },
        "task": {
            "artifact_name": ARTIFACT_NAME,
            "primary_metric": PRIMARY_METRIC,
            "goal_plus_process_metric": GOAL_PLUS_PROCESS_METRIC,
            "controller_only_official_evaluation": (
                CONTROLLER_ONLY_OFFICIAL_EVALUATION
            ),
            "evaluation_mode": EVALUATION_MODE,
            "requires_protected_pi_workers": REQUIRES_PROTECTED_PI_WORKERS,
            "goal_plus_early_stop": GOAL_PLUS_EARLY_STOP_CONTRACT,
            "goal_plus_posthoc_selection": GOAL_PLUS_POSTHOC_SELECTION_CONTRACT,
            "direction": DIRECTION,
            "codex_sandbox": CODEX_SANDBOX,
            "upstream_key": UPSTREAM_KEY,
            "upstream_tracking_branch": (
                upstreams[UPSTREAM_KEY]["tracking_branch"]
                if LOCAL_SOURCE_RELATIVE is None
                else None
            ),
            "upstream_commit": git_commit(benchmark_checkout),
            "case_set": CASE_SET_DESCRIPTION,
            "source_kind": source_kind,
            "official_benchmark_comparable": OFFICIAL_BENCHMARK_COMPARABLE,
            **(
                {
                    key: value
                    for key, value in load_json(workspaces[0] / "task.json").items()
                    if key
                    in {
                        "evaluator",
                        "evaluator_sha256",
                        "source_revision",
                        "suite",
                    }
                }
                if workspaces and (workspaces[0] / "task.json").is_file()
                else {}
            ),
        },
        "environment": {
            "manifest": str(args.environment_manifest.absolute()),
            "checkout_root": str(checkout_root),
            "benchmark_root": str(benchmark_root),
            "benchmark_checkout": str(benchmark_checkout),
            "benchmark_branch": checkout_branch(benchmark_checkout),
            "benchmark_commit": git_commit(benchmark_checkout),
            "benchmark_tracking_branch": (
                upstreams[UPSTREAM_KEY]["tracking_branch"]
                if LOCAL_SOURCE_RELATIVE is None
                else None
            ),
            "goal_plus_root": str(goal_plus_root),
            "goal_plus_branch": checkout_branch(goal_plus_root),
            "goal_plus_commit": git_commit(goal_plus_root),
            "goal_plus_source": external_goal_plus_source(),
            "goal_plus_tracking_branch": upstreams["goal_plus"]["tracking_branch"],
            "runtime_bin": str(runtime_bin(args.venv.expanduser().absolute())),
            **(
                {
                    "skydiscover_root": str(skydiscover_root),
                    "skydiscover_branch": checkout_branch(skydiscover_root),
                    "skydiscover_commit": git_commit(skydiscover_root),
                    "skydiscover_tracking_branch": upstreams["skydiscover"][
                        "tracking_branch"
                    ],
                }
                if skydiscover_root is not None
                else {}
            ),
        },
        "workspace": workspace_value,
        "workspaces": [str(path) for path in workspaces],
        "workspace_commits": workspace_commits,
        "prompt_contract": prompt_contract,
        "goal_plus_config": goal_plus_config,
        "search_backend": search_backend,
        "secret_policy": "credentials are inherited and never serialized",
    }
    write_json(run_dir / "experiment.json", manifest)
    print(run_dir)
    return 0


def evaluator_budget(workspace: Path) -> dict[str, Any]:
    return load_json(workspace / ".bench-runtime/budget.json")


def evaluate(
    workspace: Path,
    mode: str,
    upstream_root: Path | None = None,
) -> dict[str, Any]:
    metadata = load_json(workspace / "task.json")
    trusted_root = upstream_root
    if trusted_root is None:
        stored_root = metadata.get("upstream_root")
        if not isinstance(stored_root, str) or not stored_root:
            raise RuntimeError(
                "evaluation requires a controller-provided benchmark root"
            )
        trusted_root = Path(stored_root)
    return evaluate_workspace(
        workspace, Path(trusted_root), mode
    )


def evaluate_with_controller_runtime(
    workspace: Path,
    mode: str,
    controller_runtime: Path,
    upstream_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate without materializing mutable runtime files in a Goal workspace."""
    previous = os.environ.get("GOAL_PLUS_VERIFIER_TMPDIR")
    os.environ["GOAL_PLUS_VERIFIER_TMPDIR"] = str(controller_runtime)
    try:
        if upstream_root is None:
            return evaluate(workspace, mode)
        return evaluate(workspace, mode, upstream_root)
    finally:
        if previous is None:
            os.environ.pop("GOAL_PLUS_VERIFIER_TMPDIR", None)
        else:
            os.environ["GOAL_PLUS_VERIFIER_TMPDIR"] = previous


def controller_only_official_evaluation(manifest: dict[str, Any]) -> bool:
    prepared = (manifest.get("task") or {}).get(
        "controller_only_official_evaluation"
    )
    if type(prepared) is not bool:
        raise RuntimeError(
            "prepared task is missing its controller-only official evaluation contract"
        )
    if prepared != CONTROLLER_ONLY_OFFICIAL_EVALUATION:
        raise RuntimeError(
            "prepared official evaluation contract does not match the adapter"
        )
    return prepared


def goal_plus_early_stop_contract(
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    task = manifest.get("task") or {}
    if "goal_plus_early_stop" not in task:
        if GOAL_PLUS_EARLY_STOP_CONTRACT is None:
            return None
        raise RuntimeError("prepared task is missing its Goal Plus early-stop contract")
    prepared = task.get("goal_plus_early_stop")
    if prepared != GOAL_PLUS_EARLY_STOP_CONTRACT:
        raise RuntimeError(
            "prepared Goal Plus early-stop contract does not match the adapter"
        )
    return prepared


def goal_plus_posthoc_selection_contract(
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    task = manifest.get("task") or {}
    if "goal_plus_posthoc_selection" not in task:
        if GOAL_PLUS_POSTHOC_SELECTION_CONTRACT is None:
            return None
        raise RuntimeError(
            "prepared task is missing its Goal Plus posthoc-selection contract"
        )
    prepared = task.get("goal_plus_posthoc_selection")
    if prepared != GOAL_PLUS_POSTHOC_SELECTION_CONTRACT:
        raise RuntimeError(
            "prepared Goal Plus posthoc-selection contract does not match the adapter"
        )
    return prepared


def _iteration_satisfies_early_stop(
    iteration: Any,
    contract: dict[str, Any],
) -> bool:
    if not isinstance(iteration, dict):
        return False
    score = iteration.get("score")
    if type(score) not in {int, float} or not math.isfinite(float(score)):
        return False
    if float(score) != float(contract["target_score"]):
        return False
    if iteration.get("process_passed") is not True:
        return False
    if iteration.get("git_artifact_clean") is not True:
        return False
    if iteration.get("touched_denied_files") is not False:
        return False
    if iteration.get("changed_outside_allowed") is not False:
        return False
    if iteration.get("disposition") not in {"keep", "retain"}:
        return False
    git_head = iteration.get("git_head")
    ledger_head = iteration.get("ledger_git_head")
    settled_head = iteration.get("workspace_git_head_after_settlement")
    return (
        isinstance(git_head, str)
        and bool(git_head)
        and isinstance(ledger_head, str)
        and bool(ledger_head)
        and settled_head == ledger_head
    )


def goal_plus_early_stop_reached(
    workspace: Path,
    contract: dict[str, Any],
) -> bool:
    workspace = workspace.resolve()
    root = workspace / ".gp"
    active_states = {
        "running",
        "waiting_for_workers",
        "evaluating",
        "selecting",
        "selection_blocked",
        "ready_to_promote",
    }
    for run_path in sorted((root / "runs").glob("*/run.json")):
        if run_path.is_symlink() or not run_path.is_file():
            continue
        try:
            run = load_json(run_path)
            if not isinstance(run, dict):
                continue
            frozen_spec_id = run.get("frozen_spec_id")
            frozen = load_json(root / "specs" / frozen_spec_id / "frozen_spec.json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        spec = frozen.get("spec") if isinstance(frozen, dict) else None
        try:
            source_matches = Path(run.get("source_path", "")).resolve() == workspace
        except (OSError, RuntimeError):
            source_matches = False
        if (
            run.get("run_id") != run_path.parent.name
            or run.get("state") not in active_states
            or not source_matches
            or not isinstance(spec, dict)
            or spec.get("metric_name") != contract["metric_name"]
        ):
            continue
        for path in sorted((run_path.parent / "candidates").glob("*/candidate.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                candidate = load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                not isinstance(candidate, dict)
                or candidate.get("candidate_id") != path.parent.name
                or not isinstance(candidate.get("iterations"), list)
            ):
                continue
            if any(
                _iteration_satisfies_early_stop(iteration, contract)
                for iteration in candidate["iterations"]
            ):
                return True
    return False


def verified_early_stop_completion(
    control: dict[str, Any],
    final: dict[str, Any] | None,
    contract: dict[str, Any] | None,
) -> bool:
    """Confirm that a controlled live-pass stop survived official final scoring."""
    if (
        not isinstance(contract, dict)
        or contract.get("enabled") is not True
        or control.get("early_stop_triggered") is not True
        or control.get("deadline_reached") is not False
        or control.get("controller_interrupted") is not False
        or control.get("hard_killed") is not False
        or control.get("soft_stop_signal") != "SIGTERM"
        or control.get("returncode")
        not in {0, -signal.SIGTERM, 128 + signal.SIGTERM}
        or not isinstance(final, dict)
        or final.get("valid") is not True
        or final.get("mode") != "final"
    ):
        return False
    primary = final.get("primary_metric")
    if (
        not isinstance(primary, dict)
        or primary.get("name") != contract.get("metric_name")
    ):
        return False
    score = primary.get("value")
    return (
        type(score) in {int, float}
        and math.isfinite(float(score))
        and type(contract.get("target_score")) in {int, float}
        and math.isfinite(float(contract["target_score"]))
        and float(score) == float(contract["target_score"])
    )


@contextmanager
def controller_subprocess_environment(
    *, runtime_bin_dir: Path, verifier_tmpdir: Path
):
    """Give controller-owned Goal Plus verifiers the resolved benchmark runtime."""
    updates = {
        "PATH": str(runtime_bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "GOAL_PLUS_VERIFIER_TMPDIR": str(verifier_tmpdir),
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def score_order_key(evaluation: dict[str, Any]) -> float:
    value = primary_score(evaluation)
    return value if DIRECTION == "minimize" else -value


def copy_artifact(source: Path, destination: Path) -> None:
    """Copy one adapter artifact while preserving file or directory shape."""
    if source.is_dir():
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        shutil.copytree(source, destination)
        return
    shutil.copy2(source, destination)


def _publicly_compliant_iteration(iteration: Any) -> bool:
    """Match the frozen public-gate eligibility rules without using F1."""
    if not isinstance(iteration, dict):
        return False
    score = iteration.get("score")
    return (
        iteration.get("process_passed") is True
        and type(iteration.get("iteration")) is int
        and iteration["iteration"] >= 1
        and isinstance(iteration.get("git_head"), str)
        and (
            iteration.get("artifact_clean")
            if iteration.get("artifact_clean") is not None
            else iteration.get("git_artifact_clean")
        ) is True
        and not iteration.get("touched_denied_files", False)
        and not iteration.get("changed_outside_allowed", False)
        and type(score) in {int, float}
        and math.isfinite(float(score))
    )


def _materialize_git_directory(
    repository: Path,
    git_head: str,
    artifact_name: str,
    destination: Path,
) -> str:
    """Materialize a committed direct-file artifact and return its content hash."""
    if len(git_head) != 40 or any(
        character not in "0123456789abcdef" for character in git_head
    ):
        raise ValueError(f"invalid iteration Git head: {git_head!r}")
    verified = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            f"{git_head}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if verified != git_head:
        raise RuntimeError("iteration Git head did not resolve exactly")
    tree = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-tree",
            "-rz",
            "--full-tree",
            git_head,
            "--",
            artifact_name,
        ],
        capture_output=True,
        check=True,
    ).stdout
    destination.mkdir(mode=0o700)
    digest = hashlib.sha256()
    digest.update(b"bench-goal-plus-committed-directory-v1\0")
    seen: set[str] = set()
    for raw_entry in tree.split(b"\0"):
        if not raw_entry:
            continue
        header, separator, raw_path = raw_entry.partition(b"\t")
        if not separator:
            raise RuntimeError("malformed Git tree entry")
        mode, object_type, object_id = header.decode("ascii").split(" ")
        path = raw_path.decode("utf-8")
        relative = Path(path)
        if (
            mode not in {"100644", "100755"}
            or object_type != "blob"
            or len(relative.parts) != 2
            or relative.parts[0] != artifact_name
            or relative.parts[1] in {"", ".", ".."}
            or relative.parts[1] in seen
        ):
            raise RuntimeError(f"unsafe committed artifact entry: {path!r}")
        name = relative.parts[1]
        seen.add(name)
        payload = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "blob", object_id],
            capture_output=True,
            check=True,
        ).stdout
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        output = destination / name
        output.write_bytes(payload)
        output.chmod(0o600)
    if not seen:
        raise RuntimeError("committed artifact directory is empty or missing")
    return digest.hexdigest()


def _posthoc_metric_value(
    evaluation: dict[str, Any], contract: dict[str, Any]
) -> float:
    metric_name = contract["metric_name"]
    primary = evaluation.get("primary_metric")
    if (
        evaluation.get("mode") != "final"
        or evaluation.get("valid") is not True
        or evaluation.get("format_valid") is not True
        or not isinstance(primary, dict)
        or primary.get("name") != metric_name
        or primary.get("direction") != contract["metric_direction"]
    ):
        raise RuntimeError("official evaluator returned an invalid final result")
    primary_value = primary.get("value")
    metric_value = evaluation.get(metric_name)
    if (
        type(primary_value) not in {int, float}
        or type(metric_value) not in {int, float}
        or not math.isfinite(float(primary_value))
        or not math.isfinite(float(metric_value))
        or float(primary_value) != float(metric_value)
    ):
        raise RuntimeError("official evaluator returned an invalid primary metric")
    value = float(metric_value)
    if metric_name == "f1" and not 0.0 <= value <= 1.0:
        raise RuntimeError("official evaluator returned an out-of-range F1")
    score_payload = evaluation.get("zsoft_score")
    if (
        metric_name == "f1"
        and (
            not isinstance(score_payload, dict)
            or type(score_payload.get("f1")) not in {int, float}
            or float(score_payload["f1"]) != value
        )
    ):
        raise RuntimeError("official evaluator F1 payload is inconsistent")
    return value


def finalize_posthoc_official_selection(
    *,
    run_dir: Path,
    workspace: Path,
    benchmark_root: Path | None,
    closeout: dict[str, Any],
    contract: dict[str, Any],
    worker_shutdown_verified: bool,
) -> dict[str, Any]:
    """Score hidden committed snapshots posthoc and publish the best result."""
    if worker_shutdown_verified is not True:
        raise RuntimeError("posthoc scoring requires verified worker shutdown")
    normalized_contract = _normalize_posthoc_selection_contract(
        contract,
        primary_metric=PRIMARY_METRIC,
        direction=DIRECTION,
        controller_only=True,
    )
    if normalized_contract is None:
        raise RuntimeError("posthoc scoring requires an enabled selection contract")
    contract = normalized_contract
    started = time.monotonic()
    run_dir = Path(run_dir).resolve(strict=True)
    workspace_input = Path(workspace)
    if workspace_input.is_symlink():
        raise RuntimeError("posthoc selection rejects a symlinked workspace")
    workspace = workspace_input.resolve(strict=True)
    if workspace == run_dir or run_dir not in workspace.parents:
        raise RuntimeError("posthoc selection requires a run-owned workspace")
    if benchmark_root is None:
        raise RuntimeError("posthoc selection requires a trusted benchmark root")
    task_path = workspace / "task.json"
    if task_path.is_symlink() or not task_path.is_file():
        raise RuntimeError("posthoc selection requires a regular task manifest")

    analysis_parent = run_dir / "controller-runtime/posthoc-selection"
    analysis_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt_root = Path(
        tempfile.mkdtemp(prefix="attempt-", dir=str(analysis_parent))
    )
    score_record_path = run_dir / "posthoc-candidate-scores.json"
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[tuple[str, str, int], Path] = {}
    official_calls = 0
    cache_hits = 0
    selected: dict[str, Any] | None = None

    try:
        closeout_runs = closeout.get("runs")
        if (
            closeout.get("completed") is not True
            or not isinstance(closeout_runs, list)
            or not closeout_runs
        ):
            raise RuntimeError("posthoc selection has no closed Search run")
        run_ids: set[str] = set()
        for item in closeout_runs:
            run_id = item.get("run_id") if isinstance(item, dict) else None
            if (
                not isinstance(run_id, str)
                or not run_id
                or Path(run_id).name != run_id
                or item.get("final_state") != "promoted"
            ):
                raise RuntimeError(
                    "posthoc selection requires a safely promoted Search run"
                )
            run_ids.add(run_id)

        search_runs_root = workspace / ".gp/runs"
        if search_runs_root.is_symlink() or not search_runs_root.is_dir():
            raise RuntimeError("posthoc selection requires a real Search runs root")
        sequence = 0
        for run_id in sorted(run_ids):
            search_run = search_runs_root / run_id
            if search_run.is_symlink() or not search_run.is_dir():
                raise RuntimeError(f"Search run is unavailable: {run_id}")
            candidates_root = search_run / "candidates"
            repositories_root = search_run / "workspace"
            if (
                candidates_root.is_symlink()
                or not candidates_root.is_dir()
                or repositories_root.is_symlink()
                or not repositories_root.is_dir()
            ):
                raise RuntimeError(f"Search run roots are unavailable: {run_id}")
            candidate_paths = sorted(
                candidates_root.glob("*/candidate.json")
            )
            if not candidate_paths:
                raise RuntimeError(f"Search run has no candidates: {run_id}")
            for candidate_path in candidate_paths:
                if (
                    candidate_path.parent.is_symlink()
                    or candidate_path.is_symlink()
                    or not candidate_path.is_file()
                ):
                    raise RuntimeError("candidate metadata must be a regular file")
                candidate = load_json(candidate_path)
                candidate_id = candidate.get("candidate_id")
                iterations = candidate.get("iterations")
                if (
                    not isinstance(candidate_id, str)
                    or not candidate_id
                    or candidate_id != candidate_path.parent.name
                    or not isinstance(iterations, list)
                ):
                    raise RuntimeError("candidate iteration metadata is malformed")
                eligible = [
                    iteration
                    for iteration in iterations
                    if _publicly_compliant_iteration(iteration)
                ]
                if not eligible:
                    continue
                iteration_numbers = [item["iteration"] for item in eligible]
                if len(iteration_numbers) != len(set(iteration_numbers)):
                    raise RuntimeError(
                        f"candidate {candidate_id} has duplicate eligible iterations"
                    )
                repository = repositories_root / candidate_id
                if repository.is_symlink() or not repository.is_dir():
                    raise RuntimeError(
                        f"candidate workspace is unavailable: {candidate_id}"
                    )
                for iteration in sorted(
                    eligible, key=lambda item: item["iteration"]
                ):
                    sequence += 1
                    iteration_number = int(iteration["iteration"])
                    git_head = str(iteration["git_head"])
                    row: dict[str, Any] = {
                        "run_id": run_id,
                        "candidate_id": candidate_id,
                        "iteration": iteration_number,
                        "git_head": git_head,
                        "public_score": float(iteration["score"]),
                        "artifact_hash": iteration.get("artifact_hash"),
                    }
                    try:
                        evaluation_workspace = (
                            attempt_root
                            / "materialized"
                            / f"snapshot-{sequence:06d}"
                            / "workspace"
                        )
                        evaluation_workspace.mkdir(mode=0o700, parents=True)
                        shutil.copy2(task_path, evaluation_workspace / "task.json")
                        snapshot_sha256 = _materialize_git_directory(
                            repository,
                            git_head,
                            ARTIFACT_NAME,
                            evaluation_workspace / ARTIFACT_NAME,
                        )
                        row["snapshot_sha256"] = snapshot_sha256
                        artifact_paths[
                            (run_id, candidate_id, iteration_number)
                        ] = evaluation_workspace / ARTIFACT_NAME
                        cached = cache.get(snapshot_sha256)
                        if cached is None:
                            official_calls += 1
                            try:
                                evaluation = evaluate_with_controller_runtime(
                                    evaluation_workspace,
                                    "final",
                                    attempt_root / "scores" / snapshot_sha256,
                                    benchmark_root,
                                )
                                metric_value = _posthoc_metric_value(
                                    evaluation, contract
                                )
                                cached = {
                                    "evaluation": evaluation,
                                    "metric_value": metric_value,
                                    "error": None,
                                    "official_evaluator_call": official_calls,
                                }
                            except Exception as exc:
                                cached = {
                                    "evaluation": None,
                                    "metric_value": None,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "official_evaluator_call": official_calls,
                                }
                            cache[snapshot_sha256] = cached
                            row["score_source"] = "official_evaluator"
                        else:
                            cache_hits += 1
                            row["score_source"] = "artifact_cache"
                        row["official_evaluator_call"] = cached[
                            "official_evaluator_call"
                        ]
                        if cached["error"] is not None:
                            raise RuntimeError(str(cached["error"]))
                        evaluation = cached["evaluation"]
                        score_payload = evaluation.get("zsoft_score") or {}
                        row.update(
                            {
                                contract["metric_name"]: cached["metric_value"],
                                "precision": score_payload.get("precision"),
                                "recall": score_payload.get("recall"),
                                "tp": score_payload.get("tp"),
                                "fp": score_payload.get("fp"),
                                "fn": score_payload.get("fn"),
                            }
                        )
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        errors.append(
                            {
                                "run_id": run_id,
                                "candidate_id": candidate_id,
                                "iteration": iteration_number,
                                "error": row["error"],
                            }
                        )
                    rows.append(row)
    except Exception as exc:
        errors.append(
            {
                "scope": "candidate_discovery",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    if not rows and not errors:
        errors.append(
            {
                "scope": "candidate_selection",
                "error": "no publicly compliant candidate iteration is available",
            }
        )

    if not errors:
        metric_name = contract["metric_name"]
        selected_row = min(
            rows,
            key=lambda row: (
                (
                    float(row[metric_name])
                    if contract["metric_direction"] == "minimize"
                    else -float(row[metric_name])
                ),
                str(row["candidate_id"]),
                -int(row["iteration"]),
                str(row["run_id"]),
            ),
        )
        selected = {
            "run_id": selected_row["run_id"],
            "candidate_id": selected_row["candidate_id"],
            "iteration": selected_row["iteration"],
            "git_head": selected_row["git_head"],
            "snapshot_sha256": selected_row["snapshot_sha256"],
            metric_name: selected_row[metric_name],
        }
        snapshot = cache[str(selected_row["snapshot_sha256"])]
        selected_evaluation = json.loads(json.dumps(snapshot["evaluation"]))
        selected_evaluation["posthoc_selection"] = {
            **selected,
            "candidate_scope": contract["candidate_scope"],
            "tie_break": contract["tie_break"],
            "timing": contract["timing"],
            "visible_to_workers": False,
            "eligible_iteration_count": len(rows),
            "unique_artifact_count": len(cache),
            "official_evaluator_calls": official_calls,
            "score_record_path": str(score_record_path),
        }
        selected_artifact = artifact_paths[
            (
                str(selected_row["run_id"]),
                str(selected_row["candidate_id"]),
                int(selected_row["iteration"]),
            )
        ]
        try:
            copy_artifact(selected_artifact, run_dir / ARTIFACT_NAME)
            write_json_atomic(run_dir / "final-eval.json", selected_evaluation)
        except Exception as exc:
            errors.append(
                {
                    "scope": "final_publication",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            selected = None

    result = {
        "schema_version": 1,
        "completed": not errors and selected is not None,
        "contract": contract,
        "timing_scope": "posthoc_after_agent_exit_and_controller_closeout",
        "visible_to_workers": False,
        "affects_online_search": False,
        "worker_shutdown_verified": True,
        "eligible_iteration_count": len(rows),
        "unique_artifact_count": len(cache),
        "official_evaluator_calls": official_calls,
        "artifact_cache_hits": cache_hits,
        "selected": selected,
        "scores": rows,
        "errors": errors,
        "attempt_root": str(attempt_root),
        "score_record_path": str(score_record_path),
        "duration_seconds": time.monotonic() - started,
    }
    write_json_atomic(score_record_path, result)
    return result


def condition_incomplete_reason(
    goal_plus_state: dict[str, Any], condition: dict[str, Any] | None
) -> str | None:
    if not condition or condition.get("id") not in {"B3", "B4"}:
        return None
    expected_mode = condition.get("search_space_mode")
    linked_run_ids = {
        goal.get("linked_run_id")
        for goal in goal_plus_state.get("goals", [])
        if isinstance(goal, dict) and isinstance(goal.get("linked_run_id"), str)
    }
    active_runs = [
        run
        for run in goal_plus_state.get("runs", [])
        if isinstance(run, dict)
        and run.get("candidate_count", 0) > 0
        and (
            run.get("run_id") in linked_run_ids
            if linked_run_ids
            else run.get("status") not in {"aborted", "failed"}
        )
    ]
    spaces = [run.get("search_space") or {} for run in active_runs]
    if not spaces or not all(space.get("exists") for space in spaces):
        return f"{condition['id']} requires a Search Space for every active run"
    actual_modes = {space.get("mode") for space in spaces}
    if actual_modes != {expected_mode}:
        return (
            f"{condition['id']} requires Search Space mode {expected_mode!r}, "
            f"observed {sorted(str(mode) for mode in actual_modes)}"
        )
    return None


def codex_command(
    *,
    codex_bin: str,
    workspace: Path,
    output_last_message: Path,
    model: str,
    reasoning_effort: str,
    api_base: str | None,
    sandbox: str,
    goal_plus: bool,
    ephemeral: bool,
    max_concurrent_threads_per_session: int = 5,
) -> list[str]:
    command = [
        codex_bin,
        "exec",
        "--json",
        "--sandbox",
        sandbox,
        "--cd",
        str(workspace),
        "--output-last-message",
        str(output_last_message),
    ]
    if not goal_plus:
        command.append("--ignore-user-config")
    command.extend(
        [
            "--color",
            "never",
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
    )
    if not goal_plus:
        command.extend(["--config", 'approval_policy="never"'])
    if ephemeral:
        command.append("--ephemeral")
    if api_base:
        command.extend(codex_provider_args(api_base))
    if goal_plus:
        if max_concurrent_threads_per_session < 2:
            raise ValueError("Goal Plus Codex requires at least two agent threads")
        command.extend(
            [
                "--config",
                "features.multi_agent=true",
                "--config",
                "agents.enabled=true",
                "--config",
                "agents.max_concurrent_threads_per_session="
                f"{max_concurrent_threads_per_session}",
                "--dangerously-bypass-hook-trust",
                *codex_goal_plus_mcp_args(GOAL_PLUS_MCP_ENV_VARS),
            ]
        )
    command.extend(["--model", model, "-"])
    return command


def command_for_manifest(command: list[str], api_base: str | None) -> list[str]:
    if not api_base:
        return command.copy()
    return [part.replace(api_base, "<provider-url>") for part in command]


def pi_provider_config(args: argparse.Namespace) -> tuple[str, str, str]:
    """Keep direct helper callers compatible with the historical Pi defaults."""
    return (
        getattr(args, "pi_provider_id", PI_PROVIDER_ID),
        getattr(args, "pi_api", "openai-responses"),
        getattr(args, "pi_api_key_env", PI_API_KEY_ENV),
    )


def execute_plain(
    manifest: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any]:
    budget = manifest["budget"]
    benchmark_root_text = (manifest.get("environment") or {}).get("benchmark_root")
    benchmark_root = Path(benchmark_root_text) if benchmark_root_text else None
    controller_only = controller_only_official_evaluation(manifest)
    is_pi = manifest["method"] == "plain-pi"
    pi_provider_id, pi_api, pi_api_key_env = pi_provider_config(args)
    workspaces = [Path(path) for path in manifest["workspaces"]]
    lanes_root = run_dir / "lanes"
    lanes_root.mkdir()
    jobs = []
    seeds = []
    setup_calls = 0
    for lane_index, workspace in enumerate(workspaces):
        lane_name = f"lane-{lane_index:02d}"
        lane_dir = lanes_root / lane_name
        lane_dir.mkdir()
        seed = evaluate(workspace, "public", benchmark_root)
        write_json(lane_dir / "seed-eval.json", seed)
        seeds.append({"lane": lane_name, "evaluation": seed})
        setup_calls += evaluator_budget(workspace)["total_claimed"]
        prompt = render_plain_prompt(
            (workspace / "TASK.md").read_text(),
            budget["wall_time_seconds"],
            budget["soft_closeout_seconds"],
            public_feedback_command=PUBLIC_FEEDBACK_COMMAND,
            controller_only_official_evaluation=controller_only,
        )
        (lane_dir / "prompt.md").write_text(prompt)
        lane_environment = environment.copy()
        if is_pi:
            qualified_model = f"{pi_provider_id}/{args.model}"
            pi_home = lane_dir / "pi-home"
            write_pi_models_config(
                pi_home,
                api_base=args.api_base,
                model=args.model,
                reasoning_effort=manifest["reasoning_effort"],
                provider_id=pi_provider_id,
                api=pi_api,
                api_key_env=pi_api_key_env,
            )
            lane_environment["PI_CODING_AGENT_DIR"] = str(pi_home)
            command = [
                args.pi_bin,
                "--mode",
                "json",
                "--provider",
                pi_provider_id,
                "--model",
                qualified_model,
                "--thinking",
                manifest["reasoning_effort"],
                "--approve",
                "--session-dir",
                str(lane_dir / "pi-session"),
                "--session-id",
                f"bench-{run_dir.name}-{lane_name}",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
                prompt,
            ]
            stdin_text = None
            recorded_command = [*command[:-1], "<task-prompt>"]
        else:
            command = codex_command(
                codex_bin=args.codex_bin,
                workspace=workspace,
                output_last_message=lane_dir / "final-message.txt",
                model=args.model,
                reasoning_effort=manifest["reasoning_effort"],
                api_base=args.api_base,
                sandbox=CODEX_SANDBOX,
                goal_plus=False,
                ephemeral=True,
            )
            stdin_text = prompt
            recorded_command = command_for_manifest(command, args.api_base)
        jobs.append(
            {
                "name": lane_name,
                "command": command,
                "recorded_command": recorded_command,
                "cwd": workspace,
                "stdin_text": stdin_text,
                "environment": lane_environment,
                "stdout_path": lane_dir / "events.jsonl",
                "stderr_path": lane_dir / "stderr.log",
            }
        )
    write_json(run_dir / "seed-evals.json", {"lanes": seeds})
    control = run_controlled_many(
        jobs,
        environment=environment,
        wall_time_seconds=budget["wall_time_seconds"],
        hard_kill_grace_seconds=budget["hard_kill_grace_seconds"],
    )
    lane_results = []
    for lane_index, workspace in enumerate(workspaces):
        lane_name = f"lane-{lane_index:02d}"
        lane_dir = lanes_root / lane_name
        lane_evaluation = evaluate(
            workspace,
            "public" if controller_only else "final",
            benchmark_root,
        )
        write_json(
            lane_dir
            / ("public-eval.json" if controller_only else "final-eval.json"),
            lane_evaluation,
        )
        candidate = lane_dir / ARTIFACT_NAME
        copy_artifact(workspace / ARTIFACT_NAME, candidate)
        lane_results.append(
            {
                "lane": lane_name,
                "workspace": str(workspace),
                "candidate": str(candidate),
                "evaluation": lane_evaluation,
                ("pi" if is_pi else "codex"): (
                    parse_pi_events(lane_dir / "events.jsonl")
                    if is_pi
                    else parse_codex_events(lane_dir / "events.jsonl")
                ),
            }
        )
    valid_lane_results = [
        item for item in lane_results if item["evaluation"].get("valid") is True
    ]
    selection_pool = valid_lane_results or lane_results
    selected = (
        min(selection_pool, key=lambda item: item["lane"])
        if controller_only
        else min(
            selection_pool,
            key=lambda item: score_order_key(item["evaluation"]),
        )
    )
    selected_evaluation = selected["evaluation"]
    if controller_only:
        selected_evaluation = evaluate(
            Path(selected["workspace"]), "final", benchmark_root
        )
        selected["official_evaluation"] = selected_evaluation
        write_json(
            lanes_root / selected["lane"] / "final-eval.json",
            selected_evaluation,
        )
    write_json(run_dir / "lane-results.json", {"lanes": lane_results})
    write_json(run_dir / "final-eval.json", selected_evaluation)
    copy_artifact(Path(selected["candidate"]), run_dir / ARTIFACT_NAME)
    control["selected_lane"] = selected["lane"]
    control["selected_score"] = primary_score(selected_evaluation)
    agent_key = "pi" if is_pi else "codex"
    control[agent_key] = {
        "lanes": [
            {"lane": item["lane"], **item[agent_key]} for item in lane_results
        ],
        "coverage": f"top-level {'Pi' if is_pi else 'Codex'} usage for every independent lane",
    }
    control["evaluator_calls"] = {
        "lane_count": len(lane_results),
        "total_claimed": (
            setup_calls + len(lane_results) + 1
            if controller_only
            else sum(
                item["evaluation"]["budget"]["total_claimed"]
                for item in lane_results
            )
        ),
        "setup_claimed_before_t": setup_calls,
        "controller_final_claimed": 1 if controller_only else len(lane_results),
    }
    bad = [
        lane["name"]
        for lane in control["lanes"]
        if lane["returncode"] != 0 or lane["hard_killed"]
    ]
    if bad:
        control["result_incomplete_reason"] = (
            f"{manifest['method']} lanes did not exit cleanly: " + ", ".join(bad)
        )
    if not valid_lane_results or selected_evaluation.get("valid") is not True:
        control["result_incomplete_reason"] = (
            f"official final evaluator rejected every {manifest['method']} lane"
        )
    return control


def _controller_only_closeout_incomplete_reason(
    closeout: Any, *, deterministic_public_gate: bool = True
) -> str | None:
    if not isinstance(closeout, dict) or closeout.get("completed") is not True:
        error = closeout.get("error") if isinstance(closeout, dict) else None
        return (
            "controller-only Goal Plus closeout did not complete: "
            f"{error or 'unknown error'}"
        )
    runs = closeout.get("runs")
    if not isinstance(runs, list) or not runs:
        return "controller-only Goal Plus closeout has no completed Search run"
    for item in runs:
        if not isinstance(item, dict):
            return "controller-only Goal Plus closeout contains malformed Search evidence"
        selection = item.get("selection")
        promotion = item.get("promotion")
        goal_statuses = item.get("goal_statuses")
        if (
            not isinstance(selection, dict)
            or not isinstance(selection.get("selected_candidate_id"), str)
            or not selection["selected_candidate_id"]
            or (deterministic_public_gate and selection.get("selection_rule") != PUBLIC_GATE_SELECTION_RULE)
        ):
            return (
                "controller-only Goal Plus closeout lacks deterministic selection evidence"
            )
        if (
            not isinstance(promotion, dict)
            or not isinstance(promotion.get("artifact_path"), str)
            or not promotion["artifact_path"]
            or item.get("final_state") != "promoted"
        ):
            return "controller-only Goal Plus closeout lacks completed promotion evidence"
        if (
            not isinstance(goal_statuses, dict)
            or not goal_statuses
            or any(status != "complete" for status in goal_statuses.values())
        ):
            return (
                "controller-only Goal Plus closeout lacks complete Goal Plus terminal evidence"
            )
    return None


def _goal_plus_agent_wall_time_seconds(
    budget: dict[str, Any], controller_only: bool
) -> int:
    wall_time = int(budget["wall_time_seconds"])
    if not controller_only:
        return wall_time
    closeout_reserve = int(budget["soft_closeout_seconds"])
    agent_wall_time = wall_time - closeout_reserve
    if agent_wall_time < 1:
        raise ValueError(
            "controller-only Goal Plus closeout reserve must fit inside wall time"
        )
    return agent_wall_time


def _candidate_cleanup_incomplete_reason(cleanup: Any) -> str | None:
    if not isinstance(cleanup, list) or not cleanup:
        return "posthoc selection lacks candidate session cleanup evidence"
    for receipt in cleanup:
        if (
            not isinstance(receipt, dict)
            or not receipt.get("run_id")
            or receipt.get("active_count") != 0
        ):
            return "candidate sessions were not closed before posthoc selection"
    return None


def _posthoc_prerequisite_incomplete_reason(
    closeout_reason: str | None, candidate_cleanup: Any
) -> str | None:
    """Preserve an earlier closeout failure before checking session cleanup."""
    if closeout_reason is not None:
        return closeout_reason
    return _candidate_cleanup_incomplete_reason(candidate_cleanup)


def execute_goal_plus(
    manifest: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any]:
    budget = manifest["budget"]
    benchmark_root_text = (manifest.get("environment") or {}).get("benchmark_root")
    benchmark_root = Path(benchmark_root_text) if benchmark_root_text else None
    workspace = Path(manifest["workspace"])
    controller_only = controller_only_official_evaluation(manifest)
    early_stop = goal_plus_early_stop_contract(manifest)
    posthoc_selection = goal_plus_posthoc_selection_contract(manifest)
    prepared_goal_config = manifest.get("goal_plus_config") or {}
    if prepared_goal_config.get("early_stop") != early_stop:
        raise RuntimeError(
            "prepared Goal Plus config does not match the task early-stop contract"
        )
    if prepared_goal_config.get("posthoc_selection") != posthoc_selection:
        raise RuntimeError(
            "prepared Goal Plus config does not match the task posthoc-selection contract"
        )
    is_pi = manifest.get("method", "goal-plus-codex") == "goal-plus-pi"
    if is_pi and controller_only and EVALUATION_MODE == "blind":
        environment[CONTROLLER_ONLY_CLOSEOUT_ENV] = "1"
    else:
        environment.pop(CONTROLLER_ONLY_CLOSEOUT_ENV, None)
    if controller_only:
        environment.pop(GOAL_PLUS_EXTERNAL_EVIDENCE_DIR_ENV, None)
        environment[GOAL_PLUS_GLOBAL_EVIDENCE_MODE_ENV] = "manual"
        environment[GOAL_PLUS_SUPPLEMENTAL_EVALUATION_ENABLED_ENV] = "0"
        environment[GOAL_PLUS_SUPPLEMENTAL_EVALUATION_REQUIRED_ENV] = "0"
    pi_provider_id, pi_api, pi_api_key_env = pi_provider_config(args)
    if (workspace / ".gp").exists():
        raise RuntimeError("standard Goal Plus run must start without .gp")
    seed = evaluate_with_controller_runtime(
        workspace,
        "public",
        run_dir / "controller-runtime/seed",
        benchmark_root,
    )
    write_json(run_dir / "seed-eval.json", seed)
    setup_calls = seed["budget"]["total_claimed"]
    if seed.get("valid") is not True:
        return {
            "returncode": 2,
            "duration_seconds": seed.get("duration_seconds"),
            "deadline_reached": False,
            "hard_killed": False,
            "controller_interrupted": False,
            "preflight_failed": True,
            "result_incomplete_reason": (
                "seed evaluator did not complete; Agent execution was not started"
            ),
            "evaluator_calls": {
                "total_claimed": setup_calls,
                "setup_claimed_before_t": setup_calls,
                "process_verifier_commands": 0,
                "promotion_verifier_commands": 0,
                "controller_final_claimed": 0,
                "coverage": "preflight seed only; Agent and final evaluator not started",
            },
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "coverage": "Agent execution was not started after preflight failure",
            },
        }
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=budget["wall_time_seconds"]
    )
    environment["GOAL_PLUS_OUTER_DEADLINE_AT"] = deadline.isoformat()
    environment["GOAL_PLUS_VERIFIER_TMPDIR"] = str(
        run_dir / "controller-runtime/goal-plus"
    )
    configure_isolated_codex_home(environment, run_dir)
    configure_evidence_annotator_environment(
        environment,
        model=(
            f"{pi_provider_id}/{args.model}" if is_pi else args.model
        ),
        reasoning_effort=manifest.get(
            "reasoning_effort", DEFAULT_REASONING_EFFORT
        ),
        api_base=None if is_pi else args.api_base,
    )
    prompt = render_goal(
        task_text=(workspace / "TASK.md").read_text(),
        public_feedback_command=PUBLIC_FEEDBACK_COMMAND,
        artifact_name=ARTIFACT_NAME,
        artifact_is_directory=(workspace / ARTIFACT_NAME).is_dir(),
        metric_name=GOAL_PLUS_PROCESS_METRIC,
        metric_direction=DIRECTION,
        wall_seconds=budget["wall_time_seconds"],
        closeout_seconds=budget["soft_closeout_seconds"],
        concurrency=budget["concurrency"],
        agent_harness="pi" if is_pi else "codex",
        worker_model=f"{pi_provider_id}/{args.model}" if is_pi else args.model,
        reasoning_effort=manifest.get(
            "reasoning_effort", DEFAULT_REASONING_EFFORT
        ),
        worker_runtime_seconds=budget["worker_runtime_seconds"],
        worker_min_runtime_seconds=budget.get("worker_min_runtime_seconds"),
        verifier_timeout_seconds=VERIFIER_TIMEOUT_SECONDS,
        process_verifier_timeout_seconds=PROCESS_VERIFIER_TIMEOUT_SECONDS,
        coordination_condition=(manifest.get("condition") or {}).get("id"),
        search_space_mode=(manifest.get("condition") or {}).get("search_space_mode"),
        shared_dir_enabled=bool(
            (manifest.get("goal_plus_config") or {}).get("shared_dir_enabled")
        ),
        shared_cache_enabled=bool(
            (manifest.get("goal_plus_config") or {}).get("shared_cache_enabled")
        ),
        controller_only_official_evaluation=controller_only,
        evaluation_mode=EVALUATION_MODE,
        early_stop_contract=early_stop,
        outer_deadline_at=deadline.isoformat(),
    )
    (run_dir / "prompt.md").write_text(prompt)
    reasoning_effort = manifest.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
    if is_pi:
        qualified_model = f"{pi_provider_id}/{args.model}"
        pi_home = run_dir / "pi-home"
        write_pi_models_config(
            pi_home,
            api_base=args.api_base,
            model=args.model,
            reasoning_effort=reasoning_effort,
            provider_id=pi_provider_id,
            api=pi_api,
            api_key_env=pi_api_key_env,
        )
        environment["PI_CODING_AGENT_DIR"] = str(pi_home)
        environment["GOAL_PLUS_PI_MODEL"] = qualified_model
        command = [
            args.pi_bin,
            "--mode",
            "json",
            "--provider",
            pi_provider_id,
            "--model",
            qualified_model,
            "--thinking",
            reasoning_effort,
            "--approve",
            "--session-dir",
            str(run_dir / "pi-main-session"),
            "--session-id",
            f"bench-{run_dir.name}",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            prompt,
        ]
        stdin_text = None
        recorded_command = [*command[:-1], "<goal-prompt>"]
    else:
        command = codex_command(
            codex_bin=args.codex_bin,
            workspace=workspace,
            output_last_message=run_dir / "final-message.txt",
            model=args.model,
            reasoning_effort=reasoning_effort,
            api_base=args.api_base,
            sandbox=CODEX_SANDBOX,
            goal_plus=True,
            ephemeral=False,
            max_concurrent_threads_per_session=budget["concurrency"] + 1,
        )
        stdin_text = prompt
        recorded_command = command_for_manifest(command, args.api_base)
    agent_wall_time_seconds = _goal_plus_agent_wall_time_seconds(
        budget, controller_only
    )
    bind_goal_plus_environment(environment, run_dir)
    control = run_controlled(
        command,
        cwd=workspace,
        environment=environment,
        stdin_text=stdin_text,
        stdout_path=run_dir / "events.jsonl",
        stderr_path=run_dir / "stderr.log",
        wall_time_seconds=agent_wall_time_seconds,
        hard_kill_grace_seconds=budget["hard_kill_grace_seconds"],
        recorded_command=recorded_command,
        stop_predicate=(
            (lambda: goal_plus_early_stop_reached(workspace, early_stop))
            if early_stop is not None
            else None
        ),
    )
    control["agent_wall_time_seconds"] = agent_wall_time_seconds
    control["controller_closeout_reserve_seconds"] = (
        budget["soft_closeout_seconds"] if controller_only else 0
    )
    control["candidate_session_cleanup"] = close_candidate_sessions(
        workspace, budget["hard_kill_grace_seconds"]
    )
    try:
        with controller_subprocess_environment(
            runtime_bin_dir=Path(manifest["environment"]["runtime_bin"]),
            verifier_tmpdir=run_dir / "controller-runtime/goal-plus",
        ):
            closeout = finalize_goal_plus_search(
                workspace,
                deterministic_public_gate=controller_only and EVALUATION_MODE == "blind",
                verify_unsettled_candidates=not control.get(
                    "early_stop_triggered", False
                ),
            )
    except Exception as exc:
        closeout = {
            "completed": False,
            "runs": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    control["goal_plus_controller_closeout"] = closeout
    closeout_reason = (
        _controller_only_closeout_incomplete_reason(
            closeout, deterministic_public_gate=EVALUATION_MODE == "blind"
        )
        if controller_only
        else None
    )
    if closeout.get("completed") is not True:
        closeout_reason = (
            "Goal Plus controller closeout failed: "
            + closeout.get("error", "unknown error")
        )
    if is_pi and posthoc_selection is not None:
        closeout_reason = _posthoc_prerequisite_incomplete_reason(
            closeout_reason,
            control.get("candidate_session_cleanup"),
        )
    final: dict[str, Any] | None = None
    posthoc_result: dict[str, Any] | None = None
    if posthoc_selection is None and (closeout_reason is None or controller_only):
        final = evaluate_with_controller_runtime(
            workspace,
            "final",
            run_dir / "controller-runtime/final",
            benchmark_root,
        )
        write_json(run_dir / "final-eval.json", final)
        copy_artifact(workspace / ARTIFACT_NAME, run_dir / ARTIFACT_NAME)
        if closeout_reason is not None:
            control["result_incomplete_reason"] = closeout_reason
    elif closeout_reason is None:
        if posthoc_selection is not None:
            try:
                posthoc_result = finalize_posthoc_official_selection(
                    run_dir=run_dir,
                    workspace=workspace,
                    benchmark_root=benchmark_root,
                    closeout=closeout,
                    contract=posthoc_selection,
                    worker_shutdown_verified=True,
                )
            except Exception as exc:
                posthoc_result = {
                    "completed": False,
                    "visible_to_workers": False,
                    "affects_online_search": False,
                    "official_evaluator_calls": 0,
                    "errors": [
                        {
                            "scope": "posthoc_selection",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                }
            control["posthoc_official_selection"] = posthoc_result
            if posthoc_result.get("completed") is True:
                final = load_json(run_dir / "final-eval.json")
            else:
                control["official_evaluation_withheld"] = True
                control["result_incomplete_reason"] = (
                    "controller posthoc official selection failed"
                )
    else:
        control["official_evaluation_withheld"] = True
        control["result_incomplete_reason"] = closeout_reason
    if is_pi:
        control["pi"] = parse_pi_events(run_dir / "events.jsonl")
    else:
        control["codex"] = parse_codex_events(run_dir / "events.jsonl")
    control["goal_plus"] = collect_goal_plus_state(workspace)
    control["evidence_annotator_usage"] = collect_evidence_annotator_usage(
        workspace
    )
    goal_runs = control["goal_plus"]["runs"]
    process_calls = sum(
        item.get("process_verifier_command_count", 0) for item in goal_runs
    )
    promotion_calls = sum(
        item.get("promotion_verifier_command_count", 0) for item in goal_runs
    )
    final_claims = 1 if final is not None else 0
    control["evaluator_calls"] = {
        "total_claimed": (
            setup_calls
            + process_calls
            + promotion_calls
            + final_claims
        ),
        "setup_claimed_before_t": setup_calls,
        "process_verifier_commands": process_calls,
        "promotion_verifier_commands": promotion_calls,
        "controller_final_claimed": final_claims,
        "coverage": (
            "seed + Goal Plus verifier command logs + controller posthoc scorer ledger"
            if posthoc_result is not None
            else "seed + Goal Plus verifier command logs + controller final ledger"
        ),
    }
    early_stop_completion_verified = verified_early_stop_completion(
        control, final, early_stop
    )
    control["early_stop_completion_verified"] = early_stop_completion_verified
    if early_stop_completion_verified:
        control["minimum_lease_completion_waived"] = True
    reason = goal_plus_incomplete_reason(
        control["goal_plus"],
        expected_concurrency=budget["concurrency"],
        minimum_worker_verified_candidates=1,
    )
    if reason:
        control["result_incomplete_reason"] = reason
    condition_reason = condition_incomplete_reason(
        control["goal_plus"], manifest.get("condition")
    )
    if condition_reason:
        control["result_incomplete_reason"] = condition_reason
    if not control["goal_plus_controller_closeout"].get("completed"):
        control["result_incomplete_reason"] = (
            "Goal Plus controller closeout failed: "
            + control["goal_plus_controller_closeout"].get("error", "unknown error")
        )
    if closeout_reason is not None:
        control["result_incomplete_reason"] = closeout_reason
    if control["hard_killed"]:
        control["result_incomplete_reason"] = "Goal Plus exceeded hard-kill grace"
    if final is not None and final.get("valid") is not True:
        control["result_incomplete_reason"] = "official final evaluator rejected the artifact"
    return control


def execute_skydiscover(
    manifest: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Run one standalone benchmark through a native SkyDiscover method."""
    budget = manifest["budget"]
    benchmark_root_text = (manifest.get("environment") or {}).get("benchmark_root")
    benchmark_root = Path(benchmark_root_text) if benchmark_root_text else None
    workspace = Path(manifest["workspace"])
    seed = evaluate(workspace, "public", benchmark_root)
    write_json(run_dir / "seed-eval.json", seed)
    setup_calls = seed["budget"]["total_claimed"]

    bin_dir = Path(manifest["environment"]["runtime_bin"])
    sky_executable = bin_dir / "skydiscover-run"
    if not sky_executable.is_file():
        raise FileNotFoundError(
            f"SkyDiscover runtime is missing: {sky_executable}; "
            "run scripts/repro_env.py bootstrap --only skydiscover"
        )

    output = run_dir / "skydiscover-output"
    evaluation_root = run_dir / "skydiscover-evaluations"
    evaluation_root.mkdir()
    environment["BENCH_SKYDISCOVER_TEMPLATE_WORKSPACE"] = str(workspace)
    environment["BENCH_SKYDISCOVER_EVALUATION_ROOT"] = str(evaluation_root)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    algorithm = sky_backend.algorithm_for_method(manifest["method"])
    command = [
        str(sky_executable),
        str(workspace / ARTIFACT_NAME),
        str(ROOT / "adapters/skydiscover_bridge.py"),
        "--config",
        str(run_dir / "skydiscover-config.yaml"),
        "--output",
        str(output),
        "--iterations",
        str(budget["iterations_ceiling"]),
        "--api-base",
        args.api_base,
        "--model",
        args.model,
        "--search",
        algorithm,
        "--log-level",
        "INFO",
    ]
    control = run_controlled(
        command,
        cwd=workspace,
        environment=environment,
        stdin_text=None,
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        wall_time_seconds=budget["wall_time_seconds"],
        hard_kill_grace_seconds=budget["hard_kill_grace_seconds"],
        recorded_command=command_for_manifest(command, args.api_base),
    )

    best = sky_backend.best_candidate(output, Path(ARTIFACT_NAME).suffix)
    best_info = sky_backend.collect_best_info(output)
    if best.is_file():
        shutil.copy2(best, workspace / ARTIFACT_NAME)
        shutil.copy2(best, run_dir / ARTIFACT_NAME)
        final = evaluate(workspace, "final", benchmark_root)
        write_json(run_dir / "final-eval.json", final)
        if final.get("valid") is not True:
            control["result_incomplete_reason"] = (
                "official final evaluator rejected the SkyDiscover artifact"
            )
    else:
        final = seed
        control["result_incomplete_reason"] = (
            "SkyDiscover best candidate was not saved"
        )

    evaluation_workspace_count = sum(
        1 for path in evaluation_root.iterdir() if path.is_dir()
    )
    controller_calls = final["budget"]["total_claimed"]
    total_calls = controller_calls + evaluation_workspace_count
    control["evaluator_calls"] = {
        "total_claimed": total_calls,
        "public_claimed": (
            final["budget"]["public_claimed"] + evaluation_workspace_count
        ),
        "final_claimed": final["budget"]["final_claimed"],
        "setup_claimed_before_t": setup_calls,
        "timed_plus_closeout_claimed": total_calls - setup_calls,
        "coverage": (
            "controller seed/final ledger plus one preserved workspace per "
            "SkyDiscover evaluator call"
        ),
    }
    control["usage"] = {
        "coverage": (
            "missing: SkyDiscover OpenAI-compatible client does not persist "
            "response usage metadata"
        )
    }
    control["skydiscover"] = {
        "algorithm": algorithm,
        "output_dir": str(output),
        "best_info": best_info,
        "requested_concurrency_cap": budget["concurrency"],
        "observed_peak_concurrency": None,
        "evaluation_workspace_count": evaluation_workspace_count,
        "protocol_coverage": (
            "functional smoke: native seed and best test evaluations still "
            "occur inside the timed SkyDiscover runtime"
        ),
        "determinism_coverage": manifest["search_backend"][
            "determinism_coverage"
        ],
    }
    control["telemetry_coverage"] = {
        "evaluator_calls": control["evaluator_calls"]["coverage"],
        "tokens": control["usage"]["coverage"],
        "iterations": (
            "native best_program_info is persisted when a best candidate exists"
        ),
        "actual_concurrency": (
            "missing: runtime does not persist an observed peak"
        ),
    }
    return control


def execute(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().absolute()
    manifest_path = run_dir / "experiment.json"
    manifest = load_json(manifest_path)
    require_prepared_goal_plus_source((manifest.get("environment") or {}).get("goal_plus_source"))
    configure_adapter(
        manifest.get("benchmark_adapter", "heurigym"),
        task_id=manifest.get("benchmark_task_selector"),
        module_name=manifest.get("benchmark_adapter_module"),
    )
    validate_controller_only_method(manifest["method"])
    if manifest["status"] != "prepared":
        raise RuntimeError(f"run is not prepared: {manifest['status']}")
    if args.model != manifest["model"]:
        raise ValueError(f"model mismatch: prepared {manifest['model']}, got {args.model}")
    is_pi = manifest["method"] in {"plain-pi", "goal-plus-pi"}
    pi_provider_id, pi_api, pi_api_key_env = pi_provider_config(args)
    if is_pi:
        prepared_provider = manifest.get("pi_provider") or {
            "id": PI_PROVIDER_ID,
            "api": "openai-responses",
            "api_key_env": PI_API_KEY_ENV,
        }
        runtime_provider = {
            "id": pi_provider_id,
            "api": pi_api,
            "api_key_env": pi_api_key_env,
        }
        if runtime_provider != prepared_provider:
            raise ValueError(
                "Pi provider mismatch: prepared "
                f"{prepared_provider}, got {runtime_provider}"
            )
    if (
        sky_backend.is_method(manifest["method"])
        or manifest["method"] in {"plain-pi", "goal-plus-pi"}
    ) and not args.api_base:
        raise ValueError(f"--api-base is required for {manifest['method']}")
    credential_env = pi_api_key_env if is_pi else "OPENAI_API_KEY"
    if args.api_base and not os.environ.get(credential_env):
        raise RuntimeError(f"{credential_env} is required with --api-base")
    environment = configure_temp_environment(os.environ.copy())
    bin_dir = Path(manifest["environment"]["runtime_bin"])
    environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
    prepend_to_path(environment)
    _configure_pi_worker_sandbox_environment(
        manifest,
        environment,
        pi_api_key_env,
        args.pi_bin,
    )
    manifest["status"] = "running"
    manifest["execution_started_at"] = utc_now()
    write_json(manifest_path, manifest)
    if manifest["method"] in {"plain-codex", "plain-pi"}:
        control = execute_plain(manifest, run_dir, args, environment)
    elif sky_backend.is_method(manifest["method"]):
        control = execute_skydiscover(manifest, run_dir, args, environment)
    else:
        control = execute_goal_plus(manifest, run_dir, args, environment)

    expected_controlled_stop = (
        (control.get("deadline_reached") or control.get("early_stop_triggered"))
        and not control.get("controller_interrupted")
        and not control.get("hard_killed")
        and control.get("returncode") in {0, -signal.SIGTERM, 128 + signal.SIGTERM}
    )
    if (
        control.get("returncode", 0) != 0
        and not expected_controlled_stop
        and not control.get("result_incomplete_reason")
    ):
        control["result_incomplete_reason"] = (
            f"controlled process exited nonzero ({control['returncode']})"
        )
    if control.get("controller_interrupted"):
        control["result_incomplete_reason"] = "campaign controller was interrupted"
    manifest["status"] = (
        "incomplete" if control.get("result_incomplete_reason") else "finished"
    )
    manifest["provider_mode"] = (
        "openai_compatible" if args.api_base else "codex_native_auth"
    )
    version_command = (
        [str(bin_dir / "skydiscover-run"), "--version"]
        if sky_backend.is_method(manifest["method"])
        else [args.pi_bin, "--version"]
        if manifest["method"] in {"plain-pi", "goal-plus-pi"}
        else [args.codex_bin, "--version"]
    )
    version = subprocess.run(
        version_command, capture_output=True, text=True, check=False
    )
    manifest["runner_version"] = (version.stdout or version.stderr).strip() or None
    manifest["execution"] = control
    write_json(manifest_path, manifest)
    print(json.dumps(control, indent=2))
    return 0 if manifest["status"] == "finished" else 2


def seed_smoke(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().absolute()
    manifest = load_json(run_dir / "experiment.json")
    configure_adapter(
        manifest.get("benchmark_adapter", "heurigym"),
        task_id=manifest.get("benchmark_task_selector"),
        module_name=manifest.get("benchmark_adapter_module"),
    )
    validate_controller_only_method(manifest["method"])
    benchmark_root_text = (manifest.get("environment") or {}).get("benchmark_root")
    benchmark_root = Path(benchmark_root_text) if benchmark_root_text else None
    results = []
    for workspace_text in manifest["workspaces"]:
        workspace = Path(workspace_text)
        evaluation = (
            evaluate_with_controller_runtime(
                workspace,
                "public",
                run_dir / "controller-runtime/seed-smoke" / workspace.name,
                benchmark_root,
            )
            if manifest["method"] in {"goal-plus-codex", "goal-plus-pi"}
            else evaluate(workspace, "public", benchmark_root)
        )
        results.append(
            {"workspace": str(workspace), "evaluation": evaluation}
        )
    payload = {"task_id": TASK_ID, "results": results}
    write_json(run_dir / "seed-smoke.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if all(item["evaluation"]["valid"] for item in results) else 2


def repair_closeout(args: argparse.Namespace) -> int:
    """Re-audit an interrupted or conservatively classified Goal Plus run."""
    run_dir = args.run_dir.expanduser().absolute()
    manifest_path = run_dir / "experiment.json"
    manifest = load_json(manifest_path)
    configure_adapter(
        manifest.get("benchmark_adapter", "heurigym"),
        task_id=manifest.get("benchmark_task_selector"),
        module_name=manifest.get("benchmark_adapter_module"),
    )
    validate_controller_only_method(manifest["method"])
    if manifest["method"] not in {"goal-plus-codex", "goal-plus-pi"}:
        raise ValueError("closeout is only valid for Goal Plus runs")
    benchmark_root_text = (manifest.get("environment") or {}).get("benchmark_root")
    benchmark_root = Path(benchmark_root_text) if benchmark_root_text else None
    workspace = Path(manifest["workspace"])
    controller_only = controller_only_official_evaluation(manifest)
    early_stop = goal_plus_early_stop_contract(manifest)
    posthoc_selection = goal_plus_posthoc_selection_contract(manifest)
    prepared_goal_config = manifest.get("goal_plus_config") or {}
    if prepared_goal_config.get("posthoc_selection") != posthoc_selection:
        raise RuntimeError(
            "prepared Goal Plus config does not match the task posthoc-selection contract"
        )
    control = dict(manifest.get("execution") or {})
    control["candidate_session_cleanup_repair"] = close_candidate_sessions(
        workspace, manifest["budget"]["hard_kill_grace_seconds"]
    )
    try:
        with controller_subprocess_environment(
            runtime_bin_dir=Path(manifest["environment"]["runtime_bin"]),
            verifier_tmpdir=run_dir / "controller-runtime/goal-plus",
        ):
            closeout = finalize_goal_plus_search(
                workspace, deterministic_public_gate=controller_only and EVALUATION_MODE == "blind"
            )
    except Exception as exc:
        if not controller_only:
            raise
        closeout = {
            "completed": False,
            "runs": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    control["goal_plus_controller_closeout_repair"] = closeout
    controller_only_closeout_reason = (
        _controller_only_closeout_incomplete_reason(
            closeout, deterministic_public_gate=EVALUATION_MODE == "blind"
        )
        if controller_only
        else None
    )
    if manifest["method"] == "goal-plus-pi" and posthoc_selection is not None:
        controller_only_closeout_reason = _posthoc_prerequisite_incomplete_reason(
            controller_only_closeout_reason,
            control.get("candidate_session_cleanup_repair"),
        )
    final: dict[str, Any] | None = None
    posthoc_result: dict[str, Any] | None = None
    if posthoc_selection is None and (
        controller_only_closeout_reason is None or controller_only
    ):
        final_path = run_dir / "final-eval.json"
        if final_path.is_file():
            final = load_json(final_path)
        else:
            final = evaluate_with_controller_runtime(
                workspace,
                "final",
                run_dir / "controller-runtime/final",
                benchmark_root,
            )
            write_json(final_path, final)
            copy_artifact(workspace / ARTIFACT_NAME, run_dir / ARTIFACT_NAME)
        control.pop("official_evaluation_withheld", None)
    elif controller_only_closeout_reason is None:
        if posthoc_selection is not None:
            try:
                posthoc_result = finalize_posthoc_official_selection(
                    run_dir=run_dir,
                    workspace=workspace,
                    benchmark_root=benchmark_root,
                    closeout=closeout,
                    contract=posthoc_selection,
                    worker_shutdown_verified=True,
                )
            except Exception as exc:
                posthoc_result = {
                    "completed": False,
                    "visible_to_workers": False,
                    "affects_online_search": False,
                    "official_evaluator_calls": 0,
                    "errors": [
                        {
                            "scope": "posthoc_selection",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                }
            control["posthoc_official_selection"] = posthoc_result
            if posthoc_result.get("completed") is True:
                final = load_json(run_dir / "final-eval.json")
                control.pop("official_evaluation_withheld", None)
            else:
                control["official_evaluation_withheld"] = True
    else:
        control["official_evaluation_withheld"] = True
    control["goal_plus"] = collect_goal_plus_state(workspace)
    control["evidence_annotator_usage"] = collect_evidence_annotator_usage(
        workspace
    )
    goal_runs = control["goal_plus"]["runs"]
    process_calls = sum(
        item.get("process_verifier_command_count", 0) for item in goal_runs
    )
    promotion_calls = sum(
        item.get("promotion_verifier_command_count", 0) for item in goal_runs
    )
    setup_calls = (control.get("evaluator_calls") or {}).get(
        "setup_claimed_before_t", 1
    )
    final_claims = 1 if final is not None else 0
    control["evaluator_calls"] = {
        "total_claimed": (
            setup_calls
            + process_calls
            + promotion_calls
            + final_claims
        ),
        "setup_claimed_before_t": setup_calls,
        "process_verifier_commands": process_calls,
        "promotion_verifier_commands": promotion_calls,
        "controller_final_claimed": final_claims,
        "coverage": (
            "seed + Goal Plus verifier command logs + controller posthoc scorer ledger"
            if posthoc_result is not None
            else "seed + Goal Plus verifier command logs + controller final ledger"
        ),
    }
    budget = manifest["budget"]
    early_stop_completion_verified = verified_early_stop_completion(
        control, final, early_stop
    )
    control["early_stop_completion_verified"] = early_stop_completion_verified
    if early_stop_completion_verified:
        control["minimum_lease_completion_waived"] = True
    reason = goal_plus_incomplete_reason(
        control["goal_plus"],
        expected_concurrency=budget["concurrency"],
        minimum_worker_verified_candidates=1,
    )
    if not control["goal_plus_controller_closeout_repair"].get("completed"):
        reason = (
            "Goal Plus controller closeout failed: "
            + control["goal_plus_controller_closeout_repair"].get(
                "error", "unknown error"
            )
        )
    if controller_only_closeout_reason is not None:
        reason = controller_only_closeout_reason
    if posthoc_result is not None and posthoc_result.get("completed") is not True:
        reason = "controller posthoc official selection failed"
    if final is not None and final.get("valid") is not True:
        reason = "official final evaluator rejected the artifact"
    if control.get("hard_killed"):
        reason = "Goal Plus exceeded hard-kill grace"
    if reason:
        control["result_incomplete_reason"] = reason
        manifest["status"] = "incomplete"
    else:
        control.pop("result_incomplete_reason", None)
        manifest["status"] = "finished"
    manifest["execution"] = control
    manifest["closeout_repaired_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(json.dumps(control["goal_plus_controller_closeout_repair"], indent=2))
    return 0 if manifest["status"] == "finished" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--benchmark", choices=tuple(BENCHMARK_ADAPTERS), default="heurigym"
    )
    prepare_parser.add_argument(
        "--task-id",
        help="adapter-specific task selector; persisted in the experiment manifest",
    )
    prepare_parser.add_argument("--method", choices=METHODS, required=True)
    prepare_parser.add_argument(
        "--condition",
        choices=tuple(CONDITIONS),
        help="freeze a B0-B4 ablation condition; omitted commands keep legacy inference",
    )
    prepare_parser.add_argument(
        "--coordination-variant",
        choices=tuple(VARIANT_LIMITATIONS),
        help="optional way0/way1/way2 assertion for a supported condition",
    )
    add_runtime_prepare_arguments(prepare_parser)
    prepare_parser.add_argument("--iterations-ceiling", type=int, default=1)
    prepare_parser.add_argument("--seed", type=int, default=1)
    prepare_parser.add_argument("--llm-max-tokens", type=int, default=8192)
    prepare_parser.add_argument("--run-dir", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--codex-bin", default="codex")
    run_parser.add_argument("--pi-bin", default="pi")
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--api-base")
    run_parser.add_argument("--pi-provider-id", default=PI_PROVIDER_ID)
    run_parser.add_argument("--pi-api", choices=PI_APIS, default="openai-responses")
    run_parser.add_argument("--pi-api-key-env", default=PI_API_KEY_ENV)

    smoke_parser = subparsers.add_parser("seed-smoke")
    smoke_parser.add_argument("--run-dir", type=Path, required=True)

    closeout_parser = subparsers.add_parser("closeout")
    closeout_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "seed-smoke":
        return seed_smoke(args)
    if args.command == "closeout":
        return repair_closeout(args)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
