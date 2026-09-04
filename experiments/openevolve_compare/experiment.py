#!/usr/bin/env python3
"""Run one OpenEvolve example through OE, Codex-only, or Goal Plus hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_artifacts import read_json as load_json  # noqa: E402
from bench_artifacts import utc_now, write_json  # noqa: E402
from bench_goal_plus.codex_provider import (  # noqa: E402
    codex_responses_provider_args,
)
from bench_goal_plus.goal_plus_command import (  # noqa: E402
    goal_plus_command_config,
    goal_plus_entrypoint,
    render_goal_plus_command,
)
from bench_goal_plus.upstreams import upstream_source_path  # noqa: E402
from bench_runtime_paths import configure_temp_environment  # noqa: E402
from adapters.openevolve_examples.adapter import (  # noqa: E402
    describe_task,
    evaluate_workspace,
    git_branch,
    git_commit,
    list_catalog_tasks,
    materialize_workspace,
    resolve_task,
    run_worker,
)
from experiments.openevolve_compare.reporting import write_campaign_report  # noqa: E402
from experiments.backends import skydiscover as sky_backend  # noqa: E402


DEFAULT_ENV_MANIFEST = ROOT / "environment/upstreams.json"
DEFAULT_VENV = ROOT / ".bench-env/venv"
DEFAULT_CHECKOUT_ROOT = ROOT / "third_party"
DEFAULT_RUNS = ROOT / "runs/openevolve-compare"
METHODS = (
    "openevolve",
    "plain-codex",
    "goal-plus-codex",
    "goal-plus-pi",
    *sky_backend.METHODS,
)
DEFAULT_BATCH_METHODS = (
    "openevolve",
    "plain-codex",
    "goal-plus-codex",
    "goal-plus-pi",
)
METHOD_ALIASES = {"goal-plus": "goal-plus-codex"}
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_WALL_TIME_SECONDS = 300
DEFAULT_CONCURRENCY = 2
DEFAULT_REASONING_EFFORT = "high"
BLIND_SELECTION_RULE = "lowest_candidate_id_latest_compliant_iteration"
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
CODEX_SANDBOX = "danger-full-access"
CODEX_PROVIDER_ID = "bench_proxy"
PI_PROVIDER_ID = "bench-openai"
PI_APIS = ("openai-responses", "openai-completions", "anthropic-messages")
PI_API_KEY_ENV = "OPENAI_API_KEY"
ANNOTATOR_PROVIDER_ID = "bench_evidence"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def refresh_campaign_report(run_root: Path) -> None:
    """Refresh derived reports without interrupting an expensive campaign."""
    try:
        write_campaign_report(run_root)
    except Exception as error:
        print(
            f"WARNING: campaign report refresh failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )


def default_run_dir(task_id: str, method: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS / f"{timestamp}-{task_id}-{method}-seed{seed}"


def runtime_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_bin(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def runner_version_command(
    method: str,
    *,
    python: Path,
    codex_bin: str,
    pi_bin: str,
) -> list[str]:
    if method == "openevolve":
        package = "openevolve"
    elif sky_backend.is_method(method):
        package = "skydiscover"
    else:
        if method == "goal-plus-pi":
            return [pi_bin, "--version"]
        return [codex_bin, "--version"]
    return [
        str(python),
        "-c",
        f"import importlib.metadata; print(importlib.metadata.version({package!r}))",
    ]


def checkout_dirty(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(completed.stdout.strip())


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def copy_goal_plus_assets(goal_plus_root: Path, workspace: Path) -> None:
    source = goal_plus_root / ".codex"
    required = (source / "skills", source / "config.example.toml")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    hook_source = next(
        (
            path
            for path in (source / "hooks.example.json", source / "hooks.json")
            if path.is_file()
        ),
        None,
    )
    if hook_source is None:
        raise FileNotFoundError(source / "hooks.example.json")

    target = workspace / ".codex"
    target.mkdir()
    if (source / "agents").is_dir():
        shutil.copytree(source / "agents", target / "agents")
    shutil.copytree(source / "skills", target / "skills")
    shutil.copy2(hook_source, target / "hooks.json")
    shutil.copy2(source / "config.example.toml", target / "config.toml")


def copy_goal_plus_pi_assets(goal_plus_root: Path, workspace: Path) -> None:
    source = goal_plus_root / ".pi"
    required = (
        source / "extensions" / "goal-plus.ts",
        source / "skills" / "goal-plus" / "SKILL.md",
        source / "prompts",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    target = workspace / ".pi"
    target.mkdir()
    shutil.copytree(source / "extensions", target / "extensions")
    shutil.copytree(source / "skills", target / "skills")
    shutil.copytree(source / "prompts", target / "prompts")


def append_unique_lines(path: Path, lines: list[str]) -> None:
    existing = path.read_text().splitlines() if path.is_file() else []
    for line in lines:
        if line not in existing:
            existing.append(line)
    path.write_text("\n".join(existing) + "\n")


def commit_workspace(workspace: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-m", message],
        check=True,
        capture_output=True,
    )
    return git_commit(workspace)


def render_common_task_prompt(
    task_text: str,
    wall_seconds: int,
    closeout_seconds: int,
) -> str:
    exploration_seconds = max(1, wall_seconds - closeout_seconds)
    return (
        f"{task_text.strip()}\n\n"
        "# Common experiment contract\n\n"
        f"- Total outer wall-clock budget: {wall_seconds} seconds.\n"
        f"- Use about {exploration_seconds} seconds for exploration and reserve "
        f"{closeout_seconds} seconds to make the best evaluator-verified artifact ready for "
        "final evaluation.\n"
        "- Use only `python3 evaluate.py` for public task feedback. Do not modify the evaluator "
        "or any controller-owned verifier.\n"
        "- Public evaluator calls are not hard-capped. The experiment records actual calls, "
        "tokens, cost coverage, and wall time after the run.\n"
        "- The wall-clock value is the total budget, not a success criterion. Stop when the "
        "objective is satisfied or the budget is exhausted, leaving or promoting the best "
        "verified artifact for the experiment controller.\n"
    )


def render_blind_task_prompt(
    task_text: str,
    wall_seconds: int,
    closeout_seconds: int,
) -> str:
    exploration_seconds = max(1, wall_seconds - closeout_seconds)
    return (
        f"{task_text.strip()}\n\n"
        "# Common experiment contract\n\n"
        f"- Total outer wall-clock budget: {wall_seconds} seconds.\n"
        f"- Use about {exploration_seconds} seconds for independent exploration and reserve "
        f"{closeout_seconds} seconds to leave the selected artifact ready for controller closeout.\n"
        "- Use only `python3 public_check.py` for public format and structure diagnostics. "
        "The checker provides no behavioral quality signal.\n"
        "- Hidden evaluation is unavailable during exploration and selection. Do not infer, "
        "request, or optimize against hidden results.\n"
        "- The wall-clock value is the total budget, not an outcome criterion. Stop when the "
        "budget is exhausted, leaving the latest committed artifact in each candidate.\n"
    )


def render_goal(
    *,
    task_text: str,
    artifact_name: str,
    metric_name: str,
    metric_direction: str,
    wall_seconds: int,
    closeout_seconds: int,
    concurrency: int,
    worker_host: str,
    worker_model: str,
    artifact_is_directory: bool = False,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    worker_runtime_seconds: int | None = None,
    worker_min_runtime_seconds: int | None = None,
    verifier_timeout_seconds: int = 60,
    coordination_condition: str | None = None,
    search_space_mode: str | None = None,
    shared_dir_enabled: bool = False,
    evaluation_mode: str = "visible",
) -> str:
    """Add the host-native Goal Plus entrypoint and config to the common prompt."""
    exploration_seconds = max(1, wall_seconds - closeout_seconds)
    dispatch_seconds = (
        worker_runtime_seconds
        if worker_runtime_seconds is not None
        else max(30, min(60, exploration_seconds // 3))
    )
    if dispatch_seconds < 1 or dispatch_seconds > exploration_seconds:
        raise ValueError("worker runtime must fit inside the exploration budget")
    if worker_min_runtime_seconds is not None and not (
        1 <= worker_min_runtime_seconds <= dispatch_seconds
    ):
        raise ValueError("worker minimum runtime must fit inside the worker budget")
    if verifier_timeout_seconds < 1:
        raise ValueError("verifier timeout must be positive")
    if search_space_mode not in {None, "observe", "enforce"}:
        raise ValueError(f"unsupported search-space mode: {search_space_mode}")
    if evaluation_mode not in {"visible", "blind"}:
        raise ValueError(f"unsupported evaluation mode: {evaluation_mode}")
    if coordination_condition in {"B3", "B4"} and search_space_mode is None:
        raise ValueError(
            f"{coordination_condition} requires an explicit search-space mode"
        )
    goal_plus_command = render_goal_plus_command(
        worker_host,
        max_parallel=concurrency,
        strategy="agent_guided",
        worker_model=worker_model,
        annotator_model=worker_model,
        workspace_backend="git_worktree",
        promotion_mode=(
            "artifact_only" if evaluation_mode == "blind" else "apply"
        ),
    )
    coordination_text = ""
    if search_space_mode is not None:
        mode_behavior = (
            "Observe mode records plan visibility, reviewer decisions, and Evidence updates "
            "but must not block a candidate when the reviewer would reject it.\n"
            if search_space_mode == "observe"
            else "Enforce mode must reject duplicate or colliding plans and reserve accepted work.\n"
        )
        coordination_text = (
            f"- Ablation condition: `{coordination_condition}`. After creating the Search run "
            "and before dispatching workers, call `search_space_open` exactly once with "
            f"`mode=\"{search_space_mode}\"`, `reviewer_model=\"{worker_model}\"`, and "
            f"`reviewer_reasoning_effort=\"{reasoning_effort}\"`. Every candidate must propose one minimal "
            "AtomicPlan before each material edit or evaluator call and close an accepted plan "
            f"with its verifier result. {mode_behavior}"
        )
    minimum_lease_enforcement = (
        "the Pi pool supervisor automatically resumes the same native session "
        "until the cumulative minimum is satisfied"
        if worker_host == "pi-rpc"
        else "SubagentStop blocks an early worker return"
    )
    initial_launch_contract = (
        "- After the one initial `search_plan_next` call, call `search_start_batch` "
        "once, then open exactly one `pi_search_pool_open` for the returned launch "
        "payloads. Treat the pool's persisted jobs and native session ids as the "
        "only evidence that workers started; use `pi_search_pool_wait_any`, continue "
        "ready candidates while useful, and close the pool before selection.\n"
        if worker_host == "pi-rpc"
        else "- A successful `search_start_agent_session` only allocates a durable "
        "Goal Plus session and returns a launch payload; it does not start a Codex "
        "worker. For every initial candidate, immediately map that payload to an "
        "actual `spawn_agent` call. Only bind the handle returned by the successful "
        "spawn. Do not claim workers are running or call `wait_agent` until all "
        "initial spawn calls have returned real agent handles. If a spawn is "
        "unavailable or fails, leave the run incomplete and report the launch "
        "failure instead of simulating worker progress.\n"
    )
    common_prompt = (
        render_blind_task_prompt(task_text, wall_seconds, closeout_seconds)
        if evaluation_mode == "blind"
        else render_common_task_prompt(task_text, wall_seconds, closeout_seconds)
    )
    edit_surface_limit = (
        "omit `max_file_changes` because this artifact is a directory and multiple "
        "changed files inside it are allowed.\n"
        if artifact_is_directory
        else "allow at most one changed file.\n"
    )
    if evaluation_mode == "blind":
        return (
            f"{goal_plus_command}\n\n"
            f"{common_prompt.rstrip()}\n\n"
            "# Goal Plus configuration\n\n"
            "Use the current workspace and construct a public-format-only Goal Plus search "
            "from the configuration below. The benchmark-owned Pi boundary keeps worker "
            "feedback opaque; Goal Plus remains an unmodified generic runtime and never "
            "receives the official evaluator or official metric.\n\n"
            "- Honor every leading typed command field in the SearchSpec and omit "
            "deprecated `budget.max_candidates`.\n"
            f"- Set `strategy.worker_host=\"{worker_host}\"` and "
            "`strategy.orchestration_mode=\"parallel_loops\"`.\n"
            + "- Set top-level `shared_dir.enabled=false` and "
            "`strategy.config.global_evidence_mode=\"independent\"`.\n"
            + f"- `strategy.worker_budget.max_runtime_seconds={dispatch_seconds}` and "
            "`strategy.worker_budget.on_exceed=\"interrupt\"`; continue the same candidate "
            "lineages while useful work and outer time remain.\n"
            + (
                f"- `strategy.worker_budget.min_runtime_seconds={worker_min_runtime_seconds}` "
                "and `strategy.worker_budget.min_verifier_runs=1`; preserve this minimum "
                f"AutoResearch lease so {minimum_lease_enforcement}. Do not place either "
                "field in `strategy.config`.\n"
                if worker_min_runtime_seconds is not None
                else ""
            )
            + "- Keep `strategy.worker_launch.model` aligned with "
            "`command_config.workers` and set "
            f"`strategy.worker_launch.reasoning_effort=\"{reasoning_effort}\"`.\n"
            f"{initial_launch_contract}"
            f"{coordination_text}"
            f"- Metric: `{metric_name}` with direction `{metric_direction}`; it is a public "
            "format gate only. No official evaluation value may enter this Search run.\n"
            "- Process verifier: `python3 public_check.py`, role `validity_gate`, feedback "
            f"policy `final_only`, timeout {verifier_timeout_seconds} seconds.\n"
            "- Promotion verifier: the same command, role `promotion_gate`, feedback policy "
            f"`final_only`, timeout {verifier_timeout_seconds} seconds.\n"
            "- The benchmark-owned Pi shim makes verifier feedback, worker context history, "
            "`results.tsv`, Git administration, and Global Evidence opaque to workers. "
            "Workers must not request or infer those values.\n"
            "- Each worker must commit its latest artifact and submit its own final process "
            "verifier result. Do not run duplicate parent-side process verification when "
            "matching durable evidence already exists.\n"
            f"- Edit surface: allow only `{artifact_name}`; deny `public_check.py`, "
            "`task.json`, `TASK.md`, `AGENTS.md`, and `GOAL.md`; "
            f"{edit_surface_limit}"
            "- Workspace: use `source_path=\".\"`; backend and promotion mode come "
            "from the typed command config.\n"
            "- Constraints: no network; preserve all controller-owned public task files.\n"
            f"- `strategy.config.closeout_reserve_seconds={closeout_seconds}` so host "
            "supervisors stop worker continuation before final completion work.\n"
            f"- Outer budget: {wall_seconds} seconds total, with about {exploration_seconds} "
            f"seconds for exploration and {closeout_seconds} seconds reserved for completion. "
            "Treat `GOAL_PLUS_OUTER_DEADLINE_AT` as the authoritative upper deadline.\n"
            "- Predeclare the controller-checked selection rule: among candidates with a publicly compliant "
            "`process_passed` iteration, choose the lowest candidate ID and that candidate's latest "
            "such commit. Compliance is determined only by the public format checker and is "
            "never disclosed to workers. Goal Plus may settle only on this public gate; after "
            "closeout the benchmark controller independently rejects any selection that does "
            "not match this rule before invoking the official evaluator. Promote the selected "
            "commit, complete the full goal audit, and write the final Goal Plus report.\n"
        )
    return (
        f"{goal_plus_command}\n\n"
        f"{common_prompt.rstrip()}\n\n"
        "# Goal Plus configuration\n\n"
        "Use the current workspace and construct the verifier-backed Goal Plus search from "
        "the configuration below. Goal Plus owns intake, triage, SearchSpec freezing, candidate "
        "workspaces, selection, promotion, and final reporting.\n\n"
        "- Honor every leading typed command field in the SearchSpec and omit "
        "deprecated `budget.max_candidates`.\n"
        f"- Set `strategy.worker_host=\"{worker_host}\"` and "
        "`strategy.orchestration_mode=\"parallel_loops\"`.\n"
        + (
            "- Set top-level `shared_dir.enabled=true`.\n"
            if shared_dir_enabled
            else ""
        )
        + f"- `strategy.worker_budget.max_runtime_seconds={dispatch_seconds}` and "
        "`strategy.worker_budget.on_exceed=\"interrupt\"`; continue the same candidate "
        "lineages while useful work and outer time remain.\n"
        + (
            f"- `strategy.worker_budget.min_runtime_seconds={worker_min_runtime_seconds}` "
            "and `strategy.worker_budget.min_verifier_runs=1`; preserve this minimum "
            f"AutoResearch lease so {minimum_lease_enforcement}. Do not place either "
            "field in `strategy.config`.\n"
            if worker_min_runtime_seconds is not None
            else ""
        )
        + "- Keep `strategy.worker_launch.model` aligned with "
        "`command_config.workers` and set "
        f"`strategy.worker_launch.reasoning_effort=\"{reasoning_effort}\"`.\n"
        f"{initial_launch_contract}"
        f"{coordination_text}"
        f"- Metric: `{metric_name}` with direction `{metric_direction}`.\n"
        "- Process verifier: `python3 .goal-plus-verifiers/primary_metric.py`, role "
        "`ranking_signal`, feedback policy `summary_only`, timeout "
        f"{verifier_timeout_seconds} seconds.\n"
        "- Promotion verifier: the same command, role `promotion_gate`, feedback policy "
        f"`final_only`, timeout {verifier_timeout_seconds} seconds. The verifier is "
        "controller-owned and immutable.\n"
        "- Each candidate worker must submit its own final process verifier result. After a "
        "worker returns with that durable result, do not run a duplicate parent-side process "
        "verification for the same candidate; wait for all workers, then select and let the "
        "promotion verifier perform the final gate.\n"
        f"- Edit surface: allow only `{artifact_name}`; deny `evaluate.py`, "
        "`.goal-plus-verifiers/**`, `task.json`, `TASK.md`, `AGENTS.md`, and `GOAL.md`; "
        f"{edit_surface_limit}"
        "- Workspace: use `source_path=\".\"`; backend and promotion mode come from "
        "the typed command config.\n"
        "- Constraints: no network; preserve the artifact's controller-checked fixed regions.\n"
        f"- `strategy.config.closeout_reserve_seconds={closeout_seconds}` so host "
        "supervisors stop worker continuation before final completion work.\n"
        f"- Outer budget: {wall_seconds} seconds total, with about {exploration_seconds} "
        f"seconds for exploration and {closeout_seconds} seconds reserved for completion. "
        "Treat `GOAL_PLUS_OUTER_DEADLINE_AT` as the authoritative upper deadline.\n"
        f"- Promotion rule: select the valid verifier-backed candidate with the best "
        f"`{metric_name}`, promote it, complete the full goal audit, and write the "
        "final Goal Plus report.\n"
    )


def render_plain_prompt(
    task_text: str,
    wall_seconds: int,
    closeout_seconds: int,
    evaluation_mode: str = "visible",
) -> str:
    if evaluation_mode == "blind":
        return render_blind_task_prompt(task_text, wall_seconds, closeout_seconds)
    if evaluation_mode != "visible":
        raise ValueError(f"unsupported evaluation mode: {evaluation_mode}")
    return render_common_task_prompt(task_text, wall_seconds, closeout_seconds)


def codex_provider_args(api_base: str) -> list[str]:
    return codex_responses_provider_args(api_base, provider_id=CODEX_PROVIDER_ID)


def codex_model_args(
    model: str,
    api_base: str | None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> list[str]:
    """Pin model/reasoning identically for native and explicit providers."""
    args = [
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
    ]
    if api_base:
        args.extend(codex_provider_args(api_base))
    args.extend(["--model", model])
    return args


def codex_execution_args() -> list[str]:
    """Run benchmark Codex lanes non-interactively with host-level access."""
    return [
        "--sandbox",
        CODEX_SANDBOX,
        "--config",
        'approval_policy="never"',
    ]


def codex_goal_plus_mcp_args(extra_env_vars: tuple[str, ...] = ()) -> list[str]:
    """Register and non-interactively approve Goal Plus for `codex exec`."""
    default_env_vars = (
        "CODEX_HOME",
        "OPENAI_API_KEY",
        "SFORGE_AGENT_API_KEY",
        "GOAL_PLUS_OUTER_DEADLINE_AT",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_MODEL",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_REASONING_EFFORT",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_BASE_URL",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_PROVIDER_ID",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_PROVIDER_NAME",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_API_KEY_ENV",
        "GOAL_PLUS_EVIDENCE_ANNOTATOR_WIRE_API",
    )
    env_vars = list(dict.fromkeys((*default_env_vars, *extra_env_vars)))
    return [
        "--config",
        'mcp_servers.goal-plus.command="goal-plus"',
        "--config",
        'mcp_servers.goal-plus.args=["--root", ".gp"]',
        "--config",
        f"mcp_servers.goal-plus.env_vars={json.dumps(env_vars)}",
        "--config",
        "mcp_servers.goal-plus.startup_timeout_sec=10",
        "--config",
        "mcp_servers.goal-plus.tool_timeout_sec=300",
        "--config",
        'mcp_servers.goal-plus.default_tools_approval_mode="approve"',
        "--config",
        "mcp_servers.goal-plus.enabled=true",
    ]


def configure_isolated_codex_home(
    environment: dict[str, str], run_dir: Path
) -> Path:
    """Load project hooks without inheriting the user's Codex configuration."""
    codex_home = run_dir / "controller-runtime" / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=False)
    environment["CODEX_HOME"] = str(codex_home)
    return codex_home


def configure_evidence_annotator_environment(
    environment: dict[str, str],
    *,
    model: str,
    reasoning_effort: str,
    api_base: str | None,
) -> None:
    environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_MODEL"] = model
    environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_REASONING_EFFORT"] = reasoning_effort
    if api_base:
        environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_BASE_URL"] = api_base
        environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_PROVIDER_ID"] = ANNOTATOR_PROVIDER_ID
        environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_PROVIDER_NAME"] = (
            "Benchmark Evidence provider"
        )
        environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_API_KEY_ENV"] = "OPENAI_API_KEY"
        environment["GOAL_PLUS_EVIDENCE_ANNOTATOR_WIRE_API"] = "responses"


def write_pi_models_config(
    target: Path,
    *,
    api_base: str,
    model: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    provider_id: str = PI_PROVIDER_ID,
    api: str = "openai-responses",
    api_key_env: str = PI_API_KEY_ENV,
) -> None:
    if not provider_id or "/" in provider_id:
        raise ValueError("Pi provider id must be non-empty and cannot contain '/'")
    if api not in PI_APIS:
        raise ValueError(f"unsupported Pi API: {api}")
    if not api_key_env or not api_key_env.replace("_", "A").isalnum():
        raise ValueError("Pi API key environment variable name is invalid")
    target.mkdir(parents=True, exist_ok=True)
    model_config = {
        "id": model,
        "name": f"{model} benchmark proxy",
        "reasoning": True,
        "input": ["text"],
        "contextWindow": 200000 if api == "anthropic-messages" else 272000,
        "maxTokens": 131072 if api == "anthropic-messages" else 32000,
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
    }
    if api != "anthropic-messages":
        model_config["thinkingLevelMap"] = {reasoning_effort: reasoning_effort}
        model_config["compat"] = {
            "supportsDeveloperRole": provider_id.casefold() != "deepseek",
            "supportsReasoningEffort": True,
        }
    payload = {
        "providers": {
            provider_id: {
                "baseUrl": api_base,
                "api": api,
                "apiKey": f"${api_key_env}",
                "authHeader": True,
                "models": [model_config],
            }
        }
    }
    models_path = target / "models.json"
    models_path.write_text(json.dumps(payload, indent=2) + "\n")
    models_path.chmod(0o600)


def write_openevolve_config(
    source: Path,
    target: Path,
    *,
    concurrency: int,
    iterations_ceiling: int,
    seed: int,
    reasoning_effort: str,
) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is missing; run scripts/repro_env.py bootstrap"
        ) from error

    payload = yaml.safe_load(source.read_text())
    payload["max_iterations"] = iterations_ceiling
    payload["random_seed"] = seed
    payload.setdefault("database", {})["random_seed"] = seed
    payload.setdefault("evaluator", {})["parallel_evaluations"] = concurrency
    llm = payload.setdefault("llm", {})
    llm.pop("api_key", None)
    llm["secondary_model_weight"] = 0.0
    llm["reasoning_effort"] = reasoning_effort
    target.write_text(yaml.safe_dump(payload, sort_keys=False))


def prepare(args: argparse.Namespace) -> int:
    method = canonical_method(args.method)
    is_sky = sky_backend.is_method(method)
    if args.wall_time_seconds <= args.soft_closeout_seconds:
        raise ValueError("wall time must be greater than the soft closeout reserve")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.hard_kill_grace_seconds < 1:
        raise ValueError("hard-kill grace must be positive")
    if args.iterations_ceiling < 1:
        raise ValueError("iterations ceiling must be positive")

    environment = load_json(args.environment_manifest)
    checkout_root = args.checkout_root.expanduser().absolute()
    upstreams = environment["upstreams"]
    openevolve_root = upstream_source_path(
        checkout_root,
        upstreams["openevolve"],
        upstream_key="openevolve",
    )
    goal_plus_root = upstream_source_path(
        checkout_root,
        upstreams["goal_plus"],
        upstream_key="goal_plus",
    )
    skydiscover_root = upstream_source_path(
        checkout_root,
        upstreams["skydiscover"],
        upstream_key="skydiscover",
    )
    managed_checkouts = [
        ("openevolve", openevolve_root),
        ("goal_plus", goal_plus_root),
    ]
    if is_sky:
        managed_checkouts.append(("skydiscover", skydiscover_root))
    for name, path in managed_checkouts:
        expected_branch = upstreams[name]["tracking_branch"]
        actual_branch = git_branch(path)
        if actual_branch != expected_branch:
            raise RuntimeError(
                f"{name} branch mismatch: expected {expected_branch}, got {actual_branch}"
            )
        if checkout_dirty(path):
            raise RuntimeError(f"managed {name} checkout has local changes: {path}")
    python = runtime_python(args.venv.expanduser().absolute())
    if not python.is_file():
        raise FileNotFoundError(f"reproducible runtime is missing: {python}")

    task = resolve_task(args.task_id, openevolve_root)
    run_dir = (
        args.run_dir or default_run_dir(args.task_id, method, args.seed)
    ).absolute()
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = run_dir / "openevolve-config.yaml"
    write_openevolve_config(
        task.config,
        run_config,
        concurrency=args.concurrency,
        iterations_ceiling=args.iterations_ceiling,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
    )
    run_task = replace(task, config=run_config)
    description = describe_task(run_task, python)
    workspaces: list[Path] = []
    workspace_commits: list[str] = []
    goal_plus_config: dict[str, Any] | None = None
    prompt_contract: dict[str, Any] | None = None

    if method == "plain-codex":
        for lane_index in range(args.concurrency):
            workspace = run_dir / "workspaces" / f"lane-{lane_index:02d}"
            materialized = materialize_workspace(
                run_task,
                workspace,
                python,
                max_evaluator_calls=None,
                reserved_final_calls=1,
                description=description,
                controller_runtime_dir=(
                    run_dir / "controller-runtime" / f"lane-{lane_index:02d}"
                ),
            )
            workspaces.append(workspace)
            workspace_commits.append(materialized["workspace_commit"])
        task_text = (workspaces[0] / "TASK.md").read_text()
        common_prompt = render_common_task_prompt(
            task_text,
            args.wall_time_seconds,
            args.soft_closeout_seconds,
        )
        prompt_contract = {
            "mode": "plain_codex_common_prompt",
            "common_prompt_sha256": sha256_text(common_prompt),
            "transform": "identity",
        }
    elif method in {"goal-plus-codex", "goal-plus-pi"}:
        workspace = run_dir / "workspace"
        materialized = materialize_workspace(
            run_task,
            workspace,
            python,
            max_evaluator_calls=None,
            reserved_final_calls=1,
            description=description,
            controller_runtime_dir=run_dir / "controller-runtime",
        )
        if method == "goal-plus-codex":
            copy_goal_plus_assets(goal_plus_root, workspace)
            append_unique_lines(workspace / ".gitignore", [".gp/", ".codex-log/"])
            worker_host = "codex"
            worker_model = args.model
        else:
            copy_goal_plus_pi_assets(goal_plus_root, workspace)
            append_unique_lines(workspace / ".gitignore", [".gp/", ".pi-log/"])
            worker_host = "pi-rpc"
            worker_model = f"{PI_PROVIDER_ID}/{args.model}"
        task_text = (workspace / "TASK.md").read_text()
        goal = render_goal(
            task_text=task_text,
            artifact_name=task.artifact_name,
            metric_name=description["evaluation"]["primary_metric"],
            metric_direction=description["evaluation"]["direction"],
            wall_seconds=args.wall_time_seconds,
            closeout_seconds=args.soft_closeout_seconds,
            concurrency=args.concurrency,
            worker_host=worker_host,
            worker_model=worker_model,
            reasoning_effort=args.reasoning_effort,
        )
        (workspace / "GOAL.md").write_text(goal)
        workspaces.append(workspace)
        workspace_commits.append(
            commit_workspace(workspace, "install managed Goal Plus host assets")
        )
        common_prompt = render_common_task_prompt(
            task_text,
            args.wall_time_seconds,
            args.soft_closeout_seconds,
        )
        prompt_contract = {
            "mode": "natural_goal_plus_entry",
            "common_prompt_sha256": sha256_text(common_prompt),
            "transform": (
                f"{goal_plus_entrypoint(worker_host)} typed config prefix plus "
                "Goal Plus SearchSpec-only configuration suffix"
            ),
            "goal_prompt_sha256": sha256_text(goal),
        }
        goal_plus_config = {
            "entrypoint": goal_plus_entrypoint(worker_host),
            "command_config": goal_plus_command_config(
                max_parallel=args.concurrency,
                strategy="agent_guided",
                worker_model=worker_model,
                annotator_model=worker_model,
                workspace_backend="git_worktree",
                promotion_mode="apply",
            ),
            "worker_host": worker_host,
            "worker_model": worker_model,
            "base_model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "metric_name": description["evaluation"]["primary_metric"],
            "metric_direction": description["evaluation"]["direction"],
            "artifact_name": task.artifact_name,
            "state_at_t0": "absent; Goal Plus intake starts from the natural prompt",
        }
    elif is_sky:
        workspace = run_dir / "workspace"
        materialized = materialize_workspace(
            run_task,
            workspace,
            python,
            max_evaluator_calls=None,
            reserved_final_calls=1,
            description=description,
            controller_runtime_dir=run_dir / "controller-runtime",
        )
        workspaces.append(workspace)
        workspace_commits.append(materialized["workspace_commit"])
        task_text = (workspace / "TASK.md").read_text()
        algorithm = sky_backend.algorithm_for_method(method)
        sky_config_path = run_dir / "skydiscover-config.yaml"
        sky_backend.write_config(
            sky_config_path,
            algorithm=algorithm,
            task_prompt=task_text,
            file_suffix=task.initial_program.suffix,
            evaluator_timeout_seconds=int(
                description["evaluation"]["timeout_seconds"]
            ),
            concurrency=args.concurrency,
            iterations_ceiling=args.iterations_ceiling,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
        )
        prompt_contract = {
            "mode": "skydiscover_native_context",
            "task_prompt_sha256": sha256_text(task_text),
            "backend_control_prompt": (
                "SkyDiscover native context builder adds search history and "
                "mutation instructions to the fixed task prompt"
            ),
            "backend_config_sha256": sha256_file(sky_config_path),
        }
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "prepared_at": utc_now(),
        "method": method,
        "task_id": args.task_id,
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "budget": {
            "wall_time_seconds": args.wall_time_seconds,
            "concurrency": args.concurrency,
            "soft_closeout_seconds": args.soft_closeout_seconds,
            "hard_kill_grace_seconds": args.hard_kill_grace_seconds,
            "iterations_ceiling": args.iterations_ceiling,
            "evaluator_call_cap": None,
        },
        "task": {
            "artifact_name": task.artifact_name,
            "primary_metric": description["evaluation"]["primary_metric"],
            "direction": description["evaluation"]["direction"],
            "execution_profile": task.profile,
            "upstream_tracking_branch": upstreams["openevolve"][
                "tracking_branch"
            ],
            "upstream_commit": task.upstream_commit,
            "initial_program": str(task.initial_program),
            "evaluator": str(task.evaluator),
            "config": str(run_config),
            "upstream_config": str(task.config),
            "backend_config": (
                str(run_dir / "skydiscover-config.yaml") if is_sky else None
            ),
            "initial_program_sha256": sha256_file(task.initial_program),
            "evaluator_sha256": sha256_file(task.evaluator),
        },
        "environment": {
            "manifest": str(args.environment_manifest.absolute()),
            "runtime_python": str(python),
            "openevolve_root": str(openevolve_root),
            "openevolve_tracking_branch": upstreams["openevolve"][
                "tracking_branch"
            ],
            "openevolve_branch": git_branch(openevolve_root),
            "openevolve_commit": git_commit(openevolve_root),
            "goal_plus_root": str(goal_plus_root),
            "goal_plus_tracking_branch": upstreams["goal_plus"][
                "tracking_branch"
            ],
            "goal_plus_branch": git_branch(goal_plus_root),
            "goal_plus_commit": git_commit(goal_plus_root),
            **(
                {
                    "skydiscover_root": str(skydiscover_root),
                    "skydiscover_tracking_branch": upstreams["skydiscover"][
                        "tracking_branch"
                    ],
                    "skydiscover_branch": git_branch(skydiscover_root),
                    "skydiscover_commit": git_commit(skydiscover_root),
                }
                if is_sky
                else {}
            ),
        },
        "workspace": str(workspaces[0]) if len(workspaces) == 1 else None,
        "workspace_commit": (
            workspace_commits[0] if len(workspace_commits) == 1 else None
        ),
        "workspaces": [str(path) for path in workspaces],
        "workspace_commits": workspace_commits,
        "prompt_contract": prompt_contract,
        "goal_plus_config": goal_plus_config,
        "search_backend": (
            {
                "family": "skydiscover",
                "algorithm": algorithm,
                "config": str(run_dir / "skydiscover-config.yaml"),
                "config_sha256": sha256_file(run_dir / "skydiscover-config.yaml"),
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
            if is_sky
            else None
        ),
        "codex_sandbox": (
            CODEX_SANDBOX
            if method in {"plain-codex", "goal-plus-codex"}
            else None
        ),
        "codex_approval_policy": (
            "never" if method in {"plain-codex", "goal-plus-codex"} else None
        ),
        "secret_policy": (
            "credentials are inherited from the process environment and never serialized"
        ),
    }
    write_json(run_dir / "experiment.json", manifest)
    print(run_dir)
    return 0


def prepare_batch(args: argparse.Namespace) -> int:
    """Prepare every task/method cell in a reusable experiment campaign."""
    run_root = args.run_root.expanduser().absolute()
    methods = list(dict.fromkeys(canonical_method(item) for item in args.methods))
    tasks = list_catalog_tasks(args.task_set)
    run_root.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []

    for task in tasks:
        task_id = task["task_id"]
        for method in methods:
            run_dir = run_root / task_id / method
            prepare_args = argparse.Namespace(
                method=method,
                task_id=task_id,
                seed=args.seed,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                wall_time_seconds=args.wall_time_seconds,
                concurrency=args.concurrency,
                soft_closeout_seconds=args.soft_closeout_seconds,
                hard_kill_grace_seconds=args.hard_kill_grace_seconds,
                iterations_ceiling=args.iterations_ceiling,
                run_dir=run_dir,
                environment_manifest=args.environment_manifest,
                checkout_root=args.checkout_root,
                venv=args.venv,
            )
            entry = {
                "task_id": task_id,
                "method": method,
                "run_dir": str(run_dir),
                "prepared": False,
                "error": None,
            }
            try:
                prepare(prepare_args)
                entry["prepared"] = True
            except Exception as error:  # Keep the rest of the batch reproducible.
                entry["error"] = f"{type(error).__name__}: {error}"
            entries.append(entry)

    prepared_count = sum(bool(item["prepared"]) for item in entries)
    campaign = {
        "schema_version": 1,
        "prepared_at": utc_now(),
        "task_set": args.task_set,
        "task_count": len(tasks),
        "methods": methods,
        "cell_count": len(entries),
        "prepared_count": prepared_count,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "budget": {
            "wall_time_seconds": args.wall_time_seconds,
            "concurrency": args.concurrency,
            "soft_closeout_seconds": args.soft_closeout_seconds,
            "hard_kill_grace_seconds": args.hard_kill_grace_seconds,
            "iterations_ceiling": args.iterations_ceiling,
        },
        "entries": entries,
        "secret_policy": (
            "credentials are inherited only by run-batch and are never serialized"
        ),
    }
    write_json(run_root / "campaign.json", campaign)
    refresh_campaign_report(run_root)
    print(json.dumps(campaign, indent=2))
    return 0 if prepared_count == len(entries) else 2


def run_batch(args: argparse.Namespace) -> int:
    """Execute prepared campaign cells sequentially and preserve every result."""
    campaign_path = args.campaign.expanduser().absolute()
    if campaign_path.is_dir():
        campaign_path = campaign_path / "campaign.json"
    campaign = load_json(campaign_path)
    selected_methods = {
        canonical_method(item) for item in (args.methods or campaign["methods"])
    }
    model = args.model or campaign["model"]
    results_path = campaign_path.parent / "campaign-results.json"
    results: list[dict[str, Any]] = []
    if results_path.is_file():
        existing = load_json(results_path)
        results = list(existing.get("results") or [])
    recorded_cells = {
        (item["task_id"], item["method"])
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("task_id"), str)
        and isinstance(item.get("method"), str)
    }

    for item in campaign["entries"]:
        if item["method"] not in selected_methods:
            continue
        cell = (item["task_id"], item["method"])
        if cell in recorded_cells:
            continue
        result = {
            "task_id": item["task_id"],
            "method": item["method"],
            "run_dir": item["run_dir"],
            "started_at": utc_now(),
            "returncode": None,
            "status": "not_run",
            "error": None,
        }
        if not item["prepared"]:
            result["status"] = "prepare_failed"
            result["error"] = item["error"]
        else:
            try:
                result["returncode"] = execute(
                    argparse.Namespace(
                        run_dir=Path(item["run_dir"]),
                        venv=args.venv,
                        codex_bin=args.codex_bin,
                        pi_bin=args.pi_bin,
                        model=model,
                        api_base=args.api_base,
                    )
                )
                result["status"] = (
                    "finished" if result["returncode"] == 0 else "incomplete"
                )
            except Exception as error:  # Preserve the remaining campaign cells.
                result["status"] = "error"
                result["error"] = f"{type(error).__name__}: {error}"
        result["finished_at"] = utc_now()
        results.append(result)
        write_json(
            results_path,
            {
                "schema_version": 1,
                "campaign": str(campaign_path),
                "updated_at": utc_now(),
                "model": model,
                "methods": sorted(selected_methods),
                "results": results,
                "secret_policy": "provider credentials and API base are not serialized",
            },
        )
        refresh_campaign_report(campaign_path.parent)
        recorded_cells.add(cell)
        if args.fail_fast and result["status"] != "finished":
            break

    selected_results = [
        item for item in results if item["method"] in selected_methods
    ]
    failed = [item for item in selected_results if item["status"] != "finished"]
    print(json.dumps({"results": results}, indent=2))
    return 0 if selected_results and not failed else 2


def send_soft_stop(process: subprocess.Popen[str]) -> None:
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        pass


def send_hard_stop(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def parse_codex_events(path: Path) -> dict[str, Any]:
    from bench_goal_plus.agent_events import parse_codex_event_file

    return parse_codex_event_file(path)


def parse_pi_events(path: Path) -> dict[str, Any]:
    usage = {
        "assistant_messages": 0,
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "cost_total": 0.0,
    }
    event_count = 0
    terminal_events = 0
    models: set[str] = set()
    providers: set[str] = set()
    for line in path.read_text().splitlines() if path.is_file() else []:
        if not line.strip():
            continue
        event_count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "agent_end":
            terminal_events += 1
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        message_usage = message.get("usage")
        if not isinstance(message_usage, dict):
            continue
        usage["assistant_messages"] += 1
        for source, target in (
            ("input", "input"),
            ("output", "output"),
            ("cacheRead", "cache_read"),
            ("cacheWrite", "cache_write"),
        ):
            value = message_usage.get(source)
            if isinstance(value, (int, float)):
                usage[target] += int(value)
        cost = message_usage.get("cost")
        if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
            usage["cost_total"] += float(cost["total"])
        if isinstance(message.get("model"), str):
            models.add(message["model"])
        if isinstance(message.get("provider"), str):
            providers.add(message["provider"])
    usage["cost_total"] = round(float(usage["cost_total"]), 12)
    return {
        "event_count": event_count,
        "terminal_events": terminal_events,
        "usage": usage,
        "models": sorted(models),
        "providers": sorted(providers),
        "coverage": "top-level Pi JSON usage plus Goal Plus worker metadata under workspace/.gp",
    }


def collect_evidence_annotator_usage(workspace: Path) -> dict[str, Any]:
    totals: dict[str, int | float] = {}
    tasks = 0
    attempts = 0
    states: dict[str, int] = {}
    for path in sorted(
        (workspace / ".gp" / "runs").glob(
            "run_*/candidates/*/evidence-annotations/iteration-*.json"
        )
    ):
        task = load_json(path)
        tasks += 1
        attempts += int(task.get("attempts") or 0)
        state = str(task.get("state") or "unknown")
        states[state] = states.get(state, 0) + 1
        usage = task.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return {
        **totals,
        "tasks": tasks,
        "attempts": attempts,
        "states": states,
        "coverage": "persisted Goal Plus Evidence annotator turns",
    }


def _plan_footprint(plan: dict[str, Any]) -> set[str]:
    footprint = (plan.get("proposal") or {}).get("footprint") or {}
    return {
        f"{view}:{value}"
        for view, values in footprint.items()
        if isinstance(values, list)
        for value in values
        if isinstance(value, str)
    }


def collect_search_space_state(run_dir: Path) -> dict[str, Any]:
    """Collect coordination metrics without invoking or mutating the runtime."""
    roots = [run_dir / "search-space", run_dir / "space-experiment"]
    root = next((path for path in roots if (path / "config.json").is_file()), None)
    if root is None:
        return {"exists": False}
    config = load_json(root / "config.json")
    state = load_json(root / "state.json") if (root / "state.json").is_file() else {}
    plans = [load_json(path) for path in sorted((root / "plans").glob("*.json"))]
    events = [load_json(path) for path in sorted((root / "events").glob("*.json"))]
    plan_counts: dict[str, int] = {}
    for plan in plans:
        status = str(plan.get("status", "unknown"))
        plan_counts[status] = plan_counts.get(status, 0) + 1

    event_candidates = {
        event.get("event_id"): event.get("candidate_id")
        for event in events
        if isinstance(event.get("event_id"), str)
    }
    evidence_references = 0
    cross_lineage_references = 0
    for plan in plans:
        proposal = plan.get("proposal") or {}
        refs = [
            ref
            for key in ("evidence_refs", "relation_evidence_refs")
            for ref in proposal.get(key, [])
            if isinstance(ref, str)
        ]
        evidence_references += len(refs)
        cross_lineage_references += sum(
            event_candidates.get(ref) not in {None, plan.get("candidate_id")}
            for ref in refs
        )

    overlaps: list[float] = []
    for index, left in enumerate(plans):
        left_footprint = _plan_footprint(left)
        if not left_footprint:
            continue
        for right in plans[index + 1 :]:
            if left.get("candidate_id") == right.get("candidate_id"):
                continue
            right_footprint = _plan_footprint(right)
            union = left_footprint | right_footprint
            if union:
                overlaps.append(len(left_footprint & right_footprint) / len(union))

    reviewed = [plan for plan in plans if isinstance(plan.get("review"), dict)]
    duplicate_reviews = [
        plan
        for plan in reviewed
        if (plan.get("review") or {}).get("decision") == "reject"
    ]
    reviewer_usage: dict[str, int | float] = {}
    for source in [
        *(plan.get("reviewer_usage") or {} for plan in plans),
        state.get("schema_reviewer_usage") or {},
    ]:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                reviewer_usage[key] = reviewer_usage.get(key, 0) + value
    return {
        "exists": True,
        "root": str(root),
        "mode": config.get("mode"),
        "protocol_version": config.get("protocol_version"),
        "reviewer_model": config.get("reviewer_model"),
        "reviewer_reasoning_effort": config.get("reviewer_reasoning_effort"),
        "reviewer_timeout_seconds": config.get("reviewer_timeout_seconds"),
        "reviewer_usage": reviewer_usage,
        "plans_total": len(plans),
        "plan_counts": plan_counts,
        "reviewed_plans": len(reviewed),
        "semantic_duplicate_reviews": len(duplicate_reviews),
        "semantic_duplicate_probability": (
            len(duplicate_reviews) / len(reviewed) if reviewed else None
        ),
        "enforced_rejections": sum(plan.get("status") == "rejected" for plan in plans),
        "evidence_event_count": len(events),
        "evidence_revision": state.get("evidence_revision"),
        "evidence_references": evidence_references,
        "cross_lineage_evidence_references": cross_lineage_references,
        "cross_lineage_evidence_reuse_rate": (
            cross_lineage_references / evidence_references
            if evidence_references
            else None
        ),
        "cross_lineage_footprint_pairs": len(overlaps),
        "mean_cross_lineage_footprint_jaccard": (
            sum(overlaps) / len(overlaps) if overlaps else None
        ),
        "shared_tool_reuse": None,
        "shared_tool_reuse_coverage": (
            "not yet attributable: tool provenance is not persisted in Search Evidence"
        ),
    }


def collect_goal_plus_state(workspace: Path) -> dict[str, Any]:
    root = workspace / ".gp"
    pi_pool_jobs_by_run: dict[str, list[dict[str, Any]]] = {}
    for job_path in sorted(
        (root / "host-pools" / "pi").glob("pool_*/jobs/job_*/job.json")
    ):
        job = load_json(job_path)
        run_id = job.get("run_id")
        if not isinstance(run_id, str):
            continue
        result_path = job_path.parent / "result.json"
        result = load_json(result_path) if result_path.is_file() else {}
        pi_pool_jobs_by_run.setdefault(run_id, []).append(
            {
                "job_id": job.get("job_id"),
                "candidate_id": job.get("candidate_id"),
                "status": job.get("status"),
                "lease": result.get("lease") if isinstance(result, dict) else None,
            }
        )
    goals = []
    for path in sorted((root / "goal-plus").glob("gp_*/goal.json")):
        payload = load_json(path)
        goals.append(
            {
                "goal_plus_id": payload.get("goal_plus_id"),
                "status": payload.get("status"),
                "phase": payload.get("phase"),
                "goal_revision": payload.get("goal_revision"),
                "linked_run_id": (payload.get("linked_search") or {}).get("run_id"),
            }
        )
    runs = []
    for path in sorted((root / "runs").glob("run_*/run.json")):
        payload = load_json(path)
        run_dir = path.parent
        metric_direction = None
        worker_host = None
        worker_budget = None
        frozen_spec_id = payload.get("frozen_spec_id")
        if isinstance(frozen_spec_id, str):
            frozen_spec_path = root / "specs" / frozen_spec_id / "frozen_spec.json"
            if frozen_spec_path.is_file():
                spec = load_json(frozen_spec_path).get("spec") or {}
                metric_direction = spec.get("metric_direction")
                strategy = spec.get("strategy") or {}
                worker_host = strategy.get("worker_host")
                worker_budget = strategy.get("worker_budget")
        candidate_paths = sorted(run_dir.glob("candidates/*/candidate.json"))
        session_paths = sorted(run_dir.glob("agent_sessions/agent_*.json"))
        process_verifier_logs = sorted(
            run_dir.glob("candidates/*/logs/process/*.log")
        )
        promotion_verifier_logs = sorted(
            run_dir.glob("candidates/*/logs/promotion/*.log")
        )
        iteration_count = 0
        best_scores: list[float] = []
        hosts: set[str] = set()
        bound_candidate_ids: set[str] = set()
        worker_verified_candidate_ids: set[str] = set()
        unbound_agent_session_count = 0
        session_counts_by_candidate: dict[str, int] = {}
        bound_session_counts_by_candidate: dict[str, int] = {}
        same_agent_continuation_session_count = 0
        for candidate_path in candidate_paths:
            candidate = load_json(candidate_path)
            iterations = candidate.get("iterations")
            if isinstance(iterations, list):
                iteration_count += len(iterations)
                best_scores.extend(
                    float(item["score"])
                    for item in iterations
                    if isinstance(item, dict)
                    and isinstance(item.get("score"), (int, float))
                    and item.get("process_passed") is not False
                )
        for session_path in session_paths:
            session = load_json(session_path)
            if isinstance(session.get("host"), str):
                hosts.add(session["host"])
            candidate_id = session.get("candidate_id")
            if isinstance(candidate_id, str):
                session_counts_by_candidate[candidate_id] = (
                    session_counts_by_candidate.get(candidate_id, 0) + 1
                )
            host_handle = session.get("host_handle") or {}
            launch = session.get("launch") or {}
            if launch.get("tool") in {"followup_task", "pi_search_pool_continue"}:
                same_agent_continuation_session_count += 1
            external_id = host_handle.get("external_id")
            task_name = host_handle.get("task_name")
            is_bound = (
                isinstance(candidate_id, str)
                and (
                    (isinstance(external_id, str) and external_id)
                    or (
                        isinstance(task_name, str)
                        and task_name.startswith("/")
                    )
                )
            )
            if is_bound:
                bound_candidate_ids.add(candidate_id)
                bound_session_counts_by_candidate[candidate_id] = (
                    bound_session_counts_by_candidate.get(candidate_id, 0) + 1
                )
            else:
                unbound_agent_session_count += 1
            counters = session.get("counters") or {}
            if (
                isinstance(candidate_id, str)
                and isinstance(counters.get("verifier_runs"), int)
                and counters["verifier_runs"] > 0
            ):
                worker_verified_candidate_ids.add(candidate_id)
        search_space = collect_search_space_state(run_dir)
        runs.append(
            {
                "run_id": payload.get("run_id"),
                "status": payload.get("state", payload.get("status")),
                "candidate_count": len(candidate_paths),
                "agent_session_count": len(session_paths),
                "bound_agent_session_count": (
                    len(session_paths) - unbound_agent_session_count
                ),
                "bound_candidate_count": len(bound_candidate_ids),
                "worker_verified_candidate_count": len(
                    worker_verified_candidate_ids
                ),
                "worker_verified_candidate_ids": sorted(
                    worker_verified_candidate_ids
                ),
                "unbound_agent_session_count": unbound_agent_session_count,
                "session_counts_by_candidate": session_counts_by_candidate,
                "bound_session_counts_by_candidate": (
                    bound_session_counts_by_candidate
                ),
                "same_agent_continuation_session_count": (
                    same_agent_continuation_session_count
                ),
                "iteration_count": iteration_count,
                "process_verifier_command_count": len(process_verifier_logs),
                "promotion_verifier_command_count": len(promotion_verifier_logs),
                "metric_direction": metric_direction,
                "worker_host": worker_host,
                "worker_budget": worker_budget,
                "pi_pool_jobs": pi_pool_jobs_by_run.get(str(payload.get("run_id")), []),
                "best_recorded_score": (
                    min(best_scores)
                    if best_scores and metric_direction == "minimize"
                    else max(best_scores) if best_scores else None
                ),
                "recorded_score_min": min(best_scores) if best_scores else None,
                "recorded_score_max": max(best_scores) if best_scores else None,
                "selected_score": payload.get("selected_score"),
                "hosts": sorted(hosts),
                "report_exists": (run_dir / "report.md").is_file(),
                "search_space": search_space,
            }
        )
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "goals": goals,
        "runs": runs,
    }


def goal_plus_settled_selection(
    state: dict[str, Any], *, target_score: float | None = None
) -> bool:
    """Return True when a Search run has materialized a real business result.

    A run is "settled" once Goal Plus committed a ``selected_score`` (even 0.0,
    which is a genuine NOT_PASS) or a verified candidate reached ``target_score``.
    Such a run has a real PASS/NOT_PASS outcome and must never be reported as an
    INFRA_ERROR just because the host cut a pi worker's minimum lease short.
    """

    runs = state.get("runs") or []
    if not runs:
        return False
    for run in runs:
        selected_score = run.get("selected_score")
        if selected_score is not None:
            try:
                float(selected_score)
                return True
            except (TypeError, ValueError):
                pass
        best = run.get("best_recorded_score")
        if best is not None:
            try:
                best_value = float(best)
            except (TypeError, ValueError):
                continue
            if target_score is None or best_value == float(target_score):
                verified = run.get("worker_verified_candidate_count")
                if isinstance(verified, int) and verified >= 1:
                    return True
    return False


def goal_plus_incomplete_reason(
    state: dict[str, Any],
    *,
    expected_concurrency: int | None = None,
    minimum_worker_verified_candidates: int | None = None,
    expected_worker_min_runtime_seconds: int | None = None,
    expected_worker_min_verifier_runs: int | None = None,
    expected_goal_plus_id: str | None = None,
    expected_run_id: str | None = None,
    require_satisfied_pi_minimum_lease: bool = True,
    codex_events: dict[str, Any] | None = None,
) -> str | None:
    goals = state.get("goals") or []
    if not goals:
        return "Goal Plus did not create a durable goal record"
    noncomplete = [
        f"{item.get('goal_plus_id')}:{item.get('status')}"
        for item in goals
        if item.get("status") != "complete"
    ]
    if noncomplete:
        return "Goal Plus did not finish cleanly: " + ", ".join(noncomplete)
    if expected_goal_plus_id is not None:
        matching = [
            item for item in goals if item.get("goal_plus_id") == expected_goal_plus_id
        ]
        if len(matching) != 1:
            return f"expected exactly one prepared goal {expected_goal_plus_id}"
        duplicates = [
            item.get("goal_plus_id")
            for item in goals
            if item.get("goal_plus_id") != expected_goal_plus_id
            and item.get("linked_run_id") == expected_run_id
        ]
        if duplicates:
            return (
                "duplicate Goal Plus records linked to the prepared run: "
                + ", ".join(str(item) for item in duplicates)
            )
    runs = state.get("runs") or []
    if expected_run_id is not None:
        runs = [item for item in runs if item.get("run_id") == expected_run_id]
        if len(runs) != 1:
            return f"expected exactly one prepared Search run {expected_run_id}"
    else:
        # A natural /goal-plus invocation may leave an aborted spec-discovery
        # attempt in the append-only history before it creates the successful
        # Search run.  The experiment result is defined by the run currently
        # linked from the completed goal, not by historical aborted attempts.
        linked_run_ids = {
            item.get("linked_run_id")
            for item in goals
            if isinstance(item.get("linked_run_id"), str)
            and item.get("linked_run_id")
        }
        if not linked_run_ids:
            return "completed Goal Plus record did not link a Search run"
        runs = [item for item in runs if item.get("run_id") in linked_run_ids]
        if len(runs) != len(linked_run_ids):
            return "one or more Goal Plus linked Search runs are missing"
    expected_lease = {
        key: value
        for key, value in {
            "min_runtime_seconds": expected_worker_min_runtime_seconds,
            "min_verifier_runs": expected_worker_min_verifier_runs,
        }.items()
        if value is not None
    }
    for run in runs:
        actual_budget = run.get("worker_budget") or {}
        mismatches = [
            f"{key}={actual_budget.get(key)!r} (expected {expected!r})"
            for key, expected in expected_lease.items()
            if actual_budget.get(key) != expected
        ]
        if mismatches:
            return (
                f"Search run {run.get('run_id')} frozen worker budget mismatch: "
                + ", ".join(mismatches)
            )
        if (
            require_satisfied_pi_minimum_lease
            and expected_lease
            and run.get("worker_host") == "pi-rpc"
        ):
            jobs = run.get("pi_pool_jobs") or []
            unsatisfied = [
                str(job.get("job_id") or job.get("candidate_id") or "unknown")
                for job in jobs
                if job.get("status") != "completed"
                or not isinstance(job.get("lease"), dict)
                or job["lease"].get("satisfied") is not True
            ]
            if not jobs:
                return (
                    f"Search run {run.get('run_id')} has no Pi minimum lease evidence"
                )
            if unsatisfied:
                return (
                    f"Search run {run.get('run_id')} did not satisfy the Pi minimum lease "
                    "for jobs: " + ", ".join(unsatisfied)
                )
    if expected_concurrency is not None:
        if codex_events is not None:
            spawned_workers = int(
                codex_events.get("spawned_agent_thread_count") or 0
            )
            bound_workers = int(
                (codex_events.get("goal_plus") or {}).get(
                    "bound_worker_handle_count"
                )
                or 0
            )
            if max(spawned_workers, bound_workers) < expected_concurrency:
                return (
                    "Codex recorded "
                    f"{spawned_workers} distinct spawned worker threads; "
                    f"Goal Plus recorded {bound_workers} distinct bound worker handles; "
                    "expected at least "
                    f"{expected_concurrency} actual workers"
                )
        required_worker_evidence = (
            expected_concurrency
            if minimum_worker_verified_candidates is None
            else minimum_worker_verified_candidates
        )
        if not 0 <= required_worker_evidence <= expected_concurrency:
            return (
                "minimum worker verifier evidence must be between zero and "
                "expected concurrency"
            )
        for run in runs:
            if run.get("candidate_count") != expected_concurrency:
                return (
                    f"Search run {run.get('run_id')} materialized "
                    f"{run.get('candidate_count')} candidates; expected {expected_concurrency}"
                )
            session_counts = (
                run.get("bound_session_counts_by_candidate")
                or run.get("session_counts_by_candidate")
                or {}
            )
            duplicate_sessions = {
                candidate_id: count
                for candidate_id, count in session_counts.items()
                if count != 1
            }
            if len(session_counts) != expected_concurrency or duplicate_sessions:
                return (
                    f"Search run {run.get('run_id')} did not keep exactly one bound session per "
                    f"candidate: {session_counts}"
                )
            worker_verified_count = run.get("worker_verified_candidate_count")
            if (
                not isinstance(worker_verified_count, int)
                or worker_verified_count < required_worker_evidence
            ):
                return (
                    f"Search run {run.get('run_id')} has completed worker verifier "
                    f"evidence for {worker_verified_count} candidates; required at "
                    f"least {required_worker_evidence}"
                )
    return None


def close_pi_pools(workspace: Path, timeout_seconds: int) -> list[dict[str, Any]]:
    from goal_plus.pi_pool import close_pi_search_pool

    root = workspace / ".gp"
    summaries = []
    for path in sorted((root / "host-pools" / "pi").glob("pool_*/pool.json")):
        pool_id = path.parent.name
        snapshot = close_pi_search_pool(
            root_dir=root,
            pool_id=pool_id,
            mode="interrupt",
            timeout_seconds=timeout_seconds,
        )
        summaries.append(
            {
                "pool_id": pool_id,
                "state": snapshot.get("state"),
                "active_count": snapshot.get("active_count"),
                "terminal_count": snapshot.get("terminal_count"),
                "close_timed_out": bool(snapshot.get("close_timed_out")),
                "jobs": [
                    {
                        "job_id": job.get("job_id"),
                        "candidate_id": job.get("candidate_id"),
                        "status": job.get("status"),
                        "lease": (job.get("result") or {}).get("lease"),
                    }
                    for job in snapshot.get("jobs", [])
                ],
            }
        )
    return summaries


def apply_promotion_patch(source_workspace: Path, patch_path: Path) -> str:
    """Apply a Goal Plus promotion artifact once to the task source workspace."""
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    if not patch_path.read_text().strip():
        return "empty_patch"
    check = subprocess.run(
        ["git", "-C", str(source_workspace), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        subprocess.run(
            ["git", "-C", str(source_workspace), "apply", str(patch_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return "applied"
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(source_workspace),
            "apply",
            "--reverse",
            "--check",
            str(patch_path),
        ],
        capture_output=True,
        text=True,
    )
    if reverse.returncode == 0:
        return "already_applied"
    raise RuntimeError(
        "promotion patch does not apply cleanly: "
        + (check.stderr.strip() or reverse.stderr.strip())
    )


def _existing_promotion(
    run_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str]] | None:
    run_data = load_json(run_path)
    if run_data.get("state") != "promoted":
        return None
    candidate_id = run_data.get("selected_candidate_id")
    if not candidate_id:
        raise RuntimeError("promoted Search run has no selected candidate")
    patch_path = run_path.parent / "promotion" / f"{candidate_id}.patch"
    if not patch_path.is_file():
        raise RuntimeError("promoted Search run has no promotion artifact")
    selection = {
        "selected_candidate_id": candidate_id,
        "selected_score": run_data.get("selected_score"),
        "selected_iteration": run_data.get("selected_iteration"),
        "selected_git_head": run_data.get("selected_git_head"),
        "selected_artifact_hash": run_data.get("selected_artifact_hash"),
        "reused_existing_promotion": True,
    }
    return run_data, candidate_id, selection, {"artifact_path": str(patch_path)}


def _existing_selection(
    run_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    run_data = load_json(run_path)
    if run_data.get("state") != "ready_to_promote":
        return None
    candidate_id = run_data.get("selected_candidate_id")
    if not candidate_id:
        raise RuntimeError("ready-to-promote Search run has no selected candidate")
    selection = {
        "selected_candidate_id": candidate_id,
        "selected_score": run_data.get("selected_score"),
        "selected_iteration": run_data.get("selected_iteration"),
        "selected_git_head": run_data.get("selected_git_head"),
        "selected_artifact_hash": run_data.get("selected_artifact_hash"),
        "reused_existing_selection": True,
    }
    return run_data, candidate_id, selection


def _goal_plus_runtime_types() -> tuple[type[Any], type[Any], type[Any]]:
    from goal_plus.goal_plus import FileGoalPlusRuntime
    from goal_plus.runtime import FileSearchRuntime
    from goal_plus.tools import SearchTools

    return FileGoalPlusRuntime, FileSearchRuntime, SearchTools


def _validate_existing_blind_selection(
    run_path: Path, selection: dict[str, Any]
) -> None:
    expected: tuple[str, int, str] | None = None
    candidates = [
        load_json(candidate_path)
        for candidate_path in (run_path.parent / "candidates").glob(
            "*/candidate.json"
        )
    ]
    if any(
        not isinstance(candidate, dict)
        or not isinstance(candidate.get("candidate_id"), str)
        or not isinstance(candidate.get("iterations"), list)
        for candidate in candidates
    ):
        raise RuntimeError("blind candidate evidence is malformed")
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        compliant = [
            iteration
            for iteration in candidate.get("iterations", [])
            if isinstance(iteration, dict)
            and iteration.get("process_passed") is True
            and type(iteration.get("iteration")) is int
            and isinstance(iteration.get("git_head"), str)
            and iteration.get("git_artifact_clean") is True
            and not iteration.get("touched_denied_files", False)
            and not iteration.get("changed_outside_allowed", False)
        ]
        if compliant:
            latest = max(compliant, key=lambda item: item["iteration"])
            expected = (
                str(candidate["candidate_id"]),
                int(latest["iteration"]),
                str(latest["git_head"]),
            )
            break
    if expected is None:
        raise RuntimeError("no publicly compliant candidate iteration is available")
    actual = (
        selection.get("selected_candidate_id"),
        selection.get("selected_iteration"),
        selection.get("selected_git_head"),
    )
    if actual != expected:
        raise RuntimeError("existing promotion violates the frozen blind selection rule")


def finalize_goal_plus_search(
    workspace: Path, evaluation_mode: str = "visible"
) -> dict[str, Any]:
    """Controller-owned post-deadline drain/select/promote, outside search T."""
    if evaluation_mode not in {"visible", "blind"}:
        raise ValueError(f"unsupported evaluation mode: {evaluation_mode}")
    FileGoalPlusRuntime, FileSearchRuntime, SearchTools = _goal_plus_runtime_types()

    started = time.monotonic()
    root = workspace / ".gp"
    goal_runtime = FileGoalPlusRuntime(root)
    search_runtime = FileSearchRuntime(root)
    tools = SearchTools(search_runtime)
    result: dict[str, Any] = {
        "completed": False,
        "duration_seconds": None,
        "runs": [],
    }
    try:
        goal_paths = sorted((root / "goal-plus").glob("gp_*/goal.json"))
        if not goal_paths:
            raise RuntimeError("no Goal Plus goal record exists")
        goals_by_run: dict[str, list[str]] = {}
        for goal_path in goal_paths:
            goal = goal_runtime.status(goal_path.parent.name)
            if goal.linked_search is not None and goal.linked_search.run_id:
                goals_by_run.setdefault(goal.linked_search.run_id, []).append(
                    goal.goal_plus_id
                )
        for run_id, goal_ids in goals_by_run.items():
            run_path = root / "runs" / run_id / "run.json"
            run_data = load_json(run_path)
            initial_state = run_data.get("state")
            candidate_paths = sorted(
                (run_path.parent / "candidates").glob("*/candidate.json")
            )
            if not candidate_paths:
                continue
            verified_in_closeout: list[str] = []
            existing = _existing_promotion(run_path)
            if existing is not None:
                run_data, candidate_id, selection, promotion = existing
                if evaluation_mode == "blind":
                    _validate_existing_blind_selection(run_path, selection)
                    selection["selection_rule"] = BLIND_SELECTION_RULE
            else:
                try:
                    selected = _existing_selection(run_path)
                    if selected is None:
                        if evaluation_mode == "blind":
                            selection = tools.search_select(run_id)
                            _validate_existing_blind_selection(run_path, selection)
                            selection["selection_rule"] = BLIND_SELECTION_RULE
                        else:
                            for candidate_path in candidate_paths:
                                candidate = load_json(candidate_path)
                                if not candidate.get("iterations"):
                                    tools.search_run_verifier(
                                        run_id,
                                        candidate["candidate_id"],
                                        hypothesis="controller post-deadline final verification",
                                    )
                                    verified_in_closeout.append(
                                        candidate["candidate_id"]
                                    )
                            selection = tools.search_select(run_id)
                        candidate_id = selection["selected_candidate_id"]
                        run_data = load_json(run_path)
                    else:
                        run_data, candidate_id, selection = selected
                        if evaluation_mode == "blind":
                            _validate_existing_blind_selection(run_path, selection)
                            selection["selection_rule"] = BLIND_SELECTION_RULE
                    promotion = tools.search_promote(run_id, candidate_id)
                except RuntimeError:
                    existing = _existing_promotion(run_path)
                    if existing is not None:
                        run_data, candidate_id, selection, promotion = existing
                        if evaluation_mode == "blind":
                            _validate_existing_blind_selection(run_path, selection)
                            selection["selection_rule"] = BLIND_SELECTION_RULE
                    else:
                        if evaluation_mode == "blind":
                            raise
                        selected = _existing_selection(run_path)
                        if selected is None:
                            raise
                        run_data, candidate_id, selection = selected
                        promotion = tools.search_promote(run_id, candidate_id)
            patch_status = apply_promotion_patch(
                Path(run_data["source_path"]), Path(promotion["artifact_path"])
            )
            for goal_plus_id in goal_ids:
                goal = goal_runtime.status(goal_plus_id)
                linked = goal.linked_search
                if linked is None or linked.selected_candidate_id is None:
                    goal_runtime.record_search_result(
                        goal_plus_id,
                        run_id=run_id,
                        selected_candidate_id=candidate_id,
                        promotion_artifact_path=promotion["artifact_path"],
                        summary="Controller finalized the drained fixed-budget Search run.",
                    )
                goal = goal_runtime.status(goal_plus_id)
                if goal.status != "complete":
                    goal_runtime.set_status(
                        goal_plus_id,
                        status="complete",
                        reason=(
                            "fixed wall-clock search budget ended and controller closeout passed"
                        ),
                        evidence=[
                            {
                                "type": "controller_closeout",
                                "run_id": run_id,
                                "selected_candidate_id": candidate_id,
                                "selected_score": selection.get("selected_score"),
                            }
                        ],
                    )
            goal_statuses = {
                goal_plus_id: goal_runtime.status(goal_plus_id).status
                for goal_plus_id in goal_ids
            }
            final_run_data = load_json(run_path)
            report = tools.search_report(run_id)
            result["runs"].append(
                {
                    "goal_plus_ids": goal_ids,
                    "run_id": run_id,
                    "initial_state": initial_state,
                    "candidate_count": len(candidate_paths),
                    "verified_in_closeout": verified_in_closeout,
                    "selection": selection,
                    "promotion": promotion,
                    "source_patch_status": patch_status,
                    "final_state": final_run_data.get("state"),
                    "goal_statuses": goal_statuses,
                    "report": report,
                }
            )
        if not result["runs"]:
            raise RuntimeError(
                "no linked Search run with materialized candidates exists"
            )
        result["completed"] = True
    except Exception as error:  # preserve the raw run as diagnostic evidence
        result["error"] = f"{type(error).__name__}: {error}"
    result["duration_seconds"] = time.monotonic() - started
    return result


def primary_score(evaluation: dict[str, Any]) -> float:
    metric = evaluation.get("primary_metric")
    if not isinstance(metric, dict) or not isinstance(
        metric.get("value"), (int, float)
    ):
        raise RuntimeError("final evaluation did not return a numeric primary metric")
    return float(metric["value"])


def select_best_lane(
    lane_evaluations: list[dict[str, Any]], *, maximize: bool
) -> tuple[dict[str, Any] | None, list[str]]:
    scored: list[dict[str, Any]] = []
    invalid_lanes: list[str] = []
    for item in lane_evaluations:
        try:
            score = primary_score(item["evaluation"])
        except (KeyError, RuntimeError):
            invalid_lanes.append(item["lane"])
            continue
        scored.append({**item, "_primary_score": score})
    if not scored:
        return None, invalid_lanes
    selected = sorted(
        scored,
        key=lambda item: item["_primary_score"],
        reverse=maximize,
    )[0]
    selected.pop("_primary_score")
    return selected, invalid_lanes


def evaluator_budget_for_workspace(workspace: Path) -> dict[str, Any]:
    metadata = load_json(workspace / "task.json")
    return load_json(Path(metadata["controller_runtime_dir"]) / "budget.json")


def run_controlled(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdin_text: str | None,
    stdout_path: Path,
    stderr_path: Path,
    wall_time_seconds: int,
    hard_kill_grace_seconds: int,
    recorded_command: list[str] | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    soft_stopped = False
    hard_killed = False
    controller_interrupted = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        if stdin_text is not None and process.stdin is not None:
            process.stdin.write(stdin_text)
            process.stdin.close()
        try:
            process.wait(timeout=wall_time_seconds)
        except subprocess.TimeoutExpired:
            soft_stopped = True
            send_soft_stop(process)
            try:
                process.wait(timeout=hard_kill_grace_seconds)
            except subprocess.TimeoutExpired:
                hard_killed = True
                send_hard_stop(process)
                process.wait()
        except KeyboardInterrupt:
            controller_interrupted = True
            soft_stopped = True
            send_soft_stop(process)
            try:
                process.wait(timeout=hard_kill_grace_seconds)
            except subprocess.TimeoutExpired:
                hard_killed = True
                send_hard_stop(process)
                process.wait()

    return {
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "deadline_reached": soft_stopped,
        "controller_interrupted": controller_interrupted,
        "soft_stop_signal": "SIGTERM" if soft_stopped else None,
        "hard_killed": hard_killed,
        "hard_kill_grace_seconds": hard_kill_grace_seconds,
        "command": recorded_command or command,
    }


def run_controlled_many(
    jobs: list[dict[str, Any]],
    *,
    environment: dict[str, str],
    wall_time_seconds: int,
    hard_kill_grace_seconds: int,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    running: list[dict[str, Any]] = []
    for job in jobs:
        stdout = Path(job["stdout_path"]).open("w")
        stderr = Path(job["stderr_path"]).open("w")
        stdin_text = job.get("stdin_text")
        process = subprocess.Popen(
            job["command"],
            cwd=job["cwd"],
            env=job.get("environment", environment),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        if stdin_text is not None:
            assert process.stdin is not None
            process.stdin.write(stdin_text)
            process.stdin.close()
        running.append(
            {
                **job,
                "process": process,
                "stdout_handle": stdout,
                "stderr_handle": stderr,
                "started_at": utc_now(),
                "started_monotonic": time.monotonic(),
            }
        )

    deadline = started + wall_time_seconds
    controller_interrupted = False
    try:
        while time.monotonic() < deadline and any(
            item["process"].poll() is None for item in running
        ):
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        controller_interrupted = True

    soft_stopped = []
    for item in running:
        if item["process"].poll() is None:
            soft_stopped.append(item["name"])
            send_soft_stop(item["process"])

    grace_deadline = time.monotonic() + hard_kill_grace_seconds
    while time.monotonic() < grace_deadline and any(
        item["process"].poll() is None for item in running
    ):
        time.sleep(0.2)

    hard_killed = []
    lane_results = []
    for item in running:
        process = item["process"]
        if process.poll() is None:
            hard_killed.append(item["name"])
            send_hard_stop(process)
        process.wait()
        item["stdout_handle"].close()
        item["stderr_handle"].close()
        lane_results.append(
            {
                "name": item["name"],
                "started_at": item["started_at"],
                "finished_at": utc_now(),
                "duration_seconds": time.monotonic() - item["started_monotonic"],
                "returncode": process.returncode,
                "deadline_reached": item["name"] in soft_stopped,
                "soft_stop_signal": (
                    "SIGTERM" if item["name"] in soft_stopped else None
                ),
                "hard_killed": item["name"] in hard_killed,
                "command": item.get("recorded_command") or item["command"],
            }
        )
    return {
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "deadline_reached": bool(soft_stopped),
        "controller_interrupted": controller_interrupted,
        "soft_stopped_lanes": soft_stopped,
        "hard_killed_lanes": hard_killed,
        "hard_kill_grace_seconds": hard_kill_grace_seconds,
        "lanes": lane_results,
    }


def evaluate_native(
    task: Any,
    python: Path,
    artifact: Path,
    config: Path,
) -> dict[str, Any]:
    return run_worker(
        python,
        "evaluate",
        [
            "--upstream-root",
            str(task.upstream_root),
            "--config",
            str(config),
            "--evaluator",
            str(task.evaluator),
            "--artifact",
            str(artifact),
        ],
    )


def execute(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().absolute()
    manifest_path = run_dir / "experiment.json"
    manifest = load_json(manifest_path)
    if manifest["status"] != "prepared":
        raise RuntimeError(f"run is not prepared: status={manifest['status']}")

    method = canonical_method(manifest["method"])
    reasoning_effort = manifest.get(
        "reasoning_effort", DEFAULT_REASONING_EFFORT
    )
    if manifest.get("goal_plus_prepared") is not None:
        raise RuntimeError(
            "legacy controller-prepared Goal Plus runs cannot be executed by the standard "
            "prompt runner; preserve them as historical evidence or use closeout"
        )
    goal_plus_config = manifest.get("goal_plus_config")
    if goal_plus_config is not None and args.model != goal_plus_config["base_model"]:
        raise ValueError(
            "run model differs from the Goal Plus prompt configuration: "
            f"{args.model!r} != {goal_plus_config['base_model']!r}"
        )
    if method in {"goal-plus-codex", "goal-plus-pi"} and (
        Path(manifest["workspace"]) / ".gp"
    ).exists():
        raise RuntimeError(
            "standard Goal Plus execution must start without pre-created .gp state"
        )
    if not args.model:
        raise ValueError(
            "--model is required so every comparable run has explicit identity"
        )
    if (
        method in {"openevolve", "goal-plus-pi"}
        or sky_backend.is_method(method)
    ) and not args.api_base:
        raise ValueError(f"--api-base is required for {method}")
    budget = manifest["budget"]
    python = Path(manifest["environment"]["runtime_python"])
    task = resolve_task(
        manifest["task_id"],
        Path(manifest["environment"]["openevolve_root"]),
    )
    run_config = Path(manifest["task"]["config"])
    environment = configure_temp_environment(os.environ.copy())
    bin_dir = runtime_bin(args.venv.expanduser().absolute())
    environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
    if method in {"goal-plus-codex", "goal-plus-pi"}:
        configure_isolated_codex_home(environment, run_dir)
        configure_evidence_annotator_environment(
            environment,
            model=args.model,
            reasoning_effort=reasoning_effort,
            api_base=args.api_base,
        )

    if args.api_base and not environment.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required when --api-base selects the unified provider"
        )

    if method in {"plain-codex", "goal-plus-codex"}:
        manifest["codex_sandbox"] = CODEX_SANDBOX
        manifest["codex_approval_policy"] = "never"
    manifest["status"] = "running"
    manifest["execution_started_at"] = utc_now()
    write_json(manifest_path, manifest)

    if method == "openevolve":
        baseline = evaluate_native(task, python, task.initial_program, run_config)
        write_json(run_dir / "seed-eval.json", baseline)
        output = run_dir / "native-output"
        command = [
            str(bin_dir / "openevolve-run"),
            str(task.initial_program),
            str(task.evaluator),
            "--config",
            str(run_dir / "openevolve-config.yaml"),
            "--output",
            str(output),
            "--iterations",
            str(budget["iterations_ceiling"]),
            "--api-base",
            args.api_base,
            "--primary-model",
            args.model,
            "--secondary-model",
            args.model,
            "--log-level",
            "INFO",
        ]
        control = run_controlled(
            command,
            cwd=task.source_dir,
            environment=environment,
            stdin_text=None,
            stdout_path=run_dir / "stdout.log",
            stderr_path=run_dir / "stderr.log",
            wall_time_seconds=budget["wall_time_seconds"],
            hard_kill_grace_seconds=budget["hard_kill_grace_seconds"],
        )
        best = output / "best" / f"best_program{task.initial_program.suffix}"
        if best.is_file():
            shutil.copy2(best, run_dir / "final-candidate.py")
            write_json(
                run_dir / "final-eval.json",
                evaluate_native(task, python, best, run_config),
            )
            info = output / "best" / "best_program_info.json"
            if info.is_file():
                control["native_best"] = load_json(info)
        else:
            control["result_incomplete_reason"] = "native best program was not saved"
        control["telemetry_coverage"] = {
            "evaluator_calls": "missing: native upstream does not expose an exact completed-call ledger",
            "tokens": "missing: native upstream OpenAI-compatible client does not persist usage",
            "iterations": "best_program_info iteration is available when graceful shutdown saves a best",
        }
    elif sky_backend.is_method(method):
        workspace = Path(manifest["workspace"])
        seed_evaluation = evaluate_workspace(workspace, "public")
        write_json(run_dir / "seed-eval.json", seed_evaluation)
        setup_evaluator_calls = evaluator_budget_for_workspace(workspace)[
            "total_claimed"
        ]

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
        algorithm = sky_backend.algorithm_for_method(method)
        command = [
            str(sky_executable),
            str(workspace / task.artifact_name),
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
        )
        best = sky_backend.best_candidate(output, task.initial_program.suffix)
        best_info = sky_backend.collect_best_info(output)
        if best.is_file():
            shutil.copy2(best, workspace / task.artifact_name)
            shutil.copy2(best, run_dir / "final-candidate.py")
            final = evaluate_workspace(workspace, "final")
            write_json(run_dir / "final-eval.json", final)
        else:
            control["result_incomplete_reason"] = (
                "SkyDiscover best program was not saved"
            )

        evaluator_calls = evaluator_budget_for_workspace(workspace)
        evaluator_calls["setup_claimed_before_t"] = setup_evaluator_calls
        evaluator_calls["timed_plus_closeout_claimed"] = (
            evaluator_calls["total_claimed"] - setup_evaluator_calls
        )
        control["evaluator_calls"] = evaluator_calls
        control["usage"] = {
            "coverage": (
                "missing: SkyDiscover OpenAI-compatible client does not "
                "persist response usage metadata"
            )
        }
        control["skydiscover"] = {
            "algorithm": algorithm,
            "output_dir": str(output),
            "best_info": best_info,
            "requested_concurrency_cap": budget["concurrency"],
            "observed_peak_concurrency": None,
            "evaluation_workspace_count": sum(
                1 for path in evaluation_root.iterdir() if path.is_dir()
            ),
            "protocol_coverage": (
                "functional smoke: native seed and best test evaluations still "
                "occur inside the timed SkyDiscover runtime"
            ),
            "determinism_coverage": manifest["search_backend"][
                "determinism_coverage"
            ],
        }
        control["telemetry_coverage"] = {
            "evaluator_calls": (
                "exact controller-owned ledger, including native seed/best "
                "re-evaluations and controller final evaluation"
            ),
            "tokens": control["usage"]["coverage"],
            "iterations": (
                "native best_program_info is persisted when a best candidate exists"
            ),
            "actual_concurrency": (
                "missing: runtime does not persist an observed peak"
            ),
        }
    elif method == "plain-codex":
        workspaces = [Path(path) for path in manifest.get("workspaces") or []]
        if len(workspaces) != budget["concurrency"]:
            raise RuntimeError(
                f"plain Codex expected {budget['concurrency']} lanes, got {len(workspaces)}"
            )
        lanes_root = run_dir / "lanes"
        lanes_root.mkdir()
        seed_evaluations = []
        jobs = []
        setup_evaluator_calls = 0
        for lane_index, workspace in enumerate(workspaces):
            lane_name = f"lane-{lane_index:02d}"
            lane_dir = lanes_root / lane_name
            lane_dir.mkdir()
            seed_evaluation = evaluate_workspace(workspace, "public")
            write_json(lane_dir / "seed-eval.json", seed_evaluation)
            setup_evaluator_calls += evaluator_budget_for_workspace(workspace)[
                "total_claimed"
            ]
            seed_evaluations.append({"lane": lane_name, "evaluation": seed_evaluation})
            prompt = render_plain_prompt(
                (workspace / "TASK.md").read_text(),
                budget["wall_time_seconds"],
                budget["soft_closeout_seconds"],
            )
            (lane_dir / "prompt.md").write_text(prompt)
            command = [
                args.codex_bin,
                "exec",
                "--json",
                *codex_execution_args(),
                "--cd",
                str(workspace),
                "--output-last-message",
                str(lane_dir / "final-message.txt"),
                "--ignore-user-config",
                "--color",
                "never",
                "--ephemeral",
                *codex_model_args(
                    args.model,
                    args.api_base,
                    reasoning_effort,
                ),
                "-",
            ]
            jobs.append(
                {
                    "name": lane_name,
                    "command": command,
                    "cwd": workspace,
                    "stdin_text": prompt,
                    "stdout_path": lane_dir / "events.jsonl",
                    "stderr_path": lane_dir / "stderr.log",
                }
            )
        write_json(run_dir / "seed-evals.json", {"lanes": seed_evaluations})
        control = run_controlled_many(
            jobs,
            environment=environment,
            wall_time_seconds=budget["wall_time_seconds"],
            hard_kill_grace_seconds=budget["hard_kill_grace_seconds"],
        )
        lane_evaluations = []
        for lane_index, workspace in enumerate(workspaces):
            lane_name = f"lane-{lane_index:02d}"
            lane_dir = lanes_root / lane_name
            final = evaluate_workspace(workspace, "final")
            write_json(lane_dir / "final-eval.json", final)
            candidate_path = lane_dir / "final-candidate.py"
            shutil.copy2(workspace / task.artifact_name, candidate_path)
            lane_evaluations.append(
                {
                    "lane": lane_name,
                    "workspace": str(workspace),
                    "candidate": str(candidate_path),
                    "evaluation": final,
                    "codex": parse_codex_events(lane_dir / "events.jsonl"),
                }
            )
        write_json(run_dir / "lane-results.json", {"lanes": lane_evaluations})
        selected, invalid_lanes = select_best_lane(
            lane_evaluations,
            maximize=manifest["task"]["direction"] == "maximize",
        )
        control["invalid_evaluation_lanes"] = invalid_lanes
        if selected is None:
            control["result_incomplete_reason"] = (
                "plain Codex produced no lane with a numeric final metric"
            )
        else:
            write_json(run_dir / "final-eval.json", selected["evaluation"])
            shutil.copy2(selected["candidate"], run_dir / "final-candidate.py")
            control["selected_lane"] = selected["lane"]
            control["selected_score"] = primary_score(selected["evaluation"])
        control["codex"] = {
            "lanes": [
                {"lane": item["lane"], **item["codex"]} for item in lane_evaluations
            ],
            "coverage": "top-level Codex usage for every independent lane",
        }
        control["evaluator_calls"] = {
            "lane_count": len(lane_evaluations),
            "total_claimed": sum(
                item["evaluation"]["budget"]["total_claimed"]
                for item in lane_evaluations
            ),
            "public_claimed": sum(
                item["evaluation"]["budget"]["public_claimed"]
                for item in lane_evaluations
            ),
            "final_claimed": sum(
                item["evaluation"]["budget"]["final_claimed"]
                for item in lane_evaluations
            ),
            "setup_claimed_before_t": setup_evaluator_calls,
            "timed_plus_closeout_claimed": (
                sum(
                    item["evaluation"]["budget"]["total_claimed"]
                    for item in lane_evaluations
                )
                - setup_evaluator_calls
            ),
        }
        bad_lanes = [
            item["name"]
            for item in control["lanes"]
            if item["returncode"] != 0 or item["hard_killed"]
        ]
        if bad_lanes:
            control["result_incomplete_reason"] = (
                "plain Codex lanes did not exit cleanly: " + ", ".join(bad_lanes)
            )
    else:
        workspace = Path(manifest["workspace"])
        write_json(run_dir / "seed-eval.json", evaluate_workspace(workspace, "public"))
        setup_evaluator_calls = evaluator_budget_for_workspace(workspace)[
            "total_claimed"
        ]
        deadline = datetime.now(timezone.utc) + timedelta(
            seconds=budget["wall_time_seconds"]
        )
        environment["GOAL_PLUS_OUTER_DEADLINE_AT"] = deadline.isoformat()
        if method == "goal-plus-codex":
            prompt = render_goal(
                task_text=(workspace / "TASK.md").read_text(),
                artifact_name=manifest["task"]["artifact_name"],
                metric_name=manifest["task"]["primary_metric"],
                metric_direction=manifest["task"]["direction"],
                wall_seconds=budget["wall_time_seconds"],
                closeout_seconds=budget["soft_closeout_seconds"],
                concurrency=budget["concurrency"],
                worker_host="codex",
                worker_model=args.model,
                reasoning_effort=reasoning_effort,
            )
            command = [
                args.codex_bin,
                "exec",
                "--json",
                *codex_execution_args(),
                "--cd",
                str(workspace),
                "--output-last-message",
                str(run_dir / "final-message.txt"),
                "--color",
                "never",
                "--dangerously-bypass-hook-trust",
                *codex_model_args(
                    args.model,
                    args.api_base,
                    reasoning_effort,
                ),
                *codex_goal_plus_mcp_args(),
                "-",
            ]
            stdout_path = run_dir / "events.jsonl"
            recorded_command = None
        else:
            qualified_model = f"{PI_PROVIDER_ID}/{args.model}"
            prompt = render_goal(
                task_text=(workspace / "TASK.md").read_text(),
                artifact_name=manifest["task"]["artifact_name"],
                metric_name=manifest["task"]["primary_metric"],
                metric_direction=manifest["task"]["direction"],
                wall_seconds=budget["wall_time_seconds"],
                closeout_seconds=budget["soft_closeout_seconds"],
                concurrency=budget["concurrency"],
                worker_host="pi-rpc",
                worker_model=qualified_model,
                reasoning_effort=reasoning_effort,
            )
            pi_home = run_dir / "pi-home"
            write_pi_models_config(
                pi_home,
                api_base=args.api_base,
                model=args.model,
                reasoning_effort=reasoning_effort,
            )
            environment["PI_CODING_AGENT_DIR"] = str(pi_home)
            environment["GOAL_PLUS_PI_MODEL"] = qualified_model
            command = [
                args.pi_bin,
                "--mode",
                "json",
                "--provider",
                PI_PROVIDER_ID,
                "--model",
                qualified_model,
                "--thinking",
                reasoning_effort,
                "--approve",
                "--session-dir",
                str(run_dir / "pi-main-session"),
                "--session-id",
                f"bench-{run_dir.name}",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
                "--extension",
                str(workspace / ".pi/extensions/goal-plus.ts"),
                "--skill",
                str(workspace / ".pi/skills/goal-plus/SKILL.md"),
                prompt,
            ]
            stdout_path = run_dir / "events.jsonl"
            recorded_command = [*command[:-1], "<goal-prompt>"]
        (run_dir / "prompt.md").write_text(prompt)
        control = run_controlled(
            command,
            cwd=workspace,
            environment=environment,
            stdin_text=prompt if method == "goal-plus-codex" else None,
            stdout_path=stdout_path,
            stderr_path=run_dir / "stderr.log",
            wall_time_seconds=budget["wall_time_seconds"],
            hard_kill_grace_seconds=budget["hard_kill_grace_seconds"],
            recorded_command=recorded_command,
        )
        if method == "goal-plus-pi":
            control["pi_pool_cleanup"] = close_pi_pools(
                workspace, budget["hard_kill_grace_seconds"]
            )
        control["goal_plus_controller_closeout"] = finalize_goal_plus_search(workspace)
        final = evaluate_workspace(workspace, "final")
        write_json(run_dir / "final-eval.json", final)
        shutil.copy2(workspace / task.artifact_name, run_dir / "final-candidate.py")
        if method == "goal-plus-codex":
            control["codex"] = parse_codex_events(run_dir / "events.jsonl")
        else:
            control["pi"] = parse_pi_events(run_dir / "events.jsonl")
        evaluator_calls = dict(final["budget"])
        evaluator_calls["setup_claimed_before_t"] = setup_evaluator_calls
        evaluator_calls["timed_plus_closeout_claimed"] = (
            evaluator_calls["total_claimed"] - setup_evaluator_calls
        )
        control["evaluator_calls"] = evaluator_calls
        control["goal_plus"] = collect_goal_plus_state(workspace)
        control["evidence_annotator_usage"] = collect_evidence_annotator_usage(
            workspace
        )
        if control["hard_killed"]:
            control["result_incomplete_reason"] = (
                f"{method} process group exceeded the shutdown grace"
            )
        goal_reason = goal_plus_incomplete_reason(
            control["goal_plus"],
            expected_concurrency=budget["concurrency"],
            codex_events=(
                control.get("codex") if method == "goal-plus-codex" else None
            ),
            expected_worker_min_runtime_seconds=budget.get(
                "worker_min_runtime_seconds"
            ),
            expected_worker_min_verifier_runs=(
                1 if budget.get("worker_min_runtime_seconds") is not None else None
            ),
        )
        if goal_reason and not control.get("result_incomplete_reason"):
            control["result_incomplete_reason"] = goal_reason
        if not control["goal_plus_controller_closeout"].get("completed"):
            control["result_incomplete_reason"] = (
                "Goal Plus controller closeout failed: "
                + control["goal_plus_controller_closeout"].get(
                    "error", "unknown closeout error"
                )
            )

    expected_deadline_stop = (
        control.get("deadline_reached")
        and not control.get("hard_killed")
        and control.get("returncode") in {0, -signal.SIGTERM, 128 + signal.SIGTERM}
    )
    if (
        control.get("returncode", 0) != 0
        and not expected_deadline_stop
        and not control.get("result_incomplete_reason")
    ):
        control["result_incomplete_reason"] = (
            f"controlled process exited nonzero ({control['returncode']})"
        )

    manifest["status"] = (
        "finished" if not control.get("result_incomplete_reason") else "incomplete"
    )
    manifest["model"] = args.model
    manifest["api_base"] = args.api_base
    manifest["provider_mode"] = (
        "openai_compatible" if args.api_base else "codex_native_auth"
    )
    manifest["reasoning_effort"] = reasoning_effort
    version_command = runner_version_command(
        method,
        python=python,
        codex_bin=args.codex_bin,
        pi_bin=args.pi_bin,
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
    python = Path(manifest["environment"]["runtime_python"])
    task = resolve_task(
        manifest["task_id"],
        Path(manifest["environment"]["openevolve_root"]),
    )
    payload = evaluate_native(
        task,
        python,
        task.initial_program,
        Path(manifest["task"]["config"]),
    )
    write_json(run_dir / "seed-eval.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


def repair_closeout(args: argparse.Namespace) -> int:
    """Idempotently recover Goal Plus selection/promotion after host interruption."""
    run_dir = args.run_dir.expanduser().absolute()
    manifest_path = run_dir / "experiment.json"
    manifest = load_json(manifest_path)
    method = canonical_method(manifest["method"])
    if method not in {"goal-plus-codex", "goal-plus-pi"}:
        raise ValueError("closeout is only valid for Goal Plus runs")
    workspace = Path(manifest["workspace"])
    control = dict(manifest.get("execution") or {})
    if method == "goal-plus-pi":
        control["pi_pool_cleanup_repair"] = close_pi_pools(
            workspace, manifest["budget"]["hard_kill_grace_seconds"]
        )
    control["goal_plus_controller_closeout_repair"] = finalize_goal_plus_search(
        workspace
    )
    control["goal_plus"] = collect_goal_plus_state(workspace)
    control["evidence_annotator_usage"] = collect_evidence_annotator_usage(
        workspace
    )
    reason = goal_plus_incomplete_reason(
        control["goal_plus"],
        expected_concurrency=manifest["budget"]["concurrency"],
        codex_events=(
            control.get("codex") if method == "goal-plus-codex" else None
        ),
        expected_worker_min_runtime_seconds=manifest["budget"].get(
            "worker_min_runtime_seconds"
        ),
        expected_worker_min_verifier_runs=(
            1
            if manifest["budget"].get("worker_min_runtime_seconds") is not None
            else None
        ),
    )
    if reason is None and not control.get("hard_killed"):
        control.pop("result_incomplete_reason", None)
        manifest["status"] = "finished"
    else:
        control["result_incomplete_reason"] = reason or (
            "process group exceeded the shutdown grace"
        )
        manifest["status"] = "incomplete"
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
        "--method", choices=(*METHODS, *METHOD_ALIASES), required=True
    )
    prepare_parser.add_argument("--task-id", default="function_minimization")
    prepare_parser.add_argument("--seed", type=int, default=1)
    prepare_parser.add_argument("--model", default=DEFAULT_MODEL)
    prepare_parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
    )
    prepare_parser.add_argument(
        "--wall-time-seconds", type=int, default=DEFAULT_WALL_TIME_SECONDS
    )
    prepare_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    prepare_parser.add_argument("--soft-closeout-seconds", type=int, default=60)
    prepare_parser.add_argument("--hard-kill-grace-seconds", type=int, default=30)
    prepare_parser.add_argument("--iterations-ceiling", type=int, default=1_000_000)
    prepare_parser.add_argument("--run-dir", type=Path)
    prepare_parser.add_argument(
        "--environment-manifest", type=Path, default=DEFAULT_ENV_MANIFEST
    )
    prepare_parser.add_argument(
        "--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT
    )
    prepare_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)

    prepare_batch_parser = subparsers.add_parser("prepare-batch")
    prepare_batch_parser.add_argument("--task-set", default="cpu_portable")
    prepare_batch_parser.add_argument(
        "--methods",
        nargs="+",
        choices=(*METHODS, *METHOD_ALIASES),
        default=list(DEFAULT_BATCH_METHODS),
    )
    prepare_batch_parser.add_argument("--seed", type=int, default=1)
    prepare_batch_parser.add_argument("--model", default=DEFAULT_MODEL)
    prepare_batch_parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
    )
    prepare_batch_parser.add_argument(
        "--wall-time-seconds", type=int, default=DEFAULT_WALL_TIME_SECONDS
    )
    prepare_batch_parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY
    )
    prepare_batch_parser.add_argument("--soft-closeout-seconds", type=int, default=60)
    prepare_batch_parser.add_argument("--hard-kill-grace-seconds", type=int, default=30)
    prepare_batch_parser.add_argument(
        "--iterations-ceiling", type=int, default=1_000_000
    )
    prepare_batch_parser.add_argument("--run-root", type=Path, required=True)
    prepare_batch_parser.add_argument(
        "--environment-manifest", type=Path, default=DEFAULT_ENV_MANIFEST
    )
    prepare_batch_parser.add_argument(
        "--checkout-root", type=Path, default=DEFAULT_CHECKOUT_ROOT
    )
    prepare_batch_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    run_parser.add_argument("--codex-bin", default="codex")
    run_parser.add_argument("--pi-bin", default="pi")
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--api-base")

    run_batch_parser = subparsers.add_parser("run-batch")
    run_batch_parser.add_argument("--campaign", type=Path, required=True)
    run_batch_parser.add_argument("--methods", nargs="+", choices=METHODS)
    run_batch_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    run_batch_parser.add_argument("--codex-bin", default="codex")
    run_batch_parser.add_argument("--pi-bin", default="pi")
    run_batch_parser.add_argument("--model")
    run_batch_parser.add_argument("--api-base")
    run_batch_parser.add_argument("--fail-fast", action="store_true")

    smoke_parser = subparsers.add_parser("seed-smoke")
    smoke_parser.add_argument("--run-dir", type=Path, required=True)
    closeout_parser = subparsers.add_parser("closeout")
    closeout_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "prepare-batch":
        return prepare_batch(args)
    if args.command == "run-batch":
        return run_batch(args)
    if args.command == "seed-smoke":
        return seed_smoke(args)
    if args.command == "closeout":
        return repair_closeout(args)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
