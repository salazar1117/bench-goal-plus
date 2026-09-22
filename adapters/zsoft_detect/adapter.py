#!/usr/bin/env python3
"""Adapter for the cybergym-zsoft-detect static detection benchmark.

Worker-visible validation checks only the public finding format. After every
worker has exited, the trusted benchmark controller scores the compliant
committed snapshots and selects the one with the highest official F1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
from adapters import zsoft_blind  # noqa: E402
from adapters.zsoft_blind import (  # noqa: E402
    DETECT_VALIDATION_KIND,
    PUBLIC_CHECKER_NAME,
    PUBLIC_METRIC,
    diagnostics_valid,
    ensure_single_final_claim,
    validate_detect_submission,
)

ZSOFT_ROOT = Path(
    os.environ.get("BENCH_GOAL_PLUS_ZSOFT_ROOT", ROOT / "third_party" / "zsoft-bench")
).expanduser().resolve()
BENCHMARK_ROOT = ZSOFT_ROOT / "benchmarks" / "vulnerability" / "zsoft-detect"

BENCHMARK_NAME = "zsoft-detect"
DEFAULT_PROJECT = "civetweb"
TASK_ID = f"{DEFAULT_PROJECT}-detect"
UPSTREAM_KEY = "zsoft_l1"
UPSTREAM_SUBDIR = "benchmarks/vulnerability/zsoft-detect"
ARTIFACT_NAME = "submission"
PRIMARY_METRIC = "f1"
GOAL_PLUS_PROCESS_METRIC = PUBLIC_METRIC
PUBLIC_FORMAT_METRIC = PUBLIC_METRIC
CONTROLLER_ONLY_OFFICIAL_EVALUATION = True
EVALUATION_MODE = "blind"
REQUIRES_PROTECTED_PI_WORKERS = True
DIRECTION = "maximize"
GOAL_PLUS_POSTHOC_SELECTION_CONTRACT = {
    "enabled": True,
    "metric_name": PRIMARY_METRIC,
    "metric_direction": DIRECTION,
    "candidate_scope": "all_publicly_compliant_iterations",
    "tie_break": "lowest_candidate_id_then_latest_iteration",
    "timing": "after_agent_exit_and_controller_closeout",
    "visible_to_workers": False,
}
CASE_SET_DESCRIPTION = (
    "one zsoft-detect project bench: static findings on a pinned commit"
)
CODEX_SANDBOX = "workspace-write"
PI_WORKER_SANDBOX = {
    "engine": "bubblewrap",
    "evaluation_mode": EVALUATION_MODE,
    "workspace_access": "read_only",
    "read_only_workspace_paths": ["source", "schemas"],
    "writable_workspace_paths": [ARTIFACT_NAME],
    "pass_env": [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ],
}
VERIFIER_TIMEOUT_SECONDS = 30
OFFICIAL_EVALUATOR_TIMEOUT_SECONDS = 600
SOURCE_CACHE_ENV = "BENCH_GOAL_PLUS_ZSOFT_DETECT_SOURCE_CACHE"

# project -> bench commit (from projects/<p>/versions/*)
PROJECT_COMMITS: dict[str, str] = {
    "civetweb": "d7ba35bbb649209c66e582d5a0244ba988a15159",
    "jiuwenswarm": "f5e0e4c7b64c2ffbbde05762f949659e08457f56",
    "libxml2": "d583ca9bbfbfa58f8f2c400c7eade02dfc70bbbd",
    "linux-rxrpc-sample": "1f318b96cc84d7c2ab792fcc0bfd42a7ca890681",
    "umdk": "8ea80d2c35fd9312dd81ace11e2836a2665311ff",
}

ACTIVE_PROJECT = DEFAULT_PROJECT


class AdapterError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(
    arguments: list[str],
    *,
    cwd: Path = BENCHMARK_ROOT,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path.home() / "bin") + os.pathsep + environment.get("PATH", "")
    )
    return subprocess.run(
        arguments,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def list_projects() -> list[str]:
    projects = BENCHMARK_ROOT / "projects"
    return sorted(
        path.name
        for path in projects.iterdir()
        if (path / "project.json").is_file()
    )


def configure_task(task_id: str | None) -> None:
    """Accept either a bare project id or '<project>-detect'."""
    global ACTIVE_PROJECT, TASK_ID
    if task_id is None:
        ACTIVE_PROJECT = DEFAULT_PROJECT
    else:
        project = task_id.removesuffix("-detect")
        if project not in PROJECT_COMMITS:
            raise AdapterError(f"unknown zsoft-detect project: {task_id}")
        ACTIVE_PROJECT = project
    TASK_ID = f"{ACTIVE_PROJECT}-detect"


def project_commit(project: str) -> str:
    return PROJECT_COMMITS[project]


def _resolve_benchmark_root(source_root: Path) -> Path:
    source_root = Path(source_root).expanduser().absolute()
    if (source_root / "scripts" / "show_bench.py").is_file():
        return source_root
    candidate = source_root / UPSTREAM_SUBDIR
    if (candidate / "scripts" / "show_bench.py").is_file():
        return candidate
    raise FileNotFoundError(f"zsoft-detect scripts missing under {source_root}")


def bench_contract(
    project: str, benchmark_root: Path = BENCHMARK_ROOT
) -> dict[str, Any]:
    completed = _run(
        [
            sys.executable,
            "scripts/show_bench.py",
            "--project",
            project,
            "--commit",
            project_commit(project),
        ],
        cwd=benchmark_root,
    )
    if completed.returncode != 0:
        raise AdapterError(f"show_bench failed: {completed.stderr[-500:]}")
    return json.loads(completed.stdout)


TASK_SKILLS_SOURCE = Path(__file__).resolve().parent / "skills"


def copy_task_skills(workspace: Path) -> list[str]:
    """Copy adapter-owned task-local Skills into the workspace.

    Goal Plus discovers Git-tracked ``.agents/skills/<name>/SKILL.md`` bundles
    at freeze time and instructs every candidate worker to read them, so the
    copies must exist before ``init_git`` commits the workspace.
    """
    copied: list[str] = []
    if not TASK_SKILLS_SOURCE.is_dir():
        return copied
    for skill_dir in sorted(TASK_SKILLS_SOURCE.iterdir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_dir.is_dir() or not skill_file.is_file():
            continue
        if skill_dir.is_symlink() or skill_file.is_symlink():
            raise AdapterError(
                f"task-local Skill cannot be a symlink: {skill_dir.name}"
            )
        destination = workspace / ".agents" / "skills" / skill_dir.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            skill_dir,
            destination,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        copied.append(f".agents/skills/{skill_dir.name}/SKILL.md")
    return copied


def task_text(project: str, contract: dict[str, Any]) -> str:
    roots = ", ".join(contract["scan_roots"]) or "(whole repository)"
    bug_types = ", ".join(
        item["id"] for item in contract["project_bug_types"]
    )
    return "\n".join(
        [
            f"# ZSoft detect bench: {contract['display_name']} @ {contract['version_label']}",
            "",
            "Perform a static security audit of the source tree in `source/`.",
            f"Scan roots: {roots}.",
            f"Allowed bug types (CWE ids or aliases): {bug_types}.",
            "",
            f"Write each finding as one JSON file under `{ARTIFACT_NAME}/`,",
            "matching `schemas/finding.schema.json` in the bench contract",
            "copy (`bench-contract.json`): location (repo-relative path,",
            "function, 1-based inclusive line range), bug_type, and",
            "root_cause (cause/trigger/impact).",
            "",
            f"Use `python3 {PUBLIC_CHECKER_NAME}` to check only the public",
            "JSON structure. It does not assess whether a reported issue is",
            "real. Submit only findings supported by your source audit.",
        ]
    ) + "\n"


def materialize_workspace(
    source_root: Path,
    workspace: Path,
) -> dict[str, Any]:
    benchmark_root = _resolve_benchmark_root(source_root)
    workspace = Path(workspace).expanduser().absolute()
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError(workspace)
    project = ACTIVE_PROJECT
    commit = project_commit(project)
    contract = bench_contract(project, benchmark_root)

    source_checkout = validated_source_cache(project, commit)
    source_materialization = "validated_local_cache"
    if source_checkout is None:
        source_checkout = workspace.parent / f"{workspace.name}-source"
        if source_checkout.exists() or source_checkout.is_symlink():
            source_checkout = validate_source_checkout(
                source_checkout,
                project,
                commit,
                label="campaign-local source checkout",
            )
        else:
            fetch_source_checkout(project, commit, source_checkout)
            source_checkout = validate_source_checkout(
                source_checkout,
                project,
                commit,
                label="fetched campaign-local source checkout",
            )
        source_materialization = "campaign_local"

    _require_disjoint_paths(
        workspace=workspace,
        source_checkout=source_checkout,
        benchmark_root=benchmark_root,
    )
    scan_roots = _validated_scan_roots(contract, source_checkout)
    workspace.mkdir(parents=True)
    from adapters.portable import copytree_confined

    source_destination = workspace / "source"
    if scan_roots:
        source_destination.mkdir()
        for relative, source_path in scan_roots:
            destination = source_destination / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_dir():
                copytree_confined(
                    source_path,
                    destination,
                    label=f"ZSoft detect scan root {relative.as_posix()}",
                    ignore=shutil.ignore_patterns(".git"),
                )
            else:
                shutil.copy2(source_path, destination, follow_symlinks=False)
        source_materialization += "_scan_roots"
    else:
        copytree_confined(
            source_checkout,
            source_destination,
            label="ZSoft detect source checkout",
            ignore=shutil.ignore_patterns(".git"),
        )
        source_materialization += "_copy"
    (workspace / "bench-contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    schema_relative = _submission_schema_path(contract)
    schema_source = (benchmark_root / schema_relative).resolve(strict=True)
    resolved_benchmark_root = benchmark_root.resolve(strict=True)
    if (
        schema_source == resolved_benchmark_root
        or resolved_benchmark_root not in schema_source.parents
        or not schema_source.is_file()
    ):
        raise AdapterError(
            f"public submission schema escapes the benchmark root: {schema_relative}"
        )
    schema_destination = workspace / schema_relative
    schema_destination.parent.mkdir(parents=True)
    shutil.copy2(schema_source, schema_destination, follow_symlinks=False)
    (workspace / ARTIFACT_NAME).mkdir()
    (workspace / ARTIFACT_NAME / ".gitkeep").write_text("")
    (workspace / "TASK.md").write_text(task_text(project, contract))
    (workspace / "AGENTS.md").write_text(
        "# ZSoft detect task rules\n\n"
        "- Audit only the tree under `source/`.\n"
        f"- Write finding JSON files into `{ARTIFACT_NAME}/`.\n"
        "- The bench contract in `bench-contract.json` is read-only reference.\n"
        f"- `{PUBLIC_CHECKER_NAME}` checks structure only and is read-only.\n"
        "- Hidden task data and other run directories are forbidden.\n"
    )
    shutil.copy2(
        Path(zsoft_blind.__file__),
        workspace / PUBLIC_CHECKER_NAME,
        follow_symlinks=False,
    )
    copy_task_skills(workspace)
    (workspace / ".gitignore").write_text(
        ".bench-runtime/\n.gp/\n.codex-log/\n__pycache__/\n*.pyc\n"
    )

    metadata = {
        "schema_version": 1,
        "adapter": "zsoft-detect",
        "task_id": f"{project}-detect",
        "project_id": project,
        "commit": commit,
        "artifact_name": ARTIFACT_NAME,
        "upstream_commit": commit,
        "source_revision": commit,
        "source_materialization": source_materialization,
        "framework_version": _framework_version(benchmark_root),
        "controller_only_official_evaluation": True,
        "evaluation_mode": EVALUATION_MODE,
        "requires_protected_pi_workers": REQUIRES_PROTECTED_PI_WORKERS,
        "public_validation_kind": DETECT_VALIDATION_KIND,
        "scan_roots": list(contract.get("scan_roots") or []),
        "primary_metric": GOAL_PLUS_PROCESS_METRIC,
        "direction": DIRECTION,
    }
    (workspace / "task.json").write_text(json.dumps(metadata, indent=2) + "\n")

    from adapters.portable import init_git

    workspace_commit = init_git(workspace, f"zsoft-detect {project}@{commit[:12]}")
    return {
        "task_id": f"{project}-detect",
        "workspace_commit": workspace_commit,
        "upstream_commit": commit,
    }


def validated_source_cache(project: str, commit: str) -> Path | None:
    """Return an explicitly configured, clean checkout at the pinned commit."""
    configured = os.environ.get(SOURCE_CACHE_ENV)
    if not configured:
        return None
    source = Path(configured).expanduser().resolve()
    return validate_source_checkout(
        source,
        project,
        commit,
        label=SOURCE_CACHE_ENV,
    )


def _validated_scan_roots(
    contract: dict[str, Any], source_checkout: Path
) -> list[tuple[Path, Path]]:
    """Resolve public scan roots without allowing aliases outside the checkout."""
    values = contract.get("scan_roots")
    if not isinstance(values, list):
        raise AdapterError("bench contract scan_roots must be a list")
    if not values:
        return []

    checkout = Path(source_checkout).resolve(strict=True)
    resolved_roots: list[tuple[Path, Path]] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise AdapterError(f"invalid scan root: {value!r}")
        relative = Path(value)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise AdapterError(f"invalid scan root: {value!r}")

        candidate = checkout / relative
        if candidate.is_symlink():
            raise AdapterError(f"scan root must not be a symlink: {value!r}")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise AdapterError(f"scan root does not exist: {value!r}") from exc
        if resolved == checkout or checkout not in resolved.parents:
            raise AdapterError(f"scan root escapes source checkout: {value!r}")
        mode = candidate.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise AdapterError(f"scan root is not a regular file or directory: {value!r}")

        for existing, _ in resolved_roots:
            if (
                relative == existing
                or relative in existing.parents
                or existing in relative.parents
            ):
                raise AdapterError(
                    f"scan roots overlap: {existing.as_posix()!r} and {value!r}"
                )
        resolved_roots.append((relative, resolved))
    return resolved_roots


def validate_source_checkout(
    source: Path,
    project: str,
    commit: str,
    *,
    label: str,
) -> Path:
    """Require one exact, clean Git top level before it can be copied."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise AdapterError(f"{label} is not a directory: {source}")
    top_level = _run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        cwd=source,
    )
    head = _run(["git", "-C", str(source), "rev-parse", "HEAD"], cwd=source)
    status = _run(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=source,
    )
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != source:
        raise AdapterError(f"{label} is not its Git top level: {source}")
    if head.returncode != 0 or head.stdout.strip() != commit:
        actual = head.stdout.strip() or "unavailable"
        raise AdapterError(
            f"{label} has wrong commit for {project}: expected {commit}, got {actual}"
        )
    if status.returncode != 0 or status.stdout.strip():
        raise AdapterError(f"{label} is not clean: {source}")
    return source


def _submission_schema_path(contract: dict[str, Any]) -> Path:
    value = contract.get("submission_schema")
    if not isinstance(value, str) or not value:
        raise AdapterError("bench contract does not declare a submission schema")
    relative = Path(value)
    if relative.is_absolute() or value == "." or ".." in relative.parts:
        raise AdapterError(f"invalid public submission schema path: {value!r}")
    return relative


def _require_disjoint_paths(
    *,
    workspace: Path,
    source_checkout: Path,
    benchmark_root: Path,
) -> None:
    resolved = {
        "workspace": workspace.resolve(strict=False),
        "source checkout": source_checkout.resolve(strict=False),
        "benchmark root": benchmark_root.resolve(strict=True),
    }
    items = list(resolved.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise AdapterError(
                    f"materialization paths overlap: {left_name}={left} and "
                    f"{right_name}={right}"
                )


def fetch_source_checkout(project: str, commit: str, destination: Path) -> None:
    """Materialize the clean source checkout the runner requires."""
    from adapters.zsoft_l1.adapter import BENCHMARK_ROOT as L1_ROOT

    locks = L1_ROOT / "source-locks"
    mapping = {
        "civetweb": "https://github.com/civetweb/civetweb",
        "libxml2": "https://gitlab.gnome.org/GNOME/libxml2",
        "umdk": None,
        "jiuwenswarm": None,
        "linux-rxrpc-sample": None,
    }
    url = mapping.get(project)
    lock = locks / f"{project}-{commit}.json"
    if not lock.is_file():
        matches = sorted(locks.glob(f"*-{commit}.json"))
        if len(matches) == 1:
            lock = matches[0]
    if lock.is_file():
        url = json.loads(lock.read_text()).get("url")
    if url is None:
        raise AdapterError(
            f"no source URL for {project}@{commit}; fetch a clean checkout to"
            f" {destination} manually (git clone + checkout {commit})"
        )
    if url.endswith(".tar.gz"):
        # tarball: fetch via mirror, then fetch the real commit object so
        # HEAD can equal the bench commit exactly (the launcher enforces it)
        mirror = url.replace(
            "https://github.com", "https://gh-proxy.com/https://github.com"
        )
        archive = destination.with_suffix(".tar.gz")
        completed = subprocess.run(
            ["curl", "-sL", "--max-time", "600", "-o", str(archive), mirror],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AdapterError(f"source download failed: {completed.stderr[-300:]}")
        destination.mkdir(parents=True)
        subprocess.run(
            ["tar", "xzf", str(archive), "-C", str(destination),
             "--strip-components=1"],
            check=True,
        )
        archive.unlink(missing_ok=True)
        repo_url = url.rsplit("/archive/", 1)[0]
        subprocess.run(
            ["git", "-C", str(destination), "init", "-q"], check=True
        )
        subprocess.run(
            ["git", "-C", str(destination), "remote", "add", "origin", repo_url],
            check=True,
        )
        fetch = subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin",
             commit],
            capture_output=True, text=True,
        )
        if fetch.returncode != 0:
            raise AdapterError(
                f"cannot fetch commit {commit}: {fetch.stderr[-300:]}"
            )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "-q", "-f", commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "remote", "remove", "origin"],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "clone", "-q", url, str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "-q", commit],
            check=True,
        )


def evaluate_workspace(
    workspace: Path, upstream_root: Path, mode: str
) -> dict[str, Any]:
    from adapters.portable import append_history, claim_evaluator_call

    if mode not in {"public", "final"}:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    started = time.monotonic()
    workspace = Path(workspace).expanduser().absolute()
    metadata = json.loads((workspace / "task.json").read_text(encoding="utf-8"))
    runtime_dir, budget = claim_evaluator_call(workspace, mode)
    ensure_single_final_claim(mode, budget)
    submission = workspace / ARTIFACT_NAME
    public_diagnostics = validate_detect_submission(
        submission,
        scan_roots=zsoft_blind.validated_scan_roots_config(
            metadata.get("scan_roots")
        ),
    )
    format_valid = diagnostics_valid(public_diagnostics)
    if mode == "public":
        report = {
            "schema_version": 1,
            "task_id": metadata["task_id"],
            "mode": mode,
            "valid": format_valid,
            "primary_metric": {
                "name": GOAL_PLUS_PROCESS_METRIC,
                "value": 1.0 if format_valid else 0.0,
                "direction": "maximize",
            },
            GOAL_PLUS_PROCESS_METRIC: 1.0 if format_valid else 0.0,
            "public_diagnostics": public_diagnostics,
            "budget": budget,
            "duration_seconds": time.monotonic() - started,
            "evaluated_at": _utc_now(),
        }
        append_history(runtime_dir, report)
        return report

    benchmark_root = _resolve_benchmark_root(upstream_root)
    project = metadata["project_id"]
    commit = metadata["commit"]
    valid = format_valid
    score_payload: dict[str, Any] | None = None
    message = "ok" if valid else "candidate artifact failed public validation"
    if valid:
        try:
            scored_submission = _snapshot_submission(
                submission,
                runtime_dir / f"submission-{budget['total_claimed']:06d}",
            )
        except (AdapterError, OSError) as exc:
            valid = False
            message = f"candidate artifact is unsafe: {exc}"
        else:
            scored = _run(
                [
                    sys.executable,
                    "scripts/score_submission.py",
                    str(scored_submission),
                    "--project",
                    project,
                    "--commit",
                    commit,
                    "--release",
                    "0.1.0",
                    "--track",
                    "tp",
                ],
                cwd=benchmark_root,
                timeout=OFFICIAL_EVALUATOR_TIMEOUT_SECONDS,
            )
            (runtime_dir / "score.stdout").write_text(
                scored.stdout, encoding="utf-8"
            )
            (runtime_dir / "score.stderr").write_text(
                scored.stderr, encoding="utf-8"
            )
            if scored.returncode != 0:
                valid = False
                message = f"scoring failed: {scored.stderr[-300:]}"
            else:
                try:
                    score_payload = json.loads(scored.stdout)
                except json.JSONDecodeError:
                    valid = False
                    message = "official scorer did not emit JSON"

    f1 = float(score_payload.get("f1", 0.0)) if score_payload else 0.0
    report = {
        "schema_version": 1,
        "task_id": metadata["task_id"],
        "mode": mode,
        "valid": valid,
        "primary_metric": {
            "name": PRIMARY_METRIC, "value": f1, "direction": DIRECTION,
        },
        PRIMARY_METRIC: f1,
        "zsoft_score": score_payload,
        "message": message,
        "format_valid": format_valid,
        "public_diagnostics": public_diagnostics,
        "budget": budget,
        "duration_seconds": time.monotonic() - started,
        "evaluated_at": _utc_now(),
    }
    append_history(runtime_dir, report)
    return report


def _snapshot_submission(source: Path, destination: Path) -> Path:
    """Copy direct regular files without following worker-created links."""
    destination.mkdir(mode=0o700, exist_ok=False)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AdapterError("submission snapshots require O_NOFOLLOW support")
    with os.scandir(source) as entries:
        for entry in entries:
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise AdapterError(f"submission entry is a symlink: {entry.name}")
            if not entry.is_file(follow_symlinks=False):
                raise AdapterError(
                    f"submission entry is not a regular file: {entry.name}"
                )
            descriptor = os.open(entry_path, os.O_RDONLY | nofollow)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise AdapterError(
                        f"submission entry changed type while reading: {entry.name}"
                    )
                with os.fdopen(descriptor, "rb", closefd=False) as candidate:
                    payload = candidate.read()
            finally:
                os.close(descriptor)
            (destination / entry.name).write_bytes(payload)
    return destination


def _framework_version(benchmark_root: Path) -> str:
    return (benchmark_root / "FRAMEWORK_VERSION").read_text(
        encoding="utf-8"
    ).strip()


def git_commit(path: Path) -> str:
    """Report the framework ref, or a normal commit for shared runtimes."""
    target = Path(path).expanduser().absolute()
    completed = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    benchmark_root = _resolve_benchmark_root(target)
    return f"zsoft-detect-framework-{_framework_version(benchmark_root)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--upstream-root", type=Path, required=True)
    materialize.add_argument("--workspace", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--workspace", type=Path, required=True)
    evaluate.add_argument("--upstream-root", type=Path, required=True)
    evaluate.add_argument("--mode", choices=("public", "final"), default="public")
    evaluate.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "materialize":
        print(
            json.dumps(
                materialize_workspace(args.upstream_root, args.workspace), indent=2
            )
        )
        return 0
    report = evaluate_workspace(args.workspace, args.upstream_root, args.mode)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
