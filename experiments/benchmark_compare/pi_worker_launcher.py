#!/usr/bin/env python3
"""Benchmark-owned Bubblewrap launcher for Goal Plus Pi workers."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socketserver
import stat
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_GOAL_PLUS_WORKER_LAUNCHER_ENV = "GOAL_PLUS_PI_WORKER_LAUNCHER"
SANDBOX_POLICY_ENV = "BENCH_GOAL_PLUS_PI_SANDBOX"
TOOL_SOCKET_ENV = "BENCH_GOAL_PLUS_PI_TOOL_SOCKET"
REAL_PI_BIN_ENV = "BENCH_GOAL_PLUS_REAL_PI_BIN"
LAUNCH_CONTEXT_VERSION = 1
_MAX_PROXY_REQUEST_BYTES = 1024 * 1024
_MAX_UNIX_SOCKET_PATH_BYTES = 103
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PATH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_RESERVED_ENV_NAMES = {
    "GIT_DIR",
    "GIT_OPTIONAL_LOCKS",
    "GIT_WORK_TREE",
    "GOAL_PLUS_AGENT_SESSION_ID",
    "GOAL_PLUS_PI_ROLE",
    "GOAL_PLUS_ROOT",
    "HOME",
    "PATH",
    "TMPDIR",
    LEGACY_GOAL_PLUS_WORKER_LAUNCHER_ENV,
    SANDBOX_POLICY_ENV,
    TOOL_SOCKET_ENV,
    REAL_PI_BIN_ENV,
    "BENCH_GOAL_PLUS_PACKAGE",
    "BENCH_GOAL_PLUS_BUILD",
}
_WORKER_TOOLS = {
    "goal_plus_search_get_agent_context",
    "goal_plus_search_get_global_evidence",
    "goal_plus_search_reference_open",
    "goal_plus_search_stage_shared_tool",
    "goal_plus_search_copy_shared_tool",
    "goal_plus_search_get_evidence_detail",
    "goal_plus_search_read_shared_cache",
    "goal_plus_search_append_shared_cache",
    "goal_plus_search_run_verifier",
    "goal_plus_search_list_iterations",
}
_SESSION_SCOPED_TOOLS = _WORKER_TOOLS
_TOOL_PROXY_BIN = Path(__file__).resolve().parent / "bin" / "goal-plus-pi-tool"
_SANDBOX_TOOL_BIN = Path("/opt/bench-goal-plus/bin")
_SANDBOX_GIT_DIR = Path("/opt/bench-goal-plus/git-admin")
_BLIND_PUBLIC_METRIC = "format_valid"
_BLIND_RESPONSE_REJECTED = {
    "ok": False,
    "error": "worker tool response is unavailable",
}
_BLIND_BLOCKED_TOOLS = {
    "goal_plus_search_get_evidence_detail",
}
_BLIND_SYSTEM_PROMPT = (
    "This ZSoft Search worker has a permanent benchmark-owned confidentiality "
    "boundary. The official evaluator and official metric never enter the worker "
    "trajectory. Treat direct verifier and iteration responses as opaque submission "
    "receipts. Global Evidence contains only schema-filtered public format-check "
    "Evidence and safe peer Views; use it as reference and independently verify it. "
    "Do not infer official outcomes from timing, Git metadata, workspace state, or "
    "tool errors. Work only from public task files and source mounted in this sandbox."
)
_OPAQUE_RESULTS_LEDGER = "iteration\tcommit\tstate\n"
_BLIND_CACHE_SECTION_FIELDS = {
    "key",
    "title",
    "requirement",
    "cardinality",
    "requires_evidence",
}
_BLIND_CACHE_ENTRY_FIELDS = {
    "key",
    "title",
    "section_key",
    "content",
    "writer",
    "record_id",
    "superseded_by",
    "evidence_ref",
}
_BLIND_CONTEXT_SOURCE_FIELDS = {
    "agent_session_id",
    "best_iteration",
    "candidate_id",
    "candidate_task",
    "execution_generation",
    "agent_harness",
    "runtime_provider",
    "inner_agent",
    "iteration_count",
    "latest_result",
    "metric_direction",
    "metric_name",
    "objective",
    "recent_iterations",
    "reference_workspaces",
    "result_count",
    "result_ledger_source",
    "results_tsv",
    "resume",
    "run_id",
    "shared_cache",
    "supplemental_evaluation_enabled",
    "tool_family_catalog",
    "workspace_access",
    "workspace_ledger_projection",
}
_BLIND_CANDIDATE_TASK_SOURCE_FIELDS = {
    "acceptance",
    "allowed_files",
    "denied_files",
    "hypothesis",
    "instructions",
    "process_environment",
    "share_out_dir",
    "workspace",
    "task_skill_paths",
}
_BLIND_CANDIDATE_TASK_OUTPUT_FIELDS = {
    "allowed_files",
    "denied_files",
    "workspace",
}
_BLIND_VERIFIER_SOURCE_FIELDS = {
    "allocation_decision",
    "agent_session_id",
    "aggregate_score",
    "attempt_artifact_ref",
    "attempt_id",
    "attempt_iteration",
    "best_artifact_ref",
    "best_git_head",
    "best_iteration",
    "candidate_id",
    "changed_outside_allowed",
    "commit",
    "disposition",
    "failure_class",
    "global_evidence_entry_count",
    "global_evidence_injected",
    "global_evidence_snapshot",
    "global_evidence_warning",
    "hardcoding_suspected",
    "iteration",
    "parent_id",
    "process_passed",
    "promotion_passed",
    "run_id",
    "shared_cache_injected",
    "shared_cache_snapshot",
    "shared_cache_warning",
    "shared_tool_consumed_entries",
    "shared_tool_deduplicated_entries",
    "shared_tool_errors",
    "shared_tool_publish_status",
    "shared_tool_staged_bytes",
    "shared_tool_staged_entries",
    "shared_tool_staged_file_count",
    "state",
    "toolization_advisories",
    "toolization_decision",
    "touched_denied_files",
    "validity_passed",
    "verifier_results",
    "workspace_git_head_after_settlement",
    "workspace_artifact_after_settlement",
}
_BLIND_VERIFIER_RECEIPT_FIELDS = {
    "agent_session_id",
    "candidate_id",
    "commit",
    "iteration",
    "run_id",
    "state",
}
_BLIND_ITERATION_SOURCE_FIELDS = {
    "allocation_decision_error",
    "allocation_decision_id",
    "adopted_tools",
    "adoption_confounded",
    "adapter_version",
    "agent_session_id",
    "artifact_hash",
    "artifact_clean",
    "artifact_ref",
    "artifact_status",
    "attempt_base_artifact_ref",
    "attempt_base_git_head",
    "attempt_changed_files",
    "candidate_id",
    "changed_files",
    "changed_outside_allowed",
    "commit",
    "created_at",
    "disposition",
    "exact_model_ref",
    "failure_class",
    "git_artifact_clean",
    "git_head",
    "git_status",
    "hypothesis",
    "iteration",
    "ledger_git_head",
    "ledger_artifact_ref",
    "log_paths",
    "metrics",
    "model_provenance",
    "process_passed",
    "provider_request_ids",
    "restored_to_artifact_ref",
    "restored_to_git_head",
    "restored_to_iteration",
    "run_id",
    "score",
    "settlement_id",
    "selected_model",
    "shared_tool_consumed_entries",
    "shared_tool_deduplicated_entries",
    "shared_tool_errors",
    "shared_tool_publish_status",
    "shared_tool_staged_bytes",
    "shared_tool_staged_entries",
    "shared_tool_staged_file_count",
    "shared_tools",
    "state",
    "summary",
    "toolization_advisories",
    "toolization_decision",
    "touched_denied_files",
    "workspace_git_head_after_settlement",
    "workspace_artifact_after_settlement",
}
_BLIND_ITERATION_RECEIPT_FIELDS = {
    "agent_session_id",
    "candidate_id",
    "commit",
    "iteration",
    "run_id",
    "state",
}
_BLIND_ITERATION_LEGACY_FIELDS = _BLIND_ITERATION_SOURCE_FIELDS - {
    "candidate_id",
    "commit",
    "run_id",
    "state",
}
_GLOBAL_EVIDENCE_FIELDS = {
    "annotation_result_available",
    "artifact_availability",
    "artifact_ref",
    "candidate_id",
    "commit",
    "disposition",
    "iteration",
    "reference_available",
    "score",
    "shared_tools",
    "view",
    "view_created_at",
}
_GLOBAL_EVIDENCE_DISPOSITIONS = {"keep", "retain", "discard", "failure"}
_SHARED_TOOL_FIELDS = {
    "candidate_id",
    "capability_ids",
    "coverage_keys",
    "created_at",
    "entrypoint",
    "family_id",
    "files",
    "iteration",
    "name",
    "publication_intent",
    "size_bytes",
    "snapshot_hash",
    "source_commit",
    "source_relative_path",
    "summary",
    "supersedes_tool_id",
    "tool_id",
    "tool_view",
    "version",
}
_TOOL_VIEW_FIELDS = {
    "adoption_steps",
    "capabilities",
    "dependencies",
    "entrypoint",
    "evidence_scope",
    "inputs",
    "limitations",
    "outputs",
    "snapshot_hash",
    "source_commit",
    "summary",
    "tool_id",
    "when_to_use",
}
_STAGED_SHARED_TOOL_FIELDS = {
    "capability_ids",
    "coverage_keys",
    "entrypoint",
    "file_count",
    "files",
    "name",
    "path_count",
    "publication_intent",
    "size_bytes",
    "source_paths",
    "staged_name",
    "staging_path",
    "supersedes_tool_id",
}
_COPIED_SHARED_TOOL_FIELDS = {
    "agent_session_id",
    "candidate_base_git_head",
    "copied_at",
    "inbox_path",
    "receipt_id",
    "snapshot_hash",
    "source_commit",
    "tool_id",
}
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_INVALID_BLIND_RESPONSE = object()


@dataclass(frozen=True)
class LaunchContext:
    run_id: str
    candidate_id: str
    agent_session_id: str
    workspace: Path

    @classmethod
    def from_json(cls, raw: str) -> LaunchContext:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid worker launch context JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError("worker launch context must be a JSON object")
        expected = {
            "schema_version",
            "run_id",
            "candidate_id",
            "agent_session_id",
            "workspace",
        }
        if set(payload) != expected:
            raise ValueError(
                "worker launch context fields do not match schema version 1"
            )
        if payload["schema_version"] != LAUNCH_CONTEXT_VERSION:
            raise ValueError(
                f"unsupported worker launch context version: {payload['schema_version']!r}"
            )
        values = {
            name: payload[name]
            for name in ("run_id", "candidate_id", "agent_session_id", "workspace")
        }
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError(
                "worker launch context identity fields must be non-empty strings"
            )
        for name in ("run_id", "candidate_id"):
            if not _PATH_ID.fullmatch(values[name]) or values[name] in {".", ".."}:
                raise ValueError(
                    f"worker launch context {name} is not a safe path identity"
                )
        workspace = Path(values["workspace"])
        if not workspace.is_absolute() or not workspace.is_dir():
            raise ValueError(
                "worker launch workspace must be an existing absolute path"
            )
        return cls(
            run_id=values["run_id"],
            candidate_id=values["candidate_id"],
            agent_session_id=values["agent_session_id"],
            workspace=workspace.resolve(),
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        root: Path,
        workspace: Path,
        session_id: str,
    ) -> LaunchContext:
        root = root.expanduser().absolute()
        if root.is_symlink():
            raise ValueError("GOAL_PLUS_ROOT must not be a symlink")
        root = root.resolve(strict=True)
        workspace = workspace.resolve(strict=True)
        try:
            relative = workspace.relative_to(root / "runs")
        except ValueError as exc:
            raise ValueError(
                "Pi worker cwd is outside the declared Goal Plus runtime"
            ) from exc
        if len(relative.parts) != 3 or relative.parts[1] != "workspace":
            raise ValueError("Pi worker cwd is not a Goal Plus candidate workspace")
        run_id, _, candidate_id = relative.parts
        for name, value in (("run_id", run_id), ("candidate_id", candidate_id)):
            if not _PATH_ID.fullmatch(value) or value in {".", ".."}:
                raise ValueError(f"Pi worker {name} is not a safe path identity")
        if not _PATH_ID.fullmatch(session_id) or session_id in {".", ".."}:
            raise ValueError("Pi worker session id is not a safe identity")

        identity_path = root
        for part in ("runs", run_id, "workspace", candidate_id):
            identity_path = identity_path / part
            if identity_path.is_symlink():
                raise ValueError("Pi worker runtime identity must not contain symlinks")

        session_path = root / "runs" / run_id / "agent_sessions" / f"{session_id}.json"
        if session_path.is_symlink() or not session_path.is_file():
            raise RuntimeError("Pi worker has no trusted Goal Plus session record")
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Pi worker session record is unreadable") from exc
        if not isinstance(session, dict):
            raise RuntimeError("Pi worker session record is malformed")
        expected = {
            "agent_session_id": session_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "agent_harness": "pi",
            "runtime_provider": "direct",
            "workspace": str(workspace),
        }
        if any(key in session for key in ("host", "host_handle")) or any(
            session.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("Pi worker session record does not match its process identity")
        launch = session.get("launch")
        if not isinstance(launch, dict) or launch.get("role", "worker") != "worker":
            raise RuntimeError("Pi worker session record does not describe a worker launch")
        return cls(
            run_id=run_id,
            candidate_id=candidate_id,
            agent_session_id=session_id,
            workspace=workspace,
        )


@dataclass(frozen=True)
class SandboxPolicy:
    engine: str
    workspace_access: str
    read_only_workspace_paths: tuple[str, ...]
    writable_workspace_paths: tuple[str, ...]
    pass_env: tuple[str, ...]
    # The launcher historically served only blind ZSoft workers. Keep missing
    # policy fields fail-closed; L1 opts into live binary feedback explicitly.
    evaluation_mode: str = "blind"

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> SandboxPolicy:
        raw = environment.get(SANDBOX_POLICY_ENV)
        if raw is None:
            raise RuntimeError(f"{SANDBOX_POLICY_ENV} is required")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{SANDBOX_POLICY_ENV} must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"{SANDBOX_POLICY_ENV} must be a JSON object")
        allowed = {
            "engine",
            "evaluation_mode",
            "workspace_access",
            "read_only_workspace_paths",
            "writable_workspace_paths",
            "pass_env",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                f"{SANDBOX_POLICY_ENV} has unsupported fields: {', '.join(unknown)}"
            )
        engine = payload.get("engine")
        if engine != "bubblewrap":
            raise ValueError(
                f"{SANDBOX_POLICY_ENV}.engine must be 'bubblewrap', got {engine!r}"
            )
        evaluation_mode = payload.get("evaluation_mode", "blind")
        if evaluation_mode not in {"visible", "blind"}:
            raise ValueError(
                f"{SANDBOX_POLICY_ENV}.evaluation_mode must be 'visible' or "
                f"'blind', got {evaluation_mode!r}"
            )
        workspace_access = payload.get("workspace_access")
        if workspace_access != "read_only":
            raise ValueError(
                f"{SANDBOX_POLICY_ENV}.workspace_access must be 'read_only', "
                f"got {workspace_access!r}"
            )
        read_only_paths = _workspace_path_list(
            payload.get("read_only_workspace_paths", []),
            field="read_only_workspace_paths",
        )
        writable_paths = _workspace_path_list(
            payload.get("writable_workspace_paths", []),
            field="writable_workspace_paths",
        )
        for read_only in read_only_paths:
            for writable in writable_paths:
                if _relative_paths_overlap(read_only, writable):
                    raise ValueError(
                        "sandbox read-only and writable workspace paths overlap: "
                        f"{read_only!r} and {writable!r}"
                    )
        for index, left in enumerate(writable_paths):
            for right in writable_paths[index + 1 :]:
                if _relative_paths_overlap(left, right):
                    raise ValueError(
                        "sandbox writable workspace paths overlap: "
                        f"{left!r} and {right!r}"
                    )
        for value in writable_paths:
            if value == ".tmp" or value.startswith(".tmp/"):
                raise ValueError(
                    "writable_workspace_paths must not override launcher-owned .tmp"
                )
        pass_env = _string_list(payload.get("pass_env", []), field="pass_env")
        invalid_env = [name for name in pass_env if not _ENV_NAME.fullmatch(name)]
        if invalid_env:
            raise ValueError(f"invalid pass_env name: {invalid_env[0]!r}")
        reserved_env = sorted(set(pass_env) & _RESERVED_ENV_NAMES)
        if reserved_env:
            raise ValueError(
                f"{SANDBOX_POLICY_ENV}.pass_env cannot override reserved variables: "
                + ", ".join(reserved_env)
            )
        return cls(
            engine=engine,
            workspace_access=workspace_access,
            read_only_workspace_paths=tuple(read_only_paths),
            writable_workspace_paths=tuple(writable_paths),
            pass_env=tuple(pass_env),
            evaluation_mode=evaluation_mode,
        )


@dataclass(frozen=True)
class PrivateGitAdmin:
    git_dir: Path
    work_tree: Path


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{SANDBOX_POLICY_ENV}.{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{SANDBOX_POLICY_ENV}.{field} must not contain duplicates")
    return value


def _workspace_path_list(value: Any, *, field: str) -> list[str]:
    paths = _string_list(value, field=field)
    for item in paths:
        candidate = Path(item)
        if candidate.is_absolute() or item in {"", "."} or ".." in candidate.parts:
            raise ValueError(
                f"{field} entries must be non-empty relative paths without '..': "
                f"{item!r}"
            )
    return paths


def _relative_paths_overlap(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def _blind_context_response(
    result: Any, context: LaunchContext
) -> dict[str, Any] | object:
    if not isinstance(result, dict) or not set(result) <= _BLIND_CONTEXT_SOURCE_FIELDS:
        return _INVALID_BLIND_RESPONSE
    required = {
        "agent_session_id",
        "run_id",
        "candidate_id",
        "candidate_task",
        "metric_name",
        "metric_direction",
    }
    if not required <= set(result):
        return _INVALID_BLIND_RESPONSE
    if (
        result["agent_session_id"] != context.agent_session_id
        or result["run_id"] != context.run_id
        or result["candidate_id"] != context.candidate_id
        or result["metric_name"] != _BLIND_PUBLIC_METRIC
        or result["metric_direction"] != "maximize"
    ):
        return _INVALID_BLIND_RESPONSE

    candidate_task = result["candidate_task"]
    if (
        not isinstance(candidate_task, dict)
        or not set(candidate_task) <= _BLIND_CANDIDATE_TASK_SOURCE_FIELDS
        or candidate_task.get("workspace") != str(context.workspace)
    ):
        return _INVALID_BLIND_RESPONSE
    projected_task: dict[str, Any] = {}
    for key in _BLIND_CANDIDATE_TASK_OUTPUT_FIELDS:
        if key not in candidate_task:
            continue
        value = candidate_task[key]
        if key in {
            "allowed_files",
            "denied_files",
            "expected_artifacts",
            "instructions",
            "parent_candidate_ids",
        }:
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                return _INVALID_BLIND_RESPONSE
        elif value is not None and not isinstance(value, str):
            return _INVALID_BLIND_RESPONSE
        projected_task[key] = value

    projected: dict[str, Any] = {
        "agent_session_id": context.agent_session_id,
        "run_id": context.run_id,
        "candidate_id": context.candidate_id,
        "workspace": str(context.workspace),
        "candidate_task": projected_task,
    }
    projected["metric_name"] = _BLIND_PUBLIC_METRIC
    projected["metric_direction"] = "maximize"
    shared_cache = result.get("shared_cache")
    if shared_cache is not None:
        projected_cache = _blind_shared_cache_snapshot(shared_cache)
        if projected_cache is _INVALID_BLIND_RESPONSE:
            return _INVALID_BLIND_RESPONSE
        projected["shared_cache"] = projected_cache
    return projected


def _blind_verifier_receipt(
    result: Any, context: LaunchContext
) -> dict[str, Any] | object:
    if isinstance(result, dict) and set(result) == _BLIND_VERIFIER_RECEIPT_FIELDS:
        if (
            result["run_id"] != context.run_id
            or result["candidate_id"] != context.candidate_id
            or result["agent_session_id"] != context.agent_session_id
            or type(result["iteration"]) is not int
            or result["iteration"] < 1
            or not isinstance(result["commit"], str)
            or _GIT_COMMIT.fullmatch(result["commit"]) is None
            or result["state"] != "recorded"
        ):
            return _INVALID_BLIND_RESPONSE
        return dict(result)
    if isinstance(result, dict) and (
        set(result)
        & (_BLIND_VERIFIER_RECEIPT_FIELDS - {"run_id", "candidate_id"})
    ):
        return _INVALID_BLIND_RESPONSE
    if (
        not isinstance(result, dict)
        or not set(result) <= _BLIND_VERIFIER_SOURCE_FIELDS
        or result.get("run_id") != context.run_id
        or result.get("candidate_id") != context.candidate_id
    ):
        return _INVALID_BLIND_RESPONSE
    receipt: dict[str, Any] = {
        "run_id": context.run_id,
        "candidate_id": context.candidate_id,
        "recorded": True,
    }
    if result.get("shared_cache_injected") is True:
        projected_cache = _blind_shared_cache_snapshot(
            result.get("shared_cache_snapshot")
        )
        if projected_cache is _INVALID_BLIND_RESPONSE:
            return _INVALID_BLIND_RESPONSE
        receipt["shared_cache_snapshot"] = projected_cache
    return receipt


def _blind_shared_cache_snapshot(
    snapshot: Any,
) -> dict[str, Any] | object:
    """Project an operator-frozen shared cache snapshot for a blind worker.

    The cache is an operator-defined fact plane: sections and entries are
    written only by peer candidates and the LLM verifier from public task
    material, never by the official evaluator. The projection keeps the
    template and the projected entries but performs a closed-field check so
    the host cannot smuggle arbitrary shapes through the cache channel.
    """
    if not isinstance(snapshot, dict) or not set(snapshot) <= {
        "sections", "entries", "total_records", "updates_allowed",
    }:
        return _INVALID_BLIND_RESPONSE
    sections = snapshot.get("sections")
    entries = snapshot.get("entries")
    if not isinstance(sections, list) or not isinstance(entries, list):
        return _INVALID_BLIND_RESPONSE
    projected_sections: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict) or not set(section) <= _BLIND_CACHE_SECTION_FIELDS:
            return _INVALID_BLIND_RESPONSE
        if not isinstance(section.get("key"), str) or not section["key"]:
            return _INVALID_BLIND_RESPONSE
        if not isinstance(section.get("title"), str):
            return _INVALID_BLIND_RESPONSE
        if not isinstance(section.get("requirement"), str):
            return _INVALID_BLIND_RESPONSE
        if section.get("cardinality") not in {"any", "latest"}:
            return _INVALID_BLIND_RESPONSE
        if type(section.get("requires_evidence")) is not bool:
            return _INVALID_BLIND_RESPONSE
        projected_sections.append(dict(section))
    projected_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not set(entry) <= _BLIND_CACHE_ENTRY_FIELDS:
            return _INVALID_BLIND_RESPONSE
        for field in ("key", "title", "section_key", "content", "writer", "record_id"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                return _INVALID_BLIND_RESPONSE
        for field in ("superseded_by", "evidence_ref"):
            if entry.get(field) is not None and not isinstance(entry[field], str):
                return _INVALID_BLIND_RESPONSE
        projected_entries.append(dict(entry))
    projected: dict[str, Any] = {
        "sections": projected_sections,
        "entries": projected_entries,
    }
    total_records = snapshot.get("total_records")
    if total_records is not None and type(total_records) is not int:
        return _INVALID_BLIND_RESPONSE
    if total_records is not None:
        projected["total_records"] = total_records
    return projected


def _blind_appended_shared_cache(
    result: Any,
) -> dict[str, Any] | object:
    if not isinstance(result, dict) or not set(result) <= {
        "section", "writer", "view",
    }:
        return _INVALID_BLIND_RESPONSE
    if not isinstance(result.get("section"), str) or not result["section"]:
        return _INVALID_BLIND_RESPONSE
    if not isinstance(result.get("writer"), str) or not result["writer"]:
        return _INVALID_BLIND_RESPONSE
    view = result.get("view")
    if not isinstance(view, list):
        return _INVALID_BLIND_RESPONSE
    projected_view = []
    for entry in view:
        if not isinstance(entry, dict) or not set(entry) <= _BLIND_CACHE_ENTRY_FIELDS:
            return _INVALID_BLIND_RESPONSE
        for field in ("key", "title", "section_key", "content", "writer", "record_id"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                return _INVALID_BLIND_RESPONSE
        for field in ("superseded_by", "evidence_ref"):
            if entry.get(field) is not None and not isinstance(entry[field], str):
                return _INVALID_BLIND_RESPONSE
        projected_view.append(dict(entry))
    return {
        "section": result["section"],
        "writer": result["writer"],
        "view": projected_view,
    }


def _blind_iteration_receipts(
    result: Any, context: LaunchContext
) -> list[dict[str, Any]] | object:
    if not isinstance(result, list):
        return _INVALID_BLIND_RESPONSE
    receipts: list[dict[str, Any]] = []
    for item in result:
        if isinstance(item, dict) and set(item) == _BLIND_ITERATION_RECEIPT_FIELDS:
            if (
                item["run_id"] != context.run_id
                or item["candidate_id"] != context.candidate_id
                or item["agent_session_id"] != context.agent_session_id
                or type(item["iteration"]) is not int
                or item["iteration"] < 1
                or not isinstance(item["commit"], str)
                or _GIT_COMMIT.fullmatch(item["commit"]) is None
                or item["state"] != "recorded"
            ):
                return _INVALID_BLIND_RESPONSE
            receipts.append(dict(item))
            continue
        if (
            not isinstance(item, dict)
            or not set(item) <= _BLIND_ITERATION_LEGACY_FIELDS
        ):
            return _INVALID_BLIND_RESPONSE
        iteration = item.get("iteration")
        if type(iteration) is not int or iteration < 1:
            return _INVALID_BLIND_RESPONSE
        receipts.append({"iteration": iteration, "recorded": True})
    return receipts


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _project_tool_view(result: Any, tool: dict[str, Any]) -> dict[str, Any] | object:
    if not isinstance(result, dict) or set(result) != _TOOL_VIEW_FIELDS:
        return _INVALID_BLIND_RESPONSE
    if (
        result.get("tool_id") != tool.get("tool_id")
        or result.get("snapshot_hash") != tool.get("snapshot_hash")
        or result.get("source_commit") != tool.get("source_commit")
    ):
        return _INVALID_BLIND_RESPONSE
    for field in (
        "adoption_steps",
        "capabilities",
        "dependencies",
        "inputs",
        "limitations",
        "outputs",
    ):
        if not _is_string_list(result.get(field)):
            return _INVALID_BLIND_RESPONSE
    for field in ("evidence_scope", "summary", "tool_id", "when_to_use"):
        if not isinstance(result.get(field), str) or not result[field]:
            return _INVALID_BLIND_RESPONSE
    if result.get("entrypoint") is not None and not isinstance(
        result["entrypoint"], str
    ):
        return _INVALID_BLIND_RESPONSE
    return dict(result)


def _project_shared_tool(
    result: Any, evidence_candidate_id: str
) -> dict[str, Any] | object:
    if not isinstance(result, dict) or set(result) != _SHARED_TOOL_FIELDS:
        return _INVALID_BLIND_RESPONSE
    if (
        result.get("candidate_id") != evidence_candidate_id
        or not isinstance(result.get("tool_id"), str)
        or not result["tool_id"]
        or not isinstance(result.get("family_id"), str)
        or not result["family_id"]
        or type(result.get("version")) is not int
        or result["version"] < 1
        or type(result.get("iteration")) is not int
        or result["iteration"] < 1
        or not isinstance(result.get("snapshot_hash"), str)
        or _HEX_DIGEST.fullmatch(result["snapshot_hash"]) is None
        or type(result.get("size_bytes")) is not int
        or result["size_bytes"] < 0
        or not isinstance(result.get("created_at"), str)
        or not _is_safe_relative_path(result.get("source_relative_path"))
    ):
        return _INVALID_BLIND_RESPONSE
    source_commit = result.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str)
        or _GIT_COMMIT.fullmatch(source_commit) is None
    ):
        return _INVALID_BLIND_RESPONSE
    for field in ("capability_ids", "coverage_keys", "files"):
        if not _is_string_list(result.get(field)):
            return _INVALID_BLIND_RESPONSE
    if not all(_is_safe_relative_path(value) for value in result["files"]):
        return _INVALID_BLIND_RESPONSE
    for field in ("entrypoint", "summary", "supersedes_tool_id"):
        if result.get(field) is not None and not isinstance(result[field], str):
            return _INVALID_BLIND_RESPONSE
    if result.get("publication_intent") not in {
        "new",
        "capability_extension",
        "adoption_fix",
        "contract_change",
    }:
        return _INVALID_BLIND_RESPONSE
    tool_view = _project_tool_view(result.get("tool_view"), result)
    if tool_view is _INVALID_BLIND_RESPONSE:
        return _INVALID_BLIND_RESPONSE
    return {**result, "tool_view": tool_view}


def _blind_global_evidence(result: Any) -> list[dict[str, Any]] | object:
    if not isinstance(result, list):
        return _INVALID_BLIND_RESPONSE
    projected: list[dict[str, Any]] = []
    for entry in result:
        if (
            not isinstance(entry, dict)
            or set(entry) != _GLOBAL_EVIDENCE_FIELDS
        ):
            return _INVALID_BLIND_RESPONSE
        candidate_id = entry.get("candidate_id")
        score = entry.get("score")
        view = entry.get("view")
        view_created_at = entry.get("view_created_at")
        if (
            not isinstance(candidate_id, str)
            or _PATH_ID.fullmatch(candidate_id) is None
            or type(entry.get("iteration")) is not int
            or entry["iteration"] < 1
            or not isinstance(entry.get("commit"), str)
            or _GIT_COMMIT.fullmatch(entry["commit"]) is None
            or entry.get("disposition") not in _GLOBAL_EVIDENCE_DISPOSITIONS
            or (
                score is not None
                and (
                    type(score) not in {int, float}
                    or float(score) not in {0.0, 1.0}
                )
            )
            or (view is not None and not isinstance(view, str))
            or (view_created_at is not None and not isinstance(view_created_at, str))
            or ((view is None) != (view_created_at is None))
            or not isinstance(entry.get("shared_tools"), list)
        ):
            return _INVALID_BLIND_RESPONSE
        shared_tools: list[dict[str, Any]] = []
        for tool in entry["shared_tools"]:
            projected_tool = _project_shared_tool(tool, candidate_id)
            if projected_tool is _INVALID_BLIND_RESPONSE:
                return _INVALID_BLIND_RESPONSE
            shared_tools.append(projected_tool)
        projected_entry = dict(entry)
        for field in (
            "annotation_result_available", "artifact_availability",
            "artifact_ref", "reference_available",
        ):
            projected_entry.pop(field)
        projected_entry["shared_tools"] = shared_tools
        projected.append(projected_entry)
    return projected


def _blind_staged_shared_tool(
    result: Any, context: LaunchContext
) -> dict[str, Any] | object:
    if not isinstance(result, dict) or set(result) != _STAGED_SHARED_TOOL_FIELDS:
        return _INVALID_BLIND_RESPONSE
    staged_name = result.get("staged_name")
    if (
        not isinstance(staged_name, str)
        or _PATH_ID.fullmatch(staged_name) is None
        or staged_name in {".", ".."}
        or not _is_string_list(result.get("source_paths"))
        or not _is_string_list(result.get("files"))
        or any(
            type(result.get(field)) is not int or result[field] < 0
            for field in ("file_count", "path_count", "size_bytes")
        )
        or not _is_string_list(result.get("capability_ids"))
        or not _is_string_list(result.get("coverage_keys"))
    ):
        return _INVALID_BLIND_RESPONSE
    expected = context.workspace / ".tmp" / "share-out" / staged_name
    staging_path = result.get("staging_path")
    try:
        staging_matches = (
            isinstance(staging_path, str)
            and Path(staging_path).resolve(strict=True) == expected.resolve(strict=True)
        )
    except (OSError, RuntimeError):
        staging_matches = False
    if not staging_matches:
        return _INVALID_BLIND_RESPONSE
    if not all(
        _is_safe_relative_path(value) and value.startswith(".tmp/tool-drafts/")
        for value in result["source_paths"]
    ) or not all(_is_safe_relative_path(value) for value in result["files"]):
        return _INVALID_BLIND_RESPONSE
    projected = dict(result)
    projected.pop("staging_path")
    projected["staged"] = True
    return projected


def _blind_copied_shared_tool(
    result: Any, context: LaunchContext
) -> dict[str, Any] | object:
    if not isinstance(result, dict) or set(result) != _COPIED_SHARED_TOOL_FIELDS:
        return _INVALID_BLIND_RESPONSE
    receipt_id = result.get("receipt_id")
    if (
        result.get("agent_session_id") != context.agent_session_id
        or not isinstance(receipt_id, str)
        or _PATH_ID.fullmatch(receipt_id) is None
        or not isinstance(result.get("tool_id"), str)
        or not result["tool_id"]
        or not isinstance(result.get("snapshot_hash"), str)
        or _HEX_DIGEST.fullmatch(result["snapshot_hash"]) is None
        or not isinstance(result.get("candidate_base_git_head"), str)
        or _GIT_COMMIT.fullmatch(result["candidate_base_git_head"]) is None
        or not isinstance(result.get("copied_at"), str)
    ):
        return _INVALID_BLIND_RESPONSE
    source_commit = result.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str)
        or _GIT_COMMIT.fullmatch(source_commit) is None
    ):
        return _INVALID_BLIND_RESPONSE
    relative_inbox = Path(".tmp") / "shared-tools" / receipt_id
    expected = context.workspace / relative_inbox
    inbox_path = result.get("inbox_path")
    try:
        inbox_matches = (
            isinstance(inbox_path, str)
            and Path(inbox_path).resolve(strict=True) == expected.resolve(strict=True)
            and expected.is_dir()
        )
    except (OSError, RuntimeError):
        inbox_matches = False
    if not inbox_matches:
        return _INVALID_BLIND_RESPONSE
    projected = dict(result)
    projected.pop("agent_session_id")
    projected.pop("candidate_base_git_head")
    projected["inbox_path"] = relative_inbox.as_posix()
    return projected


def _blind_tool_response(
    tool: str, result: Any, context: LaunchContext
) -> Any:
    if tool == "goal_plus_search_get_agent_context":
        return _blind_context_response(result, context)
    if tool == "goal_plus_search_run_verifier":
        return _blind_verifier_receipt(result, context)
    if tool == "goal_plus_search_list_iterations":
        return _blind_iteration_receipts(result, context)
    if tool == "goal_plus_search_get_global_evidence":
        return _blind_global_evidence(result)
    if tool == "goal_plus_search_stage_shared_tool":
        return _blind_staged_shared_tool(result, context)
    if tool == "goal_plus_search_copy_shared_tool":
        return _blind_copied_shared_tool(result, context)
    if tool == "goal_plus_search_read_shared_cache":
        return _blind_shared_cache_snapshot(result)
    if tool == "goal_plus_search_append_shared_cache":
        return _blind_appended_shared_cache(result)
    return _INVALID_BLIND_RESPONSE


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _run_host_tool(
    root: Path,
    tool: str,
    args: dict[str, Any],
    environment: Mapping[str, str],
    *,
    host_capability: str | None = None,
) -> Any:
    python = environment.get("GOAL_PLUS_PYTHON")
    if not python or not Path(python).is_absolute():
        raise RuntimeError("worker proxy requires the installed Goal Plus Python")
    command = [python, "-I", "-m", "goal_plus.pi_tool", "--root", str(root), tool]
    if host_capability is not None:
        command.extend(["--host-entrypoint", "--host-capability", host_capability])
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        input=json.dumps(args, ensure_ascii=False, separators=(",", ":")),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        if len(detail) > 500:
            detail = f"{detail[:500]}..."
        raise RuntimeError(detail or "host tool call failed")
    return json.loads(completed.stdout)


class WorkerToolProxy:
    def __init__(
        self,
        *,
        root: Path,
        context: LaunchContext,
        socket_dir: Path,
        environment: Mapping[str, str] | None = None,
        evaluation_mode: str = "blind",
    ) -> None:
        self.root = root
        self.context = context
        self.socket_dir = socket_dir
        self.socket_path = socket_dir / "tool.sock"
        if evaluation_mode not in {"visible", "blind"}:
            raise ValueError(f"unsupported proxy evaluation mode: {evaluation_mode}")
        self.evaluation_mode = evaluation_mode
        self.host_environment = dict(os.environ if environment is None else environment)
        for name in (
            "GIT_DIR",
            "GIT_OPTIONAL_LOCKS",
            "GIT_WORK_TREE",
            "GOAL_PLUS_PI_WORKER_CONTINUE_UNTIL_MS",
            LEGACY_GOAL_PLUS_WORKER_LAUNCHER_ENV,
            TOOL_SOCKET_ENV,
        ):
            self.host_environment.pop(name, None)
        self.host_environment["GOAL_PLUS_PI_ROLE"] = "worker"
        self.host_environment["GOAL_PLUS_AGENT_SESSION_ID"] = context.agent_session_id
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.socket_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        (self.socket_dir / "runtime/runs").mkdir(parents=True)
        proxy = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline(_MAX_PROXY_REQUEST_BYTES + 1)
                if len(raw) > _MAX_PROXY_REQUEST_BYTES:
                    response = {"ok": False, "error": "proxy request is too large"}
                else:
                    try:
                        request = json.loads(raw.decode("utf-8"))
                        response = proxy.dispatch(request)
                    except Exception as exc:  # noqa: BLE001
                        response = (
                            dict(_BLIND_RESPONSE_REJECTED)
                            if proxy.evaluation_mode == "blind"
                            else {
                                "ok": False,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            }
                        )
                self.wfile.write(
                    (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                )

        self._server = _ThreadingUnixServer(str(self.socket_path), Handler)
        os.chmod(self.socket_path, 0o600)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"zsoft-pi-tool-proxy-{self.context.agent_session_id}",
            daemon=True,
        )
        self._thread.start()

    def dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("proxy request must be a JSON object")
        if request.get("host_entrypoint") is True:
            return self._dispatch_host_callback(request)
        tool = request.get("tool")
        args = request.get("args")
        if tool == "host_mcp_manifest" and args == {}:
            manifest = _run_host_tool(self.root, tool, {}, self.host_environment)
            allowed = _WORKER_TOOLS - (
                _BLIND_BLOCKED_TOOLS if self.evaluation_mode == "blind" else set()
            )
            tools = []
            for item in manifest["tools"]:
                if item["name"] not in allowed:
                    continue
                if not isinstance(item.get("inputSchema"), dict):
                    raise ValueError("worker MCP tool requires an inputSchema object")
                # Python's manifest includes absent optional fields as null; MCP's
                # JavaScript client requires them to be omitted, not null. The worker
                # bridge also returns JSON as text, without structuredContent.
                tools.append(
                    {
                        key: value
                        for key, value in item.items()
                        if value is not None and key != "outputSchema"
                    }
                )
            return {"ok": True, "result": {"tools": tools}}
        if "native_session_id" in request and request["native_session_id"] != self.context.agent_session_id:
            raise PermissionError("Pi worker MCP requires the bound native session")
        if tool not in _WORKER_TOOLS:
            raise PermissionError(f"Pi worker proxy does not allow tool {tool!r}")
        if not isinstance(args, dict):
            raise TypeError("proxy tool args must be a JSON object")
        self._authorize(str(tool), args)
        if self.evaluation_mode == "blind" and tool in _BLIND_BLOCKED_TOOLS:
            return dict(_BLIND_RESPONSE_REJECTED)

        try:
            result = _run_host_tool(
                self.root,
                str(tool),
                args,
                self.host_environment,
            )
            if (
                tool == "goal_plus_search_get_agent_context"
                and self.evaluation_mode == "visible"
            ):
                self._project_worker_generation(result)
        except Exception:  # workers must not receive raw host exceptions
            return dict(_BLIND_RESPONSE_REJECTED)
        if self.evaluation_mode == "blind":
            result = _blind_tool_response(str(tool), result, self.context)
            if result is _INVALID_BLIND_RESPONSE:
                return dict(_BLIND_RESPONSE_REJECTED)
        return {
            "ok": True,
            "result": result,
        }

    def _project_worker_generation(self, result: Any) -> None:
        if (
            not isinstance(result, dict)
            or result.get("agent_session_id") != self.context.agent_session_id
            or result.get("run_id") != self.context.run_id
            or result.get("candidate_id") != self.context.candidate_id
            or type(result.get("execution_generation")) is not int
            or result["execution_generation"] < 0
            or not isinstance(result.get("candidate_task"), dict)
            or result["candidate_task"].get("workspace") != str(self.context.workspace)
        ):
            raise ValueError("worker generation requires a bound host context")
        destination = (
            self.socket_dir / "runtime/runs" / self.context.run_id
            / "candidates" / self.context.candidate_id / "candidate.json"
        )
        projection = {"execution_generation": result["execution_generation"]}
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != projection:
                raise ValueError("worker generation changed within one sandbox launch")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(projection), encoding="utf-8")

    def _dispatch_host_callback(self, request: dict[str, Any]) -> dict[str, Any]:
        tool = request.get("tool")
        args = request.get("args")
        token = request.get("host_capability")
        if (
            tool != "goal_plus_host_kick_internal_agents"
            or not isinstance(args, dict)
            or not set(args) <= {"run_id", "owner_pid"}
            or args.get("run_id") != self.context.run_id
            or not isinstance(token, str)
            or re.fullmatch(r"[0-9a-f-]{36}", token) is None
        ):
            raise PermissionError("Pi worker host callback requires its bound run")
        source = self.socket_dir / "runtime/host-entrypoint/pi" / f"{token}.json"
        destination = self.root / "host-entrypoint/pi" / f"{token}.json"
        created = False
        try:
            # Transfer only this worker's one-time native callback capability.
            if not source.resolve(strict=True).is_relative_to(self.socket_dir.resolve(strict=True)):
                raise ValueError("host callback capability escapes the worker proxy")
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("host callback capability must be a regular file")
                payload = stream.read(_MAX_PROXY_REQUEST_BYTES + 1)
            if len(payload) > _MAX_PROXY_REQUEST_BYTES:
                raise ValueError("host callback capability exceeds its size limit")
            decoded = json.loads(payload)
            if not isinstance(decoded, dict) or set(decoded) != {"token", "issued_at_ms"}:
                raise ValueError("invalid host callback capability")
            if decoded["token"] != token:
                raise ValueError("host callback capability identity mismatch")
            source.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(decoded, stream)
            _run_host_tool(
                self.root, tool, {"run_id": self.context.run_id},
                self.host_environment, host_capability=token,
            )
            return {"ok": True, "result": {"ok": True, "run_id": self.context.run_id}}
        except Exception:
            return dict(_BLIND_RESPONSE_REJECTED)
        finally:
            if created:
                destination.unlink(missing_ok=True)

    def _authorize(self, tool: str, args: dict[str, Any]) -> None:
        if (
            tool in _SESSION_SCOPED_TOOLS
            and args.get("agent_session_id") != self.context.agent_session_id
        ):
            raise PermissionError("Pi worker proxy requires the bound agent_session_id")
        if "run_id" in args and args["run_id"] != self.context.run_id:
            raise PermissionError("Pi worker proxy rejected a different run_id")
        if (
            tool == "goal_plus_search_run_verifier"
            and args.get("candidate_id") != self.context.candidate_id
        ):
            raise PermissionError("Pi worker proxy rejected a different candidate_id")
        if (
            tool == "goal_plus_search_list_iterations"
            and set(args) != {"agent_session_id"}
        ):
            raise PermissionError(
                "Pi iteration listing accepts only the bound agent_session_id"
            )
        if (
            tool == "goal_plus_search_read_shared_cache"
            and set(args) != {"agent_session_id"}
        ):
            raise PermissionError(
                "Pi shared cache reads accept only the bound agent_session_id"
            )
        if tool == "goal_plus_search_append_shared_cache":
            if not set(args) <= {
                "agent_session_id",
                "section",
                "content",
                "supersedes",
                "evidence_ref",
            }:
                raise PermissionError(
                    "Pi shared cache appends accept only the documented fields"
                )
            if not isinstance(args.get("section"), str) or not args["section"]:
                raise PermissionError("Pi shared cache appends require a section")
            if not isinstance(args.get("content"), str) or not args["content"]:
                raise PermissionError("Pi shared cache appends require content")
            for field in ("supersedes", "evidence_ref"):
                if args.get(field) is not None and not isinstance(args[field], str):
                    raise PermissionError(
                        "Pi shared cache appends require string optional fields"
                    )
        if (
            tool == "goal_plus_search_run_verifier"
            and args.get("scope", "process") != "process"
        ):
            raise PermissionError("Pi workers may only run process verifiers")

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.socket_path.unlink(missing_ok=True)
        try:
            self.socket_dir.rmdir()
        except OSError:
            pass


class BubblewrapWorker:
    def __init__(
        self,
        *,
        context: LaunchContext,
        policy: SandboxPolicy,
        command: list[str],
        environment: Mapping[str, str],
    ) -> None:
        self.context = context
        self.policy = policy
        self.command = command
        self.environment = dict(environment)
        self.root = _runtime_root(self.environment, context)
        proxy_base = _worker_proxy_base(self.environment)
        self.proxy = WorkerToolProxy(
            root=self.root,
            context=context,
            socket_dir=proxy_base / f"bgp-pi-{uuid.uuid4().hex[:16]}",
            environment=self.environment,
            evaluation_mode=policy.evaluation_mode,
        )
        self.private_git_admin: PrivateGitAdmin | None = None

    def prepare(self) -> tuple[list[str], dict[str, str]]:
        try:
            self.proxy.start()
            return self._build_command()
        except Exception:
            self.close()
            raise

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        if not self.command:
            raise ValueError("external worker launcher requires an original command")
        bwrap = shutil.which("bwrap", path=self.environment.get("PATH"))
        if bwrap is None:
            raise FileNotFoundError("ZSoft Pi worker launcher requires bwrap on PATH")
        executable = shutil.which(self.command[0], path=self.environment.get("PATH"))
        if executable is None:
            raise FileNotFoundError(f"Pi executable not found: {self.command[0]}")
        executable_path = Path(executable).absolute()
        executable_entrypoint = _executable_entrypoint(executable_path)
        pi_runtime_roots = tuple(
            dict.fromkeys(
                (
                    _executable_runtime_root(executable_path),
                    _executable_runtime_root(executable_entrypoint),
                )
            )
        )
        pi_runtime = pi_runtime_roots[0]
        extension = _command_path_argument(self.command, "-e")
        extension_bundle = extension.parent
        package = extension.parents[3]
        receipt_path = package / ".goal-plus-runtime.json"
        session_root = _command_path_argument(self.command, "--session-dir")
        session_id = _command_argument(self.command, "--session-id")
        if not extension.is_file():
            raise FileNotFoundError(f"Pi extension not found: {extension}")
        if _inside(extension_bundle, self.root) or _inside(self.root, extension_bundle):
            raise ValueError("Pi extension bundle must be disjoint from GOAL_PLUS_ROOT")
        if self.root not in session_root.parents:
            raise ValueError(
                "Pi worker session directory must live under GOAL_PLUS_ROOT"
            )
        isolated_session = session_root / _safe_name(session_id)
        if isolated_session.is_symlink():
            raise RuntimeError("Pi worker session directory must not be a symlink")
        isolated_session.mkdir(parents=True, exist_ok=True)
        if (
            not isolated_session.is_dir()
            or session_root not in isolated_session.resolve(strict=True).parents
        ):
            raise RuntimeError(
                "Pi worker session directory escapes its declared session root"
            )
        command = _replace_command_argument(
            self.command,
            "--session-dir",
            str(isolated_session),
        )
        if not _TOOL_PROXY_BIN.is_file() or not os.access(_TOOL_PROXY_BIN, os.X_OK):
            raise FileNotFoundError(
                f"worker tool proxy is not executable: {_TOOL_PROXY_BIN}"
            )

        args = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--share-net",
            "--unshare-user",
        ]
        if _bwrap_supports_option(bwrap, "--disable-userns", self.environment):
            args.append("--disable-userns")
        args.extend(
            [
                "--cap-drop",
                "ALL",
                "--hostname",
                "zsoft-goal-plus-worker",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/run",
                "--dir",
                "/home",
                "--dir",
                "/home/pi",
            ]
        )
        created = {"/proc", "/dev", "/tmp", "/run", "/home", "/home/pi"}
        _mount_system(args)
        for runtime in pi_runtime_roots:
            if not _is_system_path(runtime):
                _add_bind(args, runtime, runtime, readonly=True, created=created)
        _add_tmpfs(args, self.root, created)
        protected_paths = _validated_workspace_paths(
            self.context.workspace,
            self.policy.read_only_workspace_paths,
            access="read-only",
            validate_links=True,
        )
        writable_paths = _validated_workspace_paths(
            self.context.workspace,
            self.policy.writable_workspace_paths,
            access="writable",
            validate_links=False,
        )
        scratch = _prepare_workspace_scratch(self.context.workspace)
        self.private_git_admin = _prepare_private_git_admin(
            self.context.workspace,
            self.root
            / "worker-sandbox-state"
            / self.context.run_id
            / self.context.candidate_id
            / "git-admin",
        )
        if self.private_git_admin is not None:
            _add_bind(
                args,
                self.private_git_admin.git_dir,
                _SANDBOX_GIT_DIR,
                readonly=False,
                created=created,
            )
        _add_bind(
            args,
            self.context.workspace,
            self.context.workspace,
            readonly=True,
            created=created,
        )
        _mount_private_workspace_metadata(
            args,
            workspace=self.context.workspace,
            private_git_admin=self.private_git_admin,
            private_root=self.proxy.socket_dir,
            created=created,
        )
        _add_bind(
            args,
            isolated_session,
            isolated_session,
            readonly=False,
            created=created,
        )
        _add_bind(
            args,
            self.proxy.socket_dir,
            self.proxy.socket_dir,
            readonly=False,
            created=created,
        )
        generation_root = self.proxy.socket_dir / "runtime/runs"
        _add_bind(
            args, generation_root, generation_root, readonly=True, created=created,
        )
        _add_bind(
            args,
            extension_bundle,
            extension_bundle,
            readonly=True,
            created=created,
        )
        # Expose installed assets and a transport shim, never the host store or Python runtime.
        receipt = json.loads(receipt_path.read_text())
        python = Path(receipt["python"])
        if (
            receipt.get("schema") != "goal-plus.managed-runtime.v1"
            or not receipt.get("build") or not python.is_absolute()
            or self.environment.get("GOAL_PLUS_PYTHON", str(python)) != str(python)
        ):
            raise ValueError("Pi worker requires its installed Goal Plus runtime receipt")
        _add_bind(args, package / "assets", package / "assets", readonly=True, created=created)
        _add_bind(args, receipt_path, receipt_path, readonly=True, created=created)
        _add_bind(args, _TOOL_PROXY_BIN, python, readonly=True, created=created)
        _add_bind(
            args,
            _TOOL_PROXY_BIN.parent,
            _SANDBOX_TOOL_BIN,
            readonly=True,
            created=created,
        )

        pi_home_text = self.environment.get("PI_CODING_AGENT_DIR")
        if not pi_home_text:
            raise RuntimeError("PI_CODING_AGENT_DIR is required for ZSoft Pi workers")
        pi_home = Path(pi_home_text).expanduser().resolve()
        if not pi_home.is_dir():
            raise FileNotFoundError(f"PI_CODING_AGENT_DIR not found: {pi_home}")
        private_pi_home = _prepare_private_pi_home(
            pi_home,
            self.proxy.socket_dir / "pi-home",
        )
        _add_bind(args, private_pi_home, pi_home, readonly=False, created=created)
        private_models = private_pi_home / "models.json"
        if private_models.is_file():
            _add_bind(
                args,
                private_models,
                pi_home / "models.json",
                readonly=True,
                created=created,
            )

        for path in protected_paths:
            _add_bind(args, path, path, readonly=True, created=created)
        for path in writable_paths:
            _add_bind(args, path, path, readonly=False, created=created)
        _add_bind(args, scratch, scratch, readonly=False, created=created)

        sandbox_env = _sandbox_environment(
            self.environment,
            policy=self.policy,
            pi_runtime=pi_runtime,
            socket_path=self.proxy.socket_path,
            runtime_root=self.proxy.socket_dir / "runtime",
            agent_session_id=self.context.agent_session_id,
            private_git_admin=self.private_git_admin,
        )
        sandbox_env["BENCH_GOAL_PLUS_PACKAGE"] = str(package)
        sandbox_env["BENCH_GOAL_PLUS_BUILD"] = receipt["build"]
        args.extend(
            [
                "--chdir",
                str(self.context.workspace),
                "--",
                str(executable_path),
                *command[1:],
            ]
        )
        return args, sandbox_env

    def close(self) -> None:
        self.proxy.close()
        _remove_private_runtime_tree(self.proxy.socket_dir)


def _runtime_root(
    environment: Mapping[str, str],
    context: LaunchContext,
) -> Path:
    value = environment.get("GOAL_PLUS_ROOT")
    if not value:
        raise RuntimeError("GOAL_PLUS_ROOT is required for ZSoft Pi workers")
    root = Path(value).resolve()
    if not root.is_dir():
        raise ValueError("GOAL_PLUS_ROOT must be an existing runtime directory")
    expected_workspace_path = (
        root / "runs" / context.run_id / "workspace" / context.candidate_id
    )
    current = root
    for part in ("runs", context.run_id, "workspace", context.candidate_id):
        current = current / part
        if current.is_symlink():
            raise ValueError("worker runtime identity path must not contain symlinks")
    expected_workspace = expected_workspace_path.resolve(strict=False)
    if (
        not expected_workspace_path.is_dir()
        or context.workspace.resolve(strict=True) != expected_workspace
    ):
        raise ValueError(
            "worker workspace does not match GOAL_PLUS_ROOT identity: "
            f"{expected_workspace}"
        )
    return root


def _worker_proxy_base(environment: Mapping[str, str]) -> Path:
    candidates: list[Path] = []
    configured = environment.get("XDG_RUNTIME_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    user_runtime = Path("/run") / "user" / str(os.geteuid())
    if user_runtime not in candidates:
        candidates.append(user_runtime)

    for candidate in candidates:
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_dir()
        ):
            continue
        metadata = candidate.stat()
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or not os.access(candidate, os.W_OK | os.X_OK)
        ):
            continue
        base = candidate / "bench-goal-plus"
        if base.is_symlink():
            continue
        base.mkdir(mode=0o700, exist_ok=True)
        base.chmod(0o700)
        sample = base / "bgp-pi-0000000000000000" / "tool.sock"
        if len(os.fsencode(sample)) <= _MAX_UNIX_SOCKET_PATH_BYTES:
            return base
    raise RuntimeError(
        "ZSoft Pi worker launcher requires a private short XDG_RUNTIME_DIR or "
        "/run/user/<uid> for its Unix socket"
    )


def _command_argument(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
        value = command[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Pi worker command is missing {option}") from exc
    if not value:
        raise ValueError(f"Pi worker command has an empty {option}")
    return value


def _command_path_argument(command: list[str], option: str) -> Path:
    path = Path(_command_argument(command, option))
    if not path.is_absolute():
        raise ValueError(f"Pi worker command {option} path must be absolute")
    return path.resolve()


def _replace_command_argument(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    result[result.index(option) + 1] = value
    return result


def _safe_name(value: str) -> str:
    if not _PATH_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError("Pi worker session id does not contain a safe path name")
    return value


def _executable_runtime_root(executable: Path) -> Path:
    if executable.parent.name == "bin":
        return executable.parent.parent.resolve()
    if (
        executable.parent.name == ".bin"
        and executable.parent.parent.name == "node_modules"
    ):
        return executable.parent.parent.parent.resolve()
    resolved = executable.resolve()
    for parent in resolved.parents:
        if parent.name == "node_modules":
            return parent.parent
    if resolved.parent.name == "bin":
        return resolved.parent.parent
    return resolved


def _executable_entrypoint(executable: Path) -> Path:
    return executable.resolve(strict=True)


def _bwrap_supports_option(
    executable: str,
    option: str,
    environment: Mapping[str, str],
) -> bool:
    completed = subprocess.run(
        [executable, "--help"],
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to inspect Bubblewrap options: {completed.stderr.strip()}"
        )
    return option in f"{completed.stdout}\n{completed.stderr}".split()


def _is_system_path(path: Path) -> bool:
    roots = map(Path, ("/usr", "/bin", "/sbin", "/lib", "/lib64"))
    return any(path == root or root in path.parents for root in roots)


def _mount_system(args: list[str]) -> None:
    args.extend(["--ro-bind", "/usr", "/usr"])
    for value in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(value)
        if path.is_symlink():
            args.extend(["--symlink", os.readlink(path), value])
        elif path.exists():
            args.extend(["--ro-bind", value, value])
    args.extend(["--dir", "/etc"])
    for value in (
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/etc/localtime",
        "/etc/ssl/certs",
        "/etc/alternatives",
    ):
        if Path(value).exists():
            args.extend(["--ro-bind", value, value])


def _ensure_parent_dirs(args: list[str], destination: Path, created: set[str]) -> None:
    for parent in reversed(destination.parents[:-1]):
        value = str(parent)
        if value != "/" and value not in created:
            args.extend(["--dir", value])
            created.add(value)


def _add_bind(
    args: list[str],
    source: Path,
    destination: Path,
    *,
    readonly: bool,
    created: set[str],
) -> None:
    source = source.resolve()
    destination = destination.absolute()
    _ensure_parent_dirs(args, destination, created)
    args.extend(["--ro-bind" if readonly else "--bind", str(source), str(destination)])
    created.add(str(destination))


def _add_tmpfs(args: list[str], destination: Path, created: set[str]) -> None:
    destination = destination.absolute()
    _ensure_parent_dirs(args, destination, created)
    args.extend(["--tmpfs", str(destination)])
    created.add(str(destination))


def _git_output(cwd: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_confined_symlinks(root: Path, *, label: str) -> None:
    root = root.absolute()
    if root.is_symlink():
        raise RuntimeError(f"{label} root must not be a symlink")
    resolved_root = root.resolve(strict=True)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in (*dirnames, *filenames):
            path = parent / name
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{label} contains an unresolvable symlink: "
                    f"{path.relative_to(root)}"
                ) from exc
            if not _inside(target, resolved_root):
                raise RuntimeError(
                    f"{label} symlink escapes its root: "
                    f"{path.relative_to(root)} -> {os.readlink(path)}"
                )


def _validated_workspace_paths(
    workspace: Path,
    relative_paths: tuple[str, ...],
    *,
    access: str,
    validate_links: bool,
) -> list[Path]:
    resolved_workspace = workspace.resolve(strict=True)
    result: list[Path] = []
    for relative in relative_paths:
        path = workspace / relative
        if not path.exists():
            raise FileNotFoundError(
                f"sandbox {access} workspace path not found: {relative}"
            )
        if path.is_symlink():
            raise RuntimeError(
                f"sandbox {access} workspace path must not be a symlink: {relative}"
            )
        resolved = path.resolve(strict=True)
        if not _inside(resolved, resolved_workspace):
            raise RuntimeError(
                f"sandbox {access} workspace path escapes the workspace: {relative}"
            )
        if validate_links and path.is_dir():
            _validate_confined_symlinks(
                path,
                label=f"sandbox {access} workspace path {relative}",
            )
        if access == "writable":
            required_access = os.W_OK | (os.X_OK if path.is_dir() else 0)
            if not os.access(path, required_access):
                raise PermissionError(
                    f"sandbox writable workspace path is not writable: {relative}"
                )
        result.append(path)
    return result


def _prepare_workspace_scratch(workspace: Path) -> Path:
    scratch = workspace / ".tmp"
    if scratch.is_symlink():
        raise RuntimeError("launcher-owned workspace .tmp must not be a symlink")
    scratch.mkdir(mode=0o700, exist_ok=True)
    scratch.chmod(0o700)
    if not scratch.is_dir() or not os.access(scratch, os.W_OK | os.X_OK):
        raise PermissionError("launcher-owned workspace .tmp is not writable")
    return scratch


def _remove_private_runtime_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists() and not path.is_symlink():
        return
    _restore_private_tree_owner_access(path)
    shutil.rmtree(path)


def _restore_private_tree_owner_access(path: Path) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return
    owner_access = stat.S_IRUSR | stat.S_IWUSR
    if stat.S_ISDIR(mode):
        path.chmod(mode | owner_access | stat.S_IXUSR)
        with os.scandir(path) as entries:
            for entry in entries:
                _restore_private_tree_owner_access(Path(entry.path))
    else:
        path.chmod(mode | owner_access)


def _prepare_private_pi_home(source: Path, destination: Path) -> Path:
    symlink = next((path for path in source.rglob("*") if path.is_symlink()), None)
    if symlink is not None:
        raise RuntimeError(
            "PI_CODING_AGENT_DIR must not contain symlinks: "
            f"{symlink.relative_to(source)}"
        )
    shutil.copytree(source, destination)
    destination.chmod(0o700)
    return destination


def _prepare_private_git_admin(
    workspace: Path,
    destination: Path,
) -> PrivateGitAdmin | None:
    workspace = workspace.resolve()
    if not _git_output(workspace, "rev-parse", "--verify", "HEAD"):
        raise RuntimeError(
            "ZSoft Pi worker sandbox cannot isolate the candidate Git state"
        )

    if destination.is_symlink():
        raise RuntimeError("candidate-private Git directory must not be a symlink")
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError("candidate-private Git path is not a directory")
        _validate_confined_symlinks(
            destination,
            label="candidate-private Git directory",
        )
        configured_worktree = _git_output(
            workspace,
            f"--git-dir={destination}",
            "config",
            "--path",
            "core.worktree",
        )
        private_head = _git_output(
            workspace,
            f"--git-dir={destination}",
            "rev-parse",
            "--verify",
            "HEAD",
        )
        isolated = _git_output(
            workspace,
            f"--git-dir={destination}",
            "config",
            "--get",
            "bench-goal-plus.isolated",
        )
        if (
            configured_worktree is None
            or Path(configured_worktree).resolve() != workspace
            or private_head is None
            or isolated != "true"
            or (destination / "objects" / "info" / "alternates").exists()
        ):
            raise RuntimeError(
                "existing candidate-private Git directory has an invalid identity"
            )
        return PrivateGitAdmin(
            git_dir=destination,
            work_tree=workspace,
        )

    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if destination.parent.is_symlink():
        raise RuntimeError("candidate-private Git parent must not be a symlink")
    subprocess.run(
        ["git", "init", "--bare", "-q", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    destination.chmod(0o700)
    git_command = ["git", f"--git-dir={destination}", f"--work-tree={workspace}"]
    identity_environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Bench Goal Plus",
        "GIT_AUTHOR_EMAIL": "bench-goal-plus@localhost",
        "GIT_COMMITTER_NAME": "Bench Goal Plus",
        "GIT_COMMITTER_EMAIL": "bench-goal-plus@localhost",
    }
    for command in (
        [*git_command, "config", "core.bare", "false"],
        [*git_command, "config", "core.worktree", str(workspace)],
        [*git_command, "config", "bench-goal-plus.isolated", "true"],
        [
            *git_command,
            "add",
            "-A",
            "--",
            ".",
            ":(exclude)results.tsv",
            ":(exclude).tmp",
            ":(exclude).tmp/**",
        ],
    ):
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=identity_environment,
        )
    blob = subprocess.run(
        [*git_command, "hash-object", "-w", "--stdin"],
        input=_OPAQUE_RESULTS_LEDGER,
        check=True,
        capture_output=True,
        text=True,
        env=identity_environment,
    ).stdout.strip()
    subprocess.run(
        [
            *git_command,
            "update-index",
            "--add",
            "--cacheinfo",
            "100644",
            blob,
            "results.tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=identity_environment,
    )
    tree = subprocess.run(
        [*git_command, "write-tree"],
        check=True,
        capture_output=True,
        text=True,
        env=identity_environment,
    ).stdout.strip()
    private_head = subprocess.run(
        [*git_command, "commit-tree", tree, "-m", "public candidate baseline"],
        check=True,
        capture_output=True,
        text=True,
        env=identity_environment,
    ).stdout.strip()
    for command in (
        [*git_command, "update-ref", "refs/heads/sandbox", private_head],
        [*git_command, "symbolic-ref", "HEAD", "refs/heads/sandbox"],
        [*git_command, "reset", "--mixed", "HEAD"],
    ):
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=identity_environment,
        )
    return PrivateGitAdmin(
        git_dir=destination,
        work_tree=workspace,
    )


def _mount_private_workspace_metadata(
    args: list[str],
    *,
    workspace: Path,
    private_git_admin: PrivateGitAdmin | None,
    private_root: Path,
    created: set[str],
) -> None:
    if private_git_admin is None:
        raise RuntimeError("ZSoft worker requires a candidate-private Git view")
    results_path = workspace / "results.tsv"
    if results_path.is_symlink() or not results_path.is_file():
        raise RuntimeError("ZSoft worker results.tsv must be a regular file")
    opaque_results = private_root / "opaque-results.tsv"
    opaque_results.write_text(_OPAQUE_RESULTS_LEDGER, encoding="utf-8")
    opaque_results.chmod(0o400)
    _add_bind(args, opaque_results, results_path, readonly=True, created=created)

    workspace_git = workspace / ".git"
    if workspace_git.is_symlink() or not workspace_git.exists():
        raise RuntimeError("ZSoft worker workspace must have Git metadata")
    if workspace_git.is_dir():
        _add_bind(
            args,
            private_git_admin.git_dir,
            workspace_git,
            readonly=False,
            created=created,
        )
    elif workspace_git.is_file():
        private_git_link = private_root / "workspace.git"
        private_git_link.write_text(
            f"gitdir: {_SANDBOX_GIT_DIR}\n",
            encoding="utf-8",
        )
        private_git_link.chmod(0o400)
        _add_bind(
            args,
            private_git_link,
            workspace_git,
            readonly=True,
            created=created,
        )
    else:
        raise RuntimeError("ZSoft worker workspace Git metadata is unsupported")


def _sandbox_environment(
    environment: Mapping[str, str],
    *,
    policy: SandboxPolicy,
    pi_runtime: Path,
    socket_path: Path,
    runtime_root: Path,
    agent_session_id: str,
    private_git_admin: PrivateGitAdmin | None,
) -> dict[str, str]:
    runtime_bin = pi_runtime / "bin" if pi_runtime.is_dir() else pi_runtime.parent
    result = {
        "HOME": "/home/pi",
        "TMPDIR": "/tmp",
        "LANG": environment.get("LANG", "C.UTF-8"),
        "LC_ALL": environment.get("LC_ALL", "C.UTF-8"),
        "TZ": environment.get("TZ", "UTC"),
        "PATH": os.pathsep.join(
            (
                str(_SANDBOX_TOOL_BIN),
                str(runtime_bin),
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            )
        ),
        TOOL_SOCKET_ENV: str(socket_path),
        "GOAL_PLUS_ROOT": str(runtime_root),
        "GOAL_PLUS_PI_ROLE": "worker",
        "GOAL_PLUS_AGENT_SESSION_ID": agent_session_id,
    }
    if private_git_admin is not None:
        result.update(
            {
                "GIT_DIR": str(_SANDBOX_GIT_DIR),
                "GIT_WORK_TREE": str(private_git_admin.work_tree),
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
    inherited_names = {
        "PI_CODING_AGENT_DIR",
        "GOAL_PLUS_PI_MODEL",
        "GOAL_PLUS_PI_WORKER_CONTINUE_UNTIL_MS",
        *policy.pass_env,
    }
    for name in sorted(inherited_names):
        if name in environment:
            result[name] = environment[name]
    return result


def run_launcher(
    context: LaunchContext,
    command: list[str],
    environment: Mapping[str, str],
) -> int:
    policy = SandboxPolicy.from_environment(environment)
    sandbox = BubblewrapWorker(
        context=context,
        policy=policy,
        command=command,
        environment=environment,
    )
    wrapped_command, wrapped_environment = sandbox.prepare()
    process: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, Any] = {}

    def stop_child(signum: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    try:
        process = subprocess.Popen(
            wrapped_command,
            cwd=context.workspace,
            env=wrapped_environment,
        )
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, stop_child)
        returncode = process.wait()
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        sandbox.close()


def _real_pi_binary(environment: Mapping[str, str]) -> Path:
    value = environment.get(REAL_PI_BIN_ENV)
    if not value:
        raise RuntimeError(f"{REAL_PI_BIN_ENV} is required by the bench Pi shim")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{REAL_PI_BIN_ENV} must be an absolute path")
    path = path.absolute()
    resolved = path.resolve(strict=True)
    shim = Path(__file__).resolve().parent / "bin" / "pi"
    if path == shim.absolute() or resolved == shim.resolve():
        raise RuntimeError("bench Pi shim cannot point back to itself")
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"{REAL_PI_BIN_ENV} is not executable")
    return path


def _pi_mode(command: list[str]) -> str | None:
    try:
        return _command_argument(command, "--mode")
    except ValueError:
        return None


def _shim_worker_launch(
    command: list[str],
    environment: Mapping[str, str],
    cwd: Path,
) -> tuple[LaunchContext, list[str]] | None:
    if environment.get("GOAL_PLUS_PI_ROLE") != "worker":
        return None
    if _pi_mode(command) != "rpc":
        raise RuntimeError("Goal Plus Pi worker must use RPC mode")
    root_value = environment.get("GOAL_PLUS_ROOT")
    if not root_value:
        raise RuntimeError("GOAL_PLUS_ROOT is required for a Goal Plus Pi worker")
    session_id = _command_argument(command, "--session-id")
    if not _PATH_ID.fullmatch(session_id) or session_id in {".", ".."}:
        raise ValueError("Goal Plus Pi worker session id is not a safe identity")
    context = LaunchContext.from_runtime(
        root=Path(root_value),
        workspace=cwd,
        session_id=session_id,
    )
    real_pi = _real_pi_binary(environment)
    wrapped = [str(real_pi), *command]
    policy = SandboxPolicy.from_environment(environment)
    if policy.evaluation_mode == "blind":
        wrapped.extend(["--append-system-prompt", _BLIND_SYSTEM_PROMPT])
    return context, wrapped


def run_pi_shim(
    argv: list[str],
    environment: Mapping[str, str],
    cwd: Path,
) -> int:
    command = list(argv)
    planned = _shim_worker_launch(command, environment, cwd)
    if planned is not None:
        context, worker_command = planned
        return run_launcher(context, worker_command, environment)
    real_pi = _real_pi_binary(environment)
    os.execve(str(real_pi), [str(real_pi), *command], dict(environment))
    raise AssertionError("os.execve unexpectedly returned")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-json", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        context = LaunchContext.from_json(args.context_json)
        return run_launcher(context, command, os.environ)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
