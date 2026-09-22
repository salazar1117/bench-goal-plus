#!/usr/bin/env python3
"""Self-contained public structure checks for ZSoft workspaces."""

from __future__ import annotations

import ast
import json
import os
import stat
from pathlib import Path
from typing import Any

PUBLIC_CHECKER_NAME = "public_check.py"
PUBLIC_METRIC = "format_valid"
DETECT_VALIDATION_KIND = "detect_json_findings"
L1_VALIDATION_KIND = "l1_python_script"


def read_regular_file(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes | None, int | None, str | None]:
    """Read one direct regular file without following a link."""
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None, None, "file is missing"
    except OSError:
        return None, None, "file metadata is unavailable"
    if stat.S_ISLNK(file_stat.st_mode):
        return None, None, "file must not be a symlink"
    if not stat.S_ISREG(file_stat.st_mode):
        return None, None, "file is not a regular file"
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        return None, None, "platform cannot enforce non-symlink reads"
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError:
        return None, None, "file could not be opened safely"
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            return None, None, "file changed type while being read"
        if max_bytes is not None and opened_stat.st_size > max_bytes:
            return None, opened_stat.st_size, "file exceeds configured size limit"
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = (
                handle.read(max_bytes + 1)
                if max_bytes is not None
                else handle.read()
            )
        if max_bytes is not None and len(payload) > max_bytes:
            return None, len(payload), "file exceeds configured size limit"
    except OSError:
        return None, None, "file could not be read safely"
    finally:
        os.close(descriptor)
    return payload, len(payload), None


def _normalized_repo_path(value: str) -> str:
    """Mirror the evaluator's path normalization for repo-relative paths."""
    result = value.strip()
    while result.startswith("./"):
        result = result[2:]
    return result


def validated_scan_roots_config(value: Any) -> list[str]:
    """Validate the scan-root list from task metadata, fail closed."""
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError("task scan_roots must be a list of non-empty strings")
    roots: list[str] = []
    for item in value:
        normalized = _normalized_repo_path(item)
        relative = Path(normalized)
        if not normalized or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"task scan root is not a confined relative path: {item!r}"
            )
        roots.append(relative.as_posix())
    return roots


def _within_scan_roots(path: str, roots: list[str]) -> bool:
    normalized = _normalized_repo_path(path)
    return any(
        normalized == root or normalized.startswith(root + "/") for root in roots
    )


def _finding_shape_errors(
    payload: Any, scan_roots: list[str] | None = None
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["finding must be a JSON object"]
    expected = {"location", "bug_type", "root_cause"}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        if missing:
            errors.append("missing fields: " + ", ".join(missing))
        if extra:
            errors.append("unexpected fields: " + ", ".join(extra))

    location = payload.get("location")
    location_fields = {"path", "function", "start_line", "end_line"}
    if not isinstance(location, dict):
        errors.append("location must be an object")
    else:
        missing = sorted(location_fields - set(location))
        extra = sorted(set(location) - location_fields)
        if missing:
            errors.append("location missing fields: " + ", ".join(missing))
        if extra:
            errors.append("location has unexpected fields: " + ", ".join(extra))
        location_path = location.get("path")
        if not isinstance(location_path, str) or not location_path:
            errors.append("location.path must be a non-empty string")
        elif location_path.startswith("/") or ".." in Path(location_path).parts:
            errors.append("location.path must be a confined relative path")
        elif scan_roots and not _within_scan_roots(location_path, scan_roots):
            errors.append(
                "location.path must be repo-relative (no workspace prefix)"
                " and start with one of the scan roots: "
                + ", ".join(scan_roots)
            )
        function = location.get("function")
        if function is not None and (
            not isinstance(function, str) or not function
        ):
            errors.append("location.function must be null or a non-empty string")
        for field in ("start_line", "end_line"):
            value = location.get(field)
            if type(value) is not int or value < 1:
                errors.append(f"location.{field} must be an integer of at least 1")

    bug_type = payload.get("bug_type")
    if not isinstance(bug_type, str) or not bug_type:
        errors.append("bug_type must be a non-empty string")

    root_cause = payload.get("root_cause")
    root_cause_fields = {"cause", "trigger", "impact"}
    if not isinstance(root_cause, dict):
        errors.append("root_cause must be an object")
    else:
        missing = sorted(root_cause_fields - set(root_cause))
        extra = sorted(set(root_cause) - root_cause_fields)
        if missing:
            errors.append("root_cause missing fields: " + ", ".join(missing))
        if extra:
            errors.append(
                "root_cause has unexpected fields: " + ", ".join(extra)
            )
        for field in sorted(root_cause_fields):
            value = root_cause.get(field)
            if not isinstance(value, str) or not value:
                errors.append(f"root_cause.{field} must be a non-empty string")
    return errors


def validate_detect_submission(
    directory: Path, *, scan_roots: list[str] | None = None
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "check": DETECT_VALIDATION_KIND,
        "json_file_count": 0,
        "errors": [],
    }
    errors: list[dict[str, Any]] = diagnostics["errors"]
    try:
        directory_stat = directory.lstat()
    except FileNotFoundError:
        errors.append({"message": "submission directory is missing"})
        return diagnostics
    except OSError:
        errors.append({"message": "submission directory metadata is unavailable"})
        return diagnostics
    if stat.S_ISLNK(directory_stat.st_mode):
        errors.append({"message": "submission directory must not be a symlink"})
        return diagnostics
    if not stat.S_ISDIR(directory_stat.st_mode):
        errors.append({"message": "submission artifact is not a directory"})
        return diagnostics

    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        errors.append({"message": "submission directory could not be read safely"})
        return diagnostics
    for entry in entries:
        entry_path = Path(entry.path)
        if entry.name == ".gitkeep":
            placeholder, size, placeholder_error = read_regular_file(entry_path)
            if placeholder_error is not None or size != 0 or placeholder != b"":
                errors.append(
                    {"file": entry.name, "message": "invalid directory placeholder"}
                )
            continue
        if entry.is_symlink():
            errors.append({"file": entry.name, "message": "entry must not be a symlink"})
            continue
        if not entry.is_file(follow_symlinks=False):
            errors.append(
                {"file": entry.name, "message": "entry is not a direct regular file"}
            )
            continue
        if not entry.name.endswith(".json"):
            errors.append(
                {"file": entry.name, "message": "finding file must end in .json"}
            )
            continue
        diagnostics["json_file_count"] += 1
        raw, _size, read_error = read_regular_file(entry_path)
        if read_error is not None or raw is None:
            errors.append({"file": entry.name, "message": read_error})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append({"file": entry.name, "message": "file is not UTF-8"})
            continue
        try:
            finding = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "file": entry.name,
                    "message": f"invalid JSON at line {exc.lineno} column {exc.colno}",
                }
            )
            continue
        for message in _finding_shape_errors(finding, scan_roots):
            errors.append({"file": entry.name, "message": message})
    return diagnostics


def validate_l1_artifact(artifact: Path, max_bytes: int) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "check": L1_VALIDATION_KIND,
        "size_limit_bytes": max_bytes,
        "size_bytes": None,
        "errors": [],
    }
    errors: list[dict[str, Any]] = diagnostics["errors"]
    raw, size, read_error = read_regular_file(artifact, max_bytes=max_bytes)
    diagnostics["size_bytes"] = size
    if read_error == "file exceeds configured size limit":
        errors.append(
            {"message": f"artifact exceeds the {max_bytes}-byte public limit"}
        )
        return diagnostics
    if read_error is not None or raw is None or size is None:
        errors.append({"message": read_error})
        return diagnostics
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        errors.append({"message": "artifact is not UTF-8"})
        return diagnostics
    try:
        ast.parse(text, filename=artifact.name, mode="exec")
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        suffix = f" at line {line}" if isinstance(line, int) else ""
        errors.append({"message": "artifact is not parseable Python" + suffix})
    return diagnostics


def diagnostics_valid(diagnostics: dict[str, Any]) -> bool:
    return diagnostics.get("errors") == []


def ensure_single_final_claim(mode: str, budget: dict[str, Any]) -> None:
    """Fail closed before a repeated official-evaluator invocation."""
    if mode == "final" and budget.get("final_claimed") != 1:
        raise RuntimeError("official final evaluation may only be claimed once")


def _safe_artifact_path(workspace: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("task artifact_name must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("task artifact_name must be a confined relative path")
    return workspace / relative


def run_public_check(workspace: Path) -> dict[str, Any]:
    metadata = json.loads((workspace / "task.json").read_text(encoding="utf-8"))
    artifact = _safe_artifact_path(workspace, metadata.get("artifact_name"))
    kind = metadata.get("public_validation_kind")
    if kind == DETECT_VALIDATION_KIND:
        scan_roots = validated_scan_roots_config(metadata.get("scan_roots"))
        diagnostics = validate_detect_submission(artifact, scan_roots=scan_roots)
    elif kind == L1_VALIDATION_KIND:
        max_bytes = metadata.get("submission_max_bytes")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("task submission_max_bytes must be a positive integer")
        diagnostics = validate_l1_artifact(artifact, max_bytes)
    else:
        raise ValueError("task public_validation_kind is unsupported")
    valid = diagnostics_valid(diagnostics)
    return {
        PUBLIC_METRIC: 1.0 if valid else 0.0,
        "valid": valid,
        "public_diagnostics": diagnostics,
    }


def main() -> int:
    workspace = Path(__file__).resolve().parent
    try:
        report = run_public_check(workspace)
    except (OSError, ValueError, json.JSONDecodeError):
        report = {
            PUBLIC_METRIC: 0.0,
            "valid": False,
            "public_diagnostics": {
                "check": "configuration",
                "errors": [{"message": "public checker configuration is invalid"}],
            },
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
