"""Validate the native checkpoint for an explicitly requested two-stage Pi fork."""

from __future__ import annotations

import json
from pathlib import Path


def preparation_ticket(root: Path, sessions: Path) -> dict:
    from goal_plus.candidate_host import candidate_host_queries
    from goal_plus.execution.pi_checkpoint import active_ancestry, select_completed_checkpoint
    from goal_plus.goal_plus import FileGoalPlusRuntime

    runtime = FileGoalPlusRuntime(root)
    records = [runtime.status(p.parent.name) for p in (root / "goal-plus").glob("*/goal.json")]
    records = [r for r in records if not runtime.is_terminal(r)]
    if len(records) != 1:
        raise RuntimeError("ForkPreparationRejected: ambiguous or terminal Goal")
    record = records[0]
    active = record.active_session
    if (record.command_config.spawn_mode != "fork" or record.linked_search is None
            or active is None or active.agent_harness != "pi" or active.state != "attached"):
        raise RuntimeError("ForkPreparationRejected: not the attached Pi fork Main")
    run_id = record.linked_search.run_id
    run = json.loads((root / "runs" / run_id / "run.json").read_text())
    if run.get("dispatch_group") is not None or run.get("invalidated_at"):
        raise RuntimeError("ForkPreparationRejected: group already started or run invalidated")
    candidate_host_queries(root).dispatch_batch(run_id)
    preparation = _current_preparation_event(runtime, record, run_id)
    files = list(sessions.glob(f"*_{active.session_id}.jsonl"))
    if len(files) != 1:
        raise RuntimeError("ForkPreparationRejected: missing exact native Main file")
    entries = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    ancestry = active_ancestry(entries, entries[-1].get("id"))
    marker = next((e for e in ancestry if e.get("type") == "custom"
                   and e.get("customType") == "goal-plus-fork-checkpoint"
                   and e.get("data", {}).get("run_id") == run_id), None)
    expected_marker = {"run_id": run_id, "goal_plus_id": record.goal_plus_id,
                       "goal_revision": record.goal_revision, "control_version": record.control.version}
    if marker is None or any(marker.get("data", {}).get(k) != v for k, v in expected_marker.items()):
        raise RuntimeError("ForkPreparationRejected: missing current preparation marker")
    select_completed_checkpoint(entries, entries[-1].get("id"), run_id=run_id,
                                require_preparation_marker=True)
    if runtime.status(record.goal_plus_id) != record:
        raise RuntimeError("ForkPreparationRejected: Goal changed during admission")
    return {"goal_plus_id": record.goal_plus_id, "goal_revision": record.goal_revision,
            "session_id": active.session_id, "control_version": record.control.version,
            "fork_preparation_event_id": preparation["event_id"]}


def _current_preparation_event(runtime: object, record: object, run_id: str) -> dict:
    events = runtime.state.read_events(record.goal_plus_id)
    latest = events[-1] if events else None
    payload = (latest or {}).get("payload") or {}
    if ((latest or {}).get("event_type") != "status_changed"
            or payload.get("reason") != "fork_checkpoint_ready"
            or not any(item.get("run_id") == run_id
                       for item in payload.get("evidence") or [] if isinstance(item, dict))):
        raise RuntimeError("ForkPreparationRejected: ordinary stop or changed preparation")
    return latest


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    args = parser.parse_args()
    try:
        ticket = preparation_ticket(args.root, args.sessions)
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(json.dumps(ticket, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
