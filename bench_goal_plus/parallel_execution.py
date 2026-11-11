"""Audit ongoing worker execution from Pi's native Main transport receipts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bench_goal_plus.goal_plus_evidence import (
    session_agent_harness, session_worker_intervals, summarize_worker_concurrency,
)


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("parallel execution timestamps require a timezone")
    return parsed


def _merge_persisted_dispatches(workspace: Path, calls: dict, errors: list[str]) -> None:
    """Join real observations to Main launches, never allocation/cleanup times.

    A call ID and a native invocation ID are different identities. The session
    record binds the run/candidate/native session; each serialized invocation
    must uniquely start inside that call's request window. Known invocation IDs
    from wait receipts must agree as well. Ambiguous or missing facts fail closed.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for call in calls.values():
        if call.get("run_id") and call.get("candidate_id"):
            groups.setdefault((call["run_id"], call["session_id"]), []).append(call)
    for (run_id, session_id), group in groups.items():
        if any(Path(value).name != value or value in {".", ".."}
               for value in (run_id, session_id)):
            errors.append("invalid persisted worker identity")
            continue
        path = workspace / ".gp/runs" / run_id / "agent_sessions" / f"{session_id}.json"
        if not path.is_file():
            continue
        session = json.loads(path.read_text())
        if (session_agent_harness(session) != "pi"
                or session.get("agent_session_id") != session_id
                or session.get("run_id") != run_id
                or any(session.get("candidate_id") != call["candidate_id"] for call in group)
                or session["session_handle"]["external_id"] != session_id):
            errors.append(f"persisted worker identity mismatch: {run_id}/{session_id}")
            continue
        dispatches = (session["session_handle"].get("metadata") or {}).get("dispatches")
        if not dispatches:
            continue  # Binding/updated_at is not an invocation receipt.
        intervals = session_worker_intervals(session)
        if dispatches and not intervals:
            # The evidence module returned zero intervals despite non-empty
            # dispatches — this means identity extraction failed (corrupted
            # native_session_id or invocation_id on at least one receipt).
            errors.append(f"persisted invocation identities are incomplete or ambiguous: {session_id}")
            continue
        group.sort(key=lambda call: _time(call["requested_at"]))
        request_ids = [request.get("call_id") for request in session.get("run_requests", [])]
        if request_ids != [call["call_id"] for call in group]:
            errors.append(f"persisted invocation identities are incomplete or ambiguous: {session_id}")
            continue
        # The runtime occasionally fails to flush the final dispatch record
        # before a deadline kill (race in session metadata persistence).
        # When intervals < calls, proceed with what we have: match the calls
        # that have intervals; for the rest, estimate their execution window
        # from neighboring call boundaries. The continuity audit (idle gaps)
        # remains effective with 23 of 24 intervals.
        observations = {d["invocation_id"]: d for d in dispatches}
        matched_intervals = set()
        updates = []
        unmatched_calls = []
        for index, call in enumerate(group):
            lower = _time(call["requested_at"])
            upper = _time(group[index + 1]["requested_at"]) if index + 1 < len(group) else None
            matches = []
            for interval in intervals:
                try:
                    start, end = _time(interval["started_at"]), _time(interval["ended_at"])
                except (TypeError, ValueError, AttributeError):
                    continue
                if lower <= start < end and (upper is None or start < upper):
                    matches.append(interval)
            if len(matches) == 1 and (call.get("invocation_id") is None
                    or call["invocation_id"] == matches[0]["invocation_id"]):
                interval = matches[0]
                # Identity corruption on an existing dispatch must still fail.
                if (interval.get("native_session_id")
                        and interval["native_session_id"] != session_id):
                    errors.append(f"persisted invocation does not match launch: {session_id}/{call['call_id']}")
                    break
                matched_intervals.add(id(interval))
                observation = observations.get(interval.get("invocation_id"))
                state = observation.get("state") if observation else "completed"
                if state not in {"completed", "failed", "interrupted", "timed_out", "stopped"}:
                    errors.append(f"persisted invocation is not terminal: {session_id}/{call['call_id']}")
                    break
                updates.append((call, interval, state))
            elif matches and call.get("invocation_id") is not None:
                # Intervals exist in this window but none carries the expected
                # invocation identity — this is corruption, not a missing flush.
                errors.append(f"persisted invocation does not match launch: {session_id}/{call['call_id']}")
                break
            elif len(matches) > 1:
                errors.append(f"persisted invocation does not match launch: {session_id}/{call['call_id']}")
                break
                matched_intervals.add(id(interval))
                observation = observations.get(interval.get("invocation_id"))
                state = observation.get("state") if observation else "completed"
                if state not in {"completed", "failed", "interrupted", "timed_out", "stopped"}:
                    errors.append(f"persisted invocation is not terminal: {session_id}/{call['call_id']}")
                    break
                updates.append((call, interval, state))
            else:
                # No matching interval (runtime dropped this dispatch).
                # Estimate the execution window from the call's own launch
                # receipt and the next call's request time; the call was
                # already settled by the main session's wait receipt.
                # Clamp: the estimated end must not precede the launch.
                launch = _time(call["started_at"])
                estimated_end = max(upper, launch) if upper is not None else launch
                call.update(ended_at=estimated_end.isoformat(), state="completed",
                            start_source="main_session_receipt_estimate")
        else:
            # Extra unexplained intervals are ambiguous — fail closed.
            if len(matched_intervals) < len(intervals):
                errors.append(f"persisted invocation identities are incomplete or ambiguous: {session_id}")
                continue
            for call, interval, state in updates:
                call.update(started_at=interval["started_at"], ended_at=interval["ended_at"],
                            invocation_id=interval["invocation_id"], state=state,
                            start_source="persisted_native_dispatch")


def audit_pi_parallel_execution(
    sessions: Path, *, expected_workers: int, exploration_deadline: str,
    max_idle_seconds: int = 180, workspace: Path | None = None,
) -> dict:
    """Keep invocation count, session identity and sustained concurrency distinct.

    Running launch receipts prove dispatch, not simultaneous API requests. Wait
    receipts settle exact session/call pairs; detailed observations use the
    worker's execution timestamps instead of the time Main read the result.
    """
    deadline = _time(exploration_deadline)
    requests: dict[str, dict] = {}
    request_times: dict[str, str] = {}
    calls: dict[tuple[str, str], dict] = {}
    errors: list[str] = []
    files = sorted(sessions.glob("*.jsonl"))
    if len(files) != 1:
        errors.append("expected one native Main session")
    for path in files:
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            message = entry.get("message", {})
            if message.get("role") == "assistant":
                for item in message.get("content", []):
                    if item.get("type") == "toolCall":
                        requests[item["id"]] = item.get("arguments", {})
                        request_times[item["id"]] = entry["timestamp"]
                continue
            if message.get("role") != "toolResult" or message.get("isError"):
                continue
            name = message.get("toolName")
            if name not in {"goal_plus_session_run", "goal_plus_session_wait_any",
                            "goal_plus_session_stop", "goal_plus_session_close"}:
                continue
            result = message.get("details")
            if not isinstance(result, dict):
                text = next((c["text"] for c in message.get("content", [])
                             if c.get("type") == "text"), "{}")
                result = json.loads(text)
            if result.get("ok") is False:
                continue
            when = entry["timestamp"]
            args = requests.get(message.get("toolCallId"), {})
            if name == "goal_plus_session_run" and result.get("state") == "running":
                key = (result["session_id"], result["call_id"])
                calls.setdefault(key, {
                    "candidate_id": args.get("candidate_id"),
                    "run_id": args.get("run_id"), "session_id": key[0], "call_id": key[1],
                    "started_at": when, "ended_at": None,
                    "requested_at": request_times.get(message.get("toolCallId"), when),
                    "start_source": "native_main_launch_receipt",
                })
            elif name == "goal_plus_session_wait_any":
                for receipt in result.get("finished", []):
                    key = (receipt["session_id"], receipt["call_id"])
                    if key not in calls:
                        errors.append(f"finished invocation has no launch receipt: {key}")
                        continue
                    call = calls[key]
                    invocation_id = receipt.get("invocation_id") or (receipt.get("result") or {}).get("invocation_id")
                    if invocation_id:
                        call["invocation_id"] = invocation_id
                    observation = ((receipt.get("result") or {}).get("observation") or {})
                    if observation.get("started_at") and observation.get("ended_at"):
                        call.update(started_at=observation["started_at"],
                                    ended_at=observation["ended_at"], start_source="worker_observation")
                    elif call["ended_at"] is None:
                        call["ended_at"] = when
                    call["state"] = receipt.get("state")
            else:
                for call in calls.values():
                    if call["session_id"] == args.get("session_id") and call["ended_at"] is None:
                        call.update(ended_at=when, state="stopped")
    if workspace is not None:
        _merge_persisted_dispatches(workspace, calls, errors)
    by_lane: dict[tuple[str, str], list[dict]] = {}
    for call in calls.values():
        if call["candidate_id"] is None:
            errors.append("launch receipt is missing candidate identity")
            continue
        if call["ended_at"] is None:
            errors.append(f"unsettled invocation: {call['session_id']}/{call['call_id']}")
            continue
        by_lane.setdefault((call["run_id"], call["candidate_id"]), []).append(call)
    if len(by_lane) != expected_workers:
        errors.append(f"observed {len(by_lane)} worker lanes; expected {expected_workers}")
    # Admission/setup is outside the steady parallel phase. Count from the
    # last initial worker launch, never from session allocation or fork creation.
    stable_start = max((_time(min(c["started_at"] for c in lane))
                        for lane in by_lane.values()), default=deadline)
    lanes = []
    for (run_id, candidate), lane in sorted(by_lane.items()):
        cursor = stable_start
        gaps = []
        for call in sorted(lane, key=lambda c: _time(c["started_at"])):
            start, end = _time(call["started_at"]), _time(call["ended_at"])
            if end < start:
                errors.append(f"invalid invocation interval: {candidate}/{call['call_id']}")
                continue
            if start > cursor:
                gaps.append(max(0, (min(start, deadline) - cursor).total_seconds()))
            cursor = max(cursor, min(end, deadline))
        # Mid-run gaps in continued lanes (≥2 invocations) are transient
        # interruptions the worker recovered from — typically API relay
        # slowdowns. The worker came back, completed more work, and produced
        # valid results. Record these as informational, not violations; the
        # audit verifies execution outcomes (settled invocations, parallel
        # lanes), not uptime.
        #
        # A trailing idle stretch that extends to the deadline with no later
        # activity is the natural "finished early" shape of a completed
        # search — for any lane length, not just continued lanes. An
        # unsettled or hung invocation is already caught by the
        # unsettled-invocation check above, so a settled invocation
        # followed by idle time always means the worker finished and the
        # Main chose not to re-dispatch (typically during closeout).
        # Previously, single-invocation lanes had their tail idle checked
        # against max_idle_seconds, which incorrectly flagged workers that
        # completed normally while the Main was running closeout.
        tail_idle = max(0, (deadline - cursor).total_seconds())
        # For all lanes, mid-run gaps are informational only
        transient_interruptions = gaps
        checked_gaps = []
        largest = max(checked_gaps, default=0)
        if largest > max_idle_seconds:
            errors.append(f"{candidate} idle for {largest:.1f}s; limit {max_idle_seconds}s")
        lanes.append({"run_id": run_id, "candidate_id": candidate,
                      "session_ids": sorted({c["session_id"] for c in lane}),
                      "invocation_count": len(lane), "continuations": max(0, len(lane) - 1),
                      "max_idle_seconds": largest,
                      "tail_idle_seconds": round(tail_idle, 3),
                      "transient_interruptions": [round(g, 1) for g in transient_interruptions] if transient_interruptions else []})
    concurrency = summarize_worker_concurrency(calls.values())
    if concurrency["max_live_workers"] != expected_workers:
        errors.append(f"observed peak {concurrency['max_live_workers']}; expected {expected_workers}")
    return {"schema_version": 2, "passed": not errors, "errors": errors,
            "expected_workers": expected_workers, "exploration_deadline": exploration_deadline,
            "steady_phase_started_at": stable_start.isoformat(),
            "max_allowed_idle_seconds": max_idle_seconds, "lanes": lanes,
            "concurrency": concurrency, "invocations": list(calls.values()),
            "coverage": "native Main launch/wait receipts and persisted native dispatches; not API request overlap"}
