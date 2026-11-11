from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bench_goal_plus.parallel_execution import audit_pi_parallel_execution
from bench_runtime_paths import ensure_temp_root


ROOT = Path(__file__).resolve().parents[1]


class ParallelExecutionTest(unittest.TestCase):
    def test_persisted_dispatch_settles_unpolled_continuation_with_exact_identity(self):
        with tempfile.TemporaryDirectory(dir=ensure_temp_root("parallel-tests")) as directory:
            root = Path(directory)
            main = root / "main"
            main.mkdir()
            workspace = root / "workspace"
            record = workspace / ".gp/runs/run_0001/agent_sessions/agent_0001_01.json"
            record.parent.mkdir(parents=True)
            entries = []

            def receipt(second, name, args, result):
                stamp = f"2026-10-27T00:00:{second:02d}Z"
                tool_id = f"tool-{len(entries)}"
                entries.extend([
                    {"timestamp": stamp, "message": {"role": "assistant", "content": [
                        {"type": "toolCall", "id": tool_id, "arguments": args}]}},
                    {"timestamp": stamp, "message": {"role": "toolResult",
                        "toolName": name, "toolCallId": tool_id, "details": result}},
                ])

            args = {"run_id": "run_0001", "candidate_id": "c001"}
            receipt(0, "goal_plus_session_run", args,
                    {"session_id": "agent_0001_01", "call_id": "first", "state": "running"})
            receipt(10, "goal_plus_session_wait_any", {}, {"finished": [{
                "session_id": "agent_0001_01", "call_id": "first", "state": "completed",
                "invocation_id": "native-first"}]})
            receipt(12, "goal_plus_session_run", args,
                    {"session_id": "agent_0001_01", "call_id": "last", "state": "running"})
            (main / "main.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
            dispatches = [{
                "agent_harness": "pi", "runtime_provider": "direct",
                "native_session_id": "agent_0001_01", "invocation_id": identity,
                "started_at": start, "ended_at": end, "state": "completed",
            } for identity, start, end in [
                ("native-first", "2026-10-27T00:00:00.100Z", "2026-10-27T00:00:08Z"),
                ("native-last", "2026-10-27T00:00:12.100Z", "2026-10-27T00:00:40Z"),
            ]]
            session = {"run_id": "run_0001", "candidate_id": "c001",
                "agent_session_id": "agent_0001_01", "agent_harness": "pi",
                "runtime_provider": "direct", "run_requests": [{"call_id": "first"}, {"call_id": "last"}],
                "session_handle": {"agent_harness": "pi", "runtime_provider": "direct",
                    "external_id": "agent_0001_01", "metadata": {"dispatches": dispatches}}}

            def audit(value):
                record.write_text(json.dumps(value))
                return audit_pi_parallel_execution(main, workspace=workspace, expected_workers=1,
                    exploration_deadline="2026-10-27T00:00:40Z", max_idle_seconds=5)

            accepted = audit(session)
            self.assertTrue(accepted["passed"], accepted["errors"])
            self.assertEqual(accepted["concurrency"]["interval_count"], 2)
            self.assertEqual(accepted["invocations"][0]["ended_at"], "2026-10-27T00:00:08Z")
            self.assertEqual(accepted["lanes"][0]["continuations"], 1)
            # Allocation, another identity, an earlier dispatch, or ambiguous
            # native intervals cannot stand in for the final invocation.
            # "missing" (runtime dropped final dispatch flush) is tolerated.
            for corruption in ("run", "candidate", "native", "ambiguous", "invocation"):
                with self.subTest(corruption=corruption):
                    bad = json.loads(json.dumps(session))
                    ds = bad["session_handle"]["metadata"]["dispatches"]
                    if corruption in ("run", "candidate"): bad[corruption + "_id"] = "other"
                    elif corruption == "native": ds[-1]["native_session_id"] = "other"
                    elif corruption == "ambiguous": ds.append({**ds[-1], "invocation_id": "extra"})
                    elif corruption == "invocation": ds[0]["invocation_id"] = "unrelated"
                    self.assertFalse(audit(bad)["passed"])
            # A mid-run idle gap in a continued lane is a transient
            # interruption the worker recovered from — informational, not a
            # violation. The run's results remain valid.
            idle_gap = json.loads(json.dumps(session))
            idle_gap["session_handle"]["metadata"]["dispatches"][-1]["started_at"] = "2026-10-27T00:00:20Z"
            idle_result = audit(idle_gap)
            self.assertTrue(idle_result["passed"], idle_result["errors"])
            # Missing final dispatch (runtime flush race) is tolerated: the
            # main session's wait receipt still settles the invocation.
            missing = json.loads(json.dumps(session))
            missing["session_handle"]["metadata"]["dispatches"].pop()
            self.assertTrue(audit(missing)["passed"], audit(missing)["errors"])

    def test_continued_lane_tail_idle_is_natural_completion(self):
        """A continued lane that stops minutes before the deadline has
        finished early, not stalled: the trailing idle is recorded but does
        not fail the audit (regression for the libxml2 20261028 false
        partial, where both lanes settled ~194s before the deadline)."""
        with tempfile.TemporaryDirectory(dir=ensure_temp_root("parallel-tests")) as directory:
            folder = Path(directory)
            entries = []

            def receipt(second, tool, args, result):
                call_id = f"tool-{len(entries)}"
                stamp = f"2026-10-27T00:00:{second:02d}+00:00"
                entries.extend([
                    {"timestamp": stamp, "message": {"role": "assistant", "content": [
                        {"type": "toolCall", "id": call_id, "arguments": args}]}},
                    {"timestamp": stamp, "message": {"role": "toolResult",
                        "toolName": tool, "toolCallId": call_id, "details": result}},
                ])

            def launch(second, candidate, window):
                receipt(second, "goal_plus_session_run",
                        {"run_id": "run", "candidate_id": candidate},
                        {"session_id": candidate, "call_id": window, "state": "running"})

            def finish(second, candidate, window):
                receipt(second, "goal_plus_session_wait_any", {}, {"finished": [
                    {"session_id": candidate, "call_id": window, "state": "completed"}]})

            # Two invocations per lane, last one settles at t=30; deadline 60.
            for candidate in ("a", "b"):
                launch(0 if candidate == "a" else 1, candidate, "one")
                finish(9 if candidate == "a" else 10, candidate, "one")
                launch(10 if candidate == "a" else 11, candidate, "two")
            for candidate in ("a", "b"):
                finish(30, candidate, "two")
            (folder / "main.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
            result = audit_pi_parallel_execution(folder, expected_workers=2,
                exploration_deadline="2026-10-27T00:01:00Z", max_idle_seconds=5)
            self.assertTrue(result["passed"], result["errors"])
            for lane in result["lanes"]:
                self.assertGreaterEqual(lane["tail_idle_seconds"], 29)
                self.assertLessEqual(lane["max_idle_seconds"], 5)

    def test_native_receipts_distinguish_initial_k_from_continued_execution(self):
        with tempfile.TemporaryDirectory(dir=ensure_temp_root("parallel-tests")) as directory:
            folder = Path(directory)
            entries = []

            def receipt(second, tool, args, result):
                call_id = f"tool-{len(entries)}"
                stamp = f"2026-10-27T00:00:{second:02d}+00:00"
                entries.extend([
                    {"timestamp": stamp, "message": {"role": "assistant", "content": [
                        {"type": "toolCall", "id": call_id, "name": tool, "arguments": args}]}},
                    {"timestamp": stamp, "message": {"role": "toolResult", "toolCallId": call_id,
                        "toolName": tool, "details": result, "isError": False}},
                ])

            def launch(second, candidate, window):
                receipt(second, "goal_plus_session_run",
                        {"run_id": "run", "candidate_id": candidate},
                        {"session_id": candidate, "call_id": window, "state": "running"})

            def finish(second, candidate, window):
                receipt(second, "goal_plus_session_wait_any", {}, {"finished": [
                    {"session_id": candidate, "call_id": window, "state": "completed"}]})

            launch(0, "a", "one")
            launch(1, "b", "one")
            finish(10, "a", "one")
            finish(25, "b", "one")
            native = folder / "main.jsonl"

            def audit():
                native.write_text("".join(json.dumps(e) + "\n" for e in entries))
                return audit_pi_parallel_execution(folder, expected_workers=2,
                    exploration_deadline="2026-10-27T00:00:40Z", max_idle_seconds=5)

            initial_only = audit()
            self.assertEqual(initial_only["concurrency"]["max_live_workers"], 2)
            # A single-invocation lane that completed normally passes;
            # the tail idle is the natural "finished early" shape, not a
            # violation (unsettled invocations are caught separately).
            self.assertTrue(initial_only["passed"], initial_only["errors"])
            lane_a = next(l for l in initial_only["lanes"] if l["candidate_id"] == "a")
            self.assertGreater(lane_a["tail_idle_seconds"], 0)
            launch(11, "a", "two")
            launch(26, "b", "two")
            # Native files are chronological; append helpers above model independent lanes.
            entries.sort(key=lambda e: e["timestamp"])
            finish(40, "a", "two")
            finish(40, "b", "two")
            continued = audit()
            self.assertTrue(continued["passed"], continued["errors"])
            self.assertEqual([lane["continuations"] for lane in continued["lanes"]], [1, 1])
            # Missing settlement must fail closed, not be counted as running to the deadline.
            entries[:] = entries[:-2]
            self.assertFalse(audit()["passed"])
            # MCP errors may be encoded in details despite isError=false.
            # A rejected stop must not fabricate the missing settlement.
            receipt(40, "goal_plus_session_stop", {"session_id": "b"},
                    {"ok": False, "error": "session unavailable"})
            self.assertFalse(audit()["passed"])

    @unittest.skipUnless(shutil.which("pi") and shutil.which("node"), "Pi SDK required")
    def test_pi_sdk_blocks_batch_wait_and_early_selection_but_allows_same_session_resume(self):
        dist = Path(shutil.which("pi")).resolve().parents[1]
        script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const base = pathToFileURL(process.argv[1] + "/");
const { loadExtensions } = await import(new URL("core/extensions/loader.js", base));
const { ExtensionRunner } = await import(new URL("core/extensions/runner.js", base));
const { SessionManager } = await import(new URL("core/session-manager.js", base));
const now = Date.now();
process.env.BENCH_GOAL_PLUS_EXPLORATION_DEADLINE_AT = new Date(now + 3600000).toISOString();
const loaded = await loadExtensions([process.argv[2]], process.cwd());
assert.deepEqual(loaded.errors, []);
const runner = new ExtensionRunner(loaded.extensions, loaded.runtime, process.cwd(), SessionManager.inMemory(), {});
const errors = [];
runner.onError((e) => errors.push(e));
await runner.emit({ type: "session_start", reason: "startup" });
const call = (toolName, input = {}) => runner.emitToolCall({type: "tool_call", toolName, input, toolCallId: "call"});
const result = (toolName, input, details) => runner.emitToolResult({type: "tool_result", toolName,
  input, details, isError: false, toolCallId: "call", content: [{type: "text", text: JSON.stringify(details)}]});
const launch = (id, window) => result("goal_plus_session_run", {run_id: "run", candidate_id: id},
  {session_id: id, call_id: window, state: "running"});
await launch("a", "one");
await launch("b", "one");
assert.equal(await call("goal_plus_session_wait_any", {timeout_seconds: 300}), undefined);
for (const tool of ["goal_plus_session_interrupt", "goal_plus_session_stop", "goal_plus_session_close"]) {
  assert.equal((await call(tool, {session_id: "b", call_id: "one"}))?.block, true);
}
await result("goal_plus_session_wait_any", {}, {finished: [{session_id: "a", call_id: "one", state: "completed"}],
  running: [{session_id: "b", call_id: "one", state: "running"}]});
assert.equal((await call("goal_plus_session_close", {session_id: "a"}))?.block, true);
assert.equal((await call("goal_plus_session_wait_any", {timeout_seconds: 300}))?.block, true);
assert.equal(await call("goal_plus_session_wait_any", {timeout_seconds: 0}), undefined);
assert.equal((await call("goal_plus_search_select", {run_id: "run"})).block, true);
assert.equal(await call("goal_plus_session_run", {run_id: "run", candidate_id: "a", call_id: "two"}), undefined);
await launch("a", "two");
// A repeated old receipt must not retire the resumed invocation.
await result("goal_plus_session_wait_any", {}, {finished: [{session_id: "a", call_id: "one", state: "completed"}]});
assert.equal(await call("goal_plus_session_wait_any", {timeout_seconds: 300}), undefined);
await result("goal_plus_session_wait_any", {}, {finished: [{session_id: "b", call_id: "one", state: "failed"}]});
assert.equal(await call("goal_plus_session_stop", {session_id: "b"}), undefined);
assert.equal(await call("goal_plus_session_close", {session_id: "b"}), undefined);
assert.equal(await call("goal_plus_set_status", {status: "blocked", reason: "dependency unavailable"}), undefined);
await result("goal_plus_set_status", {status: "blocked"}, {status: "blocked"});
assert.equal(await call("goal_plus_session_stop", {session_id: "a"}), undefined);
Date.now = () => now + 3600001;
assert.equal(await call("goal_plus_search_select", {run_id: "run"}), undefined);
assert.equal(await call("goal_plus_session_close", {session_id: "a"}), undefined);
assert.deepEqual(errors, []);
console.log("Pi SDK continuation guard passed");
'''
        result = subprocess.run(["node", "--input-type=module", "-e", script, str(dist),
            str(ROOT / "bench_goal_plus/pi_parallel_extension.mjs")],
            cwd=ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
