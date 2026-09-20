"""Runner adapter for the shared artifact campaign controller."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..errors import ContractError, UnsupportedOperation
from ..models import CampaignRef, CampaignSpec, StatusSnapshot
from ..paths import ROOT, RUNS_ROOT, managed_python
from ..state import ensure_under
from .base import BenchmarkRunner


TERMINAL = {"finished", "partial", "failed"}


def process_alive(pid: int | None) -> bool | None:
    if not isinstance(pid, int) or pid < 1:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CommonMatrixRunner(BenchmarkRunner):
    def provision_commands(
        self, spec: CampaignSpec, *, skip_provision: bool
    ) -> list[list[str]]:
        return []

    def prepare_commands(self, spec: CampaignSpec) -> tuple[list[list[str]], CampaignRef]:
        if spec.model is None or spec.reasoning_effort is None:
            raise ContractError("common-matrix requires model and reasoning effort")
        if spec.wall_time_seconds is None or spec.live_search_concurrency is None:
            raise ContractError("common-matrix requires T and K")
        if spec.cell_concurrency not in (None, 1):
            raise ContractError("common-matrix has not proven cross-cell concurrency; use C=1")
        if spec.methods and spec.conditions:
            raise ContractError("common-matrix accepts methods or conditions, not both")
        destination = spec.campaign_dir or (
            RUNS_ROOT / "benchmark-campaigns" / spec.campaign_id
        )
        destination = ensure_under(destination, RUNS_ROOT, label="campaign directory")
        controller = str(self.definition.controller.relative_to(ROOT))
        command = [
            str(managed_python()),
            controller,
            "prepare",
            "--campaign-dir",
            str(destination),
            "--benchmarks",
            *spec.target_ids,
        ]
        if spec.task_id is not None:
            command.extend(["--task-id", spec.task_id])
        if spec.shared_dir:
            command.append("--shared-dir")
        if spec.shared_cache:
            command.append("--shared-cache")
        if spec.methods:
            command.extend(["--methods", *spec.methods])
        else:
            command.extend(["--conditions", *(spec.conditions or ("B0",))])
        command.extend([
            "--seeds",
            *(str(item) for item in spec.seeds),
            "--model",
            spec.model,
            "--reasoning-effort",
            spec.reasoning_effort,
            "--wall-time-seconds",
            str(spec.wall_time_seconds),
            "--concurrency",
            str(spec.live_search_concurrency),
        ])
        if spec.worker_runtime_seconds is not None:
            command.extend(
                ["--worker-runtime-seconds", str(spec.worker_runtime_seconds)]
            )
        if spec.worker_min_runtime_seconds is not None:
            command.extend(
                ["--worker-min-runtime-seconds", str(spec.worker_min_runtime_seconds)]
            )
        if spec.pi_provider_id is not None:
            command.extend(
                [
                    "--pi-provider-id",
                    spec.pi_provider_id,
                    "--pi-api",
                    str(spec.pi_api),
                    "--pi-api-key-env",
                    str(spec.pi_api_key_env),
                    "--pi-api-base-env",
                    str(spec.pi_api_base_env),
                ]
            )
        campaign = CampaignRef(
            campaign_id=spec.campaign_id,
            path=destination,
            target_id=spec.targets[0].target_id,
            runner_id=self.definition.runner_id,
        )
        return [command], campaign

    def start_command(
        self, spec: CampaignSpec, campaign: CampaignRef, *, detach: bool
    ) -> list[str]:
        if detach:
            raise UnsupportedOperation("common-matrix controller does not support detach")
        if not spec.model:
            raise ContractError("common-matrix run requires model")
        command = [
            str(managed_python()),
            str(self.definition.controller.relative_to(ROOT)),
            "run",
            "--campaign",
            str(campaign.path),
            "--model",
            spec.model,
        ]
        if spec.pi_provider_id is not None:
            command.extend(
                [
                    "--pi-provider-id",
                    spec.pi_provider_id,
                    "--pi-api",
                    str(spec.pi_api),
                    "--pi-api-key-env",
                    str(spec.pi_api_key_env),
                    "--pi-api-base-env",
                    str(spec.pi_api_base_env),
                ]
            )
        return command

    def resume_command(self, state: dict, campaign: CampaignRef) -> list[str]:
        if not self.definition.capabilities.resume:
            raise UnsupportedOperation("common-matrix runner does not support resume")
        model = (state.get("resolved_spec") or {}).get("model")
        if not model:
            raise ContractError("agent state does not record the campaign model")
        command = [
            str(managed_python()),
            str(self.definition.controller.relative_to(ROOT)),
            "run",
            "--campaign",
            str(campaign.path),
            "--model",
            str(model),
        ]
        provider = (state.get("resolved_spec") or {}).get("pi_provider")
        if isinstance(provider, dict) and provider.get("id"):
            command.extend(
                [
                    "--pi-provider-id",
                    str(provider["id"]),
                    "--pi-api",
                    str(provider["api"]),
                    "--pi-api-key-env",
                    str(provider["api_key_env"]),
                    "--pi-api-base-env",
                    str(provider["api_base_env"]),
                ]
            )
        return command

    def status(self, campaign: CampaignRef) -> StatusSnapshot:
        manifest_path = campaign.path / "campaign.json"
        if not manifest_path.is_file():
            raise ContractError(f"campaign manifest does not exist: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = str(payload.get("state") or "unknown")
        counts: dict[str, int] = {}
        for cell in payload.get("cells", []):
            cell_state = str(cell.get("state") or "unknown")
            counts[cell_state] = counts.get(cell_state, 0) + 1
        controller = payload.get("controller") or {}
        pid = controller.get("pid") if isinstance(controller.get("pid"), int) else None
        terminal = raw in TERMINAL
        normalized = {
            "prepared": "pending",
            "running": "running",
            "finished": "succeeded",
            "partial": "partial",
            "failed": "failed",
            "interrupted": "interrupted",
        }.get(raw, "unknown")
        return StatusSnapshot(
            state=normalized,
            raw_state=raw,
            terminal=terminal,
            controller_pid=pid,
            controller_alive=process_alive(pid),
            counts=counts,
            can_resume=raw in {"prepared", "interrupted"},
            can_stop=False,
            can_finalize=terminal,
        )

    def stop_command(self, campaign: CampaignRef) -> list[str]:
        raise UnsupportedOperation(
            "common-matrix has no external stop command; interrupt its foreground controller"
        )

    def finalize_command(self, campaign: CampaignRef) -> list[str]:
        return [
            str(managed_python()),
            str(self.definition.controller.relative_to(ROOT)),
            "summarize",
            "--campaign",
            str(campaign.path),
        ]

    def evidence_source(self, campaign: CampaignRef) -> Path:
        return campaign.path / "campaign-summary.json"
