"""Typed contracts shared by catalogs, runners, state, and CLI code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class RunnerCapabilities:
    provision: bool
    detach: bool
    stop: bool
    resume: bool
    repeat_seeds: bool
    cell_concurrency: bool
    retain_containers: bool
    official_evaluator: bool
    attempt_seed: bool
    resume_semantics: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DockerContract:
    requirement: str
    owner: str
    provision_mode: str
    scope: str

    @property
    def required(self) -> bool:
        return self.requirement == "required"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunnerDefinition:
    runner_id: str
    kind: str
    controller: Path
    evidence_filename: str
    supported_methods: tuple[str, ...]
    capabilities: RunnerCapabilities
    method_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.runner_id,
            "kind": self.kind,
            "controller": str(self.controller),
            "evidence_filename": self.evidence_filename,
            "supported_methods": list(self.supported_methods),
            "capabilities": self.capabilities.as_dict(),
            "method_contracts": self.method_contracts,
        }


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    runner_id: str
    adapter_id: str | None
    bootstrap_targets: tuple[str, ...]
    docker: DockerContract
    local_asset_inventory: bool
    default_inventory_profile: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.target_id,
            "runner": self.runner_id,
            "adapter": self.adapter_id,
            "bootstrap_targets": list(self.bootstrap_targets),
            "docker": self.docker.as_dict(),
            "local_asset_inventory": self.local_asset_inventory,
            "default_inventory_profile": self.default_inventory_profile,
        }


@dataclass(frozen=True)
class AssetPackDefinition:
    pack_id: str
    controller: Path
    bootstrap_targets: tuple[str, ...]
    default_profile: str
    provision: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.pack_id,
            "controller": str(self.controller),
            "bootstrap_targets": list(self.bootstrap_targets),
            "default_profile": self.default_profile,
            "provision": self.provision,
            "local_asset_inventory": True,
        }


@dataclass(frozen=True)
class PresetDefinition:
    preset_id: str
    description: str
    benchmarks: tuple[str, ...]
    profile: str | None
    campaign_id_template: str | None
    expected_profile: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.preset_id,
            "description": self.description,
            "benchmarks": list(self.benchmarks),
            "profile": self.profile,
            "campaign_id_template": self.campaign_id_template,
            "expected_profile": self.expected_profile,
        }


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    targets: tuple[TargetDefinition, ...]
    runner: RunnerDefinition
    preset_id: str | None = None
    profile: str | None = None
    task_id: str | None = None
    shared_dir: bool = False
    shared_cache: bool = False
    methods: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (1,)
    model: str | None = None
    reasoning_effort: str | None = None
    pi_provider_id: str | None = None
    pi_api: str | None = None
    pi_api_key_env: str | None = None
    pi_api_base_env: str | None = None
    wall_time_seconds: int | None = None
    live_search_concurrency: int | None = None
    cell_concurrency: int | None = None
    worker_runtime_seconds: int | None = None
    worker_min_runtime_seconds: int | None = None
    retain_containers: bool = False
    campaign_dir: Path | None = None

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(item.target_id for item in self.targets)

    def concurrency(self) -> dict[str, Any]:
        return {
            "T": self.wall_time_seconds if self.wall_time_seconds is not None else "profile",
            "K": (
                self.live_search_concurrency
                if self.live_search_concurrency is not None
                else "profile"
            ),
            "C": self.cell_concurrency if self.cell_concurrency is not None else "profile",
            "R": len(self.seeds),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "campaign_id": self.campaign_id,
            "targets": list(self.target_ids),
            "runner_id": self.runner.runner_id,
            "runner_kind": self.runner.kind,
            "preset": self.preset_id,
            "profile": self.profile,
            "task_id": self.task_id,
            "shared_dir": self.shared_dir,
            "shared_cache": self.shared_cache,
            "methods": list(self.methods),
            "conditions": list(self.conditions),
            "seeds": list(self.seeds),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "pi_provider": (
                {
                    "id": self.pi_provider_id,
                    "api": self.pi_api,
                    "api_key_env": self.pi_api_key_env,
                    "api_base_env": self.pi_api_base_env,
                }
                if self.pi_provider_id is not None
                else None
            ),
            "worker_runtime_seconds": self.worker_runtime_seconds,
            "worker_min_runtime_seconds": self.worker_min_runtime_seconds,
            "retain_containers": self.retain_containers,
            "concurrency": self.concurrency(),
        }
        return payload


@dataclass(frozen=True)
class CampaignRef:
    campaign_id: str
    path: Path
    target_id: str
    runner_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "campaign_id": self.campaign_id,
            "path": str(self.path),
            "target_id": self.target_id,
            "runner_id": self.runner_id,
        }


@dataclass(frozen=True)
class StatusSnapshot:
    state: str
    raw_state: str
    terminal: bool
    controller_pid: int | None = None
    controller_alive: bool | None = None
    counts: dict[str, int] = field(default_factory=dict)
    can_resume: bool = False
    can_stop: bool = False
    can_finalize: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBundle:
    source: Path
    markdown: Path
    workbook: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "source": str(self.source),
            "markdown": str(self.markdown),
            "xlsx": str(self.workbook),
        }
