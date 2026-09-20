"""Application service behind the repository's benchmark Agent Skills."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from adapters.registry import load_adapter

from .catalog import Catalog, read_json
from .errors import ContractError, UnsupportedOperation
from .models import (
    AssetPackDefinition,
    CampaignRef,
    CampaignSpec,
    EvidenceBundle,
    TargetDefinition,
)
from .paths import ROOT, RUNS_ROOT
from .runners.factory import create_runner
from .runners.openevolve_batch import DEFAULT_METHODS as OPENEVOLVE_METHODS
from .runtime import CommandExecutor, RuntimeManager, command_text
from .state import (
    STATE_FILE,
    campaign_ref_from_state,
    create_agent_state,
    load_agent_state,
    resolve_campaign_path,
    update_observation,
    update_phase,
    write_json_atomic,
)


SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M")


def validate_pi_provider(
    *, runner_kind: str, methods: tuple[str, ...], model: str | None,
    provider_id: str | None, api: str | None,
    api_key_env: str | None, api_base_env: str | None,
) -> None:
    fields = {"--pi-provider-id": provider_id, "--pi-api": api,
              "--pi-api-key-env": api_key_env, "--pi-api-base-env": api_base_env}
    if not any(value is not None for value in fields.values()):
        return
    missing = [flag for flag, value in fields.items() if value is None]
    if missing:
        raise ContractError("explicit Pi provider selection requires " + ", ".join(missing))
    if (runner_kind not in {"common-matrix", "openevolve-batch"}
            or not methods or any(not method.endswith("-pi") for method in methods)):
        raise ContractError("explicit Pi provider selection requires a common-matrix or OpenEvolve Pi method")
    if not provider_id or "/" in provider_id:
        raise ContractError("Pi provider id must be non-empty and cannot contain '/'")
    if api not in {"openai-responses", "openai-completions", "anthropic-messages"}:
        raise ContractError(f"unsupported Pi API: {api}")
    if not model or "/" in model:
        raise ContractError("explicit Pi provider selection requires a bare --model ID")
    for flag, value in (("--pi-api-key-env", api_key_env), ("--pi-api-base-env", api_base_env)):
        if not value or not value.replace("_", "A").isalnum():
            raise ContractError(f"{flag} must name an environment variable")


class BenchmarkAgent:
    def __init__(
        self,
        *,
        catalog: Catalog | None = None,
        executor: CommandExecutor | None = None,
        runtime: RuntimeManager | None = None,
    ) -> None:
        self.catalog = catalog or Catalog()
        self.executor = executor or CommandExecutor()
        self.runtime = runtime or RuntimeManager()

    def resolve_targets(
        self,
        *,
        target_ids: Iterable[str] = (),
        preset_id: str | None = None,
    ) -> tuple[tuple[TargetDefinition, ...], Any | None]:
        ids = tuple(target_ids)
        if ids and preset_id:
            raise ContractError("use either --preset or --benchmark, not both")
        preset = self.catalog.presets.get(preset_id) if preset_id else None
        if preset_id and preset is None:
            raise ContractError(f"unknown preset: {preset_id}")
        ids = ids or (preset.benchmarks if preset else ())
        if not ids:
            raise ContractError("choose --benchmark or --preset")
        if len(set(ids)) != len(ids):
            raise ContractError("benchmark ids must be unique")
        unknown = set(ids) - set(self.catalog.targets)
        if unknown:
            raise ContractError("unknown benchmark target(s): " + ", ".join(sorted(unknown)))
        if preset:
            self._validate_preset_profile(preset)
        return tuple(self.catalog.targets[item] for item in ids), preset

    def resolve_asset_packs(
        self, pack_ids: Iterable[str]
    ) -> tuple[AssetPackDefinition, ...]:
        ids = tuple(pack_ids)
        if not ids:
            raise ContractError("choose --asset-pack")
        if len(set(ids)) != len(ids):
            raise ContractError("asset pack ids must be unique")
        unknown = set(ids) - set(self.catalog.asset_packs)
        if unknown:
            raise ContractError(
                "unknown asset pack(s): " + ", ".join(sorted(unknown))
            )
        return tuple(self.catalog.asset_packs[item] for item in ids)

    def resolve_spec(
        self,
        *,
        target_ids: Iterable[str] = (),
        preset_id: str | None = None,
        profile: str | None = None,
        task_id: str | None = None,
        shared_dir: bool = False,
        shared_cache: bool = False,
        campaign_id: str | None = None,
        campaign_dir: Path | None = None,
        methods: Iterable[str] = (),
        conditions: Iterable[str] = (),
        seeds: Iterable[int] = (),
        model: str | None = None,
        reasoning_effort: str | None = None,
        pi_provider_id: str | None = None,
        pi_api: str | None = None,
        pi_api_key_env: str | None = None,
        pi_api_base_env: str | None = None,
        wall_time_seconds: int | None = None,
        live_search_concurrency: int | None = None,
        cell_concurrency: int | None = None,
        worker_runtime_seconds: int | None = None,
        worker_min_runtime_seconds: int | None = None,
        retain_containers: bool = False,
    ) -> CampaignSpec:
        targets, preset = self.resolve_targets(target_ids=target_ids, preset_id=preset_id)
        runners = {item.runner_id for item in targets}
        if len(runners) != 1:
            raise ContractError("one campaign cannot mix different runner families")
        runner_definition = self.catalog.runners[next(iter(runners))]
        if runner_definition.kind == "native-profile" and len(targets) != 1:
            raise ContractError("native-profile campaigns accept exactly one benchmark")
        selected_profile = profile or (preset.profile if preset else None)
        if task_id is not None:
            if runner_definition.kind == "openevolve-batch":
                from adapters.openevolve_examples.adapter import list_catalog_tasks

                if task_id not in {item["task_id"] for item in list_catalog_tasks(selected_profile or "cpu_portable")}:
                    raise ContractError(f"unknown task in OpenEvolve task set: {task_id}")
            elif runner_definition.kind != "common-matrix" or len(targets) != 1:
                raise ContractError("--task-id requires exactly one common-matrix benchmark")
            else:
                adapter_id = targets[0].adapter_id
                if adapter_id is None:
                    raise ContractError(f"target {targets[0].target_id} has no task adapter")
                loaded = load_adapter(adapter_id)
                try:
                    loaded.configure_task(task_id)
                except (KeyError, RuntimeError, ValueError) as error:
                    raise ContractError(str(error)) from error
                finally:
                    loaded.configure_task(None)
        selected_methods = tuple(methods)
        selected_seeds = tuple(seeds) or (1,)
        selected_conditions = tuple(conditions)
        if shared_dir and (
            runner_definition.kind != "common-matrix"
            or not selected_methods
            or any(not method.startswith("goal-plus-") for method in selected_methods)
        ):
            raise ContractError(
                "--shared-dir requires an explicit common-matrix Goal Plus method"
            )
        if shared_cache and (
            runner_definition.kind != "common-matrix"
            or not selected_methods
            or any(not method.startswith("goal-plus-") for method in selected_methods)
        ):
            raise ContractError(
                "--shared-cache requires an explicit common-matrix Goal Plus method"
            )
        values = (
            wall_time_seconds,
            live_search_concurrency,
            cell_concurrency,
            worker_runtime_seconds,
            worker_min_runtime_seconds,
            *selected_seeds,
        )
        if any(value is not None and value < 1 for value in values):
            raise ContractError("T, K, C, and seeds must be positive integers")
        if len(set(selected_seeds)) != len(selected_seeds):
            raise ContractError("seeds must be unique")
        if len(selected_seeds) > 1 and not runner_definition.capabilities.repeat_seeds:
            raise ContractError(
                f"runner {runner_definition.runner_id} does not support R>1 seed matrices"
            )
        if len(set(selected_conditions)) != len(selected_conditions):
            raise ContractError("conditions must be unique")

        if preset:
            expected = preset.expected_profile
            overrides = {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "wall_time_seconds": wall_time_seconds,
                "concurrency": live_search_concurrency,
                "cell_concurrency": cell_concurrency,
            }
            drift = {
                key: {"expected": expected.get(key), "requested": value}
                for key, value in overrides.items()
                if value is not None and value != expected.get(key)
            }
            if selected_methods and list(selected_methods) != expected.get("methods"):
                drift["methods"] = {
                    "expected": expected.get("methods"),
                    "requested": list(selected_methods),
                }
            if drift:
                raise ContractError(
                    f"preset {preset.preset_id} is frozen; use --benchmark for overrides:\n"
                    + json.dumps(drift, indent=2)
                )
            selected_methods = selected_methods or tuple(expected.get("methods") or ())
            model = model if model is not None else expected.get("model")
            reasoning_effort = (
                reasoning_effort
                if reasoning_effort is not None
                else expected.get("reasoning_effort")
            )
            wall_time_seconds = (
                wall_time_seconds
                if wall_time_seconds is not None
                else expected.get("wall_time_seconds")
            )
            live_search_concurrency = (
                live_search_concurrency
                if live_search_concurrency is not None
                else expected.get("concurrency")
            )
            cell_concurrency = (
                cell_concurrency
                if cell_concurrency is not None
                else expected.get("cell_concurrency")
            )

        if runner_definition.kind == "native-profile" and not selected_profile:
            raise ContractError("native-profile campaigns require --profile or a preset")
        if (
            runner_definition.kind == "native-profile"
            and runner_definition.capabilities.attempt_seed
            and len(selected_seeds) != 1
        ):
            raise ContractError("native-profile campaigns support one attempt seed")
        if (
            runner_definition.kind == "native-profile"
            and not runner_definition.capabilities.attempt_seed
            and not runner_definition.capabilities.repeat_seeds
            and selected_seeds != (1,)
        ):
            raise ContractError(
                f"runner {runner_definition.runner_id} does not support attempt seeds"
            )
        if (
            cell_concurrency is not None
            and cell_concurrency > 1
            and not runner_definition.capabilities.cell_concurrency
        ):
            raise ContractError(
                f"{runner_definition.runner_id} has not proven cross-cell "
                "concurrency; use C=1"
            )
        if retain_containers and not runner_definition.capabilities.retain_containers:
            raise ContractError(
                f"runner {runner_definition.runner_id} does not support retained "
                "debug containers"
            )
        if runner_definition.kind in {"common-matrix", "openevolve-batch"}:
            required = {
                "--model": model,
                "--reasoning-effort": reasoning_effort,
                "--wall-time-seconds": wall_time_seconds,
                "--live-search-concurrency": live_search_concurrency,
            }
            missing = [flag for flag, value in required.items() if value is None]
            if missing:
                raise ContractError("common-matrix requires " + ", ".join(missing))
            cell_concurrency = 1
            if (
                worker_min_runtime_seconds is not None
                and worker_runtime_seconds is not None
                and worker_min_runtime_seconds > worker_runtime_seconds
            ):
                raise ContractError(
                    "worker minimum runtime cannot exceed worker maximum runtime"
                )
        if (
            runner_definition.kind == "common-matrix"
            and selected_methods
            and selected_conditions
        ):
            raise ContractError("common-matrix accepts --method or --condition, not both")
        if runner_definition.kind == "openevolve-batch" and len(selected_seeds) != 1:
            raise ContractError("openevolve-batch currently supports one seed per campaign")
        if runner_definition.kind == "openevolve-batch" and not selected_methods:
            selected_methods = OPENEVOLVE_METHODS
        if len(set(selected_methods)) != len(selected_methods):
            raise ContractError("methods must be unique")
        unsupported_methods = set(selected_methods) - set(
            runner_definition.supported_methods
        )
        if unsupported_methods:
            supported = ", ".join(runner_definition.supported_methods)
            rejected = ", ".join(sorted(unsupported_methods))
            raise ContractError(
                f"runner {runner_definition.runner_id} does not support method(s): "
                f"{rejected}; supported: {supported}"
            )
        validate_pi_provider(
            runner_kind=runner_definition.kind, methods=selected_methods, model=model,
            provider_id=pi_provider_id, api=pi_api,
            api_key_env=pi_api_key_env, api_base_env=pi_api_base_env,
        )
        if live_search_concurrency is not None and live_search_concurrency > 1:
            unsupported_parallel_methods = [
                method
                for method in selected_methods
                if not (
                    method.startswith("goal-plus-") or method.startswith("plain-")
                )
            ]
            non_goal_plus_conditions = [
                condition
                for condition in selected_conditions
                if condition not in {"B3", "B4"}
            ]
            if (
                unsupported_parallel_methods
                or non_goal_plus_conditions
                or not (selected_methods or selected_conditions)
            ):
                raise ContractError(
                    "K>1 is unsupported: methods without a Plain outer-trajectory "
                    "or Goal Plus internal-subagent topology require K=1"
                )
        for method in selected_methods:
            contract = runner_definition.method_contracts.get(method, {})
            if contract.get("model_format") == "provider/model":
                provider, separator, model_id = str(model or "").partition("/")
                if not separator or not provider or not model_id:
                    raise ContractError(
                        f"method {method} requires --model PROVIDER/MODEL; "
                        "a bare model ID cannot select a Pi provider"
                    )

        selected_id = campaign_id
        if not selected_id and preset and preset.campaign_id_template:
            selected_id = preset.campaign_id_template.format(timestamp=timestamp())
        selected_id = selected_id or f"{'-'.join(item.target_id for item in targets)}-{timestamp()}"
        if SAFE_ID.fullmatch(selected_id) is None:
            raise ContractError(f"unsafe campaign id: {selected_id!r}")
        return CampaignSpec(
            campaign_id=selected_id,
            targets=targets,
            runner=runner_definition,
            preset_id=preset.preset_id if preset else None,
            profile=selected_profile,
            task_id=task_id,
            shared_dir=shared_dir,
            shared_cache=shared_cache,
            methods=selected_methods,
            conditions=selected_conditions,
            seeds=selected_seeds,
            model=model,
            reasoning_effort=reasoning_effort,
            pi_provider_id=pi_provider_id,
            pi_api=pi_api,
            pi_api_key_env=pi_api_key_env,
            pi_api_base_env=pi_api_base_env,
            wall_time_seconds=wall_time_seconds,
            live_search_concurrency=live_search_concurrency,
            cell_concurrency=cell_concurrency,
            worker_runtime_seconds=worker_runtime_seconds,
            worker_min_runtime_seconds=worker_min_runtime_seconds,
            retain_containers=retain_containers,
            campaign_dir=campaign_dir,
        )

    def setup(
        self,
        targets: tuple[TargetDefinition, ...],
        *,
        profile: str | None,
        methods: tuple[str, ...] = (),
        model: str | None = None,
        reasoning_effort: str | None = None,
        pi_provider_id: str | None = None,
        pi_api: str | None = None,
        pi_api_key_env: str | None = None,
        pi_api_base_env: str | None = None,
        skip_bootstrap: bool,
        skip_provision: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        warnings = self.runtime.validate_host(
            targets, dry_run=dry_run, require_uv=not skip_bootstrap
        )
        commands: list[list[str]] = []
        if profile:
            for target in targets:
                if target.local_asset_inventory:
                    commands.extend(
                        self._local_asset_check_commands(
                            target,
                            profile,
                            allow_missing=not skip_provision,
                        )
                    )
        commands.extend(self.runtime.setup_commands(
            targets,
            skip_bootstrap=skip_bootstrap,
            skip_provision=skip_provision,
            bootstrap_targets=self._bootstrap_targets_for_methods(targets, methods),
            runner_owns_doctor=all(
                self.catalog.runners[target.runner_id].kind == "native-profile"
                for target in targets
            ),
        ))
        groups: dict[str, list[TargetDefinition]] = {}
        for target in targets:
            groups.setdefault(target.runner_id, []).append(target)
        for runner_id, members in groups.items():
            definition = self.catalog.runners[runner_id]
            runner = create_runner(definition)
            selected_methods = tuple(
                method for method in methods if method in definition.supported_methods
            )
            unsupported = set(methods) - {
                method
                for target in targets
                for method in self.catalog.runners[
                    target.runner_id
                ].supported_methods
            }
            if unsupported:
                raise ContractError(
                    "setup method(s) are not supported by the selected runner(s): "
                    + ", ".join(sorted(unsupported))
                )
            validate_pi_provider(
                runner_kind=definition.kind, methods=selected_methods, model=model,
                provider_id=pi_provider_id, api=pi_api,
                api_key_env=pi_api_key_env, api_base_env=pi_api_base_env,
            )
            spec = CampaignSpec(
                campaign_id="setup",
                targets=tuple(members),
                runner=definition,
                profile=profile,
                methods=selected_methods,
                model=model,
                reasoning_effort=reasoning_effort,
                pi_provider_id=pi_provider_id, pi_api=pi_api,
                pi_api_key_env=pi_api_key_env, pi_api_base_env=pi_api_base_env,
            )
            commands.extend(
                runner.provision_commands(
                    spec, skip_provision=skip_provision
                )
            )
        result = {
            "schema_version": 1,
            "action": "setup",
            "benchmarks": [item.target_id for item in targets],
            "profile": profile,
            "docker": [item.docker.as_dict() for item in targets],
            "warnings": warnings,
            "commands": [command_text(item) for item in commands],
        }
        self.executor.execute(commands, dry_run=dry_run)
        if not dry_run and not skip_bootstrap and not (ROOT / ".bench-env/state.json").is_file():
            raise ContractError("setup completed without .bench-env/state.json")
        return result

    def start(
        self,
        spec: CampaignSpec,
        *,
        skip_bootstrap: bool,
        skip_provision: bool,
        prepare_only: bool,
        foreground: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        warnings = self.runtime.validate_host(
            spec.targets, dry_run=dry_run, require_uv=not skip_bootstrap
        )
        runner = create_runner(spec.runner)
        runtime_metadata = runner.runtime_metadata(spec)
        resolved_spec = {**spec.as_dict(), **runtime_metadata}
        setup_commands: list[list[str]] = []
        if spec.profile:
            for target in spec.targets:
                if target.local_asset_inventory:
                    setup_commands.extend(
                        self._local_asset_check_commands(
                            target,
                            spec.profile,
                            allow_missing=not skip_provision,
                        )
                    )
        setup_commands.extend(self.runtime.setup_commands(
            spec.targets,
            skip_bootstrap=skip_bootstrap,
            skip_provision=skip_provision,
            bootstrap_targets=self._bootstrap_targets_for_methods(
                spec.targets, spec.methods
            ),
            runner_owns_doctor=spec.runner.kind == "native-profile",
        ))
        setup_commands.extend(
            runner.provision_commands(spec, skip_provision=skip_provision)
        )
        prepare_commands, campaign = runner.prepare_commands(spec)
        detach = spec.runner.capabilities.detach and not foreground
        run_commands = [] if prepare_only else [runner.start_command(spec, campaign, detach=detach)]
        commands = [*setup_commands, *prepare_commands, *run_commands]
        follow_up = self._follow_up(campaign, spec.runner.capabilities)
        plan = {
            "schema_version": 1,
            "action": "start",
            "resolved_spec": resolved_spec,
            "campaign": campaign.as_dict(),
            "docker": [item.docker.as_dict() for item in spec.targets],
            "warnings": warnings,
            "commands": [command_text(item) for item in commands],
            "follow_up": follow_up,
        }
        if dry_run:
            self.executor.execute(commands, dry_run=True)
            return plan

        self.executor.execute(setup_commands, dry_run=False)
        self.executor.execute(prepare_commands, dry_run=False)
        if not campaign.path.is_dir() or not (campaign.path / "campaign.json").is_file():
            raise ContractError(f"runner did not create campaign state: {campaign.path}")
        state = create_agent_state(
            spec,
            campaign,
            commands=plan["commands"],
            follow_up=follow_up,
            resolved_spec=resolved_spec,
        )
        if run_commands:
            try:
                self.executor.execute(run_commands, dry_run=False)
                if detach:
                    state = update_phase(campaign.path, state, "running")
            finally:
                state = update_observation(campaign.path, state, runner.status(campaign))
        plan["agent_state"] = str(campaign.path / STATE_FILE)
        plan["agent_phase"] = state["agent_phase"]
        return plan

    def _bootstrap_targets_for_methods(
        self,
        targets: tuple[TargetDefinition, ...],
        methods: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        if not methods:
            return None
        selected: set[str] = set()
        for target in targets:
            runner = self.catalog.runners[target.runner_id]
            applicable = [method for method in methods if method in runner.supported_methods]
            if not applicable:
                return None
            for method in applicable:
                override = runner.method_contracts.get(method, {}).get(
                    "bootstrap_targets"
                )
                if override is None:
                    return None
                selected.update(override)
        return tuple(sorted(selected))

    def status(self, campaign_value: str | Path, *, benchmark: str | None = None) -> dict[str, Any]:
        campaign, state, ref, runner = self._campaign_context(campaign_value, benchmark)
        snapshot = runner.status(ref)
        state = update_observation(campaign, state, snapshot)
        return {
            "campaign": ref.as_dict(),
            "agent_phase": state["agent_phase"],
            "runner": snapshot.as_dict(),
            "follow_up": state.get("follow_up", {}),
            "artifacts": state.get("artifacts", {}),
        }

    def stop(self, campaign_value: str | Path, *, benchmark: str | None, dry_run: bool) -> dict[str, Any]:
        campaign, state, ref, runner = self._campaign_context(campaign_value, benchmark)
        command = runner.stop_command(ref)
        if dry_run:
            self.executor.execute([command], dry_run=True)
        else:
            try:
                self.executor.execute([command], dry_run=False)
            finally:
                state = update_observation(campaign, state, runner.status(ref))
        return {"campaign": ref.as_dict(), "command": command_text(command), "agent_phase": state["agent_phase"]}

    def resume(self, campaign_value: str | Path, *, benchmark: str | None, dry_run: bool) -> dict[str, Any]:
        campaign, state, ref, runner = self._campaign_context(campaign_value, benchmark)
        command = runner.resume_command(state, ref)
        if dry_run:
            self.executor.execute([command], dry_run=True)
        else:
            try:
                self.executor.execute([command], dry_run=False)
            finally:
                state = update_observation(campaign, state, runner.status(ref))
        return {"campaign": ref.as_dict(), "command": command_text(command), "agent_phase": state["agent_phase"]}

    def finish(
        self,
        campaign_value: str | Path,
        *,
        benchmark: str | None,
        markdown_out: Path | None,
        xlsx_out: Path | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        campaign, state, ref, runner = self._campaign_context(campaign_value, benchmark)
        snapshot = runner.status(ref)
        if not snapshot.terminal:
            raise ContractError(f"campaign is not terminal: state={snapshot.raw_state!r}")
        finalize = runner.finalize_command(ref)
        report = [sys.executable, "scripts/benchmark_report.py", "--campaign", str(campaign)]
        if markdown_out:
            report.extend(["--markdown-out", str(markdown_out.expanduser().resolve())])
        if xlsx_out:
            report.extend(["--xlsx-out", str(xlsx_out.expanduser().resolve())])
        if dry_run:
            self.executor.execute([finalize, report], dry_run=True)
        else:
            self.executor.execute([finalize], dry_run=False)
            state = update_phase(campaign, state, "finalized")
            self.executor.execute([report], dry_run=False)
        markdown = markdown_out.expanduser().resolve() if markdown_out else campaign / "report.md"
        workbook = xlsx_out.expanduser().resolve() if xlsx_out else campaign / f"{campaign.name}.xlsx"
        evidence = EvidenceBundle(runner.evidence_source(ref), markdown, workbook)
        if not dry_run:
            missing = [path for path in (evidence.source, evidence.markdown, evidence.workbook) if not path.is_file()]
            if missing:
                raise ContractError("finish did not create: " + ", ".join(str(path) for path in missing))
            state = update_phase(campaign, state, "reported", evidence=evidence)
        return {
            "campaign": ref.as_dict(),
            "agent_phase": state["agent_phase"],
            "artifacts": evidence.as_dict(),
            "commands": [command_text(finalize), command_text(report)],
        }

    def check(
        self, target_id: str, *, profile: str | None = None, dry_run: bool
    ) -> dict[str, Any]:
        try:
            target = self.catalog.targets[target_id]
        except KeyError as error:
            raise ContractError(f"unknown benchmark target: {target_id}") from error
        loaded = load_adapter(target.adapter_id) if target.adapter_id else None
        adapter = loaded.manifest_contract() if loaded else None
        if loaded and target.docker.provision_mode == "eager":
            missing_hooks = [
                name
                for name in ("provision_environment", "doctor_environment")
                if not callable(getattr(loaded.module, name, None))
            ]
            if missing_hooks:
                raise ContractError(
                    f"{target_id}: eager adapter Docker is missing "
                    + ", ".join(missing_hooks)
                )
        result = {
            "benchmark": target.as_dict(),
            "runner": self.catalog.runners[target.runner_id].as_dict(),
            "adapter": adapter,
            "validated": not dry_run,
        }
        if profile is None:
            command = [sys.executable, "scripts/status.py", "--check"]
            self.executor.execute([command], dry_run=dry_run)
            result["repository_check"] = command_text(command)
            return result

        if not target.local_asset_inventory:
            raise UnsupportedOperation(
                f"target {target_id} does not support profiled local-asset checks"
            )
        commands = self._local_asset_check_commands(target, profile)
        if not commands:
            raise ContractError(
                f"runner {target.runner_id} returned no local-asset check command"
            )
        self.executor.execute(commands, dry_run=dry_run)
        result["local_asset_inventory"] = {
            "profile": profile,
            "read_only": True,
            "acquisition_attempted": False,
            "commands": [command_text(command) for command in commands],
        }
        return result

    def check_environment(self, *, assume_yes: bool, dry_run: bool) -> dict[str, Any]:
        commands: list[list[str]] = [
            [sys.executable, "scripts/status.py", "--check"]
        ]
        inventories: list[dict[str, str]] = []
        for target in self.catalog.targets.values():
            if not target.local_asset_inventory:
                continue
            profile = target.default_inventory_profile
            if profile is None:
                raise ContractError(
                    f"{target.target_id}: missing default inventory profile"
                )
            target_commands = self._local_asset_check_commands(target, profile)
            commands.extend(target_commands)
            inventories.append(
                {
                    "kind": "benchmark",
                    "id": target.target_id,
                    "profile": profile,
                }
            )
        for pack in self.catalog.asset_packs.values():
            commands.append(
                self._asset_pack_inventory_command(pack, pack.default_profile)
            )
            inventories.append(
                {
                    "kind": "asset-pack",
                    "id": pack.pack_id,
                    "profile": pack.default_profile,
                }
            )
        update_command = [
            sys.executable,
            "scripts/repro_env.py",
            "check",
            "--inventory-gated",
        ]
        if assume_yes:
            update_command.append("--yes")
        commands.append(update_command)
        self.executor.execute(commands, dry_run=dry_run)
        return {
            "schema_version": 1,
            "action": "environment-check",
            "inventory_gates": inventories,
            "update_policy": "accept" if assume_yes else "prompt",
            "commands": [command_text(command) for command in commands],
            "validated": not dry_run,
        }

    def _local_asset_check_commands(
        self,
        target: TargetDefinition,
        profile: str,
        *,
        allow_missing: bool = False,
    ) -> list[list[str]]:
        definition = self.catalog.runners[target.runner_id]
        if target.docker.owner == "adapter":
            return [
                [
                    sys.executable,
                    "-m",
                    "bench_goal_plus.docker_hooks",
                    "inventory",
                    "--target",
                    target.target_id,
                    "--profile",
                    profile,
                ]
            ]
        runner = create_runner(definition)
        return runner.local_asset_check_commands(
            profile, allow_missing=allow_missing
        )

    def check_asset_pack(
        self,
        pack: AssetPackDefinition,
        *,
        profile: str | None,
        dry_run: bool,
    ) -> dict[str, Any]:
        selected_profile = profile or pack.default_profile
        command = self._asset_pack_inventory_command(pack, selected_profile)
        self.executor.execute([command], dry_run=dry_run)
        return {
            "asset_pack": pack.as_dict(),
            "profile": selected_profile,
            "read_only": True,
            "acquisition_attempted": False,
            "commands": [command_text(command)],
        }

    @staticmethod
    def _asset_pack_inventory_command(
        pack: AssetPackDefinition, profile: str
    ) -> list[str]:
        return [
            sys.executable,
            str(pack.controller.relative_to(ROOT)),
            "inventory",
            "--profile",
            profile,
        ]

    def setup_asset_packs(
        self,
        packs: tuple[AssetPackDefinition, ...],
        *,
        profile: str | None,
        skip_bootstrap: bool,
        skip_provision: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        if len(packs) != 1:
            raise ContractError("setup currently accepts exactly one asset pack")
        pack = packs[0]
        selected_profile = profile or pack.default_profile
        inventory_command = [
            sys.executable,
            str(pack.controller.relative_to(ROOT)),
            "inventory",
            "--profile",
            selected_profile,
        ]
        commands: list[list[str]] = [inventory_command, ["docker", "info"]]
        if not skip_bootstrap:
            bootstrap = [sys.executable, "scripts/repro_env.py", "bootstrap"]
            for upstream in pack.bootstrap_targets:
                bootstrap.extend(["--only", upstream])
            commands.append(bootstrap)
        doctor = [sys.executable, "scripts/repro_env.py", "doctor"]
        for upstream in pack.bootstrap_targets:
            doctor.extend(["--only", upstream])
        commands.append(doctor)
        if not skip_provision:
            if not pack.provision:
                raise UnsupportedOperation(
                    f"asset pack {pack.pack_id} does not support provision"
                )
            commands.append(
                [
                    sys.executable,
                    str(pack.controller.relative_to(ROOT)),
                    "provision",
                    "--profile",
                    selected_profile,
                ]
            )
        commands.append(
            [
                sys.executable,
                str(pack.controller.relative_to(ROOT)),
                "doctor",
                "--profile",
                selected_profile,
            ]
        )
        self.executor.execute(commands, dry_run=dry_run)
        return {
            "schema_version": 1,
            "action": "setup",
            "asset_packs": [pack.pack_id],
            "profile": selected_profile,
            "commands": [command_text(command) for command in commands],
        }

    def _campaign_context(self, value: str | Path, benchmark: str | None):
        campaign = resolve_campaign_path(value)
        if not (campaign / STATE_FILE).is_file():
            if not benchmark:
                raise ContractError(
                    f"{STATE_FILE} is missing; pass --benchmark once to adopt this legacy campaign"
                )
            self._adopt_legacy(campaign, benchmark)
        state = load_agent_state(campaign)
        ref = campaign_ref_from_state(campaign, state)
        try:
            definition = self.catalog.runners[ref.runner_id]
        except KeyError as error:
            raise ContractError(f"agent state names unknown runner {ref.runner_id!r}") from error
        return campaign, state, ref, create_runner(definition)

    def _adopt_legacy(self, campaign: Path, benchmark: str) -> None:
        try:
            target = self.catalog.targets[benchmark]
        except KeyError as error:
            raise ContractError(f"unknown benchmark target: {benchmark}") from error
        manifest = read_json(campaign / "campaign.json")
        runner = create_runner(self.catalog.runners[target.runner_id])
        ref = CampaignRef(str(manifest.get("campaign_id") or campaign.name), campaign, target.target_id, target.runner_id)
        snapshot = runner.status(ref)
        phase = "terminal" if snapshot.terminal else ("running" if snapshot.state == "running" else "prepared")
        payload = {
            "schema_version": 1,
            "campaign_id": ref.campaign_id,
            "campaign_path": str(campaign),
            "target_id": target.target_id,
            "runner_id": target.runner_id,
            "agent_phase": phase,
            "created_at": datetime.now().astimezone().isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
            "resolved_spec": {
                "campaign_id": ref.campaign_id,
                "targets": manifest.get("benchmarks") or [target.target_id],
                "model": manifest.get("model"),
                "reasoning_effort": manifest.get("reasoning_effort"),
                "methods": manifest.get("methods") or [],
                "concurrency": manifest.get("budget") or {},
                "adopted": True,
            },
            "runner_manifest": "campaign.json",
            "commands": [],
            "follow_up": self._follow_up(ref, self.catalog.runners[target.runner_id].capabilities),
            "last_observed": snapshot.as_dict(),
            "artifacts": {},
            "secret_policy": "credentials are inherited only and are never serialized",
        }
        write_json_atomic(campaign / STATE_FILE, payload)

    @staticmethod
    def _follow_up(campaign: CampaignRef, capabilities: Any) -> dict[str, str]:
        base = [sys.executable, "scripts/bench.py"]
        result = {
            "status": command_text([*base, "status", "--campaign", str(campaign.path)]),
            "finish": command_text([*base, "finish", "--campaign", str(campaign.path)]),
        }
        if capabilities.stop:
            result["stop"] = command_text([*base, "stop", "--campaign", str(campaign.path)])
        if capabilities.resume:
            result["resume"] = command_text([*base, "resume", "--campaign", str(campaign.path)])
        return result

    def _validate_preset_profile(self, preset: Any) -> None:
        if not preset.expected_profile or not preset.profile:
            return
        runner_ids = {
            self.catalog.targets[target_id].runner_id
            for target_id in preset.benchmarks
        }
        if len(runner_ids) != 1:
            raise ContractError(
                f"preset {preset.preset_id} cannot mix profile roots"
            )
        runner = self.catalog.runners[next(iter(runner_ids))]
        profile = read_json(
            runner.controller.parent / "profiles" / f"{preset.profile}.json"
        )
        observed = {
            "task_count": len(profile.get("task_ids", [])),
            "methods": profile.get("methods"),
            "model": profile.get("model"),
            "reasoning_effort": profile.get("reasoning_effort"),
            "wall_time_seconds": profile.get("wall_time_seconds"),
            "concurrency": profile.get("concurrency"),
            "cell_concurrency": profile.get("cell_concurrency"),
        }
        if "agent_provider" in preset.expected_profile:
            observed["agent_provider"] = profile.get("agent_provider")
        if "supplemental_evaluation_enabled" in preset.expected_profile:
            observed["supplemental_evaluation_enabled"] = (
                (profile.get("goal_plus") or {}).get(
                    "supplemental_evaluation_enabled"
                )
            )
        if observed != preset.expected_profile:
            raise ContractError(
                f"preset {preset.preset_id} profile drifted:\n"
                + json.dumps({"expected": preset.expected_profile, "observed": observed}, indent=2)
            )
