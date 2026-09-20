"""Command-line interface for the benchmark Agent Skill engine."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .application import BenchmarkAgent
from .errors import BenchGoalPlusError, ContractError
from .scaffold import scaffold_benchmark


def add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--preset")
    parser.add_argument("--profile")


def add_pi_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pi-provider-id")
    parser.add_argument(
        "--pi-api",
        choices=("openai-responses", "openai-completions", "anthropic-messages"),
    )
    parser.add_argument("--pi-api-key-env")
    parser.add_argument("--pi-api-base-env")


def add_start_arguments(parser: argparse.ArgumentParser) -> None:
    add_selection(parser)
    parser.add_argument(
        "--task-id",
        help="select one task from a common adapter that exposes a task catalog",
    )
    parser.add_argument(
        "--shared-dir",
        action="store_true",
        help="enable Goal Plus shared_dir for a common-matrix Goal Plus run",
    )
    parser.add_argument(
        "--shared-cache",
        action="store_true",
        help="enable the Goal Plus shared_cache fact plane and LLM-verifier "
        "path-feasibility feedback for a common-matrix Goal Plus run",
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"))
    add_pi_provider_arguments(parser)
    parser.add_argument("--wall-time-seconds", type=int)
    parser.add_argument("--live-search-concurrency", type=int)
    parser.add_argument("--cell-concurrency", type=int)
    parser.add_argument("--worker-runtime-seconds", type=int)
    parser.add_argument("--worker-min-runtime-seconds", type=int)
    parser.add_argument(
        "--retain-containers",
        action="store_true",
        help="stop and retain runner-owned debug containers when supported",
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-provision", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up, run, resume, finalize, and report registered benchmarks."
    )
    children = parser.add_subparsers(dest="command", required=True)
    catalog = children.add_parser("catalog")
    catalog.add_argument("--json", action="store_true")

    setup = children.add_parser("setup")
    add_selection(setup)
    setup.add_argument("--asset-pack", action="append", default=[])
    setup.add_argument("--method", action="append", default=[])
    setup.add_argument("--model")
    add_pi_provider_arguments(setup)
    setup.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh")
    )
    setup.add_argument("--skip-bootstrap", action="store_true")
    setup.add_argument("--skip-provision", action="store_true")
    setup.add_argument("--dry-run", action="store_true")

    for name in ("plan", "start", "launch"):
        add_start_arguments(children.add_parser(name))
    e2e = children.add_parser("e2e")
    add_start_arguments(e2e)
    e2e.add_argument("--markdown-out", type=Path)
    e2e.add_argument("--xlsx-out", type=Path)

    for name in ("status", "stop", "resume"):
        child = children.add_parser(name)
        child.add_argument("--campaign", required=True)
        child.add_argument("--benchmark")
        child.add_argument("--dry-run", action="store_true")

    finish = children.add_parser("finish")
    finish.add_argument("--campaign", required=True)
    finish.add_argument("--benchmark")
    finish.add_argument("--markdown-out", type=Path)
    finish.add_argument("--xlsx-out", type=Path)
    finish.add_argument("--dry-run", action="store_true")

    check = children.add_parser("check")
    add_selection(check)
    check.add_argument("--asset-pack", action="append", default=[])
    check.add_argument(
        "--environment",
        action="store_true",
        help="check all asset inventories and managed Git repositories",
    )
    check.add_argument(
        "--yes",
        action="store_true",
        help="accept a fast-forward environment update without prompting",
    )
    check.add_argument("--dry-run", action="store_true")
    scaffold = children.add_parser("scaffold")
    scaffold.add_argument("--benchmark-id", required=True)
    scaffold.add_argument("--shape", choices=("common", "native"), required=True)
    scaffold.add_argument("--module")
    scaffold.add_argument("--write", action="store_true")
    return parser


def spec_from_args(agent: BenchmarkAgent, args: argparse.Namespace):
    return agent.resolve_spec(
        target_ids=args.benchmark,
        preset_id=args.preset,
        profile=args.profile,
        task_id=args.task_id,
        shared_dir=args.shared_dir,
        shared_cache=args.shared_cache,
        campaign_id=args.campaign_id,
        campaign_dir=args.campaign_dir,
        methods=args.method,
        conditions=args.condition,
        seeds=args.seed,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        pi_provider_id=args.pi_provider_id,
        pi_api=args.pi_api,
        pi_api_key_env=args.pi_api_key_env,
        pi_api_base_env=args.pi_api_base_env,
        wall_time_seconds=args.wall_time_seconds,
        live_search_concurrency=args.live_search_concurrency,
        cell_concurrency=args.cell_concurrency,
        worker_runtime_seconds=args.worker_runtime_seconds,
        worker_min_runtime_seconds=args.worker_min_runtime_seconds,
        retain_containers=args.retain_containers,
    )


def render_catalog(agent: BenchmarkAgent, *, as_json: bool) -> int:
    payload = agent.catalog.as_dict()
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0
    for runner in payload["runners"]:
        capabilities = runner["capabilities"]
        print(
            f"runner {runner['id']}: {runner['kind']} "
            f"detach={capabilities['detach']} stop={capabilities['stop']} "
            f"resume={capabilities['resume']} R={capabilities['repeat_seeds']} "
            f"C={capabilities['cell_concurrency']} "
            f"retain={capabilities['retain_containers']} "
            f"methods={','.join(runner['supported_methods'])}"
        )
    for target in payload["targets"]:
        docker = target["docker"]
        print(
            f"{target['id']}: runner={target['runner']} adapter={target['adapter'] or '-'} "
            f"docker={docker['requirement']}/{docker['owner']}/{docker['provision_mode']} "
            f"assets={target['local_asset_inventory']} "
            f"asset-profile={target['default_inventory_profile'] or '-'}"
        )
    for pack in payload["asset_packs"]:
        print(
            f"asset-pack {pack['id']}: profile={pack['default_profile']} "
            f"provision={pack['provision']} assets=True"
        )
    for preset in payload["presets"]:
        print(f"preset {preset['id']}: {preset['description']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = BenchmarkAgent()
    if args.command == "catalog":
        return render_catalog(agent, as_json=args.json)
    if args.command == "scaffold":
        result = scaffold_benchmark(
            benchmark_id=args.benchmark_id,
            shape=args.shape,
            module=args.module,
            write=args.write,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "setup":
        if args.asset_pack:
            if (
                args.benchmark
                or args.preset
                or args.method
                or args.model
                or args.reasoning_effort
                or args.pi_provider_id
                or args.pi_api
                or args.pi_api_key_env
                or args.pi_api_base_env
            ):
                raise ContractError(
                    "asset-pack setup does not accept benchmark, preset, method, "
                    "model, or reasoning effort"
                )
            packs = agent.resolve_asset_packs(args.asset_pack)
            result = agent.setup_asset_packs(
                packs,
                profile=args.profile,
                skip_bootstrap=args.skip_bootstrap,
                skip_provision=args.skip_provision,
                dry_run=args.dry_run,
            )
        else:
            targets, preset = agent.resolve_targets(
                target_ids=args.benchmark, preset_id=args.preset
            )
            result = agent.setup(
                targets,
                pi_provider_id=args.pi_provider_id,
                pi_api=args.pi_api,
                pi_api_key_env=args.pi_api_key_env,
                pi_api_base_env=args.pi_api_base_env,
                profile=args.profile or (preset.profile if preset else None),
                methods=tuple(
                    args.method
                    or (
                        (preset.expected_profile.get("methods") or [])
                        if preset
                        else []
                    )
                ),
                model=(
                    args.model
                    or (
                        preset.expected_profile.get("model") if preset else None
                    )
                ),
                reasoning_effort=(
                    args.reasoning_effort
                    or (
                        preset.expected_profile.get("reasoning_effort")
                        if preset
                        else None
                    )
                ),
                skip_bootstrap=args.skip_bootstrap,
                skip_provision=args.skip_provision,
                dry_run=args.dry_run,
            )
    elif args.command in {"plan", "start", "launch", "e2e"}:
        if args.command == "e2e" and args.prepare_only:
            raise ContractError("e2e cannot be combined with --prepare-only")
        spec = spec_from_args(agent, args)
        dry_run = args.dry_run or args.command == "plan"
        result = agent.start(
            spec,
            skip_bootstrap=args.skip_bootstrap,
            skip_provision=args.skip_provision,
            prepare_only=args.prepare_only,
            foreground=args.foreground or args.command == "e2e",
            dry_run=dry_run,
        )
        if args.command == "e2e" and not dry_run:
            result = {
                "start": result,
                "finish": agent.finish(
                    result["campaign"]["path"],
                    benchmark=None,
                    markdown_out=args.markdown_out,
                    xlsx_out=args.xlsx_out,
                    dry_run=False,
                ),
            }
    elif args.command == "status":
        result = agent.status(args.campaign, benchmark=args.benchmark)
    elif args.command == "stop":
        result = agent.stop(args.campaign, benchmark=args.benchmark, dry_run=args.dry_run)
    elif args.command == "resume":
        result = agent.resume(args.campaign, benchmark=args.benchmark, dry_run=args.dry_run)
    elif args.command == "finish":
        result = agent.finish(
            args.campaign,
            benchmark=args.benchmark,
            markdown_out=args.markdown_out,
            xlsx_out=args.xlsx_out,
            dry_run=args.dry_run,
        )
    elif args.command == "check":
        if args.environment:
            if args.benchmark or args.preset or args.profile or args.asset_pack:
                raise ContractError(
                    "use --environment by itself; it checks every managed repository"
                )
            result = agent.check_environment(
                assume_yes=args.yes,
                dry_run=args.dry_run,
            )
        elif args.asset_pack:
            if args.yes:
                raise ContractError("--yes is only valid with --environment")
            if args.benchmark or args.preset:
                raise ContractError(
                    "use --asset-pack or --benchmark/--preset, not both"
                )
            packs = agent.resolve_asset_packs(args.asset_pack)
            if len(packs) != 1:
                raise ContractError("check accepts exactly one asset pack")
            result = agent.check_asset_pack(
                packs[0], profile=args.profile, dry_run=args.dry_run
            )
        else:
            if args.yes:
                raise ContractError("--yes is only valid with --environment")
            targets, preset = agent.resolve_targets(
                target_ids=args.benchmark, preset_id=args.preset
            )
            if len(targets) != 1:
                raise ContractError("check accepts exactly one benchmark target")
            result = agent.check(
                targets[0].target_id,
                profile=args.profile or (preset.profile if preset else None),
                dry_run=args.dry_run,
            )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))
    return 0


def entrypoint(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (BenchGoalPlusError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2
