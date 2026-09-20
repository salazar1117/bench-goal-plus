#!/usr/bin/env bash

set -uo pipefail
umask 077

# Locate the bench-goal-plus repository root (WORK_DIR) so this script runs from
# wherever it lives — including its committed home under adapters/zsoft_detect/.
# Resolution order:
#   1. $BENCH_GOAL_PLUS_WORK_DIR if set (explicit override)
#   2. walk up from this script's real directory to the first ancestor that
#      looks like a bench-goal-plus checkout (has scripts/bench.py).
resolve_work_dir() {
    if [[ -n "${BENCH_GOAL_PLUS_WORK_DIR:-}" ]]; then
        printf '%s' "${BENCH_GOAL_PLUS_WORK_DIR%/}"
        return 0
    fi
    local src="${BASH_SOURCE[0]}"
    # Resolve symlinks to the script's real path.
    while [[ -L "$src" ]]; do
        local dir
        dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
        src="$(readlink "$src")"
        [[ "$src" != /* ]] && src="$dir/$src"
    done
    local dir
    dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/scripts/bench.py" ]]; then
            printf '%s' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

readonly WORK_DIR="$(resolve_work_dir)"
[[ -n "$WORK_DIR" && -f "$WORK_DIR/scripts/bench.py" ]] \
    || { printf '错误: 无法定位 bench-goal-plus 仓库根 (设置 BENCH_GOAL_PLUS_WORK_DIR 可覆盖)\n' >&2; exit 1; }
readonly CAMPAIGN_BASE="runs/benchmark-campaigns-v2-detect"
readonly ZSOFT_CHECKOUT="$WORK_DIR/third_party/zsoft-bench"
readonly DETECT_ROOT="$ZSOFT_CHECKOUT/benchmarks/vulnerability/zsoft-detect"
readonly PROJECTS_DIR="$DETECT_ROOT/projects"
readonly L1_SOURCE_CACHE="$ZSOFT_CHECKOUT/benchmarks/vulnerability/zsoft-l1/.cache/sources"
readonly PYTHON="$WORK_DIR/.bench-env/venv/bin/python"
readonly EXPECTED_CASES="${DETECT_EXPECTED_CASES:-5}"
readonly SHARED_CACHE_ENABLED="${DETECT_SHARED_CACHE:-0}"

MODE="all"
REQUESTED_CASE_ID=""
WALL_TIME_SECONDS=1800
LIVE_SEARCH_CONCURRENCY=4
CELL_CONCURRENCY=1
REPEAT_COUNT=1
MAX_INFRA_ATTEMPTS="${DETECT_MAX_INFRA_ATTEMPTS:-2}"
STATUS_INTERVAL_SECONDS="${DETECT_STATUS_INTERVAL_SECONDS:-60}"
FETCH_MISSING_SOURCES="${DETECT_FETCH_MISSING_SOURCES:-1}"
ACTIVE_HEARTBEAT_PID=""
SOURCE_CACHE_ROOT=""
SOURCE_CACHE_STORAGE_MODE=""
ACTIVE_SOURCE_CHECKOUT=""

usage() {
    cat <<'USAGE'
用法:
  ./run_all_detect_cases.sh
  ./run_all_detect_cases.sh --one
  ./run_all_detect_cases.sh --case-id PROJECT_OR_TASK_ID
  ./run_all_detect_cases.sh --case-id civetweb-detect -T 3600 -K 2 -C 1 -R 3
  ./run_all_detect_cases.sh --list-cases

选项:
  --one                         优先选择已有源码缓存的 detect 项目做真实验证
  --case-id ID                  只运行指定项目；支持 civetweb 或 civetweb-detect
  --list-cases                  列出当前 ZSoft checkout 中全部可用 CASE_ID 后退出
  -T, --wall-time-seconds       T: 每个 search 的墙钟预算，默认 1800，当前最小 1260
  -K, --live-search-concurrency
                                K: 每个 task cell 内部 subagent 数，默认 4
  -C, --cell-concurrency        C: 同时运行的 task cell 数，默认 1（当前仅支持 1）
  -R, --repeats                 R: 每个 case 的独立串行 campaign 数，使用 seed 1..R，默认 1
  -A, --max-infra-attempts      基础设施失败时的最大尝试次数，默认 2
  -h, --help                    显示帮助

可选环境变量:
  DEEPSEEK_API_KEY              已设置时不再交互读取
  DETECT_SOURCE_CACHE_ROOT      项目源码缓存根目录，每个 checkout 使用 PROJECT-COMMIT 命名
  DETECT_FETCH_MISSING_SOURCES  缺少源码时是否自动获取，1（默认）或 0
  DETECT_MAX_INFRA_ATTEMPTS     基础设施失败最大尝试次数
  DETECT_STATUS_INTERVAL_SECONDS 实验运行心跳间隔，默认 60 秒
  DETECT_SHARED_CACHE           1 时为 Goal Plus 运行开启 shared_cache 事实面与
                                LLM verifier 逐 finding 路径可行性分析（默认 0）

实验合同:
  默认配置为 model=deepseek-v4-flash、reasoning=high、T=1800、K=4、C=1、R=1。
  detect 搜索过程中只使用公开格式校验；不查看 F1，也不提前终止。
  所有 agent 退出且收尾完成后，控制器对合规提交评分；final-eval 记录最高 F1。
  F1 同分时选择 candidate ID 最小者，同一 candidate 选择较晚 iteration。
  有效 F1（包括 0）记为 COMPLETED；只有实验合同不完整才记为 INFRA_ERROR。
USAGE
}

die() {
    printf '错误: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

while (($# > 0)); do
    case "$1" in
        --one)
            [[ "$MODE" == "all" && -z "$REQUESTED_CASE_ID" ]] \
                || die "--one 不能重复，也不能和 --case-id 或 --list-cases 同时使用"
            MODE="one"
            shift
            ;;
        --case-id)
            (($# >= 2)) || die "--case-id 缺少参数"
            [[ "$MODE" == "all" && -z "$REQUESTED_CASE_ID" ]] \
                || die "--case-id 不能重复，也不能和 --one 或 --list-cases 同时使用"
            MODE="case"
            REQUESTED_CASE_ID="$2"
            shift 2
            ;;
        --list-cases)
            [[ "$MODE" == "all" && -z "$REQUESTED_CASE_ID" ]] \
                || die "--list-cases 不能重复，也不能和 --one 或 --case-id 同时使用"
            MODE="list"
            shift
            ;;
        -A|--max-infra-attempts)
            (($# >= 2)) || die "$1 缺少参数"
            MAX_INFRA_ATTEMPTS="$2"
            shift 2
            ;;
        -T|--wall-time-seconds)
            (($# >= 2)) || die "$1 缺少参数"
            WALL_TIME_SECONDS="$2"
            shift 2
            ;;
        -K|--live-search-concurrency)
            (($# >= 2)) || die "$1 缺少参数"
            LIVE_SEARCH_CONCURRENCY="$2"
            shift 2
            ;;
        -C|--cell-concurrency)
            (($# >= 2)) || die "$1 缺少参数"
            CELL_CONCURRENCY="$2"
            shift 2
            ;;
        -R|--repeats)
            (($# >= 2)) || die "$1 缺少参数"
            REPEAT_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $1"
            ;;
    esac
done

[[ "$EXPECTED_CASES" =~ ^[1-9][0-9]*$ ]] \
    || die "DETECT_EXPECTED_CASES 必须是正整数"
[[ "$WALL_TIME_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || die "T (--wall-time-seconds) 必须是正整数"
((WALL_TIME_SECONDS >= 1260)) \
    || die "T (--wall-time-seconds) 必须至少为 1260 秒（1200 秒 worker 预算 + 60 秒 closeout）"
[[ "$LIVE_SEARCH_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] \
    || die "K (--live-search-concurrency) 必须是正整数"
[[ "$CELL_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] \
    || die "C (--cell-concurrency) 必须是正整数"
((CELL_CONCURRENCY == 1)) \
    || die "当前 zsoft-detect common-matrix runner 尚未支持 C>1；--cell-concurrency 必须为 1"
[[ "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]] \
    || die "R (--repeats) 必须是正整数"
[[ "$MAX_INFRA_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
    || die "最大基础设施尝试次数必须是正整数"
[[ "$STATUS_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || die "DETECT_STATUS_INTERVAL_SECONDS 必须是正整数"
((STATUS_INTERVAL_SECONDS >= 10)) \
    || die "DETECT_STATUS_INTERVAL_SECONDS 不能小于 10 秒"
[[ "$FETCH_MISSING_SOURCES" == "0" || "$FETCH_MISSING_SOURCES" == "1" ]] \
    || die "DETECT_FETCH_MISSING_SOURCES 只能是 0 或 1"

cd "$WORK_DIR" || die "无法进入工作目录: $WORK_DIR"

for required in find sort; do
    require_command "$required"
done
[[ -d "$PROJECTS_DIR" ]] || die "detect projects 目录不存在: $PROJECTS_DIR"

mapfile -t ALL_PROJECT_IDS < <(
    find "$PROJECTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        LC_ALL=C sort
)

declare -A PROJECT_COMMITS=()
declare -a ALL_CASE_IDS=()
for project_id in "${ALL_PROJECT_IDS[@]}"; do
    [[ "$project_id" =~ ^[a-z0-9][a-z0-9-]*$ ]] \
        || die "发现不安全的 detect 项目 ID: $project_id"
    mapfile -t version_commits < <(
        find "$PROJECTS_DIR/$project_id/versions" \
            -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
            LC_ALL=C sort
    )
    ((${#version_commits[@]} == 1)) \
        || die "项目 $project_id 必须且只能有一个固定版本"
    [[ "${version_commits[0]}" =~ ^[0-9a-f]{40}$ ]] \
        || die "项目 $project_id 的固定 commit 非法"
    PROJECT_COMMITS["$project_id"]="${version_commits[0]}"
    ALL_CASE_IDS+=("${project_id}-detect")
done

if [[ "$MODE" == "list" ]]; then
    printf '可用 Detect CASE_ID（%d 个；也可省略 -detect 后缀）:\n' \
        "${#ALL_CASE_IDS[@]}"
    printf '  %s\n' "${ALL_CASE_IDS[@]}"
    exit 0
fi

((${#ALL_PROJECT_IDS[@]} == EXPECTED_CASES)) \
    || die "detect 项目数量为 ${#ALL_PROJECT_IDS[@]}，预期为 $EXPECTED_CASES"

for required in awk curl date flock git jq mktemp mv realpath rm sleep stat tar tee; do
    require_command "$required"
done
[[ -x "$PYTHON" ]] || die "Python 环境不存在: $PYTHON"

mkdir -p "$WORK_DIR/.tmp"
exec 9>"$WORK_DIR/.tmp/run-detect-cases.lock"
flock -n 9 || die "已有另一个 detect 批量或验证脚本正在运行"

if [[ -n "${DETECT_SOURCE_CACHE_ROOT:-}" ]]; then
    [[ "$DETECT_SOURCE_CACHE_ROOT" == /* && "$DETECT_SOURCE_CACHE_ROOT" != "/" ]] \
        || die "DETECT_SOURCE_CACHE_ROOT 必须是非根绝对路径"
    SOURCE_CACHE_ROOT="${DETECT_SOURCE_CACHE_ROOT%/}"
    SOURCE_CACHE_STORAGE_MODE="explicit"
elif [[ -d "$L1_SOURCE_CACHE" ]]; then
    l1_cache_resolved="$(realpath "$L1_SOURCE_CACHE")" \
        || die "无法解析已有 L1 源码缓存"
    SOURCE_CACHE_ROOT="$(dirname "$l1_cache_resolved")/detect-source-cache"
    SOURCE_CACHE_STORAGE_MODE="l1-cache-device"
else
    SOURCE_CACHE_ROOT="$WORK_DIR/.cache/zsoft-detect-sources"
    SOURCE_CACHE_STORAGE_MODE="worktree-local"
fi
mkdir -p "$SOURCE_CACHE_ROOT" || die "无法创建 detect 源码缓存: $SOURCE_CACHE_ROOT"
SOURCE_CACHE_ROOT="$(realpath "$SOURCE_CACHE_ROOT")" \
    || die "无法解析 detect 源码缓存路径"
[[ "$SOURCE_CACHE_ROOT" != "/" ]] || die "拒绝使用根目录作为源码缓存"

normalize_case_id() {
    local requested="$1"
    if [[ "$requested" == *-detect ]]; then
        printf '%s\n' "$requested"
    else
        printf '%s-detect\n' "$requested"
    fi
}

is_known_case() {
    local requested="$1"
    local case_id
    for case_id in "${ALL_CASE_IDS[@]}"; do
        [[ "$case_id" == "$requested" ]] && return 0
    done
    return 1
}

source_checkout_valid() {
    local source="$1"
    local expected_commit="$2"
    local resolved=""
    local top_level=""
    local actual_commit=""
    local dirty=""

    [[ -d "$source" ]] || return 1
    resolved="$(realpath "$source" 2>/dev/null)" || return 1
    top_level="$(git -C "$resolved" rev-parse --show-toplevel 2>/dev/null)" \
        || return 1
    [[ "$(realpath "$top_level" 2>/dev/null)" == "$resolved" ]] || return 1
    actual_commit="$(git -C "$resolved" rev-parse HEAD 2>/dev/null)" || return 1
    [[ "$actual_commit" == "$expected_commit" ]] || return 1
    dirty="$(git -C "$resolved" status --porcelain=v1 --untracked-files=all 2>/dev/null)" \
        || return 1
    [[ -z "$dirty" ]]
}

cached_source_path() {
    local project="$1"
    local commit="$2"
    local candidate
    for candidate in \
        "$SOURCE_CACHE_ROOT/$project-$commit" \
        "$SOURCE_CACHE_ROOT/$project"; do
        if source_checkout_valid "$candidate" "$commit"; then
            realpath "$candidate"
            return 0
        fi
    done
    return 1
}

select_one_case() {
    local case_id
    local project
    local commit
    for case_id in "${ALL_CASE_IDS[@]}"; do
        project="${case_id%-detect}"
        commit="${PROJECT_COMMITS[$project]}"
        if cached_source_path "$project" "$commit" >/dev/null; then
            printf '%s\n' "$case_id"
            return 0
        fi
    done
    printf '%s\n' "${ALL_CASE_IDS[0]}"
}

declare -a CASE_IDS=()
case "$MODE" in
    all)
        CASE_IDS=("${ALL_CASE_IDS[@]}")
        ;;
    one)
        CASE_IDS=("$(select_one_case)")
        ;;
    case)
        REQUESTED_CASE_ID="$(normalize_case_id "$REQUESTED_CASE_ID")"
        is_known_case "$REQUESTED_CASE_ID" \
            || die "未知 detect case: $REQUESTED_CASE_ID"
        CASE_IDS=("$REQUESTED_CASE_ID")
        ;;
esac

readonly TOTAL_TASK_CELLS=$((${#CASE_IDS[@]} * REPEAT_COUNT))

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    [[ -t 0 ]] || die "非交互运行前必须先 export DEEPSEEK_API_KEY"
    read -rsp '请输入 DEEPSEEK_API_KEY: ' DEEPSEEK_API_KEY
    printf '\n'
fi
[[ -n "$DEEPSEEK_API_KEY" ]] || die "DEEPSEEK_API_KEY 不能为空"
export DEEPSEEK_API_KEY
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
unset OPENAI_BASE_URL OPENAI_API_BASE_URL OPENAI_API_BASE

cleanup_runtime() {
    if [[ -n "$ACTIVE_HEARTBEAT_PID" ]]; then
        kill "$ACTIVE_HEARTBEAT_PID" >/dev/null 2>&1 || true
        wait "$ACTIVE_HEARTBEAT_PID" 2>/dev/null || true
        ACTIVE_HEARTBEAT_PID=""
    fi
    unset BENCH_GOAL_PLUS_ZSOFT_DETECT_SOURCE_CACHE
    unset DEEPSEEK_API_KEY
}

handle_interrupt() {
    printf '\n收到中断信号；不会自动重启当前 campaign。\n' >&2
    exit 130
}

trap cleanup_runtime EXIT
trap handle_interrupt INT TERM HUP

readonly RUN_STAMP="$(date +%Y%m%d-%H%M%S)-$$"
readonly LOG_DIR="$WORK_DIR/logs/detect_cases_$RUN_STAMP"
readonly SUMMARY_FILE="$LOG_DIR/summary.txt"
readonly PROGRESS_FILE="$LOG_DIR/progress.log"
mkdir -p "$LOG_DIR/cases"
: >"$PROGRESS_FILE"

status_message() {
    local message="$*"
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$message" \
        | tee -a "$PROGRESS_FILE"
}

case_status() {
    local log_file="$1"
    shift
    local message="$*"
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$message" \
        | tee -a "$PROGRESS_FILE" "$log_file"
}

{
    printf 'ZSoft Detect 运行汇总\n'
    printf '开始时间: %s\n' "$(date --iso-8601=seconds)"
    printf '配置: model=deepseek-v4-flash reasoning=high T=%d K=%d C=%d R=%d\n' \
        "$WALL_TIME_SECONDS" "$LIVE_SEARCH_CONCURRENCY" \
        "$CELL_CONCURRENCY" "$REPEAT_COUNT"
    printf '评分策略: 搜索期间仅 format_valid；收尾后隐藏评分并记录最高 F1\n'
    printf '源码缓存: %s\n' "$SOURCE_CACHE_ROOT"
    printf '源码缓存模式: %s\n' "$SOURCE_CACHE_STORAGE_MODE"
    printf '计划 case 数: %d\n' "${#CASE_IDS[@]}"
    printf '计划 task cell 数: %d\n' "$TOTAL_TASK_CELLS"
    printf 'case_id\tseed\tstatus\tf1\tattempts\tcampaign_id\treason\tcase_log\n'
} >"$SUMMARY_FILE"

status_message "INIT work_dir=$WORK_DIR"
status_message "INIT log_dir=$LOG_DIR progress_file=$PROGRESS_FILE"
status_message "INIT source_cache=$SOURCE_CACHE_ROOT storage_mode=$SOURCE_CACHE_STORAGE_MODE"
status_message \
    "CONFIG model=deepseek-v4-flash reasoning=high T=$WALL_TIME_SECONDS K=$LIVE_SEARCH_CONCURRENCY C=$CELL_CONCURRENCY R=$REPEAT_COUNT scoring=controller_posthoc_best_f1"
status_message \
    "QUEUE cases=${#CASE_IDS[@]} task_cells=$TOTAL_TASK_CELLS execution=serial"

PREFLIGHT_LOG="$LOG_DIR/preflight.log"
preflight_started="$(date +%s)"
status_message "PREFLIGHT_START log=$PREFLIGHT_LOG"
if {
    "$PYTHON" scripts/bench.py catalog
    "$PYTHON" scripts/bench.py setup \
        --benchmark zsoft-detect \
        --method goal-plus-pi \
        --model deepseek-v4-flash \
        --reasoning-effort high \
        --pi-provider-id deepseek \
        --pi-api openai-completions \
        --pi-api-key-env DEEPSEEK_API_KEY \
        --pi-api-base-env DEEPSEEK_BASE_URL \
        --skip-bootstrap \
        --skip-provision
} >"$PREFLIGHT_LOG" 2>&1; then
    status_message "PREFLIGHT_OK elapsed_seconds=$(($(date +%s) - preflight_started))"
else
    status_message "PREFLIGHT_FAILED elapsed_seconds=$(($(date +%s) - preflight_started)) log=$PREFLIGHT_LOG"
    die "运行前环境检查失败，详见 $PREFLIGHT_LOG"
fi

prepare_source_checkout() {
    local case_id="$1"
    local log_file="$2"
    local project="${case_id%-detect}"
    local commit="${PROJECT_COMMITS[$project]}"
    local target="$SOURCE_CACHE_ROOT/$project-$commit"
    local cached=""
    local staging_parent=""
    local staging=""
    local started=0

    ACTIVE_SOURCE_CHECKOUT=""
    cached="$(cached_source_path "$project" "$commit" 2>/dev/null)" || cached=""
    if [[ -n "$cached" ]]; then
        ACTIVE_SOURCE_CHECKOUT="$cached"
        case_status "$log_file" \
            "SOURCE_CACHE_REUSED case=$case_id commit=$commit path=$cached"
        return 0
    fi

    if [[ -e "$target" || -L "$target" ]]; then
        case_status "$log_file" \
            "SOURCE_CACHE_INVALID case=$case_id commit=$commit path=$target"
        return 1
    fi
    if [[ "$FETCH_MISSING_SOURCES" != "1" ]]; then
        case_status "$log_file" \
            "SOURCE_CACHE_MISSING case=$case_id commit=$commit auto_fetch=false"
        return 1
    fi

    staging_parent="$(mktemp -d "$SOURCE_CACHE_ROOT/.prepare-${project}.XXXXXX")" \
        || return 1
    staging="$staging_parent/checkout"
    started="$(date +%s)"
    case_status "$log_file" \
        "SOURCE_PREPARE_START case=$case_id commit=$commit staging_root=$staging_parent"
    if ! PYTHONPATH="$WORK_DIR" "$PYTHON" - \
        "$project" "$commit" "$staging" >>"$log_file" 2>&1 <<'PY'
import sys
from pathlib import Path

from adapters.zsoft_detect import adapter

adapter.fetch_source_checkout(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
PY
    then
        case_status "$log_file" \
            "SOURCE_PREPARE_FAILED case=$case_id elapsed_seconds=$(($(date +%s) - started))"
        case "$staging_parent" in
            "$SOURCE_CACHE_ROOT"/.prepare-*) rm -rf -- "$staging_parent" ;;
        esac
        return 1
    fi
    if ! source_checkout_valid "$staging" "$commit"; then
        case_status "$log_file" \
            "SOURCE_PREPARE_FAILED case=$case_id reason=checkout_validation_failed"
        case "$staging_parent" in
            "$SOURCE_CACHE_ROOT"/.prepare-*) rm -rf -- "$staging_parent" ;;
        esac
        return 1
    fi
    if ! mv "$staging" "$target"; then
        if ! source_checkout_valid "$target" "$commit"; then
            case_status "$log_file" \
                "SOURCE_PREPARE_FAILED case=$case_id reason=cache_publish_failed"
            case "$staging_parent" in
                "$SOURCE_CACHE_ROOT"/.prepare-*) rm -rf -- "$staging_parent" ;;
            esac
            return 1
        fi
    fi
    case "$staging_parent" in
        "$SOURCE_CACHE_ROOT"/.prepare-*) rm -rf -- "$staging_parent" ;;
    esac
    ACTIVE_SOURCE_CHECKOUT="$(realpath "$target")"
    case_status "$log_file" \
        "SOURCE_PREPARE_OK case=$case_id elapsed_seconds=$(($(date +%s) - started)) path=$ACTIVE_SOURCE_CHECKOUT"
    return 0
}

build_bench_args() {
    local case_id="$1"
    local campaign_id="$2"
    local seed="$3"
    BENCH_ARGS=(
        --benchmark zsoft-detect
        --task-id "$case_id"
        --campaign-id "$campaign_id"
        --campaign-dir "$WORK_DIR/$CAMPAIGN_BASE/$campaign_id"
        --method goal-plus-pi
        --seed "$seed"
        --model deepseek-v4-flash
        --reasoning-effort high
        --pi-provider-id deepseek
        --pi-api openai-completions
        --pi-api-key-env DEEPSEEK_API_KEY
        --pi-api-base-env DEEPSEEK_BASE_URL
        --wall-time-seconds "$WALL_TIME_SECONDS"
        --live-search-concurrency "$LIVE_SEARCH_CONCURRENCY"
        --cell-concurrency "$CELL_CONCURRENCY"
        --worker-runtime-seconds 1200
        --worker-min-runtime-seconds 600
        --skip-bootstrap
        --skip-provision
        --foreground
    )
    if [[ "$SHARED_CACHE_ENABLED" == "1" ]]; then
        BENCH_ARGS+=(--shared-cache)
    fi
}

start_launch_heartbeat() {
    local case_id="$1"
    local seed="$2"
    local campaign_dir="$3"
    local log_file="$4"
    local started_epoch="$5"

    (
        local campaign_state="not_created"
        local cell_state="not_created"
        local experiment_status="not_created"
        local final_eval_status="absent"
        local run_dir=""
        while true; do
            sleep "$STATUS_INTERVAL_SECONDS"
            campaign_state="not_created"
            cell_state="not_created"
            experiment_status="not_created"
            final_eval_status="absent"
            run_dir=""
            if [[ -f "$campaign_dir/campaign.json" ]]; then
                campaign_state="$(jq -r '.state // "unknown"' \
                    "$campaign_dir/campaign.json" 2>/dev/null)" \
                    || campaign_state="unreadable"
                cell_state="$(jq -r '
                    if (.cells | type) == "array" and (.cells | length) == 1
                    then .cells[0].state // "unknown"
                    else "unavailable"
                    end
                ' "$campaign_dir/campaign.json" 2>/dev/null)" \
                    || cell_state="unreadable"
                run_dir="$(jq -r '
                    if (.cells | type) == "array" and (.cells | length) == 1
                    then .cells[0].run_dir // empty
                    else empty
                    end
                ' "$campaign_dir/campaign.json" 2>/dev/null)" || run_dir=""
            fi
            case "$run_dir/" in
                "$campaign_dir"/cells/*/)
                    if [[ -f "$run_dir/experiment.json" ]]; then
                        experiment_status="$(jq -r '.status // "unknown"' \
                            "$run_dir/experiment.json" 2>/dev/null)" \
                            || experiment_status="unreadable"
                    fi
                    [[ -f "$run_dir/final-eval.json" ]] \
                        && final_eval_status="present"
                    ;;
            esac
            case_status "$log_file" \
                "EXPERIMENT_HEARTBEAT case=$case_id seed=$seed elapsed_seconds=$(($(date +%s) - started_epoch)) campaign_state=$campaign_state cell_state=$cell_state experiment_status=$experiment_status final_eval=$final_eval_status"
        done
    ) &
    ACTIVE_HEARTBEAT_PID=$!
}

stop_launch_heartbeat() {
    if [[ -n "$ACTIVE_HEARTBEAT_PID" ]]; then
        kill "$ACTIVE_HEARTBEAT_PID" >/dev/null 2>&1 || true
        wait "$ACTIVE_HEARTBEAT_PID" 2>/dev/null || true
        ACTIVE_HEARTBEAT_PID=""
    fi
}

declare -a BENCH_ARGS=()
ATTEMPT_RESULT="INFRA_ERROR"
ATTEMPT_F1=""
ATTEMPT_CAMPAIGN_ID=""
ATTEMPT_REASON="not_started"
ATTEMPT_LOG_FILE=""

campaign_contract_valid() {
    local case_id="$1"
    local seed="$2"
    local campaign="$3"

    jq -e \
        --arg case_id "$case_id" \
        --argjson seed "$seed" \
        --argjson wall_time "$WALL_TIME_SECONDS" \
        --argjson live_concurrency "$LIVE_SEARCH_CONCURRENCY" \
        --argjson cell_concurrency "$CELL_CONCURRENCY" '
        .task_id == $case_id and
        .methods == ["goal-plus-pi"] and
        .seeds == [$seed] and
        .model == "deepseek-v4-flash" and
        .reasoning_effort == "high" and
        .budget.wall_time_seconds == $wall_time and
        .budget.live_search_concurrency == $live_concurrency and
        .budget.requested_live_concurrency == $live_concurrency and
        .budget.cell_concurrency == $cell_concurrency and
        .budget.attempts == 1 and
        .budget.worker_runtime_seconds == 1200 and
        .budget.worker_min_runtime_seconds == 600 and
        (.cells | type) == "array" and
        (.cells | length) == 1 and
        .cells[0].benchmark_id == "zsoft-detect" and
        .cells[0].method == "goal-plus-pi" and
        .cells[0].seed == $seed and
        .cells[0].effective_concurrency == $live_concurrency
    ' "$campaign" >/dev/null 2>&1
}

run_case_attempt() {
    local case_id="$1"
    local seed="$2"
    local attempt="$3"
    local task_cell_number="$4"
    local project="${case_id%-detect}"
    local campaign_id=""
    if ((REPEAT_COUNT == 1)); then
        campaign_id="detect-${project}-${RUN_STAMP}-a${attempt}"
    else
        campaign_id="detect-${project}-seed-${seed}-${RUN_STAMP}-a${attempt}"
    fi
    local campaign_dir="$WORK_DIR/$CAMPAIGN_BASE/$campaign_id"
    local case_log_dir="$LOG_DIR/cases/$case_id"
    if ((REPEAT_COUNT > 1)); then
        case_log_dir="$case_log_dir/seed-$seed"
    fi
    local log_file="$case_log_dir/attempt-${attempt}.log"
    local launch_rc=0
    local campaign_state=""
    local cell_state=""
    local run_dir=""
    local experiment=""
    local final_eval=""
    local score_record=""
    local stage_started=0
    local core_reason=""

    ATTEMPT_RESULT="INFRA_ERROR"
    ATTEMPT_F1=""
    ATTEMPT_CAMPAIGN_ID="$campaign_id"
    ATTEMPT_REASON="attempt_started"
    ATTEMPT_LOG_FILE="$log_file"
    mkdir -p "$case_log_dir"
    : >"$log_file"
    case_status "$log_file" \
        "CASE_ATTEMPT_START case=$case_id seed=$seed position=$task_cell_number/$TOTAL_TASK_CELLS attempt=$attempt/$MAX_INFRA_ATTEMPTS campaign_id=$campaign_id"

    stage_started="$(date +%s)"
    case_status "$log_file" "SOURCE_CHECK_START case=$case_id"
    if ! prepare_source_checkout "$case_id" "$log_file"; then
        ATTEMPT_REASON="source_checkout_unavailable"
        case_status "$log_file" \
            "SOURCE_CHECK_FAILED case=$case_id elapsed_seconds=$(($(date +%s) - stage_started))"
        return 0
    fi
    export BENCH_GOAL_PLUS_ZSOFT_DETECT_SOURCE_CACHE="$ACTIVE_SOURCE_CHECKOUT"
    case_status "$log_file" \
        "SOURCE_CHECK_OK case=$case_id elapsed_seconds=$(($(date +%s) - stage_started))"

    build_bench_args "$case_id" "$campaign_id" "$seed"
    stage_started="$(date +%s)"
    case_status "$log_file" "PLAN_START case=$case_id campaign_id=$campaign_id"
    "$PYTHON" scripts/bench.py plan "${BENCH_ARGS[@]}" >>"$log_file" 2>&1
    if (($? != 0)); then
        ATTEMPT_REASON="plan_failed"
        case_status "$log_file" \
            "PLAN_FAILED case=$case_id elapsed_seconds=$(($(date +%s) - stage_started))"
        return 0
    fi
    case_status "$log_file" \
        "PLAN_OK case=$case_id elapsed_seconds=$(($(date +%s) - stage_started))"

    stage_started="$(date +%s)"
    case_status "$log_file" \
        "EXPERIMENT_LAUNCH_START case=$case_id campaign_id=$campaign_id campaign_dir=$campaign_dir fresh_campaign=true"
    start_launch_heartbeat \
        "$case_id" "$seed" "$campaign_dir" "$log_file" "$stage_started"
    "$PYTHON" scripts/bench.py launch "${BENCH_ARGS[@]}" >>"$log_file" 2>&1
    launch_rc=$?
    stop_launch_heartbeat
    case_status "$log_file" \
        "EXPERIMENT_LAUNCH_EXIT case=$case_id exit_code=$launch_rc elapsed_seconds=$(($(date +%s) - stage_started))"

    if [[ ! -f "$campaign_dir/campaign.json" || -L "$campaign_dir/campaign.json" ]]; then
        ATTEMPT_REASON="campaign_manifest_missing_after_launch_${launch_rc}"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    fi
    campaign_state="$(jq -er '.state // empty' "$campaign_dir/campaign.json" 2>/dev/null)" || {
        ATTEMPT_REASON="campaign_manifest_invalid"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    }
    if ! campaign_contract_valid \
        "$case_id" "$seed" "$campaign_dir/campaign.json"; then
        ATTEMPT_REASON="campaign_contract_invalid"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id seed=$seed reason=$ATTEMPT_REASON"
        return 0
    fi
    cell_state="$(jq -r '
        if (.cells | type) == "array" and (.cells | length) == 1
        then .cells[0].state // empty
        else empty
        end
    ' "$campaign_dir/campaign.json" 2>/dev/null)"
    run_dir="$(jq -er '
        if (.cells | type) == "array" and (.cells | length) == 1
        then .cells[0].run_dir
        else empty
        end
    ' "$campaign_dir/campaign.json" 2>/dev/null)" || {
        ATTEMPT_REASON="run_dir_missing"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    }
    case "$run_dir/" in
        "$campaign_dir"/cells/*/) ;;
        *)
            ATTEMPT_REASON="run_dir_outside_campaign"
            case_status "$log_file" \
                "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
            return 0
            ;;
    esac
    experiment="$run_dir/experiment.json"
    final_eval="$run_dir/final-eval.json"
    score_record="$run_dir/posthoc-candidate-scores.json"

    if ((launch_rc != 0)) \
        || [[ "$campaign_state" != "finished" ]] \
        || [[ "$cell_state" != "finished" ]]; then
        core_reason="unavailable"
        if [[ -f "$experiment" && ! -L "$experiment" ]]; then
            core_reason="$(jq -r \
                '.execution.result_incomplete_reason // "unavailable"' \
                "$experiment" 2>/dev/null)" || core_reason="unreadable"
            core_reason="${core_reason//$'\t'/ }"
            core_reason="${core_reason//$'\n'/ }"
        fi
        if [[ "$campaign_state" != "finished" ]]; then
            ATTEMPT_REASON="campaign_state_${campaign_state:-missing}"
        elif [[ "$cell_state" != "finished" ]]; then
            ATTEMPT_REASON="cell_state_${cell_state:-missing}"
        else
            ATTEMPT_REASON="launch_exit_code_${launch_rc}"
        fi
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON launch_exit_code=$launch_rc cell_state=${cell_state:-missing} core_reason=$core_reason"
        return 0
    fi

    if [[ ! -f "$experiment" || -L "$experiment" ]]; then
        ATTEMPT_REASON="experiment_manifest_missing"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    fi
    if [[ ! -f "$final_eval" || -L "$final_eval" ]]; then
        ATTEMPT_REASON="final_eval_missing"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    fi
    if [[ ! -f "$score_record" || -L "$score_record" ]]; then
        ATTEMPT_REASON="posthoc_score_record_missing"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    fi

    case_status "$log_file" "RESULT_VALIDATION_START case=$case_id"
    jq -e \
        --arg case_id "$case_id" \
        --argjson seed "$seed" \
        --argjson wall_time "$WALL_TIME_SECONDS" \
        --argjson live_concurrency "$LIVE_SEARCH_CONCURRENCY" '
        .status == "finished" and
        .benchmark_adapter == "zsoft-detect" and
        .benchmark_task_selector == $case_id and
        .task_id == $case_id and
        .method == "goal-plus-pi" and
        .seed == $seed and
        .model == "deepseek-v4-flash" and
        .reasoning_effort == "high" and
        .budget.wall_time_seconds == $wall_time and
        .budget.concurrency == $live_concurrency and
        .budget.worker_runtime_seconds == 1200 and
        .budget.worker_min_runtime_seconds == 600 and
        .task.primary_metric == "f1" and
        .task.goal_plus_process_metric == "format_valid" and
        .task.controller_only_official_evaluation == true and
        .task.evaluation_mode == "blind" and
        .task.requires_protected_pi_workers == true and
        .task.goal_plus_posthoc_selection == {
            "enabled": true,
            "metric_name": "f1",
            "metric_direction": "maximize",
            "candidate_scope": "all_publicly_compliant_iterations",
            "tie_break": "lowest_candidate_id_then_latest_iteration",
            "timing": "after_agent_exit_and_controller_closeout",
            "visible_to_workers": false
        } and
        .goal_plus_config.command_config.max_parallel == $live_concurrency and
        .goal_plus_config.metric_name == "format_valid" and
        .goal_plus_config.controller_only_official_evaluation == true and
        ((.goal_plus_config.early_stop // null) == null) and
        .goal_plus_config.posthoc_selection == .task.goal_plus_posthoc_selection and
        .pi_provider.id == "deepseek" and
        .pi_provider.api == "openai-completions" and
        .pi_provider.api_key_env == "DEEPSEEK_API_KEY" and
        .execution.goal_plus_controller_closeout.completed == true and
        .execution.evaluator_calls.setup_claimed_before_t == 1 and
        .execution.posthoc_official_selection.completed == true and
        .execution.posthoc_official_selection.visible_to_workers == false and
        .execution.posthoc_official_selection.affects_online_search == false and
        .execution.posthoc_official_selection.worker_shutdown_verified == true and
        .execution.posthoc_official_selection.official_evaluator_calls >= 1 and
        .execution.evaluator_calls.controller_final_claimed == 1 and
        .execution.posthoc_official_selection.official_evaluator_calls ==
            .execution.posthoc_official_selection.unique_artifact_count and
        (.execution.candidate_session_cleanup | type) == "array" and
        (.execution.candidate_session_cleanup | length) > 0 and
        all(.execution.candidate_session_cleanup[]; .active_count == 0) and
        (.execution.early_stop_triggered != true) and
        ((.execution.result_incomplete_reason // null) == null) and
        (.execution.goal_plus_controller_closeout.runs | type) == "array" and
        (.execution.goal_plus_controller_closeout.runs | length) > 0 and
        all(.execution.goal_plus_controller_closeout.runs[];
            .selection.selection_rule == "lowest_candidate_id_latest_compliant_iteration")
    ' "$experiment" >/dev/null 2>&1 || {
        ATTEMPT_REASON="experiment_contract_invalid"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    }

    jq -e --arg case_id "$case_id" '
        .schema_version == 1 and
        .task_id == $case_id and
        .mode == "final" and
        .valid == true and
        .format_valid == true and
        (.f1 | type) == "number" and
        .f1 >= 0 and .f1 <= 1 and
        .primary_metric.name == "f1" and
        .primary_metric.direction == "maximize" and
        .primary_metric.value == .f1 and
        (.zsoft_score | type) == "object" and
        .zsoft_score.f1 == .f1 and
        .budget.total_claimed == 1 and
        .budget.final_claimed == 1 and
        .posthoc_selection.visible_to_workers == false and
        .posthoc_selection.timing == "after_agent_exit_and_controller_closeout" and
        .posthoc_selection.tie_break == "lowest_candidate_id_then_latest_iteration" and
        .posthoc_selection.official_evaluator_calls >= 1
    ' "$final_eval" >/dev/null 2>&1 || {
        ATTEMPT_REASON="final_eval_contract_invalid"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    }

    jq -e -s --arg score_record "$score_record" '
        .[0] as $experiment |
        .[1] as $final |
        .[2] as $scores |
        $experiment.execution.posthoc_official_selection as $posthoc |
        ($scores.scores | map(.f1) | max) as $best_f1 |
        ($scores.scores | map(select(.f1 == $best_f1)) |
            map(.candidate_id) | min) as $best_candidate |
        ($scores.scores | map(select(
            .f1 == $best_f1 and .candidate_id == $best_candidate
        )) | map(.iteration) | max) as $best_iteration |
        $scores.completed == true and
        $scores.errors == [] and
        $scores.score_record_path == $score_record and
        $scores.contract == $experiment.task.goal_plus_posthoc_selection and
        $scores.selected == $posthoc.selected and
        $experiment.execution.evaluator_calls.controller_final_claimed == 1 and
        $scores.unique_artifact_count == $scores.official_evaluator_calls and
        $scores.eligible_iteration_count == ($scores.scores | length) and
        $scores.artifact_cache_hits ==
            ($scores.eligible_iteration_count - $scores.unique_artifact_count) and
        $final.f1 == $best_f1 and
        $scores.selected.f1 == $best_f1 and
        $scores.selected.candidate_id == $best_candidate and
        $scores.selected.iteration == $best_iteration and
        $final.posthoc_selection.candidate_id == $scores.selected.candidate_id and
        $final.posthoc_selection.iteration == $scores.selected.iteration and
        $final.posthoc_selection.git_head == $scores.selected.git_head and
        $final.posthoc_selection.snapshot_sha256 == $scores.selected.snapshot_sha256
    ' "$experiment" "$final_eval" "$score_record" >/dev/null 2>&1 || {
        ATTEMPT_REASON="posthoc_selection_contract_invalid"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    }

    ATTEMPT_F1="$(jq -er '.f1' "$final_eval")" || {
        ATTEMPT_REASON="f1_unreadable"
        case_status "$log_file" \
            "RESULT_VALIDATION_FAILED case=$case_id reason=$ATTEMPT_REASON"
        return 0
    }
    ATTEMPT_RESULT="COMPLETED"
    ATTEMPT_REASON="valid_f1"
    case_status "$log_file" \
        "RESULT_VALIDATION_OK case=$case_id seed=$seed result=COMPLETED f1=$ATTEMPT_F1"
}

COMPLETED_COUNT=0
INFRA_COUNT=0
declare -a INFRA_CASES=()

for case_index in "${!CASE_IDS[@]}"; do
    case_id="${CASE_IDS[$case_index]}"
    for ((seed = 1; seed <= REPEAT_COUNT; seed++)); do
        task_cell_number=$((case_index * REPEAT_COUNT + seed))
        case_result="INFRA_ERROR"
        case_f1=""
        attempts_used=0
        final_campaign_id=""
        final_reason="not_started"
        final_log_file=""

        for ((attempt = 1; attempt <= MAX_INFRA_ATTEMPTS; attempt++)); do
            attempts_used="$attempt"
            run_case_attempt "$case_id" "$seed" "$attempt" "$task_cell_number"
            case_result="$ATTEMPT_RESULT"
            case_f1="$ATTEMPT_F1"
            final_campaign_id="$ATTEMPT_CAMPAIGN_ID"
            final_reason="$ATTEMPT_REASON"
            final_log_file="$ATTEMPT_LOG_FILE"
            [[ "$case_result" == "COMPLETED" ]] && break
            if ((attempt < MAX_INFRA_ATTEMPTS)); then
                status_message \
                    "CASE_RETRY case=$case_id seed=$seed position=$task_cell_number/$TOTAL_TASK_CELLS attempt=$attempt reason=$final_reason"
            fi
        done

        if [[ "$case_result" == "COMPLETED" ]]; then
            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            status_message \
                "CASE_COMPLETE case=$case_id seed=$seed position=$task_cell_number/$TOTAL_TASK_CELLS result=COMPLETED f1=$case_f1 attempts=$attempts_used"
        else
            INFRA_COUNT=$((INFRA_COUNT + 1))
            INFRA_CASES+=("$case_id:seed-$seed")
            status_message \
                "CASE_COMPLETE case=$case_id seed=$seed position=$task_cell_number/$TOTAL_TASK_CELLS result=INFRA_ERROR attempts=$attempts_used reason=$final_reason"
        fi

        printf '%s\t%d\t%s\t%s\t%d\t%s\t%s\t%s\n' \
            "$case_id" "$seed" "$case_result" "${case_f1:-n/a}" \
            "$attempts_used" "$final_campaign_id" "$final_reason" \
            "$final_log_file" >>"$SUMMARY_FILE"

        unset BENCH_GOAL_PLUS_ZSOFT_DETECT_SOURCE_CACHE
        if ((task_cell_number < TOTAL_TASK_CELLS)); then
            sleep 2
        fi
    done
done

if ((COMPLETED_COUNT > 0)); then
    MEAN_F1="$(awk -F '\t' '
        $3 == "COMPLETED" { total += $4; count++ }
        END { if (count > 0) printf "%.6f", total / count; else print "n/a" }
    ' "$SUMMARY_FILE")"
else
    MEAN_F1="n/a"
fi

{
    printf '\n结束时间: %s\n' "$(date --iso-8601=seconds)"
    printf '计划 task cell: %d\n' "$TOTAL_TASK_CELLS"
    printf '有效完成: %d\n' "$COMPLETED_COUNT"
    printf 'INFRA_ERROR: %d\n' "$INFRA_COUNT"
    printf '有效结果平均 F1: %s\n' "$MEAN_F1"
    if ((INFRA_COUNT > 0)); then
        printf '全量平均 F1: unavailable（仍有基础设施失败需要重跑）\n'
        printf '待重跑 case/seed: %s\n' "${INFRA_CASES[*]}"
    else
        printf '全量平均 F1: %s\n' "$MEAN_F1"
    fi
} >>"$SUMMARY_FILE"

status_message \
    "RUN_COMPLETE planned=$TOTAL_TASK_CELLS completed=$COMPLETED_COUNT infra_error=$INFRA_COUNT mean_f1=$MEAN_F1 summary=$SUMMARY_FILE"

printf '\n运行结束：有效完成 %d/%d，INFRA_ERROR=%d。\n' \
    "$COMPLETED_COUNT" "$TOTAL_TASK_CELLS" "$INFRA_COUNT"
printf '有效结果平均 F1: %s\n' "$MEAN_F1"
printf '汇总文件: %s\n' "$SUMMARY_FILE"

if ((INFRA_COUNT > 0)); then
    exit 2
fi
exit 0
