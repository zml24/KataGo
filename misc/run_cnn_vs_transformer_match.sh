#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  run_cnn_vs_transformer_match.sh KATAGO_BIN CNN_MODEL TF_MODEL OUTPUT_DIR [NUM_GAMES] [MAX_VISITS]

参数:
  KATAGO_BIN   katago 可执行文件
  CNN_MODEL    CNN 模型路径，例如 .bin.gz
  TF_MODEL     Transformer 模型路径，例如 .kgtr.gz
  OUTPUT_DIR   输出目录
  NUM_GAMES    可选，默认 500
  MAX_VISITS   可选，默认 500

环境变量:
  CNN_NAME                       默认取 CNN_MODEL 文件名
  TF_NAME                        默认取 TF_MODEL 文件名
  CNN_USE_FP16                   CNN 路径精度开关，默认 auto
  TF_PRECISION                   Transformer 精度，默认 fp16，可选 fp32/fp16/bf16
  NUM_GAME_THREADS               默认 100
  NUM_SEARCH_THREADS             默认 1
  NUM_NN_SERVER_THREADS_PER_MODEL 默认 1
  NN_MAX_BATCH_SIZE              默认 64
  NN_CACHE_SIZE_POWER_OF_TWO     默认 24
  NN_MUTEX_POOL_SIZE_POWER_OF_TWO 默认 17
  MAX_MOVES_PER_GAME             默认 1200
  B_SIZES                        默认 19
  B_SIZE_REL_PROBS               默认 1
  ALLOW_RESIGNATION              默认 true
  RESIGN_THRESHOLD               默认 -0.95
  RESIGN_CONSEC_TURNS            默认 6
  RULES_KO                       默认 SIMPLE
  RULES_SCORING                  默认 AREA
  RULES_TAX                      默认 NONE
  RULES_MULTI_STONE_SUICIDE      默认 false
  RULES_HAS_BUTTON               默认 true
  KOMI_AUTO                      默认 false
  KOMI_MEAN                      默认 7.0
  HANDICAP_PROB                  默认 0.0
  CHOSEN_MOVE_TEMPERATURE_EARLY  默认 0.60
  CHOSEN_MOVE_TEMPERATURE        默认 0.20
  CHOSEN_MOVE_TEMPERATURE_HALFLIFE 可选，默认不写入
  CHOSEN_MOVE_SUBTRACT           可选，默认不写入
  CHOSEN_MOVE_PRUNE              可选，默认不写入
  ROOT_NUM_SYMMETRIES_TO_SAMPLE  可选，默认不写入
  USE_LCB_FOR_SELECTION          可选，默认不写入
  LCB_STDEVS                     可选，默认不写入
  MIN_VISIT_PROP_FOR_LCB         可选，默认不写入
  EXTRA_CONFIG_FILE              可选，追加一个额外配置文件内容
  EXTRA_CONFIG_TEXT              可选，直接追加一段额外配置文本
  CUDA_GPU_TO_USE                可选，写入 cudaGpuToUse

示例:
  TF_PRECISION=bf16 \
  NUM_GAME_THREADS=16 \
  ./misc/run_cnn_vs_transformer_match.sh \
    /path/to/katago \
    /path/to/cnn.bin.gz \
    /path/to/model.kgtr.gz \
    /tmp/cnn-vs-tf \
    400 \
    800
EOF
}

if [[ $# -lt 4 || $# -gt 6 ]]; then
  usage
  exit 1
fi

KATAGO_BIN=$1
CNN_MODEL=$2
TF_MODEL=$3
OUTPUT_DIR=$4
NUM_GAMES=${5:-500}
MAX_VISITS=${6:-500}

CNN_NAME=${CNN_NAME:-$(basename "$CNN_MODEL")}
TF_NAME=${TF_NAME:-$(basename "$TF_MODEL")}
CNN_USE_FP16=${CNN_USE_FP16:-auto}
TF_PRECISION=${TF_PRECISION:-fp16}
NUM_GAME_THREADS=${NUM_GAME_THREADS:-100}
NUM_SEARCH_THREADS=${NUM_SEARCH_THREADS:-1}
NUM_NN_SERVER_THREADS_PER_MODEL=${NUM_NN_SERVER_THREADS_PER_MODEL:-1}
NN_MAX_BATCH_SIZE=${NN_MAX_BATCH_SIZE:-64}
NN_CACHE_SIZE_POWER_OF_TWO=${NN_CACHE_SIZE_POWER_OF_TWO:-24}
NN_MUTEX_POOL_SIZE_POWER_OF_TWO=${NN_MUTEX_POOL_SIZE_POWER_OF_TWO:-17}
MAX_MOVES_PER_GAME=${MAX_MOVES_PER_GAME:-1200}
B_SIZES=${B_SIZES:-19}
B_SIZE_REL_PROBS=${B_SIZE_REL_PROBS:-1}
ALLOW_RESIGNATION=${ALLOW_RESIGNATION:-true}
RESIGN_THRESHOLD=${RESIGN_THRESHOLD:--0.95}
RESIGN_CONSEC_TURNS=${RESIGN_CONSEC_TURNS:-6}
RULES_KO=${RULES_KO:-SIMPLE}
RULES_SCORING=${RULES_SCORING:-AREA}
RULES_TAX=${RULES_TAX:-NONE}
RULES_MULTI_STONE_SUICIDE=${RULES_MULTI_STONE_SUICIDE:-false}
RULES_HAS_BUTTON=${RULES_HAS_BUTTON:-true}
KOMI_AUTO=${KOMI_AUTO:-false}
KOMI_MEAN=${KOMI_MEAN:-7.0}
HANDICAP_PROB=${HANDICAP_PROB:-0.0}
CHOSEN_MOVE_TEMPERATURE_EARLY=${CHOSEN_MOVE_TEMPERATURE_EARLY:-0.60}
CHOSEN_MOVE_TEMPERATURE=${CHOSEN_MOVE_TEMPERATURE:-0.20}
CHOSEN_MOVE_TEMPERATURE_HALFLIFE=${CHOSEN_MOVE_TEMPERATURE_HALFLIFE:-}
CHOSEN_MOVE_SUBTRACT=${CHOSEN_MOVE_SUBTRACT:-}
CHOSEN_MOVE_PRUNE=${CHOSEN_MOVE_PRUNE:-}
ROOT_NUM_SYMMETRIES_TO_SAMPLE=${ROOT_NUM_SYMMETRIES_TO_SAMPLE:-}
USE_LCB_FOR_SELECTION=${USE_LCB_FOR_SELECTION:-}
LCB_STDEVS=${LCB_STDEVS:-}
MIN_VISIT_PROP_FOR_LCB=${MIN_VISIT_PROP_FOR_LCB:-}
EXTRA_CONFIG_FILE=${EXTRA_CONFIG_FILE:-}
EXTRA_CONFIG_TEXT=${EXTRA_CONFIG_TEXT:-}
CUDA_GPU_TO_USE=${CUDA_GPU_TO_USE:-}

if [[ ! -x "$KATAGO_BIN" ]]; then
  echo "KATAGO_BIN 不可执行: $KATAGO_BIN" >&2
  exit 1
fi
if [[ ! -f "$CNN_MODEL" ]]; then
  echo "CNN_MODEL 不存在: $CNN_MODEL" >&2
  exit 1
fi
if [[ ! -f "$TF_MODEL" ]]; then
  echo "TF_MODEL 不存在: $TF_MODEL" >&2
  exit 1
fi
if [[ -n "$EXTRA_CONFIG_FILE" && ! -f "$EXTRA_CONFIG_FILE" ]]; then
  echo "EXTRA_CONFIG_FILE 不存在: $EXTRA_CONFIG_FILE" >&2
  exit 1
fi
case "$TF_PRECISION" in
  fp32|fp16|bf16) ;;
  *)
    echo "TF_PRECISION 必须是 fp32 / fp16 / bf16，当前为: $TF_PRECISION" >&2
    exit 1
    ;;
esac
case "$CNN_USE_FP16" in
  auto|true|false) ;;
  *)
    echo "CNN_USE_FP16 必须是 auto / true / false，当前为: $CNN_USE_FP16" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"
SGF_DIR="$OUTPUT_DIR/sgfs"
mkdir -p "$SGF_DIR"

CONFIG_FILE="$OUTPUT_DIR/match_cnn_vs_transformer.cfg"
LOG_FILE="$OUTPUT_DIR/match.log"
SUMMARY_FILE="$OUTPUT_DIR/match_summary.txt"

cat > "$CONFIG_FILE" <<EOF
logTimeStamp = true
logSearchInfo = false
logMoves = false
logGamesEvery = 20
logToStdout = true

numBots = 2
botName0 = $CNN_NAME
botName1 = $TF_NAME

nnModelFile0 = $CNN_MODEL
nnModelFile1 = $TF_MODEL
nnModelType0 = cnn
nnModelType1 = tf

useFP16-0 = $CNN_USE_FP16
nnPrecision-1 = $TF_PRECISION

numGameThreads = $NUM_GAME_THREADS
numGamesTotal = $NUM_GAMES
maxMovesPerGame = $MAX_MOVES_PER_GAME

allowResignation = $ALLOW_RESIGNATION
resignThreshold = $RESIGN_THRESHOLD
resignConsecTurns = $RESIGN_CONSEC_TURNS

koRules = $RULES_KO
scoringRules = $RULES_SCORING
taxRules = $RULES_TAX
multiStoneSuicideLegals = $RULES_MULTI_STONE_SUICIDE
hasButtons = $RULES_HAS_BUTTON

bSizes = $B_SIZES
bSizeRelProbs = $B_SIZE_REL_PROBS

komiAuto = $KOMI_AUTO
komiMean = $KOMI_MEAN
handicapProb = $HANDICAP_PROB
handicapCompensateKomiProb = 1.0

maxVisits = $MAX_VISITS
numSearchThreads = $NUM_SEARCH_THREADS

nnMaxBatchSize = $NN_MAX_BATCH_SIZE
nnCacheSizePowerOfTwo = $NN_CACHE_SIZE_POWER_OF_TWO
nnMutexPoolSizePowerOfTwo = $NN_MUTEX_POOL_SIZE_POWER_OF_TWO
nnRandomize = true
numNNServerThreadsPerModel = $NUM_NN_SERVER_THREADS_PER_MODEL

chosenMoveTemperatureEarly = $CHOSEN_MOVE_TEMPERATURE_EARLY
chosenMoveTemperature = $CHOSEN_MOVE_TEMPERATURE
EOF

if [[ -n "$CHOSEN_MOVE_TEMPERATURE_HALFLIFE" ]]; then
  echo "chosenMoveTemperatureHalflife = $CHOSEN_MOVE_TEMPERATURE_HALFLIFE" >> "$CONFIG_FILE"
fi
if [[ -n "$CHOSEN_MOVE_SUBTRACT" ]]; then
  echo "chosenMoveSubtract = $CHOSEN_MOVE_SUBTRACT" >> "$CONFIG_FILE"
fi
if [[ -n "$CHOSEN_MOVE_PRUNE" ]]; then
  echo "chosenMovePrune = $CHOSEN_MOVE_PRUNE" >> "$CONFIG_FILE"
fi
if [[ -n "$ROOT_NUM_SYMMETRIES_TO_SAMPLE" ]]; then
  echo "rootNumSymmetriesToSample = $ROOT_NUM_SYMMETRIES_TO_SAMPLE" >> "$CONFIG_FILE"
fi
if [[ -n "$USE_LCB_FOR_SELECTION" ]]; then
  echo "useLcbForSelection = $USE_LCB_FOR_SELECTION" >> "$CONFIG_FILE"
fi
if [[ -n "$LCB_STDEVS" ]]; then
  echo "lcbStdevs = $LCB_STDEVS" >> "$CONFIG_FILE"
fi
if [[ -n "$MIN_VISIT_PROP_FOR_LCB" ]]; then
  echo "minVisitPropForLCB = $MIN_VISIT_PROP_FOR_LCB" >> "$CONFIG_FILE"
fi

if [[ -n "$CUDA_GPU_TO_USE" ]]; then
  {
    echo
    echo "cudaGpuToUse = $CUDA_GPU_TO_USE"
  } >> "$CONFIG_FILE"
fi

if [[ -n "$EXTRA_CONFIG_FILE" ]]; then
  {
    echo
    cat "$EXTRA_CONFIG_FILE"
  } >> "$CONFIG_FILE"
fi

if [[ -n "$EXTRA_CONFIG_TEXT" ]]; then
  {
    echo
    printf '%s\n' "$EXTRA_CONFIG_TEXT"
  } >> "$CONFIG_FILE"
fi

echo "写入配置: $CONFIG_FILE"
echo "日志文件: $LOG_FILE"
echo "SGF 目录: $SGF_DIR"
echo "汇总文件: $SUMMARY_FILE"

set +e
"$KATAGO_BIN" match \
  -config "$CONFIG_FILE" \
  -log-file "$LOG_FILE" \
  -sgf-output-dir "$SGF_DIR"
MATCH_EXIT_CODE=$?
set -e

if compgen -G "$SGF_DIR/*.sgfs" > /dev/null || compgen -G "$SGF_DIR/*.sgf" > /dev/null; then
  python3 - "$SGF_DIR" "$CNN_NAME" "$TF_NAME" "$SUMMARY_FILE" <<'PY'
import math
import re
import sys
from pathlib import Path

sgf_dir = Path(sys.argv[1])
cnn_name = sys.argv[2]
tf_name = sys.argv[3]
summary_file = Path(sys.argv[4])


def split_sgf_records(text: str):
    records = []
    start = None
    depth = 0
    in_value = False
    escaped = False
    for i, ch in enumerate(text):
        if in_value:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "]":
                in_value = False
            continue
        if ch == "[":
            in_value = True
        elif ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    records.append(text[start:i + 1])
                    start = None
    return records


def extract_prop(record: str, key: str):
    match = re.search(rf"{key}\[((?:\\\\.|[^\]])*)\]", record)
    if match is None:
        return None
    value = match.group(1)
    value = value.replace("\\]", "]").replace("\\\\", "\\")
    return value


def classify_result(result: str):
    if result is None:
        return "no_result"
    raw = result.strip()
    lowered = raw.lower()
    if raw.startswith("B+"):
        return "black"
    if raw.startswith("W+"):
        return "white"
    if lowered in {"draw", "jigo", "tie", "0"} or lowered.startswith("draw"):
        return "draw"
    if lowered in {"void", "noresult", "no result", "unknown", ""}:
        return "no_result"
    if lowered.startswith("void") or lowered.startswith("no result") or lowered.startswith("noresult"):
        return "no_result"
    return "no_result"


def new_stats():
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "no_result": 0,
        "black_games": 0,
        "black_wins": 0,
        "black_losses": 0,
        "black_draws": 0,
        "black_no_result": 0,
        "white_games": 0,
        "white_wins": 0,
        "white_losses": 0,
        "white_draws": 0,
        "white_no_result": 0,
    }


def add_color_result(stats, color: str, outcome: str):
    prefix = "black" if color == "black" else "white"
    stats[f"{prefix}_games"] += 1
    if outcome == "win":
        stats[f"{prefix}_wins"] += 1
    elif outcome == "loss":
        stats[f"{prefix}_losses"] += 1
    elif outcome == "draw":
        stats[f"{prefix}_draws"] += 1
    else:
        stats[f"{prefix}_no_result"] += 1


def rate_to_elo(rate: float):
    eps = 1e-9
    rate = min(max(rate, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / rate - 1.0)


stats = {
    cnn_name: new_stats(),
    tf_name: new_stats(),
}
total_records = 0
parsed_games = 0
skipped_games = 0
draw_games = 0
no_result_games = 0
tf_scores = []

paths = sorted(list(sgf_dir.glob("*.sgfs")) + list(sgf_dir.glob("*.sgf")))
for path in paths:
    text = path.read_text(encoding="utf-8", errors="replace")
    for record in split_sgf_records(text):
        total_records += 1
        black_player = extract_prop(record, "PB")
        white_player = extract_prop(record, "PW")
        result = extract_prop(record, "RE")
        if black_player not in stats or white_player not in stats:
            skipped_games += 1
            continue

        parsed_games += 1
        stats[black_player]["games"] += 1
        stats[white_player]["games"] += 1

        outcome = classify_result(result)
        if outcome == "black":
            winner = black_player
            loser = white_player
            stats[winner]["wins"] += 1
            stats[loser]["losses"] += 1
            add_color_result(stats[winner], "black", "win")
            add_color_result(stats[loser], "white", "loss")
            tf_scores.append(1.0 if winner == tf_name else 0.0)
        elif outcome == "white":
            winner = white_player
            loser = black_player
            stats[winner]["wins"] += 1
            stats[loser]["losses"] += 1
            add_color_result(stats[winner], "white", "win")
            add_color_result(stats[loser], "black", "loss")
            tf_scores.append(1.0 if winner == tf_name else 0.0)
        elif outcome == "draw":
            draw_games += 1
            stats[black_player]["draws"] += 1
            stats[white_player]["draws"] += 1
            add_color_result(stats[black_player], "black", "draw")
            add_color_result(stats[white_player], "white", "draw")
            tf_scores.append(0.5)
        else:
            no_result_games += 1
            stats[black_player]["no_result"] += 1
            stats[white_player]["no_result"] += 1
            add_color_result(stats[black_player], "black", "no_result")
            add_color_result(stats[white_player], "white", "no_result")

lines = []
lines.append("=== 对战统计 ===")
lines.append(f"总 SGF 记录数: {total_records}")
lines.append(f"成功解析对局: {parsed_games}")
if skipped_games > 0:
    lines.append(f"跳过对局: {skipped_games}")
lines.append("")

for name in (cnn_name, tf_name):
    s = stats[name]
    lines.append(f"{name}:")
    lines.append(
        f"  总计 {s['games']} 局, 胜 {s['wins']} 负 {s['losses']} 和 {s['draws']} 无结果 {s['no_result']}"
    )
    lines.append(
        f"  执黑 {s['black_games']} 局, 胜 {s['black_wins']} 负 {s['black_losses']} 和 {s['black_draws']} 无结果 {s['black_no_result']}"
    )
    lines.append(
        f"  执白 {s['white_games']} 局, 胜 {s['white_wins']} 负 {s['white_losses']} 和 {s['white_draws']} 无结果 {s['white_no_result']}"
    )
    lines.append("")

lines.append(f"和棋: {draw_games}")
lines.append(f"无结果: {no_result_games}")

if tf_scores:
    n = len(tf_scores)
    tf_score = sum(tf_scores)
    tf_rate = tf_score / n
    tf_elo = rate_to_elo(tf_rate)
    lines.append(f"Transformer 相对 CNN 记分胜率: {tf_score:.1f}/{n} = {tf_rate:.4f}")
    lines.append(f"Transformer 相对 CNN Elo 估计: {tf_elo:.1f}")
    if n > 1:
        mean = tf_rate
        variance = sum((x - mean) ** 2 for x in tf_scores) / (n - 1)
        stderr = math.sqrt(variance / n)
        ci_low = max(0.0, mean - 1.96 * stderr)
        ci_high = min(1.0, mean + 1.96 * stderr)
        elo_low = rate_to_elo(ci_low)
        elo_high = rate_to_elo(ci_high)
        lines.append(f"Transformer 相对 CNN 记分胜率 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        lines.append(f"Transformer 相对 CNN Elo 95% CI: [{elo_low:.1f}, {elo_high:.1f}]")
else:
    lines.append("Transformer 相对 CNN Elo 估计: N/A（没有可用于 Elo 的有效对局）")

summary_text = "\n".join(lines) + "\n"
summary_file.write_text(summary_text, encoding="utf-8")
sys.stdout.write(summary_text)
PY
else
  echo "未找到 SGF 输出，跳过汇总统计" >&2
fi

if [[ $MATCH_EXIT_CODE -ne 0 ]]; then
  echo "katago match 退出码非 0: $MATCH_EXIT_CODE" >&2
  exit "$MATCH_EXIT_CODE"
fi
