#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  validate_transformer_precisions.sh KATAGO_BIN MODEL_KGTR CHECKPOINT_CKPT OUTPUT_DIR COMPARE_SCRIPT [SAMPLE_NAME] [SYMMETRY]

参数:
  KATAGO_BIN         katago 可执行文件路径
  MODEL_KGTR         导出的 Transformer 模型 .kgtr.gz
  CHECKPOINT_CKPT    Torch checkpoint .ckpt
  OUTPUT_DIR         输出目录
  COMPARE_SCRIPT     compare_backend_dump.py 路径
  SAMPLE_NAME        可选，默认 canary_opening_18
  SYMMETRY           可选，默认 0

环境变量:
  PRECISIONS         要验证的精度列表，默认: "fp32 fp16 bf16"
  USE_EMA            设为 1 时给 compare_backend_dump.py 追加 --use-ema
  SCORE_MODE         compare_backend_dump.py 的 --score-mode，默认 mixop

示例:
  PRECISIONS="fp32 fp16 bf16" \
  ./misc/validate_transformer_precisions.sh \
    /tmp/katago-cuda-build/katago \
    /tmp/checkpoint.kgtr.gz \
    /data/checkpoint.ckpt \
    /tmp/transformer-precision-check \
    /path/to/compare_backend_dump.py
EOF
}

if [[ $# -lt 5 || $# -gt 7 ]]; then
  usage
  exit 1
fi

KATAGO_BIN=$1
MODEL_KGTR=$2
CHECKPOINT_CKPT=$3
OUTPUT_DIR=$4
COMPARE_SCRIPT=$5
SAMPLE_NAME=${6:-canary_opening_18}
SYMMETRY=${7:-0}

PRECISIONS_STRING=${PRECISIONS:-"fp32 fp16 bf16"}
SCORE_MODE_VALUE=${SCORE_MODE:-mixop}

if [[ ! -x "$KATAGO_BIN" ]]; then
  echo "KATAGO_BIN 不可执行: $KATAGO_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_KGTR" ]]; then
  echo "MODEL_KGTR 不存在: $MODEL_KGTR" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_CKPT" ]]; then
  echo "CHECKPOINT_CKPT 不存在: $CHECKPOINT_CKPT" >&2
  exit 1
fi
if [[ ! -f "$COMPARE_SCRIPT" ]]; then
  echo "COMPARE_SCRIPT 不存在: $COMPARE_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

IFS=' ' read -r -a PRECISIONS_ARRAY <<< "$PRECISIONS_STRING"

TORCH_EXTRA_ARGS=(--score-mode "$SCORE_MODE_VALUE")
if [[ "${USE_EMA:-0}" == "1" ]]; then
  TORCH_EXTRA_ARGS+=(--use-ema)
fi

compare_backend_dumps() {
  local baseline_dump=$1
  local candidate_dump=$2
  local output_json=$3
  python3 - "$baseline_dump" "$candidate_dump" "$output_json" <<'PY'
import json
import sys

baseline_path, candidate_path, output_path = sys.argv[1:4]

with open(baseline_path, "r", encoding="utf-8") as f:
    baseline = json.load(f)
with open(candidate_path, "r", encoding="utf-8") as f:
    candidate = json.load(f)


def new_diff_stats():
    return {
        "count": 0,
        "sumAbs": 0.0,
        "sum": 0.0,
        "maxAbs": 0.0,
    }


def update_diff_stats(stats, diff):
    diff = float(diff)
    abs_diff = abs(diff)
    stats["count"] += 1
    stats["sumAbs"] += abs_diff
    stats["sum"] += diff
    stats["maxAbs"] = max(stats["maxAbs"], abs_diff)


def compare_scalar(lhs, rhs, stats):
    diff = float(lhs) - float(rhs)
    update_diff_stats(stats, diff)
    return abs(diff)


def compare_list(lhs, rhs, stats):
    if len(lhs) != len(rhs):
        raise ValueError(f"length mismatch: {len(lhs)} vs {len(rhs)}")
    max_abs = 0.0
    for a, b in zip(lhs, rhs):
        diff = float(a) - float(b)
        update_diff_stats(stats, diff)
        max_abs = max(max_abs, abs(diff))
    return max_abs


def finalize_diff_stats(stats):
    count = int(stats["count"])
    if count <= 0:
        return {
            "numComparedValues": 0,
            "maxAbsErr": 0.0,
            "meanAbsErr": 0.0,
            "meanDiff": 0.0,
        }
    return {
        "numComparedValues": count,
        "maxAbsErr": float(stats["maxAbs"]),
        "meanAbsErr": float(stats["sumAbs"] / count),
        "meanDiff": float(stats["sum"] / count),
    }


sample_reports = []
global_stats = new_diff_stats()

if len(baseline["samples"]) != len(candidate["samples"]):
    raise ValueError("sample count mismatch")

for baseline_sample, candidate_sample in zip(baseline["samples"], candidate["samples"]):
    if baseline_sample["name"] != candidate_sample["name"]:
        raise ValueError(
            f"sample mismatch: {baseline_sample['name']} vs {candidate_sample['name']}"
        )

    report = {
        "name": baseline_sample["name"],
        "raw": {
            "policy": compare_list(baseline_sample["raw"]["policy"], candidate_sample["raw"]["policy"], global_stats),
            "policyPass": compare_list(baseline_sample["raw"]["policyPass"], candidate_sample["raw"]["policyPass"], global_stats),
            "value": compare_list(baseline_sample["raw"]["value"], candidate_sample["raw"]["value"], global_stats),
            "scoreValue": compare_list(baseline_sample["raw"]["scoreValue"], candidate_sample["raw"]["scoreValue"], global_stats),
            "ownership": compare_list(baseline_sample["raw"]["ownership"], candidate_sample["raw"]["ownership"], global_stats),
        },
        "nnOutput": {
            "policyProbs": compare_list(baseline_sample["nnOutput"]["policyProbs"], candidate_sample["nnOutput"]["policyProbs"], global_stats),
            "whiteOwnerMap": compare_list(baseline_sample["nnOutput"]["whiteOwnerMap"], candidate_sample["nnOutput"]["whiteOwnerMap"], global_stats),
            "whiteWinProb": compare_scalar(baseline_sample["nnOutput"]["whiteWinProb"], candidate_sample["nnOutput"]["whiteWinProb"], global_stats),
            "whiteLossProb": compare_scalar(baseline_sample["nnOutput"]["whiteLossProb"], candidate_sample["nnOutput"]["whiteLossProb"], global_stats),
            "whiteNoResultProb": compare_scalar(baseline_sample["nnOutput"]["whiteNoResultProb"], candidate_sample["nnOutput"]["whiteNoResultProb"], global_stats),
            "whiteScoreMean": compare_scalar(baseline_sample["nnOutput"]["whiteScoreMean"], candidate_sample["nnOutput"]["whiteScoreMean"], global_stats),
            "whiteScoreMeanSq": compare_scalar(baseline_sample["nnOutput"]["whiteScoreMeanSq"], candidate_sample["nnOutput"]["whiteScoreMeanSq"], global_stats),
            "whiteLead": compare_scalar(baseline_sample["nnOutput"]["whiteLead"], candidate_sample["nnOutput"]["whiteLead"], global_stats),
            "varTimeLeft": compare_scalar(baseline_sample["nnOutput"]["varTimeLeft"], candidate_sample["nnOutput"]["varTimeLeft"], global_stats),
            "shorttermWinlossError": compare_scalar(baseline_sample["nnOutput"]["shorttermWinlossError"], candidate_sample["nnOutput"]["shorttermWinlossError"], global_stats),
            "shorttermScoreError": compare_scalar(baseline_sample["nnOutput"]["shorttermScoreError"], candidate_sample["nnOutput"]["shorttermScoreError"], global_stats),
        },
    }

    if "rawFull" in baseline_sample and "rawFull" in candidate_sample:
        report["rawFull"] = {
            "policy": compare_list(baseline_sample["rawFull"]["policy"], candidate_sample["rawFull"]["policy"], global_stats),
            "policyPass": compare_list(baseline_sample["rawFull"]["policyPass"], candidate_sample["rawFull"]["policyPass"], global_stats),
            "value": compare_list(baseline_sample["rawFull"]["value"], candidate_sample["rawFull"]["value"], global_stats),
            "misc": compare_list(baseline_sample["rawFull"]["misc"], candidate_sample["rawFull"]["misc"], global_stats),
            "moreMisc": compare_list(baseline_sample["rawFull"]["moreMisc"], candidate_sample["rawFull"]["moreMisc"], global_stats),
            "ownership": compare_list(baseline_sample["rawFull"]["ownership"], candidate_sample["rawFull"]["ownership"], global_stats),
            "scoring": compare_list(baseline_sample["rawFull"]["scoring"], candidate_sample["rawFull"]["scoring"], global_stats),
            "futurePos": compare_list(baseline_sample["rawFull"]["futurePos"], candidate_sample["rawFull"]["futurePos"], global_stats),
            "seki": compare_list(baseline_sample["rawFull"]["seki"], candidate_sample["rawFull"]["seki"], global_stats),
            "scoreBelief": compare_list(baseline_sample["rawFull"]["scoreBelief"], candidate_sample["rawFull"]["scoreBelief"], global_stats),
        }

    sample_reports.append(report)

aggregate = finalize_diff_stats(global_stats)

result = {
    "baselineDump": baseline_path,
    "candidateDump": candidate_path,
    "maxAbsErr": aggregate["maxAbsErr"],
    "meanAbsErr": aggregate["meanAbsErr"],
    "meanDiff": aggregate["meanDiff"],
    "numComparedValues": aggregate["numComparedValues"],
    "samples": sample_reports,
}

text = json.dumps(result, indent=2)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)
print(text)
PY
}

extract_summary_metrics() {
  local report_json=$1
  python3 - "$report_json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
required_keys = ["maxAbsErr", "meanAbsErr", "meanDiff", "numComparedValues"]
missing = [key for key in required_keys if key not in data]
if missing:
    raise SystemExit(
        f'{sys.argv[1]} 缺少字段: {", ".join(missing)}; 需要同步最新的 compare_backend_dump.py'
    )
print(f'{data["maxAbsErr"]}\t{data["meanAbsErr"]}\t{data["meanDiff"]}\t{data["numComparedValues"]}')
PY
}

SUMMARY_FILE="$OUTPUT_DIR/summary.tsv"
printf "precision\tcompare_target\tattention_mode\tmax_abs_err\tmean_abs_err\tmean_diff\tnum_compared_values\treport_json\n" > "$SUMMARY_FILE"

FP32_DUMP_JSON=""

for precision in "${PRECISIONS_ARRAY[@]}"; do
  dump_json="$OUTPUT_DIR/transformer_dump_${precision}.json"
  echo "==> 生成 ${precision} dump"
  "$KATAGO_BIN" runtransformerdump "$MODEL_KGTR" "$dump_json" "$SYMMETRY" "$SAMPLE_NAME" "$precision"

  if [[ "$precision" == "fp32" ]]; then
    FP32_DUMP_JSON="$dump_json"
  fi

  attention_mode="default"
  report_json="$OUTPUT_DIR/compare_torch_${precision}_${attention_mode}.json"
  echo "==> 对照 Torch: precision=${precision}, attention_mode=${attention_mode}"
  python3 "$COMPARE_SCRIPT" \
    --checkpoint "$CHECKPOINT_CKPT" \
    --dump-json "$dump_json" \
    --output-json "$report_json" \
    --attention-mode "$attention_mode" \
    "${TORCH_EXTRA_ARGS[@]}"
  metrics=$(extract_summary_metrics "$report_json")
  printf "%s\ttorch\t%s\t%s\t%s\n" "$precision" "$attention_mode" "$metrics" "$report_json" >> "$SUMMARY_FILE"
done

if [[ -n "$FP32_DUMP_JSON" ]]; then
  for precision in "${PRECISIONS_ARRAY[@]}"; do
    if [[ "$precision" == "fp32" ]]; then
      continue
    fi
    candidate_dump="$OUTPUT_DIR/transformer_dump_${precision}.json"
    if [[ ! -f "$candidate_dump" ]]; then
      continue
    fi
    report_json="$OUTPUT_DIR/compare_backend_fp32_vs_${precision}.json"
    echo "==> 对照后端 fp32: candidate=${precision}"
    compare_backend_dumps "$FP32_DUMP_JSON" "$candidate_dump" "$report_json"
    metrics=$(extract_summary_metrics "$report_json")
    printf "%s\tbackend_fp32\t-\t%s\t%s\n" "$precision" "$metrics" "$report_json" >> "$SUMMARY_FILE"
  done
fi

echo
echo "验证完成。汇总文件:"
echo "  $SUMMARY_FILE"
echo
cat "$SUMMARY_FILE"
