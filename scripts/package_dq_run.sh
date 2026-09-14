#!/usr/bin/env bash
# package_dq_run.sh — P-DECISION-QUALITY 运行产物的自动打包(回传用)。
#
# 用法:
#   bash scripts/package_dq_run.sh [RUN_DIR]
#     RUN_DIR 缺省 = cloud_run_decision_quality.sh 的产物目录
#                (cloud_dq_<ts>/decision_quality_out)
#
# 打包内容(全部进 zip,路径保持仓库相对结构,便于本地直接解包覆盖):
#   results/openillumination/decision_quality.json   主产物(含 manifest)
#   docs/img/decision_quality.png                    图
#   configs/decision_quality.yaml                    运行时 config(身份链)
#   RUN_INFO.txt                                     运行指纹(git_sha、config sha256、
#                                                    时间、宿主、python/numpy 版本)
#   checkpoints/(若存在且保留)                       逐物体 checkpoint(审计/对拍用)
# 输出: decision_quality_v1_1_<gitsha8>_<ts>.zip(与 RUN_DIR 同级)
set -euo pipefail

RUN_DIR="${1:-$(ls -dt cloud_dq_*/decision_quality_out 2>/dev/null | head -1)}"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "usage: bash scripts/package_dq_run.sh <RUN_DIR>  (or run inside the cloud_dq_* parent)" >&2
    exit 1
fi
RUN_DIR="$(readlink -f "$RUN_DIR")"
REPO="$(dirname "$(dirname "$(readlink -f "$0")")")"   # scripts/ 的上级

# --- 从产物 manifest 提取身份 ---
ART="$RUN_DIR/decision_quality.json"
if [ ! -f "$ART" ]; then
    # cloud 脚本的 collect 段产物与仓库路径并存
    ART="$RUN_DIR/results/openillumination/decision_quality.json"
fi
GITSHA8=$(python3 - "$ART" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))["manifest"]
print(m["git_sha"][:8])
PY
)
TS=$(date +%Y%m%d_%H%M%S)
STAGE="$RUN_DIR/package_staging"
rm -rf "$STAGE"
mkdir -p "$STAGE/results/openillumination" "$STAGE/docs/img" "$STAGE/configs"

# --- 主产物 + 图 + config(保仓库相对路径) ---
find "$RUN_DIR" -name decision_quality.json -exec cp {} "$STAGE/results/openillumination/" \;
find "$RUN_DIR" -name decision_quality.png   -exec cp {} "$STAGE/docs/img/" \;
CFG_HITS=$(find "$REPO/configs" "$RUN_DIR" -maxdepth 3 -name "decision_quality.yaml" 2>/dev/null | head -2)
for c in $CFG_HITS; do cp "$c" "$STAGE/configs/decision_quality.yaml" && break; done

# --- checkpoint(若调用方用 --ckpt-dir 保留过) ---
CKPTS=$(find "$RUN_DIR" "$REPO/results/openillumination" -maxdepth 2 -type d -name ".dq_checkpoints" 2>/dev/null | head -1)
if [ -n "$CKPTS" ] && [ -d "$CKPTS" ]; then
    mkdir -p "$STAGE/checkpoints"
    cp "$CKPTS"/*.json "$STAGE/checkpoints/" 2>/dev/null || true
fi

# --- 运行指纹 ---
python3 - "$ART" > "$STAGE/RUN_INFO.txt" <<'PY'
import json, sys, platform, datetime
j = json.load(open(sys.argv[1], encoding="utf-8"))
m = j["manifest"]
print("P-DECISION-QUALITY run package")
print("analysis_status :", j["analysis_status"])
print("rows            :", len(j["rows"]))
print("git_sha         :", m["git_sha"])
print("git_sha_at_launch:", m.get("git_sha_at_launch", "(field absent in this run)"))
print("config_sha256   :", m["config_sha256"])
print("elapsed_s       :", m.get("elapsed_s"))
print("packaged_utc    :", datetime.datetime.utcnow().isoformat())
print("packaged_on     :", platform.platform())
try:
    import numpy
    print("numpy           :", numpy.__version__)
except Exception:
    pass
PY

# --- zip ---
OUT="$(dirname "$RUN_DIR")/decision_quality_${GITSHA8}_${TS}.zip"
(cd "$STAGE" && zip -q -r "$OUT" .)
rm -rf "$STAGE"
echo "PACKAGED: $OUT"
unzip -l "$OUT"
