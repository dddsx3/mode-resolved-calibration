#!/usr/bin/env bash
# cloud_run_decision_quality.sh — P-DECISION-QUALITY v1.1 cloud runner
# (32 physical cores / 64 GB RAM assumed; any Linux x86_64 with Python 3.10+ works).
#
# 用法(在云实例上):
#   DATA_SOURCE=user@host:/path/to/OpenIllumination bash cloud_run_decision_quality.sh
#   # 或数据已在本机:
#   DATA_DIR=/data/OpenIllumination bash scripts/cloud_run_decision_quality.sh
#
# 环境变量:
#   DATA_SOURCE  (必填二选一) rsync 来源(本地机的数据根,含 11 个 obj_*/)
#   DATA_DIR     (必填二选一) 云实例上已就位的数据根
#   DATA_META    可选,meta 目录(缺省 $DATA_DIR/../OpenIllumination_meta 或 rsync 自带)
#   REPO_REF     可选,默认 5196fe7(--workers 支持提交;产物的 git_sha 将记录它)
#   WORKERS      可选,默认 11(每对象一个 worker;64GB 内存余量巨大)
#
# 产物:results/openillumination/decision_quality.json + docs/img/decision_quality.png
#       回传后在本地仓库做 commit B(台账/pin 测试/登记在本地完成,保持纪律)。
# 数值不变性:逐对象独立播种,workers 只影响墙钟(见 --workers 的帮助文本)。
set -euo pipefail

REPO_URL="https://github.com/dddsx3/mode-resolved-calibration.git"
REPO_REF="${REPO_REF:-5196fe7}"
WORKERS="${WORKERS:-11}"
ROOT="$(pwd)/cloud_dq_$(date +%s)"
mkdir -p "$ROOT"
cd "$ROOT"

echo "=== [1/5] clone @ $REPO_REF"
git clone --quiet "$REPO_URL" mrc
cd mrc
git checkout --quiet "$REPO_REF"

echo "=== [2/5] environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev,examples]"   # numpy/scipy/pyyaml/pytest + matplotlib(图)

echo "=== [3/5] data (195 MB + 314 KB meta)"
# config 里 preregistered 的 data_root 是 Windows 字面路径 "D:/data/OpenIllumination"。
# 在 Linux 上这是相对路径:从仓库根建同名符号链接,config 字节保持不变
# (config_sha256 / git_sha 与本地跑完全一致,产物逐位可比)。
mkdir -p "D:/data"
if [ -n "${DATA_SOURCE:-}" ]; then
    rsync -a "$DATA_SOURCE" "D:/data/OpenIllumination"
    rsync -a "${DATA_SOURCE}_meta" "D:/data/OpenIllumination_meta" || true
elif [ -n "${DATA_DIR:-}" ]; then
    ln -s "$(readlink -f "$DATA_DIR")" "D:/data/OpenIllumination"
    META="${DATA_META:-$(dirname "$DATA_DIR")/OpenIllumination_meta}"
    ln -s "$(readlink -f "$META")" "D:/data/OpenIllumination_meta"
else
    echo "set DATA_SOURCE (rsync source) or DATA_DIR (local path)" >&2; exit 1
fi
ls "D:/data/OpenIllumination" | head -3   # sanity:应看到 obj_*/

echo "=== [4/5] run (workers=$WORKERS; ~100 min on 32 cores)"
# 每对象一个 worker:11 物体 × 8 (level,regime) × (e/a/d ~230s + mode ~50s) ≈ 97 min/对象
# / 11 workers ≈ 1.7h;bootstrap B=10000 × 48 keys 在 aggregate 段再花几分钟。
time python experiments/decision_quality.py --workers "$WORKERS"

echo "=== [5/5] collect"
OUT="$ROOT/decision_quality_out"
mkdir -p "$OUT"
cp results/openillumination/decision_quality.json "$OUT/"
cp docs/img/decision_quality.png "$OUT/"
python - << 'PY'
import json
j = json.load(open("results/openillumination/decision_quality.json", encoding="utf-8"))
print("analysis_status:", j["analysis_status"])
print("rows:", len(j["rows"]), "| auc_rows:", len(j["auc_rows"]),
      "| git_sha:", j["manifest"]["git_sha"][:12])
print("informed pairwise pooled:", j["informed_pairwise_sign_pooled"])
PY
echo
echo "DONE. Artifact bundle: $OUT"
echo "回传本地后,在本地仓库: cp 回 results/ 与 docs/img/,更新台账/pin 测试,commit B。"
echo "(云/本地两次运行的逐位一致性本身是一次跨机器确定性验证——若都在,建议都跑完对比。)"
