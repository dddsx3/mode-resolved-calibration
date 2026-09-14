#!/usr/bin/env bash
# cloud_run_decision_quality.sh — P-DECISION-QUALITY v1.1 cloud runner
# (32-core/64GB assumed; any Linux x86_64 with Python 3.10+ works).
#
# 用法(在云实例上):
#   DATA_SOURCE=user@host:/path/to/OpenIllumination bash cloud_run_decision_quality.sh
#   # 或数据已在本机:
#   DATA_DIR=/data/OpenIllumination bash scripts/cloud_run_decision_quality.sh
#
# 环境变量:
#   DATA_SOURCE  (必填二选一) rsync 来源(含 11 个 obj_*/;195MB)
#   DATA_DIR     (必填二选一) 云实例上已就位的数据根
#   DATA_META    可选,meta 目录(缺省 $DATA_DIR/../OpenIllumination_meta)
#   REPO_REF     可选,默认 5196fe7(--workers + checkpoint 支持提交)
#   WORKERS      可选,默认 11(每对象一个 worker;64GB 内存余量巨大)
#
# 优化(严格不改数值结构;详见提交说明):
#   - workers=11(11 个独立物体;每物体内部 ~88% 串行,更多核无用)
#   - OPENBLAS_NUM_THREADS=4:eigh 在 1→4 线程 1.49×,8 线程倒退(实测);
#     11 workers × 4 BLAS 线程 = 44 逻辑核占用,32 物理核恰好饱和
#   - 逐物体 checkpoint(原子写,resume 不重算):崩溃只丢在算的那一个物体
#   - 运行结束自动打包 .zip(scripts/package_dq_run.sh)
#
# 监控:进度打印逐对象 flush(每 ~2h 一行);或另开终端看
#   watch -n 30 "wc -l < <(ls results/openillumination/.dq_checkpoints/ 2>/dev/null | wc -l)"
#
# 产物:results/.../decision_quality.json + docs/img/...png + RUN_DIR/*.zip
# 数值不变性:逐对象独立播种,workers/BLAS 线程只影响墙钟。
set -euo pipefail

REPO_URL="https://github.com/dddsx3/mode-resolved-calibration.git"
REPO_REF="${REPO_REF:-5196fe7}"
WORKERS="${WORKERS:-11}"
ROOT="$(pwd)/cloud_dq_$(date +%s)"
mkdir -p "$ROOT"
cd "$ROOT"

echo "=== [1/6] clone @ $REPO_REF"
git clone --quiet "$REPO_URL" mrc
cd mrc
git checkout --quiet "$REPO_REF"

echo "=== [2/6] environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev,examples]"

echo "=== [3/6] data (195 MB + 314 KB meta)"
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

echo "=== [4/6] run (workers=$WORKERS, BLAS=4 threads; ~1.5-2h on 32 cores)"
# BLAS 线程上限:worker 数 × BLAS 线程 > 逻辑核也无妨(OpenBLAS 自排队),
# 但 4 是实测甜点(1.49×;8 倒退)。export 对 spawn 的 worker 子进程生效。
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
# nohup + 心跳:防止 ssh 断开杀掉整个作业(上次本地事故的教训)。
nohup python -u experiments/decision_quality.py --workers "$WORKERS" \
    > "$ROOT/run.log" 2>&1 &
RUN_PID=$!
echo "run PID=$RUN_PID; log=$ROOT/run.log"
echo "监控: tail -f $ROOT/run.log   (每完成一个物体打一行)"
wait "$RUN_PID"
tail -3 "$ROOT/run.log"

echo "=== [5/6] verify artifact"
python - << 'PY'
import json
j = json.load(open("results/openillumination/decision_quality.json", encoding="utf-8"))
print("analysis_status:", j["analysis_status"])
print("rows:", len(j["rows"]), "| auc_rows:", len(j["auc_rows"]),
      "| git_sha:", j["manifest"]["git_sha"][:12])
print("informed pairwise pooled:", j["informed_pairwise_sign_pooled"])
PY

echo "=== [6/6] package (auto-zip)"
OUT_DIR="$ROOT/decision_quality_out"
mkdir -p "$OUT_DIR"
cp results/openillumination/decision_quality.json "$OUT_DIR/"
cp docs/img/decision_quality.png "$OUT_DIR/"
bash scripts/package_dq_run.sh "$OUT_DIR"
echo
echo "DONE. Artifact bundle + zip under: $ROOT/"
echo "回传本地后:解包 zip 到仓库根覆盖,然后本地做 commit B(台账/pin/登记)。"
