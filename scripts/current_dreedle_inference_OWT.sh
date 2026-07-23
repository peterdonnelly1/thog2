#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT=""; PROMPT="The meaning of life is"; DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"; TOKENS=120; SAMPLES=1; TEMP=0.8; TOP_K=200; DTYPE=float16; BACKEND=sdpa; DEVICE=cuda; SEED=1337
usage(){ echo "Usage: $0 -f checkpoint [-p prompt] [-t data_dir] [-N tokens] [-s samples] [-T temperature] [-k top_k] [-d device] [-y dtype] [-K auto|flash2|sdpa|math]"; }
while getopts ":f:p:t:N:s:T:k:d:y:K:S:h" opt; do case "$opt" in f) CHECKPOINT="$OPTARG";; p) PROMPT="$OPTARG";; t) DATA_DIR="$OPTARG";; N) TOKENS="$OPTARG";; s) SAMPLES="$OPTARG";; T) TEMP="$OPTARG";; k) TOP_K="$OPTARG";; d) DEVICE="$OPTARG";; y) DTYPE="$OPTARG";; K) BACKEND="$OPTARG";; S) SEED="$OPTARG";; h) usage; exit 0;; *) usage; exit 2;; esac; done
[[ -n "$CHECKPOINT" ]] || { usage; exit 2; }
python -m run_thog2_owt_inference --checkpoint "$CHECKPOINT" --data-dir "$DATA_DIR" --prompt "$PROMPT" --max-new-tokens "$TOKENS" --num-samples "$SAMPLES" --temperature "$TEMP" --top-k "$TOP_K" --device "$DEVICE" --dtype "$DTYPE" --attention-backend "$BACKEND" --seed "$SEED"
