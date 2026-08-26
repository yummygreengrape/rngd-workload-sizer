#!/usr/bin/env bash
# 평가 서버의 모든 측정 산출물을 로컬 저장소로 회수한다.
# rsync 가 서버에 없어 tar 파이프를 쓴다.
#
# 회수 대상
#   data/raw/{run_id}/     요청별 JSONL · metrics 시계열 · meta · summary
#   data/server_logs/serve/    furiosa-llm serve 의 stdout/stderr (설정별)
#   data/server_logs/runs/     벤치마크 실행 stdout (조건별 진행 로그)
#   data/server_logs/configs/  실행에 쓰인 config (동적 생성분 포함 — 실험 정의 그 자체)
#   data/server_logs/system/   furiosa-smi 스냅샷, 커널·패키지 정보
set -euo pipefail
HOST="${1:-edu-hlu-1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$REPO/data/server_logs"/{serve,runs,configs,system}

echo "[1/5] 측정 결과 (data/raw)"
ssh -o BatchMode=yes "$HOST" 'cd /root/rngd-bench && tar czf - data/raw' | tar xzf - -C "$REPO"

echo "[2/5] 서버 기동 로그"
ssh -o BatchMode=yes "$HOST" 'cd /root/bench_logs && tar czf - . 2>/dev/null' \
  | tar xzf - -C "$REPO/data/server_logs/serve"

echo "[3/5] 벤치마크 실행 로그"
ssh -o BatchMode=yes "$HOST" 'cd /root/rngd-bench/logs && tar czf - . 2>/dev/null' \
  | tar xzf - -C "$REPO/data/server_logs/runs"

echo "[4/5] 실행에 쓰인 config (동적 생성분 포함)"
ssh -o BatchMode=yes "$HOST" 'cd /tmp && tar czf - $(ls c1_*.json c2_*.json 2>/dev/null) 2>/dev/null || true' \
  | tar xzf - -C "$REPO/data/server_logs/configs" 2>/dev/null || true
ssh -o BatchMode=yes "$HOST" 'cd /root/rngd-bench && tar czf - benchmark/configs' \
  | tar xzf - -C "$REPO/data/server_logs/configs" --strip-components=1

echo "[5/5] 시스템 스냅샷"
ssh -o BatchMode=yes "$HOST" '
  echo "### date"; date -u; echo
  echo "### furiosa-smi info"; furiosa-smi info; echo
  echo "### furiosa-smi status"; furiosa-smi status 2>&1 | head -40; echo
  echo "### furiosa-smi ps"; furiosa-smi ps; echo
  echo "### furiosa-smi topo"; furiosa-smi topo; echo
  echo "### furiosa-smi version"; furiosa-smi version; echo
  echo "### uname"; uname -a; echo
  echo "### pip (furiosa/torch/transformers)"; pip3 list 2>/dev/null | grep -iE "furiosa|torch|transformers|numpy"; echo
  echo "### free"; free -g; echo
  echo "### loadavg"; cat /proc/loadavg
' > "$REPO/data/server_logs/system/snapshot-$STAMP.txt" 2>&1

echo "완료"
find "$REPO/data" -type f | wc -l | xargs echo "  파일 수:"
du -sh "$REPO/data"
