"""raw run → planner 가 읽는 실측 테이블.

`data/raw/{run_id}/summary.json` 들을 모아 `data/processed/benchmark_rows.json` 을 만든다.
planner 는 이 파일만 읽고, 출처가 measured_local 이 아닌 행은 로드 단계에서 거부한다.

카드 수(n_cards)는 capacity 계산의 분모라 반드시 필요하다. meta.json 의
`target.n_cards` 를 쓰되, 그 필드가 생기기 전에 측정한 run 은 아래 표로 보정한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 서버가 물고 있던 동시 요청이 이 배수를 넘으면 같은 서버에 다른 클라이언트가 있었다는 뜻.
# 임계값은 실측 분포에서 뽑았다 — 비 1.05~1.20 구간이 0건이고 0.8~1.05 에 58건,
# 1.2 이상에 11건으로 깨끗하게 갈린다. 경계에 아무것도 없어 1.25 가 안전하다.
FOREIGN_LOAD_RATIO = 1.25

# **실험 처치군.** 의도적으로 교란을 넣어 그 효과를 재는 run 이라, 측정 자체는 유효하지만
# 용량 산정의 근거로 뽑히면 안 된다. A3dualA/B 는 `FOREIGN_LOAD` 로 자동으로 걸러지지만
# 폴러를 띄운 run 은 데이터에 흔적이 안 남는다 — 실제로 빠름 등급의 근거로 뽑혔었다.
#
# 앞으로 측정할 때는 `run_llm --not-for-capacity` 로 실행 시점에 표시한다.
# 아래는 그 플래그가 생기기 전에 측정한 run 을 위한 보정표다.
EXPERIMENT_ARMS: dict[str, str] = {
    "furiosa-ai_Qwen3-8B-FP8_A3poll2_20260826-055358":
        "폴링 비용 측정의 처치군 — furiosa-smi 폴러 2개를 의도적으로 띄웠다 (세션 2 A-3(c))",
}

# 무효 처리 규칙은 하네스와 **같은 정의**를 쓴다. 여기서 다시 적으면 두 곳이 갈린다.
# `python analysis/process.py` 로 직접 실행해도 import 되도록 경로를 넣는다.
sys.path.insert(0, REPO_ROOT)
from benchmark.schema import capacity_blocks  # noqa: E402

# --cards 플래그를 붙이기 전에 실행한 run 들의 카드 수. **최후 수단이다.**
# 이름 표에 의존하면 새 실험 코드로 측정한 데이터가 조용히 버려진다.
LEGACY_RUN_CARDS: dict[str, int] = {
    "C1x2": 2, "C1x4": 4, "C1x8": 8, "C1x8b": 8,
    "C2x1": 1, "C2x2": 2, "C2x4": 4,
    "B1": 1, "A3": 1, "A4": 1, "E1": 1, "S0": 1, "B4soak": 1,
}

_MESH = re.compile(r"Device mesh:\s*(\d+)\s*DP group")
_UVICORN = re.compile(r"Uvicorn running on https?://[^:]+:(\d+)")


def server_log_mesh(log_dir: str) -> list[tuple[int, int]]:
    """서버 기동 로그에서 (포트, DP 그룹 수) 를 뽑는다.

    `--cards` 를 안 붙이고 측정한 run 의 카드 수를 되찾는 경로다.
    로그 하나에 기동이 여러 번 기록될 수 있으므로 등장 순서대로 모은다.
    """
    out: list[tuple[int, int]] = []
    if not os.path.isdir(log_dir):
        return out
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".log"):
            continue
        try:
            with open(os.path.join(log_dir, name), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        dp = None
        for line in text.splitlines():
            m = _MESH.search(line)
            if m:
                dp = int(m.group(1))
                continue
            u = _UVICORN.search(line)
            if u and dp is not None:
                out.append((int(u.group(1)), dp))
                dp = None
    return out


def infer_cards(meta: dict[str, Any], mesh: list[tuple[int, int]] | None = None) -> tuple[int | None, str]:
    """카드 수와 그 출처를 돌려준다.

    capacity 계산의 분모이므로 추측하지 않는다. 순서:

    1. `meta.target.n_cards` — 하네스가 `--cards` 로 기록한 값 (가장 확실)
    2. 서버 기동 로그의 `Device mesh: N DP group(s)` — 포트로 대조
    3. 실험 코드 이름 표 — 최후 수단
    """
    n = (meta.get("target") or {}).get("n_cards")
    if n:
        return int(n), "meta"

    if mesh:
        base = (meta.get("target") or {}).get("base_url") or ""
        m = re.search(r":(\d+)", base.split("//")[-1])
        if m:
            port = int(m.group(1))
            hits = [dp for p, dp in mesh if p == port]
            if len(set(hits)) == 1:          # 그 포트에서 항상 같은 구성이었을 때만 신뢰
                return hits[0], "server_log"

    legacy = LEGACY_RUN_CARDS.get(meta.get("experiment", ""))
    if legacy:
        return legacy, "legacy_table"
    return None, "unknown"


def load_runs(raw_dir: str) -> list[dict[str, Any]]:
    runs = []
    for name in sorted(os.listdir(raw_dir)):
        d = os.path.join(raw_dir, name)
        meta_p, sum_p = os.path.join(d, "meta.json"), os.path.join(d, "summary.json")
        if not (os.path.isfile(meta_p) and os.path.isfile(sum_p)):
            continue
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        with open(sum_p, encoding="utf-8") as f:
            summary = json.load(f)
        runs.append({"run_id": name, "meta": meta, "summary": summary})
    return runs


def run_spans(runs: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """run 마다 (시작, 끝) 벽시계 시각을 구한다.

    meta.started_at 을 기준점으로 삼고, 조건들의 측정 구간 끝을 더한다.
    run 이 겹쳤는지 판정하는 데 쓴다.
    """
    import datetime as _dt
    spans: dict[str, tuple[float, float]] = {}
    for run in runs:
        started = (run["meta"] or {}).get("started_at")
        if not started:
            continue
        try:
            t0 = _dt.datetime.fromisoformat(started).timestamp()
        except ValueError:
            continue
        last = 0.0
        for cond in run["summary"]:
            w = cond.get("window_s") or 0
            last = max(last, w)
        # 조건이 여러 개면 순차 실행이므로 전체 길이는 합에 가깝다
        total = sum((c.get("window_s") or 0) for c in run["summary"])
        spans[run["run_id"]] = (t0, t0 + max(total, last))
    return spans


def overlapping_runs(spans: dict[str, tuple[float, float]]) -> dict[str, list[str]]:
    """서로 시간이 겹치는 run 들.

    다른 측정이 같은 머신에서 동시에 돌면 부하 생성기와 호스트 자원을 나눠 쓰게 되어
    처리량이 낮게 나온다. 실측에서 soak 단독 ~707 tok/s 가 동시 실행 중 ~521 tok/s 로
    떨어졌다. 겹친 run 은 절대 처리량 근거로 쓰지 않는다.
    """
    out: dict[str, list[str]] = {k: [] for k in spans}
    ids = list(spans)
    for i, a in enumerate(ids):
        a0, a1 = spans[a]
        for b in ids[i + 1:]:
            b0, b1 = spans[b]
            if a0 < b1 and b0 < a1:
                out[a].append(b)
                out[b].append(a)
    return out


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    return v[max(0, math.ceil(len(v) * q) - 1)]


def recover_percentiles(raw_dir: str, run_id: str, spec: dict[str, Any]
                        ) -> dict[str, Any] | None:
    """warm-up 이 동시성보다 적었던 측정의 백분위를 원본에서 다시 낸다.

    워커 N개에 warm-up 이 M건(M<N)이면 워커 N−M 개의 첫 요청이 측정에 들어오고,
    그것들은 큐가 형성되는 동안 대기해 지연이 부풀려진다. **앞 N건을 빼면** 참값이
    복원된다 (실측: B1 conc=32 p95 1,437ms → 247ms, 참값 252ms).

    재측정하지 않고 기존 데이터를 살리는 경로다.
    """
    conc = spec.get("concurrency") or 1
    warm = spec.get("warmup_requests") or 0
    if warm >= conc:
        return None
    path = os.path.join(raw_dir, run_id, "requests.jsonl")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    sel = [r for r in recs
           if r.get("concurrency") == conc
           and r.get("target_input_tokens") == spec.get("input_tokens")
           and r.get("target_output_tokens") == spec.get("output_tokens")
           and not r.get("is_warmup") and r.get("error") is None]
    if len(sel) <= conc * 2:          # 뺄 만큼 남지 않으면 손대지 않는다
        return None
    sel.sort(key=lambda r: r.get("t_send", 0.0))
    kept = sel[conc:]
    ttft = [r["ttft_ms"] for r in kept if r.get("ttft_ms") is not None]
    e2e = [r["e2e_ms"] for r in kept if r.get("e2e_ms") is not None]
    if not ttft:
        return None
    return {
        "ramp_excluded": conc,
        "n_after_exclusion": len(kept),
        "ttft_ms_p50": _pct(ttft, 0.50),
        "ttft_ms_p95": _pct(ttft, 0.95),
        "e2e_ms_p95": _pct(e2e, 0.95) if e2e else None,
    }


def _condition_requests(raw_dir: str, run_id: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """이 조건의 측정 요청(warm-up·에러 제외). 창 재구성과 in-flight 계산에 쓴다."""
    path = os.path.join(raw_dir, run_id, "requests.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [r for r in recs
            if r.get("concurrency") == spec.get("concurrency")
            and r.get("target_input_tokens") == spec.get("input_tokens")
            and r.get("target_output_tokens") == spec.get("output_tokens")
            and not r.get("is_warmup") and r.get("error") is None
            and r.get("t_last_token") is not None]


def _peak_key(run_id: str, spec: dict[str, Any]) -> str:
    return (f"{run_id}|{spec.get('input_tokens')}|{spec.get('output_tokens')}"
            f"|{spec.get('concurrency')}")


def load_recovered_peaks(path: str) -> dict[str, dict[str, Any]]:
    """이전에 원본 시계열로 재계산해 저장해 둔 값.

    `metrics_timeseries.jsonl` 은 용량이 커서(run 하나에 108MB) 커밋 대상이 아니다.
    그래서 clone 한 환경에는 시계열이 없고 재계산을 못 한다. 그때 저장된 summary 값을
    쓰면 **커밋된 집계와 다른 결과가 나온다** — 세션 1 의 누적 최댓값이 되살아나
    9개 조건이 다시 오염되고, 그중 일부가 잘못 무효 처리된다.
    README 가 안내하는 재현 명령이 커밋된 것과 다른 데이터를 만들면 안 된다.

    그래서 재계산 결과를 이 작은 파일에 실어 커밋한다. 시계열이 있으면 다시 계산하고,
    없으면 여기서 읽는다.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["key"]: r for r in json.load(f)["peaks"]}


def recover_peaks(raw_dir: str, run_id: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """조건별 창으로 running·waiting·kv 최댓값을 원본 시계열에서 다시 낸다.

    세션 1 하네스는 `peak_sum` 에 창을 넘기지 않아 **run 시작부터의 누적 최댓값**을
    조건마다 기록했다. B1 run 17개 조건 중 9개가 틀렸다 — 입력 512 블록은 동시성이
    1→64 로 단조 증가해 누적과 구간이 우연히 같았고, 입력 1024 가 conc=1 로 되돌아가는
    순간 64 에 굳었다.

    그 값이 `diagnose_bottleneck` 을 6건 뒤집는다 (동시성 1 에서 대기 0 인데 41 로
    기록돼 "스케줄링이 병목" 으로 판정). 원본 시계열이 있으면 재측정 없이 복구된다.

    시계열은 gitignore 대상이라 clone 환경에는 없다. 그때는 None 을 돌려주고
    호출부가 저장된 값을 쓰되 `peaks_source` 로 그 사실을 드러낸다.
    """
    path = os.path.join(raw_dir, run_id, "metrics_timeseries.jsonl")
    if not os.path.isfile(path):
        return None
    ok = _condition_requests(raw_dir, run_id, spec)
    if not ok:
        return None
    lo = min(r["t_send"] for r in ok)
    hi = max(r["t_last_token"] for r in ok)

    run_tot: list[float] = []
    wait_tot: list[float] = []
    kv_max: list[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            t = s.get("t", 0.0)
            if t < lo or t > hi:
                continue
            m = s.get("metrics") or {}
            # 엔진별로 나뉘어 나오므로 running·waiting 은 합산한다 (dp 배포).
            # kv 는 사용률이라 합산하지 않고 최댓값을 쓴다.
            r_vals = [v for k, v in m.items() if "num_requests_running" in k]
            w_vals = [v for k, v in m.items() if "num_requests_waiting" in k]
            k_vals = [v for k, v in m.items() if "kv_cache_usage" in k]
            if r_vals:
                run_tot.append(sum(r_vals))
            if w_vals:
                wait_tot.append(sum(w_vals))
            if k_vals:
                kv_max.append(max(k_vals))
    if not run_tot:
        return None
    return {
        "peak_running": max(run_tot),
        "peak_waiting": max(wait_tot) if wait_tot else None,
        "kv_cache_peak": max(kv_max) if kv_max else None,
        "window": (lo, hi),
    }


def mean_inflight(raw_dir: str, run_id: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """창 동안 실제로 떠 있던 평균 동시성.

    `peak_running` 은 "목표에 한 번 닿았다" 를 말하지 "유지했다" 를 말하지 않는다.
    세션 1 의 다중 카드 측정은 창이 짧아(8장 14.6초) 파이프가 안 찬 채로 끝났고,
    그것이 그대로 "카드가 늘수록 효율이 떨어진다" 로 찍혔다 (8장 유지율 85.6% vs
    1장 97.6%). 확장효율을 명목 동시성으로 나누면 이 차이가 섞인다.

    디코드 in-flight 를 따로 내는 것은 처리량이 디코드에서 나오기 때문이다 —
    성능 정규화는 이쪽으로 해야 한다.
    """
    ok = _condition_requests(raw_dir, run_id, spec)
    if not ok:
        return None
    lo = min(r["t_send"] for r in ok)
    hi = max(r["t_last_token"] for r in ok)
    win = hi - lo
    if win <= 0:
        return None
    total = sum((r["e2e_ms"] or 0.0) / 1000.0 for r in ok)
    decode = sum(max(0.0, r["t_last_token"] - r["t_first_token"])
                 for r in ok if r.get("t_first_token") is not None)
    conc = spec.get("concurrency") or 1
    return {
        "mean_inflight": total / win,
        "mean_decode_inflight": decode / win,
        "load_retention": (total / win) / conc,
    }


def to_rows(runs: list[dict[str, Any]], raw_dir: str | None = None,
            recovered: dict[str, dict[str, Any]] | None = None,
            fresh_peaks: list[dict[str, Any]] | None = None
            ) -> tuple[list[dict], list[dict]]:
    """(사용 가능한 행, 건너뛴 행) 을 돌려준다. 건너뛴 이유를 남긴다."""
    rows, skipped = [], []
    recovered = recovered or {}
    if fresh_peaks is None:
        fresh_peaks = []
    overlaps = overlapping_runs(run_spans(runs))
    mesh = server_log_mesh(os.path.join(REPO_ROOT, "data", "server_logs", "serve"))
    for run in runs:
        meta, run_id = run["meta"], run["run_id"]
        model = (meta.get("target") or {}).get("model")
        source = meta.get("source")
        cards, card_source = infer_cards(meta, mesh)

        for cond in run["summary"]:
            spec = cond.get("spec") or {}
            # 임베딩 결과는 스키마가 달라 별도 처리한다
            if "batch_size" in spec:
                continue
            if cond.get("aggregate_output_tps") is None:
                skipped.append({"run_id": run_id, "spec": spec, "reason": "집계 실패"})
                continue
            # hosted 측정에는 카드 수라는 개념 자체가 없다(멀티테넌트, 구성 비공개).
            # 그러므로 카드 수를 찾았는지와 **무관하게** 출처만으로 판정한다.
            # 이름 표에 우연히 같은 실험 코드가 있으면 호스팅 행에 카드 수가 붙어버린다
            # (실제로 E1 이 legacy_table 로 1장이 붙었다).
            #
            # 그리고 run 단위 변수(cards, card_source)를 이 루프 안에서 덮어쓰면 안 된다.
            # 첫 조건이 자리표시를 넣으면 다음 조건부터 판정이 뒤집힌다.
            hosted = source != "measured_local"
            cond_cards = 1 if hosted else cards          # 자리표시. 게이트가 먼저 거부한다
            cond_card_source = "정의되지 않음 (hosted)" if hosted else card_source
            if cond_cards is None:
                skipped.append({"run_id": run_id, "spec": spec, "source": source,
                                "reason": "카드 수를 알 수 없음 (실측 데이터 유실)"})
                continue
            v = cond.get("validation") or {}
            conc_total = spec.get("concurrency", 1)

            # 하네스가 남긴 capacity_usable 을 **믿지 않고 다시 계산한다.**
            # 옛 run 은 출처만 보던 시절의 규칙으로 쓰인 값이라 오염을 놓친다.
            # 행을 버리지는 않는다 — 버리면 왜 사라졌는지가 남지 않는다.
            # 거부는 planner 의 로드 게이트가 이 플래그를 보고 한다.
            # 세션 1 하네스는 조건별 창을 안 걸러 누적 최댓값을 기록했다(B1 17개 중 9개).
            # 원본 시계열이 있으면 다시 계산하고, 없으면 저장값을 쓰되 출처를 밝힌다.
            peaks = recover_peaks(raw_dir, run_id, spec) if raw_dir else None
            cached = recovered.get(_peak_key(run_id, spec))
            if peaks:
                pk_run, pk_wait = peaks["peak_running"], peaks["peak_waiting"]
                pk_kv, peaks_src = peaks["kv_cache_peak"], "recomputed_condition_window"
                fresh_peaks.append({"key": _peak_key(run_id, spec), "run_id": run_id,
                                    "input_tokens": spec.get("input_tokens"),
                                    "output_tokens": spec.get("output_tokens"),
                                    "concurrency": spec.get("concurrency"),
                                    "peak_running": pk_run, "peak_waiting": pk_wait,
                                    "kv_cache_peak": pk_kv})
            elif cached:
                # 시계열이 없는 환경(clone). 커밋된 재계산 결과를 쓴다.
                pk_run, pk_wait = cached["peak_running"], cached["peak_waiting"]
                pk_kv, peaks_src = cached["kv_cache_peak"], "recovered_from_committed"
            else:
                pk_run = (v.get("concurrency") or {}).get("peak_running_samples")
                pk_wait = (v.get("concurrency") or {}).get("peak_waiting_samples")
                pk_kv, peaks_src = v.get("kv_cache_peak"), "summary"
            flow = mean_inflight(raw_dir, run_id, spec) if raw_dir else None

            # 판정은 **저장된 verdict 문자열이 아니라 값에서** 다시 낸다.
            # 옛 run 은 상한 검사가 없던 시절의 "ok" 를 달고 있어 문자열을 믿으면 못 잡는다.
            # (soak 두 행: requested 16 · peak_running 32 · verdict "ok")
            v = dict(v)
            if pk_run is not None:
                cc = dict(v.get("concurrency") or {})
                cc["requested"] = conc_total
                cc["peak_running_samples"] = pk_run
                ratio = pk_run / max(1, conc_total)
                if ratio > FOREIGN_LOAD_RATIO:
                    cc["verdict"] = "FOREIGN_LOAD"
                elif (cc.get("verdict") or "") == "FOREIGN_LOAD":
                    cc["verdict"] = "ok"          # 재계산으로 오탐이 걷힌 경우
                v["concurrency"] = cc
            blocks = capacity_blocks(v)
            if (meta.get("target") or {}).get("not_for_capacity"):
                blocks = blocks + ["실험 처치군으로 표시된 run 입니다 (--not-for-capacity)"]
            elif run_id in EXPERIMENT_ARMS:
                blocks = blocks + [f"실험 처치군입니다 — {EXPERIMENT_ARMS[run_id]}"]

            # warm-up 이 동시성보다 적었으면 앞 구간을 빼고 백분위를 다시 낸다.
            ttft_p50 = cond.get("ttft_ms_p50")
            ttft_p95 = cond.get("ttft_ms_p95")
            e2e_p95 = cond.get("e2e_ms_p95")
            pct_source = "summary"
            ramp_excluded = 0
            rec = recover_percentiles(raw_dir, run_id, spec) if raw_dir else None
            if rec:
                ttft_p50, ttft_p95 = rec["ttft_ms_p50"], rec["ttft_ms_p95"]
                e2e_p95 = rec["e2e_ms_p95"] if rec["e2e_ms_p95"] is not None else e2e_p95
                pct_source = "recomputed_ramp_excluded"
                ramp_excluded = rec["ramp_excluded"]
            rows.append({
                "model": model,
                "source": source,
                "n_cards": cond_cards,
                "n_cards_known": not hosted,
                # hosted 는 카드 수를 모르므로 '카드당' 이 성립하지 않는다. 총량을 그대로 둔다.
                "concurrency_per_card": conc_total if hosted else max(1, round(conc_total / cond_cards)),
                "concurrency_total": conc_total,
                "input_tokens": spec.get("input_tokens"),
                "output_tokens": spec.get("output_tokens"),
                "aggregate_output_tps": cond.get("aggregate_output_tps"),
                "per_user_output_tps": cond.get("per_user_output_tps_median"),
                "ttft_ms_p50": ttft_p50,
                "ttft_ms_p95": ttft_p95,
                "e2e_ms_p95": e2e_p95,
                "tpot_ms_p50": cond.get("tpot_ms_p50"),
                "capacity_usable": not blocks,
                "capacity_blocks": blocks,
                "kv_cache_peak": pk_kv,
                "waiting_peak": pk_wait,
                "peak_running": pk_run,
                "peaks_source": peaks_src,
                # 요청한 동시성 중 서버가 실제로 물고 있던 비율. 1.00 미만이면 포화,
                # 초과면 같은 서버에 다른 클라이언트가 붙은 것이다.
                "concurrency_achieved_ratio": (round(pk_run / max(1, conc_total), 4)
                                               if pk_run is not None else None),
                # peak 은 "닿았다", 아래 둘은 "유지했다" 를 말한다.
                "mean_inflight": round(flow["mean_inflight"], 3) if flow else None,
                "mean_decode_inflight": round(flow["mean_decode_inflight"], 3) if flow else None,
                "load_retention": round(flow["load_retention"], 4) if flow else None,
                "n_samples": cond.get("n_measured_ok", 0),
                "window_s": cond.get("window_s"),
                "card_source": cond_card_source,
                "percentile_source": pct_source,
                "ramp_excluded": ramp_excluded,
                "warmup_requests": spec.get("warmup_requests"),
                "exclusive": not overlaps.get(run_id),
                "concurrent_with": overlaps.get(run_id, []),
                "run_id": run_id,
            })
    return rows, skipped


# 확장 효율이 이 값을 넘으면 물리적으로 불가능하다. planner 에도 같은 상수가 있는데
# 그쪽은 자체 계산분에만 걸려서, 이 파일이 만드는 scaling.json 은 검사를 안 탔다.
# 실제로 공개 저장소에 "1장이 다른 1장 대비 135.7%" 가 13행 들어가 있었다.
EFFICIENCY_SANITY_LIMIT = 1.15
# 확장 효율을 비교 가능하다고 보는 최소 조건. 유지율은 세션 2 깊은 측정이 96.8~99.4%
# 였고 세션 1 다중 카드는 37~94% 였다. 창은 짧을수록 과도 구간 비중이 커진다.
SCALING_MIN_RETENTION = 0.95
# 창 하한을 과도 구간 길이에서 잡는다. 세션 2 구간 분해에서 처리량이 첫 2~3분에 걸쳐
# 떨어지고(conc=32 −9.9%, conc=16 −2.2%) 그 뒤 평탄해졌다. 창이 그보다 짧으면 과도
# 구간만 재게 된다 — 세션 1 의 C2x1(창 60초)이 정상상태보다 12.4% 높았던 이유다.
SCALING_MIN_WINDOW_S = 120.0


def scaling_table(rows: list[dict]) -> list[dict]:
    """같은 (모델, 입력, 출력, 카드당 동시성) 에서 카드 수만 다른 행을 묶어 확장 효율을 낸다."""
    base: dict[tuple, dict] = {}
    for r in rows:
        if r["source"] != "measured_local" or r["n_cards"] != 1:
            continue
        # 동시성 라벨이 실측을 서술하지 못하는 행은 기준선이 될 수 없다.
        # soak 두 행이 정확히 그랬다 — 라벨은 conc=16 인데 창의 3/4 동안 카드가 32 를
        # 받고 있어 처리량이 521 로 낮았고, 그게 카드당 16 열 전체의 기준선이 되어
        # 효율을 135~147% 로 부풀렸다.
        if not r.get("capacity_usable", True):
            continue
        key = (r["model"], r["input_tokens"], r["output_tokens"], r["concurrency_per_card"])
        # 측정창이 더 긴 쪽을 기준으로 삼는다 (짧은 창은 처리량을 과소평가한다)
        if key not in base or (r.get("window_s") or 0) > (base[key].get("window_s") or 0):
            base[key] = r

    out = []
    for r in rows:
        if r["source"] != "measured_local":
            continue
        # 무효 처리된 행은 기준선뿐 아니라 비교 대상에서도 뺀다. 그러지 않으면
        # 오염된 1장 측정이 "1장인데 효율 68%" 같은 모양으로 표에 남는다
        # (A3dualA/B: 같은 카드에 클라이언트 2개, soak: 라벨과 실제 동시성 불일치).
        if not r.get("capacity_usable", True):
            continue
        key = (r["model"], r["input_tokens"], r["output_tokens"], r["concurrency_per_card"])
        b = base.get(key)
        if not b:
            continue
        ideal = b["aggregate_output_tps"] * r["n_cards"]
        out.append({
            "model": r["model"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "concurrency_per_card": r["concurrency_per_card"],
            "n_cards": r["n_cards"],
            "aggregate_output_tps": r["aggregate_output_tps"],
            "per_card_tps": r["aggregate_output_tps"] / r["n_cards"],
            "single_card_tps": b["aggregate_output_tps"],
            "scaling_efficiency": r["aggregate_output_tps"] / ideal if ideal else None,
            "window_s": r.get("window_s"),
            "run_id": r["run_id"],
            "baseline_run_id": b["run_id"],
            # 명목 동시성으로 나눈 효율은 유지율 차이를 그대로 흡수한다. 세션 1 은
            # 카드가 늘수록 창이 짧아져(1장 60초 → 8장 14.6초) 유지율이 97.6% → 85.6%
            # 로 떨어졌고, 그것이 "8장 효율 82.1%" 로 찍혔다. 양쪽 유지율을 같이 실어
            # 읽는 사람이 그 차이를 볼 수 있게 한다.
            "load_retention": r.get("load_retention"),
            "baseline_load_retention": b.get("load_retention"),
            # 양쪽이 같은 조건에서 측정됐는가. 유지율이 다르면 그 차이가 효율로 둔갑하고,
            # 창이 너무 짧으면 run 간 산포(같은 조건 반복 시 0.92%)보다 큰 잡음이 섞인다.
            # 이 값이 false 인 행의 효율로는 결론을 내면 안 된다.
            "comparable": bool(
                (r.get("load_retention") or 0) >= SCALING_MIN_RETENTION
                and (b.get("load_retention") or 0) >= SCALING_MIN_RETENTION
                and (r.get("window_s") or 0) >= SCALING_MIN_WINDOW_S
                and (b.get("window_s") or 0) >= SCALING_MIN_WINDOW_S
            ),
        })
    # 비교 가능한 행에서만 본다. 유지율이 다르거나 창이 짧은 행의 효율은
    # 애초에 물리량이 아니라서 상한을 논할 대상이 아니다.
    over = [o for o in out
            if o["comparable"] and (o["scaling_efficiency"] or 0) > EFFICIENCY_SANITY_LIMIT]
    if over:
        worst = max(over, key=lambda o: o["scaling_efficiency"])
        print(f"⚠ 확장 효율이 물리적 상한({EFFICIENCY_SANITY_LIMIT:.0%})을 넘는 행이 "
              f"{len(over)}개 있습니다. 최댓값 {worst['scaling_efficiency']:.1%} "
              f"(카드당 {worst['concurrency_per_card']}, {worst['n_cards']}장, "
              f"기준 {worst['baseline_run_id']}). 기준행 선택이나 조건 묶기를 확인하세요.",
              file=sys.stderr)
    return sorted(out, key=lambda x: (x["concurrency_per_card"], x["n_cards"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="raw 측정 → processed 테이블")
    ap.add_argument("--raw", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "processed"))
    args = ap.parse_args(argv)

    runs = load_runs(args.raw)
    os.makedirs(args.out, exist_ok=True)
    peaks_path = os.path.join(args.out, "recovered_peaks.json")
    recovered = load_recovered_peaks(peaks_path)
    fresh_peaks: list[dict[str, Any]] = []
    rows, skipped = to_rows(runs, raw_dir=args.raw,
                            recovered=recovered, fresh_peaks=fresh_peaks)
    # 시계열로 새로 계산한 값이 있으면 저장해 둔다. 이 파일이 커밋되므로 clone 한
    # 환경에서도 같은 결과가 나온다 (시계열 자체는 용량 때문에 커밋하지 않는다).
    if fresh_peaks:
        merged = {r["key"]: r for r in recovered.values()}
        merged.update({r["key"]: r for r in fresh_peaks})
        with open(peaks_path, "w", encoding="utf-8") as f:
            json.dump({
                "note": ("조건별 창으로 다시 계산한 running/waiting/kv 최댓값. "
                         "원본 metrics_timeseries.jsonl 은 용량 때문에 커밋하지 않으므로, "
                         "clone 한 환경은 이 파일을 읽는다. 시계열이 있으면 다시 계산하고 "
                         "이 파일을 갱신한다."),
                "peaks": sorted(merged.values(), key=lambda r: r["key"]),
            }, f, ensure_ascii=False, indent=2)

    usable = [r for r in rows if r["source"] == "measured_local"]
    with open(os.path.join(args.out, "benchmark_rows.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "skipped": skipped}, f, ensure_ascii=False, indent=2)
    scal = scaling_table(rows)
    with open(os.path.join(args.out, "scaling.json"), "w", encoding="utf-8") as f:
        json.dump(scal, f, ensure_ascii=False, indent=2)

    print(f"run {len(runs)}개 → 행 {len(rows)}개 (measured_local {len(usable)}개, 건너뜀 {len(skipped)}개)")
    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    for s, n in sorted(by_src.items()):
        print(f"  {s}: {n}")

    by_card: dict[str, int] = {}
    for r in rows:
        by_card[r.get("card_source", "?")] = by_card.get(r.get("card_source", "?"), 0) + 1
    print("카드 수 출처: " + ", ".join(f"{k} {v}" for k, v in sorted(by_card.items())))

    # 실측 데이터가 버려지는 것은 조용히 넘어가면 안 된다.
    # 측정 세션 하나를 통째로 날릴 수 있다.
    lost = [s for s in skipped if s.get("source") == "measured_local"
            or "유실" in s.get("reason", "")]
    if lost:
        print()
        print(f"!! 실측 조건 {len(lost)}개가 버려졌습니다. 확인이 필요합니다:")
        for s in lost[:5]:
            print(f"   {s['run_id']}  {s['reason']}")
        print("   → 측정 시 `--cards N` 을 붙였는지, 서버 기동 로그가 회수됐는지 확인하세요.")
    print(f"확장 비교 가능한 행: {len(scal)}개")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
