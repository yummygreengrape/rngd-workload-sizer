"""raw run → planner 가 읽는 실측 테이블.

`data/raw/{run_id}/summary.json` 들을 모아 `data/processed/benchmark_rows.json` 을 만든다.
planner 는 이 파일만 읽고, 출처가 measured_local 이 아닌 행은 로드 단계에서 거부한다.

카드 수(n_cards)는 capacity 계산의 분모라 반드시 필요하다. meta.json 의
`target.n_cards` 를 쓰되, 그 필드가 생기기 전에 측정한 run 은 아래 표로 보정한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def to_rows(runs: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """(사용 가능한 행, 건너뛴 행) 을 돌려준다. 건너뛴 이유를 남긴다."""
    rows, skipped = [], []
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
                "ttft_ms_p50": cond.get("ttft_ms_p50"),
                "ttft_ms_p95": cond.get("ttft_ms_p95"),
                "e2e_ms_p95": cond.get("e2e_ms_p95"),
                "tpot_ms_p50": cond.get("tpot_ms_p50"),
                "kv_cache_peak": v.get("kv_cache_peak"),
                "waiting_peak": (v.get("concurrency") or {}).get("peak_waiting_samples"),
                "n_samples": cond.get("n_measured_ok", 0),
                "window_s": cond.get("window_s"),
                "card_source": cond_card_source,
                "exclusive": not overlaps.get(run_id),
                "concurrent_with": overlaps.get(run_id, []),
                "run_id": run_id,
            })
    return rows, skipped


def scaling_table(rows: list[dict]) -> list[dict]:
    """같은 (모델, 입력, 출력, 카드당 동시성) 에서 카드 수만 다른 행을 묶어 확장 효율을 낸다."""
    base: dict[tuple, dict] = {}
    for r in rows:
        if r["source"] != "measured_local" or r["n_cards"] != 1:
            continue
        key = (r["model"], r["input_tokens"], r["output_tokens"], r["concurrency_per_card"])
        # 측정창이 더 긴 쪽을 기준으로 삼는다 (짧은 창은 처리량을 과소평가한다)
        if key not in base or (r.get("window_s") or 0) > (base[key].get("window_s") or 0):
            base[key] = r

    out = []
    for r in rows:
        if r["source"] != "measured_local":
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
        })
    return sorted(out, key=lambda x: (x["concurrency_per_card"], x["n_cards"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="raw 측정 → processed 테이블")
    ap.add_argument("--raw", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "processed"))
    args = ap.parse_args(argv)

    runs = load_runs(args.raw)
    rows, skipped = to_rows(runs)
    os.makedirs(args.out, exist_ok=True)

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
