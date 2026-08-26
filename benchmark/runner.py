"""조건 하나를 실행하고 집계·검증한다.

동시성 모델
-----------
워커 스레드 N개가 각자 커넥션 하나를 들고 요청을 **연속으로** 던진다. 요청이
끝나면 곧바로 다음 요청을 보내므로 측정 구간 내내 N개가 항상 in-flight 다.
즉 concurrency N 은 "한 번에 N개를 보내고 다 끝나기를 기다린다"가 아니라
"항상 N개가 처리 중"을 뜻한다. 후자가 실제 서비스에 가깝고, 앞의 방식은
매 배치 끝에서 동시성이 줄어들어 throughput 을 과소평가한다.

warm-up 과 측정 구간
--------------------
dispatch 순서로 앞의 warmup_requests 건은 warm-up 이다. 기록은 하되
(is_warmup=True) 집계에서 제외한다. 삭제하지 않는 이유는 이상치를 지우기 전에
원인을 먼저 보기 위해서다 (docs/01 IMP-8).

    window_start = 첫 측정 요청의 전송 시각
    window_end   = 마지막 측정 요청의 마지막 토큰 수신 시각
    aggregate_output_tps = Σ(측정 요청의 completion_tokens) / (window_end − window_start)

wall-clock 기반 aggregate 와 요청별 latency 를 섞지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import threading
from typing import Any

from .client import StreamingClient
from .metrics import MetricsPoller
from .prompts import PromptFactory
from .schema import RequestRecord
from .target import Target

# 이 개수 미만이면 P95 를 신뢰할 수 없다고 보고 값을 내지 않는다 (docs/01 IMP-8).
MIN_SAMPLES_FOR_P95 = 100

# prompt 중 이 비율을 넘게 prefix cache 에서 나오면 오염으로 본다.
# 서로 다른 프롬프트라도 선두 토큰 2~4개는 공유되므로 0 을 기준으로 쓰면 전부 오탐이 된다.
PREFIX_CACHE_RATIO_LIMIT = 0.05


@dataclass
class ConditionSpec:
    experiment: str
    concurrency: int = 1
    input_tokens: int = 512
    output_tokens: int = 128
    warmup_requests: int = 4
    measured_requests: int = 100
    ignore_eos: bool = True
    min_samples_for_p95: int = MIN_SAMPLES_FOR_P95

    def label(self) -> str:
        return (f"{self.experiment} conc={self.concurrency} "
                f"in={self.input_tokens} out={self.output_tokens}")


@dataclass
class ConditionResult:
    spec: dict[str, Any]
    source: str
    n_measured_ok: int = 0
    n_error: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    window_s: float | None = None
    aggregate_output_tps: float | None = None
    per_user_output_tps_median: float | None = None

    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    e2e_ms_p50: float | None = None
    e2e_ms_p95: float | None = None
    tpot_ms_p50: float | None = None

    prompt_tokens_median: float | None = None
    completion_tokens_median: float | None = None

    # 검증 결과 — 값이 아니라 신뢰성에 대한 판정이다.
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(values: list[float], q: float) -> float | None:
    """nearest-rank 백분위. 표본이 적을 때 statistics 보다 해석이 명확하다."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, min(len(s), int(-(-len(s) * q // 1))))   # ceil(n*q)
    return s[k - 1]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_condition(target: Target, factory: PromptFactory, spec: ConditionSpec, *,
                  run_id: str, on_record, t0: float,
                  poller: MetricsPoller | None = None,
                  progress=None) -> ConditionResult:
    """조건 하나를 끝까지 실행한다.

    t0 는 **run 전체의 기준 시각**이다. 요청 레코드와 metrics 시계열이 같은 시계를
    써야 나중에 "이 요청이 처리될 때 running_samples 가 몇이었나"를 맞춰볼 수 있다.
    """
    total = spec.warmup_requests + spec.measured_requests
    lock = threading.Lock()
    state = {"dispatched": 0}
    records: list[RequestRecord] = []

    # 프롬프트 길이 보정은 워커 시작 전에 한 번만 (워커들이 동시에 보정하지 않도록)
    factory.calibrate(spec.input_tokens)

    def worker() -> None:
        client = StreamingClient(target)
        try:
            while True:
                with lock:
                    if state["dispatched"] >= total:
                        return
                    idx = state["dispatched"]
                    state["dispatched"] += 1
                rec = client.stream_request(
                    factory.make(spec.input_tokens), spec.output_tokens,
                    run_id=run_id, experiment=spec.experiment,
                    concurrency=spec.concurrency, t0=t0,
                    target_input_tokens=spec.input_tokens,
                    is_warmup=idx < spec.warmup_requests,
                    ignore_eos=spec.ignore_eos,
                )
                with lock:
                    records.append(rec)
                    on_record(rec)
                    if progress:
                        progress(len(records), total)
        finally:
            client.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(spec.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return _summarize(spec, target, records, poller)





def _summarize(spec: ConditionSpec, target: Target, records: list[RequestRecord],
               poller: MetricsPoller | None) -> ConditionResult:
    res = ConditionResult(spec=asdict(spec), source=target.source)

    measured = [r for r in records if not r.is_warmup]
    ok = [r for r in measured if r.error is None and r.t_last_token is not None]
    bad = [r for r in measured if r.error is not None]

    res.n_measured_ok = len(ok)
    res.n_error = len(bad)
    for r in bad:
        key = (r.error or "")[:60]
        res.errors[key] = res.errors.get(key, 0) + 1

    if not ok:
        res.validation["fatal"] = "성공한 측정 요청이 없습니다."
        return res

    # -- 처리량: wall-clock 윈도 기준 -------------------------------------
    window_start = min(r.t_send for r in ok)
    window_end = max(r.t_last_token for r in ok)          # type: ignore[type-var]
    window = window_end - window_start
    res.window_s = window
    total_out = sum(r.completion_tokens_actual or 0 for r in ok)
    if window > 0:
        res.aggregate_output_tps = total_out / window

    # -- 지연: 요청 단위 ---------------------------------------------------
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    e2es = [r.e2e_ms for r in ok if r.e2e_ms is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]

    res.ttft_ms_p50 = _median(ttfts)
    res.e2e_ms_p50 = _median(e2es)
    res.tpot_ms_p50 = _median(tpots)
    res.prompt_tokens_median = _median([float(r.prompt_tokens_actual or 0) for r in ok])
    res.completion_tokens_median = _median([float(r.completion_tokens_actual or 0) for r in ok])

    per_user = [(r.completion_tokens_actual or 0) / (r.e2e_ms / 1000.0)
                for r in ok if r.e2e_ms]
    res.per_user_output_tps_median = _median(per_user)

    # 표본이 부족하면 P95 를 내지 않는다.
    if len(ok) >= spec.min_samples_for_p95:
        res.ttft_ms_p95 = _pct(ttfts, 0.95)
        res.e2e_ms_p95 = _pct(e2es, 0.95)
    else:
        res.validation["p95"] = (
            f"insufficient_samples: 측정 성공 {len(ok)}건 < 기준 {spec.min_samples_for_p95}건"
        )

    _validate(res, spec, ok, poller, window=(window_start, window_end))
    return res


def _validate(res: ConditionResult, spec: ConditionSpec, ok: list[RequestRecord],
              poller: MetricsPoller | None,
              window: tuple[float, float] | None = None) -> None:
    """결과를 믿어도 되는지 스스로 점검한다. 실패해도 값을 지우지 않고 표시만 한다."""
    v = res.validation

    # 1) prefix cache 오염.
    #    절대 건수가 아니라 **비율**로 본다. 서로 다른 프롬프트라도 선두 토큰
    #    몇 개(BOS 등)는 공유되어 cached_tokens 가 2~4로 잡히는데, 이건 오염이
    #    아니다. 프롬프트의 상당 부분이 캐시에서 나올 때만 prefill 비용이
    #    과소평가된다 (docs/03-api-findings.md §3).
    pairs = [(r.cached_tokens, r.prompt_tokens_actual) for r in ok
             if r.cached_tokens is not None and r.prompt_tokens_actual]
    if pairs:
        ratios = [c / p for c, p in pairs]
        worst = max(ratios)
        over = [x for x in ratios if x > PREFIX_CACHE_RATIO_LIMIT]
        v["prefix_cache"] = {
            "reported": True,
            "max_cached_ratio": round(worst, 4),
            "max_cached_tokens": max(c for c, _ in pairs),
            "requests_over_limit": len(over),
            "limit": PREFIX_CACHE_RATIO_LIMIT,
            "verdict": "clean" if not over else "CONTAMINATED",
        }
    else:
        v["prefix_cache"] = {"reported": False,
                             "note": "서버가 cached_tokens 를 보고하지 않습니다. /metrics 로 확인하세요."}

    # 2) 출력 길이 고정이 실제로 먹었는지
    lens = {r.completion_tokens_actual for r in ok if r.completion_tokens_actual is not None}
    v["output_length"] = {
        "requested": spec.output_tokens,
        "distinct_observed": sorted(lens)[:5],
        "verdict": "fixed" if lens == {spec.output_tokens} else "VARIES",
    }

    # 3) 입력 길이가 의도와 맞는지 (랜덤 단어라 소폭 흔들린다)
    if res.prompt_tokens_median is not None:
        drift = abs(res.prompt_tokens_median - spec.input_tokens) / max(1, spec.input_tokens)
        v["input_length"] = {
            "target": spec.input_tokens,
            "median_actual": res.prompt_tokens_median,
            "drift_pct": round(drift * 100, 2),
            "verdict": "ok" if drift <= 0.05 else "DRIFTED",
        }

    # 4) 의도한 concurrency 가 실제로 발생했는지 — 배칭인가 큐잉인가
    if poller and poller.available:
        t_from, t_to = window if window else (None, None)
        # 엔진별 값을 합산해야 dp 배포에서 실제 동시 처리량이 나온다.
        running = poller.peak_sum("running", t_from, t_to)
        waiting = poller.peak_sum("waiting", t_from, t_to)
        v["concurrency"] = {
            "requested": spec.concurrency,
            "peak_running_samples": running,
            "peak_waiting_samples": waiting,
            "verdict": ("ok" if running is not None and running >= spec.concurrency * 0.8
                        else "QUEUED_NOT_BATCHED" if running is not None else "unknown"),
        }
        kv = poller.peak("kv_cache", t_from, t_to)   # 사용률이므로 합산하지 않고 최댓값
        if kv is not None:
            v["kv_cache_peak"] = kv
    else:
        v["concurrency"] = {"requested": spec.concurrency,
                            "note": "/metrics 없음 — 실제 동시 처리량을 확인할 수 없습니다."}

    # 5) 재시도 — 커넥션이 끊겨 다시 보낸 요청은 측정이 방해받았다는 뜻이다.
    retried = [r for r in ok if (r.attempts or 1) > 1]
    if retried:
        v["retries"] = {"requests_retried": len(retried),
                        "verdict": "DISTURBED",
                        "note": "커넥션 재수립이 발생했습니다. 해당 요청의 지연에 영향이 있을 수 있습니다."}

    # 6) 출처
    v["source"] = res.source
    if res.source != "measured_local":
        v["capacity_usable"] = False
        v["capacity_note"] = (
            "전용 서버 실측이 아닙니다. capacity 계산에 사용할 수 없습니다 "
            "(docs/03-api-findings.md §7)."
        )
    else:
        v["capacity_usable"] = True
