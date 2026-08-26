"""planner 테스트용 합성 측정 행.

여기 값은 **테스트 픽스처**이지 실측이 아니다. 계산 로직의 경계 동작을 확인하기
위해 의도적으로 단순한 숫자를 쓴다. 실측 데이터는 `data/raw/` 에 따로 있다.
"""

import pytest

from planner.models import BenchmarkRow

MODEL = "test/model"


def row(concurrency, agg_tps, *, n_cards=1, input_tokens=512, output_tokens=128,
        ttft_p95=100.0, per_user=None, e2e_p95=None, tpot=10.0,
        window_s=60.0, exclusive=True, source="measured_local", n_samples=100,
        kv=None, waiting=None, model=MODEL, run_id="fixture",
        capacity_usable=True, capacity_blocks=None):
    return BenchmarkRow(
        model=model, source=source, n_cards=n_cards,
        concurrency_per_card=concurrency,
        input_tokens=input_tokens, output_tokens=output_tokens,
        aggregate_output_tps=agg_tps,
        per_user_output_tps=per_user if per_user is not None else agg_tps / (concurrency * n_cards),
        ttft_ms_p50=ttft_p95 * 0.6, ttft_ms_p95=ttft_p95, e2e_ms_p95=e2e_p95,
        tpot_ms_p50=tpot, kv_cache_peak=kv, waiting_peak=waiting,
        n_samples=n_samples, window_s=window_s, exclusive=exclusive, run_id=run_id,
        capacity_usable=capacity_usable, capacity_blocks=list(capacity_blocks or []),
    )


@pytest.fixture
def simple_curve():
    """동시성이 오를수록 처리량은 늘고 사용자당 속도와 지연은 나빠지는 곡선.

    conc 8 이후 TTFT p95 가 급격히 악화되도록 만들어, SLA 제약이 실제로
    작동하는지 확인할 수 있게 했다.
    """
    return [
        row(1, 100.0, ttft_p95=80.0),
        row(2, 190.0, ttft_p95=90.0),
        row(4, 360.0, ttft_p95=120.0),
        row(8, 640.0, ttft_p95=200.0),
        row(16, 800.0, ttft_p95=1500.0),
        row(32, 900.0, ttft_p95=6000.0),
    ]


@pytest.fixture
def store_single(simple_curve):
    from planner.benchmark_store import BenchmarkStore
    return BenchmarkStore(simple_curve)


@pytest.fixture
def store_multicard(simple_curve):
    """1장 곡선 + 완전 선형인 2·4장 측정."""
    from planner.benchmark_store import BenchmarkStore
    rows = list(simple_curve)
    for r in simple_curve:
        for n in (2, 4):
            rows.append(row(r.concurrency_per_card, r.aggregate_output_tps * n,
                            n_cards=n, ttft_p95=r.ttft_ms_p95, run_id=f"x{n}"))
    return BenchmarkStore(rows)


@pytest.fixture
def base_requirement():
    from planner.models import ServiceRequirement
    return dict(workload="llm_chat", model=MODEL, avg_input_tokens=512,
                avg_output_tokens=128)
