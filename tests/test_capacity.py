"""필요 카드 수 계산.

기획서 검토 항목이 직접 지목한 부분이다. 특히:
- `users × target tok/s` 와 `required / capacity` 의 단위가 맞는가
- target utilization 이 중복 적용되거나 빠지지 않았는가
- 소수 카드가 나올 때 올림이 올바른가
- 처리량만 만족하면 "충분"으로 판정하는 문제가 없는가
"""

import math

import pytest

from planner.benchmark_store import BenchmarkStore
from planner.capacity import (
    best_operating_point, diagnose_bottleneck, max_concurrency_meeting_sla,
    plan, sla_checks,
)
from planner.models import ServiceRequirement
from tests.conftest import MODEL, row


def req(**kw):
    base = dict(workload="llm_chat", model=MODEL, avg_input_tokens=512,
                avg_output_tokens=128, concurrent_users=100,
                target_output_tps_per_user=10.0, target_max_ttft_ms=1000.0,
                target_utilization=1.0)
    base.update(kw)
    return ServiceRequirement(**base)


class TestUnits:
    """단위가 어긋나면 결과가 조용히 몇 배씩 틀린다."""

    def test_required_throughput_is_users_times_rate(self, store_single):
        r = plan(store_single, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert r.required_output_tps == pytest.approx(1000.0)

    def test_required_throughput_scales_with_both_factors(self, store_single):
        a = plan(store_single, req(concurrent_users=50, target_output_tps_per_user=20.0))
        b = plan(store_single, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert a.required_output_tps == b.required_output_tps == pytest.approx(1000.0)


class TestUtilization:
    """target_utilization 은 분모에 **한 번만** 적용된다."""

    def test_lower_utilization_needs_more_cards(self, store_single):
        full = plan(store_single, req(target_utilization=1.0))
        half = plan(store_single, req(target_utilization=0.5))
        assert half.n_cards_by_throughput >= full.n_cards_by_throughput

    def test_utilization_applied_exactly_once(self):
        """이용률 0.5 는 카드당 가용 처리량을 정확히 절반으로 만든다.

        두 번 곱하면 1/4 이 되어 카드 수가 배로 튄다.
        """
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=50.0,
                        target_utilization=0.5))
        # 필요 50 tok/s, 카드당 가용 100 × 0.5 = 50 → 정확히 1장
        assert r.n_cards_by_throughput == 1
        it = r.iterations[-1]
        assert it["usable_tps_per_card"] == pytest.approx(50.0)

    def test_headroom_converts_to_utilization(self):
        a = ServiceRequirement.from_headroom(0.3, workload="llm_chat", model=MODEL)
        assert a.target_utilization == pytest.approx(0.7)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_headroom_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            ServiceRequirement.from_headroom(bad, workload="llm_chat", model=MODEL)

    def test_estimated_utilization_does_not_reapply_target(self):
        """예상 이용률은 결과이지 제약이 아니다. 여기에 target 을 또 곱하면 안 된다."""
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=50.0,
                        target_utilization=0.5))
        # 필요 50 / (1장 × 100 실측) = 0.5
        assert r.estimated_utilization == pytest.approx(0.5)


class TestRounding:
    """카드는 정수다. 반올림하면 용량이 모자란다."""

    def test_fractional_rounds_up(self):
        """필요 100 / 가용 99 = 1.01장 → 2장.

        사용자당 속도 SLA 는 만족시켜 두고(그래야 지연 경로에서 먼저 걸리지 않는다),
        이용률로만 소수 카드를 만든다.
        """
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0, per_user=100.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=100.0,
                        target_utilization=0.99))
        assert r.n_cards_by_throughput == 2

    def test_exact_fit_does_not_round_up(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=100.0))
        assert r.n_cards_by_throughput == 1

    def test_never_returns_zero_cards(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=0.1))
        assert r.n_cards >= 1


class TestLatencyIsIndependentOfThroughput:
    """처리량만 만족하면 '충분' 이라고 하지 않는다.

    기획서가 명시적으로 요구한 항목이다.
    """

    def test_sla_failure_at_every_condition_is_infeasible(self, store_single):
        r = plan(store_single, req(target_max_ttft_ms=10.0))   # 어떤 조건도 못 지킴
        assert r.feasible is False
        assert r.n_cards is None
        assert r.binding_constraint == "latency_sla_infeasible"
        assert any("SLA" in w for w in r.warnings)

    def test_latency_can_force_more_cards_than_throughput(self):
        """처리량은 1장으로 충분하지만 지연 때문에 나눠야 하는 경우."""
        s = BenchmarkStore([
            row(1, 1000.0, ttft_p95=50.0),
            row(8, 8000.0, ttft_p95=9000.0),      # 처리량은 좋지만 지연이 나쁨
        ])
        r = plan(s, req(concurrent_users=8, target_output_tps_per_user=1.0,
                        target_max_ttft_ms=100.0))
        assert r.n_cards_by_latency_sla == 8      # 카드당 1명만 가능
        assert r.binding_constraint == "latency_sla"
        assert r.n_cards == 8

    def test_throughput_can_be_the_binding_constraint(self, store_single):
        """지연 SLA 는 1장으로 충분한데 처리량 여유분 때문에 2장이 되는 경우.

        사용자당 80 tok/s 는 conc=8 까지 만족하므로 지연 기준으로는 1장이면 된다.
        이용률 90% 를 적용하면 카드당 가용이 576 tok/s 로 줄어 필요 640 에 모자란다.
        """
        r = plan(store_single, req(concurrent_users=8, target_output_tps_per_user=80.0,
                                   target_max_ttft_ms=100000.0, target_utilization=0.9))
        assert r.n_cards_by_latency_sla == 1
        assert r.n_cards_by_throughput == 2
        assert r.binding_constraint == "throughput"

    def test_final_answer_is_max_of_both(self, store_single):
        r = plan(store_single, req(concurrent_users=40, target_output_tps_per_user=20.0))
        assert r.n_cards == max(r.n_cards_by_latency_sla, r.n_cards_by_throughput)

    def test_per_user_speed_is_checked_too(self):
        """TTFT 만 보고 통과시키면 안 된다. 사용자당 생성 속도도 SLA 다."""
        s = BenchmarkStore([row(4, 40.0, ttft_p95=10.0, per_user=10.0)])
        checks = sla_checks(s.rows[0], req(target_output_tps_per_user=30.0,
                                           target_max_ttft_ms=1000.0))
        speed = [c for c in checks if "출력 속도" in c.name][0]
        assert speed.passed is False

    def test_e2e_p95_checked_when_requested(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0, e2e_p95=5000.0)])
        checks = sla_checks(s.rows[0], req(target_p95_e2e_ms=1000.0))
        assert any(c.name.startswith("요청 완료") and not c.passed for c in checks)

    def test_missing_p95_does_not_silently_pass(self):
        """표본 부족으로 P95 가 없으면 통과시키지 않는다."""
        s = BenchmarkStore([row(1, 100.0)])
        s.rows[0].ttft_ms_p95 = None
        checks = sla_checks(s.rows[0], req())
        ttft = [c for c in checks if c.name == "TTFT p95"][0]
        assert ttft.passed is False and "표본" in ttft.note


class TestFixedPoint:
    """카드 수와 카드당 동시성이 서로를 결정하는 순환을 푼다."""

    def test_iterations_are_recorded_for_audit(self, store_single):
        r = plan(store_single, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert r.iterations
        for it in r.iterations:
            assert {"n_cards", "concurrency_per_card", "usable_tps_per_card",
                    "n_cards_next"} <= set(it)

    def test_converges(self, store_single):
        r = plan(store_single, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert r.iterations[-1]["n_cards"] == r.iterations[-1]["n_cards_next"]

    def test_concurrency_per_card_consistent_with_card_count(self, store_single):
        r = plan(store_single, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert r.concurrency_per_card == math.ceil(100 / r.n_cards)


class TestScalingBasis:
    def test_linear_assumption_is_disclosed(self, store_single):
        r = plan(store_single, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert r.scaling_basis == "linear_assumption"
        assert any("가정" in w for w in r.warnings)

    def test_measured_curve_is_used_when_available(self, store_multicard):
        r = plan(store_multicard, req(concurrent_users=100, target_output_tps_per_user=10.0))
        assert r.scaling_basis == "measured_curve"
        assert not any("비례해 성능이 는다고 가정" in w for w in r.warnings)


class TestOperatingPoint:
    def test_picks_max_throughput_not_max_concurrency(self):
        """동시성을 더 올리면 처리량이 떨어지는 구간이 실재한다.

        실측: 카드당 동시성 128 이 1,459 tok/s, 256 은 1,310 tok/s.
        '동시성 최대' 를 고르면 더 나쁜 지점을 운영점으로 삼게 된다.
        """
        curve = [row(64, 1200.0, ttft_p95=300.0),
                 row(128, 1459.0, ttft_p95=400.0),
                 row(256, 1310.0, ttft_p95=500.0)]
        r = req(target_max_ttft_ms=10000.0, target_output_tps_per_user=1.0)
        assert best_operating_point(curve, r).concurrency_per_card == 128
        assert max_concurrency_meeting_sla(curve, r) == 256

    def test_returns_none_when_nothing_meets_sla(self, simple_curve):
        assert best_operating_point(simple_curve, req(target_max_ttft_ms=1.0)) is None

    def test_grid_ceiling_is_warned(self, store_single):
        """실측 격자 끝에서 SLA 를 만족하면, 더 높은 동시성은 확인된 바 없다."""
        r = plan(store_single, req(concurrent_users=100, target_max_ttft_ms=100000.0,
                                   target_output_tps_per_user=1.0))
        assert any("격자의 상한" in w for w in r.warnings)


class TestBottleneck:
    def test_kv_cache_saturation(self, simple_curve):
        assert diagnose_bottleneck(row(8, 640.0, kv=0.95), simple_curve) == "memory_capacity"

    def test_queue_buildup(self, simple_curve):
        assert diagnose_bottleneck(row(8, 640.0, waiting=20.0), simple_curve) == "concurrency_scheduling"

    def test_headroom_when_still_scaling(self):
        curve = [row(4, 400.0), row(8, 800.0)]
        assert diagnose_bottleneck(curve[0], curve) == "headroom"

    def test_saturated_when_more_concurrency_adds_nothing(self):
        curve = [row(4, 400.0), row(8, 405.0)]
        assert diagnose_bottleneck(curve[0], curve) == "throughput_saturated"

    def test_unknown_at_the_end_of_the_grid(self):
        curve = [row(4, 400.0)]
        assert diagnose_bottleneck(curve[0], curve) == "unknown"


class TestTradeoffs:
    def test_relaxing_sla_saves_cards(self, store_single):
        r = plan(store_single, req(concurrent_users=64, target_output_tps_per_user=5.0,
                                   target_max_ttft_ms=150.0))
        assert r.sla_tradeoffs
        assert any(t.cards_saved > 0 for t in r.sla_tradeoffs)

    def test_tradeoff_never_claims_more_cards_than_baseline(self, store_single):
        r = plan(store_single, req(concurrent_users=64, target_output_tps_per_user=5.0,
                                   target_max_ttft_ms=150.0))
        for t in r.sla_tradeoffs:
            assert t.n_cards <= r.n_cards


class TestProvenance:
    def test_evidence_names_the_source_measurement(self, store_single):
        r = plan(store_single, req())
        assert r.evidence["run_id"] and r.evidence["source"] == "measured_local"

    def test_rejected_sources_are_disclosed(self, simple_curve):
        s = BenchmarkStore(simple_curve + [row(1, 9999.0, source="hosted_endpoint")])
        r = plan(s, req())
        # 개수와 사유가 둘 다 보여야 한다. 문구 자체가 아니라 그 두 가지를 고정한다.
        assert any("1개 측정을 제외했습니다" in w and "호스팅" in w for w in r.warnings)

    def test_self_check_rejections_are_disclosed_too(self, simple_curve):
        """출처가 아닌 이유로 빠진 측정도 결과에 드러나야 한다."""
        s = BenchmarkStore(simple_curve + [row(1, 9999.0, capacity_usable=False,
                                               capacity_blocks=["prefix cache 오염"])])
        r = plan(s, req())
        assert any("prefix cache 오염" in w for w in r.warnings)

    def test_mock_data_is_loudly_flagged(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0, source="mock")], allow_mock=True)
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=1.0))
        assert r.data_source == "mock"
        assert any("MOCK" in w for w in r.warnings)

    def test_short_window_evidence_is_flagged(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0, window_s=5.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=1.0))
        assert any("측정창" in w for w in r.warnings)

    def test_concurrent_run_evidence_is_flagged(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0, exclusive=False)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=1.0))
        assert any("동시에 실행" in w for w in r.warnings)

    def test_small_sample_is_flagged(self):
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0, n_samples=12)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=1.0))
        assert any("표본" in w for w in r.warnings)

    def test_operating_point_on_the_grid_is_measured(self, store_single):
        """입력·출력·동시성이 모두 실측 격자와 일치하면 신뢰도는 measured 다."""
        r = plan(store_single, req(concurrent_users=4, target_output_tps_per_user=10.0,
                                   target_max_ttft_ms=100000.0))
        assert r.concurrency_per_card == 4
        assert r.confidence == "measured"

    def test_operating_point_between_grid_points_is_interpolated(self, store_single):
        """동시성이 격자 사이에 떨어지면 보간으로 낮춰 표시한다."""
        r = plan(store_single, req(concurrent_users=6, target_output_tps_per_user=10.0,
                                   target_max_ttft_ms=100000.0))
        assert r.concurrency_per_card == 6
        assert r.confidence == "interpolated"

    def test_input_length_outside_measured_range_is_extrapolated(self, store_single):
        r = plan(store_single, req(concurrent_users=4, target_output_tps_per_user=10.0,
                                   target_max_ttft_ms=100000.0, avg_input_tokens=8192))
        assert r.confidence == "extrapolated"


class TestNonConvergence:
    """카드를 늘려도 처리량이 안 느는 경우가 수학적으로 존재한다.

    카드당 처리량이 동시성에 **정확히 비례**하면, 사용자를 N장에 나눠도
    총 처리량은 그대로다: N × k·(users/N) = k·users.
    이때 고정점 반복은 수렴하지 않는다. 조용히 틀린 답을 내지 않고
    보수적인 값과 경고를 내야 한다.
    """

    def test_perfectly_linear_curve_does_not_hang_or_lie(self):
        curve = [row(c, 100.0 * c, ttft_p95=10.0, per_user=100.0)
                 for c in (1, 2, 4, 8, 16, 32)]
        s = BenchmarkStore(curve)
        r = plan(s, req(concurrent_users=32, target_output_tps_per_user=100.0,
                        target_max_ttft_ms=1000.0, target_utilization=0.5))
        assert r.n_cards is not None and r.n_cards >= 1
        assert len(r.iterations) <= 20            # 무한 루프에 빠지지 않는다
        if r.iterations[-1]["n_cards"] != r.iterations[-1]["n_cards_next"]:
            assert any("수렴" in w for w in r.warnings)


class TestInterpolationCarriesProvenance:
    """보간한 행도 자기 신뢰도를 들고 있어야 한다.

    근거에 측정창·단독여부가 비어 있으면 경고 로직이 조용히 통과한다.
    """

    def test_window_and_exclusivity_survive_interpolation(self):
        s = BenchmarkStore([
            row(4, 400.0, ttft_p95=10.0, window_s=90.0, exclusive=True),
            row(16, 800.0, ttft_p95=10.0, window_s=40.0, exclusive=False),
        ])
        curve = s.concurrency_curve(MODEL, 512, 128)
        from planner.capacity import _interp_rows
        mid, level = _interp_rows(curve, 8)
        assert level == "interpolated"
        assert mid.window_s == 40.0          # 나쁜 쪽을 따른다
        assert mid.exclusive is False        # 하나라도 동시 실행이면 동시 실행

    def test_interpolated_evidence_triggers_warnings(self):
        s = BenchmarkStore([
            row(4, 400.0, ttft_p95=10.0, window_s=90.0, exclusive=True),
            row(16, 800.0, ttft_p95=10.0, window_s=40.0, exclusive=False),
        ])
        r = plan(s, req(concurrent_users=8, target_output_tps_per_user=10.0,
                        target_max_ttft_ms=1000.0))
        assert any("동시에 실행" in w for w in r.warnings)


class TestTradeoffLimitReason:
    """SLA 를 완화하면 다른 SLA 가 한계가 된다.

    이유를 안 보여주면 "2배 완화나 4배 완화나 결과가 같다" 가 설명되지 않는다.
    """

    def _curve(self):
        # 동시성이 오를수록 TTFT 는 나빠지고 사용자당 속도는 떨어진다
        return [row(4, 400.0, ttft_p95=100.0, per_user=100.0),
                row(8, 700.0, ttft_p95=400.0, per_user=87.5),
                row(16, 900.0, ttft_p95=3000.0, per_user=56.0)]

    def test_reports_which_sla_blocks_the_next_step(self):
        from planner.capacity import _next_limit
        r = req(target_max_ttft_ms=500.0, target_output_tps_per_user=50.0)
        # conc=8 다음은 conc=16 — TTFT 3000 > 500 이라 TTFT 가 먼저 걸린다
        assert _next_limit(self._curve(), r, 8) == ["TTFT p95"]

    def test_reports_speed_when_latency_is_relaxed(self):
        from planner.capacity import _next_limit
        r = req(target_max_ttft_ms=9000.0, target_output_tps_per_user=60.0)
        # TTFT 를 풀면 이제 사용자당 속도(56 < 60)가 한계다
        assert _next_limit(self._curve(), r, 8) == ["사용자당 출력 속도"]

    def test_reports_grid_end_when_no_higher_measurement(self):
        from planner.capacity import _next_limit
        r = req(target_max_ttft_ms=9000.0, target_output_tps_per_user=1.0)
        assert _next_limit(self._curve(), r, 16) == ["measurement_grid"]

    def test_tradeoffs_carry_the_reason(self, store_single):
        r = plan(store_single, req(concurrent_users=64, target_output_tps_per_user=5.0,
                                   target_max_ttft_ms=150.0))
        assert r.sla_tradeoffs
        assert all(t.limited_by and t.limited_by != ["unknown"] for t in r.sla_tradeoffs)

    def test_reports_every_constraint_that_blocks(self):
        """다음 지점에서 여러 SLA 가 동시에 막으면 전부 알려야 한다.

        교차검증 F5: 하나만 내면 "그것만 더 풀면 된다" 로 읽힌다. 실제 데이터에서
        TTFT 를 2배 완화한 지점의 conc=128 은 TTFT 와 출력 속도가 함께 막는데,
        TTFT 만 보고하면 4배로 풀면 될 것처럼 보인다.
        """
        from planner.capacity import _next_limit
        r = req(target_max_ttft_ms=500.0, target_output_tps_per_user=70.0)
        # conc=16 은 TTFT 3000>500 이고 속도 56<70 이라 둘 다 실패한다
        assert _next_limit(self._curve(), r, 8) == ["TTFT p95", "사용자당 출력 속도"]


class TestNonConvergenceReporting:
    """수렴하지 않았을 때 무엇을 말하는가.

    확장 효율을 최근접 격자로 스냅하면 경계에 불연속이 생기고 고정점이 그 위에서
    진동한다. 실측에서 카드당 48↔49 의 효율 1.0093↔0.8034 를 타고 카드 수가
    20↔25 를 무한 반복했다. 그때 "가장 큰 값을 보수적으로 채택" 이라고만 쓰면
    읽는 사람이 그 값을 답으로 받는다.
    """

    def test_seen_does_not_include_the_seed(self):
        """초기값 1 은 후보였던 적이 없다. 넣으면 범위가 항상 1~N 으로 나온다."""
        s = BenchmarkStore([row(1, 100.0, ttft_p95=10.0)])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=1.0))
        for w in r.warnings:
            assert "범위 1~" not in w or r.n_cards == 1

    def test_non_convergence_says_it_did_not_converge(self):
        """경고가 '보수적으로 채택' 이 아니라 '수렴하지 않았다' 를 먼저 말해야 한다."""
        import planner.capacity as cap
        rows = [row(c, tps, ttft_p95=10.0) for c, tps in
                [(1, 100.0), (2, 190.0), (4, 360.0), (8, 640.0), (16, 800.0), (32, 900.0)]]
        s = BenchmarkStore(rows)
        # 효율이 카드당 동시성에 따라 계단처럼 뛰도록 강제해 진동을 만든다
        orig = s.scaling_efficiency
        s.scaling_efficiency = lambda m, n, c=None: ((1.0, "x") if (c or 0) <= 24
                                                     else (0.5, "x"))
        r = plan(s, req(concurrent_users=1000, target_output_tps_per_user=10.0,
                        target_utilization=0.7))
        s.scaling_efficiency = orig
        warn = [w for w in r.warnings if "수렴" in w]
        if warn:                      # 진동이 실제로 났을 때만 검사한다
            assert "수렴하지 않았습니다" in warn[0]
            assert "근거가 없습니다" in warn[0]


class TestUntrustworthyPercentilesAreDisclosed:
    """백분위를 못 믿는 행이 근거에 있으면 결과에 드러나야 한다.

    `untrustworthy_percentile_rows` 는 정의도 테스트도 있는데 `plan()` 이
    호출하지 않아 경고가 한 줄도 안 나갔다. 판정을 만들어놓고 소비자가 없는
    구조는 오염 판정(X9)에서도 똑같이 있었다.
    """

    def test_warning_appears(self):
        bad = row(1, 100.0, ttft_p95=10.0, n_samples=5)
        bad.warmup_requests = 0
        s = BenchmarkStore([bad])
        r = plan(s, req(concurrent_users=1, target_output_tps_per_user=1.0))
        assert any("백분위를 믿을 수 없는" in w for w in r.warnings)


class TestTradeoffsUseTheSameSolver:
    """완화 시나리오의 답이 그 조건을 다시 푼 답과 같아야 한다.

    예전에는 `_tradeoffs` 가 고정점을 안 돌리고 1패스로 계산했다. 격자 전수 비교에서
    절반 이상이 어긋났고 거의 전부 **과소** 보고였다 — "완화하면 6장 절약" 이라고
    출력하는데 다시 풀면 절약이 0장인 식이다. 도구가 스스로와 모순되면 어느 쪽도
    믿을 수 없다.
    """

    def _curve(self):
        return [row(c, tps, ttft_p95=t) for c, tps, t in
                [(1, 100.0, 80.0), (2, 190.0, 90.0), (4, 360.0, 120.0),
                 (8, 640.0, 200.0), (16, 800.0, 900.0), (32, 900.0, 6000.0)]]

    def test_every_tradeoff_matches_a_full_replan(self):
        s = BenchmarkStore(self._curve())
        base = req(concurrent_users=200, target_output_tps_per_user=20.0,
                   target_max_ttft_ms=500.0, target_utilization=0.7)
        r = plan(s, base)
        assert r.sla_tradeoffs, "완화 시나리오가 하나도 안 나왔다"
        for t in r.sla_tradeoffs:
            kw = dict(base.__dict__)
            if "2배" in t.relaxed:
                kw["target_max_ttft_ms"] = base.target_max_ttft_ms * 2
            elif "4배" in t.relaxed:
                kw["target_max_ttft_ms"] = base.target_max_ttft_ms * 4
            else:
                kw["target_output_tps_per_user"] = base.target_output_tps_per_user * 0.7
            again = plan(s, ServiceRequirement(**kw))
            assert again.n_cards == t.n_cards, (t.relaxed, t.n_cards, again.n_cards)


class TestUtilizationTerminationIsBySatisfaction:
    """이용률 보정의 종료 조건은 **반복 횟수가 아니라 충족 여부**다.

    한 번에 한 장씩 올리며 20회로 끊으면 사용자 수가 클수록 부족분이 커진다.
    실제로 1,000명에서는 69.95% 였는데 3,000명에서 73.31% 를 내면서도
    `feasible=True` 를 반환했다 — 요청한 여유를 못 주면서 "가능" 이라고 말했다.
    """

    def _store(self):
        return BenchmarkStore([row(c, tps, ttft_p95=10.0) for c, tps in
                               [(1, 100.0), (2, 190.0), (4, 360.0), (8, 640.0),
                                (16, 800.0), (32, 900.0)]])

    @pytest.mark.parametrize("users", [100, 500, 1000, 3000, 10000])
    def test_target_utilization_is_respected_at_every_scale(self, users):
        s = self._store()
        r = plan(s, req(concurrent_users=users, target_output_tps_per_user=10.0,
                        target_utilization=0.7))
        if r.estimated_utilization is None:
            return
        if r.feasible:
            assert r.estimated_utilization <= 0.7 + 1e-9, (users, r.estimated_utilization)
        else:
            # 못 지킬 때는 feasible 을 내리고 그 사실을 경고에 쓴다
            assert r.binding_constraint == "target_utilization_unreachable"
            assert any("목표 이용률" in w for w in r.warnings)

    def test_unreachable_target_is_not_reported_as_feasible(self):
        """카드를 늘려도 수렴하지 않으면 '가능' 이라고 하면 안 된다."""
        s = self._store()
        r = plan(s, req(concurrent_users=1000, target_output_tps_per_user=10.0,
                        target_utilization=0.01))
        if r.estimated_utilization and r.estimated_utilization > 0.01 + 1e-9:
            assert r.feasible is False
