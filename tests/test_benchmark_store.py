"""실측 데이터 로드·선택 로직.

여기서 틀리면 planner 가 **잘못된 근거로 옳은 계산**을 한다. 값이 그럴듯해 보여
사후에 발견하기 어려운 종류다. 실제로 눈으로만 확인하다 3건을 놓쳤다.
"""

import pytest

from planner.benchmark_store import (
    BenchmarkStore, DataSourceError, PREFILL_BUCKET_EDGES, _crosses_bucket_edge,
)
from tests.conftest import MODEL, row


class TestSourceGating:
    """호스팅 측정이 capacity 계산에 새어 들어가면 안 된다.

    카드 수를 알 수 없고 멀티테넌트이며 네트워크가 포함돼 있다. 사람이 실수로
    섞는 걸 막기 위해 로드 단계에서 거부한다.
    """

    def test_measured_local_accepted(self):
        s = BenchmarkStore([row(1, 100.0)])
        assert len(s.rows) == 1 and not s.rejected

    def test_hosted_rejected_with_reason(self):
        s = BenchmarkStore([row(1, 100.0, source="hosted_endpoint")])
        assert not s.rows
        assert len(s.rejected) == 1
        assert "호스팅" in s.rejected[0]["reason"]

    def test_mock_rejected_unless_explicitly_allowed(self):
        s = BenchmarkStore([row(1, 100.0, source="mock")])
        assert not s.rows and s.rejected

    def test_mock_accepted_when_allowed(self):
        s = BenchmarkStore([row(1, 100.0, source="mock")], allow_mock=True)
        assert len(s.rows) == 1

    def test_unknown_source_rejected(self):
        s = BenchmarkStore([row(1, 100.0, source="who_knows")])
        assert not s.rows

    def test_self_check_failure_rejected_even_when_source_is_local(self):
        """출처가 실측이어도 하네스가 오염을 보고했으면 거부한다.

        이게 없으면 "적중 시 자동 무효 처리" 라는 서술이 서류상으로만 참이 된다.
        """
        s = BenchmarkStore([row(1, 100.0, capacity_usable=False,
                                capacity_blocks=["prefix cache 오염 — 적중 비율 최대 0.99"])])
        assert not s.rows and len(s.rejected) == 1
        assert "prefix cache" in s.rejected[0]["reason"]

    def test_hosted_row_is_rejected_as_hosted_not_as_self_check(self):
        """호스팅 행은 오염 판정도 같이 달고 오는데, 사유는 출처여야 한다.

        사유가 틀리면 사람이 엉뚱한 데를 고친다 (F2 에서 실제로 겪었다).
        """
        s = BenchmarkStore([row(1, 100.0, source="hosted_endpoint",
                                capacity_usable=False, capacity_blocks=["prefix cache 오염"])])
        assert "호스팅" in s.rejected[0]["reason"]

    def test_allowed_mock_still_obeys_self_check(self):
        s = BenchmarkStore([row(1, 100.0, source="mock", capacity_usable=False,
                                capacity_blocks=["출력 길이가 고정되지 않았습니다"])],
                           allow_mock=True)
        assert not s.rows and s.rejected

    def test_rejection_records_a_reason_even_without_blocks(self):
        """사유가 비어 있어도 조용히 사라지면 안 된다."""
        s = BenchmarkStore([row(1, 100.0, capacity_usable=False)])
        assert s.rejected[0]["reason"].strip()

    def test_data_source_reports_mock_so_ui_can_banner_it(self):
        assert BenchmarkStore([row(1, 100.0, source="mock")], allow_mock=True).data_source == "mock"

    def test_data_source_flags_mixture(self):
        s = BenchmarkStore([row(1, 100.0), row(2, 200.0, source="mock")], allow_mock=True)
        assert s.data_source == "mixed"

    def test_empty_store(self):
        assert BenchmarkStore([]).data_source == "empty"


class TestMeasurementQuality:
    """같은 조건이 여러 번 측정됐을 때 어느 것을 믿을 것인가.

    실측에서 배운 순서: 신뢰 가능한 측정창 → 단독 실행 → 창 길이.
    """

    def test_prefers_longer_window(self):
        s = BenchmarkStore([
            row(4, 300.0, window_s=20.0, run_id="short"),
            row(4, 360.0, window_s=90.0, run_id="long"),
        ])
        c = s.concurrency_curve(MODEL, 512, 128)
        assert len(c) == 1 and c[0].run_id == "long"

    def test_prefers_exclusive_over_concurrent(self):
        s = BenchmarkStore([
            row(4, 300.0, window_s=90.0, exclusive=False, run_id="shared"),
            row(4, 360.0, window_s=40.0, exclusive=True, run_id="alone"),
        ])
        assert s.concurrency_curve(MODEL, 512, 128)[0].run_id == "alone"

    def test_untrusted_window_loses_even_when_exclusive(self):
        """짧은 창의 램프업 오염은 단독 실행이어도 회복되지 않는다.

        실측: 같은 조건이 창 12초에서 TTFT p95 3,070ms, 63초에서 261ms.
        """
        s = BenchmarkStore([
            row(4, 360.0, window_s=12.0, exclusive=True, ttft_p95=3070.0, run_id="short_alone"),
            row(4, 340.0, window_s=63.0, exclusive=False, ttft_p95=261.0, run_id="long_shared"),
        ])
        assert s.concurrency_curve(MODEL, 512, 128)[0].run_id == "long_shared"

    def test_short_window_rows_are_reported(self):
        s = BenchmarkStore([row(1, 100.0, window_s=5.0), row(2, 200.0, window_s=90.0)])
        c = s.concurrency_curve(MODEL, 512, 128)
        assert [r.concurrency_per_card for r in s.short_window_rows(c)] == [1]

    def test_curve_is_sorted_by_concurrency(self, simple_curve):
        s = BenchmarkStore(list(reversed(simple_curve)))
        c = s.concurrency_curve(MODEL, 512, 128)
        assert [r.concurrency_per_card for r in c] == sorted(r.concurrency_per_card for r in c)


class TestBucketBoundary:
    """RNGD 는 컴파일된 shape 단위로 실행된다.

    실측에서 프롬프트 1,003 토큰은 77ms, 1,037 토큰은 96ms 였다. 경계를 넘는
    선형 보간은 실제와 다르므로 신뢰도를 따로 표시해야 한다.
    """

    def test_edges_match_artifact(self):
        assert PREFILL_BUCKET_EDGES[0] == 128 and PREFILL_BUCKET_EDGES[-1] == 1024

    @pytest.mark.parametrize("a,b,expected", [
        (130, 200, False),   # 같은 구간 안
        (200, 300, True),    # 256 을 넘음
        (900, 1100, True),   # 1024 를 넘음
        (1100, 1200, False), # 선언된 경계 위 (아티팩트 기준)
        # 경계값은 **직전 구간의 마지막 토큰**이다. 128 tok 은 a128 버킷에 맞고
        # 129 tok 부터 다음 버킷이므로 [128, 256] 은 실제로 계단을 지난다.
        # 예전 기대값(False)은 `lo < edge` 라는 잘못된 규약을 담고 있었다.
        (128, 256, True),
        (129, 256, False),   # 계단을 이미 지난 뒤에서 시작하면 안 넘는다
    ])
    def test_crossing_detection(self, a, b, expected):
        assert _crosses_bucket_edge(a, b) is expected

    def test_interval_starting_exactly_on_an_edge_is_caught(self):
        """교차검증 X2: lo < edge 로 두면 1024→2048 이 1024/1025 계단을 놓친다."""
        assert _crosses_bucket_edge(1024, 2048, "furiosa-ai/Qwen3-8B-FP8") is True

    def test_interpolation_across_boundary_is_flagged(self):
        s = BenchmarkStore([row(1, 100.0, input_tokens=200), row(1, 90.0, input_tokens=300)])
        _, level, notes = s.resolve_input_slice(MODEL, 260, 128)
        assert level == "interpolated_across_bucket_boundary"
        assert any("버킷 경계" in n for n in notes)

    def test_interpolation_within_bucket_is_plain(self):
        s = BenchmarkStore([row(1, 100.0, input_tokens=300), row(1, 95.0, input_tokens=380)])
        _, level, _ = s.resolve_input_slice(MODEL, 340, 128)
        assert level == "interpolated"

    def test_just_over_boundary_gets_actionable_advice(self):
        """'조금 줄이면 같은 비용' 은 사용자가 바로 행동할 수 있는 정보다."""
        s = BenchmarkStore([row(1, 100.0, input_tokens=1024), row(1, 90.0, input_tokens=1280)])
        _, _, notes = s.resolve_input_slice(MODEL, 1030, 128)
        assert any("1024" in n and "줄이면" in n for n in notes)


class TestInputSlice:
    def test_exact_match_is_measured(self):
        s = BenchmarkStore([row(1, 100.0, input_tokens=512)])
        _, level, notes = s.resolve_input_slice(MODEL, 512, 128)
        assert level == "measured" and not notes

    def test_outside_range_is_extrapolated_with_warning(self):
        s = BenchmarkStore([row(1, 100.0, input_tokens=512)])
        _, level, notes = s.resolve_input_slice(MODEL, 4096, 128)
        assert level == "extrapolated" and any("범위" in n for n in notes)

    def test_missing_model_raises(self):
        s = BenchmarkStore([row(1, 100.0)])
        with pytest.raises(DataSourceError):
            s.resolve_input_slice("no/such-model", 512, 128)

    def test_interpolated_row_carries_both_sources(self):
        s = BenchmarkStore([row(1, 100.0, input_tokens=300, run_id="a"),
                            row(1, 80.0, input_tokens=380, run_id="b")])
        curve, _, _ = s.resolve_input_slice(MODEL, 340, 128)
        assert "a" in curve[0].run_id and "b" in curve[0].run_id
        assert 80.0 < curve[0].aggregate_output_tps < 100.0


class TestScalingEfficiency:
    def test_linear_assumption_when_no_multicard_data(self, store_single):
        eff, basis = store_single.scaling_efficiency(MODEL, 4)
        assert eff == 1.0 and basis == "linear_assumption"

    def test_no_concurrency_given_is_labelled_as_fallback(self, store_multicard):
        """카드당 동시성을 안 주면 동시성 축을 통째로 무시한 값이 나온다.

        효율은 카드 수만이 아니라 카드당 동시성에도 의존하므로, 그 축을 못 맞춘
        결과를 `measured_curve` 로 표시하면 안 된다 — 매칭 실패가 사용자에게
        안 보인다. 실제로 그래서 카드당 4 에서 잰 효율이 카드당 50 운영점에 쓰였다.
        """
        eff, basis = store_multicard.scaling_efficiency(MODEL, 2)
        assert basis == "all_conditions_fallback"
        assert eff == pytest.approx(1.0, abs=1e-6)

    def test_exact_grid_point_is_measured_curve(self, store_multicard):
        eff, basis = store_multicard.scaling_efficiency(MODEL, 2, 8)
        assert basis == "measured_curve" and eff == pytest.approx(1.0, abs=1e-6)

    def test_between_grid_points_is_interpolated_not_snapped(self):
        """격자 사이에서는 **보간**한다. 스냅은 경계에 불연속을 만든다.

        실측: 카드당 48 -> 1.0093, 49 -> 0.8034 (20.6%p 점프). 고정점이 그 위에서
        20 <-> 25 로 무한 진동했다. 보간하면 연속이라 수렴한다.
        """
        rows = [row(16, 100.0), row(64, 100.0),
                row(16, 200.0, n_cards=2, run_id="m16"),   # 카드당16 -> 8장 100%
                row(64, 100.0, n_cards=2, run_id="m64")]   # 카드당64 -> 8장  50%
        s = BenchmarkStore(rows)
        e16, b16 = s.scaling_efficiency(MODEL, 2, 16)
        e64, _ = s.scaling_efficiency(MODEL, 2, 64)
        e40, b40 = s.scaling_efficiency(MODEL, 2, 40)      # 정확히 중간
        assert b16 == "measured_curve"
        assert e16 == pytest.approx(1.0) and e64 == pytest.approx(0.5)
        assert e40 == pytest.approx(0.75, abs=1e-6)        # 스냅이면 1.0 이나 0.5 가 나온다
        assert "interpolated_concurrency" in b40

    def test_outside_grid_is_flagged(self):
        """격자 밖은 마지막 값을 유지하되 **그 사실을 basis 로 드러낸다.**

        하강 추세에서 마지막 값을 유지하는 것은 보수적이 아니라 낙관적이다.
        배치 등급 운영점(카드당 167)이 격자 상한의 몇 배 밖인데 지금은 세 등급이
        같은 신뢰도로 보인다.
        """
        rows = [row(16, 100.0), row(32, 100.0),
                row(16, 160.0, n_cards=2, run_id="m16"),
                row(32, 160.0, n_cards=2, run_id="m32")]
        s = BenchmarkStore(rows)
        eff, basis = s.scaling_efficiency(MODEL, 2, 200)
        assert eff == pytest.approx(0.8)
        assert basis.startswith("held_above_grid")

    def test_grid_uses_intersection_not_single_card_concurrencies(self):
        """격자는 1장 곡선이 아니라 **비교쌍이 있는 동시성**이다.

        1장에만 있는 동시성을 고르면 그 조건에 다중 카드 자료가 없어 통째로
        폴백된다. 실측에서 1장은 1~256 인데 다중 카드는 4·16·32 뿐이었다.
        """
        rows = [row(16, 100.0), row(64, 100.0), row(256, 100.0),
                row(16, 200.0, n_cards=2, run_id="m16")]
        s = BenchmarkStore(rows)
        assert s._comparable_concurrencies(MODEL) == [16]

    def test_single_card_is_always_unity(self, store_multicard):
        assert store_multicard.scaling_efficiency(MODEL, 1)[0] == 1.0

    def test_efficiency_depends_on_concurrency(self):
        """실측에서 카드당 동시성 16은 8장까지 100.8%, 32는 82.1% 였다.

        전 조건을 평균내면 낮은 동시성 운영점의 카드 수가 과대 추정된다.
        이 버그로 200명 시나리오가 21장(실제 15장)으로 나왔다.
        """
        rows = [
            row(4, 100.0), row(32, 400.0),
            row(4, 780.0, n_cards=8),    # 97.5% 효율
            row(32, 2560.0, n_cards=8),  # 80.0% 효율
        ]
        s = BenchmarkStore(rows)
        assert s.scaling_efficiency(MODEL, 8, 4)[0] == pytest.approx(0.975, abs=1e-3)
        assert s.scaling_efficiency(MODEL, 8, 32)[0] == pytest.approx(0.800, abs=1e-3)

    def test_beyond_measured_holds_last_value_rather_than_extrapolating(self, store_multicard):
        """실측 밖으로 효율을 외삽하지 않는다. 마지막 실측치를 유지하는 편이 보수적이다."""
        at4, _ = store_multicard.scaling_efficiency(MODEL, 4)
        at64, _ = store_multicard.scaling_efficiency(MODEL, 64)
        assert at64 == pytest.approx(at4)

    def test_efficiency_prefers_longer_window(self):
        rows = [
            row(4, 100.0, window_s=90.0),
            row(4, 150.0, n_cards=2, window_s=5.0, run_id="short"),   # 75% (짧은 창)
            row(4, 200.0, n_cards=2, window_s=90.0, run_id="long"),   # 100%
        ]
        s = BenchmarkStore(rows)
        assert s.scaling_efficiency(MODEL, 2, 4)[0] == pytest.approx(1.0, abs=1e-6)


class TestOutputTokens:
    def test_exact_output_length(self):
        s = BenchmarkStore([row(1, 100.0, output_tokens=128)])
        assert s.nearest_output_tokens(MODEL, 128) == (128, "measured")

    def test_nearest_output_length_is_flagged(self):
        s = BenchmarkStore([row(1, 100.0, output_tokens=128), row(1, 90.0, output_tokens=512)])
        val, level = s.nearest_output_tokens(MODEL, 200)
        assert val == 128 and level == "interpolated"

    def test_no_data_raises(self):
        with pytest.raises(DataSourceError):
            BenchmarkStore([]).nearest_output_tokens(MODEL, 128)


class TestEfficiencySanityLimit:
    """확장 효율이 물리적 상한을 크게 넘으면 집계가 틀린 것이다.

    실제로 단독 실행 우선 규칙을 빼먹고 오염된 측정을 기준으로 잡았을 때 135% 가
    나온 적이 있다. 값 자체는 그럴듯해 보여서 눈으로는 안 걸린다.
    """

    def test_detected_at_load_not_at_query(self):
        """무결성 검사가 질의 시점에 있으면 무엇을 묻느냐에 따라 오염이 숨는다."""
        s = BenchmarkStore([row(4, 100.0), row(4, 300.0, n_cards=2)])   # 150%
        assert s.scaling_issues                       # 로드하자마자 잡힌다
        assert s.scaling_issues[0]["efficiency"] > 1.15

    def test_query_does_not_raise_but_falls_back(self):
        """예외로 전부 세우면 UI 가 트레이스백을 띄운다. 선형 가정으로 내려간다."""
        s = BenchmarkStore([row(4, 100.0), row(4, 300.0, n_cards=2)])
        eff, basis = s.scaling_efficiency(MODEL, 2, 4)
        assert eff == 1.0 and basis == "linear_assumption_scaling_data_rejected"

    def test_contamination_in_one_region_is_found_regardless_of_query(self):
        """정상 구간만 물어도 오염이 드러나야 한다.

        교차검증 F6: 카드당 4 는 98% 로 정상 반환되고 카드당 64 만 터졌다.
        아무도 64 를 묻지 않으면 오염이 계속 남는다.
        """
        s = BenchmarkStore([
            row(4, 100.0), row(4, 196.0, n_cards=2),        # 98% 정상
            row(64, 100.0), row(64, 300.0, n_cards=2),      # 150% 오염
        ])
        assert any(i["concurrency_per_card"] == 64 for i in s.scaling_issues)
        # 정상 구간 질의도 곡선을 쓰지 않는다 — 같은 모델의 데이터를 믿을 수 없기 때문
        assert s.scaling_efficiency(MODEL, 2, 4)[1] == "linear_assumption_scaling_data_rejected"

    def test_real_world_slight_overshoot_is_allowed(self):
        """실측에 102.4% 가 있다. 측정 산포이지 오류가 아니다."""
        s = BenchmarkStore([row(4, 100.0), row(4, 204.8, n_cards=2)])   # 102.4%
        assert not s.scaling_issues
        eff, basis = s.scaling_efficiency(MODEL, 2, 4)
        assert eff == pytest.approx(1.024, abs=1e-3) and basis == "measured_curve"


class TestPercentileTrustworthiness:
    """지연 백분위를 믿을 수 있는 조건.

    교차검증 F9: 처음에 쓴 "측정창 15초 이상" 은 원인이 아니라 대리지표였다.
    진짜 변수는 warm-up 이 동시성보다 적을 때 워커 일부의 첫 요청이 측정에 섞이는 것이다.
    그 요청 수는 고정(=동시성)이라 오래 측정하면 희석될 뿐이다.

    실측: B1 conc=32(warmup=8) p95 1,437ms → 앞 32건 제외 시 246ms (참값 252ms).
    앞 8건·16건 제외로는 복원되지 않고 정확히 동시성만큼 빼야 한다.
    """

    def _row(self, **kw):
        base = dict(concurrency=32, agg_tps=1000.0, n_samples=1000, window_s=60.0)
        base.update(kw)
        return row(base.pop("concurrency"), base.pop("agg_tps"), **base)

    def test_warmup_below_concurrency_is_untrusted(self):
        from planner.benchmark_store import percentiles_trustworthy
        r = self._row()
        r.warmup_requests = 8            # < 동시성 32
        assert percentiles_trustworthy(r) is False

    def test_warmup_at_least_concurrency_is_trusted(self):
        from planner.benchmark_store import percentiles_trustworthy
        r = self._row()
        r.warmup_requests = 32
        assert percentiles_trustworthy(r) is True

    def test_recomputed_rows_are_trusted_regardless_of_warmup(self):
        """앞 구간을 빼고 다시 계산했으면 warm-up 이 적었어도 쓸 수 있다."""
        from planner.benchmark_store import percentiles_trustworthy
        r = self._row()
        r.warmup_requests = 8
        r.percentile_source = "recomputed_ramp_excluded"
        assert percentiles_trustworthy(r) is True

    def test_shallow_samples_are_untrusted_even_with_enough_warmup(self):
        """표본이 얕으면 몇 건이 백분위를 흔든다."""
        from planner.benchmark_store import percentiles_trustworthy
        r = self._row(n_samples=100)     # 동시성 32 × 20 = 640 필요
        r.warmup_requests = 32
        assert percentiles_trustworthy(r) is False

    def test_unknown_warmup_defers_judgement(self):
        """옛 데이터는 warm-up 을 기록하지 않았다. 모르면 판단을 보류한다."""
        from planner.benchmark_store import percentiles_trustworthy
        r = self._row()
        r.warmup_requests = None
        assert percentiles_trustworthy(r) is True

    def test_trustworthy_row_wins_over_longer_window(self):
        """창 길이보다 백분위 신뢰도가 우선이다."""
        untrusted = self._row(window_s=200.0, run_id="long_but_untrusted")
        untrusted.warmup_requests = 8
        trusted = self._row(window_s=40.0, run_id="short_but_trusted")
        trusted.warmup_requests = 32
        s = BenchmarkStore([untrusted, trusted])
        assert s.concurrency_curve(MODEL, 512, 128)[0].run_id == "short_but_trusted"

    def test_untrustworthy_rows_are_listed(self):
        r = self._row()
        r.warmup_requests = 8
        s = BenchmarkStore([r])
        curve = s.concurrency_curve(MODEL, 512, 128)
        assert len(s.untrustworthy_percentile_rows(curve)) == 1


class TestMeasuredCostEdges:
    """아티팩트에 선언된 버킷과 **실제 비용 경계**는 다르다.

    교차검증 F7: A3+A4 요청 950건을 실제 토큰 수로 재정렬하니 계단이 128·256·1024·1280
    에만 있었다. 384·768·896 주변은 ±0.3% 로 평탄한데 아티팩트에는 버킷이 선언돼 있고,
    반대로 1280 은 +55% 계단인데 아티팩트 목록에 없다.
    """

    def test_measured_model_uses_measured_edges(self):
        from planner.benchmark_store import cost_edges
        assert cost_edges("furiosa-ai/Qwen3-8B-FP8") == (128, 256, 1024, 1280)

    def test_unmeasured_model_falls_back_to_artifact(self):
        from planner.benchmark_store import ARTIFACT_BUCKET_EDGES, cost_edges
        assert cost_edges("some/other-model") == ARTIFACT_BUCKET_EDGES

    @pytest.mark.parametrize("a,b", [(448, 576), (576, 704), (704, 832)])
    def test_flat_regions_are_no_longer_downgraded(self, a, b):
        """실측 변동이 2~3% 인 구간을 경계 넘김으로 강등하면 안 된다."""
        assert _crosses_bucket_edge(a, b, "furiosa-ai/Qwen3-8B-FP8") is False
        assert _crosses_bucket_edge(a, b) is True          # 아티팩트 기준으로는 넘김

    def test_1280_step_is_caught(self):
        """1280 은 +55% 계단인데 아티팩트 목록에 없어 놓치고 있었다."""
        assert _crosses_bucket_edge(1152, 1408, "furiosa-ai/Qwen3-8B-FP8") is True
        assert _crosses_bucket_edge(1152, 1408) is False

    def test_real_edges_still_caught(self):
        m = "furiosa-ai/Qwen3-8B-FP8"
        assert _crosses_bucket_edge(200, 300, m) is True      # 256
        assert _crosses_bucket_edge(1000, 1100, m) is True    # 1024

    def test_advice_points_at_a_real_edge(self):
        """'조금 줄이면 같은 비용' 은 실제 경계에서만 나와야 한다."""
        m = "furiosa-ai/Qwen3-8B-FP8"
        s = BenchmarkStore([row(1, 100.0, input_tokens=1024, model=m),
                            row(1, 90.0, input_tokens=1280, model=m)])
        _, _, notes = s.resolve_input_slice(m, 1030, 128)
        assert any("1024" in n and "줄이면" in n for n in notes)

    def test_no_advice_at_a_phantom_edge(self):
        """512 는 아티팩트에 있지만 실제 비용 경계가 아니다."""
        m = "furiosa-ai/Qwen3-8B-FP8"
        s = BenchmarkStore([row(1, 100.0, input_tokens=384, model=m),
                            row(1, 99.0, input_tokens=640, model=m)])
        _, level, notes = s.resolve_input_slice(m, 520, 128)
        assert level == "interpolated"                       # 강등되지 않는다
        assert not any("줄이면" in n for n in notes)
