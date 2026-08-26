"""혼합 부하 간섭 측정 (B3).

실측 데이터가 없는 신규 코드라 로직과 측정 설계를 테스트로 고정한다.
특히 '간섭' 의 정의 — 주입 전후 비교가 아니라 **같은 run 안에서 겹친 것과
안 겹친 것을 나눈다** — 가 제대로 구현됐는지.
"""

import pytest

from benchmark.run_mixed import MixedSpec, _summarize, annotate_overlap
from benchmark.schema import RequestRecord


def rec(*, role, t_send, t_last, ttft=100.0, tpot=10.0, tokens=128,
        warmup=False, error=None, cached=0, prompt=512):
    r = RequestRecord(run_id="r", request_id="q", experiment="B3", source="measured_local",
                      is_warmup=warmup, role=role)
    r.t_send, r.t_last_token = t_send, t_last
    r.t_first_token = t_send + ttft / 1000.0
    r.ttft_ms, r.tpot_ms = ttft, tpot
    r.e2e_ms = (t_last - t_send) * 1000.0
    r.completion_tokens_actual = tokens
    r.prompt_tokens_actual = prompt
    r.cached_tokens = cached
    r.error = error
    return r


class TestOverlapAnnotation:
    """겹침 판정이 간섭 정의의 핵심이다."""

    def test_background_inside_injection_is_marked(self):
        recs = [rec(role="injected", t_send=0.0, t_last=10.0),
                rec(role="background", t_send=2.0, t_last=4.0)]
        annotate_overlap(recs)
        assert recs[1].overlapped_injections == 1

    def test_background_outside_injection_is_clean(self):
        recs = [rec(role="injected", t_send=0.0, t_last=1.0),
                rec(role="background", t_send=5.0, t_last=6.0)]
        annotate_overlap(recs)
        assert recs[1].overlapped_injections == 0

    def test_partial_overlap_counts(self):
        """요청 생존 구간이 조금이라도 겹치면 영향을 받을 수 있다."""
        recs = [rec(role="injected", t_send=3.0, t_last=8.0),
                rec(role="background", t_send=5.0, t_last=12.0)]
        annotate_overlap(recs)
        assert recs[1].overlapped_injections == 1

    def test_touching_but_not_overlapping_is_clean(self):
        recs = [rec(role="injected", t_send=0.0, t_last=5.0),
                rec(role="background", t_send=5.0, t_last=9.0)]
        annotate_overlap(recs)
        assert recs[1].overlapped_injections == 0

    def test_counts_multiple_injections(self):
        recs = [rec(role="injected", t_send=0.0, t_last=6.0),
                rec(role="injected", t_send=3.0, t_last=9.0),
                rec(role="background", t_send=4.0, t_last=5.0)]
        annotate_overlap(recs)
        assert recs[2].overlapped_injections == 2

    def test_injected_requests_are_not_annotated(self):
        recs = [rec(role="injected", t_send=0.0, t_last=6.0),
                rec(role="injected", t_send=1.0, t_last=2.0)]
        annotate_overlap(recs)
        assert all(r.overlapped_injections == 0 for r in recs)


class TestAggregationSeparation:
    """배경 처리량에 주입 토큰을 섞으면 '긴 요청이 처리량을 올렸다' 는 착시가 생긴다."""

    def _spec(self, **kw):
        return MixedSpec(experiment="B3", min_samples_for_p95=1, **kw)

    def _target(self):
        from benchmark.target import Target
        return Target(name="t", base_url="http://h/v1", model="m", source="measured_local")

    def test_injected_tokens_excluded_from_background_throughput(self):
        recs = ([rec(role="background", t_send=float(i), t_last=float(i) + 1.0, tokens=100)
                 for i in range(10)]
                + [rec(role="injected", t_send=0.0, t_last=10.0, tokens=99999)])
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        # 배경만: 10건 × 100 토큰 / 창 10초
        assert res.background_output_tps == pytest.approx(100.0)

    def test_counts_are_split_by_role(self):
        recs = [rec(role="background", t_send=0.0, t_last=1.0),
                rec(role="injected", t_send=0.0, t_last=1.0)]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.n_background == 1 and res.n_injected == 1

    def test_warmup_excluded(self):
        recs = [rec(role="background", t_send=0.0, t_last=1.0, warmup=True),
                rec(role="background", t_send=1.0, t_last=2.0)]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.n_background == 1


class TestInterferenceMeasure:
    def _spec(self, **kw):
        return MixedSpec(experiment="B3", min_samples_for_p95=1, **kw)

    def _target(self):
        from benchmark.target import Target
        return Target(name="t", base_url="http://h/v1", model="m", source="measured_local")

    def test_penalty_is_computed_from_within_run_split(self):
        """주입과 겹친 배경 요청의 TPOT 이 40% 나쁘면 penalty 40%."""
        recs = [rec(role="injected", t_send=0.0, t_last=5.0)]
        recs += [rec(role="background", t_send=1.0 + i * 0.1, t_last=1.5 + i * 0.1,
                     tpot=14.0) for i in range(5)]          # 겹침
        recs += [rec(role="background", t_send=10.0 + i, t_last=10.5 + i,
                     tpot=10.0) for i in range(5)]          # 안 겹침
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.clean_n == 5 and res.disturbed_n == 5
        assert res.tpot_penalty_pct == pytest.approx(40.0)

    def test_degenerate_split_is_flagged(self):
        """전부 겹쳤으면 비교가 성립하지 않는다."""
        recs = [rec(role="injected", t_send=0.0, t_last=100.0)]
        recs += [rec(role="background", t_send=float(i), t_last=float(i) + 1.0)
                 for i in range(5)]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.validation["overlap_split"]["verdict"] == "DEGENERATE"

    def test_small_groups_are_flagged(self):
        recs = [rec(role="injected", t_send=0.0, t_last=2.0),
                rec(role="background", t_send=0.5, t_last=1.0),
                rec(role="background", t_send=10.0, t_last=11.0)]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.validation["group_sizes"]["verdict"].startswith("INSUFFICIENT")

    def test_no_background_is_fatal(self):
        recs = [rec(role="injected", t_send=0.0, t_last=1.0)]
        res = _summarize(self._spec(), self._target(), recs, None)
        assert "fatal" in res.validation

    def test_errors_are_counted_not_silently_dropped(self):
        recs = [rec(role="background", t_send=0.0, t_last=1.0),
                rec(role="background", t_send=1.0, t_last=2.0, error="HTTP 500")]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.n_error == 1 and sum(res.errors.values()) == 1


class TestLengthBias:
    """겹침 기준 하나만 쓰면 편향 방향이 숨는다.

    교차검증 F9 후속 지적: 주입에 의해 느려진 요청은 생존 구간이 길어지고,
    따라서 주입과 겹칠 확률이 높아진다. 간섭이 **전혀 없어도** 겹친 그룹이
    느린 쪽으로 치우친다. 생존 겹침 기준은 간섭을 과대평가한다.
    """

    def _spec(self, **kw):
        return MixedSpec(experiment="B3", min_samples_for_p95=1, **kw)

    def _target(self):
        from benchmark.target import Target
        return Target(name="t", base_url="http://h/v1", model="m", source="measured_local")

    def test_start_time_classification_ignores_duration(self):
        """시작 시각 기준은 요청이 얼마나 오래 걸렸는지와 무관해야 한다."""
        recs = [rec(role="injected", t_send=0.0, t_last=5.0),
                rec(role="background", t_send=1.0, t_last=2.0),      # 주입 중 시작, 짧음
                rec(role="background", t_send=1.0, t_last=100.0)]    # 주입 중 시작, 김
        annotate_overlap(recs)
        assert recs[1].started_during_injection is True
        assert recs[2].started_during_injection is True

    def test_started_before_injection_is_not_counted_even_if_it_overlaps(self):
        """주입 직전 시작해 길게 도는 요청 — 생존은 겹치지만 시작은 밖이다."""
        recs = [rec(role="injected", t_send=5.0, t_last=8.0),
                rec(role="background", t_send=1.0, t_last=20.0)]
        annotate_overlap(recs)
        assert recs[1].overlapped_injections == 1       # 생존 겹침: 포착
        assert recs[1].started_during_injection is False  # 시작 기준: 놓침(과소평가 방향)

    def test_length_bias_demonstrated_with_zero_true_effect(self):
        """간섭이 없는데도 생존 겹침 기준은 차이를 만들어낸다.

        느린 요청(오래 삶)과 빠른 요청을 주입과 무관하게 섞어두면,
        느린 쪽이 겹칠 확률이 높아 '겹친 그룹' 이 느려 보인다.
        """
        recs = [rec(role="injected", t_send=10.0, t_last=11.0)]
        # 느린 요청: 오래 살아서 주입 구간을 지나간다
        recs += [rec(role="background", t_send=5.0, t_last=15.0, tpot=30.0)
                 for _ in range(5)]
        # 빠른 요청: 주입 구간 밖에서 짧게 끝난다
        recs += [rec(role="background", t_send=20.0 + i, t_last=20.5 + i, tpot=10.0)
                 for i in range(5)]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        # 생존 겹침 기준은 200% 페널티를 보고한다 — 실제 인과는 없다
        assert res.tpot_penalty_pct == pytest.approx(200.0)
        # 시작 시각 기준은 겹친 그룹이 비어 페널티를 내지 않는다
        assert res.start_disturbed_n == 0
        assert res.tpot_penalty_pct_start_based is None

    def test_both_measures_are_reported(self):
        recs = [rec(role="injected", t_send=0.0, t_last=10.0)]
        recs += [rec(role="background", t_send=1.0 + i, t_last=1.5 + i, tpot=14.0)
                 for i in range(5)]
        recs += [rec(role="background", t_send=20.0 + i, t_last=20.5 + i, tpot=10.0)
                 for i in range(5)]
        annotate_overlap(recs)
        res = _summarize(self._spec(), self._target(), recs, None)
        assert res.tpot_penalty_pct is not None
        assert res.tpot_penalty_pct_start_based is not None
        assert res.start_clean_n == 5 and res.start_disturbed_n == 5
