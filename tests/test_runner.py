"""집계와 자기검증 로직.

집계가 조용히 틀리는 것을 막는 게 목적이다. 특히:
- wall-clock aggregate 와 요청별 latency 를 섞지 않는가
- 표본이 적을 때 P95 를 내지 않는가
- prefix cache 오염을 비율로 판정하는가 (건수로 하면 전부 오탐)
"""

import pytest

from benchmark.runner import (
    ConditionSpec, _median, _pct, _summarize, PREFIX_CACHE_RATIO_LIMIT,
)
from benchmark.schema import (
    RequestRecord, SOURCE_HOSTED_ENDPOINT, SOURCE_MEASURED_LOCAL, capacity_blocks,
)
from benchmark.target import Target


def make_target(source=SOURCE_MEASURED_LOCAL) -> Target:
    return Target(name="t", base_url="http://127.0.0.1:8000/v1", model="m", source=source)


def make_records(n, *, out_tokens=10, prompt_tokens=512, cached=0, span=1.0,
                 warmup=0, error_every=0):
    """t_send 를 0,1,2... 로 두고 각 요청이 span 초 걸리게 만든다."""
    recs = []
    for i in range(warmup + n):
        is_w = i < warmup
        r = RequestRecord(run_id="r", request_id=f"q{i}", experiment="E", source="measured_local",
                          is_warmup=is_w, concurrency=1,
                          target_input_tokens=prompt_tokens, target_output_tokens=out_tokens)
        if error_every and i % error_every == 0 and not is_w:
            r.error = "HTTP 500: boom"
            recs.append(r)
            continue
        r.t_send = float(i)
        r.t_first_token = float(i) + 0.1
        r.t_last_token = float(i) + span
        r.ttft_ms = 100.0
        r.e2e_ms = span * 1000.0
        r.tpot_ms = (span * 1000.0 - 100.0) / max(1, out_tokens - 1)
        r.prompt_tokens_actual = prompt_tokens
        r.completion_tokens_actual = out_tokens
        r.cached_tokens = cached
        recs.append(r)
    return recs


class TestPercentiles:
    def test_median_odd_even(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert _median([]) is None

    def test_pct_nearest_rank(self):
        vals = [float(i) for i in range(1, 101)]   # 1..100
        assert _pct(vals, 0.95) == 95.0
        assert _pct(vals, 0.50) == 50.0

    def test_pct_small_sample(self):
        assert _pct([5.0], 0.95) == 5.0


class TestAggregate:
    def test_aggregate_uses_wall_clock_window_not_sum_of_latencies(self):
        """요청별 지연을 더해서 처리량을 내면 안 된다. 윈도 wall-clock 으로 나눈다."""
        recs = make_records(10, out_tokens=10, span=1.0)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=5),
                         make_target(), recs, None)
        # window: t_send 최솟값 0.0 ~ t_last_token 최댓값 9+1.0=10.0
        assert res.window_s == pytest.approx(10.0)
        assert res.aggregate_output_tps == pytest.approx(100 / 10.0)

    def test_warmup_excluded_from_aggregate_but_kept_in_records(self):
        recs = make_records(5, warmup=3)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(), recs, None)
        assert res.n_measured_ok == 5
        assert len([r for r in recs if r.is_warmup]) == 3   # 삭제하지 않는다

    def test_errors_counted_and_grouped(self):
        recs = make_records(10, error_every=3)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(), recs, None)
        assert res.n_error > 0
        assert sum(res.errors.values()) == res.n_error

    def test_no_successful_request(self):
        recs = make_records(3, error_every=1)
        res = _summarize(ConditionSpec(experiment="E"), make_target(), recs, None)
        assert "fatal" in res.validation
        assert res.aggregate_output_tps is None


class TestValidation:
    def test_p95_withheld_when_samples_insufficient(self):
        recs = make_records(10)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=100),
                         make_target(), recs, None)
        assert res.ttft_ms_p95 is None
        assert "insufficient_samples" in res.validation["p95"]

    def test_p95_emitted_when_enough(self):
        recs = make_records(100)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=100),
                         make_target(), recs, None)
        assert res.ttft_ms_p95 is not None
        assert "p95" not in res.validation

    def test_few_shared_leading_tokens_is_not_contamination(self):
        """선두 토큰 2개 공유는 정상이다. 건수 기준이면 전부 오탐이 된다."""
        recs = make_records(10, prompt_tokens=512, cached=2)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(), recs, None)
        assert res.validation["prefix_cache"]["verdict"] == "clean"

    def test_mostly_cached_prompt_is_contamination(self):
        recs = make_records(10, prompt_tokens=512, cached=500)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(), recs, None)
        pc = res.validation["prefix_cache"]
        assert pc["verdict"] == "CONTAMINATED"
        assert pc["max_cached_ratio"] > PREFIX_CACHE_RATIO_LIMIT

    def test_output_length_variation_flagged(self):
        recs = make_records(5, out_tokens=128)
        recs[2].completion_tokens_actual = 64      # EOS 로 일찍 끝난 경우
        res = _summarize(ConditionSpec(experiment="E", output_tokens=128, min_samples_for_p95=1),
                         make_target(), recs, None)
        assert res.validation["output_length"]["verdict"] == "VARIES"

    def test_input_length_drift_flagged(self):
        recs = make_records(5, prompt_tokens=700)
        res = _summarize(ConditionSpec(experiment="E", input_tokens=512, min_samples_for_p95=1),
                         make_target(), recs, None)
        assert res.validation["input_length"]["verdict"] == "DRIFTED"

    def test_hosted_source_blocked_from_capacity(self):
        """호스팅 결과가 capacity 계산으로 흘러가지 않도록 레코드 단계에서 막는다."""
        recs = make_records(5, out_tokens=128)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(SOURCE_HOSTED_ENDPOINT), recs, None)
        assert res.validation["capacity_usable"] is False
        # 사유가 출처여야 한다. 다른 이유로 우연히 막히면 출처 게이트가 죽어도 모른다.
        assert any("출처" in b for b in res.validation["capacity_blocks"])

    def test_local_source_with_clean_checks_allowed(self):
        recs = make_records(5, out_tokens=128)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(SOURCE_MEASURED_LOCAL), recs, None)
        assert res.validation["capacity_usable"] is True
        assert "capacity_blocks" not in res.validation

    def test_concurrency_unknown_without_metrics(self):
        recs = make_records(5)
        res = _summarize(ConditionSpec(experiment="E", concurrency=8, min_samples_for_p95=1),
                         make_target(), recs, None)
        assert "note" in res.validation["concurrency"]


class TestCapacityInvalidation:
    """"적중 시 자동 무효 처리" 가 실제로 일어나는가.

    README 는 prefix cache 적중을 "기록 후 초과 시 자동 무효 처리" 한다고 적고 있었는데,
    `capacity_usable` 은 출처만 보고 있었다. 기록은 남지만 아무도 읽지 않는 방어였다.

    **여기서 고정하는 것은 경계다.** 무엇을 막고 무엇을 안 막는지 둘 다 고정한다.
    안 막는 쪽을 고정하지 않으면 나중에 "오염이면 다 버리자" 로 번져서, metric 마다
    영향 방향이 다르다는 사실을 잃는다.
    """

    def test_prefix_cache_contamination_invalidates(self):
        recs = make_records(10, prompt_tokens=512, cached=500, out_tokens=128)
        res = _summarize(ConditionSpec(experiment="E", min_samples_for_p95=1),
                         make_target(SOURCE_MEASURED_LOCAL), recs, None)
        assert res.validation["prefix_cache"]["verdict"] == "CONTAMINATED"
        assert res.validation["capacity_usable"] is False
        assert any("prefix cache" in b for b in res.validation["capacity_blocks"])

    def test_output_length_variation_invalidates(self):
        """조건 라벨이 실제와 다르면 그 라벨로 보간할 수 없다."""
        recs = make_records(5, out_tokens=128)
        recs[2].completion_tokens_actual = 64
        res = _summarize(ConditionSpec(experiment="E", output_tokens=128, min_samples_for_p95=1),
                         make_target(SOURCE_MEASURED_LOCAL), recs, None)
        assert res.validation["capacity_usable"] is False
        assert any("출력 길이" in b for b in res.validation["capacity_blocks"])

    def test_warmup_deficiency_does_not_invalidate(self):
        """warm-up 부족은 **백분위만** 망가뜨린다. 처리량은 멀쩡하다.

        여기서 행을 통째로 버리면 conc=128 처럼 복구 가능한 측정을 날린다
        (앞 동시성만큼 제외해 복원된다). 근거 선택 순위로 다루는 게 맞다.
        """
        recs = make_records(20, out_tokens=128)
        spec = ConditionSpec(experiment="E", concurrency=8, warmup_requests=0,
                             min_samples_for_p95=1)
        spec.warmup_requests = 0            # __post_init__ 의 자동 상향을 되돌린다
        res = _summarize(spec, make_target(SOURCE_MEASURED_LOCAL), recs, None)
        assert res.validation["warmup"]["verdict"] == "DEFICIENT"
        assert res.validation["capacity_usable"] is True

    def test_input_length_drift_does_not_invalidate(self):
        """계단 분석은 목표 길이가 아니라 실제 토큰 수로 재정렬해서 본다."""
        recs = make_records(5, prompt_tokens=700, out_tokens=128)
        res = _summarize(ConditionSpec(experiment="E", input_tokens=512, min_samples_for_p95=1),
                         make_target(SOURCE_MEASURED_LOCAL), recs, None)
        assert res.validation["input_length"]["verdict"] == "DRIFTED"
        assert res.validation["capacity_usable"] is True


class TestCapacityBlockRule:
    """규칙 자체. runner 와 analysis/process.py 가 같은 함수를 쓴다."""

    def test_clean_validation_has_no_blocks(self):
        assert capacity_blocks({"source": SOURCE_MEASURED_LOCAL,
                                "prefix_cache": {"verdict": "clean"},
                                "output_length": {"verdict": "fixed"}}) == []

    def test_empty_validation_is_not_blocked(self):
        """검증 블록이 없다고 무효 처리하면 손으로 만든 행까지 막힌다."""
        assert capacity_blocks({}) == []

    def test_legacy_count_based_verdict_is_kept_but_labelled(self):
        """비율 기록 이전 run 은 판정만 있고 근거 수치가 없다. 뒤집지 않되 밝힌다."""
        blocks = capacity_blocks({"source": SOURCE_MEASURED_LOCAL,
                                  "prefix_cache": {"verdict": "CONTAMINATED",
                                                   "requests_with_cache_hit": 1,
                                                   "max_cached_tokens": 2}})
        assert len(blocks) == 1
        assert "비율 기록 없음" in blocks[0]

    def test_queued_not_batched_is_not_a_block(self):
        """큐잉은 측정 결함이 아니라 서버 거동의 관찰이다."""
        assert capacity_blocks({"source": SOURCE_MEASURED_LOCAL,
                                "concurrency": {"verdict": "QUEUED_NOT_BATCHED"}}) == []

    def test_hosted_source_is_a_block(self):
        blocks = capacity_blocks({"source": SOURCE_HOSTED_ENDPOINT})
        assert len(blocks) == 1 and "출처" in blocks[0]


class TestForeignLoadDetection:
    """같은 서버에 다른 클라이언트가 붙었으면 그 사실이 행에 남아야 한다.

    세션 1 의 soak 두 건이 이 사각지대에 있었다. 라벨은 conc=16 인데 창의 3/4 동안
    카드가 32 를 받고 있었고, `/metrics` 에는 `peak_running=32` 로 이미 기록돼 있었는데
    판정식이 **하한만** 봐서 `ok` 로 통과했다. 그 row 가 곡선에 들어갔고, 그 구간의
    성능 저하가 나중에 furiosa-smi 폴링 탓으로 잘못 귀속됐다.
    """

    def test_running_above_requested_is_foreign_load(self):
        v = {"source": SOURCE_MEASURED_LOCAL,
             "concurrency": {"requested": 16, "peak_running_samples": 32.0,
                             "verdict": "FOREIGN_LOAD"}}
        blocks = capacity_blocks(v)
        assert len(blocks) == 1
        assert "다른 클라이언트" in blocks[0]

    def test_queued_not_batched_is_still_not_a_block(self):
        """하한 미달은 서버 거동의 관찰이지 결함이 아니다. 여기서 막으면 안 된다."""
        assert capacity_blocks({"source": SOURCE_MEASURED_LOCAL,
                                "concurrency": {"verdict": "QUEUED_NOT_BATCHED"}}) == []

    def test_verdict_boundaries(self):
        """0.8 미만은 큐잉, 1.25 초과는 외부 부하, 그 사이는 ok."""
        from benchmark.runner import _validate, ConditionSpec

        class FakePoller:
            available = True
            def __init__(self, running): self._r = running
            def peak_sum(self, needle, *a):
                return self._r if needle == "running" else 0.0
            def peak(self, *a, **k): return None

        cases = [(12.0, "QUEUED_NOT_BATCHED"), (13.0, "ok"), (16.0, "ok"),
                 (20.0, "ok"), (21.0, "FOREIGN_LOAD"), (32.0, "FOREIGN_LOAD")]
        for running, expected in cases:
            res = _summarize(ConditionSpec(experiment="E", concurrency=16,
                                           min_samples_for_p95=1),
                             make_target(), make_records(5, out_tokens=128),
                             FakePoller(running))
            assert res.validation["concurrency"]["verdict"] == expected, (running, expected)
