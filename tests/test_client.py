"""토큰 delta 추출과 지연 지표 계산 검증.

여기가 틀리면 모든 측정이 조용히 틀린다. 특히 추론 모델의 reasoning delta는
실제로 겪은 함정이라 회귀 테스트로 고정해둔다 (docs/03-api-findings.md §1).
"""

from benchmark.client import extract_delta_text, _finalize
from benchmark.schema import RequestRecord


def rec(**kw) -> RequestRecord:
    base = dict(run_id="r", request_id="q", experiment="T", source="measured_local")
    base.update(kw)
    return RequestRecord(**base)


class TestExtractDelta:
    def test_completions_text(self):
        chunk = {"choices": [{"text": " London", "finish_reason": None}]}
        assert extract_delta_text(chunk, is_chat=False) == " London"

    def test_chat_content(self):
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        assert extract_delta_text(chunk, is_chat=True) == "Hi"

    def test_chat_reasoning_is_a_token(self):
        """추론 모델은 content 가 비어 있고 reasoning 으로만 토큰을 낸다."""
        chunk = {"choices": [{"delta": {"reasoning": "Okay"}}]}
        assert extract_delta_text(chunk, is_chat=True) == "Okay"

    def test_role_only_chunk_is_not_a_token(self):
        """첫 chunk 는 항상 role-only 다. 이걸 토큰으로 세면 TTFT 가 과소평가된다."""
        chunk = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
        assert extract_delta_text(chunk, is_chat=True) == ""

    def test_empty_choices(self):
        assert extract_delta_text({"choices": []}, is_chat=True) == ""
        assert extract_delta_text({}, is_chat=False) == ""

    def test_content_preferred_but_reasoning_fallback(self):
        chunk = {"choices": [{"delta": {"content": "", "reasoning": "think"}}]}
        assert extract_delta_text(chunk, is_chat=True) == "think"


class TestFinalize:
    def test_ttft_e2e_tpot(self):
        r = rec(t_send=1.0, t_first_token=1.2, t_last_token=2.2, completion_tokens_actual=11)
        _finalize(r)
        assert r.ttft_ms == pytest_approx(200.0)
        assert r.e2e_ms == pytest_approx(1200.0)
        # 생성 구간 1000ms 를 (11-1)=10 개 간격으로 나눈다
        assert r.tpot_ms == pytest_approx(100.0)

    def test_tpot_undefined_for_single_token(self):
        """출력이 1토큰이면 토큰 간 간격이 없으므로 TPOT 은 정의되지 않는다."""
        r = rec(t_send=0.0, t_first_token=0.5, t_last_token=0.5, completion_tokens_actual=1)
        _finalize(r)
        assert r.ttft_ms == pytest_approx(500.0)
        assert r.tpot_ms is None

    def test_no_token_received(self):
        r = rec(t_send=0.0, completion_tokens_actual=None)
        _finalize(r)
        assert r.ttft_ms is None and r.e2e_ms is None and r.tpot_ms is None


def pytest_approx(x, tol=1e-6):
    import pytest
    return pytest.approx(x, abs=tol)


class TestRetryDoesNotCorruptTiming:
    """실측에서 만난 버그의 회귀 테스트.

    conc=32 에서 스트림이 멈춰 소켓 타임아웃 → 재시도했는데, 재시도가 t_send 만
    갱신하고 이전 시도의 t_first_token 을 남겨두어 TTFT 가 -302초로 나왔다.
    시도마다 타이밍 상태를 비워야 한다.
    """

    def test_reset_timing_clears_previous_attempt(self):
        from benchmark.client import _reset_timing
        r = rec(t_send=1.0, t_first_token=1.5, t_last_token=9.0,
                ttft_ms=500.0, e2e_ms=8000.0, tpot_ms=10.0,
                prompt_tokens_actual=512, completion_tokens_actual=128,
                cached_tokens=3, upstream_ms=12.0, finish_reason="length")
        _reset_timing(r)
        assert r.t_first_token is None and r.t_last_token is None
        assert r.ttft_ms is None and r.e2e_ms is None and r.tpot_ms is None
        assert r.prompt_tokens_actual is None and r.completion_tokens_actual is None
        assert r.cached_tokens is None and r.upstream_ms is None
        assert r.finish_reason is None

    def test_negative_latency_is_rejected_not_reported(self):
        """음수 지연은 물리적으로 불가능하다. 값을 내는 대신 에러로 드러낸다."""
        r = rec(t_send=304.4, t_first_token=1.6, t_last_token=309.1,
                completion_tokens_actual=128)
        _finalize(r)
        assert r.ttft_ms is None
        assert r.e2e_ms is None
        assert r.error is not None and "negative_latency" in r.error

    def test_normal_case_unaffected_by_guard(self):
        r = rec(t_send=1.0, t_first_token=1.2, t_last_token=2.2, completion_tokens_actual=11)
        _finalize(r)
        assert r.error is None and r.ttft_ms is not None
