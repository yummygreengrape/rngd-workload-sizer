"""프롬프트 생성 — 길이 정확성과 고유성.

둘 다 벤치마크 신뢰성의 전제다. 길이가 틀리면 축이 틀리고,
고유하지 않으면 prefix cache 가 prefill 비용을 삼킨다.
"""

import pytest

from benchmark.prompts import PromptFactory, CalibrationError


class FakeTokenizer:
    """단어당 tokens_per_word 토큰 + 고정 오버헤드. 호출 횟수를 센다."""

    def __init__(self, tokens_per_word=1.3, overhead=3):
        self.tpw = tokens_per_word
        self.overhead = overhead
        self.calls = 0

    def __call__(self, text: str):
        self.calls += 1
        return int(len(text.split()) * self.tpw) + self.overhead


class TestCalibration:
    @pytest.mark.parametrize("target", [128, 512, 1024, 2048, 8192])
    def test_converges_within_tolerance(self, target):
        tok = FakeTokenizer()
        f = PromptFactory(tok, tolerance=0.02)
        prompt = f.make(target)
        got = tok(prompt)
        assert abs(got - target) <= max(1, target * 0.02)

    @pytest.mark.parametrize("tpw", [0.8, 1.0, 1.3, 2.5])
    def test_converges_for_various_tokenizers(self, tpw):
        tok = FakeTokenizer(tokens_per_word=tpw)
        f = PromptFactory(tok, tolerance=0.03)
        got = tok(f.make(1024))
        assert abs(got - 1024) <= max(1, 1024 * 0.03)

    def test_calibration_is_cached(self):
        """같은 길이를 여러 번 요청해도 보정은 한 번만 한다 (probe 요청은 비싸다)."""
        tok = FakeTokenizer()
        f = PromptFactory(tok)
        f.make(512)
        after_first = tok.calls
        for _ in range(5):
            f.make(512)
        assert tok.calls == after_first

    def test_raises_when_counter_unavailable(self):
        f = PromptFactory(lambda _t: None)
        with pytest.raises(CalibrationError):
            f.make(512)

    def test_summary_records_probes(self):
        f = PromptFactory(FakeTokenizer())
        f.make(256)
        s = f.summary()
        assert s and s[0]["target_tokens"] == 256
        assert s[0]["probes"]


class TestUniqueness:
    def test_prompts_differ_every_call(self):
        """prefix cache 를 피하려면 매 요청 프롬프트가 달라야 한다."""
        f = PromptFactory(FakeTokenizer())
        prompts = {f.make(256) for _ in range(50)}
        assert len(prompts) == 50

    def test_no_shared_long_prefix(self):
        """앞부분이 길게 겹치면 그만큼 캐시에 적중한다."""
        f = PromptFactory(FakeTokenizer())
        a, b = f.make(512), f.make(512)
        shared = 0
        for x, y in zip(a.split(), b.split()):
            if x != y:
                break
            shared += 1
        assert shared < 5
