"""프롬프트 생성 — 길이는 정확하게, 내용은 매번 다르게.

두 가지를 동시에 만족해야 한다.

1) **매 요청 고유**해야 한다. 같은 프롬프트를 반복하면 prefix cache 가 적중해
   prefill 비용이 과소평가된다. 긴 컨텍스트에서 최대 ~21배까지 벌어지는 것을
   실측했다 (docs/03-api-findings.md §3).

2) **토큰 길이가 목표에 맞아야** 한다. 문자 수로 어림하면 안 되고, 서버 기준
   토크나이저로 세야 한다 (docs/01-spec-review.md IMP-5). 단어 수를 조금씩
   보정하며 목표 토큰 수에 수렴시킨다.

랜덤 단어를 쓰므로 요청마다 토큰 수가 소폭 흔들린다. 그래서 실제 길이는
usage.prompt_tokens 로 사후 기록하고, 의도와 다르면 그 차이를 숨기지 않는다.
"""

from __future__ import annotations

import random
import uuid
from typing import Callable

# 흔한 영어 명사. 대부분 1토큰으로 쪼개지고 모델·언어 편향이 적다.
WORDS = [
    "elephant", "harbor", "crimson", "lantern", "meadow", "quartz", "tundra",
    "violin", "zephyr", "bramble", "cobalt", "dolphin", "ember", "fossil",
    "granite", "hollow", "ivory", "jasmine", "kettle", "lagoon", "marble",
    "nectar", "opal", "prairie", "quiver", "ripple", "saffron", "thicket",
    "umber", "velvet", "walnut", "yonder", "anchor", "beacon", "cavern",
]

TokenCounter = Callable[[str], "int | None"]


class CalibrationError(RuntimeError):
    pass


class PromptFactory:
    """목표 토큰 길이별로 필요한 단어 수를 한 번 보정해두고 재사용한다."""

    def __init__(self, token_counter: TokenCounter, *, tolerance: float = 0.02,
                 max_probes: int = 8, prefix: str = ""):
        self.count_tokens = token_counter
        self.tolerance = tolerance
        self.max_probes = max_probes
        self.prefix = prefix
        self._words_for: dict[int, int] = {}    # target_tokens -> word_count
        self.calibration_log: list[dict] = []

    # -- 생성 ---------------------------------------------------------------

    def _render(self, n_words: int, rng: random.Random) -> str:
        return self.prefix + " ".join(rng.choice(WORDS) for _ in range(n_words))

    def make(self, target_tokens: int) -> str:
        """목표 길이에 맞는 **고유** 프롬프트 하나를 만든다."""
        n = self._words_for.get(target_tokens)
        if n is None:
            n = self.calibrate(target_tokens)
        # uuid4 로 시드를 잡아 요청마다 다른 단어 열을 만든다.
        return self._render(n, random.Random(uuid.uuid4().int))

    # -- 보정 ---------------------------------------------------------------

    def calibrate(self, target_tokens: int) -> int:
        """목표 토큰 수에 맞는 단어 수를 비례 보정으로 찾는다."""
        if target_tokens in self._words_for:
            return self._words_for[target_tokens]

        n = max(1, int(target_tokens / 1.3))    # 영어 단어당 대략 1.3 토큰에서 출발
        best: tuple[int, int] | None = None     # (오차, 단어수)
        probes = []

        for _ in range(self.max_probes):
            rng = random.Random(uuid.uuid4().int)
            got = self.count_tokens(self._render(n, rng))
            if got is None:
                raise CalibrationError(
                    "토큰 수를 셀 수 없습니다. 전용 서버라면 /tokenize 를, "
                    "호스팅이라면 probe 요청 경로를 확인하세요."
                )
            probes.append({"words": n, "tokens": got})
            err = abs(got - target_tokens)
            if best is None or err < best[0]:
                best = (err, n)
            if err <= max(1, int(target_tokens * self.tolerance)):
                break
            ratio = target_tokens / got if got else 1.5
            nxt = max(1, round(n * ratio))
            if nxt == n:                       # 반올림으로 멈추면 한 칸씩 민다
                nxt = n + (1 if got < target_tokens else -1)
                nxt = max(1, nxt)
            n = nxt

        assert best is not None
        self._words_for[target_tokens] = best[1]
        self.calibration_log.append(
            {"target_tokens": target_tokens, "words": best[1],
             "final_error_tokens": best[0], "probes": probes}
        )
        return best[1]

    def summary(self) -> list[dict]:
        """meta.json 에 남길 보정 결과."""
        return self.calibration_log
