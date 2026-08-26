"""실측 데이터 로드와 조건 선택.

두 가지 책임이 있다.

1) **출처 게이팅.** 호스팅 엔드포인트에서 나온 데이터는 카드 수를 알 수 없으므로
   capacity 계산에 쓸 수 없다. 사람이 실수로 섞는 걸 막기 위해 **로드 단계에서 거부**한다
   (docs/03-api-findings.md §7). mock 은 개발용으로 명시 허용해야만 들어온다.

2) **조건 선택과 신뢰도 판정.** 사용자의 ROD 와 정확히 같은 실측이 없을 때
   보간인지 외삽인지, 그리고 **컴파일 버킷 경계를 넘는 보간인지**를 구분한다.
   RNGD 는 컴파일된 shape 단위로 실행되므로 경계를 넘는 선형 보간은 신뢰할 수 없다
   (docs/01-spec-review.md IMP-1).
"""

from __future__ import annotations

import json
import os
from bisect import bisect_left
from typing import Any, Iterable

from .models import BenchmarkRow, Confidence, worst

SOURCE_MEASURED_LOCAL = "measured_local"
SOURCE_HOSTED_ENDPOINT = "hosted_endpoint"
SOURCE_MOCK = "mock"

# 전용 서버 아티팩트에서 읽은 prefill attention_size 버킷 경계 (docs/00-environment.md §6).
# 이 경계를 넘는 입력 길이 보간은 계단 구조 때문에 선형이 아니다.
PREFILL_BUCKET_EDGES = (128, 256, 384, 512, 640, 768, 896, 1024)

# 확장 효율이 이 값을 넘으면 계산이 아니라 집계가 틀린 것으로 본다.
# 실측에 102.4% 가 실제로 있으므로(측정 산포) 기준에 여유를 둔다.
EFFICIENCY_SANITY_LIMIT = 1.15


class DataSourceError(RuntimeError):
    pass


def _crosses_bucket_edge(a: int, b: int) -> bool:
    lo, hi = (a, b) if a <= b else (b, a)
    return any(lo < edge < hi for edge in PREFILL_BUCKET_EDGES)


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


class BenchmarkStore:
    """조건별 실측 행을 담고, 요청 조건에 가장 맞는 근거를 골라준다."""

    def __init__(self, rows: Iterable[BenchmarkRow], *, allow_mock: bool = False):
        self.allow_mock = allow_mock
        self.rows: list[BenchmarkRow] = []
        self.rejected: list[dict[str, Any]] = []
        for r in rows:
            ok, why = self._accept(r)
            if ok:
                self.rows.append(r)
            else:
                self.rejected.append({"run_id": r.run_id, "source": r.source, "reason": why})
        # 무결성 검사는 **로드 시점에 전량** 한다. 질의 시점에 두면 무엇을 묻느냐에 따라
        # 오염이 드러나거나 숨는다 — 아무도 그 구간을 묻지 않으면 계속 남는다.
        self.scaling_issues: list[dict[str, Any]] = self._audit_scaling()

    def _audit_scaling(self) -> list[dict[str, Any]]:
        """모든 (모델, 조건, 카드 수) 조합의 확장 효율을 훑어 불가능한 값을 찾는다.

        효율은 두 행의 비율이라 어느 쪽이 틀렸는지 특정할 수 없다. 그래서 행을 버리지 않고
        **그 모델의 확장 곡선을 사용 불가로 표시**한다. planner 는 선형 가정으로 내려가되
        결과에 강한 경고를 붙인다.
        """
        issues: list[dict[str, Any]] = []
        for model in {r.model for r in self.rows}:
            conc = {r.concurrency_per_card for r in self.rows if r.model == model}
            for c in conc:
                for n, eff in self._raw_scaling_points(model, c).items():
                    if eff > EFFICIENCY_SANITY_LIMIT:
                        issues.append({
                            "model": model, "concurrency_per_card": c, "n_cards": n,
                            "efficiency": round(eff, 4),
                            "reason": (f"확장 효율 {eff * 100:.0f}% 는 물리적으로 불가능합니다. "
                                       f"집계가 틀렸을 가능성이 높습니다 — 기준 행 선택이나 "
                                       f"조건 묶기를 확인하세요."),
                        })
        return issues

    def unusable_scaling_models(self) -> set[str]:
        return {i["model"] for i in self.scaling_issues}

    # -- 게이팅 -------------------------------------------------------------

    def _accept(self, row: BenchmarkRow) -> tuple[bool, str]:
        if row.source == SOURCE_MEASURED_LOCAL:
            return True, ""
        if row.source == SOURCE_MOCK:
            if self.allow_mock:
                return True, ""
            return False, "mock 데이터입니다. allow_mock=True 로 명시해야 사용됩니다."
        if row.source == SOURCE_HOSTED_ENDPOINT:
            return False, ("호스팅 엔드포인트 측정입니다. 카드 수를 알 수 없고 멀티테넌트이며 "
                           "네트워크가 포함되어 capacity 계산에 사용할 수 없습니다.")
        return False, f"알 수 없는 출처: {row.source!r}"

    @property
    def data_source(self) -> str:
        """이 store 가 담고 있는 데이터의 성격. UI 배너에 그대로 쓴다."""
        srcs = {r.source for r in self.rows}
        if not srcs:
            return "empty"
        if srcs == {SOURCE_MEASURED_LOCAL}:
            return SOURCE_MEASURED_LOCAL
        if srcs == {SOURCE_MOCK}:
            return SOURCE_MOCK
        return "mixed"

    # -- 적재 ---------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str, *, allow_mock: bool = False) -> "BenchmarkStore":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload["rows"] if isinstance(payload, dict) else payload
        known = set(BenchmarkRow.__dataclass_fields__)
        rows = [BenchmarkRow(**{k: v for k, v in d.items() if k in known}) for d in raw]
        return cls(rows, allow_mock=allow_mock)

    @classmethod
    def from_dir(cls, path: str, *, allow_mock: bool = False) -> "BenchmarkStore":
        rows: list[BenchmarkRow] = []
        known = set(BenchmarkRow.__dataclass_fields__)
        for name in sorted(os.listdir(path)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(path, name), encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload["rows"] if isinstance(payload, dict) else payload
            rows.extend(BenchmarkRow(**{k: v for k, v in d.items() if k in known}) for d in raw)
        return cls(rows, allow_mock=allow_mock)

    # -- 조회 ---------------------------------------------------------------

    def models(self) -> list[str]:
        return sorted({r.model for r in self.rows})

    def single_card_rows(self, model: str) -> list[BenchmarkRow]:
        return [r for r in self.rows if r.model == model and r.n_cards == 1]

    def input_lengths(self, model: str) -> list[int]:
        return sorted({r.input_tokens for r in self.single_card_rows(model)})

    # 이보다 짧은 측정창은 램프업이 지표를 지배해 신뢰할 수 없다고 본다.
    # 실측에서 같은 조건이 측정창 13.7초일 때 TTFT p95 1437ms, 60초일 때 252ms 로
    # 나온 사례가 있다. 처리량도 짧은 창에서 최대 45% 과소평가됐다.
    MIN_TRUSTED_WINDOW_S = 15.0

    def concurrency_curve(self, model: str, input_tokens: int,
                          output_tokens: int) -> list[BenchmarkRow]:
        """특정 (입력, 출력) 길이에서 concurrency 순으로 정렬된 단일 카드 측정.

        같은 concurrency 를 여러 번 측정했으면 **측정창이 가장 긴 것**을 채택한다.
        짧은 창은 시작 시 요청이 한꺼번에 몰리는 구간이 통계를 지배한다.
        """
        rows = [r for r in self.single_card_rows(model)
                if r.input_tokens == input_tokens and r.output_tokens == output_tokens]
        best: dict[int, BenchmarkRow] = {}
        for r in rows:
            c = r.concurrency_per_card
            cur = best.get(c)
            if cur is None or _quality(r) > _quality(cur):
                best[c] = r
        return sorted(best.values(), key=lambda r: r.concurrency_per_card)

    def short_window_rows(self, curve: list[BenchmarkRow]) -> list[BenchmarkRow]:
        """측정창이 기준보다 짧아 해석에 주의가 필요한 행."""
        return [r for r in curve if (r.window_s or 0) < self.MIN_TRUSTED_WINDOW_S]

    def nearest_output_tokens(self, model: str, output_tokens: int) -> tuple[int, Confidence]:
        available = sorted({r.output_tokens for r in self.single_card_rows(model)})
        if not available:
            raise DataSourceError(f"{model} 의 단일 카드 측정이 없습니다.")
        if output_tokens in available:
            return output_tokens, "measured"
        nearest = min(available, key=lambda v: abs(v - output_tokens))
        level: Confidence = ("extrapolated"
                             if output_tokens < available[0] or output_tokens > available[-1]
                             else "interpolated")
        return nearest, level

    # -- 입력 길이 슬라이스 --------------------------------------------------

    def resolve_input_slice(self, model: str, input_tokens: int, output_tokens: int
                            ) -> tuple[list[BenchmarkRow], Confidence, list[str]]:
        """요청 입력 길이에 해당하는 concurrency 곡선을 만든다.

        정확히 일치하는 측정이 있으면 그대로, 없으면 인접한 두 길이의 곡선을 보간한다.
        버킷 경계를 넘으면 별도 신뢰도로 표시한다.
        """
        notes: list[str] = []
        lengths = sorted({r.input_tokens for r in self.single_card_rows(model)
                          if r.output_tokens == output_tokens})
        if not lengths:
            raise DataSourceError(f"{model} / output={output_tokens} 측정이 없습니다.")

        if input_tokens in lengths:
            return self.concurrency_curve(model, input_tokens, output_tokens), "measured", notes

        if input_tokens < lengths[0] or input_tokens > lengths[-1]:
            nearest = lengths[0] if input_tokens < lengths[0] else lengths[-1]
            notes.append(
                f"입력 {input_tokens} tok 은 실측 범위({lengths[0]}~{lengths[-1]}) 밖입니다. "
                f"가장 가까운 {nearest} tok 측정을 사용합니다."
            )
            return self.concurrency_curve(model, nearest, output_tokens), "extrapolated", notes

        i = bisect_left(lengths, input_tokens)
        lo, hi = lengths[i - 1], lengths[i]
        level: Confidence = "interpolated"
        if _crosses_bucket_edge(lo, hi):
            level = "interpolated_across_bucket_boundary"
            notes.append(
                f"입력 {input_tokens} tok 의 보간 구간({lo}~{hi})이 컴파일 버킷 경계를 넘습니다. "
                f"RNGD 는 컴파일된 shape 단위로 실행되므로 이 구간의 성능은 계단형일 수 있고 "
                f"선형 보간이 실제와 다를 수 있습니다."
            )
        edge = _next_bucket_edge_below(input_tokens)
        if edge is not None and 0 < input_tokens - edge <= max(8, edge * 0.02):
            notes.append(
                f"입력 {input_tokens} tok 은 버킷 경계 {edge} 를 살짝 넘습니다. "
                f"{edge} tok 으로 줄이면 padding 없이 같은 비용으로 처리될 가능성이 높습니다."
            )

        curve_lo = self.concurrency_curve(model, lo, output_tokens)
        curve_hi = self.concurrency_curve(model, hi, output_tokens)
        merged = _interpolate_curves(curve_lo, curve_hi, lo, hi, input_tokens)
        return merged, level, notes

    # -- 확장 효율 ----------------------------------------------------------

    def scaling_efficiency(self, model: str, n_cards: int,
                           concurrency_per_card: int | None = None) -> tuple[float, str]:
        """카드 n장일 때의 확장 효율.

        효율은 카드 수만이 아니라 **카드당 동시성에도 의존한다**. 실측에서
        카드당 동시성 4/16에서는 8장까지 97~102%를 유지했지만, 32에서는 82%로
        떨어졌다. 전 조건을 평균내면 낮은 동시성 운영점의 카드 수가 과대 추정된다.

        다중 카드 실측이 없으면 1.0(선형 가정)을 돌려주되 근거가 가정임을 알린다
        (docs/01-spec-review.md CRIT-4).
        """
        if not self._has_multicard(model):
            return 1.0, "linear_assumption"
        if model in self.unusable_scaling_models():
            # 곡선을 믿을 수 없다. 예외로 전부 세우는 대신 선형 가정으로 내려가고
            # 그 사실을 basis 로 드러낸다 — planner 가 결과에 경고를 붙인다.
            return 1.0, "linear_assumption_scaling_data_rejected"
        if n_cards <= 1:
            return 1.0, "measured_curve"

        points = self._raw_scaling_points(model, concurrency_per_card)
        if not points:
            points = self._raw_scaling_points(model, None)   # 동시성별 자료가 없으면 전체 평균
        if not points:
            return 1.0, "linear_assumption"

        if n_cards in points:
            return points[n_cards], "measured_curve"
        ns = sorted(points)
        if n_cards < ns[0]:
            return points[ns[0]], "measured_curve"
        if n_cards > ns[-1]:
            # 실측 밖으로 외삽하지 않고 마지막 실측 효율을 유지한다 (보수적)
            return points[ns[-1]], "measured_curve"
        i = bisect_left(ns, n_cards)
        lo, hi = ns[i - 1], ns[i]
        return _lerp(n_cards, lo, hi, points[lo], points[hi]), "measured_curve"

    def _has_multicard(self, model: str) -> bool:
        return any(r.model == model and r.n_cards > 1 for r in self.rows)

    def _raw_scaling_points(self, model: str,
                            concurrency_per_card: int | None) -> dict[int, float]:
        """카드 수 -> 효율.

        같은 (입력, 출력, 카드당 동시성) 조건끼리만 비교한다. concurrency_per_card 를
        주면 그 값에 가장 가까운 실측 동시성 하나만 쓴다.
        """
        singles = [r for r in self.rows if r.model == model and r.n_cards == 1]
        if not singles:
            return {}

        if concurrency_per_card is not None:
            available = sorted({r.concurrency_per_card for r in singles})
            pick = min(available, key=lambda c: abs(c - concurrency_per_card))
            singles = [r for r in singles if r.concurrency_per_card == pick]
            wanted = pick
        else:
            wanted = None

        base: dict[tuple[int, int, int], BenchmarkRow] = {}
        for r in singles:
            key = (r.input_tokens, r.output_tokens, r.concurrency_per_card)
            # 같은 조건이 여러 번 측정됐으면 측정창이 긴 쪽을 기준으로 삼는다
            if key not in base or _quality(r) > _quality(base[key]):
                base[key] = r

        eff: dict[int, list[float]] = {}
        for r in self.rows:
            if r.model != model or r.n_cards <= 1:
                continue
            if wanted is not None and r.concurrency_per_card != wanted:
                continue
            key = (r.input_tokens, r.output_tokens, r.concurrency_per_card)
            b = base.get(key)
            if not b or not b.aggregate_output_tps:
                continue
            eff.setdefault(r.n_cards, []).append(
                (_quality(r), r.aggregate_output_tps / (b.aggregate_output_tps * r.n_cards)))
        # 카드 수마다 측정창이 가장 긴 측정 하나만 쓴다. 평균을 내면 짧은 창의
        # 과소평가가 섞여 효율이 실제보다 낮게 나온다.
        return {n: max(v, key=lambda x: x[0])[1] for n, v in eff.items() if v}


def _quality(r: BenchmarkRow) -> tuple[int, int, float]:
    """측정 신뢰도 순서. 우선순위가 높은 순으로:

    1. **측정창이 신뢰 하한 이상인가.** 짧은 창은 시작 시 요청이 몰리는 램프업이
       지연 백분위를 지배한다. 단독 실행이어도 이 오염은 회복되지 않는다.
       (실측: 같은 조건이 창 12초에서 TTFT p95 3,070ms, 창 63초에서 261ms)
    2. **단독 실행인가.** 같은 머신에서 다른 측정이 동시에 돌면 부하 생성기가
       호스트 자원을 나눠 써 처리량이 낮게 나온다. (실측: 707 → 521 tok/s)
    3. 측정창이 긴가.
    """
    trusted = 1 if (r.window_s or 0) >= BenchmarkStore.MIN_TRUSTED_WINDOW_S else 0
    return (trusted, 1 if r.exclusive else 0, r.window_s or 0.0)


def _next_bucket_edge_below(x: int) -> int | None:
    below = [e for e in PREFILL_BUCKET_EDGES if e < x]
    return below[-1] if below else None


def _interpolate_curves(curve_lo: list[BenchmarkRow], curve_hi: list[BenchmarkRow],
                        lo: int, hi: int, target: int) -> list[BenchmarkRow]:
    """두 입력 길이의 concurrency 곡선을 길이 방향으로 선형 보간한다."""
    by_lo = {r.concurrency_per_card: r for r in curve_lo}
    by_hi = {r.concurrency_per_card: r for r in curve_hi}
    out: list[BenchmarkRow] = []
    for c in sorted(set(by_lo) & set(by_hi)):
        a, b = by_lo[c], by_hi[c]
        out.append(BenchmarkRow(
            model=a.model, source=a.source, n_cards=1, concurrency_per_card=c,
            input_tokens=target, output_tokens=a.output_tokens,
            aggregate_output_tps=_lerp(target, lo, hi, a.aggregate_output_tps, b.aggregate_output_tps),
            per_user_output_tps=_lerp(target, lo, hi, a.per_user_output_tps, b.per_user_output_tps),
            ttft_ms_p50=_lerp(target, lo, hi, a.ttft_ms_p50, b.ttft_ms_p50),
            ttft_ms_p95=(None if a.ttft_ms_p95 is None or b.ttft_ms_p95 is None
                         else _lerp(target, lo, hi, a.ttft_ms_p95, b.ttft_ms_p95)),
            e2e_ms_p95=(None if a.e2e_ms_p95 is None or b.e2e_ms_p95 is None
                        else _lerp(target, lo, hi, a.e2e_ms_p95, b.e2e_ms_p95)),
            tpot_ms_p50=(None if a.tpot_ms_p50 is None or b.tpot_ms_p50 is None
                         else _lerp(target, lo, hi, a.tpot_ms_p50, b.tpot_ms_p50)),
            kv_cache_peak=(None if a.kv_cache_peak is None or b.kv_cache_peak is None
                           else _lerp(target, lo, hi, a.kv_cache_peak, b.kv_cache_peak)),
            waiting_peak=(None if a.waiting_peak is None or b.waiting_peak is None
                          else _lerp(target, lo, hi, a.waiting_peak, b.waiting_peak)),
            n_samples=min(a.n_samples, b.n_samples),
            window_s=min([w for w in (a.window_s, b.window_s) if w is not None], default=None),
            exclusive=a.exclusive and b.exclusive,
            run_id=f"interp({a.run_id},{b.run_id})",
        ))
    return out
