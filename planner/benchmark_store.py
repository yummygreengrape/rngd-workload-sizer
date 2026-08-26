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

# 아티팩트에 선언된 prefill attention_size 버킷 (docs/00-environment.md §6).
# **이 중 다수는 실제 비용 경계가 아니다.** 아래 MEASURED 값을 쓸 수 없을 때만 쓴다.
ARTIFACT_BUCKET_EDGES = (128, 256, 384, 512, 640, 768, 896, 1024)

# 실측으로 확인한 **비용** 경계 (A3+A4, 요청 950건을 실제 토큰 수로 재정렬).
#
# 단일 토큰 해상도로 확인한 값 (8토큰 단위로 묶으면 경계가 한 칸 어긋난다):
#    256 tok :  27.05 ms  →   257 tok :  80.00 ms
#   1024 tok :  77.58 ms  →  1025 tok :  96.76 ms
#   1280 tok : 103.92 ms  →  1281 tok : 160.12 ms
#
# 즉 경계값은 **직전 구간의 마지막 토큰**이다. 평탄 구간은 257–1024 이다.
#
# 반면 384·768·896 주변은 ±0.3% 로 평탄하다. 아티팩트에 버킷이 선언돼 있어도
# 비용이 뛰지 않는다. 반대로 **1280 은 실제 계단인데 아티팩트 목록에 없다.**
#
# 아티팩트 값을 그대로 쓰면 (a) 있지도 않은 경계에서 보간 신뢰도를 깎고
# (b) "조금 줄이면 같은 비용" 이라는 틀린 조언을 하며 (c) 진짜 경계인 1280 을 놓친다.
#
# 모델마다 다르므로 실측이 없는 모델은 아티팩트 값으로 내려간다.
# 1280 위는 아직 측정하지 않았다.
MEASURED_COST_EDGES: dict[str, tuple[int, ...]] = {
    "furiosa-ai/Qwen3-8B-FP8": (128, 256, 1024, 1280),
}

PREFILL_BUCKET_EDGES = ARTIFACT_BUCKET_EDGES   # 하위 호환 (모델 불명일 때)


def cost_edges(model: str | None = None) -> tuple[int, ...]:
    """이 모델의 실측 비용 경계. 없으면 아티팩트 선언값."""
    if model and model in MEASURED_COST_EDGES:
        return MEASURED_COST_EDGES[model]
    return ARTIFACT_BUCKET_EDGES

# 확장 효율이 이 값을 넘으면 계산이 아니라 집계가 틀린 것으로 본다.
# 실측에 102.4% 가 실제로 있으므로(측정 산포) 기준에 여유를 둔다.
EFFICIENCY_SANITY_LIMIT = 1.15


class DataSourceError(RuntimeError):
    pass


def _crosses_bucket_edge(a: int, b: int, model: str | None = None) -> bool:
    """[a, b] 보간 구간이 비용 계단을 지나는가.

    경계값은 **직전 구간의 마지막 토큰**이다 (256 tok = 27.05 ms, 257 tok = 80.00 ms).
    그래서 `lo <= edge < hi` 여야 한다. `lo < edge` 로 두면 1024→2048 보간이
    1024/1025 계단을 지나는데 edge == lo 라 놓친다.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    return any(lo <= edge < hi for edge in cost_edges(model))


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
            return self._accept_self_check(row)
        if row.source == SOURCE_MOCK:
            if self.allow_mock:
                return self._accept_self_check(row)
            return False, "mock 데이터입니다. allow_mock=True 로 명시해야 사용됩니다."
        if row.source == SOURCE_HOSTED_ENDPOINT:
            return False, ("호스팅 엔드포인트 측정입니다. 카드 수를 알 수 없고 멀티테넌트이며 "
                           "네트워크가 포함되어 capacity 계산에 사용할 수 없습니다.")
        return False, f"알 수 없는 출처: {row.source!r}"

    @staticmethod
    def _accept_self_check(row: BenchmarkRow) -> tuple[bool, str]:
        """출처가 통과해도, 하네스 자기검증이 조건의 오염을 보고했으면 거부한다.

        **출처를 먼저 본 뒤에 호출한다.** 호스팅 행은 오염 판정도 같이 걸리는데
        거부 사유로는 "호스팅" 이 맞다. 사유가 틀리면 사람이 엉뚱한 데를 고친다.

        여기서 걸리는 것은 조건의 정의가 깨진 경우뿐이다 (benchmark/schema.py 참조).
        warm-up 부족·짧은 측정창은 여기서 버리지 않고 근거 선택 순위로 반영한다.
        """
        if row.capacity_usable:
            return True, ""
        why = "; ".join(row.capacity_blocks) or "사유가 기록되지 않았습니다"
        return False, f"하네스 자기검증에서 무효 처리된 측정입니다 — {why}"

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
    MIN_SAMPLES_PER_WORKER = 20
    MIN_TRUSTED_WINDOW_S = 15.0     # 처리량 판정에 쓴다 (백분위 판정에는 아래 함수를 쓴다)

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
        """측정창이 기준보다 짧아 **처리량** 해석에 주의가 필요한 행."""
        return [r for r in curve if (r.window_s or 0) < self.MIN_TRUSTED_WINDOW_S]

    def untrustworthy_percentile_rows(self, curve: list[BenchmarkRow]) -> list[BenchmarkRow]:
        """지연 백분위를 믿을 수 없는 행 (warm-up 부족 또는 표본 얕음)."""
        return [r for r in curve if not percentiles_trustworthy(r)]

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
        if _crosses_bucket_edge(lo, hi, model):
            level = "interpolated_across_bucket_boundary"
            notes.append(
                f"입력 {input_tokens} tok 의 보간 구간({lo}~{hi})이 컴파일 버킷 경계를 넘습니다. "
                f"RNGD 는 컴파일된 shape 단위로 실행되므로 이 구간의 성능은 계단형일 수 있고 "
                f"선형 보간이 실제와 다를 수 있습니다."
            )
        edge = _next_bucket_edge_below(input_tokens, model)
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

        if concurrency_per_card is None:
            eff = self._efficiency_at(model, n_cards, None)
            return (eff, "all_conditions_fallback") if eff is not None else (1.0, "linear_assumption")

        grid = self._comparable_concurrencies(model)
        if not grid:
            eff = self._efficiency_at(model, n_cards, None)
            return (eff, "all_conditions_fallback") if eff is not None else (1.0, "linear_assumption")

        # **최근접 스냅을 쓰지 않는다.** 스냅은 격자 사이에 불연속을 만들고, 카드 수는
        # 그 불연속 위에서 고정점이 진동한다 — 실측으로 카드당 48↔49 에서 효율이
        # 1.0093 ↔ 0.8034 로 뛰며 카드 수가 20↔25 를 무한 반복했다. 동시성 방향으로
        # 보간하면 연속이 되고 수렴한다.
        if concurrency_per_card <= grid[0]:
            eff = self._efficiency_at(model, n_cards, grid[0])
            basis = ("measured_curve" if concurrency_per_card == grid[0]
                     else f"held_below_grid({grid[0]})")
        elif concurrency_per_card >= grid[-1]:
            # 격자 밖으로 외삽하지 않고 마지막 실측을 유지하되 **그 사실을 basis 로 드러낸다.**
            # 하강 추세에서 마지막 값을 유지하는 것은 보수적이 아니라 낙관적이다.
            eff = self._efficiency_at(model, n_cards, grid[-1])
            basis = ("measured_curve" if concurrency_per_card == grid[-1]
                     else f"held_above_grid({grid[-1]})")
        else:
            i = bisect_left(grid, concurrency_per_card)
            if grid[i] == concurrency_per_card:
                eff, basis = self._efficiency_at(model, n_cards, grid[i]), "measured_curve"
            else:
                lo, hi = grid[i - 1], grid[i]
                e_lo, e_hi = (self._efficiency_at(model, n_cards, lo),
                              self._efficiency_at(model, n_cards, hi))
                if e_lo is None or e_hi is None:
                    eff, basis = (e_lo or e_hi), f"interpolated_concurrency({lo}~{hi})"
                else:
                    eff = _lerp(concurrency_per_card, lo, hi, e_lo, e_hi)
                    basis = f"interpolated_concurrency({lo}~{hi})"
        if eff is None:
            eff2 = self._efficiency_at(model, n_cards, None)
            return (eff2, "all_conditions_fallback") if eff2 is not None else (1.0, "linear_assumption")
        return eff, basis

    def _comparable_concurrencies(self, model: str) -> list[int]:
        """1장 기준선과 다중 카드 측정이 **둘 다** 있는 카드당 동시성.

        1장 곡선에만 있는 동시성을 고르면 비교쌍이 없어 조건이 통째로 폴백된다.
        실제로 그랬다 — 운영점 카드당 50 이 1장 곡선의 64 를 골랐는데 다중 카드는
        4·16·32 에만 있어서, **카드당 4 에서 잰 효율(96.9%)이 카드당 50 운영점에**
        쓰였다. 그 값은 실측 추세(카드당 32 에서 82%)와 반대 방향이다.
        """
        singles = {r.concurrency_per_card for r in self.rows
                   if r.model == model and r.n_cards == 1}
        multi = {r.concurrency_per_card for r in self.rows
                 if r.model == model and r.n_cards > 1}
        return sorted(singles & multi)

    def _efficiency_at(self, model: str, n_cards: int,
                       concurrency_per_card: int | None) -> float | None:
        """그 카드당 동시성에서의 카드 수 방향 효율. 자료가 없으면 None."""
        points = self._raw_scaling_points(model, concurrency_per_card)
        if not points:
            return None
        if n_cards in points:
            return points[n_cards]
        ns = sorted(points)
        if n_cards < ns[0]:
            return points[ns[0]]
        if n_cards > ns[-1]:
            return points[ns[-1]]
        i = bisect_left(ns, n_cards)
        lo, hi = ns[i - 1], ns[i]
        return _lerp(n_cards, lo, hi, points[lo], points[hi])

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
            # 1장 곡선이 아니라 **비교쌍이 있는 동시성**에서 고른다.
            available = self._comparable_concurrencies(model) or sorted(
                {r.concurrency_per_card for r in singles})
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


def percentiles_trustworthy(r: BenchmarkRow) -> bool:
    """이 행의 지연 백분위를 믿을 수 있는가.

    처음에는 "측정창 15초 이상" 을 기준으로 삼았는데, 그건 **원인이 아니라 대리지표**였다.
    진짜 변수는 warm-up 이 동시성보다 적을 때 워커 일부의 첫 요청이 측정에 섞이는 것이다.
    그 요청 수는 고정(=동시성)이라 오래 측정하면 희석될 뿐이다.

    실측: B1 conc=32(warmup=8) p95 1,437ms → 앞 32건 제외 시 246ms (참값 252ms).
    앞 8건·16건 제외로는 복원되지 않고 **정확히 동시성만큼** 빼야 한다.
    """
    if r.percentile_source == "recomputed_ramp_excluded":
        ok_ramp = True
    elif r.warmup_requests is None:
        ok_ramp = True                      # 알 수 없으면 판단을 보류한다
    else:
        ok_ramp = r.warmup_requests >= r.concurrency_per_card
    deep = r.n_samples >= r.concurrency_per_card * BenchmarkStore.MIN_SAMPLES_PER_WORKER
    return ok_ramp and deep


def _quality(r: BenchmarkRow) -> tuple[int, int, int, float]:
    """측정 신뢰도 순서. 높은 순으로:

    1. **백분위를 믿을 수 있는가** (warm-up ≥ 동시성 또는 앞 구간 제외 재계산, 표본 충분)
    2. **처리량 측정창이 충분한가** (실측: 8장 conc=256 이 창 2.7초에서 4,822 tok/s,
       14.7초에서 7,003 tok/s)
    3. **단독 실행인가** (실측: 707 → 521 tok/s)
    4. 측정창 길이
    """
    return (
        1 if percentiles_trustworthy(r) else 0,
        1 if (r.window_s or 0) >= BenchmarkStore.MIN_TRUSTED_WINDOW_S else 0,
        1 if r.exclusive else 0,
        r.window_s or 0.0,
    )


def _next_bucket_edge_below(x: int, model: str | None = None) -> int | None:
    below = [e for e in cost_edges(model) if e < x]
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
