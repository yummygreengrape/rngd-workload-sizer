"""planner 의 입출력 자료형.

숫자에는 전부 단위를 붙인다. latency 는 ms, throughput 은 tokens/s,
문서 처리량은 documents/s 다 (docs/01-spec-review.md IMP-9, IMP-10).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

WorkloadType = Literal["llm_chat", "prefill_heavy", "embedding"]

# 근거 데이터를 어떻게 얻었는지. 아래로 갈수록 신뢰도가 낮다.
Confidence = Literal[
    "measured",                             # 실측 격자점과 정확히 일치
    "interpolated",                         # 격자 내부, 버킷 경계를 넘지 않음
    "interpolated_across_bucket_boundary",  # 격자 내부지만 컴파일 버킷 경계를 넘음
    "extrapolated",                         # 격자 밖
]

CONFIDENCE_ORDER: dict[str, int] = {
    "measured": 0,
    "interpolated": 1,
    "interpolated_across_bucket_boundary": 2,
    "extrapolated": 3,
}


def worst(*levels: str) -> str:
    """여러 근거가 섞이면 가장 낮은 신뢰도를 결과의 신뢰도로 삼는다."""
    return max(levels, key=lambda x: CONFIDENCE_ORDER[x])


@dataclass
class BenchmarkRow:
    """실측 격자점 하나. benchmark/ 의 summary.json 을 가공한 결과."""

    model: str
    source: str
    n_cards: int
    concurrency_per_card: int
    input_tokens: int
    output_tokens: int

    aggregate_output_tps: float          # 이 배포 전체의 출력 tokens/s
    per_user_output_tps: float           # 사용자 1명이 체감하는 tokens/s
    ttft_ms_p50: float
    ttft_ms_p95: float | None = None
    e2e_ms_p95: float | None = None
    tpot_ms_p50: float | None = None

    kv_cache_peak: float | None = None   # 0~1
    waiting_peak: float | None = None
    n_samples: int = 0
    window_s: float | None = None   # 측정창 길이. 짧으면 램프업이 지표를 지배한다
    exclusive: bool = True          # 측정 중 같은 머신에서 다른 측정이 돌지 않았는가
    run_id: str = ""

    @property
    def per_card_output_tps(self) -> float:
        return self.aggregate_output_tps / max(1, self.n_cards)

    def evidence(self) -> dict[str, Any]:
        """어떤 측정에 근거했는지 UI 에 그대로 보여줄 요약."""
        return {
            "run_id": self.run_id,
            "model": self.model,
            "source": self.source,
            "n_cards": self.n_cards,
            "concurrency_per_card": self.concurrency_per_card,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "aggregate_output_tps": round(self.aggregate_output_tps, 1),
            "ttft_ms_p95": self.ttft_ms_p95,
            "n_samples": self.n_samples,
            "window_s": self.window_s,
            "exclusive": self.exclusive,
        }


@dataclass
class ServiceRequirement:
    """사용자가 입력하는 ROD (Requirements of Deployment)."""

    workload: WorkloadType
    model: str

    concurrent_users: int = 1
    avg_input_tokens: int = 512
    avg_output_tokens: int = 128

    target_output_tps_per_user: float = 15.0
    target_max_ttft_ms: float = 1500.0
    target_p95_e2e_ms: float | None = None

    target_utilization: float = 0.7       # headroom 0.3 과 같은 뜻

    # embedding 전용
    required_docs_per_s: float | None = None

    @staticmethod
    def from_headroom(headroom: float, **kw) -> "ServiceRequirement":
        """headroom 으로 받은 입력을 target_utilization 으로 변환한다.

        둘은 같은 값을 다르게 표현한 것일 뿐이므로, 내부에서는 하나만 쓴다.
        두 번 적용되는 사고를 막기 위해서다 (docs/01-spec-review.md IMP-10).
        """
        if not 0.0 <= headroom < 1.0:
            raise ValueError("headroom 은 0 이상 1 미만이어야 합니다.")
        return ServiceRequirement(target_utilization=1.0 - headroom, **kw)


@dataclass
class SlaCheck:
    name: str
    unit: str
    target: float
    measured: float | None
    passed: bool
    note: str = ""


@dataclass
class SlaTradeoff:
    """SLA 를 완화하면 카드가 몇 장 절약되는가."""

    relaxed: str            # 무엇을 얼마나 완화했는지
    n_cards: int
    cards_saved: int
    users_per_card: int
    # 완화해도 더 못 가는 이유. 한 SLA 를 풀면 다른 SLA 가 한계가 되는데,
    # 그걸 안 보여주면 "2배 완화나 4배 완화나 결과가 같다" 가 설명되지 않는다.
    #
    # **리스트인 이유**: 다음 지점에서 여러 SLA 가 동시에 실패할 수 있다. 하나만 내면
    # "그것만 더 풀면 된다" 로 읽힌다. 실제로 TTFT 를 2배 완화한 지점에서 TTFT 와
    # 출력 속도가 함께 막는데, TTFT 만 보고하면 4배로 풀면 될 것처럼 보인다.
    limited_by: list[str] = field(default_factory=lambda: ["unknown"])


@dataclass
class CapacityResult:
    requirement: dict[str, Any]

    required_output_tps: float | None = None
    required_docs_per_s: float | None = None

    n_cards: int | None = None
    concurrency_per_card: int | None = None
    estimated_utilization: float | None = None

    # 카드 수를 결정한 것이 처리량인가 지연 SLA 인가
    n_cards_by_throughput: int | None = None
    n_cards_by_latency_sla: int | None = None
    binding_constraint: str = "unknown"

    confidence: Confidence = "extrapolated"
    scaling_basis: str = "linear_assumption"   # 또는 measured_curve
    scaling_efficiency: float | None = None

    evidence: dict[str, Any] = field(default_factory=dict)
    sla_checks: list[SlaCheck] = field(default_factory=list)
    sla_tradeoffs: list[SlaTradeoff] = field(default_factory=list)
    bottleneck: str = "unknown"

    iterations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    feasible: bool = True
    data_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
