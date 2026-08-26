"""필요한 RNGD 수 산정.

두 제약을 **따로** 계산한 뒤 더 큰 쪽을 택한다. 처리량만 보고 "충분하다"고 하지
않기 위해서다 (docs/01-spec-review.md §3).

    n_by_latency    SLA 를 지키면서 카드 하나가 받을 수 있는 최대 동시 사용자로부터
    n_by_throughput 필요 처리량 / (카드당 처리량 × 확장효율 × target_utilization)
    n = max(둘)

n_by_throughput 은 순환 의존이 있어 고정점 반복으로 푼다. 카드당 처리량은 카드당
동시성에 따라 달라지고, 카드당 동시성은 카드 수가 정해져야 나오기 때문이다
(docs/01-spec-review.md CRIT-5).

target_utilization 은 **분모에 한 번만** 적용한다 (IMP-10).
"""

from __future__ import annotations

import math
from typing import Any

from .benchmark_store import BenchmarkStore, DataSourceError
from .models import (
    BenchmarkRow, CapacityResult, ServiceRequirement, SlaCheck, SlaTradeoff, worst,
)

MAX_ITERATIONS = 20


def _interp_rows(curve: list[BenchmarkRow], concurrency: int) -> tuple[BenchmarkRow, str]:
    """concurrency 곡선에서 요청 지점의 값을 얻는다."""
    if not curve:
        raise DataSourceError("concurrency 곡선이 비어 있습니다.")
    exact = [r for r in curve if r.concurrency_per_card == concurrency]
    if exact:
        return exact[0], "measured"
    lo = [r for r in curve if r.concurrency_per_card < concurrency]
    hi = [r for r in curve if r.concurrency_per_card > concurrency]
    if not lo:
        return curve[0], "extrapolated"
    if not hi:
        return curve[-1], "extrapolated"
    a, b = lo[-1], hi[0]
    t = (concurrency - a.concurrency_per_card) / (b.concurrency_per_card - a.concurrency_per_card)

    def mix(x, y):
        return None if x is None or y is None else x + t * (y - x)

    return BenchmarkRow(
        model=a.model, source=a.source, n_cards=1, concurrency_per_card=concurrency,
        input_tokens=a.input_tokens, output_tokens=a.output_tokens,
        aggregate_output_tps=mix(a.aggregate_output_tps, b.aggregate_output_tps),
        per_user_output_tps=mix(a.per_user_output_tps, b.per_user_output_tps),
        ttft_ms_p50=mix(a.ttft_ms_p50, b.ttft_ms_p50),
        ttft_ms_p95=mix(a.ttft_ms_p95, b.ttft_ms_p95),
        e2e_ms_p95=mix(a.e2e_ms_p95, b.e2e_ms_p95),
        tpot_ms_p50=mix(a.tpot_ms_p50, b.tpot_ms_p50),
        kv_cache_peak=mix(a.kv_cache_peak, b.kv_cache_peak),
        waiting_peak=mix(a.waiting_peak, b.waiting_peak),
        n_samples=min(a.n_samples, b.n_samples),
        # 보간 결과의 신뢰도는 두 근거 중 나쁜 쪽을 따른다
        window_s=min([w for w in (a.window_s, b.window_s) if w is not None], default=None),
        exclusive=a.exclusive and b.exclusive,
        run_id=f"interp({a.run_id}@{a.concurrency_per_card},{b.run_id}@{b.concurrency_per_card})",
    ), "interpolated"


def sla_checks(row: BenchmarkRow, req: ServiceRequirement) -> list[SlaCheck]:
    checks = [
        SlaCheck("TTFT p95", "ms", req.target_max_ttft_ms, row.ttft_ms_p95,
                 row.ttft_ms_p95 is not None and row.ttft_ms_p95 <= req.target_max_ttft_ms,
                 "" if row.ttft_ms_p95 is not None else "측정 표본 부족으로 P95 없음"),
        SlaCheck("사용자당 출력 속도", "tok/s", req.target_output_tps_per_user,
                 row.per_user_output_tps,
                 row.per_user_output_tps is not None
                 and row.per_user_output_tps >= req.target_output_tps_per_user),
    ]
    if req.target_p95_e2e_ms is not None:
        checks.append(SlaCheck("요청 완료 p95", "ms", req.target_p95_e2e_ms, row.e2e_ms_p95,
                               row.e2e_ms_p95 is not None
                               and row.e2e_ms_p95 <= req.target_p95_e2e_ms))
    return checks


def sla_feasible_rows(curve: list[BenchmarkRow],
                      req: ServiceRequirement) -> list[BenchmarkRow]:
    return [r for r in curve if all(c.passed for c in sla_checks(r, req))]


def max_concurrency_meeting_sla(curve: list[BenchmarkRow],
                                req: ServiceRequirement) -> int | None:
    """SLA 를 만족하는 **최대 카드당 동시성**. 없으면 None.

    카드 수의 하한(사용자를 몇 장에 나눠야 하는가)을 정하는 데 쓴다.
    """
    ok = sla_feasible_rows(curve, req)
    return max((r.concurrency_per_card for r in ok), default=None)


def best_operating_point(curve: list[BenchmarkRow],
                         req: ServiceRequirement) -> BenchmarkRow | None:
    """SLA 를 만족하는 조건 중 **처리량이 가장 높은** 지점.

    동시성을 최대로 올린 지점과 다를 수 있다. 실측에서 카드당 동시성 128 이
    1,459 tok/s 인데 256 은 1,310 tok/s 로 오히려 낮았다 — 동시성을 더 올리면
    처리량이 떨어지는 구간이 존재한다. 용량 치트시트에는 이 지점을 쓴다.
    """
    ok = sla_feasible_rows(curve, req)
    return max(ok, key=lambda r: r.aggregate_output_tps) if ok else None


def diagnose_bottleneck(row: BenchmarkRow, curve: list[BenchmarkRow]) -> str:
    """무엇이 먼저 한계에 닿았는지. 규칙에 안 맞으면 unknown 을 그대로 낸다."""
    if row.kv_cache_peak is not None and row.kv_cache_peak >= 0.9:
        return "memory_capacity"
    if (row.waiting_peak is not None and row.waiting_peak > row.concurrency_per_card * 0.5):
        return "concurrency_scheduling"

    higher = [r for r in curve if r.concurrency_per_card > row.concurrency_per_card]
    if higher and row.aggregate_output_tps:
        nxt = higher[0]
        conc_ratio = nxt.concurrency_per_card / max(1, row.concurrency_per_card)
        tps_ratio = nxt.aggregate_output_tps / row.aggregate_output_tps
        if tps_ratio >= 1 + 0.6 * (conc_ratio - 1):
            return "headroom"                 # 아직 포화 전
        if tps_ratio <= 1.1:
            return "throughput_saturated"     # 동시성을 올려도 처리량이 안 는다
        return "throughput_soft_saturation"
    return "unknown"


def plan(store: BenchmarkStore, req: ServiceRequirement) -> CapacityResult:
    res = CapacityResult(requirement=req.__dict__.copy())
    res.data_source = store.data_source
    if store.scaling_issues:
        res.warnings.append(
            f"로드 시점 무결성 검사에서 확장 효율 이상 {len(store.scaling_issues)}건을 "
            "발견했습니다. 해당 모델의 확장 곡선은 계산에 쓰이지 않습니다."
        )
    if store.rejected:
        res.warnings.append(
            f"출처 게이팅으로 {len(store.rejected)}개 측정을 제외했습니다: "
            + " / ".join(sorted({r["reason"] for r in store.rejected}))
        )
    if store.data_source == "mock":
        res.warnings.append(
            "⚠️ MOCK 데이터입니다. 실제 RNGD 측정이 아니므로 결과를 실측으로 제시하지 마세요."
        )

    out_tokens, out_level = store.nearest_output_tokens(req.model, req.avg_output_tokens)
    if out_tokens != req.avg_output_tokens:
        res.warnings.append(
            f"출력 길이 {req.avg_output_tokens} tok 의 측정이 없어 {out_tokens} tok 측정을 사용합니다."
        )
    curve, in_level, notes = store.resolve_input_slice(req.model, req.avg_input_tokens, out_tokens)
    res.warnings.extend(notes)
    if not curve:
        res.feasible = False
        res.warnings.append("해당 조건의 실측 곡선이 없습니다.")
        return res

    shared = [r for r in curve if not r.exclusive]
    if shared:
        res.warnings.append(
            "다른 측정과 동시에 실행된 조건이 근거에 포함돼 있습니다 (동시성 "
            + ", ".join(str(r.concurrency_per_card) for r in shared)
            + "). 부하 생성기가 호스트 자원을 나눠 쓰므로 처리량이 실제보다 "
            "낮게 나왔을 수 있습니다 — 보수적인 방향의 오차입니다."
        )

    short = store.short_window_rows(curve)
    if short:
        res.warnings.append(
            "측정창이 짧은 조건이 근거에 포함돼 있습니다 (동시성 "
            + ", ".join(str(r.concurrency_per_card) for r in short)
            + f"). 창이 {store.MIN_TRUSTED_WINDOW_S:.0f}초 미만이면 시작 시 요청이 몰리는 구간이 "
            "지연 통계를 지배해 P95가 실제보다 나쁘게 나옵니다."
        )

    res.required_output_tps = req.concurrent_users * req.target_output_tps_per_user

    # --- 1) 지연 SLA 제약 ---------------------------------------------------
    c_max = max_concurrency_meeting_sla(curve, req)
    if c_max is None:
        res.feasible = False
        res.binding_constraint = "latency_sla_infeasible"
        res.sla_checks = sla_checks(curve[0], req)
        res.warnings.append(
            "실측한 모든 동시성 조건에서 SLA를 만족하지 못했습니다. "
            "가장 낮은 동시성에서도 목표 지연을 넘습니다 — SLA를 완화하거나 "
            "입력/출력 길이를 줄여야 합니다."
        )
        res.evidence = curve[0].evidence()
        res.confidence = worst(in_level, out_level, "extrapolated")
        return res
    res.n_cards_by_latency_sla = math.ceil(req.concurrent_users / c_max)
    grid_max = max(r.concurrency_per_card for r in curve)
    if c_max == grid_max:
        res.warnings.append(
            f"SLA를 만족하는 최대 카드당 동시성이 실측 격자의 상한({grid_max})과 같습니다. "
            f"더 높은 동시성에서도 SLA를 지킬 수 있는지는 측정하지 않았으므로, "
            f"필요 카드 수가 과대 추정됐을 수 있습니다."
        )

    # --- 2) 처리량 제약: 고정점 반복 ---------------------------------------
    n = 1
    conc_level = "measured"
    for _ in range(MAX_ITERATIONS):
        c_per_card = max(1, math.ceil(req.concurrent_users / n))
        row, conc_level = _interp_rows(curve, c_per_card)
        eff, basis = store.scaling_efficiency(req.model, n, c_per_card)
        usable = row.aggregate_output_tps * eff * req.target_utilization
        n_new = max(1, math.ceil(res.required_output_tps / usable)) if usable else n
        res.iterations.append({
            "n_cards": n, "concurrency_per_card": c_per_card,
            "row_aggregate_output_tps": round(row.aggregate_output_tps, 1),
            "scaling_efficiency": round(eff, 4), "scaling_basis": basis,
            "usable_tps_per_card": round(usable, 1), "n_cards_next": n_new,
        })
        if n_new == n:
            break
        n = n_new
    else:
        res.warnings.append(
            f"카드 수 계산이 {MAX_ITERATIONS}회 안에 수렴하지 않았습니다. "
            f"보수적으로 큰 값을 채택합니다."
        )
        n = max(n, n_new)
    res.n_cards_by_throughput = n

    # --- 3) 결합 -----------------------------------------------------------
    res.n_cards = max(res.n_cards_by_latency_sla, res.n_cards_by_throughput)
    if res.n_cards_by_latency_sla > res.n_cards_by_throughput:
        res.binding_constraint = "latency_sla"
    elif res.n_cards_by_throughput > res.n_cards_by_latency_sla:
        res.binding_constraint = "throughput"
    else:
        res.binding_constraint = "both"

    res.concurrency_per_card = max(1, math.ceil(req.concurrent_users / res.n_cards))
    final_row, lvl = _interp_rows(curve, res.concurrency_per_card)
    eff, basis = store.scaling_efficiency(req.model, res.n_cards, res.concurrency_per_card)
    res.scaling_efficiency = round(eff, 4)
    res.scaling_basis = basis
    if basis == "linear_assumption":
        res.warnings.append(
            "다중 카드 실측이 없어 **카드 수에 비례해 성능이 는다고 가정**했습니다. "
            "실제로는 라우팅 오버헤드와 카드당 배치 감소로 낮아질 수 있습니다."
        )
    elif basis == "linear_assumption_scaling_data_rejected":
        detail = "; ".join(
            f"카드당 동시성 {i['concurrency_per_card']}·{i['n_cards']}장에서 "
            f"{i['efficiency'] * 100:.0f}%"
            for i in store.scaling_issues if i["model"] == req.model
        )
        res.warnings.append(
            "⚠️ **이 모델의 확장 곡선을 사용하지 않았습니다.** 물리적으로 불가능한 효율이 "
            f"계산되어 데이터를 신뢰할 수 없습니다 ({detail}). 선형 가정으로 대체했으므로 "
            "카드 수가 실제와 다를 수 있습니다. 집계(기준 행 선택, 조건 묶기)를 확인하세요."
        )

    total_capacity = res.n_cards * final_row.aggregate_output_tps * eff
    res.estimated_utilization = (res.required_output_tps / total_capacity) if total_capacity else None
    res.evidence = final_row.evidence()
    res.evidence["scaling_efficiency"] = res.scaling_efficiency
    res.evidence["scaling_basis"] = basis
    res.sla_checks = sla_checks(final_row, req)
    res.bottleneck = diagnose_bottleneck(final_row, curve)
    res.confidence = worst(in_level, out_level, lvl, conc_level)
    res.sla_tradeoffs = _tradeoffs(store, curve, req, res)
    if final_row.n_samples and final_row.n_samples < 100:
        res.warnings.append(
            f"근거 측정의 표본이 {final_row.n_samples}건입니다. P95 신뢰도가 낮습니다."
        )
    return res


def _tradeoffs(store: BenchmarkStore, curve: list[BenchmarkRow],
               req: ServiceRequirement, res: CapacityResult) -> list[SlaTradeoff]:
    """SLA 를 완화하면 카드가 몇 장 절약되는가 — 의사결정에 직접 쓰이는 출력."""
    out: list[SlaTradeoff] = []
    if res.n_cards is None:
        return out

    variants = [
        ("TTFT p95 목표 2배 완화", {"target_max_ttft_ms": req.target_max_ttft_ms * 2}),
        ("TTFT p95 목표 4배 완화", {"target_max_ttft_ms": req.target_max_ttft_ms * 4}),
        ("사용자당 출력 속도 30% 완화",
         {"target_output_tps_per_user": req.target_output_tps_per_user * 0.7}),
    ]
    for label, override in variants:
        relaxed = ServiceRequirement(**{**req.__dict__, **override})
        c_max = max_concurrency_meeting_sla(curve, relaxed)
        if c_max is None:
            continue
        n_lat = math.ceil(relaxed.concurrent_users / c_max)
        required = relaxed.concurrent_users * relaxed.target_output_tps_per_user
        row, _ = _interp_rows(curve, max(1, math.ceil(relaxed.concurrent_users / max(1, n_lat))))
        eff, _ = store.scaling_efficiency(relaxed.model, n_lat, c_max)
        usable = row.aggregate_output_tps * eff * relaxed.target_utilization
        n_thr = max(1, math.ceil(required / usable)) if usable else n_lat
        n = max(n_lat, n_thr)
        out.append(SlaTradeoff(relaxed=label, n_cards=n,
                               cards_saved=res.n_cards - n,
                               users_per_card=c_max,
                               limited_by=_next_limit(curve, relaxed, c_max)))
    return out


def _next_limit(curve: list[BenchmarkRow], req: ServiceRequirement,
                c_max: int) -> list[str]:
    """완화 후 그 다음 동시성에서 걸리는 것 **전부**.

    하나만 돌려주면 "그것만 더 풀면 된다" 로 읽힌다. 실제로는 여러 SLA 가 동시에
    막는 경우가 있고, 그때 하나만 보여주면 오도한다.
    """
    higher = [r for r in curve if r.concurrency_per_card > c_max]
    if not higher:
        return ["measurement_grid"]        # 실측을 더 높은 동시성에서 하지 않았다
    failed = [c.name for c in sla_checks(higher[0], req) if not c.passed]
    return failed or ["unknown"]


def format_result(res: CapacityResult) -> str:
    """사람이 읽는 요약. UI 는 이걸 쓰거나 to_dict() 로 직접 렌더링한다."""
    lines: list[str] = []
    src = {"measured_local": "실측(전용 RNGD 서버)", "mock": "⚠️ MOCK",
           "mixed": "⚠️ 혼합", "empty": "데이터 없음"}.get(res.data_source, res.data_source)
    lines.append(f"데이터 출처 : {src}")
    if not res.feasible:
        lines.append("판정        : 실측 범위에서 이 조건을 만족할 수 없습니다")
    else:
        lines.append(f"필요 카드   : {res.n_cards}장  (지연 SLA {res.n_cards_by_latency_sla}장 / "
                     f"처리량 {res.n_cards_by_throughput}장 → 결정 요인: {res.binding_constraint})")
        lines.append(f"카드당 동시 사용자 : {res.concurrency_per_card}")
        if res.estimated_utilization is not None:
            lines.append(f"예상 이용률 : {res.estimated_utilization*100:.1f}%")
        lines.append(f"필요 처리량 : {res.required_output_tps:.1f} tok/s")
        lines.append(f"확장 근거   : {res.scaling_basis} (효율 {res.scaling_efficiency})")
        lines.append(f"예상 병목   : {res.bottleneck}")
    lines.append(f"신뢰도      : {res.confidence}")
    lines.append("")
    lines.append("SLA 점검")
    for c in res.sla_checks:
        mark = "OK  " if c.passed else "FAIL"
        m = "n/a" if c.measured is None else f"{c.measured:.1f}"
        lines.append(f"  [{mark}] {c.name}: 목표 {c.target:.1f}{c.unit} / 실측 {m}{c.unit} {c.note}")
    if res.sla_tradeoffs:
        lines.append("")
        lines.append("SLA 완화의 대가")
        for t in res.sla_tradeoffs:
            sign = f"-{t.cards_saved}장" if t.cards_saved > 0 else "변화 없음"
            lines.append(f"  {t.relaxed}: {t.n_cards}장 ({sign}), 카드당 {t.users_per_card}명"
                         f" — 여기서 더 못 가는 이유: {', '.join(t.limited_by)}")
    if res.evidence:
        lines.append("")
        lines.append(f"근거 측정   : {res.evidence}")
    if res.warnings:
        lines.append("")
        lines.append("경고")
        for w in res.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
