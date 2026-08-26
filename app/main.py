"""ROD Capacity Planner — Streamlit UI.

**여기에는 계산이 없습니다.** 입력을 받아 `planner` 에 넘기고, 돌아온 결과를 표시할 뿐입니다.
계산 로직을 UI 에 섞지 않는 이유는 (1) 테스트가 UI 없이 돌아야 하고
(2) 화면에 보이는 숫자와 스크립트로 뽑은 숫자가 달라지면 안 되기 때문입니다.

이 화면이 반드시 보여줘야 하는 것:

- **데이터 출처** — 실측인지 합성인지. 숨기면 안 됩니다.
- **무엇이 카드 수를 결정했는지** — 지연 SLA인가 처리량인가
- **어느 측정에 근거했는지** — run_id, 표본 수, 측정창, 단독 실행 여부
- **경고** — 보간 신뢰도, 짧은 측정창, 동시 실행 오염, 격자 상한

    streamlit run app/main.py
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from planner.benchmark_store import BenchmarkStore                      # noqa: E402
from planner.capacity import plan, sla_checks                           # noqa: E402
from planner.models import BenchmarkRow, ServiceRequirement             # noqa: E402

PROCESSED = os.path.join(REPO_ROOT, "data", "processed", "benchmark_rows.json")

TEAL, RUST, SLATE, MUTED = "#0B6E5F", "#A32B2B", "#41607A", "#5E6A70"

SOURCE_LABEL = {
    "measured_local": ("실측 — 전용 RNGD 서버", "success"),
    "mock": ("합성 데이터 — 실측 아님", "error"),
    "mixed": ("실측과 합성이 섞임", "warning"),
    "empty": ("데이터 없음", "error"),
}

CONFIDENCE_LABEL = {
    "measured": ("실측 격자와 정확히 일치", "success"),
    "interpolated": ("실측 사이를 보간", "info"),
    "interpolated_across_bucket_boundary":
        ("보간 구간이 컴파일 버킷 경계를 넘음 — 성능이 계단형이라 선형 보간이 어긋날 수 있음", "warning"),
    "extrapolated": ("실측 범위 밖 — 외삽", "error"),
}

LIMIT_LABEL = {
    "measurement_grid": "더 높은 동시성을 측정하지 않았습니다 — 실측 격자의 끝",
    "unknown": "판정 불가",
}

BOTTLENECK_LABEL = {
    "headroom": "여유 있음 — 동시성을 더 올릴 수 있는 구간",
    "throughput_saturated": "처리량 포화 — 동시성을 올려도 처리량이 늘지 않음",
    "throughput_soft_saturation": "처리량 둔화 — 증가폭이 꺾이는 구간",
    "memory_capacity": "메모리 용량 — KV 캐시가 한계에 근접",
    "concurrency_scheduling": "스케줄링 — 배칭되지 못하고 큐에 쌓임",
    "unknown": "판정 불가 — 규칙에 맞는 패턴이 아님",
}


@st.cache_data(show_spinner=False)
def _read_rows(_path: str, mtime: float) -> list[dict]:
    """mtime 을 캐시 키에 넣는다.

    파일 경로만 키로 쓰면 `analysis/process.py` 를 다시 돌려도 화면이 옛 데이터를
    보여준다. 실제로 hosted 행이 7 → 19 로 바뀐 뒤에도 배너가 7 을 표시했다.
    데이터 도구가 낡은 숫자를 보여주는 것은 틀린 숫자를 보여주는 것과 같다.
    """
    with open(_path, encoding="utf-8") as f:
        return json.load(f)["rows"]


def load_rows() -> tuple[list[dict], str | None]:
    if not os.path.isfile(PROCESSED):
        return [], (f"`{os.path.relpath(PROCESSED, REPO_ROOT)}` 가 없습니다. "
                    "`python -m analysis.process` 를 먼저 실행하세요.")
    return _read_rows(PROCESSED, os.path.getmtime(PROCESSED)), None


def build_store(rows: list[dict], allow_mock: bool) -> BenchmarkStore:
    known = set(BenchmarkRow.__dataclass_fields__)
    return BenchmarkStore(
        [BenchmarkRow(**{k: v for k, v in d.items() if k in known}) for d in rows],
        allow_mock=allow_mock)


def frontier_figure(store: BenchmarkStore, req: ServiceRequirement, res):
    """실측 곡선 위에 운영점을 찍는다. 어디에 서 있는지 보여주는 게 목적이다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plots import setup_style
    setup_style()

    out_tokens, _ = store.nearest_output_tokens(req.model, req.avg_output_tokens)
    curve, _, _ = store.resolve_input_slice(req.model, req.avg_input_tokens, out_tokens)
    if not curve:
        return None

    c = [r.concurrency_per_card for r in curve]
    tps = [r.aggregate_output_tps for r in curve]
    p95 = [r.ttft_ms_p95 if r.ttft_ms_p95 is not None else float("nan") for r in curve]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    a1.plot(c, tps, "-o", color=TEAL, lw=1.8, ms=4)
    a1.set_ylabel("카드당 처리량 (tok/s)")
    a2.plot(c, p95, "-o", color=RUST, lw=1.8, ms=4)
    a2.set_yscale("log")
    a2.set_ylabel("TTFT p95 (ms)")
    a2.axhline(req.target_max_ttft_ms, color=SLATE, ls="--", lw=1.2)
    a2.text(c[0], req.target_max_ttft_ms * 1.15, "목표", fontsize=8.5, color=SLATE)

    if res.feasible and res.concurrency_per_card:
        for ax in (a1, a2):
            ax.axvline(res.concurrency_per_card, color=SLATE, lw=1.4, alpha=0.7)
        a1.text(res.concurrency_per_card, max(tps) * 0.05, " 운영점",
                fontsize=9, color=SLATE, fontweight="bold")

    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(c)
        ax.set_xticklabels([str(x) for x in c])
        ax.set_xlabel("카드당 동시 요청")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="RNGD Capacity Planner", page_icon="📐", layout="wide")
    st.title("RNGD Capacity Planner")
    st.caption("실측 벤치마크로 필요한 RNGD 카드 수를 산정합니다. "
               "모든 숫자는 근거가 되는 측정을 함께 표시합니다.")

    rows, err = load_rows()
    if err:
        st.error(err)
        return

    with st.sidebar:
        st.header("서비스 조건 (ROD)")
        allow_mock = st.toggle("합성 데이터 허용", value=False,
                               help="개발용입니다. 켜면 실측이 아닌 값이 계산에 들어갑니다.")
        store = build_store(rows, allow_mock)
        models = store.models()
        if not models:
            st.error("사용 가능한 실측 데이터가 없습니다.")
            st.stop()

        # 측정이 충실한 모델이 먼저 오게 한다. 알파벳 순으로 두면 스모크 테스트용
        # 소형 모델이 기본 선택되어, 첫 화면부터 외삽 결과가 보인다.
        coverage = {m: sum(1 for r in store.rows if r.model == m and r.n_cards == 1)
                    for m in models}
        models = sorted(models, key=lambda m: -coverage[m])
        model = st.selectbox(
            "모델", models,
            format_func=lambda m: f"{m}  ({coverage[m]}개 조건 측정)")
        lengths = store.input_lengths(model)
        st.caption(f"실측 입력 길이: {min(lengths):,} – {max(lengths):,} tokens · "
                   f"단일 카드 측정 {coverage[model]}개 조건")
        if coverage[model] < 10:
            st.warning(f"이 모델은 측정 조건이 {coverage[model]}개뿐입니다. "
                       "대부분의 입력이 외삽으로 처리되어 결과 신뢰도가 낮습니다.")

        users = st.number_input("동시 사용자 수", 1, 100_000, 1000, step=50)
        col_a, col_b = st.columns(2)
        in_tok = col_a.number_input("평균 입력 (tokens)", 1, 131_072, 512, step=64)
        out_tok = col_b.number_input("평균 출력 (tokens)", 1, 131_072, 128, step=32)

        st.divider()
        st.subheader("SLA 목표")
        tps_user = st.number_input("사용자당 출력 속도 (tok/s)", 0.1, 500.0, 15.0, step=1.0)
        ttft_ms = st.number_input("첫 토큰까지 p95 (ms)", 10.0, 120_000.0, 1500.0, step=100.0)
        use_e2e = st.checkbox("요청 완료 p95 도 제한")
        e2e_ms = st.number_input("요청 완료 p95 (ms)", 10.0, 600_000.0, 10_000.0,
                                 step=500.0, disabled=not use_e2e)

        st.divider()
        headroom = st.slider("여유분 (headroom)", 0.0, 0.6, 0.3, 0.05,
                             help="목표 이용률 = 1 − 여유분. 계산에서 분모에 한 번만 적용됩니다.")
        st.caption(f"목표 이용률 **{(1 - headroom) * 100:.0f}%**")

    req = ServiceRequirement(
        workload="llm_chat", model=model, concurrent_users=int(users),
        avg_input_tokens=int(in_tok), avg_output_tokens=int(out_tok),
        target_output_tps_per_user=float(tps_user), target_max_ttft_ms=float(ttft_ms),
        target_p95_e2e_ms=float(e2e_ms) if use_e2e else None,
        target_utilization=1.0 - headroom)

    # ---- 데이터 출처 배너 (숨기지 않는다) --------------------------------
    label, kind = SOURCE_LABEL.get(store.data_source, (store.data_source, "warning"))
    getattr(st, kind)(f"**데이터 출처: {label}** · 실측 {len(store.rows)}행"
                      + (f" · 출처 게이팅으로 거부 {len(store.rejected)}행" if store.rejected else ""))

    if store.scaling_issues:
        st.warning(
            f"**로드 시점 무결성 검사에서 {len(store.scaling_issues)}건**을 발견했습니다. "
            "해당 모델의 확장 곡선은 계산에 쓰이지 않고 선형 가정으로 대체됩니다."
        )
        with st.expander("무결성 검사 상세"):
            st.dataframe(store.scaling_issues, use_container_width=True, hide_index=True)

    try:
        res = plan(store, req)
    except Exception as e:                       # noqa: BLE001
        # 데이터 도구가 트레이스백을 띄우면 안 된다. 무엇이 왜 실패했는지 말해야 한다.
        st.error(f"**계산에 실패했습니다** — {type(e).__name__}: {e}")
        st.caption("입력 조건을 바꾸거나, `python -m analysis.process` 로 데이터를 "
                   "다시 만든 뒤 시도하세요.")
        with st.expander("상세 (디버깅용)"):
            import traceback
            st.code(traceback.format_exc())
        return

    if not res.feasible:
        st.error("### 이 조건은 실측 범위에서 달성할 수 없습니다")

        # 막다른 길로 두지 않는다. 실패한 항목의 실측값이 곧 '달성 가능한 최선' 이다.
        failed = [c for c in res.sla_checks if not c.passed and c.measured is not None]
        if failed:
            st.markdown("**이 조건에서 달성 가능한 최선** — 카드를 아무리 늘려도 이보다 낫지 않습니다")
            for c in failed:
                better = "이하" if "p95" in c.name else "이상"
                st.markdown(f"- {c.name}: **{c.measured:,.1f}{c.unit}** "
                            f"(목표 {c.target:,.1f}{c.unit} {better})")
            st.caption("카드를 늘리면 카드당 사용자가 줄어 지연이 개선되지만, "
                       "동시성 1에서의 값이 하한입니다. 입력 길이를 줄이거나 SLA를 조정하세요.")

        if res.sla_checks:
            st.subheader("가장 낮은 동시성에서의 SLA 점검")
            render_sla(res)
        with st.expander("경고 전문"):
            for w in res.warnings:
                st.write(f"- {w}")
        return

    # ---- 핵심 결과 -------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("필요 RNGD", f"{res.n_cards}장")
    c2.metric("카드당 동시 사용자", f"{res.concurrency_per_card}명")
    c3.metric("예상 이용률", f"{res.estimated_utilization * 100:.0f}%"
              if res.estimated_utilization is not None else "—")
    c4.metric("필요 처리량", f"{res.required_output_tps:,.0f} tok/s")

    binding = {"latency_sla": "지연 SLA", "throughput": "처리량",
               "both": "지연 SLA와 처리량 둘 다"}.get(res.binding_constraint,
                                                res.binding_constraint)
    st.markdown(
        f"카드 수를 결정한 것은 **{binding}** 입니다 — "
        f"지연 기준 {res.n_cards_by_latency_sla}장 / 처리량 기준 {res.n_cards_by_throughput}장 중 큰 쪽."
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("SLA 점검")
        st.caption("처리량과 **독립적으로** 확인합니다. 하나라도 실패하면 '충분'이 아닙니다.")
        render_sla(res)

        st.subheader("실측 곡선에서 운영점의 위치")
        fig = frontier_figure(store, req, res)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("이 답의 근거")
        conf_text, conf_kind = CONFIDENCE_LABEL.get(res.confidence, (res.confidence, "info"))
        getattr(st, conf_kind)(f"**신뢰도: {res.confidence}** — {conf_text}")

        ev = res.evidence
        st.markdown("**근거 측정**")
        for line in _explain_run_id(ev.get("run_id", "—")):
            st.markdown(f"- {line}")
        st.markdown(f"""
| | |
|---|---|
| 조건 | 카드 {ev.get('n_cards')}장 · 동시성 {ev.get('concurrency_per_card')} · 입력 {ev.get('input_tokens')} · 출력 {ev.get('output_tokens')} |
| 실측 처리량 | {ev.get('aggregate_output_tps')} tok/s |
| 표본 수 | {ev.get('n_samples')}건 |
| 측정창 | {f"{ev['window_s']:.0f}초" if ev.get('window_s') else '—'} |
| 단독 실행 | {'예' if ev.get('exclusive') else '아니오 — 다른 측정과 동시'} |
| 확장 효율 | {ev.get('scaling_efficiency')} ({ev.get('scaling_basis')}) |
""")
        st.markdown(f"**예상 병목** — {BOTTLENECK_LABEL.get(res.bottleneck, res.bottleneck)}")

        if res.sla_tradeoffs:
            st.subheader("SLA 를 완화하면")
            st.caption("한 조건을 풀면 다른 조건이 한계가 됩니다. "
                       "그래서 더 완화해도 결과가 같아지는 지점이 생깁니다.")
            for t in res.sla_tradeoffs:
                head = (f"**{t.n_cards}장** (**{t.cards_saved}장 절약**)"
                        if t.cards_saved > 0 else f"{t.n_cards}장 (변화 없음)")
                limits = [LIMIT_LABEL.get(x, x) for x in t.limited_by]
                joined = " · ".join(limits)
                more = "  (둘 다 막습니다 — 하나만 풀어도 못 갑니다)" if len(limits) > 1 else ""
                st.markdown(f"- {t.relaxed} → {head}, 카드당 {t.users_per_card}명  \n"
                            f"　　다음 한계: {joined}{more}")

    if res.warnings:
        st.subheader("경고")
        for w in res.warnings:
            st.warning(w)

    with st.expander("카드 수를 어떻게 구했는가 — 고정점 반복 과정"):
        st.caption("카드당 처리량은 카드당 동시성에 따라 달라지고, 카드당 동시성은 카드 수가 "
                   "정해져야 나옵니다. 그래서 수렴할 때까지 반복합니다.")
        st.dataframe(res.iterations, use_container_width=True, hide_index=True)

    with st.expander("전체 결과 (JSON) — 재현·검증용"):
        st.json(res.to_dict())


def _explain_run_id(run_id: str) -> list[str]:
    """보간된 근거는 `interp(A@4,B@8)` 형태라 한 줄로 두면 읽을 수 없다.

    어느 측정 사이를 보간했는지 나눠서 보여준다.
    """
    if not run_id.startswith("interp("):
        return [f"`{run_id}`"]
    inner = run_id[len("interp("):-1] if run_id.endswith(")") else run_id[len("interp("):]
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    out = [f"두 측정 사이를 보간했습니다 ({len(parts)}개)"]
    for p in parts:
        run, _, at = p.partition("@")
        out.append(f"`{run}`" + (f" · 동시성 {at}" if at else ""))
    return out


def render_sla(res) -> None:
    for c in res.sla_checks:
        mark = "✅" if c.passed else "❌"
        measured = "측정값 없음" if c.measured is None else f"{c.measured:,.1f}{c.unit}"
        st.markdown(f"{mark} **{c.name}** — 목표 {c.target:,.1f}{c.unit} / 실측 {measured}"
                    + (f"  \n　　{c.note}" if c.note else ""))


if __name__ == "__main__":
    main()
