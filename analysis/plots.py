"""실측 데이터 → 그래프.

benchmark/ 와 분리돼 있다. 측정 코드는 그림을 모르고, 이 파일은 측정을 하지 않는다.
`data/` 만 읽으므로 RNGD 없이 재현된다.

그래프가 planner 와 같은 숫자를 쓰도록, 조건 선택은 직접 하지 않고
`planner.benchmark_store` 의 판정 로직(신뢰창 → 단독 실행 → 창 길이)을 통과시킨다.

    python -m analysis.plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from planner.benchmark_store import BenchmarkStore, _quality          # noqa: E402
from planner.capacity import max_concurrency_meeting_sla, plan        # noqa: E402
from planner.models import BenchmarkRow, ServiceRequirement           # noqa: E402

MODEL = "furiosa-ai/Qwen3-8B-FP8"

# 계측기 느낌의 차분한 팔레트. GitHub 라이트/다크 양쪽에서 읽히도록 배경은 흰색으로 고정한다.
INK = "#10151C"
MUTED = "#5E6A70"
GRID = "#DDE5E3"
TEAL = "#0B6E5F"
RUST = "#A32B2B"
AMBER = "#B8860B"
SLATE = "#41607A"
SERIES = [TEAL, SLATE, AMBER, RUST]


def setup_style() -> bool:
    """한글 폰트를 잡고 공통 스타일을 건다. 한글 폰트가 없으면 False."""
    available = {f.name for f in fm.fontManager.ttflist}
    korean = None
    # 굵기(bold)를 가진 한글 폰트를 우선한다. AppleGothic 은 regular 만 있어
    # 제목 굵기가 무시된다.
    for cand in ("Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR",
                 "Malgun Gothic", "AppleGothic"):
        if cand in available:
            korean = cand
            break
    if korean:
        plt.rcParams["font.family"] = korean
    else:
        warnings.warn("한글 폰트를 찾지 못했습니다. 라벨이 깨져 보일 수 있습니다.")
    plt.rcParams.update({
        "axes.unicode_minus": False,          # 한글 폰트는 유니코드 마이너스가 없다
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10.5,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })
    return korean is not None


def _clean(ax, *, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="-", alpha=0.9)
    ax.set_axisbelow(True)


def load_store() -> BenchmarkStore:
    path = os.path.join(REPO_ROOT, "data", "processed", "benchmark_rows.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["rows"]
    known = set(BenchmarkRow.__dataclass_fields__)
    return BenchmarkStore([BenchmarkRow(**{k: v for k, v in d.items() if k in known})
                           for d in raw])


def _summary(run_glob: str) -> list[dict]:
    import glob
    hits = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "raw", run_glob, "summary.json")))
    if not hits:
        return []
    with open(hits[-1], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. prefill 계단

def plot_prefill_steps(out: str) -> str | None:
    """입력 길이 vs TTFT. 이 프로젝트의 대표 그림."""
    rows = _summary("*_A3_*")
    if not rows:
        return None
    pts = sorted((r["prompt_tokens_median"], r["ttft_ms_p50"])
                 for r in rows if r["ttft_ms_p50"] is not None)
    xs, ys = zip(*pts)

    # 실측 토큰 수로 재정렬해 확인한 평탄 구간 (계단은 128·256·1024·1280 네 곳).
    # 아티팩트에 선언된 384·512·640·768·896 은 실제 비용 경계가 아니다.
    plateaus = [(56, 127, "~20 ms"), (128, 255, "~26 ms"),
                (257, 1024, "~78 ms"), (1025, 1280, "~95–105 ms")]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, (lo, hi, label) in enumerate(plateaus):
        ax.axvspan(lo, hi, color=TEAL if i % 2 == 0 else SLATE, alpha=0.055)
    ax.plot(xs, ys, "-o", color=TEAL, lw=1.9, ms=4.5, zorder=3)

    ax.annotate("224 tok · 26.4 ms", (224, 26.4), textcoords="offset points",
                xytext=(-8, 14), ha="right", fontsize=8.5, color=MUTED)
    ax.annotate("896 tok · 77.4 ms", (896, 77.4), textcoords="offset points",
                xytext=(0, -26), ha="center", fontsize=8.5, color=MUTED)
    ax.annotate("같은 비용 구간\n257 – 1,024 tok", (560, 78.6),
                textcoords="offset points", xytext=(0, 26), ha="center",
                fontsize=9.5, color=TEAL, fontweight="bold")
    ax.annotate("경계를 넘으면\n토큰 몇 개 차이로 +24%", (1024, 95.4),
                textcoords="offset points", xytext=(30, -6), ha="left",
                fontsize=8.5, color=RUST,
                arrowprops=dict(arrowstyle="->", color=RUST, lw=1.1))

    ax.set_xscale("log")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("입력 길이 (tokens, 로그 스케일)")
    ax.set_ylabel("첫 토큰까지 TTFT (ms)")
    ax.set_title("Prefill 비용은 토큰 수에 비례하지 않고 계단형이다")
    ax.text(0.5, -0.24, "RNGD 1장 · Qwen3-8B-FP8 · 동시성 1 · 출력 1토큰 · 조건당 30건",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    _clean(ax)
    p = os.path.join(out, "01-prefill-steps.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 2. 동시성 절벽

def plot_concurrency_cliff(store: BenchmarkStore, out: str) -> str:
    curve = store.concurrency_curve(MODEL, 512, 128)
    c = [r.concurrency_per_card for r in curve]
    tps = [r.aggregate_output_tps for r in curve]
    p95 = [r.ttft_ms_p95 for r in curve]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    a1.plot(c, tps, "-o", color=TEAL, lw=1.9, ms=5)
    peak = max(range(len(tps)), key=lambda i: tps[i])
    a1.annotate(f"최대 {tps[peak]:,.0f} tok/s", (c[peak], tps[peak]),
                textcoords="offset points", xytext=(-14, -26), ha="right",
                fontsize=9, color=MUTED)
    a1.set_ylabel("처리량 (출력 tokens/s)")
    a1.set_title("처리량은 완만하게 포화한다", fontsize=11.5)

    a2.plot(c, p95, "-o", color=RUST, lw=1.9, ms=5)
    a2.set_yscale("log")
    a2.set_ylabel("첫 토큰까지 p95 (ms, 로그 스케일)")
    a2.set_title("응답 지연은 절벽에서 무너진다", fontsize=11.5)
    a2.axhspan(0, 300, color=TEAL, alpha=0.07)
    a2.text(0.97, 0.06, "대화형 서비스 허용 범위 (p95 ≤ 300 ms)", transform=a2.transAxes,
            ha="right", fontsize=8.5, color=TEAL)

    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(c)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("카드당 동시 요청")
        ax.axvspan(64, 128, color=RUST, alpha=0.07)
        _clean(ax)
    a1.text(88, min(tps) + (max(tps) - min(tps)) * 0.06, "절벽", ha="center",
            fontsize=9, color=RUST, fontweight="bold")

    fig.suptitle("최대 처리량 지점은 운영할 수 없는 지점이다", fontsize=13,
                 fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.04, "RNGD 1장 · 입력 512 / 출력 128 토큰 · 조건당 100건 이상 "
             "· 동시성 64 이상은 세션 2 에서 단독 재측정 — 기존 값과 처리량 0~7% 차",
             ha="center", fontsize=8.5, color=MUTED)
    p = os.path.join(out, "02-concurrency-cliff.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 3. 카드 확장

def plot_card_scaling(store: BenchmarkStore, out: str) -> str:
    rows = [r for r in store.rows
            if r.model == MODEL and r.input_tokens == 512 and r.output_tokens == 128]
    best: dict[tuple[int, int], BenchmarkRow] = {}
    for r in rows:
        k = (r.concurrency_per_card, r.n_cards)
        if k not in best or _quality(r) > _quality(best[k]):
            best[k] = r

    cards = [1, 2, 4, 8]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for i, cc in enumerate([4, 16, 32]):
        have = [n for n in cards if (cc, n) in best]
        if 1 not in have:          # 1장 기준이 없으면 효율을 낼 수 없다
            continue
        base = best[(cc, 1)].aggregate_output_tps
        a1.plot(have, [best[(cc, n)].aggregate_output_tps for n in have],
                "-o", color=SERIES[i], lw=1.9, ms=5, label=f"카드당 {cc}")
        a2.plot(have, [best[(cc, n)].aggregate_output_tps / (base * n) * 100 for n in have],
                "-o", color=SERIES[i], lw=1.9, ms=5, label=f"카드당 {cc}")

    ideal_base = best[(32, 1)].aggregate_output_tps
    a1.plot(cards, [ideal_base * n for n in cards], "--", color=MUTED, lw=1.2,
            label="완전 선형 (카드당 32 기준)")
    a1.set_ylabel("총 처리량 (출력 tokens/s)")
    a1.set_title("총 처리량", fontsize=11.5)
    a1.legend()

    a2.axhline(100, color=MUTED, lw=1.1, ls="--")
    a2.set_ylim(70, 112)
    a2.set_ylabel("확장 효율 (%)")
    a2.set_title("카드당 효율 — 동시성에 따라 다르다", fontsize=11.5)
    a2.annotate("카드당 동시성이 높을 때만\n8장에서 82%로 떨어진다",
                (8, 82.1), textcoords="offset points", xytext=(-12, 20),
                ha="right", fontsize=8.5, color=RUST,
                arrowprops=dict(arrowstyle="->", color=RUST, lw=1.1))
    a2.legend(loc="lower left")

    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(cards)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}장"))
        ax.set_xlabel("RNGD 카드 수 (data parallel)")
        _clean(ax)

    fig.suptitle("카드 확장은 사실상 선형이지만, 그건 측정해야 아는 것이다",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.04, "입력 512 / 출력 128 토큰 · 카드당 동시성을 고정하고 카드 수만 변경",
             ha="center", fontsize=8.5, color=MUTED)
    p = os.path.join(out, "03-card-scaling.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 4. 지속 부하

def plot_soak(out: str) -> str | None:
    import glob
    hits = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "raw", "*_B4soak2_*",
                                         "requests.jsonl")))
    if not hits:
        return None
    with open(hits[-1], encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    m = [r for r in recs if not r["is_warmup"] and r["error"] is None and r["t_last_token"]]
    if not m:
        return None

    bucket = 60.0
    agg: dict[int, list] = {}
    for r in m:
        agg.setdefault(int(r["t_last_token"] // bucket), []).append(r)
    ks = sorted(agg)[:-1]                      # 마지막 구간은 부분 구간이라 제외
    mins = [k * bucket / 60 for k in ks]
    tps = [sum(x["completion_tokens_actual"] or 0 for x in agg[k]) / bucket for k in ks]

    def pct(vals, q):
        s = sorted(vals)
        return s[max(0, math.ceil(len(s) * q) - 1)]

    p50 = [pct([x["ttft_ms"] for x in agg[k] if x["ttft_ms"]], 0.50) for k in ks]
    p95 = [pct([x["ttft_ms"] for x in agg[k] if x["ttft_ms"]], 0.95) for k in ks]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.15]})
    a1.plot(mins, tps, "-", color=TEAL, lw=1.9)
    a1.set_ylim(0, max(tps) * 1.42)
    a1.set_ylabel("처리량 (tok/s)")
    a1.set_title("32.7분 지속 부하 — 처리량 계단은 폴링이 아니라 같은 카드를 나눠 쓴 결과")

    # 26분 지점 = 같은 카드에 붙어 있던 두 번째 클라이언트(B4soak2)가 끝난 시각.
    # 예전에는 이 지점을 furiosa-smi 폴링 루프의 종료 시각으로 적었는데, 세션 2 의
    # 통제 실험에서 폴러 2개의 비용이 0.5% 로 나와 그 귀속이 반증됐다.
    jump = 26.0
    for ax in (a1, a2):
        ax.axvline(jump, color=RUST, lw=1.2, ls="--", alpha=0.8)
    a1.axvspan(0, jump, color=RUST, alpha=0.05)
    a1.annotate("같은 카드에 클라이언트 2개\n(카드가 받는 동시성 32)",
                (jump, max(tps) * 1.30), ha="right", va="top", fontsize=8.5,
                color=RUST, xytext=(-8, 0), textcoords="offset points")
    a1.annotate(f"{tps[-2]:.0f} tok/s\n단독 구간 — 독립 측정 702~707 과 일치", (mins[-2], tps[-2]),
                ha="left", va="center", fontsize=8.5, color=TEAL,
                xytext=(-52, -26), textcoords="offset points")
    _clean(a1)

    a2.plot(mins, p95, "-", color=RUST, lw=1.7, label="TTFT p95")
    a2.plot(mins, p50, "-", color=SLATE, lw=1.7, label="TTFT p50")
    a2.set_ylim(0, max(p95) * 1.7)
    a2.set_ylabel("첫 토큰까지 (ms)")
    a2.set_xlabel("경과 시간 (분)")
    a2.legend(ncol=2)
    _clean(a2)

    fig.text(0.5, -0.02,
             f"RNGD 1장 · 입력 512 / 출력 128 · 라벨 동시성 16 · 측정 {len(m):,}건 · "
             "단독 구간 51.9–55.6 °C / 145–159 W (스로틀 카운터 미수집으로 확인 불가)\n"
             "TTFT 는 전이 전후로 159→155 ms 로 거의 변화 없으나 TPOT 은 32.8→21.9 ms 로 바뀐다 — "
             "카드가 받던 동시성이 32 에서 16 으로 줄어든 것이다 (세션 2 통제 실험으로 재현)",
             ha="center", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(hspace=0.18)
    p = os.path.join(out, "04-soak-stability.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 5. 서비스 규모 vs 카드 수

TIERS = [("대화형-빠름", 300.0, 30.0, TEAL),
         ("대화형-보통", 1500.0, 15.0, SLATE),
         ("배치", 1e9, 5.0, AMBER)]


def plot_cards_vs_users(store: BenchmarkStore, out: str) -> str:
    users = [50, 100, 200, 400, 700, 1000, 1500, 2000, 3000]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    for label, ttft, tps, color in TIERS:
        ys = []
        for u in users:
            r = plan(store, ServiceRequirement(
                workload="llm_chat", model=MODEL, concurrent_users=u,
                avg_input_tokens=512, avg_output_tokens=128,
                target_output_tps_per_user=tps, target_max_ttft_ms=ttft,
                target_utilization=0.7))
            ys.append(r.n_cards if r.feasible and r.n_cards else float("nan"))
        ax.plot(users, ys, "-o", color=color, lw=1.9, ms=4.5,
                label=f"{label} (p95 ≤{ttft/1000:.1f}s, {tps:.0f} tok/s)"
                if ttft < 1e8 else f"{label} (지연 제한 없음, {tps:.0f} tok/s)")

    ax.set_xlabel("동시 사용자 수")
    ax.set_ylabel("필요 RNGD 카드 수")
    ax.set_title("같은 사용자 수라도 SLA에 따라 카드가 몇 배 차이 난다")
    ax.legend(loc="upper left")
    ax.text(0.5, -0.24,
            "입력 512 / 출력 128 토큰 · 목표 이용률 70% · 실측 확장 곡선 사용",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    _clean(ax)
    p = os.path.join(out, "05-cards-vs-users.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- 6. SLA 완화의 대가

def plot_sla_tradeoff(store: BenchmarkStore, out: str) -> str:
    """무엇이 카드 수를 줄이는가 — 두 변수를 분리해서 본다.

    원래 이 그림은 "첫 응답 목표를 0.3초에서 1.5초로 늦추면 카드가 절반" 이라는 제목이었다.
    그런데 두 등급은 TTFT 와 사용자당 속도를 **동시에** 바꾼다. 속도를 고정하고 TTFT 만
    완화하면 카드 수가 전혀 줄지 않는다. 효과를 TTFT 에 귀속한 것이 틀렸다.
    """
    users = 1000
    speeds = [30.0, 15.0]
    ttfts = [300.0, 1500.0]
    grid = []
    for tps in speeds:
        row = []
        for ttft in ttfts:
            r = plan(store, ServiceRequirement(
                workload="llm_chat", model=MODEL, concurrent_users=users,
                avg_input_tokens=512, avg_output_tokens=128,
                target_output_tps_per_user=tps, target_max_ttft_ms=ttft,
                target_utilization=0.7))
            row.append(r.n_cards if r.feasible and r.n_cards else float("nan"))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = [0, 1]
    for i, (tps, row) in enumerate(zip(speeds, grid)):
        ax.plot(x, row, "-o", color=SERIES[i], lw=2.2, ms=9,
                label=f"사용자당 {tps:.0f} tok/s")
        for xi, v in zip(x, row):
            ax.annotate(f"{v:.0f}장", (xi, v), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=11,
                        fontweight="bold", color=SERIES[i])

    ax.annotate("", xy=(0.5, grid[1][0]), xytext=(0.5, grid[0][0]),
                arrowprops=dict(arrowstyle="<->", color=RUST, lw=1.6))
    ax.text(0.55, (grid[0][0] + grid[1][0]) / 2,
            f"속도를 절반으로 낮추면\n{grid[0][0] - grid[1][0]:.0f}장 감소",
            fontsize=9.5, color=RUST, va="center")
    ax.text(0.5, grid[0][0] + max(grid[0]) * 0.10,
            "TTFT 를 5배 완화해도 변화 없음", ha="center", fontsize=9.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([f"TTFT p95 ≤{t/1000:.1f}s" for t in ttfts])
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(0, max(max(g) for g in grid) * 1.25)
    ax.set_ylabel("필요 RNGD 카드 수")
    ax.set_title("카드를 줄이는 것은 첫 응답이 아니라 사용자당 생성 속도다")
    ax.legend(loc="center left")
    ax.text(0.5, -0.16, f"동시 사용자 {users:,}명 · 입력 512 / 출력 128 토큰 "
            "· 목표 이용률 70% · 지연·처리량 제약 모두 반영",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    _clean(ax)
    p = os.path.join(out, "06-sla-tradeoff.png")
    fig.savefig(p); plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="실측 데이터 → 그래프")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "analysis", "plots"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if not setup_style():
        print("경고: 한글 폰트 없음 — 라벨이 깨질 수 있습니다.")
    store = load_store()

    made = [
        plot_prefill_steps(args.out),
        plot_concurrency_cliff(store, args.out),
        plot_card_scaling(store, args.out),
        plot_soak(args.out),
        plot_cards_vs_users(store, args.out),
        plot_sla_tradeoff(store, args.out),
    ]
    for p in made:
        if p:
            print(f"  {os.path.relpath(p, REPO_ROOT)}  ({os.path.getsize(p)//1024} KB)")
    missing = sum(1 for p in made if p is None)
    if missing:
        print(f"  건너뜀 {missing}개 (원본 데이터 없음)")
    print("\n미측정: Embedding batch size vs 처리량 — 다음 세션 과제")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
