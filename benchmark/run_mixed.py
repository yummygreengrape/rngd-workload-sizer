"""B3 — 혼합 부하 간섭 측정.

실제 서비스는 짧은 대화 요청과 긴 문서 요청이 **섞여** 들어옵니다. 지금까지의 측정은
둘을 분리해서 쟀기 때문에 이 질문에 답할 수 없습니다.

    긴 문서 하나가 들어올 때 다른 사용자의 응답이 얼마나 느려지는가?

이 하드웨어에서 특히 중요한 이유가 있습니다. prefill 버킷은 batch=1 뿐이고
decode 버킷은 batch 256 까지 있습니다(docs/00-environment.md §6). prefill 이
온NPU 배칭되지 않는다면 긴 프롬프트 하나가 decode 흐름을 끊을 수 있습니다.
**가설이며, 이 실험이 확인 대상입니다.**

설계 판단 세 가지
------------------

**1. 배경 부하와 주입 요청을 어떻게 나눠 집계하는가**

같은 run 안에서 `role` 로 구분합니다(`background` / `injected`). 배경 처리량 집계에서
주입 요청의 토큰은 제외합니다 — 섞으면 "긴 요청이 처리량을 올렸다" 는 착시가 생깁니다.

**2. 주입 시점을 어떻게 정하는가**

고정 간격은 배경 부하와 위상이 맞아버릴 위험이 있습니다. 배경 요청 주기의 정수배로
주입되면 매번 같은 타이밍에 끼어들어 편향이 생깁니다. **지수분포 간격(포아송 도착)**
을 씁니다. 평균 간격만 지정하고 실제 간격은 매번 다릅니다.

**3. '간섭' 을 무엇으로 정의하는가 — 두 기준을 함께 냅니다**

주입 전후 비교는 시간 드리프트(캐시 워밍, 열, 다른 프로세스)와 뒤섞입니다.
대신 **같은 run 안에서 요청 단위로 나눕니다.** 같은 시간대에 섞여 있으므로 드리프트가
양쪽에 똑같이 걸립니다.

그런데 나누는 기준을 하나만 쓰면 **편향 방향을 숨기게 됩니다.**

- **생존 겹침 기준** (`overlapped_injections`) — 요청의 생존 구간이 주입과 겹쳤는가.
  **길이 편향이 있습니다.** 느려진 요청은 더 오래 살아 있으므로 겹칠 확률이 높아집니다.
  주입이 포아송(율 λ)이면 지속시간 d 인 요청이 겹칠 확률이 대략 1−e^(−λd) 라,
  간섭이 전혀 없어도 겹친 그룹이 느린 쪽으로 치우칩니다. → **간섭을 과대평가**합니다.
- **시작 시각 기준** (`started_during_injection`) — 요청이 주입 활성 구간에 시작했는가.
  요청 자신의 지속시간에 의존하지 않아 길이 편향이 없습니다. 대신 주입 직전에 시작해
  길게 도는 요청의 간섭을 놓칩니다. → **간섭을 과소평가**합니다.

둘을 함께 내면 참값이 그 사이에 있습니다. 어느 쪽이든 단일 숫자로 내면 편향 방향이
숨습니다.

> 주입의 **prefill 구간에만** 겹치는지로 좁히는 방법도 있지만 쓰지 않습니다.
> 간섭 기전이 prefill 이라는 가정을 정의에 넣는 것이라 순환이 됩니다.
> 넓게 재고 나서 "간섭이 주입의 prefill 구간에 집중되는가" 를 **결과로** 보이는 편이 낫습니다.

사용 예::

    python -m benchmark.run_mixed --target local \\
      --model furiosa-ai/Qwen3-8B-FP8 --base-url http://127.0.0.1:8000/v1 \\
      --config benchmark/configs/b3_mixed_load.json --cards 1 --devices npu:0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import env as env_mod
from .client import StreamingClient
from .metrics import MetricsPoller
from .prompts import PromptFactory
from .runner import _median, _pct
from .schema import RequestRecord, RunMeta, RunWriter
from .target import build_target

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class MixedSpec:
    """배경 부하 위에 긴 프롬프트를 주입하는 조건 하나."""

    experiment: str

    # 배경: 대화형 트래픽
    background_concurrency: int = 16
    background_input_tokens: int = 512
    background_output_tokens: int = 128

    # 주입: 긴 문서 요청
    inject_input_tokens: int = 4096
    inject_output_tokens: int = 1        # prefill 비용만 보려면 1
    mean_inject_interval_s: float = 10.0  # 포아송 평균 간격
    inject_concurrency: int = 1           # 동시에 떠 있는 주입 요청 수

    duration_s: float = 180.0
    warmup_s: float = 20.0
    min_samples_for_p95: int = 100

    def label(self) -> str:
        return (f"{self.experiment} 배경 conc={self.background_concurrency}"
                f"/in={self.background_input_tokens} · "
                f"주입 in={self.inject_input_tokens}/평균 {self.mean_inject_interval_s:.0f}s")


@dataclass
class MixedResult:
    spec: dict[str, Any]
    source: str

    n_background: int = 0
    n_injected: int = 0
    n_error: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    window_s: float | None = None
    background_output_tps: float | None = None      # 주입 토큰 제외
    injected_count_in_window: int = 0

    # 핵심 지표 — 겹친 것과 안 겹친 것을 나눠 본다
    clean_n: int = 0
    disturbed_n: int = 0
    clean_tpot_p50: float | None = None
    disturbed_tpot_p50: float | None = None
    clean_ttft_p50: float | None = None
    disturbed_ttft_p50: float | None = None
    clean_ttft_p95: float | None = None
    disturbed_ttft_p95: float | None = None

    # 생존 겹침 기준 — 길이 편향으로 **과대평가** 쪽
    tpot_penalty_pct: float | None = None
    ttft_penalty_pct: float | None = None
    # 시작 시각 기준 — 길이 편향 없음, 놓치는 요청이 있어 **과소평가** 쪽
    start_clean_n: int = 0
    start_disturbed_n: int = 0
    tpot_penalty_pct_start_based: float | None = None
    ttft_penalty_pct_start_based: float | None = None

    injected_ttft_p50: float | None = None          # 주입 요청 자신의 지연

    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def annotate_overlap(records: list[RequestRecord]) -> None:
    """배경 요청에 두 가지 겹침 판정을 채운다.

    - `overlapped_injections` — 생존 구간이 겹친 주입 수. **길이 편향 있음**(과대평가)
    - `started_during_injection` — 주입 활성 구간에 시작했는가. 길이 편향 없음(과소평가)

    두 기준의 결과가 참값을 사이에 두는 구간을 만든다.
    """
    injected = [(r.t_send, r.t_last_token) for r in records
                if r.role == "injected" and r.t_last_token is not None]
    for r in records:
        if r.role != "background" or r.t_last_token is None:
            continue
        r.overlapped_injections = sum(
            1 for s, e in injected if _overlaps(r.t_send, r.t_last_token, s, e))
        # 시작 시각만 본다 — 이 요청이 얼마나 오래 걸렸는지와 무관하다
        r.started_during_injection = any(s <= r.t_send < e for s, e in injected)


def run_condition(target, factory: PromptFactory, spec: MixedSpec, *,
                  run_id: str, on_record, t0: float,
                  poller: MetricsPoller | None = None) -> MixedResult:
    lock = threading.Lock()
    records: list[RequestRecord] = []
    stop = threading.Event()
    # 주입 간격 난수는 스레드마다 독립적인 시드로. 전역 random 을 공유하면
    # 다른 워커의 프롬프트 생성과 상태를 나눠 쓴다.
    rng = random.Random(uuid.uuid4().int)

    factory.calibrate(spec.background_input_tokens)
    factory.calibrate(spec.inject_input_tokens)

    def record(rec: RequestRecord, role: str) -> None:
        rec.role = role
        with lock:
            records.append(rec)
            on_record(rec)

    def background_worker() -> None:
        client = StreamingClient(target)
        try:
            while not stop.is_set():
                rec = client.stream_request(
                    factory.make(spec.background_input_tokens),
                    spec.background_output_tokens,
                    run_id=run_id, experiment=spec.experiment,
                    concurrency=spec.background_concurrency, t0=t0,
                    target_input_tokens=spec.background_input_tokens,
                    is_warmup=(time.perf_counter() - t0) < spec.warmup_s)
                record(rec, "background")
        finally:
            client.close()

    def injector() -> None:
        client = StreamingClient(target)
        try:
            while not stop.is_set():
                # 포아송 도착. 고정 간격이면 배경 부하와 위상이 맞아버린다.
                if stop.wait(rng.expovariate(1.0 / spec.mean_inject_interval_s)):
                    return
                rec = client.stream_request(
                    factory.make(spec.inject_input_tokens),
                    spec.inject_output_tokens,
                    run_id=run_id, experiment=spec.experiment,
                    concurrency=spec.inject_concurrency, t0=t0,
                    target_input_tokens=spec.inject_input_tokens,
                    is_warmup=(time.perf_counter() - t0) < spec.warmup_s)
                record(rec, "injected")
        finally:
            client.close()

    threads = [threading.Thread(target=background_worker, daemon=True)
               for _ in range(spec.background_concurrency)]
    threads += [threading.Thread(target=injector, daemon=True)
                for _ in range(spec.inject_concurrency)]
    for t in threads:
        t.start()
    stop.wait(spec.warmup_s + spec.duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=30)

    annotate_overlap(records)
    return _summarize(spec, target, records, poller)


def _summarize(spec: MixedSpec, target, records: list[RequestRecord],
               poller: MetricsPoller | None) -> MixedResult:
    res = MixedResult(spec=asdict(spec), source=target.source)

    measured = [r for r in records if not r.is_warmup]
    ok = [r for r in measured if r.error is None and r.t_last_token is not None]
    bad = [r for r in measured if r.error is not None]
    res.n_error = len(bad)
    for r in bad:
        key = (r.error or "")[:60]
        res.errors[key] = res.errors.get(key, 0) + 1

    bg = [r for r in ok if r.role == "background"]
    inj = [r for r in ok if r.role == "injected"]
    res.n_background, res.n_injected = len(bg), len(inj)
    if not bg:
        res.validation["fatal"] = "성공한 배경 요청이 없습니다."
        return res

    window_start = min(r.t_send for r in bg)
    window_end = max(r.t_last_token for r in bg)      # type: ignore[type-var]
    res.window_s = window_end - window_start
    if res.window_s > 0:
        # 배경 처리량에 주입 토큰을 섞지 않는다. 섞으면 "긴 요청이 처리량을 올렸다" 는
        # 착시가 생긴다.
        res.background_output_tps = sum(
            r.completion_tokens_actual or 0 for r in bg) / res.window_s
    res.injected_count_in_window = len(inj)

    clean = [r for r in bg if r.overlapped_injections == 0]
    disturbed = [r for r in bg if r.overlapped_injections > 0]
    res.clean_n, res.disturbed_n = len(clean), len(disturbed)

    # 시작 시각 기준 (길이 편향 없음)
    started_in = [r for r in bg if r.started_during_injection]
    started_out = [r for r in bg if not r.started_during_injection]
    res.start_clean_n, res.start_disturbed_n = len(started_out), len(started_in)

    def stat(rows, attr, q=None):
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        if not vals:
            return None
        return _pct(vals, q) if q else _median(vals)

    res.clean_tpot_p50 = stat(clean, "tpot_ms")
    res.disturbed_tpot_p50 = stat(disturbed, "tpot_ms")
    res.clean_ttft_p50 = stat(clean, "ttft_ms")
    res.disturbed_ttft_p50 = stat(disturbed, "ttft_ms")
    res.injected_ttft_p50 = stat(inj, "ttft_ms")

    if len(clean) >= spec.min_samples_for_p95:
        res.clean_ttft_p95 = stat(clean, "ttft_ms", 0.95)
    if len(disturbed) >= spec.min_samples_for_p95:
        res.disturbed_ttft_p95 = stat(disturbed, "ttft_ms", 0.95)

    if res.clean_tpot_p50 and res.disturbed_tpot_p50:
        res.tpot_penalty_pct = (res.disturbed_tpot_p50 / res.clean_tpot_p50 - 1) * 100
    if res.clean_ttft_p50 and res.disturbed_ttft_p50:
        res.ttft_penalty_pct = (res.disturbed_ttft_p50 / res.clean_ttft_p50 - 1) * 100

    a, b = stat(started_out, "tpot_ms"), stat(started_in, "tpot_ms")
    if a and b:
        res.tpot_penalty_pct_start_based = (b / a - 1) * 100
    a, b = stat(started_out, "ttft_ms"), stat(started_in, "ttft_ms")
    if a and b:
        res.ttft_penalty_pct_start_based = (b / a - 1) * 100

    _validate(res, spec, bg, inj, clean, disturbed, poller)
    return res


def _validate(res: MixedResult, spec: MixedSpec, bg, inj, clean, disturbed,
              poller: MetricsPoller | None) -> None:
    v = res.validation

    # 두 집단이 모두 충분해야 비교가 성립한다. 주입이 너무 잦으면 clean 이 사라지고,
    # 너무 드물면 disturbed 가 사라진다.
    v["group_sizes"] = {
        "clean": res.clean_n, "disturbed": res.disturbed_n,
        "verdict": ("ok" if min(res.clean_n, res.disturbed_n) >= 30
                    else "INSUFFICIENT — 주입 간격을 조정해 양쪽 표본을 확보하세요"),
    }

    # 주입이 실제로 일어났는가
    v["injection"] = {
        "count": res.n_injected,
        "expected_approx": round((spec.duration_s / spec.mean_inject_interval_s)
                                 * spec.inject_concurrency, 1),
        "verdict": "ok" if res.n_injected >= 5 else "TOO_FEW",
    }

    # prefix cache 오염
    pairs = [(r.cached_tokens, r.prompt_tokens_actual) for r in bg + inj
             if r.cached_tokens is not None and r.prompt_tokens_actual]
    if pairs:
        worst = max(c / p for c, p in pairs)
        v["prefix_cache"] = {"max_cached_ratio": round(worst, 4),
                             "verdict": "clean" if worst <= 0.05 else "CONTAMINATED"}

    # 겹침 판정이 실제로 갈렸는가 — 전부 겹쳤거나 전부 안 겹쳤으면 비교 불가
    v["overlap_split"] = {
        "max_overlaps": max((r.overlapped_injections for r in bg), default=0),
        "verdict": "ok" if res.clean_n and res.disturbed_n else "DEGENERATE",
    }

    v["source"] = res.source
    v["capacity_usable"] = res.source == "measured_local"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RNGD 혼합 부하 간섭 측정 (B3)")
    ap.add_argument("--target", choices=["local", "hosted"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--metrics-interval", type=float, default=1.0)
    ap.add_argument("--cards", type=int, default=None,
                    help="이 서버가 쓰는 RNGD 카드 수. capacity 계산의 분모다")
    ap.add_argument("--devices", default=None)
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    kw: dict[str, Any] = {"api_path": "/completions", "timeout_s": args.timeout}
    if args.target == "local" and args.base_url:
        kw["base_url"] = args.base_url
    target = build_target(args.target, args.model, **kw)

    defaults = cfg.get("defaults", {})
    specs = [MixedSpec(experiment=cfg["experiment"], **{**defaults, **c})
             for c in cfg["conditions"]]

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.model.replace('/', '_')}_{cfg['experiment']}_{stamp}"
    meta = RunMeta(
        run_id=run_id, experiment=cfg["experiment"], source=target.source,
        started_at=dt.datetime.now().astimezone().isoformat(),
        target={"kind": target.name, "base_url": target.base_url, "model": target.model,
                "api_path": target.api_path, "n_cards": args.cards,
                "devices": args.devices},
        config={"file": os.path.relpath(os.path.abspath(args.config), REPO_ROOT),
                "description": cfg.get("description"),
                "conditions": [asdict(s) for s in specs]},
        environment=env_mod.collect(git_root=REPO_ROOT),
        notes=list(args.note),
    )
    writer = RunWriter(args.out, meta)
    lock = threading.Lock()

    def on_record(rec):
        with lock:
            writer.write_request(rec)

    def on_metric(sample):
        with lock:
            writer.write_metric(sample)

    poller = MetricsPoller(target, on_metric, interval_s=args.metrics_interval)
    probe = StreamingClient(target)

    def count_tokens(text: str) -> int | None:
        n = probe.tokenize_count(text)
        return n if n is not None else probe.probe_tokens(text)

    factory = PromptFactory(count_tokens)

    print(f"run_id : {run_id}")
    print(f"target : {target.name} ({target.source})  {target.base_url}")
    print(f"model  : {target.model}   카드 {args.cards} ({args.devices})")
    print()

    t0 = time.perf_counter()
    poller.start(t0)
    results = []
    try:
        for i, spec in enumerate(specs, 1):
            print(f"[{i}/{len(specs)}] {spec.label()}")
            res = run_condition(target, factory, spec, run_id=run_id,
                                on_record=on_record, t0=t0, poller=poller)
            results.append(res)
            _print(res)
    finally:
        poller.stop()
        probe.close()
        meta.config["prompt_calibration"] = factory.summary()
        meta.environment["metrics_available"] = bool(poller.available)
        if poller.note:
            meta.notes.append(poller.note)
        writer.write_meta()
        with open(os.path.join(writer.dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        writer.close()
    print(f"\n결과 : {writer.dir}")
    return 0


def _print(res: MixedResult) -> None:
    if res.background_output_tps is None:
        print(f"      실패 {res.errors}")
        return
    print(f"      배경 {res.n_background}건 / 주입 {res.n_injected}건 "
          f"/ 에러 {res.n_error}  창 {res.window_s:.0f}s")
    print(f"      배경 처리량 {res.background_output_tps:7.1f} tok/s (주입 토큰 제외)")
    print(f"      {'':14}{'겹치지 않음':>12}{'겹침':>12}{'차이':>10}")
    for name, a, b, pen in [
        ("TPOT p50 (ms)", res.clean_tpot_p50, res.disturbed_tpot_p50, res.tpot_penalty_pct),
        ("TTFT p50 (ms)", res.clean_ttft_p50, res.disturbed_ttft_p50, res.ttft_penalty_pct),
    ]:
        if a is None or b is None:
            continue
        print(f"      {name:<14}{a:>12.1f}{b:>12.1f}{pen:>9.1f}%")
    print(f"      표본 {res.clean_n} / {res.disturbed_n}  "
          f"· 주입 요청 자신의 TTFT p50 {res.injected_ttft_p50 or float('nan'):.0f}ms")
    if res.tpot_penalty_pct is not None and res.tpot_penalty_pct_start_based is not None:
        lo = min(res.tpot_penalty_pct, res.tpot_penalty_pct_start_based)
        hi = max(res.tpot_penalty_pct, res.tpot_penalty_pct_start_based)
        print(f"      TPOT 간섭 추정 구간 {lo:.1f}% ~ {hi:.1f}%  "
              f"(시작시각 기준 {res.tpot_penalty_pct_start_based:.1f}% = 과소, "
              f"생존겹침 기준 {res.tpot_penalty_pct:.1f}% = 과대)")
    g = res.validation.get("group_sizes", {})
    if g.get("verdict", "").startswith("INSUFF"):
        print(f"      ! {g['verdict']}")


if __name__ == "__main__":
    raise SystemExit(main())
