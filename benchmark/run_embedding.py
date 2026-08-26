"""Embedding 벤치마크 (D1 티어).

LLM 과 다른 점:
- 스트리밍이 없다 → TTFT 는 정의되지 않고, 요청 지연과 처리량만 본다
- 용량 단위는 **documents/s** 로 통일한다 (docs/01-spec-review.md IMP-9).
  요청/s 와 섞으면 batch 배만큼 틀린 답이 나온다.
- 변수는 batch size(요청 하나에 담는 문서 수)와 문서 길이다.

사용 예:

    python -m benchmark.run_embedding --target local \
      --model furiosa-ai/Qwen3-Embedding-0.6B --base-url http://127.0.0.1:8007/v1 \
      --config benchmark/configs/d1_embedding.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
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


def _repo_relative(path: str) -> str:
    """설정 파일 경로를 저장소 기준 상대 경로로 만든다.

    절대 경로는 다른 사람에게 재현 정보가 되지 못하고, 실행한 사람의 계정명·
    머신 이름을 결과 파일에 남긴다. 저장소 밖 경로는 파일명만 남긴다.
    """
    ap = os.path.abspath(path)
    try:
        rel = os.path.relpath(ap, REPO_ROOT)
    except ValueError:
        return os.path.basename(ap)
    return rel if not rel.startswith("..") else os.path.basename(ap)


@dataclass
class EmbeddingSpec:
    experiment: str
    batch_size: int = 1
    doc_tokens: int = 128
    concurrency: int = 1
    warmup_requests: int = 4
    measured_requests: int = 40
    min_samples_for_p95: int = 100

    def label(self) -> str:
        return (f"{self.experiment} batch={self.batch_size} "
                f"doc={self.doc_tokens}tok conc={self.concurrency}")


@dataclass
class EmbeddingResult:
    spec: dict[str, Any]
    source: str
    n_measured_ok: int = 0
    n_error: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    window_s: float | None = None
    docs_per_s: float | None = None
    requests_per_s: float | None = None
    tokens_per_s: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    prompt_tokens_median: float | None = None
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_condition(target, factory: PromptFactory, spec: EmbeddingSpec, *,
                  run_id: str, on_record, t0: float,
                  poller: MetricsPoller | None = None) -> EmbeddingResult:
    total = spec.warmup_requests + spec.measured_requests
    lock = threading.Lock()
    state = {"dispatched": 0}
    records: list[RequestRecord] = []
    factory.calibrate(spec.doc_tokens)

    def worker() -> None:
        client = StreamingClient(target)
        try:
            while True:
                with lock:
                    if state["dispatched"] >= total:
                        return
                    idx = state["dispatched"]
                    state["dispatched"] += 1
                docs = [factory.make(spec.doc_tokens) for _ in range(spec.batch_size)]
                rec = client.embed_request(
                    docs, run_id=run_id, experiment=spec.experiment,
                    concurrency=spec.concurrency, t0=t0, batch_size=spec.batch_size,
                    target_doc_tokens=spec.doc_tokens,
                    is_warmup=idx < spec.warmup_requests)
                with lock:
                    records.append(rec)
                    on_record(rec)
        finally:
            client.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(spec.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return _summarize(spec, target, records)


def _summarize(spec: EmbeddingSpec, target, records: list[RequestRecord]) -> EmbeddingResult:
    res = EmbeddingResult(spec=asdict(spec), source=target.source)
    measured = [r for r in records if not r.is_warmup]
    ok = [r for r in measured if r.error is None and r.t_last_token is not None]
    bad = [r for r in measured if r.error is not None]
    res.n_measured_ok = len(ok)
    res.n_error = len(bad)
    for r in bad:
        key = (r.error or "")[:60]
        res.errors[key] = res.errors.get(key, 0) + 1
    if not ok:
        res.validation["fatal"] = "성공한 측정 요청이 없습니다."
        return res

    window_start = min(r.t_send for r in ok)
    window_end = max(r.t_last_token for r in ok)   # type: ignore[type-var]
    window = window_end - window_start
    res.window_s = window
    if window > 0:
        res.docs_per_s = sum(r.completion_tokens_actual or 0 for r in ok) / window
        res.requests_per_s = len(ok) / window
        toks = sum(r.prompt_tokens_actual or 0 for r in ok)
        res.tokens_per_s = toks / window if toks else None

    lat = [r.e2e_ms for r in ok if r.e2e_ms is not None]
    res.latency_ms_p50 = _median(lat)
    if len(ok) >= spec.min_samples_for_p95:
        res.latency_ms_p95 = _pct(lat, 0.95)
    else:
        res.validation["p95"] = (
            f"insufficient_samples: 측정 성공 {len(ok)}건 < 기준 {spec.min_samples_for_p95}건")

    res.prompt_tokens_median = _median([float(r.prompt_tokens_actual or 0) for r in ok])
    per_doc = (res.prompt_tokens_median or 0) / max(1, spec.batch_size)
    res.validation["doc_length"] = {
        "target_per_doc": spec.doc_tokens,
        "observed_per_doc": round(per_doc, 1),
        "verdict": "ok" if abs(per_doc - spec.doc_tokens) <= max(2, spec.doc_tokens * 0.05) else "DRIFTED",
    }
    res.validation["source"] = res.source
    res.validation["capacity_usable"] = res.source == "measured_local"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RNGD Embedding 벤치마크")
    ap.add_argument("--target", choices=["local", "hosted"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--metrics-interval", type=float, default=1.0)
    ap.add_argument("--cards", type=int, default=None,
                    help="이 서버가 쓰는 RNGD 카드 수")
    ap.add_argument("--devices", default=None)
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    kw: dict[str, Any] = {"api_path": "/embeddings", "timeout_s": args.timeout}
    if args.target == "local" and args.base_url:
        kw["base_url"] = args.base_url
    target = build_target(args.target, args.model, **kw)

    defaults = cfg.get("defaults", {})
    specs = [EmbeddingSpec(experiment=cfg["experiment"], **{**defaults, **c})
             for c in cfg["conditions"]]

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.model.replace('/', '_')}_{cfg['experiment']}_{stamp}"
    meta = RunMeta(
        run_id=run_id, experiment=cfg["experiment"], source=target.source,
        started_at=dt.datetime.now().astimezone().isoformat(),
        target={"kind": target.name, "base_url": target.base_url, "model": target.model,
                "api_path": target.api_path,
                "n_cards": args.cards, "devices": args.devices},
        config={"file": _repo_relative(args.config), "description": cfg.get("description"),
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
        return probe.tokenize_count(text)

    factory = PromptFactory(count_tokens)

    print(f"run_id : {run_id}")
    print(f"target : {target.name} ({target.source})  {target.base_url}")
    print(f"model  : {target.model}")
    print()

    t0 = time.perf_counter()
    poller.start(t0)
    results = []
    try:
        for i, spec in enumerate(specs, 1):
            print(f"[{i}/{len(specs)}] {spec.label()} ... ", end="", flush=True)
            started = time.perf_counter()
            res = run_condition(target, factory, spec, run_id=run_id,
                                on_record=on_record, t0=t0, poller=poller)
            took = time.perf_counter() - started
            results.append(res)
            if res.docs_per_s is None:
                print(f"실패 {res.errors}")
            else:
                p95 = f"{res.latency_ms_p95:.0f}" if res.latency_ms_p95 else "n/a"
                print(f"{took:5.1f}s  {res.docs_per_s:8.1f} docs/s  "
                      f"{res.requests_per_s:6.1f} req/s  "
                      f"lat p50={res.latency_ms_p50:7.1f}ms p95={p95:>7}  "
                      f"ok={res.n_measured_ok} err={res.n_error}")
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


if __name__ == "__main__":
    sys.exit(main())
