"""LLM 벤치마크 실행기 (E1~E5).

사용 예:

    # 전용 서버 (furiosa-llm serve 가 떠 있어야 함)
    python -m benchmark.run_llm --target local --model furiosa-ai/Qwen3-8B-FP8 \
        --config benchmark/configs/e4_decode_concurrency.json

    # 호스팅 엔드포인트 (하네스 개발/검증 전용 — capacity 계산 불가)
    set -a; . _work/.env; set +a
    python -m benchmark.run_llm --target hosted --model furiosa-ai/Qwen3-32B-FP8 \
        --config benchmark/configs/e4_decode_concurrency.json --scale 0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from dataclasses import asdict

from . import env as env_mod
from .client import StreamingClient
from .metrics import MetricsPoller
from .prompts import PromptFactory
from .runner import ConditionSpec, run_condition
from .schema import RunMeta, RunWriter
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


def make_token_counter(client: StreamingClient):
    """토큰 수 세는 함수를 고른다.

    전용 서버: POST /tokenize (정확하고 싸다)
    호스팅   : /tokenize 가 404 이므로 max_tokens=1 probe 요청으로 usage.prompt_tokens

    한 번의 실패로 전체 실행이 죽지 않게 한다. keep-alive 커넥션이 끊기면
    tokenize 가 일시적으로 None 을 돌려주는데, 이걸 치명적으로 다루면
    측정 도중에 하드 실패한다(실제로 A3 실행이 이렇게 죽었다).
    재연결 후 1회 재시도하고, 그래도 안 되면 probe 로 폴백한다.
    """
    mode = {"resolved": None, "fallbacks": 0, "retries": 0}

    def count(text: str) -> int | None:
        if mode["resolved"] is None:
            n = client.tokenize_count(text)
            mode["resolved"] = "tokenize" if n is not None else "probe"
            if n is not None:
                return n

        if mode["resolved"] == "tokenize":
            n = client.tokenize_count(text)
            if n is not None:
                return n
            mode["retries"] += 1
            n = client.tokenize_count(text)      # 커넥션은 이미 버려졌으므로 재연결된다
            if n is not None:
                return n
            mode["fallbacks"] += 1               # 그래도 안 되면 probe 로 간다
            return client.probe_tokens(text)

        return client.probe_tokens(text)

    return count, mode


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_specs(cfg: dict, scale: float) -> list[ConditionSpec]:
    defaults = cfg.get("defaults", {})
    specs = []
    for cond in cfg["conditions"]:
        merged = {**defaults, **cond}
        merged["experiment"] = cfg["experiment"]
        if scale != 1.0:
            merged["measured_requests"] = max(1, int(merged.get("measured_requests", 100) * scale))
            merged["warmup_requests"] = max(1, int(merged.get("warmup_requests", 4) * scale))
        specs.append(ConditionSpec(**merged))
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RNGD LLM 벤치마크 실행기")
    ap.add_argument("--target", choices=["local", "hosted"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-url", default=None, help="local 기본값: http://127.0.0.1:8000/v1")
    ap.add_argument("--api-path", default="/completions",
                    help="/completions (기본) 또는 /chat/completions")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="요청 수 배율. 스모크 실행에 0.05 등으로 줄인다")
    ap.add_argument("--metrics-interval", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="요청 하나의 소켓 타임아웃(초). 스트림이 멈추면 이 시간만큼 기다린다")
    ap.add_argument("--cards", type=int, default=None,
                    help="이 서버가 쓰는 RNGD 카드 수. capacity 계산의 분모이므로 반드시 기록한다")
    ap.add_argument("--devices", default=None,
                    help="serve 에 넘긴 --devices 문자열. 재현을 위해 그대로 저장한다")
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    kw = {"api_path": args.api_path}
    if args.target == "local" and args.base_url:
        kw["base_url"] = args.base_url
    kw["timeout_s"] = args.timeout
    target = build_target(args.target, args.model, **kw)

    specs = build_specs(cfg, args.scale)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "_")
    run_id = f"{safe_model}_{cfg['experiment']}_{stamp}"

    meta = RunMeta(
        run_id=run_id, experiment=cfg["experiment"], source=target.source,
        started_at=dt.datetime.now().astimezone().isoformat(),
        target={"kind": target.name, "base_url": target.base_url, "model": target.model,
                "api_path": target.api_path,
                "n_cards": args.cards, "devices": args.devices, "is_chat": target.is_chat},
        config={"file": _repo_relative(args.config), "scale": args.scale,
                "description": cfg.get("description"), "conditions": [asdict(s) for s in specs]},
        environment=env_mod.collect(git_root=REPO_ROOT),
        notes=list(args.note),
    )
    if target.source != "measured_local":
        meta.notes.append(
            "호스팅 엔드포인트 실행입니다. 카드 수 미상 + 멀티테넌트 + 네트워크 포함이므로 "
            "capacity 계산에 사용할 수 없습니다 (docs/03-api-findings.md §7)."
        )

    writer = RunWriter(args.out, meta)
    write_lock = threading.Lock()

    def on_record(rec):
        with write_lock:
            writer.write_request(rec)

    def on_metric(sample):
        with write_lock:
            writer.write_metric(sample)

    poller = MetricsPoller(target, on_metric, interval_s=args.metrics_interval)

    probe_client = StreamingClient(target)
    counter, counter_mode = make_token_counter(probe_client)
    factory = PromptFactory(counter)

    print(f"run_id : {run_id}")
    print(f"target : {target.name} ({target.source})  {target.base_url}")
    print(f"model  : {target.model}   api_path={target.api_path}")
    print(f"조건    : {len(specs)}개, scale={args.scale}")
    print()

    t0 = time.perf_counter()
    poller.start(t0)
    results = []
    try:
        for i, spec in enumerate(specs, 1):
            print(f"[{i}/{len(specs)}] {spec.label()} ... ", end="", flush=True)
            started = time.perf_counter()
            res = run_condition(target, factory, spec, run_id=run_id, on_record=on_record,
                                t0=t0, poller=poller)
            took = time.perf_counter() - started
            results.append(res)
            _print_line(res, took)
    finally:
        poller.stop()
        probe_client.close()
        meta.config["prompt_calibration"] = factory.summary()
        meta.config["token_counter_mode"] = counter_mode["resolved"]
        if poller.note:
            meta.notes.append(poller.note)
        meta.environment["metrics_available"] = bool(poller.available)
        writer.write_meta()
        with open(os.path.join(writer.dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        writer.close()

    print()
    print(f"결과 : {writer.dir}")
    _print_warnings(results)
    return 0


def _print_line(res, took: float) -> None:
    if res.aggregate_output_tps is None:
        print(f"실패 (에러 {res.n_error}건) {res.errors}")
        return
    p95 = f"{res.ttft_ms_p95:.0f}" if res.ttft_ms_p95 is not None else "n/a"
    print(f"{took:5.1f}s  agg={res.aggregate_output_tps:7.1f} tok/s  "
          f"TTFT p50={res.ttft_ms_p50:7.1f}ms p95={p95:>7}  "
          f"TPOT p50={res.tpot_ms_p50 or float('nan'):6.1f}ms  "
          f"ok={res.n_measured_ok} err={res.n_error}")


def _print_warnings(results) -> None:
    warned = False
    for res in results:
        v = res.validation
        msgs = []
        if v.get("prefix_cache", {}).get("verdict") == "CONTAMINATED":
            pc = v["prefix_cache"]
            msgs.append(f"prefix cache 오염 {pc['requests_over_limit']}건 "
                        f"(최대 {pc['max_cached_ratio']:.1%}) — prefill 비용이 과소평가됩니다")
        if v.get("output_length", {}).get("verdict") == "VARIES":
            msgs.append(f"출력 길이가 고정되지 않았습니다: {v['output_length']['distinct_observed']}")
        if v.get("input_length", {}).get("verdict") == "DRIFTED":
            msgs.append(f"입력 길이 편차 {v['input_length']['drift_pct']}%")
        if v.get("concurrency", {}).get("verdict") == "QUEUED_NOT_BATCHED":
            msgs.append("의도한 동시성만큼 동시에 처리되지 않았습니다 (큐잉)")
        if v.get("retries", {}).get("verdict") == "DISTURBED":
            msgs.append(f"커넥션 재수립 {v['retries']['requests_retried']}건")
        if "p95" in v:
            msgs.append(v["p95"])
        if not v.get("capacity_usable", True):
            msgs.append("capacity 계산 사용 불가")
        if msgs:
            warned = True
            sp = res.spec
            label = (f"{sp['experiment']} conc={sp['concurrency']} "
                     f"in={sp['input_tokens']} out={sp['output_tokens']}")
            print(f"  ! {label}: " + "; ".join(msgs))
    if not warned:
        print("  검증 경고 없음")


if __name__ == "__main__":
    sys.exit(main())
