"""스트리밍 추론 클라이언트와 지연 지표 추출.

표준 라이브러리 http.client 를 쓰는 이유:
- 서버/개발 머신 양쪽 패키지 구성이 달라 서드파티 의존을 피한다.
- 워커마다 커넥션 하나를 keep-alive 로 유지해 TLS 핸드셰이크가 매 요청 지연에
  섞이지 않게 한다 (호스팅 엔드포인트에서 핸드셰이크만 ~355ms — docs/03 §5).

지표 정의 (docs/02-plan.md §0, docs/03-api-findings.md §1):

    TTFT = 첫 non-empty 토큰 delta 의 수신 시각 − 요청 전송 시각
    E2E  = 마지막 delta 수신 시각 − 요청 전송 시각
    TPOT = (E2E − TTFT) / (completion_tokens − 1)      # completion_tokens >= 2 일 때만

`첫 non-empty 토큰 delta` 의 판정은 API 경로에 따라 다르다.

- /v1/completions      : choices[0].text
- /v1/chat/completions : choices[0].delta.content **또는** choices[0].delta.reasoning

추론 모델(Qwen3, gpt-oss)은 content 가 끝까지 비어 있고 reasoning 으로만 토큰을 낸다.
content 만 보면 TTFT 가 영원히 측정되지 않는다 — 조용히 실패하는 버그다.
"""

from __future__ import annotations

import http.client
import json
import ssl
import time
import uuid
from typing import Any

from .schema import RequestRecord
from .target import Target

_DONE = b"[DONE]"


def extract_delta_text(obj: dict[str, Any], is_chat: bool) -> str:
    """스트리밍 chunk 에서 이번에 도착한 토큰 텍스트를 뽑는다. 없으면 빈 문자열."""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    ch = choices[0]
    if not is_chat:
        return ch.get("text") or ""
    delta = ch.get("delta") or {}
    # content 가 비어 있어도 reasoning 에 토큰이 있을 수 있다. 둘 다 본다.
    return (delta.get("content") or "") or (delta.get("reasoning") or "")


def _usage_fields(usage: dict[str, Any] | None) -> dict[str, int | None]:
    if not usage:
        return {"prompt_tokens_actual": None, "completion_tokens_actual": None,
                "reasoning_tokens": None, "cached_tokens": None}
    ptd = usage.get("prompt_tokens_details") or {}
    ctd = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens_actual": usage.get("prompt_tokens"),
        "completion_tokens_actual": usage.get("completion_tokens"),
        "reasoning_tokens": ctd.get("reasoning_tokens"),
        "cached_tokens": ptd.get("cached_tokens"),
    }


class StreamingClient:
    """워커 스레드 하나가 소유하는 클라이언트. 스레드 간 공유하지 않는다."""

    def __init__(self, target: Target):
        self.target = target
        self._conn: http.client.HTTPConnection | None = None

    # -- 커넥션 -------------------------------------------------------------

    def _connect(self) -> http.client.HTTPConnection:
        p = self.target.parsed
        host, port = p.hostname, p.port
        if p.scheme == "https":
            return http.client.HTTPSConnection(
                host, port or 443, timeout=self.target.timeout_s,
                context=ssl.create_default_context())
        return http.client.HTTPConnection(host, port or 80, timeout=self.target.timeout_s)

    def _get_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _drop_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def close(self) -> None:
        self._drop_conn()

    # -- 요청 ---------------------------------------------------------------

    def _payload(self, prompt: str, max_tokens: int, *, stream: bool,
                 ignore_eos: bool, min_tokens: int | None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
        t = self.target
        body: dict[str, Any] = {
            "model": t.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,      # 재현성. 샘플링 난수를 측정에서 배제한다.
            "stream": stream,
        }
        if ignore_eos:
            # 출력 길이를 정확히 고정한다. 안 하면 EOS 시점이 요청마다 달라져
            # TPOT/throughput 비교가 무의미해진다 (docs/01 IMP-4).
            body["ignore_eos"] = True
            body["min_tokens"] = min_tokens if min_tokens is not None else max_tokens
        if stream:
            body["stream_options"] = {"include_usage": True}
        if t.is_chat:
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        if extra:
            body.update(extra)
        return body

    def stream_request(self, prompt: str, max_tokens: int, *, run_id: str,
                       experiment: str, concurrency: int, t0: float,
                       target_input_tokens: int | None = None,
                       is_warmup: bool = False, ignore_eos: bool = True,
                       min_tokens: int | None = None,
                       extra: dict[str, Any] | None = None) -> RequestRecord:
        """요청 1건을 스트리밍으로 보내고 RequestRecord 를 만든다. 예외는 삼키고 error 에 담는다."""
        t = self.target
        rec = RequestRecord(
            run_id=run_id, request_id=uuid.uuid4().hex[:12], experiment=experiment,
            source=t.source, is_warmup=is_warmup, concurrency=concurrency,
            target_input_tokens=target_input_tokens, target_output_tokens=max_tokens,
        )
        body = json.dumps(self._payload(prompt, max_tokens, stream=True,
                                        ignore_eos=ignore_eos, min_tokens=min_tokens,
                                        extra=extra)).encode()
        path = t.path_for(t.api_path)

        # keep-alive 커넥션이 끊긴 경우에만 1회 재연결한다.
        #
        # 단, **토큰을 이미 받기 시작한 뒤에는 재시도하지 않는다.** 재시도는 새로운
        # 요청이므로, 이전 시도의 t_first_token 을 남겨두면 TTFT 가 다른 요청의
        # 시각으로 계산된다(실제로 음수 TTFT 로 관측됨). 그런 경우는 측정 실패로
        # 기록하는 것이 정직하다.
        for attempt in (1, 2):
            rec.attempts = attempt
            _reset_timing(rec)
            rec.t_send = time.perf_counter() - t0
            try:
                conn = self._get_conn()
                conn.request("POST", path, body=body, headers=t.headers())
                resp = conn.getresponse()
                if resp.status != 200:
                    detail = resp.read()[:300].decode("utf-8", "replace")
                    rec.error = f"HTTP {resp.status}: {detail}"
                    self._drop_conn()
                    return rec
                rec.upstream_ms = _to_float(resp.getheader("x-envoy-upstream-service-time"))
                self._consume_stream(resp, rec, t0)
                if resp.getheader("Connection", "").lower() == "close":
                    self._drop_conn()
                break
            except (http.client.HTTPException, OSError) as e:
                started_receiving = rec.t_first_token is not None
                self._drop_conn()
                if attempt == 1 and not started_receiving:
                    continue
                rec.error = (f"{type(e).__name__}: {e}"
                             + (" (스트림 도중 끊김 — 재시도하지 않음)" if started_receiving else ""))
                _reset_timing(rec)
                return rec

        _finalize(rec)
        return rec

    def _consume_stream(self, resp, rec: RequestRecord, t0: float) -> None:
        """SSE 를 줄 단위로 읽으며 첫 토큰 시각과 마지막 토큰 시각을 잡는다."""
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == _DONE:
                continue
            now = time.perf_counter() - t0
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            if obj.get("usage"):
                for k, v in _usage_fields(obj["usage"]).items():
                    if v is not None:
                        setattr(rec, k, v)

            text = extract_delta_text(obj, self.target.is_chat)
            if text:
                if rec.t_first_token is None:
                    rec.t_first_token = now
                rec.t_last_token = now

            ch = (obj.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                rec.finish_reason = ch["finish_reason"]

    # -- Embedding ----------------------------------------------------------

    def embed_request(self, docs: list[str], *, run_id: str, experiment: str,
                      concurrency: int, t0: float, batch_size: int,
                      target_doc_tokens: int | None = None,
                      is_warmup: bool = False) -> RequestRecord:
        """문서 여러 건을 한 요청에 담아 임베딩한다.

        LLM 과 달리 스트리밍이 없으므로 TTFT 는 정의되지 않는다.
        용량 단위는 **documents/s** 로 통일한다 — 요청/s 와 섞으면 batch 배만큼 틀린다
        (docs/01-spec-review.md IMP-9).
        """
        t = self.target
        rec = RequestRecord(
            run_id=run_id, request_id=uuid.uuid4().hex[:12], experiment=experiment,
            source=t.source, is_warmup=is_warmup, concurrency=concurrency,
            target_input_tokens=target_doc_tokens, target_output_tokens=0,
        )
        rec.completion_tokens_actual = batch_size   # 이 레코드에서는 '처리한 문서 수'
        body = json.dumps({"model": t.model, "input": docs}).encode()
        rec.t_send = time.perf_counter() - t0
        try:
            conn = self._get_conn()
            conn.request("POST", t.path_for("/embeddings"), body=body, headers=t.headers())
            resp = conn.getresponse()
            raw = resp.read()
            now = time.perf_counter() - t0
            if resp.status != 200:
                rec.error = f"HTTP {resp.status}: {raw[:200].decode('utf-8', 'replace')}"
                self._drop_conn()
                return rec
            obj = json.loads(raw)
            usage = obj.get("usage") or {}
            rec.prompt_tokens_actual = usage.get("prompt_tokens")
            rec.t_first_token = now      # 임베딩은 단일 응답이므로 first == last
            rec.t_last_token = now
            rec.e2e_ms = (now - rec.t_send) * 1000.0
            n_returned = len(obj.get("data") or [])
            if n_returned != batch_size:
                rec.error = f"batch 불일치: 요청 {batch_size}건, 응답 {n_returned}건"
        except (http.client.HTTPException, OSError, json.JSONDecodeError) as e:
            self._drop_conn()
            rec.error = f"{type(e).__name__}: {e}"
        return rec

    # -- 비스트리밍 (토큰 길이 보정용) --------------------------------------

    def probe_tokens(self, prompt: str) -> int | None:
        """프롬프트의 실제 토큰 수를 서버에 물어본다.

        max_tokens=1 로 요청해 usage.prompt_tokens 를 읽는다.
        /tokenize 가 있는 전용 서버에서는 tokenize_count() 가 더 싸다.
        """
        t = self.target
        body = json.dumps(self._payload(prompt, 1, stream=False,
                                        ignore_eos=False, min_tokens=None)).encode()
        try:
            conn = self._get_conn()
            conn.request("POST", t.path_for(t.api_path), body=body, headers=t.headers())
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                self._drop_conn()
                return None
            return ((json.loads(raw).get("usage") or {}).get("prompt_tokens"))
        except (http.client.HTTPException, OSError, json.JSONDecodeError):
            self._drop_conn()
            return None

    def tokenize_count(self, prompt: str) -> int | None:
        """전용 서버의 POST /tokenize 로 정확한 토큰 수를 얻는다.

        호스팅 엔드포인트에는 이 경로가 없다(404) — 그때는 None 을 돌려주고
        호출부가 probe_tokens() 로 넘어간다 (docs/03 §5).
        """
        t = self.target
        body = json.dumps({"model": t.model, "prompt": prompt}).encode()
        try:
            conn = self._get_conn()
            conn.request("POST", t.root_path_for("/tokenize"), body=body, headers=t.headers())
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                # 커넥션을 버려 다음 호출이 새로 연결되게 한다. 남겨두면
                # 끊긴 커넥션으로 계속 실패해 보정이 하드 실패한다.
                self._drop_conn()
                return None
            obj = json.loads(raw)
            if isinstance(obj.get("count"), int):
                return obj["count"]
            if isinstance(obj.get("tokens"), list):
                return len(obj["tokens"])
            return None
        except (http.client.HTTPException, OSError, json.JSONDecodeError):
            self._drop_conn()
            return None


def _to_float(v: str | None) -> float | None:
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def _reset_timing(rec: RequestRecord) -> None:
    """시도마다 타이밍/usage 상태를 비운다. 이전 시도의 값이 남으면 지표가 오염된다."""
    rec.t_first_token = None
    rec.t_last_token = None
    rec.ttft_ms = rec.e2e_ms = rec.tpot_ms = None
    rec.prompt_tokens_actual = rec.completion_tokens_actual = None
    rec.reasoning_tokens = rec.cached_tokens = None
    rec.upstream_ms = None
    rec.finish_reason = None


def _finalize(rec: RequestRecord) -> None:
    """t_* 로부터 ms 지표를 계산한다. analysis 단계에서 재검산할 수 있도록 원본 시각도 남긴다."""
    if rec.t_first_token is not None:
        rec.ttft_ms = (rec.t_first_token - rec.t_send) * 1000.0
    if rec.t_last_token is not None:
        rec.e2e_ms = (rec.t_last_token - rec.t_send) * 1000.0
    # 음수는 물리적으로 불가능하다. 나오면 계측 버그이므로 값을 내지 않고 드러낸다.
    if (rec.ttft_ms is not None and rec.ttft_ms < 0) or (rec.e2e_ms is not None and rec.e2e_ms < 0):
        rec.error = (rec.error or "") + " | negative_latency: 계측 오류 (t_send 이후에 토큰이 와야 함)"
        rec.ttft_ms = rec.e2e_ms = rec.tpot_ms = None
        return
    n = rec.completion_tokens_actual
    if rec.ttft_ms is not None and rec.e2e_ms is not None and n is not None and n >= 2:
        rec.tpot_ms = (rec.e2e_ms - rec.ttft_ms) / (n - 1)
