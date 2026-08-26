"""서버 지표 수집.

전용 서버는 Prometheus 텍스트를 GET /metrics 로 노출한다. 여기서 얻는 값이
벤치마크 신뢰성 검증의 핵심이다 (docs/00-environment.md §4).

    running_samples        실제 동시 처리량 — 의도한 concurrency 에 도달했는지
    waiting_samples        큐 대기 — 배칭이 아니라 큐잉이면 해석이 달라진다
    kv_cache_usage         메모리 용량 병목의 직접 증거
    prefix_cache_hit_rate  측정 오염 검증

metric 의 정확한 이름은 네이티브 모듈이 만들어내므로 **런타임에 확인**해야 한다
(docs/00 미해결 항목 3). 그래서 이름을 하드코딩하지 않고 전부 담아둔 뒤
find() 로 부분 문자열 검색해서 쓴다.

호스팅 엔드포인트에는 /metrics 가 없다(404). 그때는 스스로 비활성화한다.
"""

from __future__ import annotations

import http.client
import re
import ssl
import threading
import time
from typing import Any, Callable

from .target import Target

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus 텍스트를 {키: 값} 으로 만든다. 키는 라벨을 포함한 원문 그대로."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        key = m.group("name") + (m.group("labels") or "")
        out[key] = value
    return out


def find(sample: dict[str, float], needle: str) -> dict[str, float]:
    """부분 문자열로 metric 을 찾는다. 이름이 확정되기 전까지의 안전장치."""
    n = needle.lower()
    return {k: v for k, v in sample.items() if n in k.lower()}


class MetricsPoller:
    """백그라운드 스레드로 /metrics 를 주기적으로 긁는다."""

    def __init__(self, target: Target, sink: Callable[[dict[str, Any]], None],
                 interval_s: float = 1.0):
        self.target = target
        self.sink = sink
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available: bool | None = None      # None = 아직 모름
        self.samples: list[dict[str, Any]] = []
        self.note: str | None = None

    def _fetch(self) -> str | None:
        p = self.target.parsed
        try:
            if p.scheme == "https":
                conn = http.client.HTTPSConnection(p.hostname, p.port or 443, timeout=5,
                                                   context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(p.hostname, p.port or 80, timeout=5)
            headers = {}
            if self.target.api_key:
                headers["Authorization"] = f"Bearer {self.target.api_key}"
            conn.request("GET", self.target.root_path_for("/metrics"), headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            conn.close()
            if resp.status != 200:
                return None
            return raw.decode("utf-8", "replace")
        except (http.client.HTTPException, OSError):
            return None

    def _loop(self, t0: float) -> None:
        while not self._stop.is_set():
            text = self._fetch()
            if text is None:
                if self.available is None:
                    self.available = False
                    self.note = "/metrics 를 사용할 수 없습니다 (호스팅 엔드포인트에는 없음)."
                    return          # 없는 경로를 계속 두드리지 않는다
            else:
                self.available = True
                sample = {"t": time.perf_counter() - t0, "metrics": parse_prometheus(text)}
                self.samples.append(sample)
                self.sink(sample)
            self._stop.wait(self.interval)

    def start(self, t0: float) -> None:
        self._thread = threading.Thread(target=self._loop, args=(t0,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # -- 검증에 쓰는 요약 ---------------------------------------------------

    def peak(self, needle: str, t_from: float | None = None,
             t_to: float | None = None) -> float | None:
        """해당 metric 의 최댓값.

        t_from/t_to 를 주면 **그 구간의 샘플만** 본다. 조건별 검증에는 반드시 구간을
        지정해야 한다. run 전체를 훑으면 앞 조건(예: concurrency 64)의 값이 뒤 조건
        (concurrency 1)의 검증에 새어 들어와 누적 최댓값이 되어버린다.
        """
        vals: list[float] = []
        for s in self.samples:
            t = s.get("t", 0.0)
            if t_from is not None and t < t_from:
                continue
            if t_to is not None and t > t_to:
                continue
            vals.extend(find(s["metrics"], needle).values())
        return max(vals) if vals else None

    def peak_sum(self, needle: str, t_from: float | None = None,
                 t_to: float | None = None) -> float | None:
        """샘플마다 **라벨을 가로질러 합산**한 뒤 그 최댓값을 돌려준다.

        data parallel 배포에서는 metric 이 엔진별로 따로 나온다
        (`furiosa_llm_num_requests_running{engine="0"}`, `{engine="1"}`, ...).
        동시 처리량을 보려면 엔진별 값을 더해야 한다. max 를 쓰면
        dp=8 에서 총 동시성 256 을 '한 엔진의 32' 와 비교하게 되어
        항상 큐잉으로 오판한다.
        """
        totals: list[float] = []
        for s in self.samples:
            t = s.get("t", 0.0)
            if t_from is not None and t < t_from:
                continue
            if t_to is not None and t > t_to:
                continue
            vals = find(s["metrics"], needle).values()
            if vals:
                totals.append(sum(vals))
        return max(totals) if totals else None

    def last(self, needle: str) -> dict[str, float]:
        return find(self.samples[-1]["metrics"], needle) if self.samples else {}
