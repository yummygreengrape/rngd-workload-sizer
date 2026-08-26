"""측정 레코드 스키마와 JSONL 입출력.

설계 원칙 (docs/02-plan.md §2):
- 요청 1건 = JSONL 1줄. 집계는 나중에 analysis/ 에서 한다.
- warm-up 요청도 삭제하지 않고 is_warmup=True로 남긴다.
- 모든 레코드에 source를 박아 실측/호스팅/mock이 섞이지 않게 한다.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Any

# 데이터 출처. planner는 MEASURED_LOCAL 이외를 capacity 계산에서 거부한다.
SOURCE_MEASURED_LOCAL = "measured_local"      # 전용 서버 실측 — capacity 계산에 사용 가능
SOURCE_HOSTED_ENDPOINT = "hosted_endpoint"    # 호스팅 엔드포인트 — 개발/검증 전용
SOURCE_MOCK = "mock"                          # 합성 데이터


@dataclass
class RequestRecord:
    """요청 1건의 측정 결과."""

    run_id: str
    request_id: str
    experiment: str
    source: str

    is_warmup: bool = False

    # 조건
    concurrency: int = 1
    target_input_tokens: int | None = None
    target_output_tokens: int | None = None

    # 타임스탬프 (perf_counter 기준, run 시작 시점을 0으로)
    t_send: float = 0.0
    t_first_token: float | None = None
    t_last_token: float | None = None

    # 파생 지표 (ms). t_* 로부터 계산되며 analysis 단계에서 재검산 가능하다.
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tpot_ms: float | None = None

    # 서버가 보고한 실제 토큰 수. 의도한 길이와 다를 수 있으며 그 차이를 숨기지 않는다.
    prompt_tokens_actual: int | None = None
    completion_tokens_actual: int | None = None
    reasoning_tokens: int | None = None      # 사용자에게 안 보이지만 NPU는 소비한다
    cached_tokens: int | None = None         # prefix cache 적중량. 0이 아니면 오염 신호

    # 호스팅 게이트웨이(Envoy)가 보고하는 서버측 시간. 전용 서버에는 없다.
    # 주의: 스트리밍 요청에서는 **응답 헤더까지의 시간**이지 생성 완료 시간이 아니다.
    # (실측: prompt 8192 tok 에서 upstream_ms=54 인데 TTFT=1476ms)
    # prefill 시간을 서버측 기준으로 보고 싶으면 비스트리밍 요청을 써야 한다.
    upstream_ms: float | None = None

    finish_reason: str | None = None
    error: str | None = None

    # 커넥션 재수립 횟수. 1보다 크면 측정이 방해받았다는 뜻이므로 분석에서 구분한다.
    attempts: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RunMeta:
    """재현에 필요한 모든 것. docs/00-environment.md §7 참조."""

    run_id: str
    experiment: str
    source: str
    started_at: str

    target: dict[str, Any] = field(default_factory=dict)   # base_url, model, api_path
    config: dict[str, Any] = field(default_factory=dict)   # 실험 조건 전체
    environment: dict[str, Any] = field(default_factory=dict)  # SDK/펌웨어/모델 revision
    server_argv: list[str] = field(default_factory=list)   # furiosa-llm serve 인자 전체
    artifact_buckets: dict[str, Any] = field(default_factory=dict)  # 사후 해석에 필요

    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class RunWriter:
    """run 하나의 산출물을 data/raw/{run_id}/ 아래에 쓴다."""

    def __init__(self, root: str, meta: RunMeta):
        self.dir = os.path.join(root, meta.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.meta = meta
        self._requests = open(os.path.join(self.dir, "requests.jsonl"), "a", encoding="utf-8")
        self._metrics = open(os.path.join(self.dir, "metrics_timeseries.jsonl"), "a", encoding="utf-8")

    def write_meta(self) -> None:
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            f.write(self.meta.to_json())

    def write_request(self, rec: RequestRecord) -> None:
        self._requests.write(rec.to_json() + "\n")
        self._requests.flush()

    def write_metric(self, sample: dict[str, Any]) -> None:
        self._metrics.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        self._requests.close()
        self._metrics.close()


def read_requests(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
