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


# capacity 계산에서 **자동으로 무효 처리**하는 조건. 하네스가 조건마다 남긴
# validation 블록만 보고 판정한다.
#
# 여기 들어가는 것은 **측정 조건의 정의 자체가 깨진 경우**뿐이다.
#   - 출처가 전용 서버 실측이 아님 — 카드 수라는 분모가 없다
#   - prefix cache 적중 — prefill 비용이 최대 21배 과소 측정된다 (README §측정이 틀렸던 1)
#   - 출력 길이 미고정 — 조건 라벨(output_tokens)이 실제와 다르다. 이 라벨로 보간한다
#
# **warm-up 부족·짧은 측정창·QUEUED_NOT_BATCHED 는 일부러 넣지 않았다.** metric 마다
# 영향 방향이 다르기 때문이다 — warm-up 부족은 지연 백분위를 부풀리지만 처리량은
# 멀쩡하고, 큐잉 판정은 결함이 아니라 서버 거동의 관찰이다. 한 덩어리로 묶어 버리면
# 어느 metric 이 왜 못 쓰는지를 잃는다. 그쪽은 planner 의 근거 선택 순위로 반영한다
# (percentiles_trustworthy / _quality).
def capacity_blocks(validation: dict[str, Any]) -> list[str]:
    """이 조건을 capacity 계산에서 빼야 하는 이유들. 비어 있으면 사용 가능."""
    v = validation or {}
    out: list[str] = []
    if v.get("source") not in (None, SOURCE_MEASURED_LOCAL, SOURCE_MOCK):
        out.append(f"출처가 전용 서버 실측이 아닙니다 ({v.get('source')}) "
                   f"— docs/03-api-findings.md §7")
    pc = v.get("prefix_cache") or {}
    if pc.get("verdict") == "CONTAMINATED":
        if pc.get("max_cached_ratio") is not None:
            out.append(f"prefix cache 오염 — 적중 비율 최대 {pc['max_cached_ratio']} "
                       f"(한계 {pc.get('limit')}), 초과 요청 {pc.get('requests_over_limit')}건")
        else:
            # 비율 기준 이전(건수 기준)에 쓰인 run. 그 판정은 오탐이 많았지만
            # 기록된 판정을 임의로 뒤집지 않는다. 대신 근거가 약하다는 것을 밝힌다.
            out.append(f"prefix cache 오염 — 건수 기준 옛 판정 "
                       f"(적중 요청 {pc.get('requests_with_cache_hit')}건, "
                       f"최대 {pc.get('max_cached_tokens')} 토큰). 비율 기록 없음")
    cc = v.get("concurrency") or {}
    if cc.get("verdict") == "FOREIGN_LOAD":
        out.append(f"같은 서버에 다른 클라이언트가 붙어 있었습니다 — 요청 동시성 "
                   f"{cc.get('requested')}, 서버가 실제로 처리한 최대 "
                   f"{cc.get('peak_running_samples')}. 이 행의 동시성 라벨이 실제와 다릅니다")
    ol = v.get("output_length") or {}
    if ol.get("verdict") == "VARIES":
        out.append(f"출력 길이가 고정되지 않았습니다 — 요청 {ol.get('requested')}, "
                   f"관측 {ol.get('distinct_observed')}")
    return out


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

    # -- 혼합 부하(B3) 전용 -------------------------------------------------
    # 실제 서비스는 짧은 대화 요청과 긴 문서 요청이 섞여 들어온다. 그 간섭을 재려면
    # 두 종류를 같은 run 안에서 구분해야 한다.
    role: str = "main"                  # main | background | injected
    # 이 요청이 살아 있는 동안 겹친 주입 요청 수. 사후에 채운다.
    # 주입 전후 비교 대신 **같은 run 안에서 겹친 것과 안 겹친 것을 비교**하기 위한 값이다.
    # 길이 편향이 있다 — 느려진 요청은 더 오래 살아 겹칠 확률이 높다(간섭 과대평가).
    overlapped_injections: int = 0
    # 주입 활성 구간에 **시작**했는가. 요청 자신의 지속시간에 의존하지 않아 길이 편향이
    # 없는 대신, 주입 직전 시작해 길게 도는 요청을 놓친다(간섭 과소평가).
    # 두 기준을 함께 내면 참값이 그 사이에 있다.
    started_during_injection: bool = False

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
