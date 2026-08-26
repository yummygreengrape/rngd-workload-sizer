# 00. 실측 환경 조사 결과

> 조사일: 2026-08-25 / 대상: `edu-hlu-1` — FuriosaAI RNGD 8-card 평가 서버 (접속 정보는 공개하지 않음)
> 이 문서의 모든 값은 **추측이 아니라 해당 장비에서 직접 확인한 값**이다.
> 아직 실행하지 않아 확인하지 못한 항목은 문서 마지막 "미해결 항목"에 분리해 두었다.

## 1. 하드웨어

| 항목 | 값 |
|---|---|
| NPU | **RNGD 8장** (`npu0`~`npu7`), 카드당 8 PE |
| Firmware | `2026.3.0, 2d3f72a` (8장 전부 동일) |
| PCI BDF | `03:00.0`, `04:00.0`, `44:00.0`, `45:00.0`, `83:00.0`, `84:00.0`, `c3:00.0`, `c4:00.0` |
| 상태 | 온도 33~40°C, 전력 36~41W → **전부 유휴** (`furiosa-smi ps` 결과 없음) |
| 호스트 | Ubuntu 22.04.5, Linux 6.8.0-124, x86_64 |
| CPU / RAM | 128 코어 / 1007 GB (사용 144 GB) |
| 디스크 | `/` 432G 중 395G 여유, `/root` 251G 여유 |
| 네트워크 | huggingface.co 접근 가능 (HTTP 200), HF 토큰은 미설정 |

카드 하나는 `/dev/rngd/npuXpe0`~`npuXpe7` 8개 PE와 `npuXmgmt`로 노출된다.
`npuXpe0-1`, `npuXpe0-3`, `npuXpe4-7` 같은 **PE 묶음 디바이스**도 존재 → 카드 내부를 쪼개 쓰는 것도 가능하다.

## 2. 소프트웨어 스택

| 패키지 | 버전 |
|---|---|
| `furiosa-llm` | `2026.3.0-release (rev a13b5a4)` |
| `furiosa-llm-native` / `furiosa-native-llm-common` | 2026.3.0 |
| `furiosa-kernels` / `furiosa-tcc` / `furiosa-torch-ext` | 2026.3.0 |
| `furiosa-torch` | 2026.2.0 |
| `furiosa-models` | 2026.2.0 |
| `furiosa-smi-py` | 2026.1.2 |
| `torch` | 2.10.0+**cpu** |
| `transformers` | 5.1.0 |
| Python | 3.10.12 (시스템 `/usr/bin/python3`, venv/conda 없음) |

CLI: `/usr/bin/furiosa-smi`, `/usr/local/bin/furiosa-llm`
(`furiosactl`, `furiosa-compiler` 단독 CLI는 **없음** — 컴파일은 `furiosa-llm build`로 수행)

### `furiosa-llm` 서브커맨드
`build` / `serve` / `collect-env` / `version`

### `furiosa-smi` 서브커맨드
`info` / `status` / `ps` / `version` / `topo` / `governor` / `drain`

## 3. 서빙 API — OpenAI 호환 서버

`furiosa-llm serve <model>` 이 띄우는 FastAPI 앱의 실제 라우트:

```
GET  /health
GET  /metrics                 ← Prometheus 텍스트
GET  /version
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings           ← Embedding 워크로드
POST /v1/score , /v1/rerank , /v2/rerank
POST /tokenize , POST /detokenize , GET /tokenizer_info
POST /v1/responses ...
```

**별도의 추상화 레이어가 필요 없다.** 표준 OpenAI 클라이언트(혹은 `httpx`)로 전부 측정 가능하다.

### 벤치마크 통제에 직접 쓸 수 있는 파라미터 (실제 확인됨)

`furiosa_llm/server/protocol.py` 기준으로 요청 바디에 다음이 존재한다:

- `ignore_eos: bool = False` → **출력 길이를 정확히 고정할 수 있다**
- `min_tokens: int = 0` → `max_tokens`와 같이 쓰면 출력 토큰 수 고정
- `stream_options.include_usage: bool = True` → **스트리밍에서도 `usage` 수신 가능** (정확한 tokenizer 기준 토큰 수)

### 서버 기동 옵션 중 측정에 영향이 큰 것

| 옵션 | 기본값 | 벤치마크 관점 |
|---|---|---|
| `--enable-prefix-caching` | **True** | ⚠️ 반복 측정을 오염시킨다. 반드시 끄거나 통제해야 함 |
| `--enable-overlap-scheduling` | True | CPU 스케줄링 오버헤드를 NPU 연산 뒤로 숨김 |
| `--devices` | 미지정 시 **유휴 디바이스 전부 사용** | ⚠️ 반드시 `npu:0`으로 고정해야 "1장 성능"이 된다 |
| `--max-batch-size` | None → **아티팩트 값** | 측정 조건이 암묵적이 됨. 명시 지정 필요 |
| `--max-concurrency` | None | "iteration당 최대 동시 decode 요청" |
| `--max-num-batched-tokens` | None | "iteration당 최대 배치 토큰 수" |
| `--npu-queue-limit` / `--max-processing-samples` | None → 아티팩트 값 | 동일 |
| `--enable-jit-compilation` | False (실험적) | 버킷 미스 시 동작에 영향 |
| `-tp` / `-pp` / `-dp` | 아티팩트 기본 | 스케일링 실험에 사용 |

`--devices` 표기법: `npu:X` = 카드 전체, `npu:X:Y` = 카드 X의 코어 Y (예: `npu:0:0-7`).

### 개발용 엔드포인트

`FURIOSA_SERVER_DEV_MODE=1` 환경변수로 기동하면 추가된다:

```
POST /reset_prefix_cache      ← 반복 측정 사이 prefix cache 비우기
```

**벤치마크 신뢰성 확보의 핵심 도구다.**

## 4. 수집 가능한 metric

`/metrics` (Prometheus) 의 소스인 `DpMetrics` 필드 — 실제 확인:

```
prompt_tokens         누적 prefill 토큰 수
generation_tokens     누적 decode 토큰 수
running_samples       현재 처리 중 요청 수     ← 실제 concurrency 검증용
waiting_samples       대기 중 요청 수          ← 큐잉 vs 배칭 구분용
kv_cache_usage        KV cache 사용률          ← 메모리 용량 병목 증거
prefix_cache_hit_rate prefix cache 적중률      ← 측정 오염 검증용
wire_hit_rate         wire pipeline 적중률
```

서버 로그에도 동일 지표가 주기적으로 출력된다:
`Avg prompt throughput / Avg generation throughput / Running / Waiting / RNGD KV cache usage / Prefix cache hit rate / Wire pipeline hit rate`

**스펙이 요구한 "memory/KV cache 관련 metric"은 실제로 제공된다.**

### NPU 하드웨어 metric (`furiosa_smi_py`)

```
Device, DeviceInfo, CoreStatus(es), CoreUtilization, CoreFrequency, MemoryFrequency,
PeFrequency, DeviceTemperature, DevicePerformanceCounter, PePerformanceCounter,
PcieInfo/PcieLinkInfo, ThrottleReason(THERMAL_SLOWDOWN, HW_POWER_CAP, ...),
list_devices(), driver_info(), create_default_observer()
```

`CoreUtilization`, `PePerformanceCounter`가 존재하므로 **NPU utilization 수집은 가능하다.**
단 그 값의 정확한 정의(PE busy 시간 비율인가, 연산 활용률인가)는 아직 확인하지 않았다 → 미해결 항목 4번.

## 5. 사용 가능한 모델 (HF 캐시, 총 ~60GB+)

`/root/.cache/huggingface/hub`:

| 모델 | 아티팩트 형태 | 크기 | 비고 |
|---|---|---|---|
| `furiosa-ai/Qwen3-8B-FP8` | **`.fxb` (신형)** | 9.2G | 빌드 불필요, 최신 포맷 |
| `furiosa-ai/Qwen3-Embedding-0.6B` | **`.fxb` (신형)** | 1.2G | **유일하게 컴파일된 임베딩 모델** |
| `furiosa-ai/Qwen3-Reranker-0.6B` | `.fxb` | - | rerank용 |
| `furiosa-ai/Qwen3-VL-2B-Instruct` | `.fxb` | - | 멀티모달 |
| `furiosa-ai/Llama-3.1-8B-Instruct` | `artifact.json` + `binary_bundle.zip` (구형) | 16G | ⚠️ 구버전 빌드 |
| `furiosa-ai/Qwen2.5-0.5B-Instruct` | `artifact.json` (구형) | 972M | **스모크 테스트용으로 최적** |
| `furiosa-ai/Qwen3-32B-FP8` | 구형 | - | |
| `furiosa-ai/EXAONE-4.0-32B-FP8` | - | - | |
| `furiosa-ai/Llama-3.3-70B-Instruct` | - | - | |
| `furiosa-ai-dev/solar-enkoja-100b-...-nvfp4` | - | - | + `solar.tar.zst` 60GB |

`/root/.cache/llm-engine-artifacts` 는 **비어 있다** (런타임 아티팩트 캐시).

**`furiosa-llm build` 단계를 건너뛸 수 있다** — 2일 일정의 가장 큰 리스크가 이미 제거된 상태.

## 6. ⭐ 아티팩트 버킷 구조 — 이 프로젝트의 설계를 좌우하는 사실

RNGD는 **컴파일된 버킷(bucket) 단위**로 실행된다. 아티팩트에 없는 shape는 그대로 실행되지 않는다.

### `Llama-3.1-8B-Instruct` (artifact.json)

```
tensor_parallel_size = 8      pipeline_parallel_size = 1
devices = {'0': 'npu:0:0-7'}   ← 1 아티팩트 인스턴스 = RNGD 1장 전체
max_position_embeddings = 131072   (32L, 32 heads, 8 KV heads)
```

| 종류 | 버킷 |
|---|---|
| **Prefill** (kv=0) | **batch=1만**, attention_size = **128, 256, 384, 512, 640, 768, 896, 1024** (8개) |
| **Decode** (kv>0) | batch = **1, 2, 4, 8, 16, 32, 64, 128, 256**, attention_size = 256 ~ 131072 (120개) |

### `Qwen3-8B-FP8` (.fxb, 152 엔트리)

| 종류 | 버킷 |
|---|---|
| **Prefill** | **batch=1만**, attention_size = **128, 256, 384, 512, 640, 768, 1024** |
| **Decode** | batch = **1, 2, 4, 8, 16, 32, 64, 128, 256**, attention_size = 256 ~ **40960** |
| tokenwise | `tw1, 4, 8, 16, 32, 64, 128, 256, 1024` (first / mid / last_with_lm_head) |

### 여기서 즉시 도출되는 두 가지

**(1) Prefill 버킷은 batch=1이고 최대 1024 토큰까지다.**
→ 스펙이 제시한 input length 2048 / 8192를 그대로 측정 축으로 쓸 수 없다.
   해당 프롬프트가 **어떤 경로로 실행되는지 먼저 확인해야 한다.** (미해결 항목 1번)

**(2) Prefill은 온NPU 배칭이 없고, Decode는 256까지 배칭된다.**
→ concurrency를 올리면 decode 처리량은 늘지만 **prefill은 직렬화**될 가능성이 높다.
   즉 이 하드웨어에서 서비스 capacity는 처리량이 아니라 **TTFT에 먼저 막힐 수 있다.**
   → 이것이 프로젝트의 핵심 분석 가설이 된다. ([01-spec-review.md](01-spec-review.md) IMP-2)

**(3) prefill attention_size 간격이 128이다.**
→ input length 1000과 1030은 **같은 1024 버킷**, 1030과 1100은 다른 버킷일 수 있다.
   TTFT가 입력 길이에 대해 **계단형**일 것이고, 선형 보간은 틀린다.
   → planner의 interpolation 설계에 직접 영향. ([01-spec-review.md](01-spec-review.md) IMP-1)

## 7. 재현성 메타데이터 (모든 결과 파일에 박을 값)

```json
{
  "host": "edu-hlu-1",
  "kernel": "Linux 6.8.0-124-generic",
  "os": "Ubuntu 22.04.5 LTS",
  "npu_firmware": "2026.3.0, 2d3f72a",
  "furiosa_llm": "2026.3.0-release (rev a13b5a4)",
  "furiosa_smi_py": "2026.1.2",
  "torch": "2.10.0+cpu",
  "transformers": "5.1.0",
  "python": "3.10.12",
  "model_id": "<HF id>",
  "model_revision": "<snapshot sha>",
  "artifact_furiosa_llm_version": "<artifact.json metadata>",
  "artifact_furiosa_compiler_version": "<artifact.json metadata>",
  "serve_argv": ["..."],
  "devices": "npu:0"
}
```

`furiosa-llm collect-env` 출력도 함께 저장한다.

## 8. 미해결 항목 — 실제로 실행해야만 알 수 있는 것

| # | 항목 | 왜 중요한가 | 확인 방법 |
|---|---|---|---|
| 1 | **prefill 버킷(>1024) 초과 프롬프트 처리 방식** | input length 축 전체가 여기에 달림. `chunked_prefill` 관련 코드는 grep으로 찾지 못했다 | 128→8192 스윕 실행 + 서버 로그 관찰 |
| 2 | **서버 기동 + 8B 모델 로드 소요 시간** | 2일 일정의 최대 미지수. 모델 교체마다 발생 | 스모크 테스트에서 계측 |
| 3 | `/metrics` Prometheus **실제 metric 이름** | 파싱 코드가 여기에 의존. 이름은 네이티브 모듈에서 생성됨 | 서버 기동 후 `curl /metrics` |
| 4 | `CoreUtilization`의 **정확한 정의** | "utilization 70% = compute-bound" 같은 오독을 막아야 함 | idle / full-load 두 상태에서 값 비교 |
| 5 | **HBM bandwidth counter 제공 여부** | 제공 안 되면 병목 판정을 간접 분석으로 대체해야 함 | `DevicePerformanceCounter` 필드 확인 |
| 6 | **Llama 구형 아티팩트의 최신 서버 호환성** | artifact `254c5ee` vs 설치본 `a13b5a4` 불일치 | 실제 `serve` 시도 |
| 7 | **Embedding 모델이 `furiosa-llm serve`로 뜨는지** | `.fxb`는 있으나 `/v1/embeddings` 경로 동작 미검증 | 스모크 테스트 |
| 8 | **dp≥2 스케일링 실효율** | planner의 "N장" 계산이 선형 가정에 의존 | 2카드 실측 |
| 9 | **eval 서버 이용 가능 기간 / 독점 여부** | 실험 범위 전체를 결정 | ⚠️ **사용자 확인 필요** |
