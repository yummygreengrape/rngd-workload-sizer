# 02. 구체화된 실행 계획

> 근거: [00-environment.md](00-environment.md) 실측 / [01-spec-review.md](01-spec-review.md) 검토 결과
>
> ⚠ **이 문서는 착수 전 계획서이고, 그 시점의 판단을 그대로 남깁니다.**
> §1 실험 계획은 [04-experiment-program.md](04-experiment-program.md) 로 대체됐고,
> §5 일정(2일)과 §6 착수 전 확인 사항은 소진됐습니다.
> **지금 유효한 것**은 §0 확정 전제·§2 데이터 스키마·§3 capacity 계산 정의·§4 프로젝트 구조입니다.
> 현재 상태는 [STATUS.md](STATUS.md), 결과는 [../README.md](../README.md) 를 보세요.

## 0. 확정 전제

| 항목 | 값 | 근거 |
|---|---|---|
| **1 RNGD** | `npu:X` 카드 전체 = 8 PE, tp=8 / pp=1 / dp=1 | 아티팩트가 `tensor_parallel_size=8`, `devices={'0':'npu:0:0-7'}` |
| 단일 카드 측정 | `--devices npu:0` 고정, 나머지 7장 유휴 | CRIT-3 |
| prefix cache | `--no-enable-prefix-caching` + 요청별 고유 프롬프트 + hit_rate 검증 | CRIT-2 |
| 출력 길이 통제 | `max_tokens = min_tokens = N`, `ignore_eos = true` | IMP-4 |
| 토큰 수 | `/tokenize`로 생성, `usage`로 검증 | IMP-5 |
| LLM 모델 (P0) | `furiosa-ai/Qwen3-8B-FP8` (최신 `.fxb`) | IMP-11 |
| Embedding 모델 | `furiosa-ai/Qwen3-Embedding-0.6B` | 유일한 컴파일 임베딩 모델 |
| 스모크 모델 | `furiosa-ai/Qwen2.5-0.5B-Instruct` (972M) | IMP-12 |
| **API 경로** | **`/v1/completions`** (chat 아님) | chat template·reasoning 우회, 프롬프트 토큰 정확 통제 — [03](03-api-findings.md) §2 |
| **측정 소스** | **전용 서버만.** 호스팅 엔드포인트는 개발·검증 전용 | [03](03-api-findings.md) §7 |

### 측정 정의 (코드와 문서에 동일 문구로 박는다)

```
TTFT   = (delta.content 또는 delta.reasoning 중 먼저 온 non-empty 값의 수신 시각) − (요청 전송 시각)
         # role-only 첫 chunk는 건너뛴다. 추론 모델은 content가 끝까지 비어 있다 → 03 §1
E2E    = (마지막 delta 수신 시각) − (요청 전송 시각)
TPOT   = (E2E − TTFT) / (completion_tokens − 1)          # completion_tokens ≥ 2 일 때만
per-user output tps    = completion_tokens / E2E          # 요청 단위
aggregate output tps   = Σ completion_tokens / (측정 구간 wall-clock)   # 구간 단위
```

**wall-clock 기반 aggregate와 요청별 latency를 절대 섞지 않는다.**
aggregate는 "마지막 warm-up 종료 후 ~ 마지막 측정 요청 종료" 구간으로 정의하고,
그 구간에 완료된 요청만 분자에 넣는다.

---

## 1. 실험 계획

> ⚠️ **2026-08-25 갱신.** RNGD 8장을 시간 단위로 예약할 수 있게 되어 이 절의 실험 matrix는
> [04-experiment-program.md](04-experiment-program.md)로 대체되었다.
> 아래 표는 단일 카드·짧은 세션 전제의 원안으로 남겨둔다. 측정 정의(§0), 데이터 스키마(§2),
> capacity 수식(§3), 구조(§4)는 그대로 유효하다.


전 실험 공통: **warm-up 요청 → (기록하되 `is_warmup=true`) → 측정 구간** 순서.
`/metrics`는 측정 내내 1초 간격 폴링하여 별도 시계열로 저장한다.

| ID | 실험 | 조건 | 우선순위 | 목적 |
|---|---|---|---|---|
| **E0** | 스모크 | Qwen2.5-0.5B, 요청 몇 건 | **P0** | 파이프라인 전체 검증 + 기동/로드 시간 계측 (미해결 2) |
| **E1** | **실행 경로 확인** | input 128→8192, conc=1 | **P0** | 버킷 초과 프롬프트 처리 방식 확인 → **input 축 확정** (CRIT-1) |
| **E2** | Prefill / TTFT | input ∈ 확정된 축, conc=1, output=1 | **P0** | 순수 prefill 비용. output=1이면 TTFT가 곧 prefill 시간 |
| **E3** | **버킷 경계** | input ∈ {896, 1000, 1024, 1030, 1100, 1152}, conc=1 | **P0** | 계단형 여부 확인 (ADD-1) — 저비용 |
| **E4** | Decode / concurrency | input=512 고정, output=128 고정, conc ∈ {1,2,4,8,16,32} | **P0** | capacity의 주 데이터. 조건당 완료 ≥100건 |
| **E6** | Embedding | batch ∈ {1,2,4,8,16,32,64}, seq len 2~3종 | **P1** | documents/s |
| **E5** | Long-context × conc | input=4096(가능하면), conc ∈ {1,4,16} | **P1** | `kv_cache_usage` 압박 관찰 |
| **E7** | **2카드 스케일링** | dp=2, E4의 conc ∈ {8,32} 재측정 | **P1** | 선형 가정 검증 (CRIT-4) |
| E8 | prefix cache ON/OFF 비교 | E4 일부 조건 | P2 | prefix cache의 capacity 기여 |
| E9 | 모델 비교 | Llama-3.1-8B | P2 | 로드 성공 시에만 |

**E1이 E2/E4/E5의 input 축을 결정하므로 반드시 먼저 실행한다.**
E1 결과 1024 초과가 지원되지 않으면 측정 범위를 좁히고 그 사실을 README의 한계에 기록한다 (은폐하지 않는다).

### 반복 횟수 기준
- TTFT 계열(E2/E3): 조건당 **≥30회**, median + p95 + stddev
- concurrency 계열(E4/E5): 조건당 **완료 요청 ≥100건**, 미달 시 P95를 `insufficient_samples`로 표기 (IMP-8)
- Embedding(E6): 조건당 **≥20회**

### 실행별 자동 무효화 조건
1. `prefix_cache_hit_rate ≠ 0` → 무효 (CRIT-2)
2. `running_samples` 최대값 < 의도한 concurrency → **배칭이 아니라 큐잉**으로 기록, 별도 해석 (CRIT-5 / IMP-7b)
3. `usage.prompt_tokens`가 의도 길이와 다름 → 차이를 기록 (숨기지 않음)
4. 다른 프로세스가 NPU 점유 (`furiosa-smi ps`) → 무효

---

## 2. 데이터 스키마

```
data/
  raw/
    {run_id}/
      meta.json                 # 재현 메타데이터 (환경 + serve 인자 전체)
      requests.jsonl            # 요청 1건 = 1줄
      metrics_timeseries.jsonl  # /metrics 1초 폴링
      server.log                # serve stdout/stderr
  processed/
    llm_summary.csv             # 조건별 집계
    embedding_summary.csv
  sample/                       # mock 전용. 모든 레코드에 "source":"mock" 필수
```

`run_id` = `{model}_{experiment}_{YYYYmmdd-HHMMSS}`

### `requests.jsonl` 한 줄

```json
{
  "run_id": "...", "request_id": "...", "experiment": "E4",
  "is_warmup": false,
  "concurrency": 8, "target_input_tokens": 512, "target_output_tokens": 128,
  "t_send": 0.0, "t_first_token": 0.0, "t_last_token": 0.0,
  "ttft_ms": 0.0, "e2e_ms": 0.0, "tpot_ms": 0.0,
  "prompt_tokens_actual": 512, "completion_tokens_actual": 128,
  "reasoning_tokens": 0,
  "cached_tokens": 0,
  "source": "measured_local",
  "error": null
}
```

### `meta.json`
[00-environment.md](00-environment.md) §7의 메타데이터 블록 전체 + `furiosa-llm collect-env` 출력 + 아티팩트 버킷 목록.

**아티팩트의 버킷 목록을 meta에 저장하는 것이 중요하다** — 나중에 결과를 해석할 때
"이 input length가 어느 버킷에 떨어졌는가"를 사후에 재구성할 수 있어야 한다.

---

## 3. Capacity 계산 (단일 정의)

### LLM Decode

```
required_output_tps = concurrent_users × target_output_tps_per_user          [tok/s]

# 고정점 반복 (CRIT-5)
n := 1
repeat up to 20:
    c_per_card := ceil(concurrent_users / n)
    row        := select_row(workload, model, input_len, c_per_card)
    usable     := row.aggregate_output_tps × target_utilization
    n_new      := ceil(required_output_tps / usable)
    break if n_new == n
    n := n_new

n_cards               = n
estimated_utilization = required_output_tps / (n_cards × row.aggregate_output_tps)
```

`target_utilization`은 **분모에 한 번만** 적용된다. `headroom` 입력은 `1 − headroom`으로 변환한다 (IMP-10).

### SLA 검사 — 처리량과 독립적으로 전부 통과해야 "충분"

```
PASS 조건 (AND):
  row.ttft_p95_ms          ≤ target_max_ttft_ms
  row.per_user_output_tps  ≥ target_output_tps_per_user
  row.e2e_p95_ms           ≤ target_p95_ms          (입력된 경우에만)
```

하나라도 FAIL이면:
1. 카드당 concurrency를 한 단계 낮춘 실측 조건으로 재탐색 → 카드 수 증가
2. 그래도 FAIL이면 **"이 SLA는 현재 실측 범위에서 달성 불가"**로 명시 출력
3. **어떤 경우에도 처리량만으로 "충분"이라고 표시하지 않는다**

### Embedding

```
required_docs_per_s = 총 문서 수 / 목표 처리 시간(s)      또는   입력 QPS × 요청당 문서 수
usable_docs_per_s   = row.docs_per_s × target_utilization
n_cards             = ceil(required_docs_per_s / usable_docs_per_s)
```
단위는 **documents/s로 통일**한다 (IMP-9).

### 실측 조건 선택 (`select_row`)과 confidence

| 등급 | 조건 | 표시 |
|---|---|---|
| `measured` | 요청 조건이 실측 격자점과 정확히 일치 | 높음 |
| `interpolated` | 격자 내부, **버킷 경계를 넘지 않음** | 중간 |
| `interpolated_across_bucket_boundary` | 격자 내부지만 버킷 경계를 넘음 | **낮음** — 계단 구조로 선형 보간 신뢰 불가 (IMP-1) |
| `extrapolated` | 격자 밖 | **낮음** — `Low confidence: requested workload is outside the measured benchmark range.` |

출력에는 항상 **근거가 된 raw 데이터 행의 `run_id`와 조건**을 함께 표시한다.

### 병목 판정
[01-spec-review.md](01-spec-review.md) IMP-7(b)의 규칙표만 사용한다.
어느 패턴에도 안 맞으면 **`unknown`을 그대로 출력**한다 (억지 분류 금지).

---

## 4. 프로젝트 구조

```
rngd-workload-sizer/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 00-environment.md        # 실측 환경 (본 조사 결과)
│   ├── 01-spec-review.md        # 기획 검토
│   └── 02-plan.md               # 본 문서
│
├── benchmark/                   # ← 서버에서 실행
│   ├── server.py                # serve 기동/종료, 인자를 meta.json에 기록
│   ├── client.py                # 스트리밍 요청, TTFT/TPOT 계측
│   ├── prompts.py               # /tokenize 기반 정확한 길이 + 고유 prefix
│   ├── metrics.py               # /metrics 폴링, furiosa-smi 샘플링
│   ├── run_llm.py               # E1~E5
│   ├── run_embedding.py         # E6
│   └── configs/                 # 실험 정의 YAML
│
├── analysis/                    # ← 로컬에서 실행
│   ├── process.py               # raw → processed 집계
│   ├── bottleneck.py            # IMP-7(b) 규칙 + 대역폭 역산
│   └── plots.py
│
├── planner/                     # ← UI와 완전 분리, 순수 함수
│   ├── models.py                # ROD 입력/출력 dataclass
│   ├── benchmark_store.py       # processed 로드, select_row, confidence 판정
│   └── capacity.py              # 고정점 반복 + SLA 검사
│
├── app/                         # ← UI. 계산 로직 없음
│   └── main.py
│
├── data/{raw,processed,sample}/
└── tests/
    ├── test_capacity.py         # 수식/단위/올림/utilization 이중적용 검증
    ├── test_select_row.py       # confidence 등급 판정
    └── test_metrics.py          # TTFT/TPOT 계산
```

**`tests/test_capacity.py`를 planner 구현보다 먼저 쓴다.**
기획서 검토 프롬프트가 지적한 문제(단위 불일치, utilization 이중 적용, 올림 처리)를
사후 리뷰가 아니라 **테스트로 막는다.**

---

## 5. 일정 (2일)

| 시점 | 작업 | 산출물 |
|---|---|---|
| D1 오전 | 환경 고정, **E0 스모크**, **E1 실행 경로 확인** | 기동 시간 실측, **input 축 확정** |
| D1 오후 | 벤치 하네스 완성, **E2 + E3 + E4** 실행 | `data/raw/` LLM 실측 |
| D1 저녁 | **E6 Embedding** | `data/raw/` 임베딩 실측 |
| D2 오전 | `analysis/` 집계·그래프, `planner/` + 테스트 | `data/processed/`, 그래프, 통과하는 테스트 |
| D2 오후 | `app/` UI, README | 동작하는 planner |
| D2 여유 | **E7 2카드 스케일링**, E5, E8 | 선형 가정 검증 |

**E1이 D1 오전에 끝나야 나머지가 굴러간다.** 여기서 막히면 즉시 범위를 좁힌다.

우선순위 원칙 (기획서 §9 유지): **P2 때문에 P0/P1을 희생하지 않는다.**

---

## 5.5 전용 서버 없이 지금 진행 가능한 작업

호스팅 엔드포인트로 API 동작을 확인해 둔 덕분에, 서버 접속이 확정되기 전에도 다음을 완성할 수 있다.
전용 서버 점유 시간을 **실측에만** 쓰기 위해서다.

| 작업 | 호스팅으로 검증 가능한가 |
|---|---|
| `benchmark/client.py` — SSE 파싱, TTFT 추출, `usage` 수집, 에러/타임아웃 | ✅ 이미 동작 확인 |
| `benchmark/prompts.py` — 고유 프롬프트 생성 + 토큰 길이 보정 루프 | ✅ `usage.prompt_tokens`로 수렴 확인 가능 |
| `planner/` 전체 + `tests/` | ✅ 실측 데이터 불필요 (mock으로 개발) |
| `analysis/` 집계·그래프 코드 | ✅ 스키마만 있으면 됨 |
| 실제 capacity 수치 | ❌ **전용 서버 필요** |

`--target hosted`로 얻은 레코드는 `"source": "hosted_endpoint"`가 강제로 박히며
`planner/benchmark_store.py`가 로드 단계에서 거부한다.

---

## 6. 착수 전 확인이 필요한 사항

| # | 항목 | 왜 필요한가 |
|---|---|---|
| 1 | **`edu-hlu-1` 이용 가능 기간과 독점 여부** | 다른 프로세스가 NPU를 쓰면 모든 측정이 무효다. 기간이 짧으면 E1/E2/E4만 남기고 축소한다 |
| 2 | 8장 전부 사용 가능한가 | E7(2카드 스케일링) 가능 여부를 결정 |
| 3 | LLM 모델 선택 | 권장: `Qwen3-8B-FP8` (IMP-11). Llama는 P2 |
| 4 | UI 스택 | 권장: Streamlit. planner가 순수 함수이므로 나중에 교체해도 비용이 작다 |

1번이 가장 중요하다. 나머지는 기본값으로 진행해도 되지만, **1번이 정해지지 않으면 실험 matrix를 확정할 수 없다.**

---

## 7. 이 계획이 지키는 원칙

- 확인되지 않은 SDK 동작을 가정하지 않는다 → 미해결 항목 9개를 명시적으로 분리했다
- 측정과 계산을 분리한다 → `benchmark/` ↔ `data/` ↔ `planner/`
- UI와 계산을 분리한다 → `planner/`는 순수 함수, `app/`은 표시만
- 모든 계산은 재현 가능하다 → raw 데이터 커밋, meta.json에 조건 전량 기록
- 데이터를 가설에 맞추지 않는다 → E1/E3 결과가 예상과 다르면 축과 서사를 바꾼다
- 결과가 나오기 전에 결론을 쓰지 않는다 → README "주요 관찰"은 placeholder로 둔다
