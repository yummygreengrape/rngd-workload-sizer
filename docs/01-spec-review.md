# 01. 원 스펙 검토 결과

> 대상: 최초 프로젝트 기획서 (`rngd-workload-sizer` 구현 프롬프트)
> 기준: [00-environment.md](00-environment.md)의 실측 환경
> 목적: 코드를 쓰기 전에 **기획대로 하면 깨지는 지점**을 찾아낸다.

기획서 자체의 방향은 타당하다. "peak FLOPS가 아니라 실측 기반 capacity planning"이라는 문제 설정,
mock 데이터 격리, 과잉 결론 금지 같은 원칙은 그대로 유지한다.
아래는 **실측 환경과 대조했을 때 수정이 필요한 부분**이다.

---

## Critical — 그대로 두면 측정 결과가 무의미해진다

### CRIT-1. Prefill 버킷이 batch=1 · 최대 1024 토큰 → 스펙의 input length matrix가 성립하지 않는다

**위치**: 기획서 §2 "LLM input length: 128 / 512 / 2048 / 8192"

**문제**
아티팩트 실측 결과 prefill 버킷은 두 모델 모두 **batch=1, attention_size 128~1024**뿐이다.
2048 / 8192 프롬프트가 어떤 경로로 실행되는지 확인되지 않았고, `chunked_prefill` 관련 코드도 찾지 못했다.

**왜 문제인가**
실행이 되더라도 그 숫자가 "2048 토큰 prefill 성능"인지 "1024 버킷을 여러 번 돈 결과"인지 모르는 채
그래프를 그리게 된다. 원인을 모르는 채 "8192에서 TTFT 급증" 같은 해석을 붙이면
기획서 §10이 금지한 "추측 기반 구현"을 결과 해석에서 저지르는 셈이다.

**수정**
input length 축을 고정하기 전에 **실행 경로 확인 실험(E1)을 P0의 첫 작업으로** 넣는다.
128 → 8192를 concurrency=1로 던져서 (a) 에러 여부 (b) TTFT가 길이에 대해 선형인지 계단형인지
(c) 서버 로그의 버킷 선택을 관찰한 뒤 축을 확정한다.
버킷을 넘는 길이가 지원되지 않으면 **측정 범위를 1024 이하로 좁히고 그 사실을 README의 한계에 명시**한다.
좁아진 범위는 결함이 아니라 "이 하드웨어의 실제 제약을 발견했다"는 결과다.

---

### CRIT-2. prefix caching이 기본 ON — 반복 측정이 오염된다

**위치**: 기획서 §2 "한 번의 결과를 그대로 사용하지 말고 최소한 여러 번 측정"

**문제**
`--enable-prefix-caching`의 **기본값이 True**다. 같은 프롬프트를 반복 측정하면
2회차부터 prefix cache hit로 TTFT가 비현실적으로 낮아진다.
기획서는 반복 측정을 요구하면서 이 오염원을 전혀 언급하지 않는다.

**왜 문제인가**
"여러 번 측정해서 median" 이라는 신뢰성 장치가 오히려 **낙관적 편향을 증폭시킨다.**
그리고 이 편향은 값이 이상하지 않아 보이기 때문에 사후에 발견하기 어렵다.

**수정** — 3중 방어
1. 벤치마크 서버는 `--no-enable-prefix-caching`으로 기동한다 (기본 측정 조건).
2. 매 요청의 프롬프트에 **고유 난수 prefix**를 넣는다. 길이는 `/tokenize`로 정확히 맞춘다.
3. 매 실행마다 `/metrics`의 `prefix_cache_hit_rate`를 기록하고,
   **0이 아니면 해당 실행을 무효 처리**한다. (자동 검증)

추가로 `FURIOSA_SERVER_DEV_MODE=1`로 기동하면 `POST /reset_prefix_cache`를 쓸 수 있다.
반복 사이 캐시 초기화 수단으로 확보해 둔다.

> 보너스: prefix caching ON/OFF 비교 자체를 P2 실험으로 두면
> "실제 서비스에서 prefix cache가 capacity에 얼마나 기여하는가"라는 좋은 분석이 된다.

---

### CRIT-3. "RNGD 1장"의 정의가 없다 — capacity 계산의 분모가 미정의

**위치**: 기획서 §1 B "RNGD 1장의 실측 capacity", "estimated RNGD count"

**문제**
RNGD 1장 = 8 PE이고, PE를 쪼개 쓰는 디바이스(`npu:0:0-3` 등)도 존재한다.
`--devices`를 지정하지 않으면 **유휴 디바이스를 전부 잡는다.**
정의 없이 측정하면 "1장 capacity"가 실제로는 8장 합산일 수 있다.

**왜 문제인가**
planner 출력 전체가 이 분모 위에 서 있다. 분모가 흔들리면 나머지가 전부 무의미하다.

**수정**
문서와 코드에 다음을 못박는다.

> **1 RNGD = `npu:X` 카드 전체 (8 PE), tensor_parallel_size=8, pp=1, dp=1**

근거: 아티팩트가 `tensor_parallel_size=8`, `devices={'0': 'npu:0:0-7'}`로 빌드되어 있다.
모든 단일 카드 벤치마크는 **`--devices npu:0` 고정**, 나머지 7장은 유휴로 두어 간섭을 배제한다.

---

### CRIT-4. "N장" 선형 외삽이 검증되지 않는다

**위치**: 기획서 §1 B `ceil(required_tps / (measured_tps_per_rngd × target_utilization))`

**문제**
카드 수를 늘렸을 때 처리량이 선형으로 늘어난다는 가정이 검증 없이 planner의 핵심 출력이 된다.
선형이 아닐 이유가 최소 두 가지 있다.
1. dp 라우팅/스케줄링 오버헤드 (`--data-parallel-routing-policy` 옵션이 존재한다는 것 자체가 방증)
2. **카드당 batch 감소** — 같은 동시 사용자를 N장에 나누면 카드당 concurrency가 1/N이 되고,
   decode는 batch가 클수록 효율적이므로 **카드당 효율이 떨어진다.**

**왜 문제인가**
2번은 특히 중요하다. 선형 가정은 이 효과를 무시하므로 **필요 카드 수를 과소 추정**한다.
용량 산정 도구가 낙관적으로 틀리는 것은 가장 나쁜 실패 방식이다.

**수정**
- **2카드 dp 스케일링 실험을 P1으로 승격**한다 (기획서에는 아예 없음).
  동일 총 concurrency에서 1카드 vs 2카드 aggregate throughput을 재고 scaling efficiency를 산출한다.
- planner는 카드 수를 **고정점 반복**으로 푼다 (CRIT-5 참조).
- 실험을 못 하더라도 README와 UI에 **"선형 스케일 가정"을 한계로 명시**한다.

---

### CRIT-5. 카드 수와 카드당 concurrency가 서로를 결정한다 — 순환 의존이 처리되지 않았다

**위치**: 기획서 §1 B의 capacity 수식

**문제**
`measured_output_tps_per_rngd`는 **어떤 concurrency에서 측정했느냐에 따라 달라진다.**
그런데 카드당 concurrency = `concurrent_users / n_cards` 이고, `n_cards`는 아직 구하려는 값이다.
기획서 수식은 이 순환을 인지하지 못하고 "measured_output_tps_per_rngd"를 상수처럼 쓴다.

**왜 문제인가**
어떤 실측 행을 분모에 넣느냐에 따라 답이 몇 배씩 달라진다.
기획서 §7이 요구한 "재현 가능한 계산"이 성립하지 않는다.

**수정** — 고정점 반복으로 명시적으로 푼다

```
n := 1
반복 (최대 20회):
    c_per_card := ceil(concurrent_users / n)
    row        := 실측표에서 c_per_card에 가장 가까운 조건 선택
    usable     := row.aggregate_output_tps × target_utilization
    n_new      := ceil(required_output_tps / usable)
    if n_new == n: 수렴 → 종료
    n := n_new
수렴 실패 시: 진동 구간의 상한을 채택하고 low confidence 표기
```

반복 과정 전체(각 회차의 n, c_per_card, 선택된 row)를 UI에 노출한다 → 계산 근거가 그대로 보인다.

---

## Important — 분석 품질을 크게 떨어뜨린다

### IMP-1. Bucket 경계 때문에 선형 보간이 틀린다 (기획서 §7의 interpolation 설계 결함)

prefill attention_size 간격이 **128**이다 (128, 256, 384, ..., 1024).
따라서 TTFT는 입력 길이에 대해 **연속이 아니라 계단형**일 가능성이 높다.
`measured / interpolated / extrapolated` 3분류만으로는 부족하다.

**수정**: planner에 4번째 플래그를 추가한다.
- `interpolated_across_bucket_boundary` — 보간 구간이 버킷 경계를 넘음 → 선형 보간 신뢰 불가
- 사용자 입력 input length가 버킷 경계 **바로 위**면 경고:
  "입력 1030 토큰은 1024 버킷을 초과해 다음 버킷으로 padding됩니다. 1024로 줄이면 동일 비용으로 처리됩니다."

이 기능은 다른 어떤 capacity planner에도 없고, **RNGD 구조를 이해했다는 가장 직접적인 증거**가 된다.

---

### IMP-2. 프로젝트의 핵심 가설이 기획서에 없다 — prefill/decode 배칭 비대칭

기획서 §3은 "Prefill은 compute-bound, Decode는 memory-bound"라는 **교과서 가설**에서 출발한다.
그런데 이 하드웨어의 아티팩트는 더 구체적이고 검증 가능한 가설을 이미 시사하고 있다.

> **Prefill 버킷은 batch=1뿐이고, Decode 버킷은 batch 256까지 있다.**
> → concurrency를 올리면 decode aggregate throughput은 잘 늘어나지만,
>   prefill은 직렬화되어 **TTFT가 concurrency에 대해 선형으로 악화**될 것이다.
> → 즉 이 하드웨어에서 서비스 capacity는 처리량이 아니라 **TTFT SLA에 먼저 막힌다.**

**왜 중요한가**
기획서 §1 B는 "처리량은 만족하지만 P95/TTFT가 목표를 초과하면 경고하라"고 요구한다.
위 가설이 맞다면 그 경고는 **가상의 예외 처리가 아니라 이 하드웨어의 지배적 제약**이 된다.
프로젝트의 서사가 "웹 계산기"에서 "구조 분석"으로 올라선다.

**주의**: 이것은 어디까지나 가설이다. 실측이 다르면 데이터를 따르고 가설을 폐기한다 (기획서 §10).

---

### IMP-3. TTFT 측정 방법이 정의되어 있지 않다

기획서는 TTFT를 요구하지만 측정 방법을 정하지 않았다. 스트리밍 응답의 첫 chunk는
content가 빈 role delta일 수 있어 "첫 토큰"의 정의가 모호하다.

**수정** — 정의를 하나로 못박는다.
```
TTFT = (delta.content 또는 delta.reasoning 중 먼저 도착한 non-empty 값의 수신 시각) − (요청 전송 시각)
```
role-only chunk(`{"role":"assistant","content":""}`)는 제외한다. 이 정의를 코드 주석과 문서 양쪽에 적는다.

> ⚠️ **2026-08-25 실측으로 수정됨.** 최초에는 "첫 non-empty `content` delta"로 적었으나,
> Qwen3·gpt-oss 같은 추론 모델은 `content`가 끝까지 빈 채 **`reasoning` delta로만** 토큰을 낸다.
> 원래 정의를 쓰면 이런 모델에서 TTFT가 영원히 측정되지 않는다.
> 근거와 재현 로그: [03-api-findings.md](03-api-findings.md) §1

---

### IMP-4. 출력 토큰 수를 고정하지 않으면 TPOT이 흔들린다

EOS가 일찍 나오면 output length가 요청마다 달라져 TPOT/throughput 비교가 무의미해진다.

**수정** — 실측으로 확인된 파라미터를 쓴다.
```
max_tokens = min_tokens = N,  ignore_eos = true
TPOT = (E2E − TTFT) / (N − 1)      ← N ≥ 2 일 때만 정의
```
`ignore_eos`와 `min_tokens`는 OpenAI 호환 API에 **실제로 노출되어 있다**(protocol.py 확인).

---

### IMP-5. 토큰 수는 추정하지 말고 서버에 물어본다

기획서 검토 항목 "token count를 정확한 tokenizer 기준으로 측정하는가"에 대한 구체적 수단:
- 프롬프트 생성: `POST /tokenize`로 목표 길이에 **정확히** 맞춘다 (문자 수 추정 금지)
- 결과 검증: `stream_options.include_usage=true`로 받은 `usage.prompt_tokens` / `usage.completion_tokens`
- 의도한 길이와 실제 usage가 다르면 **그 차이를 결과 파일에 기록**한다 (숨기지 않는다)

---

### IMP-6. NPU utilization을 잘못 해석하면 기획서 §5의 함정에 그대로 빠진다

`CoreUtilization`은 존재하지만 그 정의를 확인하지 않았다.
PE busy 시간 비율이라면 "utilization 90%인데 FLOPS는 낮다 → 비효율"은 **완전히 틀린 결론**이다.
(memory 대기 중에도 PE는 busy로 잡힐 수 있다.)

**수정**
1. 측정 전에 idle 상태와 full-load 상태의 값을 찍어 **관측된 정의를 문서에 기록**한다.
2. 정의가 불명확하면 지표 이름을 **"PE busy ratio"로만 부르고** FLOPS 효율로 해석하지 않는다.
3. README에 "이 수치를 연산 활용률로 읽으면 안 되는 이유"를 한 줄 적는다.

---

### IMP-7. HBM bandwidth 직접 측정 수단이 확인되지 않았다 → "예상 병목" 출력이 근거를 잃는다

기획서 §1 B는 병목을 `compute / memory-bandwidth / concurrency-scheduling / unknown`으로 출력하라고 한다.
그런데 bandwidth counter가 없으면 이 분류는 **근거 없는 추측**이 된다.

**수정 (a) — 간접 정량 분석으로 대체한다**

decode 1 step당 최소 메모리 트래픽은 계산할 수 있다:
```
bytes_per_step ≈ (모델 파라미터 바이트 / tp) + (batch × context_len × KV bytes per token)
필요 대역폭 하한 [GB/s] = bytes_per_step × (steps/s)
```
실측 tokens/s로부터 **필요 대역폭 하한**을 역산해 RNGD 스펙 대역폭과 비교하면,
counter 없이도 "memory-bound에 얼마나 가까운가"를 **실측 기반으로** 말할 수 있다.
FP8 모델(Qwen3-8B-FP8)이라 weight 바이트가 절반이라는 점도 분석 소재가 된다.

**수정 (b) — 병목 판정을 데이터 패턴 규칙으로 정의한다**

| 관측 패턴 | 판정 |
|---|---|
| batch↑ → aggregate tps 거의 선형 증가, per-user tps 유지 | 아직 포화 아님 (`headroom`) |
| batch↑ → aggregate tps 크게 증가, per-user tps는 완만히 감소 | weight 로딩 비용 분산 효과 → `memory_bandwidth` 지배 신호 |
| batch↑ → aggregate tps 포화, per-user tps 반비례 감소 | `compute` 또는 스케줄링 포화 |
| `kv_cache_usage` → 100% 근접 & `waiting_samples` 증가 | `memory_capacity` |
| `running_samples` < 요청 concurrency & `waiting_samples` 큼 | `concurrency_scheduling` |
| 위 어느 것에도 안 맞음 | **`unknown`** (억지로 분류하지 않는다) |

planner는 **이 규칙의 출력만** 사용한다. 규칙이 안 맞으면 `unknown`을 그대로 표시한다.

---

### IMP-8. P95의 모집단 크기가 정의되어 있지 않다

concurrency 32에서 요청 20건을 재고 P95를 말하는 것은 통계적으로 무의미하다.

**수정**: 조건당 **완료 요청 최소 100건**. 미달이면 P95를 출력하지 않고 `insufficient_samples`로 표기한다.
warm-up 요청은 모집단에서 제외하되 **삭제하지 않고 `is_warmup=true`로 남긴다** (기획서 §2 "이상치 삭제 금지" 원칙과 일치).

---

### IMP-9. Embedding의 "QPS" 단위가 모호하다

batch로 여러 문장을 한 요청에 담으면 requests/s와 documents/s가 다르다.
기획서 §1 B는 `required_qps / measured_qps_per_rngd`라고만 쓴다 — 단위가 섞이면 결과가 batch 배만큼 틀린다.

**수정**: **documents/s로 통일**한다. requests/s와 per-request latency는 별도 컬럼으로 유지하되
capacity 계산에는 documents/s만 쓴다. UI 입력도 "초당 문서 수" 또는 "총 문서 수 / 목표 시간"으로 받는다.

---

### IMP-10. target utilization 이중 적용 방지 — 수식을 하나만 정의한다

기획서 검토 프롬프트 스스로가 지적한 항목이다. 정의를 못박는다.

```
required_output_tps      = concurrent_users × target_output_tps_per_user        [tok/s]
usable_tps_per_card      = measured_aggregate_output_tps(row) × target_utilization
n_cards                  = ceil(required_output_tps / usable_tps_per_card)
estimated_utilization    = required_output_tps / (n_cards × measured_aggregate_output_tps(row))
```

- `target_utilization`은 **분모에만 한 번** 적용된다.
- `headroom` 입력은 `target_utilization = 1 − headroom`으로 변환해서 들어오며, 그 외 어디에도 곱하지 않는다.
- `estimated_utilization`에는 곱하지 않는다 (이건 결과이지 제약이 아니다).
- 이 수식을 검증하는 **단위 테스트를 먼저 쓴다** (`tests/test_capacity.py`).

---

### IMP-11. 모델 선택 — Llama 구형 아티팩트는 호환성 리스크가 있다

`Llama-3.1-8B-Instruct` 아티팩트는 `furiosa_llm_version = 254c5ee`로 빌드되었는데
설치본은 `a13b5a4`다. `Qwen3-8B-FP8`은 최신 `.fxb`다.

**수정**: P0 대상 모델은 **`Qwen3-8B-FP8`**을 권장한다.
Llama는 로드 검증에 성공하면 P2(모델 비교)로 돌린다.

---

### IMP-12. 서버 기동/모델 로드 시간이 미측정 — 2일 일정의 최대 미지수

**수정**: 첫 작업을 `Qwen2.5-0.5B-Instruct`(972M) **end-to-end 스모크 테스트**로 잡는다.
기동 → 요청 → 스트리밍 TTFT 계측 → `/metrics` 수집 → JSONL 저장까지가 도는지 짧게 확인한 뒤 8B로 올린다.
파이프라인 버그를 8B 로드 시간을 기다리며 디버깅하는 상황을 피한다.

---

## Improvement — 완성도와 포트폴리오 가치를 올린다

### ADD-1. Bucket 경계 실험 (저비용 · 고효과, 최우선 추가 권장)

prefill 버킷 간격이 128임을 이미 알고 있으므로, input length를 경계 주변으로 스윕한다:
`896 / 1000 / 1024 / 1030 / 1100 / 1152`

TTFT가 계단형으로 튀는 그래프 하나가 **"peak FLOPS로는 설명할 수 없는, 컴파일러 기반 NPU의 실제 성능 특성"**을 증명한다.
짧은 프롬프트 몇 개라 비용이 거의 0이고, IMP-1의 planner 경고 기능과 직결된다.

### ADD-2. 측정 데이터를 리포지토리에 커밋 → NPU 없이 재현 가능하게

벤치마크는 서버에서, 분석·planner는 로컬에서 돌린다. **`data/raw/*.jsonl`이 두 세계의 인터페이스**다.
raw 데이터를 커밋해 두면 리뷰어가 RNGD 없이도 planner와 그래프를 전부 재현할 수 있다.
포트폴리오에서 이 차이는 크다.

### ADD-3. 그래프 추가 제안

기획서 §4의 5개에 더해, 위 분석과 직결되는 것들:
- **Concurrency vs TTFT** (IMP-2 가설의 직접 증거)
- **Input length vs TTFT, 버킷 경계 표시** (ADD-1)
- **Concurrency vs KV cache usage** (메모리 용량 병목)
- **필요 대역폭 하한 vs 스펙 대역폭** (IMP-7a)
- **1카드 vs 2카드 scaling efficiency** (CRIT-4)

### ADD-4. Mock 데이터 격리 강화

기획서 §6은 mock 표시를 요구한다. 구체적으로:
- 경로를 `data/sample/`로 분리 (기획서 제안 그대로)
- 모든 mock 레코드에 `"source": "mock"` 필드를 **필수**로 넣고, 로더가 이 필드 없는 파일을 거부
- UI 상단에 데이터 출처 배너를 항상 표시: `MOCK DATA — 실측 아님` / `MEASURED — edu-hlu-1, 2026-08-25`
- 실측 데이터가 들어오면 mock 로더는 **자동 비활성화**

---

## Verified — 실측으로 확인, 기획서대로 진행 가능

| 항목 | 확인 내용 |
|---|---|
| RNGD 가용성 | 8장 인식, 전부 유휴, 정상 (`furiosa-smi info` / `ps`) |
| 서빙 경로 | OpenAI 호환 서버가 chat/completions + embeddings + tokenize + metrics 전부 제공 → **별도 추상화 불필요** |
| 빌드 생략 | 컴파일 완료 아티팩트가 캐시에 존재 → `furiosa-llm build`(장시간) 생략 가능 |
| 실험 통제 | `ignore_eos`, `min_tokens`, `stream_options.include_usage` 실제 지원 |
| 메모리 metric | `kv_cache_usage`, `prefix_cache_hit_rate`, `running/waiting_samples` 실제 제공 → 기획서의 metric 목록 대부분 충족 가능 |
| 카드 격리 | `--devices npu:X`로 카드 단위 고정 가능 |
| Embedding | `Qwen3-Embedding-0.6B` `.fxb` 존재 + `/v1/embeddings` 라우트 존재 (기동 검증은 미해결 7번) |
| 인프라 | 디스크 395G 여유, HF 네트워크 접근 가능, CPU 128코어(부하 생성기가 병목될 일 없음) |
| 프로젝트 구조 | 기획서 §5 구조는 타당. 그대로 채택 |
| 원칙 | §6 mock 격리, §7 confidence 표기, §10 실측 우선 원칙은 전부 유지 |

---

## 요약

| 등급 | 개수 | 핵심 |
|---|---|---|
| Critical | 5 | prefill 버킷 제약, prefix cache 오염, "1장" 정의, 선형 외삽, 카드 수 순환 의존 |
| Important | 12 | 측정 정의(TTFT/TPOT/토큰수), 병목 판정 근거, 단위 통일, 수식 단일화 |
| Improvement | 4 | 버킷 경계 실험, 데이터 커밋, 그래프 추가, mock 격리 |
| Verified | 10 | 환경·API·metric·아티팩트 모두 프로젝트 수행에 충분 |

**총평**: 환경은 기획서가 가정한 것보다 오히려 좋다 (빌드 생략 가능, metric 풍부, 실험 통제 파라미터 존재).
반면 **버킷 구조라는, 기획서가 몰랐던 하드웨어 제약**이 실험 설계와 planner 양쪽을 바꾼다.
이 제약을 회피 대상이 아니라 **분석 주제로 삼는 것**이 이 프로젝트의 가장 큰 개선 방향이다.
