# 03. API 실측 확인 결과 — 하네스 설계를 바꾸는 것들

> 확인일: 2026-08-25 / 대상: FuriosaAI 호스팅 추론 엔드포인트 (`/v1`, 임시 평가용 키)
> **이 문서의 수치는 capacity 계산에 사용하지 않는다.** 카드 수를 알 수 없고 멀티테넌트이며
> 네트워크가 포함되어 있다. 여기서 얻는 것은 **API의 동작 방식**과 **방법론 검증**이다.
> 실측 capacity의 유일한 소스는 전용 서버다 ([02-plan.md](02-plan.md) §0).

## 요약 — 이 확인으로 바뀐 설계 결정 4가지

| # | 발견 | 설계 변경 |
|---|---|---|
| 1 | 추론(reasoning) 모델이 `content`가 아닌 **`reasoning` delta**로 토큰을 낸다 | **TTFT 정의 수정** — `content` 또는 `reasoning` 중 먼저 오는 non-empty delta |
| 2 | `/v1/completions`는 chat template·reasoning을 **완전히 우회**한다 | **LLM 벤치마크는 `/v1/completions`로 한다** (chat 아님) |
| 3 | prefix cache가 긴 컨텍스트에서 prefill 비용을 **최대 ~21배 과소평가**시킨다 (실측 재현) | CRIT-2 방어를 **필수**로 격상, 검증 지표를 결과에 남긴다 |
| 4 | 호스팅에는 `/tokenize`·`/metrics`가 **없다** (404) | 하네스는 **로컬 tokenizer 경로**를 함께 가져야 한다 |

---

## 1. ⚠️ 추론 모델은 `content`가 비어 있다 — TTFT 정의가 깨진다

`Qwen3-32B-FP8`에 chat 요청을 스트리밍하면:

```
[0] @806.5ms  {"role": "assistant", "content": ""}
[1] @826.1ms  {"reasoning": "\n"}
[2] @839.7ms  {"reasoning": "Okay"}
[3] @850.8ms  {"reasoning": ","}
```

`max_tokens=32`로 끝까지 받아도 **`content`는 끝내 비어 있고** `text_len=0`이다.
`usage`는 `completion_tokens: 32`, `completion_tokens_details.reasoning_tokens: 31`을 보고한다.
`gpt-oss-120b`도 동일하게 `reasoning` delta를 낸다.

**왜 중요한가**
[01-spec-review.md](01-spec-review.md) IMP-3에서 정한 *"첫 non-empty `content` delta"* 정의를 그대로 쓰면
추론 모델에서 **TTFT가 영원히 측정되지 않는다.** 조용히 실패하는 종류의 버그다.

**수정된 정의**
```
TTFT = (delta.content 또는 delta.reasoning 중 먼저 도착한 non-empty 값의 수신 시각) − (요청 전송 시각)
```
- 필드명은 `reasoning`이다 (`reasoning_content` 아님).
- 첫 chunk는 항상 `{"role":"assistant","content":""}` role-only이므로 **반드시 건너뛴다.**
- 결과 레코드에 `reasoning_tokens`를 별도 컬럼으로 남긴다 — 추론 토큰은 사용자가 보지 않지만
  **NPU 자원은 그대로 소비하므로 capacity 계산에는 포함되어야 한다.**

**추론 끄기** (chat 경로를 쓸 경우)
```json
"chat_template_kwargs": {"enable_thinking": false}
```
→ `reasoning: null`, 정상 `content` 반환 확인.

---

## 2. ✅ `/v1/completions`가 벤치마크에 더 적합하다

같은 모델에 `/v1/completions`로 요청하면:

```
[0] @761.7ms  {"index":0,"text":" London","finish_reason":null}
[1] @774.7ms  {"index":0,"text":".","finish_reason":null}
usage: {prompt_tokens: 5, completion_tokens: 8, prompt_tokens_details: {cached_tokens: 0}}
```

- chat template이 개입하지 않는다 → **프롬프트 토큰 수를 정확히 통제**할 수 있다
- reasoning이 발생하지 않는다 → 출력 토큰이 곧 측정 대상이다
- delta 필드가 `text` 하나뿐이라 파싱이 단순하다

**결정: Prefill/Decode 벤치마크(E1~E5)는 `/v1/completions`를 쓴다.**
chat 경로는 "실제 서비스 형태"를 보려는 P2 비교 실험에서만 쓰고,
그때는 `enable_thinking:false`로 고정하고 chat template 오버헤드를 별도 컬럼으로 기록한다.

---

## 3. ⚠️ prefix cache 오염을 실측으로 재현했다 — CRIT-2 확정

동일 프롬프트를 3회 반복한 경우와, 매 요청 고유 프롬프트를 쓴 경우를 비교했다.
(`/v1/completions`, `max_tokens=1`, 서버 측 시간 = `x-envoy-upstream-service-time`)

| prompt tokens | 같은 프롬프트 반복 (오염) | 고유 프롬프트 (정상) | **과소평가 배율** |
|---:|---:|---:|---:|
| ~130 | 45 ms | 54 ms | 1.2× |
| ~1,000 | 55 ms | 163 ms | 3.0× |
| ~2,000 | 69 ms | 309 ms | 4.5× |
| ~4,800 | 90 ms | 653 ms | 7.3× |
| ~9,600 | 122 ms | 1,419 ms | 11.6× |
| ~19,300 | 166 ms | 3,464 ms | **20.9×** |

오염된 실행에서는 `usage.prompt_tokens_details.cached_tokens`가 `prompt_tokens - 1`이었다.
즉 **프롬프트 전체가 캐시에서 나왔다.**

**해석**
"여러 번 측정해서 median을 쓴다"는 신뢰성 장치가, prefix cache 앞에서는
**긴 컨텍스트일수록 더 크게 낙관 편향을 만든다.** 그리고 값이 그럴듯해 보이기 때문에
사후에 발견하기 어렵다. 이 표가 그 위험을 정량적으로 보여준다.

**방어 (전용 서버 기준, 변경 없음 — 다만 이제 필수)**
1. `--no-enable-prefix-caching`으로 기동
2. 매 요청 고유 프롬프트
3. `/metrics`의 `prefix_cache_hit_rate ≠ 0` 이면 실행 무효
4. 호스팅 경로에서는 `usage.prompt_tokens_details.cached_tokens`를 같은 용도로 쓴다

---

## 4. Prefill은 길이에 대해 초선형으로 증가한다 (형태 관찰)

고유 프롬프트, `max_tokens=1`, 서버 측 시간 기준:

| prompt tokens | 서버 시간 | ms / 1k tok |
|---:|---:|---:|
| 150 | 54 ms | 360.0 |
| 292 | 67 ms | 229.5 |
| 604 | 115 ms | 190.4 |
| 1,224 | 163 ms | 133.2 |
| 2,389 | 309 ms | 129.3 |
| 4,785 | 653 ms | 136.5 |
| 9,561 | 1,419 ms | 148.4 |
| 19,317 | 3,464 ms | 179.3 |

- 짧은 길이에서는 **고정 오버헤드가 지배**한다 (150 tok에서 360 ms/1k)
- 1.2k~4.8k 구간이 가장 효율적 (~130 ms/1k)
- **~5k를 넘으면 다시 악화**된다: 9,561 → 19,317 토큰은 2.02배인데 시간은 3,464/1,419 = **2.44배**
  → attention의 길이 제곱 항이 드러나는 구간으로 보인다

**이 수치를 capacity에 쓰지 않는 이유**: 카드 수 미상, 멀티테넌트, 큐 대기 포함.
**그래도 가치 있는 이유**: E1/E2의 실험 설계(길이 스윕 + `max_tokens=1` = prefill 프로브)가
실제로 신호를 뽑아낸다는 것을 전용 서버를 점유하기 전에 확인했다.

---

## 5. 호스팅 엔드포인트의 제약 (하네스가 대비해야 할 것)

### 없는 엔드포인트 — 전부 404
```
/tokenize   /detokenize   /tokenizer_info   /metrics   /health   /version
```
게이트웨이가 `/v1/*`만 통과시킨다.

→ **하네스는 토크나이저 경로를 두 개 가져야 한다.**
- `local` 모드: 서버의 `POST /tokenize` 사용 (정확, [01-spec-review.md](01-spec-review.md) IMP-5)
- `hosted` 모드: `transformers` 토크나이저를 로컬에서 로드
- 어느 쪽이든 `usage.prompt_tokens`로 **사후 검증**하고 불일치를 기록한다

### 네트워크가 관측 지연의 대부분이다
```
dns 2ms | connect 175ms | TLS 355ms(누적) | TTFB 720ms
```
`/v1/completions` 짧은 요청에서 `x-envoy-upstream-service-time: 8ms`인데
관측 wall-clock은 762ms였다 → **관측 지연의 약 99%가 네트워크·게이트웨이다.**

→ 호스팅 경로에서 얻은 wall-clock TTFT/latency는 **의미가 없다.**
   `x-envoy-upstream-service-time`(Envoy 게이트웨이 헤더)만 형태 관찰에 쓴다.
   이 헤더는 전용 서버에는 없다 — 대신 전용 서버에는 `/metrics`가 있다.

### 동시성 — rate limit은 관측되지 않았다
`/v1/completions`, 프롬프트 ~400 tok, `max_tokens=64`, `ignore_eos=true`:

| concurrency | 성공 | wall | aggregate out tok/s | e2e median |
|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.62 s | 39.6 | 1.62 s |
| 2 | 2/2 | 1.69 s | 75.8 | 1.69 s |
| 4 | 4/4 | 1.99 s | 128.8 | 1.98 s |
| 8 | 8/8 | 2.75 s | 186.4 | 2.71 s |
| 16 | 16/16 | 3.04 s | 337.1 | 2.99 s |

16까지 에러·429 없음. 다만 각 요청에 ~0.75 s의 네트워크가 포함되어 있어
**per-request 지연은 해석 불가**이고, aggregate 증가 형태만 참고값이다.

### 파라미터 지원 여부 (요청이 거부되지 않음을 확인)
| 파라미터 | 결과 |
|---|---|
| `stream_options.include_usage` | ✅ 마지막 chunk에 `usage` 도착 |
| `ignore_eos`, `min_tokens` | ✅ 400 없이 수용, 요청한 `completion_tokens` 정확히 반환 |
| `chat_template_kwargs.enable_thinking` | ✅ 동작 (reasoning 비활성화) |
| `prompt_tokens_details.cached_tokens` | ✅ prefix cache 적중량 노출 |

---

## 6. 모델 카탈로그 (`GET /v1/models`)

| 모델 | max_model_len |
|---|---:|
| `furiosa-ai/gpt-oss-120b` | 131,072 |
| `furiosa-ai/K-EXAONE-236B-A23B-NVFP4A16` | 49,483 |
| `furiosa-ai/EXAONE-4.0-32B-FP8` | 131,072 |
| `furiosa-ai/Qwen3-32B-FP8` | 40,960 |
| `furiosa-ai/Qwen3-VL-32B-Instruct` | 262,144 |
| `furiosa-ai/Qwen3-Embedding-8B` | 8,192 |
| `furiosa-ai/Qwen3-Reranker-8B` | 8,192 |

- Embedding 8B: `dim=4096`, 2문장 요청에 서버 시간 1,692 ms
- Rerank 8B: `relevance_score` 반환, 정상 동작
- P0 후보인 `Qwen3-8B-FP8`은 호스팅 목록에 없다 (공유 서비스에는 더 큰 모델을 올린 것으로 보임).
  전용 서버에 컴파일된 아티팩트가 있으므로 모델 선택을 바꿀 이유는 아니다.

---

## 7. 데이터 격리 규칙 (필수)

호스팅에서 나온 어떤 수치도 capacity 계산에 들어가서는 안 된다.

- 하네스에 `--target {local,hosted}` 스위치를 둔다
- `hosted`로 얻은 모든 레코드에 `"source": "hosted_endpoint"`를 **강제로 박는다**
- `planner/benchmark_store.py`는 `source != "measured_local"`인 행을 **로드 단계에서 거부**한다
  (mock 격리 ADD-4와 동일한 장치)
- UI 배너에 데이터 출처를 항상 표시한다

---

## 8. 하네스 검증 실행 (2026-08-25)

`benchmark/` 하네스를 호스팅 엔드포인트로 end-to-end 검증했다.
**아래 수치는 capacity 가 아니다** — 카드 수 미상, 멀티테넌트, 요청마다 ~0.3s 네트워크 포함.
확인하려는 것은 "하네스가 신호를 제대로 뽑아내는가"이다.

```
Qwen3-32B-FP8 / /v1/completions / in=512 out=128(ignore_eos) / 조건당 25건
```

| concurrency | aggregate out tok/s | TTFT p50 | TPOT p50 |
|---:|---:|---:|---:|
| 1 | 61.8 | 304 ms | 13.9 ms |
| 2 | 115.8 | 294 ms | 14.3 ms |
| 4 | 185.2 | 408 ms | 15.5 ms |
| 8 | 281.7 | 598 ms | 17.0 ms |
| 16 | 426.3 | 1,169 ms | 20.4 ms |
| 32 | 521.4 | 2,432 ms | 28.2 ms |

동시성 32배에 처리량은 8.4배, TTFT 는 8.0배 악화, TPOT 은 2.0배 악화.
**처리량은 포화하는데 TTFT 는 계속 나빠지는 형태** — IMP-2 가설이 예측한 모양이다.
전용 서버에서 같은 형태가 나오는지 확인하는 것이 E4 의 목적이다.

### 검증 블록이 실제로 동작함

`summary.json` 의 `validation` (conc=32):

```json
{
  "prefix_cache":  {"max_cached_ratio": 0.0097, "requests_over_limit": 0, "verdict": "clean"},
  "output_length": {"requested": 128, "distinct_observed": [128], "verdict": "fixed"},
  "input_length":  {"target": 512, "median_actual": 514.0, "drift_pct": 0.39, "verdict": "ok"},
  "concurrency":   {"note": "/metrics 없음 — 실제 동시 처리량을 확인할 수 없습니다."},
  "capacity_usable": false
}
```

- `ignore_eos` 로 출력이 정확히 128 토큰으로 고정됐다
- 입력 길이 편차 0.39%
- prefix cache 최대 적중 0.97% (선두 토큰 5개) → 정상
- 호스팅 출처이므로 capacity 사용이 자동으로 차단됐다

### 이 과정에서 잡은 하네스 버그 2건

**(1) 재시도가 지연 지표를 오염시킴 — TTFT 가 −302초로 관측됨**

conc=32 에서 스트림이 멈춰 소켓 타임아웃이 났고, 재연결 후 재시도했다.
그런데 재시도가 `t_send` 만 갱신하고 이전 시도의 `t_first_token` 을 남겨두어
TTFT 가 음수가 됐다. 에러로 잡히지도 않았다 — 재시도가 성공했기 때문이다.

수정:
- 시도마다 타이밍/usage 상태를 전부 비운다
- **토큰을 이미 받기 시작한 뒤에는 재시도하지 않는다.** 재시도는 다른 요청이므로
  측정 실패로 기록하는 것이 정직하다
- 음수 지연이 나오면 값을 내지 않고 `negative_latency` 에러로 드러낸다
- 회귀 테스트 3건 추가 (`tests/test_client.py`)

**(2) prefix cache 오염 판정이 전부 오탐**

`cached_tokens > 0` 을 오염으로 봤는데, 서로 다른 프롬프트라도 선두 토큰 2~5개는
공유되어 항상 0보다 크다. 전 조건에서 오탐이 떴다.
→ **비율 기준(5%)** 으로 바꿨다. 실제 오염(§3)은 90% 이상이므로 충분히 구분된다.

### 부수 확인

- `x-envoy-upstream-service-time` 은 **스트리밍에서는 응답 헤더까지의 시간**이다.
  prompt 8,192 토큰에서 `upstream_ms=54` 인데 TTFT 는 1,476 ms 였다.
  서버측 prefill 시간을 보려면 비스트리밍 요청을 써야 한다 (§4 는 비스트리밍으로 측정).
- 호스팅 엔드포인트는 conc=32 에서 스트림이 장시간 멈추는 경우가 있다.
  전용 서버에서 같은 현상이 나오는지는 별개로 확인해야 한다.
