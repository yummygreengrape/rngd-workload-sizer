# benchmark — 측정 하네스

표준 라이브러리만 사용한다. 평가 서버(Python 3.10)와 개발 머신(3.14)에서
동일하게 돌아야 하고, 평가 서버에 pip 설치를 요구하지 않기 위해서다.

## 모듈

| 파일 | 역할 |
|---|---|
| `target.py` | 측정 대상 정의. `local`/`hosted` 를 구분하고 **출처 태그를 강제**한다 |
| `prompts.py` | 목표 토큰 길이에 맞는 **고유** 프롬프트 생성 (prefix cache 회피) |
| `client.py` | 스트리밍 요청과 TTFT/TPOT 추출. 워커당 keep-alive 커넥션 1개 |
| `metrics.py` | `/metrics` 폴링과 Prometheus 파싱 |
| `runner.py` | 워커 풀, warm-up 분리, 정상상태 윈도 집계, **자기검증** |
| `run_llm.py` | CLI |
| `env.py` | 재현용 환경 정보 수집 |
| `schema.py` | 레코드 스키마와 JSONL 출력 |

## 실행

전용 서버 (`furiosa-llm serve` 가 떠 있어야 함):

```bash
python -m benchmark.run_llm --target local --model furiosa-ai/Qwen3-8B-FP8 \
  --config benchmark/configs/e4_decode_concurrency.json
```

호스팅 엔드포인트 (하네스 개발/검증 전용 — capacity 계산 불가):

```bash
set -a; . ../_work/.env; set +a
python -m benchmark.run_llm --target hosted --model furiosa-ai/Qwen3-32B-FP8 \
  --config benchmark/configs/e1_path_check.json --scale 0.3 --timeout 60
```

`--scale` 은 요청 수 배율이다. 스모크 실행에 0.05~0.3 을 쓴다.

## 산출물

```
data/raw/{run_id}/
  meta.json                 환경·서버 인자·프롬프트 보정 결과 등 재현에 필요한 전부
  requests.jsonl            요청 1건 = 1줄 (warm-up 포함, is_warmup 로 구분)
  metrics_timeseries.jsonl  /metrics 1초 폴링
  summary.json              조건별 집계 + 검증 결과
```

## 측정 정의

```
TTFT = 첫 non-empty 토큰 delta 수신 시각 − 요청 전송 시각
E2E  = 마지막 delta 수신 시각 − 요청 전송 시각
TPOT = (E2E − TTFT) / (completion_tokens − 1)      # completion_tokens >= 2 일 때만

aggregate_output_tps = Σ(측정 요청의 completion_tokens) / (윈도 wall-clock)
```

`첫 non-empty 토큰 delta` 판정은 API 경로에 따라 다르다.

- `/v1/completions` : `choices[0].text`
- `/v1/chat/completions` : `choices[0].delta.content` **또는** `.reasoning`

추론 모델(Qwen3, gpt-oss)은 `content` 가 끝까지 비어 있고 `reasoning` 으로만 토큰을 낸다.
`content` 만 보면 TTFT 가 영원히 측정되지 않는다.

## 동시성 모델

워커 N개가 각자 요청을 **연속으로** 던진다. 요청이 끝나면 곧바로 다음을 보내므로
측정 구간 내내 N개가 항상 in-flight 다. "N개 보내고 다 끝나기를 기다리는" 방식은
매 배치 끝에서 동시성이 줄어 throughput 을 과소평가한다.

## 자기검증

매 조건마다 결과를 믿어도 되는지 스스로 점검하고 `summary.json` 의 `validation` 에 남긴다.
**검증에 실패해도 값을 지우지 않고 표시만 한다.**

| 항목 | 내용 |
|---|---|
| `prefix_cache` | 프롬프트 중 캐시에서 나온 **비율**. 5% 초과면 `CONTAMINATED` |
| `output_length` | `ignore_eos` 로 출력 길이가 실제로 고정됐는지 |
| `input_length` | 의도한 입력 길이와의 편차 |
| `concurrency` | `/metrics` 의 `running_samples` 로 배칭인지 큐잉인지 |
| `retries` | 커넥션 재수립 발생 여부 |
| `p95` | 표본 100건 미만이면 P95 를 내지 않는다 |
| `capacity_usable` | 전용 서버 실측이 아니면 `false` |

## 테스트

```bash
python -m pytest tests/ -q
```
