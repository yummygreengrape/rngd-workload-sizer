"""Prometheus 파싱과 요약.

metric 이름이 런타임에야 확정되므로(docs/00 미해결 3), 이름을 하드코딩하지 않고
부분 문자열로 찾는 경로를 테스트한다.
"""

from benchmark.metrics import parse_prometheus, find

SAMPLE = """
# HELP furiosa_running_samples Number of running samples
# TYPE furiosa_running_samples gauge
furiosa_running_samples{engine="0"} 12
furiosa_waiting_samples{engine="0"} 3
furiosa_kv_cache_usage_ratio{engine="0"} 0.42
furiosa_prefix_cache_hit_rate{engine="0"} 0.0
furiosa_prompt_tokens_total{engine="0"} 1.234e5
malformed line without value
furiosa_bad_value{engine="0"} NaNsomething
"""


class TestParse:
    def test_parses_valid_lines_only(self):
        s = parse_prometheus(SAMPLE)
        assert len(s) == 5
        assert s['furiosa_running_samples{engine="0"}'] == 12.0

    def test_ignores_comments_and_garbage(self):
        s = parse_prometheus(SAMPLE)
        assert not any(k.startswith("#") for k in s)
        assert not any("malformed" in k for k in s)
        assert not any("bad_value" in k for k in s)

    def test_scientific_notation(self):
        s = parse_prometheus(SAMPLE)
        assert s['furiosa_prompt_tokens_total{engine="0"}'] == 123400.0

    def test_empty_input(self):
        assert parse_prometheus("") == {}


class TestFind:
    def test_substring_search_is_case_insensitive(self):
        s = parse_prometheus(SAMPLE)
        assert find(s, "RUNNING")
        assert list(find(s, "kv_cache").values()) == [0.42]

    def test_miss_returns_empty(self):
        assert find(parse_prometheus(SAMPLE), "gpu_utilization") == {}
