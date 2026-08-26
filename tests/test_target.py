"""target 경로 조합과 출처 태깅."""

import pytest

from benchmark.schema import SOURCE_HOSTED_ENDPOINT, SOURCE_MEASURED_LOCAL
from benchmark.target import Target, build_target, hosted_target, local_target


class TestPaths:
    def test_v1_endpoints(self):
        t = local_target("m")
        assert t.path_for("/completions") == "/v1/completions"
        assert t.path_for("/chat/completions") == "/v1/chat/completions"

    def test_root_endpoints_are_not_under_v1(self):
        """/tokenize, /metrics, /health 는 /v1 아래가 아니다."""
        t = local_target("m")
        assert t.root_path_for("/tokenize") == "/tokenize"
        assert t.root_path_for("/metrics") == "/metrics"

    def test_base_url_without_v1(self):
        t = Target(name="x", base_url="http://h:8000", model="m", source=SOURCE_MEASURED_LOCAL)
        assert t.path_for("/completions") == "/completions"
        assert t.root_path_for("/metrics") == "/metrics"


class TestChatDetection:
    def test_completions_is_not_chat(self):
        assert local_target("m", api_path="/completions").is_chat is False

    def test_chat_completions_is_chat(self):
        assert local_target("m", api_path="/chat/completions").is_chat is True


class TestSourceTagging:
    def test_local_is_capacity_usable_source(self):
        assert local_target("m").source == SOURCE_MEASURED_LOCAL

    def test_hosted_is_tagged_separately(self, monkeypatch):
        monkeypatch.setenv("FURIOSA_API_KEY", "test-key")
        assert hosted_target("m").source == SOURCE_HOSTED_ENDPOINT

    def test_hosted_requires_key_from_env(self, monkeypatch):
        """키를 코드나 설정 파일에 두지 않는다 — 환경변수에서만 읽는다."""
        monkeypatch.delenv("FURIOSA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FURIOSA_API_KEY"):
            hosted_target("m")

    def test_auth_header_only_when_key_present(self, monkeypatch):
        assert "Authorization" not in local_target("m").headers()
        monkeypatch.setenv("FURIOSA_API_KEY", "k")
        assert hosted_target("m").headers()["Authorization"] == "Bearer k"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            build_target("mock", "m")
