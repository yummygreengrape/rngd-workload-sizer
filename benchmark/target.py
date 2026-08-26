"""측정 대상(target) 정의.

전용 서버와 호스팅 엔드포인트를 같은 코드로 때리되, **출처가 절대 섞이지 않도록**
Target이 source 태그를 강제로 들고 다닌다 (docs/03-api-findings.md §7).

- local  : furiosa-llm serve 로 띄운 전용 서버. capacity 계산에 쓸 수 있는 유일한 소스.
- hosted : FuriosaAI 호스팅 엔드포인트. 카드 수 미상 + 멀티테넌트 + 네트워크 포함이라
           개발/검증에만 쓴다. 여기서 나온 수치는 planner가 거부한다.
"""

from dataclasses import dataclass
import os
from urllib.parse import urlparse

from .schema import SOURCE_HOSTED_ENDPOINT, SOURCE_MEASURED_LOCAL


@dataclass
class Target:
    name: str
    base_url: str          # 예: http://127.0.0.1:8000/v1
    model: str
    source: str
    api_key: str | None = None
    api_path: str = "/completions"   # docs/03 §2: chat 아닌 completions 가 기본
    timeout_s: float = 300.0

    @property
    def is_chat(self) -> bool:
        return self.api_path.endswith("/chat/completions") or self.api_path == "/chat/completions"

    @property
    def parsed(self):
        return urlparse(self.base_url)

    def path_for(self, endpoint: str) -> str:
        """base_url 의 경로(보통 /v1) + endpoint 를 합쳐 요청 경로를 만든다."""
        base = self.parsed.path.rstrip("/")
        return base + (endpoint if endpoint.startswith("/") else "/" + endpoint)

    def root_path_for(self, endpoint: str) -> str:
        """서버 루트 기준 경로. /tokenize, /metrics, /health 는 /v1 아래가 아니다."""
        base = self.parsed.path.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base + (endpoint if endpoint.startswith("/") else "/" + endpoint)

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


def local_target(model: str, base_url: str = "http://127.0.0.1:8000/v1", **kw) -> Target:
    return Target(name="local", base_url=base_url, model=model,
                  source=SOURCE_MEASURED_LOCAL, **kw)


def hosted_target(model: str, **kw) -> Target:
    """호스팅 엔드포인트. 키는 환경변수에서만 읽는다 — 코드/설정에 넣지 않는다."""
    key = os.environ.get("FURIOSA_API_KEY")
    if not key:
        raise RuntimeError(
            "FURIOSA_API_KEY 가 없습니다. 키는 커밋되지 않는 곳(_work/.env)에 두고 "
            "`set -a; . _work/.env; set +a` 로 주입하세요."
        )
    base = os.environ.get("FURIOSA_BASE_URL", "https://endpoint.access.furiosa.dev/v1")
    return Target(name="hosted", base_url=base, model=model,
                  source=SOURCE_HOSTED_ENDPOINT, api_key=key, **kw)


def build_target(kind: str, model: str, **kw) -> Target:
    if kind == "local":
        return local_target(model, **kw)
    if kind == "hosted":
        return hosted_target(model, **kw)
    raise ValueError(f"알 수 없는 target: {kind!r} (local|hosted)")
