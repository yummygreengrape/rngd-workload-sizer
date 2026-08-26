"""재현에 필요한 환경 정보를 모은다 (docs/00-environment.md §7).

없는 도구는 조용히 건너뛴다. 개발 머신(RNGD 없음)과 평가 서버 양쪽에서
같은 코드가 돌아야 하기 때문이다. 없는 값을 지어내지 않는다.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any

FURIOSA_PKGS = [
    "furiosa-llm", "furiosa-llm-native", "furiosa-native-llm-common",
    "furiosa-kernels", "furiosa-tcc", "furiosa-torch", "furiosa-torch-ext",
    "furiosa-smi-py", "furiosa-models", "torch", "transformers",
]


def _run(cmd: list[str], timeout: int = 30) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def package_versions() -> dict[str, str]:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        return {}
    out: dict[str, str] = {}
    for p in FURIOSA_PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def collect(git_root: str | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }

    smi_info = _run(["furiosa-smi", "info"])
    if smi_info:
        env["furiosa_smi_info"] = smi_info
        env["npu_present"] = True
    else:
        env["npu_present"] = False
        env["note_no_npu"] = "furiosa-smi 를 찾지 못했습니다 — 이 머신에 RNGD 가 없습니다."

    smi_ver = _run(["furiosa-smi", "version"])
    if smi_ver:
        env["furiosa_smi_version"] = smi_ver

    ps = _run(["furiosa-smi", "ps"])
    if ps:
        env["furiosa_smi_ps_at_start"] = ps

    llm_ver = _run(["furiosa-llm", "version"])
    if llm_ver:
        env["furiosa_llm_version"] = llm_ver

    if git_root:
        rev = _run(["git", "-C", git_root, "rev-parse", "--short", "HEAD"])
        dirty = _run(["git", "-C", git_root, "status", "--porcelain"])
        if rev:
            env["git_rev"] = rev
            env["git_dirty"] = bool(dirty)

    return env
