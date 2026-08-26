"""raw run → 실측 테이블 변환.

카드 수는 **capacity 계산의 분모**다. 여기서 틀리면 모든 결과가 배수로 틀린다.
그리고 조용히 틀린다 — 값이 그럴듯해 보인다.

교차검증 세션(2026-08-26)이 지적한 결함의 회귀 테스트를 포함한다:
실험 코드 이름 표에만 의존하면 새 코드로 측정한 데이터가 말없이 버려졌다.
"""

import pytest

from analysis.process import (
    LEGACY_RUN_CARDS, infer_cards, overlapping_runs, run_spans, server_log_mesh,
)


def meta(*, n_cards=None, experiment="X", base_url="http://127.0.0.1:8000/v1",
         source="measured_local"):
    return {"experiment": experiment, "source": source,
            "target": {"n_cards": n_cards, "base_url": base_url, "model": "m"}}


class TestCardLookupOrder:
    """meta → 서버 로그 → 이름 표. 순서가 신뢰도 순이다."""

    def test_meta_wins(self):
        n, src = infer_cards(meta(n_cards=4, experiment="C2x1"), [(8000, 2)])
        assert (n, src) == (4, "meta")          # 표(1)·로그(2)보다 우선

    def test_server_log_when_meta_missing(self):
        n, src = infer_cards(meta(experiment="없는코드", base_url="http://h:8004/v1"),
                             [(8004, 4)])
        assert (n, src) == (4, "server_log")

    def test_legacy_table_is_last_resort(self):
        n, src = infer_cards(meta(experiment="C1x8"), [])
        assert (n, src) == (8, "legacy_table")

    def test_unknown_when_nothing_resolves(self):
        n, src = infer_cards(meta(experiment="새실험"), [])
        assert n is None and src == "unknown"

    def test_ambiguous_port_is_not_guessed(self):
        """같은 포트가 서로 다른 구성으로 여러 번 쓰였으면 추측하지 않는다.

        실제로 8000 번 포트를 1장 서버와 8장 서버가 번갈아 썼다.
        """
        n, src = infer_cards(meta(experiment="새실험", base_url="http://h:8000/v1"),
                             [(8000, 1), (8000, 8)])
        assert n is None and src == "unknown"

    def test_same_config_repeated_is_trusted(self):
        n, src = infer_cards(meta(experiment="새실험", base_url="http://h:8000/v1"),
                             [(8000, 2), (8000, 2)])
        assert (n, src) == (2, "server_log")


class TestNewExperimentCodesSurvive:
    """다음 세션 계획의 실험 코드는 이름 표에 없다.

    표에만 의존하던 구현에서는 이 데이터가 조용히 사라졌다.
    """

    @pytest.mark.parametrize("code", ["B2", "B3", "D1emb", "B1hi2"])
    def test_planned_codes_are_not_in_the_table(self, code):
        assert code not in LEGACY_RUN_CARDS      # 표는 이들을 모른다

    @pytest.mark.parametrize("code", ["B2", "B3", "D1emb", "B1hi2"])
    def test_but_they_resolve_via_cards_flag(self, code):
        n, src = infer_cards(meta(n_cards=2, experiment=code), [])
        assert (n, src) == (2, "meta")

    @pytest.mark.parametrize("code", ["B2", "B3", "D1emb"])
    def test_and_via_server_log_if_flag_forgotten(self, code):
        n, src = infer_cards(meta(experiment=code, base_url="http://h:8002/v1"),
                             [(8002, 2)])
        assert (n, src) == (2, "server_log")


class TestServerLogParsing:
    def test_extracts_port_and_dp(self, tmp_path):
        (tmp_path / "serve_4card.log").write_text(
            "INFO Device mesh: 4 DP group(s), TP=8, devices=npu:4,npu:5,npu:6,npu:7\n"
            "INFO Uvicorn running on http://127.0.0.1:8004 (Press CTRL+C to quit)\n",
            encoding="utf-8")
        assert server_log_mesh(str(tmp_path)) == [(8004, 4)]

    def test_multiple_starts_in_one_log(self, tmp_path):
        (tmp_path / "s.log").write_text(
            "Device mesh: 1 DP group(s)\nUvicorn running on http://127.0.0.1:8000\n"
            "Device mesh: 8 DP group(s)\nUvicorn running on http://127.0.0.1:8008\n",
            encoding="utf-8")
        assert server_log_mesh(str(tmp_path)) == [(8000, 1), (8008, 8)]

    def test_mesh_without_uvicorn_is_ignored(self, tmp_path):
        """기동에 실패한 서버는 포트를 열지 못한다. 짝이 없으면 버린다."""
        (tmp_path / "s.log").write_text("Device mesh: 4 DP group(s)\n", encoding="utf-8")
        assert server_log_mesh(str(tmp_path)) == []

    def test_missing_directory(self):
        assert server_log_mesh("/nonexistent/path") == []


class TestRunOverlap:
    """다른 측정과 동시에 돌면 처리량이 낮게 나온다. 겹침을 표시해야 한다."""

    def _run(self, rid, started, windows):
        return {"run_id": rid, "meta": {"started_at": started},
                "summary": [{"window_s": w} for w in windows]}

    def test_detects_overlap(self):
        runs = [self._run("a", "2026-08-25T20:00:00+09:00", [600]),
                self._run("b", "2026-08-25T20:05:00+09:00", [600])]
        ov = overlapping_runs(run_spans(runs))
        assert ov["a"] == ["b"] and ov["b"] == ["a"]

    def test_sequential_runs_do_not_overlap(self):
        runs = [self._run("a", "2026-08-25T20:00:00+09:00", [60]),
                self._run("b", "2026-08-25T20:05:00+09:00", [60])]
        ov = overlapping_runs(run_spans(runs))
        assert ov["a"] == [] and ov["b"] == []

    def test_run_without_timestamp_is_skipped(self):
        runs = [{"run_id": "x", "meta": {}, "summary": [{"window_s": 10}]}]
        assert run_spans(runs) == {}


class TestHostedIsClassifiedBySourceOnly:
    """호스팅 측정에는 카드 수라는 개념이 없다.

    교차검증 F2: 카드 수를 찾았는지로 hosted 를 판정하면, 이름 표에 우연히 같은
    실험 코드가 있을 때 호스팅 행에 카드 수가 붙는다. 실제로 E1 이 legacy_table 로
    1장이 붙고 n_cards_known=true 로 표시됐다.

    추가로, run 단위 변수를 조건 루프 안에서 덮어쓰면 첫 조건 이후 판정이 뒤집힌다.
    E4 12행 중 10행이 그렇게 known=true 가 됐다.
    """

    def _runs(self, source, experiment, n_conditions):
        return [{
            "run_id": f"r_{experiment}_x",
            "meta": {"experiment": experiment, "source": source,
                     "started_at": "2026-08-25T20:00:00+09:00",
                     "target": {"model": "m", "base_url": "http://h:9999/v1"}},
            "summary": [{"spec": {"concurrency": 4, "input_tokens": 512,
                                  "output_tokens": 128},
                         "aggregate_output_tps": 100.0, "window_s": 60.0,
                         "n_measured_ok": 100, "validation": {}}
                        for _ in range(n_conditions)],
        }]

    def test_hosted_never_claims_a_card_count(self):
        from analysis.process import to_rows
        # E1 은 LEGACY_RUN_CARDS 에 1 로 들어 있다 — 그래도 호스팅이면 붙으면 안 된다
        rows, _ = to_rows(self._runs("hosted_endpoint", "E1", 3))
        assert rows and all(r["n_cards_known"] is False for r in rows)
        assert all(r["card_source"] == "정의되지 않음 (hosted)" for r in rows)

    def test_every_condition_of_a_run_is_classified_the_same(self):
        """첫 조건과 나머지가 갈리면 안 된다 (루프 반송 상태 회귀)."""
        from analysis.process import to_rows
        rows, _ = to_rows(self._runs("hosted_endpoint", "E4", 6))
        assert len(rows) == 6
        assert len({r["n_cards_known"] for r in rows}) == 1
        assert len({r["card_source"] for r in rows}) == 1

    def test_measured_local_still_resolves_normally(self):
        from analysis.process import to_rows
        rows, _ = to_rows(self._runs("measured_local", "C1x8", 3))
        assert rows and all(r["n_cards_known"] is True for r in rows)
        assert all(r["n_cards"] == 8 for r in rows)

    def test_hosted_keeps_total_concurrency_not_per_card(self):
        """카드 수를 모르므로 '카드당' 이 성립하지 않는다. 총량을 그대로 둔다."""
        from analysis.process import to_rows
        rows, _ = to_rows(self._runs("hosted_endpoint", "E1", 1))
        assert rows[0]["concurrency_per_card"] == rows[0]["concurrency_total"] == 4


class TestCapacityFlagsReachThePlanner:
    """오염 판정이 행에 실려야 planner 가 그걸 보고 거부할 수 있다.

    교차검증 X9: 하네스는 `prefix_cache.verdict = CONTAMINATED` 를 기록하고 있었는데
    `to_rows` 가 그 판정을 행에 싣지 않아, planner 까지 갈 경로 자체가 없었다.
    README 는 "초과 시 자동 무효 처리" 라고 적고 있었다.
    """

    def _run(self, source, validation):
        return [{
            "run_id": "r_B1_x",
            "meta": {"experiment": "B1", "source": source,
                     "started_at": "2026-08-25T20:00:00+09:00",
                     "target": {"model": "m", "n_cards": 1,
                                "base_url": "http://127.0.0.1:8000/v1"}},
            "summary": [{"spec": {"concurrency": 4, "input_tokens": 512,
                                  "output_tokens": 128},
                         "aggregate_output_tps": 100.0, "window_s": 60.0,
                         "n_measured_ok": 100, "validation": validation}],
        }]

    def test_clean_condition_is_usable(self):
        from analysis.process import to_rows
        rows, _ = to_rows(self._run("measured_local", {"source": "measured_local"}))
        assert rows[0]["capacity_usable"] is True
        assert rows[0]["capacity_blocks"] == []

    def test_contaminated_condition_is_flagged_with_reason(self):
        from analysis.process import to_rows
        rows, _ = to_rows(self._run("measured_local", {
            "source": "measured_local",
            "prefix_cache": {"verdict": "CONTAMINATED", "max_cached_ratio": 0.99,
                             "limit": 0.05, "requests_over_limit": 30},
        }))
        assert rows[0]["capacity_usable"] is False
        assert "prefix cache" in rows[0]["capacity_blocks"][0]

    def test_row_is_kept_not_dropped(self):
        """버리면 왜 사라졌는지가 남지 않는다. 거부는 planner 로드 게이트가 한다."""
        from analysis.process import to_rows
        rows, skipped = to_rows(self._run("measured_local", {
            "source": "measured_local",
            "prefix_cache": {"verdict": "CONTAMINATED", "max_cached_ratio": 0.99,
                             "limit": 0.05, "requests_over_limit": 30},
        }))
        assert len(rows) == 1 and not skipped

    def test_stored_flag_is_recomputed_not_trusted(self):
        """옛 run 은 출처만 보던 규칙으로 쓰인 `capacity_usable` 을 달고 있다."""
        from analysis.process import to_rows
        rows, _ = to_rows(self._run("measured_local", {
            "source": "measured_local",
            "capacity_usable": True,            # 옛 규칙의 판정
            "output_length": {"verdict": "VARIES", "requested": 128,
                              "distinct_observed": [64, 128]},
        }))
        assert rows[0]["capacity_usable"] is False
