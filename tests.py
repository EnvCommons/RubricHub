"""Unit tests for RubricHub's grading result, with a scripted grader (no API calls).

Run:  pytest tests.py -v

Uses the real corpus under data/ when it is present. Otherwise a one-row synthetic shard is
written to data/ for the duration of the run and removed afterwards.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

SYNTHETIC_ROW = {
    "prompt": [{"role": "user", "content": "Name the smallest prime number."}],
    "data_source": "Chat",
    "ability": "chat",
    "reward_model": {
        "ground_truth": "REFERENCE-ANSWER: the smallest prime is 2",
        "rubrics": [{"criterion": "unused", "points": 1}],
    },
    "extra_info": {"index": 0},
    "Rubrics": [
        {"criterion": "RUBRIC-CRITERION-A: names 2 as the smallest prime", "points": 10},
        {"criterion": "RUBRIC-CRITERION-B: answers in under 20 words", "points": 5},
    ],
}


@pytest.fixture(scope="module")
def rubrichub():
    created = not DATA_DIR.exists()
    if created:
        DATA_DIR.mkdir()
        pd.DataFrame([SYNTHETIC_ROW]).to_parquet(DATA_DIR / "rurbichub_v1_Chat.parquet")
        os.environ["RUBRICHUB_ALLOW_PARTIAL_CORPUS"] = "1"
    try:
        import rubrichub as module

        yield module
    finally:
        if created:
            shutil.rmtree(DATA_DIR)


class ScriptedGrader:
    """Stands in for openai.AsyncClient. Each reply quotes the criterion and the reference
    answer, as real grader feedback does, and scores max_points - 1 (or omits the score)."""

    def __init__(self, with_score: bool = True) -> None:
        self.with_score = with_score
        self.chat = self
        self.completions = self

    async def create(self, *, model: str, messages: list[dict[str, str]]):
        prompt = messages[0]["content"]
        criterion = re.search(r"EVALUATION CRITERION:\n(.*)\n\nMAXIMUM POINTS", prompt, re.S).group(1)
        reference = re.search(r"REFERENCE ANSWER[^\n]*\n(.*)\n\nSUBMITTED RESPONSE", prompt, re.S).group(1)
        max_points = int(re.search(r"MAXIMUM POINTS: (\d+)", prompt).group(1))
        content = f"Analysis: Against '{criterion}' and the reference '{reference}', mostly met."
        if self.with_score:
            content += f"\n<answer>{max_points - 1}</answer>"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _env(rubrichub):
    spec = rubrichub.RubricHub.list_tasks("train")[0]
    env = rubrichub.RubricHub(spec, secrets={"openai_api_key": "unused"})
    env.client = ScriptedGrader()
    return env


def _hidden(env) -> list[str]:
    hidden = [r.criterion for r in env.rubrics]
    if env.ground_truth:
        hidden.append(env.ground_truth)
    return hidden


def test_submit_result_shows_scores_not_rubric_or_feedback(rubrichub):
    env = _env(rubrichub)
    out = asyncio.run(env.submit_response(rubrichub.SubmitResponseInput(response="It is 2.")))

    visible = out.blocks[0].text + json.dumps(out.metadata)
    for hidden in _hidden(env):
        assert hidden not in visible, hidden
    assert "Analysis:" not in visible

    possible = sum(r.points for r in env.rubrics)
    earned = sum(r.points - 1 for r in env.rubrics)
    assert out.finished
    assert out.reward == earned / possible
    assert out.metadata["total_points_earned"] == earned
    assert out.metadata["total_points_possible"] == possible
    assert [c["score"] for c in out.metadata["criterion_results"]] == [
        float(r.points - 1) for r in env.rubrics
    ]


def test_unparseable_grader_error_does_not_quote_the_grader(rubrichub):
    env = _env(rubrichub)
    env.client = ScriptedGrader(with_score=False)
    with pytest.raises(ValueError) as exc:
        asyncio.run(env.submit_response(rubrichub.SubmitResponseInput(response="It is 2.")))
    for hidden in _hidden(env):
        assert hidden not in str(exc.value), hidden


def test_session_init_reads_no_parquet(rubrichub, monkeypatch):
    """Sessions read their row from the tables loaded at import, never a shard file."""
    import pyarrow.parquet as pq

    def no_read(*args, **kwargs):
        raise AssertionError("session init read a parquet file")

    monkeypatch.setattr(pd, "read_parquet", no_read)
    monkeypatch.setattr(pq, "read_table", no_read)
    monkeypatch.setattr(pq, "ParquetFile", no_read)
    env = _env(rubrichub)
    assert env.prompt_text and env.prompt_text != "No prompt available"
    assert env.rubrics and all(r.points > 0 for r in env.rubrics)


def test_task_data_matches_the_shard_row(rubrichub):
    """The parsed prompt, rubrics and reference equal what a full read of the row gives."""
    spec = rubrichub.RubricHub.list_tasks("train")[0]
    env = _env(rubrichub)
    row = pd.read_parquet(DATA_DIR / spec["file"]).iloc[spec["local_idx"]].to_dict()
    assert env.prompt_text == env._extract_prompt(row)
    assert env.rubrics == env._parse_rubrics(row)
    assert env.ground_truth == row["reward_model"]["ground_truth"]
    assert spec["data_source"] == row["data_source"] and spec["ability"] == row["ability"]


def test_submit_response_rejects_unknown_fields(rubrichub):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        rubrichub.SubmitResponseInput(response="It is 2.", score=1.0)


def test_grader_client_has_bounded_timeout_and_retries(rubrichub):
    spec = rubrichub.RubricHub.list_tasks("train")[0]
    env = rubrichub.RubricHub(spec, secrets={"openai_api_key": "unused"})
    import httpx

    assert httpx.Timeout(env.client.timeout).read <= 300
    assert env.client.max_retries >= 3


class FlakyGrader(ScriptedGrader):
    """Replies with no score `misses` times per criterion, then scores normally."""

    def __init__(self, misses: int) -> None:
        super().__init__()
        self.misses = misses
        self.calls: dict[str, int] = {}

    async def create(self, *, model: str, messages: list[dict[str, str]]):
        prompt = messages[0]["content"]
        self.calls[prompt] = self.calls.get(prompt, 0) + 1
        if self.calls[prompt] <= self.misses:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
        return await super().create(model=model, messages=messages)


def test_grader_reply_without_score_is_retried(rubrichub):
    env = _env(rubrichub)
    env.client = FlakyGrader(misses=2)
    out = asyncio.run(env.submit_response(rubrichub.SubmitResponseInput(response="It is 2.")))
    possible = sum(r.points for r in env.rubrics)
    assert out.reward == sum(r.points - 1 for r in env.rubrics) / possible
    assert set(env.client.calls.values()) == {3}


def test_grader_that_never_scores_still_raises(rubrichub):
    env = _env(rubrichub)
    env.client = FlakyGrader(misses=rubrichub.GRADER_ATTEMPTS)
    with pytest.raises(ValueError):
        asyncio.run(env.submit_response(rubrichub.SubmitResponseInput(response="It is 2.")))
    assert set(env.client.calls.values()) == {rubrichub.GRADER_ATTEMPTS}
