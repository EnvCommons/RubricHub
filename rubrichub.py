from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import openai
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

# Path handling
# Production path
import os
if os.path.exists("/orwd_data"):
    PATH = Path("/orwd_data/")
else:
    PATH = Path(__file__).parent 

# Local development (uncomment for local testing)
# PATH = Path(__file__).parent / "data"


# Expected corpus, keyed by shard filename -> row count. See DATA_UPLOAD.md.
# Validated at import so a partial upload fails loudly instead of silently serving
# a fraction of the dataset (a Chat-only mount once passed as a healthy env).
# NOTE: "rurbichub" is a typo, but it is the name the live Chat shard was uploaded
# under, so the other shards match it; renaming needs a coordinated re-upload.
EXPECTED_SHARDS: Dict[str, int] = {
    "rurbichub_v1_Chat.parquet": 9812,
    "rurbichub_v1_Instruction_Following.parquet": 95173,
    "rurbichub_v1_Medical.parquet": 29681,
    "rurbichub_v1_Science.parquet": 29418,
    "rurbichub_v1_Writing.parquet": 17444,
}

# Escape hatch for local development against a partial corpus.
ALLOW_PARTIAL_CORPUS = os.environ.get("RUBRICHUB_ALLOW_PARTIAL_CORPUS") == "1"


def _validate_corpus(parquet_dir: Path, found: Dict[str, int]) -> None:
    """Raise on a missing shard; warn on row-count drift.

    A missing shard is unambiguously a broken mount. A count mismatch is more
    likely a legitimate upstream revision, so it warns rather than taking the
    environment down over a dataset update.
    """
    missing = {n: w for n, w in EXPECTED_SHARDS.items() if n not in found}
    drift = [
        f"{n}: {found[n]} rows, expected {EXPECTED_SHARDS[n]}"
        for n in EXPECTED_SHARDS
        if n in found and found[n] != EXPECTED_SHARDS[n]
    ]
    drift += [
        f"UNEXPECTED {n} ({found[n]} rows)" for n in sorted(set(found) - set(EXPECTED_SHARDS))
    ]

    if drift:
        logger.warning(
            "RubricHub corpus row counts differ from EXPECTED_SHARDS: %s. "
            "If upstream was revised, update EXPECTED_SHARDS; see DATA_UPLOAD.md.",
            "; ".join(drift),
        )

    if not missing:
        return

    want_total = sum(EXPECTED_SHARDS.values())
    got_total = sum(found.values())
    detail = (
        f"RubricHub corpus at {parquet_dir} is incomplete: missing "
        + "; ".join(f"{n} (expected {w} rows)" for n, w in missing.items())
        + f" | rows {got_total}/{want_total} "
        f"({100.0 * got_total / want_total:.1f}% of the expected corpus). "
        "See DATA_UPLOAD.md. Set RUBRICHUB_ALLOW_PARTIAL_CORPUS=1 to proceed anyway."
    )
    if ALLOW_PARTIAL_CORPUS:
        logger.warning("%s", detail)
        return
    raise RuntimeError(detail)


def build_task_index() -> List[Dict[str, Any]]:
    """Build lightweight index with row IDs and metadata from all .parquet files in data/"""
    # Find all .parquet files in PATH / "data"
    parquet_dir = PATH / "data"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    tasks = []
    global_row_id = 0
    found: Dict[str, int] = {}

    for file in parquet_files:
        df = pd.read_parquet(file, columns=["data_source", "ability"])
        found[file.name] = len(df)

        # Create task entries with global row IDs
        for local_idx in range(len(df)):
            tasks.append({
                "row_id": global_row_id,
                "file": file.name,
                "local_idx": local_idx,
                "data_source": df.iloc[local_idx]["data_source"],
                "ability": df.iloc[local_idx]["ability"]
            })
            global_row_id += 1

    _validate_corpus(parquet_dir, found)
    return tasks


# Load task index at module level (lightweight - only ~5-10 MB)
TASK_INDEX = build_task_index()


# Grader template
GRADER_TEMPLATE = """You are an expert evaluator. Evaluate the following response against a specific criterion.

ORIGINAL PROMPT:
{prompt}

REFERENCE ANSWER (may be empty if task is open-ended):
{ground_truth}

SUBMITTED RESPONSE:
{response}

EVALUATION CRITERION:
{criterion}

MAXIMUM POINTS: {max_points}

Instructions:
1. Analyze how well the response meets the criterion
2. Consider the reference answer if provided, but evaluate based on the criterion
3. Provide a brief explanation (1-2 sentences)
4. Assign a score from 0 to {max_points}

Output format (the score MUST be wrapped in <answer></answer> tags):
Analysis: [Your 1-2 sentence explanation]
<answer>[Integer from 0 to {max_points}]</answer>
"""


# Pydantic models
# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class Rubric(BaseModel):
    criterion: str
    points: int


class TaskSpec(BaseModel):
    row_id: int
    file: str
    local_idx: int
    data_source: str
    ability: str


class SubmitResponseInput(BaseModel):
    response: str = Field(..., description="Your response to the prompt")


class RubricHub(Environment):
    """Single-turn evaluation environment for RubricHub dataset"""

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        # Validate secrets (CRITICAL)
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key required in secrets")

        self.client = openai.AsyncClient(api_key=api_key)
        self.validated = TaskSpec.model_validate(task_spec)

        # Graded submissions this session. The grading output breaks the score
        # down criterion by criterion, so an uncapped tool lets the agent read
        # which rubric items it missed and resubmit against them.
        self.submitted = 0

        # Lazy load specific task (single row only)
        self.task_data = self._load_task_data(self.validated.file, self.validated.local_idx)
        self.rubrics = self._parse_rubrics(self.task_data)
        self.prompt_text = self._extract_prompt(self.task_data)
        self.ground_truth = self.task_data.get("reward_model", {}).get("ground_truth", "")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            raise ValueError(f"Unknown split: {split}")
        return [
            {
                "row_id": task["row_id"],
                "file": task["file"],
                "local_idx": task["local_idx"],
                "data_source": task["data_source"],
                "ability": task["ability"]
            }
            for task in TASK_INDEX
        ]

    async def get_prompt(self) -> List[TextBlock]:
        # Models were responding to the raw task text in plain prose and never
        # invoking submit_response, producing zero tool-call rollouts on every
        # task (env-clinic loadtest "zero tool-call steps" failure). The task
        # prompts in this dataset are open-ended user content with no hint
        # that submission must happen via a tool, so we wrap them with an
        # explicit instruction that the only way to be graded is to call
        # submit_response.
        instructions = (
            "You will be given a task below. Compose your full response, then "
            "submit it for evaluation by calling the `submit_response` tool "
            "with your answer as the `response` argument. Do not reply with "
            "plain text — the grader only sees the tool call.\n\n"
            "=== TASK ===\n"
        )
        return [TextBlock(text=instructions + self.prompt_text)]

    def _load_task_data(self, file: str, local_idx: int) -> Dict[str, Any]:
        """Load specific task by file and local index (MEMORY EFFICIENT)"""
        parquet_dir = PATH / "data"
        file_path = parquet_dir / file

        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        # Read just the specific row we need
        df = pd.read_parquet(file_path)

        if local_idx >= len(df):
            raise IndexError(f"Local index {local_idx} out of range for file {file} (length: {len(df)})")

        return df.iloc[local_idx].to_dict()

    @staticmethod
    def _coerce_rubric_list(rubrics_data: Any) -> list:
        """Normalize a rubric column value to a list; parquet gives ndarray, not list."""
        if rubrics_data is None or isinstance(rubrics_data, str):
            return []
        if isinstance(rubrics_data, np.ndarray):
            return rubrics_data.tolist()
        if isinstance(rubrics_data, (list, tuple)):
            return list(rubrics_data)
        return []

    def _parse_rubrics(self, task_data: Dict) -> List[Rubric]:
        """Extract rubrics from task data"""
        # Try 'Rubrics' field first (flat structure)
        if "Rubrics" in task_data:
            rubrics_data = self._coerce_rubric_list(task_data["Rubrics"])
            if rubrics_data:
                return [Rubric(**dict(r)) for r in rubrics_data]

        # Fallback to reward_model rubrics
        reward_model = task_data.get("reward_model")
        if reward_model is not None and isinstance(reward_model, dict):
            rubrics_data = self._coerce_rubric_list(reward_model.get("rubrics", []))
            if rubrics_data:
                return [
                    Rubric(criterion=r["criterion"], points=r["points"])
                    for r in rubrics_data
                ]

        # Warn rather than substitute a placeholder silently.
        logger.warning(
            "RubricHub: no per-criterion rubric found for task "
            "(row_id=%s, file=%s); falling back to a single generic criterion. "
            "This erases per-criterion grading — check the task's data format.",
            getattr(getattr(self, "validated", None), "row_id", "?"),
            getattr(getattr(self, "validated", None), "file", "?"),
        )
        return [Rubric(criterion="Overall quality and correctness", points=100)]

    def _extract_prompt(self, task_data: Dict) -> str:
        """Extract prompt text from task data"""
        prompt_list = task_data.get("prompt")
        if prompt_list is None:
            return "No prompt available"

        # Handle list, numpy array, or any iterable of dicts
        if hasattr(prompt_list, '__iter__') and not isinstance(prompt_list, str):
            parts = [
                p["content"] for p in prompt_list
                if isinstance(p, dict) and "content" in p
            ]
            if parts:
                return "\n\n".join(parts)
            return "No prompt available"

        return str(prompt_list)

    async def _grade_single_criterion(
        self, response: str, criterion: str, max_points: int
    ) -> Dict[str, Any]:
        """Grade response against a single rubric criterion"""
        grader_prompt = GRADER_TEMPLATE.format(
            prompt=self.prompt_text,
            ground_truth=self.ground_truth if self.ground_truth else "(No reference answer provided)",
            response=response,
            criterion=criterion,
            max_points=max_points
        )

        # Use gpt-5-mini (NO temperature parameter per guidelines)
        res = await self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}]
        )

        grading_response = res.choices[0].message.content or ""
        score = self._parse_score(grading_response, max_points)

        return {
            "criterion": criterion,
            "max_points": max_points,
            "score": score,
            "grading_response": grading_response
        }

    async def _grade_all_criteria(self, response: str) -> List[Dict[str, Any]]:
        """Grade response against all rubrics (PARALLEL for efficiency)"""
        grading_tasks = [
            self._grade_single_criterion(response, rubric.criterion, rubric.points)
            for rubric in self.rubrics
        ]

        return await asyncio.gather(*grading_tasks)

    def _parse_score(self, grading_response: str, max_points: int) -> float:
        """Extract the score from the judge's <answer></answer> tag."""
        # Last tag wins, so a tag quoted inside the analysis can't shadow the real score.
        matches = re.findall(
            r"<answer>\s*(\d+(?:\.\d+)?)\s*</answer>", grading_response, re.IGNORECASE
        )
        if not matches:
            # Raise rather than guess: a fabricated score is indistinguishable from a real one.
            raise ValueError(
                "Grader response contained no parseable <answer></answer> score "
                f"(max_points={max_points}); response was: {grading_response!r}"
            )
        return max(0.0, min(float(max_points), float(matches[-1])))

    def _format_grading_output(
        self,
        criterion_results: List[Dict],
        total_earned: float,
        total_possible: int,
        reward: float
    ) -> str:
        """Format grading results for display"""
        lines = ["# Rubric Evaluation Results\n"]

        for i, result in enumerate(criterion_results, 1):
            lines.append(f"## Criterion {i}: {result['criterion']}")
            lines.append(f"**Score:** {result['score']}/{result['max_points']}")
            lines.append(f"**Feedback:** {result['grading_response']}\n")

        lines.append("---")
        lines.append(f"## Final Score: {total_earned:.1f}/{total_possible}")
        lines.append(f"## Normalized Reward: {reward:.3f}")

        return "\n".join(lines)

    @tool
    async def submit_response(self, params: SubmitResponseInput) -> ToolOutput:
        """
        Submit your response to be evaluated against all rubric criteria.
        Returns detailed feedback for each criterion plus total score.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="A response has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        # Grade all criteria in parallel
        criterion_results = await self._grade_all_criteria(params.response)

        # Calculate scores
        total_possible = sum(r.points for r in self.rubrics)
        total_earned = sum(r["score"] for r in criterion_results)
        reward = total_earned / total_possible if total_possible > 0 else 0.0

        # Format display output
        display_text = self._format_grading_output(
            criterion_results, total_earned, total_possible, reward
        )

        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "row_id": self.validated.row_id,
                "total_points_earned": total_earned,
                "total_points_possible": total_possible,
                "reward": reward,
                "criterion_results": criterion_results,
                "data_source": self.validated.data_source,
                "ability": self.validated.ability
            },
            reward=reward,
            finished=True
        )
