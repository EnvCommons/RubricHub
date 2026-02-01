from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List

import openai
import pandas as pd
from pydantic import BaseModel, Field

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

    for file in parquet_files:
        df = pd.read_parquet(file, columns=["data_source", "ability"])

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

Output format:
Analysis: [Your 1-2 sentence explanation]
Score: [Integer from 0 to {max_points}]
"""


# Pydantic models
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
        return [TextBlock(text=self.prompt_text)]

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

    def _parse_rubrics(self, task_data: Dict) -> List[Rubric]:
        """Extract rubrics from task data"""
        # Try 'Rubrics' field first (flat structure)
        if "Rubrics" in task_data:
            rubrics_data = task_data["Rubrics"]
            if rubrics_data is not None and isinstance(rubrics_data, list) and len(rubrics_data) > 0:
                return [Rubric(**r) for r in rubrics_data]

        # Fallback to reward_model rubrics
        reward_model = task_data.get("reward_model")
        if reward_model is not None and isinstance(reward_model, dict):
            rubrics_data = reward_model.get("rubrics", [])
            if rubrics_data is not None and isinstance(rubrics_data, list) and len(rubrics_data) > 0:
                return [
                    Rubric(criterion=r["criterion"], points=r["points"])
                    for r in rubrics_data
                ]

        # If no rubrics found, create a default one
        return [Rubric(criterion="Overall quality and correctness", points=100)]

    def _extract_prompt(self, task_data: Dict) -> str:
        """Extract prompt text from task data"""
        prompt_list = task_data.get("prompt")
        if prompt_list is None or (isinstance(prompt_list, list) and len(prompt_list) == 0):
            return "No prompt available"

        # Concatenate all prompt turns
        if isinstance(prompt_list, list):
            return "\n\n".join(
                p["content"] for p in prompt_list
                if isinstance(p, dict) and "content" in p
            )

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
        """Extract score with robust fallback parsing"""
        # Look for "Score: X" pattern
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", grading_response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0.0, min(float(max_points), score))

        # Fallback: find any number
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", grading_response)
        if numbers:
            return max(0.0, min(float(max_points), float(numbers[-1])))

        return 0.0  # Default if parsing fails

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
