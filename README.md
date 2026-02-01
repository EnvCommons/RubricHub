# RubricHub OpenReward Environment

An OpenReward environment for evaluating open-ended generation tasks using rubric-based LLM grading.

## Overview

RubricHub provides detailed evaluation across 364K tasks spanning:
- Text summarization
- Code generation
- Creative writing
- Question answering
- Logical reasoning

Each task includes 2-67 rubric criteria evaluated by LLM grading (gpt-5-mini).

## Installation

```bash
pip install -r requirements.txt
```

## Data Requirements

Requires RubricHub_v1 dataset (3.63 GB). See [DATA_UPLOAD.md](DATA_UPLOAD.md) for instructions.

## Local Testing

```bash
# Start server
python server.py

# Run test agent
export OPENAI_API_KEY="your-key"
python test_agent.py
```

## Docker

```bash
docker build -t rubrichub:latest .
docker run -p 8080:8080 -v /path/to/data:/orwd_data:ro rubrichub:latest
```

## Tool

**submit_response(response: str)**: Submit response for rubric-based evaluation

Returns:
- Detailed per-criterion feedback
- Scores for each rubric (0 to max points)
- Total score and normalized reward [0, 1]

## Example

```python
from openreward import AsyncOpenReward
import asyncio

async def run_example():
    client = AsyncOpenReward()
    env = client.environments.get(name="EnvCommons/rubrichub")

    tasks = await env.list_tasks(split="train")
    task = tasks[0]

    async with env.session(task=task, secrets={"openai_api_key": "..."}) as session:
        prompt = await session.get_prompt()
        print(f"Prompt: {prompt[0].text}")

        result = await session.call_tool("submit_response", {
            "response": "Your answer here"
        })
        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.blocks[0].text}")

asyncio.run(run_example())
```

## Dataset

Based on RubricHub_v1 from HuggingFace:
- **Paper**: [RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset](https://arxiv.org/abs/2601.08430)
- **Dataset**: [sojuL/RubricHub_v1](https://huggingface.co/datasets/sojuL/RubricHub_v1)

## License

Apache 2.0
