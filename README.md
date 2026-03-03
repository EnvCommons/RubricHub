# RubricHub

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/RubricHub) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/sojuL/RubricHub_v1)

## Description

RubricHub is an environment for evaluating open-ended generation tasks using rubric-based LLM grading. It contains 364,000 tasks spanning text summarization, code generation, creative writing, question answering, and logical reasoning. Each task includes 2-67 detailed rubric criteria for fine-grained evaluation.

## Capabilities

- Open-ended text generation evaluation
- Multi-criteria rubric-based assessment
- Code generation and summarization tasks
- Creative writing and question answering

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[Apache 2.0](https://opensource.org/licenses/Apache-2.0).

## Tasks

There are two splits in this environment:

- **train**: ~360,000 tasks
- **test**: ~4,000 tasks

Tasks span multiple domains including summarization, code generation, creative writing, Q&A, and logical reasoning.

## Reward Structure

This is a single-turn environment. The agent submits a response via the `submit_response` tool. An LLM grader (gpt-5-mini) evaluates against 2-67 rubric criteria, scoring each from 0 to its maximum points. Reward is normalized: total earned / total possible (0.0 to 1.0).

## Data

Data consists of Parquet files (3.63 GB total) sourced from [HuggingFace sojuL/RubricHub_v1](https://huggingface.co/datasets/sojuL/RubricHub_v1). Each row contains a prompt, rubric criteria with point values, and task metadata. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `submit_response` | Submit your response for rubric-based evaluation. Ends the episode. |

## Time Horizon

Single-turn. The agent reads the prompt and submits one response.

## Environment Difficulty

RubricHub evaluates open-ended generation quality across multiple domains with fine-grained rubric assessment.

## Other Environment Requirements

OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.

## Safety

Agents in RubricHub generate text responses in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@article{li2026rubrichub,
  title={RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation},
  author={Li, Sunzhu and Zhao, Jiale and Wei, Miteto and Ren, Huimin and Zhou, Yang and Yang, Jingwen and Liu, Shunyu and Zhang, Kaike and Chen, Wei},
  journal={arXiv preprint arXiv:2601.08430},
  year={2026}
}
```
