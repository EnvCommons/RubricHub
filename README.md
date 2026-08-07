# RubricHub

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/RubricHub) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/sojuL/RubricHub_v1)

## Description

RubricHub is an environment for evaluating open-ended generation tasks using rubric-based LLM grading. It contains 181,528 tasks across five domains — chat, instruction following, medical, science and writing — and each task carries 2-67 detailed rubric criteria for fine-grained evaluation. Prompts are multilingual (English and Chinese predominate), and rubric criteria are written in the language of their prompt.

## Capabilities

- Open-ended text generation evaluation
- Multi-criteria rubric-based assessment
- Multilingual prompts and rubric criteria
- Domain-stratified evaluation via the `ability` field

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[Apache 2.0](https://opensource.org/licenses/Apache-2.0).

## Tasks

There is a single split, **train**, with **181,528 tasks**. `list_splits()` returns only
`["train"]`; any other split name raises.

| domain (`ability`) | tasks |
|---|---:|
| Instruction_Following | 95,173 |
| Medical | 29,681 |
| Science | 29,418 |
| Writing | 17,444 |
| chat | 9,812 |

`ability` and `data_source` carry the same value on every row. Neither affects grading —
they are metadata, useful for stratifying results by domain. Note `chat` is lower-case
while the other four are capitalised.

## Reward Structure

This is a single-turn environment. The agent submits a response via the `submit_response` tool. An LLM grader (gpt-5-mini) evaluates against 2-67 rubric criteria, scoring each from 0 to its maximum points. Reward is normalized: total earned / total possible (0.0 to 1.0).

The grader returns each score inside `<answer></answer>` tags. If a grader response cannot be parsed, the episode raises rather than substituting a score — a couldn't-grade condition is never scored as a real result.

## Data

Five Parquet files (~995 MiB total), one per domain, sourced from the **`RuRL/`** directory
of [HuggingFace sojuL/RubricHub_v1](https://huggingface.co/datasets/sojuL/RubricHub_v1) and
stored on the OpenReward platform. Each row contains a prompt, rubric criteria with point
values, and task metadata.

The dataset repo also holds a `sft_RuFT/` directory (182,732 rows) which is **not** part of
this environment: it is supervised-fine-tuning data whose rows already carry a model answer,
a `rubric_score` and per-criterion judge verdicts. It uses a different schema and adds no new
prompts. See `DATA_UPLOAD.md` — the environment validates its corpus at import and raises if a
domain shard is missing.

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
