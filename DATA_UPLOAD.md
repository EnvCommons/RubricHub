# Data Upload Requirements for RubricHub

## Overview

This environment serves the **`RuRL/`** half of the HuggingFace dataset
`sojuL/RubricHub_v1` — 5 domain parquet files, **181,528 samples** total.

The repo contains two directories with different purposes. Only one of them belongs
in this environment:

| directory | contents | rows | use here |
|---|---|---|---|
| `RuRL/` | 5 domain files: prompt + rubrics, no answers | 181,528 | **yes — this is the env corpus** |
| `sft_RuFT/` | 2 files: prompt + model answer + rubric scores + judge verdicts | 182,732 | **no — see below** |

> **Do not upload `sft_RuFT/`.** Those rows are supervised-fine-tuning data: the same
> prompts, already answered by gpt-5.1 / Gemini 3 Pro, already graded, with
> `rubric_score` and per-criterion `rubric_judge_details` stored in the row. Serving
> them as tasks would hand the agent the answer and the judge's verdict. They are also
> a different schema (`source`, `query`, `answer`, `sample_id`, `rubrics`,
> `rubric_score`, `rubric_judge_details`) with no `ability` or `data_source` column, so
> `build_task_index()` raises on them. And they add no new prompts — a 60-query probe
> found 60/60 already present in `RuRL/`. The two files are one dataset:
> `..._6samples_156k...` holds ~6 sampled responses per prompt and
> `..._best_of_6samples_26k...` the best one per prompt (156,538 / 26,194 ≈ 5.98).

> **The dataset has ONE split, not five.** `sojuL/RubricHub_v1` publishes a single
> config (`default`) with a single split (`train`); the five "domains" are values of
> the `ability` column. Any procedure calling `load_dataset(..., split="chat")` fails —
> no such split exists. Note also that HuggingFace's auto-converted parquet view
> (`refs/convert/parquet`) flattens **both** directories into that one split, so
> downloading from there silently mixes the SFT data in. Fetch `RuRL/` explicitly.

## Re-provisioning

The data is already uploaded; this is only needed for a new environment instance, a
fork, or disaster recovery. Fetch the five `RuRL/` files and upload them **verbatim**:

```bash
huggingface-cli download sojuL/RubricHub_v1 --repo-type dataset --include 'RuRL/*.parquet'
```

Then upload those five files to the environment's namespace at
[openreward.ai](https://openreward.ai), into a `data/` folder.

Do **not** re-encode them. The files are already exactly the five domain shards, one
domain each, correctly named — no partitioning or conversion is required. Uploading
them byte-for-byte keeps them checksum-comparable against upstream, which is how a
mount can be verified later: a sha256 match proves the mount is the published data.
(Re-encoding also fragments the row groups badly — writing in small batches turns a
single row group into hundreds and inflates the file.)

## Required directory structure

The five files must sit in a top-level **`data/`** folder:

```
data/
├── rurbichub_v1_Chat.parquet                  (  9,812 samples)
├── rurbichub_v1_Instruction_Following.parquet ( 95,173 samples)
├── rurbichub_v1_Medical.parquet               ( 29,681 samples)
├── rurbichub_v1_Science.parquet               ( 29,418 samples)
└── rurbichub_v1_Writing.parquet               ( 17,444 samples)
```

Two things that have caused real problems here:

- **`data/` is the top level — there is no `rubrichub/` above it.** `rubrichub.py`
  globs `data/*.parquet` relative to the mount root, so files nested under an extra
  `rubrichub/` directory are never read, and nothing reports an error.
- **`rurbichub` is a typo — keep it.** It is the upstream spelling: the files in the
  dataset's own `RuRL/` directory are named `rurbichub_v1_*.parquet`. Our mount mirrors
  the source repo name-for-name. Do not "correct" it.

## Corpus validation

`rubrichub.py` declares `EXPECTED_SHARDS` (filename → row count) and validates the
mount at import. The two failure modes are treated differently on purpose:

- **A missing shard raises** — unambiguously a broken mount:

  ```
  RubricHub corpus at /orwd_data/data is incomplete: missing
  rurbichub_v1_Medical.parquet (expected 29681 rows) | rows 9812/181528
  (5.4% of the expected corpus). See DATA_UPLOAD.md.
  ```

- **A row-count mismatch, or an unexpected file, only warns.** Those are more likely a
  legitimate upstream revision than a broken mount, and hard-failing the environment
  over a dataset update would be a self-inflicted outage. If upstream really was
  revised, update `EXPECTED_SHARDS` to match — after confirming what changed.

Set `RUBRICHUB_ALLOW_PARTIAL_CORPUS=1` to downgrade the missing-shard error to a
warning when developing against a subset.

This guard exists because a partial upload used to be invisible. Between 2026-02-01
and 2026-08-07 only `rurbichub_v1_Chat.parquet` was mounted — 9,812 of 181,528 rows,
**5.4% of the corpus, all from the single easiest domain** — and the environment
reported healthy throughout, because the loader only raised when the directory was
missing or held zero parquet files. Three clinic trace runs graded green against it.
That lone shard was byte-identical to HuggingFace's auto-converted `0000.parquet`, so
no partitioning had ever been performed; the remaining four were uploaded 2026-08-07.

## File descriptions

- **Total**: 181,528 samples across 5 domain files
- **Content**: prompts, rubric criteria, ground-truth answers
- **Domains**: Chat, Instruction_Following, Medical, Science, Writing. Note `chat` is
  lower-case in the `ability` column while the other four are capitalised — exact-match
  filters need to account for it.
- **`ability` vs `data_source`**: both columns exist and hold **identical** values in
  every row. Neither is used for grading or prompt construction; they are passthrough
  metadata, useful for domain-stratified evaluation.
- **Rubrics**: 2–67 criteria per sample with point weights, in the `Rubrics` column
  (list of `{criterion, points}`) and mirrored under `reward_model.rubrics`.
