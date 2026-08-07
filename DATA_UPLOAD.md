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

## Download and partition

```python
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

# RuRL/ only — never sft_RuFT/, and not refs/convert/parquet (it mixes the two).
raw = Path(snapshot_download(
    "sojuL/RubricHub_v1", repo_type="dataset", allow_patterns="RuRL/*.parquet",
)) / "RuRL"

out = Path("data")
out.mkdir(exist_ok=True)
writers, counts, schema = {}, Counter(), None

for f in sorted(raw.glob("*.parquet")):
    pf = pq.ParquetFile(f)
    schema = schema or pf.schema_arrow
    for batch in pf.iter_batches(batch_size=256):     # streamed: rows carry big nested rubrics
        tbl = pa.Table.from_batches([batch])
        ability = tbl.column("ability")
        for dom in pc.unique(ability.drop_null()).to_pylist():
            sub = tbl.filter(pc.equal(ability, pa.scalar(dom)))
            if dom not in writers:
                writers[dom] = pq.ParquetWriter(
                    out / f"rurbichub_v1_{dom}.parquet", schema, compression="snappy"
                )
            writers[dom].write_table(sub)
            counts[dom] += sub.num_rows

for w in writers.values():
    w.close()
print(counts)   # -> chat 9812, Instruction_Following 95173, Medical 29681, Science 29418, Writing 17444
```

A ready-to-run version lives at `scripts/partition_by_ability.py`.

Partitioning by `ability` is currently idempotent — each `RuRL/` file already holds
exactly one domain, verified homogeneous. It is kept deliberately anyway: it derives
the domains from the data rather than trusting filenames, so it still produces the
correct five shards (and the correct row counts) if upstream ever reorganises the
files. Treat a count that disagrees with the table below as a signal that the upstream
dataset changed, not as a reason to edit the expectations.

## Required directory structure

Upload the five files so they land at **`/orwd_data/data/`**:

```
/orwd_data/
└── data/
    ├── rurbichub_v1_Chat.parquet                  (  9,812 samples)
    ├── rurbichub_v1_Instruction_Following.parquet ( 95,173 samples)
    ├── rurbichub_v1_Medical.parquet               ( 29,681 samples)
    ├── rurbichub_v1_Science.parquet               ( 29,418 samples)
    └── rurbichub_v1_Writing.parquet               ( 17,444 samples)
```

Two things that have caused real problems here:

- **The path is `/orwd_data/data/`, with no `rubrichub/` level.** `rubrichub.py` globs
  `PATH / "data" / "*.parquet"` where `PATH = /orwd_data`. Files uploaded to
  `/orwd_data/rubrichub/data/` are never read, and nothing reports an error.
- **`rurbichub` is a typo — keep it.** It is the upstream spelling: the files in the
  dataset's own `RuRL/` directory are named `rurbichub_v1_*.parquet`. Our mount mirrors
  the source repo name-for-name. Do not "correct" it.

## Corpus validation

`rubrichub.py` declares `EXPECTED_SHARDS` (filename → row count) and validates the
mount at import. A missing shard or a row-count mismatch raises with the shortfall:

```
RubricHub corpus at /orwd_data/data is incomplete or unexpected:
MISSING rurbichub_v1_Medical.parquet (expected 29681 rows) | rows 9812/181528
(5.4% of the expected corpus). See DATA_UPLOAD.md.
```

Set `RUBRICHUB_ALLOW_PARTIAL_CORPUS=1` to downgrade this to a warning when developing
against a subset.

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
