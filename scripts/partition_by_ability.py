#!/usr/bin/env python3
"""Build the five RubricHub domain shards from the HF dataset, partitioned by `ability`.

Two traps this script exists to avoid:

1. `sojuL/RubricHub_v1` has ONE split (`train`), not five. The domains are values of
   the `ability` column, so `load_dataset(..., split="chat")` fails.
2. The repo has two directories. `RuRL/` is the RL corpus this environment serves;
   `sft_RuFT/` is supervised-fine-tuning data (prompt + model answer + rubric score +
   judge verdicts) that must never be mounted. HuggingFace's auto-converted parquet
   view flattens both into one split, so fetching from `refs/convert/parquet` silently
   mixes them. We fetch `RuRL/` explicitly and additionally reject any file lacking the
   `ability` column, so an sft file can never slip through.

Partitioning by `ability` is idempotent against the current upstream layout (each
`RuRL/` file already holds one domain) and is kept deliberately: it derives the domains
from the data rather than trusting filenames.

Usage:  python3 partition_by_ability.py [OUTPUT_DIR]
"""
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "sojuL/RubricHub_v1"
BATCH_ROWS = 256  # small: rows carry large nested rubric structs
NAME = "rurbichub_v1_{domain}.parquet"  # upstream spelling, typo included — do not "fix"

EXPECTED = {
    "chat": 9812,
    "Instruction_Following": 95173,
    "Medical": 29681,
    "Science": 29418,
    "Writing": 17444,
}


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).parent / "shards"
    out.mkdir(parents=True, exist_ok=True)

    raw = Path(snapshot_download(
        REPO, repo_type="dataset", allow_patterns="RuRL/*.parquet",
    )) / "RuRL"
    files = sorted(raw.glob("*.parquet"))
    if not files:
        print(f"no parquet files under {raw}", file=sys.stderr)
        return 1

    writers: dict[str, pq.ParquetWriter] = {}
    counts: Counter = Counter()
    schema = None
    total = 0

    try:
        for f in files:
            pf = pq.ParquetFile(f)
            if "ability" not in pf.schema_arrow.names:
                # An sft_RuFT file, or an upstream schema change. Never mount these.
                print(f"  SKIP {f.name}: no 'ability' column", file=sys.stderr)
                continue
            schema = schema or pf.schema_arrow
            print(f"  reading {f.name} ({pf.metadata.num_rows} rows)", flush=True)
            for batch in pf.iter_batches(batch_size=BATCH_ROWS):
                tbl = pa.Table.from_batches([batch])
                total += tbl.num_rows
                ability = tbl.column("ability")
                for dom in pc.unique(ability.drop_null()).to_pylist():
                    sub = tbl.filter(pc.equal(ability, pa.scalar(dom)))
                    if sub.num_rows == 0:
                        continue
                    if dom not in writers:
                        path = out / NAME.format(domain=dom)
                        writers[dom] = pq.ParquetWriter(path, schema, compression="snappy")
                        print(f"    + {path.name}", flush=True)
                    writers[dom].write_table(sub)
                    counts[dom] += sub.num_rows
    finally:
        for w in writers.values():
            w.close()

    print("\n=== result ===")
    ok = True
    for dom in sorted(set(counts) | set(EXPECTED)):
        got, want = counts.get(dom, 0), EXPECTED.get(dom)
        flag = "OK" if got == want else f"MISMATCH (expected {want})"
        ok &= got == want
        print(f"  {NAME.format(domain=dom):48s} {got:>7} rows  {flag}")
    print(f"  {'TOTAL':48s} {sum(counts.values()):>7} rows (expected {sum(EXPECTED.values())})")
    if not ok:
        print("\nCounts disagree with EXPECTED — upstream changed. Do not just edit the\n"
              "expectations; confirm what changed, then update EXPECTED here and\n"
              "EXPECTED_SHARDS in rubrichub.py together.", file=sys.stderr)
        return 1
    print(f"\nWrote {len(counts)} shards to {out}. Upload them to /orwd_data/data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
