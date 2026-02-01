# Data Upload Requirements for RubricHub

## Overview
This environment requires the RubricHub_v1 dataset (1 GB, 181K samples) from HuggingFace, split across 5 domain-specific parquet files.

## Download Instructions

1. Install HuggingFace datasets library:
   ```bash
   pip install datasets
   ```

2. Download the dataset by domain:
   ```python
   from datasets import load_dataset

   # The dataset comes split by domain
   domains = ["Chat", "Instruction_Following", "Medical", "Science", "Writing"]

   for domain in domains:
       dataset = load_dataset("sojuL/RubricHub_v1", split=domain.lower())
       dataset.to_parquet(f"rurbichub_v1_{domain}.parquet")
   ```

3. Upload all files to OpenReward namespace at https://openreward.ai

## Required Directory Structure
```
/orwd_data/
└── rubrichub/
    └── data/
        ├── rurbichub_v1_Chat.parquet (9,812 samples)
        ├── rurbichub_v1_Instruction_Following.parquet (95,173 samples)
        ├── rurbichub_v1_Medical.parquet (29,681 samples)
        ├── rurbichub_v1_Science.parquet (29,418 samples)
        └── rurbichub_v1_Writing.parquet (17,444 samples)
```

## File Descriptions
- **Total**: 181,528 samples across 5 domain-specific files
- **Content**: Prompts, rubric criteria, ground truth answers
- **Domains**: Chat, instruction following, medical QA, science, writing
- **Rubrics**: 2-67 criteria per sample with point weights
