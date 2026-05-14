# NLPCC 2025 Task 1 - Chinese LLM-generated Text Detection

This repository contains my solution for NLPCC 2025 Shared Task 1: Chinese LLM-generated text detection.

I ranked 10th in Task 1. The method combines hierarchical feature fusion with heterogeneous ensemble modeling for robust detection across normal, attack, and varying-length scenarios.

## Repository Contents

- `code/`: training, inference, feature extraction, augmentation, and ensemble scripts
- `code/requirements.txt`: Python dependencies
- `figure/`: analysis and experiment figures
- `层次化特征融合与异构集成的中文 LLM生成文本检测方法.pdf`: project report

Large datasets, trained checkpoints, generated result dumps, logs, archives, and videos are intentionally excluded from GitHub. Some trained-model artifacts were larger than GitHub's normal file limits.

## Setup

```bash
pip install -r code/requirements.txt
```

## Main Scripts

```bash
python code/run.py
python code/ensemble_detector.py
python code/enhanced_detector.py
```

## Notes

The original competition page is available at: http://tcci.ccf.org.cn/conference/2025/cfpt.php

The official task repository and leaderboard information are available from the NLPCC 2025 Task 1 organizers.

## Citation / Acknowledgement

This work was developed for NLPCC 2025 Shared Task 1. Please refer to the official task page and report PDF for task details and method description.
