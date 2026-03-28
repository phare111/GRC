# GRC: Risk-Controlled Generative OCR for Vision-Language Models

## Overview

![teaser](teaser.png)

*Overview of the proposed Geometric Risk Controller (GRC).*

This repository provides a minimal, reproducible implementation of **Geometric Risk Controller (GRC)** for generative OCR using frozen vision-language models (VLMs).

Since the VLM is treated as a frozen black-box generator, we use Ollama for convenient local deployment and inference. The controller enforces a system-level accept/abstain contract based on cross-view consistency and stability.

Related paper:
[From Plausibility to Verifiability: Risk-Controlled Generative OCR for Vision-Language Models](https://arxiv.org/abs/2603.19790)

## Quick Start 

```bash
python run.py \
  --input_dir example \
  --gt_csv example.csv \
  --model llava-phi3:3.8b \
  --output_dir ./out \
  --limit 300 \
  --sleep 0.0 \
  --baseline_tokens 64 \
  --system_tokens 64 \
  --meltdown_max_len 256 \
  --meltdown_t 2.0 \
  --k_views 5 \
  --stability_tau 0.60 \
  --vote_tau 0.40 \
  --length_alpha 2 \
  --length_safe_buffer 2
```

## Repository Structure

```text
grc/
├── run.py                  # entry
├── requirements.txt
├── example.csv
├── example/
│
├── grc/
│   ├── inference_tools/
│   │   ├── engine.py       # vlm interface
│   │   ├── image_views.py  # multi-view
│   │   ├── verifier.py     # accept/abstain
│   │   ├── stability.py    # stability
│   │   ├── parsing.py      # parsing
│   │   ├── schemas.py      # schemas
│   │   └── meltdown.py     # meltdown
│   │
│   ├── length_tools/
│   │   ├── estimator.py    # length est
│   │   └── geo_bound.py    # geom bound
│   │
│   └── ocr_tools/
│       ├── eval_tools/
│       │   └── evaluator.py    # evaluation
│       └── text_tools/
│           └── text_metrics.py # cer
```

## Key Parameters

* `k_views`: number of geometric probes
* `vote_tau`: consensus threshold
* `stability_tau`: stability threshold
* `meltdown_t`: CER threshold for extreme-error reporting
* `meltdown_max_len`: maximum length for meltdown detection
* `length_alpha`: scaling factor for geometric length bound
* `length_safe_buffer`: safety margin for length constraint

## Notes

* This repository provides a **minimal reproducible implementation** of the proposed system.
* The implementation focuses on a simplified instantiation of the controller.
* The system treats the VLM as a **black-box generator** and only uses returned strings.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt