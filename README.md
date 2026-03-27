# Video Model Evaluation — PoC

A Rust-based evaluation pipeline that benchmarks **Vision-Language Models (VLMs)** and **person detection models** for use in a live, latency-sensitive video pipeline.

Built as part of the Async Coding Challenge described in [docs/requirement.md](docs/requirement.md).

---

## Overview

The pipeline:
1. Extracts key frames from short video clips (via `ffmpeg`).
2. Runs each frame through multiple VLM and person-detection models in parallel.
3. Scores VLM captions using an LLM judge (quality 1–5).
4. Evaluates person detection against ground-truth counts (MAE, precision, recall, F1).
5. Measures end-to-end latency, cost, and reliability per model.
6. Produces three recommendations: best quality, best real-time, best overall.
7. Writes results to JSON, CSV, and Markdown.

---

## Quick Start

### Prerequisites

- **Rust** (edition 2021, stable) — [install](https://rustup.rs/)
- **ffmpeg** — must be on `PATH` for video frame extraction
- **OpenCV** (for the `opencv` crate) — see [docs/architecture.md](docs/architecture.md)
- **OpenRouter API key** — [openrouter.ai](https://openrouter.ai)

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd ChallengeVideoModelEvaluation

# Copy the environment template and fill in your API key
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your-key>

# (Optional) add your own .mp4 clips to the data folder
cp your_clips/*.mp4 video_model_eval/data/
```

### Run the full evaluation

```bash
./run.sh -- --evaluate
```

Results are written to `video_model_eval/results/`:
- `output.json` — full detail per clip and model
- `results.csv` — flat rows: `run_id, clip_name, model_type, model_name, metric_name, metric_value`
- `summary.md` — Markdown tables and recommendations

To write results to a custom directory:
```bash
./run.sh -- --evaluate --out /path/to/output/
```

### Run tests

```bash
./test.sh
```

---

## CLI Reference

All commands are run from the repo root via `./run.sh`:

| Command | Description |
|---|---|
| `./run.sh -- --list-clips` | List video clips found in `data/` |
| `./run.sh -- --vlm-test` | Single VLM inference on `data/sample.jpg` |
| `./run.sh -- --vlm-all-models` | Run all VLM models + judge scoring on `data/sample.jpg` |
| `./run.sh -- --detect-test` | Person detection on the first `.mp4` in `data/` |
| `./run.sh -- --detect-all-models` | Run all detection models on `data/sample.jpg` |
| `./run.sh -- --detect-ground-truth` | Evaluate detection models against ground truth; print MAE per model |
| `./run.sh -- --evaluate [--out <dir>]` | Full evaluation pipeline |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **yes** | — | API key for OpenRouter |
| `OPENROUTER_URL` | no | `https://openrouter.ai/api/v1/chat/completions` | Override the OpenRouter endpoint |
| `VLM_MODEL` | no | `reka/reka-edge` | Model used by `--vlm-test` |
| `VLM_JUDGE_MODEL` | no | `google/gemini-2.0-flash-lite-001` | Judge model for quality scoring |
| `PERSON_DETECTION_MODEL` | no | `reka/reka-edge` | Default model for `--detect-test` |
| `DETECTION_THRESHOLD` | no | `0.5` | Minimum confidence threshold for person detection |
| `DATA_PATH` | no | `data/` | Directory with video clips |
| `RESULT_PATH` | no | `results/` | Output directory for evaluation artifacts |
| `GROUND_TRUTH_PATH` | no | `tests/data/ground_truth.json` | Ground truth annotations for detection evaluation |
| `MAX_CLIPS` | no | `5` | Maximum number of clips to evaluate |
| `MIN_QUALITY_THRESHOLD` | no | `3.0` | Minimum judge score (1–5) for real-time candidate eligibility |
| `MIN_RELIABILITY_THRESHOLD` | no | `0.8` | Minimum reliability rate for real-time candidate eligibility |
| `WEIGHT_QUALITY` | no | `0.35` | Weight for quality dimension in composite score |
| `WEIGHT_LATENCY` | no | `0.25` | Weight for latency dimension in composite score |
| `WEIGHT_COST` | no | `0.20` | Weight for cost dimension in composite score |
| `WEIGHT_RELIABILITY` | no | `0.15` | Weight for reliability dimension in composite score |
| `WEIGHT_PRACTICALITY` | no | `0.05` | Weight for practicality dimension in composite score |
| `RUST_LOG` | no | `info` | Log level: `trace`, `debug`, `info`, `warn`, `error` |

You can place these in a `.env` file at `video_model_eval/` — loaded automatically via `dotenv`.

---

## Models Evaluated

### VLM (Vision-Language)

| Model | Provider | Est. Cost/call |
|---|---|---|
| `reka/reka-edge` | Reka | $0.00100 |
| `google/gemini-2.0-flash-lite-001` | Google | $0.00020 |
| `meta-llama/llama-3.2-11b-vision-instruct` | Meta | $0.00005 |

### Person Detection

| Model | Provider | Est. Cost/call |
|---|---|---|
| `reka/reka-edge` | Reka | $0.00100 |
| `google/gemini-2.0-flash-lite-001` | Google | $0.00020 |

All models are accessed through [OpenRouter](https://openrouter.ai).

---

## Project Structure

```
ChallengeVideoModelEvaluation/
├── run.sh                        # Run wrapper (cargo run from repo root)
├── test.sh                       # Test wrapper (cargo test from repo root)
├── docs/
│   ├── requirement.md            # Original challenge spec
│   ├── coco_subset.md            # Ground truth dataset description
│   ├── architecture.md           # System design and data flows
│   ├── business-rules.md         # Evaluation rules and thresholds
│   └── reports/
│       └── summary.md            # Results write-up (generated + manual)
└── video_model_eval/
    ├── Cargo.toml
    ├── data/                     # Input video clips and test images
    ├── results/                  # Evaluation output (JSON, CSV, Markdown)
    ├── src/
    │   ├── main.rs               # CLI entrypoint
    │   ├── lib.rs                # Library re-exports
    │   ├── vlm.rs                # VLM inference + judge scoring
    │   ├── person_detection.rs   # Person detection via VLM APIs
    │   ├── evaluator.rs          # Metrics, scoring, recommendations
    │   └── models.rs             # Shared data structures
    └── tests/
        ├── integration_test.rs
        └── data/
            ├── ground_truth.json # Ground truth person counts
            ├── persons_frame.jpg # Sample frame with persons
            └── no_persons_frame.jpg
```

---

## Results Summary

See [docs/reports/summary.md](docs/reports/summary.md) for the full write-up with methodology, results tables, recommendations, and tradeoffs.

See [video_model_eval/results/summary.md](video_model_eval/results/summary.md) for the auto-generated metrics table from the last evaluation run.

---

## Logging

The pipeline uses structured logging via `tracing`. Set the `RUST_LOG` environment variable to control verbosity:

```bash
RUST_LOG=debug ./run.sh -- --evaluate
```

Each evaluation run generates a unique `run_id` (e.g., `run-1774555503`) included in all log entries and output artifacts for correlation.
