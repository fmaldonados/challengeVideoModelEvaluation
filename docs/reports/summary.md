# VLM & Person Detection Evaluation — Write-up

> **PoC de evaluación de VLM y detección de personas**  
> Run ID: run-1774555503 · Frames processed: 35 · Date: 2026-03-26

---

## 1. What Was Tested

### Models evaluated

| Type | Model | Provider |
|------|-------|----------|
| VLM | `google/gemini-2.0-flash-lite-001` | OpenRouter |
| VLM | `reka/reka-edge` | OpenRouter |
| VLM | `meta-llama/llama-3.2-11b-vision-instruct` | OpenRouter |
| Detection | `google/gemini-2.0-flash-lite-001` | OpenRouter |
| Detection | `reka/reka-edge` | OpenRouter |

### Dataset

- 5 MP4 video clips stored in `video_model_eval/data/` (personal recordings, mixed indoor/outdoor scenes).
- 2 annotated COCO-subset image frames in `tests/data/` with ground-truth person counts (`tests/data/ground_truth.json`).
- One frame was extracted per video clip using OpenCV (`person_detection::extract_frame`).
- Total frames processed: **35** (7 per model × 5 frames).

---

## 2. Methodology

### VLM quality (no ground truth)

VLM quality is subjective by nature; there are no captions ground-truth labels for arbitrary video frames. The approach used is a **LLM-as-judge** pattern:

1. Run inference on each frame with all 3 VLM models using the prompt: *"Describe the image in detail."*
2. For each generated caption, send the original image **and** the caption to a judge model (`google/gemini-2.0-flash-lite-001`) that scores the description 1–5 on:
   - **Accuracy** – does the description match what's visible?
   - **Completeness** – are all salient objects/people/actions mentioned?
   - **Detail** – level of semantic richness.
3. Parse the judge's JSON output and average scores across all frames per model.

### Person detection

1. Call each detection model with a vision prompt designed for person counting.
2. Parse the structured JSON response to extract `person_count`.
3. Compute **MAE**, **Precision**, **Recall**, and **F1** comparing predictions against the COCO-subset ground truth (2 frames).

### Latency

Measured end-to-end wall-clock time (including HTTP round-trip to OpenRouter cloud) per call,
stored as raw millisecond arrays, then aggregated to P50/P95/P99 and mean at the end.

### Cost

Estimated from published OpenRouter pricing (approximate, verify at <https://openrouter.ai/models>):

| Model | $/call (with image) |
|-------|---------------------|
| `google/gemini-2.0-flash-lite-001` | $0.00020 |
| `reka/reka-edge` | $0.00100 |
| `meta-llama/llama-3.2-11b-vision-instruct` | $0.00005 |

### Reliability

`reliability_rate = successful_calls / total_calls` — no HTTP errors were encountered during this run.

---

## 3. Results

### VLM Models

| Model | Calls | Errors | Reliability | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Cost/Call | Total Cost | Avg Quality |
|-------|-------|--------|-------------|----------|----------|----------|-----------|-----------|------------|-------------|
| `google/gemini-2.0-flash-lite-001` | 7 | 0 | 100.0% | 2298 | 4583 | 4583 | 2665 | $0.00020 | $0.0014 | 4.86/5 |
| `reka/reka-edge` | 7 | 0 | 100.0% | 1868 | 4392 | 4392 | 2518 | $0.00100 | $0.0070 | 3.86/5 |
| `meta-llama/llama-3.2-11b-vision-instruct` | 7 | 0 | 100.0% | 1057 | 3667 | 3667 | 1620 | $0.00005 | $0.0003 | 3.86/5 |

### Detection Models

| Model | Calls | Errors | Reliability | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Cost/Call | Total Cost | MAE | Precision | Recall | F1 |
|-------|-------|--------|-------------|----------|----------|----------|-----------|-----------|------------|-----|-----------|--------|-----|
| `google/gemini-2.0-flash-lite-001` | 7 | 0 | 100.0% | 3039 | 5233 | 5233 | 3214 | $0.00020 | $0.0014 | 0.500 | 1.000 | 1.000 | 1.000 |
| `reka/reka-edge` | 7 | 0 | 100.0% | 1958 | 7775 | 7775 | 3106 | $0.00100 | $0.0070 | 1.000 | 0.500 | 1.000 | 0.667 |

---

## 4. Recommendations

### Best Quality

| Type | Model | Justification |
|------|-------|---------------|
| vlm | `google/gemini-2.0-flash-lite-001` | Avg quality score: **4.86/5** — consistently the highest-rated captions by the judge. Despite mid-range latency (P95 4583ms), its image understanding translates to richer, more detailed descriptions. Reliability: 100%. Cost/call: $0.00020. |
| detection | `google/gemini-2.0-flash-lite-001` | **F1: 1.000** (perfect) on the 2-frame COCO subset. MAE: 0.500 — off by half a person on average. Precision and Recall both 1.000 for binary presence/absence. |

### Best Real-Time

| Type | Model | Justification |
|------|-------|---------------|
| vlm | `meta-llama/llama-3.2-11b-vision-instruct` | **P95: 3667ms**, mean: 1620ms — the fastest VLM by a wide margin. Quality score 3.86/5 meets the minimum threshold of 3.0. At $0.00005/call it is also the most cost-efficient option for latency-sensitive pipelines. |
| detection | `reka/reka-edge` | **P50: 1958ms** — faster median than Gemini Flash (3039ms) though with higher P95 (7775ms due to occasional slow requests). Acceptable when tail-latency outliers can be tolerated. |

### Best Overall

| Type | Model | Justification |
|------|-------|---------------|
| vlm | `google/gemini-2.0-flash-lite-001` | **Composite score: ~0.72** (weights — quality:0.35 latency:0.25 cost:0.20 reliability:0.15 practicality:0.05). Highest quality (4.86/5) dominates the quality dimension. Mid-tier cost ($0.00020/call) and 100% reliability make it the strongest overall candidate. |
| detection | `google/gemini-2.0-flash-lite-001` | **Perfect F1 (1.000)**, lower MAE (0.500 vs 1.000), 100% reliability, and 5× cheaper than `reka/reka-edge`. Highest composite score across all 5 dimensions. |

---

## 5. Tradeoffs

| Dimension | Observation |
|-----------|-------------|
| **Quality vs Latency (VLM)** | `google/gemini-2.0-flash-lite-001` scores highest quality (4.86/5) but `meta-llama/llama-3.2-11b-vision-instruct` is faster (P95: 3667ms vs 4583ms). For real-time use cases favor Llama; for archival / review tasks favor Gemini Flash. |
| **Quality vs Cost (VLM)** | `google/gemini-2.0-flash-lite-001` is the best-quality model at $0.00020/call — actually the second cheapest. The best quality/cost ratio in this evaluation; Llama is 4× cheaper but produces lower-quality captions. |
| **Accuracy vs Latency (Detection)** | `google/gemini-2.0-flash-lite-001` achieves perfect F1 and is faster at P50 (3039ms vs 1958ms for reka) but `reka/reka-edge` has a lower P50. Gemini Flash is more consistent (P95 = 5233ms) vs reka's spiky P95 (7775ms). |
| **Reliability** | All models accessed via OpenRouter cloud API — real-world latency includes network round-trip (Berlin → US/EU data centers). Local/edge deployment would reduce P95 significantly. No errors were observed in this run; retry/fallback logic is not implemented in the PoC. |
| **Cost at Scale** | At 1 000 calls/day: `meta-llama` costs ~$1.83/month, `gemini-2.0-flash-lite` ~$7.30/month, `reka-edge` ~$36.50/month. For cost-sensitive production workloads, Llama or Gemini Flash are preferable. |
| **Ground Truth Coverage** | Person-count MAE and F1 are evaluated on only **2 annotated frames** (COCO subset). Results are indicative, not statistically conclusive. Expanding `tests/data/ground_truth.json` to ≥50 frames would yield reliable accuracy estimates. |
| **VLM Judge Bias** | Quality scores are produced by `gemini-2.0-flash-lite` itself (`VLM_JUDGE_MODEL`). This may introduce self-preference bias — Gemini's own outputs could be scored more favourably. Re-run with an independent judge (e.g. `openai/gpt-4o`) to cross-validate. |

---

## 6. How to Reproduce

```bash
# Copy .env.example and set your API key
cp .env.example .env
echo "OPENROUTER_API_KEY=sk-or-..." >> .env

# Place video clips in video_model_eval/data/
# (MP4, H264, any resolution/duration)

# Run the full evaluation pipeline
cd video_model_eval
cargo run --release -- --evaluate --out results/

# Results are written to:
#   results/output.json   – raw per-frame per-model data + recommendations
#   results/results.csv   – tidy CSV rows for further analysis
#   results/summary.md    – auto-generated Markdown report
```

Environment variables that affect output:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter authentication key |
| `DATA_PATH` | `data` | Directory with MP4 clips |
| `RESULT_PATH` | `results` | Output directory |
| `MAX_CLIPS` | `5` | Maximum video clips to process |
| `VLM_JUDGE_MODEL` | `google/gemini-2.0-flash-lite-001` | Judge model for quality scoring |
| `MIN_QUALITY_THRESHOLD` | `3.0` | Minimum quality score for real-time eligibility |
| `MIN_RELIABILITY_THRESHOLD` | `0.8` | Minimum reliability for real-time eligibility |
| `WEIGHT_QUALITY` | `0.35` | Weight for quality dimension in composite score |
| `WEIGHT_LATENCY` | `0.25` | Weight for latency dimension |
| `WEIGHT_COST` | `0.20` | Weight for cost dimension |
| `WEIGHT_RELIABILITY` | `0.15` | Weight for reliability dimension |
| `WEIGHT_PRACTICALITY` | `0.05` | Weight for practicality dimension |
