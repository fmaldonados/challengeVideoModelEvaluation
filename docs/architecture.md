# Architecture

This document describes the system design, data flows, and key technical decisions for the Video Model Evaluation PoC.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CLI (main.rs)                       │
│  --list-clips  --vlm-test  --detect-test  --evaluate    │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     evaluate_all()          │
          │  (orchestration loop)       │
          └──┬──────────────────────┬──┘
             │                      │
    ┌────────▼──────┐     ┌─────────▼──────────┐
    │   vlm.rs      │     │ person_detection.rs │
    │               │     │                    │
    │ infer()       │     │ detect_video()     │
    │ infer_all_    │     │ detect()           │
    │   models()    │     │ detect_all_models()│
    │ judge_quality │     │ extract_frame()    │
    └────────┬──────┘     └─────────┬──────────┘
             │                      │
             └──────────┬───────────┘
                        │
               ┌────────▼────────┐
               │  evaluator.rs   │
               │                 │
               │ score()         │
               │ generate_       │
               │  recommendations│
               └────────┬────────┘
                        │
          ┌─────────────▼─────────────┐
          │      Output Writers       │
          │  output.json              │
          │  results.csv              │
          │  summary.md               │
          └───────────────────────────┘
```

---

## Data Flow

### Full Evaluation Pipeline (`--evaluate`)

```
1. Load config from env vars
   (DATA_PATH, MAX_CLIPS, RESULT_PATH, GROUND_TRUTH_PATH)

2. Load ground_truth.json
   → Vec<DetectionGroundTruth>

3. Scan DATA_PATH for *.mp4 files

4. For each clip:
   a. extract_frame(clip)  ← ffmpeg shells out, returns JPEG path
   b. Parallel dispatch:
      - infer_all_models(frame)    → Vec<(model, Result<Inference>)>
      - detect_all_models(frame)   → Vec<(model, Result<Vec<PersonDetection>>)>
   c. For each successful VLM inference:
      - judge_quality(caption, frame)  → Result<u8>  (score 1–5)
   d. Accumulate into ModelRawData per model

5. evaluator::score(&ModelRawData) per model
   → EvaluationResult

6. evaluator::generate_recommendations(&[EvaluationResult])
   → Vec<Recommendation>

7. Write output.json, results.csv, summary.md
```

### VLM Inference Flow

```
frame.jpg
   │
   ▼ base64 encode
   │
   ▼ build VlmRequest { model, messages: [{ role: user, content: [text, image_url] }] }
   │
   ▼ POST https://openrouter.ai/api/v1/chat/completions
     Authorization: Bearer $OPENROUTER_API_KEY
   │
   ▼ parse VlmResponse → choices[0].message.content
   │
   ▼ return Inference { model_name, description, latency_ms }
```

### Person Detection Flow

```
clip.mp4  OR  frame.jpg
   │
   ├── (if video) ffmpeg → frame.jpg
   │
   ▼ base64 encode
   │
   ▼ build DetectRequest { model, messages: [text prompt + image_url] }
     (model-specific prompt: Gemini → explicit count, Reka → presence/count)
   │
   ▼ POST https://openrouter.ai/api/v1/chat/completions
   │
   ▼ parse raw response → extract {...} JSON block
   │
   ▼ deserialize DetectionAnswer { persons_detected, count, confidence }
   │
   ▼ return Vec<PersonDetection> (empty if persons_detected=false)
```

### Quality Scoring (Judge)

```
caption (string) + frame.jpg
   │
   ▼ build prompt with structured criteria:
     - accuracy vs. image content
     - completeness (objects, actions, context)
     - level of detail
   │
   ▼ send to VLM_JUDGE_MODEL (default: gemini-2.0-flash-lite-001)
   │
   ▼ extract {"score": N} from raw response
   │
   ▼ clamp to [1, 5] → u8
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.rs` | CLI parsing, orchestration, output writing |
| `vlm.rs` | VLM inference, multi-model dispatch, LLM judge |
| `person_detection.rs` | Detection via VLM API, frame extraction via ffmpeg |
| `evaluator.rs` | Metrics (MAE, precision/recall, latency percentiles, cost), recommendations |
| `models.rs` | Shared data structures (serializable types, raw accumulators) |
| `lib.rs` | Library crate for integration tests |

---

## Concurrency Model

- The runtime is **Tokio** (async, full features).
- Within each clip, VLM inference and person detection calls are dispatched in parallel using `tokio::join!` and `futures::future::join_all`.
- Judge quality calls are fired in parallel after all VLM inferences for a clip complete.
- Clips are processed **sequentially** to keep the evaluation run predictable and to avoid rate-limiting on OpenRouter.

---

## External Dependencies

| Crate | Purpose |
|---|---|
| `reqwest` | Async HTTP client for OpenRouter API calls (rustls-tls) |
| `serde` / `serde_json` | JSON serialization for API requests and output artifacts |
| `tokio` | Async runtime |
| `futures` | `join_all` for parallel async tasks |
| `opencv` | OpenCV bindings (available for image/video processing extensions) |
| `csv` | CSV output writer |
| `base64` | Image encoding for multimodal API requests |
| `dotenv` | `.env` file loading |
| `tracing` / `tracing-subscriber` | Structured logging with `RUST_LOG` |
| `anyhow` | Ergonomic error propagation |

---

## API Interface (OpenRouter)

All model calls use the **OpenAI-compatible chat completions endpoint**:

```
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer <OPENROUTER_API_KEY>
Content-Type: application/json
```

Request body schema:

```json
{
  "model": "provider/model-name",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "..." },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
      ]
    }
  ]
}
```

Response: `choices[0].message.content` is the raw text output.

- **Timeout:** 60 seconds per request.
- **Auth:** Bearer token from `OPENROUTER_API_KEY` environment variable.
- **No retries:** As agreed for this PoC; failed calls are counted toward the error rate metric.

---

## Output Schema

### `output.json`

```json
{
  "run_id": "run-1774555503",
  "clip_results": [ <ClipResult>... ],
  "model_summaries": [ <EvaluationResult>... ],
  "recommendations": [ <Recommendation>... ]
}
```

### `results.csv`

Columns: `run_id, clip_name, model_type, model_name, metric_name, metric_value`

One row per (clip × model × metric). Suitable for SQL import (see schema in [plan-desarrollo-1.md](plans/plan-desarrollo-1.md)).

### `summary.md`

Auto-generated Markdown with:
- VLM results table
- Detection results table
- Recommendations (best quality / best real-time / best overall) per model type
- Tradeoffs section

---

## Design Decisions

### VLMs for person detection
Person detection is implemented via VLM API calls with a structured JSON prompt rather than a dedicated detection API (e.g. YOLO, COCO API). This was chosen for simplicity in the PoC and to keep the OpenRouter integration as the single external dependency. The tradeoff is that bounding boxes are not available — only counts and confidence.

### Judge-based quality evaluation
There is no ground truth for VLM captions, so quality is assessed using an LLM judge (another multimodal model) that scores each caption against the source image on criteria of accuracy, completeness, and detail. This is an established proxy technique, though it introduces judge model bias.

### Sequential clip processing
Clips are processed one at a time to avoid hammering OpenRouter with many concurrent requests and exceeding rate limits. Within each clip, all model calls are parallelized.

### No retries
Per PoC scope, failed API calls are surfaced as errors and counted toward the reliability metric without automatic retry or fallback.
