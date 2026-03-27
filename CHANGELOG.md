# Changelog

## [0.1.0] — 2026-03-26

PoC de evaluación de VLM y detección de personas.

### Added

#### Phase 1 — Project structure & test data
- Initialized Rust project (`cargo new video_model_eval --bin`) with lib + binary targets.
- Defined module layout: `main.rs`, `vlm.rs`, `person_detection.rs`, `evaluator.rs`, `models.rs`, `lib.rs`.
- Added `run.sh` and `test.sh` wrapper scripts for running from the repo root.
- CLI flag `--list-clips` lists video clips found in `data/`.

#### Phase 2 — VLM integration (OpenRouter)
- `vlm::infer(text, frame_path, model)` — multimodal inference via OpenRouter API; base64-encoded image, bearer auth, 60s timeout.
- `vlm::infer_all_models(text, frame_path)` — runs all models in `VLM_MODELS` sequentially.
- `vlm::judge_quality(caption, frame_path)` — LLM-as-judge scoring (1–5) for caption quality evaluation.
- `VLM_MODELS` constant: `reka/reka-edge`, `google/gemini-2.0-flash-lite-001`, `meta-llama/llama-3.2-11b-vision-instruct`.
- CLI flags `--vlm-test` and `--vlm-all-models`.
- `VlmQualityScore` struct in `models.rs`.
- Unit tests: model list validation, judge response parsing.

#### Phase 3 — Person detection
- `person_detection::extract_frame(video_path)` — extracts a representative frame via `ffmpeg`.
- `person_detection::detect_with_model(frame_path, model)` — VLM-based person counting with model-specific prompts.
- `person_detection::detect_all_models(frame_path)` — runs all `DETECTION_MODELS`.
- `DETECTION_MODELS` constant: `reka/reka-edge`, `google/gemini-2.0-flash-lite-001`.
- `PersonDetection` model with `person_count` and `model_name` fields.
- `tests/data/ground_truth.json` — annotated 2-frame COCO subset for detection evaluation.
- `evaluator::calculate_mae` implementation.
- CLI flags `--detect-test`, `--detect-all-models`, `--detect-ground-truth`.

#### Phase 4 — Evaluation & metrics
- `evaluator::score(&ModelRawData) -> EvaluationResult` — full metrics aggregation: reliability rate, latency percentiles (p50/p95/p99/mean), cost, MAE, precision, recall, F1, avg quality score.
- `evaluator::calculate_precision_recall` and `evaluator::calculate_latency_percentiles` helpers.
- `evaluator::cost_per_call_usd(model)` — hard-coded cost estimates per model.
- `models.rs` updated with `EvaluationResult`, `ModelRawData`, `ClipResult`, `Recommendation`.
- `--evaluate [--out <path>]` CLI command — full pipeline: frame extraction → parallel inference → judge scoring → metric aggregation.
- Output artifacts: `results/output.json`, `results/results.csv`, `results/summary.md`.
- Integration tests for synthetic evaluation scenarios.

#### Phase 5 — Recommendations & report
- `evaluator::generate_recommendations(&[EvaluationResult])` — three recommendations per model type: best quality, best real-time, best overall.
- Best overall uses a 5-dimension weighted composite score with configurable env-var weights.
- `docs/reports/summary.md` — full write-up with methodology, results tables, recommendations, and tradeoffs.

#### Documentation
- `README.md` — setup, CLI reference, env vars, project structure, results summary.
- `docs/architecture.md` — system design, data flows, module responsibilities, API schema, design decisions.
- `docs/business-rules.md` — evaluation rules, thresholds, scoring logic.
- `docs/coco_subset.md` — ground truth dataset description and COCO expansion instructions.
- `CHANGELOG.md` (this file).
