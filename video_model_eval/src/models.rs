// src/models.rs
// Define common data structures

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Inference {
    pub model_name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub latency_ms: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VlmQualityScore {
    pub model_name: String,
    pub clip: String,
    pub score: u8,        // 1-5, rated by VLM judge
    pub judge_model: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PersonDetection {
    pub confidence: f32,
    pub bbox: (u32, u32, u32, u32), // x, y, w, h
    pub person_count: u32,           // total persons detected in this frame
    pub model_name: String,          // model that generated this detection
}

/// Entry in a ground-truth dataset (e.g. COCO subset)
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DetectionGroundTruth {
    pub image_path: String,
    pub person_count: u32,
    #[serde(default)]
    pub notes: String,
}

/// Aggregated evaluation results for a single model.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EvaluationResult {
    pub model_name: String,
    pub model_type: String,          // "vlm" or "detection"
    pub total_calls: u32,
    pub error_count: u32,
    pub reliability_rate: f32,       // successful_calls / total_calls

    // Latency (milliseconds)
    pub latency_mean: f64,
    pub latency_p50: u64,
    pub latency_p95: u64,
    pub latency_p99: u64,

    // Cost (approximate USD)
    pub cost_per_call_usd: f64,
    pub cost_total_usd: f64,

    // Detection-specific (None when no ground truth available)
    pub mae: Option<f32>,
    pub precision: Option<f32>,
    pub recall: Option<f32>,
    pub f1: Option<f32>,

    // VLM-specific (None when judge could not run)
    pub avg_quality_score: Option<f32>,
}

/// Raw data collected for a single model across all evaluated frames.
/// Passed to `evaluator::score()` to compute aggregated metrics.
#[derive(Debug)]
pub struct ModelRawData {
    pub model_name: String,
    pub model_type: String,           // "vlm" or "detection"
    pub latencies_ms: Vec<u64>,
    pub error_count: u32,
    // Detection-specific: parallel slices of (prediction, ground_truth) person counts
    pub person_predictions: Vec<u32>,
    pub person_ground_truth: Vec<u32>,
    // VLM-specific: judge scores (1–5)
    pub quality_scores: Vec<u8>,
}

/// A model recommendation for a specific usage category.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Recommendation {
    /// "best_quality" | "best_real_time" | "best_overall"
    pub category: String,
    /// "vlm" or "detection"
    pub model_type: String,
    pub model_name: String,
    pub justification: String,
    pub weighted_score: f32,
}

/// Per-frame, per-model result used for JSON output and CSV rows.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ClipResult {
    pub run_id: String,
    pub clip_name: String,
    pub model_name: String,
    pub model_type: String,
    pub latency_ms: u64,
    pub success: bool,
    pub error_msg: Option<String>,
    // VLM fields
    pub description: Option<String>,
    pub quality_score: Option<u8>,
    // Detection fields
    pub person_count: Option<u32>,
    pub ground_truth_count: Option<u32>,
}