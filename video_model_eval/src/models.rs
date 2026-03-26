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
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EvaluationResult {
    pub model_name: String,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub latency_p95: u64,
    pub cost_estimate: f32,
}