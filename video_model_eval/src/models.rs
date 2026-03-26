// src/models.rs
// Define common data structures

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Inference {
    pub description: String,
    pub tags: Vec<String>,
    pub latency_ms: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PersonDetection {
    pub confidence: f32,
    pub bbox: (u32, u32, u32, u32), // x, y, w, h
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