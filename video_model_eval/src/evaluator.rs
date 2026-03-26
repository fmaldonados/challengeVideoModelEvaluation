// src/evaluator.rs
// Evaluation and metrics calculation

use crate::models::EvaluationResult;
use anyhow::Result;

pub fn score(_model_name: &str, _predictions: Vec<f32>, _ground_truth: Vec<f32>) -> Result<EvaluationResult> {
    // Placeholder implementation
    // TODO: Calculate actual metrics
    Ok(EvaluationResult {
        model_name: _model_name.to_string(),
        accuracy: 0.85,
        precision: 0.8,
        recall: 0.9,
        f1: 0.85,
        latency_p95: 150,
        cost_estimate: 0.1,
    })
}

/// Calcula el Mean Absolute Error entre conteos predichos y ground truth.
/// Ambos slices deben tener la misma longitud.
pub fn calculate_mae(predictions: &[u32], ground_truth: &[u32]) -> f32 {
    assert_eq!(
        predictions.len(),
        ground_truth.len(),
        "predictions and ground_truth must have the same length"
    );
    if predictions.is_empty() {
        return 0.0;
    }
    let sum: f32 = predictions
        .iter()
        .zip(ground_truth.iter())
        .map(|(&p, &g)| (p as f32 - g as f32).abs())
        .sum();
    sum / predictions.len() as f32
}
