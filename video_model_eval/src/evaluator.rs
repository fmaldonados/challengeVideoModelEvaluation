// src/evaluator.rs
// Evaluation and metrics calculation

use crate::models::{EvaluationResult, ModelRawData, Recommendation};
use anyhow::Result;

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

/// Calcula precision y recall binarios (presencia de persona: count > 0 → positivo).
/// Devuelve (precision, recall). Cuando el denominador es 0, retorna 0.0.
pub fn calculate_precision_recall(predictions: &[u32], ground_truth: &[u32]) -> (f32, f32) {
    assert_eq!(
        predictions.len(),
        ground_truth.len(),
        "predictions and ground_truth must have the same length"
    );
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut false_neg = 0u32;
    for (&p, &g) in predictions.iter().zip(ground_truth.iter()) {
        match (p > 0, g > 0) {
            (true, true)  => tp += 1,
            (true, false) => fp += 1,
            (false, true) => false_neg += 1,
            (false, false) => {}
        }
    }
    let precision = if tp + fp == 0 { 0.0 } else { tp as f32 / (tp + fp) as f32 };
    let recall    = if tp + false_neg == 0 { 0.0 } else { tp as f32 / (tp + false_neg) as f32 };
    (precision, recall)
}

/// Calcula percentiles de latencia (p50, p95, p99) y la media aritmética.
/// Devuelve (p50_ms, p95_ms, p99_ms, mean_ms).
pub fn calculate_latency_percentiles(latencies: &[u64]) -> (u64, u64, u64, f64) {
    if latencies.is_empty() {
        return (0, 0, 0, 0.0);
    }
    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p50 = sorted[(n.saturating_sub(1)) / 2];
    let p95 = sorted[((n * 95) / 100).min(n - 1)];
    let p99 = sorted[((n * 99) / 100).min(n - 1)];
    let mean = sorted.iter().sum::<u64>() as f64 / n as f64;
    (p50, p95, p99, mean)
}

/// Devuelve el costo estimado en USD por llamada API para el modelo dado.
/// Valores aproximados basados en precios de OpenRouter (verificar en https://openrouter.ai/models).
pub fn cost_per_call_usd(model: &str) -> f64 {
    if model.contains("gemini-2.0-flash-lite") {
        0.0002   // ~$0.0002/call con imagen (bajo costo, alta velocidad)
    } else if model.contains("reka-edge") {
        0.001    // ~$0.001/call con imagen
    } else if model.contains("llama-3.2-11b") {
        0.00005  // free tier / muy económico en OpenRouter
    } else {
        0.001    // fallback conservador
    }
}

/// Calcula métricas agregadas para un modelo a partir de los datos crudos recopilados
/// durante la evaluación de todos los frames.
pub fn score(data: &ModelRawData) -> Result<EvaluationResult> {
    let total_calls = data.latencies_ms.len() as u32 + data.error_count;
    let reliability_rate = if total_calls == 0 {
        1.0_f32
    } else {
        data.latencies_ms.len() as f32 / total_calls as f32
    };

    let (p50, p95, p99, mean) = calculate_latency_percentiles(&data.latencies_ms);

    let cpc = cost_per_call_usd(&data.model_name);
    let cost_total = cpc * total_calls as f64;

    let (mae, precision, recall, f1) =
        if !data.person_predictions.is_empty()
            && data.person_predictions.len() == data.person_ground_truth.len()
        {
            let mae = calculate_mae(&data.person_predictions, &data.person_ground_truth);
            let (prec, rec) =
                calculate_precision_recall(&data.person_predictions, &data.person_ground_truth);
            let f1 = if prec + rec < 1e-6 {
                0.0
            } else {
                2.0 * prec * rec / (prec + rec)
            };
            (Some(mae), Some(prec), Some(rec), Some(f1))
        } else {
            (None, None, None, None)
        };

    let avg_quality_score = if !data.quality_scores.is_empty() {
        let sum: f32 = data.quality_scores.iter().map(|&s| s as f32).sum();
        Some(sum / data.quality_scores.len() as f32)
    } else {
        None
    };

    Ok(EvaluationResult {
        model_name: data.model_name.clone(),
        model_type: data.model_type.clone(),
        total_calls,
        error_count: data.error_count,
        reliability_rate,
        latency_mean: mean,
        latency_p50: p50,
        latency_p95: p95,
        latency_p99: p99,
        cost_per_call_usd: cpc,
        cost_total_usd: cost_total,
        mae,
        precision,
        recall,
        f1,
        avg_quality_score,
    })
}

/// Generates model recommendations across three categories for each model-type group
/// present in `results`:
///
/// - **`best_quality`**: highest avg quality score (VLM) or highest F1 / lowest MAE (detection).
/// - **`best_real_time`**: fastest p95 latency that still meets the minimum quality and
///   reliability thresholds (`MIN_QUALITY_THRESHOLD`, default 3.0; `MIN_RELIABILITY_THRESHOLD`,
///   default 0.8).
/// - **`best_overall`**: weighted composite score across 5 dimensions — quality, latency, cost,
///   reliability, practicality.  Weights are configurable via env vars `WEIGHT_QUALITY` (0.35),
///   `WEIGHT_LATENCY` (0.25), `WEIGHT_COST` (0.20), `WEIGHT_RELIABILITY` (0.15),
///   `WEIGHT_PRACTICALITY` (0.05).
pub fn generate_recommendations(results: &[EvaluationResult]) -> Vec<Recommendation> {
    let mut recs = Vec::new();

    for model_type in &["vlm", "detection"] {
        let group: Vec<&EvaluationResult> = results
            .iter()
            .filter(|r| r.model_type == *model_type)
            .collect();

        if group.is_empty() {
            continue;
        }

        // ── 1. Best Quality ──────────────────────────────────────────────────
        let best_q = if *model_type == "vlm" {
            group.iter().copied().max_by(|a, b| {
                let qa = a.avg_quality_score.unwrap_or(0.0);
                let qb = b.avg_quality_score.unwrap_or(0.0);
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
        } else {
            group.iter().copied().max_by(|a, b| {
                let fa = a.f1.unwrap_or(0.0);
                let fb = b.f1.unwrap_or(0.0);
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            })
        };

        if let Some(m) = best_q {
            let score = if *model_type == "vlm" {
                m.avg_quality_score.unwrap_or(0.0)
            } else {
                m.f1.unwrap_or(0.0)
            };
            let justification = if *model_type == "vlm" {
                format!(
                    "Avg quality score: {:.2}/5 | Reliability: {:.1}% | Latency P95: {}ms | Cost/call: ${:.5}",
                    m.avg_quality_score.unwrap_or(0.0),
                    m.reliability_rate * 100.0,
                    m.latency_p95,
                    m.cost_per_call_usd,
                )
            } else {
                format!(
                    "F1: {} | MAE: {} | Precision: {} | Recall: {} | Reliability: {:.1}%",
                    m.f1.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                    m.mae.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                    m.precision.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                    m.recall.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                    m.reliability_rate * 100.0,
                )
            };
            recs.push(Recommendation {
                category: "best_quality".to_string(),
                model_type: model_type.to_string(),
                model_name: m.model_name.clone(),
                justification,
                weighted_score: score,
            });
        }

        // ── 2. Best Real-Time ────────────────────────────────────────────────
        let min_quality: f32 = std::env::var("MIN_QUALITY_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3.0_f32);
        let min_reliability: f32 = std::env::var("MIN_RELIABILITY_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.8_f32);

        let eligible_rt: Vec<&EvaluationResult> = group
            .iter()
            .copied()
            .filter(|r| {
                r.reliability_rate >= min_reliability
                    && if *model_type == "vlm" {
                        r.avg_quality_score.unwrap_or(0.0) >= min_quality
                    } else {
                        true
                    }
            })
            .collect();

        let best_rt = if eligible_rt.is_empty() {
            // Fall back to fastest regardless of quality
            group.iter().copied().min_by_key(|r| r.latency_p95)
        } else {
            eligible_rt.iter().copied().min_by_key(|r| r.latency_p95)
        };

        if let Some(m) = best_rt {
            let quality_suffix = if *model_type == "vlm" {
                format!(" | Quality: {:.2}/5", m.avg_quality_score.unwrap_or(0.0))
            } else {
                m.f1.map(|f| format!(" | F1: {:.3}", f)).unwrap_or_default()
            };
            recs.push(Recommendation {
                category: "best_real_time".to_string(),
                model_type: model_type.to_string(),
                model_name: m.model_name.clone(),
                justification: format!(
                    "Latency P95: {}ms | P50: {}ms | Mean: {:.0}ms | Reliability: {:.1}%{}",
                    m.latency_p95,
                    m.latency_p50,
                    m.latency_mean,
                    m.reliability_rate * 100.0,
                    quality_suffix,
                ),
                weighted_score: 1.0 / (1.0 + m.latency_p95 as f32 / 1000.0),
            });
        }

        // ── 3. Best Overall (weighted 5-dimension score) ─────────────────────
        let w_quality: f32 = std::env::var("WEIGHT_QUALITY")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.35);
        let w_latency: f32 = std::env::var("WEIGHT_LATENCY")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
        let w_cost: f32 = std::env::var("WEIGHT_COST")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.20);
        let w_reliability: f32 = std::env::var("WEIGHT_RELIABILITY")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.15);
        let w_practicality: f32 = std::env::var("WEIGHT_PRACTICALITY")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);

        // Normalize within this type group
        let max_p95 = group.iter().map(|r| r.latency_p95).max().unwrap_or(1) as f32;
        let min_p95 = group.iter().map(|r| r.latency_p95).min().unwrap_or(0) as f32;
        let max_cost = group.iter().map(|r| r.cost_per_call_usd as f32)
            .fold(f32::NEG_INFINITY, f32::max).max(1e-9_f32);
        let min_cost = group.iter().map(|r| r.cost_per_call_usd as f32)
            .fold(f32::INFINITY, f32::min);

        let scored: Vec<(&EvaluationResult, f32)> = group.iter().map(|&r| {
            let quality_dim = if *model_type == "vlm" {
                r.avg_quality_score.unwrap_or(0.0) / 5.0
            } else {
                r.f1.unwrap_or(0.0)
            };

            let latency_dim = if (max_p95 - min_p95).abs() < 1e-6 {
                1.0
            } else {
                (max_p95 - r.latency_p95 as f32) / (max_p95 - min_p95)
            };

            let cost_dim = if (max_cost - min_cost).abs() < 1e-9 {
                1.0
            } else {
                (max_cost - r.cost_per_call_usd as f32) / (max_cost - min_cost)
            };

            let reliability_dim = r.reliability_rate;
            // Practicality: reliability + cost-efficiency proxy
            let practicality_dim = (reliability_dim + cost_dim) / 2.0;

            let total = w_quality * quality_dim
                + w_latency * latency_dim
                + w_cost * cost_dim
                + w_reliability * reliability_dim
                + w_practicality * practicality_dim;

            (r, total)
        }).collect();

        if let Some((m, &total_score)) = scored.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, s)| (r, s))
        {
            let quality_str = if *model_type == "vlm" {
                m.avg_quality_score.map(|s| format!("{:.2}/5", s)).unwrap_or_else(|| "N/A".to_string())
            } else {
                m.f1.map(|f| format!("F1={:.3}", f)).unwrap_or_else(|| "N/A".to_string())
            };
            recs.push(Recommendation {
                category: "best_overall".to_string(),
                model_type: model_type.to_string(),
                model_name: m.model_name.clone(),
                justification: format!(
                    "Composite score: {:.3} (weights — quality:{:.2} latency:{:.2} cost:{:.2} reliability:{:.2} practicality:{:.2}) | {} | P95: {}ms | Cost/call: ${:.5}",
                    total_score,
                    w_quality, w_latency, w_cost, w_reliability, w_practicality,
                    quality_str,
                    m.latency_p95,
                    m.cost_per_call_usd,
                ),
                weighted_score: total_score,
            });
        }
    }

    recs
}
