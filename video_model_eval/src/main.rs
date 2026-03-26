use std::fs;

mod models;
mod vlm;
mod person_detection;
mod evaluator;


#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--list-clips" {
        list_clips();
    } else if args.len() > 1 && args[1] == "--vlm-test" {
        vlm_test().await;
    } else if args.len() > 1 && args[1] == "--vlm-all-models" {
        vlm_all_models_test().await;
    } else if args.len() > 1 && args[1] == "--detect-test" {
        detect_test().await;
    } else if args.len() > 1 && args[1] == "--detect-all-models" {
        detect_all_models_test().await;
    } else if args.len() > 1 && args[1] == "--detect-ground-truth" {
        detect_ground_truth_eval().await;
    } else if args.len() > 1 && args[1] == "--evaluate" {
        // Support optional --out <path> argument
        let out_path = args.iter()
            .position(|a| a == "--out")
            .and_then(|i| args.get(i + 1))
            .cloned();
        evaluate_all(out_path).await;
    } else {
        println!("Hello, world!");
    }
async fn detect_test() {
    use crate::person_detection;
    use std::env;
    // Buscar el primer archivo mp4 en data/
    let data_path = "data";
    let mut test_clip = None;
    if let Ok(entries) = fs::read_dir(data_path) {
        for entry in entries.flatten() {
        // Cargar variables de entorno desde .env si existe
        let _ = dotenv::dotenv();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("mp4") {
                test_clip = Some(path);
                break;
            }
        }
    }
    let clip_path = match test_clip {
        Some(p) => p,
        None => {
            eprintln!("No se encontró ningún video en data/");
            return;
        }
    };
    // Usar umbral configurable
    let threshold: f32 = env::var("DETECTION_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    // Llamar a la detección
    match person_detection::detect_video(clip_path.to_str().unwrap()).await {
        Ok(detections) => {
            let found = detections.iter().any(|d| d.confidence >= threshold);
            let person_count = detections.first().map(|d| d.person_count).unwrap_or(0);
            println!("detected={} person_count={}", found, person_count);
            // Opcional: imprimir JSON de detecciones
            // println!("{}", serde_json::to_string_pretty(&detections).unwrap());
        }
        Err(e) => {
            eprintln!("Error en la detección: {}", e);
        }
    }
}

async fn detect_all_models_test() {
    use crate::person_detection;
    use serde_json;
    let frame_path = "data/sample.jpg";
    println!("Running all {} detection models...\n", person_detection::DETECTION_MODELS.len());
    let results = person_detection::detect_all_models(frame_path).await;
    for (model, result) in &results {
        match result {
            Ok(detections) => {
                let count = detections.first().map(|d| d.person_count).unwrap_or(0);
                println!("=== {} → {} person(s) detected ===", model, count);
                println!("{}\n", serde_json::to_string_pretty(detections).unwrap());
            }
            Err(e) => eprintln!("=== {} ERROR ===\n{}\n", model, e),
        }
    }
}

async fn detect_ground_truth_eval() {
    use crate::person_detection;
    use crate::evaluator;
    use crate::models::DetectionGroundTruth;

    let gt_path = "tests/data/ground_truth.json";
    let gt_raw = match std::fs::read_to_string(gt_path) {
        Ok(s) => s,
        Err(e) => { eprintln!("No se pudo leer {}: {}", gt_path, e); return; }
    };
    let ground_truth_entries: Vec<DetectionGroundTruth> = match serde_json::from_str(&gt_raw) {
        Ok(v) => v,
        Err(e) => { eprintln!("JSON inválido en {}: {}", gt_path, e); return; }
    };

    println!("Evaluando {} imágenes contra ground truth...\n", ground_truth_entries.len());

    for model in person_detection::DETECTION_MODELS {
        let mut predictions: Vec<u32> = Vec::new();
        let mut gt_counts: Vec<u32> = Vec::new();
        println!("--- Modelo: {} ---", model);
        for gt_entry in &ground_truth_entries {
            let result = person_detection::detect_with_model(&gt_entry.image_path, model).await;
            let predicted_count = match &result {
                Ok(detections) => detections.first().map(|d| d.person_count).unwrap_or(0),
                Err(e) => {
                    eprintln!("  {} → ERROR: {}", gt_entry.image_path, e);
                    continue;
                }
            };
            println!(
                "  {} → predicted={} gt={}",
                gt_entry.image_path, predicted_count, gt_entry.person_count
            );
            predictions.push(predicted_count);
            gt_counts.push(gt_entry.person_count);
        }
        let mae = evaluator::calculate_mae(&predictions, &gt_counts);
        println!("  MAE = {:.3}\n", mae);
    }
}
}
async fn vlm_test() {
    use crate::vlm;
    use serde_json;
    let _ = dotenv::dotenv();
    // Usar el primer modelo de la lista por defecto
    let model = std::env::var("VLM_MODEL")
        .unwrap_or_else(|_| vlm::VLM_MODELS[0].to_string());
    let frame_path = "data/sample.jpg";
    let text = "Describe the image in detail.";
    match vlm::infer(text, frame_path, &model).await {
        Ok(inf) => println!("{}", serde_json::to_string_pretty(&inf).unwrap()),
        Err(e) => eprintln!("Error: {}", e),
    }
}

async fn vlm_all_models_test() {
    use crate::vlm;
    use serde_json;
    let frame_path = "data/sample.jpg";
    let text = "Describe the image in detail.";
    println!("Running all {} VLM models...\n", vlm::VLM_MODELS.len());
    let results = vlm::infer_all_models(text, frame_path).await;
    for (model, result) in &results {
        match result {
            Ok(inf) => {
                println!("=== {} ({}ms) ===", model, inf.latency_ms);
                println!("{}\n", inf.description);
            }
            Err(e) => eprintln!("=== {} ERROR ===\n{}\n", model, e),
        }
    }
    // Calcular puntuaciones de calidad con el juez VLM
    println!("--- VLM Quality Judge Scores ---");
    for (model, result) in &results {
        if let Ok(inf) = result {
            match vlm::judge_quality(&inf.description, frame_path).await {
                Ok(score) => println!("{}: {}/5", model, score),
                Err(e) => eprintln!("{}: judge error — {}", model, e),
            }
        }
    }
    println!("\nFull JSON output:");
    let summaries: Vec<serde_json::Value> = results.iter().map(|(model, r)| {
        match r {
            Ok(inf) => serde_json::json!({
                "model": model,
                "ok": true,
                "description": inf.description,
                "latency_ms": inf.latency_ms,
            }),
            Err(e) => serde_json::json!({
                "model": model,
                "ok": false,
                "error": e.to_string(),
            }),
        }
    }).collect();
    println!("{}", serde_json::to_string_pretty(&summaries).unwrap());
}

async fn evaluate_all(out_path: Option<String>) {
    use crate::evaluator;
    use crate::person_detection;
    use crate::vlm;
    use crate::models::{ClipResult, DetectionGroundTruth, ModelRawData};
    use std::collections::HashMap;

    let _ = dotenv::dotenv();

    // ── Configuration ─────────────────────────────────────────────────────────
    let run_id = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("run-{}", secs)
    };

    let data_path = std::env::var("DATA_PATH").unwrap_or_else(|_| "data".to_string());
    let result_path = out_path
        .or_else(|| std::env::var("RESULT_PATH").ok())
        .unwrap_or_else(|| "results".to_string());
    let gt_path = std::env::var("GROUND_TRUTH_PATH")
        .unwrap_or_else(|_| "tests/data/ground_truth.json".to_string());
    let max_clips: usize = std::env::var("MAX_CLIPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    std::fs::create_dir_all(&result_path).expect("Cannot create results directory");
    println!("=== Evaluation run: {} ===", run_id);
    println!("Results will be written to: {}/", result_path);

    // ── Load ground truth ─────────────────────────────────────────────────────
    let ground_truth: Vec<DetectionGroundTruth> = std::fs::read_to_string(&gt_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    // ── Collect frames to process ──────────────────────────────────────────────
    struct FrameInfo {
        path: String,
        clip_name: String,
        gt_count: Option<u32>,
        temp: bool, // delete after use
    }

    let mut frames: Vec<FrameInfo> = Vec::new();

    // Pre-extracted ground truth images
    for entry in &ground_truth {
        if std::fs::metadata(&entry.image_path).is_ok() {
            frames.push(FrameInfo {
                path: entry.image_path.clone(),
                clip_name: entry.image_path.clone(),
                gt_count: Some(entry.person_count),
                temp: false,
            });
        }
    }

    // Video clips (extract frame, sorted, limited to max_clips)
    let mut extracted = 0usize;
    if let Ok(entries) = std::fs::read_dir(&data_path) {
        let mut clips: Vec<_> = entries
            .flatten()
            .filter(|e| {
                e.path().extension().and_then(|s| s.to_str()) == Some("mp4")
            })
            .collect();
        clips.sort_by_key(|e| e.file_name());
        for entry in clips.into_iter().take(max_clips) {
            let path = entry.path();
            let clip_name = path.file_name().unwrap().to_str().unwrap().to_string();
            print!("Extracting frame from {}... ", clip_name);
            match person_detection::extract_frame(path.to_str().unwrap()) {
                Ok(frame_path) => {
                    println!("OK");
                    frames.push(FrameInfo {
                        path: frame_path,
                        clip_name,
                        gt_count: None,
                        temp: true,
                    });
                    extracted += 1;
                }
                Err(e) => println!("ERROR: {}", e),
            }
        }
    }

    println!(
        "\nProcessing {} frames ({} from GT images, {} from video clips)\n",
        frames.len(),
        ground_truth.len(),
        extracted
    );

    // ── Per-model raw data accumulators ───────────────────────────────────────
    let mut vlm_raw: HashMap<String, ModelRawData> = vlm::VLM_MODELS
        .iter()
        .map(|&m| {
            (
                m.to_string(),
                ModelRawData {
                    model_name: m.to_string(),
                    model_type: "vlm".to_string(),
                    latencies_ms: vec![],
                    error_count: 0,
                    person_predictions: vec![],
                    person_ground_truth: vec![],
                    quality_scores: vec![],
                },
            )
        })
        .collect();

    let mut det_raw: HashMap<String, ModelRawData> = person_detection::DETECTION_MODELS
        .iter()
        .map(|&m| {
            (
                m.to_string(),
                ModelRawData {
                    model_name: m.to_string(),
                    model_type: "detection".to_string(),
                    latencies_ms: vec![],
                    error_count: 0,
                    person_predictions: vec![],
                    person_ground_truth: vec![],
                    quality_scores: vec![],
                },
            )
        })
        .collect();

    let mut all_clip_results: Vec<ClipResult> = Vec::new();
    let vlm_text = "Describe the image in detail.";

    // ── Run inference on each frame ───────────────────────────────────────────
    for frame in &frames {
        println!("--- Frame: {} ---", frame.clip_name);

        // Lanzar todas las inferencias (VLM + detection) en paralelo
        let vlm_futures: Vec<_> = vlm::VLM_MODELS
            .iter()
            .map(|&model| {
                let fp = frame.path.clone();
                let txt = vlm_text;
                async move {
                    let result = vlm::infer(txt, &fp, model).await;
                    (model, result, fp)
                }
            })
            .collect();

        let det_futures: Vec<_> = person_detection::DETECTION_MODELS
            .iter()
            .map(|&model| {
                let fp = frame.path.clone();
                async move {
                    let start = std::time::Instant::now();
                    let result = person_detection::detect_with_model(&fp, model).await;
                    let latency_ms = start.elapsed().as_millis() as u64;
                    (model, result, latency_ms)
                }
            })
            .collect();

        let (vlm_results, det_results) = tokio::join!(
            futures::future::join_all(vlm_futures),
            futures::future::join_all(det_futures),
        );

        // ── Collect VLM results + run judge calls in parallel ─────────────────
        let judge_futures: Vec<_> = vlm_results
            .iter()
            .filter_map(|(model, result, fp)| {
                if let Ok(inf) = result {
                    let desc = inf.description.clone();
                    let fp = fp.clone();
                    Some(async move {
                        let score = vlm::judge_quality(&desc, &fp).await;
                        (*model, score)
                    })
                } else {
                    None
                }
            })
            .collect();

        let judge_scores: Vec<(_, _)> = futures::future::join_all(judge_futures).await;
        let judge_map: std::collections::HashMap<&str, Result<u8, _>> =
            judge_scores.into_iter().collect();

        for (model, result, _fp) in &vlm_results {
            match result {
                Ok(inf) => {
                    println!("  [VLM] {} OK ({}ms)", model, inf.latency_ms);
                    let raw = vlm_raw.get_mut(*model).unwrap();
                    raw.latencies_ms.push(inf.latency_ms);

                    let quality = match judge_map.get(model) {
                        Some(Ok(s)) => {
                            raw.quality_scores.push(*s);
                            Some(*s)
                        }
                        Some(Err(e)) => {
                            eprintln!("  (judge error for {}: {})", model, e);
                            None
                        }
                        None => None,
                    };

                    all_clip_results.push(ClipResult {
                        run_id: run_id.clone(),
                        clip_name: frame.clip_name.clone(),
                        model_name: model.to_string(),
                        model_type: "vlm".to_string(),
                        latency_ms: inf.latency_ms,
                        success: true,
                        error_msg: None,
                        description: Some(inf.description.clone()),
                        quality_score: quality,
                        person_count: None,
                        ground_truth_count: None,
                    });
                }
                Err(e) => {
                    println!("  [VLM] {} ERROR: {}", model, e);
                    vlm_raw.get_mut(*model).unwrap().error_count += 1;
                    all_clip_results.push(ClipResult {
                        run_id: run_id.clone(),
                        clip_name: frame.clip_name.clone(),
                        model_name: model.to_string(),
                        model_type: "vlm".to_string(),
                        latency_ms: 0,
                        success: false,
                        error_msg: Some(e.to_string()),
                        description: None,
                        quality_score: None,
                        person_count: None,
                        ground_truth_count: None,
                    });
                }
            }
        }

        // ── Collect detection results ─────────────────────────────────────────
        for (model, result, latency_ms) in &det_results {
            match result {
                Ok(detections) => {
                    let count = detections.first().map(|d| d.person_count).unwrap_or(0);
                    println!("  [DET] {} OK ({} person(s), {}ms)", model, count, latency_ms);
                    let raw = det_raw.get_mut(*model).unwrap();
                    raw.latencies_ms.push(*latency_ms);
                    if let Some(gt) = frame.gt_count {
                        raw.person_predictions.push(count);
                        raw.person_ground_truth.push(gt);
                    }
                    all_clip_results.push(ClipResult {
                        run_id: run_id.clone(),
                        clip_name: frame.clip_name.clone(),
                        model_name: model.to_string(),
                        model_type: "detection".to_string(),
                        latency_ms: *latency_ms,
                        success: true,
                        error_msg: None,
                        description: None,
                        quality_score: None,
                        person_count: Some(count),
                        ground_truth_count: frame.gt_count,
                    });
                }
                Err(e) => {
                    println!("  [DET] {} ERROR: {}", model, e);
                    det_raw.get_mut(*model).unwrap().error_count += 1;
                    all_clip_results.push(ClipResult {
                        run_id: run_id.clone(),
                        clip_name: frame.clip_name.clone(),
                        model_name: model.to_string(),
                        model_type: "detection".to_string(),
                        latency_ms: 0,
                        success: false,
                        error_msg: Some(e.to_string()),
                        description: None,
                        quality_score: None,
                        person_count: None,
                        ground_truth_count: frame.gt_count,
                    });
                }
            }
        }

        // Cleanup temporary frame
        if frame.temp {
            let _ = std::fs::remove_file(&frame.path);
        }
    }

    // ── Aggregate metrics per model ────────────────────────────────────────────
    let mut eval_results = Vec::new();
    for (_, raw) in &vlm_raw {
        match evaluator::score(raw) {
            Ok(r) => eval_results.push(r),
            Err(e) => eprintln!("Score error for {}: {}", raw.model_name, e),
        }
    }
    for (_, raw) in &det_raw {
        match evaluator::score(raw) {
            Ok(r) => eval_results.push(r),
            Err(e) => eprintln!("Score error for {}: {}", raw.model_name, e),
        }
    }

    // ── Write output.json ─────────────────────────────────────────────────────
    let json_path = format!("{}/output.json", result_path);
    let recommendations = evaluator::generate_recommendations(&eval_results);
    let json_out = serde_json::json!({
        "run_id": run_id,
        "clip_results": all_clip_results,
        "model_summaries": eval_results,
        "recommendations": recommendations,
    });
    match std::fs::write(&json_path, serde_json::to_string_pretty(&json_out).unwrap()) {
        Ok(_) => println!("\nWrote {}", json_path),
        Err(e) => eprintln!("Error writing {}: {}", json_path, e),
    }

    // ── Write results.csv ─────────────────────────────────────────────────────
    let csv_path = format!("{}/results.csv", result_path);
    let mut csv_rows: Vec<String> = vec![
        "run_id,clip_name,model_type,model_name,metric_name,metric_value".to_string(),
    ];
    for r in &all_clip_results {
        let base = format!(
            "{},{},{},{}",
            r.run_id, r.clip_name, r.model_type, r.model_name
        );
        csv_rows.push(format!("{},latency_ms,{}", base, r.latency_ms));
        csv_rows.push(format!("{},success,{}", base, r.success as u8));
        if let Some(qs) = r.quality_score {
            csv_rows.push(format!("{},quality_score,{}", base, qs));
        }
        if let Some(pc) = r.person_count {
            csv_rows.push(format!("{},person_count,{}", base, pc));
        }
        if let Some(gt) = r.ground_truth_count {
            csv_rows.push(format!("{},person_count_gt,{}", base, gt));
        }
    }
    match std::fs::write(&csv_path, csv_rows.join("\n")) {
        Ok(_) => println!("Wrote {}", csv_path),
        Err(e) => eprintln!("Error writing {}: {}", csv_path, e),
    }

    // ── Write summary.md ──────────────────────────────────────────────────────
    let md_path = format!("{}/summary.md", result_path);
    let summary_md = build_summary_md(&run_id, &eval_results, &recommendations, all_clip_results.len());
    match std::fs::write(&md_path, &summary_md) {
        Ok(_) => println!("Wrote {}", md_path),
        Err(e) => eprintln!("Error writing {}: {}", md_path, e),
    }

    println!("\n=== Evaluation complete ===");
}

fn build_summary_md(
    run_id: &str,
    results: &[crate::models::EvaluationResult],
    recommendations: &[crate::models::Recommendation],
    total_frames: usize,
) -> String {
    use std::fmt::Write as FmtWrite;
    let mut md = String::new();

    let _ = writeln!(md, "# Evaluation Summary\n");
    let _ = writeln!(md, "**Run ID:** {}  ", run_id);
    let _ = writeln!(md, "**Frames processed:** {}  \n", total_frames);

    // VLM models table
    let vlm: Vec<_> = results.iter().filter(|r| r.model_type == "vlm").collect();
    if !vlm.is_empty() {
        let _ = writeln!(md, "## VLM Models\n");
        let _ = writeln!(
            md,
            "| Model | Calls | Errors | Reliability | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Cost/Call | Total Cost | Avg Quality |"
        );
        let _ = writeln!(
            md,
            "|-------|-------|--------|-------------|----------|----------|----------|-----------|-----------|------------|-------------|"
        );
        for r in &vlm {
            let quality = r
                .avg_quality_score
                .map(|s| format!("{:.2}/5", s))
                .unwrap_or_else(|| "N/A".to_string());
            let _ = writeln!(
                md,
                "| {} | {} | {} | {:.1}% | {} | {} | {} | {:.0} | ${:.5} | ${:.4} | {} |",
                r.model_name,
                r.total_calls,
                r.error_count,
                r.reliability_rate * 100.0,
                r.latency_p50,
                r.latency_p95,
                r.latency_p99,
                r.latency_mean,
                r.cost_per_call_usd,
                r.cost_total_usd,
                quality
            );
        }
        let _ = writeln!(md);
    }

    // Detection models table
    let det: Vec<_> = results.iter().filter(|r| r.model_type == "detection").collect();
    if !det.is_empty() {
        let _ = writeln!(md, "## Detection Models\n");
        let _ = writeln!(
            md,
            "| Model | Calls | Errors | Reliability | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Cost/Call | Total Cost | MAE | Precision | Recall | F1 |"
        );
        let _ = writeln!(
            md,
            "|-------|-------|--------|-------------|----------|----------|----------|-----------|-----------|------------|-----|-----------|--------|-----|"
        );
        for r in &det {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {:.1}% | {} | {} | {} | {:.0} | ${:.5} | ${:.4} | {} | {} | {} | {} |",
                r.model_name,
                r.total_calls,
                r.error_count,
                r.reliability_rate * 100.0,
                r.latency_p50,
                r.latency_p95,
                r.latency_p99,
                r.latency_mean,
                r.cost_per_call_usd,
                r.cost_total_usd,
                r.mae.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                r.precision.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                r.recall.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
                r.f1.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "N/A".to_string()),
            );
        }
        let _ = writeln!(md);
    }

    // ── Recommendations ───────────────────────────────────────────────────────
    if !recommendations.is_empty() {
        let _ = writeln!(md, "## Recommendations\n");
        for category in &["best_quality", "best_real_time", "best_overall"] {
            let label = match *category {
                "best_quality"   => "Best Quality",
                "best_real_time" => "Best Real-Time",
                "best_overall"   => "Best Overall",
                _                => category,
            };
            let _ = writeln!(md, "### {}\n", label);
            let _ = writeln!(md, "| Type | Model | Justification |");
            let _ = writeln!(md, "|------|-------|---------------|");
            for rec in recommendations.iter().filter(|r| r.category == *category) {
                let _ = writeln!(md, "| {} | `{}` | {} |", rec.model_type, rec.model_name, rec.justification);
            }
            let _ = writeln!(md);
        }
    }

    // ── Tradeoffs ─────────────────────────────────────────────────────────────
    let _ = writeln!(md, "## Tradeoffs\n");
    let _ = writeln!(md, "| Dimension | Observation |");
    let _ = writeln!(md, "|-----------|-------------|");
    // VLM tradeoffs
    let vlm: Vec<_> = results.iter().filter(|r| r.model_type == "vlm").collect();
    if vlm.len() >= 2 {
        let best_q = vlm.iter().max_by(|a, b| {
            a.avg_quality_score.unwrap_or(0.0).partial_cmp(&b.avg_quality_score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let fastest = vlm.iter().min_by_key(|r| r.latency_p95);
        let cheapest = vlm.iter().min_by(|a, b| {
            a.cost_per_call_usd.partial_cmp(&b.cost_per_call_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let (Some(bq), Some(ft)) = (best_q, fastest) {
            if bq.model_name != ft.model_name {
                let _ = writeln!(
                    md,
                    "| Quality vs Latency (VLM) | `{}` scores highest quality ({:.2}/5) but `{}` is fastest (P95: {}ms vs {}ms). Choose based on real-time requirements. |",
                    bq.model_name,
                    bq.avg_quality_score.unwrap_or(0.0),
                    ft.model_name,
                    ft.latency_p95,
                    bq.latency_p95,
                );
            }
        }
        if let (Some(bq), Some(ch)) = (best_q, cheapest) {
            if bq.model_name != ch.model_name {
                let _ = writeln!(
                    md,
                    "| Quality vs Cost (VLM) | `{}` achieves best quality but costs ${:.5}/call. `{}` costs ${:.5}/call — {:.0}× cheaper. |",
                    bq.model_name,
                    bq.cost_per_call_usd,
                    ch.model_name,
                    ch.cost_per_call_usd,
                    bq.cost_per_call_usd / ch.cost_per_call_usd.max(1e-9),
                );
            }
        }
    }
    // Detection tradeoffs
    let det: Vec<_> = results.iter().filter(|r| r.model_type == "detection").collect();
    if det.len() >= 2 {
        let best_f1 = det.iter().max_by(|a, b| {
            a.f1.unwrap_or(0.0).partial_cmp(&b.f1.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let fastest = det.iter().min_by_key(|r| r.latency_p95);
        if let (Some(bf), Some(ft)) = (best_f1, fastest) {
            if bf.model_name != ft.model_name {
                let _ = writeln!(
                    md,
                    "| Accuracy vs Latency (Detection) | `{}` achieves best F1 ({}) but `{}` is faster (P95: {}ms vs {}ms). |",
                    bf.model_name,
                    bf.f1.map(|f| format!("{:.3}", f)).unwrap_or_else(|| "N/A".to_string()),
                    ft.model_name,
                    ft.latency_p95,
                    bf.latency_p95,
                );
            }
        }
    }
    let _ = writeln!(md, "| Reliability | All models accessed via OpenRouter cloud API — real-world latency includes network round-trip. Local/edge deployment would reduce P95 significantly. |");
    let _ = writeln!(md, "| Cost at Scale | At 1 000 calls/day, even the cheapest model (~$0.00005/call) costs ~$1.83/month vs the most expensive (~$0.001/call) at ~$36.5/month. |");
    let _ = writeln!(md, "| Ground Truth Coverage | Person-count MAE is computed only on the COCO subset (2 annotated frames). Expand `tests/data/ground_truth.json` for statistically significant results. |");
    let _ = writeln!(md);

    md
}

fn list_clips() {
    let data_path = "data";
    if let Ok(entries) = fs::read_dir(data_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("mp4") {
                    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                        println!("{}", file_name);
                    }
                }
            }
        }
    } else {
        eprintln!("Error reading data directory");
    }
}