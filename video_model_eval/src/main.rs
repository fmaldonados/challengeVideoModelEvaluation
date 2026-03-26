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