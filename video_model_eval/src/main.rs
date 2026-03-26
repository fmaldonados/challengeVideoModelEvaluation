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
    } else if args.len() > 1 && args[1] == "--detect-test" {
        detect_test().await;
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
            println!("{}", found);
            // Opcional: imprimir JSON de detecciones
            // println!("{}", serde_json::to_string_pretty(&detections).unwrap());
        }
        Err(e) => {
            eprintln!("Error en la detección: {}", e);
        }
    }
}
}
async fn vlm_test() {
    use crate::vlm;
    use serde_json;
    // Usar un archivo de imagen de ejemplo
    let frame_path = "data/sample.jpg";
    let text = "Describe la imagen";
    match vlm::infer(text, frame_path).await {
        Ok(inf) => println!("{}", serde_json::to_string_pretty(&inf).unwrap()),
        Err(e) => eprintln!("Error: {}", e),
    }
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