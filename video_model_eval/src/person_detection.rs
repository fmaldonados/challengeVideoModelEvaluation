
// src/person_detection.rs
// Person detection via VLM (OpenRouter) using a structured detection prompt

use crate::models::PersonDetection;
use anyhow::{Result, anyhow};
use std::env;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Models available for person detection.
/// Index 0 is the default when PERSON_DETECTION_MODEL is not set.
pub const DETECTION_MODELS: &[&str] = &[
    "reka/reka-edge",
    "google/gemini-2.0-flash-lite-001",
];

// ── Structs para el request a OpenRouter ──────────────────────────────────────

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: Vec<ContentPart>,
}

#[derive(Serialize)]
struct DetectRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct VlmResponse {
    choices: Vec<Choice>,
}

// ── Extracción de frame con ffmpeg ────────────────────────────────────────────

/// Extrae el frame central del video usando ffmpeg.
/// Devuelve la ruta del archivo de imagen temporal generado.
pub fn extract_frame(video_path: &str) -> Result<String> {
    let frame_path = format!("{}.frame.jpg", video_path);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-i", video_path, "-vf", "thumbnail", "-frames:v", "1", &frame_path])
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| anyhow!("ffmpeg no encontrado: {e}"))?;
    if !status.success() {
        return Err(anyhow!("ffmpeg falló al extraer frame de: {}", video_path));
    }
    Ok(frame_path)
}

// ── Pipeline completo: video → frame → VLM ───────────────────────────────────

/// Extrae un frame del video y lo envía al VLM para detección de personas.
pub async fn detect_video(video_path: &str) -> Result<Vec<PersonDetection>> {
    let frame_path = extract_frame(video_path)?;
    let result = detect(&frame_path).await;
    let _ = std::fs::remove_file(&frame_path);
    result
}

/// Envía un frame (imagen .jpg) al VLM de OpenRouter con prompt de detección.
/// Retorna Vec<PersonDetection> — confidence = 1.0 si detecta personas, 0.0 si no.
pub async fn detect(frame_path: &str) -> Result<Vec<PersonDetection>> {
    let model = env::var("PERSON_DETECTION_MODEL")
        .unwrap_or_else(|_| DETECTION_MODELS[0].to_string());
    detect_with_model(frame_path, &model).await
}

/// Envía un frame al modelo especificado y retorna detecciones de personas.
pub async fn detect_with_model(frame_path: &str, model: &str) -> Result<Vec<PersonDetection>> {
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow!("OPENROUTER_API_KEY no configurada"))?;
    let url = "https://openrouter.ai/api/v1/chat/completions";

    // Leer imagen y codificar en base64
    let img_bytes = std::fs::read(frame_path)
        .map_err(|e| anyhow!("Error leyendo frame: {e}"))?;
    let img_b64 = base64::encode(&img_bytes);
    let data_url = format!("data:image/jpeg;base64,{}", img_b64);

    // El segundo modelo usa un prompt alternativo centrado en conteo explícito
    let prompt_text = if model.contains("gemini") {
        "Count every person (human being) visible in this image, including partial views. \
         Respond with JSON only: \
         {\"persons_detected\": true|false, \"count\": <number>, \"confidence\": <0.0-1.0>, \
         \"method\": \"direct_count\"}".to_string()
    } else {
        "Are there any people (humans) visible in this image? \
         Reply with JSON only in this format: \
         {\"persons_detected\": true|false, \"count\": <number>, \"confidence\": <0.0-1.0>}"
            .to_string()
    };

    let req_body = DetectRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentPart::Text { text: prompt_text },
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url: data_url },
                },
            ],
        }],
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_default();
    let start = Instant::now();
    let resp = client
        .post(url)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&req_body)
        .send()
        .await
        .map_err(|e| anyhow!("Request fallido: {e}"))?;

    let latency_ms = start.elapsed().as_millis() as u64;

    if !resp.status().is_success() {
        return Err(anyhow!("VLM API error: {} (latency: {}ms)", resp.status(), latency_ms));
    }

    let vlm_resp: VlmResponse = resp.json().await
        .map_err(|e| anyhow!("Error parseando respuesta VLM: {e}"))?;

    let raw = vlm_resp.choices.first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    parse_detection_response(&raw, latency_ms, model)
}

/// Ejecuta todos los modelos en `DETECTION_MODELS` sobre el mismo frame.
/// Retorna Vec<(model_name, Result<Vec<PersonDetection>>)>.
pub async fn detect_all_models(frame_path: &str) -> Vec<(String, Result<Vec<PersonDetection>>)> {
    let mut results = Vec::new();
    for &model in DETECTION_MODELS {
        let result = detect_with_model(frame_path, model).await;
        results.push((model.to_string(), result));
    }
    results
}

// ── Parseo de respuesta del modelo ────────────────────────────────────────────

#[derive(Deserialize)]
struct DetectionAnswer {
    persons_detected: bool,
    #[serde(default)]
    count: u32,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_confidence() -> f32 { 0.9 }

fn parse_detection_response(raw: &str, _latency_ms: u64, model: &str) -> Result<Vec<PersonDetection>> {
    // Extraer el bloque JSON aunque venga rodeado de texto
    let json_str = if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        &raw[start..=end]
    } else {
        return Err(anyhow!("Respuesta del modelo no contiene JSON: {}", raw));
    };

    let answer: DetectionAnswer = serde_json::from_str(json_str)
        .map_err(|e| anyhow!("JSON inválido en respuesta: {e} — raw: {json_str}"))?;

    if !answer.persons_detected {
        return Ok(vec![]);
    }

    // Generar una detección por persona encontrada (sin bbox real, confidence del modelo)
    let count = answer.count.max(1);
    let detections = (0..count).map(|_| PersonDetection {
        confidence: answer.confidence,
        bbox: (0, 0, 0, 0), // bbox no disponible vía VLM text-only
        person_count: count,
        model_name: model.to_string(),
    }).collect();

    Ok(detections)
}