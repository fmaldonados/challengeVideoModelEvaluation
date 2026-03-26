
// src/person_detection.rs
// Person detection via VLM (OpenRouter) using a structured detection prompt

use crate::models::PersonDetection;
use anyhow::{Result, anyhow};
use std::env;
use std::time::Instant;
use serde::{Deserialize, Serialize};

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
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow!("OPENROUTER_API_KEY no configurada"))?;
    let model = env::var("PERSON_DETECTION_MODEL")
        .unwrap_or_else(|_| "reka/reka-edge".to_string());
    let url = "https://openrouter.ai/api/v1/chat/completions";

    // Leer imagen y codificar en base64
    let img_bytes = std::fs::read(frame_path)
        .map_err(|e| anyhow!("Error leyendo frame: {e}"))?;
    let img_b64 = base64::encode(&img_bytes);
    let data_url = format!("data:image/jpeg;base64,{}", img_b64);

    let req_body = DetectRequest {
        model,
        messages: vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentPart::Text {
                    text: "Are there any people (humans) visible in this image? \
                           Reply with JSON only in this format: \
                           {\"persons_detected\": true|false, \"count\": <number>, \"confidence\": <0.0-1.0>}"
                        .to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url: data_url },
                },
            ],
        }],
    };

    let client = reqwest::Client::new();
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

    // Parsear la respuesta JSON del modelo
    parse_detection_response(&raw, latency_ms)
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

fn parse_detection_response(raw: &str, _latency_ms: u64) -> Result<Vec<PersonDetection>> {
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
    }).collect();

    Ok(detections)
}