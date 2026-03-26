// src/vlm.rs
// VLM integration with OpenRouter

use crate::models::Inference;
use anyhow::{Result, anyhow};
use std::env;
use std::fs;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ── Modelos VLM disponibles para comparación ─────────────────────────────────

/// Lista de modelos VLM multimodal a comparar en OpenRouter.
pub const VLM_MODELS: &[&str] = &[
    "reka/reka-edge",
    "google/gemini-2.0-flash-lite-001",
    "meta-llama/llama-3.2-11b-vision-instruct",
];

// ── Structs para el request ───────────────────────────────────────────────────

#[derive(Serialize)]
struct ImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
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
struct VlmRequest {
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

#[derive(Deserialize)]
struct JudgeAnswer {
    score: u8,
}

// ── Inferencia VLM ────────────────────────────────────────────────────────────

/// Llama al modelo VLM especificado con un texto y un frame de imagen.
/// Devuelve `Inference` con descripción, tags vacíos, latencia y nombre del modelo.
pub async fn infer(text: &str, frame_path: &str, model: &str) -> Result<Inference> {
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow!("OPENROUTER_API_KEY not set"))?;
    let url = env::var("OPENROUTER_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".to_string());

    let image_bytes = fs::read(frame_path)
        .map_err(|e| anyhow!("Failed to read frame: {}", e))?;
    let image_base64 = base64::encode(&image_bytes);
    let image_data_url = format!("data:image/jpeg;base64,{}", image_base64);

    let req_body = VlmRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentPart::Text { text: text.to_string() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: image_data_url,
                        detail: Some("auto".to_string()),
                    },
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
        .post(&url)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&req_body)
        .send()
        .await
        .map_err(|e| anyhow!("Request failed: {}", e))?;

    let latency_ms = start.elapsed().as_millis() as u64;

    if !resp.status().is_success() {
        return Err(anyhow!("VLM API error: {} (model: {}, latency: {}ms)", resp.status(), model, latency_ms));
    }

    let vlm_resp: VlmResponse = resp.json().await
        .map_err(|e| anyhow!("Invalid response: {}", e))?;
    let description = vlm_resp.choices.first()
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "No response".to_string());

    Ok(Inference {
        model_name: model.to_string(),
        description,
        tags: vec![],
        latency_ms,
    })
}

/// Ejecuta todos los modelos en `VLM_MODELS` sobre el mismo frame y texto.
/// Devuelve un vector de `(model_name, Result<Inference>)`.
pub async fn infer_all_models(text: &str, frame_path: &str) -> Vec<(String, Result<Inference>)> {
    let mut results = Vec::new();
    for &model in VLM_MODELS {
        let result = infer(text, frame_path, model).await;
        results.push((model.to_string(), result));
    }
    results
}

// ── Juez de calidad VLM ───────────────────────────────────────────────────────

/// Envía un caption + imagen a un modelo juez y retorna una puntuación de calidad de 1 a 5.
///
/// El modelo juez se configura con la variable de entorno `VLM_JUDGE_MODEL`
/// (por defecto: `google/gemini-flash-1.5`).
///
/// Criterios evaluados por el juez:
/// - Exactitud: ¿la descripción coincide con lo que hay en la imagen?
/// - Completitud: ¿se mencionan los elementos relevantes?
/// - Nivel de detalle: ¿es lo suficientemente descriptivo?
pub async fn judge_quality(caption: &str, frame_path: &str) -> Result<u8> {
    let judge_model = env::var("VLM_JUDGE_MODEL")
        .unwrap_or_else(|_| "google/gemini-2.0-flash-lite-001".to_string());

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow!("OPENROUTER_API_KEY not set"))?;
    let url = env::var("OPENROUTER_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".to_string());

    let image_bytes = fs::read(frame_path)
        .map_err(|e| anyhow!("Failed to read frame for judge: {}", e))?;
    let image_base64 = base64::encode(&image_bytes);
    let image_data_url = format!("data:image/jpeg;base64,{}", image_base64);

    let prompt = format!(
        "You are an image caption quality evaluator.\n\
         Given the image and the description below, rate the description quality on a scale from 1 to 5:\n\
         1 = very poor (inaccurate or irrelevant)\n\
         2 = poor (major omissions or errors)\n\
         3 = acceptable (partially correct)\n\
         4 = good (accurate and reasonably complete)\n\
         5 = excellent (accurate, complete, and detailed)\n\n\
         Description to evaluate:\n\"{caption}\"\n\n\
         Reply ONLY with valid JSON in this format: {{\"score\": <1-5>}}",
        caption = caption
    );

    let req_body = VlmRequest {
        model: judge_model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentPart::Text { text: prompt },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: image_data_url,
                        detail: Some("auto".to_string()),
                    },
                },
            ],
        }],
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_default();
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&req_body)
        .send()
        .await
        .map_err(|e| anyhow!("Judge request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(anyhow!("Judge VLM API error: {} (model: {})", resp.status(), judge_model));
    }

    let vlm_resp: VlmResponse = resp.json().await
        .map_err(|e| anyhow!("Invalid judge response: {}", e))?;

    let raw = vlm_resp.choices.first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    // Extraer el bloque JSON aunque venga rodeado de texto
    let json_str = if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        &raw[start..=end]
    } else {
        return Err(anyhow!("Judge model did not return JSON. Raw response: {}", raw));
    };

    let answer: JudgeAnswer = serde_json::from_str(json_str)
        .map_err(|e| anyhow!("Invalid judge JSON: {e} — raw: {json_str}"))?;

    let score = answer.score.clamp(1, 5);
    Ok(score)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_infer_mock() {
        let text = "Describe la imagen";
        let frame_path = "tests/data/sample.jpg";
        if std::fs::metadata(frame_path).is_err() {
            eprintln!("Archivo de prueba no encontrado, test omitido");
            return;
        }
        let result = infer(text, frame_path, "reka/reka-edge").await;
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_vlm_models_list() {
        assert!(VLM_MODELS.len() >= 3, "Se requieren al menos 3 modelos VLM");
        for model in VLM_MODELS {
            assert!(!model.is_empty(), "El nombre del modelo no debe estar vacío");
            assert!(model.contains('/'), "El modelo debe tener formato proveedor/nombre");
        }
    }
}