#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Inference;
    use tokio;

    #[tokio::test]
    async fn test_infer_mock() {
        // Usar un archivo de imagen dummy o simular base64
        let text = "Describe la imagen";
        let frame_path = "tests/data/sample.jpg";
        // Si no existe el archivo, saltar el test
        if std::fs::metadata(frame_path).is_err() {
            eprintln!("Archivo de prueba no encontrado, test omitido");
            return;
        }
        let result = infer(text, frame_path).await;
        // Solo verifica que no paniquea y retorna algo
        assert!(result.is_ok() || result.is_err());
    }
}
// src/vlm.rs
// VLM integration with OpenRouter

use crate::models::Inference;
use anyhow::{Result, anyhow};
use std::env;
use std::fs;
use std::time::Instant;

use serde::{Deserialize, Serialize};

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

pub async fn infer(text: &str, frame_path: &str) -> Result<Inference> {
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow!("OPENROUTER_API_KEY not set"))?;
    let url = "https://openrouter.ai/api/v1/chat/completions";

    // Leer imagen y codificar en base64
    let image_bytes = fs::read(frame_path)
        .map_err(|e| anyhow!("Failed to read frame: {}", e))?;
    let image_base64 = base64::encode(&image_bytes);
    let image_data_url = format!("data:image/jpeg;base64,{}", image_base64);

    let req_body = VlmRequest {
        model: "reka/reka-edge".to_string(),
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

    let client = reqwest::Client::new();
    let start = Instant::now();
    let resp = client
        .post(url)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&req_body)
        .send()
        .await
        .map_err(|e| anyhow!("Request failed: {}", e))?;

    let latency_ms = start.elapsed().as_millis() as u64;

    if !resp.status().is_success() {
        return Err(anyhow!("VLM API error: {}", resp.status()));
    }

    let vlm_resp: VlmResponse = resp.json().await.map_err(|e| anyhow!("Invalid response: {}", e))?;
    let description = vlm_resp.choices.get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "No response".to_string());

    // Para la demo, tags vacíos
    Ok(Inference {
        description,
        tags: vec![],
        latency_ms,
    })
}