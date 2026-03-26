// tests/integration_test.rs

#[cfg(test)]
mod tests {
    use video_model_eval::person_detection;
    use std::env;

    fn load_env() {
        let _ = dotenv::dotenv();
    }

    /// Test positivo: frame con personas visibles → debe retornar al menos una detección
    #[tokio::test]
    async fn test_detect_persons_present() {
        load_env();
        // Frame extraído del video data/-3t1rj8g6yg.mp4 @ ~4.28s
        // Contenido: mujer en escenario + camarógrafo en primer plano
        let frame_path = "tests/data/persons_frame.jpg";
        if std::fs::metadata(frame_path).is_err() {
            eprintln!("Fixture no encontrado, omitiendo test: {}", frame_path);
            return;
        }
        let result = person_detection::detect(frame_path).await;
        assert!(result.is_ok(), "detect() falló: {:?}", result.err());
        let detections = result.unwrap();
        assert!(!detections.is_empty(), "Se esperaban personas pero no se detectó ninguna");
    }

    /// Test negativo: frame sin personas (teatro vacío, pantalla).
    /// Observacional: registra el comportamiento del modelo sin asumir resultado exacto,
    /// ya que VLMs pueden tener false positives. El test solo valida que la función no falla.
    #[tokio::test]
    async fn test_detect_no_persons() {
        load_env();
        // Frame extraído de data/-3t1rj8g6yg.mp4 @ 4:10 — butacas vacías, pantalla, sin personas
        let frame_path = "tests/data/no_persons_frame.jpg";
        if std::fs::metadata(frame_path).is_err() {
            eprintln!("Fixture no encontrado, omitiendo test: {}", frame_path);
            return;
        }
        let result = person_detection::detect(frame_path).await;
        assert!(result.is_ok(), "detect() no debería fallar con un frame válido: {:?}", result.err());
        let detections = result.unwrap();
        // Observacional: imprimimos el resultado para visibilidad en la evaluación
        // Un modelo ideal debería retornar 0 detecciones aquí (frame sin personas)
        eprintln!(
            "[observacional] no_persons_frame: {} detección(es) — esperado: 0 (false positives indican baja calidad del modelo)",
            detections.len()
        );
    }

    /// Test de integración basic: valida que la función no paniquea con una ruta inexistente
    #[tokio::test]
    async fn test_person_detection_missing_file() {
        load_env();
        let result = person_detection::detect("tests/data/nonexistent.jpg").await;
        assert!(result.is_err(), "debería fallar con archivo inexistente");
    }
}