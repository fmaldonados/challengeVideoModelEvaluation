// tests/integration_test.rs

#[cfg(test)]
mod tests {
    use video_model_eval::person_detection;
    use video_model_eval::evaluator;
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
        // Validar campos nuevos
        let first = &detections[0];
        assert!(first.person_count > 0, "person_count debe ser > 0 cuando hay detecciones");
        assert!(!first.model_name.is_empty(), "model_name no debe estar vacío");
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

    /// Valida que DETECTION_MODELS tenga al menos 2 modelos configurados
    #[test]
    fn test_detection_models_list() {
        assert!(
            person_detection::DETECTION_MODELS.len() >= 2,
            "Se requieren al menos 2 modelos de detección para comparación"
        );
        for model in person_detection::DETECTION_MODELS {
            assert!(!model.is_empty(), "Ningún modelo debe estar vacío");
        }
    }

    /// Valida el cálculo de MAE con datos sintéticos conocidos
    #[test]
    fn test_calculate_mae_synthetic() {
        // MAE([3, 1, 0], [2, 1, 0]) = (1 + 0 + 0) / 3 = 0.333...
        let predictions   = vec![3u32, 1, 0];
        let ground_truth  = vec![2u32, 1, 0];
        let mae = evaluator::calculate_mae(&predictions, &ground_truth);
        assert!((mae - 1.0 / 3.0).abs() < 1e-5, "MAE esperado ~0.333, obtenido {}", mae);
    }

    /// Valida que MAE con predicción perfecta sea 0
    #[test]
    fn test_calculate_mae_perfect() {
        let counts = vec![0u32, 2, 5];
        let mae = evaluator::calculate_mae(&counts, &counts);
        assert_eq!(mae, 0.0, "MAE debe ser 0.0 con predicción perfecta");
    }

    /// Valida que MAE con slices vacíos devuelva 0
    #[test]
    fn test_calculate_mae_empty() {
        let mae = evaluator::calculate_mae(&[], &[]);
        assert_eq!(mae, 0.0, "MAE debe ser 0.0 con slices vacíos");
    }
}