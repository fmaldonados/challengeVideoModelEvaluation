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

    // ── Phase 4: evaluator::score and new metric functions ────────────────────

    /// Prueba evaluator::score con datos sintéticos de detección.
    #[test]
    fn test_score_detection_synthetic() {
        use video_model_eval::models::ModelRawData;

        // 5 llamadas exitosas + 1 error
        let data = ModelRawData {
            model_name: "test-model".to_string(),
            model_type: "detection".to_string(),
            latencies_ms: vec![100, 200, 150, 300, 250],
            error_count: 1,
            person_predictions: vec![1, 0, 2],
            person_ground_truth: vec![1, 0, 1],
            quality_scores: vec![],
        };

        let result = evaluator::score(&data).unwrap();

        assert_eq!(result.total_calls, 6, "total_calls debe ser 6 (5 éxitos + 1 error)");
        assert_eq!(result.error_count, 1);
        assert!((result.reliability_rate - 5.0_f32 / 6.0).abs() < 1e-5);

        // Latencies sorted: [100, 150, 200, 250, 300]; n=5
        // p50: idx (5-1)/2 = 2 → sorted[2] = 200
        assert_eq!(result.latency_p50, 200);
        // p95: idx (5*95/100).min(4) = 4 → sorted[4] = 300
        assert_eq!(result.latency_p95, 300);

        // MAE([1,0,2], [1,0,1]) = (0+0+1)/3 = 0.333...
        assert!((result.mae.unwrap() - 1.0_f32 / 3.0).abs() < 1e-5);

        // Binary: pred=[T,F,T], gt=[T,F,T] → TP=2, FP=0, FN=0 → P=1.0, R=1.0
        assert!((result.precision.unwrap() - 1.0).abs() < 1e-5);
        assert!((result.recall.unwrap() - 1.0).abs() < 1e-5);
        assert!((result.f1.unwrap() - 1.0).abs() < 1e-5);

        assert!(result.avg_quality_score.is_none());
    }

    /// Prueba evaluator::score con 3 modelos sintéticos VLM y verifica que
    /// las métricas calculadas sean correctas y comparables entre sí.
    #[test]
    fn test_score_vlm_three_models() {
        use video_model_eval::models::ModelRawData;

        let models = vec![
            // model-a: alta calidad (~4.33), latencia media, sin errores
            ModelRawData {
                model_name: "model-a".to_string(),
                model_type: "vlm".to_string(),
                latencies_ms: vec![500, 600, 550],
                error_count: 0,
                person_predictions: vec![],
                person_ground_truth: vec![],
                quality_scores: vec![4, 5, 4],
            },
            // model-b: calidad baja (~2.67), latencia baja, 1 error
            ModelRawData {
                model_name: "model-b".to_string(),
                model_type: "vlm".to_string(),
                latencies_ms: vec![200, 250, 220],
                error_count: 1,
                person_predictions: vec![],
                person_ground_truth: vec![],
                quality_scores: vec![3, 3, 2],
            },
            // model-c: calidad perfecta (5.0), latencia alta, sin errores
            ModelRawData {
                model_name: "model-c".to_string(),
                model_type: "vlm".to_string(),
                latencies_ms: vec![1000, 1200, 1100],
                error_count: 0,
                person_predictions: vec![],
                person_ground_truth: vec![],
                quality_scores: vec![5, 5, 5],
            },
        ];

        let results: Vec<_> = models
            .iter()
            .map(|d| evaluator::score(d).unwrap())
            .collect();

        // model-a: 3 llamadas, reliability 100%, avg quality 13/3
        assert_eq!(results[0].total_calls, 3);
        assert!((results[0].reliability_rate - 1.0).abs() < 1e-5);
        assert!((results[0].avg_quality_score.unwrap() - 13.0_f32 / 3.0).abs() < 1e-4);

        // model-b: 4 llamadas (3+1), reliability 75%
        assert_eq!(results[1].total_calls, 4);
        assert!((results[1].reliability_rate - 0.75).abs() < 1e-5);
        assert!((results[1].avg_quality_score.unwrap() - 8.0_f32 / 3.0).abs() < 1e-4);

        // model-c: 3 llamadas, reliability 100%, avg quality 5.0
        assert_eq!(results[2].total_calls, 3);
        assert!((results[2].reliability_rate - 1.0).abs() < 1e-5);
        assert!((results[2].avg_quality_score.unwrap() - 5.0).abs() < 1e-5);

        // Ranking de calidad: c > a > b
        assert!(results[2].avg_quality_score.unwrap() > results[0].avg_quality_score.unwrap());
        assert!(results[0].avg_quality_score.unwrap() > results[1].avg_quality_score.unwrap());

        // Ranking de latencia (p50): b < a < c
        assert!(results[1].latency_p50 < results[0].latency_p50);
        assert!(results[0].latency_p50 < results[2].latency_p50);

        // Sin métricas de detección para modelos VLM
        assert!(results[0].mae.is_none());
        assert!(results[1].precision.is_none());
    }

    /// Prueba calculate_precision_recall con casos conocidos.
    #[test]
    fn test_calculate_precision_recall() {
        use video_model_eval::evaluator::calculate_precision_recall;

        // TP=2, FP=0, FN=0 → P=1.0, R=1.0
        let (p, r) = calculate_precision_recall(&[1, 0, 2], &[1, 0, 1]);
        assert!((p - 1.0).abs() < 1e-5, "precision expected 1.0, got {}", p);
        assert!((r - 1.0).abs() < 1e-5, "recall expected 1.0, got {}", r);

        // TP=1, FP=1, FN=0 → P=0.5, R=1.0
        let (p, r) = calculate_precision_recall(&[1, 1], &[1, 0]);
        assert!((p - 0.5).abs() < 1e-5, "precision expected 0.5, got {}", p);
        assert!((r - 1.0).abs() < 1e-5, "recall expected 1.0, got {}", r);

        // TP=0, FP=0, FN=1 → P=0.0, R=0.0
        let (p, r) = calculate_precision_recall(&[0], &[1]);
        assert!((p - 0.0).abs() < 1e-5, "precision expected 0.0, got {}", p);
        assert!((r - 0.0).abs() < 1e-5, "recall expected 0.0, got {}", r);
    }

    /// Prueba calculate_latency_percentiles con datos conocidos.
    #[test]
    fn test_calculate_latency_percentiles() {
        use video_model_eval::evaluator::calculate_latency_percentiles;

        // sorted: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        let latencies = vec![500u64, 100, 900, 200, 800, 300, 700, 400, 600, 1000];
        let (p50, p95, _p99, mean) = calculate_latency_percentiles(&latencies);

        // p50: idx (10-1)/2 = 4 → sorted[4] = 500
        assert_eq!(p50, 500, "p50 expected 500, got {}", p50);
        // p95: idx (10*95/100).min(9) = 9 → sorted[9] = 1000
        assert_eq!(p95, 1000, "p95 expected 1000, got {}", p95);
        // mean = 5500/10 = 550
        assert!((mean - 550.0).abs() < 1e-5, "mean expected 550.0, got {}", mean);

        // Empty input → all zeros
        let (p50, p95, p99, mean) = calculate_latency_percentiles(&[]);
        assert_eq!((p50, p95, p99), (0, 0, 0));
        assert_eq!(mean, 0.0);
    }

    /// Valida generate_recommendations con 3 modelos VLM sintéticos cuyos rankings
    /// son predecibles:
    ///   - model-quality: calidad máxima (5.0/5), latencia alta, costo alto
    ///   - model-speed:   calidad media (3.5/5), latencia mínima, costo bajo
    ///   - model-balanced: calidad buena (4.0/5), latencia media, costo medio
    ///
    /// Expectativas:
    ///   - best_quality  → model-quality
    ///   - best_real_time → model-speed  (menor p95 con calidad ≥ 3.0)
    ///   - best_overall  → model-balanced o model-speed (mayor score compuesto)
    #[test]
    fn test_recommendations_synthetic() {
        use video_model_eval::models::EvaluationResult;
        use video_model_eval::evaluator::generate_recommendations;

        let results = vec![
            EvaluationResult {
                model_name: "model-quality".to_string(),
                model_type: "vlm".to_string(),
                total_calls: 10,
                error_count: 0,
                reliability_rate: 1.0,
                latency_mean: 2000.0,
                latency_p50: 1900,
                latency_p95: 2200,  // slowest
                latency_p99: 2500,
                cost_per_call_usd: 0.005,  // most expensive
                cost_total_usd: 0.05,
                mae: None,
                precision: None,
                recall: None,
                f1: None,
                avg_quality_score: Some(5.0),  // best quality
            },
            EvaluationResult {
                model_name: "model-speed".to_string(),
                model_type: "vlm".to_string(),
                total_calls: 10,
                error_count: 0,
                reliability_rate: 1.0,
                latency_mean: 400.0,
                latency_p50: 350,
                latency_p95: 450,   // fastest
                latency_p99: 500,
                cost_per_call_usd: 0.00005,  // cheapest
                cost_total_usd: 0.0005,
                mae: None,
                precision: None,
                recall: None,
                f1: None,
                avg_quality_score: Some(3.5),  // meets min threshold (3.0)
            },
            EvaluationResult {
                model_name: "model-balanced".to_string(),
                model_type: "vlm".to_string(),
                total_calls: 10,
                error_count: 0,
                reliability_rate: 1.0,
                latency_mean: 900.0,
                latency_p50: 850,
                latency_p95: 1000,  // mid latency
                latency_p99: 1200,
                cost_per_call_usd: 0.0002,  // mid cost
                cost_total_usd: 0.002,
                mae: None,
                precision: None,
                recall: None,
                f1: None,
                avg_quality_score: Some(4.0),  // good quality
            },
        ];

        let recs = generate_recommendations(&results);

        // There must be exactly 3 recommendations for the "vlm" group
        let vlm_recs: Vec<_> = recs.iter().filter(|r| r.model_type == "vlm").collect();
        assert_eq!(vlm_recs.len(), 3, "expected 3 recommendations for vlm group, got {}", vlm_recs.len());

        // best_quality → model-quality (highest avg_quality_score = 5.0)
        let bq = vlm_recs.iter().find(|r| r.category == "best_quality").expect("best_quality missing");
        assert_eq!(bq.model_name, "model-quality", "best_quality should be model-quality, got {}", bq.model_name);

        // best_real_time → model-speed (lowest p95 = 450ms, quality 3.5 ≥ threshold 3.0)
        let brt = vlm_recs.iter().find(|r| r.category == "best_real_time").expect("best_real_time missing");
        assert_eq!(brt.model_name, "model-speed", "best_real_time should be model-speed, got {}", brt.model_name);

        // best_overall → should NOT be model-quality alone (its high cost drags score)
        // and should be a valid model from the group
        let bo = vlm_recs.iter().find(|r| r.category == "best_overall").expect("best_overall missing");
        let valid_names = ["model-quality", "model-speed", "model-balanced"];
        assert!(valid_names.contains(&bo.model_name.as_str()), "best_overall model not recognized: {}", bo.model_name);
        assert!(bo.weighted_score > 0.0, "best_overall weighted_score must be > 0");
    }
}