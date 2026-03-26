# Evaluation Summary

**Run ID:** run-1774555503  
**Frames processed:** 35  

## VLM Models

| Model | Calls | Errors | Reliability | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Cost/Call | Total Cost | Avg Quality |
|-------|-------|--------|-------------|----------|----------|----------|-----------|-----------|------------|-------------|
| google/gemini-2.0-flash-lite-001 | 7 | 0 | 100.0% | 2298 | 4583 | 4583 | 2665 | $0.00020 | $0.0014 | 4.86/5 |
| reka/reka-edge | 7 | 0 | 100.0% | 1868 | 4392 | 4392 | 2518 | $0.00100 | $0.0070 | 3.86/5 |
| meta-llama/llama-3.2-11b-vision-instruct | 7 | 0 | 100.0% | 1057 | 3667 | 3667 | 1620 | $0.00005 | $0.0003 | 3.86/5 |

## Detection Models

| Model | Calls | Errors | Reliability | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Cost/Call | Total Cost | MAE | Precision | Recall | F1 |
|-------|-------|--------|-------------|----------|----------|----------|-----------|-----------|------------|-----|-----------|--------|-----|
| google/gemini-2.0-flash-lite-001 | 7 | 0 | 100.0% | 3039 | 5233 | 5233 | 3214 | $0.00020 | $0.0014 | 0.500 | 1.000 | 1.000 | 1.000 |
| reka/reka-edge | 7 | 0 | 100.0% | 1958 | 7775 | 7775 | 3106 | $0.00100 | $0.0070 | 1.000 | 0.500 | 1.000 | 0.667 |

