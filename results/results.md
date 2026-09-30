## Quality (SciFact test, 300 queries)

| config | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|:--|--:|--:|--:|--:|--:|
| BM25 | 0.546 | 0.743 | 0.812 | 0.653 | 0.686 |
| Dense (BGE-small) | 0.578 | 0.761 | 0.838 | 0.684 | 0.715 |
| Hybrid RRF | 0.592 | 0.798 | 0.860 | 0.705 | 0.737 |
| Hybrid + rerank top 10 | 0.567 | 0.771 | 0.860 | 0.678 | 0.716 |
| Hybrid + rerank top 20 | 0.564 | 0.763 | 0.838 | 0.673 | 0.706 |
| Hybrid + rerank top 30 | 0.564 | 0.759 | 0.829 | 0.672 | 0.704 |

## Latency, one request at a time (client-side ms; stage columns are server-side p50)

| config | p50_ms | p95_ms | mean_ms | embed_p50_ms | retrieve_p50_ms | rerank_p50_ms |
|:--|--:|--:|--:|--:|--:|--:|
| BM25 | 22.4 | 44.6 | 26.5 | 16.2 | 3.100 | 0.000 |
| Dense (BGE-small) | 20.6 | 35.4 | 22.5 | 14.4 | 3.600 | 0.000 |
| Hybrid RRF | 25.3 | 35.0 | 26.7 | 14.2 | 8.500 | 0.000 |
| Hybrid + rerank top 10 | 612.7 | 799.2 | 624.5 | 17.4 | 11.9 | 577.6 |
| Hybrid + rerank top 20 | 1301.5 | 1834.3 | 1344.8 | 20.4 | 17.2 | 1250.8 |
| Hybrid + rerank top 30 | 2353.3 | 4193.4 | 2594.0 | 32.8 | 30.3 | 2275.8 |

## Load test

| config | concurrency | p50_ms | p95_ms | mean_ms | qps |
|:--|--:|--:|--:|--:|--:|
| Hybrid RRF | 4 | 89.1 | 180.1 | 99.3 | 39.7 |
| Hybrid + rerank top 20 | 4 | 6824.7 | 10597.8 | 7204.0 | 0.552 |

## Significance (paired randomization test on per-query nDCG@10)

| A | B | ndcg@10 A - B | 95% CI | p-value | significant |
|:--|--:|--:|--:|--:|--:|
| Hybrid RRF | Dense (BGE-small) | 0.022 | [+0.002, +0.042] | 0.033 | yes |
| Hybrid RRF | BM25 | 0.051 | [+0.029, +0.073] | 0.000 | yes |
| Hybrid + rerank top 10 | Hybrid RRF | -0.021 | [-0.042, -0.001] | 0.043 | yes |
| Hybrid + rerank top 20 | Hybrid RRF | -0.031 | [-0.057, -0.005] | 0.022 | yes |
| Hybrid + rerank top 30 | Hybrid RRF | -0.034 | [-0.060, -0.007] | 0.015 | yes |
| Hybrid + rerank top 20 | Hybrid + rerank top 10 | -0.009 | [-0.021, +0.003] | 0.132 | no |
| Hybrid + rerank top 30 | Hybrid + rerank top 20 | -0.003 | [-0.008, +0.003] | 0.372 | no |
