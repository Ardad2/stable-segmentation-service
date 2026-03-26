Direct-vs-Served Benchmark  backend=mock  n=20  warmup=3
  served endpoint: http://127.0.0.1:8002/api/v1/segment
  Building direct runner for 'mock'…
  Warming up (3 iterations)…
  Measuring direct   (20 iterations)…
  Measuring served   (20 iterations)…
 Direct vs Served — mock / mock-point (ms) 
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Metric     ┃ Direct ┃ Served ┃ Overhead ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ min (ms)   │   0.04 │   0.16 │    +0.12 │
│ p50 (ms)   │   0.04 │   0.19 │    +0.15 │
│ p95 (ms)   │   0.06 │   0.29 │    +0.22 │
│ p99 (ms)   │   0.06 │   0.29 │    +0.22 │
│ max (ms)   │   0.18 │   0.52 │    +0.34 │
│ mean (ms)  │   0.05 │   0.21 │    +0.16 │
│ stdev (ms) │   0.03 │   0.08 │    +0.05 │
└────────────┴────────┴────────┴──────────┘
Service overhead (p50):  +0.15 ms  (+379.5%)
