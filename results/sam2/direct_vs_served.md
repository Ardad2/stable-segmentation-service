Direct-vs-Served Benchmark  backend=sam2  n=5  warmup=1
  served endpoint: http://127.0.0.1:8003/api/v1/segment
  Building direct runner for 'sam2'…
  Warming up (1 iterations)…
  Measuring direct   (5 iterations)…
  Measuring served   (5 iterations)…
  Direct vs Served — sam2 / sam2-point (ms)  
┏━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric     ┃  Direct ┃  Served ┃ Overhead ┃
┡━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ min (ms)   │ 3742.81 │ 3827.73 │   +84.93 │
│ p50 (ms)   │ 3780.00 │ 3843.49 │   +63.49 │
│ p95 (ms)   │ 3798.99 │ 3853.88 │   +54.89 │
│ p99 (ms)   │ 3798.99 │ 3853.88 │   +54.89 │
│ max (ms)   │ 3807.16 │ 3876.32 │   +69.16 │
│ mean (ms)  │ 3777.59 │ 3847.67 │   +70.08 │
│ stdev (ms) │   26.90 │   18.64 │    -8.26 │
└────────────┴─────────┴─────────┴──────────┘
Service overhead (p50):  +63.49 ms  (+1.7%)
