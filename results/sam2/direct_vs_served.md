Direct-vs-Served Benchmark  backend=sam2  n=3  warmup=1
  served endpoint: http://127.0.0.1:8003/api/v1/segment
  Building direct runner for 'sam2'…
  Warming up (1 iterations)…
  Measuring direct   (3 iterations)…
  Measuring served   (3 iterations)…
  Direct vs Served — sam2 / sam2-point (ms)  
┏━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric     ┃  Direct ┃  Served ┃ Overhead ┃
┡━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ min (ms)   │ 3762.73 │ 3786.43 │   +23.70 │
│ p50 (ms)   │ 3777.33 │ 3797.72 │   +20.39 │
│ p95 (ms)   │ 3777.33 │ 3797.72 │   +20.39 │
│ p99 (ms)   │ 3777.33 │ 3797.72 │   +20.39 │
│ max (ms)   │ 3829.65 │ 3828.38 │    -1.27 │
│ mean (ms)  │ 3789.90 │ 3804.18 │   +14.28 │
│ stdev (ms) │   35.19 │   21.71 │   -13.48 │
└────────────┴─────────┴─────────┴──────────┘
Service overhead (p50):  +20.39 ms  (+0.5%)
