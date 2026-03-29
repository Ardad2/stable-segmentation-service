Direct-vs-Served Benchmark  backend=clipseg  n=5  warmup=2
  served endpoint: http://127.0.0.1:8004/api/v1/segment
  Building direct runner for 'clipseg'…
  Warming up (2 iterations)…
  Measuring direct   (5 iterations)…
  Measuring served   (5 iterations)…
 Direct vs Served — clipseg / clipseg-text 
                   (ms)                    
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Metric     ┃ Direct ┃ Served ┃ Overhead ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ min (ms)   │ 209.07 │ 208.25 │    -0.82 │
│ p50 (ms)   │ 230.71 │ 218.10 │   -12.61 │
│ p95 (ms)   │ 230.89 │ 222.59 │    -8.30 │
│ p99 (ms)   │ 230.89 │ 222.59 │    -8.30 │
│ max (ms)   │ 249.35 │ 226.54 │   -22.81 │
│ mean (ms)  │ 227.55 │ 216.96 │   -10.59 │
│ stdev (ms) │  15.28 │   8.05 │    -7.24 │
└────────────┴────────┴────────┴──────────┘
Service overhead (p50):  -12.61 ms  (-5.5%)
