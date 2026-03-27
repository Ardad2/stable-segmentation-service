Direct-vs-Served Benchmark  backend=clipseg  n=3  warmup=1
  served endpoint: http://127.0.0.1:8004/api/v1/segment
  Building direct runner for 'clipseg'…
  Warming up (1 iterations)…
  Measuring direct   (3 iterations)…
  Measuring served   (3 iterations)…
 Direct vs Served — clipseg / clipseg-text 
                   (ms)                    
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Metric     ┃ Direct ┃ Served ┃ Overhead ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ min (ms)   │ 225.86 │ 215.02 │   -10.85 │
│ p50 (ms)   │ 227.89 │ 220.34 │    -7.55 │
│ p95 (ms)   │ 227.89 │ 220.34 │    -7.55 │
│ p99 (ms)   │ 227.89 │ 220.34 │    -7.55 │
│ max (ms)   │ 230.94 │ 237.63 │    +6.69 │
│ mean (ms)  │ 228.23 │ 224.33 │    -3.90 │
│ stdev (ms) │   2.56 │  11.82 │    +9.27 │
└────────────┴────────┴────────┴──────────┘
Service overhead (p50):  -7.55 ms  (-3.3%)
