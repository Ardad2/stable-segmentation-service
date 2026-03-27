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
│ min (ms)   │ 219.87 │ 214.78 │    -5.09 │
│ p50 (ms)   │ 220.77 │ 223.11 │    +2.34 │
│ p95 (ms)   │ 220.77 │ 223.11 │    +2.34 │
│ p99 (ms)   │ 220.77 │ 223.11 │    +2.34 │
│ max (ms)   │ 225.41 │ 229.29 │    +3.88 │
│ mean (ms)  │ 222.02 │ 222.39 │    +0.38 │
│ stdev (ms) │   2.97 │   7.28 │    +4.31 │
└────────────┴────────┴────────┴──────────┘
Service overhead (p50):  +2.34 ms  (+1.1%)
