# Final Evaluation Summary

| Backend | Supported Prompts | Compatibility | Correctness | Latency (p50) | Throughput (req/s) | Direct vs Served (p50) |
|--------|-------------------|--------------|-------------|---------------|--------------------|------------------------|
| mock | point, box, text | PASS | PASS | 0.17 ms | 2070.17 req/s | +0.15 ms |
| sam2 | point, box | PASS | PASS | 3798.05 ms | 0.26 req/s | +20.39 ms |
| clipseg | text | PASS | PASS | 224.77 ms | 4.48 req/s | -12.61 ms* |

## Notes
- mock used synthetic stub masks for development/testing.
- sam2 was evaluated on CPU because the Spark environment had CPU-only PyTorch.
- clipseg was evaluated on CPU.
- versioning was evaluated separately via the v1/v2 compatibility scripts.
- *Negative direct-vs-served overhead for clipseg at this small sample size should be treated as measurement noise / negligible overhead, not proof that the service is inherently faster than direct invocation.
