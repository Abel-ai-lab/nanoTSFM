---
team: baseline
description: Simplified Toto 2.0, 5,000 steps on GEP-M
members:
  - name: Shu Wan
    github: Shu-Wan
ai_disclosure: Runs and report prepared with Claude Code under the author's direction.
---

# baseline

`configs/baseline.yaml`, unchanged: a 3.3M-parameter model trained for 5,000 steps (about 2 minutes)
on GEP-M with one A100 80GB. The record holds seeds 7 (the config's default), 1 and 2.

| Seed | GIFT-Eval CRPS | GIFT-Eval MASE | GEP-Val | GEP-Test | Training |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.662 | 0.962 | 0.628 | 0.623 | 113 s |
| 1 | 0.676 | 0.971 | 0.630 | 0.618 | 107 s |
| 2 | 0.671 | 0.967 | 0.628 | 0.629 | 106 s |
| $\text{Mean} \pm \text{sd}$ | $0.670 \pm 0.007$ | $0.967 \pm 0.004$ | 0.628 | 0.623 | 109 s |

Seeds move GIFT-Eval by about $\pm 0.007$, so smaller differences are noise.
