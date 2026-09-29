---
team: baseline
members:
  - name: Shu Wan
    email: 15952765+Shu-Wan@users.noreply.github.com
repository: https://github.com/Abel-ai-lab/nanoTSFM
ai_disclosure: Runs and report prepared with Claude Code under the author's direction.
---

# baseline

`configs/baseline.yaml`, unchanged: a 3.3M-parameter model trained for 5,000 steps (about 2 minutes)
on GEP-M with one A100 80GB. `result.json` is seed 7; seeds 1 and 2 measure the spread.

| Seed | GIFT-Eval CRPS | GIFT-Eval MASE | GEP-Val | GEP-Test | Training |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.665 | 0.956 | 0.625 | 0.622 | 116 s |
| 1 | 0.663 | 0.957 | 0.622 | 0.611 | 107 s |
| 2 | 0.670 | 0.962 | 0.626 | 0.621 | 108 s |
| Mean ± sd | 0.666 ± 0.004 | 0.958 ± 0.003 | 0.624 | 0.618 | 110 s |

Seeds move GIFT-Eval by about ±0.004, so smaller differences are noise.
