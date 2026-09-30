---
name: todo
description: Open work on nanoTSFM itself - launch, rules and data.
meta:
  type: knowledge
---

# To do

Open work, most useful first.

## Launch

- [ ] Announce the internship's partner and terms on the [course](course.md) page.
- [x] Test the Runpod credit code end to end, and provide a ready-to-train template, notebook and
  measured costs in [Run on Runpod](course.md#run-on-runpod).

## Rules

Open questions that [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) answers for
its own record.

- [ ] Keep notable submissions that break a rule or miss the record in a separate list, so good ideas
  are not lost.
- [ ] Record rulings on edge cases in one place, a FAQ in the [rules](rules.md).
- [ ] Announce each record in a short post that credits its contributors.

## Data

- [ ] Detect near-duplicate series stored under two sources (for example related PEMS datasets),
  which the name-based split cannot see.

## Background

The numbers in these pages come from about 50 pilot runs on A100 80GB GPUs in September 2026:
5,000–40,000 steps, six data slices (GEP-S, GEP-M and GEP-L among them), contexts of 512–2,048,
3.3M and 12.7M parameters and three sampling schemes, each scored on GIFT-Eval, GEP-Val and
GEP-Test.
