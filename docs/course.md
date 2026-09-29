---
name: course
description: Everything specific to the course - schedule, compute credit, final evaluation, grading and awards.
meta:
  type: knowledge
---

# Course

The course runs nanoTSFM as a five-week project. Teams of one or two students improve the baseline
under the [rules](rules.md) and submit one final model each; the organizers evaluate
every submission the same way. The top-ranked team earns an internship opportunity and a
publication opportunity. Items marked *TBA* are announced with the course.

## Schedule

| Week | Work | Checkpoint |
| --- | --- | --- |
| 1 | Set up, run `./run.sh toy`, reproduce the baseline with three seeds | Baseline GIFT-Eval within ±0.01 of 0.666 |
| 2 | Choose a hypothesis from [directions](directions.md); run cheap pilots, reading GEP-Val and GIFT-Eval together | One-paragraph proposal |
| 3 | Compare your change with the baseline | — |
| 4 | Run the ablation; start the report | — |
| 5 | Final runs with three seeds, GIFT-Eval, report, `./run.sh submit` | Submission by the deadline (*TBA*) |

## Compute

Each team receives a $50 Runpod credit code; a typical project uses $20–40 of A100 time. Students
with an ASU Research Computing account can also use the Sol cluster at no cost.

## Hand-in

One member opens the team's submission pull request, as [contributing](../CONTRIBUTING.md)
describes. The team's repository must be public. Deadline: *TBA*.

## Final evaluation

1. The organizers verify every submission by retraining it, as the [review](submission.md#review)
   describes. A submission counts if its retrained score is within 0.01 of the reported one; if it
   fails, its team is contacted once to fix packaging problems.
2. Submissions are ranked by their verified GIFT-Eval relative CRPS; relative MASE breaks ties within
   0.001.

## Awards

The top-ranked team earns:

- **An internship opportunity** with the organizing group (partner and terms *TBA*).
- **A publication opportunity:** co-authoring a write-up of the results.

Rankings decide awards, not grades.

## Grading

| Component | Weight |
| --- | ---: |
| Research idea and hypothesis | 20% |
| Experimental method | 20% |
| Ablation and analysis | 20% |
| Improvement and final score | 20% |
| Report | 10% |
| Reproducibility | 10% |

A rigorous negative result can earn full credit on every component except the final score.

## Rules

- The [rules](rules.md) apply.
- Work alone or in a team of two. The report states what each member did.
- Disclose AI assistance in the report, and cite borrowed code and ideas.
