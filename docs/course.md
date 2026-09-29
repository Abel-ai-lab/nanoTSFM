---
name: course
description: Everything specific to the course - schedule, compute credit, final evaluation, grading and awards.
meta:
  type: knowledge
---

# Course

The course runs nanoTSFM as a five-week project. Teams of one or two students try to set the
[record](submission.md) under the [rules](rules.md), and the organizers verify every submission the
same way. Teams earn points for beating the course bar and for each record they set. The
top-ranked team earns an internship opportunity and a publication opportunity. Items marked *TBA*
are announced with the course.

## Schedule

| Week | Work | Checkpoint |
| --- | --- | --- |
| 1 | Set up, run `./run.sh toy`, reproduce the current record with three seeds | Its GIFT-Eval within ±0.01 of the record |
| 2 | Choose a hypothesis from [directions](directions.md); run cheap pilots, reading GEP-Val and GIFT-Eval together | One-paragraph proposal |
| 3 | Compare your change with the record | — |
| 4 | Run the ablation; start the report | — |
| 5 | Final runs with three seeds, GIFT-Eval, report, `./run.sh submit` | Pull request by the deadline (*TBA*) |

## Compute

Each team receives a $50 Runpod credit code; a typical project uses $20–40 of A100 time. Students
with an ASU Research Computing account can also use the Sol cluster at no cost.

## Hand-in

One member opens the team's pull request, as [submission](submission.md) describes, whether or not
it beats the record. Teams may open submissions during the course, one at a time, until the deadline
(*TBA*). The team's repository must be public.

## Final evaluation

1. The organizers verify each submission by retraining every run, as the
   [review](submission.md#review) describes. Runs count if their retrained scores are within 0.01
   of the reported ones; if they fail, the team is contacted once to fix packaging problems.
2. Points:
   - **1 point** if any of the team's submissions has a lower verified mean GIFT-Eval CRPS than the
     course bar: record 2, `Shu-Wan`, at 0.633.
   - **1 point** for each record the team sets during the course, under the
     [record rule](submission.md#the-record-rule).
3. Teams are ranked by points, then by the verified mean CRPS of their best submission, then by its
   mean MASE.

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
| Points and final score | 20% |
| Report | 10% |
| Reproducibility | 10% |

A rigorous negative result can earn full credit on every component except points and score.

## Rules

- The [rules](rules.md) apply.
- Work alone or in a team of two. The report states what each member did.
- Disclose AI assistance in the report, and cite borrowed code and ideas.
