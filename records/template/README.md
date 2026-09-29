---
team: YOUR_TEAM
description: YOUR_CHANGE  # the change in a few words, for the record table
members:
  - name: YOUR_NAME
    github: YOUR_GITHUB
  - name: YOUR_PARTNER_NAME  # delete this member if you work alone
    github: YOUR_PARTNER_GITHUB
ai_disclosure: YOUR_AI_USE  # any AI assistance and how you checked it, or none
---

# YOUR_TEAM

About four pages. Every number should come from a run.

## Hypothesis

What did you change, why should it lower GIFT-Eval within the one-hour budget, and what does your
ablation test? Name the record you built on.

## Method

The model, objective, data and sampling, configuration, parameter count, training time, and seeds.
Explain how you chose your final model and which scores you looked at.

## Results

A table of the record you built on, your change and the ablation: GIFT-Eval (mean and spread over
every seed you ran), GEP-Val or GEP-Test, training seconds, and steps. Where the two disagree, say
what you think fits the corpus without transferring.

## Discussion

Did the results support the hypothesis? Where does the change help or hurt (for example by
frequency or horizon)? What would you try next, and what are the limitations?

## Contributions and attribution

Each member's contributions, borrowed code or ideas with references, and any AI assistance.
