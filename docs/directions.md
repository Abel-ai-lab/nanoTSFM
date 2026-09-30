---
name: directions
description: Improvement directions by pipeline stage, with pilot results for each.
meta:
  type: knowledge
---

# Directions for improvement

Every stage except evaluation is open. In the pilot, more steps and a bigger model improved GEP-Test
but hurt GIFT-Eval, while data diversity, source weighting and longer context improved both. Numbers
are relative CRPS (lower is better); the baseline scores 0.670 on GIFT-Eval and 0.623 on GEP-Test.

## Data

Data choices moved GIFT-Eval most and are the main defense against overfitting the corpus.

- **Mixture:** sampling sources by the square root of their series count, in place of uniformly,
  scored 0.644 at 5,000 steps against 0.665 for uniform sampling. Per-frequency or per-domain balance is untested.
- **Size and diversity:** more series let longer training keep its transfer: at 20,000 steps,
  GEP-L scored 0.661 against 0.693 for GEP-M (see [data](data.md#training-slices)). The full
  corpus is 975 GB.
- **Coverage:** longer training hurt GIFT-Eval most where the corpus is thin: yearly, 10- and
  15-minute, and monthly data, and the Econ/Fin domain. Resampling those sources can fill the
  gaps.
- **Windows:** crop length and position, and how wide multivariate series are subsampled.
- **Duplicates:** some series appear under two sources, for example related PEMS datasets, which
  the split by name cannot see; removing them may change what the model learns.

## Model

The baseline follows Toto 2.0's layout but not all its details; each missing detail is a change with
a published reference. Capacity alone overfits: 12.7M parameters improved GEP-Test (0.538
against 0.579) but not GIFT-Eval (0.725 against 0.705), on GEP-70M at 20,000 steps.

- **Size and shape:** width, depth and heads, together with data and regularization.
- **Scaling:** a causal running mean and standard deviation per patch, then asinh. Alternatives:
  robust statistics, a scale floor relative to the level, backfilling the first patches (Toto 2.0).
- **Patching:** fixed 32-step patches. Alternatives: patch size by frequency, several sizes at once.
- **Attention layout:** three causal time layers, then one variate layer. Alternatives: interleave
  the variate layer, drop it, or mask all-missing padding patches out of attention (Toto 2.0).
- **Blocks and positions:** GELU MLP and full rotary encoding. Alternatives: SwiGLU, QK-norm,
  xPos (Toto 2.0), ALiBi.
- **Output head:** one linear layer to $32 \times 9$ quantiles. Alternatives: a residual MLP head, more
  quantile levels, a mixture distribution (Toto 1.0).
- **Backbones:** recurrent models such as xLSTM (TiRex) or Mamba; sparse experts (Moirai-MoE).

## Context and inference

A longer context helped both scores: 2,048 steps scored 0.673 on GIFT-Eval against 0.683 for 1,024
(20,000 steps on GEP-500M), at 1.5 times the training cost.

- **Long horizons:** a longer one-pass horizon, or block decoding instead of median rollout beyond
  512 steps; medium and long GIFT-Eval terms are the baseline's weakest.
- **Short series:** forward-fill gaps and give short series a scale from their first observations.
- **Post-processing:** clip quantiles to a band around the context, or average several contexts.

## Training objective

- **Masking:** runs of up to 8 hidden patches covering up to 40% of a series. Toto 2.0 uses up to 16;
  inference hides 16 in a row.
- **Which positions count:** every position (baseline) or only hidden ones (Toto 2.0).
- **Loss:** pinball loss on asinh-scaled values. Alternatives: loss in original units scaled per
  series (closer to the CRPS score), more quantile levels, auxiliary reconstruction or latent losses.

## Optimization and compute

The baseline uses 2 of its 60 minutes, and spending the rest is not free: plain longer training
improved GEP-Test (0.618 to 0.581 at 40,000 steps) and worsened GIFT-Eval (0.666 to 0.702).

- **Spending the budget:** steps, batch size, model size, context and data compete for the same
  hour; the baseline trains at 46 steps per second on an A100 with the GPU 96% busy.
- **Regularization and schedule:** weight decay, dropout, weight averaging, or stopping on a
  transfer signal. AdamW at $10^{-3}$ with cosine decay is the baseline; Muon or NorMuon (Toto 2.0) and
  warmup-stable-decay are alternatives.
- **Throughput:** `torch.compile`, fewer padding tokens (short series are left-padded to 1,536
  steps), packing short series end to end.

## Agents and recursive self-improvement

nanoTSFM is also a small environment for recursive self-improvement, where an AI system improves the
training of another model. One iteration takes minutes and ends in one verified number: the baseline
trains in 2 minutes and GIFT-Eval scores it in about 5. An agent can run the loop that
[autoresearch](https://github.com/karpathy/autoresearch) runs on nanochat: change the code, train,
score, keep or discard. The
[Automated LLM Speedrunning Benchmark](https://arxiv.org/abs/2506.22419) tests agents on
modded-nanogpt's records in the same way. A submission made with an agent follows the same rules,
and its `ai_disclosure` field says how the agent was used.

## Pilot runs

| Change | Setting | GIFT-Eval | GEP-Test | Compared with (GIFT-Eval, GEP-Test) |
| --- | --- | --- | --- | --- |
| $\sqrt{\text{series count}}$ sampling | GEP-M, 5,000 steps | 0.644 | 0.607 | 0.665, 0.622 uniform |
| GEP-L | 20,000 steps | 0.661 | 0.587 | 0.693, 0.586 on GEP-M |
| Context 2,048 | GEP-500M, 20,000 steps | 0.673 | 0.568 | 0.683, 0.594 at 1,024 |
| 20,000 steps | GEP-M | 0.693 | 0.586 | 0.666, 0.618 at 5,000 |
| 40,000 steps | GEP-M | 0.702 | 0.581 | 0.666, 0.618 at 5,000 |
| 12.7M parameters | GEP-70M, 20,000 steps | 0.725 | 0.538 | 0.705, 0.579 at 3.3M |
| Context 512 | GEP-70M, 20,000 steps | 0.712 | 0.634 | 0.705, 0.579 at 1,024 |

Seed spread in the pilot was about $\pm 0.004$ on GIFT-Eval and $\pm 0.006$ on GEP-Test. GEP-70M and GEP-500M were pilot
slices; the dataset's `build/build.py` rebuilds them.
