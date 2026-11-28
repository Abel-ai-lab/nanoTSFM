# Period-locked phase history and CDF pooling

This research snapshot reproduces the training recipe of a 12-layer, 23,350,176-parameter model
and integrates its long-history inference into nanoTSFM's forecast interface. It uses one
checkpoint for every scale and view. The GIFT task manifest, evaluator and scorer retain their
upstream versions.

## Training

The candidate starts from random weights, samples GEP-L at revision
`b8eee2fc1f7d71cf4d40001c39ac69e1af0d8cf7`, and draws each source proportional to the square
root of its series count. Training uses at most one channel per window. A quarter of the
windows use stride-2 or stride-4 phases. The other window transforms are sign flip, stride,
block mean, offset and rescale of the same series. The configuration fixes their probabilities.

The network uses 32-point patches and causal patch-level normalization. It fits nine sorted
quantiles with pinball loss in asinh-normalized coordinates. The first 42,972 steps use 512-point
contexts and the next 32,000 use 1,024. AdamW uses a 5% warmup followed by cosine decay to 10%
of peak learning rate. The short-context curriculum reduces the cost of the early steps.

## Forecast

The native network reads its latest 1,024 points. The period detector reads up to 4,096 recent
points to choose the phase stride. For an eligible stride above four, the first forecast block
can look further back to fill up to 1,024 samples per phase. The phase cap leaves the maximum
visible history at 12,288 points. The period, transforms, and eligibility come from the observed
input. Later blocks use the ordinary context and feed back the original quantile blend's median.

For the published quantiles, let the native and interlaced phase forecasts define two CDFs
$F_0$ and $F_s$ by piecewise-linear interpolation between deciles. The outer deciles also bound
the support, with the unreported tails assigned to the endpoints. The emitted deciles use the
generalized inverse

$$
q_\tau = \inf\{x : (1-\alpha)F_0(x) + \alpha F_s(x) \ge \tau\},
\qquad \alpha=0.25.
$$

If no phase is available, the emitted forecast equals the native forecast. CDF pooling retains
the mixture of the two predictive distributions when their centers disagree. Original,
sign-flipped and eligible square-root views share the same network and use the usual view
aggregation. The native first block is reused across the long rollout and view aggregation.

## Archived comparison

These are historical full 97-task GIFT-Eval scores on one seed-7 checkpoint. Lower is better.

| Inference on the same checkpoint | Relative CRPS |
| --- | ---: |
| Original quantile averaging | 0.522491 |
| CDF pooling with the original history | 0.519860 |
| CDF pooling with first-block locked-phase history | 0.518834 |

The last row has relative MASE 0.759713. The checkpoint finished 74,972 training steps in
3,050.41 seconds on a single A100 80 GB. Its SHA-256 is
`659e492522d40010a47c9124045bf580e57909e6f5bd6eb38cec7246a8a13217`.
The archived score report has SHA-256
`e54223d64ce653f3a5219aa01dbf1fb1d016b6c0819900d49bcad566e294e169`.
That forecast run used selective recomputation from same-checkpoint predictions for unaffected
rows. Treat these scores as an archived single-seed comparison. A new full score and three
same-commit runs are needed for a leaderboard record.

On the archived checkpoint, the integrated forecast matched the archived implementation
bit for bit in four CPU fixtures covering grouped channels, missing values, short and extended
histories, positive and signed series, and horizons crossing the first block.

## Reproduce on a RunPod A100

This branch enables CUDA Graph training. The DataLoader joins each batch before pinning it.
The training thread pins the tensors ahead of capture, so the DataLoader workers only run CPU
work while capture is active.
The optimizer, learning-rate schedule and the random mask draw run on each normal training
step. Forward, backward and gradient clipping replay on the fixed input buffers. Capture
resets when the active context or tensor shape changes.

In a from-scratch, batch-256 A100 comparison using the production GEP-L DataLoader, both paths
completed the same 192 seed-1 updates, split evenly between 512 and 1,024 points. Recorded
training time fell from 15.50 to 8.54 seconds. Model tensors, AdamW moments, CPU and CUDA RNG states,
and terminal validation loss matched exactly. Peak PyTorch allocation was 3.36 and 3.40 GiB.
The 97-task scores in the archived comparison above come from the earlier seed-7 checkpoint.

Set the artifact and Hugging Face cache directories to your RunPod project volume. Run from the
repository root.

```bash
export TORCH=cuda
runpod_volume=/workspace
export RUNS="$runpod_volume/runs"
export HF_HOME="$runpod_volume/hf-cache"
./run.sh setup
./run.sh train locked-s7 configs/locked-history-a100.yaml 7
./run.sh test locked-s7
./run.sh eval locked-s7
```

For lower evaluation concurrency, invoke the same fixed modules directly with
`--workers 1` for forecasting and a suitable CPU worker count for scoring. The predictions are
computed afresh from the selected checkpoint. The upstream GIFT-Eval code revision is
`9a014e9e8ea130ba39c100c60d5dcbab7db57ac9` and the data revision is
`30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`.
