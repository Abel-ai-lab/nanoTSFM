---
name: course
description: Everything specific to the course - schedule, compute credit, final evaluation, grading and awards.
meta:
  type: knowledge
---

# Course

nanoTSFM is a Project II option in CSE 472: Social Media Mining (Fall 2026, Prof. Huan Liu; TA Ali
Beigi), offered with the DMML lab. Teams of one or two students try to set the
[record](submission.md) under the [rules](rules.md), and the organizers verify every submission the
same way. Teams earn points for beating the course bar and for each record they set. The
top-ranked team earns an internship opportunity and a publication opportunity.

When this page and the course syllabus disagree, follow the syllabus.

## Dates

| Date | What |
| --- | --- |
| September 29, 2026 | Project II is released |
| October 7, 2026 | Choose a teammate and this project, as the syllabus describes |
| November 23, 2026, 11:59 PM | Deadline: the team's pull request is open |

Times are Arizona time. There are no extensions.

## Schedule

The project is planned for about two-thirds of the eight weeks; the last two are a buffer.

| Week | Dates | Work | Checkpoint |
| --- | --- | --- | --- |
| 1 | Sep 29 – Oct 5 | Find a teammate; set up and run `./run.sh toy` | Team and project chosen by October 7 |
| 2 | Oct 6 – Oct 12 | Reproduce the current record with three repeated runs | Its GIFT-Eval within $\pm 0.01$ of the record |
| 3 | Oct 13 – Oct 19 | Choose a hypothesis from [directions](directions.md); run cheap pilots, reading GEP-Val and GIFT-Eval together | One-paragraph proposal |
| 4 | Oct 20 – Oct 26 | Implement the change and compare it with the record | — |
| 5 | Oct 27 – Nov 2 | Run the ablation; start the report | — |
| 6 | Nov 3 – Nov 9 | Three repeated final runs, GIFT-Eval, report, `./run.sh submit` | A first pull request |
| 7–8 | Nov 10 – Nov 23 | Fix what the checks and the review find; improve and submit again | Pull request by November 23, 11:59 PM |

## Compute

Each team receives a \$50 Runpod credit code; a typical project uses \$20–40 of A100 time. Students
with an ASU Research Computing account can also use the Sol cluster at no cost. Teams that need
more compute can ask for it; each request is reviewed.

## Run on Runpod

### Before you start

1. [Runpod](https://www.runpod.io) rents cloud GPUs; [create an account](https://runpod.io?ref=j5jduepq). [Runpod docs](https://docs.runpod.io/).
2. Send your Runpod account email to the course project mentor (email provided in class) and accept
   the invitation to the nanoTSFM team; see the
   [account guide](https://docs.runpod.io/accounts-billing/manage-accounts).
3. In your personal account, open **Billing -> Credit codes** and use **Redeem code** to redeem the
   course code; check that your balance increased. Join the nanoTSFM team before redeeming.
4. You join nanoTSFM with the **Basic** role, which cannot create team pods; create pods in your
   **personal account**, where the redeemed credit pays for them.
5. For a two-person project team, one member redeems the code, then
   [converts that personal account to a team account](https://docs.runpod.io/accounts-billing/manage-accounts#convert-to-a-team-account)
   and invites the collaborator with the **Admin** role so both members can create pods there.

### Deploy a pod

Deploy [manually](https://docs.runpod.io/get-started) or with a coding agent using the
[official Runpod skills](https://github.com/runpod/runpod-plugins-official).

Use the [nanoTSFM template](https://console.runpod.io/hub/template/2zp46nns9i) with one **A100 80GB**,
at least 8 CPU cores, and on-demand pricing; check the hourly price before deploying.

Wait for setup to finish in the pod logs, then connect to the pod. For an optional walkthrough,
open **Connect -> Jupyter Lab**, open `nanoTSFM/runpod.ipynb`, and select **nanoTSFM (Python 3.13)**.

If you prefer to work in the terminal, use any of Runpod's
[connection options](https://docs.runpod.io/pods/connect-to-a-pod): web terminal, SSH, JupyterLab,
VS Code or Cursor. Coding agents can follow the [agent setup](https://docs.runpod.io/agent-setup).
Then run:

```shell
cd /workspace/nanoTSFM
./run.sh toy
./run.sh data
./run.sh train baseline
./run.sh eval baseline
```

Measured September 30, 2026 on one Secure Cloud A100 80GB PCIe in EU-RO-1, with 16 allocated
vCPUs, Python 3.13.15 and PyTorch 2.11.0+cu128, at \$1.59/hour:

| Command | Wall time | Compute cost |
| --- | ---: | ---: |
| `./run.sh toy` | 151 s | \$0.067 |
| `./run.sh data` | 99 s | \$0.044 |
| `./run.sh train baseline` | 283 s | \$0.125 |
| `./run.sh eval baseline` | 362 s | \$0.160 |

The four commands took 14.9 minutes and cost about **\$0.40 in compute**, after environment setup.
Costs are measured wall seconds times \$1.59/3,600, excluding storage, pod startup and idle time;
they are not an invoice total. Prices and download speeds vary. These first-run timings include
imports and dataset downloads; training includes GEP-Val, while the 5,000 GPU training steps alone
took 136.7 seconds. Allow several extra minutes for the initial Python/CUDA environment install.

### Save your results

Commit the small record folder (`README.md` and `result.json`) and push it to your fork, as
[submission](submission.md) describes. Keep run folders out of Git: the measured baseline run was
about 40 MB, mostly its checkpoint, and submissions reject files over 1 MB. Download any runs you
want to keep through JupyterLab, `runpodctl send` or `scp`; see
[transferring files](https://docs.runpod.io/pods/storage/transfer-files).

**Terminate the pod when finished** to end GPU and pod-storage charges. Download your files first:
termination deletes this template's pod storage. Stopping retains paid storage. Setup, downloads
and idle time are billed while the pod runs.

For help, open an [issue](https://github.com/Abel-ai-lab/nanoTSFM/issues) or contact the course
project mentor (email provided in class).

## Hand-in

One member opens the team's pull request, as [submission](submission.md) describes, whether or not
it beats the record. Teams may open submissions during the course, one at a time, until
November 23, 2026, 11:59 PM. The team's repository must be public. Hand in anything else the
syllabus asks for through the course's own channel.

## Final evaluation

1. The organizers verify each submission by retraining it three times with new seeds, as the
   [review](submission.md#review) describes; the mean of those runs is its verified score. If a
   retrain fails, the team is contacted once to fix packaging problems.
2. Points:
   - **1 point** if any of the team's submissions has a lower verified mean GIFT-Eval CRPS than the
     course bar: record 2, `Shu-Wan`, at 0.632.
   - **1 point** for each record the team sets during the course, under the
     [record rule](submission.md#the-record-rule).
3. Teams are ranked by points, then by the verified mean CRPS of their best submission, then by its
   mean MASE.

## Awards

The top-ranked team earns:

- **An internship opportunity** with the organizing group; its partner and terms will be announced.
- **A publication opportunity:** co-authoring a write-up of the results.

Rankings decide awards, not grades.

## Grading

| Component | Weight | What earns it |
| --- | ---: | --- |
| Research idea and hypothesis | 20% | A clear, testable hypothesis, grounded in the pilot results or the literature |
| Experimental method | 20% | Three repeated runs for every configuration compared, and a fair comparison with the record built on |
| Ablation and analysis | 20% | An ablation that isolates the change, and where the change helps or hurts |
| Result | 20% | 10% for a submission that passes verification, and 10% more if it beats the course bar |
| Report | 10% | Follows the template; every number comes from a run |
| Reproducibility | 10% | Retrained from the public repository with new seeds, the runs match the reported mean within run-to-run noise |

Each record a team sets adds 2% of extra credit, up to 10%. A rigorous negative result can earn full
credit on every component except the result.

## Rules

- The [rules](rules.md) apply.
- Work alone or in a team of two. The report states what each member did.
- Disclose AI assistance in the report, and cite borrowed code and ideas.
