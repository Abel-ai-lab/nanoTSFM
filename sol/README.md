# Running on Sol

One-time setup on a Sol login node:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs uv into ~/.local/bin
git clone <our fork> nanoTSFM && cd nanoTSFM && git checkout track-a-data
export HF_HOME=/scratch/$USER/hf TORCH=cuda
./run.sh setup && ./run.sh data                            # 730 MB into /scratch
uv run --locked --extra gift --extra cuda python -c "from nanotsfm.data import load; load('GEP-L')"  # GEP-L, 6.2 GB, once
```

Jobs (each trains, then scores GEP-Test and GIFT-Eval; logs in `logs/`, `grep RESULT logs/*`):

```shell
sol/pilots.sh reproduce   # record 2 (configs/series.yaml), seeds 7, 1, 2: expect 0.633 +- 0.01
sol/pilots.sh pilots      # seed 7 of every configs/track-a/*.yaml
sol/pilots.sh seeds sp1.5 # seeds 1 and 2 of a promising config
```

Track A configs (all build on record 2, `source_power: 1`, GEP-M, 5,000 steps unless noted):

| Config | Change | Hypothesis |
| --- | --- | --- |
| `sp1.5`, `sp2` | `source_power` 1.5, 2 | More weight on large sources kept helping (0 to 0.5 to 1) |
| `sp1-fp0.5`, `sp1-fp0` | `freq_power` 0.5, 0 | Balancing frequency groups lifts thin ones (second, month+) |
| `sp1-gepm-20k` | 20,000 steps | Control for the next row |
| `sp1-gepl-20k` | GEP-L, 20,000 steps | More series keep longer training's gains on GIFT-Eval |
