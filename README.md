# Low-rank & spectral regularisation for small-data machine-sound anomaly detection

Code for the study described in `task.md`: do nuclear-norm and spectral-norm regularisers improve
generalisation when only a small fraction of normal training clips is available, and what do they do
to the singular-value spectrum of the embeddings?

> **Status: untested on real DCASE data.** The pure-numpy parts (spectrum metrics, scoring, DCASE-style
> metrics) have unit tests. The PyTorch pipeline was written without access to torch in the authoring
> environment, so run the smoke test below first and expect to fix small things.

## Install
```bash
pip install -r requirements.txt
```

## 1. Smoke test (synthetic data, a minute on CPU)
```bash
pytest -q                                   # unit tests + end-to-end smoke test
# or manually:
python -m scripts.make_synthetic_data --out data/synthetic
python -m src.train --config configs/smoke.yaml
```

## 2. Real data
1. Download the DCASE 2026 Task 2 dev set from the official task page and put it at
   `data/dcase2026_dev/<machine>/{train,test}/*.wav`.
2. **Check the file naming and channel layout against `src/data.py`** (see its docstring). The 2026 data
   is two-channel (near/far mic); if channels are separate files or names differ, edit `parse_filename`,
   `scan_machine` and `AudioDataset._load` only. If test labels are not in the filenames, pass a CSV via
   `data.test_labels_csv`.
3. Baseline sanity check on one machine: it should land in a sensible range relative to the official
   baseline before you trust any comparison.
```bash
python -m src.train --config configs/baseline_strong.yaml --set data.machine=fan data.fraction=0.1
```

## 3. Pilot (do this first)
One machine, 10% data, 5 seeds, baseline vs nuclear norm on embeddings:
```bash
python -m scripts.run_grid --exps baseline_strong nuclear_emb --machines fan \
    --fractions 0.1 --seeds 0 1 2 3 4 --out runs/pilot
python -m scripts.aggregate --runs runs/pilot --baseline baseline_strong
python -m scripts.plot_svd --runs runs/pilot --machine fan --fraction 0.1 --out runs/pilot/svd.png
```

## 4. Full grid + lambda sweep
```bash
python -m scripts.run_grid --exps baseline_plain baseline_strong nuclear_emb spectral_penalty spectral_norm \
    --machines fan gearbox --fractions 0.05 0.1 0.25 1.0 --seeds 0 1 2 3 4 --out runs/main \
    --sweep reg.nuclear_emb=1e-4,1e-3,1e-2,1e-1 --sweep-exp nuclear_emb
python -m scripts.aggregate --runs runs/main --baseline baseline_strong
```
Select lambda using training-side data only (holdout), not the test set, and fix the primary comparison
before running the grid (see `task.md`).

## Layout
| Path | What |
|---|---|
| `src/data.py` | file parsing, stratified holdout/subsampling, log-mel dataset (ch1 / mean / stack) |
| `src/models.py` | `ResNetLite` with an `embed()` method (pretrained EAT/BEATs backbone is a stub) |
| `src/regularizers.py` | nuclear norm (embeddings, weights), spectral-norm penalty, spectral normalisation |
| `src/spectrum.py`, `src/svd_logger.py` | effective rank, stable rank, train-vs-heldout spectrum logging |
| `src/metrics.py` | kNN / Mahalanobis scoring, AUC, pAUC (0.1), harmonic means |
| `src/train.py` | one run = one (machine, fraction, seed, config); writes `results.json`, `svd.jsonl` |
| `scripts/` | grid runner, aggregation with paired tests, SVD plots, synthetic data |
| `configs/` | `base` -> `baseline_plain` / `baseline_strong` -> regulariser configs |

## Known limitations / design notes
- **BatchNorm caveat:** weight-space penalties on layers followed by BN are scale-invariant and act like
  weight decay; default `reg.scope: linear` applies them to the embedding and classifier layers only.
- Evaluation uses the **final-epoch model** (no test-based checkpoint selection).
- Small datasets are repeated within an epoch (`train.min_steps_per_epoch`) so all fractions get a
  comparable optimisation budget.
- Pretrained backbone (Backbone B) is not implemented; add it in `src/models.py`.
- Official DCASE score is a harmonic mean over machines and domains; `aggregate.py` reproduces the
  style but verify against the official evaluation script before reporting numbers.
