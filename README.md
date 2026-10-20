# bolus-transformer

Predicting when a person with type 1 diabetes is about to log a meal bolus, using CGM and wearable data from [OhioT1DM](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/). The model is a small causal transformer, pretrained to forecast the signals 30 minutes ahead and then fine-tuned to classify bolus timing. It is compared against a no-pretraining version, gradient-boosted trees, and a time-of-day-only model.

At every 5-minute CGM reading t, the target is whether a bolus is logged in (t, t + 20 min]. The labels are logged patient behavior, not a clinical ground truth for when a bolus should be given.

## Results

Leave-one-subject-out, 6 folds × 3 seeds, mean ± std.

| Method | AUROC | AP | Bolus recall @ ≤4 false alarms/day | Median lead (min) |
|---|---|---|---|---|
| Transformer (pretrained) | | | | |
| Transformer (no pretraining) | | | | |
| Gradient-boosted trees | | | | |
| Time of day only | | | | |

Earlier versions of this repo reported AUROC 0.760 / AP 0.328. Those numbers came from a pipeline with a label leak (described below) and have been replaced by the table above.

## Label leak fix

The first version labeled the first CGM reading after each bolus as positive. Bolus and CGM timestamps rarely line up exactly, so that reading's inputs already included the bolus (its insulin-on-board jump and carb entry). A guard excluded doses at exactly t = 0, but that case almost never occurs. On synthetic data in the same format, about 25% of positive rows were affected.

Now every feature at time t uses only boluses logged at or before t, and every label at t refers to boluses strictly after t. `tests/test_pipeline.py` checks that removing all future boluses never changes a feature.

## Method

**Data.** OhioT1DM 2018 cohort (subjects 559, 563, 570, 575, 588, 591). Inputs:

- glucose
- heart rate, GSR (log), skin temperature, air temperature
- insulin-on-board from the dose log (peak 75 min, duration 300 min)
- carbs entered with each bolus (log)
- time of day as sin/cos

Wearables are aligned to CGM timestamps with a backward as-of merge, and each recording is trimmed to where all wearables have data. Z-scores are fitted on training subjects only, after the log transforms.

**Model.**

- Per-signal linear embeddings fused into one token per timestep.
- Learned positions and 3 causal transformer encoder blocks (d_model 32, 4 heads), with a 3-hour context.
- Pretraining forecasts each signal 30 minutes ahead with a masked, weighted MSE.
- Fine-tuning adds a per-timestep bolus head trained with class-weighted BCE.

**Baselines.**

- The same transformer without pretraining.
- `HistGradientBoostingClassifier` on glucose and its 15/30/60-minute changes, insulin-on-board, carbs in the last 3 hours, minutes since the last bolus, heart rate, and time of day.
- Logistic regression on time of day alone.

**Evaluation.**

- Each fold holds out one test subject and uses the next subject as validation; the rest train.
- The validation subject picks the checkpoint (best forecast MSE, then best AP) and the alarm threshold. The test subject is never used for any of these choices.
- Both of the test subject's recordings (training and testing splits) are scored, every timestep once.
- An alarm fires when the predicted probability crosses the threshold. A bolus counts as caught if an alarm fired in the 20 minutes before it, and an alarm with no bolus in the next 20 minutes is a false alarm.
- Pretraining forecast error is compared against persistence (predict no change) on the same rows.

## Running it

In Colab, open `insulin_model_bolus.ipynb`, upload the OhioT1DM XML files to `bolus_data/`, and run all cells.

Locally:

```bash
cd bolus_transformer_py
pip install -r requirements.txt
python main.py --data-dir /path/to/OhioT1DM
python main.py --data-dir /path/to/OhioT1DM --quick
python -m pytest tests
```

The tests use synthetic data, so they run without the dataset. OhioT1DM requires a data use agreement with its authors.

## Layout

```
insulin_model_bolus.ipynb       self-contained notebook (same code as the package)
bolus_transformer_py/
  main.py                       CLI, writes results/results.csv and results/summary.md
  bolus_transformer/
    config.py  data_loading.py  features.py  preprocessing.py
    model.py   dataset.py       training.py  metrics.py
    baselines.py  loso.py
  tests/
```

## Limitations

- Six subjects, so per-subject variance is large.
- Timesteps are treated as evenly spaced, so CGM gaps compress time within a window.
- The insulin-on-board curve is the same for everyone and ignores basal insulin.
- The 2020 cohort (Empatica wearable) is not used yet.

Research code, not for clinical use.

## License

MIT
