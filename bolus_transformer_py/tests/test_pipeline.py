import numpy as np
import pandas as pd
import pytest

from bolus_transformer.config import NORMALIZED_SIGNALS, SIGNALS, Config
from bolus_transformer.data_loading import load_data
from bolus_transformer.dataset import sweep_windows
from bolus_transformer.features import make_labels
from bolus_transformer.metrics import alarm_metrics
from bolus_transformer.preprocessing import build_features, fit_norm_stats, load_sequence, normalize
from synthetic import write_cohort

CONFIG = Config()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("ohio"))
    write_cohort(d)
    return d


def test_features_never_see_future_boluses(data_dir):
    df, bolus_df = load_data(1, "training", data_dir)
    full = build_features(df, bolus_df, CONFIG)
    rng = np.random.default_rng(0)
    for p in rng.choice(len(df), 60, replace=False):
        t = df["minutes"].iloc[p]
        past_only = build_features(df, bolus_df[bolus_df["minutes"] <= t], CONFIG)
        pd.testing.assert_series_equal(full.iloc[p], past_only.iloc[p])


def test_rows_right_after_a_bolus_are_not_labeled_by_it(data_dir):
    df, bolus_df = load_data(1, "training", data_dir)
    seq = load_sequence(1, "training", CONFIG, data_dir)
    for b in bolus_df["minutes"]:
        after = np.searchsorted(seq.minutes, b, side="left")
        if after < len(seq) and seq.minutes[after] - b < 5:
            later = bolus_df["minutes"][(bolus_df["minutes"] > seq.minutes[after])]
            if (later <= seq.minutes[after] + CONFIG.label_window_minutes).any():
                continue
            assert seq.labels[after] == 0


def test_labels_match_definition():
    rows = np.arange(0, 200, 5.0)
    boluses = np.array([42.3, 120.0, 180.5])
    labels = make_labels(rows, boluses, 20.0)
    expected = [any(t < b <= t + 20 for b in boluses) for t in rows]
    np.testing.assert_array_equal(labels.astype(bool), expected)


def test_normalization_fits_after_log_transform(data_dir):
    seqs = [load_sequence(s, "training", CONFIG, data_dir) for s in (1, 2, 3)]
    stats = fit_norm_stats(seqs)
    x = np.concatenate([normalize(s, stats)["x"] for s in seqs])
    pooled = pd.DataFrame(x, columns=SIGNALS)
    assert np.allclose(pooled[NORMALIZED_SIGNALS].mean(), 0, atol=1e-3)
    assert (pooled["basis_gsr"].std() > 0.8) and (pooled["basis_gsr"].std() < 1.05)


def test_missing_inputs_use_learned_embedding():
    import torch
    from bolus_transformer.model import CausalTransformer
    torch.manual_seed(0)
    model = CausalTransformer(CONFIG).eval()
    x = torch.randn(2, CONFIG.context_length, len(SIGNALS))
    x_missing, x_zero = x.clone(), x.clone()
    x_missing[:, 20, 1] = float("nan")
    x_zero[:, 20, 1] = 0.0

    with torch.no_grad():
        out_missing, out_zero = model(x_missing), model(x_zero)
    assert torch.isfinite(out_missing).all()
    assert not torch.allclose(out_missing[:, 20:], out_zero[:, 20:])
    assert torch.allclose(out_missing[:, :20], out_zero[:, :20], atol=1e-6)

    model.train()
    model.bolus_logits(x_missing).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.missing_embeddings.grad[1].abs().sum() > 0


def test_alarm_metric_does_not_reward_staying_high():
    minutes = np.arange(0, 300, 5.0)
    boluses = np.array([100.0, 200.0])
    always_high = [{"probs": np.ones_like(minutes), "minutes": minutes, "bolus_minutes": boluses}]
    r = alarm_metrics(always_high, 20.0, 0.5)
    assert r["recall_at_fa"] == 0 and r["n_boluses"] == 2

    spikes = np.isin(minutes, [85.0, 190.0]).astype(float)
    r = alarm_metrics([{"probs": spikes, "minutes": minutes, "bolus_minutes": boluses}], 20.0, 0.5)
    assert r["recall_at_fa"] == 1 and r["fa_per_day"] == 0 and r["median_lead_min"] == 12.5


@pytest.mark.parametrize("n", [36, 37, 59, 60, 61, 500, 1013])
def test_sweep_covers_each_row_once(n):
    covered = []
    for s, keep in sweep_windows(n, 36, 12):
        assert keep >= 12
        covered += list(range(s + keep, s + 36))
    assert covered == list(range(12, n))


def test_end_to_end(data_dir, tmp_path):
    import main
    df = main.main(["--data-dir", data_dir, "--subjects", "1", "2", "3", "4", "--seeds", "0",
                    "--quick", "--quiet", "--out-dir", str(tmp_path)])
    assert len(df) == 4 * 4
    assert df["test_auroc"].notna().all()
    assert (tmp_path / "summary.md").exists()
    assert df[df["method"] == "gbm"]["test_auroc"].mean() > 0.7
    assert df["pretrain_test_mse"].dropna().notna().all()
