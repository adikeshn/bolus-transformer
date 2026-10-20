import os
from datetime import datetime, timedelta

import numpy as np

FMT = "%d-%m-%Y %H:%M:%S"
WEARABLES = {
    "basis_heart_rate": (70, 8),
    "basis_gsr": (0.5, 0.3),
    "basis_skin_temperature": (91, 1.5),
    "basis_air_temperature": (78, 2),
}


def _events(name, rows, value_attr="value"):
    body = "".join(f'<event ts="{t.strftime(FMT)}" {value_attr}="{v}"/>' for t, v in rows)
    return f"<{name}>{body}</{name}>"


def write_subject(data_dir: str, subject: int, split: str, days: int = 4, seed: int = 0) -> str:
    rng = np.random.default_rng(seed * 1000 + subject + (7 if split == "testing" else 0))
    start = datetime(2027, 1, 1, 0, 1, 13)
    n = days * 288
    times = [start + timedelta(minutes=5 * i, seconds=int(rng.integers(-20, 20))) for i in range(n)]
    minutes = np.arange(n) * 5.0

    boluses = []
    for d in range(days):
        for hour in (7.5, 12.5, 18.5):
            m = d * 1440 + hour * 60 + rng.normal(0, 25)
            boluses.append((m, round(float(rng.uniform(3, 8)), 1), int(rng.uniform(30, 80))))

    glucose = 120 + rng.normal(0, 4, n)
    for m, _, carbs in boluses:
        dt = minutes - m
        glucose += np.where(dt > 0, carbs * 1.2 * (dt / 60) * np.exp(1 - dt / 60), 0)
    keep = rng.random(n) > 0.03
    xml = [_events("glucose_level", [(t, round(float(g))) for t, g, k in zip(times, glucose, keep) if k])]

    for name, (mu, sd) in WEARABLES.items():
        vals = np.abs(mu + rng.normal(0, sd, n))
        xml.append(_events(name, [(t - timedelta(seconds=40), round(float(v), 3))
                                  for t, v, k in zip(times, vals, rng.random(n) > 0.05) if k]))

    b_events = "".join(
        f'<event ts_begin="{(start + timedelta(minutes=m, seconds=17)).strftime(FMT)}" '
        f'ts_end="{(start + timedelta(minutes=m, seconds=17)).strftime(FMT)}" type="normal" '
        f'dose="{dose}" bwz_carb_input="{carbs}"/>'
        for m, dose, carbs in boluses
    )
    xml.append(f"<bolus>{b_events}</bolus>")

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{subject}-ws-{split}.xml")
    with open(path, "w") as f:
        f.write(f'<patient id="{subject}">' + "".join(xml) + "</patient>")
    return path


def write_cohort(data_dir: str, subjects=(1, 2, 3, 4), days: int = 4) -> None:
    for s in subjects:
        write_subject(data_dir, s, "training", days)
        write_subject(data_dir, s, "testing", max(2, days // 2))
