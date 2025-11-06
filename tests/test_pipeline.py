import numpy as np
import pandas as pd
from core.pipeline import optimize_erc_from_csv

def test_pipeline_smoke(tmp_path):
    dates = pd.date_range("2020-01-01", periods=120, freq="W")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "date": dates,
        "AAA": 100 * (1 + 0.001*rng.standard_normal(len(dates))).cumprod(),
        "BBB": 50  * (1 + 0.0015*rng.standard_normal(len(dates))).cumprod(),
        "CCC": 80  * (1 + 0.0008*rng.standard_normal(len(dates))).cumprod(),
    })
    csvf = tmp_path / "prices.csv"
    prices.to_csv(csvf, index=False)

    res = optimize_erc_from_csv(str(csvf), tickers=["AAA","BBB","CCC"], start="2020-01-01", end="2022-12-31")
    w = res["weights"]
    assert np.isclose(w.sum(), 1.0)
    assert len(w) == 3
    assert res["Sigma"].shape == (3,3)
