import pandas as pd

from srisk_result.analyze_portfolio_risk import compute_srisk_from_portfolio
from srisk_result.wallstreet_srisk_test import compute_srisk_from_portfolio as compute_wallstreet_srisk


METRICS = pd.DataFrame(
    {
        "Ticker": ["LOW", "HIGH"],
        "Sector": ["Tech", "Tech"],
        "Sigma": [0.2, 0.2],
        "MDD": [-0.1, -0.5],
        "Beta": [1.0, 1.0],
    }
)


for compute in (compute_srisk_from_portfolio, compute_wallstreet_srisk):
    low_mdd_risk = compute({"LOW": 1.0}, METRICS)["Srisk"]
    high_mdd_risk = compute({"HIGH": 1.0}, METRICS)["Srisk"]
    assert high_mdd_risk > low_mdd_risk

print("PASS: larger MDD increases Srisk in both calculators")
