"""
src/data_generator.py - synthetic market data for Artisano.

Run from the project root:   python src/data_generator.py

Hidden pricing law the ML model has to rediscover:
    price = (material_cost * 1.15 + craft_rate * labor_hours ** 1.05) * tier_multiplier * noise
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "artisan_market_data.csv"

# Value of one hour of skilled labour (INR/hour) per craft
CRAFT_RATE = {
    "Pottery": 45,
    "Weaving": 60,
    "Woodwork": 75,
    "Metalwork": 90,
    "Embroidery": 65,
    "Jewellery": 110,
}

# Market tiers: 0 = village haat, 1 = city market, 2 = premium / export
TIER_NAMES = {0: "Local Village Market", 1: "Urban Market", 2: "Premium / Export"}
TIER_MULTIPLIER = {0: 0.90, 1: 1.25, 2: 1.80}

MATERIAL_MARKUP = 1.15
SKILL_EXPONENT = 1.05


def generate_dataset(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of synthetic artisan sales records."""
    rng = np.random.default_rng(seed)

    craft = rng.choice(list(CRAFT_RATE), size=n_samples)
    material_cost = np.clip(rng.lognormal(np.log(250), 0.65, n_samples), 30, 4000)
    labor_hours = rng.uniform(0.5, 16.0, n_samples)
    market_tier = rng.choice([0, 1, 2], size=n_samples, p=[0.45, 0.40, 0.15])

    rate = np.array([CRAFT_RATE[c] for c in craft])
    tier_mult = np.array([TIER_MULTIPLIER[t] for t in market_tier])

    base_value = material_cost * MATERIAL_MARKUP + rate * labor_hours ** SKILL_EXPONENT
    noise = rng.lognormal(mean=0.0, sigma=0.08, size=n_samples)  # ~8 % haggling noise
    price = base_value * tier_mult * noise

    return pd.DataFrame(
        {
            "craft": craft,
            "material_cost": material_cost.round(2),
            "labor_hours": labor_hours.round(2),
            "market_tier": market_tier,
            "selling_price": price.round(2),
        }
    )


def main() -> None:
    df = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
