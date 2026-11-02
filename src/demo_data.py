"""Syntetiske testdata med samme format som freMTPL2.

Brukes KUN til å teste koden. Fordi dataene lages fra kjente sannheter, kan vi
sjekke at modellen finner dem igjen. Tallene sier ingenting om virkelig risiko.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_BASE_FREQUENCY = 0.07
TRUE_FREQ_DRIVER_AGE = {"18-25": 2.2, "26-30": 1.4, "31-40": 1.0, "41-50": 1.05,
                        "51-60": 1.0, "61-70": 0.9, "71+": 1.1}
TRUE_FREQ_VEHICLE_AGE = {"0": 1.5, "1-3": 1.1, "4-7": 1.0, "8-12": 0.85, "13+": 0.7}
TRUE_FREQ_DIESEL = 1.15
TRUE_FREQ_AREA = {"A": 0.8, "B": 0.9, "C": 1.0, "D": 1.1, "E": 1.25, "F": 1.4}


def generate(n_policies: int = 300_000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_policies

    driver_age = rng.integers(18, 90, n)
    vehicle_age = np.minimum(rng.poisson(6.5, n), 30)
    vehicle_power = rng.integers(4, 14, n)
    diesel = rng.random(n) < 0.5
    area = rng.choice(list("ABCDEF"), n, p=[0.15, 0.11, 0.28, 0.22, 0.20, 0.04])
    region = rng.choice([f"R{r}" for r in (11, 24, 31, 52, 53, 72, 82, 93)], n)
    brand = rng.choice(["B1", "B2", "B3", "B5", "B12"], n)
    exposure = np.clip(rng.beta(1.2, 0.9, n), 0.01, 1.0)
    bonus_malus = np.clip(50 + rng.exponential(14, n) + (driver_age < 26) * 30, 50, 230).round()

    driver_band = pd.cut(driver_age, [17, 25, 30, 40, 50, 60, 70, 200],
                         labels=list(TRUE_FREQ_DRIVER_AGE)).astype(str)
    vehicle_band = pd.cut(vehicle_age, [-1, 0, 3, 7, 12, 200],
                          labels=list(TRUE_FREQ_VEHICLE_AGE)).astype(str)

    frequency = (
        TRUE_BASE_FREQUENCY
        * pd.Series(driver_band).map(TRUE_FREQ_DRIVER_AGE).to_numpy()
        * pd.Series(vehicle_band).map(TRUE_FREQ_VEHICLE_AGE).to_numpy()
        * np.where(diesel, TRUE_FREQ_DIESEL, 1.0)
        * pd.Series(area).map(TRUE_FREQ_AREA).to_numpy()
        # BonusMalus påvirker risiko, men er bevisst utelatt fra prisingsmodellen.
        * np.exp(0.012 * (bonus_malus - bonus_malus.mean()))
    )
    claim_counts = rng.poisson(frequency * exposure)

    policy_ids = np.arange(1, n + 1)
    freq = pd.DataFrame({
        "IDpol": policy_ids, "ClaimNb": claim_counts, "Exposure": exposure,
        "Area": area, "VehPower": vehicle_power, "VehAge": vehicle_age,
        "DrivAge": driver_age, "BonusMalus": bonus_malus, "VehBrand": brand,
        "VehGas": np.where(diesel, "Diesel", "Regular"),
        "Density": 500, "Region": region,
    })

    claim_policy = np.repeat(policy_ids, claim_counts)
    amounts = rng.gamma(1.5, 1100, len(claim_policy))
    is_large = rng.random(len(claim_policy)) < 0.01
    amounts[is_large] += rng.pareto(1.5, is_large.sum()) * 30_000
    sev = pd.DataFrame({"IDpol": claim_policy, "ClaimAmount": amounts})
    return freq, sev
