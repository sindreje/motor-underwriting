"""Tester som sjekker logikken. Kjør med: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from src import data, demo_data, pricing


@pytest.fixture(scope="module")
def policies():
    freq, sev = demo_data.generate(n_policies=200_000)
    cleaned, _, _ = data.clean(freq, sev)
    return cleaned


def relativity(table, feature, category):
    return table[(table["feature"] == feature) & (table["category"] == category)]["relativity"].iloc[0]


def test_frequency_model_finds_the_known_truth(policies):
    table = pricing.fit_frequency(policies).relativities()
    diesel = relativity(table, "fuel", "Diesel") / relativity(table, "fuel", "Bensin")
    area = relativity(table, "area", "E") / relativity(table, "area", "C")
    vehicle = relativity(table, "vehicle_age", "13+") / relativity(table, "vehicle_age", "4-7")
    assert diesel == pytest.approx(demo_data.TRUE_FREQ_DIESEL, rel=0.08)
    assert area == pytest.approx(1.25, rel=0.12)
    assert vehicle == pytest.approx(0.70, rel=0.12)


def test_model_total_matches_actual_claims(policies):
    predicted = pricing.fit_frequency(policies).predict(policies) * policies["exposure"]
    assert predicted.sum() == pytest.approx(policies["claims"].sum(), rel=0.001)


def test_every_policy_gets_exactly_one_fleet(policies):
    fleet_ids = data.build_fleets(policies)
    assert len(fleet_ids) == len(policies)
    assert fleet_ids.notna().all()


def test_cleaning_removes_claims_without_policy_and_keeps_totals():
    freq = pd.DataFrame({
        "IDpol": [1, 2], "ClaimNb": [1, 0], "Exposure": [1.5, 0.5], "Area": ["A", "B"],
        "VehPower": [6, 7], "VehAge": [1, 2], "DrivAge": [40, 50], "BonusMalus": [50, 60],
        "VehBrand": ["B1", "B2"], "VehGas": ["Regular", "Diesel"], "Density": [100, 200],
        "Region": ["R11", "R24"],
    })
    sev = pd.DataFrame({"IDpol": [1, 1, 99], "ClaimAmount": [100.0, 300.0, 900.0]})
    policies, claims, report = data.clean(freq, sev)
    assert report["claims_without_policy"] == 1
    assert policies.loc[policies["IDpol"] == 1, "claims"].iloc[0] == 2
    assert policies["exposure"].max() == 1.0
    assert report["large_loss_load"] >= 1.0


def test_standardize_handles_the_formats_datasets_come_in():
    raw = pd.DataFrame({
        "idpol": [1.0, 2.0], "exposure": [0.5, 1.0],
        "area": pd.Categorical(["'A'", "'B'"]), "VehGas": ["'Regular'", '"Diesel"'],
    })
    result = data._standardize(raw)
    assert result["IDpol"].tolist() == [1, 2]
    assert result["Area"].tolist() == ["A", "B"]
    assert result["VehGas"].tolist() == ["Regular", "Diesel"]


def test_credibility_formula():
    assert pricing.credibility(50, 50) == 0.5
    assert pricing.credibility(0, 50) == 0


def test_k_estimate_finds_known_fleet_differences():
    rng = np.random.default_rng(1)
    expected = np.full(20_000, 10.0)
    true_variance = 0.05
    fleet_effect = rng.gamma(1 / true_variance, true_variance, len(expected))
    actual = rng.poisson(expected * fleet_effect)
    assert pricing.estimate_credibility_k(actual, expected) == pytest.approx(20, rel=0.15)


def test_k_is_infinite_when_fleets_do_not_differ():
    expected = np.full(5_000, 10.0)
    actual = np.full(5_000, 10.0)
    assert pricing.estimate_credibility_k(actual, expected) == float("inf")


DEDUCTIBLES = pd.DataFrame({"deductible": [0, 500, 1000], "cost_factor": [1.0, 0.8, 0.6]})
ASSUMPTIONS = pricing.Assumptions()  # Målskadeprosent 70 %.


def fleet(cost, premium, z=0.5):
    return pd.Series({"expected_cost": cost, "current_premium": premium, "credibility": z})


@pytest.mark.parametrize("cost, premium, action, deductible", [
    (700, 1000, pricing.ACTION_KEEP, 0),         # Teknisk pris = dagens pris.
    (840, 1000, pricing.ACTION_INCREASE, 0),     # Trenger +20 %, innenfor grensen.
    (1050, 1000, pricing.ACTION_TERMS, 500),     # Trenger +50 %, løses med egenandel.
    (2000, 1000, pricing.ACTION_REFER, 1000),    # Lar seg ikke løse.
    (490, 1000, pricing.ACTION_DECREASE, 0),     # Trenger −30 %, grensen er −25 %.
])
def test_recommendation_rules(cost, premium, action, deductible):
    result = pricing.recommend(fleet(cost, premium), DEDUCTIBLES, ASSUMPTIONS)
    assert result["action"] == action
    assert result["deductible"] == deductible
    assert abs(result["premium_change"]) <= ASSUMPTIONS.max_rate_change + 1e-9


def test_recommendation_reaches_target_when_it_says_so():
    result = pricing.recommend(fleet(840, 1000), DEDUCTIBLES, ASSUMPTIONS)
    assert result["combined_ratio"] == pytest.approx(0.90)
    terms = pricing.recommend(fleet(1050, 1000), DEDUCTIBLES, ASSUMPTIONS)
    assert terms["combined_ratio"] <= 0.90 + 1e-9


def test_higher_deductible_never_raises_cost():
    amounts = pd.Series(np.random.default_rng(3).gamma(1.5, 1000, 10_000))
    factors = pricing.deductible_table(amounts)["cost_factor"]
    assert factors.iloc[0] == 1.0
    assert factors.is_monotonic_decreasing
