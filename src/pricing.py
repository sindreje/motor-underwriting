"""Prisingsmodell, kredibilitet og fornyelsesanbefaling.

Logikken i korte trekk:
  forventet skadekostnad = forventet antall skader x forventet skadestørrelse
  flåtejustert kostnad   = forventet skadekostnad x erfaringsfaktor
  teknisk pris           = flåtejustert kostnad / mål for skadeprosent
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# BonusMalus er bevisst utelatt. Se build_fleets i data.py.
FREQUENCY_FEATURES = ["driver_age", "vehicle_age", "vehicle_power", "fuel", "area", "region", "brand"]
SEVERITY_FEATURES = ["driver_age", "vehicle_age", "fuel"]

SIMULATED_TARGET_LOSS_RATIO = 0.70  # Brukes bare til å simulere dagens premie.

FEATURE_LABELS = {
    "driver_age": "Førerens alder", "vehicle_age": "Kjøretøyets alder",
    "vehicle_power": "Motoreffekt", "fuel": "Drivstoff", "area": "Tetthetsområde",
    "region": "Region", "brand": "Merke",
}


class GLM:
    """En GLM med logg-link. Kategorier kodes som 0/1-kolonner.

    Med logg-link er modellen multiplikativ: hver risikofaktor får en relativitet,
    for eksempel 1,30 = 30 % høyere enn referansegruppen.
    """

    def __init__(self, family: str, features: list[str]):
        model_class = PoissonRegressor if family == "poisson" else GammaRegressor
        self.model = model_class(alpha=1e-6, solver="newton-cholesky", max_iter=300)
        self.features = features
        self.encoder = None

    def fit(self, df: pd.DataFrame, target: np.ndarray, weight: np.ndarray) -> "GLM":
        # Referansegruppen er den største gruppen i hver faktor.
        text = df[self.features].astype(str)
        base = [pd.Series(weight, index=df.index).groupby(text[f]).sum().idxmax() for f in self.features]
        self.encoder = OneHotEncoder(drop=base, handle_unknown="ignore", sparse_output=False)
        X = self.encoder.fit_transform(text)
        self.model.fit(X, target, sample_weight=weight)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.encoder.transform(df[self.features].astype(str)))

    def relativities(self) -> pd.DataFrame:
        """Returnerer faktorene. Referansegruppen har alltid relativitet 1,00."""
        rows, position = [], 0
        for feature, categories, dropped in zip(
            self.features, self.encoder.categories_, self.encoder.drop_idx_
        ):
            for index, category in enumerate(categories):
                if index == dropped:
                    rows.append((feature, category, 1.0, True))
                else:
                    rows.append((feature, category, float(np.exp(self.model.coef_[position])), False))
                    position += 1
        return pd.DataFrame(rows, columns=["feature", "category", "relativity", "is_base"])


def fit_frequency(policies: pd.DataFrame) -> GLM:
    """Poisson-GLM for skader per forsikringsår, vektet med eksponering."""
    target = (policies["claims"] / policies["exposure"]).to_numpy()
    return GLM("poisson", FREQUENCY_FEATURES).fit(policies, target, policies["exposure"].to_numpy())


def fit_severity(policies: pd.DataFrame) -> GLM:
    """Gamma-GLM for kappet skadestørrelse, vektet med antall skader."""
    with_claims = policies[policies["claims"] > 0]
    target = (with_claims["cost_capped"] / with_claims["claims"]).to_numpy()
    return GLM("gamma", SEVERITY_FEATURES).fit(with_claims, target, with_claims["claims"].to_numpy())


def evaluate_frequency(policies: pd.DataFrame, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Tester modellen på 20 % av polisene som den ikke har sett."""
    train, test = train_test_split(policies, test_size=0.2, random_state=seed)
    model = fit_frequency(train)

    predicted_rate = model.predict(test)
    actual_rate = (test["claims"] / test["exposure"]).to_numpy()
    weight = test["exposure"].to_numpy()
    null_rate = np.full(len(test), train["claims"].sum() / train["exposure"].sum())

    model_deviance = mean_poisson_deviance(actual_rate, predicted_rate, sample_weight=weight)
    null_deviance = mean_poisson_deviance(actual_rate, null_rate, sample_weight=weight)

    # Løftkurve: grupper polisene etter predikert risiko og sjekk mot faktiske skader.
    lift = pd.DataFrame({"rate": predicted_rate, "exposure": weight, "claims": test["claims"].to_numpy()})
    lift = lift.sort_values("rate")
    lift["group"] = np.minimum((lift["exposure"].cumsum() / weight.sum() * 10).astype(int), 9) + 1
    lift["expected"] = lift["rate"] * lift["exposure"]
    table = lift.groupby("group")[["expected", "claims", "exposure"]].sum().reset_index()
    table["predicted_frequency"] = table["expected"] / table["exposure"]
    table["actual_frequency"] = table["claims"] / table["exposure"]

    metrics = {
        "holdout_policies": int(len(test)),
        "deviance_explained": float(1 - model_deviance / null_deviance),
        "actual_vs_expected_holdout": float(test["claims"].sum() / lift["expected"].sum()),
    }
    return metrics, table


def estimate_credibility_k(actual: np.ndarray, expected: np.ndarray) -> float:
    """Anslår k med Bühlmann-Straub, når antall skader antas Poisson-fordelt.

    Vi spør: varierer flåtenes A/E mer enn tilfeldigheter alene skulle tilsi?
    Det overskytende er ekte forskjeller mellom flåter (a). Da er k = 1 / a.
    Liten k = flåtene er ulike, og historikk teller mye.
    Uendelig k = ingen målbare forskjeller, og historikk teller ikke.
    """
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    expected = expected * actual.sum() / expected.sum()  # Totalt A/E settes til 1.
    ratio = actual / expected
    total = expected.sum()
    numerator = np.sum(expected * (ratio - 1) ** 2) - (len(ratio) - 1)
    denominator = total - np.sum(expected ** 2) / total
    between_fleet_variance = numerator / denominator
    return float(1 / between_fleet_variance) if between_fleet_variance > 0 else float("inf")


def credibility(expected_claims, k: float):
    """Z = n / (n + k), der n er forventet antall skader for flåten."""
    return expected_claims / (expected_claims + k)


def validate_credibility(policies: pd.DataFrame, seed: int = 42) -> dict:
    """Sjekker om kredibilitet faktisk gir bedre spådommer.

    Datasettet dekker bare ett år, så hver flåte deles i to tilfeldige halvdeler.
    Halvdel A er «historikken», og halvdel B er «fremtiden» som skal spås.
    Tre metoder sammenlignes med Poisson-avvik på flåtenivå i halvdel B.
    """
    rng = np.random.default_rng(seed)
    in_a = rng.random(len(policies)) < 0.5
    grouped = policies.assign(half=np.where(in_a, "a", "b")).groupby(["fleet_id", "half"])
    table = grouped[["claims", "expected_claims"]].sum().unstack(fill_value=0)
    a_claims, a_expected = table[("claims", "a")].to_numpy(), table[("expected_claims", "a")].to_numpy()
    b_claims, b_expected = table[("claims", "b")].to_numpy(), table[("expected_claims", "b")].to_numpy()

    keep = (a_expected > 0) & (b_expected > 0)
    a_claims, a_expected, b_claims, b_expected = (
        a_claims[keep], a_expected[keep], b_claims[keep], b_expected[keep]
    )
    k = estimate_credibility_k(a_claims, a_expected)
    ratio = a_claims / a_expected
    z = credibility(a_expected, k) if np.isfinite(k) else np.zeros_like(a_expected)

    model_only = b_expected
    experience_only = np.maximum(b_expected * ratio, 1e-3)
    blended = b_expected * (1 + z * (ratio - 1))

    def deviance(prediction: np.ndarray) -> float:
        return float(mean_poisson_deviance(b_claims, prediction, sample_weight=b_expected))

    return {
        "k_half_sample": k,
        "deviance_model_only": deviance(model_only),
        "deviance_experience_only": deviance(experience_only),
        "deviance_credibility": deviance(blended),
    }


def bootstrap_credibility_k(actual, expected, draws: int = 500, seed: int = 42) -> tuple[float, float]:
    """Gir et 90 %-intervall for k ved å trekke flåtene på nytt mange ganger."""
    rng = np.random.default_rng(seed)
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    estimates = []
    for _ in range(draws):
        sample = rng.integers(0, len(actual), len(actual))
        estimates.append(estimate_credibility_k(actual[sample], expected[sample]))
    low, high = np.percentile(estimates, [5, 95], method="lower")  # Tåler uendelig k.
    return float(low), float(high)


def apply_credibility(fleets: pd.DataFrame, k: float) -> pd.DataFrame:
    """Blander modell og flåtens historikk. Kalles på nytt når k endres i appen."""
    fleets = fleets.copy()
    fleets["credibility"] = credibility(fleets["expected_claims"], k) if np.isfinite(k) else 0.0
    fleets["experience_factor"] = 1 + fleets["credibility"] * (fleets["actual_vs_expected"] - 1)
    fleets["expected_cost"] = fleets["model_cost"] * fleets["experience_factor"]
    return fleets


def build_fleet_table(policies: pd.DataFrame, k: float, large_loss_load: float, seed: int = 42) -> pd.DataFrame:
    """Lager én rad per flåte med historikk, forventning og simulert dagens premie."""
    grouped = policies.groupby("fleet_id")
    fleets = grouped.agg(
        region=("region", "first"),
        vehicles=("IDpol", "count"),
        exposure=("exposure", "sum"),
        claims=("claims", "sum"),
        cost_capped=("cost_capped", "sum"),
        expected_claims=("expected_claims", "sum"),
        model_cost=("expected_cost", "sum"),
        avg_vehicle_age=("VehAge", "mean"),
        avg_driver_age=("DrivAge", "mean"),
        diesel_share=("fuel", lambda s: (s == "Diesel").mean()),
    ).reset_index()

    fleets["actual_vs_expected"] = fleets["claims"] / fleets["expected_claims"]
    fleets["model_cost"] *= large_loss_load  # Tillegg for kappede storskader.
    fleets = apply_credibility(fleets, k)

    # Datasettet har ikke premier. Dagens premie simuleres som modellpris med et
    # tilfeldig avvik, slik at noen flåter er for dyrt og andre for billig priset.
    rng = np.random.default_rng(seed)
    pricing_noise = rng.lognormal(0, 0.15, len(fleets))
    fleets["current_premium"] = fleets["model_cost"] / SIMULATED_TARGET_LOSS_RATIO * pricing_noise
    fleets["fleet_name"] = "Flåte " + fleets["fleet_id"].astype(str).str.zfill(5)
    return fleets


@dataclass(frozen=True)
class Assumptions:
    """Forutsetninger som kan endres i appen. Dette er eksempelverdier."""
    target_combined_ratio: float = 0.90
    expense_ratio: float = 0.20
    max_rate_change: float = 0.25
    tolerance: float = 0.05
    low_credibility: float = 0.25

    @property
    def target_loss_ratio(self) -> float:
        return self.target_combined_ratio - self.expense_ratio


ACTION_KEEP = "Forny uendret"
ACTION_INCREASE = "Øk prisen"
ACTION_TERMS = "Øk pris og egenandel"
ACTION_DECREASE = "Rom for prisreduksjon"
ACTION_REFER = "Vurder særskilt"
ACTIONS = [ACTION_REFER, ACTION_TERMS, ACTION_INCREASE, ACTION_KEEP, ACTION_DECREASE]


def deductible_table(claim_amounts: pd.Series, deductibles=(0, 250, 500, 1000, 2000)) -> pd.DataFrame:
    """Viser hvor mye av skadekostnaden som gjenstår etter hver egenandel.

    Beregningen er: gjennomsnittet av max(skade - egenandel, 0) / gjennomsnittlig skade.
    Den antar at dataene er uten egenandel, og at atferden ikke endres.
    """
    amounts = claim_amounts.to_numpy()
    rows = [(d, float(np.maximum(amounts - d, 0).mean() / amounts.mean())) for d in deductibles]
    return pd.DataFrame(rows, columns=["deductible", "cost_factor"])


def technical_premium(expected_cost: float, assumptions: Assumptions, cost_factor: float = 1.0) -> float:
    return expected_cost * cost_factor / assumptions.target_loss_ratio


def recommend(fleet, deductibles: pd.DataFrame, assumptions: Assumptions) -> dict:
    """Gir en fornyelsesanbefaling for én flåte (rad med expected_cost og current_premium)."""
    current = float(fleet["current_premium"])
    cost = float(fleet["expected_cost"])
    technical = technical_premium(cost, assumptions)
    gap = technical / current - 1
    ceiling = current * (1 + assumptions.max_rate_change)
    floor = current * (1 - assumptions.max_rate_change)

    deductible, cost_factor = 0, 1.0
    if abs(gap) <= assumptions.tolerance:
        action, premium = ACTION_KEEP, current
    elif gap < 0:
        action, premium = ACTION_DECREASE, max(technical, floor)
    elif technical <= ceiling:
        action, premium = ACTION_INCREASE, technical
    else:
        # Prisen alene strekker ikke til: finn laveste egenandel som lukker gapet.
        action, premium = ACTION_REFER, ceiling
        deductible, cost_factor = deductibles.iloc[-1][["deductible", "cost_factor"]]
        for _, row in deductibles.iloc[1:].iterrows():
            needed = technical_premium(cost, assumptions, row["cost_factor"])
            if needed <= ceiling:
                action = ACTION_TERMS
                deductible, cost_factor = row["deductible"], row["cost_factor"]
                premium = min(ceiling, max(needed, current))
                break

    return {
        "action": action,
        "technical_premium": technical,
        "required_change": gap,
        "premium": float(premium),
        "premium_change": float(premium) / current - 1,
        "deductible": int(deductible),
        "cost_factor": float(cost_factor),
        "low_credibility": float(fleet["credibility"]) < assumptions.low_credibility,
        **outcome(cost, float(premium), float(cost_factor), assumptions),
    }


def outcome(cost: float, premium: float, cost_factor: float, assumptions: Assumptions) -> dict:
    """Regner ut forventet skadeprosent og combined ratio for valgt pris og egenandel."""
    loss_ratio = cost * cost_factor / premium
    return {"loss_ratio": loss_ratio, "combined_ratio": loss_ratio + assumptions.expense_ratio}


def recommend_all(fleets: pd.DataFrame, deductibles: pd.DataFrame, assumptions: Assumptions) -> pd.DataFrame:
    results = pd.DataFrame([recommend(row, deductibles, assumptions) for _, row in fleets.iterrows()])
    results.index = fleets.index
    return results


def segment_table(policies: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Viser faktisk frekvens, skadestørrelse og A/E for hver gruppe."""
    table = policies.groupby(feature).agg(
        exposure=("exposure", "sum"), claims=("claims", "sum"),
        cost=("cost_capped", "sum"), expected_claims=("expected_claims", "sum"),
    ).reset_index().rename(columns={feature: "category"})
    table.insert(0, "feature", feature)
    table["frequency"] = table["claims"] / table["exposure"]
    table["severity"] = table["cost"] / table["claims"].replace(0, np.nan)
    table["burning_cost"] = table["cost"] / table["exposure"]
    table["actual_vs_expected"] = table["claims"] / table["expected_claims"]
    return table
