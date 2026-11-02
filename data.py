"""Henter, renser og strukturerer data for prosjektet.

Datasettet er freMTPL2: franske bilansvarsforsikringer. Det består av to tabeller:
  - frekvensfilen: én rad per polise (risikoinformasjon og eksponering)
  - skadefilen:    én rad per skade (polise-ID og skadebeløp i euro)

Datasettet inneholder IKKE bedriftsflåter, premier eller egenandeler.
Flåtene lages derfor her, og det står tydelig merket i appen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Dataset-ID-er på OpenML.
OPENML_FREQ_ID = 41214
OPENML_SEV_ID = 41215

# Kolonnenavnene varierer litt mellom kopier av datasettet.
COLUMN_NAMES = {
    "idpol": "IDpol", "claimnb": "ClaimNb", "exposure": "Exposure",
    "area": "Area", "vehpower": "VehPower", "vehage": "VehAge",
    "drivage": "DrivAge", "bonusmalus": "BonusMalus", "vehbrand": "VehBrand",
    "vehgas": "VehGas", "density": "Density", "region": "Region",
    "claimamount": "ClaimAmount",
}

MAX_CLAIMS_PER_POLICY = 4     # Flere enn dette regnes som datafeil.
LARGE_CLAIM_QUANTILE = 0.995  # Enkeltskader kappes her.

FEATURE_BANDS = {
    "driver_age": ("DrivAge", [17, 25, 30, 40, 50, 60, 70, 200],
                   ["18-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]),
    "vehicle_age": ("VehAge", [-1, 0, 3, 7, 12, 200],
                    ["0", "1-3", "4-7", "8-12", "13+"]),
    "vehicle_power": ("VehPower", [0, 5, 7, 9, 100],
                      ["4-5", "6-7", "8-9", "10+"]),
}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Gir kolonnene like navn og fjerner anførselstegn fra tekstverdier."""
    df = df.rename(columns={c: COLUMN_NAMES.get(str(c).lower(), c) for c in df.columns})
    for column in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        if not is_numeric:
            df[column] = df[column].astype(str).str.strip("'\" ")
    df["IDpol"] = pd.to_numeric(df["IDpol"]).astype("int64")
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bruker lokale CSV-filer hvis de finnes, ellers lastes data fra OpenML."""
    freq_file, sev_file = RAW_DIR / "freMTPL2freq.csv", RAW_DIR / "freMTPL2sev.csv"
    if freq_file.exists() and sev_file.exists():
        print("Bruker lokale filer i data/raw/")
        return _standardize(pd.read_csv(freq_file)), _standardize(pd.read_csv(sev_file))

    from sklearn.datasets import fetch_openml

    print("Laster ned data fra OpenML (første gang kan ta noen minutter) ...")
    try:
        freq = fetch_openml(data_id=OPENML_FREQ_ID, as_frame=True).frame
        sev = fetch_openml(data_id=OPENML_SEV_ID, as_frame=True).frame
    except Exception as error:
        raise SystemExit(
            "Klarte ikke å laste ned fra OpenML. Last ned freMTPL2freq.csv og "
            f"freMTPL2sev.csv manuelt og legg dem i {RAW_DIR}.\nFeil: {error}"
        )
    return _standardize(freq), _standardize(sev)


def clean(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Kobler poliser og skader, kapper storskader og lager risikogrupper.

    Antall skader telles fra skadefilen, slik at antall og beløp henger sammen.
    Skader uten kjent polise fjernes. Rapporten viser hva som ble gjort.
    """
    report = {"policies_raw": int(len(freq)), "claims_raw": int(len(sev))}

    sev = sev[sev["ClaimAmount"] > 0].copy()
    known_policy = sev["IDpol"].isin(freq["IDpol"])
    report["claims_without_policy"] = int((~known_policy).sum())
    sev = sev[known_policy].copy()

    policies = freq.drop(columns=["ClaimNb"], errors="ignore").drop_duplicates("IDpol").copy()
    claim_counts = sev.groupby("IDpol").size()
    policies["claims"] = policies["IDpol"].map(claim_counts).fillna(0).astype(int)

    too_many = policies["claims"] > MAX_CLAIMS_PER_POLICY
    report["policies_removed_too_many_claims"] = int(too_many.sum())
    policies = policies[~too_many].copy()
    sev = sev[sev["IDpol"].isin(policies["IDpol"])].copy()

    policies["exposure"] = policies["Exposure"].clip(upper=1.0)
    policies = policies[policies["exposure"] > 0].copy()

    # Storskader kappes. Det som kappes bort fordeles jevnt på alle via et tillegg.
    cap = float(sev["ClaimAmount"].quantile(LARGE_CLAIM_QUANTILE))
    sev["claim_capped"] = sev["ClaimAmount"].clip(upper=cap)
    report["large_claim_cap"] = cap
    report["large_loss_load"] = float(sev["ClaimAmount"].sum() / sev["claim_capped"].sum())

    costs = sev.groupby("IDpol")[["ClaimAmount", "claim_capped"]].sum()
    policies["cost_raw"] = policies["IDpol"].map(costs["ClaimAmount"]).fillna(0.0)
    policies["cost_capped"] = policies["IDpol"].map(costs["claim_capped"]).fillna(0.0)

    for name, (source, bins, labels) in FEATURE_BANDS.items():
        policies[name] = pd.cut(policies[source], bins=bins, labels=labels).astype(str)
    policies["fuel"] = policies["VehGas"].replace({"Regular": "Bensin", "Diesel": "Diesel"})
    policies["area"] = policies["Area"].astype(str)
    policies["region"] = policies["Region"].astype(str)
    policies["brand"] = policies["VehBrand"].astype(str)

    report["policies_clean"] = int(len(policies))
    report["claims_clean"] = int(policies["claims"].sum())
    report["exposure_years"] = float(policies["exposure"].sum())
    return policies.reset_index(drop=True), sev.reset_index(drop=True), report


def build_fleets(
    policies: pd.DataFrame,
    median_size: int = 40,
    max_size: int = 500,
    seed: int = 42,
) -> pd.Series:
    """Grupperer poliser i fiktive flåter og returnerer flåte-ID per polise.

    Hvorfor sorteres det på BonusMalus?
    Hvis polisene ble fordelt helt tilfeldig, ville ingen flåte vært bedre eller
    dårligere enn andre, og flåtens egen historikk hadde vært uten verdi. I
    virkeligheten har flåter ulik sjåførkvalitet og risikokultur, og det ser ikke
    underwriteren direkte. Her etterlignes det slik:
      1. Poliser med lignende BonusMalus havner ofte i samme flåte.
      2. BonusMalus holdes UTENFOR prisingsmodellen.
    Da finnes det risiko som bare flåtens skadehistorikk kan avsløre.
    """
    rng = np.random.default_rng(seed)
    fleet_ids = pd.Series(index=policies.index, dtype="int64")
    next_id = 1

    for _, group in policies.groupby("region", sort=True):
        # Støyen gjør at flåtene er blandede, ikke perfekt sorterte.
        order = np.log(group["BonusMalus"].to_numpy()) + rng.normal(0, 0.12, len(group))
        indices = group.index.to_numpy()[np.argsort(order)]
        start = 0
        while start < len(indices):
            size = int(np.clip(rng.lognormal(np.log(median_size), 0.9), 5, max_size))
            fleet_ids.loc[indices[start:start + size]] = next_id
            next_id += 1
            start += size
    return fleet_ids.astype("int64")
