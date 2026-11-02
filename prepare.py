"""Kjører hele analysen og lagrer tabellene som appen leser.

Ekte data:              python prepare.py
Syntetiske testdata:    python prepare.py --demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import data, demo_data, pricing

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="bruk syntetiske testdata")
    arguments = parser.parse_args()

    raw_freq, raw_sev = demo_data.generate() if arguments.demo else data.load_raw()
    policies, claims, report = data.clean(raw_freq, raw_sev)
    print(f"Renset data: {report['policies_clean']:,} poliser, {report['claims_clean']:,} skader")

    policies["fleet_id"] = data.build_fleets(policies)

    print("Tester frekvensmodellen på data den ikke har sett ...")
    metrics, lift = pricing.evaluate_frequency(policies)

    print("Trener endelige modeller på alle data ...")
    frequency_model = pricing.fit_frequency(policies)
    severity_model = pricing.fit_severity(policies)
    policies["expected_claims"] = frequency_model.predict(policies) * policies["exposure"]
    policies["expected_severity"] = severity_model.predict(policies)
    policies["expected_cost"] = policies["expected_claims"] * policies["expected_severity"]

    fleet_totals = policies.groupby("fleet_id")[["claims", "expected_claims"]].sum()
    k = pricing.estimate_credibility_k(fleet_totals["claims"], fleet_totals["expected_claims"])
    k_low, k_high = pricing.bootstrap_credibility_k(fleet_totals["claims"], fleet_totals["expected_claims"])
    validation = pricing.validate_credibility(policies)
    print(f"Kredibilitet: k = {k:.1f} forventede skader (90 %-intervall {k_low:.1f} til {k_high:.1f})")

    fleets = pricing.build_fleet_table(policies, k, report["large_loss_load"])

    relativities = pd.concat([
        frequency_model.relativities().assign(model="frequency"),
        severity_model.relativities().assign(model="severity"),
    ])
    segments = pd.concat(
        [pricing.segment_table(policies, f) for f in pricing.FREQUENCY_FEATURES]
    )
    deductibles = pricing.deductible_table(claims["claim_capped"])

    metadata = {
        "source": "demo" if arguments.demo else "freMTPL2",
        "cleaning": report,
        "model": metrics,
        "credibility_k": k,
        "credibility_k_interval": [k_low, k_high],
        "credibility_validation": validation,
        "fleets": int(len(fleets)),
        "portfolio_frequency": float(policies["claims"].sum() / policies["exposure"].sum()),
        "portfolio_severity": float(policies["cost_capped"].sum() / policies["claims"].sum()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fleets.to_csv(OUTPUT_DIR / "fleets.csv.gz", index=False)
    segments.to_csv(OUTPUT_DIR / "segments.csv", index=False)
    relativities.to_csv(OUTPUT_DIR / "relativities.csv", index=False)
    deductibles.to_csv(OUTPUT_DIR / "deductibles.csv", index=False)
    lift.to_csv(OUTPUT_DIR / "lift.csv", index=False)
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Ferdig: {len(fleets):,} flåter lagret i {OUTPUT_DIR}")
    if arguments.demo:
        print("OBS: Dette er syntetiske testdata. Ikke bruk tallene i søknaden.")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
