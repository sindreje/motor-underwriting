"""Streamlit-appen. Start med: streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, pricing
from src.charts import number, percent, signed_percent

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
METRICS = {
    "Skadefrekvens": ("frequency", True),
    "Gjennomsnittlig skade, kappet (€)": ("severity", False),
    "Skadekostnad per forsikringsår, kappet (€)": ("burning_cost", False),
}

st.set_page_config(page_title="Fornyelse av bilflåter", layout="wide")


def show(figure) -> None:
    st.pyplot(figure)
    plt.close(figure)


def points_delta(points: float, text: str) -> str | None:
    """Lager endringsteksten til en metric. Ingen endring gir ingen pil."""
    rounded = round(points, 1)
    if rounded == 0:
        return None
    sign = "-" if rounded < 0 else ""
    return f"{sign}{number(abs(rounded), 1)} {text}"


def area_note(feature: str) -> None:
    """Forklarer bokstavene i tetthetsområde."""
    if feature == "area":
        st.caption("A = mest landlig, F = mest urbant.")


@st.cache_data
def load_data():
    return {
        "fleets": pd.read_csv(DATA_DIR / "fleets.csv.gz"),
        "segments": pd.read_csv(DATA_DIR / "segments.csv", dtype={"category": str}),
        "relativities": pd.read_csv(DATA_DIR / "relativities.csv", dtype={"category": str}),
        "deductibles": pd.read_csv(DATA_DIR / "deductibles.csv"),
        "lift": pd.read_csv(DATA_DIR / "lift.csv"),
        "metadata": json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8")),
    }


@st.cache_data
def calculate(fleets: pd.DataFrame, deductibles: pd.DataFrame, k: float, target: float, costs: float, cap: float):
    assumptions = pricing.Assumptions(target_combined_ratio=target, expense_ratio=costs, max_rate_change=cap)
    adjusted = pricing.apply_credibility(fleets, k)
    recommendations = pricing.recommend_all(adjusted, deductibles, assumptions)
    return pd.concat([adjusted, recommendations], axis=1)


if not (DATA_DIR / "metadata.json").exists():
    st.title("Dataene mangler")
    st.write("Kjør `python prepare.py` i terminalen, og start appen på nytt.")
    st.stop()

data = load_data()
metadata = data["metadata"]
is_demo = metadata["source"] == "demo"
estimated_k = metadata["credibility_k"]

with st.sidebar:
    st.header("Forutsetninger")
    st.caption("Eksempelverdier som du kan endre. De er ikke Storebrands tall.")
    target_cr = st.slider("Mål for combined ratio (%)", 80, 100, 90) / 100
    expense = st.slider("Kostnadsprosent (%)", 10, 35, 20) / 100
    rate_cap = st.slider("Største prisendring (%)", 5, 50, 25, step=5) / 100
    default_k = int(round(np.clip(estimated_k, 1, 200))) if np.isfinite(estimated_k) else 200
    k = st.slider("Kredibilitet, k", 1, 200, default_k,
                  help="Hvor mange forventede skader en flåte trenger for å få 50 % vekt på egen historikk.")

assumptions = pricing.Assumptions(
    target_combined_ratio=target_cr, expense_ratio=expense, max_rate_change=rate_cap
)
fleets = calculate(data["fleets"], data["deductibles"], float(k), target_cr, expense, rate_cap)

st.title("Fornyelse av bilflåter")
st.markdown("**Hvilken pris og hvilke vilkår bør vi tilby når en flåte skal fornyes?**")
st.caption(
    "Et personlig demonstrasjonsprosjekt om underwriting av motorforsikring, "
    "ikke et Storebrand-verktøy. Bygget på åpne franske bilansvarsdata med "
    "konstruerte flåter og simulert premie. Beløp i euro."
)
if is_demo:
    st.error(
        "**SYNTETISKE TESTDATA.** Disse tallene er laget for å teste koden. "
        "Kjør `python prepare.py` for å bruke ekte data."
    )

portfolio_tab, fleet_tab, method_tab = st.tabs(["Portefølje", "Flåtefornyelse", "Metode og begrensninger"])

# ---------------------------------------------------------------- Portefølje
with portfolio_tab:
    total_premium = fleets["current_premium"].sum()
    current_loss_ratio = fleets["expected_cost"].sum() / total_premium
    current_cr = current_loss_ratio + expense
    proposed_premium = fleets["premium"].sum()
    proposed_cost = (fleets["expected_cost"] * fleets["cost_factor"]).sum()
    proposed_cr = proposed_cost / proposed_premium + expense

    columns = st.columns(4)
    columns[0].metric("Flåter", number(len(fleets)))
    columns[1].metric("Skadefrekvens", percent(metadata["portfolio_frequency"]))
    columns[2].metric(
        "Gjennomsnittlig skade (kappet)",
        f"{number(metadata['portfolio_severity'])} €",
        help=f"Enkeltskader er kappet på {number(metadata['cleaning']['large_claim_cap'])} €. "
             "Det som kappes bort, legges på prisen som et tillegg.",
    )
    columns[3].metric(
        "Forventet combined ratio",
        percent(proposed_cr),
        points_delta((proposed_cr - current_cr) * 100, "poeng mot dagens pris"),
        delta_color="inverse",
        help="Beregnet med simulert premie. Viser hva anbefalingene ville gjort med porteføljen.",
    )

    order = [a for a in pricing.ACTIONS if a in set(fleets["action"])]
    action_summary = fleets.groupby("action").agg(
        fleets=("fleet_id", "count"), premium=("current_premium", "sum"), proposed=("premium", "sum"),
    ).reindex(order)

    needs_action = fleets["action"].isin([pricing.ACTION_INCREASE, pricing.ACTION_TERMS, pricing.ACTION_REFER])
    can_decrease = fleets["action"] == pricing.ACTION_DECREASE
    st.subheader("Hovedfunn")
    st.markdown(
        f"Dagens simulerte pris gir en forventet combined ratio på **{percent(current_cr)}**, "
        f"mot målet på {percent(target_cr, 0)}. **{number(needs_action.sum())} av {number(len(fleets))} flåter** "
        f"({percent(fleets.loc[needs_action, 'current_premium'].sum() / total_premium, 0)} av premien) ligger under "
        f"teknisk pris og trenger tiltak, mens **{number(can_decrease.sum())}** ligger over og har rom for lavere pris. "
        f"Hvis alle anbefalingene aksepteres og ingen kunder forsvinner, "
        f"blir forventet combined ratio **{percent(proposed_cr)}**."
    )

    left, right = st.columns(2)
    with left:
        show(charts.action_chart(action_summary["fleets"]))
    with right:
        show(charts.funnel_chart(fleets))
        st.caption("Stiplet linje viser ventet tilfeldig variasjon. Punkter utenfor kan være reelle avvik.")

    st.subheader("Hva driver risikoen?")
    feature_names = {pricing.FEATURE_LABELS[f]: f for f in pricing.FREQUENCY_FEATURES}
    choice_left, choice_right = st.columns(2)
    feature_label = choice_left.selectbox("Risikofaktor", list(feature_names))
    metric_label = choice_right.selectbox("Mål", list(METRICS))
    feature = feature_names[feature_label]
    metric, is_percent = METRICS[metric_label]
    segment = data["segments"][data["segments"]["feature"] == feature]
    average = {
        "frequency": metadata["portfolio_frequency"],
        "severity": metadata["portfolio_severity"],
        "burning_cost": segment["cost"].sum() / segment["exposure"].sum(),
    }[metric]
    show(charts.segment_chart(segment, feature, metric, f"{metric_label} etter {feature_label.lower()}", average, is_percent))
    area_note(feature)

    table = segment.assign(
        Forsikringsår=segment["exposure"].round(0), Skader=segment["claims"],
        Frekvens=(segment["frequency"] * 100).round(1), Snittskade=segment["severity"].round(0),
        AE=segment["actual_vs_expected"].round(2),
    ).rename(columns={"category": "Gruppe", "Snittskade": "Snittskade (kappet)"})[
        ["Gruppe", "Forsikringsår", "Skader", "Frekvens", "Snittskade (kappet)", "AE"]
    ]
    with st.expander("Vis tabell"):
        st.dataframe(table, hide_index=True)

# ------------------------------------------------------------ Flåtefornyelse
with fleet_tab:
    filter_column, fleet_column = st.columns([1, 2])
    action_filter = filter_column.selectbox("Vis flåter med anbefaling", ["Alle"] + order, index=0)
    candidates = fleets if action_filter == "Alle" else fleets[fleets["action"] == action_filter]
    candidates = candidates.sort_values("current_premium", ascending=False).head(300)
    fleet_name = fleet_column.selectbox("Velg flåte (de 300 største vises)", candidates["fleet_name"].tolist())
    fleet = fleets[fleets["fleet_name"] == fleet_name].iloc[0]

    st.subheader(f"Anbefaling: {fleet['action']}")
    change_text = signed_percent(fleet["premium_change"])
    deductible_text = f", egenandel {number(fleet['deductible'])} €" if fleet["deductible"] else ""
    st.markdown(
        f"**Premie {number(fleet['premium'])} € ({change_text}{deductible_text}).** "
        f"Forventet combined ratio blir {percent(fleet['combined_ratio'])}."
    )
    if fleet["action"] == pricing.ACTION_REFER:
        st.warning("Selv største prisendring og høyeste egenandel når ikke målet. Saken bør vurderes særskilt.")
    if fleet["low_credibility"]:
        st.info(
            f"Flåten har lite egen erfaring (Z = {percent(fleet['credibility'], 0)}), "
            "så prisen bygger mest på modellen."
        )

    columns = st.columns(5)
    columns[0].metric("Kjøretøy", number(fleet["vehicles"]))
    columns[1].metric("Faktiske skader", number(fleet["claims"]))
    columns[2].metric("Forventede skader", number(fleet["expected_claims"], 1))
    columns[3].metric("A/E", number(fleet["actual_vs_expected"], 2))
    columns[4].metric("Erfaringsfaktor", number(fleet["experience_factor"], 2))

    left, right = st.columns(2)
    with left:
        show(charts.price_chart(fleet["current_premium"], fleet["technical_premium"], fleet["premium"]))
        st.caption(
            f"Modellen alene gir {number(fleet['model_cost'])} € i forventet skadekostnad. "
            f"Med flåtens historikk blir den {number(fleet['expected_cost'])} €. "
            f"Teknisk pris = skadekostnad / {percent(assumptions.target_loss_ratio, 0)} målskadeprosent."
        )
    with right:
        show(charts.credibility_chart(float(k), fleet["expected_claims"], fleet["credibility"]))

    st.subheader("Prøv andre vilkår")
    choice_left, choice_right = st.columns(2)
    premium_change = choice_left.slider("Prisendring (%)", -60, 60, int(round(fleet["premium_change"] * 100)), key=f"p{fleet_name}")
    deductible_options = data["deductibles"]["deductible"].astype(int).tolist()
    chosen_deductible = choice_right.selectbox(
        "Egenandel (€, illustrativ)", deductible_options,
        index=deductible_options.index(int(fleet["deductible"])), key=f"d{fleet_name}",
    )
    factor = float(data["deductibles"].set_index("deductible").loc[chosen_deductible, "cost_factor"])
    test_premium = fleet["current_premium"] * (1 + premium_change / 100)
    result = pricing.outcome(fleet["expected_cost"], test_premium, factor, assumptions)

    columns = st.columns(3)
    columns[0].metric("Premie", f"{number(test_premium)} €")
    columns[1].metric("Forventet skadeprosent", percent(result["loss_ratio"]))
    columns[2].metric(
        "Forventet combined ratio", percent(result["combined_ratio"]),
        points_delta((result["combined_ratio"] - target_cr) * 100, "poeng mot målet"), delta_color="inverse",
    )
    show(charts.funnel_chart(fleets, fleet))

# ------------------------------------------------------------------- Metode
with method_tab:
    model = metadata["model"]
    validation = metadata["credibility_validation"]
    cleaning = metadata["cleaning"]
    low, high = metadata["credibility_k_interval"]

    def k_text(value: float) -> str:
        return number(value, 0) if np.isfinite(value) else "uendelig"

    st.subheader("Slik beregnes prisen")
    st.markdown(
        "1. **Frekvens:** Poisson-GLM med eksponering som vekt.\n"
        "2. **Skadestørrelse:** Gamma-GLM på kappede skader.\n"
        f"3. **Storskader:** Skader kappes på {number(cleaning['large_claim_cap'])} €. "
        f"Det som kappes bort fordeles med et tillegg på {percent(cleaning['large_loss_load'] - 1)}.\n"
        "4. **Flåtens historikk:** Erfaringsfaktor = 1 + Z × (A/E − 1), der Z = n / (n + k). "
        "Bare skadefrekvens brukes, slik at én stor skade ikke styrer prisen.\n"
        "5. **Teknisk pris:** Skadekostnad / (mål for combined ratio − kostnadsprosent).\n"
        "6. **Anbefaling:** Regler for toleranse, prisgrense og egenandel gjør gapet om til et forslag."
    )

    st.subheader("Hvor god er modellen?")
    columns = st.columns(3)
    columns[0].metric("Testpoliser", number(model["holdout_policies"]))
    columns[1].metric("Forklart avvik", percent(model["deviance_explained"]),
                      help="Lav verdi er normalt på polisenivå, fordi skader i stor grad er tilfeldige.")
    columns[2].metric("A/E på testdata", number(model["actual_vs_expected_holdout"], 3))
    show(charts.lift_chart(data["lift"]))

    st.markdown("**Hjelper flåtens historikk?** Hver flåte ble delt i to tilfeldige halvdeler for å spå den ene med den andre:")
    st.dataframe(pd.DataFrame({
        "Metode": ["Bare modell", "Bare flåtehistorikk", "Kredibilitet (blanding)"],
        "Avvik (lavere er bedre)": [
            number(validation["deviance_model_only"], 3),
            number(validation["deviance_experience_only"], 3),
            number(validation["deviance_credibility"], 3),
        ],
    }), hide_index=True)
    if np.isfinite(estimated_k):
        st.caption(
            f"Anslått k er {k_text(estimated_k)} forventede skader, med 90 %-intervall "
            f"{k_text(low)} til {k_text(high)}. Anslaget er usikkert, så k kan justeres i menyen."
        )
    else:
        st.caption("Dataene viser ingen målbare forskjeller mellom flåtene. k kan justeres i menyen.")

    st.subheader("Prisfaktorer i frekvensmodellen")
    relativity_labels = {pricing.FEATURE_LABELS[f]: f for f in pricing.FREQUENCY_FEATURES}
    relativity_label = st.selectbox("Faktor", list(relativity_labels), key="relativity")
    relativity_feature = relativity_labels[relativity_label]
    relativities = data["relativities"]
    selected = relativities[(relativities["model"] == "frequency") & (relativities["feature"] == relativity_feature)]
    show(charts.relativity_chart(selected, relativity_feature, relativity_label))
    area_note(relativity_feature)

    st.subheader("Begrensninger")
    st.markdown(
        "- **Data:** Franske privatbiler med bare ansvarsdekning, ikke norske bedriftsflåter med kasko.\n"
        "- **Flåtene er konstruert:** Poliser med lignende BonusMalus grupperes, og BonusMalus holdes "
        "utenfor modellen. Slik etterlignes risiko som bare skadehistorikken avslører.\n"
        "- **Ett år:** Ekte fornyelse bør bygge på flere år, skadeutvikling og inflasjon.\n"
        "- **Premien er simulert:** Combined ratio og anbefalinger viser metoden, ikke faktisk lønnsomhet.\n"
        "- **Egenandelen er illustrativ:** Dataene har ingen egenandeler, og ansvar fungerer annerledes enn kasko.\n"
        "- **Ingen kundeatferd:** Modellen tar ikke hensyn til at kunder kan slutte etter en prisøkning."
    )
    st.caption(
        f"Datagrunnlag: {number(cleaning['policies_clean'])} poliser og {number(cleaning['claims_clean'])} skader "
        f"etter rensing. {number(cleaning['claims_without_policy'])} skader uten polise og "
        f"{number(cleaning['policies_removed_too_many_claims'])} poliser med flere enn fire skader ble fjernet."
    )
