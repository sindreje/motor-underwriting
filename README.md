# Fornyelse av bilflåter

Et personlig prosjekt om underwriting av motorforsikring. Det svarer på ett spørsmål:

**Hvilken pris og hvilke vilkår bør vi tilby når en flåte skal fornyes?**

Prosjektet er laget på åpne data og er ikke knyttet til noe forsikringsselskap.

## Hva appen gjør

- **Portefølje:** Viser hva som driver risiko, og hvilke flåter som skiller seg ut.
- **Flåtefornyelse:** Regner ut teknisk pris og gir en anbefaling om pris og egenandel for en valgt flåte. Du kan selv teste andre vilkår.
- **Metode og begrensninger:** Viser hvordan modellen er bygget, hvor godt den treffer og hva den ikke kan brukes til.

## Metode

1. **Skadefrekvens:** Poisson-GLM med eksponering som vekt.
2. **Skadestørrelse:** Gamma-GLM på kappede skader, med et tillegg for storskader.
3. **Flåtens historikk:** Bühlmann-Straub-kredibilitet, der vekten er `Z = n / (n + k)`.
4. **Teknisk pris:** Skadekostnad / (mål for combined ratio − kostnadsprosent).
5. **Anbefaling:** Regler for toleranse, prisgrense og egenandel gjør prisgapet om til et forslag.

Modellen testes på poliser den ikke har sett. Kredibiliteten testes ved å bruke halvparten av hver flåte til å spå den andre halvparten.

## Data og begrensninger

Dataene er [freMTPL2](https://www.openml.org/d/41214), franske bilansvarsforsikringer fra R-pakken CASdatasets.

- Dataene gjelder franske privatbiler med ansvarsdekning, ikke norske bedriftsflåter med kasko.
- Flåtene er konstruert. Poliser med lignende BonusMalus grupperes, og BonusMalus holdes utenfor modellen. Slik etterlignes risiko som bare skadehistorikken avslører.
- Dataene dekker omtrent ett år.
- Dagens premie er simulert, og egenandelen er illustrativ.
- Forutsetningene er eksempelverdier. Kundeatferd etter en prisendring er ikke modellert.

Prosjektet viser derfor en metode og en måte å tenke på, ikke faktisk lønnsomhet.

## Kjør selv

```bash
pip install -r requirements.txt
python prepare.py
streamlit run app.py
```

Testene kjøres med `python -m pytest`.

## Struktur

| Fil | Innhold |
|---|---|
| `prepare.py` | Kjører analysen og lagrer tabellene appen bruker |
| `app.py` | Streamlit-appen |
| `src/data.py` | Henter og renser data og lager flåter |
| `src/pricing.py` | Modeller, kredibilitet og fornyelsesregler |
| `src/charts.py` | Grafer |
| `src/demo_data.py` | Syntetiske data for testing av koden |
| `tests/` | Automatiske tester |
