# Philly Home Valuator

Pricing model for residential properties in the Philadelphia suburbs (Delaware, Montgomery, Chester, and Bucks Counties), built to support data-driven offer decisions during the home search process.

## What it does

Given a property's zip code, beds, baths, square footage, lot size, and year built, the model returns:

- **Fair value estimate** — LightGBM hedonic regression prediction
- **80% prediction interval** — empirical residual-based confidence band
- **"Highest & best" offer** — fair value + 1σ (≈ 84th percentile of comparable sales)
- **DOM discount** — suggested discount for stale listings (≥ 45 days on market)
- **5/10/15-year projected value** — linear appreciation trend estimated from training data
- **Offer strength** — probabilistic win-probability estimate, anchored to list price in hot markets
- **Mortgage & bankroll analysis** — payment breakdown, manual tax/insurance inputs, PMI, DTI, equity build-up

A **3-tab Dash web app** wraps all of this into a single interface usable while touring homes.

## Project structure

```
philly-home-valuator/
├── 01_data_scraper.ipynb        # Redfin scraper + Census ACS enrichment
├── 02_modeling.ipynb            # Feature engineering, model comparison, Optuna tuning
├── 03_evaluation.ipynb          # CV diagnostics, geographic analysis, interactive Plotly charts
├── 04_offer_simulator.ipynb     # Probabilistic offer strength simulator
├── 05_dashboard_app.py          # Standalone 3-tab Dash app (run directly)
├── data/
│   ├── redfin_sold_homes.csv           # Raw Redfin scrape (~31k sold listings)
│   ├── census_zcta.csv                 # ACS 5-year ZCTA-level socioeconomics
│   ├── redfin_with_census.csv          # Joined dataset for modeling
│   ├── model_results.csv               # Full model predictions + residuals
│   └── model_results_filtered.csv      # Filtered model predictions + residuals
├── models/
│   ├── hedonic_model.joblib            # Full 4-county model
│   └── hedonic_model_filtered.joblib   # Filtered (SFR, Del+Mont, $300k–$750k)
├── plots/                       # Saved evaluation visualizations
├── requirements.txt
└── README.md
```

## Methodology

### Data pipeline

1. **Redfin scraper** — hits the internal `/stingray/api/gis-csv` endpoint using price-band pagination to stay under the 350-result cap. Bands that hit the cap are automatically bisected and re-scraped. Exponential back-off with jitter handles rate limits.

2. **Census enrichment** — pulls ACS 5-year estimates at the ZCTA level (median income, home value, education, vacancy, commute time, etc.) and joins to each listing on zip code.

### Modeling

| Step | Detail |
|------|--------|
| **Framework** | Hedonic pricing — decomposes home price into additive contributions from property attributes and neighbourhood characteristics |
| **Train/test split** | 80/20, stratified by county |
| **Cross-validation** | 5-fold CV on training set for model comparison |
| **Hyperparameter tuning** | Optuna Bayesian search (80 trials) on LightGBM |
| **Feature engineering** | `home_age`, `log_home_age`, `bath_bed_ratio`, `sqft_per_bed`, `sqft_per_lot`, `is_historic`, log transforms, `income_sqft_idx`, temporal trend (`months_since_base`) |
| **Appreciation estimate** | Residualized regression: Ridge on all non-time features, OLS on residuals vs. `months_since_base` (removes composition bias) |
| **Reproducibility** | `SEED = 51` throughout |

The **filtered model** restricts to Single Family Residential in Delaware + Montgomery Counties, $300k–$750k — the segment most relevant to our search. This yields a lower MAE than the full 4-county model within that range.

### Offer strength model

For a given zip code, we compute the median dollar premium buyers pay above the model's fair-value estimate (the "hotness premium") and the standard deviation of actual sale prices around the model prediction.

In **hot markets** (positive zip-level hot score), the clearing price is anchored to the list price — reflecting that sellers in a seller's market price to compete and buyers must bid at or above ask. The win probability is:

```
P(win) = Φ((bid − (list_price + hotness_premium × DOM_factor)) / price_std)
```

In **neutral/cool markets**, fair value serves as the anchor instead:

```
P(win) = Φ((bid − (fair_value + hotness_premium × DOM_factor + list_gap_adj)) / price_std)
```

This prevents unrealistically high win probabilities for below-ask bids in hot zip codes (e.g., 19083 Havertown).

### Key assumptions

1. Sale prices from the past 12 months are representative of current market conditions.
2. Census ACS 5-year estimates proxy neighbourhood quality at the ZCTA level.
3. The hedonic framework assumes approximate additive separability of attribute contributions — reasonable for suburban housing but may under-weight hyper-local effects (school catchment, block-level desirability).
4. `pct_white` is intentionally excluded from the feature set.
5. PMI is estimated at 0.20%/month of the loan balance (≈ 2.4% annually) for down payments < 20%.

## Quick start

```bash
# Clone and install
git clone https://github.com/<your-username>/philly-home-valuator.git
cd philly-home-valuator
pip install -r requirements.txt

# Run the dashboard
python 05_dashboard_app.py
# → http://127.0.0.1:8050
```

To re-scrape data or retrain models, run the notebooks in order (01 → 02 → 03 → 04). The scraper requires a Census API key in a `.env` file at the project root:

```
CENSUS_API_KEY=your_key_here
```

## Dashboard tabs

### Home Valuation
Enter property details manually or use the **Address Lookup** button to pre-populate fields from Redfin. All fields remain editable after lookup. Outputs: fair value gauge, prediction interval, highest & best offer, DOM discount, zip hotness signal, appreciation projections, and offer strength analysis.

### Market Map
Scatter map of sold listings from the model results dataset. When viewing **Model Residual**, dot size encodes sale price and color encodes residual direction — red dots sold above the model's prediction, green dots sold below. Includes a zip-level summary table.

### Mortgage & Bankroll
Configurable payment breakdown:
- Down payment as either **% of purchase price** or a **flat dollar amount**
- Manual inputs for **annual property taxes** and **annual home insurance** (replaces assumed rates)
- PMI baked in automatically when down payment < 20%
- DTI and equity build-up chart

## Deployment

The Dash app is ready for hosting on Render, Railway, or Heroku. Set the start command to:

```bash
gunicorn 05_dashboard_app:server --bind 0.0.0.0:$PORT
```

Ensure `models/` and `data/` (at minimum `census_zcta.csv` and `model_results_filtered.csv`) are included in the deployment.

## Tech stack

Python · pandas · scikit-learn · LightGBM · Optuna · Dash / Plotly · scipy · Census API · Redfin GIS-CSV API

## License

MIT
