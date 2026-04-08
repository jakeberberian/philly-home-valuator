"""
Philly Home Valuator — Dash Web App v2
Three tabs: Home Valuation | Market Map | Mortgage & Bankroll
Run:  python 05_dashboard_app.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import scipy.stats as scipy_stats
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ── Load model artefacts ────────────────────────────────────────────────────
MODEL_PATH   = "models/hedonic_model_filtered.joblib"
CENSUS_PATH  = "data/census_zcta.csv"
RESULTS_PATH = "data/model_results_filtered.csv"

saved        = joblib.load(MODEL_PATH)
model        = saved["model"]
feature_cols = saved["feature_cols"]
resid_std    = saved["residual_std"]

census_df = pd.read_csv(CENSUS_PATH)
census_df["zip"] = census_df["zip"].astype(str).str.zfill(5)

CENSUS_FEATURES = [
    "median_household_income", "median_home_value", "median_gross_rent",
    "population", "median_year_built_neighborhood", "mean_commute_time",
    "vacancy_rate", "pct_bachelors_plus", "poverty_rate", "homeownership_rate",
]

# ── Load results for map & zip stats ────────────────────────────────────────
try:
    results_df = pd.read_csv(RESULTS_PATH)
    results_df["zip"] = results_df["zip"].astype(str).str.zfill(5)
    cv_col = "cv_predicted" if "cv_predicted" in results_df.columns else "predicted"
    results_df["resid_eval"] = results_df["PRICE"] - results_df[cv_col]

    zip_grp = results_df.groupby("zip").agg(
        resid_median=("resid_eval", "median"),
        resid_std=("resid_eval", "std"),
        n=("PRICE", "size"),
        median_price=("PRICE", "median"),
    ).query("n >= 5")
    zip_grp["hot_score"] = zip_grp["resid_median"] / resid_std
    zip_grp["resid_std"]  = zip_grp["resid_std"].fillna(resid_std)
    zip_hot_dict     = zip_grp["hot_score"].to_dict()
    zip_hotprem_dict = zip_grp["resid_median"].to_dict()   # $ premium above fair value
    zip_pstd_dict    = zip_grp["resid_std"].to_dict()      # price std for CDF model

    zip_city_dict = (
        results_df.dropna(subset=["CITY", "zip"])
        .groupby("zip")["CITY"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    ) if "CITY" in results_df.columns else {}

    map_available = (
        "LATITUDE" in results_df.columns
        and "LONGITUDE" in results_df.columns
        and results_df[["LATITUDE", "LONGITUDE"]].notna().all(axis=1).any()
    )
except Exception:
    results_df       = pd.DataFrame()
    zip_hot_dict     = {}
    zip_hotprem_dict = {}
    zip_pstd_dict    = {}
    zip_city_dict    = {}
    map_available    = False

# ── PA property tax rates ────────────────────────────────────────────────────
COUNTY_TAX_RATES = {
    "Delaware County":   0.027,
    "Montgomery County": 0.022,
    "Chester County":    0.018,
    "Bucks County":      0.016,
}

# ── Styling helpers ──────────────────────────────────────────────────────────
INPUT_STYLE = {
    "width": "100%", "padding": "8px", "borderRadius": "6px",
    "border": "1px solid #ccc", "fontSize": "15px", "boxSizing": "border-box",
}
LABEL_STYLE = {
    "fontWeight": "600", "marginBottom": "4px", "display": "block",
    "fontSize": "13px", "color": "#34495e",
}
CARD = {"padding": "12px 16px", "borderRadius": "8px", "marginBottom": "8px", "textAlign": "center"}

def zlabel(z):
    city = zip_city_dict.get(str(z), "")
    return f"{z} – {city}" if city else str(z)

# ── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Philly Home Valuator",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# ── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "maxWidth": "960px", "margin": "0 auto", "padding": "16px 20px",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "backgroundColor": "#f8f9fa",
    },
    children=[
        html.H2("Philly Home Valuator",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "4px"}),
        html.P("Delaware · Montgomery · Chester · Bucks Counties",
               style={"textAlign": "center", "color": "#95a5a6", "marginBottom": "18px",
                      "fontSize": "13px"}),

        dcc.Tabs(
            id="tabs", value="tab-valuation",
            colors={"border": "#dee2e6", "primary": "#2980b9", "background": "#f8f9fa"},
            children=[
                dcc.Tab(label="Home Valuation",      value="tab-valuation"),
                dcc.Tab(label="Market Map",          value="tab-map"),
                dcc.Tab(label="Mortgage & Bankroll", value="tab-mortgage"),
            ],
        ),

        # Pre-rendered tab panels (toggled via visibility)
        html.Div(id="panel-valuation", style={"marginTop": "20px"}, children=[
            # ── Inputs ────────────────────────────────────────────────────
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                children=[
                    html.Div([html.Label("Zip Code", style=LABEL_STYLE),
                              dcc.Input(id="zip", type="text", value="19083", style=INPUT_STYLE)]),
                    html.Div([html.Label("Year Built", style=LABEL_STYLE),
                              dcc.Input(id="year_built", type="number", value=1940, style=INPUT_STYLE)]),
                    html.Div([html.Label("Beds", style=LABEL_STYLE),
                              dcc.Input(id="beds", type="number", value=3, style=INPUT_STYLE)]),
                    html.Div([html.Label("Baths", style=LABEL_STYLE),
                              dcc.Input(id="baths", type="number", value=1.5, step=0.5, style=INPUT_STYLE)]),
                    html.Div([html.Label("Sq Ft", style=LABEL_STYLE),
                              dcc.Input(id="sqft", type="number", value=1500, style=INPUT_STYLE)]),
                    html.Div([html.Label("Lot Size (sqft)", style=LABEL_STYLE),
                              dcc.Input(id="lot_size", type="number", value=5000, style=INPUT_STYLE)]),
                    html.Div([html.Label("Days on Market", style=LABEL_STYLE),
                              dcc.Input(id="dom", type="number", value=0, style=INPUT_STYLE)]),
                    html.Div([html.Label("Property Type", style=LABEL_STYLE),
                              dcc.Dropdown(
                                  id="prop_type",
                                  options=[
                                      {"label": "Single Family", "value": "Single Family Residential"},
                                      {"label": "Townhouse",     "value": "Townhouse"},
                                      {"label": "Condo/Co-op",   "value": "Condo/Co-op"},
                                  ],
                                  value="Single Family Residential",
                                  clearable=False, style={"fontSize": "14px"},
                              )]),
                ],
            ),

            # ── Offer strength inputs ──────────────────────────────────────
            html.Div(
                style={"marginTop": "12px", "display": "grid",
                       "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
                children=[
                    html.Div([
                        html.Label("List / Asking Price ($)", style=LABEL_STYLE),
                        dcc.Input(id="list_price", type="number",
                                  placeholder="e.g. 419900", style=INPUT_STYLE),
                        html.P("Affects expected competition level.",
                               style={"fontSize": "10px", "color": "#aaa", "marginTop": "3px"}),
                    ]),
                    html.Div([
                        html.Label("Your Bid ($)", style=LABEL_STYLE),
                        dcc.Input(id="your_bid", type="number",
                                  placeholder="e.g. 440000", style=INPUT_STYLE),
                        html.P("Leave blank to skip offer analysis.",
                               style={"fontSize": "10px", "color": "#aaa", "marginTop": "3px"}),
                    ]),
                    html.Div([
                        html.Label("Max Budget ($)", style=LABEL_STYLE),
                        dcc.Input(id="max_bid", type="number",
                                  placeholder="e.g. 460000", style=INPUT_STYLE),
                        html.P("Hard ceiling shown on bid chart.",
                               style={"fontSize": "10px", "color": "#aaa", "marginTop": "3px"}),
                    ]),
                ],
            ),

            # ── Appreciation slider ────────────────────────────────────────
            html.Div(style={"marginTop": "16px"}, children=[
                html.Label("Annual Appreciation Assumption",
                           style={**LABEL_STYLE, "marginBottom": "8px"}),
                dcc.Slider(
                    id="appr_rate", min=0, max=8, step=0.5, value=3.0,
                    marks={i: f"{i}%" for i in range(0, 9)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.P("Overrides data-driven trend stored in the model. "
                       "Philly suburbs historical avg: ~3–5% annually.",
                       style={"fontSize": "11px", "color": "#95a5a6", "marginTop": "6px"}),
            ]),

            html.Button(
                "Estimate Value", id="btn", n_clicks=0,
                style={
                    "width": "100%", "padding": "13px", "marginTop": "18px",
                    "backgroundColor": "#2980b9", "color": "white",
                    "border": "none", "borderRadius": "8px",
                    "fontSize": "17px", "fontWeight": "bold", "cursor": "pointer",
                },
            ),
            html.Div(id="output-area", style={"marginTop": "20px"}),
        ]),

        html.Div(id="panel-map", style={"display": "none", "marginTop": "20px"}, children=[
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr", "gap": "12px",
                       "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.Label("Filter Zip Codes", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="map-zip-filter",
                            options=[{"label": zlabel(z), "value": z}
                                     for z in sorted(zip_hot_dict.keys())],
                            multi=True, placeholder="All zips",
                            style={"fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Color By", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="map-color-by",
                            options=[
                                {"label": "Sale Price",    "value": "PRICE"},
                                {"label": "Model Residual","value": "resid_eval"},
                            ],
                            value="PRICE", clearable=False, style={"fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Price Range ($k)", style=LABEL_STYLE),
                        dcc.RangeSlider(
                            id="map-price-range", min=300, max=750, step=25,
                            value=[300, 750],
                            marks={i: f"{i}k" for i in range(300, 800, 150)},
                            tooltip={"placement": "bottom"},
                        ),
                    ]),
                ],
            ),
            dcc.Graph(id="market-map", style={"height": "520px"}),
            html.Div(id="map-zip-table", style={"marginTop": "14px"}),
        ]),

        html.Div(id="panel-mortgage", style={"display": "none", "marginTop": "20px"}, children=[
            html.H4("Monthly Payment Calculator",
                    style={"color": "#2c3e50", "marginBottom": "16px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "14px"},
                children=[
                    html.Div([html.Label("Purchase Price ($)", style=LABEL_STYLE),
                              dcc.Input(id="mort-price", type="number", value=500000,
                                        style=INPUT_STYLE)]),
                    html.Div([html.Label("Down Payment ($)", style=LABEL_STYLE),
                              dcc.Input(id="mort-down", type="number", value=100000,
                                        style=INPUT_STYLE)]),
                    html.Div([html.Label("Interest Rate (%)", style=LABEL_STYLE),
                              dcc.Input(id="mort-rate", type="number", value=6.75,
                                        step=0.125, style=INPUT_STYLE)]),
                    html.Div([html.Label("Loan Term (years)", style=LABEL_STYLE),
                              dcc.Dropdown(
                                  id="mort-term",
                                  options=[{"label": f"{y} years", "value": y}
                                           for y in [30, 20, 15]],
                                  value=30, clearable=False, style={"fontSize": "14px"},
                              )]),
                    html.Div([html.Label("County (tax rate)", style=LABEL_STYLE),
                              dcc.Dropdown(
                                  id="mort-county",
                                  options=[{"label": k, "value": k}
                                           for k in COUNTY_TAX_RATES],
                                  value="Delaware County", clearable=False,
                                  style={"fontSize": "14px"},
                              )]),
                    html.Div([html.Label("Monthly Gross Income ($)", style=LABEL_STYLE),
                              dcc.Input(id="mort-income", type="number", value=15000,
                                        placeholder="For DTI", style=INPUT_STYLE)]),
                ],
            ),
            html.Div(style={"marginTop": "12px"}, children=[
                html.Label("Other Monthly Debts (car, student loans, etc.) ($)",
                           style=LABEL_STYLE),
                dcc.Input(id="mort-other-debts", type="number", value=500,
                          style={**INPUT_STYLE, "width": "220px"}),
            ]),
            html.Button(
                "Calculate", id="mort-btn", n_clicks=0,
                style={
                    "padding": "11px 32px", "marginTop": "16px",
                    "backgroundColor": "#27ae60", "color": "white",
                    "border": "none", "borderRadius": "8px",
                    "fontSize": "16px", "fontWeight": "bold", "cursor": "pointer",
                },
            ),
            html.Div(id="mortgage-output", style={"marginTop": "20px"}),
        ]),
    ],
)


# ── Tab visibility ───────────────────────────────────────────────────────────
@app.callback(
    Output("panel-valuation", "style"),
    Output("panel-map",       "style"),
    Output("panel-mortgage",  "style"),
    Input("tabs", "value"),
)
def switch_tab(tab):
    show = {"marginTop": "20px"}
    hide = {"display": "none"}
    return (
        show if tab == "tab-valuation" else hide,
        show if tab == "tab-map"       else hide,
        show if tab == "tab-mortgage"  else hide,
    )


# ── Valuation helpers ────────────────────────────────────────────────────────
def _has_new_feature(name):
    """Check if a feature exists in the current model (handles re-trained vs old model)."""
    return name in feature_cols


def build_input_row(zip_code, beds, baths, sqft, lot_size, year_built):
    zip_str = str(zip_code).zfill(5)[:5]
    zmatch  = census_df[census_df["zip"] == zip_str]
    zrow    = (zmatch.iloc[0].to_dict() if not zmatch.empty
               else census_df.median(numeric_only=True).to_dict())

    BASE_DATE = pd.Timestamp("2024-01-01")
    months    = max((pd.Timestamp.today() - BASE_DATE).days / 30.44, 0)
    home_age  = 2026 - year_built

    row = {
        "BATHS": baths, "SQUARE_FEET": sqft, "LOT_SIZE": lot_size,
        "home_age":        home_age,
        "bath_bed_ratio":  baths / beds if beds > 0 else 0,
        "sqft_per_bed":    sqft / beds  if beds > 0 else 0,
        "log_sqft":        np.log1p(sqft),
        "log_lot_size":    np.log1p(lot_size),
        "months_since_base": months,
        # New features (no-ops if model predates these)
        "log_home_age":    np.log1p(home_age),
        "sqft_per_lot":    sqft / lot_size if lot_size > 0 else 0,
        "is_historic":     int(year_built < 1940),
    }
    for feat in CENSUS_FEATURES:
        row[feat] = zrow.get(feat, np.nan)

    # Derived census feature
    inc = zrow.get("median_household_income", np.nan)
    row["income_sqft_idx"] = inc / sqft if (sqft > 0 and not np.isnan(inc)) else np.nan

    return row


# ── Valuation callback ────────────────────────────────────────────────────────
@app.callback(
    Output("output-area", "children"),
    Input("btn", "n_clicks"),
    [
        State("zip", "value"), State("beds", "value"), State("baths", "value"),
        State("sqft", "value"), State("lot_size", "value"), State("year_built", "value"),
        State("dom", "value"), State("prop_type", "value"),
        State("list_price", "value"), State("your_bid", "value"), State("max_bid", "value"),
        State("appr_rate", "value"),
    ],
    prevent_initial_call=True,
)
def estimate(n_clicks, zip_code, beds, baths, sqft, lot_size,
             year_built, dom, prop_type, list_price, your_bid, max_bid, appr_rate):
    try:
        beds, baths, sqft = int(beds), float(baths), int(sqft)
        lot_size, year_built = int(lot_size), int(year_built)
        dom = int(dom) if dom else 0
    except (TypeError, ValueError):
        return html.P("Please fill in all fields with valid numbers.",
                      style={"color": "red"})

    row = build_input_row(zip_code, beds, baths, sqft, lot_size, year_built)
    input_row = {col: 0 for col in feature_cols}
    input_row.update({k: v for k, v in row.items() if k in feature_cols})
    pt_col = f"PROPERTY_TYPE_{prop_type}"
    if pt_col in feature_cols:
        input_row[pt_col] = 1
    input_df = pd.DataFrame([input_row])[feature_cols]

    fair = model.predict(input_df)[0]
    low  = fair - 1.28 * resid_std
    high = fair + 1.28 * resid_std
    hb   = fair + resid_std

    dom_disc = 0
    if dom >= 45:
        pct = min(0.05, 0.02 + (dom - 45) / 1000)
        dom_disc = fair * pct

    ann_rate = (appr_rate or 3.0) / 100

    zip_str   = str(zip_code).zfill(5)[:5]
    city_name = zip_city_dict.get(zip_str, "")
    zip_disp  = f"{zip_str} – {city_name}" if city_name else zip_str
    hot_score = zip_hot_dict.get(zip_str, 0)

    # ── Gauge ─────────────────────────────────────────────────────────────
    gauge_min = max(0, low * 0.90)
    gauge_max = high * 1.10
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fair,
        number={"prefix": "$", "valueformat": ",.0f"},
        title={"text": f"Fair Value — {zip_disp}"},
        gauge={
            "axis": {"range": [gauge_min, gauge_max], "tickformat": "$,.0f"},
            "bar":  {"color": "#2980b9"},
            "steps": [
                {"range": [gauge_min, low],  "color": "#fadbd8"},
                {"range": [low,  fair],       "color": "#d5f5e3"},
                {"range": [fair, hb],         "color": "#fdebd0"},
                {"range": [hb,  gauge_max],   "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "value": hb, "thickness": 0.8,
            },
        },
    ))
    gauge.update_layout(height=260, margin=dict(t=60, b=10, l=20, r=20))

    cards = [dcc.Graph(figure=gauge, config={"displayModeBar": False})]

    # Core range + highest & best
    cards.append(html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
        children=[
            html.Div(style={**CARD, "backgroundColor": "#eaf2f8"}, children=[
                html.Div("80% Prediction Range",
                         style={"fontSize": "12px", "color": "#5d6d7e"}),
                html.Div(f"${low:,.0f}  —  ${high:,.0f}",
                         style={"fontSize": "17px", "fontWeight": "bold"}),
            ]),
            html.Div(style={**CARD, "backgroundColor": "#fef9e7"}, children=[
                html.Div("Highest & Best Offer",
                         style={"fontSize": "12px", "color": "#7d6608"}),
                html.Div(f"${hb:,.0f}",
                         style={"fontSize": "19px", "fontWeight": "bold", "color": "#d4ac0d"}),
                html.Div("≈ 84th percentile of comps",
                         style={"fontSize": "10px", "color": "#999"}),
            ]),
        ],
    ))

    # DOM discount
    if dom_disc > 0:
        cards.append(html.Div(style={**CARD, "backgroundColor": "#d5f5e3"}, children=[
            html.Div(f"DOM Discount ({dom} days on market)",
                     style={"fontSize": "12px", "color": "#1e8449"}),
            html.Div(f"Suggested: ${fair - dom_disc:,.0f}  (−${dom_disc:,.0f})",
                     style={"fontSize": "17px", "fontWeight": "bold", "color": "#1e8449"}),
        ]))

    # Zip hotness pill
    if abs(hot_score) > 0.2:
        hot_color = "#e74c3c" if hot_score > 0 else "#27ae60"
        label = "Hot Market" if hot_score > 0 else "Cool Market"
        delta_k = abs(hot_score * resid_std / 1000)
        direction = "above" if hot_score > 0 else "below"
        note = "Expect competitive offers." if hot_score > 0 else "Negotiation room possible."
        cards.append(html.Div(
            style={**CARD, "backgroundColor": "#fdfefe",
                   "border": f"1px solid {hot_color}"},
            children=[
                html.Div(f"{label} — {zip_str}",
                         style={"fontWeight": "bold", "color": hot_color, "fontSize": "14px"}),
                html.Div(
                    f"Homes in this zip typically sell ${delta_k:.0f}k {direction} "
                    f"model estimate. {note}",
                    style={"fontSize": "12px", "color": "#555", "marginTop": "4px"},
                ),
            ],
        ))

    # ── 5 / 10 / 15 Year Projections ──────────────────────────────────────
    proj_rows = []
    for yrs in [5, 10, 15]:
        p_bear = fair * (1 + ann_rate * 0.5)  ** yrs
        p_base = fair * (1 + ann_rate)        ** yrs
        p_bull = fair * (1 + ann_rate * 1.5)  ** yrs
        proj_rows.append(html.Tr([
            html.Td(f"{yrs} yrs",
                    style={"padding": "6px 10px", "fontWeight": "600"}),
            html.Td(f"${p_bear:,.0f}",
                    style={"padding": "6px 10px", "color": "#c0392b", "textAlign": "right"}),
            html.Td(f"${p_base:,.0f}",
                    style={"padding": "6px 10px", "fontWeight": "bold", "textAlign": "right"}),
            html.Td(f"${p_bull:,.0f}",
                    style={"padding": "6px 10px", "color": "#27ae60", "textAlign": "right"}),
        ]))

    cards.append(html.Div(style={"marginTop": "8px"}, children=[
        html.Div(f"Projected Value — {appr_rate:.1f}% annual",
                 style={"fontWeight": "700", "fontSize": "13px",
                        "color": "#6c3483", "marginBottom": "6px"}),
        html.Table(
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "14px",
                   "backgroundColor": "#f4ecf7", "borderRadius": "8px", "overflow": "hidden"},
            children=[
                html.Thead(html.Tr([
                    html.Th("Horizon",     style={"padding": "6px 10px", "textAlign": "left",
                                                   "color": "#6c3483"}),
                    html.Th("Bear (½×)",   style={"padding": "6px 10px", "textAlign": "right",
                                                   "color": "#c0392b"}),
                    html.Th("Base",        style={"padding": "6px 10px", "textAlign": "right",
                                                   "color": "#6c3483"}),
                    html.Th("Bull (1.5×)", style={"padding": "6px 10px", "textAlign": "right",
                                                   "color": "#27ae60"}),
                ])),
                html.Tbody(proj_rows),
            ],
        ),
        html.P("Bear = ½× rate · Bull = 1.5× rate · Not a guarantee.",
               style={"fontSize": "10px", "color": "#999", "marginTop": "4px"}),
    ]))

    # ── Offer strength ────────────────────────────────────────────────────
    if your_bid:
        try:
            bid        = float(your_bid)
            budget     = float(max_bid) if max_bid else None

            # Empirical model:
            #   μ = fair_value + hotness_premium × DOM_factor + list_price_gap_adj
            # If list < fair (underpriced): more competing bids → higher clearing price
            # If list > fair (overpriced):  fewer bids → lower clearing price
            hotness_prem = zip_hotprem_dict.get(zip_str, 0.0)
            price_std    = zip_pstd_dict.get(zip_str, resid_std)
            dom_factor   = max(0.5, 1.0 - (dom - 30) / 300) if dom >= 30 else 1.0
            list_val     = float(list_price) if list_price else fair
            gap_adj      = (fair - list_val) * 0.35   # +adj if underpriced, -adj if overpriced
            mu_clear     = fair + hotness_prem * dom_factor + gap_adj
            prob_win     = scipy_stats.norm.cdf(bid, loc=mu_clear, scale=price_std)

            if   prob_win >= 0.80: strength, s_color = "Very Strong",  "#1e8449"
            elif prob_win >= 0.65: strength, s_color = "Strong",       "#27ae60"
            elif prob_win >= 0.50: strength, s_color = "Competitive",  "#f39c12"
            elif prob_win >= 0.35: strength, s_color = "Moderate",     "#e67e22"
            else:                  strength, s_color = "Weak",         "#e74c3c"

            bid_vs_fv = (bid - fair) / fair * 100

            # ── P(win) card ───────────────────────────────────────────────
            cards.append(html.Div(
                style={**CARD, "backgroundColor": "#fafafa",
                       "border": f"2px solid {s_color}"},
                children=[
                    html.Div("Offer Strength",
                             style={"fontWeight": "700", "fontSize": "13px", "color": "#2c3e50",
                                    "marginBottom": "8px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                               "gap": "6px", "marginBottom": "10px"},
                        children=[
                            html.Div([
                                html.Div("Your Bid", style={"fontSize": "11px", "color": "#888"}),
                                html.Div(f"${bid:,.0f}",
                                         style={"fontWeight": "600", "fontSize": "14px"}),
                                html.Div(f"{bid_vs_fv:+.1f}% vs FV",
                                         style={"fontSize": "10px", "color": "#aaa"}),
                            ]),
                            html.Div([
                                html.Div("Expected Clear", style={"fontSize": "11px", "color": "#888"}),
                                html.Div(f"${mu_clear:,.0f}",
                                         style={"fontWeight": "600", "fontSize": "14px"}),
                                html.Div(
                                    f"hotness {hotness_prem:+,.0f}"
                                    + (f" · list {gap_adj:+,.0f}" if list_price else ""),
                                    style={"fontSize": "10px", "color": "#aaa"},
                                ),
                            ]),
                            html.Div(style={"textAlign": "center"}, children=[
                                html.Div(f"{prob_win * 100:.0f}%",
                                         style={"fontSize": "28px", "fontWeight": "bold",
                                                "color": s_color, "lineHeight": "1"}),
                                html.Div(strength,
                                         style={"fontSize": "11px", "color": s_color,
                                                "fontWeight": "600"}),
                            ]),
                        ],
                    ),

                    # ── Bid sweep sparkline ───────────────────────────────
                    dcc.Graph(
                        figure=_offer_sweep_fig(fair, mu_clear, price_std, bid, budget, list_val),
                        config={"displayModeBar": False},
                        style={"height": "160px", "marginTop": "4px"},
                    ),

                    # ── Bid-for-probability mini table ────────────────────
                    html.Div(style={"marginTop": "10px"}, children=[
                        html.Div("Bids needed for target win probability",
                                 style={"fontSize": "11px", "color": "#888",
                                        "marginBottom": "4px"}),
                        html.Div(
                            style={"display": "grid",
                                   "gridTemplateColumns": "repeat(5, 1fr)",
                                   "gap": "4px", "textAlign": "center"},
                            children=[
                                html.Div(children=[
                                    html.Div(f"{int(p*100)}%",
                                             style={"fontSize": "10px", "color": "#888"}),
                                    html.Div(
                                        f"${scipy_stats.norm.ppf(p, mu_clear, price_std):,.0f}",
                                        style={
                                            "fontSize": "12px", "fontWeight": "600",
                                            "color": s_color if abs(
                                                scipy_stats.norm.ppf(p, mu_clear, price_std) - bid
                                            ) < price_std * 0.4 else "#2c3e50",
                                        },
                                    ),
                                ])
                                for p in [0.40, 0.55, 0.65, 0.75, 0.85]
                            ],
                        ),
                    ]),
                ],
            ))
        except Exception:
            pass

    return cards


def _offer_sweep_fig(fair, mu_clear, price_std, your_bid, budget, list_val=None):
    """Compact bid sweep chart embedded in the offer strength card."""
    lo   = fair * 0.85
    hi   = (budget * 1.02) if budget else fair * 1.22
    bids = np.linspace(lo, hi, 300)
    probs = scipy_stats.norm.cdf(bids, mu_clear, price_std) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bids, y=probs, mode="lines",
        line=dict(color="#2980b9", width=2),
        hovertemplate="$%{x:,.0f} → %{y:.0f}%<extra></extra>",
    ))
    for lo_p, hi_p, col in [
        (65, 100, "rgba(39,174,96,0.10)"),
        (40, 65,  "rgba(243,156,18,0.10)"),
        (0,  40,  "rgba(231,76,60,0.10)"),
    ]:
        fig.add_hrect(y0=lo_p, y1=hi_p, fillcolor=col, line_width=0)
    pw = scipy_stats.norm.cdf(your_bid, mu_clear, price_std) * 100
    fig.add_vline(x=your_bid, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                  annotation_text=f"${your_bid:,.0f} ({pw:.0f}%)",
                  annotation_font_size=10, annotation_position="top right")
    if list_val and list_val != fair and lo <= list_val <= hi:
        fig.add_vline(x=list_val, line_dash="dot", line_color="#7f8c8d", line_width=1,
                      annotation_text=f"List ${list_val:,.0f}",
                      annotation_font_size=10, annotation_position="top left")
    if budget and lo <= budget <= hi:
        fig.add_vline(x=budget, line_dash="dot", line_color="#8e44ad", line_width=1,
                      annotation_text=f"Max ${budget:,.0f}",
                      annotation_font_size=10, annotation_position="top left")
    fig.update_layout(
        margin=dict(t=8, b=28, l=40, r=8),
        xaxis=dict(tickformat="$,.0f", tickfont_size=9),
        yaxis=dict(range=[0, 105], ticksuffix="%", tickfont_size=9),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ── Market Map callback ───────────────────────────────────────────────────────
@app.callback(
    Output("market-map",      "figure"),
    Output("map-zip-table",   "children"),
    Input("map-zip-filter",   "value"),
    Input("map-color-by",     "value"),
    Input("map-price-range",  "value"),
)
def update_map(zip_filter, color_by, price_range):
    if not map_available or results_df.empty:
        fig = go.Figure()
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=39.98, lon=-75.38), zoom=9),
            margin=dict(t=0, b=0, l=0, r=0), height=500,
            annotations=[dict(text="Run 02_modeling.ipynb to generate results data.",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=14))],
        )
        return fig, ""

    df = results_df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    df = df[(df["PRICE"] >= price_range[0] * 1000) & (df["PRICE"] <= price_range[1] * 1000)]
    if zip_filter:
        df = df[df["zip"].isin(zip_filter)]
    if df.empty:
        return go.Figure(), html.P("No sales match the current filters.",
                                   style={"color": "#888"})

    cv_col_local = "cv_predicted" if "cv_predicted" in df.columns else "predicted"
    hover = df.apply(
        lambda r: (
            f"<b>{r.get('ADDRESS', '')}</b><br>"
            f"{r.get('CITY', '')} {r.get('zip', '')}<br>"
            f"Sold: ${r['PRICE']:,.0f}<br>"
            f"Model: ${r[cv_col_local]:,.0f}<br>"
            f"Residual: ${r['resid_eval']:+,.0f}<br>"
            f"Date: {r.get('SOLD_DATE', '')}"
        ), axis=1
    )

    color_data   = df[color_by]
    colorscale   = "RdYlGn" if color_by == "resid_eval" else "Blues"
    colorbar_ttl = "Residual ($)" if color_by == "resid_eval" else "Sale Price ($)"

    fig = go.Figure(go.Scattermapbox(
        lat=df["LATITUDE"], lon=df["LONGITUDE"],
        mode="markers",
        marker=dict(
            size=7, opacity=0.7,
            color=color_data,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=colorbar_ttl, thickness=14),
        ),
        text=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=39.98, lon=-75.38), zoom=9.5),
        margin=dict(t=0, b=0, l=0, r=0),
        height=520,
    )

    # Zip summary table
    by_zip = (
        df.groupby("zip")
        .agg(n=("PRICE", "size"), median_price=("PRICE", "median"),
             median_resid=("resid_eval", "median"))
        .reset_index()
        .sort_values("median_price", ascending=False)
    )
    by_zip["city"] = by_zip["zip"].map(zip_city_dict).fillna("")

    th = {"padding": "5px 10px", "textAlign": "left", "backgroundColor": "#eaf2f8",
          "fontSize": "12px", "fontWeight": "600"}
    td = {"padding": "5px 10px", "fontSize": "12px", "borderBottom": "1px solid #eee"}

    rows = [html.Tr([html.Th(c, style=th)
                     for c in ["Zip", "City", "Sales", "Median Price", "Median vs Model"]])]
    for _, r in by_zip.head(20).iterrows():
        resid_color = "#e74c3c" if r["median_resid"] > 0 else "#27ae60"
        rows.append(html.Tr([
            html.Td(r["zip"],                   style=td),
            html.Td(r["city"],                  style=td),
            html.Td(int(r["n"]),                style=td),
            html.Td(f"${r['median_price']:,.0f}", style=td),
            html.Td(f"${r['median_resid']:+,.0f}",
                    style={**td, "color": resid_color, "fontWeight": "600"}),
        ]))

    return fig, html.Table(rows, style={"width": "100%", "borderCollapse": "collapse",
                                        "marginTop": "8px"})


# ── Mortgage callback ─────────────────────────────────────────────────────────
@app.callback(
    Output("mortgage-output", "children"),
    Input("mort-btn", "n_clicks"),
    [
        State("mort-price",       "value"),
        State("mort-down",        "value"),
        State("mort-rate",        "value"),
        State("mort-term",        "value"),
        State("mort-county",      "value"),
        State("mort-income",      "value"),
        State("mort-other-debts", "value"),
    ],
    prevent_initial_call=True,
)
def calculate_mortgage(n_clicks, price, down, rate, term, county, gross_income, other_debts):
    try:
        price       = float(price       or 500_000)
        down        = float(down        or 100_000)
        rate        = float(rate        or 6.75)
        term        = int(term          or 30)
        gross       = float(gross_income or 0)
        other_debts = float(other_debts  or 0)
    except (TypeError, ValueError):
        return html.P("Please fill in valid numbers.", style={"color": "red"})

    loan     = price - down
    down_pct = down / price * 100

    # Monthly P&I
    r = rate / 100 / 12
    n = term * 12
    pi = loan * r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else loan / n

    # Property tax (PA varies by county)
    tax_rate    = COUNTY_TAX_RATES.get(county, 0.022)
    monthly_tax = price * tax_rate / 12

    # Insurance (~0.4% of value)
    monthly_ins = max(125, price * 0.004 / 12)

    # PMI if < 20% down
    pmi = (loan * 0.008 / 12) if down_pct < 20 else 0

    total = pi + monthly_tax + monthly_ins + pmi

    # Bankroll check — $650k target mortgage
    MAX_MORTGAGE = 650_000
    over_budget  = loan - MAX_MORTGAGE
    loan_color   = "#27ae60" if loan <= MAX_MORTGAGE else "#e74c3c"
    loan_status  = (
        "Within $650k mortgage target"
        if loan <= MAX_MORTGAGE
        else f"${over_budget:,.0f} over $650k target"
    )

    # DTI
    dti_h = total / gross * 100       if gross > 0 else None
    dti_t = (total + other_debts) / gross * 100 if gross > 0 else None

    def dti_color(v, warn, bad):
        return "#27ae60" if v < warn else ("#f39c12" if v < bad else "#e74c3c")

    # Equity build-up (amortization milestones)
    balance  = loan
    eq_years = []
    eq_bal   = []
    eq_equit = []
    for month in range(1, n + 1):
        interest  = balance * r
        principal = pi - interest
        balance  -= principal
        yr = month // 12
        if month % 12 == 0 and yr in {1, 5, 10, 15, 20, 25, 30}:
            eq_years.append(yr)
            eq_bal.append(max(0, balance))
            eq_equit.append(price - max(0, balance))

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Bar(name="Loan Balance", x=eq_years,
                            y=[b / 1e3 for b in eq_bal],
                            marker_color="#e74c3c", opacity=0.8))
    eq_fig.add_trace(go.Bar(name="Equity", x=eq_years,
                            y=[e / 1e3 for e in eq_equit],
                            marker_color="#27ae60", opacity=0.8))
    eq_fig.update_layout(
        barmode="stack", height=230,
        xaxis_title="Year", yaxis_title="$k",
        title=f"Equity Build-up — {term}-yr @ {rate:.2f}%",
        title_font_size=13,
        margin=dict(t=40, b=30, l=40, r=20),
        legend=dict(orientation="h", y=1.15),
        template="plotly_white",
    )

    C2 = {**CARD, "fontSize": "14px"}
    output = [
        # Monthly breakdown grid
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
            children=[
                html.Div(style={**C2, "backgroundColor": "#eaf2f8"}, children=[
                    html.Div("Principal & Interest", style={"fontSize": "11px", "color": "#555"}),
                    html.Div(f"${pi:,.0f}/mo", style={"fontWeight": "bold", "fontSize": "18px"}),
                ]),
                html.Div(style={**C2, "backgroundColor": "#fef9e7"}, children=[
                    html.Div("Property Tax", style={"fontSize": "11px", "color": "#555"}),
                    html.Div(f"${monthly_tax:,.0f}/mo",
                             style={"fontWeight": "bold", "fontSize": "18px"}),
                    html.Div(f"{tax_rate * 100:.1f}% annual ({county})",
                             style={"fontSize": "10px", "color": "#999"}),
                ]),
                html.Div(style={**C2, "backgroundColor": "#f4ecf7"}, children=[
                    html.Div("Insurance (est.)", style={"fontSize": "11px", "color": "#555"}),
                    html.Div(f"${monthly_ins:,.0f}/mo",
                             style={"fontWeight": "bold", "fontSize": "18px"}),
                ]),
            ],
        ),
    ]

    if pmi > 0:
        output.append(html.Div(style={**C2, "backgroundColor": "#fdebd0"}, children=[
            html.Div(f"PMI ({down_pct:.1f}% down — below 20%)",
                     style={"fontSize": "11px", "color": "#935116"}),
            html.Div(f"+${pmi:,.0f}/mo  (until 80% LTV reached)",
                     style={"fontWeight": "bold", "fontSize": "15px", "color": "#935116"}),
        ]))

    # Total
    output.append(html.Div(style={**C2, "backgroundColor": "#2c3e50", "color": "white"}, children=[
        html.Div("Total Monthly Payment", style={"fontSize": "12px", "opacity": "0.8"}),
        html.Div(f"${total:,.0f}/mo", style={"fontWeight": "bold", "fontSize": "26px"}),
        html.Div(f"Loan ${loan:,.0f}  ·  {down_pct:.1f}% down  ·  {term}yr @ {rate:.2f}%",
                 style={"fontSize": "11px", "opacity": "0.7", "marginTop": "4px"}),
    ]))

    # Bankroll check
    output.append(html.Div(
        style={**C2, "backgroundColor": "#fafafa", "border": f"2px solid {loan_color}"},
        children=[
            html.Div("Bankroll Check — $650k Mortgage Target",
                     style={"fontWeight": "700", "fontSize": "13px", "color": "#2c3e50"}),
            html.Div(f"Loan Amount: ${loan:,.0f}",
                     style={"fontWeight": "bold", "color": loan_color,
                            "fontSize": "17px", "marginTop": "6px"}),
            html.Div(loan_status, style={"color": loan_color, "fontSize": "13px"}),
            html.Div(f"Pre-approval headroom: ${700_000 - loan:+,.0f} vs $700k cap",
                     style={"fontSize": "11px", "color": "#888", "marginTop": "4px"}),
        ],
    ))

    # DTI
    if dti_h is not None:
        output.append(html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
            children=[
                html.Div(style={**C2, "backgroundColor": "#fafafa", "border": "1px solid #ddd"},
                         children=[
                    html.Div("Housing DTI  (target <28%)",
                             style={"fontSize": "11px", "color": "#555"}),
                    html.Div(f"{dti_h:.1f}%",
                             style={"fontWeight": "bold", "fontSize": "22px",
                                    "color": dti_color(dti_h, 28, 36)}),
                ]),
                html.Div(style={**C2, "backgroundColor": "#fafafa", "border": "1px solid #ddd"},
                         children=[
                    html.Div("Total DTI  (target <36%)", style={"fontSize": "11px", "color": "#555"}),
                    html.Div(f"{dti_t:.1f}%",
                             style={"fontWeight": "bold", "fontSize": "22px",
                                    "color": dti_color(dti_t, 36, 43)}),
                ]),
            ],
        ))

    output.append(dcc.Graph(figure=eq_fig, config={"displayModeBar": False}))
    return output


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
