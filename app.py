"""Telecom Churn Prediction (Streamlit) - Home / Predict / Models / About.

Run:  python -m streamlit run app.py
Folder must contain: app.py, bg.jpg, churn_model.pkl (the fitted pipe_4 pipeline).
"""
import base64
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
BG_CANDIDATES = ["bg.jpg", "bg.jpeg", "bg.png"]
PAGES = ["Home", "Predict", "Models", "About"]
FONT = "'Times New Roman', Times, 'Liberation Serif', serif"

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")


NAV_RULES = [
    ('div[role="radiogroup"]', "gap: .4rem; justify-content: flex-end; flex-wrap: wrap;"),
    ('label[data-baseweb="radio"]',
     "background: rgba(8,20,36,.55); border: 1px solid rgba(110,200,255,.25); border-radius: 999px; "
     "padding: .35rem 1.2rem; margin: 0; cursor: pointer;"),
    ('label[data-baseweb="radio"] > div:first-child', "display: none;"),
    ('label[data-baseweb="radio"] p', "color: #B9CCDD; font-weight: 700; font-size: 1.08rem;"),
    ('label[data-baseweb="radio"]:has(input:checked)', "background: #4FD8FF; border-color: #4FD8FF;"),
    ('label[data-baseweb="radio"]:has(input:checked) p', "color: #06121F;"),
]
NAV_PREFIXES = [
    'div[data-testid="column"]:has(.nav-marker)',
    'div[data-testid="stColumn"]:has(.nav-marker)',
    ".st-key-page",
]
NAV_CSS = "\n".join(
    ", ".join(f"{pre} {sel}" for pre in NAV_PREFIXES) + " { " + decl + " }" for sel, decl in NAV_RULES
)


def go(page: str) -> None:
    st.session_state["page"] = page


# ---------- styling ----------
@st.cache_data
def build_css(home: bool) -> str:
    bg = ""
    for name in BG_CANDIDATES:
        path = BASE_DIR / name
        if path.exists():
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode()
            bg = f', url("data:{mime};base64,{encoded}")'
            break
    # Home: the picture stays clearly visible and fades to dark further down the page.
    overlay = (
        "linear-gradient(90deg, rgba(4,10,22,.76) 0%, rgba(4,10,22,.30) 52%, rgba(4,10,22,0) 100%), "
        "linear-gradient(180deg, rgba(4,10,22,.05) 0%, rgba(4,10,22,.50) 45%, rgba(4,10,22,.95) 100%)"
        if home
        else "linear-gradient(rgba(4,10,22,.74), rgba(4,10,22,.93))"
    )
    css = """
<style>
.stApp {
    background-color: #050B16;
    background-image: __OVERLAY____BG__;
    background-size: cover;
    background-position: center 30%;
    background-attachment: fixed;
    font-size: 17px;
}
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]) { font-family: __FONT__ !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1200px; }
[data-testid="stWidgetLabel"] p { font-size: 1.05rem; color: #D5E3F0; }
div[data-baseweb="radio"] p, div[data-baseweb="select"] div, input { font-size: 1.02rem; }

/* brand + navigation pills */
.brand { font-weight: 700; font-size: 2.3rem; line-height: 1.1; color: #F2F8FE; padding-top: .1rem; text-shadow: 0 2px 20px rgba(0,0,0,.5); }
__NAVCSS__
/* hide the heading anchor (chain) icons */
[data-testid="stHeaderActionElements"], a.anchor-link { display: none !important; }
.nav-marker { height: 0; margin-bottom: -1rem; }

/* form panel */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(8,20,36,.64);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(110,200,255,.22); border-radius: 16px;
}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: #4FD8FF; color: #06121F; border: none; font-weight: 700; font-size: 1.05rem;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #8BE6FF; color: #06121F; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: rgba(8,20,36,.55); color: #E8F1FA; border: 1px solid rgba(110,200,255,.40); font-weight: 700; font-size: 1.05rem;
}
.lead { color: #C3D4E4; font-size: 1.15rem; margin: -.2rem 0 1rem; max-width: 46rem; }
.page-title { font-weight: 700; font-size: 2.3rem; color: #F2F8FE; margin: 1.1rem 0 .3rem; }

/* hero */
.hero { padding-top: 2.2rem; }
.hero-title {
    font-weight: 700; font-size: 4.3rem; line-height: 1.02; color: #F7FBFF; margin: 0 0 .6rem;
    text-shadow: 0 3px 30px rgba(0,0,0,.65);
}
.hero-tag { font-size: 1.75rem; font-weight: 700; color: #4FD8FF; margin: 0 0 .9rem; text-shadow: 0 2px 18px rgba(0,0,0,.6); }
.hero-sub { color: #E1ECF6; font-size: 1.2rem; line-height: 1.55; max-width: 34rem; margin-bottom: 1.5rem; text-shadow: 0 1px 14px rgba(0,0,0,.6); }
@media (max-width: 800px) { .hero-title { font-size: 2.7rem; } .brand { font-size: 1.7rem; } }

/* glass panel */
.panel {
    background: rgba(8,20,36,.68);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(110,200,255,.22); border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
}
.panel h3 { margin: 0 0 .7rem; font-size: 1.4rem; font-weight: 700; color: #F2F8FE; }
.panel p { color: #D3E2F0; line-height: 1.6; margin: 0 0 .5rem; font-size: 1.08rem; }
.panel ol { color: #D3E2F0; line-height: 1.75; padding-left: 1.2rem; margin: 0; font-size: 1.08rem; }
.muted { color: #A5BACE !important; font-size: .98rem !important; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin-top: .6rem; color: #D3E2F0; font-size: .98rem; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: .4rem; }

/* gauge */
.gauge-card { text-align: center; padding: 1.5rem 1.4rem 1.4rem; }
.gauge-card svg { width: 100%; max-width: 340px; overflow: visible; }
.verdict-pill {
    display: inline-flex; align-items: center; gap: .55rem; padding: .5rem 1.2rem; border-radius: 999px;
    font-weight: 700; font-size: 1.15rem; background: rgba(255,255,255,.08); color: #F2F8FE; margin-top: .3rem;
}
.verdict-pill i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.chip-title { color: #A5BACE; font-size: .95rem; margin: 1.3rem 0 .5rem; }
.chips { display: flex; flex-wrap: wrap; gap: .45rem; justify-content: center; }
.chip { padding: .3rem .85rem; border-radius: 999px; font-size: .95rem; color: #F2F8FE; background: rgba(79,216,255,.12); border: 1px solid rgba(79,216,255,.32); }
.empty-card { color: #C3D4E4; font-size: 1.1rem; }
.caption-c { text-align: center; color: #A5BACE; font-size: .95rem; margin-top: -.5rem; }

/* contribution bars */
.crow { display: grid; grid-template-columns: 13rem 1fr 4.2rem; gap: .8rem; align-items: center; margin: .6rem 0; color: #E1ECF6; font-size: 1.05rem; }
.crow em { font-style: normal; font-weight: 700; font-size: .98rem; text-align: right; }

/* stats strip */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: .9rem; margin: 1.6rem 0 1.1rem; }
.stat { background: rgba(8,20,36,.68); border: 1px solid rgba(110,200,255,.22); border-radius: 14px; padding: 1rem 1.2rem; }
.stat b { display: block; font-size: 2.1rem; color: #4FD8FF; }
.stat span { color: #C3D4E4; font-size: 1.02rem; }
@media (max-width: 800px) { .stats { grid-template-columns: repeat(2, 1fr); } }

/* steps + drivers (home) */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.6rem; }
.steps .n {
    display: inline-flex; width: 32px; height: 32px; border-radius: 50%; align-items: center; justify-content: center;
    background: #4FD8FF; color: #06121F; font-weight: 700; margin-bottom: .6rem;
}
.steps b { display: block; color: #F2F8FE; margin-bottom: .25rem; font-size: 1.15rem; }
.drivers { display: grid; grid-template-columns: repeat(3, 1fr); }
.drivers > div { padding: .2rem 1.4rem; border-left: 1px solid rgba(110,200,255,.24); }
.drivers > div:first-child { border-left: none; padding-left: 0; }
.drivers b { display: block; font-size: 3rem; color: #4FD8FF; line-height: 1.1; margin-bottom: .3rem; }
@media (max-width: 800px) { .steps, .drivers { grid-template-columns: 1fr; } .drivers > div { border-left: none; padding-left: 0; } }

/* bars (churn rate by group) */
.barrow { display: grid; grid-template-columns: 10rem 1fr 3.4rem; gap: .7rem; align-items: center; margin: .45rem 0; font-size: 1.02rem; color: #E1ECF6; }
.track { height: 10px; border-radius: 5px; background: rgba(255,255,255,.12); overflow: hidden; }
.track > span { display: block; height: 100%; border-radius: 5px; }
.barrow em { font-style: normal; text-align: right; color: #F2F8FE; font-weight: 700; }
.bar-group { color: #A5BACE; font-size: .98rem; margin: 1rem 0 .2rem; }

/* model table */
.mtable { width: 100%; border-collapse: collapse; }
.mtable th { text-align: left; color: #A5BACE; font-weight: 700; font-size: 1rem; padding: .5rem .6rem; border-bottom: 1px solid rgba(110,200,255,.2); }
.mtable td { padding: .75rem .6rem; color: #E1ECF6; border-bottom: 1px solid rgba(110,200,255,.09); font-size: 1.05rem; vertical-align: middle; }
.mtable td.best { color: #4FD8FF; font-weight: 700; }
.mtable tr.used td:first-child { color: #4FD8FF; font-weight: 700; }
.mini { height: 4px; border-radius: 3px; background: rgba(255,255,255,.12); margin-top: .35rem; overflow: hidden; }
.mini > span { display: block; height: 100%; background: rgba(79,216,255,.8); }
.tag { margin-left: .5rem; font-size: .8rem; padding: .1rem .55rem; border-radius: 999px; background: rgba(79,216,255,.18); color: #4FD8FF; font-weight: 700; }

/* confusion matrices */
.cm-title { color: #F2F8FE; font-weight: 700; font-size: 1.1rem; margin: .8rem 0 .1rem; }
.cm { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-top: .5rem; }
.cm div { border-radius: 10px; padding: .75rem .6rem; text-align: center; background: rgba(255,255,255,.07); }
.cm div b { display: block; font-size: 1.6rem; color: #F2F8FE; }
.cm div span { font-size: .9rem; color: #A5BACE; }
.cm .good { background: rgba(61,220,151,.15); }
.cm .bad { background: rgba(255,107,107,.15); }

.chart-box svg { width: 100%; height: auto; }
</style>
"""
    return (
        css.replace("__OVERLAY__", overlay).replace("__BG__", bg).replace("__FONT__", FONT).replace("__NAVCSS__", NAV_CSS)
    )


st.markdown(build_css(st.session_state.get("page", "Home") == "Home"), unsafe_allow_html=True)


# ---------- model ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "churn_model.pkl was not found next to app.py. In your notebook run "
        "`joblib.dump(pipe_4, 'churn_model.pkl')` and copy the file into this folder."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:  # version mismatch, corrupted file, etc.
    st.error(
        "The model file could not be loaded. Run the app in the same environment you "
        f"used for the notebook (same scikit-learn version).\n\nDetails: {exc}"
    )
    st.stop()


# ---------- prediction helpers ----------
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "AdditionalServicesCount", "StreamingServices",
]


def build_row(inp: dict) -> pd.DataFrame:
    if inp["internet"] == "No":
        security = backup = protection = support = "No internet service"
        tv = movies = "No"
    else:
        security, backup = inp["security"], inp["backup"]
        protection, support = inp["protection"], inp["support"]
        tv, movies = inp["tv"], inp["movies"]

    if tv == "Yes" and movies == "Yes":
        streaming = "Both"
    elif tv == "Yes":
        streaming = "TV only"
    elif movies == "Yes":
        streaming = "Movie only"
    else:
        streaming = "Neither"

    extra_count = sum(v == "Yes" for v in (security, backup, protection, support))

    row = {
        "gender": inp["gender"],
        "SeniorCitizen": 1 if inp["senior"] == "Yes" else 0,
        "Partner": inp["partner"],
        "Dependents": inp["dependents"],
        "tenure": inp["tenure"],
        "InternetService": inp["internet"],
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "Contract": inp["contract"],
        "PaperlessBilling": inp["paperless"],
        "PaymentMethod": inp["payment"],
        "MonthlyCharges": inp["monthly"],
        "TotalCharges": round(inp["tenure"] * inp["monthly"], 2),
        "AdditionalServicesCount": extra_count,
        "StreamingServices": streaming,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def risk_tier(p: float):
    if p < 0.30:
        return "Low risk", "#3DDC97"
    if p < 0.50:
        return "Moderate risk", "#FFC857"
    return "High risk", "#FF6B6B"


def churn_traits(row: pd.Series) -> list:
    """Fallback chips: traits linked to higher churn in the EDA."""
    traits = []
    if row["Contract"] == "Month-to-month":
        traits.append(("Month-to-month contract", "42.7% churn vs 2.8% on two-year contracts"))
    if row["tenure"] <= 12:
        traits.append(("First 12 months", "47.4% churn in the first year"))
    if row["PaymentMethod"] == "Electronic check":
        traits.append(("Electronic check", "45.3% churn, highest of any payment method"))
    if row["InternetService"] == "Fiber optic":
        traits.append(("Fiber optic", "41.9% churn vs 19.0% on DSL"))
    if row["OnlineSecurity"] == "No":
        traits.append(("No Online Security", "41.8% churn vs 14.6% with it"))
    if row["TechSupport"] == "No":
        traits.append(("No Tech Support", "41.6% churn vs 15.2% with it"))
    if row["SeniorCitizen"] == 1:
        traits.append(("Senior citizen", "41.7% churn vs 23.6% non-senior"))
    if row["Dependents"] == "No":
        traits.append(("No dependents", "31.3% churn vs 15.5% with dependents"))
    if row["PaperlessBilling"] == "Yes":
        traits.append(("Paperless billing", "33.6% churn vs 16.3% without"))
    return traits[:3]


def _spaced(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def top_contributions(row_df: pd.DataFrame, n: int = 6) -> list:
    """Per-feature push on this customer's churn score (logistic regression coef x value)."""
    try:
        pre = model.named_steps["p1"]
        clf = model.named_steps["lr"]
        x = pre.transform(row_df)
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=float).ravel()
        names = pre.get_feature_names_out()
        contrib = x * clf.coef_[0]
        items = []
        for name, value in zip(names, contrib):
            if abs(value) < 1e-9:
                continue
            clean = name.split("__", 1)[-1]
            if "_" in clean:
                col, val = clean.split("_", 1)
                label = f"{_spaced(col)}: {val}"
            else:
                label = _spaced(clean).capitalize()
            items.append((label, float(value)))
        items.sort(key=lambda t: abs(t[1]), reverse=True)
        return items[:n]
    except Exception:
        return []


# ---------- html / svg builders ----------
ARC_LEN = 282.74  # length of the half-circle path (pi * 90)


def gauge_html(p: float, color: str, label: str, verdict: str, traits=None) -> str:
    filled = max(0.0, min(1.0, p)) * ARC_LEN
    chips = ""
    if traits is not None:
        if traits:
            inner = "".join(f'<span class="chip" title="{tip}">{name}</span>' for name, tip in traits)
            chips = f'<div class="chip-title">Pushing the risk up</div><div class="chips">{inner}</div>'
        else:
            chips = '<div class="chip-title">No major risk traits in this profile</div>'
    return f"""
<div class="panel gauge-card">
  <svg viewBox="0 0 220 135">
    <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="14" stroke-linecap="round"/>
    <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
          stroke-dasharray="{filled:.1f} {ARC_LEN}" style="filter: drop-shadow(0 0 7px {color}99);"/>
    <text x="110" y="100" text-anchor="middle" font-weight="700" font-size="40" fill="{color}">{p*100:.0f}%</text>
    <text x="110" y="124" text-anchor="middle" font-size="12" fill="#A5BACE">chance of leaving</text>
    <text x="20" y="130" text-anchor="middle" font-size="10" fill="#8CA3B8">0%</text>
    <text x="200" y="130" text-anchor="middle" font-size="10" fill="#8CA3B8">100%</text>
  </svg>
  <div class="verdict-pill"><i style="background:{color}"></i>{verdict}</div>
  <div class="muted" style="margin-top:.55rem;color:{color} !important;font-size:1.05rem !important;">{label}</div>
  {chips}
</div>
"""


def contrib_html(items: list) -> str:
    mx = max(abs(v) for _, v in items)
    rows = ""
    for name, v in items:
        up = v > 0
        col = "#FF6B6B" if up else "#3DDC97"
        rows += (
            f'<div class="crow"><span>{name}</span>'
            f'<div class="track"><span style="width:{abs(v)/mx*100:.0f}%;background:{col};"></span></div>'
            f'<em style="color:{col}">{"raises" if up else "lowers"}</em></div>'
        )
    return (
        '<div class="panel"><h3>What moves this prediction</h3>'
        '<p class="muted">Red bars push the risk up, green bars pull it down. Longer means a bigger push.</p>'
        f"{rows}</div>"
    )


BAR_COLORS = [
    "#4FD8FF", "#FF8FB1", "#FFD166", "#3DDC97", "#B28DFF", "#FF9F5A",
    "#7C9CFF", "#2DD4BF", "#E879F9", "#A3E635", "#FB7185", "#A8B8CC",
]


def bars_html(title: str, rows: list, start: int = 0) -> str:
    out = f'<div class="bar-group">{title}</div>'
    for i, (name, val) in enumerate(rows):
        shade = BAR_COLORS[(start + i) % len(BAR_COLORS)]
        out += (
            f'<div class="barrow"><span>{name}</span>'
            f'<div class="track"><span style="width:{min(val*1.8,100):.1f}%;background:{shade};"></span></div>'
            f"<em>{val:.1f}%</em></div>"
        )
    return out


def cm_html(tn: int, fp: int, fn: int, tp: int) -> str:
    return f"""
<div class="cm">
  <div class="good"><b>{tn}</b><span>Stayed, predicted stay</span></div>
  <div class="bad"><b>{fp}</b><span>Stayed, predicted leave</span></div>
  <div class="bad"><b>{fn}</b><span>Left, predicted stay</span></div>
  <div class="good"><b>{tp}</b><span>Left, predicted leave</span></div>
</div>"""


MODELS = [
    ("SVM (tuned)", 0.806, 0.649, 0.579, 0.612),
    ("Random Forest (tuned)", 0.812, 0.691, 0.523, 0.595),
    ("Naive Bayes", 0.714, 0.480, 0.840, 0.610),
    ("Logistic Regression (original)", 0.822, 0.688, 0.598, 0.640),
    ("Logistic Regression (tuned)", 0.750, 0.517, 0.834, 0.639),
]
MODEL_COLORS = {
    "SVM (tuned)": "#A78BFA",
    "Random Forest (tuned)": "#34D399",
    "Naive Bayes": "#FBBF24",
    "Logistic Regression (original)": "#38BDF8",
    "Logistic Regression (tuned)": "#FB7185",
}


def grouped_bars_svg() -> str:
    W, H, left, right, top, bottom = 760, 330, 46, 10, 18, 40
    pw, ph = W - left - right, H - top - bottom
    metrics = ["Accuracy", "Precision", "Recall", "F1 score"]
    group_w = pw / len(metrics)
    bar_w = group_w * 0.82 / len(MODELS)
    s = [f'<svg viewBox="0 0 {W} {H}">']
    for i in range(6):
        v = i / 5
        y = top + ph * (1 - v)
        s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-right}" y2="{y:.1f}" stroke="rgba(255,255,255,.11)"/>')
        s.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="13" fill="#A5BACE">{v:.1f}</text>')
    for gi, metric in enumerate(metrics):
        gx = left + gi * group_w + group_w * 0.09
        for mi, (name, *vals) in enumerate(MODELS):
            v = vals[gi]
            h = ph * v
            x = gx + mi * bar_w
            y = top + ph - h
            s.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{h:.1f}" rx="3" fill="{MODEL_COLORS[name]}"/>'
            )
            s.append(
                f'<text x="{x+(bar_w-3)/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-size="11" fill="#E1ECF6">{v:.2f}</text>'
            )
        s.append(
            f'<text x="{left + gi*group_w + group_w/2:.1f}" y="{H-12}" text-anchor="middle" font-size="15" '
            f'font-weight="700" fill="#F2F8FE">{metric}</text>'
        )
    s.append("</svg>")
    return "".join(s)


def legend_html() -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{n}</span>' for n, c in MODEL_COLORS.items()
    ) + "</div>"


def scatter_svg() -> str:
    W, H, left, right, top, bottom = 520, 330, 52, 20, 16, 46
    pw, ph = W - left - right, H - top - bottom
    x0, x1, y0, y1 = 0.44, 0.74, 0.46, 0.90
    offsets = {  # label dx, dy, anchor
        "SVM (tuned)": (-12, 4, "end"),
        "Random Forest (tuned)": (-12, 20, "end"),
        "Naive Bayes": (-12, 4, "end"),
        "Logistic Regression (original)": (0, -16, "middle"),
        "Logistic Regression (tuned)": (12, 4, "start"),
    }
    short = {
        "SVM (tuned)": "SVM",
        "Random Forest (tuned)": "Random Forest",
        "Naive Bayes": "Naive Bayes",
        "Logistic Regression (original)": "LR (original)",
        "Logistic Regression (tuned)": "LR (tuned)",
    }

    def px(v):
        return left + (v - x0) / (x1 - x0) * pw

    def py(v):
        return top + ph - (v - y0) / (y1 - y0) * ph

    s = [f'<svg viewBox="0 0 {W} {H}">']
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        s.append(f'<line x1="{left}" y1="{py(v):.1f}" x2="{W-right}" y2="{py(v):.1f}" stroke="rgba(255,255,255,.10)"/>')
        s.append(f'<text x="{left-8}" y="{py(v)+4:.1f}" text-anchor="end" font-size="12" fill="#A5BACE">{v:.1f}</text>')
    for v in (0.5, 0.6, 0.7):
        s.append(f'<line x1="{px(v):.1f}" y1="{top}" x2="{px(v):.1f}" y2="{top+ph}" stroke="rgba(255,255,255,.07)"/>')
        s.append(f'<text x="{px(v):.1f}" y="{top+ph+18}" text-anchor="middle" font-size="12" fill="#A5BACE">{v:.1f}</text>')
    s.append(f'<text x="{left+pw/2:.1f}" y="{H-8}" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F8FE">Precision</text>')
    s.append(
        f'<text x="14" y="{top+ph/2:.1f}" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F8FE" '
        f'transform="rotate(-90 14 {top+ph/2:.1f})">Recall</text>'
    )
    for name, _acc, prec, rec, _f1 in MODELS:
        dx, dy, anchor = offsets[name]
        c = MODEL_COLORS[name]
        s.append(f'<circle cx="{px(prec):.1f}" cy="{py(rec):.1f}" r="8" fill="{c}" style="filter: drop-shadow(0 0 6px {c}aa);"/>')
        s.append(
            f'<text x="{px(prec)+dx:.1f}" y="{py(rec)+dy:.1f}" text-anchor="{anchor}" font-size="13" '
            f'font-weight="700" fill="#F2F8FE">{short[name]}</text>'
        )
    s.append("</svg>")
    return "".join(s)


def donut_svg(stayed: int, churned: int) -> str:
    total = stayed + churned
    r = 70
    circ = 2 * 3.14159265 * r
    dash = circ * churned / total
    return f"""
<svg viewBox="0 0 220 200" style="max-width:240px;display:block;margin:0 auto;">
  <circle cx="110" cy="100" r="{r}" fill="none" stroke="#3DDC97" stroke-width="22"/>
  <circle cx="110" cy="100" r="{r}" fill="none" stroke="#FF6B6B" stroke-width="22"
          stroke-dasharray="{dash:.1f} {circ:.1f}" transform="rotate(-90 110 100)"/>
  <text x="110" y="104" text-anchor="middle" font-size="34" font-weight="700" fill="#F2F8FE">{churned/total*100:.1f}%</text>
  <text x="110" y="126" text-anchor="middle" font-size="13" fill="#A5BACE">churned</text>
</svg>
<div class="legend" style="justify-content:center;">
  <span><i style="background:#FF6B6B"></i>Churned: {churned:,}</span>
  <span><i style="background:#3DDC97"></i>Stayed: {stayed:,}</span>
</div>"""


def line_svg(labels: list, vals: list) -> str:
    W, H, left, right, top, bottom = 520, 270, 46, 18, 22, 40
    pw, ph = W - left - right, H - top - bottom
    ymax = 60.0
    inset = 30
    pts = []
    for i, v in enumerate(vals):
        x = left + inset + (pw - 2 * inset) * i / (len(vals) - 1)
        y = top + ph * (1 - v / ymax)
        pts.append((x, y, v))
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
    area = f"{pts[0][0]:.1f},{top+ph} {path} {pts[-1][0]:.1f},{top+ph}"
    s = [
        f'<svg viewBox="0 0 {W} {H}"><defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#4FD8FF" stop-opacity=".45"/>'
        '<stop offset="100%" stop-color="#4FD8FF" stop-opacity="0"/></linearGradient></defs>'
    ]
    for v in (0, 10, 20, 30, 40, 50, 60):
        y = top + ph * (1 - v / ymax)
        s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-right}" y2="{y:.1f}" stroke="rgba(255,255,255,.10)"/>')
        s.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#A5BACE">{v}%</text>')
    s.append(f'<polygon points="{area}" fill="url(#ag)"/>')
    s.append(f'<polyline points="{path}" fill="none" stroke="#4FD8FF" stroke-width="3" stroke-linejoin="round"/>')
    for (x, y, v), lab in zip(pts, labels):
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#4FD8FF"/>')
        s.append(f'<text x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#F2F8FE">{v:.1f}%</text>')
        s.append(f'<text x="{x:.1f}" y="{H-12}" text-anchor="middle" font-size="12" fill="#C3D4E4">{lab}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- top bar ----------
current = st.session_state.get("page", "Home")
brand_col, nav_col = st.columns([3, 2])
if current != "Home":
    brand_col.markdown('<div class="brand">Telecom Churn Prediction</div>', unsafe_allow_html=True)
nav_col.markdown('<div class="nav-marker"></div>', unsafe_allow_html=True)
page = nav_col.radio("Page", PAGES, horizontal=True, key="page", label_visibility="collapsed")


# ---------- Home ----------
def render_home() -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            '<div class="hero"><h1 class="hero-title">Telecom Churn Prediction</h1>'
            '<div class="hero-tag">Spot the customers who are about to leave.</div>'
            '<p class="hero-sub">Enter a customer’s plan, billing and services. The model estimates the '
            "chance they will churn and shows what is pushing the risk up.</p></div>",
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1.3, 1.1, 2.6])
        b1.button("Predict a customer", type="primary", on_click=go, args=("Predict",))
        b2.button("How it works", on_click=go, args=("About",))
    with right:
        sample_label, sample_color = risk_tier(0.78)
        st.markdown(
            gauge_html(
                0.78, sample_color, sample_label, "Likely to leave",
                [("Month-to-month contract", ""), ("First 12 months", ""), ("Electronic check", "")],
            )
            + '<div class="caption-c">Example result</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="stats">
  <div class="stat"><b>7,043</b><span>customers analysed</span></div>
  <div class="stat"><b>26.5%</b><span>left the company</span></div>
  <div class="stat"><b>4</b><span>models compared</span></div>
  <div class="stat"><b>82%</b><span>accuracy (Logistic Regression)</span></div>
</div>
<div class="panel">
  <h3>How it works</h3>
  <div class="steps">
    <div><span class="n">1</span><b>Enter the details</b><p>Contract, billing, tenure and the services the customer uses.</p></div>
    <div><span class="n">2</span><b>The model scores it</b><p>A logistic regression trained on 7,043 customers estimates the chance of leaving.</p></div>
    <div><span class="n">3</span><b>Read the result</b><p>See the risk meter and what pushes the risk up or down.</p></div>
  </div>
</div>
<div class="panel">
  <h3>What the data says about leaving</h3>
  <div class="drivers">
    <div><b>42.7%</b><p>of month-to-month customers churned, vs 2.8% on two-year contracts.</p></div>
    <div><b>47.4%</b><p>of customers in their first 12 months churned.</p></div>
    <div><b>45.3%</b><p>of electronic-check payers churned, the highest of any payment method.</p></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- Predict ----------
def render_predict() -> None:
    st.markdown('<div class="page-title">Predict customer churn</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lead">Fill in the customer details, press Check churn risk, and read the result below.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Customer")
        c = st.columns(4)
        gender = c[0].selectbox("Gender", ["Female", "Male"])
        senior = c[1].radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = c[2].radio("Has a partner", ["No", "Yes"], horizontal=True)
        dependents = c[3].radio("Has dependents", ["No", "Yes"], horizontal=True)

        st.subheader("Account")
        c = st.columns([1.5, 1, 1.25, 1, 1])
        tenure = c[0].slider("Tenure (months)", 0, 72, 12)
        contract = c[1].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = c[2].selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = c[3].number_input("Monthly charges", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
        paperless = c[4].radio("Paperless billing", ["Yes", "No"], horizontal=True)
        st.caption(f"Total charges are estimated as tenure × monthly charges: {tenure * monthly:,.2f}")

        st.subheader("Services")
        c = st.columns(4)
        internet = c[0].selectbox(
            "Internet service",
            ["Fiber optic", "DSL", "No"],
            format_func=lambda x: "No internet" if x == "No" else x,
        )
        off = internet == "No"
        security = c[1].radio("Online Security", ["No", "Yes"], horizontal=True, disabled=off)
        backup = c[2].radio("Online Backup", ["No", "Yes"], horizontal=True, disabled=off)
        protection = c[3].radio("Device Protection", ["No", "Yes"], horizontal=True, disabled=off)
        c = st.columns(4)
        support = c[0].radio("Tech Support", ["No", "Yes"], horizontal=True, disabled=off)
        tv = c[1].radio("Streaming TV", ["No", "Yes"], horizontal=True, disabled=off)
        movies = c[2].radio("Streaming Movies", ["No", "Yes"], horizontal=True, disabled=off)
        if off:
            st.caption("Internet add-ons are not available without internet service.")

        check = st.button("Check churn risk", type="primary")

    if check:
        row_df = build_row(
            dict(
                gender=gender, senior=senior, partner=partner, dependents=dependents,
                tenure=tenure, contract=contract, payment=payment, monthly=monthly,
                paperless=paperless, internet=internet, security=security, backup=backup,
                protection=protection, support=support, tv=tv, movies=movies,
            )
        )
        st.session_state["result"] = {
            "proba": float(model.predict_proba(row_df)[0][1]),
            "pred": int(model.predict(row_df)[0]),
            "traits": churn_traits(row_df.iloc[0]),
            "contrib": top_contributions(row_df),
        }

    result = st.session_state.get("result")
    st.markdown('<div class="page-title" style="font-size:1.8rem;margin-top:1.4rem;">Result</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown(
            '<div class="panel empty-card">Your result will appear here after you press '
            "<b>Check churn risk</b>.</div>",
            unsafe_allow_html=True,
        )
        return

    p = result["proba"]
    label, color = risk_tier(p)
    verdict = "Likely to leave" if result["pred"] == 1 else "Likely to stay"
    g_col, d_col = st.columns([2, 3], gap="large")
    if result["contrib"]:
        g_col.markdown(gauge_html(p, color, label, verdict), unsafe_allow_html=True)
        d_col.markdown(contrib_html(result["contrib"]), unsafe_allow_html=True)
    else:
        g_col.markdown(gauge_html(p, color, label, verdict, result["traits"]), unsafe_allow_html=True)


# ---------- Models ----------
def render_models() -> None:
    st.markdown('<div class="page-title">Model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><h3>How the models compared</h3>'
        "<p>Four models were compared: SVM, Random Forest, Naive Bayes and Logistic Regression, which is shown "
        "in two versions (original and tuned). They learned from 80% of the customers (5,634) and were then "
        "tested on the other 20% (1,409 customers they had never seen). In that test group, 373 customers "
        "actually left, which is 26.5%, the same rate as in the full data. The app uses the original "
        "<b>Logistic Regression</b>, the best balance of precision and recall.</p></div>",
        unsafe_allow_html=True,
    )

    best = [max(m[i] for m in MODELS) for i in range(1, 5)]
    head = "".join(f"<th>{h}</th>" for h in ["Model", "Accuracy", "Precision", "Recall", "F1 score"])
    body = ""
    for name, *vals in MODELS:
        used = name == "Logistic Regression (original)"
        tag = '<span class="tag">in this app</span>' if used else ""
        cells = f"<td>{name}{tag}</td>"
        for i, v in enumerate(vals):
            cls = ' class="best"' if abs(v - best[i]) < 1e-9 else ""
            cells += f'<td{cls}>{v:.3f}<div class="mini"><span style="width:{v*100:.0f}%;background:{MODEL_COLORS[name]};"></span></div></td>'
        body += f'<tr class="{"used" if used else ""}">{cells}</tr>'
    st.markdown(
        f'<div class="panel"><table class="mtable"><tr>{head}</tr>{body}</table>'
        '<p class="muted" style="margin-top:.8rem;">Best value in each column is highlighted. '
        "Precision: of the customers flagged as leaving, how many really left. "
        "Recall: of the customers who really left, how many were caught.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel chart-box"><h3>Scores side by side</h3>'
        '<p class="muted">Every model on the same four measures.</p>'
        + grouped_bars_svg()
        + legend_html()
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="panel chart-box"><h3>Precision vs recall</h3>'
            '<p class="muted">Closer to the top right is better. LR means Logistic Regression. Naive Bayes and tuned LR catch the most churners but raise more false alarms.</p>'
            + scatter_svg()
            + "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="panel"><h3>Logistic Regression: original vs tuned</h3>'
            '<div class="cm-title">Original</div>'
            '<p class="muted">Catches 223 of 373 churners with 101 false alarms.</p>'
            + cm_html(935, 101, 150, 223)
            + '<div class="cm-title" style="margin-top:1.1rem;">Tuned</div>'
            '<p class="muted">Catches 311 of 373 churners, but with 290 false alarms.</p>'
            + cm_html(746, 290, 62, 311)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="panel"><h3>Which one to use?</h3>'
        "<p>Tuning Logistic Regression barely changed its F1 score (0.640 original vs 0.639 tuned), but it changed "
        "the trade-off: the tuned version catches more churners (recall 0.83) with more false alarms (precision 0.52). "
        "Use the original when retention offers are costly, and the tuned version when losing a customer costs more "
        "than a wasted offer.</p></div>",
        unsafe_allow_html=True,
    )


# ---------- About ----------
def render_about() -> None:
    st.markdown('<div class="page-title">About this project</div>', unsafe_allow_html=True)

    t1, t2 = st.columns([3, 2], gap="large")
    with t1:
        st.markdown(
            """
<div class="panel">
  <h3>What it is</h3>
  <p>Customer churn means a customer stops using a company’s service. This project studies the Telco
  Customer Churn dataset (7,043 customers) to find which customers are most likely to leave, and uses
  a logistic regression model to estimate the risk for a new customer.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            '<div class="panel chart-box"><h3>How many left</h3>' + donut_svg(5174, 1869) + "</div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="panel"><h3>What the data showed</h3>'
            '<p class="muted">Share of customers who churned in each group.</p>'
            + bars_html("Contract", [("Month-to-month", 42.7), ("One year", 11.3), ("Two year", 2.8)], start=0)
            + bars_html("Internet service", [("Fiber optic", 41.9), ("DSL", 19.0), ("No internet", 7.4)], start=3)
            + bars_html(
                "Payment method",
                [("Electronic check", 45.3), ("Mailed check", 19.1), ("Bank transfer", 16.7), ("Credit card", 15.2)],
                start=6,
            )
            + bars_html("Age group", [("Senior citizen", 41.7), ("Not senior", 23.6)], start=10)
            + "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="panel chart-box"><h3>Churn falls the longer they stay</h3>'
            '<p class="muted">Share of customers who churned, by months with the company.</p>'
            + line_svg(["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"], [47.4, 28.7, 21.6, 19.0, 14.4, 6.6])
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="panel">
  <h3>How it was built</h3>
  <ol>
    <li>Cleaned the data: fixed the TotalCharges column and handled the 11 customers with zero tenure.</li>
    <li>Explored 23 questions about churn across contract, tenure, services, billing and demographics.</li>
    <li>Created two features: the number of add-on services, and a single streaming-services column.</li>
    <li>Encoded the categories and scaled the numbers, then split the data 80% train and 20% test.</li>
    <li>Trained and tuned Random Forest, Naive Bayes, SVM and Logistic Regression.</li>
    <li>Chose Logistic Regression for the app because it balanced precision and recall best.</li>
  </ol>
  <p class="muted" style="margin-top:.9rem;">These patterns show association, not proven cause. The prediction is an estimate, not a certainty.</p>
</div>
""",
        unsafe_allow_html=True,
    )


{"Home": render_home, "Predict": render_predict, "Models": render_models, "About": render_about}[page]()
