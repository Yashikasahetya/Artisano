
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

try:                                    # layout: src/optimizer.py
    from src.optimizer import optimize_production
except ModuleNotFoundError:             # layout: optimizer.py in project root
    from optimizer import optimize_production
from src.data_generator import CRAFT_RATE, TIER_NAMES
from src.auth import require_login, render_account_sidebar

MODEL_PATH = Path(__file__).resolve().parent / "models" / "price_model.pkl"
CRAFTS = list(CRAFT_RATE)
TIERS = list(TIER_NAMES.values())
TIER_BY_NAME = {v: k for k, v in TIER_NAMES.items()}


TRAIN_MATERIAL_RANGE = (30, 4000)
TRAIN_HOURS_RANGE = (0.5, 16.0)

st.set_page_config(page_title="Artisano", page_icon="🧵", layout="wide")

st.markdown(
    """
    <style>
    h1 { color: #2D3A47; }
    h2, h3, h4 { color: #B46A72; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid #F7C8D3;
        border-radius: 14px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #5C6B7A; }
    .stButton > button { border-radius: 999px; border: 1px solid #B46A72; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #A9B7C6; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    [data-testid="stExpander"] { border: 1px solid #A9B7C6; border-radius: 12px; }

    /* --- Authentication card (same palette as metrics / expanders) --- */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid #F7C8D3;
        border-radius: 14px;
        padding: 20px 24px;
    }
    [data-testid="stFormSubmitButton"] > button { border-radius: 999px; border: 1px solid #B46A72; width: 100%; }
    [data-testid="stSidebar"] .stButton > button { width: 100%; }
    [data-testid="stSidebar"] { border-right: 1px solid #F7C8D3; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artefact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_many(artefact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """df columns: craft, material_cost, labor_hours, market_tier (0/1/2)."""
    pred = artefact["model"].predict(df[artefact["features"]])
    lo, hi = artefact["price_band"]
    out = df.copy()
    out["predicted_price"] = pred.round(2)
    out["low"] = (pred * lo).round(2)
    out["high"] = (pred * hi).round(2)
    return out


def predict_price(artefact, craft, material_cost, labor_hours, market_tier) -> dict:
    row = pd.DataFrame([{"craft": craft, "material_cost": material_cost,
                         "labor_hours": labor_hours, "market_tier": market_tier}])
    r = predict_many(artefact, row).iloc[0]
    return {"price": float(r["predicted_price"]), "low": float(r["low"]), "high": float(r["high"])}


# ------------------------------------------------------------------ Authentication gate
# Shows the Log in / Sign up card and stops the script until the user is logged in.
user = require_login(CRAFTS, TIERS)
render_account_sidebar(user, TIERS)

if not MODEL_PATH.exists():
    st.error("Trained model not found. In the project root run:\n\n"
             "`python src/data_generator.py` then `python src/train_model.py`, then restart the app.")
    st.stop()

artefact = load_artefact()

DEFAULT_PRODUCTS = pd.DataFrame(
    {
        "Product": ["Clay Diya Set", "Handloom Scarf", "Wooden Toy Set",
                    "Brass Lamp", "Embroidered Cushion Cover", "Silver-plated Earrings"],
        "Craft": ["Pottery", "Weaving", "Woodwork", "Metalwork", "Embroidery", "Jewellery"],
        "Material_Cost": [120.0, 300.0, 450.0, 900.0, 250.0, 600.0],
        "Labor_Hours": [2.0, 4.0, 6.0, 8.0, 3.0, 5.0],
        "Market": [TIERS[0], TIERS[1], TIERS[1], TIERS[2], TIERS[0], TIERS[2]],
        "Selling_Price": [210.0, 750.0, 1300.0, 3200.0, 520.0, 2400.0],
        "Max_Units": [8, 5, 4, 3, 6, 4],
    }
)

DEFAULT_MATERIALS = pd.DataFrame(
    {"Material": ["Main raw material"], "Quantity": [1.0], "Unit_Price": [300.0]}
)

if "products" not in st.session_state:
    st.session_state.products = DEFAULT_PRODUCTS.copy()
    st.session_state.editor_version = 0

st.title("🧵 Artisano — Production, Pricing & Zero-Waste Decision Support")
st.caption(f"Welcome, {user['full_name'].split()[0]}! Plan today's production with Linear Programming "
           "and price your work with Machine Learning.")

tab_opt, tab_price, tab_about = st.tabs(["📦 Resource Optimizer", "💰 My Product & Price", "✅ How reliable is the price?"])

# ------------------------------------------------------------------ Optimizer
with tab_opt:
    st.subheader("What should I make today?")
    c1, c2 = st.columns(2)
    budget = c1.number_input("Daily material budget (₹)", min_value=0.0, value=5000.0, step=100.0)
    hours = c2.number_input("Working hours available", min_value=0.0, value=24.0, step=1.0)

    st.markdown("**Your product catalogue** (edit, add or delete rows)")
    edited = st.data_editor(
        st.session_state.products,
        key=f"editor_{st.session_state.editor_version}",
        num_rows="dynamic",
        column_config={
            "Craft": st.column_config.SelectboxColumn(options=CRAFTS, required=True),
            "Market": st.column_config.SelectboxColumn(options=TIERS, required=True),
            "Material_Cost": st.column_config.NumberColumn("Material cost (₹)", min_value=0.0),
            "Labor_Hours": st.column_config.NumberColumn("Labour hours", min_value=0.0),
            "Selling_Price": st.column_config.NumberColumn("Selling price (₹)", min_value=0.0),
            "Max_Units": st.column_config.NumberColumn("Max units (demand)", min_value=0, step=1),
        },
    )
    st.session_state.latest_edit = edited   # lets the other tab add rows without losing edits

    b1, b2, _ = st.columns([1.3, 1, 3])
    if b1.button("🤖 Fill prices with ML predictor"):
        df = edited.dropna(subset=["Craft", "Material_Cost", "Labor_Hours", "Market"]).copy()
        if df.empty:
            st.warning("Add at least one complete product row first.")
        else:
            q = pd.DataFrame({
                "craft": df["Craft"],
                "material_cost": df["Material_Cost"],
                "labor_hours": df["Labor_Hours"],
                "market_tier": df["Market"].map(TIER_BY_NAME),
            })
            df["Selling_Price"] = predict_many(artefact, q)["predicted_price"].values
            st.session_state.products = df.reset_index(drop=True)
            st.session_state.editor_version += 1
            st.rerun()
    if b2.button("↩️ Reset catalogue"):
        st.session_state.products = DEFAULT_PRODUCTS.copy()
        st.session_state.editor_version += 1
        st.rerun()

    if st.button("🚀 Optimize my day", type="primary"):
        clean = edited.dropna(subset=["Product", "Material_Cost", "Labor_Hours",
                                      "Selling_Price", "Max_Units"]).copy()
        try:
            res = optimize_production(clean, budget, hours)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        if res.status != "Optimal":
            st.error(f"Solver status: {res.status}")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Maximum profit", f"₹{res.total_profit:,.0f}",
                      help="Selling income minus material cost for the best possible plan today.")
            k2.metric("Revenue", f"₹{res.total_revenue:,.0f}")
            k3.metric("Budget used", f"₹{res.budget_used:,.0f}",
                      f"{res.budget_used / budget:.0%}" if budget else None, delta_color="off")
            k4.metric("Hours used", f"{res.hours_used:.1f} h",
                      f"{res.hours_used / hours:.0%}" if hours else None, delta_color="off")

            w1, w2 = st.columns(2)
            w1.metric("♻️ Unused budget (idle money)", f"₹{res.unused_budget:,.0f}",
                      help="Money you do not need to spend today. It is not lost; it stays with you.")
            w2.metric("♻️ Idle working hours", f"{res.idle_hours:.1f} h",
                      help="Working hours left over with nothing to make.")

            made = res.plan[res.plan["Units_To_Make"] > 0]
            if made.empty:
                st.info("With these limits, no product can be made profitably.")
            else:
                st.markdown("### Optimal production plan")
                st.dataframe(
                    made[["Product", "Units_To_Make", "Total_Cost", "Total_Hours", "Revenue", "Profit"]].round(2),
                    hide_index=True)
                ch1, ch2 = st.columns(2)
                ch1.markdown("**Units to make**")
                ch1.bar_chart(made.set_index("Product")["Units_To_Make"], color="#A8B58A")
                ch2.markdown("**Profit contribution (₹)**")
                ch2.bar_chart(made.set_index("Product")["Profit"], color="#B46A72")

            with st.expander("💡 What would help me earn more?"):
                tips = []
                if res.shadow_price_time > 0.005:
                    tips.append(f"- One **extra working hour** could add about **₹{res.shadow_price_time:,.0f}** profit.")
                else:
                    tips.append("- Extra working hours would not raise your profit right now.")
                if res.shadow_price_budget > 0.005:
                    tips.append(f"- **₹100 more budget** could add about **₹{res.shadow_price_budget * 100:,.0f}** profit.")
                elif res.shadow_price_time > 0.005:
                    tips.append("- More budget would not help right now. Your time, not your money, is the limit.")
                else:
                    tips.append("- More budget would not raise your profit right now.")
                st.markdown("\n".join(tips))
                st.caption(
                    f"Your plan earns ₹{res.total_profit:,.0f}. The theoretical best, if you could make part-pieces, "
                    f"is ₹{res.lp_upper_bound:,.0f}. (Technical name: sensitivity analysis with shadow prices; "
                    "the figures are approximate.)")


with tab_price:
    st.subheader("Enter your own product and get its cost and fair price")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    n1, n2, n3 = st.columns(3)
    pname = n1.text_input("Product name", placeholder="e.g. Terracotta flower pot")
    craft = n2.selectbox("Craft", CRAFTS, index=CRAFTS.index(user["craft"]) if user["craft"] in CRAFTS else 0)
    tier_name = n3.selectbox("Where will you sell it?", TIERS, index=user["market_tier"])

    h1, h2 = st.columns(2)
    lab = h1.number_input("Hours you took to make ONE piece", min_value=0.5, max_value=100.0,
                          value=4.0, step=0.5)
    wage = h2.number_input("Value of your labour (₹ per hour)", min_value=0.0,
                           value=float(CRAFT_RATE[craft]), step=5.0, key=f"wage_{craft}",
                           help="Default is the typical hourly value for this craft. Change it to your own.")

    st.markdown("**Materials used for ONE piece** (add a row for each material)")
    mats = st.data_editor(
        DEFAULT_MATERIALS,
        key="materials",
        num_rows="dynamic",
        column_config={
            "Material": st.column_config.TextColumn("Material"),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0),
            "Unit_Price": st.column_config.NumberColumn("Price per unit (₹)", min_value=0.0),
        },
    )
    mat_cost = float((mats["Quantity"].fillna(0) * mats["Unit_Price"].fillna(0)).sum())

    if mat_cost <= 0:
        st.info("Enter at least one material with a quantity and a price to get the estimate.")
    else:
        labour_value = lab * wage
        making_cost = mat_cost + labour_value
        out = predict_price(artefact, craft, mat_cost, lab, TIER_BY_NAME[tier_name])
        profit = out["price"] - mat_cost

        st.markdown("#### Cost of making one piece")
        a1, a2, a3 = st.columns(3)
        a1.metric("Material cost", f"₹{mat_cost:,.0f}")
        a2.metric("Labour value", f"₹{labour_value:,.0f}", f"{lab:g} h × ₹{wage:,.0f}", delta_color="off")
        a3.metric("Total making cost (break-even price)", f"₹{making_cost:,.0f}",
                  help="The lowest price at which you do not lose money (materials plus your labour).")

        st.markdown("#### Fair selling price (ML prediction)")
        b1_, b2_, b3_ = st.columns(3)
        b1_.metric("Fair selling price", f"₹{out['price']:,.0f}")
        b2_.metric("Likely price range", f"₹{out['low']:,.0f} – ₹{out['high']:,.0f}",
                   help="About 8 out of 10 similar items sell within this range.")
        b3_.metric("Profit over material cost", f"₹{profit:,.0f}", f"{profit / mat_cost:.0%} of material",
                   delta_color="off")

        if not (TRAIN_MATERIAL_RANGE[0] <= mat_cost <= TRAIN_MATERIAL_RANGE[1]) or \
           not (TRAIN_HOURS_RANGE[0] <= lab <= TRAIN_HOURS_RANGE[1]):
            st.warning(
                f"Your material cost or hours are outside the range the model learned "
                f"(material ₹{TRAIN_MATERIAL_RANGE[0]}–₹{TRAIN_MATERIAL_RANGE[1]}, "
                f"hours {TRAIN_HOURS_RANGE[0]}–{TRAIN_HOURS_RANGE[1]}). Treat this price as a rough guide.")
        if out["price"] < making_cost:
            st.warning("The predicted market price is below your full making cost (material + labour). "
                       "Consider a higher-value market, a faster method, or cheaper materials.")

        st.markdown("**How the price changes with labour hours, per market**")
        grid = pd.DataFrame({"labor_hours": [h / 2 for h in range(1, 33)]})
        curves = {}
        for t, name in TIER_NAMES.items():
            g = grid.assign(craft=craft, material_cost=float(mat_cost), market_tier=t)
            curves[name] = predict_many(artefact, g).set_index("labor_hours")["predicted_price"]
        st.line_chart(pd.DataFrame(curves), color=["#A8B58A", "#B46A72", "#2D3A47"])

        st.markdown("#### Use this product in the optimizer")
        u1, u2 = st.columns([1, 3])
        max_units = u1.number_input("Max units you can sell per day", min_value=1, value=5, step=1)
        if u2.button("➕ Add this product to the optimizer catalogue"):
            name = pname.strip() or f"{craft} item"
            new = pd.DataFrame([{
                "Product": name, "Craft": craft, "Material_Cost": round(mat_cost, 2),
                "Labor_Hours": float(lab), "Market": tier_name,
                "Selling_Price": out["price"], "Max_Units": int(max_units),
            }])
            base = st.session_state.get("latest_edit", st.session_state.products)
            df = pd.concat([base, new], ignore_index=True)
            df["Max_Units"] = df["Max_Units"].fillna(0).astype(int)
            st.session_state.products = df
            st.session_state.editor_version += 1
            st.session_state.flash = f"'{name}' added. Open the Resource Optimizer tab to include it in today's plan."
            st.rerun()


with tab_about:
    st.subheader("How reliable are the price estimates?")
    mt = artefact["metrics"]
    st.info(
        "We checked the price predictor on products it had never seen before. Its estimate was usually "
        f"within about **{mt['MAPE_%']:.0f}%** of the real price (around **₹{mt['MAE']:.0f}** on average). "
        "Use it as a fair starting price, then adjust for your own situation."
    )
    a, b, c = st.columns(3)
    a.metric("Typical error", f"±{mt['MAPE_%']:.1f}%",
             help="On average the estimate is off by this share of the price. Lower is better.")
    b.metric("Average miss", f"₹{mt['MAE']:.0f}",
             help="On average the estimate is this many rupees above or below the real price. Lower is better.")
    c.metric("Price differences explained", f"{mt['R2']:.0%}",
             help="How much of the difference between cheap and expensive items the model captures. Higher is better.")

    st.markdown("**What affects the price most?**")
    raw = pd.Series(artefact["feature_importance"])
    groups = {
        "Craft type": raw[[i for i in raw.index if i.startswith("craft__")]].sum(),
        "Material cost": raw.get("remainder__material_cost", 0.0),
        "Labour hours": raw.get("remainder__labor_hours", 0.0),
        "Market (where you sell)": raw.get("remainder__market_tier", 0.0),
    }
    imp = (pd.Series(groups) * 100).sort_values(ascending=False)
    st.bar_chart(imp, color="#2D3A47")
    st.caption("Share of the price decision (%) that each factor accounts for.")

    with st.expander("🔬 Technical details (for evaluation)"):
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("R²", f"{mt['R2']:.3f}")
        t2.metric("MAE", f"₹{mt['MAE']:.0f}")
        t3.metric("RMSE", f"₹{mt['RMSE']:.0f}")
        t4.metric("MAPE", f"{mt['MAPE_%']:.1f}%")
        st.caption(
            "Measured on a held-out 20% test set. R² = share of price variance explained; "
            "MAE = mean absolute error; RMSE = root mean squared error (penalises large misses); "
            "MAPE = mean absolute percentage error. "
            f"5-fold cross-validated R² on the training set: {mt['CV_R2_mean']:.3f}."
        )
    st.caption("The model is trained on synthetic data; real sales records would be needed for real-world use.")