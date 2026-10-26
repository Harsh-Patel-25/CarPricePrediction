import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
import sklearn.compose._column_transformer as ct

# Backward compatibility shim for model serialization across scikit-learn versions
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _RemainderColsList.__module__ = "sklearn.compose._column_transformer"
    ct._RemainderColsList = _RemainderColsList

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoValuate | AI Car Valuation & Market Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CUSTOM CSS / DESIGN SYSTEM
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container clean padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    /* Custom Header Banner */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    .hero-badge {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Glass Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        border-color: rgba(14, 165, 233, 0.3);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }

    .metric-sub {
        font-size: 0.82rem;
        color: #10b981;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* Section Header */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .section-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }

    /* Valuation Result Card */
    .valuation-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.3);
    }

    .valuation-title {
        color: #38bdf8;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .valuation-price {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0;
        letter-spacing: -0.03em;
    }

    .valuation-inr {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 500;
    }

    .range-pill {
        display: inline-block;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.25);
        color: #7dd3fc;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Form styling */
    div[data-testid="stForm"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.8rem;
        background: rgba(255, 255, 255, 0.02);
    }

    /* Submit Button styling */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.55);
    }

    /* Tabs partition custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        color: #94a3b8;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: #0284c7 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Load and repair scikit-learn pipeline."""
    model_obj = joblib.load("car_price_model.pkl")

    def fix_imputer(obj):
        if hasattr(obj, "statistics_") and not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = getattr(obj, "_fit_dtype", getattr(obj.statistics_, "dtype", None))
        if hasattr(obj, "named_steps"):
            for s in obj.named_steps.values():
                fix_imputer(s)
        if hasattr(obj, "transformers_"):
            for _, t, _ in obj.transformers_:
                fix_imputer(t)
        if hasattr(obj, "steps"):
            for _, s in obj.steps:
                fix_imputer(s)

    fix_imputer(model_obj)
    return model_obj


@st.cache_data
def load_market_data():
    """Load and preprocess market dataset."""
    data = pd.read_csv("cleaned_car_data.csv")

    current_yr = pd.Timestamp.now().year
    if "Car_Age" not in data.columns:
        data["Car_Age"] = current_yr - data["Year"]

    data["Brand"] = data["Name"].astype(str).str.split().str[0]
    return data


# Load artifacts
try:
    model = load_model()
    market_df = load_market_data()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    model_loaded = False
    market_df = pd.DataFrame()


# ---------------------------------------------------------
# HERO BANNER / BRANDING
# ---------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div>
        <h1 class="hero-title">Car Price Prediction & Market Analysis</h1>
        <div class="hero-subtitle">Machine Learning Car Valuation Engine & Used Automobile Market Intelligence</div>
    </div>
    <div>
        <span class="hero-badge">● Production Model Active</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TOP NAVIGATION PARTITION TABS
# ---------------------------------------------------------
tab_predict, tab_analytics, tab_inventory, tab_info = st.tabs([
    "🚗 Price Predictor",
    "📊 Market Insights & Analytics",
    "🔍 Vehicle Inventory Explorer",
    "ℹ️ Model & Methodology"
])


# =========================================================
# TAB 1: PRICE PREDICTOR
# =========================================================
with tab_predict:
    st.markdown('<div class="section-title"><span>🎯</span> Instant Resale Value Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Provide vehicle specifications to generate a data-driven fair market valuation based on thousands of real-world resale transactions.</div>', unsafe_allow_html=True)

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        with st.form("car_valuation_form"):
            st.markdown("#### 1. Identity & Registration")
            id_c1, id_c2 = st.columns(2)
            with id_c1:
                location = st.selectbox(
                    "Metro / City",
                    ["Mumbai", "Delhi", "Bangalore", "Hyderabad",
                     "Chennai", "Pune", "Kolkata", "Ahmedabad",
                     "Jaipur", "Kochi"],
                    index=0,
                    help="Regional demand heavily influences resale price."
                )
                year = st.selectbox(
                    "Manufacturing Year",
                    list(range(2025, 1999, -1)),
                    index=7,  # 2018 default
                    help="Year of manufacture affects age and depreciation."
                )
            with id_c2:
                owner_type = st.selectbox(
                    "Ownership Record",
                    ["First", "Second", "Third", "Fourth & Above"],
                    index=0,
                    help="Single-owner vehicles command a significant premium."
                )
                seats = st.selectbox(
                    "Seating Capacity",
                    [2, 4, 5, 6, 7, 8, 9, 10],
                    index=2,  # 5 seats default
                    help="Factory certified seating capacity."
                )

            st.divider()

            st.markdown("#### 2. Powertrain & Transmission")
            pt_c1, pt_c2 = st.columns(2)
            with pt_c1:
                fuel_type = st.selectbox(
                    "Fuel Type",
                    ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
                    index=0
                )
                engine = st.number_input(
                    "Engine Displacement (CC)",
                    min_value=500.0,
                    max_value=10000.0,
                    value=1197.0,
                    step=50.0,
                    help="E.g. 1197cc (1.2L) for hatchbacks, 1995cc (2.0L) for sedans/SUVs."
                )
            with pt_c2:
                transmission = st.selectbox(
                    "Transmission Type",
                    ["Manual", "Automatic"],
                    index=0
                )
                power = st.number_input(
                    "Peak Power Output (bhp)",
                    min_value=30.0,
                    max_value=1200.0,
                    value=83.0,
                    step=5.0,
                    help="Engine brake horsepower (bhp)."
                )

            st.divider()

            st.markdown("#### 3. Usage & Efficiency")
            us_c1, us_c2 = st.columns(2)
            with us_c1:
                kilometers = st.number_input(
                    "Odometer (Kilometers Driven)",
                    min_value=0,
                    max_value=1000000,
                    value=45000,
                    step=1000,
                    help="Total recorded kilometers on the odometer."
                )
            with us_c2:
                mileage = st.number_input(
                    "Certified Fuel Economy (kmpl / km/kg)",
                    min_value=5.0,
                    max_value=50.0,
                    value=18.5,
                    step=0.5,
                    help="ARA-rated fuel consumption efficiency."
                )

            predict_submitted = st.form_submit_button("⚡ Calculate Fair Market Valuation", use_container_width=True)

    with result_col:
        current_year = pd.Timestamp.now().year
        car_age = current_year - year

        if predict_submitted and model_loaded:
            input_data = pd.DataFrame([{
                "Location": location,
                "Year": year,
                "Kilometers_Driven": kilometers,
                "Fuel_Type": fuel_type,
                "Transmission": transmission,
                "Owner_Type": owner_type,
                "Mileage": mileage,
                "Engine": engine,
                "Power": power,
                "Seats": seats,
                "Car_Age": car_age
            }])

            try:
                prediction = float(model.predict(input_data)[0])

                if prediction <= 0:
                    st.warning("⚠️ The inputs produced a negative valuation. Please review extreme odometer readings or car age.")
                else:
                    inr_value = prediction * 100000
                    low_range = prediction * 0.93
                    high_range = prediction * 1.07

                    st.markdown(f"""
                    <div class="valuation-box">
                        <div class="valuation-title">Estimated Fair Market Resale Value</div>
                        <div class="valuation-price">₹ {prediction:.2f} Lakh</div>
                        <div class="valuation-inr">≈ ₹ {int(inr_value):,} INR</div>
                        <div class="range-pill">Suggested Deal Window: ₹ {low_range:.2f}L — ₹ {high_range:.2f}L</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")
                    # Key Vehicle Health Indicators
                    v1, v2 = st.columns(2)
                    with v1:
                        avg_km_per_year = kilometers / max(1, car_age)
                        usage_status = "Light" if avg_km_per_year < 8000 else "Average" if avg_km_per_year < 15000 else "Heavy"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Annualized Usage</div>
                            <div class="metric-value">{int(avg_km_per_year):,} <span style="font-size:0.9rem;color:#94a3b8">km/yr</span></div>
                            <div class="metric-sub">{usage_status} Wear & Tear Profile</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with v2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Vehicle Age</div>
                            <div class="metric-value">{car_age} <span style="font-size:0.9rem;color:#94a3b8">years</span></div>
                            <div class="metric-sub">Manufactured {year}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Quick comparison against market
                    if not market_df.empty:
                        similars = market_df[
                            (market_df["Fuel_Type"] == fuel_type) &
                            (market_df["Transmission"] == transmission)
                        ]
                        if not similars.empty:
                            median_sim = similars["Price"].median()
                            diff = prediction - median_sim
                            rel_text = f"+₹{diff:.2f}L vs {fuel_type} {transmission} median" if diff >= 0 else f"-₹{abs(diff):.2f}L vs {fuel_type} {transmission} median"
                            st.info(f"💡 **Market Benchmark**: {rel_text} (Segment median: ₹{median_sim:.2f} Lakhs).")

            except Exception as ex:
                st.error(f"Valuation calculation failed: {ex}")

        else:
            # Standby Preview Card
            st.markdown(f"""
            <div class="valuation-box" style="border-style: dashed; border-color: rgba(255, 255, 255, 0.15);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚘</div>
                <div class="valuation-title" style="color: #94a3b8;">Valuation Engine Ready</div>
                <p style="color: #64748b; font-size: 0.95rem; margin-top: 0.5rem;">
                    Configure your car's details on the left and click <b>Calculate Fair Market Valuation</b> to produce an instant ML price estimate.
                </p>
                <div class="range-pill" style="background: rgba(255, 255, 255, 0.05); color: #cbd5e1; border-color: rgba(255,255,255,0.1);">
                    Trained on 6,000+ Verified Market Transactions
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("##### 📌 Key Resale Determinants")
            st.markdown("""
            - **Depreciation Curve**: Vehicles typically lose 15-20% value in Year 1, stabilizing to ~8-10% annually thereafter.
            - **Fuel & Transmission**: Automatics and high-efficiency Petrol/Diesels command sustained resale liquidity in metro hubs.
            - **Ownership History**: First-owner cars maintain up to 12% higher market confidence than multi-transfer vehicles.
            """)


# =========================================================
# TAB 2: MARKET INSIGHTS & ANALYTICS
# =========================================================
with tab_analytics:
    st.markdown('<div class="section-title"><span>📊</span> Comprehensive Used Car Market Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Analyze pricing trends, depreciation trajectories, and consumer demand patterns across Indian urban centers.</div>', unsafe_allow_html=True)

    if not market_df.empty:
        # Filter Partition Card
        with st.expander("🔍 Interactive Market Filters & Slicers", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)

            with f_col1:
                city_options = ["All Cities"] + sorted(market_df["Location"].dropna().unique().tolist())
                f_city = st.selectbox("Location", city_options, key="analytics_city")

            with f_col2:
                fuel_options = ["All Fuels"] + sorted(market_df["Fuel_Type"].dropna().unique().tolist())
                f_fuel = st.selectbox("Fuel Type", fuel_options, key="analytics_fuel")

            with f_col3:
                trans_options = ["All Transmissions"] + sorted(market_df["Transmission"].dropna().unique().tolist())
                f_trans = st.selectbox("Transmission", trans_options, key="analytics_trans")

            with f_col4:
                owner_options = ["All Owners"] + sorted(market_df["Owner_Type"].dropna().unique().tolist())
                f_owner = st.selectbox("Ownership", owner_options, key="analytics_owner")

        # Apply Filters
        adf = market_df.copy()
        if f_city != "All Cities":
            adf = adf[adf["Location"] == f_city]
        if f_fuel != "All Fuels":
            adf = adf[adf["Fuel_Type"] == f_fuel]
        if f_trans != "All Transmissions":
            adf = adf[adf["Transmission"] == f_trans]
        if f_owner != "All Owners":
            adf = adf[adf["Owner_Type"] == f_owner]

        # Top KPI Ribbon
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Listings</div>
                <div class="metric-value">{len(adf):,}</div>
                <div class="metric-sub">{(len(adf)/len(market_df)*100):.1f}% of total inventory</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            avg_p = adf["Price"].mean() if not adf.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Selling Price</div>
                <div class="metric-value">₹ {avg_p:.2f}L</div>
                <div class="metric-sub">≈ ₹ {int(avg_p * 100000):,} INR</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            med_p = adf["Price"].median() if not adf.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Median Selling Price</div>
                <div class="metric-value">₹ {med_p:.2f}L</div>
                <div class="metric-sub">Midpoint transaction value</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            avg_a = adf["Car_Age"].mean() if not adf.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Fleet Age</div>
                <div class="metric-value">{avg_a:.1f} <span style="font-size:0.9rem;color:#94a3b8">years</span></div>
                <div class="metric-sub">Mean lifecycle duration</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        if not adf.empty:
            # Row 1 Charts: Price Distribution & Fuel Types
            c1, c2 = st.columns(2)
            with c1:
                fig_dist = px.histogram(
                    adf,
                    x="Price",
                    nbins=40,
                    title="📈 Resale Price Distribution (Lakhs)",
                    color_discrete_sequence=["#0284c7"],
                    template="plotly_dark",
                    labels={"Price": "Price (₹ Lakhs)"}
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with c2:
                fuel_agg = (
                    adf.groupby("Fuel_Type", as_index=False)
                    .agg(Average_Price=("Price", "mean"), Count=("Price", "size"))
                    .sort_values("Average_Price", ascending=False)
                )
                fig_fuel = px.bar(
                    fuel_agg,
                    x="Fuel_Type",
                    y="Average_Price",
                    color="Average_Price",
                    color_continuous_scale="Viridis",
                    title="⛽ Average Resale Price by Fuel Type",
                    template="plotly_dark",
                    labels={"Average_Price": "Avg Price (₹ Lakhs)", "Fuel_Type": "Fuel Type"}
                )
                fig_fuel.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_fuel, use_container_width=True)

            # Row 2 Charts: Depreciation Curve & Top Brands
            c3, c4 = st.columns(2)
            with c3:
                age_grp = (
                    adf.groupby("Car_Age", as_index=False)
                    .agg(Median_Price=("Price", "median"), Volume=("Price", "size"))
                )
                age_grp = age_grp[age_grp["Volume"] >= 3]

                fig_deprec = px.line(
                    age_grp,
                    x="Car_Age",
                    y="Median_Price",
                    markers=True,
                    title="📉 Vehicle Depreciation Curve (Car Age vs Median Price)",
                    template="plotly_dark",
                    labels={"Car_Age": "Car Age (Years)", "Median_Price": "Median Price (₹ Lakhs)"}
                )
                fig_deprec.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=8, color="#0ea5e9"))
                fig_deprec.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_deprec, use_container_width=True)

            with c4:
                brand_agg = (
                    adf.groupby("Brand", as_index=False)
                    .agg(Listings=("Price", "size"), Median_Price=("Price", "median"))
                    .sort_values("Listings", ascending=False)
                    .head(10)
                )
                fig_brand = px.bar(
                    brand_agg,
                    x="Brand",
                    y="Listings",
                    color="Median_Price",
                    color_continuous_scale="Teal",
                    title="🏆 Top 10 Car Brands by Listing Volume",
                    template="plotly_dark",
                    labels={"Listings": "Volume (Count)", "Median_Price": "Median Price (₹L)"}
                )
                fig_brand.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_brand, use_container_width=True)

            # Row 3: Metro Price Comparison
            loc_agg = (
                adf.groupby("Location", as_index=False)
                .agg(Average_Price=("Price", "mean"), Median_Price=("Price", "median"), Listings=("Price", "size"))
                .sort_values("Average_Price", ascending=False)
            )
            fig_loc = px.bar(
                loc_agg,
                x="Location",
                y="Average_Price",
                hover_data=["Median_Price", "Listings"],
                color="Average_Price",
                color_continuous_scale="Blues",
                title="🏙️ Resale Valuation Across Major Indian Metro Hubs",
                template="plotly_dark",
                labels={"Average_Price": "Average Price (₹ Lakhs)", "Location": "Metro City"}
            )
            fig_loc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                margin=dict(l=20, r=20, t=50, b=20),
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        else:
            st.warning("No listings match the chosen filter combinations. Try loosening the filter criteria.")

    else:
        st.warning("Market dataset not found.")


# =========================================================
# TAB 3: VEHICLE INVENTORY EXPLORER
# =========================================================
with tab_inventory:
    st.markdown('<div class="section-title"><span>🔍</span> Verified Vehicle Inventory Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Search, filter, and inspect verified secondary market listings to benchmark real-world vehicle offerings.</div>', unsafe_allow_html=True)

    if not market_df.empty:
        s_col1, s_col2, s_col3 = st.columns([1.5, 1, 1])
        with s_col1:
            search_query = st.text_input("🔎 Search Vehicle Name or Model", "", placeholder="e.g. Swift, Creta, City, Fortuner...")
        with s_col2:
            sort_by = st.selectbox("Sort By", ["Price (High to Low)", "Price (Low to High)", "Year (Newest First)", "Kilometers (Lowest First)"])
        with s_col3:
            price_bracket = st.slider("Max Price (₹ Lakhs)", min_value=1.0, max_value=float(market_df["Price"].max()), value=float(market_df["Price"].max()))

        # Filter dataset
        idf = market_df[market_df["Price"] <= price_bracket].copy()
        if search_query.strip():
            idf = idf[idf["Name"].str.contains(search_query.strip(), case=False, na=False)]

        # Sort
        if sort_by == "Price (High to Low)":
            idf = idf.sort_values("Price", ascending=False)
        elif sort_by == "Price (Low to High)":
            idf = idf.sort_values("Price", ascending=True)
        elif sort_by == "Year (Newest First)":
            idf = idf.sort_values("Year", ascending=False)
        elif sort_by == "Kilometers (Lowest First)":
            idf = idf.sort_values("Kilometers_Driven", ascending=True)

        st.caption(f"Showing {len(idf):,} matching records.")

        display_cols = [
            "Name", "Location", "Year", "Fuel_Type", "Transmission",
            "Owner_Type", "Kilometers_Driven", "Mileage", "Engine", "Power", "Price"
        ]
        available_cols = [c for c in display_cols if c in idf.columns]

        st.dataframe(
            idf[available_cols].rename(columns={
                "Kilometers_Driven": "Kms Driven",
                "Price": "Price (₹ Lakhs)",
                "Owner_Type": "Owner",
                "Fuel_Type": "Fuel",
                "Transmission": "Gearbox"
            }),
            use_container_width=True,
            height=450
        )

        # CSV Download Button
        csv_bytes = idf[available_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Inventory (CSV)",
            data=csv_bytes,
            file_name="used_cars_filtered_inventory.csv",
            mime="text/csv"
        )
    else:
        st.warning("Inventory dataset not available.")


# =========================================================
# TAB 4: MODEL & METHODOLOGY
# =========================================================
with tab_info:
    st.markdown('<div class="section-title"><span>ℹ️</span> Valuation Architecture & Machine Learning Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Technical specification of the underlying machine learning model, feature encoding, and data engineering workflow.</div>', unsafe_allow_html=True)

    info_c1, info_c2 = st.columns(2, gap="large")

    with info_c1:
        st.markdown("""
        #### 🤖 Estimator Architecture
        - **Model Algorithm**: `RandomForestRegressor` with an ensemble of decision trees to capture non-linear depreciation dynamics.
        - **Preprocessing Pipeline**:
          - Numerical Features: Median Imputation (`SimpleImputer(strategy='median')`)
          - Categorical Features: Categorical imputation followed by one-hot vectorization (`OneHotEncoder(handle_unknown='ignore')`).
        - **Target Variable**: Resale transaction price in Indian Lakhs (1 Lakh = ₹100,000 INR).
        """)

        st.markdown("""
        #### 📊 Feature Matrix
        | Feature Name | Type | Description |
        | :--- | :--- | :--- |
        | **Location** | Categorical | Metro region where car is registered |
        | **Year / Car_Age** | Numerical | Vehicle vintage & age in years |
        | **Kilometers_Driven** | Numerical | Total odometer distance |
        | **Fuel_Type** | Categorical | Petrol, Diesel, CNG, LPG, Electric |
        | **Transmission** | Categorical | Manual vs Automatic gearbox |
        | **Owner_Type** | Categorical | First, Second, Third, Fourth+ |
        | **Mileage** | Numerical | Fuel efficiency (kmpl or km/kg) |
        | **Engine** | Numerical | Engine displacement in CC |
        | **Power** | Numerical | Maximum engine power in bhp |
        | **Seats** | Numerical | Certified passenger seating count |
        """)

    with info_c2:
        st.markdown("""
        #### 📈 Valuation Governance & Guidelines
        - **Market Dynamics**: Prices generated represent fair dealer and private-party resale benchmarks in standard roadworthy condition.
        - **Outlier Mitigation**: Vehicles with salvage titles, severe accidental damage, or extreme odometer tampering require manual mechanical appraisal.
        - **Metro Variance**: High-density urban corridors (e.g. Mumbai, Bangalore, Delhi) exhibit higher price elasticity for automatic and compact vehicles.
        """)

        st.info("💡 **Tip**: When selling privately, cars typically trade 5%–10% higher than dealer trade-in offers, corresponding to the estimated dealer margin.")

st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    AutoValuate AI &copy; 2026 &bull; Advanced Machine Learning Automobile Valuation System &bull; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
