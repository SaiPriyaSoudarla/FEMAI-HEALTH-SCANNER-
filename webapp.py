import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import pickle
from PIL import Image

# -----------------------------
# Page setup & theming
# -----------------------------
st.set_page_config(
    page_title="FemAI Health Scanner",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Subtle glassmorphism styling with lavender & pastel bubbles
st.markdown(
    """
    <style>
        :root {
            --primary: #8b5cf6;
            --secondary: #c084fc;
            --mint: #b5f5ec;
            --surface: rgba(255, 255, 255, 0.8);
            --border: rgba(255, 255, 255, 0.4);
            --text: #1f1b2d;
            --heading: #6b4ae6;
        }
        * {
            font-family: "Calibri", "Cambria", "Gabriola", "Segoe UI", system-ui, -apple-system, sans-serif;
        }
        h1, h2, h3, h4 {
            color: var(--heading);
            letter-spacing: -0.01em;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: var(--heading) !important;
        }
        .css-10trblm, .css-1v3fvcr {
            color: var(--heading) !important;
        }
        .stApp {
            background:
                radial-gradient(120% 70% at 15% 0%, rgba(203, 197, 255, 0.55), transparent 60%),
                radial-gradient(140% 80% at 85% 0%, rgba(255, 255, 255, 0.75), transparent 60%),
                linear-gradient(135deg, #f8f7ff 0%, #f4f4ff 35%, #f8fbff 70%, #eef7ff 100%);
            color: var(--text);
            position: relative;
            overflow: hidden;
            min-height: 100vh;
            padding: 8px 0 28px 0;
        }
        /* floating pastel bubbles / snowflakes */
        .stApp::before,
        .stApp::after {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                radial-gradient(circle at 10% 20%, rgba(196, 181, 255, 0.28) 0 18px, transparent 20px),
                radial-gradient(circle at 80% 18%, rgba(181, 240, 255, 0.25) 0 14px, transparent 16px),
                radial-gradient(circle at 30% 65%, rgba(201, 255, 240, 0.25) 0 18px, transparent 20px),
                radial-gradient(circle at 70% 72%, rgba(210, 194, 255, 0.22) 0 22px, transparent 24px);
            animation: drift 18s linear infinite;
            opacity: 0.75;
            pointer-events: none;
            z-index: 0;
        }
        .stApp::after {
            animation-duration: 24s;
            filter: blur(2px);
            opacity: 0.55;
            z-index: 0;
        }
        @keyframes drift {
            from { transform: translateY(0px); }
            to { transform: translateY(-40px); }
        }
        /* ensure content stays above decorative layers */
        .main, .block-container {
            position: relative;
            z-index: 1;
        }
        .glass-card {
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: 0 20px 60px rgba(82, 67, 170, 0.12);
            border-radius: 18px;
            padding: 22px 24px;
            backdrop-filter: blur(10px);
        }
        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #8b5cf6 0%, #c084fc 40%, #9ef3e7 100%);
            color: white;
            font-weight: 600;
            box-shadow: 0 10px 28px rgba(139, 92, 246, 0.35);
        }
        .section-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #7c3aed;
            font-weight: 700;
            margin-bottom: -6px;
        }
        .stButton>button {
            border-radius: 14px;
            background: linear-gradient(120deg, #7c3aed, #c084fc, #9ef3e7);
            color: white;
            border: none;
            font-weight: 700;
            padding: 12px 18px;
            box-shadow: 0 12px 32px rgba(124, 58, 237, 0.35);
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(124, 58, 237, 0.4);
        }
        .stNumberInput input, .stTextInput input, .stSelectbox select {
            border-radius: 12px;
            border: 1px solid #e6e0ff;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.12);
        }
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #e2d9ff, transparent);
            margin: 6px 0 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load model and dataset stats
# -----------------------------
# Load the trained ML model
pickled_model = pickle.load(open("modelfinal_new.pkl", "rb"))

# Load dataset to get feature medians (for default values)
df_raw = pd.read_excel("PCOS_data_without_infertility.xlsx", sheet_name=1)

# Drop unnecessary columns (same as in training)
cols_to_drop = [c for c in df_raw.columns if "Sl" in c or "File" in c or "Unnamed" in c]
df = df_raw.drop(columns=cols_to_drop, errors="ignore")

TARGET_COL_NAME = "PCOS (Y/N)"
X_cols = [c for c in df.columns if c != TARGET_COL_NAME]

# Convert to numeric and compute medians for features
df_X = df[X_cols].apply(pd.to_numeric, errors="coerce")
feature_medians = df_X.median(numeric_only=True)

# -----------------------------
# UI: Title and intro
# -----------------------------
hero_left, hero_right = st.columns([1.25, 0.75])
with hero_left:
    st.markdown('<div class="section-label">PCOS companion</div>', unsafe_allow_html=True)
    st.title("✨ FemAI Health Scanner")
    st.subheader("Smarter PCOS screening with calm, guided inputs")
    st.write(
        """
        FemAI blends a clinician-trained model with gentle guidance to help you
        understand PCOS risk in seconds. Thoughtful defaults keep things simple,
        while your entries refine the assessment.
        """
    )
    st.markdown('<span class="metric-pill">💡 Simple • Fast • Private</span>', unsafe_allow_html=True)
with hero_right:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight:700;font-size:1rem;color:#6c5ce7;">Wellness snapshot</div>
            <p style="margin:4px 0 10px;color:#475569;">A friendly checklist that auto-fills advanced fields so you can focus on what matters.</p>
            <ul style="padding-left:18px; margin: 0; color:#0f172a;">
                <li>Evidence-based defaults from dataset medians</li>
                <li>Instant BMI + lifestyle cues</li>
                <li>Visual feedback & risk probability</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Age vs share chart (kept from original)
# -----------------------------
st.markdown('<div class="section-label">Snapshot</div>', unsafe_allow_html=True)
chart_col, tips_col = st.columns([1.4, 1])
with chart_col:
    st.subheader("Age vs Respondent Share")
    chart_df = pd.DataFrame(
        {
            "Age group": ["<19", "20-29", "30-44", "45-59", "60>"],
            "Percentage": [3.8, 16.81, 11.58, 1.44, 0.55],
        }
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Age group", title="Age group"),
            y=alt.Y("Percentage", title="Percentage"),
            color=alt.Color("Age group", scale=alt.Scale(scheme="tableau20")),
            text=alt.Text("Percentage", format=".1f"),
        )
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
        .properties(width=alt.Step(46))
    )
    st.altair_chart(chart, use_container_width=True)
with tips_col:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight:700;margin-bottom:6px;">Quick tips</div>
            <ul style="padding-left:18px; margin: 0; color:#0f172a;">
                <li>Balanced diet + regular movement reduce metabolic risk.</li>
                <li>Track cycles to spot irregularities early.</li>
                <li>Consult a specialist for personalized care.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Simplified diagnosis section
# -----------------------------
st.markdown('<div class="section-label">Assessment</div>', unsafe_allow_html=True)
st.title("Start your diagnosis")
st.write(
    "Fill these guided fields; advanced medical values are auto-handled in the background."
)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    basic_col, lifestyle_col = st.columns(2)
    with basic_col:
        st.write("### Basics")
        age = st.number_input(
            "Age (years)", min_value=10, max_value=50, value=22, step=1, help="Typical screening range"
        )
        weight = st.number_input(
            "Weight (Kg)", min_value=30.0, max_value=150.0, value=55.0, step=0.5
        )
        height = st.number_input(
            "Height (Cm)", min_value=130.0, max_value=190.0, value=160.0, step=1.0
        )
        cycle_type = st.selectbox("Menstrual Cycle", ["Regular", "Irregular"])
        cycle_length = st.slider("Average cycle length (days)", 20, 60, 30)

    with lifestyle_col:
        st.write("### Lifestyle & symptoms")
        weight_gain = st.selectbox("Recent weight gain?", ["No", "Yes"])
        hair_growth = st.selectbox("Increased facial/body hair?", ["No", "Yes"])
        skin_dark = st.selectbox("Skin darkening (neck / underarms)?", ["No", "Yes"])
        pimples = st.selectbox("Pimples / acne?", ["No", "Yes"])
        fast_food = st.selectbox("Frequent fast food intake?", ["No", "Yes"])
        exercise = st.selectbox("Regular exercise?", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

# Helper to convert Yes/No to 0/1
def yn(x: str) -> int:
    return 1 if x == "Yes" else 0

# -----------------------------
# Build feature vector for model
# -----------------------------
# Start from dataset medians (so complex clinical fields get safe defaults)
feat = feature_medians.copy()

# Override with user-friendly inputs mapped to correct column names
# NOTE: these column names come directly from your Excel file
if " Age (yrs)" in feat.index:
    feat[" Age (yrs)"] = age
if "Weight (Kg)" in feat.index:
    feat["Weight (Kg)"] = weight
if "Height(Cm) " in feat.index:
    feat["Height(Cm) "] = height  # note the space in original column

# BMI calculation
bmi = weight / ((height / 100) ** 2)
if "BMI" in feat.index:
    feat["BMI"] = bmi

# Cycle info
if "Cycle(R/I)" in feat.index:
    feat["Cycle(R/I)"] = 0 if cycle_type == "Regular" else 1
if "Cycle length(days)" in feat.index:
    feat["Cycle length(days)"] = cycle_length

# Symptom-based Y/N fields
mapping_updates = {
    "Weight gain(Y/N)": yn(weight_gain),
    "hair growth(Y/N)": yn(hair_growth),
    "Skin darkening (Y/N)": yn(skin_dark),
    "Pimples(Y/N)": yn(pimples),
    "Fast food (Y/N)": yn(fast_food),
    "Reg.Exercise(Y/N)": yn(exercise),
}
for col_name, val in mapping_updates.items():
    if col_name in feat.index:
        feat[col_name] = val

# Prepare model input in the same column order as training
input_row = [feat[c] for c in X_cols]
input_df = pd.DataFrame([input_row], columns=X_cols)

# -----------------------------
# Prediction button
# -----------------------------
action_col, info_col = st.columns([1.1, 1])
with action_col:
    if st.button("Check PCOS Risk"):
        try:
            pred = pickled_model.predict(input_df)[0]
            prob_text = ""
            if hasattr(pickled_model, "predict_proba"):
                prob = pickled_model.predict_proba(input_df)[0][1]
                prob_text = f" (estimated risk: {prob*100:.1f}%)"

            st.success("Diagnosis completed ✅")

            if int(pred) == 1:
                st.error(
                    f"High likelihood of PCOS detected{prob_text}. Please consult a gynecologist for further evaluation."
                )
            else:
                st.success(
                    f"Low likelihood of PCOS{prob_text}. Maintain a healthy lifestyle and regular check-ups."
                )
        except Exception as e:
            st.error("Something went wrong while generating the prediction.")
            st.write(e)
with info_col:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight:700;">How results are built</div>
            <p style="margin:6px 0 0; color:#475569;">
                We start from evidence-based medians for complex lab values,
                then personalize with your inputs. Probability is shown when the
                model supports it.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Show key derived info
# -----------------------------
st.markdown(
    f"""
    <div class="glass-card" style="margin-top:12px;">
        <div style="font-weight:700; color:#1f2937;">Key derived metric</div>
        <p style="margin:6px 0 2px;">Calculated BMI: <b>{bmi:.2f}</b></p>
        <p style="margin:0; color:#475569;">BMI is auto-calculated from your height and weight.</p>
        <p style="margin:6px 0 0; color:#0f172a;">✅ Advanced hormone and scan values are handled internally using dataset statistics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
