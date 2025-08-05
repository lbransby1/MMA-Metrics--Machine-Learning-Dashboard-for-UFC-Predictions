import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import requests
import pickle
from bs4 import BeautifulSoup
from datetime import datetime
from comparison_table import render_comparison_table
import matplotlib.pyplot as plt

# Set wide layout and dashboard title
st.set_page_config(layout="wide", page_title="UFC Fight Predictor Dashboard")
st.title("UFC Fight Predictor Dashboard")
st.markdown("Compare fighters, analyze stats, and predict outcomes using AI and SHAP explanations.")

# Load and clean fighter data
fighters_df = pd.read_csv("processed_data/fighter_averages.csv")
fighters_df["Name"] = fighters_df["Name"].astype(str)
fighter_names = sorted(fighters_df["Name"].dropna().unique())

# Convert DOB to Age
fighters_df["DOB"] = pd.to_datetime(fighters_df["DOB"], errors='coerce')
today = pd.Timestamp.today()
fighters_df["Age"] = fighters_df["DOB"].apply(lambda dob: (today - dob).days // 365 if pd.notnull(dob) else np.nan)

# Averaged stats only
avg_stats = [col for col in fighters_df.columns if (
    "PerMin" in col or "Per15Min" in col or "AccuracyPct" in col or "DefencePct" in col or "Pct" in col)
    and col != "KnockdownPct"]

@st.cache_data(show_spinner=False)
def get_ufc_image(fighter_name):
    base_url = "https://www.ufc.com/athlete/"
    slug = fighter_name.lower().replace(" ", "-")
    url = base_url + slug
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', class_='hero-profile__image')
        return img_tag['src'] if img_tag else None
    except Exception:
        return None

# Default fighters
f1_default = "Alexander Volkanovski"
f2_default = "Max Holloway"

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Select Fighters")
    fighter_1 = st.selectbox("Red Corner", fighter_names, index=fighter_names.index(f1_default))
    fighter_2 = st.selectbox("Blue Corner", fighter_names, index=fighter_names.index(f2_default))

    with st.expander("Developer Options", expanded=False):
        debug_mode = st.checkbox("Enable Dev Mode")


    if debug_mode:
        st.sidebar.success("Developer mode is ON")



def display_fighter_card(name, corner_color, image_width = 200):
    import streamlit as st
    fighter_row = fighters_df[fighters_df["Name"] == name].iloc[0]
    # Map color to label for display
    color_label = "Red Corner" if corner_color == "red" else "Blue Corner"

    # Extract data
    name = fighter_row["Name"]
    image_url = get_ufc_image(name)  # Fallback if missing
    details = {
        "Age": fighter_row.get("Age", "N/A"),
        "Height": fighter_row.get("Height", "N/A"),
        "Reach": fighter_row.get("Reach", "N/A"),
        "Weight": fighter_row.get("Weight", "N/A"),
        "Stance": fighter_row.get("Stance", "N/A"),
        "Wins": fighter_row.get("Wins", "N/A"),
        "Losses": fighter_row.get("Losses", "N/A")
    }
    font_size = max(12, int(image_width * 0.11)) 
    st.markdown(f"### {color_label}")
    # Create two columns: one for image, one for stats
    col1, col2 = st.columns([1, 2])  # Adjust width ratio as needed

    with col1:
        st.image(image_url, width=200, caption=name)

    with col2:
        # Use HTML for font size control
        stats_html = ""
        for stat_name, stat_value in details.items():
            stats_html += f"<div style='font-size:{font_size}px; margin-bottom:4px'>{stat_name}:<b> {stat_value}</b></div>"

        st.markdown(stats_html, unsafe_allow_html=True)

# Fighter profile images
col1, col2 = st.columns(2)
with col1:
    display_fighter_card(fighter_1, corner_color="red")
with col2:
    display_fighter_card(fighter_2, corner_color="blue")

# Variance Weights
variance_weights = {
    'PerMin': 2.0, 'Per15Min': 1.5, 'Pct': 1.0, 'AccuracyPct': 1.0, 'DefencePct': 1.0
}

def get_weight(stat):
    for key, weight in variance_weights.items():
        if key in stat:
            return weight
    return 1.0

def compute_percentile_ratings(df, stats, invert_stats=None):
    if invert_stats is None:
        invert_stats = []
    ratings = pd.DataFrame(index=df.index)
    for stat in stats:
        percentiles = df[stat].rank(pct=True)
        if stat in invert_stats:
            percentiles = 1 - percentiles
        ratings[stat] = (percentiles * 10).round(2)
    return ratings

roster_ratings = compute_percentile_ratings(fighters_df, avg_stats, invert_stats=["StrikesAbsorbedPerMin"])

f1_row = fighters_df[fighters_df["Name"] == fighter_1].index[0]
f2_row = fighters_df[fighters_df["Name"] == fighter_2].index[0]

f1_ratings = roster_ratings.loc[f1_row]
f2_ratings = roster_ratings.loc[f2_row]

f1_raw = fighters_df.loc[f1_row, avg_stats]
f2_raw = fighters_df.loc[f2_row, avg_stats]
comparison_df = pd.DataFrame({fighter_1: f1_raw.values, "Stat": avg_stats, fighter_2: f2_raw.values})

# Stat Highlighting

def highlight_row(row):
    if len(row) < 3 or pd.isna(row[1]) or pd.isna(row[2]):
        return [''] * len(row)
    val1, stat, val2 = row[0], row[1], row[2]
    return [
        'background-color: #006400' if val1 > val2 else 'background-color: #8B0000' if val1 < val2 else '',
        '',
        'background-color: #006400' if val2 > val1 else 'background-color: #8B0000' if val2 < val1 else ''
    ]

st.markdown("---")
st.subheader("Fighter Stat Comparison")
render_comparison_table(comparison_df, highlight_row)

# Radar chart
st.markdown("---")
st.subheader("Fighter Performance Radar Chart")
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=f1_ratings, theta=avg_stats, fill='toself', name=fighter_1, line=dict(color='crimson')))
fig.add_trace(go.Scatterpolar(r=f2_ratings, theta=avg_stats, fill='toself', name=fighter_2, line=dict(color='royalblue')))
fig.update_layout(
    polar=dict(
        bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(visible=True, range=[0, 10], tickvals=[0, 2.5, 5, 7.5, 10]),
        angularaxis=dict(tickfont=dict(size=13))
    ),
    showlegend=True,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Prediction
@st.cache_resource
def load_model():
    with open("models/ufc_fight_predictor.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

def create_features_from_df(f1, f2, df):
    red = df[df["Name"] == f1].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
    blue = df[df["Name"] == f2].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
    red.index = ["red_" + col for col in red.index]
    blue.index = ["blue_" + col for col in blue.index]
    return pd.concat([red, blue]).to_frame().T



original_X = create_features_from_df(fighter_1, fighter_2, fighters_df)
swapped_X = create_features_from_df(fighter_2, fighter_1, fighters_df)
numeric_cols = [
    'red_StrikesLandedPerMin', 'red_StrikesAbsorbedPerMin', 'red_TakedownsPer15Min', 'red_SubmissionsPer15Min',
    'red_ControlPer15Min', 'red_StrikingAccuracyPct', 'red_StrikeDefencePct', 'red_TakedownAccuracyPct', 'red_TakedownDefencePct',
    'blue_StrikesLandedPerMin', 'blue_StrikesAbsorbedPerMin', 'blue_TakedownsPer15Min', 'blue_SubmissionsPer15Min',
    'blue_ControlPer15Min', 'blue_StrikingAccuracyPct', 'blue_StrikeDefencePct', 'blue_TakedownAccuracyPct', 'blue_TakedownDefencePct',
    'red_OpponentTakedownsPer15Min', 'blue_OpponentTakedownsPer15Min',
    'red_Height', 'red_Weight', 'red_Reach', 'blue_Height', 'blue_Weight', 'blue_Reach', 'red_Elo', 'blue_Elo', 'red_Age', 'blue_Age']

for col in numeric_cols:
    if col in original_X.columns:
        original_X[col] = pd.to_numeric(original_X[col], errors='coerce')
        swapped_X[col] = pd.to_numeric(swapped_X[col], errors='coerce')

categorical_cols = ["red_Stance", "blue_Stance"]
for col in categorical_cols:
    original_X[col] = original_X[col].astype("category")
    swapped_X[col] = swapped_X[col].astype("category")


import numpy as np

def check_corner_bias(original_winner, swapped_winner, original_proba, swapped_proba, threshold=0.10):
    # 1) Numeric confidence drop\
    conf_orig = np.max(original_proba)
    conf_swap = np.max(swapped_proba)
    confidence_drop = abs(conf_orig - conf_swap)
    
    # 2) Was the drop “significant”?
    significant_drop = confidence_drop > threshold
    
    # 3) Did the predicted class flip?
    prediction_flipped =  original_winner!= swapped_winner
    
    return {
        "confidence_drop": confidence_drop,
        "significant_drop": significant_drop,
        "prediction_flipped": prediction_flipped,

    }




#original_X = original_X.fillna(0)

if st.button("Predict Winner"):
    original_X = original_X.reindex(columns=model.feature_names_in_, fill_value=0)
    original_pred = model.predict(original_X)[0]
    original_confidence = model.predict_proba(original_X)[0].max()
    original_winner = fighter_1 if original_pred == 0 else fighter_2

    swapped_X = swapped_X.reindex(columns=model.feature_names_in_, fill_value=0)
    swapped_pred = model.predict(swapped_X)[0]
    swapped_confidence = model.predict_proba(swapped_X)[0].max()
    swapped_winner = fighter_1 if swapped_pred == 0 else fighter_2

    st.success(f"Original Prediction: 🏆 {original_winner} would win with confidence **{original_confidence:.2%}**")
    st.success(f"Swapped Prediction:  🏆 {swapped_winner} would win with confidence **{swapped_confidence:.2%}**")

    

    st.write("### Bias Test: Swap Fighter Corners")

    check = check_corner_bias(original_winner, swapped_winner, original_confidence, swapped_confidence, threshold=0.10)

    if check["prediction_flipped"]:
        st.error("Prediction flipped when corners were swapped — potential corner bias!")
    elif check["significant_drop"]:
        st.warning("Confidence dropped by over 10% when corners were swapped — potential bias!")
    else:
        st.success("Prediction stable and no large confidence drop when corners swapped.")


    st.subheader("SHAP Model Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(original_X)
    sample = shap_values[0]

    indices = np.argsort(np.abs(sample.values))[::-1][:15]
    feature_names = np.array(sample.feature_names)[indices]
    shap_vals = sample.values[indices]
    input_vals = original_X.iloc[0][feature_names].values
    formatted_labels = [
    f"{name} ({float(val):.2f})" if isinstance(val, (int, float)) or val.replace('.', '', 1).isdigit() else f"{name} ({val})"
    for name, val in zip(feature_names, input_vals)
]


    colors = ['#1f77b4' if val > 0 else '#d62728' for val in shap_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(formatted_labels))
    bars = ax.barh(y_pos, shap_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(formatted_labels)
    ax.invert_yaxis()
    ax.set_title("SHAP Feature Impact with Input Values")
    ax.set_xlabel("SHAP value (impact on model output)")
    st.pyplot(fig)

