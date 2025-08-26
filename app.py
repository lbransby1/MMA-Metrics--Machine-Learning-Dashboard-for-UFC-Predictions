import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import requests
import pickle
from bs4 import BeautifulSoup
from datetime import datetime
import streamlit.components.v1 as components
from comparison_table import render_comparison_table
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set wide layout and dashboard title
st.set_page_config(layout="wide", page_title="UFC Fight Predictor Dashboard")
st.title("UFC Fight Predictor Dashboard")
st.markdown("Compare fighters, analyze stats, and predict outcomes using ML")



# Load and clean fighter data
fighters_df = pd.read_csv("processed_data/fighter_averages.csv")
fighters_df["Name"] = fighters_df["Name"].astype(str)
fighter_names = sorted(fighters_df["Name"].dropna().unique())

# Load fight styles
styles_df = pd.read_csv("processed_data/fight_style_descriptions.csv")
fighters_df = fighters_df.merge(styles_df, on="Name", how="left")

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

    st.markdown(f"<p style='font-size:{font_size}px'><b>Style</b>: {fighter_row.get('Style','N/a')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{font_size}px'><b>Strengths</b>: {fighter_row.get('Strengths','N/a')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{font_size}px'><b>Weaknesses</b> : {fighter_row.get('Weaknesses','N/a')}</p>", unsafe_allow_html=True)

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
# Round all numeric columns to 2 decimal places
# Only round numeric columns, keep 'Stat' as-is
numeric_cols = [c for c in comparison_df.columns if c != "Stat"]
comparison_df[numeric_cols] = comparison_df[numeric_cols].applymap(lambda x: f"{x:.2f}")


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

# Create two columns
col_table, col_radar = st.columns([1, 1])  # Equal width, can adjust ratio if needed

with col_table:
    st.markdown("### Stat Comparison")
    render_comparison_table(comparison_df, highlight_row)  # Your styled comparison table

with col_radar:
    st.markdown("### Performance Radar Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=f1_ratings, theta=avg_stats, fill='toself',
        name=fighter_1, line=dict(color='crimson')
    ))
    fig.add_trace(go.Scatterpolar(
        r=f2_ratings, theta=avg_stats, fill='toself',
        name=fighter_2, line=dict(color='royalblue')
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 10], 
                tickvals=[0, 2.5, 5, 7.5, 10],
                tickfont=dict(size=14)  # radial axis labels
            ),
            angularaxis=dict(
                tickfont=dict(size=16)  # stats labels around the chart
            )
        ),
        showlegend=False,
        height=450  # smaller than before
    )
    st.plotly_chart(fig, use_container_width=True)

# Prediction
@st.cache_resource
def load_model():
    with open("models/ufc_fight_predictor.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

@st.cache_resource
def load_models():
    models = {}
    model_names = [
        "RandomForest_Opt",
        "XGBoost_Opt",
        "LightGBM_Opt",
        "CatBoost_Opt",
        "HistGB_Opt",
        "Votingclf",
        "Stacking",
    ]
    for name in model_names:
        path = f"models/{name}_model.pkl"
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_models()

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

def swap_averaged_all(models, X_orig, X_swapped, y_true=None, fighters=None):
    """
    Returns a dictionary of results for each model, including:
    - pre-swap probabilities
    - swap-averaged probabilities
    - predicted winner
    - accuracy (if y_true provided)
    """
    results = {}
    for name, model in models.items():
        # Predict probabilities
        orig_probs = model.predict_proba(X_orig)
        swap_probs = model.predict_proba(X_swapped)
        swap_probs_corrected = swap_probs[:, [1, 0]]  # flip to align with original

        # Swap-averaged probabilities
        final_probs = (orig_probs + swap_probs_corrected) / 2
        print(name, "og prob", orig_probs, "swap prob:", swap_probs, final_probs)
        # Single fight winner
        if fighters and final_probs.shape[0] == 1:
            f1, f2 = fighters
            winner = f1 if final_probs[0][0] > final_probs[0][1] else f2
            results[name] = {
                "PreSwapProbs": orig_probs[0],
                "Probs": final_probs[0],
                "Winner": winner
            }
        else:
            # Dataset
            y_pred_final = np.argmax(final_probs, axis=1)
            acc = None
            if y_true is not None:
                acc = (y_pred_final == y_true).mean()
            results[name] = {
                "PreSwapProbs": orig_probs,
                "Probs": final_probs,
                "Accuracy": acc
            }
    return results

# fighter_1 = "Jack Della Maddalena"
# fighter_2 = "Ilia Topuria"

# --- Save the feature order from training ---

def create_features_from_df(f1, f2, df):
    red = df[df["Name"] == f1].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
    blue = df[df["Name"] == f2].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
    red.index = ["red_" + col for col in red.index]
    blue.index = ["blue_" + col for col in blue.index]
    return pd.concat([red, blue]).to_frame().T
    
st.markdown(
    """
    <style>
    .stButton > button {
        transform: scale(1.5);  /* make button + text 1.5x bigger */
        font-weight: bold;
        width: 70%;
        margin: 0 auto;  /* centers the button */
        display: block;  /* required for margin auto to work */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Button
predict_button = st.button("Predict")


if predict_button:
    fighter_1_sel = st.session_state.get("Red Corner", fighter_1)
    fighter_2_sel = st.session_state.get("Blue Corner", fighter_2)
    # Build feature rows for both orderings
    original_X = create_features_from_df(fighter_1_sel, fighter_2_sel, fighters_df)
    swapped_X = create_features_from_df(fighter_2_sel, fighter_1_sel, fighters_df)

    # # Ensure numeric cols are numeric
    numeric_cols = [
        'red_StrikesLandedPerMin', 'red_StrikesAbsorbedPerMin', 'red_TakedownsPer15Min', 'red_SubmissionsPer15Min',
        'red_ControlPer15Min', 'red_StrikingAccuracyPct', 'red_StrikeDefencePct', 'red_TakedownAccuracyPct', 'red_TakedownDefencePct',
        'blue_StrikesLandedPerMin', 'blue_StrikesAbsorbedPerMin', 'blue_TakedownsPer15Min', 'blue_SubmissionsPer15Min',
        'blue_ControlPer15Min', 'blue_StrikingAccuracyPct', 'blue_StrikeDefencePct', 'blue_TakedownAccuracyPct', 'blue_TakedownDefencePct',
        'red_OpponentTakedownsPer15Min', 'blue_OpponentTakedownsPer15Min',
        'red_Height', 'red_Weight', 'red_Reach', 'blue_Height', 'blue_Weight', 'blue_Reach',
        'red_Elo', 'blue_Elo', 'red_Age', 'blue_Age'
    ]
    for col in numeric_cols:
        if col in original_X.columns:
            original_X[col] = pd.to_numeric(original_X[col], errors='coerce')
            swapped_X[col]  = pd.to_numeric(swapped_X[col], errors='coerce')

    categorical_cols = ["red_Stance", "blue_Stance"]

    # One-hot encode all categorical columns
    original_X = pd.get_dummies(original_X, columns=categorical_cols)
    swapped_X = pd.get_dummies(swapped_X, columns=categorical_cols)

    original_X = original_X.reindex(columns=feature_columns, fill_value=0)
    swapped_X = swapped_X.reindex(columns=feature_columns, fill_value=0)

    print(original_X.head())
    # Run ensemble predictions
    results = swap_averaged_all(models, original_X, swapped_X, fighters=(fighter_1, fighter_2))

    X_test = pd.read_csv("processed_data/X_test.csv")
    y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()  # flatten if needed

    # Evaluate each model's accuracy on test data once
    accuracies = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc


    # Tabulate results
    rows = []
    for name, res in results.items():
        prob_red, prob_blue = res["Probs"]
        rows.append({
            "Model": name,
            f"{fighter_1} Win %": f"{prob_red:.2%}",
            f"{fighter_2} Win %": f"{prob_blue:.2%}",
            "Predicted Winner": res["Winner"],
            "Test Accuracy": f"{accuracies[name]:.2%}"
        })

    
    results_df = pd.DataFrame(rows)
    

    # Convert to HTML
    def color_win_percent(val):
        # Convert string to float if needed
        if isinstance(val, str):
            val = val.strip('%')  # Remove %
            try:
                val = float(val) / 100  # convert to 0-1
            except:
                return ""
        if val >= 60:
            return "color: ##3ffc3f; font-weight:bold;"  # very green
        elif val >= 0.55:
            return "color: #80ff80; font-weight:bold;"  # green
        elif val >= 0.50:
            return "color: #b8ffb8; font-weight:bold;"  # green ish
        elif val >= 0.45:
            return "color: #ffabab; font-weight:bold;"  # red
        elif val >= 0.40:
            return "color: #ff7070; font-weight:bold;"  # red
        elif val <0.40:
            return "color: #fa3939; font-weight:bold;"  # very red
        else:
            return ""


    html_table = results_df.to_html(index=False, escape=False)

    # Apply CSS manually
    html_string = f"""
    <html>
    <head>
    <style>
        body {{
            background-color: #1e1e1e;
            color: #f0f0f0;
            font-family: Arial, sans-serif;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 16px;
        }}
        th, td {{
            padding: 10px;
            border: 1px solid #444;
            text-align: center;
        }}
        th {{
            background-color: #333;
            color: #fff;
        }}
        tr:nth-child(even) {{
            background-color: #2a2a2a;
        }}
        tr:nth-child(odd) {{
            background-color: #242424;
        }}
    </style>
    </head>
    <body>
    <table>
        <thead>
            <tr>
                {''.join(f'<th>{c}</th>' for c in results_df.columns)}
            </tr>
        </thead>
        <tbody>
    """

    for _, row in results_df.iterrows():
        html_string += "<tr>"
        for col in results_df.columns:
            val = row[col]

            # Handle percentages for fighters
            if col in [f"{fighter_1} Win %", f"{fighter_2} Win %"]:
                style = color_win_percent(val)
                # Convert string like "60.00%" to float
                if isinstance(val, str):
                    val = val.strip('%')
                    val = float(val) / 100
                html_string += f'<td style="{style}">{val:.2%}</td>'

            # Handle Test Accuracy
            elif col == "Test Accuracy":
                # Convert string to float if necessary
                if isinstance(val, str):
                    val = val.strip('%')
                    val = float(val) / 100
                html_string += f'<td>{val:.2%}</td>'
            else:
                html_string += f'<td>{val}</td>'
        html_string += "</tr>"


    html_string += "</tbody></table></body></html>"
    

    stack_result = results.get("Stacking", None)
    if stack_result:
        prob_red, prob_blue = stack_result["Probs"]
        winner = stack_result["Winner"]

        # Nice styled "Winner Winner"
        st.markdown(
            f"""
            <div style="
                background-color:#222;
                color:#fff;
                border: 2px solid #444;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 20px;">
                🏆 Winner Prediction: <span style="color:#3fcf5f">{winner}</span><br>
                Confidence: {max(prob_red, prob_blue):.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
    # Render in Streamlit
    components.html(html_string, height=400, scrolling=False)





