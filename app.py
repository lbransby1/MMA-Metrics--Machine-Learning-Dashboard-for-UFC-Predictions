import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load and clean fighter data
fighters_df = pd.read_csv("fighter_averages.csv")
fighters_df["Name"] = fighters_df["Name"].astype(str)
fighter_names = sorted(fighters_df["Name"].dropna().unique())

# Select only averaged stats
avg_stats = [col for col in fighters_df.columns if (
    "PerMin" in col or 
    "Per15Min" in col or 
    "AccuracyPct" in col or 
    "DefencePct" in col or
    "Pct" in col
) and col != "KnockdownPct"]

# Dropdowns side-by-side
col1, col2 = st.columns(2)
with col1:
    fighter_1 = st.selectbox("Select Red Fighter", fighter_names, key="f1")
with col2:
    fighter_2 = st.selectbox("Select Blue Fighter", fighter_names, key="f2")

# Variance weights for more meaningful stat spread
variance_weights = {
    'PerMin': 2.0,
    'Per15Min': 1.5,
    'Pct': 1.0,
    'AccuracyPct': 1.0,
    'DefencePct': 1.0
}

def get_weight(stat):
    for key, weight in variance_weights.items():
        if key in stat:
            return weight
    return 1.0

# Calculate weighted z-score based normalized ratings (scaled to 0-10)
def compute_percentile_ratings(df, stats, invert_stats=None):
    if invert_stats is None:
        invert_stats = []

    ratings = pd.DataFrame(index=df.index)
    for stat in stats:
        percentiles = df[stat].rank(pct=True)  # 0–1 percentile
        if stat in invert_stats:
            percentiles = 1 - percentiles  # invert so lower is better
        ratings[stat] = (percentiles * 10).round(2)  # scale to 0–10
    return ratings


# Get normalized ratings for all fighters
roster_ratings = compute_percentile_ratings(fighters_df, avg_stats, invert_stats=["StrikesAbsorbedPerMin"])


# Get selected fighter ratings
f1_row = fighters_df[fighters_df["Name"] == fighter_1].index[0]
f2_row = fighters_df[fighters_df["Name"] == fighter_2].index[0]

f1_ratings = roster_ratings.loc[f1_row]
f2_ratings = roster_ratings.loc[f2_row]

# Combine comparison DataFrame (original values, not ratings)
f1_raw = fighters_df.loc[f1_row, avg_stats]
f2_raw = fighters_df.loc[f2_row, avg_stats]
comparison_df = pd.DataFrame({
    "Stat": avg_stats,
    fighter_1: f1_raw.values,
    fighter_2: f2_raw.values
})

# Highlight function

def highlight_row(row):
    if len(row) < 3 or pd.isna(row[1]) or pd.isna(row[2]):
        return [''] * len(row)

    stat_name = row[0]
    val1 = row[1]
    val2 = row[2]
    highlight = []
    for i in range(len(row)):
        if i == 1 or i == 2:
            if i == 1:
                if val1 > val2:
                    highlight.append('background-color: #006400')
                elif val1 < val2:
                    highlight.append('background-color: #8B0000')
                else:
                    highlight.append('')
            else:
                if val2 > val1:
                    highlight.append('background-color: #006400')
                elif val2 < val1:
                    highlight.append('background-color: #8B0000')
                else:
                    highlight.append('')
        else:
            highlight.append('')
    return highlight

# Show styled comparison table
st.write(
    comparison_df.style.apply(highlight_row, axis=1).hide(axis="index")
)

# Radar Chart (using full stat labels)
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=f1_ratings,
    theta=avg_stats,
    fill='toself',
    name=fighter_1,
    line=dict(color='rgba(128, 0, 0, 0.7)')
))
fig.add_trace(go.Scatterpolar(
    r=f2_ratings,
    theta=avg_stats,
    fill='toself',
    name=fighter_2,
    line=dict(color='rgba(0, 0, 128, 0.7)')
))

fig.update_layout(
    polar=dict(
        bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(
            visible=True,
            range=[0, 10],
            tickvals=[0, 2.5, 5, 7.5, 10],
            ticktext=['0', '2.5', '5', '7.5', '10']
        ),
        angularaxis=dict(
            tickfont=dict(size=13, color='white'),
            rotation=90,
            direction="clockwise"
        )
    ),
    showlegend=True,
    title="Fighter Ratings (Relative to Roster Avg, Weighted by Stat Type)",
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(fig, use_container_width=True)
