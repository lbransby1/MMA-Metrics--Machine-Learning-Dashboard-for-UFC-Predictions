import streamlit as st
import pandas as pd

def greet(name):
    return "Hello " + name + "!! :O"

st.title("Greeting App")

name = st.text_input("Enter your name")

if st.button("Greet"):
    greeting = greet(name)
    st.write(greeting)



# Load the fighter averages CSV
fighters_df = pd.read_csv("fighter_averages.csv")

# Get list of fighter names
fighter_names = fighters_df["Name"].sort_values().tolist()

# Streamlit dropdown
selected_fighter = st.selectbox("Choose a fighter", fighter_names)

# Display selected fighter stats
fighter_stats = fighters_df[fighters_df["Name"] == selected_fighter]

st.write(f"### Stats for {selected_fighter}")
st.dataframe(fighter_stats.transpose())
