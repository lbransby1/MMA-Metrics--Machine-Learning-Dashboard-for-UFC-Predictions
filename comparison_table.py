# table_styles.py
import streamlit as st

def render_comparison_table(comparison_df, highlight_func):
    # Apply highlighting and hide index
    styled_html = (
        comparison_df.style
            .apply(highlight_func, axis=1)
            .hide(axis="index")
            .to_html(index=False)
    )

    # CSS for nicer dark table
    html = f"""
<style>
    table {{
        border-collapse: collapse;
        margin: 0 auto;
        font-family: Arial, sans-serif;
        width: 90%;
        max-width: 900px;
        table-layout: fixed;
        background-color: #1e1e1e;
        color: #f0f0f0;
        border-radius: 8px;
        overflow: hidden;
    }}
    th, td {{
        border: 1px solid #444;
        padding: 8px 12px;
        text-align: center;
        font-size: 14px;
        vertical-align: middle;
        word-wrap: break-word;
    }}
    th {{
        background-color: #333;
    }}
    tbody tr:hover {{
        background-color: #2a2a2a;
    }}
</style>
<div>
    {styled_html}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
